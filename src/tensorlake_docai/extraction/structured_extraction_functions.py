# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import re
import time
from typing import List, Optional

from tensorlake_docai.pipeline.api import (
    Chunk,
    ParsedDocumentRef,
    PatternChunking,
    StructuredData,
    Usage,
)
from tensorlake_docai.extraction.chunking_functions import chunk_pages
from tensorlake_docai.postprocess.formatter import document_layout_to_document
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.providers.model_provider_utils import (
    OPENAI_LLM_MODEL_NAME,
    get_gemini_async_client_and_model,
    get_openai_client_and_model,
)
from tensorlake_docai.extraction.openai_schema_enricher import pydantic_converter
from tensorlake_docai.pipeline.output_formatter import format_final_output
from pydantic import BaseModel
from tensorlake_docai.extraction.schema_chunker import split_schema
from tensorlake_docai.extraction.citation_handler import StructuredExtractionCitationHandler
from tensorlake_docai.extraction.tabular_content_splitter import (
    estimate_tokens,
    is_output_token_limit_error,
    merge_extraction_results,
    should_split_dense_table,
    split_dense_table_content,
)
from tensorlake.applications import RequestError as RequestException
from tensorlake.applications import Retries, cls, function
from tensorlake_docai.vlm.workflow_images import structured_extraction_image
from tensorlake_docai.pipeline.routing import stream_with_timeout, update_progress_if_needed

# Safe token limit to avoid hitting context limits
SAFE_TOKEN_LIMIT = 8000

SYSTEM_PROMPT = """You are an expert at extracting information from text."""

# Azure OpenAI Configuration:
# Set USE_AZURE_OPENAI=true to use Azure OpenAI instead of regular OpenAI
# Required environment variables when using Azure OpenAI:
# - AZURE_OPENAI_ENDPOINT: The endpoint URL for your Azure OpenAI resource
# - AZURE_OPENAI_API_KEY: The API key for your Azure OpenAI resource
# - AZURE_OPENAI_MODEL_DEPLOYMENT_NAME: The deployment name of your model in Azure OpenAI


def _redact_provider_names(raw_error: str) -> str:
    """Minimally redact provider names from an error string to comply with
    service abstraction rules while preserving the original error details.
    """
    if not raw_error:
        return ""
    redactions = [
        "openai",
        "anthropic",
        "azure",
        "textract",
        "gemini",
        "google",
        "aws",
        "qwen",
        "vllm",
        "document intelligence",
    ]
    sanitized = raw_error
    for term in redactions:
        sanitized = re.sub(term, "service", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def _create_token_limit_error_message(error_str: str) -> str:
    """Create a user-friendly error message when token limits are exceeded.

    Detects token limit issues from error messages and provides helpful guidance
    including token usage information and chunking recommendations.
    Handles two types of errors:
    1. Output length limit (completion tokens exceeded)
    2. Input context length limit (prompt too large)
    """
    import re

    # Separate error indicators by type
    output_limit_indicators = [
        "length limit was reached",
        "completion_tokens",
        "token limit",
    ]

    input_context_indicators = ["maximum context length", "context length exceeded"]

    error_str_lower = error_str.lower()

    # Check for output limit errors first
    is_output_limit_error = any(
        indicator in error_str_lower for indicator in output_limit_indicators
    )
    is_input_context_error = any(
        indicator in error_str_lower for indicator in input_context_indicators
    )

    if not (is_output_limit_error or is_input_context_error):
        return None

    # Extract token information from the error message
    completion_tokens_match = re.search(r"completion_tokens[=:](\d+)", error_str)
    prompt_tokens_match = re.search(r"prompt_tokens[=:](\d+)", error_str)
    total_tokens_match = re.search(r"total_tokens[=:](\d+)", error_str)

    completion_tokens = completion_tokens_match.group(1) if completion_tokens_match else "unknown"
    prompt_tokens = prompt_tokens_match.group(1) if prompt_tokens_match else "unknown"
    total_tokens = total_tokens_match.group(1) if total_tokens_match else "unknown"

    if is_output_limit_error:
        # Output token limit exceeded - model hit max generation limit
        max_output_tokens = 16384  # Common limit for many models

        user_friendly_msg = (
            f"Output token limit exceeded during structured extraction. "
            f"The model reached its maximum output generation limit.\n\n"
            f"Token Usage Details:\n"
            f"• Input tokens (prompt): {prompt_tokens}\n"
            f"• Output tokens generated: {completion_tokens}\n"
            f"• Total tokens: {total_tokens}\n"
            f"• Max output token limit: {max_output_tokens}\n\n"
            f"Recommendations:\n"
            f"• Use the Chunking Strategy to split your pages into smaller sections\n"
            f"• This reduces the amount of data the model needs to extract per chunk\n"
            f"• Consider simplifying your extraction schema to require less output\n"
            f"• Break complex schemas into smaller, focused extraction tasks"
        )

    else:  # is_input_context_error
        # Input context length exceeded - prompt is too large
        max_context_tokens = 128000  # Common context limit for many models

        user_friendly_msg = (
            f"Input context length exceeded during structured extraction. "
            f"The document content is too large for the model to process.\n\n"
            f"Token Usage Details:\n"
            f"• Input tokens (prompt): {prompt_tokens}\n"
            f"• Output tokens generated: {completion_tokens}\n"
            f"• Total tokens: {total_tokens}\n"
            f"• Max context length limit: {max_context_tokens}\n\n"
            f"Recommendations:\n"
            f"• Use the Chunking Strategy to split your pages into smaller sections\n"
            f"• This is essential - your document content exceeds the model's input capacity\n"
            f"• Reduce the number of pages processed in a single extraction\n"
            f"• Consider page classification to filter out irrelevant pages"
        )

    return user_friendly_msg


def is_input_token_limit_error(exception: Exception) -> bool:
    """Check if the exception is due to input token limit exceeded."""
    error_str = str(exception)
    if "Input context length exceeded" in error_str:
        return True
    token_limit_msg = _create_token_limit_error_message(error_str)
    return token_limit_msg is not None and "Input context length exceeded" in token_limit_msg


def add_page_marker_to_content(content: str, page_number: int) -> str:
    """Add page marker to content for better LLM context during structured extraction."""
    return f"--- Page {page_number} ---\n{content}"


def should_add_page_markers(
    parsed_pages: List[Chunk],
    actual_page_numbers: List[int],
    chunking_strategy: Optional[str] = None,
) -> bool:
    """
    Determine if page markers should be added based on chunking strategy.
    Add markers ONLY for:
    - Page-level chunking (chunking_strategy == "page")
    - Document-level chunking (chunking_strategy == None/default and multiple pages)

    Do NOT add markers for:
    - Section-level chunking (sections may span multiple pages)
    - Fragment-level chunking (fragments may span multiple pages)
    """
    # Only add page markers for page-level and document-level chunking
    if chunking_strategy == "page":
        # Page-level chunking: each chunk represents a single page
        return True
    elif chunking_strategy in [None, "none"]:
        # Document-level chunking: single chunk with multiple pages
        return len(parsed_pages) == 1 and len(actual_page_numbers) > 1
    else:
        # Section or fragment chunking: don't add page markers
        # because sections/fragments may span multiple pages
        return False


def build_document_content_with_page_markers(pages: List, request) -> str:
    """
    Build document content with individual page markers for each page.
    This is used for document-level chunking to provide clear page boundaries.
    """
    from tensorlake_docai.postprocess.formatter import page_to_markdown

    text = ""
    for page in pages:
        # Add page marker for this page
        page_content = page_to_markdown(page, request)
        text += add_page_marker_to_content(page_content, page.page_number)
        text += "\n\n"

    return text


EXTRACT_PROMPT = """Extract the following information into the given json schema.

* If information for a field is not present, return "null" for that field. Don't make up any information that isn't present in the given text.
* Don't exclude any information from the source text which is relevant to the schema. Pay special attention to objects and arrays.
* Return only the json and nothing else.
* It should be directly parseable by json.loads, so don't add any formatting like ```json

JSON SCHEMA:
{json_schema}

TEXT:
{text}
"""

EXTRACT_PROMPT_OAI_STRICT_JSON_ADHERENCE = """Extract the following information into the given json schema.

* If information for a field is not present, return "null" for that field. Don't make up any information that isn't present in the given text.
* Don't exclude any information from the source text which is relevant to the schema. Pay special attention to objects and arrays.
* Return only the json and nothing else.
* It should be directly parseable by json.loads, so don't add any formatting like ```json

TEXT:
{text}
"""


def extract_chunking_info(partition_strategy):
    """Extract chunking strategy string and patterns from PartitionStrategy."""
    if partition_strategy is None:
        return None, None, None

    if isinstance(partition_strategy, str):
        # Handle string values directly
        return partition_strategy, None, None

    if isinstance(partition_strategy, PatternChunking):
        # Extract from PatternChunking object
        return (
            "patterns",
            partition_strategy.start_patterns,
            partition_strategy.end_patterns,
        )

    # If it's not a PatternChunking object, treat as string
    return str(partition_strategy), None, None


SECRETS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "USE_AZURE_OPENAI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME",
]


class ModelResponse(BaseModel):
    resp: Optional[dict] = None
    input_tokens: int = 0
    output_tokens: int = 0


@cls()
class StructuredExtraction:
    def __init__(self):
        self._initialize_client()
        # Initialize citation handler
        self.citation_handler = StructuredExtractionCitationHandler()
        self.enable_citations = False

    def _initialize_client(self):
        import anthropic

        # Store OpenAI API key (Azure config is handled by get_openai_client_and_model helper)
        self._oai_api_key: str = os.environ.get("OPENAI_API_KEY")

        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._anthropic = anthropic.AsyncAnthropic(
            api_key=anthropic_api_key,
        )

        self._gemini_api_key: str = os.environ.get("GEMINI_API_KEY")

    def _enrich_schema_for_anthropic(self, schema: dict) -> dict:
        """
        Add 'required' fields to reduce optional parameters for Anthropic.
        Anthropic has a limit of 24 optional parameters for structured outputs.
        Note: transform_schema() handles other constraints/validations.
        """

        def add_required_fields(obj: dict) -> None:
            if not isinstance(obj, dict):
                return

            schema_type = obj.get("type")

            # Make all properties required for objects
            if schema_type == "object" or (
                isinstance(schema_type, list) and "object" in schema_type
            ):
                if "properties" in obj and isinstance(obj["properties"], dict):
                    all_properties = list(obj["properties"].keys())
                    existing_required = set(obj.get("required", []))
                    obj["required"] = sorted(existing_required.union(all_properties))

                    for prop_value in obj["properties"].values():
                        add_required_fields(prop_value)

            # Handle array items
            elif schema_type == "array" or (
                isinstance(schema_type, list) and "array" in schema_type
            ):
                if "items" in obj:
                    add_required_fields(obj["items"])

            # Recurse into definitions
            for key in ["$defs", "definitions"]:
                if key in obj:
                    for def_value in obj[key].values():
                        add_required_fields(def_value)

        add_required_fields(schema)
        return schema

    def _json_schema_to_pydantic(self, json_schema: dict):
        """Convert JSON schema to Pydantic model dynamically"""
        from typing import Any, List, Optional

        from pydantic import create_model

        def json_type_to_python(prop_schema: dict):
            """Convert JSON schema type to Python type"""
            json_type = prop_schema.get("type", "string")

            if json_type == "string":
                return (str, ...)
            elif json_type == "number":
                return (float, ...)
            elif json_type == "integer":
                return (int, ...)
            elif json_type == "boolean":
                return (bool, ...)
            elif json_type == "array":
                items = prop_schema.get("items", {})
                item_type, _ = json_type_to_python(items)
                return (List[item_type], ...)
            elif json_type == "object":
                # For nested objects, create a nested model
                nested_props = prop_schema.get("properties", {})
                if nested_props:
                    nested_fields = {k: json_type_to_python(v) for k, v in nested_props.items()}
                    nested_model = create_model(
                        prop_schema.get("title", "NestedModel"), **nested_fields
                    )
                    return (nested_model, ...)
                return (dict, ...)
            else:
                return (Any, ...)

        # Build field definitions from JSON schema properties
        fields = {}
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        for field_name, field_schema in properties.items():
            python_type, default = json_type_to_python(field_schema)
            # If field is not required, make it optional
            if field_name not in required:
                python_type = Optional[python_type]
                default = None
            fields[field_name] = (python_type, default)

        # Create the Pydantic model
        model_name = json_schema.get("title", "DynamicSchema")
        return create_model(model_name, **fields)

    async def _make_anthropic_request(
        self, system_prompt: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> ModelResponse:
        # Time the LLM request
        start_time = time.time()

        try:
            # Convert JSON schema to Pydantic model for Anthropic streaming
            # Anthropic's streaming API expects a Pydantic model class
            DynamicModel = self._json_schema_to_pydantic(json_schema)

            # Accumulate streamed response with activity-based timeout
            parsed_data = None
            accumulated_chars = 0
            input_tokens = 0
            output_tokens = 0
            chunk_count = 0
            last_progress_update = start_time

            # Create streaming request with Pydantic model (context manager, not awaitable)
            async with self._anthropic.beta.messages.stream(
                model="claude-sonnet-4-5",
                system=system_prompt,
                max_tokens=16000,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                betas=["structured-outputs-2025-11-13"],
                output_format=DynamicModel,
            ) as stream:
                # Process stream with timeout that resets on each chunk
                async for event in stream_with_timeout(stream, timeout_seconds=120):
                    chunk_count += 1

                    # Get the latest parsed snapshot from various event types
                    # Structured outputs can arrive via text, input_json, or other event types
                    if event.type in ("text", "input_json", "content_block_delta"):
                        try:
                            parsed_data = event.parsed_snapshot()
                            # Estimate character count from serialized model
                            if parsed_data:
                                accumulated_chars = len(
                                    json.dumps(
                                        parsed_data.model_dump()
                                        if hasattr(parsed_data, "model_dump")
                                        else dict(parsed_data)
                                    )
                                )
                        except Exception:
                            # Not all events have parsed_snapshot, continue
                            pass

                    # Update progress to keep function-level timeout alive
                    last_progress_update = update_progress_if_needed(
                        chunk_count, accumulated_chars, start_time, last_progress_update
                    )

                # Get final message with token usage
                final_message = await stream.get_final_message()
                if final_message.usage:
                    input_tokens = final_message.usage.input_tokens or 0
                    output_tokens = final_message.usage.output_tokens or 0

            llm_time = time.time() - start_time
            print(f"Anthropic streaming response time: {llm_time:.2f}s")
            print(f"Anthropic tokens used - Input: {input_tokens}, Output: {output_tokens}")

            # Check if we received no chunks at all (indicates a problem)
            if chunk_count == 0:
                raise Exception("Anthropic returned no response chunks")

            # Convert Pydantic model to dict
            if parsed_data is None:
                raise Exception(
                    "Anthropic returned empty response - invalid for structured extraction"
                )

            resp_json = (
                parsed_data.model_dump()
                if hasattr(parsed_data, "model_dump")
                else dict(parsed_data)
            )

            return ModelResponse(
                resp=resp_json, input_tokens=input_tokens, output_tokens=output_tokens
            )

        except asyncio.TimeoutError as e:
            print(f"Anthropic streaming timeout: {str(e)}")
            raise Exception(f"Anthropic streaming timed out due to inactivity: {str(e)}")

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            self._handle_llm_error(e, "Anthropic")

    async def _make_gemini_request(
        self, system_prompt: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> ModelResponse:
        from google.genai import types

        # Time the LLM request
        start_time = time.time()
        client, model_name = get_gemini_async_client_and_model(
            api_key=self._gemini_api_key, model_name="gemini-3-flash-preview"
        )

        # Prepare configuration with JSON schema
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=64000,
            response_mime_type="application/json",
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        )

        # Add JSON schema if provided
        if json_schema:
            config.response_json_schema = json_schema

        # Make the async streaming request with activity timeout
        try:
            stream = await client.models.generate_content_stream(
                model=model_name,
                contents=user_prompt,
                config=config,
            )

            try:
                # Accumulate streamed response with activity-based timeout
                accumulated_text = ""
                input_tokens = 0
                output_tokens = 0
                activity_timeout = 5 * 60  # 5 minutes in seconds
                chunk_count = 0
                last_progress_update = start_time
                finish_reason = None

                # Process stream with timeout that resets on each chunk
                async for chunk in stream_with_timeout(stream, timeout_seconds=activity_timeout):
                    chunk_count += 1
                    if chunk.text:
                        accumulated_text += chunk.text

                    # Update progress to keep function-level timeout alive
                    last_progress_update = update_progress_if_needed(
                        chunk_count,
                        len(accumulated_text),
                        start_time,
                        last_progress_update,
                    )

                    # Extract token usage and finish reason from chunks
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        input_tokens = getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0
                        total_tokens = getattr(chunk.usage_metadata, "total_token_count", 0) or 0
                        output_tokens = total_tokens - input_tokens

                    # Track finish reason
                    if hasattr(chunk, "candidates") and chunk.candidates:
                        candidate_finish = getattr(chunk.candidates[0], "finish_reason", None)
                        if candidate_finish is not None:
                            finish_reason = candidate_finish
            finally:
                # Ensure stream cleanup
                if hasattr(stream, "aclose"):
                    try:
                        await stream.aclose()
                    except Exception as e:
                        print(f"WARNING: Failed to close Gemini stream: {e}")
                elif hasattr(stream, "close"):
                    try:
                        close_result = stream.close()
                        if hasattr(close_result, "__await__"):
                            await close_result
                    except Exception as e:
                        print(f"WARNING: Failed to close Gemini stream: {e}")

            llm_time = time.time() - start_time
            print(f"Gemini streaming response time: {llm_time:.2f}s")
            print(f"Gemini tokens used - Input: {input_tokens}, Output: {output_tokens}")

            # Check if we received no chunks at all (indicates a problem)
            if chunk_count == 0:
                raise Exception("Gemini returned no response chunks")

            # Check if response is empty
            if not accumulated_text:
                raise Exception(
                    "Gemini returned empty response - invalid for structured extraction"
                )

            # Check if output was truncated due to token limit
            finish_reason_str = str(finish_reason).upper() if finish_reason else ""
            if "MAX_TOKENS" in finish_reason_str or output_tokens >= 63500:
                # Raise exception with token info that _handle_llm_error can parse
                raise Exception(
                    f"completion_tokens={output_tokens} prompt_tokens={input_tokens} total_tokens={input_tokens + output_tokens} length limit was reached"
                )

            # Parse the JSON response
            try:
                resp_json = json.loads(accumulated_text)
            except json.JSONDecodeError as e:
                # If JSON parse fails and we're near token limit, it's likely truncation
                if output_tokens >= 60000:
                    # Raise exception with token info that _handle_llm_error can parse
                    raise Exception(
                        f"completion_tokens={output_tokens} prompt_tokens={input_tokens} total_tokens={input_tokens + output_tokens} token limit. JSON parse error: {str(e)}"
                    )
                raise e

            return ModelResponse(
                resp=resp_json, input_tokens=input_tokens, output_tokens=output_tokens
            )

        except asyncio.TimeoutError as e:
            print(f"Gemini streaming timeout: {str(e)}")
            raise Exception(f"Gemini streaming timed out due to inactivity: {str(e)}")

        except Exception as e:
            self._handle_llm_error(e, "Gemini")

    async def _make_openrouter_request(
        self, system_prompt: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> ModelResponse:
        """Run the same Gemini model via OpenRouter (OpenAI-compatible), matching the
        Gemini-direct contract Document AI uses: the schema is ENFORCED via
        response_format json_schema (OpenRouter forwards it to Gemini's native
        responseSchema), and the caller passes the schema-less prompt. strict=False keeps
        complex schemas (oneOf / optionals) working, as Gemini handles them natively.
        Falls back to loose json_object only when no schema is supplied."""
        import openai

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise Exception("OPENROUTER_API_KEY not set; cannot fall back to OpenRouter")
        model_name = os.environ.get("OPENROUTER_GEMINI_MODEL", "google/gemini-3-flash-preview")
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_extraction",
                    "strict": False,
                    "schema": json_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        start_time = time.time()
        client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        try:
            async with client:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=64000,
                    response_format=response_format,
                    # Match Gemini-direct's thinking_config=ThinkingLevel.LOW: OpenRouter
                    # maps reasoning.effort → Gemini thinkingLevel (effort "low" = LOW).
                    extra_body={"reasoning": {"effort": "low"}},
                )
            content = resp.choices[0].message.content or ""
            print(f"OpenRouter fallback response time: {time.time() - start_time:.2f}s")
            if not content:
                raise Exception("OpenRouter returned empty response")
            usage = getattr(resp, "usage", None)
            return ModelResponse(
                resp=json.loads(content),
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        except Exception as e:
            self._handle_llm_error(e, "OpenRouter")

    async def _make_openai_request(
        self,
        oai_client,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
        model_name: str = OPENAI_LLM_MODEL_NAME,
    ) -> ModelResponse:
        import openai

        oai_client: openai.AsyncOpenAI

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        req_params = {
            "model": model_name,
            "messages": messages,
            "reasoning_effort": "low",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        json_schema = pydantic_converter(
            json_schema, flavor="openai_output_schema", include_title=True
        )

        # Use smaller schema chunks when citations are enabled to improve adherence
        max_fields = 50 if getattr(self, "enable_citations", False) else 100
        schemas = split_schema(json_schema, max_fields=max_fields)

        try:
            structured_data = {}
            total_input_tokens = 0
            total_output_tokens = 0
            total_llm_time = 0

            for schema_idx, schema in enumerate(schemas):
                req_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema,
                }

                start_time = time.time()

                # stream the response with timeout management
                accumulated_text = ""
                usage_data = None
                chunk_count = 0
                last_progress_update = start_time

                # Create stream and wrap in try/finally for cleanup
                stream = await oai_client.chat.completions.create(**req_params)
                try:
                    async for chunk in stream_with_timeout(stream, timeout_seconds=120):
                        chunk_count += 1
                        # process each chunk
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                accumulated_text += delta.content

                        # Update progress to keep function-level timeout alive
                        last_progress_update = update_progress_if_needed(
                            chunk_count,
                            len(accumulated_text),
                            start_time,
                            last_progress_update,
                        )

                        # Capture usage data (comes in final chunk)
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_data = chunk.usage
                finally:
                    # Ensure stream cleanup
                    if hasattr(stream, "aclose"):
                        try:
                            await stream.aclose()
                        except Exception as e:
                            print(f"WARNING: Failed to close OpenAI stream: {e}")
                    elif hasattr(stream, "close"):
                        try:
                            close_result = stream.close()
                            # If close() returns a coroutine, await it
                            if hasattr(close_result, "__await__"):
                                await close_result
                        except Exception as e:
                            print(f"WARNING: Failed to close OpenAI stream: {e}")

                llm_time = time.time() - start_time
                total_llm_time += llm_time
                print(f"OpenAI response time for schema {schema_idx + 1}: {llm_time:.2f}s")

                # Capture token usage
                if usage_data:
                    total_input_tokens += usage_data.prompt_tokens or 0
                    total_output_tokens += usage_data.completion_tokens or 0

                # Check if we received no chunks at all (indicates a problem)
                if chunk_count == 0:
                    raise Exception("OpenAI returned no response chunks")

                # Check if response is empty
                if not accumulated_text:
                    raise Exception(
                        "OpenAI returned empty response - invalid for structured extraction"
                    )

                # Parse and accumulate the JSON response
                structured_data.update(json.loads(accumulated_text))

            print(f"OpenAI total response time: {total_llm_time:.2f}s")
            print(
                f"OpenAI tokens used - Input: {total_input_tokens}, Output: {total_output_tokens}"
            )

            return ModelResponse(
                resp=structured_data,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )

        except asyncio.TimeoutError as e:
            raise Exception(f"OpenAI streaming timed out due to inactivity: {str(e)}")
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            self._handle_llm_error(e, "OpenAI")

    def _handle_llm_error(self, e: Exception, provider: str):
        """Common error handling for all LLM providers"""
        print(f"Caught exception in {provider} call: {str(e)}")

        # Check if this is a token limit error and provide user-friendly message
        token_limit_msg = _create_token_limit_error_message(str(e))
        if token_limit_msg:
            raise RequestException(message=token_limit_msg)
        else:
            raise RequestException(
                message=("Structured extraction failed. " + _redact_provider_names(str(e)))
            )

    async def _extract_json(
        self,
        model_provider: str,
        text: str,
        json_schema: Optional[dict] = None,
        prompt: Optional[str] = None,
    ) -> ModelResponse:

        if prompt:
            if json_schema is not None:
                prompt = f"{prompt}\n\n{json_schema}\n\n{text}\n\nRespond in valid JSON."
            else:
                prompt = f"{prompt}\n\n{text}\n\nRespond in valid JSON."

            sonnet_prompt = f"{prompt}\n\n{text}\n\nRespond in valid JSON."
        else:
            if json_schema is None:
                raise RequestException(
                    message="Json schema should not be None if prompt is not provided"
                )

            if model_provider == "openai":
                prompt = EXTRACT_PROMPT_OAI_STRICT_JSON_ADHERENCE.format(text=text)
            else:
                prompt = EXTRACT_PROMPT.format(json_schema=json_schema, text=text)

            # for sonnet we do not pass the json schema to the prompt
            sonnet_prompt = EXTRACT_PROMPT.format(json_schema="", text=text)

        # Retry logic matching run_clients pattern - applies to all providers
        retries = 3
        for attempt in range(retries):
            try:
                if model_provider == "openai":
                    # Recreate OAI client for each request to workaround
                    # https://github.com/openai/openai-python/issues/1254
                    oai_client, model_name = get_openai_client_and_model(
                        api_key=self._oai_api_key, default_model=OPENAI_LLM_MODEL_NAME
                    )
                    async with oai_client:
                        return await self._make_openai_request(
                            oai_client=oai_client,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=prompt,
                            json_schema=json_schema,
                            model_name=model_name,
                        )
                elif model_provider == "anthropic":
                    return await self._make_anthropic_request(
                        SYSTEM_PROMPT, sonnet_prompt, json_schema
                    )
                elif model_provider == "gemini":
                    # EXACTLY as upstream OpenIngest / Document AI: Gemini-direct with
                    # Gemini's NATIVE response_json_schema (handles oneOf/optionals; no
                    # OpenAI-strict constraints). OpenRouter is OUR addition (not in
                    # upstream) and only a fallback when Google's API fails (503/quota) —
                    # so the default path is byte-faithful to Document AI. Set
                    # OPENROUTER_API_KEY to enable the fallback.
                    try:
                        return await self._make_gemini_request(
                            SYSTEM_PROMPT, sonnet_prompt, json_schema
                        )
                    except Exception as gem_err:
                        if os.environ.get("OPENROUTER_API_KEY"):
                            print(
                                f"Gemini-direct failed ({type(gem_err).__name__}); "
                                "falling back to OpenRouter"
                            )
                            return await self._make_openrouter_request(
                                SYSTEM_PROMPT, sonnet_prompt, json_schema
                            )
                        raise
                else:
                    print(f"Unknown model provider: {model_provider}")
                    raise RequestException(message=f"Unrecognized model provider: {model_provider}")

            except RequestException:
                # If this was the last attempt, re-raise the error
                if attempt == retries - 1:
                    raise

                # Otherwise retry with exponential backoff
                delay = 2**attempt
                print(
                    f"Provider error, retrying after {delay}s (attempt {attempt + 1}/{retries})..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                if is_input_token_limit_error(e):
                    # Input token limit errors are not retryable
                    raise RequestException(
                        message=f"Input token limit exceeded: {_redact_provider_names(str(e))}, no retry attempted."
                    )

                # Unexpected error - if last attempt, convert to RequestException
                if attempt == retries - 1:
                    error_str = str(e)
                    if "timed out due to inactivity" in error_str:
                        raise RequestException(
                            message="Structured extraction failed due to a streaming timeout from the AI provider. Please try again. If the issue persists, consider using chunking strategies to split the document into smaller sections."
                        )
                    raise RequestException(
                        message=f"Unexpected error: {_redact_provider_names(error_str)}"
                    )

                # Otherwise retry with exponential backoff
                delay = 2**attempt
                print(
                    f"Unexpected error, retrying after {delay}s (attempt {attempt + 1}/{retries})..."
                )
                await asyncio.sleep(delay)

    async def _extract_json_with_dense_table_splitting(
        self,
        model_provider: str,
        text: str,
        json_schema: Optional[dict] = None,
        prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Extract JSON by chunking dense tabular content into row-based chunks with headers."""
        print("Starting dense table chunking for CSV/Excel-like content")

        # Calculate schema tokens for chunk sizing
        schema_tokens = estimate_tokens(str(json_schema) if json_schema else "", model_provider)
        prompt_tokens = estimate_tokens(prompt or "", model_provider) + estimate_tokens(
            SYSTEM_PROMPT, model_provider
        )

        # Calculate max tokens per chunk: use conservative limit minus overhead
        # Target: schema_tokens * rows_per_chunk < 10000 (conservative threshold)
        overhead_tokens = prompt_tokens + 1000  # Buffer for response formatting

        # Calculate max tokens for table content per chunk
        max_tokens_per_chunk = SAFE_TOKEN_LIMIT - overhead_tokens

        if max_tokens_per_chunk <= 0:
            raise RequestException(
                message="Schema and prompts are too large to fit in model context"
            )

        # Chunk the table content
        table_chunks = split_dense_table_content(text, max_tokens_per_chunk, schema_tokens)

        print(f"Split dense table into {len(table_chunks)} chunks")

        # Process all chunks in parallel
        async def process_chunk(i, chunk_text):
            print(f"Processing table chunk {i + 1}/{len(table_chunks)}")
            try:
                return await self._extract_json(model_provider, chunk_text, json_schema, prompt)
            except RequestException as e:
                error_msg = str(e)
                if "Token limit exceeded" in error_msg or is_output_token_limit_error(error_msg):
                    print(f"Table chunk {i + 1} still too large, skipping")
                    return None
                else:
                    raise e

        # Create tasks for all chunks
        tasks = [process_chunk(i, chunk_text) for i, chunk_text in enumerate(table_chunks)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        chunk_results = []
        total_input_tokens = 0
        total_output_tokens = 0

        for ind_res, response in enumerate(responses):
            if response is not None and not isinstance(response, Exception) and response.resp:
                chunk_results.append(response.resp)
                # print(f"####Chunk result {ind_res}: {response.resp}")
                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens

        if not chunk_results:
            raise RequestException(message="Failed to process any table chunks successfully")

        # Merge results - for tabular data, typically merge arrays
        merged_result = merge_extraction_results(chunk_results, json_schema or {})

        return ModelResponse(
            resp=merged_result,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def _filter_page_layouts_for_chunk(
        self, parsed_chunk: Chunk, page_layouts: List, chunking_strategy: Optional[str]
    ) -> List:
        """
        Filter page_layouts to only include elements relevant to this chunk.

        Filtering strategy:
        - Section/Fragment: Element-level filtering (most precise)
        - Document-level (none): No filtering (all pages)
        - Page/Pattern: Page-level filtering
        """
        # Section/Fragment: Filter to specific elements in the chunk
        if chunking_strategy in ["section", "fragment"] and parsed_chunk.element_ids:
            element_id_set = set(parsed_chunk.element_ids)
            filtered_layouts = []
            for pl in page_layouts:
                filtered_elements = [e for e in pl.elements if e.ref_id in element_id_set]
                if filtered_elements:
                    from tensorlake_docai.models.layout_objects import PageLayout

                    filtered_layouts.append(
                        PageLayout(
                            elements=filtered_elements,
                            shape=pl.shape,
                            page_number=pl.page_number,
                            page_class=pl.page_class,
                            classification_reason=pl.classification_reason,
                            page_dimensions=pl.page_dimensions,
                        )
                    )
            return filtered_layouts

        # Document-level: No filtering (use all pages)
        if chunking_strategy in [None, "none"]:
            return page_layouts

        # Page/Pattern: Filter to relevant pages
        chunk_page_numbers = (
            parsed_chunk.page_numbers if parsed_chunk.page_numbers else [parsed_chunk.page_number]
        )
        return [pl for pl in page_layouts if pl.page_number in chunk_page_numbers]

    def _get_allowed_citation_filter(
        self, parsed_chunk: Chunk, chunking_strategy: Optional[str]
    ) -> tuple[Optional[set], Optional[set]]:
        """
        Determine which citations are allowed for this chunk.

        Returns: (allowed_ref_ids, allowed_pages)
        - If allowed_ref_ids is set: use element-level filtering
        - If allowed_pages is set: use page-level filtering
        - If both None: no filtering (document-level)
        """
        # Section/Fragment: Element-level filtering
        if chunking_strategy in ["section", "fragment"] and parsed_chunk.element_ids:
            return set(parsed_chunk.element_ids), None

        # Document-level: No filtering
        if chunking_strategy in [None, "none"]:
            return None, None

        # Page/Pattern: Page-level filtering
        chunk_page_numbers = (
            parsed_chunk.page_numbers if parsed_chunk.page_numbers else [parsed_chunk.page_number]
        )
        return None, set(chunk_page_numbers)

    def extract_data(
        self,
        model_provider: str,
        parsed_pages: List[Chunk],
        json_schema: Optional[str] = None,
        prompt: Optional[str] = None,
        actual_page_numbers: List[int] = [],
        schema_name: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        pages: Optional[List] = None,
        request: Optional[object] = None,
        page_layouts: Optional[List] = None,
        enable_citation: bool = False,
    ) -> tuple[List[StructuredData], int, int]:
        print(
            f"Extracting data with model provider: {model_provider}, number of pages: {len(parsed_pages)}, schema_name: {schema_name}, enable_citation: {enable_citation}"
        )

        # Set the citation flag for this extraction
        self.enable_citations = enable_citation

        try:
            if json_schema:
                json_schema = json.loads(json_schema)
                # Handle double-encoded JSON
                if isinstance(json_schema, str):
                    json_schema = json.loads(json_schema)
                # Enforce that the result is a dict (JSON object)
                if not isinstance(json_schema, dict):
                    raise ValueError("JSON schema must be a JSON object, got str.")

        except Exception as e:
            print(f"Error decoding JSON: {e}\n\n")
            raise RequestException(message=f"Invalid JSON schema: {e}\n\n")

        # Enhance schema for citations if enabled
        if self.enable_citations and json_schema:
            json_schema = self.citation_handler.enhance_schema_for_citations(json_schema)
            # Inline $ref/$defs to make schema explicit for response_format
            json_schema = self.citation_handler.inline_refs_for_response_format(json_schema)

        # Enrich schema for Anthropic to reduce optional parameters
        if model_provider == "anthropic" and json_schema:
            json_schema = self._enrich_schema_for_anthropic(json_schema)

        async def async_task(chunk_number, parsed_chunk, json_schema, prompt):

            try:
                # Add page markers for better LLM context during structured extraction
                content_to_extract = parsed_chunk.content

                # Add citation references if enabled and page_layouts provided
                if self.enable_citations and page_layouts:
                    # Get table_output_mode from request if available
                    table_output_mode = "markdown"  # default
                    if request and hasattr(request, "table_output_mode"):
                        table_output_mode = request.table_output_mode

                    # Filter page_layouts to only include elements relevant to this chunk
                    filtered_layouts = self._filter_page_layouts_for_chunk(
                        parsed_chunk, page_layouts, chunking_strategy
                    )

                    content_with_refs, has_citations = (
                        self.citation_handler.prepare_text_with_citations(
                            content_to_extract, filtered_layouts, table_output_mode
                        )
                    )

                    content_to_extract = content_with_refs

                elif should_add_page_markers(parsed_pages, actual_page_numbers, chunking_strategy):
                    if chunking_strategy == "page":
                        # Page-level chunking: Add marker for this specific page
                        content_to_extract = add_page_marker_to_content(
                            content_to_extract, parsed_chunk.page_number
                        )
                    elif chunking_strategy in [None, "none"]:
                        # Document-level chunking: Build content with individual page markers for each page
                        if pages and request and len(actual_page_numbers) > 1:
                            # Rebuild content with individual page markers
                            content_to_extract = build_document_content_with_page_markers(
                                pages, request
                            )

                full_initial_prompt = None
                if prompt:
                    if json_schema is not None:
                        full_initial_prompt = f"{prompt}\n\n{json_schema}\n\n{content_to_extract}"
                    else:
                        full_initial_prompt = f"{prompt}\n\n{content_to_extract}"
                else:
                    full_initial_prompt = None

                enhanced_prompt = (
                    self.citation_handler.add_citation_instructions(prompt)
                    if self.enable_citations
                    else prompt
                )

                if self.enable_citations:
                    try:
                        # Reconstruct the full enriched prompt matching the final model call content
                        if enhanced_prompt and json_schema is not None:
                            full_enriched_prompt = (
                                f"{enhanced_prompt}\n\n{json_schema}\n\n{content_to_extract}"
                            )
                        elif enhanced_prompt:
                            full_enriched_prompt = f"{enhanced_prompt}\n\n{content_to_extract}"
                        else:
                            full_enriched_prompt = None

                        if full_initial_prompt is None:
                            if json_schema is not None:
                                full_initial_prompt = (
                                    EXTRACT_PROMPT_OAI_STRICT_JSON_ADHERENCE.format(
                                        text=content_to_extract
                                    )
                                )
                            else:
                                full_initial_prompt = content_to_extract

                        if full_enriched_prompt is not None:
                            self.citation_handler.snapshot_prompts(
                                full_initial_prompt, full_enriched_prompt
                            )
                    except Exception:
                        pass

                # Time the individual extraction
                extraction_start = time.time()

                print("#####file mime_type in structured extraction: ", request.mime_type)

                # Check for dense tabular content that needs special chunking only in CSV or Excel inputs
                if (
                    request
                    and request.mime_type in ["text/table", "text/csv"]
                    and json_schema
                    and should_split_dense_table(json_schema, content_to_extract, model_provider)
                ):
                    print(
                        f"Dense tabular content detected for chunk {chunk_number}, using table splitting approach"
                    )
                    model_response = await self._extract_json_with_dense_table_splitting(
                        model_provider=model_provider,
                        text=content_to_extract,
                        json_schema=json_schema,
                        prompt=enhanced_prompt,
                    )
                else:
                    # Regular extraction
                    model_response = await self._extract_json(
                        model_provider=model_provider,
                        text=content_to_extract,
                        json_schema=json_schema,
                        prompt=enhanced_prompt,
                    )

                extraction_time = time.time() - extraction_start
                print(f"Extraction for chunk {chunk_number} completed in {extraction_time:.2f}s")
            except RequestException as e:
                raise e
            except Exception as e:
                print(f"error calling model: {e}")

                # Check if this is a token limit error and provide user-friendly message
                token_limit_msg = _create_token_limit_error_message(str(e))
                if token_limit_msg:
                    raise RequestException(message=token_limit_msg)
                else:
                    raise RequestException(
                        message=("Structured extraction failed. " + _redact_provider_names(str(e)))
                    )

            # print(
            #     f"extracted JSON: {model_response.resp}, chunk number: {chunk_number}"
            # )

            try:
                json_result = model_response.resp

                if self.enable_citations:
                    # Get allowed citation filter based on chunking strategy
                    allowed_ref_ids, allowed_pages = self._get_allowed_citation_filter(
                        parsed_chunk, chunking_strategy
                    )

                    json_result = self.citation_handler.resolve_citations(
                        json_result,
                        allowed_ref_ids=allowed_ref_ids,
                        allowed_pages=allowed_pages,
                    )

                # Use actual_page_numbers if provided when no chunking, otherwise use page_number in chunks or chunk_number
                if len(parsed_pages) == 1:
                    # No chunking - one chunk for all pages
                    page_nums = actual_page_numbers
                else:
                    # Per-page or per-section chunking - use chunk's own page numbers (supports cross-page chunks)
                    page_nums = (
                        parsed_chunk.page_numbers
                        if parsed_chunk.page_numbers
                        else [parsed_chunk.page_number]
                    )

                structured_data = StructuredData(
                    data=json_result,
                    page_numbers=page_nums,
                    schema_name=schema_name,
                )

                # Return structured data along with token usage
                return (
                    structured_data,
                    model_response.input_tokens,
                    model_response.output_tokens,
                )

            except json.JSONDecodeError as e:
                print(f"error decoding JSON: {e}\n\n")
                raise RequestException(
                    message="Data extraction failed. Please try again or contact Tensorlake support with the trace ID of the job."
                )

        async def parse_all_pages():
            tasks = []
            try:
                for chunk_number, parsed_chunk in enumerate(parsed_pages):
                    tasks.append(
                        asyncio.create_task(
                            async_task(
                                chunk_number=chunk_number,
                                parsed_chunk=parsed_chunk,
                                json_schema=json_schema,
                                prompt=prompt,
                            )
                        )
                    )

                print(f"Awaiting tasks of len {len(tasks)}")

                return await asyncio.gather(*tasks)
            except Exception as e:
                print(f"Task failed with error: {e}")

                # Cancel all pending tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete their cleanup (finally blocks)
                # This ensures streams are properly closed before propagating the exception
                await asyncio.gather(*tasks, return_exceptions=True)
                print("All tasks cleaned up after cancellation")

                raise RequestException(message=str(e))

        try:
            results = asyncio.run(parse_all_pages())
        except Exception as e:
            raise e

        structured_data_pages = []
        total_input_tokens = 0
        total_output_tokens = 0

        for structured_data_page, input_tokens, output_tokens in results:
            structured_data_pages.append(structured_data_page)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        print(f"Total tokens used - Input: {total_input_tokens}, Output: {total_output_tokens}")
        return structured_data_pages, total_input_tokens, total_output_tokens

    @function(
        image=structured_extraction_image,
        timeout=30 * 60,  # 30 minutes
        cpu=2,
        memory=8,
        # output_encoder = "json"
        # The function is not using /tmp disk space, just reserve a small amount
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        max_containers=200,
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, result: ParseResult) -> ParsedDocumentRef:
        """
        This method now ONLY performs LLM-based structured extraction.
        Output formatting is handled by a separate OutputFormatterTask in the workflow.
        """
        print("=== LLM-BASED STRUCTURED EXTRACTION ONLY ===")

        # Start timing the overall extraction
        extraction_start_time = time.time()

        # If VLM already processed everything, skip LLM extraction
        if result.structured_outputs_by_page:
            print("VLM outputs detected, skipping LLM extraction")
            return format_final_output(result)

        structured_extraction_requests = result.request.structured_extraction_requests or []

        # If no structured extraction requests, return as-is
        if not structured_extraction_requests:
            print("No LLM-based structured extraction requests, returning as-is")
            return format_final_output(result)

        # Perform LLM-based structured extraction
        all_structured_data = []
        total_input_tokens = 0
        total_output_tokens = 0

        for structured_extraction_request in structured_extraction_requests:
            try:
                # Filter pages by classes if specified
                if structured_extraction_request.page_classes:
                    page_layouts = []
                    page_numbers = []
                    for page in result.document_layout.pages:
                        page_classes = page.page_class
                        if isinstance(page_classes, str):
                            page_classes = [page_classes]
                        if any(
                            page_class in structured_extraction_request.page_classes
                            for page_class in page_classes
                        ):
                            page_layouts.append(page)
                            page_numbers.append(page.page_number)
                else:
                    page_layouts = [page for page in result.document_layout.pages]
                    page_numbers = [page.page_number for page in page_layouts]

                chunking_strategy, start_patterns, end_patterns = extract_chunking_info(
                    structured_extraction_request.chunking_strategy
                )

                # Get pages for document-level chunking
                merged_tables = (
                    result.document_layout.merged_tables if result.request.table_merging else None
                )

                # Create chunks
                chunks = chunk_pages(
                    page_layouts,
                    chunking_strategy,
                    result,
                    start_patterns,
                    end_patterns,
                    merged_tables=merged_tables,
                )

                # Skip if no content
                if not chunks or all(not chunk.content.strip() for chunk in chunks):
                    print("No content found, skipping")
                    continue

                pages = document_layout_to_document(
                    page_layouts,
                    result.document_layout.scale_factor,
                    result.request.ignore_sections,
                    merged_tables=merged_tables,
                    chunking_strategy="none",
                )

                # Perform extraction
                structured_data, input_tokens, output_tokens = self.extract_data(
                    model_provider=structured_extraction_request.model_provider,
                    parsed_pages=chunks,
                    json_schema=structured_extraction_request.json_schema,
                    actual_page_numbers=page_numbers,
                    prompt=structured_extraction_request.prompt,
                    schema_name=structured_extraction_request.schema_name,
                    chunking_strategy=chunking_strategy,
                    pages=pages,
                    request=result.request,
                    page_layouts=page_layouts,  # pass page layouts for citation tracking
                    enable_citation=(
                        structured_extraction_request.enable_citation
                        if hasattr(structured_extraction_request, "enable_citation")
                        else False
                    ),  # Pass enable_citation flag
                )

                # Accumulate results
                all_structured_data.extend(structured_data)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            except Exception as e:
                print(f"Caught exception during structured extraction {str(e)}")
                raise RequestException(message=str(e))

        # Store LLM extraction results in the ParseResult for the output formatter
        # Convert StructuredData objects to the structured_outputs_by_page format
        if all_structured_data:
            if not result.structured_outputs_by_page:
                result.structured_outputs_by_page = {}

            for chunk_idx, data in enumerate(all_structured_data):
                # Create unique key that prevents chunks from overwriting each other
                # Use (page_numbers, chunk_idx) to ensure uniqueness
                if len(data.page_numbers) == 1:
                    # Single page: (page_number, chunk_index)
                    page_key = (data.page_numbers[0], chunk_idx)
                else:
                    # Multiple pages: ((page1, page2, ...), chunk_index)
                    page_key = (tuple(data.page_numbers), chunk_idx)

                if page_key not in result.structured_outputs_by_page:
                    result.structured_outputs_by_page[page_key] = {}
                result.structured_outputs_by_page[page_key][data.schema_name] = data.data

        # Update token usage in result
        if not result.usage:
            result.usage = Usage(
                pages_parsed=(
                    len(result.document_layout.pages)
                    if result.document_layout and result.document_layout.pages
                    else 0
                ),
                signature_detection=result.request.detect_signature,
            )

        # Add LLM tokens to existing usage
        existing_input_tokens = result.usage.extraction_input_tokens_used or 0
        existing_output_tokens = result.usage.extraction_output_tokens_used or 0

        result.usage.extraction_input_tokens_used = existing_input_tokens + total_input_tokens
        result.usage.extraction_output_tokens_used = existing_output_tokens + total_output_tokens

        # Calculate and print total extraction time
        extraction_time = time.time() - extraction_start_time
        print(
            f"LLM extraction completed. Added {total_input_tokens} input tokens and {total_output_tokens} output tokens"
        )
        print(f"✅ Structured extraction completed in {extraction_time:.2f}s")

        return format_final_output(result)
