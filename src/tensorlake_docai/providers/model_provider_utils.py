# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import sys
import time
from typing import Callable, List, Optional

from PIL import Image

from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.pipeline.routing import stream_with_timeout, update_progress_if_needed

# NOTE: if the keys are not found or un-set a new deployment/executor will fail.

GEMINI_TIMEOUT = 10 * 60  # 10 minutes in total
DEFAULT_VLM_TIMEOUT = 10 * 60  # 10 minutes in total
OPENAI_LLM_MODEL_NAME = os.environ.get(
    "OPENAI_LLM_MODEL_NAME", "gpt-6.1-sol"
)  # For structured extraction
OPENAI_VLM_MODEL_NAME = os.environ.get("OPENAI_VLM_MODEL_NAME", "gpt-6-luna")  # For _make_oai_call


def get_gemini_async_client_and_model(
    api_key: Optional[str] = None, model_name: str = "gemini-3.1-flash-lite"
):
    """
    Return (async Gemini client (.aio), model_name).

    Args:
        api_key: Optional API key. If None, uses "GEMINI_API_KEY" from env.
        model_name: Model name to use. Defaults to "gemini-3.1-flash-lite".

    Returns:
        Tuple of (client.aio, model_name)
    """
    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )
    return client.aio, model_name


AZURE_OPENAI_API_VERSION = "2025-04-01-preview"


def _load_azure_openai_config() -> Optional[tuple]:
    """Return (endpoint, api_key, deployment) when USE_AZURE_OPENAI is enabled, else None.

    Raises RequestException when enabled but any of the three required vars is missing.
    """
    if os.environ.get("USE_AZURE_OPENAI", "false").lower() != "true":
        return None

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME")

    if not endpoint or not api_key or not deployment:
        raise RequestException(
            message="Azure OpenAI configuration incomplete. Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"
        )
    return endpoint, api_key, deployment


def get_openai_client_and_model(
    api_key: Optional[str] = None, default_model: str = OPENAI_LLM_MODEL_NAME
):
    """
    Return (AsyncOpenAI client, model_name) for either Azure or regular OpenAI.

    Args:
        api_key: Optional API key for regular OpenAI. If None, uses "OPENAI_API_KEY" from env.
        default_model: Default model name for regular OpenAI (ignored for Azure). Defaults to OPENAI_LLM_MODEL_NAME.

    Returns:
        Tuple of (client, model_name) where:
        - client is AsyncAzureOpenAI for Azure or AsyncOpenAI for regular
        - model_name is the deployment name for Azure or default_model for regular

    Raises:
        RequestException if Azure is enabled but config is incomplete.
    """
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    azure = _load_azure_openai_config()
    if azure:
        endpoint, azure_api_key, deployment = azure
        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=azure_api_key,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        return client, deployment

    key = api_key or os.environ.get("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=key), default_model


def get_openai_sync_client_and_model(
    api_key: Optional[str] = None, default_model: str = OPENAI_LLM_MODEL_NAME
):
    """Sync counterpart of get_openai_client_and_model. See that function for semantics."""
    from openai import AzureOpenAI, OpenAI

    azure = _load_azure_openai_config()
    if azure:
        endpoint, azure_api_key, deployment = azure
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=azure_api_key,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        return client, deployment

    key = api_key or os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=key), default_model


# Global semaphore to limit concurrent API calls across all models
# Keep a per-loop semaphore to avoid "bound to different event loop" errors
def _get_api_semaphore():
    """Get or create a semaphore for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    if not hasattr(loop, "_vlm_api_semaphore"):
        loop._vlm_api_semaphore = asyncio.Semaphore(10)
    return loop._vlm_api_semaphore


async def _make_gemini_call(
    user_prompt,
    images,
    page_image=None,
    json_schema=None,
    job_type=None,
    timeout: Optional[int] = None,
    pdf_bytes=None,
    model_name: str = "gemini-3.1-flash-lite",
    system_instruction=None,
    config_overrides=None,
):
    import io

    from google.genai import types

    client, model_name = get_gemini_async_client_and_model(model_name=model_name)

    # preprocess the json schema
    schema = None
    if json_schema:
        schema = json.loads(json_schema)
        schema.pop("$schema", None)  # Safely remove $schema if exists

    # Base generation config with defaults
    generation_config = {
        "temperature": 0.2,  # reduce temperature to 0.2 to make the output more deterministic
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 64000,  # 8192, double the max tokens so it can handle some super dense tables
        "response_mime_type": (
            "application/json"
            if job_type in ["structured_extraction", "page_classification", "ocr"]
            else "text/plain"
        ),
        # "thinking_config": types.ThinkingConfig(thinking_budget=0)
    }

    # Apply config overrides if provided (for OCR more deterministic settings)
    if config_overrides:
        generation_config.update(config_overrides)

    # Add system instruction if provided
    if system_instruction:
        generation_config["system_instruction"] = system_instruction

    # Conditionally add the appropriate schema field based on job type
    if schema:
        if job_type in ["structured_extraction", "page_classification"]:
            generation_config["response_json_schema"] = schema
        else:
            generation_config["response_schema"] = schema

    if "thinking_config" not in generation_config:
        # turn off thinking for other tasks except page classification
        if job_type != "page_classification":
            generation_config["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            )
        # limit thinking budget to 2048 tokens for page classification
        # https://ai.google.dev/gemini-api/docs/thinking
        # I personally think page classification is a medium level task
        else:
            generation_config["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM
            )

    # Handle PDF vs image inputs
    if pdf_bytes:
        # Use PDF native support with media resolution
        generation_config["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        contents = [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            user_prompt,
        ]
    else:
        # Use high resolution for images
        generation_config["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
        contents = [user_prompt]
        for image in images:
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")  # Explicitly force PNG
            png_bytes = byte_io.getvalue()
            contents.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
        if page_image is not None:
            byte_io = io.BytesIO()
            page_image.save(byte_io, format="PNG")  # Explicitly force PNG
            png_bytes = byte_io.getvalue()
            contents.append(png_bytes)

    # Time the LLM request
    start_time = time.time()

    try:
        # Make async streaming request with activity timeout
        stream = await client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generation_config,
        )

        try:
            # Accumulate streamed response with activity-based timeout
            accumulated_text = ""
            input_tokens = 0
            output_tokens = 0
            activity_timeout = timeout or 120  # 2 minutes in seconds
            chunk_count = 0
            last_progress_update = start_time
            finish_reason = None
            consecutive_whitespace_len = 0

            # Process stream with timeout that resets on each chunk
            async for chunk in stream_with_timeout(stream, timeout_seconds=activity_timeout):
                chunk_count += 1

                # Safe access to text
                text_content = None
                try:
                    if chunk.text:
                        text_content = chunk.text
                except ValueError:
                    # chunk.text raises ValueError if no text is present (e.g. usage metadata only)
                    pass

                if text_content:
                    accumulated_text += text_content

                    # Check for hallucinated spaces loop
                    if text_content.isspace():
                        consecutive_whitespace_len += len(text_content)
                    else:
                        consecutive_whitespace_len = 0

                    if consecutive_whitespace_len > 2500:
                        print("Aborting stream due to excessive repetitive generation patterns).")
                        raise RequestException(
                            message="Stream aborted due to repetitive generation patterns. Aborting."
                        )

                # Update progress to keep function-level timeout alive
                last_progress_update = update_progress_if_needed(
                    chunk_count, len(accumulated_text), start_time, last_progress_update
                )

                # Extract token usage from chunks
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    prompt_token_count = getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0
                    candidates_token_count = (
                        getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0
                    )
                    total_token_count = getattr(chunk.usage_metadata, "total_token_count", 0) or 0

                    if total_token_count > 0 and (prompt_token_count + candidates_token_count) > 0:
                        # Gemini billing calculation
                        input_ratio = prompt_token_count / (
                            prompt_token_count + candidates_token_count
                        )
                        output_ratio = 1 - input_ratio

                        input_tokens = int(total_token_count * input_ratio)
                        output_tokens = int(total_token_count * output_ratio)
                    else:
                        # Fallback to direct counts if calculation fails
                        input_tokens = prompt_token_count
                        output_tokens = candidates_token_count

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

        # Warn if response is empty but chunks were received (might be valid)
        if not accumulated_text:
            print("WARNING: Gemini returned empty text despite receiving chunks")

        # Check if output was truncated due to token limit
        finish_reason_str = str(finish_reason).upper() if finish_reason else ""
        if "MAX_TOKENS" in finish_reason_str or output_tokens >= 63500:
            # Raise exception with token info that error handler can parse
            raise Exception(
                f"completion_tokens={output_tokens} prompt_tokens={input_tokens} total_tokens={input_tokens + output_tokens} length limit was reached. Finish reason: {finish_reason}"
            )

        # Parse JSON response if needed
        if job_type in ["structured_extraction", "page_classification", "ocr"]:
            try:
                _ = json.loads(accumulated_text)
            except json.JSONDecodeError as e:
                # If JSON parse fails and we're near token limit, it's likely truncation
                print(f"Finish reason: {finish_reason}", flush=True)

                if output_tokens >= 60000:
                    raise Exception(
                        f"completion_tokens={output_tokens} prompt_tokens={input_tokens} total_tokens={input_tokens + output_tokens} token limit. JSON parse error: {str(e)}"
                    )
                raise e

        return (
            (accumulated_text.strip() if accumulated_text else ""),
            input_tokens,
            output_tokens,
        )

    except asyncio.TimeoutError as e:
        error_message = f"VLM timed out due to inactivity: {str(e)}"
        print(f"Timeout error in Gemini streaming call: {error_message}")
        raise RequestException(message=error_message)

    except Exception as e:
        import traceback

        # Clean up the error message by removing verbose prefixes
        error_message = str(e)

        # Only print traceback for non-transient errors to reduce log noise
        if not any(
            indicator in error_message
            for indicator in [
                "503",
                "UNAVAILABLE",
                "overloaded",
                "Resource has been exhausted",
                "RESOURCE_EXHAUSTED",
            ]
        ):
            full_traceback = traceback.format_exc()
            print(full_traceback)

        print(f"Error in Gemini call: {e}")
        original_error = error_message  # Keep original for detection

        # Extract just the actual error message from Gemini's verbose format
        if "'message':" in error_message:
            try:
                import re

                # Extract the message content between quotes
                match = re.search(r"'message': '([^']+)'", error_message)
                if match:
                    error_message = match.group(1)
            except Exception:
                pass

        # Clean up the verbose Gemini prefix from each line
        if "GenerateContentRequest.generation_config.response_schema.properties" in error_message:
            # Split by lines and clean each line
            lines = error_message.split("\\n")
            cleaned_lines = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    cleaned_line = line.replace(
                        "GenerateContentRequest.generation_config.response_schema.properties",
                        "",
                    )
                    cleaned_line = cleaned_line.replace("* ", "").strip()
                    if cleaned_line:  # Only add non-empty lines
                        cleaned_lines.append(cleaned_line)
            error_message = "\n".join(cleaned_lines)

            # Add a marker to indicate this shouldn't be retried since it's a schema error
            error_message = "SCHEMA_ERROR: " + error_message

        # Mark server overload errors for special retry handling
        if any(
            indicator in original_error
            for indicator in [
                "503",
                "UNAVAILABLE",
                "overloaded",
                "Resource has been exhausted",
                "RESOURCE_EXHAUSTED",
            ]
        ):
            error_message = "SERVER_OVERLOAD: " + error_message

        raise RequestException(message=error_message)


async def _make_oai_call(
    user_prompt,
    images,
    page_image=None,
    json_schema=None,
    job_type=None,
    pdf_bytes=None,
):
    import base64
    from io import BytesIO

    schema = None
    if json_schema:
        schema = json.loads(json_schema)
        schema.update({"additionalProperties": False})

    # Build image list with base64 encoding
    image_urls = []
    for image in images:
        if image.width == 0 or image.height == 0:
            print(f"Skipping empty image ({image.width}x{image.height}) in _make_oai_call")
            continue
        buff = BytesIO()
        image.convert("RGB").save(buff, format="JPEG")
        b64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
        image_urls.append(f"data:image/jpeg;base64,{b64_img}")

    if page_image is not None:
        if page_image.width > 0 and page_image.height > 0:
            page_buff = BytesIO()
            page_image.convert("RGB").save(page_buff, format="JPEG")
            page_b64_img = base64.b64encode(page_buff.getvalue()).decode("utf-8")
            image_urls.append(f"data:image/jpeg;base64,{page_b64_img}")
        else:
            print(
                f"Skipping empty page_image ({page_image.width}x{page_image.height}) in _make_oai_call"
            )

    # Time the LLM request
    start_time = time.time()

    try:
        # Get configured OpenAI client and model name
        use_azure_openai = os.environ.get("USE_AZURE_OPENAI", "false").lower() == "true"
        client_context, model_name = get_openai_client_and_model(
            default_model=OPENAI_VLM_MODEL_NAME
        )

        async with client_context as oai_client:
            # Azure OpenAI uses chat.completions API
            chat_content = [{"type": "text", "text": user_prompt}]
            for img_url in image_urls:
                chat_content.append({"type": "image_url", "image_url": {"url": img_url}})

            # Prepare streaming parameters
            req_params = {
                "model": model_name,
                "max_completion_tokens": 8192,
                "messages": [{"role": "user", "content": chat_content}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            if job_type == "page_classification" and schema:
                req_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "page_classification",
                        "schema": schema,
                        "strict": True,
                    },
                }

            # Stream the response with timeout management
            accumulated_text = ""
            usage_data = None
            chunk_count = 0
            last_progress_update = start_time

            # Create stream and wrap in try/finally for cleanup
            stream = await oai_client.chat.completions.create(**req_params)
            try:
                async for chunk in stream_with_timeout(stream, timeout_seconds=120):
                    chunk_count += 1
                    # Process each chunk
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
                        if hasattr(close_result, "__await__"):
                            await close_result
                    except Exception as e:
                        print(f"WARNING: Failed to close OpenAI stream: {e}")

        llm_time = time.time() - start_time
        provider_name = "Azure OpenAI" if use_azure_openai else "OpenAI"
        print(f"{provider_name} streaming response time: {llm_time:.2f}s")

        # Extract token usage
        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0
        print(f"{provider_name} tokens used - Input: {input_tokens}, Output: {output_tokens}")

        # Check if we received no chunks at all (indicates a problem)
        if chunk_count == 0:
            raise Exception(f"{provider_name} returned no response chunks")

        # Warn if response is empty but chunks were received (might be valid)
        if not accumulated_text:
            print(f"WARNING: {provider_name} returned empty text despite receiving chunks")

        return (
            accumulated_text.strip() if accumulated_text else "",
            input_tokens,
            output_tokens,
        )

    except asyncio.TimeoutError as e:
        provider_name = "Azure OpenAI" if use_azure_openai else "OpenAI"
        error_message = f"{provider_name} VLM timed out due to inactivity: {str(e)}"
        print(f"Timeout error in {provider_name} streaming call: {error_message}")
        raise RequestException(message=error_message)

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"Error in OpenAI call: {e}")

        # Clean up the error message
        error_message = str(e)

        raise RequestException(message=f"LLM provider error: {error_message}")


async def _make_anthropic_call(
    user_prompt,
    images,
    page_image=None,
    json_schema=None,
    job_type=None,
    timeout: Optional[int] = None,
):
    import base64
    from io import BytesIO

    import anthropic

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    ANTHROPIC_CLIENT = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    if timeout is None:
        timeout = DEFAULT_VLM_TIMEOUT

    image_urls = []
    for image in images:
        buff = BytesIO()
        image.save(buff, format="JPEG")
        b64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
        image_urls.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64_img,
                },
            }
        )

    content = [
        *image_urls,
        {"type": "text", "text": user_prompt},
    ]

    if page_image is not None:
        page_buff = BytesIO()
        page_image.save(page_buff, format="JPEG")
        page_b64_img = base64.b64encode(page_buff.getvalue()).decode("utf-8")

        page_img_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": page_b64_img,
            },
        }
        content.insert(0, page_img_content)

    response = await ANTHROPIC_CLIENT.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_img,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
        timeout=timeout,
    )

    # Extract token usage
    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0
    print(f"Anthropic tokens used - Input: {input_tokens}, Output: {output_tokens}")

    return response.content[0].text, input_tokens, output_tokens


async def run_clients(
    user_prompt: str,
    images: List[Image.Image],
    models: List[Callable],
    page_image: Optional[Image.Image] = None,
    json_schema: Optional[str] = None,
    job_type: Optional[str] = None,
    timeout: Optional[int] = None,
    pdf_bytes: Optional[bytes] = None,
) -> tuple[str, int, int]:
    # Use configured timeout if none provided
    if timeout is None:
        timeout = DEFAULT_VLM_TIMEOUT

    # Use only the first model (should be gemini_call)
    # Later we will enable model selection
    # switch model usage from gemini to openAI for summarization and page classification tasks
    def _is_server_issue(error_msg: str) -> bool:
        """Check if error indicates a server issue requiring longer backoff."""
        error_lower = error_msg.lower()
        return (
            error_msg.startswith("SERVER_OVERLOAD:")
            or "timed out" in error_lower
            or any(indicator in error_lower for indicator in ["503", "unavailable", "overload"])
        )

    def _calculate_backoff_delay(attempt: int, is_server_issue: bool) -> int:
        """Calculate exponential backoff delay based on error type."""
        if is_server_issue:
            # Server issues: use longer backoff (10s, 20s, 40s, capped at 60s)
            return min(60, 10 * (2**attempt))
        else:
            # Regular errors: standard exponential backoff (1s, 2s, 4s)
            return 2**attempt

    model = models[0] if job_type in ["structured_extraction"] else models[1]
    model_name = model.__name__

    # Get or create semaphore for current event loop
    semaphore = _get_api_semaphore()

    retries = 3

    for attempt in range(retries):
        start_s = time.time()
        wait_start_s = time.time()
        call_start_s = None
        try:
            pdf_info = (
                f" with PDF ({len(pdf_bytes)} bytes)"
                if pdf_bytes
                else f" with {len(images)} images"
            )
            print(f"Calling model: {model_name}{pdf_info} (attempt {attempt + 1}/{retries})")

            # Use semaphore to limit concurrent API calls to 10
            if semaphore:
                async with semaphore:
                    call_start_s = time.time()
                    if model in [_make_oai_call, _make_gemini_call]:
                        result = await model(
                            user_prompt,
                            images,
                            page_image,
                            json_schema,
                            job_type,
                            pdf_bytes=pdf_bytes,
                        )
                    else:
                        result = await model(
                            user_prompt,
                            images,
                            page_image,
                            json_schema,
                            job_type,
                            timeout=timeout,
                            pdf_bytes=pdf_bytes,
                        )
            else:
                call_start_s = time.time()
                if model in [_make_oai_call, _make_gemini_call]:
                    result = await model(
                        user_prompt,
                        images,
                        page_image,
                        json_schema,
                        job_type,
                        pdf_bytes=pdf_bytes,
                    )
                else:
                    result = await model(
                        user_prompt,
                        images,
                        page_image,
                        json_schema,
                        job_type,
                        timeout=timeout,
                        pdf_bytes=pdf_bytes,
                    )

            # Handle the new return format with token usage
            if isinstance(result, tuple) and len(result) == 3:
                output_text, input_tokens, output_tokens = result
            else:
                output_text = result
                input_tokens = output_tokens = 0

            # Handle silent failure when model can't fulfill json schema
            if output_text is None:
                raise RequestException(
                    message="Unable to generate valid output with the given schema"
                )

            return output_text, input_tokens, output_tokens

        except RequestException as e:
            clean_error = str(e)
            print(
                f"RequestException calling model: {model_name} (attempt {attempt + 1}/{retries}): {clean_error}\n",
                file=sys.stderr,
            )

            # Don't retry schema errors - fail immediately
            if clean_error.startswith("SCHEMA_ERROR:"):
                raise RequestException(message=clean_error)

            if "length limit was reached" in clean_error:
                raise RequestException(message=clean_error)

            if "repetitive generation patterns" in clean_error:
                raise RequestException(message=clean_error)

            # If this was the last attempt, raise the exception
            if attempt == retries - 1:
                raise RequestException(message=clean_error)

            # Adaptive backoff and retry
            is_server_issue = _is_server_issue(clean_error)
            delay = _calculate_backoff_delay(attempt, is_server_issue)
            issue_type = "Server overload/timeout" if is_server_issue else "Error"
            print(f"{issue_type} detected. Waiting {delay}s before retry...")
            await asyncio.sleep(delay)

        except Exception as e:
            error_str = str(e)
            print(
                f"Error calling model: {model_name} (attempt {attempt + 1}/{retries}): {error_str}\n",
                file=sys.stderr,
            )

            if "length limit was reached" in error_str:
                raise RequestException(message=f"Failed to call VLM: {error_str}")

            if "Model hallucinating spaces" in error_str:
                raise RequestException(message=f"Failed to call VLM: {error_str}")

            # If this was the last attempt, raise as RequestException
            if attempt == retries - 1:
                raise RequestException(message=f"Failed to call VLM: {error_str}")

            # Adaptive backoff and retry
            is_server_issue = _is_server_issue(error_str)
            delay = _calculate_backoff_delay(attempt, is_server_issue)
            issue_type = "Server issue" if is_server_issue else "Error"
            print(f"{issue_type} detected. Waiting {delay}s before retry...")
            await asyncio.sleep(delay)

        finally:
            end_s = time.time()
            wait_time_s = (call_start_s - wait_start_s) if call_start_s else (end_s - wait_start_s)
            call_time_s = (end_s - call_start_s) if call_start_s else 0
            print(
                f"Model {model_name} attempt {attempt + 1} took: {end_s - start_s} s "
                f"(waited {wait_time_s:.2f}s, call {call_time_s:.2f}s)"
            )
