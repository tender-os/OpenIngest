# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Gemini Header Correction Service
Service for correcting document header hierarchy levels using Google's Gemini AI.
"""

import json
import time
from typing import List, Dict, Tuple, Optional
import re

from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.models.layout_objects import PageLayoutElement
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.providers.model_provider_utils import get_openai_sync_client_and_model
from tensorlake.applications import RequestError as RequestException

# OpenAI API Configuration (used when USE_AZURE_OPENAI != "true";
# for Azure, the deployment name from AZURE_OPENAI_MODEL_DEPLOYMENT_NAME is used instead)
OPENAI_HEADER_CORRECTION_MODEL_NAME = "gpt-6-luna"


def correct_document_headers(
    parse_result: ParseResult, api_key: Optional[str] = None
) -> ParseResult:
    """
    Correct header hierarchy levels in a ParseResult using an LLM.

    Args:
        parse_result: ParseResult object with document_layout
        api_key: Optional API key.

    Returns:
        ParseResult: The same object, potentially modified with corrected header levels
    """
    print("Header correction is enabled.")

    if not parse_result.document_layout or not parse_result.document_layout.pages:
        return parse_result

    # Initialize token tracking
    header_correction_input_tokens = 0
    header_correction_output_tokens = 0

    try:
        # Extract headers
        headers_json, header_map = _extract_headers(parse_result)
        if not header_map:
            return parse_result

        # Get corrections
        corrections, input_tokens, output_tokens = _get_openai_corrections(headers_json, api_key)
        header_correction_input_tokens += input_tokens
        header_correction_output_tokens += output_tokens

        if not corrections:
            return parse_result

        # Apply corrections
        applied = _apply_corrections(parse_result, corrections, header_map)
        print(f"✅ Applied {applied} header corrections")

        # Update token usage in parse_result
        _update_token_usage(
            parse_result, header_correction_input_tokens, header_correction_output_tokens
        )

        return parse_result

    except RequestException:
        raise
    except Exception as e:
        print(f"⚠️ Header correction failed: {e}")
        # Unknown error
        raise RequestException(message="Header correction failed.") from e


def _extract_headers(
    parse_result: ParseResult,
) -> Tuple[str, Dict[str, Tuple[int, int, PageLayoutElement]]]:
    """Extract headers and create mapping for corrections."""
    header_info = []
    header_map = {}

    for page_idx, page in enumerate(parse_result.document_layout.pages):
        page_headers = []
        sorted_elements = sorted(page.elements, key=lambda x: x.reading_order)

        for element_idx, element in enumerate(sorted_elements):
            if element.fragment_type in [PageFragmentType.SECTION_HEADER, PageFragmentType.TITLE]:
                header_text = element.ocr_text.strip()
                if not header_text:
                    continue
                # Normalize markdown headers only when '#' is followed by space
                if re.match(r"^#{1,6}\s", header_text):
                    header_text = re.sub(r"^#{1,6}\s+", "", header_text)
                    element.ocr_text = header_text

                # Get content that follows this header (next 2-3 elements for context)
                following_content = []
                for next_idx in range(element_idx + 1, min(element_idx + 4, len(sorted_elements))):
                    next_element = sorted_elements[next_idx]
                    if next_element.fragment_type in [
                        PageFragmentType.SECTION_HEADER,
                        PageFragmentType.TITLE,
                    ]:
                        break  # Stop at next header
                    if next_element.ocr_text and next_element.ocr_text.strip():
                        following_content.append(next_element.ocr_text.strip()[:200])

                content_preview = " ".join(following_content)[:500]

                # Use a position-based key that is unique and language-agnostic
                header_key = f"p{page_idx}_h{len(header_map)}"

                header_map[header_key] = (page_idx, element_idx, element)
                page_headers.append(
                    {
                        "text": header_text,
                        "current_level": element.hierarchy_level or 1,
                        "type": (
                            "title"
                            if element.fragment_type == PageFragmentType.TITLE
                            else "section_header"
                        ),
                        "key": header_key,
                        "content_preview": content_preview,
                    }
                )

        if page_headers:
            header_info.append({"page": page_idx + 1, "headers": page_headers})

    document_structure = {"pages": header_info}
    return json.dumps(document_structure, indent=2), header_map


def _get_openai_corrections(
    headers_json: str, api_key: Optional[str] = None
) -> Tuple[List[Dict], int, int]:
    try:
        client, model_name = get_openai_sync_client_and_model(
            api_key=api_key, default_model=OPENAI_HEADER_CORRECTION_MODEL_NAME
        )

        correction_schema = {
            "name": "HeaderCorrections",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "current_level": {"type": "integer"},
                                "corrected_level": {"type": "integer"},
                            },
                            "required": ["key", "current_level", "corrected_level"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["corrections"],
                "additionalProperties": False,
            },
        }

        prompt = (
            "You are analyzing the complete header structure of a multi-page document.\n"
            "Assign correct hierarchy levels to ALL headers so the document structure is logical and consistent.\n\n"
            "HIERARCHY LEVELS:\n"
            "- Level 0: Document title — main title, appears once near the start, usually 5+ words, descriptive\n"
            "- Level 1: Major sections — top-level chapters or primary divisions\n"
            "- Level 2: Subsections within major sections\n"
            "- Level 3+: Sub-subsections and deeper nesting\n\n"
            "ANALYSIS APPROACH — follow these steps in order:\n"
            "1. Scan ALL headers across the entire document before assigning any levels.\n"
            "   Identify recurring patterns such as:\n"
            "   - Numbering schemes (e.g. '1.', '1.1', '1.1.1', 'A.', 'Appendix A')\n"
            "   - Capitalization (ALL CAPS often signals higher level; Title Case mid-level)\n"
            "   - Header length and style (short/directive vs. descriptive/verbose)\n"
            "2. Headers with the same format or numbering pattern must get the same level throughout the document.\n"
            "3. If numbering resets mid-document (e.g. Appendix, new chapter), treat each reset as a new Level 1 — not a level 0.\n"
            "4. Use content_preview to resolve ambiguity when header text alone is insufficient.\n\n"
            "CONSISTENCY RULES:\n"
            "- Structurally equivalent headers must share the same level even if they appear far apart.\n"
            "- Do not let a level drift or reset without a clear structural reason.\n"
            "- An inline label or callout (e.g. 'Note:', 'Example:') is not a section header — assign it a deeper level or treat it as the same level as its surrounding context.\n\n"
            "Return a correction entry for EVERY header in the input, even if the level is unchanged.\n\n"
            f"{headers_json}"
        )

        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a document structure expert. "
                                "You analyze header hierarchies across multi-page documents and assign consistent, "
                                "logically correct level numbers. "
                                "Always examine the full document pattern before assigning any individual levels."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": correction_schema,
                    },
                )
                break
            except Exception as api_error:
                if attempt == max_retries - 1:
                    raise api_error
                delay = base_delay * (2**attempt)
                print(f"OpenAI API attempt {attempt + 1} failed, retrying in {delay}s: {api_error}")
                time.sleep(delay)

        input_tokens = 0
        output_tokens = 0
        if completion.usage:
            input_tokens = completion.usage.prompt_tokens or 0
            output_tokens = completion.usage.completion_tokens or 0
            print(
                f"OpenAI header correction tokens - Input: {input_tokens}, Output: {output_tokens}"
            )

        response_text = (
            completion.choices[0].message.content.strip()
            if completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
            else ""
        )
        if not response_text:
            raise RequestException(message="header correction model returned empty response")

        try:
            result = json.loads(response_text)
            corrections = result.get("corrections", [])
            print(f"OpenAI suggested levels: {len(corrections)}")
            return corrections, input_tokens, output_tokens
        except json.JSONDecodeError as e:
            raise RequestException(
                message=f"Failed to parse header correction model JSON response: {str(e)}"
            )
    except RequestException:
        raise
    except Exception as e:
        raise RequestException(message=("Header correction failed. " + str(e))) from e


def _apply_corrections(
    parse_result: ParseResult,
    corrections: List[Dict],
    header_map: Dict[str, Tuple[int, int, PageLayoutElement]],
) -> int:
    """Apply corrections to PageLayoutElement objects."""
    applied = 0

    for correction in corrections:
        try:
            header_key = correction.get("key")
            corrected_level = correction.get("corrected_level")

            if not header_key or header_key not in header_map:
                continue

            if not isinstance(corrected_level, int) or corrected_level < 0 or corrected_level > 6:
                continue

            page_idx, element_idx, element = header_map[header_key]

            # Validate indices
            if page_idx >= len(parse_result.document_layout.pages) or element_idx >= len(
                parse_result.document_layout.pages[page_idx].elements
            ):
                continue

            # Apply correction
            old_level = element.hierarchy_level or 1
            element.hierarchy_level = corrected_level

            display_text = element.ocr_text.strip()[:80]
            print(f"📝 '{display_text}': Level {old_level} → {corrected_level}")
            applied += 1

        except Exception:
            continue

    return applied


def _update_token_usage(parse_result: ParseResult, input_tokens: int, output_tokens: int):
    """Update token usage in parse_result with header correction specific fields."""
    if parse_result.usage:
        # If usage object exists, set header correction tokens
        parse_result.usage.header_correction_input_tokens_used = input_tokens
        parse_result.usage.header_correction_output_tokens_used = output_tokens
    else:
        from tensorlake_docai.pipeline.api import Usage

        parse_result.usage = Usage(
            pages_parsed=0,
            header_correction_input_tokens_used=input_tokens,
            header_correction_output_tokens_used=output_tokens,
        )

    print(f"📊 Header correction token usage - Input: {input_tokens}, Output: {output_tokens}")
