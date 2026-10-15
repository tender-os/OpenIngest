# SPDX-License-Identifier: Apache-2.0
"""Langfuse tracing for the OpenAI calls, grouped per parse."""

import os
import re
from contextlib import contextmanager
from functools import cache
from typing import Iterator, Optional
from urllib.parse import urlsplit

from langfuse import Langfuse, propagate_attributes
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

INLINE_IMAGE = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=]+")
IMAGE_PLACEHOLDER = "[image omitted, rendered from metadata.source_file]"

# ponytail: process-wide, correct while a container serves one parse at a time (Modal's
# default); pass the context through the call chain if inputs ever run concurrently.
_current_parse: dict[str, str] = {}


def strip_inline_images(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    """Replace base64 images in exported span attributes with a placeholder."""
    patches = {}
    for identifier, span in params.spans.items():
        stripped = {
            key: INLINE_IMAGE.sub(IMAGE_PLACEHOLDER, value)
            for key, value in span.attributes.items()
            if isinstance(value, str) and ";base64," in value
        }
        if stripped:
            patches[identifier] = OtelSpanPatch(set_attributes=stripped)
    return MaskOtelSpansResult(span_patches=patches)


@cache
def langfuse_client() -> Langfuse:
    """Langfuse client that never uploads page images; disabled without LANGFUSE_PUBLIC_KEY."""
    os.environ.setdefault("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    return Langfuse(mask_otel_spans=strip_inline_images)


def unsigned_url(url: Optional[str]) -> str:
    """The URL without its query string, so a signed link never reaches the trace."""
    if not url:
        return ""
    return urlsplit(url)._replace(query="", fragment="").geturl()


@contextmanager
def traced_parse(
    *, parse_id: str, stage: str, labels: Optional[dict] = None, file_url: Optional[str] = None
) -> Iterator[None]:
    """Attribute every traced call inside to one parse stage, then flush."""
    langfuse_client()
    labels = labels or {}
    _current_parse.update(
        parse_id=parse_id,
        stage=stage,
        source_file=unsigned_url(file_url),
        organization_id=str(labels.get("organization_id", "")),
        tender_id=str(labels.get("tender_id", "")),
    )
    try:
        yield
    finally:
        langfuse_client().flush()
        _current_parse.clear()


@contextmanager
def traced_call(job_type: str) -> Iterator[None]:
    """Tag every observation created inside with the current parse and job type."""
    langfuse_client()
    metadata = {key: value for key, value in _current_parse.items() if value}
    with propagate_attributes(
        session_id=_current_parse.get("parse_id"),
        tags=["document-parsing", job_type, _current_parse.get("stage", "local")],
        metadata=metadata,
        trace_name=job_type,
    ):
        yield
