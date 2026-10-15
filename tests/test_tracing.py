# SPDX-License-Identifier: Apache-2.0
"""Tests for providers.tracing — image stripping, URL signing removal, OpenAI passthrough."""

import asyncio
import json
from types import SimpleNamespace

import httpx
from langfuse.openai import AsyncOpenAI

from tensorlake_docai.providers.tracing import (
    IMAGE_PLACEHOLDER,
    end_parse,
    start_parse,
    strip_inline_images,
    traced_call,
    unsigned_url,
)


def test_strip_inline_images_replaces_base64_and_leaves_text():
    message = (
        'see [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ+Zw=="}}]'
    )
    params = SimpleNamespace(
        spans={
            "with_image": SimpleNamespace(attributes={"input": message, "tokens": 12}),
            "text_only": SimpleNamespace(attributes={"input": "classify this page"}),
        }
    )

    result = strip_inline_images(params=params)

    assert list(result.span_patches) == ["with_image"]
    patched = result.span_patches["with_image"].set_attributes["input"]
    assert IMAGE_PLACEHOLDER in patched
    assert "base64" not in patched


def test_unsigned_url_drops_signature():
    signed = (
        "https://acc.r2.cloudflarestorage.com/tenders/a/b.pdf?X-Amz-Signature=abc&X-Amz-Expires=600"
    )

    assert unsigned_url(signed) == "https://acc.r2.cloudflarestorage.com/tenders/a/b.pdf"
    assert unsigned_url(None) == ""


def test_traced_openai_call_sends_no_langfuse_fields_to_openai():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-6-luna",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def call():
        client = AsyncOpenAI(
            api_key="sk-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        with traced_call("page_classification"):
            return await client.chat.completions.create(
                model="gpt-6-luna",
                messages=[{"role": "user", "content": "hi"}],
                name="page_classification",
            )

    start_parse(
        parse_id="fc-1",
        stage="post_ocr",
        labels={"organization_id": "org_1"},
        file_url="https://x/y.pdf?sig=1",
    )
    try:
        response = asyncio.run(call())
    finally:
        end_parse()

    assert response.choices[0].message.content == "ok"
    assert sorted(sent) == ["messages", "model"]
