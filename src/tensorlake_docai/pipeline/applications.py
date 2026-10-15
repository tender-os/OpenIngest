# SPDX-License-Identifier: Apache-2.0
"""Split entrypoints for the GPU/CPU throughput optimization (plan T1).

The single-container pipeline runs convert → OCR → classify/extract/format in one
process, holding a GPU through the (CPU/API) post-OCR steps. To isolate the GPU to
OCR only, we split the existing DAG at the OCR boundary:

* **OCR stage** — reuse ``normalize_file_type_and_upload`` with ``ocr_only=True``
  (see :func:`build_ocr_only_request`). ``route_after_ocr`` short-circuits and the
  app returns the OCR'd :class:`ParseResult` instead of dispatching the post-OCR DAG.
  Run this on the GPU function; the A100 is released the instant OCR returns.
* **Post-OCR stage** — :func:`resume_post_ocr_app` re-injects that ``ParseResult``
  and resumes from ``route_after_ocr`` (TableMerging → VLM classify/summarize →
  StructuredExtraction → format_final_output). Run this on a CPU function; it holds
  the LLM/API waits, not the GPU.

The intermediate ``ParseResult`` is plain Pydantic (verified JSON-lossless in
``tests/test_split_roundtrip.py``), so it crosses the function boundary as a dict.
"""

from tensorlake.applications import application, function, Retries

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.pipeline.routing import route_after_ocr
from tensorlake_docai.vlm.workflow_images import file_convertion_image, ocr_gpu_cuda_image

SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
]


def build_ocr_only_request(raw_request: dict) -> dict:
    """Return a copy of the request flagged so the pipeline stops after OCR.

    The GPU stage runs ``normalize_file_type_and_upload`` with this request; the
    OCR backend fills ``ocr_text`` and ``route_after_ocr`` returns the ParseResult.
    """
    return {**raw_request, "ocr_only": True}


def build_dots_only_request(raw_request: dict) -> dict:
    """OCR-only request that ALSO defers Ovis (two-GPU split).

    The dots-ocr GPU stage runs with this and stops after dots — figure-bearing docs
    keep their figure markers but do NOT co-load Ovis. Figure OCR then runs in its own
    GPU container via :func:`run_ovis_app`.
    """
    return {**raw_request, "ocr_only": True, "defer_ovis": True}


@application()
@function(
    description=(
        "Resume the parse pipeline from a completed (OCR'd) ParseResult: TableMerging, "
        "VLM classification/summarization, StructuredExtraction, and final formatting. "
        "Runs on CPU — no GPU is held during the LLM/API steps."
    ),
    image=file_convertion_image,
    secrets=SECRETS,
    timeout=30 * 60,
    cpu=2,
    memory=4,
    retries=Retries(max_retries=2),
)
def resume_post_ocr_app(parse_result: dict) -> dict:
    """Resume the post-OCR DAG from a serialized ParseResult.

    Args:
        parse_result: ``ParseResult.model_dump()`` produced by the OCR-only stage.

    Returns:
        The OpenIngest document dict (``{"document": ..., "usage": ...}``), identical
        to a single-call ``normalize_file_type_and_upload`` run.
    """
    pr = ParseResult.model_validate(parse_result)
    # Clear the OCR-only flag so route_after_ocr resumes the DAG instead of
    # short-circuiting again.
    pr.request.ocr_only = False
    # The OCR-only seam base64-encoded the raw PDF bytes so they survived the JSON
    # output boundary; decode them back so the VLM steps can render page images.
    fb = pr.request.file_bytes
    if isinstance(fb, str) and fb:
        import base64

        try:
            pr.request.file_bytes = base64.b64decode(fb)
        except Exception:
            pass  # not base64 (e.g. caller-supplied text); leave as-is
    return route_after_ocr(
        pr,
        log_prefix="RESUME_POST_OCR",
        dots_ocr=(pr.request.ocr_model == "dots-ocr"),
    )


@application()
@function(
    description=(
        "Run ONLY Ovis figure OCR on a dots-ocr'd ParseResult, in an isolated GPU "
        "container (two-GPU split). The dots stage deferred Ovis (defer_ovis=True); "
        "this runs it alone so the two models never co-load on one GPU."
    ),
    image=ocr_gpu_cuda_image,
    secrets=SECRETS,
    timeout=30 * 60,
    cpu=2,
    memory=24,  # Ovis alone needs ~24 GB (OVIS_MEMORY_IN_GB)
    ephemeral_disk=40,
    gpu=["H100", "A100-80GB"],
    retries=Retries(max_retries=2),
)
def run_ovis_app(parse_result: dict) -> ParseResult | dict:
    """Run Ovis figure OCR in isolation on a dots-ocr'd ParseResult, then stop.

    The dots stage (``defer_ovis=True``) returned the ParseResult with figure regions
    marked but Ovis not run. We run ONLY ``OvisFigureOCRTask`` here; ``ocr_only`` stays
    True so Ovis stops after figure OCR and the post-OCR DAG resumes separately on CPU
    via :func:`resume_post_ocr_app`.

    Args:
        parse_result: ``ParseResult.model_dump()`` from the dots-only stage.

    Returns:
        The same ParseResult with figure elements OCR'd by Ovis (base64 file_bytes
        preserved across the JSON boundary, as in the OCR-only seam).
    """
    # Lazy import: OvisFigureOCRTask pulls heavy GPU deps; keep applications.py
    # importable in the CPU-only post-OCR container.
    from tensorlake_docai.ocr.figure_ocr import OvisFigureOCRTask

    pr = ParseResult.model_validate(parse_result)
    # The dots seam base64-encoded the raw PDF bytes; Ovis renders figure crops from
    # them, so decode back to bytes (ocr_only_stop re-encodes on the way out).
    fb = pr.request.file_bytes
    if isinstance(fb, str) and fb:
        import base64

        try:
            pr.request.file_bytes = base64.b64decode(fb)
        except Exception:
            pass
    return OvisFigureOCRTask().run.future(pr)
