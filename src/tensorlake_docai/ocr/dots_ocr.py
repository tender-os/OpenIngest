# SPDX-License-Identifier: Apache-2.0
import os

# Force offline mode for DotsOCR

from tensorlake.applications import function, Retries, cls, RequestContext
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.vlm.workflow_images import ocr_gpu_cuda_image
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.pipeline.routing import route_after_ocr, ocr_only_stop
from tensorlake_docai.postprocess.header_correction import correct_document_headers
from tensorlake_docai.pipeline.simple_page_creator import ImageDimensions
from tensorlake_docai.ocr.utils import (
    BatchProcessor,
    create_page_layout_from_dotsocr_elements,
    detect_consecutive_repetition,
    create_masked_image,
    merge_partial_outputs,
    scale_bbox_to_original_coordinates,
    create_page_elements_from_dotsocr_output,
)
from tensorlake_docai.postprocess.output_cleaner import OutputCleaner
from tensorlake_docai.models.layout_objects import PageLayout
from tensorlake_docai.prompts.dotsocr_prompts import DOTSOCR_LAYOUT_PROMPT
from tensorlake_docai.ocr.figure_ocr import OvisFigureOCRTask

SECRETS = [
    "OPENAI_API_KEY",
    "USE_AZURE_OPENAI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME",
]

# GPU-specific config.
# Env-configurable: when DotsOCR and Ovis figure-OCR run in the SAME process on ONE
# GPU (e.g. a single-container run_local_application deployment) they cannot each take
# 0.85 of VRAM. Lowered default (0.4) lets both vLLM engines co-reside on an 80GB GPU.
# Only affects KV-cache size (throughput), not model output.
OCR_GPU_MEMORY_UTILIZATION = float(os.getenv("OCR_GPU_MEMORY_UTILIZATION", "0.4"))
MEMORY_IN_GB = int(os.getenv("OCR_GPU_MEMORY_IN_GB", "32"))
GPU_MODELS = ["H100", "A100-80GB"]


# ---------------------------------------------------------------------------
# Module-global dots.ocr vLLM engine singleton.
#
# The vLLM engine is built ONCE per process and stored here. Every DotsOCRTask
# instance adopts this shared engine instead of constructing its own (a fresh
# build is a ~800s load), and the engine can be put to sleep / woken between
# requests (level-1 sleep frees VRAM while keeping the engine resident) so a
# co-resident model (e.g. Ovis figure OCR) can use the GPU in the meantime.
# ---------------------------------------------------------------------------
_DOTS_OCR_ENGINE = None
_DOTS_OCR_TOKENIZER = None
_DOTS_OCR_SAMPLING_PARAMS = None
_DOTS_OCR_MODEL_DIR = None


def _download_model_dir():
    """Download the DotsOCR1.5 model from Hugging Face Hub if needed.

    Module-level so it can be shared by the engine builder and patched in tests.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="rednote-hilab/dots.mocr")


def build_dots_ocr_engine(gpu_memory_utilization=None, enforce_eager=False):
    """Build the dots.ocr vLLM engine ONCE into module globals.

    Idempotent: returns immediately if the engine already exists. Carries the
    exact engine kwargs used by ``DotsOCRTask`` and additionally enables
    ``enable_sleep_mode`` so the engine can be slept/woken between requests.

    ``enforce_eager=True`` disables CUDA graph capture. The Modal GPU memory
    snapshot path uses this because captured CUDA graphs leave live GPU state
    that the snapshotter cannot capture cleanly; output is unchanged, decode is
    marginally slower. The normal (warm-pool) path leaves it False for speed.
    """
    global _DOTS_OCR_ENGINE, _DOTS_OCR_TOKENIZER, _DOTS_OCR_SAMPLING_PARAMS
    global _DOTS_OCR_MODEL_DIR

    if _DOTS_OCR_ENGINE is not None:
        return

    if gpu_memory_utilization is None:
        gpu_memory_utilization = OCR_GPU_MEMORY_UTILIZATION

    from vllm import SamplingParams, LLM
    from transformers import AutoTokenizer

    model_dir = _download_model_dir()
    print(f"Initializing DotsOCR1.5 from: {model_dir}")

    _DOTS_OCR_ENGINE = LLM(
        model=model_dir,
        trust_remote_code=True,
        download_dir=model_dir,
        dtype="bfloat16",
        max_model_len=28800,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=32768,
        enable_chunked_prefill=True,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
        enable_sleep_mode=True,
        enforce_eager=enforce_eager,
    )

    _DOTS_OCR_TOKENIZER = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    _DOTS_OCR_SAMPLING_PARAMS = SamplingParams(
        temperature=0.1,
        top_p=1.0,
        max_tokens=16384,
    )

    _DOTS_OCR_MODEL_DIR = model_dir


def sleep_dots_ocr_engine():
    """Put the shared dots.ocr engine to sleep (level 1: frees VRAM, keeps engine)."""
    if _DOTS_OCR_ENGINE is not None:
        _DOTS_OCR_ENGINE.sleep(level=1)


def wake_dots_ocr_engine():
    """Wake the shared dots.ocr engine after a sleep."""
    if _DOTS_OCR_ENGINE is not None:
        _DOTS_OCR_ENGINE.wake_up()


@cls()
class DotsOCRTask(BatchProcessor):
    def __init__(self):
        # Don't pass batch_size=None explicitly - let BatchProcessor handle it
        super().__init__(memory_gb=MEMORY_IN_GB / 8)
        # GPU-only task - initialize components and prepare for model loading

        # Lazy initialization attributes
        self.model_dir = None
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None

    def _initialize_local_model(self):
        """Adopt the module-global vLLM engine (build it once if needed)."""
        if self.llm is not None:
            return

        print("🔄 Initializing DotsOCR1.5 model for GPU processing...")

        import gc
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(
                f"GPU memory before model load: {free_mem / 1024**3:.2f} GiB free / {total_mem / 1024**3:.2f} GiB total"
            )

        # Build the shared engine once into module globals, then adopt it. This
        # avoids a second ~800s engine load per task instance.
        build_dots_ocr_engine()

        self.llm = _DOTS_OCR_ENGINE
        self.tokenizer = _DOTS_OCR_TOKENIZER
        self.sampling_params = _DOTS_OCR_SAMPLING_PARAMS
        self.model_dir = _DOTS_OCR_MODEL_DIR

        print("✅ DotsOCR1.5 model initialized successfully (GPU)")

    def _download_model(self):
        """Download the DotsOCR1.5 model from Hugging Face Hub if needed."""
        return _download_model_dir()

    def _check_repetition_during_generation(self, current_text: str, context: str = "") -> bool:
        """
        Check if current generation shows consecutive repetition patterns.

        Args:
            current_text: Current generated text
            context: Context string for logging (e.g., "Page 2")

        Returns:
            True if repetition detected, False otherwise
        """
        # Need at least 2000 characters to check
        if len(current_text) < 2000:
            return False

        if detect_consecutive_repetition(current_text):
            print(f"⚠️ {context}: Consecutive repetition detected in last 2000 characters")
            return True

        return False

    def _generate_with_repetition_detection(
        self, img, formatted_prompt, sampling_params, request_id: str, context: str = ""
    ) -> tuple:
        """
        Run vLLM generation with post-hoc repetition detection.

        Args:
            img: PIL Image for generation
            formatted_prompt: Formatted prompt string
            sampling_params: vLLM SamplingParams
            request_id: Unique request ID (unused with sync engine, kept for API compat)
            context: Context string for logging (e.g., "Page 2")

        Returns:
            Tuple of (result_text, stopped_early)
        """
        from vllm import TextPrompt

        outputs = self.llm.generate(
            TextPrompt(prompt=formatted_prompt, multi_modal_data={"image": img}),
            sampling_params=sampling_params,
        )

        result_text = outputs[0].outputs[0].text if outputs else ""

        # Check for repetition in the completed output
        stopped_early = self._check_repetition_during_generation(result_text, context)

        return result_text, stopped_early

    def _attempt_recovery_with_crop(
        self,
        original_text: str,
        img,
        page_num: int,
        base_request_id: str,
        formatted_prompt: str,
        sampling_params,
        pdf_sizes: dict,
    ):
        """
        Recovery strategy: Iteratively mask and retry until no repetition detected.

        Args:
            original_text: Partial output from first generation (layout_all with content)
            img: Original PIL Image
            page_num: Page number
            base_request_id: Base request ID for generating retry ID
            formatted_prompt: Formatted prompt for generation
            sampling_params: vLLM SamplingParams
            pdf_sizes: Dict of page numbers to PDF sizes

        Returns:
            Merged PageLayout if successful, None otherwise
        """
        print(f"🔄 Page {page_num}: Attempting iterative recovery with masking")

        cleaner = OutputCleaner()

        # Accumulate all partial outputs from iterations
        accumulated_elements = []
        max_iterations = 5
        iteration = 0

        # Start with original partial output (we know it had repetition)
        partial_elements = cleaner.clean_model_output(original_text)

        # Mark last element as Picture - it had repetitive generation
        if partial_elements:
            partial_elements[-1]["category"] = "Picture"
            partial_elements[-1]["text"] = ""
            print(
                f"   Iteration 0: Marked last element (bbox {partial_elements[-1].get('bbox')}) as Picture for Ovis"
            )

        accumulated_elements.extend(partial_elements)
        print(f"   Iteration 0 (original): {len(partial_elements)} elements")

        # Iteratively mask and retry until no repetition or max iterations
        while iteration < max_iterations:
            iteration += 1

            # Create masked image from all accumulated elements
            masked_img, masked_count = create_masked_image(img, accumulated_elements)
            print(f"   Iteration {iteration}: Masked {masked_count} regions, retrying...")

            # Run layout_all on masked image
            retry_request_id = f"{base_request_id}_iter{iteration}"
            retry_text, stopped_early = self._generate_with_repetition_detection(
                masked_img,
                formatted_prompt,
                sampling_params,
                retry_request_id,
                context=f"Page {page_num} Iter{iteration}",
            )

            if not retry_text:
                print(f"   ⚠️ Iteration {iteration} failed - using accumulated results")
                break

            # Parse new output
            new_elements = cleaner.clean_model_output(retry_text)

            # Calculate total content length BEFORE clearing text
            total_content_length = sum(len(elem.get("text", "")) for elem in new_elements)

            # Mark last element as Picture if repetition detected
            marked_idx = None
            if stopped_early and new_elements:
                new_elements[-1]["category"] = "Picture"
                new_elements[-1]["text"] = ""  # Clear after calculating content length
                marked_idx = len(new_elements) - 1
                print(
                    f"   Iteration {iteration}: Marked last element (bbox {new_elements[-1].get('bbox')}) as Picture for Ovis"
                )

            # Process elements: filter only spurious full-page figures, keep intentional Pictures
            valid_elements = []
            spurious_count = 0
            for idx, elem in enumerate(new_elements):
                category = elem.get("category", "")

                # Skip spurious full-page figures from late iterations (DotsOCR sees mostly blank)
                # But DON'T skip the Picture we intentionally marked
                if category == "Picture" and iteration >= 3 and idx != marked_idx:
                    bbox = elem.get("bbox", [])
                    if len(bbox) == 4:
                        elem_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        page_area = img.width * img.height
                        coverage = elem_area / page_area if page_area > 0 else 0
                        if coverage > 0.5:
                            spurious_count += 1
                            continue

                valid_elements.append(elem)

            if spurious_count > 0:
                print(
                    f"   Iteration {iteration}: Skipped {spurious_count} spurious full-page figures"
                )

            print(f"   Iteration {iteration}: {len(valid_elements)} elements total")

            # Stop if iteration generated very little content (< 100 chars total)
            # This usually means we've masked everything and model sees blank page
            if total_content_length < 100 and iteration > 1:
                print(
                    f"   ⚠️ Iteration {iteration}: Generated very little content ({total_content_length} chars), stopping"
                )
                break

            # Add valid elements to accumulated list
            accumulated_elements.extend(valid_elements)

            # Check if we're done
            if not stopped_early:
                print(f"   ✓ Iteration {iteration}: No repetition detected, success!")
                break
            else:
                print(f"   ⚠️ Iteration {iteration}: Still has repetition, continuing...")

        # Merge all accumulated elements (remove duplicates, keep best content)
        merged_elements = merge_partial_outputs(accumulated_elements)

        # Count final element types
        figure_count = sum(1 for elem in merged_elements if elem.get("category") == "Picture")
        print(f"   Final merge: {len(merged_elements)} elements ({figure_count} figures for Ovis)")

        pdf_size = pdf_sizes.get(page_num, img.size)

        # Scale from image coordinates to PDF coordinates
        scale_bbox_to_original_coordinates(merged_elements, img.size, pdf_size)

        # Create page elements
        page_elements = create_page_elements_from_dotsocr_output(merged_elements, page_num)

        # Create final layout
        page_layout = PageLayout(
            elements=page_elements,
            shape=(int(pdf_size[0]), int(pdf_size[1])),
            page_number=page_num,
            page_dimensions={"width": int(pdf_size[0]), "height": int(pdf_size[1])},
        )

        return page_layout

    async def process_batch(self, processing_batch, batch_number):
        """
        Process a batch of page images using GPU inference.
        Implementation of BatchProcessor.process_batch()
        Works with pre-loaded images from get_images_generator.

        Note: Must be async because BatchProcessor.run_batch_processing awaits it,
        but uses sync vLLM LLM engine internally
        """
        import time
        import uuid
        from datetime import datetime

        # Initialize local model if not already done
        self._initialize_local_model()

        # Extract images from processing batch (payload is list of PIL images)
        page_numbers = processing_batch.page_numbers
        page_images = processing_batch.payload  # List of PIL images
        original_sizes = processing_batch.original_sizes  # Original PDF/image sizes

        print(f"\n=== Processing batch {batch_number}: {len(page_images)} pages ===")

        # Prepare prompts for DotsOCR
        prompt_str = f"<|img|><|imgpad|><|endofimg|>{DOTSOCR_LAYOUT_PROMPT}"
        messages = [{"role": "user", "content": prompt_str}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Build images_by_page dict for figure processing
        images_by_page = {pn: img for pn, img in zip(page_numbers, page_images)}

        # Get original page sizes from the processing batch (these are the original PDF/image dimensions)
        pdf_sizes = {
            pn: original_sizes.get(pn, (img.width, img.height))
            for pn, img in images_by_page.items()
        }

        # Batch all pages in a single llm.generate() call for efficiency
        t0 = time.time()
        from vllm import TextPrompt

        prompts = [
            TextPrompt(prompt=formatted_prompt, multi_modal_data={"image": img})
            for img in page_images
        ]

        try:
            batch_outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        except Exception as e:
            raise RequestException(
                message=f"Failed to process batch {batch_number}: {str(e)}. This could be due to model issues or GPU memory constraints."
            )

        # Post-process each page: check repetition, attempt recovery if needed
        layouts = []
        for page_idx, (output, img, page_num) in enumerate(
            zip(batch_outputs, page_images, page_numbers)
        ):
            try:
                result_text = output.outputs[0].text if output.outputs else ""
                request_id = f"dots_ocr_page_{page_num}_{page_idx}_{uuid.uuid4().hex[:8]}"

                # Check for repetition in the completed output
                stopped_early = self._check_repetition_during_generation(
                    result_text, context=f"Page {page_num}"
                )

                # Handle repetition with masking and re-inference
                if stopped_early:
                    merged_layout = self._attempt_recovery_with_crop(
                        result_text,
                        img,
                        page_num,
                        request_id,
                        formatted_prompt,
                        self.sampling_params,
                        pdf_sizes,
                    )
                    if merged_layout:
                        layouts.append(merged_layout)
                        continue
                    # If recovery failed, fall through to create layout from partial results

                # Create layout from the result (normal path or fallback)
                layout = create_page_layout_from_dotsocr_elements(
                    page_num=page_num,
                    dotsocr_elements=result_text,
                    image_size=(img.width, img.height),
                    pdf_size=pdf_sizes.get(page_num, (img.width, img.height)),
                )
                layouts.append(layout)
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                import traceback

                traceback.print_exc()
                raise RequestException(
                    message=f"Failed to process page {page_num}: {str(e)}. This could be due to model issues or GPU memory constraints."
                )

        dt = time.time() - t0
        end_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{end_time}] DotsOCR1.5 GPU timer: {dt:.2f}s for {len(page_images)} pages")

        # Note: Figure processing will be handled separately via routing
        # after all pages are processed

        return layouts

    @function(
        image=ocr_gpu_cuda_image,
        timeout=30 * 60,
        cpu=4,
        memory=MEMORY_IN_GB,
        ephemeral_disk=25,
        gpu=GPU_MODELS,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        min_containers=1,
        max_containers=1,
    )
    def run(self, parse_result: ParseResult) -> ParseResult:

        import torch

        if not torch.cuda.is_available():
            raise RequestException(
                message=(
                    "ocr_model='dots-ocr' requires a CUDA-equipped worker, but no GPU "
                    "is available. Pick a cloud OCR backend (azure-di, textract, gemini) "
                    "or run on a GPU host."
                )
            )

        print(f"Start OCR inference with DotsOCR1.5, model directory: {self.model_dir}")

        # Store parse_result for access in process_batch
        ctx: RequestContext = RequestContext.get()
        self._parse_result = parse_result

        image_dimensions = ImageDimensions(
            min_pixels=3136,
            max_pixels=11289600,
            target_dpi=200,
            upgrade_image_dpi=True,
        )

        # Use the BatchProcessor's run_batch_processing method
        parse_result = self.run_batch_processing(ctx, parse_result, image_dimensions)

        # Optional: header correction
        if (
            hasattr(parse_result.request, "xpage_header_detection")
            and parse_result.request.xpage_header_detection
        ):
            try:
                parse_result = correct_document_headers(
                    parse_result, api_key=os.getenv("OPENAI_API_KEY")
                )
            except RequestException:
                raise
            except Exception as e:
                print(f"[OP-GPU] Header correction skipped: {e}")

        # Check if we need figure OCR processing (route to separate GPU)
        has_figures = any(
            element.fragment_type == PageFragmentType.FIGURE
            for page in parse_result.document_layout.pages
            for element in page.elements
        )

        if has_figures:
            # Two-GPU split (wrapper): when defer_ovis is set, STOP after dots-ocr
            # without co-loading Ovis. The figure elements stay marked in the
            # ParseResult; the wrapper runs OvisFigureOCRTask in its own GPU container
            # via run_ovis_app. ocr_only_stop returns the ParseResult only when
            # request.ocr_only is set (always true on the split path).
            if getattr(parse_result.request, "defer_ovis", False):
                stopped = ocr_only_stop(parse_result, "DotsOCRTask (defer Ovis → separate GPU)")
                if stopped is not None:
                    return stopped
            print("🔀 DotsOCRTask → OvisFigureOCRTask (separate GPU)")
            return OvisFigureOCRTask().run.future(parse_result)

        return route_after_ocr(parse_result, log_prefix="DotsOCRTask", dots_ocr=True)
