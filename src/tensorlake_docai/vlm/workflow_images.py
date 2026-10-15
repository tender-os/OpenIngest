# SPDX-License-Identifier: Apache-2.0
from tensorlake.applications import Image

CUDA_BASE_IMAGE = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
VLLM_BASE_IMAGE = "vllm/vllm-openai:v0.13.0"

# vllm==0.13.0 requires pydantic>=2.12.0. Exact pin across all images ensures
# byte-identical serialization of shared Pydantic models (ParseResult etc.).
PYDANTIC_PIN = "pydantic==2.12.5"

table_merging_image = (
    Image(base_image="python:3.12-slim", name="documentai/table-merging")
    .run("apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*")
    .run("python -m pip install --upgrade pip wheel setuptools")
    .run("pip install google-genai==1.51.0")
    .run("pip install requests boto3")  # File download + S3 access
    .run("pip install pymupdf psutil markdownify")
    .run(f"pip install '{PYDANTIC_PIN}'")  # Data models
    .run("pip install pillow numpy jdeskew")
    .run("pip install pytesseract")
    .run("pip cache purge")
)

structured_extraction_image = (
    Image(base_image="python:3.12-slim", name="documentai/structured-extraction")
    .run("python -m pip install --upgrade pip wheel setuptools")
    .run("pip install openai anthropic==0.74.1 google-genai==1.51.0 langfuse==5.0.1")
    .run(f"pip install '{PYDANTIC_PIN}' markdownify")  # Data validation + markdown output
    .run("pip install requests boto3")  # File download + S3 access
    .run("pip install tiktoken==0.11.0 beautifulsoup4==4.13.4")
    .run("pip install pillow")  # imported by providers.model_provider_utils
    .run("pip cache purge")
)

vlm_extraction_image = (
    Image(base_image="python:3.12-slim", name="documentai/vlm-extraction")
    .run("apt-get update && apt-get install -y libxcb1 && rm -rf /var/lib/apt/lists/*")
    .run("python -m pip install --upgrade pip wheel setuptools")
    .run("pip install pillow numpy")
    .run("pip install psutil pymupdf")  # PDF handling
    .run("pip install opencv-python-headless")  # Headless OpenCV for jdeskew
    .run("pip install jdeskew==0.3.0 --no-deps")
    .run(f"pip install '{PYDANTIC_PIN}'")  # Data validation
    .run("pip install boto3 requests markdownify")  # AWS and HTTP
    .run("pip install openai==2.3.0 google-genai==1.51.0 langfuse==5.0.1")
    .run(
        "pip install amazon-textract-textractor"
    )  # Removed azure-ai-documentintelligence - not used
    .run("pip cache purge")
)

# CUDA-enabled image for GPU OCR task with full dependencies
# Uses vllm/vllm-openai as base so vllm is pre-installed; no need to pip install it.
ocr_gpu_cuda_image = (
    Image(base_image=VLLM_BASE_IMAGE, name="documentai/ocr-gpu-cuda")
    .env("DEBIAN_FRONTEND", "noninteractive")
    .run(
        "apt-get update && apt-get install -y libgl1 libglib2.0-0 git poppler-utils libzbar0 default-jdk && rm -rf /var/lib/apt/lists/*"
    )
    .run(
        f"pip install '{PYDANTIC_PIN}'"
    )  # enforce exact version after vllm's transitive resolution
    .run("pip install pyzbar")
    .run("pip install s3fs boto3 cryptography==46.0.5")
    .run("pip install qwen-vl-utils")
    .run("pip install markdownify zxing==1.0.3")
    .run("pip install pillow numpy pandas")
    .run("pip install pdf2image pypdf")
    .run("pip install opencv-python-headless")  # Headless OpenCV for jdeskew
    .run("pip install jdeskew==0.3.0 --no-deps")
    .run("pip install pymupdf")  # image preprocessing
    .run("pip cache purge")
)


file_convertion_image = (
    Image(base_image="python:3.12-slim", name="documentai/file-convert")
    .env("DEBIAN_FRONTEND", "noninteractive")
    .run(
        "apt-get update && apt-get install -y poppler-utils libreoffice libmagic1 openssl && rm -rf /var/lib/apt/lists/*"
    )
    .run("python -m pip install --upgrade pip wheel setuptools")
    .run("pip install python-magic==0.4.27 markdownify")
    .run("pip install requests boto3 google-genai==1.51.0")
    .run(f"pip install '{PYDANTIC_PIN}'")
    .run("pip install lxml==6.0.2 beautifulsoup4 cryptography==46.0.5 python-docx chardet")
    .run("pip install pillow pandas openpyxl xlrd PyMuPDF pypdf pdf2image")
    .run("pip cache purge")
)

simple_page_creator_image = (
    Image(base_image="python:3.12-slim", name="documentai/simple-page-creator")
    .run("apt-get update && apt-get install -y libxcb1 && rm -rf /var/lib/apt/lists/*")
    .run("python -m pip install --upgrade pip wheel setuptools")
    .run("pip install pillow numpy")
    .run("pip install pypdf img2pdf==0.6.3 psutil pymupdf")  # PDF handling
    .run(
        "pip install opencv-python-headless"
    )  # Headless OpenCV for jdeskew (used by simple_page_creator)
    .run("pip install jdeskew==0.3.0 --no-deps")
    .run(f"pip install '{PYDANTIC_PIN}' markdownify")  # Data validation
    .run("pip install boto3 requests")  # AWS + HTTP
    .run("pip install openai google-genai==1.51.0 langfuse==5.0.1")
    .run("pip install amazon-textract-textractor azure-ai-documentintelligence")
    .run("pip cache purge")
)
