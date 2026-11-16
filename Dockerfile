# syntax=docker/dockerfile:1
# frames-extractor: FastAPI webapp + in-process worker.
# Default build is CPU-only torch (small image, right for Azure with no GPU).
# GPU build:  docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t frames-extractor:gpu .

FROM python:3.11-slim

ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    FRAMES_DATA_DIR=/app/data \
    FRAMES_DEVICE=auto \
    HF_HOME=/app/data/.hf-cache

# Runtime lib for OpenCV. No ffmpeg install needed: opencv wheels bundle it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1 (biggest, changes rarely): torch.
RUN pip install "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX}

# Layer 2: the other dependencies, mirrored from pyproject.toml.
# opencv-python-headless: no GUI libs (the cv2 review window is CLI-only, not used by the webapp).
RUN pip install \
    "opencv-python-headless>=4.9" \
    "numpy>=1.26" \
    "imagehash>=4.3.1" \
    "transformers==5.14.1" \
    "accelerate==1.14.0" \
    "requests>=2.31" \
    "fastapi>=0.115" \
    "uvicorn>=0.30" \
    "python-multipart>=0.0.9"

# Layer 3 (changes often): your code. Imported via PYTHONPATH, not pip-installed,
# so the pinned "torch==2.6.0+cu124" in pyproject.toml is never resolved here.
COPY src ./src

# Non-root user; /app/data is where the Azure Files share will be mounted.
RUN useradd --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]
