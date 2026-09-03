# syntax=docker/dockerfile:1
#
# Production image for the Vector Embedding Visualizer backend.
# Uses a slim Python base; PyTorch + transformers are installed from the
# pinned requirements. Build context is the backend/ directory.

FROM python:3.12-slim AS base

# - PYTHONDONTWRITEBYTECODE: no .pyc files in the image.
# - PYTHONUNBUFFERED: stream logs straight to stdout for the container runtime.
# - HF_HOME: keep the Hugging Face cache in a predictable, mountable location.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application source.
COPY app ./app

# Create a non-root user and a writable model cache directory.
RUN useradd --create-home appuser \
    && mkdir -p /cache/huggingface \
    && chown -R appuser:appuser /app /cache
USER appuser

EXPOSE 8000

# A basic container healthcheck hitting the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
