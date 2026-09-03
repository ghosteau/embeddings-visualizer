# syntax=docker/dockerfile:1

FROM node:24-alpine AS frontend-build
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit

COPY frontend/ ./
# Empty means same-origin API calls in production.
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    ENVIRONMENT=production \
    STATIC_DIR=/app/static

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
# torch resolves to the CUDA build on default PyPI, dragging in ~2.5 GB of
# NVIDIA wheels that a GPU-less host can never use. Install it from PyTorch's
# CPU index first; the requirements install below then sees it satisfied and
# leaves it alone. This is the difference between a ~250 MB and a ~3 GB image.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" \
    && pip install -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /cache/huggingface /app/static \
    && chown -R appuser:appuser /app /cache

COPY --chown=appuser:appuser backend/app ./app
COPY --from=frontend-build --chown=appuser:appuser /build/frontend/dist ./static

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
