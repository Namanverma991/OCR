# ==========================================
# Multi-Stage Production Dockerfile (CPU)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for OpenCV, spatial libs, and image codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV OCR_ENABLED=true
ENV OCR_VERSION=PP-OCRv4
ENV OCR_DEVICE=cpu
ENV USE_GPU=false
ENV FLAGS_enable_pir_api=0
ENV FLAGS_use_mkldnn=0
ENV FLAGS_enable_onednn=0
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
