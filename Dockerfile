# ============================================================
#  AI Proctor — Dockerfile
#  Multi-stage build with optional GPU support
# ============================================================

# ── Stage 1: Base ════════════════════════════════════════════
FROM python:3.11-slim AS base

# System deps for OpenCV, audio, and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Dependencies ═══════════════════════════════════
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application ════════════════════════════════════
COPY . .

# Download YOLO weights at build time (cached in image)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# ── Environment ═════════════════════════════════════════════
ENV PROCTOR_API_HOST=0.0.0.0
ENV PROCTOR_API_PORT=8000
ENV PROCTOR_CAMERA_INDEX=0
ENV PROCTOR_DEMO_MODE=false

EXPOSE 8000

# ── Healthcheck ═════════════════════════════════════════════
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.ok else 1)" || exit 1

# ── Entrypoint ══════════════════════════════════════════════
# Default: API-only mode (camera requires --device flag at runtime)
CMD ["python", "-m", "ai_proctor.main", "--api-only"]

# ── GPU variant ─────────────────────────────────────────────
# To build with GPU support:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1-runtime-ubuntu22.04 -t ai-proctor-gpu .
# Then run with:
#   docker run --gpus all -p 8000:8000 --device /dev/video0 ai-proctor-gpu
