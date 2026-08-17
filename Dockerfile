# ==============================================================================
# AI Therapy Notes Maker - Production Container Image
# Multi-arch compatible, lightweight Python 3.11 with system FFmpeg & Gunicorn
# ==============================================================================

FROM python:3.11-slim

# Set environment variables for clean, unbuffered Python runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    ASR_BACKEND=gemini \
    STORAGE_DIR=/app

# Install system dependencies (FFmpeg for audio conversion, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition first (leverages Docker layer caching)
COPY requirements.txt .

# Install Python packages and Gunicorn production WSGI server
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code and assets
COPY app.py routes.py ./
COPY config/ config/
COPY asr/ asr/
COPY summarizer/ summarizer/
COPY templates/ templates/
COPY static/ static/

# Create runtime directories for storage with appropriate permissions
RUN mkdir -p uploads downloads transcripts && \
    chmod -R 777 uploads downloads transcripts

# Expose web application port
EXPOSE 5000

# Health check to ensure service responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/quota_status || exit 1

# Launch with Gunicorn production server (2 workers, 120s timeout for AI synthesis)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
