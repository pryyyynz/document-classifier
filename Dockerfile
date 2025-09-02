FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# System deps for OCR and DOC support
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr \
      antiword \
      poppler-utils \
      build-essential \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what we need to run the API and models
COPY api ./api
COPY src ./src
COPY preprocessing ./preprocessing
COPY enhanced_models_output ./enhanced_models_output


EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


