FROM python:3.10-slim

# System dependencies for OpenCV, EasyOCR, and PDF processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    poppler-utils \
    tesseract-ocr \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY app/ ./app/

# Download yolov8n.pt model at build time
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -O yolov8n.pt

# Create data directories
RUN mkdir -p qdrant_data uploads

# Expose port (7860 for HuggingFace Spaces, 8000 default)
EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
