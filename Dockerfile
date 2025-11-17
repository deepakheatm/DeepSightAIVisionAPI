# DeepSight Vision API Service Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application files
COPY deepSightVision_api.py .
COPY config.py .
COPY locator_db.py .

# Create temp directory
RUN mkdir -p temp

# Copy trained models directory
COPY trained_models/ ./trained_models/

# Expose API port
EXPOSE 8000

# Set environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV YOLO_MODEL_PATH=/app/trained_models/sample_trained_model.pt
ENV CONFIDENCE_THRESHOLD=0.05

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["python", "-m", "uvicorn", "deepSightVision_api:app", "--host", "0.0.0.0", "--port", "8000"]

