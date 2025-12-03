# DeepSight AI Vision REST API - Installation & Usage Guide

Complete guide for installing, configuring, and using the DeepSight Vision REST API service for UI element detection with semantic text matching.

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: Local Installation](#method-1-local-installation)
  - [Method 2: Docker Installation](#method-2-docker-installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## Overview

The **DeepSight Vision REST API** is a computer vision service that detects UI elements in screenshots using:

- **YOLO** - For non-text UI elements (buttons, fields, icons)
- **PaddleOCR** - For text detection and recognition
- **Semantic Matching** - AI-powered text similarity using sentence transformers
- **Multilingual Support** - 50+ languages with feature-flagged model selection

**Key Features:**
-  REST API with FastAPI
-  Base64 image input
-  Semantic text matching with fuzzy fallback
-  Multilingual model support
-  Thread-safe caching
-  Multi-tenant ready
-  Production-ready with Docker support

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux, macOS, Windows 10+ |
| **Python** | 3.12+ |
| **RAM** | 4GB (8GB recommended) |
| **Disk Space** | 5GB free |
| **CPU** | 2 cores minimum |

### Optional

- **Docker** (for containerized deployment)
- **GPU** (not required, CPU-only by default)

---

## Installation Methods

### Method 1: Local Installation

#### Step 1: Clone or Download the Project

```bash
cd /path/to/your/workspace
# If using git
git clone <repository-url>
cd DeepSightVision

# Or download and extract the ZIP
```

#### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Install API dependencies
pip install -r requirements_api.txt
```

**Dependencies installed:**
- FastAPI & Uvicorn (REST API framework)
- OpenCV (image processing)
- YOLO (object detection)
- PaddleOCR (text detection)
- Sentence Transformers (semantic matching)
- RapidFuzz (fuzzy string matching)
- CacheTools (TTL-based caching)

**Installation time:** ~5-10 minutes (depending on internet speed)

#### Step 4: Prepare Model Files

Ensure your trained YOLO model is in place:

```bash
# Default location
mkdir -p trained_models
# Place your model file here:
# trained_models/sample_trained_model.pt
```

If you don't have a trained model, you'll need to train one or obtain a pre-trained model for your specific UI elements.

#### Step 5: Verify Installation

```bash
# Test imports
python3 -c "
from fastapi import FastAPI
from ultralytics import YOLO
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
print('✓ All dependencies installed successfully!')
"
```

---

### Method 2: Docker Installation

#### Step 1: Install Docker

**On Ubuntu/Debian:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes
```

**On macOS:**
```bash
# Install Docker Desktop from:
# https://www.docker.com/products/docker-desktop
```

**On Windows:**
```powershell
# Install Docker Desktop from:
# https://www.docker.com/products/docker-desktop
```

#### Step 2: Create Dockerfile

Create `Dockerfile` in your project root:

```dockerfile
FROM python:3.12-slim-bookworm

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY deepSightVision_api.py
COPY trained_models/ ./trained_models/

# Create temp directory
RUN mkdir -p temp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "deepSightVision_api.py"]
```

#### Step 3: Build Docker Image

```bash
# Build image
docker build -t deepsight-vision-api:latest .

# Build time: ~10-15 minutes (first time)
```

#### Step 4: Run Docker Container

**Basic run:**
```bash
docker run -d \
  --name deepsight-vision-api \
  -p 8000:8000 \
  deepsight-vision-api:latest
```

**With environment variables:**
```bash
docker run -d \
  --name deepsight-vision-api \
  -p 8000:8000 \
  -e ENABLE_SEMANTIC_MATCH=true \
  -e SEMANTIC_THRESHOLD=85.0 \
  -e SEMANTIC_CACHE_MAX_SIZE=10000 \
  deepsight-vision-api:latest
```

**With volume mounts (for model updates):**
```bash
docker run -d \
  --name deepsight-vision-api \
  -p 8000:8000 \
  -v $(pwd)/trained_models:/app/trained_models \
  -v $(pwd)/temp:/app/temp \
  deepsight-vision-api:latest
```

#### Step 5: Verify Docker Container

```bash
# Check container status
docker ps

# View logs
docker logs deepsight-vision-api

# Test health endpoint
curl http://localhost:8000/health
```

---

## Configuration

### Environment Variables

Configure the API behavior using environment variables:

#### Core Configuration

```bash
# YOLO Model Path
export YOLO_MODEL_PATH=/app/trained_models/sample_trained_model.pt

# API Server Settings
export API_HOST=0.0.0.0
export API_PORT=8000

# Detection Confidence
export CONFIDENCE_THRESHOLD=0.05
```

#### Semantic Matching Configuration

```bash
# Enable/Disable Semantic Matching
export ENABLE_SEMANTIC_MATCH=false  # Default: false

# Semantic Match Threshold (0-100)
export SEMANTIC_THRESHOLD=90.0

# Alpha Weight (0-1: 0=fuzzy only, 1=semantic only)
export SEMANTIC_ALPHA=0.7

# Rollout Percentage (gradual deployment)
export SEMANTIC_ROLLOUT_PERCENT=100

# Allow Per-Request Overrides
export ALLOW_CLIENT_OVERRIDE=true
```

#### Multilingual Model Configuration

```bash
# Enable Multilingual Model
export ENABLE_MULTILINGUAL_MODEL=false  # Default: false

# Model Names
export SEMANTIC_MODEL_NAME=all-MiniLM-L6-v2
export MULTILINGUAL_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
```

#### Cache Configuration

```bash
# Cache Settings
export SEMANTIC_CACHE_MAX_SIZE=10000
export SEMANTIC_CACHE_TTL=3600  # 1 hour in seconds
```

### Configuration File (Optional)

Create `.env` file in project root:

```bash
# .env file
ENABLE_SEMANTIC_MATCH=true
SEMANTIC_THRESHOLD=85.0
SEMANTIC_ALPHA=0.7
ENABLE_MULTILINGUAL_MODEL=false
SEMANTIC_CACHE_MAX_SIZE=10000
SEMANTIC_CACHE_TTL=3600
```

Load with:
```bash
# Install python-dotenv
pip install python-dotenv

# Then in your code or startup script
export $(cat .env | xargs)
```

---

## Running the Service

### Local Run

**Basic run:**
```bash
python deepSightVision_api.py
```

**With custom configuration:**
```bash
ENABLE_SEMANTIC_MATCH=true \
SEMANTIC_THRESHOLD=85 \
python deepSightVision_api.py
```

**Output:**
```
INFO - Semantic Config: enabled=True, rollout=100%, threshold=85.0, alpha=0.7
INFO - DeepSight Vision API started successfully
INFO - Model path: /app/trained_models/sample_trained_model.pt
INFO - Starting DeepSight Vision API on 0.0.0.0:8000
```

### Access API Documentation

Once running, access interactive API docs:

```
http://localhost:8000/docs     # Swagger UI
http://localhost:8000/redoc    # ReDoc
http://localhost:8000/health   # Health check
```

### Docker Run

```bash
# Start container
docker start deepsight-vision-api

# Stop container
docker stop deepsight-vision-api

# Restart container
docker restart deepsight-vision-api

# View real-time logs
docker logs -f deepsight-vision-api

# Remove container
docker stop deepsight-vision-api
docker rm deepsight-vision-api
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check API status and model availability

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "yolo_model_loaded": true,
  "paddle_ocr_loaded": true,
  "model_path": "/app/trained_models/sample_trained_model.pt"
}
```

---

### 2. Analyze Page

**Endpoint:** `POST /analyze/page`

**Description:** Detect all UI elements in a screenshot with optional semantic text matching

**Request Body:**
```json
{
  "screenshot_base64": "iVBORw0KGgo...",
  "viewport_size": {
    "width": 1920,
    "height": 1080
  },
  "confidence_threshold": 0.05,
  "text": "Login",
  "element_type": "floating_label",
  "enable_semantic_match": true,
  "semantic_threshold": 80.0,
  "semantic_alpha": 0.9,
  "use_multilingual_model": false,
  "tenant_id": "customer_123",
  "request_id": "req_456"
}
```

**Field Descriptions:**

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `screenshot_base64` |  Yes | string | Base64-encoded screenshot |
| `viewport_size` |  Yes | object | Browser viewport dimensions |
| `confidence_threshold` |  No | float | YOLO confidence (default: 0.05) |
| `text` |  No | string | Text to search for (triggers semantic matching) |
| `element_type` |  No | string | Element type filter: "floating_label" or "non-text" |
| `enable_semantic_match` |  No | boolean | Override server semantic enable flag |
| `semantic_threshold` |  No | float | Match threshold 0-100 (default: 90.0) |
| `semantic_alpha` |  No | float | Semantic weight 0-1 (default: 0.7) |
| `use_multilingual_model` |  No | boolean | Use multilingual model (default: false) |
| `tenant_id` |  No | string | Tenant identifier for cache isolation |
| `request_id` |  No | string | Request ID for tracking/rollout |

**Sample Response:**
```json
{
  "screenshot_size": {
    "width": 1920,
    "height": 1080
  },
  "detected_elements": [
    {
      "type": "floating_label",
      "text": "Login",
      "x": 1467,
      "y": 637,
      "width": 88,
      "height": 40,
      "confidence": 99.98944401741028,
      "x1": 1467,
      "x2": 1555,
      "y1": 637,
      "y2": 677,
      "scaled_center_x": 1512,
      "scaled_center_y": 650,
      "text_source": "semantic_query",
      "semantic_query": "SIGN IN",
      "semantic_pct": 89.58,
      "fuzzy_pct": 50,
      "combined_pct": 85.62,
      "model": "all-MiniLM-L6-v2",
      "method": "semantic"
    },
    {
      "type": "non-text",
      "class_name": "field",
      "x": 724,
      "y": 204,
      "width": 471,
      "height": 54,
      "x1": 724,
      "y1": 1195,
      "x2": 204,
      "y2": 258,
      "scaled_center_x": 959,
      "scaled_center_y": 230
    } ```
 ],
  "total_text_elements": 13,
  "total_non_text_elements": 0,
  "matched": true,
  "method": "semantic",
  "reason": null,
  "semantic_result": {
    "best_candidate": {
      "text": "Login",
      "semantic_pct": 89.58,
      "fuzzy_pct": 50,
      "combined_pct": 85.62
    },
    "total_matches": 1,
    "matched_elements": [
      {
        "text": "Login",
        "combined_pct": 85.62,
        "position": "(1467, 637)"
      }
    ],
    "model": "all-MiniLM-L6-v2",
    "alpha": 0.9,
    "threshold": 80
  }
}
```

---

### 3. Find Element

**Endpoint:** `POST /find/element`

**Description:** Find a specific element (returns first match only)

**Request Body:**
```json
{
  "screenshot_base64": "iVBORw0KGgo...",
  "viewport_size": {
    "width": 1920,
    "height": 1080
  },
  "search_criteria": {
    "text": "Submit",
    "element_type": "floating_label"
  }
}
```

**Response:**
```json
{
  "found": true,
  "element": {
    "type": "floating_label",
    "text": "Submit",
    "x": 500,
    "y": 600,
    "width": 100,
    "height": 40
  },
  "message": "Element found successfully"
}
```

---

### 4. Filter Elements

**Endpoint:** `POST /filter/elements`

**Description:** Get all elements matching criteria

**Request Body:**
```json
{
  "screenshot_base64": "iVBORw0KGgo...",
  "viewport_size": {
    "width": 1920,
    "height": 1080
  },
  "text": "Login",
  "element_type": "floating_label",
  "class_name": "button"
}
```

**Response:**
```json
{
  "filtered_elements": [
    {
      "type": "floating_label",
      "text": "Login",
      "x": 100,
      "y": 200
    },
    {
      "type": "floating_label",
      "text": "Login",
      "x": 500,
      "y": 800
    }
  ],
  "count": 2
}
```

---

## Usage Examples

### Example 1: Basic Element Detection

```python
import requests
import base64

# Read and encode screenshot
with open("screenshot.png", "rb") as f:
    screenshot_base64 = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post(
    "http://localhost:8000/analyze/page",
    json={
        "screenshot_base64": screenshot_base64,
        "viewport_size": {"width": 1920, "height": 1080}
    }
)

result = response.json()
print(f"Found {result['total_text_elements']} text elements")
print(f"Found {result['total_non_text_elements']} non-text elements")
```

### Example 2: Semantic Text Matching (English)

```python
import requests
import base64

with open("screenshot.png", "rb") as f:
    screenshot_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/analyze/page",
    json={
        "screenshot_base64": screenshot_base64,
        "viewport_size": {"width": 1920, "height": 1080},
        "text": "Sign in",  # Search for "Sign in"
        "element_type": "floating_label",
        "enable_semantic_match": True,
        "semantic_threshold": 85.0,
        "semantic_alpha": 0.9  # Favor semantic over fuzzy
    }
)

result = response.json()
if result['matched']:
    print(f"Method: {result['method']}")
    print(f"Found {result['semantic_result']['total_matches']} matches")
    for elem in result['detected_elements']:
        if elem.get('text_source') == 'semantic_fallback':
            print(f"Matched '{elem['matched_candidate_text']}' to '{elem['text']}'")
            print(f"Position: ({elem['x']}, {elem['y']})")
```

### Example 3: Multilingual Text Matching

```python
import requests
import base64

with open("spanish_page.png", "rb") as f:
    screenshot_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/analyze/page",
    json={
        "screenshot_base64": screenshot_base64,
        "viewport_size": {"width": 1920, "height": 1080},
        "text": "Iniciar sesión",  # Spanish for "Login"
        "element_type": "floating_label",
        "enable_semantic_match": True,
        "use_multilingual_model": True,  # Enable multilingual
        "semantic_threshold": 80.0,
        "tenant_id": "spanish_customer"
    }
)

result = response.json()
print(f"Model used: {result['semantic_result']['model']}")
print(f"Matches: {result['semantic_result']['total_matches']}")
```

### Example 4: Find All Buttons

```python
import requests
import base64

with open("screenshot.png", "rb") as f:
    screenshot_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/filter/elements",
    json={
        "screenshot_base64": screenshot_base64,
        "viewport_size": {"width": 1920, "height": 1080},
        "element_type": "non-text",
        "class_name": "button"
    }
)

result = response.json()
print(f"Found {result['count']} buttons")
for button in result['filtered_elements']:
    print(f"Button at ({button['x']}, {button['y']})")
```

### Example 5: Using cURL

```bash
# Basic detection
curl -X POST http://localhost:8000/analyze/page \
  -H "Content-Type: application/json" \
  -d @request.json

# Semantic matching
curl -X POST http://localhost:8000/analyze/page \
  -H "Content-Type: application/json" \
  -d '{
    "screenshot_base64": "iVBORw0KGgo...",
    "viewport_size": {"width": 1920, "height": 1080},
    "text": "Login",
    "element_type": "floating_label",
    "enable_semantic_match": true
  }'

# Health check
curl http://localhost:8000/health
```

---

## Advanced Features

### 1. Semantic Matching

**What it does:** Finds UI elements even when text doesn't exactly match

**Use cases:**
- Typo tolerance: "Registr" → finds "Register"
- Synonym matching: "Sign Up" → finds "Register"
- Partial matching: "Submit" → finds "Submit Form"

**Configuration:**
```bash
export ENABLE_SEMANTIC_MATCH=true
export SEMANTIC_THRESHOLD=85.0  # 0-100
export SEMANTIC_ALPHA=0.7       # 0=fuzzy, 1=semantic
```

**Scoring:**
```
combined_score = alpha × semantic_score + (1-alpha) × fuzzy_score

alpha=0.9 → 90% semantic, 10% fuzzy (favor meaning)
alpha=0.5 → 50% semantic, 50% fuzzy (balanced)
alpha=0.3 → 30% semantic, 70% fuzzy (favor spelling)
```

### 2. Multilingual Support

**Supported languages:** 50+ including:
- European: English, Spanish, French, German, Italian, Portuguese
- Asian: Chinese, Japanese, Korean, Thai, Vietnamese
- Middle Eastern: Arabic, Hebrew, Turkish

**Models:**
- Default: `all-MiniLM-L6-v2` (English, 80MB)
- Multilingual: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 420MB)

**Usage:**
```json
{
  "text": "登录",  // Chinese
  "use_multilingual_model": true
}
```

### 3. Multi-Tenant Support

**Cache isolation per tenant:**
```json
{
  "tenant_id": "customer_123",
  "text": "Login"
}
```

**Benefits:**
- Separate cache partitions
- Per-tenant monitoring
- Billing/usage tracking

### 4. Rollout Control

**Gradual feature deployment:**
```bash
export SEMANTIC_ROLLOUT_PERCENT=25  # 25% of requests
```

**Deterministic rollout:**
- Based on `request_id` hash
- Same request_id always gets same decision

---

## Troubleshooting

### Issue 1: Port Already in Use

**Error:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
export API_PORT=8001
python deepSightVision_api.py
```

### Issue 2: Model Not Found

**Error:** `FileNotFoundError: Model file not found`

**Solution:**
```bash
# Check model path
ls -la trained_models/

# Set correct path
export YOLO_MODEL_PATH=/path/to/your/model.pt

# Or place model in default location
mkdir -p trained_models
cp your_model.pt trained_models/sample_trained_model.pt
```

### Issue 3: Out of Memory

**Error:** `MemoryError` or container OOM killed

**Solution:**
```bash
# Reduce cache size
export SEMANTIC_CACHE_MAX_SIZE=5000

# Or increase Docker memory
docker run -d \
  --memory=4g \
  --name deepsight-vision-api \
  deepsight-vision-api:latest
```

### Issue 4: Slow First Request

**Symptom:** First request takes 10-15 seconds

**Explanation:** This is normal! First request downloads models:
- PaddleOCR models (~20MB)
- Sentence transformer models (~80-420MB)

**Solution:** Models are cached after first use. Subsequent requests are fast.

### Issue 5: Semantic Model Download Fails

**Error:** `ConnectionError` or `HTTPError` from HuggingFace

**Solution:**
```bash
# Pre-download models
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model downloaded successfully')
"

# Or use offline mode
export TRANSFORMERS_OFFLINE=1
```

### Issue 6: Docker Container Exits Immediately

**Debug:**
```bash
# Check logs
docker logs deepsight-vision-api

# Run in foreground to see errors
docker run --rm deepsight-vision-api:latest

# Check health
docker inspect deepsight-vision-api
```

---

## Production Deployment

### Load Balancing with Nginx

**nginx.conf:**
```nginx
upstream ai_vision_backend {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://ai_vision_backend;
        proxy_set_header Host $host;
        client_max_body_size 50M;
    }
}
```

### Docker Compose (Multi-Instance)

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api1
      - api2
      - api3

  api1:
    build: .
    environment:
      - ENABLE_SEMANTIC_MATCH=true
      - SEMANTIC_THRESHOLD=85.0
    deploy:
      resources:
        limits:
          memory: 4G

  api2:
    build: .
    environment:
      - ENABLE_SEMANTIC_MATCH=true
      - SEMANTIC_THRESHOLD=85.0
    deploy:
      resources:
        limits:
          memory: 4G

  api3:
    build: .
    environment:
      - ENABLE_SEMANTIC_MATCH=true
      - SEMANTIC_THRESHOLD=85.0
    deploy:
      resources:
        limits:
          memory: 4G
```

**Deploy:**
```bash
docker-compose up -d
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepsight-vision-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepsight-vision-api
  template:
    metadata:
      labels:
        app: deepsight-vision-api
    spec:
      containers:
      - name: api
        image: deepsight-vision-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: deepsight-vision-service
spec:
  type: LoadBalancer
  selector:
    app: deepsight-vision-api
  ports:
  - port: 80
    targetPort: 8000
```

### Monitoring

**Prometheus metrics (future):**
```python
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
response_time = Histogram('api_response_seconds', 'Response time')
```

**Health check:**
```bash
# Add to monitoring
curl http://localhost:8000/health
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Response Time** (no semantic) | 2-3 seconds |
| **Response Time** (with semantic) | 3-5 seconds |
| **First Request** (model loading) | 10-15 seconds |
| **Concurrent Requests** (single instance) | 10-20 |
| **Memory Usage** (default model) | ~500MB |
| **Memory Usage** (both models) | ~1.6GB |
| **Cache Hit Rate** | 70-90% |

---

## API Rate Limits

**Recommended limits:**
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req zone=api_limit burst=20 nodelay;
```

---

## Security Considerations

1. **API Authentication:** Add API key validation (not implemented by default)
2. **Rate Limiting:** Use Nginx or API Gateway
3. **Input Validation:** Base64 size limits (50MB max recommended)
4. **HTTPS:** Use SSL/TLS in production
5. **Network:** Run in private VPC, expose via load balancer only

---

## Support & Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Quick Reference

### Start Service (Local)
```bash
python deepSightVision_api.py
```

### Start Service (Docker)
```bash
docker run -d -p 8000:8000 deepsight-vision-api
```

### Health Check
```bash
curl http://localhost:8000/health
```

### API Docs
```
http://localhost:8000/docs
```

### Common Ports
- API Service: `8000`
- Swagger UI: `8000/docs`
- Health Check: `8000/health`

---

## Version Information

- **API Version:** 1.0.0
- **FastAPI:** 0.104.0+
- **Python:** 3.12
- **Default semantic Model fallback:** all-MiniLM-L6-v2
- **Multilingual semantic Model fallback:** paraphrase-multilingual-MiniLM-L12-v2

---
