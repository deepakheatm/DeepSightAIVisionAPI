#!/bin/bash
# Start DeepSight Vision API Service
# Usage: ./start_api.sh

set -e

echo " Starting DeepSight Vision API Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo " Installing dependencies..."
    pip install -r requirements_api.txt
fi

# Check if model file exists
if [ ! -f "trained_models/sample_trained_model.pt" ]; then
    echo " Error: Model file not found at trained_models/sample_trained_model.pt"
    exit 1
fi

# Set environment variables
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.05}

echo " Configuration:"
echo "   - Host: $API_HOST"
echo "   - Port: $API_PORT"
echo "   - Confidence Threshold: $CONFIDENCE_THRESHOLD"

# Start the API
echo ""
echo " Starting API server at http://$API_HOST:$API_PORT"
echo " API docs available at http://localhost:$API_PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python deepSightVision_api.py

