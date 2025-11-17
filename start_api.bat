@echo off
REM Start DeepSight Vision API Service (Windows)
REM Usage: start_api.bat

echo Starting DeepSight Vision API Service...

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_api.txt
)

REM Check if model file exists
if not exist "trained_models\sample_trained_model.pt" (
    echo Error: Model file not found at trained_models\sample_trained_model.pt
    exit /b 1
)

REM Set environment variables
if not defined API_HOST set API_HOST=0.0.0.0
if not defined API_PORT set API_PORT=8000
if not defined CONFIDENCE_THRESHOLD set CONFIDENCE_THRESHOLD=0.05

echo Configuration:
echo    - Host: %API_HOST%
echo    - Port: %API_PORT%
echo    - Confidence Threshold: %CONFIDENCE_THRESHOLD%
echo.
echo Starting API server at http://%API_HOST%:%API_PORT%
echo API docs available at http://localhost:%API_PORT%/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python deepSightVision_api.py

