@echo off
echo.
echo   +---------------------------------+
echo   ^|  SENTINEL v2.0 - Local Startup  ^|
echo   +---------------------------------+
echo.

REM Create virtual environment if needed
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements-local.txt -q

REM Check for .env
if not exist ".env" (
    echo.
    echo WARNING: No .env file found. Creating from template...
    copy .env.example .env
    echo.
    echo  ACTION REQUIRED:
    echo  Edit .env and add your API keys:
    echo    GEMINI_API_KEY  -^> https://aistudio.google.com/apikey
    echo    GROQ_API_KEY    -^> https://console.groq.com
    echo.
    echo  Then run this script again.
    pause
    exit /b 1
)

REM Check for MediaPipe model
if not exist "pose_landmarker_heavy.task" (
    echo.
    echo Downloading MediaPipe model (25MB, one time only)...
    curl -L -o pose_landmarker_heavy.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    echo Model downloaded.
)

echo.
echo Starting SENTINEL...
echo.
echo   Web UI     -^>  http://localhost:5000
echo   History    -^>  http://localhost:5000/history-page
echo   API Status -^>  http://localhost:5000/api/status
echo.

set FLASK_DEBUG=true
python app.py
pause
