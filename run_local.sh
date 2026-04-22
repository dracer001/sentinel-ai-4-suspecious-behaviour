#!/bin/bash
# ─── SENTINEL LOCAL RUN SCRIPT ────────────────────────────────────────────────

set -e

echo ""
echo "  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     "
echo "  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     "
echo "  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     "
echo "  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     "
echo "  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗ "
echo "  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝"
echo ""
echo "  AI Security Analysis System — Local Mode"
echo ""

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "▶ Creating virtual environment..."
  python3 -m venv .venv
fi

echo "▶ Activating virtual environment..."
source .venv/bin/activate

echo "▶ Installing dependencies..."
pip install -r requirements-local.txt -q

# Check for .env
if [ ! -f ".env" ]; then
  echo "⚠️  No .env file found. Copying from .env.example..."
  cp .env.example .env
  echo "✏️  Edit .env and set your GROQ_API_KEY, then re-run this script."
  exit 1
fi

# Load .env
export $(grep -v '^#' .env | xargs)

# Check for MediaPipe model
if [ ! -f "${MEDIAPIPE_MODEL_PATH:-pose_landmarker_heavy.task}" ]; then
  echo ""
  echo "⚠️  MediaPipe model not found. Downloading..."
  echo "    This is ~25MB and only needed once."
  curl -L -o pose_landmarker_heavy.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
  echo "✅ MediaPipe model downloaded."
fi

echo ""
echo "✅ Starting Sentinel on http://localhost:5000"
echo "   Web UI:     http://localhost:5000"
echo "   History:    http://localhost:5000/history-page"
echo "   API Status: http://localhost:5000/api/status"
echo ""

FLASK_DEBUG=true python app.py
