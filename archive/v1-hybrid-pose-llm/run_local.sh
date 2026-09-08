#!/bin/bash
# SENTINEL — Local startup script (Linux / Mac)
# Run: chmod +x run_local.sh && ./run_local.sh

set -e

echo ""
echo "  ┌─────────────────────────────────┐"
echo "  │  SENTINEL v2.0 — Local Startup  │"
echo "  └─────────────────────────────────┘"
echo ""

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "▶ Creating virtual environment..."
  python3 -m venv .venv
fi

echo "▶ Activating virtual environment..."
source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
echo "▶ Installing dependencies..."
pip install -r requirements-local.txt -q

# ── Check .env ────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo ""
  echo "⚠️  No .env file found. Creating from template..."
  cp .env.example .env
  echo ""
  echo "  ACTION REQUIRED:"
  echo "  Edit .env and add your API keys:"
  echo "    GEMINI_API_KEY  → https://aistudio.google.com/apikey  (free, no card)"
  echo "    GROQ_API_KEY    → https://console.groq.com            (free, no card)"
  echo ""
  echo "  Then run this script again."
  exit 1
fi

# Load .env
export $(grep -v '^#' .env | grep -v '^$' | xargs)

# ── Download MediaPipe model if missing ───────────────────────────────────────
MODEL="${MEDIAPIPE_MODEL_PATH:-pose_landmarker_heavy.task}"
if [ ! -f "$MODEL" ]; then
  echo ""
  echo "▶ Downloading MediaPipe pose model (~25MB, one time only)..."
  curl -L --progress-bar -o "$MODEL" \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
  echo "✅ Model downloaded."
fi

# ── Start server ──────────────────────────────────────────────────────────────
echo ""
echo "✅ Starting SENTINEL..."
echo ""
echo "   Web UI      →  http://localhost:5000"
echo "   History     →  http://localhost:5000/history-page"
echo "   API Status  →  http://localhost:5000/api/status"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

FLASK_DEBUG=true python app.py
