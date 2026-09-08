# SENTINEL v2.0 — AI Security Analysis System

## Project Structure

```
SENTINEL/
│
├── app.py                   ← Flask web server (run this to start)
├── analyzer.py              ← All AI logic (Gemini + Groq + MediaPipe)
│
├── .env                     ← YOUR API KEYS GO HERE (create this yourself)
├── .env.example             ← Template showing what keys are needed
│
├── requirements.txt         ← Python packages (online/no-MediaPipe)
├── requirements-local.txt   ← Python packages (local + MediaPipe)
│
├── run_local.sh             ← One-command startup (Linux/Mac)
├── run_local.bat            ← One-command startup (Windows)
│
├── pose_landmarker_heavy.task  ← MediaPipe model (download separately, see README)
│
├── templates/
│   ├── index.html           ← Main web UI (drag & drop, analyze, results)
│   └── history.html         ← History viewer with time filters
│
├── static/
│   ├── css/                 ← (empty — styles are inline in templates)
│   └── js/                  ← (empty — scripts are inline in templates)
│
├── uploads/                 ← Uploaded images saved here (auto-created)
└── logs/
    └── history.json         ← All scan records saved here (auto-created)
```

---

## Setup (Local — Full Mode with MediaPipe)

### Step 1 — Get API Keys (free, no credit card)

**Gemini (Primary — 1,500 free requests/day):**
1. Go to https://aistudio.google.com/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

**Groq (Backup — 1,000 free requests/day):**
1. Go to https://console.groq.com
2. Sign up with email
3. Go to API Keys → Create key
4. Copy the key

### Step 2 — Create your .env file

Copy `.env.example` to `.env` and fill in your keys:
```
GEMINI_API_KEY=AIzaSy...your_key_here
GROQ_API_KEY=gsk_...your_key_here
```

### Step 3 — Download MediaPipe Model (one time, ~25MB)

```bash
curl -L -o pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
```

Or run `run_local.sh` — it downloads automatically.

### Step 4 — Start the server

**Linux/Mac:**
```bash
chmod +x run_local.sh
./run_local.sh
```

**Windows:**
```
run_local.bat
```

Open browser at: http://localhost:5000

---

## Setup (Online — Render.com Free Tier)

1. Push to GitHub
2. Go to render.com → New Web Service → connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
5. Add environment variables in Render dashboard:
   - `GEMINI_API_KEY` = your key
   - `GROQ_API_KEY` = your key
6. Deploy

Note: MediaPipe is disabled on Render free tier (not enough RAM).
The system auto-detects and runs API-only mode gracefully.

---

## Analysis Modes

| Mode | MediaPipe | Gemini | Groq | Token cost |
|------|-----------|--------|------|------------|
| `hybrid` (default) | ✅ Always | ✅ If threat ≥ 4 | Backup if Gemini fails | Low |
| `api` | ❌ | ✅ Always | Backup | Every call |
| `local` | ✅ | ❌ | ❌ | Zero |

**Hybrid mode** is recommended: MediaPipe runs free and fast on every frame.
Only escalates to Gemini (using your free quota) when the pose data looks suspicious.

---

## API Reference

### POST /analyze
Upload image for analysis.

```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@photo.jpg" \
  -F "mode=hybrid"
```

Response:
```json
{
  "id": "a3b2c1d4",
  "threat_level": 7,
  "scene_type": "fighting",
  "behavior": "Hostile",
  "subjects": 2,
  "reason": "Two subjects engaged in physical altercation, one has arm around other's neck",
  "confidence": "high",
  "status": "HIGH",
  "status_icon": "🚨",
  "engines_used": ["mediapipe", "gemini:gemini-2.5-flash"],
  "timestamp": "2026-07-03T09:01:22+00:00",
  "processing_time_s": 1.8
}
```

### GET /api/history
```
/api/history?last=10
/api/history?start=1700000000&end=1800000000&limit=100
/api/history?scene_type=fighting
/api/history/last
```

### GET /api/status
Returns system status and which engines are available.
