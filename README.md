# 🔴 SENTINEL — AI Security Analysis System

A dual-engine security threat detection system that combines **MediaPipe local pose analysis** with **Groq LLM vision AI** to analyze images for security threats in real-time.

---

## ✨ Features

- **Dual-Engine Analysis** — MediaPipe (local, offline) + Groq LLM (cloud vision)
- **4 Analysis Modes** — Hybrid, Both, Local Only, API Only
- **Token-Optimized Hybrid Mode** — Local runs first; only escalates to API if threat detected
- **Web UI** — Drag-and-drop interface with real-time results and threat visualization
- **REST API** — Fully documented JSON API for ESP32-CAM, Postman, or any HTTP client
- **History Log** — All scans saved with timestamps; filterable by date/time range
- **Threat Classification** — 0–10 numeric scale + 5 status levels (CLEAR → CRITICAL)
- **Free Hosting Ready** — Deploy to Render free tier with a single config file

---

## 📦 Project Structure

```
sentinel/
├── app.py                  # Main Flask application
├── requirements.txt        # Production (API-only, no MediaPipe)
├── requirements-local.txt  # Local (includes MediaPipe + OpenCV)
├── run_local.sh            # One-command local startup script
├── Procfile                # For Render/Heroku deployment
├── render.yaml             # Render deployment config
├── .env.example            # Environment variable template
├── .gitignore
├── templates/
│   ├── index.html          # Main analysis UI
│   └── history.html        # History & log viewer
├── uploads/                # Auto-created — uploaded images stored here
└── logs/
    └── history.json        # Auto-created — scan records (JSON)
```

---

## 🚀 Local Setup (Full Hybrid Mode)

### 1. Clone & Run
```bash
git clone <your-repo>
cd sentinel
chmod +x run_local.sh
./run_local.sh
```

The script will:
- Create a Python virtual environment
- Install all dependencies (including MediaPipe)
- Download the MediaPipe pose model (~25MB, one time)
- Start the server on `http://localhost:5000`

### 2. Manual Setup (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt

# Download MediaPipe model
curl -L -o pose_landmarker_heavy.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Set environment variables
copy .env.example .env
# Edit .env and set GROQ_API_KEY

python app.py
```

---

## ☁️ Deploy to Render (Free Tier)

> **Note:** Render free tier has limited RAM. Use `requirements.txt` (API-only, no MediaPipe) for online deployment. The system auto-detects and runs in API-only mode gracefully.

### Steps:
1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Render will auto-detect `render.yaml`
5. Set environment variable:
   - `GROQ_API_KEY` = your Groq key
6. Deploy!

**Free tier limitations:**
- Spins down after 15 mins of inactivity (first request may be slow)
- 512MB RAM — MediaPipe is disabled; runs in API-only mode
- 750 hours/month free

---

## 📡 API Reference

### POST `/analyze`
Upload an image for security analysis.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | ✅ | Image file (jpg, png, webp, bmp, max 10MB) |
| `mode` | String | ❌ | `hybrid` (default), `both`, `local`, `api` |

**Response:**
```json
{
  "id": "a3b2c1d4",
  "timestamp": "2024-01-15T10:30:00+00:00",
  "timestamp_unix": 1705314600.0,
  "threat_level": 8,
  "behavior": "Weaponized",
  "reason": "Arms raised above shoulders with full extension — overhead strike detected",
  "status": "HIGH",
  "status_color": "red",
  "status_icon": "🚨",
  "fusion_mode": "both_alert",
  "local_threat_level": 9,
  "api_threat_level": 8,
  "local_source": "local_mediapipe",
  "api_source": "groq:meta-llama/llama-4-scout-17b-16e-instruct",
  "mediapipe_available": true,
  "mode": "hybrid",
  "processing_time_s": 1.234,
  "filename": "20240115_103000_a3b2c1d4_photo.jpg",
  "original_filename": "photo.jpg"
}
```

**Threat Levels:**
| Level | Status | Meaning |
|-------|--------|---------|
| 0 | CLEAR | No threat |
| 1–3 | LOW | Minor anomaly |
| 4–6 | MODERATE | Suspicious behavior |
| 7–8 | HIGH | Active threat |
| 9–10 | CRITICAL | Immediate danger |

---

### GET `/api/history`
Retrieve scan history with optional filters.

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `limit` | int | Max records to return (default: 50, max: 500) |
| `last` | int | Return last N records (ignores start/end) |
| `start` | float | Unix timestamp — filter from this time |
| `end` | float | Unix timestamp — filter until this time |

```bash
# Last 10 records
GET /api/history?last=10

# Today's records
GET /api/history?start=1705276800&limit=500

# Time range
GET /api/history?start=1705000000&end=1705200000&limit=100
```

---

### GET `/api/history/last`
Returns the single most recent scan record.

---

### GET `/api/status`
Returns system status including available engines.

---

## 🤖 ESP32-CAM Integration

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char* SERVER = "http://YOUR_IP:5000/analyze";

void sendToSentinel(uint8_t* buf, size_t len) {
  HTTPClient http;
  http.begin(SERVER);
  http.setTimeout(15000);
  
  // Build multipart body
  String boundary = "SentinelBoundary";
  String header = "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
    "Content-Type: image/jpeg\r\n\r\n";
  String modeField = "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"mode\"\r\n\r\nhybrid\r\n";
  String footer = "--" + boundary + "--\r\n";
  
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  
  // Assemble full body
  int totalLen = header.length() + len + 2 + modeField.length() + footer.length();
  uint8_t* body = (uint8_t*)malloc(totalLen);
  int pos = 0;
  memcpy(body + pos, header.c_str(), header.length()); pos += header.length();
  memcpy(body + pos, buf, len); pos += len;
  body[pos++] = '\r'; body[pos++] = '\n';
  memcpy(body + pos, modeField.c_str(), modeField.length()); pos += modeField.length();
  memcpy(body + pos, footer.c_str(), footer.length());
  
  int code = http.POST(body, totalLen);
  if (code == 200) {
    String response = http.getString();
    // Parse JSON response
    // Check "threat_level" >= 6 to trigger alarm
    Serial.println(response);
  }
  free(body);
  http.end();
}
```

---

## 🔧 Analysis Modes Explained

| Mode | LocalAI | GroqAPI | Token Use | Best For |
|------|---------|---------|-----------|----------|
| **hybrid** | ✅ First | ✅ If threat | Low | Production (default) |
| **both** | ✅ Always | ✅ Always | High | Maximum accuracy |
| **local** | ✅ Always | ❌ Never | None | Offline / No credits |
| **api** | ❌ Never | ✅ Always | Every call | Online deploy (no MediaPipe) |

### Hybrid Fusion Logic
```
Local runs → threat_level < 6? → DONE (no API call)
                               → threat_level >= 6? → API verifies → results fused
```

### Fusion Rules (when both run)
- Either engine ≥ 6 → ALERT (conservative)
- Both agree (similar scores) → averaged → CONSENSUS
- Large disagreement (≥4 apart) → higher score wins → DISPUTED flag

---

## 🌐 Postman Collection

Import this to Postman:

**Analyze Image:**
- Method: POST
- URL: `{{base_url}}/analyze`
- Body: form-data
  - `image`: File
  - `mode`: `hybrid`

**Get History:**
- Method: GET
- URL: `{{base_url}}/api/history?last=10`

**Last Record:**
- Method: GET
- URL: `{{base_url}}/api/history/last`

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `MEDIAPIPE_MODEL_PATH` | `pose_landmarker_heavy.task` | Path to model file |

---

## 📝 History Storage

All scan records are saved to `logs/history.json` as a JSON array. Each record contains:
- Unique ID, timestamp (ISO + Unix)
- Full analysis result (threat level, behavior, reason)
- Both engine scores and sources
- Processing time, filename, mode used

Up to **10,000 records** are retained (oldest auto-pruned).

---

## 🔑 Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Go to API Keys → Create new key
4. Copy into `.env` as `GROQ_API_KEY`

Groq free tier: **~14,400 requests/day** on Llama 4 Scout (generous for security systems).

---

*SENTINEL — For authorized security monitoring use only.*
