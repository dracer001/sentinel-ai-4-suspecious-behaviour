import os
import json
import base64
import uuid
import time
import logging
import urllib.request  # <--- Added for the auto-download logic
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Ensure this matches what you set in Render Dashboard
GROQ_KEY = os.getenv("GROQ_API_KEY") 
UPLOAD_FOLDER = Path("uploads")
LOG_FILE = Path("logs/history.json")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

# MediaPipe model path
MEDIAPIPE_MODEL = os.environ.get("MEDIAPIPE_MODEL_PATH", "pose_landmarker_heavy.task")

# ─── MODEL DOWNLOADER ────────────────────────────────────────────────────────
def ensure_model_exists(model_path):
    if not Path(model_path).exists():
        log.info(f"📡 Downloading MediaPipe model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            log.info("✅ Download complete.")
        except Exception as e:
            log.error(f"❌ Failed to download model: {e}")

# Call the downloader before initializing MediaPipe
ensure_model_exists(MEDIAPIPE_MODEL)


# Groq models in priority order (vision-capable first)
GROQ_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
]

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

UPLOAD_FOLDER.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)
if not LOG_FILE.exists():
    LOG_FILE.write_text("[]")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_KEY)

# ─── MEDIAPIPE LOCAL ANALYZER ─────────────────────────────────────────────────
mediapipe_available = False
SecurityAI_instance = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import numpy as np
    import cv2

    class SecurityAI:
        def __init__(self, model_path):
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(base_options=base_options)
            self.detector = mp_vision.PoseLandmarker.create_from_options(options)
            log.info("✅ MediaPipe SecurityAI loaded.")

        def get_angle(self, a, b, c):
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            norm = np.linalg.norm(ba) * np.linalg.norm(bc)
            if norm == 0:
                return 0
            return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)))

        def analyze_frame(self, image_path: str) -> dict:
            image = mp.Image.create_from_file(image_path)
            res = self.detector.detect(image)
            if not res.pose_landmarks:
                return {
                    "threat_level": 0,
                    "behavior": "NO_SUBJECT",
                    "reason": "No human pose detected in frame",
                    "source": "local_mediapipe",
                }

            lm = res.pose_landmarks[0]
            nose = lm[0]
            l_sh, r_sh = lm[11], lm[12]
            l_wr, r_wr = lm[15], lm[16]
            l_hip, r_hip = lm[23], lm[24]
            l_knee, r_knee = lm[25], lm[26]
            l_ank, r_ank = lm[27], lm[28]

            avg_hip_y = (l_hip.y + r_hip.y) / 2
            l_arm_angle = self.get_angle(l_sh, lm[13], l_wr)
            r_arm_angle = self.get_angle(r_sh, lm[14], r_wr)
            ground_clearance = abs(nose.y - ((l_ank.y + r_ank.y) / 2))
            sh_center_x = (l_sh.x + r_sh.x) / 2
            look_offset = abs(nose.x - sh_center_x)

            if (l_wr.y < l_sh.y or r_wr.y < r_sh.y) and (l_arm_angle > 140 or r_arm_angle > 140):
                return {
                    "threat_level": 9,
                    "behavior": "Weaponized",
                    "reason": "Arms raised above shoulders with full extension — overhead strike or weapon threat detected",
                    "source": "local_mediapipe",
                }
            if ground_clearance < 0.4:
                return {
                    "threat_level": 7,
                    "behavior": "Stealth",
                    "reason": "Head near ground level — crawling or crouching/stealth movement detected",
                    "source": "local_mediapipe",
                }
            if look_offset > (abs(l_sh.x - r_sh.x) * 0.8) and ground_clearance < 0.7:
                return {
                    "threat_level": 5,
                    "behavior": "Suspicious",
                    "reason": "Unusual head offset with low body posture — suspicious peeping or loitering",
                    "source": "local_mediapipe",
                }
            if (l_arm_angle < 90 or r_arm_angle < 90) and (l_wr.y < avg_hip_y):
                return {
                    "threat_level": 7,
                    "behavior": "Hostile",
                    "reason": "Arms bent at acute angles with fists raised — combat-ready stance",
                    "source": "local_mediapipe",
                }

            return {
                "threat_level": 1,
                "behavior": "Normal",
                "reason": "No threatening postures or gestures detected — subject appears calm",
                "source": "local_mediapipe",
            }

    if Path(MEDIAPIPE_MODEL).exists():
        SecurityAI_instance = SecurityAI(MEDIAPIPE_MODEL)
        mediapipe_available = True
    else:
        log.warning(f"⚠️ MediaPipe model not found at '{MEDIAPIPE_MODEL}'. Local analysis disabled.")

except ImportError:
    log.warning("⚠️ MediaPipe not installed. Running API-only mode.")
except Exception as e:
    log.error(f"⚠️ MediaPipe init error: {e}")


# ─── GROQ API ANALYZER ────────────────────────────────────────────────────────
def get_security_prompt():
    return """
[SECURITY AUDIT PROTOCOL — RESPOND IN JSON ONLY]
Analyze the subject's body language, posture, and any held objects for security threats.
Ignore background furniture, electronics, and environment.

CRITICAL RED FLAGS:
- POSTURE: Bladed stance, raised fists, crouching/stealth movement, combat-ready stance
- OBJECTS: Anything held in hands — blades, sticks, tools, weapons
- INTENT: Any gesture that is not passive standing or sitting

Return ONLY a raw JSON object with no markdown or explanation:
{
  "threat_level": <integer 0-10>,
  "behavior": "<one of: Hostile | Stealth | Weaponized | Suspicious | Normal>",
  "reason": "<specific physical evidence observed>"
}
"""

def run_groq_audit(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    prompt = get_security_prompt()

    for model_id in GROQ_MODELS:
        try:
            log.info(f"⚡ Trying Groq model: {model_id}")
            completion = groq_client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
                    ],
                }],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=256,
            )
            result = json.loads(completion.choices[0].message.content)
            result["source"] = f"groq:{model_id}"
            return result
        except Exception as e:
            log.warning(f"❌ {model_id} failed: {str(e)[:80]}")
            continue

    return {
        "threat_level": -1,
        "behavior": "Error",
        "reason": "All Groq models exhausted",
        "source": "groq:none",
    }


# ─── FUSION LOGIC ─────────────────────────────────────────────────────────────
def fuse_results(local_result: dict | None, api_result: dict | None) -> dict:
    """
    Fusion strategy:
    - If local is unavailable → use API only
    - If API is unavailable → use local only
    - If both available:
        • Either detects threat_level >= 6 → ALERT (conservative approach)
        • Both agree on normal (<4) → NORMAL
        • Disagreement → use higher threat_level, flag as 'DISPUTED'
    """
    if local_result is None and api_result is None:
        return {"threat_level": -1, "behavior": "Error", "reason": "No analysis completed", "fusion": "none"}

    if local_result is None:
        return {**api_result, "fusion": "api_only"}

    if api_result is None:
        return {**local_result, "fusion": "local_only"}

    local_tl = local_result.get("threat_level", 0)
    api_tl = api_result.get("threat_level", 0)

    if local_tl >= 6 or api_tl >= 6:
        dominant = local_result if local_tl >= api_tl else api_result
        return {
            **dominant,
            "threat_level": max(local_tl, api_tl),
            "fusion": "both_alert" if (local_tl >= 6 and api_tl >= 6) else "single_alert",
            "local_threat_level": local_tl,
            "api_threat_level": api_tl,
        }

    if abs(local_tl - api_tl) >= 4:
        dominant = local_result if local_tl >= api_tl else api_result
        return {
            **dominant,
            "threat_level": max(local_tl, api_tl),
            "fusion": "disputed",
            "local_threat_level": local_tl,
            "api_threat_level": api_tl,
        }

    avg_tl = round((local_tl + api_tl) / 2)
    dominant = local_result if local_tl >= api_tl else api_result
    return {
        **dominant,
        "threat_level": avg_tl,
        "fusion": "consensus",
        "local_threat_level": local_tl,
        "api_threat_level": api_tl,
    }


def classify_threat(threat_level: int) -> dict:
    if threat_level <= 0:
        return {"status": "CLEAR", "color": "green", "icon": "✅"}
    elif threat_level <= 3:
        return {"status": "LOW", "color": "yellow", "icon": "🟡"}
    elif threat_level <= 6:
        return {"status": "MODERATE", "color": "orange", "icon": "⚠️"}
    elif threat_level <= 8:
        return {"status": "HIGH", "color": "red", "icon": "🚨"}
    else:
        return {"status": "CRITICAL", "color": "darkred", "icon": "🔴"}


# ─── HISTORY ──────────────────────────────────────────────────────────────────
def save_record(record: dict):
    try:
        data = json.loads(LOG_FILE.read_text())
        data.append(record)
        # Keep last 10,000 records
        if len(data) > 10000:
            data = data[-10000:]
        LOG_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.error(f"Failed to save record: {e}")


def load_history(start_ts=None, end_ts=None, limit=None):
    try:
        data = json.loads(LOG_FILE.read_text())
        if start_ts:
            data = [r for r in data if r.get("timestamp_unix", 0) >= start_ts]
        if end_ts:
            data = [r for r in data if r.get("timestamp_unix", 0) <= end_ts]
        data = sorted(data, key=lambda r: r.get("timestamp_unix", 0), reverse=True)
        if limit:
            data = data[:limit]
        return data
    except Exception:
        return []


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_full_analysis(image_path: str, mode: str = "hybrid") -> dict:
    """
    mode: 'local' | 'api' | 'hybrid'
    hybrid = local first, escalate to API if threat >= 6 or local unavailable
    both = run both always and fuse
    """
    local_result = None
    api_result = None

    if mode in ("local", "hybrid", "both") and mediapipe_available:
        local_result = SecurityAI_instance.analyze_frame(image_path)

    if mode == "api":
        api_result = run_groq_audit(image_path)

    elif mode == "both":
        api_result = run_groq_audit(image_path)

    elif mode == "hybrid":
        # Run API if local is unavailable OR local detected a threat
        if local_result is None:
            api_result = run_groq_audit(image_path)
        elif local_result.get("threat_level", 0) >= 6:
            log.info("🔁 Local flagged threat — verifying with Groq API...")
            api_result = run_groq_audit(image_path)

    fused = fuse_results(local_result, api_result)
    classification = classify_threat(fused.get("threat_level", 0))

    return {
        "threat_level": fused.get("threat_level", 0),
        "behavior": fused.get("behavior", "Unknown"),
        "reason": fused.get("reason", ""),
        "status": classification["status"],
        "status_color": classification["color"],
        "status_icon": classification["icon"],
        "fusion_mode": fused.get("fusion", "unknown"),
        "local_threat_level": fused.get("local_threat_level"),
        "api_threat_level": fused.get("api_threat_level"),
        "local_source": local_result.get("source") if local_result else None,
        "api_source": api_result.get("source") if api_result else None,
        "mediapipe_available": mediapipe_available,
    }


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", mediapipe_available=mediapipe_available)


@app.route("/history-page")
def history_page():
    return render_template("history.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Upload endpoint.
    Accepts: multipart/form-data with 'image' file field
    Optional: 'mode' field = 'local' | 'api' | 'hybrid' | 'both'
    Returns: JSON analysis result
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    mode = request.form.get("mode", "hybrid")
    if mode not in ("local", "api", "hybrid", "both"):
        mode = "hybrid"

    # Save upload
    request_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc)
    safe_name = secure_filename(file.filename)
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{request_id}_{safe_name}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    start_time = time.time()
    try:
        result = run_full_analysis(str(save_path), mode=mode)
    except Exception as e:
        log.error(f"Analysis error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        elapsed = round(time.time() - start_time, 3)

    record = {
        "id": request_id,
        "timestamp": timestamp.isoformat(),
        "timestamp_unix": timestamp.timestamp(),
        "filename": filename,
        "original_filename": safe_name,
        "mode": mode,
        "processing_time_s": elapsed,
        **result,
    }

    save_record(record)

    return jsonify(record), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Query params:
    - start: unix timestamp (float)
    - end: unix timestamp (float)
    - limit: int (default 50, max 500)
    - last: int — return last N records (overrides start/end)
    """
    try:
        limit = min(int(request.args.get("limit", 50)), 500)
        last = request.args.get("last")
        if last:
            records = load_history(limit=int(last))
        else:
            start_ts = float(request.args.get("start", 0)) or None
            end_ts = float(request.args.get("end", 0)) or None
            records = load_history(start_ts=start_ts, end_ts=end_ts, limit=limit)
        return jsonify({"count": len(records), "records": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/last", methods=["GET"])
def api_last():
    """Return the single most recent record."""
    records = load_history(limit=1)
    if not records:
        return jsonify({"error": "No records found"}), 404
    return jsonify(records[0])


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "online",
        "mediapipe_available": mediapipe_available,
        "groq_models": GROQ_MODELS,
        "version": "1.0.0",
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "True"
    log.info(f"🚀 Sentinel starting on port {port} | MediaPipe: {mediapipe_available}")
    app.run(host="0.0.0.0", port=port, debug=debug)
