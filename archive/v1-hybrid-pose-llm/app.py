"""
SENTINEL app.py
===============
Flask web server. Handles file uploads, routing, history.
All AI logic is in analyzer.py.
"""

import os, json, uuid, time, logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from analyzer import analyze_image, mediapipe_available, GEMINI_API_KEY, GROQ_API_KEY

# ─── SETUP ───────────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = Path("uploads")
LOG_FILE           = Path("logs/history.json")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_UPLOAD_BYTES   = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)

UPLOAD_FOLDER.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)
if not LOG_FILE.exists():
    LOG_FILE.write_text("[]")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ─── HISTORY HELPERS ─────────────────────────────────────────────────────────

def save_record(record: dict):
    try:
        data = json.loads(LOG_FILE.read_text())
        data.append(record)
        if len(data) > 10_000:
            data = data[-10_000:]
        LOG_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.error(f"save_record: {e}")


def load_history(start_ts=None, end_ts=None, limit=50,
                 scene_type=None, min_threat=None):
    try:
        data = json.loads(LOG_FILE.read_text())

        if start_ts:
            data = [r for r in data if r.get("timestamp_unix", 0) >= start_ts]
        if end_ts:
            data = [r for r in data if r.get("timestamp_unix", 0) <= end_ts]
        if scene_type:
            data = [r for r in data if r.get("scene_type") == scene_type]
        if min_threat is not None:
            data = [r for r in data if r.get("threat_level", 0) >= min_threat]

        data = sorted(data, key=lambda r: r.get("timestamp_unix", 0), reverse=True)
        return data[:limit]
    except Exception:
        return []


def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        mediapipe_available=mediapipe_available,
        gemini_available=bool(GEMINI_API_KEY),
        groq_available=bool(GROQ_API_KEY),
    )


@app.route("/history-page")
def history_page():
    return render_template("history.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Form fields:
      image  — image file (required)
      mode   — 'hybrid' | 'api' | 'local'  (default: hybrid)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    mode = request.form.get("mode", "hybrid")
    if mode not in ("local", "api", "hybrid"):
        mode = "hybrid"

    # Save upload
    request_id = str(uuid.uuid4())[:8]
    ts         = datetime.now(timezone.utc)
    safe_name  = secure_filename(file.filename)
    filename   = f"{ts.strftime('%Y%m%d_%H%M%S')}_{request_id}_{safe_name}"
    save_path  = UPLOAD_FOLDER / filename
    file.save(save_path)

    # Run analysis
    t0 = time.time()
    try:
        result = analyze_image(str(save_path), mode=mode)
    except Exception as e:
        log.exception("analyze_image failed")
        return jsonify({"error": str(e)}), 500
    elapsed = round(time.time() - t0, 3)

    # Build record
    record = {
        "id":                request_id,
        "timestamp":         ts.isoformat(),
        "timestamp_unix":    ts.timestamp(),
        "filename":          filename,
        "original_filename": safe_name,
        "mode":              mode,
        "processing_time_s": elapsed,
        **result,
    }
    save_record(record)
    return jsonify(record), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    GET /api/history
    Query params:
      last        — return last N records (overrides start/end)
      start       — unix timestamp float
      end         — unix timestamp float
      limit       — max records (default 50, max 500)
      scene_type  — filter by scene type
      min_threat  — filter by minimum threat level
    """
    try:
        limit      = min(int(request.args.get("limit", 50)), 500)
        last       = request.args.get("last")
        scene_type = request.args.get("scene_type")
        min_threat = request.args.get("min_threat")
        if min_threat is not None:
            min_threat = int(min_threat)

        if last:
            records = load_history(limit=int(last), scene_type=scene_type,
                                   min_threat=min_threat)
        else:
            start_ts = float(request.args.get("start", 0)) or None
            end_ts   = float(request.args.get("end",   0)) or None
            records  = load_history(start_ts=start_ts, end_ts=end_ts,
                                    limit=limit, scene_type=scene_type,
                                    min_threat=min_threat)

        return jsonify({"count": len(records), "records": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/last", methods=["GET"])
def api_last():
    """GET /api/history/last — single most recent record."""
    records = load_history(limit=1)
    if not records:
        return jsonify({"error": "No records found"}), 404
    return jsonify(records[0])


@app.route("/api/status", methods=["GET"])
def api_status():
    """GET /api/status — system health and engine availability."""
    return jsonify({
        "status":              "online",
        "version":             "2.0.0",
        "mediapipe_available": mediapipe_available,
        "gemini_available":    bool(GEMINI_API_KEY),
        "groq_available":      bool(GROQ_API_KEY),
        "engines": {
            "primary":   "gemini-2.5-flash" if GEMINI_API_KEY else "none",
            "backup":    "groq-llama4-scout" if GROQ_API_KEY else "none",
            "local":     "mediapipe" if mediapipe_available else "none",
        }
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    log.info("╔══════════════════════════════════╗")
    log.info("║  SENTINEL v2.0 — Starting        ║")
    log.info("╚══════════════════════════════════╝")
    log.info(f"  MediaPipe : {'✅' if mediapipe_available else '❌'}")
    log.info(f"  Gemini    : {'✅' if GEMINI_API_KEY else '❌ (set GEMINI_API_KEY)'}")
    log.info(f"  Groq      : {'✅' if GROQ_API_KEY else '❌ (set GROQ_API_KEY)'}")
    log.info(f"  Port      : {port}")

    app.run(host="0.0.0.0", port=port, debug=debug)
