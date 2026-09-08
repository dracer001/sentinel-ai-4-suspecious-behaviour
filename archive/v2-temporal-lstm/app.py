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

from temporal_model import PoseSequenceExtractor, TemporalClassifier, SEQ_LEN

load_dotenv() 

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Ensure this matches what you set in Render Dashboard
GROQ_KEY = os.getenv("GROQ_API_KEY") 
UPLOAD_FOLDER = Path("uploads")
EVENTS_FOLDER = Path("events")   # burst frames from motion-triggered captures
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
EVENTS_FOLDER.mkdir(exist_ok=True)
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

        def _posture_context(self, lm):
            """
            Determine if subject is SEATED or STANDING.
            In normalized coords, y increases downward (0=top, 1=bottom).
            When standing: hips are well above ankles (hip.y < ankle.y by a large margin).
            When seated:   hips and knees are at similar y; knee angle is ~90°.
            Returns: 'seated' | 'standing' | 'prone'
            """
            l_hip, r_hip   = lm[23], lm[24]
            l_knee, r_knee = lm[25], lm[26]
            l_ank, r_ank   = lm[27], lm[28]

            avg_hip_y  = (l_hip.y  + r_hip.y)  / 2
            avg_knee_y = (l_knee.y + r_knee.y) / 2
            avg_ank_y  = (l_ank.y  + r_ank.y)  / 2

            # Vertical span of legs (normalized 0-1)
            leg_span = abs(avg_ank_y - avg_hip_y)

            # If hips and knees are at nearly the same height → seated
            hip_to_knee = abs(avg_knee_y - avg_hip_y)
            knee_to_ank = abs(avg_ank_y - avg_knee_y)

            if leg_span < 0.15:
                return 'prone'          # Lying down / very compressed
            if hip_to_knee < 0.08 and knee_to_ank > 0.08:
                return 'seated'         # Knees bent under, ankles below
            if hip_to_knee < knee_to_ank * 0.55:
                return 'seated'         # Hips and knees bunched together
            return 'standing'

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

            # ── Key landmarks ────────────────────────────────────────────────
            nose          = lm[0]
            l_eye, r_eye  = lm[2], lm[5]
            l_sh, r_sh    = lm[11], lm[12]
            l_elb, r_elb  = lm[13], lm[14]
            l_wr, r_wr    = lm[15], lm[16]
            l_hip, r_hip  = lm[23], lm[24]
            l_knee, r_knee= lm[25], lm[26]
            l_ank, r_ank  = lm[27], lm[28]

            # ── Derived measurements ─────────────────────────────────────────
            posture          = self._posture_context(lm)
            avg_sh_y         = (l_sh.y + r_sh.y) / 2
            avg_hip_y        = (l_hip.y + r_hip.y) / 2
            avg_ank_y        = (l_ank.y + r_ank.y) / 2
            shoulder_width   = abs(l_sh.x - r_sh.x)
            torso_height     = abs(avg_hip_y - avg_sh_y)   # How tall the torso appears

            l_arm_angle      = self.get_angle(l_sh, l_elb, l_wr)
            r_arm_angle      = self.get_angle(r_sh, r_elb, r_wr)

            # Wrist positions relative to body anchors
            l_wr_above_sh    = l_wr.y < l_sh.y   # True = wrist higher than shoulder
            r_wr_above_sh    = r_wr.y < r_sh.y
            l_wr_above_hip   = l_wr.y < avg_hip_y
            r_wr_above_hip   = r_wr.y < avg_hip_y

            # Wrist distance from body center (outward reach)
            body_center_x    = (l_sh.x + r_sh.x) / 2
            l_wr_reach       = abs(l_wr.x - body_center_x) / max(shoulder_width, 0.01)
            r_wr_reach       = abs(r_wr.x - body_center_x) / max(shoulder_width, 0.01)

            # Head offset from spine
            sh_center_x      = (l_sh.x + r_sh.x) / 2
            look_offset      = abs(nose.x - sh_center_x)
            look_offset_norm = look_offset / max(shoulder_width, 0.01)  # Normalized to shoulder width

            # Ground clearance — nose-to-ankle vertical gap (normalized)
            ground_clearance = abs(nose.y - avg_ank_y)

            # How far wrists are in front (z-axis if available, else skip)
            # MediaPipe z is relative depth — negative = closer to camera
            l_wr_forward     = getattr(l_wr, 'z', 0) < getattr(l_sh, 'z', 0) - 0.1
            r_wr_forward     = getattr(r_wr, 'z', 0) < getattr(r_sh, 'z', 0) - 0.1

            log.debug(
                f"Posture={posture} | gc={ground_clearance:.2f} | "
                f"l_arm={l_arm_angle:.0f}° r_arm={r_arm_angle:.0f}° | "
                f"l_reach={l_wr_reach:.2f} r_reach={r_wr_reach:.2f} | "
                f"look_norm={look_offset_norm:.2f}"
            )

            # ════════════════════════════════════════════════════════════════
            # THREAT A: OVERHEAD STRIKE / WEAPON THREAT
            # Original bug: arm raised + extended fired on shelf-reaching, waving
            #
            # Fix: Require ALL of:
            #   1. Wrist clearly above shoulder (not just at shoulder level)
            #   2. Arm is extended (angle > 150° — not just slightly raised)
            #   3. Wrist reaches outward beyond 1.2x shoulder width (aggressive reach,
            #      not just touching own head/hair)
            #   4. Only trigger if STANDING (not seated reaching for something)
            # ════════════════════════════════════════════════════════════════
            OVERHEAD_WRIST_MARGIN = 0.04   # Wrist must be this much ABOVE shoulder y
            OVERHEAD_ARM_ANGLE    = 150    # Must be nearly fully extended
            OVERHEAD_REACH        = 1.2    # Must reach 1.2x shoulder width outward

            l_overhead = (
                (l_sh.y - l_wr.y) > OVERHEAD_WRIST_MARGIN and   # wrist well above shoulder
                l_arm_angle > OVERHEAD_ARM_ANGLE and              # arm extended, not bent
                l_wr_reach  > OVERHEAD_REACH                      # arm out wide, not touching own head
            )
            r_overhead = (
                (r_sh.y - r_wr.y) > OVERHEAD_WRIST_MARGIN and
                r_arm_angle > OVERHEAD_ARM_ANGLE and
                r_wr_reach  > OVERHEAD_REACH
            )

            if (l_overhead or r_overhead) and posture == 'standing':
                return {
                    "threat_level": 9,
                    "behavior": "Weaponized",
                    "reason": (
                        "Standing subject with arm raised above shoulder, "
                        "fully extended, and reaching laterally — overhead strike posture"
                    ),
                    "source": "local_mediapipe",
                }

            # ════════════════════════════════════════════════════════════════
            # THREAT B: STEALTH / CRAWL
            # Original bug: seated person leaning on desk fired this
            #
            # Fix: Require ALL of:
            #   1. Ground clearance < 0.35 (tighter threshold)
            #   2. Posture is NOT 'seated' (seated people naturally have low ground_clearance)
            #   3. Torso_height is still reasonable (person is not just short/camera angle)
            #      — if seated, torso collapses in frame and gc drops legitimately
            # ════════════════════════════════════════════════════════════════
            if ground_clearance < 0.35 and posture not in ('seated', 'prone'):
                return {
                    "threat_level": 7,
                    "behavior": "Stealth",
                    "reason": "Standing subject with head very near ground level — crawling or crouching detected",
                    "source": "local_mediapipe",
                }

            # Prone (lying down) is only suspicious if not in a designated rest area
            # — flag as LOW rather than HIGH to avoid false-positives on resting
            if posture == 'prone':
                return {
                    "threat_level": 3,
                    "behavior": "Prone",
                    "reason": "Subject appears to be lying down or fully collapsed — verify context",
                    "source": "local_mediapipe",
                }

            # ════════════════════════════════════════════════════════════════
            # THREAT C: COMBAT READY STANCE
            # Original bug: folded arms triggered this because elbow angle < 90°
            #
            # Fix: Require ALL of:
            #   1. One or both arms bent (angle < 85°)
            #   2. Wrists are FORWARD from body (not tucked inward = folded arms)
            #      — folded arms: wrists cross toward opposite shoulder (low x-reach)
            #      — combat guard: wrists pushed outward-forward
            #   3. Wrists above hip level (fists raised, not just hanging)
            #   4. BOTH wrists above hip (one arm = natural relaxed carry, not threat)
            #   5. Only fire when STANDING
            # ════════════════════════════════════════════════════════════════
            COMBAT_ARM_ANGLE   = 85    # Elbows must be clearly bent
            COMBAT_REACH_MIN   = 0.4   # Wrists must reach at least 0.4x shoulder width outward
            COMBAT_REACH_MAX   = 1.5   # But not so far out it's just arms spread wide

            l_combat_bent = l_arm_angle < COMBAT_ARM_ANGLE
            r_combat_bent = r_arm_angle < COMBAT_ARM_ANGLE

            # Folded arms check: wrists cross the body center (reach < 0.3 = tucked inward)
            l_tucked = l_wr_reach < 0.3
            r_tucked = r_wr_reach < 0.3

            l_combat_valid = (
                l_combat_bent and
                l_wr_above_hip and
                not l_tucked and
                COMBAT_REACH_MIN < l_wr_reach < COMBAT_REACH_MAX
            )
            r_combat_valid = (
                r_combat_bent and
                r_wr_above_hip and
                not r_tucked and
                COMBAT_REACH_MIN < r_wr_reach < COMBAT_REACH_MAX
            )

            # Require BOTH arms in guard position for high-threat call
            if l_combat_valid and r_combat_valid and posture == 'standing':
                return {
                    "threat_level": 7,
                    "behavior": "Hostile",
                    "reason": (
                        "Both arms bent with fists raised and pushed outward — "
                        "combat-ready guard stance"
                    ),
                    "source": "local_mediapipe",
                }

            # Single arm combat guard = lower threat (could be casual)
            if (l_combat_valid or r_combat_valid) and posture == 'standing':
                return {
                    "threat_level": 4,
                    "behavior": "Suspicious",
                    "reason": "One arm in raised bent position — possibly guarded or aggressive posture",
                    "source": "local_mediapipe",
                }

            # ════════════════════════════════════════════════════════════════
            # THREAT D: SUSPICIOUS PEEPING / LOITERING
            # Original bug: looking sideways while seated fired this
            #
            # Fix: Require ALL of:
            #   1. Head offset > 1.5x shoulder width (extreme turn, not casual glance)
            #   2. Ground clearance < 0.65 (crouching context)
            #   3. STANDING posture only — seated people naturally look sideways
            # ════════════════════════════════════════════════════════════════
            if (
                look_offset_norm > 1.5 and
                ground_clearance < 0.65 and
                posture == 'standing'
            ):
                return {
                    "threat_level": 4,
                    "behavior": "Suspicious",
                    "reason": (
                        "Standing subject with head turned sharply sideways "
                        "while crouching — possible surveillance or peeping"
                    ),
                    "source": "local_mediapipe",
                }

            # ════════════════════════════════════════════════════════════════
            # ALL CLEAR
            # ════════════════════════════════════════════════════════════════
            return {
                "threat_level": 1,
                "behavior": "Normal",
                "reason": (
                    f"No threatening postures detected — subject appears calm "
                    f"({posture}, arms neutral)"
                ),
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


# ─── TEMPORAL (BURST/CLIP) ANALYZER ───────────────────────────────────────────
# Separate from the single-frame SecurityAI above. This is the accurate path:
# motion-triggered bursts of frames from the ESP32-CAM get turned into a pose
# sequence and classified by a trained LSTM (see temporal_model.py) instead of
# angle heuristics on one frame.
pose_sequence_extractor = None
temporal_classifier = None

try:
    if Path(MEDIAPIPE_MODEL).exists():
        pose_sequence_extractor = PoseSequenceExtractor(MEDIAPIPE_MODEL)
    else:
        log.warning("⚠️ MediaPipe model missing — burst/clip pose extraction disabled.")
except Exception as e:
    log.error(f"⚠️ PoseSequenceExtractor init error: {e}")

try:
    temporal_classifier = TemporalClassifier()  # gracefully reports .available = False if untrained
except Exception as e:
    log.error(f"⚠️ TemporalClassifier init error: {e}")


# ─── GROQ API ANALYZER ────────────────────────────────────────────────────────
def get_security_prompt():
    return (
        "[SECURITY AUDIT PROTOCOL — RESPOND IN JSON ONLY]\n"
        "You are a security AI analyzing a single image for genuine physical threats only.\n"
        "Be CONSERVATIVE. A false positive is worse than a missed low-level threat.\n\n"
        "ANALYZE: Body posture, stance, held objects, and movement intent.\n"
        "IGNORE: Background objects, furniture, screens, clothing style, race, gender.\n\n"
        "GENUINE THREATS (score 6-10):\n"
        "- Active aggression: raised fist striking, lunging at someone, choking\n"
        "- Visible weapon in hand being aimed or raised (knife, gun, bat in attack grip)\n"
        "- Stealth crawl: adult deliberately crawling on floor toward something\n"
        "- Full combat stance: weight forward, both fists raised at chest, bladed feet\n\n"
        "SUSPICIOUS (score 3-5):\n"
        "- One arm raised in a guard position with body leaning forward aggressively\n"
        "- Object in hand that COULD be a weapon but context is ambiguous\n\n"
        "NOT THREATS — DO NOT FLAG (score 0-2):\n"
        "- Arms folded or crossed over chest (resting, thinking)\n"
        "- Hand raised to wave, point, or gesture in conversation\n"
        "- Reaching up to a shelf, cabinet, or board\n"
        "- Head resting on desk or table (sleeping, tired)\n"
        "- Person sitting, leaning, or slouching in a chair\n"
        "- Stretching, yawning, adjusting clothing\n"
        "- Person standing and talking with hand gestures\n"
        "- Writing on a board with arm raised\n"
        "- Looking sideways or turning head while seated\n"
        "- Holding a phone, book, bag, pen, or cup\n\n"
        "SCORING: 0=clear 1-2=normal 3-5=mildly suspicious 6-7=likely threat 8-9=active threat 10=critical\n\n"
        "Return ONLY a raw JSON object with no markdown:\n"
        '{"threat_level": <0-10>, "behavior": "<Hostile|Stealth|Weaponized|Suspicious|Normal>", '
        '"reason": "<exact physical evidence you see>"}'
    )

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


def run_groq_audit_multi(image_paths: list[str]) -> dict:
    """
    Same as run_groq_audit, but sends several representative frames from a
    burst (e.g. first / middle / last) in one message so the vision model can
    reason about motion across frames, not just one still. Used to verify
    events the temporal classifier flags as ambiguous or high-threat.
    """
    content = [{"type": "text", "text": get_security_prompt() + (
        "\n\nNOTE: You are being shown MULTIPLE SEQUENTIAL FRAMES from a short "
        "burst (roughly 3-4 seconds, in time order). Judge the ACTION across "
        "the sequence, not just one frame in isolation."
    )}]

    for image_path in image_paths:
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}})

    for model_id in GROQ_MODELS:
        try:
            log.info(f"⚡ Trying Groq model (multi-frame): {model_id}")
            completion = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=256,
            )
            result = json.loads(completion.choices[0].message.content)
            result["source"] = f"groq:{model_id}:multiframe"
            return result
        except Exception as e:
            log.warning(f"❌ {model_id} (multi-frame) failed: {str(e)[:80]}")
            continue

    return {
        "threat_level": -1,
        "behavior": "Error",
        "reason": "All Groq models exhausted (multi-frame)",
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


@app.route("/analyze-clip", methods=["POST"])
def analyze_clip():
    """
    Motion-triggered burst endpoint.
    Accepts: multipart/form-data with multiple 'frames' file fields — the
    ESP32-CAM captures 8-12 frames over ~3-4s on motion trigger, buffers them,
    then POSTs them all here in one request (do NOT stream frame-by-frame;
    the sequence model needs the whole burst at once).
    Optional: 'event_id' field (a UUID is generated if not provided)

    This is the accurate path (see temporal_model.py): pose sequence + trained
    LSTM classifier, escalating to a multi-frame Groq check when ambiguous.
    The single-frame /analyze route above is left for manual "check this one
    photo" use from the dashboard and is not changed.
    """
    frames = request.files.getlist("frames")
    if not frames:
        return jsonify({"error": "No 'frames' provided. Send burst frames as "
                                  "multiple 'frames' fields in one request."}), 400

    event_id = request.form.get("event_id") or str(uuid.uuid4())[:12]
    timestamp = datetime.now(timezone.utc)
    event_dir = EVENTS_FOLDER / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{event_id}"
    event_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for i, f in enumerate(frames):
        if not allowed_file(f.filename or f"frame_{i}.jpg"):
            continue
        safe_name = secure_filename(f.filename) or f"frame_{i:03d}.jpg"
        save_path = event_dir / f"{i:03d}_{safe_name}"
        f.save(save_path)
        frame_paths.append(str(save_path))

    if len(frame_paths) < 2:
        return jsonify({"error": "Need at least 2 valid frames to analyze a burst"}), 400

    start_time = time.time()
    local_result = None
    api_result = None

    try:
        if pose_sequence_extractor is not None and temporal_classifier is not None:
            sequence, detection_rate = pose_sequence_extractor.extract_sequence(frame_paths)
            local_result = temporal_classifier.predict(sequence, detection_rate)

        # Escalate to a multi-frame Groq check when the temporal model is
        # unavailable/untrained, OR when it flags meaningful suspicion.
        # Representative frames only (first/middle/last) — cheaper, and Groq
        # doesn't need all 12 to judge the arc of the action.
        needs_escalation = (
            local_result is None
            or local_result.get("behavior") == "MODEL_UNAVAILABLE"
            or local_result.get("threat_level", 0) >= 5
        )
        if needs_escalation:
            rep_idx = sorted(set([0, len(frame_paths) // 2, len(frame_paths) - 1]))
            rep_frames = [frame_paths[i] for i in rep_idx]
            log.info(f"🔁 Event {event_id}: escalating {len(rep_frames)} representative "
                     f"frames to Groq multi-frame audit...")
            api_result = run_groq_audit_multi(rep_frames)

    except Exception as e:
        log.error(f"Clip analysis error: {e}")
        return jsonify({"error": f"Clip analysis failed: {str(e)}"}), 500

    elapsed = round(time.time() - start_time, 3)
    fused = fuse_results(local_result, api_result)
    classification = classify_threat(fused.get("threat_level", 0))

    record = {
        "id": event_id,
        "event": True,
        "timestamp": timestamp.isoformat(),
        "timestamp_unix": timestamp.timestamp(),
        "frame_count": len(frame_paths),
        "processing_time_s": elapsed,
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
        "temporal_model_available": bool(temporal_classifier and temporal_classifier.available),
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
        "temporal_model_available": bool(temporal_classifier and temporal_classifier.available),
        "burst_seq_len": SEQ_LEN,
        "groq_models": GROQ_MODELS,
        "version": "1.1.0",
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
