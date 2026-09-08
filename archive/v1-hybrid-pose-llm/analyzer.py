"""
SENTINEL analyzer.py
====================
All AI analysis logic in one place.

Engine priority:
  1. MediaPipe  — always runs locally (free, fast, no internet needed)
                  contributes pose context to the LLM prompt
  2. Gemini 2.5 Flash — primary vision LLM (free 1,500 req/day, no card)
  3. Groq Llama 4 Scout — backup (free 1,000 req/day, no card)

Scene types detected:
  fighting, stealing, weapon_present, snooping,
  vandalism, loitering, threatening, normal
"""

import os
import json
import base64
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ─── LOAD .env IF PRESENT ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE 1 — MEDIAPIPE  (local, always free, no internet)
# ═══════════════════════════════════════════════════════════════════════════════

mediapipe_available = False
_pose_detector      = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import numpy as np

    _MODEL_PATH = os.environ.get("MEDIAPIPE_MODEL_PATH", "pose_landmarker_heavy.task")

    if Path(_MODEL_PATH).exists():
        _opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            num_poses=4,
        )
        _pose_detector      = mp_vision.PoseLandmarker.create_from_options(_opts)
        mediapipe_available = True
        log.info("✅ MediaPipe loaded — up to 4 subjects")
    else:
        log.warning(f"MediaPipe model not found at '{_MODEL_PATH}'")

except ImportError:
    log.info("MediaPipe not installed — local pose disabled")
except Exception as e:
    log.error(f"MediaPipe init error: {e}")


def _mediapipe_context(image_path: str) -> str:
    """
    Run pose detection. Returns a plain-English description of what's
    happening physically — fed into the LLM prompt as extra context.
    Returns "" if unavailable.
    """
    if not mediapipe_available or _pose_detector is None:
        return ""
    try:
        image  = mp.Image.create_from_file(image_path)
        result = _pose_detector.detect(image)

        if not result.pose_landmarks:
            return "Pose sensor: no humans detected."

        n = len(result.pose_landmarks)
        parts = []

        for i, lm in enumerate(result.pose_landmarks):
            nose             = lm[0]
            l_sh, r_sh       = lm[11], lm[12]
            l_elb, r_elb     = lm[13], lm[14]
            l_wr, r_wr       = lm[15], lm[16]
            l_hip, r_hip     = lm[23], lm[24]
            l_knee, r_knee   = lm[25], lm[26]
            l_ank, r_ank     = lm[27], lm[28]

            avg_sh_y   = (l_sh.y  + r_sh.y)  / 2
            avg_hip_y  = (l_hip.y + r_hip.y) / 2
            avg_ank_y  = (l_ank.y + r_ank.y) / 2
            sh_width   = max(abs(l_sh.x - r_sh.x), 0.01)
            knee_avg_y = (l_knee.y + r_knee.y) / 2

            # Posture
            leg_span     = abs(avg_ank_y - avg_hip_y)
            hip_knee_gap = abs(avg_hip_y - knee_avg_y)
            if leg_span < 0.15:
                posture = "prone/lying down"
            elif hip_knee_gap < 0.08:
                posture = "seated"
            else:
                posture = "standing"

            # Arm descriptors
            arms = []
            if l_wr.y < avg_sh_y - 0.05:
                arms.append("left arm raised above shoulder")
            if r_wr.y < avg_sh_y - 0.05:
                arms.append("right arm raised above shoulder")

            l_reach = abs(l_wr.x - (l_sh.x + r_sh.x) / 2) / sh_width
            r_reach = abs(r_wr.x - (l_sh.x + r_sh.x) / 2) / sh_width
            if l_reach < 0.25 and r_reach < 0.25:
                arms.append("arms tucked close/folded")
            elif (l_reach > 1.5 or r_reach > 1.5):
                arms.append("arm(s) extended wide outward")

            # Arm angles
            def angle(a, b, c):
                import numpy as np
                ba = np.array([a.x - b.x, a.y - b.y])
                bc = np.array([c.x - b.x, c.y - b.y])
                n  = np.linalg.norm(ba) * np.linalg.norm(bc)
                return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1, 1))) if n else 0

            l_angle = angle(l_sh, l_elb, l_wr)
            r_angle = angle(r_sh, r_elb, r_wr)
            if l_angle < 80 or r_angle < 80:
                arms.append("elbows sharply bent")

            # Head
            head_notes = []
            gc = abs(nose.y - avg_ank_y)
            if gc < 0.25:
                head_notes.append("head near floor level")
            elif nose.y > avg_sh_y + 0.08:
                head_notes.append("head bent downward")

            arm_str  = ", ".join(arms) if arms else "arms in neutral position"
            head_str = (", " + ", ".join(head_notes)) if head_notes else ""
            parts.append(f"Person {i+1}: {posture}{head_str}, {arm_str}")

        return f"Pose sensor — {n} person(s): " + "; ".join(parts) + "."

    except Exception as e:
        log.warning(f"MediaPipe context error: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(pose_context: str) -> str:
    pose_section = (
        f"\n\nPose sensor data (use as supporting evidence, not sole verdict):\n{pose_context}"
        if pose_context else ""
    )
    return (
        "You are a security camera AI. Analyze this image and identify exactly "
        "what is happening. Be precise about what you SEE, not what you guess.\n\n"

        "SCENE TYPES — pick exactly one:\n"
        "  fighting     — active physical assault: punching, choking, wrestling violently\n"
        "  stealing     — taking objects, pickpocketing, grabbing belongings covertly\n"
        "  weapon_present — knife, firearm, bat, or dangerous tool visibly held/displayed\n"
        "  snooping     — going through drawers/bags/files/screens without permission\n"
        "  vandalism    — breaking, scratching, spray-painting, damaging property\n"
        "  loitering    — suspicious lingering near sensitive area with no clear purpose\n"
        "  threatening  — aggressive confrontation, intimidation, without contact yet\n"
        "  normal       — working, walking, talking, sitting — any everyday activity\n\n"

        "THREAT LEVEL 0-10:\n"
        "  0-1  Clearly normal — people working, talking, laughing together\n"
        "  2-3  Slightly unusual but almost certainly innocent\n"
        "  4-5  Genuinely suspicious — warrants attention\n"
        "  6-7  Likely threat — strong reason for concern\n"
        "  8-9  Active threat — immediate action needed\n"
        "  10   Critical emergency\n\n"

        "CRITICAL — DO NOT FLAG THESE AS THREATS:\n"
        "  • Two or more people laughing, smiling, or appear relaxed = normal\n"
        "  • Friendly physical contact, hugging, leaning on someone = normal\n"
        "  • Person leaning over desk/table to look at something = normal\n"
        "  • Someone bending down to pick something up = normal\n"
        "  • Person walking through a room = normal\n"
        "  • Hand gestures during conversation = normal\n"
        "  • Person using a computer, phone, or any work device = normal\n"
        "  • Arms crossed or folded = normal (resting posture)\n\n"

        "FLAG THESE:\n"
        "  • Someone hitting, choking, or wrestling aggressively with another person\n"
        "  • Someone covertly taking an object that belongs to someone else\n"
        "  • A weapon (knife, gun, bat) being actively held or brandished\n"
        "  • Someone rifling through another person's belongings without permission\n"
        "  • Someone clearly damaging property\n\n"

        "CONFIDENCE: Rate how clear the image is and how certain you are.\n"
        "  high   — image clear, action unambiguous\n"
        "  medium — image ok, some ambiguity\n"
        "  low    — image blurry/dark, or action unclear\n"
        + pose_section + "\n\n"

        "Return ONLY valid JSON — no markdown, no explanation, nothing else:\n"
        '{\n'
        '  "threat_level": <integer 0-10>,\n'
        '  "scene_type": "<scene type from list above>",\n'
        '  "behavior": "<one of: Normal|Suspicious|Threatening|Hostile|Stealing|Snooping|Vandalism>",\n'
        '  "subjects": <number of people visible>,\n'
        '  "reason": "<1-2 sentences: exactly what you see happening>",\n'
        '  "confidence": "<high|medium|low>"\n'
        '}'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE 2 — GEMINI 2.5 FLASH (primary, 1500 free req/day)
# ═══════════════════════════════════════════════════════════════════════════════

_gemini_client = None

def _get_gemini():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_client = genai.GenerativeModel("gemini-2.5-flash")
            log.info("✅ Gemini 2.5 Flash ready")
        except Exception as e:
            log.error(f"Gemini init failed: {e}")
    return _gemini_client


def _run_gemini(image_path: str, prompt: str) -> dict | None:
    """Returns parsed dict or None on failure."""
    client = _get_gemini()
    if client is None:
        return None

    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        with open(image_path, "rb") as f:
            image_data = f.read()

        ext  = Path(image_path).suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        image_part = {"mime_type": mime, "data": image_data}

        response = client.generate_content(
            [prompt, image_part],
            generation_config={
                "temperature":     0.1,
                "max_output_tokens": 400,
                "response_mime_type": "application/json",
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        raw = response.text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        data["source"] = "gemini:gemini-2.5-flash"
        return data

    except Exception as e:
        log.warning(f"Gemini failed: {str(e)[:100]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE 3 — GROQ (backup, 1000 free req/day for vision models)
# ═══════════════════════════════════════════════════════════════════════════════

_groq_client = None

# Vision-capable models on Groq, in priority order
GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]


def _get_groq():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
            log.info("✅ Groq client ready")
        except Exception as e:
            log.error(f"Groq init failed: {e}")
    return _groq_client


def _run_groq(image_path: str, prompt: str) -> dict | None:
    """Returns parsed dict or None on failure."""
    client = _get_groq()
    if client is None:
        return None

    ext  = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    for model in GROQ_VISION_MODELS:
        try:
            log.info(f"Trying Groq model: {model}")
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400,
            )
            raw  = resp.choices[0].message.content
            data = json.loads(raw)
            data["source"] = f"groq:{model}"
            return data
        except Exception as e:
            log.warning(f"Groq {model} failed: {str(e)[:80]}")
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  FUSION + CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _classify(threat_level: int) -> dict:
    if threat_level <= 1:
        return {"status": "CLEAR",    "color": "green",   "icon": "✅"}
    elif threat_level <= 3:
        return {"status": "LOW",      "color": "yellow",  "icon": "🟡"}
    elif threat_level <= 5:
        return {"status": "MODERATE", "color": "orange",  "icon": "⚠️"}
    elif threat_level <= 7:
        return {"status": "HIGH",     "color": "red",     "icon": "🚨"}
    else:
        return {"status": "CRITICAL", "color": "darkred", "icon": "🔴"}


def _ensure_fields(data: dict, source: str) -> dict:
    """Fill in any missing fields with safe defaults."""
    data.setdefault("threat_level", 0)
    data.setdefault("scene_type",   "unknown")
    data.setdefault("behavior",     "Normal")
    data.setdefault("subjects",     1)
    data.setdefault("reason",       "No details available")
    data.setdefault("confidence",   "low")
    data.setdefault("source",       source)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_image(image_path: str, mode: str = "hybrid") -> dict:
    """
    Full analysis pipeline.

    mode = 'hybrid'  — MediaPipe always; LLM only if pose flags suspicion (saves quota)
    mode = 'api'     — MediaPipe for context + always call LLM
    mode = 'local'   — MediaPipe only, no LLM calls

    Returns a flat dict ready to be JSON-serialised and returned to the client.
    """
    engines_used  = []
    pose_context  = ""
    llm_result    = None

    # ── Step 1: MediaPipe pose context (always, free) ─────────────────────────
    if mediapipe_available and mode != "api":
        pose_context = _mediapipe_context(image_path)
        if pose_context:
            engines_used.append("mediapipe")
        log.info(f"Pose: {pose_context[:80]}")

    # ── Step 2: Decide whether to call LLM ───────────────────────────────────
    should_call_llm = True
    if mode == "local":
        should_call_llm = False
    elif mode == "hybrid" and mediapipe_available:
        # In hybrid mode, skip LLM for clearly normal poses to save quota.
        # Only call LLM if:
        #   - pose context says something unusual, OR
        #   - MediaPipe detected no subjects (LLM might see something it missed)
        normal_indicators = [
            "arms tucked close/folded",
            "arms in neutral position",
            "seated",
        ]
        unusual_indicators = [
            "raised above shoulder",
            "prone",
            "extended wide",
            "elbows sharply bent",
            "head near floor",
            "no humans detected",
        ]
        has_unusual = any(ind in pose_context for ind in unusual_indicators)
        has_normal  = any(ind in pose_context for ind in normal_indicators)

        if has_normal and not has_unusual:
            # Looks calm — skip LLM, but still allow it for multi-person scenes
            # (two people could be fighting even with neutral pose readings due to camera angle)
            if "Person 2" not in pose_context:
                should_call_llm = False
                log.info("Hybrid: pose looks normal, skipping LLM")

    # ── Step 3: LLM analysis (Gemini → Groq fallback) ────────────────────────
    if should_call_llm:
        prompt = _build_prompt(pose_context)

        # Primary: Gemini 2.5 Flash
        if GEMINI_API_KEY:
            log.info("Calling Gemini 2.5 Flash...")
            llm_result = _run_gemini(image_path, prompt)
            if llm_result:
                engines_used.append(llm_result.get("source", "gemini"))

        # Backup: Groq (if Gemini failed or no key)
        if llm_result is None and GROQ_API_KEY:
            log.info("Gemini unavailable — falling back to Groq...")
            llm_result = _run_groq(image_path, prompt)
            if llm_result:
                engines_used.append(llm_result.get("source", "groq"))

        if llm_result is None:
            log.warning("All LLM engines failed")

    # ── Step 4: Build final result ────────────────────────────────────────────
    if llm_result:
        result = _ensure_fields(llm_result, llm_result.get("source", "unknown"))
    elif mediapipe_available and pose_context:
        # LLM not called or failed — derive minimal result from pose only
        result = _pose_fallback_result(pose_context)
        engines_used.append("mediapipe_only")
    else:
        # Nothing worked
        result = {
            "threat_level": -1,
            "scene_type":   "error",
            "behavior":     "Error",
            "subjects":     0,
            "reason":       "No analysis engines available. Check API keys and MediaPipe model.",
            "confidence":   "low",
            "source":       "none",
        }

    # ── Step 5: Enrich and return ─────────────────────────────────────────────
    tl  = max(result.get("threat_level", 0), 0)
    cls = _classify(tl)
    result.update({
        "status":              cls["status"],
        "status_color":        cls["color"],
        "status_icon":         cls["icon"],
        "engines_used":        engines_used,
        "pose_context":        pose_context,
        "mediapipe_available": mediapipe_available,
        "gemini_available":    bool(GEMINI_API_KEY),
        "groq_available":      bool(GROQ_API_KEY),
        "mode":                mode,
    })
    return result


def _pose_fallback_result(pose_context: str) -> dict:
    """
    Minimal threat assessment when only MediaPipe ran (no LLM available).
    Very conservative — only flags obvious geometric threats.
    """
    tl = 1
    reason = "Pose-only analysis (no LLM). " + pose_context

    if "raised above shoulder" in pose_context and "extended wide" in pose_context:
        tl = 5
        reason = "Pose sensor: arm raised and extended. LLM not available for confirmation."
    elif "prone/lying down" in pose_context:
        tl = 3
        reason = "Pose sensor: subject appears prone/lying down. Verify context."
    elif "Person 2" in pose_context:
        tl = 2
        reason = "Pose sensor: multiple subjects detected. LLM not available for scene analysis."

    return {
        "threat_level": tl,
        "scene_type":   "unknown",
        "behavior":     "Suspicious" if tl >= 4 else "Normal",
        "subjects":     pose_context.count("Person"),
        "reason":       reason,
        "confidence":   "low",
        "source":       "mediapipe_only",
    }
