"""
SENTINEL analyzer.py
=====================
Behavior analysis engine, rebuilt on the CLIP-embedding system
(see the behavior-threat-detection project: embed_dataset.py / infer_single.py).

No external API calls (no Gemini/Groq) and no MediaPipe pose detection.
Instead:
  1. The uploaded image is embedded with a pretrained CLIP model.
  2. It's compared against your labeled dataset via nearest-neighbor
     (mode="nn") or a trained classifier (mode="classifier").
  3. A threat score/level is computed from the predicted class + confidence.

Requires the models/ folder from the behavior-threat-detection project
(embeddings.npy, labels.json, and optionally classifier.joblib) to be
present relative to wherever this app is run from -- or point MODELS_DIR
at it via the MODELS_DIR environment variable.
"""

import os
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))

# Kept for template/route backward-compatibility (index.html previously
# showed MediaPipe/Gemini/Groq badges -- these are no longer used for
# analysis but the names are kept so existing imports in app.py don't break
# if you haven't updated it yet).
mediapipe_available = False
GEMINI_API_KEY = None
GROQ_API_KEY = None

# New availability flags the updated app.py/index.html actually check.
embeddings_available = (MODELS_DIR / "embeddings.npy").exists()
classifier_available = (MODELS_DIR / "classifier.joblib").exists()

THREAT_WEIGHTS = {
    "normal":   0.0,
    "peeping":  0.5,
    "stealing": 0.7,
    "fighting": 0.9,
}

# (low, high, internal_level, display_status)
THREAT_LEVELS = [
    (0.00, 0.15, "none",     "CLEAR"),
    (0.15, 0.40, "low",      "LOW"),
    (0.40, 0.65, "medium",   "MODERATE"),
    (0.65, 0.85, "high",     "HIGH"),
    (0.85, 1.01, "critical", "CRITICAL"),
]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


# ─── CLIP LOADING / EMBEDDING ────────────────────────────────────────────────

def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def _unwrap_features(features):
    """
    Some transformers versions return a plain tensor from get_image_features(),
    others return a wrapped output object. Normalize either case to a tensor.
    """
    if torch.is_tensor(features):
        return features
    if hasattr(features, "image_embeds"):
        return features.image_embeds
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    raise TypeError(
        f"Unexpected output type from get_image_features(): {type(features)}"
    )


def _embed_image(image_path):
    model, processor = _load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = _unwrap_features(features)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).cpu().numpy()


# ─── PREDICTION ──────────────────────────────────────────────────────────────

def _predict_nn(query_embedding, top_k=5):
    if not (MODELS_DIR / "embeddings.npy").exists():
        raise FileNotFoundError(
            f"No dataset embeddings found at {MODELS_DIR}/embeddings.npy. "
            "Run embed_dataset.py first."
        )
    embeddings = np.load(MODELS_DIR / "embeddings.npy")
    with open(MODELS_DIR / "labels.json") as f:
        labels = json.load(f)

    sims = embeddings @ query_embedding
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_labels = [labels[i] for i in top_idx]
    top_sims = [float(sims[i]) for i in top_idx]

    scores = {}
    for lbl, sim in zip(top_labels, top_sims):
        scores[lbl] = scores.get(lbl, 0.0) + max(sim, 0.0)

    predicted_class = max(scores, key=scores.get)
    confidence = scores[predicted_class] / sum(scores.values())
    breakdown = list(zip(top_labels, [round(s, 3) for s in top_sims]))
    return predicted_class, confidence, breakdown


def _predict_classifier(query_embedding):
    clf_path = MODELS_DIR / "classifier.joblib"
    if not clf_path.exists():
        raise FileNotFoundError(
            f"No trained classifier found at {clf_path}. "
            "Run train_classifier.py first, or use mode='nn'."
        )
    clf = joblib.load(clf_path)
    probs = clf.predict_proba([query_embedding])[0]
    classes = clf.classes_
    best_idx = int(np.argmax(probs))
    breakdown = list(zip(classes.tolist(), [round(float(p), 3) for p in probs]))
    return classes[best_idx], float(probs[best_idx]), breakdown


def _compute_threat(predicted_class, confidence, context_multiplier=1.0):
    base_weight = THREAT_WEIGHTS.get(predicted_class, 0.5)
    score = min(base_weight * confidence * context_multiplier, 1.0)

    level, status = "none", "CLEAR"
    for low, high, lvl_name, status_name in THREAT_LEVELS:
        if low <= score < high:
            level, status = lvl_name, status_name
            break

    return round(score, 3), level, status


# ─── PUBLIC ENTRY POINT ──────────────────────────────────────────────────────

def analyze_image(image_path, mode="nn", context_multiplier=1.0):
    """
    Analyze a single image, returning a dict matching the record schema
    that app.py / index.html / history.html expect.

    mode:
      "nn"          nearest-neighbor against your dataset embeddings (default,
                    works with as few as a handful of labeled images/class)
      "classifier"  trained Logistic Regression on top of the embeddings
                    (needs train_classifier.py to have been run first)
    """
    if mode not in ("nn", "classifier"):
        mode = "nn"

    query_embedding = _embed_image(image_path)

    if mode == "classifier":
        predicted_class, confidence, breakdown = _predict_classifier(query_embedding)
        engine = "clip-classifier"
    else:
        predicted_class, confidence, breakdown = _predict_nn(query_embedding)
        engine = "clip-nearest-neighbor"

    score, level, status = _compute_threat(predicted_class, confidence, context_multiplier)
    threat_level_10 = round(score * 10)
    breakdown_str = ", ".join(f"{cls}: {p:.2f}" for cls, p in breakdown)

    return {
        "scene_type":   predicted_class,
        "behavior":     f"Detected pattern consistent with '{predicted_class}'",
        "threat_level": threat_level_10,
        "status":       status,
        "confidence":   f"{confidence * 100:.1f}%",
        "engines_used": [engine],
        "subjects":     None,
        "reason":       f"Class scores — {breakdown_str}",
        "pose_context": "N/A",
        "source":       "upload",
    }
