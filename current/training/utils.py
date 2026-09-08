"""
Shared utilities: CLIP model loading, image embedding, and threat-level scoring.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
# You can swap this for a larger CLIP variant if you have the compute/VRAM:
#   "openai/clip-vit-base-patch32"   -> fastest, good starting point (default)
#   "openai/clip-vit-large-patch14"  -> more accurate, slower, needs more RAM/VRAM
#   "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" -> trained on a larger dataset (LAION-2B)
MODEL_NAME = "openai/clip-vit-base-patch32"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


def load_model():
    """Lazy-load CLIP model + processor once, reuse across calls."""
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def embed_image(image_path):
    """Return a normalized CLIP embedding (numpy array) for a single image."""
    model, processor = load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = _unwrap_features(features)
    features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
    return features.squeeze(0).cpu().numpy()


def embed_images_batch(image_paths, batch_size=16):
    """Embed a list of image paths in batches. Returns a numpy array (N, D)."""
    import numpy as np

    model, processor = load_model()
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(_device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        features = _unwrap_features(features)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def _unwrap_features(features):
    """
    Some transformers versions return a plain tensor from get_image_features(),
    others return a wrapped output object (e.g. BaseModelOutputWithPooling).
    Normalize either case to a plain tensor.
    """
    if torch.is_tensor(features):
        return features
    if hasattr(features, "image_embeds"):
        return features.image_embeds
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    raise TypeError(
        f"Unexpected output type from get_image_features(): {type(features)}. "
        "Check your installed 'transformers' version."
    )


# ---------------------------------------------------------------------------
# Threat level scoring
# ---------------------------------------------------------------------------
# Base severity weight per behavior class (0-1). Tune these to your use case.
THREAT_WEIGHTS = {
    "normal": 0.0,
    "peeping": 0.5,
    "stealing": 0.7,
    "fighting": 0.9,
}

THREAT_LEVELS = [
    (0.0, 0.15, "none"),
    (0.15, 0.4, "low"),
    (0.4, 0.65, "medium"),
    (0.65, 0.85, "high"),
    (0.85, 1.01, "critical"),
]


def compute_threat_level(predicted_class, confidence, context_multiplier=1.0):
    """
    Combine class severity weight, model confidence, and an optional
    context multiplier (e.g. 1.3x if it's a restricted zone / night time)
    into a single threat score and label.
    """
    base_weight = THREAT_WEIGHTS.get(predicted_class, 0.5)
    score = base_weight * confidence * context_multiplier
    score = min(score, 1.0)

    label = "none"
    for low, high, name in THREAT_LEVELS:
        if low <= score < high:
            label = name
            break

    return {"score": round(score, 3), "level": label}
