"""
temporal_model.py
──────────────────────────────────────────────────────────────────────────────
Sequence-based action recognition to replace single-frame pose-angle heuristics.

Why this exists:
    A single photo of "arm raised" cannot tell you if it's a punch, a wave, or
    someone reaching for a shelf. Fighting / stealing / shoplifting are actions
    that unfold over TIME. This module extracts MediaPipe pose landmarks across
    a short burst of frames (an "event") and classifies the whole sequence with
    a trained LSTM, instead of hand-written angle thresholds on one frame.

Two halves live in this file:
    1. PoseSequenceExtractor — turns a list of image paths into a fixed-length
       landmark sequence tensor. Used by BOTH training and inference so they
       stay perfectly consistent.
    2. LSTMActionClassifier   — the actual model architecture.
    3. TemporalClassifier     — inference wrapper the Flask app calls. Loads a
       trained checkpoint if present; degrades gracefully (like the existing
       mediapipe_available pattern) if no checkpoint has been trained yet.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SEQ_LEN = 12                     # frames per event, after padding/sampling
NUM_LANDMARKS = 33               # MediaPipe pose landmarks
FEATURES_PER_LANDMARK = 4        # x, y, z, visibility
FEATURES_PER_FRAME = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 132

# Class order MUST match training. Index 0 should always be the "safe" class.
CLASSES = ["normal", "loitering", "stealing", "fighting"]

# Base severity per class — final threat_level = round(base * confidence),
# so a low-confidence "fighting" guess doesn't scream CRITICAL.
CLASS_BASE_THREAT = {
    "normal": 0,
    "loitering": 3,
    "stealing": 7,
    "fighting": 9,
}

CHECKPOINT_PATH = Path("temporal_model.pt")


# ─── POSE SEQUENCE EXTRACTION ─────────────────────────────────────────────────
class PoseSequenceExtractor:
    """
    Wraps a MediaPipe PoseLandmarker to turn a burst of frames into a fixed
    (SEQ_LEN, FEATURES_PER_FRAME) numpy array. Frames with no detected person
    are zero-filled (not dropped) so timing/order stays intact.
    """

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(base_options=base_options)
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)
        log.info("PoseSequenceExtractor ready.")

    def _extract_single(self, image_path: str) -> np.ndarray | None:
        image = self._mp.Image.create_from_file(image_path)
        res = self.detector.detect(image)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        flat = np.array(
            [[p.x, p.y, getattr(p, "z", 0.0), getattr(p, "visibility", 0.0)] for p in lm],
            dtype=np.float32,
        ).flatten()
        return flat  # shape (132,)

    def extract_sequence(self, image_paths: list[str]) -> tuple[np.ndarray, float]:
        """
        Returns:
            sequence: np.ndarray shape (SEQ_LEN, FEATURES_PER_FRAME)
            detection_rate: fraction of frames where a person was detected
                             (low rate = unreliable sequence, e.g. camera moved)
        """
        raw_frames = []
        hits = 0
        for p in image_paths:
            feat = self._extract_single(p)
            if feat is not None:
                hits += 1
                raw_frames.append(feat)
            else:
                raw_frames.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))

        detection_rate = hits / max(len(image_paths), 1)
        sequence = self._resample(raw_frames)
        return sequence, detection_rate

    @staticmethod
    def _resample(frames: list[np.ndarray]) -> np.ndarray:
        """Uniformly sample/pad a variable-length frame list to exactly SEQ_LEN."""
        n = len(frames)
        if n == 0:
            return np.zeros((SEQ_LEN, FEATURES_PER_FRAME), dtype=np.float32)
        if n == SEQ_LEN:
            return np.stack(frames)
        idx = np.linspace(0, n - 1, SEQ_LEN).round().astype(int)
        return np.stack([frames[i] for i in idx])


# ─── MODEL ────────────────────────────────────────────────────────────────────
def _build_model():
    """Lazily imports torch so the rest of the app still runs if torch isn't
    installed (mirrors the optional-mediapipe pattern already in app.py)."""
    import torch
    import torch.nn as nn

    class LSTMActionClassifier(nn.Module):
        def __init__(self, input_size=FEATURES_PER_FRAME, hidden_size=128,
                     num_layers=2, num_classes=len(CLASSES), dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_size * 2)
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            # x: (batch, SEQ_LEN, FEATURES_PER_FRAME)
            out, _ = self.lstm(x)
            last = out[:, -1, :]           # final timestep, both directions
            last = self.norm(last)
            return self.head(last)         # (batch, num_classes) raw logits

    return torch, LSTMActionClassifier


# ─── INFERENCE WRAPPER ────────────────────────────────────────────────────────
class TemporalClassifier:
    """
    Loads a trained checkpoint if available. If not (no training run yet),
    `available` is False and the Flask app should skip straight to Groq
    escalation for clip analysis — same graceful-degradation pattern as
    `mediapipe_available` elsewhere in this codebase.
    """

    def __init__(self, checkpoint_path: str | Path = CHECKPOINT_PATH):
        self.available = False
        self.model = None
        self.torch = None
        self.feature_mean = None
        self.feature_std = None
        checkpoint_path = Path(checkpoint_path)

        try:
            torch, LSTMActionClassifier = _build_model()
        except ImportError:
            log.warning("PyTorch not installed — temporal classifier disabled. "
                        "Install with: pip install torch --break-system-packages")
            return

        if not checkpoint_path.exists():
            log.warning(f"No trained checkpoint at '{checkpoint_path}'. "
                        "Run train_temporal_model.py first. "
                        "Temporal classifier disabled until then.")
            self.torch = torch
            return

        try:
            model = LSTMActionClassifier()
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state["model_state"] if "model_state" in state else state)
            model.eval()
            self.model = model
            self.torch = torch
            self.feature_mean = state.get("feature_mean")
            self.feature_std = state.get("feature_std")
            self.available = True
            log.info(f"Temporal action classifier loaded from '{checkpoint_path}' "
                     f"(val_acc={state.get('val_acc', '?')}).")
        except Exception as e:
            log.error(f"Failed to load temporal classifier checkpoint: {e}")

    def predict(self, sequence: np.ndarray, detection_rate: float) -> dict:
        """
        sequence: (SEQ_LEN, FEATURES_PER_FRAME) numpy array from PoseSequenceExtractor
        Returns a dict matching the existing result schema used elsewhere in app.py.
        """
        if not self.available:
            return {
                "threat_level": 0,
                "behavior": "MODEL_UNAVAILABLE",
                "reason": "Temporal classifier not trained yet — see train_temporal_model.py",
                "source": "local_temporal_lstm",
                "confidence": 0.0,
            }

        if detection_rate < 0.3:
            return {
                "threat_level": 0,
                "behavior": "NO_SUBJECT",
                "reason": f"Person detected in only {detection_rate:.0%} of burst frames — "
                          "too unreliable to classify",
                "source": "local_temporal_lstm",
                "confidence": 0.0,
            }

        torch = self.torch
        if self.feature_mean is not None and self.feature_std is not None:
            sequence = (sequence - self.feature_mean) / self.feature_std

        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LEN, F)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        base = CLASS_BASE_THREAT[pred_class]
        threat_level = min(10, round(base * confidence + base * 0.3))  # floor so it isn't zeroed by confidence alone
        if pred_class == "normal":
            threat_level = round(base * confidence)  # keep normal near 0 regardless

        return {
            "threat_level": threat_level,
            "behavior": pred_class.upper(),
            "reason": f"Temporal model: {pred_class} (confidence {confidence:.0%}) "
                      f"over {SEQ_LEN}-frame burst, {detection_rate:.0%} detection rate",
            "source": "local_temporal_lstm",
            "confidence": confidence,
            "class_probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        }
