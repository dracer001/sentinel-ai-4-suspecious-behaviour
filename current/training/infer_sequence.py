"""
Predict the behavior class + threat level for a SEQUENCE of frames
(e.g. frames extracted from a few seconds of video, or a burst of stills).

Strategy: embed every frame with CLIP, average the embeddings into a single
"sequence embedding" (mean pooling), then run the same nearest-neighbor or
classifier lookup used for single images. This captures how the scene
evolves over the frames rather than judging off one still.

For best results, use 5-15 frames spanning ~2-5 seconds of the event.

Usage:
    python scripts/infer_sequence.py path/to/frame_folder --mode nn
    python scripts/infer_sequence.py path/to/frame_folder --mode classifier
"""

import argparse
import os

import numpy as np

from utils import embed_images_batch, compute_threat_level
from infer_single import predict_nn, predict_classifier

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def load_frame_paths(folder):
    frames = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(VALID_EXT)
    ]
    if not frames:
        raise ValueError(f"No image frames found in {folder}")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_folder", help="Folder containing sequential frame images")
    parser.add_argument("--mode", choices=["nn", "classifier"], default="nn")
    parser.add_argument("--context-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    frame_paths = load_frame_paths(args.frame_folder)
    print(f"Found {len(frame_paths)} frames. Embedding...")

    frame_embeddings = embed_images_batch(frame_paths)

    # Mean-pool across the time dimension, then re-normalize
    sequence_embedding = frame_embeddings.mean(axis=0)
    sequence_embedding = sequence_embedding / np.linalg.norm(sequence_embedding)

    if args.mode == "nn":
        predicted_class, confidence, breakdown = predict_nn(sequence_embedding)
    else:
        predicted_class, confidence, breakdown = predict_classifier(sequence_embedding)

    threat = compute_threat_level(predicted_class, confidence, args.context_multiplier)

    print(f"\nSequence: {args.frame_folder} ({len(frame_paths)} frames)")
    print(f"Predicted behavior: {predicted_class}  (confidence: {confidence:.3f})")
    print(f"Threat level: {threat['level']}  (score: {threat['score']})")
    print(f"\nClass breakdown: {breakdown}")


if __name__ == "__main__":
    main()
