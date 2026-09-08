"""
Predict the behavior class + threat level for a SINGLE image.

Two modes:
  --mode nn         Nearest-neighbor against your embedded dataset. No training
                     needed -- works as soon as you've run embed_dataset.py.
                     Best when you have very few labeled images per class.

  --mode classifier  Uses the trained classifier from train_classifier.py.
                     Needs you to have run embed_dataset.py + train_classifier.py.
                     Better once you have 20-50+ images per class.

Usage:
    python scripts/infer_single.py path/to/image.jpg --mode nn
    python scripts/infer_single.py path/to/image.jpg --mode classifier
"""

import argparse
import json
import os

import joblib
import numpy as np

from utils import embed_image, compute_threat_level

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def predict_nn(query_embedding, top_k=5):
    embeddings = np.load(os.path.join(MODELS_DIR, "embeddings.npy"))
    with open(os.path.join(MODELS_DIR, "labels.json")) as f:
        labels = json.load(f)

    # cosine similarity (embeddings are already L2-normalized)
    sims = embeddings @ query_embedding
    top_idx = np.argsort(sims)[::-1][:top_k]

    top_labels = [labels[i] for i in top_idx]
    top_sims = [float(sims[i]) for i in top_idx]

    # majority vote among top_k neighbors, weighted by similarity
    scores = {}
    for lbl, sim in zip(top_labels, top_sims):
        scores[lbl] = scores.get(lbl, 0.0) + max(sim, 0.0)

    predicted_class = max(scores, key=scores.get)
    # normalize confidence to 0-1 range across the summed similarity scores
    confidence = scores[predicted_class] / sum(scores.values())

    return predicted_class, confidence, list(zip(top_labels, top_sims))


def predict_classifier(query_embedding):
    clf = joblib.load(os.path.join(MODELS_DIR, "classifier.joblib"))
    probs = clf.predict_proba([query_embedding])[0]
    classes = clf.classes_
    best_idx = np.argmax(probs)
    return classes[best_idx], float(probs[best_idx]), list(zip(classes, probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image to classify")
    parser.add_argument("--mode", choices=["nn", "classifier"], default="nn")
    parser.add_argument("--context-multiplier", type=float, default=1.0,
                         help="Optional multiplier for threat score, e.g. 1.3 "
                              "for a restricted zone or nighttime event")
    args = parser.parse_args()

    print("Embedding image...")
    query_embedding = embed_image(args.image_path)

    if args.mode == "nn":
        predicted_class, confidence, breakdown = predict_nn(query_embedding)
    else:
        predicted_class, confidence, breakdown = predict_classifier(query_embedding)

    threat = compute_threat_level(predicted_class, confidence, args.context_multiplier)

    print(f"\nImage: {args.image_path}")
    print(f"Predicted behavior: {predicted_class}  (confidence: {confidence:.3f})")
    print(f"Threat level: {threat['level']}  (score: {threat['score']})")
    print(f"\nClass breakdown: {breakdown}")


if __name__ == "__main__":
    main()
