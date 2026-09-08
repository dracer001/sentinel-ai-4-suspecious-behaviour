"""
Train a lightweight classifier (Logistic Regression) on top of the CLIP
embeddings produced by embed_dataset.py. This step is OPTIONAL -- if you
don't have enough labeled images yet, skip it and use infer_single.py in
nearest-neighbor mode instead.

As a rule of thumb: with 20-50+ images per class this classifier will
outperform plain nearest-neighbor matching. With fewer than that, stick
to nearest-neighbor.

Usage:
    python scripts/train_classifier.py
"""

import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    embeddings_path = os.path.join(MODELS_DIR, "embeddings.npy")
    labels_path = os.path.join(MODELS_DIR, "labels.json")

    if not os.path.exists(embeddings_path):
        print("No embeddings found. Run scripts/embed_dataset.py first.")
        return

    X = np.load(embeddings_path)
    with open(labels_path) as f:
        y = json.load(f)

    class_counts = {c: y.count(c) for c in set(y)}
    print(f"Class counts: {class_counts}")

    min_count = min(class_counts.values())
    if min_count < 4:
        print("Warning: some classes have very few images. The train/test split "
              "below may fail or be unreliable. Consider adding more images or "
              "using nearest-neighbor mode (infer_single.py --mode nn) instead.")

    # Stratified split so each class is represented in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nEvaluation on held-out test split:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier.joblib"))
    print(f"Saved trained classifier to {MODELS_DIR}/classifier.joblib")


if __name__ == "__main__":
    main()
