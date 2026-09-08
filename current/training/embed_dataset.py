"""
Walk data/dataset/<class_name>/*.jpg, embed every image with CLIP, and save:
  - models/embeddings.npy   (N, D) float array
  - models/labels.json      list of class names, one per row of embeddings.npy
  - models/image_paths.json list of source file paths (for nearest-neighbor lookup)

Run this whenever you add/change images in data/dataset/.

Usage:
    python scripts/embed_dataset.py
"""

import json
import os

import numpy as np

from utils import embed_images_batch

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def collect_dataset():
    image_paths = []
    labels = []
    for class_name in sorted(os.listdir(DATASET_DIR)):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(VALID_EXT):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    return image_paths, labels


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    image_paths, labels = collect_dataset()

    if not image_paths:
        print(f"No images found under {DATASET_DIR}. "
              f"Add images into class subfolders (e.g. data/dataset/fighting/img1.jpg) and re-run.")
        return

    print(f"Found {len(image_paths)} images across classes: {sorted(set(labels))}")
    print("Embedding with CLIP... (first run downloads the model, ~600MB)")

    embeddings = embed_images_batch(image_paths)

    np.save(os.path.join(MODELS_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(MODELS_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(MODELS_DIR, "image_paths.json"), "w") as f:
        json.dump(image_paths, f)

    print(f"Saved embeddings to {MODELS_DIR}/embeddings.npy")
    print("Done. You can now either:")
    print("  1. Run infer_single.py in nearest-neighbor mode (no training needed), or")
    print("  2. Run train_classifier.py to train a classifier on top of these embeddings.")


if __name__ == "__main__":
    main()
