"""
train_temporal_model.py
──────────────────────────────────────────────────────────────────────────────
Trains the LSTM action classifier on pose sequences produced by
extract_pose_sequences.py, and saves temporal_model.pt for the Flask server
to load.

Run this on your PC / Google Colab (GPU recommended but not required — the
model is small enough to train on CPU in a reasonable time for a few hundred
clips). Then copy the resulting temporal_model.pt to the Raspberry Pi, next
to app.py.

USAGE:
    python train_temporal_model.py --sequences_dir ./sequences --epochs 40

Install requirements first:
    pip install torch numpy scikit-learn --break-system-packages
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from temporal_model import CLASSES, _build_model, FEATURES_PER_FRAME, SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_dataset(sequences_dir: Path):
    X, y = [], []
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    for npy_path in sorted(sequences_dir.glob("*.npy")):
        class_name = npy_path.stem.rsplit("_", 1)[0]
        if class_name not in class_to_idx:
            log.warning(f"Skipping {npy_path.name} — unrecognized class prefix '{class_name}'")
            continue
        seq = np.load(npy_path)
        if seq.shape != (SEQ_LEN, FEATURES_PER_FRAME):
            log.warning(f"Skipping {npy_path.name} — unexpected shape {seq.shape}")
            continue
        X.append(seq)
        y.append(class_to_idx[class_name])

    if not X:
        raise RuntimeError(
            f"No valid sequences found in {sequences_dir}. "
            "Run extract_pose_sequences.py first."
        )

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    log.info(f"Loaded {len(X)} sequences. Class counts: {counts}")
    for c, n in counts.items():
        if n < 20:
            log.warning(f"  '{c}' has only {n} examples — accuracy for this class will "
                        "be unreliable. Aim for 50+ per class minimum, more for 'normal'.")

    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences_dir", default="./sequences")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out", default="temporal_model.pt")
    args = ap.parse_args()

    torch, LSTMActionClassifier = _build_model()
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    X, y = load_dataset(Path(args.sequences_dir))

    # Per-frame normalization stats saved alongside the model so inference
    # can apply the same scaling.
    mean = X.reshape(-1, FEATURES_PER_FRAME).mean(axis=0)
    std = X.reshape(-1, FEATURES_PER_FRAME).std(axis=0) + 1e-6
    X_norm = (X - mean) / std

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=args.val_split, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LSTMActionClassifier().to(device)

    # Class weighting — "normal" clips are usually much easier to find than
    # "fighting"/"stealing", so weight the loss to avoid the model just
    # always predicting "normal".
    class_counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float32)
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * len(CLASSES)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(total, 1)
        scheduler.step(val_acc)

        log.info(f"Epoch {epoch:3d}/{args.epochs} | train_loss={total_loss/len(train_ds):.4f} "
                  f"| val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "feature_mean": mean,
                "feature_std": std,
                "val_acc": val_acc,
            }, args.out)
            log.info(f"  -> saved new best checkpoint ({val_acc:.3f} val acc) to {args.out}")

    log.info(f"Training complete. Best val_acc={best_val_acc:.3f}. Model saved to {args.out}")
    log.info("Copy this .pt file next to app.py on the Raspberry Pi to enable the temporal classifier.")


if __name__ == "__main__":
    main()
