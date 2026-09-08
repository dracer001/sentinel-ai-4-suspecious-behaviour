# Temporal Action Detection — Setup Guide

This adds a second, more accurate detection path alongside your existing
single-frame `/analyze` route. It does **not** touch your frontend upload
flow — that keeps working exactly as before.

## What changed

| File | Status | Purpose |
|---|---|---|
| `app.py` | modified | Added `/analyze-clip` endpoint + temporal classifier init |
| `temporal_model.py` | new | Pose-sequence extraction + LSTM model + inference wrapper |
| `extract_pose_sequences.py` | new | One-time script: turns labeled video clips into training data |
| `train_temporal_model.py` | new | Trains the LSTM, saves `temporal_model.pt` |

## Install

```bash
pip install torch scikit-learn --break-system-packages
```
(You already have `mediapipe`, `opencv-python`, `numpy` from the original app.)

## Step 1 — Get labeled clips

Folder structure (folder names must match exactly):
```
dataset/
    normal/       clip001.mp4, clip002.mp4, ...
    loitering/    clip001.mp4, ...
    stealing/     clip001.mp4, ...
    fighting/     clip001.mp4, ...
```
Sources: RWF-2000 (fighting, on GitHub), Kaggle's "DCSASS Dataset" (has a
shoplifting category), Kaggle's UCF-Crime dataset. **Record your own
"normal" footage from the actual camera/location this will run in** —
that one class matters more for accuracy than any public dataset, since
false positives on ordinary browsing/shopping behavior are what will erode
trust in the system fastest.

## Step 2 — Extract pose sequences (run once)

```bash
python extract_pose_sequences.py --dataset_dir ./dataset --out_dir ./sequences \
    --model_path pose_landmarker_heavy.task --fps 3
```
This samples each clip at ~3fps (matching how the ESP32-CAM burst will
capture), runs MediaPipe on every sampled frame, and saves the resulting
pose sequences as `.npy` files.

## Step 3 — Train

```bash
python train_temporal_model.py --sequences_dir ./sequences --epochs 40
```
Do this on your PC or Google Colab — CPU is fine for a few hundred clips,
GPU speeds it up if you have more. Watch `val_acc` in the logs; it saves
the best checkpoint automatically to `temporal_model.pt`.

## Step 4 — Deploy

Copy `temporal_model.pt` next to `app.py` on the Raspberry Pi. On next
`app.py` startup you'll see in the logs:
```
Temporal action classifier loaded from 'temporal_model.pt' (val_acc=0.87).
```
Until you do this, `/analyze-clip` still works — it just always escalates
to the Groq multi-frame check instead of using the (untrained) local model,
same graceful-degradation pattern your `mediapipe_available` flag already uses.

## Step 5 — ESP32-CAM firmware change needed

`/analyze-clip` expects **one HTTP POST per motion event**, containing
8–12 JPEG frames as multiple `frames` fields (not one request per frame):

```
POST /analyze-clip
Content-Type: multipart/form-data
  frames: frame_000.jpg
  frames: frame_001.jpg
  ...
  frames: frame_011.jpg
  event_id: (optional, auto-generated if omitted)
```

On the firmware side: trigger on PIR/motion, capture 8-12 JPEGs into PSRAM
at ~2-3fps, then send them all in one multipart request when the burst is
complete. I didn't write this firmware in this round since you asked for
the model + server first — happy to build it next when you're ready.

## Class design

Current classes: `normal`, `loitering`, `stealing`, `fighting`
(`temporal_model.py` → `CLASSES`). Add/remove classes there and re-extract
+ re-train if your use case needs different categories — just keep the
folder names in `dataset/` matching exactly.
