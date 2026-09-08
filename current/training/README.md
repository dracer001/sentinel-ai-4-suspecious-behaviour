# Behavior Threat Detection

Upload a dataset of labeled images, then classify new images (or a sequence of
frames over time) into a behavior category (e.g. `fighting`, `stealing`,
`peeping`, `normal`) and get an assigned threat level. Built on pretrained
CLIP embeddings from Hugging Face so you don't need to train a model from
scratch, and don't need thousands of labeled examples to get something working.

## How it works

1. You put labeled example images into `data/dataset/<class_name>/`.
2. `embed_dataset.py` runs every image through a pretrained CLIP model and
   saves the resulting embeddings (numeric fingerprints of each image).
3. From there you have two options:
   - **No training (nearest-neighbor):** compare a new image's embedding to
     your labeled examples directly and take the closest match(es). Works
     with very few images per class (even single digits).
   - **Train a classifier:** fit a small Logistic Regression model on top of
     the embeddings. More accurate once you have 20-50+ images per class.
4. A threat-level score is computed from the predicted class, the model's
   confidence, and any context multiplier you supply (e.g. restricted zone,
   nighttime).
5. For a series of images/frames instead of one photo, frame embeddings are
   averaged (mean-pooled) into a single "sequence embedding" before the same
   lookup/classification is applied.

## Folder structure

```
behavior-threat-detection/
├── README.md
├── requirements.txt
├── data/
│   └── dataset/                 # put your labeled training images here
│       ├── fighting/
│       ├── stealing/
│       ├── peeping/
│       └── normal/
├── models/                       # generated automatically, do not hand-edit
│   ├── embeddings.npy            # CLIP embeddings for your dataset
│   ├── labels.json                # class label per embedding row
│   ├── image_paths.json           # source image path per embedding row
│   └── classifier.joblib          # trained classifier (only if you train one)
├── sample_sequence/                # example folder for a frame-sequence test
└── scripts/
    ├── utils.py                    # CLIP loading + embedding + threat scoring
    ├── embed_dataset.py            # step 1: embed your labeled dataset
    ├── train_classifier.py         # step 2 (optional): train a classifier
    ├── infer_single.py             # predict on a single image
    └── infer_sequence.py           # predict on a folder of sequential frames
```

## Setup

```bash
cd behavior-threat-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run any script, it will download the CLIP model from
Hugging Face (~600MB) and cache it locally — you don't need an API key,
it's a public model.

## Models used (all free, from Hugging Face)

- **Default:** [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
  — fast, good starting point, runs fine on CPU.
- **More accurate, slower:** [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)
- **Trained on a larger public dataset:** [`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)

Swap the model by changing `MODEL_NAME` in `scripts/utils.py` — everything
else stays the same.

If you later want to move to full video-based action recognition instead of
mean-pooled frames, these are strong pretrained options to fine-tune:
- [`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
- [`facebook/timesformer-base-finetuned-k400`](https://huggingface.co/facebook/timesformer-base-finetuned-k400)

## Public datasets to bootstrap training data

- [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) — real-world
  anomaly/crime videos including fighting, stealing/shoplifting, and more.
- [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) —
  2,000 video clips specifically for violence/fighting detection.
- "Peeping"/voyeurism behavior is not well covered by any public dataset —
  you'll likely need to collect and label your own clips for that class
  (e.g. someone lingering and oriented toward a window/doorway for an
  extended period).

## Usage

### 1. Add your training images

Drop labeled images into the matching folder:

```
data/dataset/fighting/img1.jpg
data/dataset/stealing/img1.jpg
data/dataset/peeping/img1.jpg
data/dataset/normal/img1.jpg
```

More images per class = better accuracy, but this approach can work with as
few as a handful of images per class using nearest-neighbor mode.

### 2. Embed the dataset

```bash
python scripts/embed_dataset.py
```

Re-run this any time you add or change images in `data/dataset/`.

### 3. Choose: train a classifier, or skip straight to nearest-neighbor

**Option A — No training needed (best for small datasets):**
Skip straight to inference with `--mode nn`.

**Option B — Train a classifier (best once you have 20-50+ images/class):**

```bash
python scripts/train_classifier.py
```

This prints an accuracy report on a held-out test split and saves the model.

### 4. Predict on a single image

```bash
# Nearest-neighbor (no training required)
python scripts/infer_single.py path/to/new_image.jpg --mode nn

# Trained classifier
python scripts/infer_single.py path/to/new_image.jpg --mode classifier
```

Add `--context-multiplier 1.3` if the event is in a higher-risk context
(e.g. restricted zone, nighttime) to bump the threat score accordingly.

### 5. Predict on a sequence of frames (when a single image isn't enough)

Put a handful of frames from the same event (5-15 frames spanning a few
seconds works well) into a folder, then:

```bash
python scripts/infer_sequence.py path/to/frame_folder --mode nn
```

## Threat level scoring

Threat level is computed from three things, configurable in `scripts/utils.py`:

1. **Base severity weight per class** (`THREAT_WEIGHTS`) — e.g. fighting
   is weighted higher than peeping by default.
2. **Model confidence** — how sure the model is about its prediction.
3. **Context multiplier** — an optional number you pass in (e.g. `1.3` for
   a restricted area or off-hours event) to scale the score up or down.

These combine into a 0-1 score, mapped to a label: `none`, `low`, `medium`,
`high`, `critical`. Adjust the weights/thresholds in `utils.py` to match
your real-world risk tolerance — the defaults are a starting point, not a
fixed standard.

## When to move beyond this setup

- If accuracy on stills plateaus and you consistently have access to short
  video clips (not just occasional bursts), fine-tuning a video-action model
  (VideoMAE/TimeSformer, linked above) directly on your labeled clips will
  outperform the mean-pooled CLIP approach here.
- If you need real-time processing on an edge device (Jetson/Pi), consider
  a smaller/distilled CLIP variant or exporting the classifier to ONNX for
  faster inference.
