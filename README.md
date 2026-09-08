# Sentinel

Sentinel is a behavior-based threat detection system for camera footage. It looks at an
image (or a short sequence of frames from a motion event) and classifies what's
happening — `fighting`, `stealing`, `peeping`, `normal`, etc. — then assigns a threat
level so a downstream alert/notification system can decide what to do about it.

This repo tracks five iterations of the project, each one a different approach to the
same problem: turning raw pixels into a reliable behavior label without needing a huge
labeled video dataset. The current version (`/current`) is the one actually worth
running; `/archive` keeps the earlier approaches for reference — see
[CHANGELOG.md](./CHANGELOG.md) for why each one was tried and why it was replaced.

## How it works (current version)

1. A camera or upload sends a frame (or a burst of frames from one motion event) to the
   Flask backend.
2. The image is embedded with a pretrained **CLIP** model from Hugging Face — no
   training required to get started.
3. That embedding is compared against a small labeled dataset, either via
   **nearest-neighbor** matching (works with a handful of examples per class) or a
   **trained classifier** (a Logistic Regression head on top of the embeddings, once
   you have 20-50+ examples per class).
4. A threat score is derived from the predicted class and the model's confidence.
5. Results, including a running history, are served through a small web UI.

See [`current/training/README.md`](./current/training/README.md) for the full
dataset → embeddings → classifier pipeline.

## Repo layout

```
Sentinel/
├── current/              ← the active system (CLIP embeddings + Flask app)
│   ├── app.py            ← web server, upload handling, history API
│   ├── analyzer.py        ← embeds an image and scores it against the dataset
│   ├── models/            ← generated embeddings/labels/classifier
│   ├── training/          ← scripts to build/rebuild the dataset + models
│   └── templates/         ← web UI
├── archive/               ← earlier approaches, kept for reference (see CHANGELOG)
│   ├── v1-hybrid-pose-llm/
│   ├── v2-temporal-lstm/
│   └── v3-keras-cnn/
├── CHANGELOG.md
└── LICENSE
```

## Quickstart

```bash
cd current
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open `http://localhost:5000`. The first run downloads the CLIP model
(`openai/clip-vit-base-patch32`, ~600MB) from Hugging Face and caches it locally.

To add your own labeled images and (re)build the dataset the app scores against, see
[`current/training/README.md`](./current/training/README.md).

## Status

Actively evolving. The nearest-neighbor / classifier split (`mode=nn` vs
`mode=classifier` in `analyzer.py`) is there so the dataset can grow from "a handful of
labeled examples" to "a proper trained classifier" without a rewrite.

## License

[MIT](./LICENSE) — see file for details.
