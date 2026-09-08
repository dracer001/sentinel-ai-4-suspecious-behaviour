# Changelog

Sentinel went through five distinct approaches before landing on the current
CLIP-embedding design. Each entry below explains what the version did, why it was
tried, and why it was moved on from.

## v5 — CLIP embeddings + classifier, integrated into the web app (current → `/current`)

The nearest-neighbor/classifier engine from v4 replaces the MediaPipe + Gemini/Groq
engine from v1 inside the same Flask app and web UI, so the front end (upload, history,
threat-level display) didn't need to change.

- `analyzer.py` rebuilt around CLIP embeddings: no external API calls (no Gemini/Groq),
  no MediaPipe pose detection.
- Supports two scoring modes: `nn` (nearest-neighbor against labeled examples) and
  `classifier` (trained Logistic Regression on top of embeddings) — selectable per
  request.
- Ships with a trained `classifier.joblib` plus the embeddings/labels it was trained on
  (dataset: fighting, stealing, normal).
- Threat level is computed from predicted class + model confidence, same shape as v1's
  scoring so the UI didn't need rework.
- Drops the API-key dependency entirely — everything runs locally once the CLIP model
  is cached.

## v4 — Dataset-upload CLIP pipeline (standalone toolkit → `/current/training`)

A step back from live inference to fix the actual bottleneck: v1-v3 all assumed either
enough labeled video to train a classifier, or good-enough general-purpose pose
detection. Neither was true yet. v4 is a small toolkit, not a running app:

- `embed_dataset.py` — embeds a folder of labeled images with a pretrained CLIP model
  (`openai/clip-vit-base-patch32` by default) and saves the embeddings.
- `infer_single.py` / `infer_sequence.py` — classify a new image, or a folder of
  sequential frames (mean-pooled into one embedding), against the saved dataset.
- `train_classifier.py` — optional: once there are enough labeled examples, fit a
  classifier on top of the embeddings instead of doing nearest-neighbor lookup.
- Point: get useful classification with as few as single-digit examples per class,
  using a pretrained model instead of training one from scratch.

This became the training pipeline for v5 rather than a separate app.

## v3 — Custom CNN classifier on office footage (`/archive/v3-keras-cnn`)

A parallel experiment: instead of pose or embeddings, train a standard image classifier
(Keras/TensorFlow, 224×224 input) directly on labeled office footage, with classes
`Fighting`, `Harvoc`, `NormalVideos`, `Stealing`.

- Straightforward Flask + Keras `model.predict()` app.
- Two model checkpoints were tried (`baseline_office_model.keras`,
  `baseline_office_model1.keras`).
- Shelved in favor of the CLIP-embedding approach (v4/v5): a from-scratch CNN needed
  far more labeled footage per class than was available, whereas CLIP embeddings gave
  usable results with a fraction of the data.

## v2 — Temporal (LSTM) layer on top of pose sequences (`/archive/v2-temporal-lstm`)

An attempt to fix v1's single-frame pose limitation by looking at *sequences* of poses
over time instead of one frame at a time, added as a second endpoint
(`/analyze-clip`) alongside v1's existing `/analyze` — the upload flow kept working
unchanged.

- `extract_pose_sequences.py` — turns labeled video clips into MediaPipe pose-sequence
  training data.
- `temporal_model.py` / `train_temporal_model.py` — an LSTM trained on those sequences.
- Designed to match an ESP32-CAM capturing ~8-12 frames per motion event rather than
  one frame per request.
- Graceful degradation: without a trained `temporal_model.pt`, `/analyze-clip` still
  worked by falling back to a multi-frame LLM check.
- Superseded once pose-based detection in general was dropped (see v1 note below) —
  the sequence-modeling idea carried over conceptually into v4/v5's frame-averaging
  for sequences, without the pose dependency.

## v1 — Hybrid MediaPipe pose + Gemini/Groq vision LLM (`/archive/v1-hybrid-pose-llm`)

The first end-to-end working version.

- MediaPipe pose landmarks run on every frame, locally and for free.
- Only escalates to a vision LLM (Gemini primary, Groq backup, both free-tier) when the
  pose data looks suspicious (`threat ≥ 4`), to conserve free API quota.
- Three modes: `hybrid` (default, described above), `api` (always call the LLM),
  `local` (pose only, no LLM, zero cost).
- Full history API (`/api/history`, filters by time range / scene type / threat) and a
  web UI with drag-and-drop upload.
- Deployable to Render's free tier, with MediaPipe auto-disabled there (not enough RAM)
  and the app falling back to API-only mode.
- Pose-based detection ultimately didn't generalize well enough on its own — a raised
  arm looks the same in a high-five and a punch without more context — which motivated
  moving to appearance-based classification (v3) and then embeddings (v4/v5) instead of
  refining pose further.

---

**Note on `/archive`:** large model binaries (MediaPipe's `.task` file, the v3 `.keras`
weights) were left out of this repo to keep it lightweight — see the `MODEL_DOWNLOAD.md`
/ `MODEL_WEIGHTS.md` notes in each archived folder for how to get or regenerate them.
