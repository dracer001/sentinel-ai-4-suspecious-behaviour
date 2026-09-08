"""
extract_pose_sequences.py
──────────────────────────────────────────────────────────────────────────────
Run this ONCE per dataset to convert labeled video clips into pose-landmark
sequences (.npy) that train_temporal_model.py trains on. Keeps training fast
(no re-running MediaPipe every epoch) and keeps extraction logic identical to
what runs at inference time (via PoseSequenceExtractor).

EXPECTED INPUT FOLDER STRUCTURE:
    dataset/
        normal/
            clip001.mp4
            clip002.mp4
        loitering/
            clip001.mp4
        stealing/
            clip001.mp4
        fighting/
            clip001.mp4
            ...

    Folder names must exactly match temporal_model.CLASSES.

WHERE TO GET CLIPS:
    - Fighting:            RWF-2000 (GitHub: mchengny/RWF2000-Video-Database-for-Violence-Detection)
    - Fighting/robbery:    UCF-Crime dataset (Kaggle: "UCF Crime Dataset")
    - Shoplifting:         Kaggle: "DCSASS Dataset" (has a Shoplifting category),
                            or Kaggle: "Shoplifting Detection Dataset"
    - Normal/loitering:    Sample your own footage — plain walking, shopping,
                            standing around, talking — from your own camera.
                            This matters a lot: your camera angle/lighting/
                            store layout will differ from any public dataset,
                            so 50-100 of your OWN "normal" clips will help
                            accuracy more than another 500 public ones.

USAGE:
    python extract_pose_sequences.py --dataset_dir ./dataset --out_dir ./sequences \
        --model_path pose_landmarker_heavy.task --fps 3

Output:
    sequences/
        normal_0000.npy, normal_0001.npy, ...
        fighting_0000.npy, ...
    Each .npy is shape (SEQ_LEN, FEATURES_PER_FRAME) — same format used at inference.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2

from temporal_model import PoseSequenceExtractor, SEQ_LEN, CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def sample_frames_from_video(video_path: str, target_fps: float, tmp_dir: Path) -> list[str]:
    """Extracts frames from a video at target_fps and writes them as temp jpgs
    (PoseSequenceExtractor works on file paths, matching the inference path)."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(int(round(src_fps / target_fps)), 1)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    i = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            out_path = tmp_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(str(out_path))
            saved += 1
        i += 1
    cap.release()
    return frame_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", default="./sequences")
    ap.add_argument("--model_path", default="pose_landmarker_heavy.task")
    ap.add_argument("--fps", type=float, default=3.0,
                     help="Frames per second to sample from each clip before "
                          "resampling to SEQ_LEN — should roughly match how "
                          "the ESP32-CAM bursts are captured.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("./_tmp_frames")

    extractor = PoseSequenceExtractor(args.model_path)

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            log.warning(f"Skipping missing class folder: {class_dir}")
            continue

        video_files = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
        )
        log.info(f"[{class_name}] {len(video_files)} clips found")

        for idx, video_path in enumerate(video_files):
            try:
                frame_paths = sample_frames_from_video(video_path, args.fps, tmp_dir)
                if len(frame_paths) < 2:
                    log.warning(f"  Skipping {video_path.name} — too few frames extracted")
                    continue

                sequence, detection_rate = extractor.extract_sequence(frame_paths)
                if detection_rate < 0.2:
                    log.warning(f"  Skipping {video_path.name} — low pose detection rate "
                                f"({detection_rate:.0%}), likely no clear person in frame")
                    continue

                out_path = out_dir / f"{class_name}_{idx:04d}.npy"
                np.save(out_path, sequence)
                log.info(f"  {video_path.name} -> {out_path.name} "
                         f"(detection rate {detection_rate:.0%})")

            finally:
                for f in tmp_dir.glob("*.jpg"):
                    f.unlink(missing_ok=True)

    log.info(f"Done. Sequences written to {out_dir}/")


if __name__ == "__main__":
    main()
