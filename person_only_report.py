#!/usr/bin/env python3
"""
person_clip_extractor.py

Scans videos and records timestamp segments (ms) where **exactly one** person is
visible. Output is one JSON per video with:
    {
      "person_id": <tracker‑id>,
      "start_ms": <int>,
      "end_ms": <int>
    }

Default paths:
  input : /media/mili/EXTERNAL_USB/Editor_videos/procesar
  output: /media/mili/EXTERNAL_USB/Editor_videos/output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

import cv2
from ultralytics import YOLO

# ───────────────────────── helpers ──────────────────────────


def _ms(frame: int, fps: float) -> int:
    return int(frame / fps * 1000)


def _segments(video: Path, model: YOLO, conf: float = 0.3) -> List[Dict]:
    """Return list of segments with exactly one person on screen."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    segs: List[Dict] = []
    active_start = active_end = None
    active_id = None

    stream = model.track(
        source=str(video), stream=True, classes=[0], conf=conf, persist=True
    )

    for idx, r in enumerate(stream):
        ids = r.boxes.id.cpu().tolist() if r.boxes.id is not None else []
        ids = list(set(ids))

        if len(ids) == 1:
            pid = int(ids[0])
            if active_start is None:
                active_start = idx
                active_id = pid
            elif pid != active_id:
                # person changed → close previous segment
                segs.append(
                    {
                        "person_id": active_id,
                        "start_ms": _ms(active_start, fps),
                        "end_ms": _ms(active_end, fps),
                    }
                )
                active_start = idx
                active_id = pid
            active_end = idx
        else:
            if active_start is not None:
                segs.append(
                    {
                        "person_id": active_id,
                        "start_ms": _ms(active_start, fps),
                        "end_ms": _ms(active_end, fps),
                    }
                )
                active_start = active_end = active_id = None

    if active_start is not None:
        segs.append(
            {
                "person_id": active_id,
                "start_ms": _ms(active_start, fps),
                "end_ms": _ms(active_end, fps),
            }
        )

    return segs


def _videos(dir_: Path, patterns: Iterable[str]) -> List[Path]:
    vids: List[Path] = []
    for p in patterns:
        vids.extend(dir_.glob(p))
    return sorted(vids)


# ───────────────────────── main flow ──────────────────────────


def process_video(
    video: Path,
    out_dir: Path,
    *,
    model: YOLO | None = None,
    weights: str | Path = "yolov8n.pt",
    conf: float = 0.3,
):
    if model is None:
        model = YOLO(str(weights))

    segs = _segments(video, model, conf)
    if not segs:
        print(f"[skip] {video.name}: no single‑person segments")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{video.stem}.json"
    out_file.write_text(json.dumps(segs, indent=2))
    print(f"[json] {out_file.name}: {len(segs)} segment(s)")



def process_dir(
    in_dir: Path,
    out_dir: Path,
    *,
    weights: str | Path = "yolov8n.pt",
    conf: float = 0.3,
    patterns: Iterable[str] = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"),
):
    vids = _videos(in_dir, patterns)
    if not vids:
        print(f"No videos in {in_dir}")
        return

    model = YOLO(str(weights))
    for v in vids:
        process_video(v, out_dir, model=model, conf=conf)


# ────────────────────────── CLI ─────────────────────────────


def _cli():
    p = argparse.ArgumentParser(description="Save timestamps where only one person is visible")
    p.add_argument("--input", help="Video file to process")
    p.add_argument("--output-dir", help="Where to write JSON")
    p.add_argument(
        "--input-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/testing/procesar"
    )
    p.add_argument(
        "--default-out-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/testing/output"
    )
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()

    if args.input:
        if not args.output_dir:
            p.error("--output-dir is required when using --input")
        process_video(
            Path(args.input), Path(args.output_dir), weights=args.model, conf=args.conf
        )
    else:
        out_dir = Path(args.output_dir or args.default_out_dir)
        process_dir(Path(args.input_dir), out_dir, weights=args.model, conf=args.conf)


if __name__ == "__main__":
    _cli()
