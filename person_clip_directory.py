#!/usr/bin/env python3
"""
person_clip_extractor.py

Cuts video segments where exactly **one** person is visible.

Default paths:
  input : /media/mili/EXTERNAL_USB/Editor_videos/procesar
  output: /media/mili/EXTERNAL_USB/Editor_videos/output
"""
from __future__ import annotations

import argparse
import subprocess as sp
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
from ultralytics import YOLO


# ───────────────────────── helpers ──────────────────────────

def _ms(frame_idx: int, fps: float) -> int:
    return int(frame_idx / fps * 1000)


def _segments(video: Path, model: YOLO, conf: float) -> List[Tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    segs: List[Tuple[int, int]] = []
    active_start = active_end = None

    for idx, r in enumerate(
        model.track(source=str(video), stream=True, classes=[0], conf=conf, persist=True)
    ):
        ids = r.boxes.id.cpu().tolist() if r.boxes.id is not None else []
        ids = list(set(ids))

        if len(ids) == 1:
            if active_start is None:
                active_start = idx
            active_end = idx
        else:
            if active_start is not None:
                segs.append((_ms(active_start, fps), _ms(active_end, fps)))
                active_start = active_end = None

    if active_start is not None:
        segs.append((_ms(active_start, fps), _ms(active_end, fps)))

    return segs


def _save_clip(inp: Path, out: Path, start_ms: int, end_ms: int):
    if end_ms <= start_ms:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(inp),
        "-ss",
        f"{start_ms/1000:.3f}",
        "-to",
        f"{end_ms/1000:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        str(out),
    ]
    sp.run(cmd, check=False)


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

    for i, (s, e) in enumerate(segs, 1):
        clip_name = f"{video.stem}_clip_{i:03d}.mp4"
        _save_clip(video, out_dir / clip_name, s, e)
    print(f"[done] {video.name}: {len(segs)} clip(s)")


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
    p = argparse.ArgumentParser(description="Cut clips with exactly one person visible")
    p.add_argument("--input", help="Video file to process")
    p.add_argument("--output-dir", help="Where to write clips")
    p.add_argument("--input-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/testing/procesar")
    p.add_argument("--default-out-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/testing/output")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()

    if args.input:
        if not args.output_dir:
            p.error("--output-dir is required when using --input")
        process_video(Path(args.input), Path(args.output_dir), weights=args.model, conf=args.conf)
    else:
        out_dir = Path(args.output_dir or args.default_out_dir)
        process_dir(Path(args.input_dir), out_dir, weights=args.model, conf=args.conf)


if __name__ == "__main__":
    _cli()
