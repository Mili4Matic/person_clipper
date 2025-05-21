#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU‑ready

Scans videos and stores JSON clips where **exactly one** person appears.

JSON schema (one file per video):
[
  {"video_id": "<filename_without_ext>", "person_id": <int>, "start_ms": <int>, "end_ms": <int>},
  ...
]

Each element corresponds to a single continuous appearance of one person.

CLI extras:
  --device 0   # GPU (default)
  --device cpu # force CPU

Default dirs:
  input  : /media/mili/EXTERNAL_USB/Editor_videos/procesar
  output : /media/mili/EXTERNAL_USB/Editor_videos/output
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Iterable, List, Dict, Union

import cv2
from ultralytics import YOLO

# ───────── helpers ──────────

def _ms(frame: int, fps: float) -> int:
    return int(frame / fps * 1000)


def _segments(video: Path, model: YOLO, *, conf: float, device: Union[str, int]) -> List[Dict]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    segs: List[Dict] = []
    st = end = None  # segment start/end frames
    pid = None       # active person id

    stream = model.track(
        source=str(video), stream=True, classes=[0], conf=conf, persist=True, device=device
    )
    for idx, r in enumerate(stream):
        ids = r.boxes.id.cpu().tolist() if r.boxes.id is not None else []
        ids = list(set(ids))

        if len(ids) == 1:
            cur = int(ids[0])
            if st is None:
                st, pid = idx, cur
            elif cur != pid:
                segs.append({"person_id": pid, "start_ms": _ms(st, fps), "end_ms": _ms(end, fps)})
                st, pid = idx, cur
            end = idx
        else:
            if st is not None:
                segs.append({"person_id": pid, "start_ms": _ms(st, fps), "end_ms": _ms(end, fps)})
                st = end = pid = None

    if st is not None:
        segs.append({"person_id": pid, "start_ms": _ms(st, fps), "end_ms": _ms(end, fps)})

    return segs


def _videos(dir_: Path, pats: Iterable[str]) -> List[Path]:
    vids: List[Path] = []
    for p in pats:
        vids.extend(dir_.glob(p))
    return sorted(vids)

# ───────── processing ──────────

def _load_model(weights: str | Path, device: Union[str, int]):
    m = YOLO(str(weights))
    if device != "cpu" and str(device) != "-1":
        d = f"cuda:{device}" if str(device).isdigit() else device
        m.to(d)
    return m


def process_video(video: Path, out_dir: Path, *, model: YOLO | None, conf: float, device):
    if model is None:
        raise ValueError("model must be provided")

    segs = _segments(video, model, conf=conf, device=device)
    if not segs:
        print(f"[skip] {video.name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [ {"video_id": video.stem, **seg} for seg in segs ]
    (out_dir / f"{video.stem}.json").write_text(json.dumps(clips, indent=2))
    print(f"[json] {video.name}: {len(clips)} clip(s)")


def process_dir(
    in_dir: Path,
    out_dir: Path,
    *,
    weights: str | Path,
    conf: float,
    device: Union[str, int],
    patterns: Iterable[str] = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"),
):
    vids = _videos(in_dir, patterns)
    if not vids:
        print(f"No videos in {in_dir}")
        return

    model = _load_model(weights, device)
    for v in vids:
        process_video(v, out_dir, model=model, conf=conf, device=device)

# ────────── CLI ───────────

def _cli():
    p = argparse.ArgumentParser(description="Save timestamps where exactly one person is visible (GPU ready)")
    p.add_argument("--input")
    p.add_argument("--output-dir")
    p.add_argument("--input-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/procesar")
    p.add_argument("--default-out-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/output")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="0", help="GPU id or 'cpu'")
    args = p.parse_args()

    if args.input:
        if not args.output_dir:
            p.error("--output-dir is required with --input")
        m = _load_model(args.model, args.device)
        process_video(Path(args.input), Path(args.output_dir), model=m, conf=args.conf, device=args.device)
    else:
        out_dir = Path(args.output_dir or args.default_out_dir)
        process_dir(Path(args.input_dir), out_dir, weights=args.model, conf=args.conf, device=args.device)


if __name__ == "__main__":
    _cli()
