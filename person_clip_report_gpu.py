#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU ready, single JSON output

Scan one video or a whole directory and extract segments where **exactly one** person
is visible. All clips from all processed videos are written into **one** JSON file.

Output example (clips.json):
[
  {"video_id": "video1", "person_id": 3, "start_ms": 10000, "end_ms": 20000},
  {"video_id": "video1", "person_id": 7, "start_ms": 21000, "end_ms": 30000},
  {"video_id": "video2", "person_id": 5, "start_ms": 5000,  "end_ms": 15000}
]

CLI example (defaults use GPU 0):
    python person_clip_extractor.py           # dir → clips.json
    python person_clip_extractor.py --input video.mp4 --out-file clip.json
    python person_clip_extractor.py --device cpu

Default paths:
  input dir : /media/mili/EXTERNAL_USB/Editor_videos/procesar
  output dir: /media/mili/EXTERNAL_USB/Editor_videos/output
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

        if len(ids) == 1:  # exactly one person in frame
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


def process_video(video: Path, *, model: YOLO, conf: float, device) -> List[Dict]:
    segs = _segments(video, model, conf=conf, device=device)
    for s in segs:
        s["video_id"] = video.stem
    print(f"[done] {video.name}: {len(segs)} clip(s)")
    return segs


def process_many(
    in_dir: Path,
    *,
    weights: str | Path,
    conf: float,
    device: Union[str, int],
    patterns: Iterable[str],
) -> List[Dict]:
    vids = _videos(in_dir, patterns)
    if not vids:
        print(f"No videos in {in_dir}")
        return []

    model = _load_model(weights, device)
    all_clips: List[Dict] = []
    for v in vids:
        all_clips.extend(process_video(v, model=model, conf=conf, device=device))
    return all_clips

# ────────── CLI ───────────

def _cli():
    p = argparse.ArgumentParser(description="Save timestamps (single JSON) where exactly one person is visible")
    p.add_argument("--input", help="single video file")
    p.add_argument("--input-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/procesar")
    p.add_argument("--out-file", help="JSON path to write")
    p.add_argument("--output-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/output")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="0", help="GPU id or 'cpu'")
    p.add_argument("--patterns", nargs="*", default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"],
                   help="glob patterns for videos")
    args = p.parse_args()

    out_file = Path(args.out_file) if args.out_file else Path(args.output_dir) / "clips.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        model = _load_model(args.model, args.device)
        clips = process_video(Path(args.input), model=model, conf=args.conf, device=args.device)
    else:
        clips = process_many(Path(args.input_dir), weights=args.model, conf=args.conf, device=args.device, patterns=args.patterns)

    if clips:
        out_file.write_text(json.dumps(clips, indent=2))
        print(f"[json] wrote {len(clips)} clip(s) to {out_file}")
    else:
        print("No clips detected; nothing written")


if __name__ == "__main__":
    _cli()
