#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU ready, persistent IDs

Extract segments (start_ms / end_ms) where exactly **one** person appears across
multiple videos.  All clips are written into a single JSON file.

Person identities persist across files by comparing face embeddings.

Dependencies (Python ≥ 3.9)
──────────────────────────
Option A – pre‑built wheels (recommended, no compilation):
    pip install dlib-bin face_recognition opencv-python ultralytics

Option B – build dlib from source (Linux):
    sudo apt install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
    pip install dlib face_recognition opencv-python ultralytics

If **face_recognition** is missing at runtime, the script still works but will
assign incremental IDs per appearance (no cross‑video matching).

Default paths:
  input dir : /media/mili/EXTERNAL_USB/Editor_videos/procesar
  output dir: /media/mili/EXTERNAL_USB/Editor_videos/output
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Iterable, Union

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import face_recognition  # type: ignore
    FACE_OK = True
except ImportError as e:
    FACE_OK = False
    print("[warn] face_recognition not available – persistent IDs disabled", file=sys.stderr)

# ───────── helpers ──────────

def _ms(frame: int, fps: float) -> int:
    return int(frame / fps * 1000)


def _face_encoding(img: np.ndarray, box: list[float]):
    if not FACE_OK:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)
    return enc[0] if enc else None


def _match(enc, known: List[Dict], next_id: list[int], tol: float = 0.55) -> int:
    """Return persistent id for this encoding, adding a new one if needed."""
    if enc is None or not known or not FACE_OK:
        pid = next_id[0]
        if enc is not None and FACE_OK:
            known.append({"id": pid, "enc": enc})
        next_id[0] += 1
        return pid

    encs = [k["enc"] for k in known]
    match = face_recognition.compare_faces(encs, enc, tolerance=tol)
    if True in match:
        return known[match.index(True)]["id"]

    pid = next_id[0]
    known.append({"id": pid, "enc": enc})
    next_id[0] += 1
    return pid


# ───────── core detection ──────────

def _segments(video: Path, model: YOLO, *, conf: float, device: Union[str, int],
              known: List[Dict], next_id: list[int]) -> List[Dict]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    segs: List[Dict] = []
    st = end = None
    pid = None

    stream = model.track(source=str(video), stream=True, classes=[0], conf=conf,
                         persist=True, device=device)

    for idx, r in enumerate(stream):
        ids_t = r.boxes.id
        ids = [] if ids_t is None else list(set(ids_t.cpu().tolist()))

        if len(ids) == 1:
            tid = ids[0]
            boxes = r.boxes.xyxy.cpu().tolist()
            tids = ids_t.cpu().tolist()
            bbox = next(b for b, t in zip(boxes, tids) if t == tid)

            if st is None:
                enc = _face_encoding(r.orig_img, bbox)
                pid = _match(enc, known, next_id)
                st = idx
            end = idx
        else:
            if st is not None:
                segs.append({"person_id": pid, "start_ms": _ms(st, fps),
                             "end_ms": _ms(end, fps)})
                st = end = pid = None

    if st is not None:
        segs.append({"person_id": pid, "start_ms": _ms(st, fps),
                     "end_ms": _ms(end, fps)})

    return segs


def _videos(dir_: Path, pats: Iterable[str]) -> List[Path]:
    vids: List[Path] = []
    for p in pats:
        vids.extend(dir_.glob(p))
    return sorted(vids)


# ───────── model ──────────

def _load(weights: str | Path, device: Union[str, int]):
    m = YOLO(str(weights))
    if device not in ("cpu", "-1"):
        m.to(f"cuda:{device}" if str(device).isdigit() else device)
    return m


# ───────── processing ──────────

def process_video(video: Path, *, model: YOLO, conf: float, device, known, next_id) -> List[Dict]:
    segs = _segments(video, model, conf=conf, device=device, known=known, next_id=next_id)
    for s in segs:
        s["video_id"] = video.stem
    print(f"[done] {video.name}: {len(segs)} clip(s)")
    return segs


def process_many(dir_in: Path, *, weights, conf, device, patterns) -> List[Dict]:
    vids = _videos(dir_in, patterns)
    if not vids:
        print(f"No videos in {dir_in}")
        return []

    model = _load(weights, device)
    known: List[Dict] = []
    next_id = [1]
    clips: List[Dict] = []

    for v in vids:
        clips.extend(process_video(v, model=model, conf=conf, device=device,
                                   known=known, next_id=next_id))
    return clips

# ────────── CLI ───────────

def _cli():
    p = argparse.ArgumentParser(description="Detect clips with exactly one person; output single JSON; persistent IDs if face_recognition is available")
    p.add_argument("--input")
    p.add_argument("--input-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/procesar")
    p.add_argument("--out-file")
    p.add_argument("--output-dir", default="/media/mili/EXTERNAL_USB/Editor_videos/output")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="0")
    p.add_argument("--patterns", nargs="*", default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"])
    args = p.parse_args()

    of = Path(args.out_file) if args.out_file else Path(args.output_dir) / "clips.json"
    of.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        model = _load(args.model, args.device)
        known: List[Dict] = []
        next_id = [1]
        clips = process_video(Path(args.input), model=model, conf=args.conf,
                               device=args.device, known=known, next_id=next_id)
    else:
        clips = process_many(Path(args.input_dir), weights=args.model, conf=args.conf,
                             device=args.device, patterns=args.patterns)

    if clips:
        of.write_text(json.dumps(clips, indent=2))
        print(f"[json] wrote {len(clips)} clip(s) → {of}")
    else:
        print("No clips detected; nothing written")


if __name__ == "__main__":
    _cli()
