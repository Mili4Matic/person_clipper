#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU ready, persistent IDs

Extract segments (start_ms / end_ms) where exactly **one** person appears across
multiple videos.  All clips are written into a single JSON file.

Person identities persist across files by comparing face embeddings.

Dependencies (Python ≥ 3.9)
──────────────────────────
Option A – pre‑built wheels (recommended, no compilation):
    pip install dlib-bin face_recognition opencv-python ultralytics

Option B – build dlib from source (Linux):
    sudo apt install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
    pip install dlib face_recognition opencv-python ultralytics

If **face_recognition** is missing at runtime, the script still works but will
assign incremental IDs per appearance (no cross‑video matching).

Default paths:
  input dir : /media/linuxbida/EXTERNAL_USB/Editor_videos/procesar
  output dir: /media/linuxbida/EXTERNAL_USB/Editor_videos/output
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


def _filter_and_merge_clips(clips: List[Dict], min_duration_ms: int = 100, merge_gap_ms: int = 500) -> List[Dict]:
    """
    Filter out clips shorter than min_duration_ms and merge consecutive clips 
    from the same person that are separated by less than merge_gap_ms.
    """
    if not clips:
        return []
    
    # First, filter out clips that are too short
    filtered_clips = []
    for clip in clips:
        duration = clip["end_ms"] - clip["start_ms"]
        if duration >= min_duration_ms:
            filtered_clips.append(clip)
    
    if not filtered_clips:
        return []
    
    # Sort clips by video_id and start_ms to ensure proper ordering
    filtered_clips.sort(key=lambda x: (x["video_id"], x["start_ms"]))
    
    # Group clips by video_id first
    video_groups = {}
    for clip in filtered_clips:
        video_id = clip["video_id"]
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(clip)
    
    # Merge clips within each video
    merged_clips = []
    for video_id, video_clips in video_groups.items():
        if not video_clips:
            continue
            
        # Sort clips by start time
        video_clips.sort(key=lambda x: x["start_ms"])
        
        current_clip = video_clips[0].copy()
        
        for next_clip in video_clips[1:]:
            # Check if clips are from same person and close enough to merge
            if (current_clip["person_id"] == next_clip["person_id"] and 
                next_clip["start_ms"] - current_clip["end_ms"] <= merge_gap_ms):
                # Merge clips by extending the end time
                current_clip["end_ms"] = next_clip["end_ms"]
            else:
                # Save current clip and start a new one
                merged_clips.append(current_clip)
                current_clip = next_clip.copy()
        
        # Don't forget the last clip
        merged_clips.append(current_clip)
    
    return merged_clips


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
    
    # Apply filtering and merging
    clips = _filter_and_merge_clips(clips)
    
    return clips

# ────────── CLI ───────────

def _cli():
    p = argparse.ArgumentParser(description="Detect clips with exactly one person; output single JSON; persistent IDs if face_recognition is available")
    p.add_argument("--input")
    p.add_argument("--input-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/procesar")
    p.add_argument("--out-file")
    p.add_argument("--output-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/output")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="0")
    p.add_argument("--patterns", nargs="*", default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"])
    p.add_argument("--min-duration", type=int, default=100, help="Minimum clip duration in ms (default: 100)")
    p.add_argument("--merge-gap", type=int, default=500, help="Maximum gap in ms to merge consecutive clips from same person (default: 500)")
    args = p.parse_args()

    of = Path(args.out_file) if args.out_file else Path(args.output_dir) / "clips.json"
    of.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        model = _load(args.model, args.device)
        known: List[Dict] = []
        next_id = [1]
        clips = process_video(Path(args.input), model=model, conf=args.conf,
                               device=args.device, known=known, next_id=next_id)
        # Apply filtering and merging for single video processing too
        clips = _filter_and_merge_clips(clips, args.min_duration, args.merge_gap)
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
