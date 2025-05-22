#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU ready, persistent IDs, min‑duration filter

Extract segments where exactly **one** person appears. All clips from every
video are saved into a single JSON file.

* Clips shorter than 100 ms are ignored.
* The same logical person keeps the same `person_id` across videos using simple
  face‑embedding re‑identification.
* GPU/CPU autodetect: pass `--device 0` (default), `--device 1`, or
  `--device cpu`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR  = Path("/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/procesar")
DEFAULT_OUTPUT_DIR = Path("/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/output")
DEFAULT_OUT_FILE   = "clips.json"
MIN_DURATION_MS    = 100  # ignore segments shorter than this

# ---------------------------------------------------------------------------
# face‑recognition dependency (fail fast with clear hint)
# ---------------------------------------------------------------------------
try:
    import face_recognition as fr  # type: ignore
except ImportError:
    print(
        "Missing dependency: face_recognition.\n"
        "Install pre‑built wheels with:\n"
        "    pip install dlib-bin==19.24.2 face_recognition",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_device(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("cpu"):
        return "cpu"
    if raw.isdigit():
        return f"cuda:{raw}"
    if raw.startswith("cuda"):
        return raw
    return "cpu"


def load_model(device: str):
    torch_device = parse_device(device)
    return YOLO("yolov8n.pt").to(torch_device), torch_device


def get_face_id(frame, box_xyxy, known_faces: List[Tuple[any, int]], tol: float = 0.55) -> int:
    """Return a stable ID for the face inside *box_xyxy* (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = map(int, box_xyxy)
    rgb = frame[:, :, ::-1]  # BGR → RGB
    # face_recognition expects (top, right, bottom, left)
    encodings = fr.face_encodings(rgb, [(y1, x2, y2, x1)])
    if not encodings:
        return -1  # no face found
    enc = encodings[0]
    matches = fr.compare_faces([f[0] for f in known_faces], enc, tolerance=tol)
    if True in matches:
        return known_faces[matches.index(True)][1]
    new_id = len(known_faces) + 1
    known_faces.append((enc, new_id))
    return new_id


def process_video(model: YOLO, video_path: Path, device: str, known_faces: List[Tuple[any, int]]) -> List[dict]:
    clips: List[dict] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return clips

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ms_per_frame = 1000.0 / fps
    current_id = None
    start_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t_ms = frame_idx * ms_per_frame

        detections = model(frame, verbose=False, device=device)[0]
        persons = [d for d in detections.boxes if int(d.cls[0]) == 0]

        if len(persons) == 1:
            x1, y1, x2, y2 = map(float, persons[0].xyxy[0])
            pid = get_face_id(frame, (x1, y1, x2, y2), known_faces)

            if current_id is None:
                current_id = pid
                start_ms = t_ms
            elif pid != current_id and pid != -1:
                duration = t_ms - start_ms
                if duration >= MIN_DURATION_MS:
                    clips.append({
                        "video_id": video_path.stem,
                        "person_id": current_id,
                        "start_ms": int(start_ms),
                        "end_ms": int(t_ms)
                    })
                current_id = pid
                start_ms = t_ms
        else:
            if current_id is not None:
                duration = t_ms - start_ms
                if duration >= MIN_DURATION_MS:
                    clips.append({
                        "video_id": video_path.stem,
                        "person_id": current_id,
                        "start_ms": int(start_ms),
                        "end_ms": int(t_ms)
                    })
            current_id = None

    # flush last segment
    if current_id is not None:
        end_ms = cap.get(cv2.CAP_PROP_POS_FRAMES) * ms_per_frame
        if end_ms - start_ms >= MIN_DURATION_MS:
            clips.append({
                "video_id": video_path.stem,
                "person_id": current_id,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms)
            })

    cap.release()
    return clips

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--input", type=Path, help="single video file")
    g.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="directory of videos")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--out-file", default=DEFAULT_OUT_FILE)
    ap.add_argument("--device", default="0", help="CUDA device id like 0,1 or 'cpu'")
    args = ap.parse_args()

    torch_device = parse_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, torch_device = load_model(args.device)
    known_faces: List[Tuple[any, int]] = []
    videos = [args.input] if args.input else sorted(args.input_dir.glob("*.mp4"))

    all_clips: List[dict] = []
    for vid in videos:
        all_clips.extend(process_video(model, vid, torch_device, known_faces))
        print(f"Processed {vid.name}")

    out_path = args.output_dir / args.out_file
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_clips, fh, indent=2)
    print(f"Saved {len(all_clips)} clips → {out_path}")


if __name__ == "__main__":
    main()
