#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU ready, stable IDs, min‑duration filter

Detects segments where **exactly one** person is visible and stores every clip
from every video into a single JSON file.

Improvements
------------
* **Robust face re‑identification** – instead of handing YOLO person boxes
directly to dlib (which caused signature errors on some builds), we now:
  1. Crop the detected person box.
  2. Run `face_recognition.face_locations()` on the crop.
  3. Compute the encoding only if *exactly one* face is found.
* Clips shorter than `MIN_DURATION_MS` (100 ms) are skipped.
* GPU/CPU autodetect: `--device 0` (default), `--device 1`, or `--device cpu`.

Output (`clips.json`)
---------------------
Each line is a dict:
```
{
  "video_id": "video_name",
  "person_id": 3,
  "start_ms": 12000,
  "end_ms"  : 22000
}
```
The same logical person keeps the same `person_id` across clips and videos.
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
DEFAULT_INPUT_DIR  = Path("/media/mili/EXTERNAL_USB/Editor_videos/procesar")
DEFAULT_OUTPUT_DIR = Path("/media/mili/EXTERNAL_USB/Editor_videos/output")
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
    """Return a stable ID for the face inside *box_xyxy* (x1, y1, x2, y2).

    Strategy: crop the person box, detect faces inside that crop, encode the
    first face (if any). This avoids the dlib signature mismatch seen when
    passing our own landmarks.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    # sanity check crop bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return -1

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return -1

    rgb_crop = crop[:, :, ::-1]  # BGR → RGB
    faces = fr.face_locations(rgb_crop, model="hog")  # fast HOG detector
    if len(faces) != 1:
        return -1

    enc = fr.face_encodings(rgb_crop, faces)[0]
    if not known_faces:
        new_id = 1
        known_faces.append((enc, new_id))
        return new_id

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

        preds = model(frame, verbose=False, device=device)[0]
        persons = [d for d in preds.boxes if int(d.cls[0]) == 0]

        if len(persons) == 1:
            x1, y1, x2, y2 = persons[0].xyxy[0].tolist()
            pid = get_face_id(frame, (x1, y1, x2, y2), known_faces)

            if current_id is None and pid != -1:
                current_id = pid
                start_ms = t_ms
            elif pid != current_id and pid != -1:
                # close previous
                if t_ms - start_ms >= MIN_DURATION_MS:
                    clips.append({
                        "video_id": video_path.stem,
                        "person_id": current_id,
                        "start_ms": int(start_ms),
                        "end_ms": int(t_ms)
                    })
                current_id = pid
                start_ms = t_ms
        else:
            # zero or multiple persons – close current segment if any
            if current_id is not None:
                if t_ms - start_ms >= MIN_DURATION_MS:
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
    g.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="directory of videos (.mp4)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--out-file", default=DEFAULT_OUT_FILE)
    ap.add_argument("--device", default="0", help="CUDA id like 0,1 or 'cpu'")
    args = ap.parse_args()

    model, torch_device = load_model(args.device)

    # gather videos
    videos = [args.input] if args.input else sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        print("No videos found", file=sys.stderr)
        sys.exit(1)

    known_faces: List[Tuple[any, int]] = []
    all_clips: List[dict] = []

    for vid in videos:
        print(f"Processing {vid.name}…")
        all_clips.extend(process_video(model, vid, torch_device, known_faces))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.out_file
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_clips, fh, indent=2)
    print(f"Saved {len(all_clips)} clips → {out_path}")


if __name__ == "__main__":
    main()
