#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU ready, persistent IDs, min‑duration filter

Extracts segments (start_ms, end_ms) where exactly **one** person appears in a
video or a directory of videos. All clips are saved to a single JSON file.

New: clips shorter than MIN_DURATION_MS (=100 ms) are ignored.

JSON clip format:
    {"video_id": str, "person_id": int, "start_ms": int, "end_ms": int}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR  = Path("/media/mili/EXTERNAL_USB/Editor_videos/procesar")
DEFAULT_OUTPUT_DIR = Path("/media/mili/EXTERNAL_USB/Editor_videos/output")
DEFAULT_OUT_FILE   = "clips.json"
MIN_DURATION_MS    = 100  # <‑‑ ignore segments shorter than this

# ---------------------------------------------------------------------------
# (face recognition / re‑id helper imported lazily to avoid heavy deps if unused)
# ---------------------------------------------------------------------------
try:
    import face_recognition as fr  # type: ignore
except ImportError:
    print("face_recognition not installed. Install with: pip install dlib-bin face_recognition", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_model(device: str):
    return YOLO("yolov8n.pt").to(device)


def process_video(model: YOLO, video_path: Path, device: str, known_faces: List[tuple]) -> List[dict]:
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
            person = persons[0]
            # face crop for embedding (simple center crop around bbox head)
            x1, y1, x2, y2 = map(int, person.xyxy[0])
            face_crop = frame[max(y1,0):max(y1,0)+min((y2-y1)//2, frame.shape[0]-y1), x1:x2]
            if face_crop.size == 0:
                pid = -1  # fallback
            else:
                enc = fr.face_encodings(face_crop)
                if enc:
                    enc = enc[0]
                    matches = fr.compare_faces([f[0] for f in known_faces], enc, tolerance=0.55)
                    if True in matches:
                        pid = known_faces[matches.index(True)][1]
                    else:
                        pid = len(known_faces) + 1
                        known_faces.append((enc, pid))
                else:
                    pid = -1

            if current_id is None:
                # start new segment
                current_id = pid
                start_ms = t_ms
            elif pid != current_id:
                # person changed, close old segment first if long enough
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
            # Not exactly one person: close any open segment
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

    # flush last segment at EOF
    if current_id is not None:
        end_ms = cap.get(cv2.CAP_PROP_POS_FRAMES) * ms_per_frame
        duration = end_ms - start_ms
        if duration >= MIN_DURATION_MS:
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
    inp = ap.add_mutually_exclusive_group()
    inp.add_argument("--input", type=Path, help="single video file")
    inp.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="directory of videos")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--out-file", default=DEFAULT_OUT_FILE)
    ap.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_clips: List[dict] = []
    model = load_model(args.device)
    known_faces: List[tuple] = []  # list of (embedding, person_id)

    videos = [args.input] if args.input else sorted(args.input_dir.glob("*.mp4"))
    for vid in videos:
        all_clips.extend(process_video(model, vid, args.device, known_faces))
        print(f"Done {vid.name}")

    out_path = args.output_dir / args.out_file
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_clips, fh, indent=2)
    print(f"Saved {len(all_clips)} clips → {out_path}")


if __name__ == "__main__":
    main()
