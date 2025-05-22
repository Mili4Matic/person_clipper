#!/usr/bin/env python3
"""
person_clip_extractor.py – GPU optimized for RTX 4070, persistent IDs

Extract segments (start_ms / end_ms) where exactly **one** person appears across
multiple videos. Optimized for batch processing on modern GPUs.

Dependencies (Python ≥ 3.9)
──────────────────────────
Option A – pre‑built wheels (recommended, no compilation):
    pip install dlib-bin face_recognition opencv-python ultralytics torch torchvision

Option B – build dlib from source (Linux):
    sudo apt install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
    pip install dlib face_recognition opencv-python ultralytics torch torchvision

If **face_recognition** is missing at runtime, the script still works but will
assign incremental IDs per appearance (no cross‑video matching).

Default paths:
  input dir : /media/linuxbida/EXTERNAL_USB/Editor_videos/procesar
  output dir: /media/linuxbida/EXTERNAL_USB/Editor_videos/output
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from typing import List, Dict, Iterable, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import torch
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


class FrameBatch:
    """Handles batched frame processing for GPU optimization"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.frames = []
        self.frame_indices = []
        
    def add_frame(self, frame: np.ndarray, idx: int):
        self.frames.append(frame)
        self.frame_indices.append(idx)
        
    def is_full(self) -> bool:
        return len(self.frames) >= self.batch_size
        
    def clear(self):
        self.frames.clear()
        self.frame_indices.clear()
        
    def get_batch(self):
        return self.frames.copy(), self.frame_indices.copy()


def _preload_frames(video_path: Path, max_frames: int = None, skip_frames: int = 1) -> tuple[List[np.ndarray], float]:
    """
    Preload frames into memory for batch processing.
    skip_frames: process every Nth frame (1 = all frames, 2 = every other frame, etc.)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    frames = []
    frame_idx = 0
    actual_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % skip_frames == 0:
            frames.append(frame)
            if max_frames and actual_idx >= max_frames:
                break
            actual_idx += 1
            
        frame_idx += 1
    
    cap.release()
    print(f"[preload] {video_path.name}: loaded {len(frames)} frames (skip={skip_frames})")
    return frames, fps


def _segments_optimized(video: Path, model: YOLO, *, conf: float, device: Union[str, int],
                       known: List[Dict], next_id: list[int], batch_size: int = 32,
                       max_frames: int = None, skip_frames: int = 1, use_half: bool = True) -> List[Dict]:
    """
    Optimized segment detection with batch processing and memory preloading
    """
    # Enable half precision for RTX 4070 (significant speedup)
    if use_half and device != "cpu":
        model.model.half()
    
    # Preload frames for faster access
    frames, fps = _preload_frames(video, max_frames, skip_frames)
    if not frames:
        return []
    
    segs: List[Dict] = []
    st = end = None
    pid = None
    
    # Process frames in batches
    batch = FrameBatch(batch_size)
    all_results = {}
    
    print(f"[processing] {video.name}: {len(frames)} frames in batches of {batch_size}")
    start_time = time.time()
    
    # Batch processing
    for idx, frame in enumerate(frames):
        batch.add_frame(frame, idx * skip_frames)  # Adjust index for skipped frames
        
        if batch.is_full() or idx == len(frames) - 1:
            batch_frames, batch_indices = batch.get_batch()
            
            # Process entire batch at once
            results = model(batch_frames, classes=[0], conf=conf, device=device, verbose=False)
            
            # Store results
            for result, frame_idx in zip(results, batch_indices):
                all_results[frame_idx] = result
                
            batch.clear()
    
    processing_time = time.time() - start_time
    print(f"[batch] {video.name}: processed in {processing_time:.2f}s ({len(frames)/processing_time:.1f} fps)")
    
    # Now process results sequentially to maintain temporal consistency
    for idx in range(0, len(frames) * skip_frames, skip_frames):
        if idx not in all_results:
            continue
            
        r = all_results[idx]
        
        # Extract person detections
        if r.boxes is None or len(r.boxes) == 0:
            person_count = 0
            bbox = None
        else:
            person_boxes = []
            for box in r.boxes:
                if box.cls is not None and int(box.cls[0]) == 0:  # person class
                    person_boxes.append(box.xyxy[0].cpu().tolist())
            
            person_count = len(person_boxes)
            bbox = person_boxes[0] if person_boxes else None
        
        if person_count == 1:
            if st is None:
                # Get original frame for face encoding (not from results)
                orig_frame = frames[idx // skip_frames] if idx // skip_frames < len(frames) else None
                enc = _face_encoding(orig_frame, bbox) if orig_frame is not None else None
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


def _segments_streaming(video: Path, model: YOLO, *, conf: float, device: Union[str, int],
                       known: List[Dict], next_id: list[int]) -> List[Dict]:
    """
    Original streaming method - kept for comparison and as fallback
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    segs: List[Dict] = []
    st = end = None
    pid = None

    stream = model.track(source=str(video), stream=True, classes=[0], conf=conf,
                         persist=True, device=device, verbose=False)

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

def _load(weights: str | Path, device: Union[str, int], optimize: bool = True):
    m = YOLO(str(weights))
    if device not in ("cpu", "-1"):
        m.to(f"cuda:{device}" if str(device).isdigit() else device)
        
    # RTX 4070 optimizations
    if optimize and device != "cpu":
        # Enable optimized attention and memory efficient attention
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Compile model for better performance (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                m.model = torch.compile(m.model, mode="reduce-overhead")
                print("[opt] Model compiled with torch.compile")
        except Exception as e:
            print(f"[warn] Could not compile model: {e}")
    
    return m


# ───────── processing ──────────

def process_video(video: Path, *, model: YOLO, conf: float, device, known, next_id, 
                 use_batch: bool = True, batch_size: int = 32, skip_frames: int = 1,
                 max_frames: int = None, use_half: bool = True) -> List[Dict]:
    
    if use_batch:
        segs = _segments_optimized(video, model, conf=conf, device=device, known=known, 
                                  next_id=next_id, batch_size=batch_size, 
                                  max_frames=max_frames, skip_frames=skip_frames, use_half=use_half)
    else:
        segs = _segments_streaming(video, model, conf=conf, device=device, known=known, next_id=next_id)
    
    for s in segs:
        s["video_id"] = video.stem
    print(f"[done] {video.name}: {len(segs)} clip(s)")
    return segs


def process_many(dir_in: Path, *, weights, conf, device, patterns, use_batch: bool = True,
                batch_size: int = 32, skip_frames: int = 1, max_frames: int = None,
                use_half: bool = True, parallel_videos: int = 1) -> List[Dict]:
    """
    Process multiple videos with GPU optimizations
    parallel_videos: Process multiple videos in parallel (be careful with VRAM usage)
    """
    vids = _videos(dir_in, patterns)
    if not vids:
        print(f"No videos in {dir_in}")
        return []

    model = _load(weights, device, optimize=True)
    clips: List[Dict] = []
    
    if parallel_videos > 1 and len(vids) > 1:
        print(f"[parallel] Processing {len(vids)} videos with {parallel_videos} parallel workers")
        
        # Shared state for face recognition
        known: List[Dict] = []
        next_id = [1]
        lock = threading.Lock()
        
        def process_video_thread(video):
            with lock:
                local_known = known.copy()
                local_next_id = [next_id[0]]
            
            video_clips = process_video(video, model=model, conf=conf, device=device,
                                      known=local_known, next_id=local_next_id,
                                      use_batch=use_batch, batch_size=batch_size,
                                      skip_frames=skip_frames, max_frames=max_frames,
                                      use_half=use_half)
            
            with lock:
                # Update global state
                for new_person in local_known[len(known):]:
                    known.append(new_person)
                next_id[0] = max(next_id[0], local_next_id[0])
            
            return video_clips
        
        with ThreadPoolExecutor(max_workers=parallel_videos) as executor:
            futures = [executor.submit(process_video_thread, v) for v in vids]
            for future in futures:
                clips.extend(future.result())
    else:
        # Sequential processing
        known: List[Dict] = []
        next_id = [1]
        for v in vids:
            clips.extend(process_video(v, model=model, conf=conf, device=device,
                                     known=known, next_id=next_id, use_batch=use_batch,
                                     batch_size=batch_size, skip_frames=skip_frames,
                                     max_frames=max_frames, use_half=use_half))
    
    # Apply filtering and merging
    clips = _filter_and_merge_clips(clips)
    
    return clips

# ────────── CLI ───────────

def _cli():
    p = argparse.ArgumentParser(description="GPU-optimized person clip detector for RTX 4070")
    p.add_argument("--input")
    p.add_argument("--input-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/procesar")
    p.add_argument("--out-file")
    p.add_argument("--output-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/output")
    p.add_argument("--model", default="yolov8n.pt", help="Model size: yolov8n.pt (fastest) to yolov8x.pt (most accurate)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="0")
    p.add_argument("--patterns", nargs="*", default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"])
    p.add_argument("--min-duration", type=int, default=100, help="Minimum clip duration in ms")
    p.add_argument("--merge-gap", type=int, default=500, help="Max gap in ms to merge consecutive clips")
    
    # GPU optimization options
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU processing (32-64 recommended for RTX 4070)")
    p.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (1=all, 2=half, etc.)")
    p.add_argument("--max-frames", type=int, help="Limit frames per video (for testing)")
    p.add_argument("--no-batch", action="store_true", help="Disable batch processing (use streaming)")
    p.add_argument("--no-half", action="store_true", help="Disable half precision (FP16)")
    p.add_argument("--parallel-videos", type=int, default=1, help="Process multiple videos in parallel")
    
    args = p.parse_args()

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[gpu] Using {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        print(f"[gpu] Batch size: {args.batch_size}, Half precision: {not args.no_half}")

    of = Path(args.out_file) if args.out_file else Path(args.output_dir) / "clips.json"
    of.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        model = _load(args.model, args.device, optimize=True)
        known: List[Dict] = []
        next_id = [1]
        clips = process_video(Path(args.input), model=model, conf=args.conf,
                             device=args.device, known=known, next_id=next_id,
                             use_batch=not args.no_batch, batch_size=args.batch_size,
                             skip_frames=args.skip_frames, max_frames=args.max_frames,
                             use_half=not args.no_half)
        clips = _filter_and_merge_clips(clips, args.min_duration, args.merge_gap)
    else:
        clips = process_many(Path(args.input_dir), weights=args.model, conf=args.conf,
                           device=args.device, patterns=args.patterns,
                           use_batch=not args.no_batch, batch_size=args.batch_size,
                           skip_frames=args.skip_frames, max_frames=args.max_frames,
                           use_half=not args.no_half, parallel_videos=args.parallel_videos)

    if clips:
        of.write_text(json.dumps(clips, indent=2))
        print(f"[json] wrote {len(clips)} clip(s) → {of}")
    else:
        print("No clips detected; nothing written")


if __name__ == "__main__":
    _cli()
