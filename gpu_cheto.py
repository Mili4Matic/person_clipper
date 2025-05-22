#!/usr/bin/env python3
"""
person_clip_extractor.py – RTX 4070 BEAST MODE 🔥 (memory-safe edition)

Extract segments where exactly one person appears – tuned for maximal GPU
utilisation but with aggressive RAM/VRAM safeguards.

Dependencies
------------
    pip install ultralytics torch torchvision opencv-python face_recognition dlib-bin
    # If you have CUDA 11.8:
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""
from __future__ import annotations

import argparse, json, sys, time, gc, psutil, os, threading, multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue  # noqa: F401 (kept in case you use Queue later)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------------- optional face-ID support ----------------------------
try:
    import face_recognition
    FACE_OK = True
except ImportError:
    FACE_OK = False
    print("[warn] face_recognition not available – persistent IDs disabled", file=sys.stderr)

# ═════════════════════════ GPU BEAST MODE CONFIG ════════════════════════════
class GPUOptimizer:
    def __init__(self, device: str = "0"):
        self.device = device
        self.setup_beast_mode()

    def setup_beast_mode(self):
        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "memory_pool"):
            torch.cuda.memory_pool.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        print(f"[🔥] BEAST MODE ACTIVATED for RTX 4070!")
        self.print_gpu_info()

    def print_gpu_info(self):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1024**3
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU] {props.name}")
            print(f"[MEM] Total: {total:.1f} GB | Alloc: {alloc:.1f} GB | Reserved: {reserved:.1f} GB")
            print(f"[CUDA] Cores: {props.multi_processor_count} | CC: {props.major}.{props.minor}")

# ══════════════════════ EFFICIENT VIDEO READER ══════════════════════════════
class MemoryEfficientVideoReader:
    def __init__(self, video_path: Path, skip_frames: int = 1, max_frames: int | None = None):
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.cap: cv2.VideoCapture | None = None
        tmp = cv2.VideoCapture(str(video_path))
        self.fps = tmp.get(cv2.CAP_PROP_FPS) or 30
        total = int(tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        tmp.release()
        self.frame_indices = list(range(0, total, skip_frames))
        if max_frames:
            self.frame_indices = self.frame_indices[:max_frames]
        print(f"[reader] {video_path.name}: {len(self.frame_indices)} frames")

    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        return self

    def __exit__(self, *_):
        if self.cap:
            self.cap.release()
            self.cap = None

    # read a batch by seeking only when needed
    def read_batch(self, start_idx: int, batch_size: int):
        frames, indices = [], []
        end_idx = min(start_idx + batch_size, len(self.frame_indices))
        for i in range(start_idx, end_idx):
            fidx = self.frame_indices[i]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = self.cap.read()
            if ok:
                frames.append(frame)
                indices.append(fidx)
        return frames, indices

    def __len__(self):
        return len(self.frame_indices)

# ═══════════════════════ MEMORY MANAGEMENT HELPERS ═════════════════════════
class MemoryManager:
    @staticmethod
    def cleanup_all():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_usage():
        proc = psutil.Process(os.getpid())
        ram_mb = proc.memory_info().rss / 1024 / 1024
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        return ram_mb, vram_mb

    @staticmethod
    def check_memory_limit(max_ram_gb: float = 8.0):
        ram_mb, _ = MemoryManager.get_memory_usage()
        if ram_mb / 1024 > max_ram_gb:
            print(f"[⚠️] High RAM ({ram_mb/1024:.1f} GB) – forcing cleanup")
            MemoryManager.cleanup_all()
            return True
        return False

# ═════════════════════ CLIP UTILITIES (unchanged) ══════════════════════════
def _ms(frame: int, fps: float) -> int:  # frame index → milliseconds
    return int(frame / fps * 1000)

def _match_fast(enc, known: List[Dict], next_id: list[int], tol: float = 0.6) -> int:
    if enc is None or not known or not FACE_OK:
        pid = next_id[0]
        next_id[0] += 1
        return pid
    if len(known) > 10:
        encs = np.array([k["enc"] for k in known])
        dists = np.linalg.norm(encs - enc, axis=1)
        idx = np.argmin(dists)
        if dists[idx] <= tol:
            return known[idx]["id"]
    else:
        matches = face_recognition.compare_faces([k["enc"] for k in known], enc, tolerance=tol)
        if True in matches:
            return known[matches.index(True)]["id"]
    pid = next_id[0]
    known.append({"id": pid, "enc": enc})
    next_id[0] += 1
    return pid

def _filter_and_merge_clips(clips: List[Dict], min_dur_ms: int = 100, gap_ms: int = 500):
    if not clips:
        return []
    clips = [c for c in clips if (c["end_ms"] - c["start_ms"]) >= min_dur_ms]
    if not clips:
        return []
    clips.sort(key=lambda x: (x["video_id"], x["person_id"], x["start_ms"]))
    merged, cur = [], None
    for c in clips:
        if (cur is None or c["video_id"] != cur["video_id"] or c["person_id"] != cur["person_id"]
                or c["start_ms"] - cur["end_ms"] > gap_ms):
            if cur is not None:
                merged.append(cur)
            cur = c.copy()
        else:
            cur["end_ms"] = c["end_ms"]
    if cur is not None:
        merged.append(cur)
    return merged

# ═════════════════════════ PROCESSOR CLASS ═════════════════════════════════
class RTX4070BeastProcessor:
    def __init__(self, weights: str, device: str = "0", conf: float = 0.25):
        self.device = device
        self.conf = conf
        self.gpu_opt = GPUOptimizer(device)
        self.model = self._load_model(weights)
    # ----------------------------------------------------------------------
    def _load_model(self, weights: str):
        print("[🚀] Loading YOLO model…")
        model = YOLO(weights).to(self.device)
        if self.device != "cpu":
            try:
                model.model.half()
                print("[⚡] Half precision enabled")
            except Exception as e:
                print("[warn] Could not switch to FP16:", e)
            try:
                if hasattr(torch, "compile"):
                    model.model = torch.compile(model.model, mode="max-autotune", fullgraph=True, dynamic=False)
                    print("[🔥] Model compiled with max-autotune")
            except Exception as e:
                print("[warn] torch.compile failed:", e)
        return model
    # ----------------------------------------------------------------------
    def process_video_memory_safe(
        self,
        video_path: Path,
        batch_size: int = 32,
        skip_frames: int = 2,
        max_frames: int | None = None,
        known: List[Dict] | None = None,
        next_id: list[int] | None = None,
    ) -> List[Dict]:
        if known is None:
            known = []
        if next_id is None:
            next_id = [1]

        print(f"\n[🔥] MEMORY-SAFE PROCESSING: {video_path.name}")
        segs: List[Dict] = []
        st = end = pid = None
        t0 = time.time()
        processed = 0

        with MemoryEfficientVideoReader(video_path, skip_frames, max_frames) as reader:
            batches = (len(reader) + batch_size - 1) // batch_size
            for b in range(batches):
                batch_start = b * batch_size
                frames, idxs = reader.read_batch(batch_start, batch_size)
                if not frames:
                    continue
                try:
                    with torch.cuda.amp.autocast(True):
                        res = self.model(
                            frames, classes=[0], conf=self.conf, device=self.device,
                            verbose=False, agnostic_nms=True, max_det=5, half=True
                        )
                except Exception as e:
                    print(f"[error] Batch {b}: {e}")
                    del frames
                    MemoryManager.cleanup_all()
                    continue

                for r, fi in zip(res, idxs):
                    count = sum(1 for box in r.boxes if int(box.cls[0]) == 0) if r.boxes else 0
                    if count == 1:
                        if st is None:
                            pid = next_id[0]
                            next_id[0] += 1
                            st = int(fi)
                        end = int(fi)
                    else:
                        if st is not None:
                            segs.append({"person_id": pid, "start_ms": _ms(st, reader.fps), "end_ms": _ms(end, reader.fps)})
                            st = end = pid = None

                del frames, res
                processed += len(idxs)

                if b % 10 == 0:
                    MemoryManager.cleanup_all()
                    ram_mb, vram_mb = MemoryManager.get_memory_usage()
                    fps = processed / (time.time() - t0)
                    print(f"[⚡] Batch {b}/{batches}: {fps:.1f} fps | RAM {ram_mb:.0f} MB | VRAM {vram_mb:.0f} MB")
                    MemoryManager.check_memory_limit(max_ram_gb=6.0)

        if st is not None:
            segs.append({"person_id": pid, "start_ms": _ms(st, reader.fps), "end_ms": _ms(end, reader.fps)})

        MemoryManager.cleanup_all()
        dt = time.time() - t0
        print(f"[🏁] {video_path.name}: {processed} frames in {dt:.1f}s ({processed/dt:.1f} fps)")
        print(f"[📊] Found {len(segs)} raw segments")
        return segs

# ═════════════════════ WRAPPERS FOR CLI (NEW) ═══════════════════════════════
def process_video_safe(
    video: Path,
    processor: RTX4070BeastProcessor,
    batch_size: int,
    skip_frames: int,
    max_frames: int | None,
    known: List[Dict],
    next_id: list[int],
) -> List[Dict]:
    clips = processor.process_video_memory_safe(video, batch_size, skip_frames, max_frames, known, next_id)
    for c in clips:
        c["video_id"] = video.stem
    return clips

def process_many_memory_safe(
    dir_in: Path,
    weights: str,
    conf: float,
    device: str,
    patterns: List[str],
    batch_size: int,
    skip_frames: int,
    max_frames: int | None,
    save_interval: int = 5,
) -> List[Dict]:
    vids: List[Path] = []
    for pat in patterns:
        vids.extend(dir_in.glob(pat))
    vids.sort()
    if not vids:
        print(f"No videos found in {dir_in}")
        return []

    print(f"[🎯] Will scan {len(vids)} videos")
    proc = RTX4070BeastProcessor(weights, device, conf)
    known: List[Dict] = []
    next_id = [1]
    clips_all: List[Dict] = []

    for i, vid in enumerate(vids, 1):
        clips = process_video_safe(vid, proc, batch_size, skip_frames, max_frames, known, next_id)
        clips_all.extend(clips)
        if i % save_interval == 0:
            print(f"[💾] Checkpoint after {i} videos – {len(clips_all)} clips so far")

    return _filter_and_merge_clips(clips_all)

# ═════════════════════════════ CLI ═════════════════════════════════════════
def _cli():
    ap = argparse.ArgumentParser(
        prog="person_clip_extractor.py",
        description="RTX 4070 BEAST MODE – single-person clip extractor (memory-safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔥 PRESETS
  --preset ultra   : Highest speed  (batch 48, skip 3)
  --preset beast   : Balanced       (batch 32, skip 2)
  --preset turbo   : Conservative   (batch 24, skip 1)
""",
    )
    # IO
    ap.add_argument("--input", help="Single video file")
    ap.add_argument("--input-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/procesar")
    ap.add_argument("--out-file")
    ap.add_argument("--output-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/output")
    # model/GPU
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patterns", nargs="*", default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"])
    # filtering/processing
    ap.add_argument("--min-duration", type=int, default=100)
    ap.add_argument("--merge-gap", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-frames", type=int, default=2)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--save-interval", type=int, default=5)
    ap.add_argument("--max-ram-gb", type=float, default=6.0)
    ap.add_argument("--preset", choices=["ultra", "beast", "turbo"])
    args = ap.parse_args()

    # apply presets (memory-safe values)
    if args.preset == "ultra":
        args.batch_size, args.skip_frames = 48, 3
        print("[🔥] ULTRA preset")
    elif args.preset == "beast":
        args.batch_size, args.skip_frames = 32, 2
        print("[⚡] BEAST preset")
    elif args.preset == "turbo":
        args.batch_size, args.skip_frames = 24, 1
        print("[🚀] TURBO preset")

    # system info
    print(f"[💻] {mp.cpu_count()} CPU cores | {psutil.virtual_memory().total/1024**3:.1f} GB RAM")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[🎮] GPU: {name} ({vram:.1f} GB)")

    out_path = Path(args.out_file) if args.out_file else Path(args.output_dir) / "clips.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.input:
        proc = RTX4070BeastProcessor(args.model, args.device, args.conf)
        clips = process_video_safe(
            Path(args.input), proc,
            args.batch_size, args.skip_frames, args.max_frames, [], [1]
        )
        clips = _filter_and_merge_clips(clips, args.min_duration, args.merge_gap)
    else:
        clips = process_many_memory_safe(
            Path(args.input_dir), args.model, args.conf, args.device,
            args.patterns, args.batch_size, args.skip_frames,
            args.max_frames, args.save_interval
        )

    # save
    if clips:
        out_path.write_text(json.dumps(clips, indent=2))
        total_dur = sum(c["end_ms"] - c["start_ms"] for c in clips) / 1000
        print(f"\n[🏆] Saved {len(clips)} clips → {out_path}")
        print(f"[📊] Total single-person content: {total_dur:.1f} s")
    else:
        print("[❌] No clips found")
    print(f"[⏱️] Finished in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    _cli()
