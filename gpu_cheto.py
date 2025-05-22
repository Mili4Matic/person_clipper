#!/usr/bin/env python3
"""
person_clip_extractor.py – RTX 4070 BEAST MODE 🔥

Extract segments where exactly one person appears - MAXIMUM GPU UTILIZATION
Designed to push RTX 4070 laptop to its limits.

Dependencies:
    pip install ultralytics torch torchvision opencv-python face_recognition dlib-bin
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""
from __future__ import annotations

import argparse, json, sys, time, gc, psutil, os
from pathlib import Path
from typing import List, Dict, Iterable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import multiprocessing as mp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

try:
    import face_recognition
    FACE_OK = True
except ImportError:
    FACE_OK = False
    print("[warn] face_recognition not available – persistent IDs disabled", file=sys.stderr)

# ═══════════ GPU BEAST MODE CONFIGURATION ═══════════

class GPUOptimizer:
    def __init__(self, device: str = "0"):
        self.device = device
        self.setup_beast_mode()
    
    def setup_beast_mode(self):
        """Configure PyTorch for MAXIMUM performance"""
        if not torch.cuda.is_available():
            return
            
        # Maximum CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        # Memory management
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_pool'):
            torch.cuda.memory_pool.empty_cache()
        
        # Set memory fraction to use almost all VRAM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"[🔥] BEAST MODE ACTIVATED for RTX 4070!")
        self.print_gpu_info()
    
    def print_gpu_info(self):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU] {props.name}")
            print(f"[MEM] Total: {total_memory:.1f}GB | Allocated: {allocated:.1f}GB | Cached: {cached:.1f}GB")
            print(f"[CUDA] Cores: {props.multi_processor_count} | Capability: {props.major}.{props.minor}")


# ═══════════ ULTRA-FAST DATASET ═══════════

class VideoFrameDataset(Dataset):
    """Custom dataset for ultra-fast batch loading"""
    
    def __init__(self, video_path: Path, skip_frames: int = 1, max_frames: int = None):
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        
        # Pre-calculate frame indices
        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.frame_indices = list(range(0, total_frames, skip_frames))
        if max_frames:
            self.frame_indices = self.frame_indices[:max_frames]
        
        print(f"[dataset] {video_path.name}: {len(self.frame_indices)} frames to process")
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        
        # Fast frame extraction
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return black frame if failed
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return frame, frame_idx


class MultiThreadVideoLoader:
    """Asynchronous video frame loader with thread pool"""
    
    def __init__(self, video_path: Path, batch_size: int, skip_frames: int = 1, 
                 max_frames: int = None, num_workers: int = 4):
        self.dataset = VideoFrameDataset(video_path, skip_frames, max_frames)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
    def __iter__(self):
        return iter(self.dataloader)


# ═══════════ TURBO PROCESSING ENGINE ═══════════

def _ms(frame: int, fps: float) -> int:
    return int(frame / fps * 1000)


def _match_fast(enc, known: List[Dict], next_id: list[int], tol: float = 0.6) -> int:
    """Super fast face matching with optimizations"""
    if enc is None or not known or not FACE_OK:
        pid = next_id[0]
        next_id[0] += 1
        return pid

    # Use numpy for faster comparison
    if len(known) > 10:  # Only for larger face databases
        encs = np.array([k["enc"] for k in known])
        distances = np.linalg.norm(encs - enc, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= tol:
            return known[min_idx]["id"]
    else:
        # Use face_recognition for smaller databases
        encs = [k["enc"] for k in known]
        matches = face_recognition.compare_faces(encs, enc, tolerance=tol)
        if True in matches:
            return known[matches.index(True)]["id"]

    # Add new person
    pid = next_id[0]
    known.append({"id": pid, "enc": enc})
    next_id[0] += 1
    return pid


def _filter_and_merge_clips(clips: List[Dict], min_duration_ms: int = 100, merge_gap_ms: int = 500) -> List[Dict]:
    """Optimized clip filtering and merging"""
    if not clips:
        return []
    
    # Vectorized filtering
    filtered_clips = [c for c in clips if (c["end_ms"] - c["start_ms"]) >= min_duration_ms]
    
    if not filtered_clips:
        return []
    
    # Fast sorting
    filtered_clips.sort(key=lambda x: (x["video_id"], x["person_id"], x["start_ms"]))
    
    # Optimized merging
    merged_clips = []
    current_clip = None
    
    for clip in filtered_clips:
        if (current_clip is None or 
            current_clip["video_id"] != clip["video_id"] or
            current_clip["person_id"] != clip["person_id"] or
            clip["start_ms"] - current_clip["end_ms"] > merge_gap_ms):
            
            if current_clip is not None:
                merged_clips.append(current_clip)
            current_clip = clip.copy()
        else:
            # Merge clips
            current_clip["end_ms"] = clip["end_ms"]
    
    if current_clip is not None:
        merged_clips.append(current_clip)
    
    return merged_clips


# ═══════════ BEAST MODE PROCESSING ═══════════

class RTX4070BeastProcessor:
    """RTX 4070 optimized processing engine"""
    
    def __init__(self, weights: str, device: str = "0", conf: float = 0.25):
        self.gpu_opt = GPUOptimizer(device)
        self.device = device
        self.conf = conf
        
        # Load and optimize model
        self.model = self._load_beast_model(weights)
        
        # Processing stats
        self.total_frames = 0
        self.total_time = 0
        
    def _load_beast_model(self, weights: str):
        """Load YOLO model with MAXIMUM optimizations"""
        print(f"[🚀] Loading model in BEAST MODE...")
        
        model = YOLO(weights)
        model.to(self.device)
        
        # Enable all optimizations
        if self.device != "cpu":
            # Use half precision if possible
            try:
                model.model.half()
                print("[⚡] Half precision enabled")
            except:
                print("[⚠️] Half precision failed, using float32")
            
            # Compile model for maximum speed
            try:
                if hasattr(torch, 'compile'):
                    model.model = torch.compile(
                        model.model, 
                        mode="max-autotune",  # Maximum optimization
                        fullgraph=True,
                        dynamic=False
                    )
                    print("[🔥] Model compiled with max-autotune")
            except Exception as e:
                print(f"[warn] Compilation failed: {e}")
        
        return model
    
    def process_video_beast_mode(self, video_path: Path, batch_size: int = 64, 
                                skip_frames: int = 1, max_frames: int = None,
                                known: List[Dict] = None, next_id: list[int] = None) -> List[Dict]:
        """Process video with MAXIMUM RTX 4070 utilization"""
        
        if known is None:
            known = []
        if next_id is None:
            next_id = [1]
        
        print(f"\n[🔥] BEAST MODE PROCESSING: {video_path.name}")
        print(f"[⚙️] Batch size: {batch_size} | Skip frames: {skip_frames}")
        
        # Create turbo loader
        loader = MultiThreadVideoLoader(
            video_path, batch_size, skip_frames, max_frames, num_workers=6
        )
        
        segs: List[Dict] = []
        st = end = None
        pid = None
        
        start_time = time.time()
        processed_batches = 0
        
        # MAXIMUM SPEED PROCESSING
        with torch.cuda.amp.autocast(enabled=True):  # Mixed precision
            for batch_frames, batch_indices in loader:
                processed_batches += 1
                batch_start = time.time()
                
                # Convert to tensor for GPU processing
                if isinstance(batch_frames, (list, tuple)):
                    frames_tensor = torch.stack([torch.from_numpy(f) for f in batch_frames])
                else:
                    frames_tensor = batch_frames
                
                # Ultra-fast inference
                try:
                    results = self.model(
                        batch_frames.numpy() if hasattr(batch_frames, 'numpy') else batch_frames,
                        classes=[0],  # Only person class
                        conf=self.conf,
                        device=self.device,
                        verbose=False,
                        half=True,
                        agnostic_nms=True,  # Faster NMS
                        max_det=10  # Limit detections for speed
                    )
                except Exception as e:
                    print(f"[error] Batch processing failed: {e}")
                    continue
                
                # Process results FAST
                for result, frame_idx in zip(results, batch_indices.numpy() if hasattr(batch_indices, 'numpy') else batch_indices):
                    person_count = 0
                    bbox = None
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        person_boxes = []
                        for box in result.boxes:
                            if box.cls is not None and int(box.cls[0]) == 0:
                                person_boxes.append(box.xyxy[0].cpu().tolist())
                        
                        person_count = len(person_boxes)
                        bbox = person_boxes[0] if person_boxes else None
                    
                    # State machine for clip detection
                    if person_count == 1:
                        if st is None:
                            pid = next_id[0]  # Skip face encoding for speed
                            next_id[0] += 1
                            st = int(frame_idx)
                        end = int(frame_idx)
                    else:
                        if st is not None:
                            segs.append({
                                "person_id": pid, 
                                "start_ms": _ms(st, loader.dataset.fps),
                                "end_ms": _ms(end, loader.dataset.fps)
                            })
                            st = end = pid = None
                
                # Performance monitoring
                batch_time = time.time() - batch_start
                if processed_batches % 10 == 0:
                    fps = batch_size / batch_time
                    gpu_util = self._get_gpu_utilization()
                    print(f"[⚡] Batch {processed_batches}: {fps:.1f} fps | GPU: {gpu_util:.1f}%")
                
                # Memory management
                if processed_batches % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Final clip
        if st is not None:
            segs.append({
                "person_id": pid,
                "start_ms": _ms(st, loader.dataset.fps),
                "end_ms": _ms(end, loader.dataset.fps)
            })
        
        # Performance stats
        total_time = time.time() - start_time
        total_frames = len(loader.dataset)
        avg_fps = total_frames / total_time
        
        print(f"[🏁] COMPLETED: {total_frames} frames in {total_time:.2f}s")
        print(f"[📊] Average FPS: {avg_fps:.1f} | Found {len(segs)} segments")
        
        return segs
    
    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
        except:
            pass
        return 0.0


# ═══════════ MAIN PROCESSING ═══════════

def process_video_beast(video: Path, processor: RTX4070BeastProcessor, 
                        batch_size: int, skip_frames: int, max_frames: int,
                        known: List[Dict], next_id: list[int]) -> List[Dict]:
    """Process single video in beast mode"""
    
    segs = processor.process_video_beast_mode(
        video, batch_size, skip_frames, max_frames, known, next_id
    )
    
    for s in segs:
        s["video_id"] = video.stem
    
    return segs


def process_many_beast(dir_in: Path, weights: str, conf: float, device: str,
                      patterns: List[str], batch_size: int, skip_frames: int,
                      max_frames: int, parallel_videos: int) -> List[Dict]:
    """Process multiple videos in BEAST MODE"""
    
    # Find videos
    vids = []
    for pattern in patterns:
        vids.extend(dir_in.glob(pattern))
    vids = sorted(vids)
    
    if not vids:
        print(f"No videos found in {dir_in}")
        return []
    
    print(f"[🎯] Found {len(vids)} videos to process")
    
    # Create beast processor
    processor = RTX4070BeastProcessor(weights, device, conf)
    
    # Global face database
    known: List[Dict] = []
    next_id = [1]
    lock = threading.Lock()
    
    all_clips = []
    
    if parallel_videos > 1 and len(vids) > 1:
        print(f"[🚀] PARALLEL BEAST MODE: {parallel_videos} videos simultaneously")
        
        def process_video_thread(video):
            with lock:
                local_known = known.copy()
                local_next_id = [next_id[0]]
            
            clips = process_video_beast(
                video, processor, batch_size, skip_frames, max_frames,
                local_known, local_next_id
            )
            
            with lock:
                # Update global state
                for new_person in local_known[len(known):]:
                    known.append(new_person)
                next_id[0] = max(next_id[0], local_next_id[0])
            
            return clips
        
        with ThreadPoolExecutor(max_workers=parallel_videos) as executor:
            futures = {executor.submit(process_video_thread, vid): vid for vid in vids}
            
            for future in as_completed(futures):
                video = futures[future]
                try:
                    clips = future.result()
                    all_clips.extend(clips)
                    print(f"[✅] Completed: {video.name}")
                except Exception as e:
                    print(f"[❌] Failed: {video.name} - {e}")
    else:
        # Sequential processing
        for vid in vids:
            clips = process_video_beast(
                vid, processor, batch_size, skip_frames, max_frames, known, next_id
            )
            all_clips.extend(clips)
    
    # Apply filtering and merging
    filtered_clips = _filter_and_merge_clips(all_clips)
    
    return filtered_clips


def _videos(dir_: Path, pats: Iterable[str]) -> List[Path]:
    vids: List[Path] = []
    for p in pats:
        vids.extend(dir_.glob(p))
    return sorted(vids)


# ═══════════ CLI BEAST MODE ═══════════

def _cli():
    parser = argparse.ArgumentParser(
        description="RTX 4070 BEAST MODE - Maximum GPU utilization for person detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔥 BEAST MODE PRESETS:
  --preset ultra    : Maximum speed (batch=96, skip=3)
  --preset beast    : Balanced (batch=64, skip=2) 
  --preset turbo    : Conservative (batch=32, skip=1)
        """
    )
    
    # Basic options
    parser.add_argument("--input", help="Single video file")
    parser.add_argument("--input-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/data/videos")
    parser.add_argument("--out-file", help="Output JSON file")
    parser.add_argument("--output-dir", default="/media/linuxbida/EXTERNAL_USB/Editor_videos/testing/output")
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt (fastest) to yolov8x.pt (accurate)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument("--patterns", nargs="*", default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"])
    
    # Filtering
    parser.add_argument("--min-duration", type=int, default=100, help="Min clip duration (ms)")
    parser.add_argument("--merge-gap", type=int, default=500, help="Max gap to merge clips (ms)")
    
    # BEAST MODE options
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (32-128 for RTX 4070)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, help="Limit frames (for testing)")
    parser.add_argument("--parallel-videos", type=int, default=1, help="Process N videos in parallel")
    
    # Presets
    parser.add_argument("--preset", choices=["ultra", "beast", "turbo"], 
                       help="Performance preset")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == "ultra":
        args.batch_size = 96
        args.skip_frames = 3
        print("[🔥] ULTRA PRESET: Maximum speed, may sacrifice accuracy")
    elif args.preset == "beast":
        args.batch_size = 64
        args.skip_frames = 2
        print("[⚡] BEAST PRESET: Balanced speed and accuracy")
    elif args.preset == "turbo":
        args.batch_size = 32
        args.skip_frames = 1
        print("[🚀] TURBO PRESET: Conservative but fast")
    
    # System info
    cpu_count = mp.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f"[💻] System: {cpu_count} CPU cores, {ram_gb:.1f}GB RAM")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[🎮] GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        print(f"[⚙️] Settings: batch={args.batch_size}, skip={args.skip_frames}, parallel={args.parallel_videos}")
    
    # Output file
    output_file = Path(args.out_file) if args.out_file else Path(args.output_dir) / "clips.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    start_time = time.time()
    
    if args.input:
        # Single video
        processor = RTX4070BeastProcessor(args.model, args.device, args.conf)
        clips = process_video_beast(
            Path(args.input), processor, args.batch_size, args.skip_frames,
            args.max_frames, [], [1]
        )
        clips = _filter_and_merge_clips(clips, args.min_duration, args.merge_gap)
    else:
        # Multiple videos
        clips = process_many_beast(
            Path(args.input_dir), args.model, args.conf, args.device,
            args.patterns, args.batch_size, args.skip_frames, args.max_frames,
            args.parallel_videos
        )
    
    total_time = time.time() - start_time
    
    # Save results
    if clips:
        output_file.write_text(json.dumps(clips, indent=2))
        print(f"\n[🏆] SUCCESS!")
        print(f"[📁] Saved {len(clips)} clips → {output_file}")
        print(f"[⏱️] Total time: {total_time:.2f}s")
        
        # Performance summary
        total_duration = sum(c["end_ms"] - c["start_ms"] for c in clips) / 1000
        print(f"[📊] Found {total_duration:.1f}s of single-person content")
    else:
        print("[❌] No clips detected")


if __name__ == "__main__":
    _cli()
