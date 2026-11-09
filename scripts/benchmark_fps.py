#!/usr/bin/env python3
"""
Measures how fast YOLO26n runs on this machine (meant for the Orange Pi 5 Plus).

What it measures: the complete detection step for each frame, from reading
the image off disk to having the swimmer boxes (load + preprocess + model +
postprocess). That's the number that matters for the boat.

Usage (from the repo root):
    python3 scripts/benchmark_fps.py                    # 3-minute run on dataset/images
    python3 scripts/benchmark_fps.py --minutes 5        # longer run (checks heat throttling)
    python3 scripts/benchmark_fps.py --imgsz 320        # smaller input = faster, less accurate
    python3 scripts/benchmark_fps.py --model yolo26n_rknn_model   # NPU model, if exported
    python3 scripts/benchmark_fps.py --video some_clip.mp4        # use a video instead

It prints the average FPS for every 30-second window, so you can see whether
it slows down as the board heats up, plus the CPU temperature when the board
reports it. A summary is saved to dataset/eval_results/fps_<model>_<imgsz>.csv.
"""

import argparse
import csv
import glob
import os
import platform
import time

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def cpu_temp_c():
    """Hottest thermal zone in degrees C, or None if the board doesn't report it."""
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path) as f:
                temps.append(int(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def frame_source(args):
    """Yields frames forever: loops over the images (or the video) until time is up."""
    if args.video:
        while True:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"Can't open video: {args.video}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()
    else:
        files = sorted(
            os.path.join(args.images, f) for f in os.listdir(args.images)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not files:
            raise SystemExit(f"No images found in {args.images}")
        while True:
            for path in files:
                # Reading from disk is part of the timed step on purpose
                yield cv2.imread(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(BASE_DIR, "yolo26n.pt"))
    p.add_argument("--images", default=os.path.join(BASE_DIR, "dataset", "images"))
    p.add_argument("--video", default=None, help="use a video file instead of the images")
    p.add_argument("--imgsz", type=int, default=640, help="model input size in pixels (default 640)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--minutes", type=float, default=3.0)
    p.add_argument("--window", type=float, default=30.0, help="seconds per progress line")
    p.add_argument("--warmup", type=int, default=10, help="frames to skip before timing")
    args = p.parse_args()

    model = YOLO(args.model)
    frames = frame_source(args)

    # Warm-up: the first few frames are slow while things load, so don't time them
    for _ in range(args.warmup):
        model.predict(next(frames), imgsz=args.imgsz, conf=args.conf, classes=[0], verbose=False)

    print("=" * 60)
    print(f"  FPS BENCHMARK  model={os.path.basename(args.model)}  imgsz={args.imgsz}")
    print(f"  machine={platform.node()} ({platform.machine()})  run={args.minutes} min")
    print("=" * 60)

    rows = []
    run_end = time.perf_counter() + args.minutes * 60
    total_frames = 0
    total_time = 0.0

    while time.perf_counter() < run_end:
        win_frames = 0
        win_start = time.perf_counter()
        while time.perf_counter() - win_start < args.window and time.perf_counter() < run_end:
            t0 = time.perf_counter()
            frame = next(frames)  # includes reading the image
            model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[0], verbose=False)
            total_time += time.perf_counter() - t0
            win_frames += 1
        elapsed = time.perf_counter() - win_start
        fps = win_frames / elapsed if elapsed > 0 else 0.0
        temp = cpu_temp_c()
        total_frames += win_frames
        minute = len(rows) * args.window / 60
        temp_txt = f"{temp:.0f} C" if temp is not None else "n/a"
        print(f"  t={minute:5.1f} min   {fps:6.2f} FPS   temp {temp_txt}")
        rows.append({"minute": round(minute, 2), "fps": round(fps, 2),
                     "temp_c": round(temp, 1) if temp is not None else ""})

    avg_fps = total_frames / total_time if total_time > 0 else 0.0
    first, last = rows[0]["fps"], rows[-1]["fps"]
    drop = (first - last) / first * 100 if first else 0.0

    print("-" * 60)
    print(f"  Average FPS (full detection step): {avg_fps:.2f}")
    print(f"  Average time per frame:            {1000 / avg_fps:.0f} ms" if avg_fps else "")
    print(f"  First vs last window:              {first:.2f} -> {last:.2f} FPS ({drop:+.0f}% drop)")
    print(f"  Input size: {args.imgsz}x{args.imgsz}   Frames timed: {total_frames}")
    print("  Device: record whether this used the CPU or the NPU (rknn model = NPU)")

    out_dir = os.path.join(BASE_DIR, "dataset", "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.model.rstrip("/")))[0]
    out_csv = os.path.join(out_dir, f"fps_{name}_{args.imgsz}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["minute", "fps", "temp_c"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved per-window results to {out_csv}")


if __name__ == "__main__":
    main()
