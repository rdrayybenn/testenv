#!/usr/bin/env python3
"""
AI Inference: CPU vs GPU (Simulated) Demo
==========================================
Suggestion 1: Real-Time AI Inference + CPU vs GPU
"What happens when we move an AI workload from general-purpose CPU
computing to specialized GPU computing?"

This script runs YOLOv8 object detection on a video file (or webcam)
and lets you toggle live between:

    [c]  CPU MODE  - real inference, timed on your machine's CPU
    [g]  GPU MODE  - the SAME detections, but with realistic simulated
                      CUDA/Tensor-core acceleration numbers overlaid,
                      so no NVIDIA GPU is required to run the demo

On exit ([q] or ESC), a bar chart (performance_comparison.png) is
generated comparing average FPS / inference time between the two modes.


"""

import argparse
import os
import random
import sys
import time
from collections import deque

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend, chart is saved not shown
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] The 'ultralytics' package is not installed.")
    print("        Run:  pip install -r requirements.txt")
    sys.exit(1)



MODEL_NAME = "yolov8s.pt"           # lightweight YOLOv8 "small" model
DEFAULT_VIDEO = "sample_video.mp4"  # bundled/provided sample video
WINDOW_NAME = "AI Inference: CPU vs GPU Demo"

# Realistic published-benchmark speedup range for YOLOv8n CPU -> GPU
# (CUDA / Tensor Core) inference. Actual numbers vary by hardware
# generation (e.g. NVIDIA T4/RTX vs Huawei Ascend/Atlas NPUs); tune to taste.
GPU_SPEEDUP_MIN = 4.5
GPU_SPEEDUP_MAX = 7.5
GPU_JITTER = 0.06  # +/- 6% frame-to-frame jitter so numbers look organic

METRICS_WINDOW = 45      # rolling average window, in frames
CHART_PATH = "performance_comparison.png"

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, mode, inference_ms, fps, num_objects, simulated):
    """Draws the metrics HUD onto the frame in-place and returns it."""
    h, w = frame.shape[:2]
    panel_h = 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    mode_color = (80, 170, 255) if mode == "CPU" else (60, 220, 130)
    mode_label = f"{mode} MODE" + ("  (CUDA Simulated)" if simulated else "")

    cv2.putText(frame, mode_label, (16, 32), FONT, 0.8, mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {inference_ms:6.1f} ms", (16, 62),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (16, 88),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects detected: {num_objects}", (260, 62),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    controls = "[c] CPU   [g] GPU (sim)   [q] Quit"
    cv2.putText(frame, controls, (260, 88), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def open_source(source_arg):
    """Opens a webcam index (int-like string) or a video file path."""
    is_webcam = str(source_arg).isdigit()
    cap_source = int(source_arg) if is_webcam else source_arg

    if not is_webcam and not os.path.exists(source_arg):
        print(f"[ERROR] Video source '{source_arg}' was not found.")
        print("        Place a video named 'sample_video.mp4' in this folder,")
        print("        pass --source <path-to-video>, or pass --source 0 for webcam.")
        sys.exit(1)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_arg}")
        sys.exit(1)

    return cap, is_webcam


def main():
    parser = argparse.ArgumentParser(
        description="Real-time YOLO inference demo: CPU vs GPU."
    )
    parser.add_argument("--source", default=DEFAULT_VIDEO,
                         help="Path to a video file, or a camera index (e.g. 0). "
                              f"Default: {DEFAULT_VIDEO}")
    parser.add_argument("--model", default=MODEL_NAME,
                         help=f"Ultralytics YOLO model to load. Default: {MODEL_NAME}")
    parser.add_argument("--conf", type=float, default=0.35,
                         help="Detection confidence threshold. Default: 0.35")
    parser.add_argument("--no-loop", action="store_true",
                         help="Don't loop the video when it reaches the end.")
    args = parser.parse_args()

    print("Loading YOLO model (first run downloads weights automatically)...")
    model = YOLO(args.model)

    cap, is_webcam = open_source(args.source)
    loop_video = (not is_webcam) and (not args.no_loop)

    mode = "CPU"          # "CPU" or "GPU" (GPU is always simulated here)
    cpu_fps_hist = deque(maxlen=METRICS_WINDOW)
    gpu_fps_hist = deque(maxlen=METRICS_WINDOW)

    # Full-run metrics kept for the final comparison chart
    all_cpu_fps, all_gpu_fps = [], []
    all_cpu_ms, all_gpu_ms = [], []

    print("\n=== AI Inference: CPU vs GPU Demo ===")
    print("Controls:  [c] CPU mode   [g] GPU mode   [q]/[ESC] quit\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # --- Real inference always runs on CPU under the hood; "GPU mode"
        # --- reuses the same detections but overlays simulated CUDA timing.
        t0 = time.perf_counter()
        results = model.predict(frame, conf=args.conf, device="cpu", verbose=False)
        cpu_infer_s = time.perf_counter() - t0

        annotated = results[0].plot()
        num_objects = len(results[0].boxes)

        cpu_ms = cpu_infer_s * 1000.0
        cpu_fps = 1.0 / cpu_infer_s if cpu_infer_s > 0 else 0.0

        speedup = random.uniform(GPU_SPEEDUP_MIN, GPU_SPEEDUP_MAX)
        jitter = 1.0 + random.uniform(-GPU_JITTER, GPU_JITTER)
        gpu_infer_s = max(cpu_infer_s / speedup * jitter, 1e-4)
        gpu_ms = gpu_infer_s * 1000.0
        gpu_fps = 1.0 / gpu_infer_s

        if mode == "CPU":
            cpu_fps_hist.append(cpu_fps)
            all_cpu_fps.append(cpu_fps)
            all_cpu_ms.append(cpu_ms)
            shown_ms, shown_fps = cpu_ms, sum(cpu_fps_hist) / len(cpu_fps_hist)
        else:
            gpu_fps_hist.append(gpu_fps)
            all_gpu_fps.append(gpu_fps)
            all_gpu_ms.append(gpu_ms)
            shown_ms, shown_fps = gpu_ms, sum(gpu_fps_hist) / len(gpu_fps_hist)

        annotated = draw_overlay(annotated, mode, shown_ms, shown_fps,
                                  num_objects, simulated=(mode == "GPU"))

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            mode = "CPU"
        elif key == ord("g"):
            mode = "GPU"

    cap.release()
    cv2.destroyAllWindows()

    generate_chart(all_cpu_fps, all_gpu_fps, all_cpu_ms, all_gpu_ms)


# ---------------------------------------------------------------------------
# Performance comparison chart
# ---------------------------------------------------------------------------

def generate_chart(cpu_fps, gpu_fps, cpu_ms, gpu_ms):
    if not cpu_fps and not gpu_fps:
        print("No metrics were collected (demo closed too quickly) - skipping chart.")
        return

    avg_cpu_fps = sum(cpu_fps) / len(cpu_fps) if cpu_fps else 0
    avg_gpu_fps = sum(gpu_fps) / len(gpu_fps) if gpu_fps else 0
    avg_cpu_ms = sum(cpu_ms) / len(cpu_ms) if cpu_ms else 0
    avg_gpu_ms = sum(gpu_ms) / len(gpu_ms) if gpu_ms else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Real-Time AI Inference: CPU vs GPU", fontsize=14, fontweight="bold")

    labels = ["CPU MODE", "GPU MODE"]
    colors = ["#5AAAFF", "#3CDC82"]

    # --- FPS subplot
    ax = axes[0]
    bars = ax.bar(labels, [avg_cpu_fps, avg_gpu_fps], color=colors)
    ax.set_title("Average FPS")
    ax.set_ylabel("Frames per second")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.1f}", ha="center", va="bottom", fontweight="bold")

    # --- Inference time subplot
    ax = axes[1]
    bars = ax.bar(labels, [avg_cpu_ms, avg_gpu_ms], color=colors)
    ax.set_title("Average Inference Time")
    ax.set_ylabel("Milliseconds per frame")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.1f} ms", ha="center", va="bottom", fontweight="bold")

    if avg_cpu_fps and avg_gpu_fps:
        speedup = avg_gpu_fps / avg_cpu_fps
        fig.text(0.5, 0.02,
                  f"GPU speedup vs CPU",
                  ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(CHART_PATH, dpi=150)
    print(f"\nSaved performance comparison chart -> {CHART_PATH}")


if __name__ == "__main__":
    main()