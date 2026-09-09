#!/usr/bin/env python3
"""
AI Inference: CPU vs REAL GPU Demo
====================================
Suggestion 1 (Real Hardware Version): Real-Time AI Inference + CPU vs GPU

This is the "no simulation" counterpart to cpu_vs_gpu_demo.py. Instead of
estimating GPU performance, this script actually runs YOLOv8 inference on
your laptop's real GPU (NVIDIA CUDA, or Apple Silicon via MPS) if one is
available, and on CPU otherwise - then times both for real.

IMPORTANT: This requires actual GPU hardware + the correct PyTorch build
to show a difference. If your laptop only has an integrated (non-NVIDIA,
non-Apple-Silicon) GPU, PyTorch/CUDA won't be able to use it, and this
script will report "no compatible GPU detected" and run CPU-only. For a
guaranteed-to-work presentation demo regardless of hardware, use
cpu_vs_gpu_demo.py (the simulated-GPU version) instead.

Controls:
    [c]  CPU MODE  - force inference on CPU
    [g]  GPU MODE  - force inference on GPU (if available)
    [q]/[ESC]  Quit and generate performance_comparison_real.png
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is not installed.")
    print("        Run: pip install -r requirements_real_gpu.txt")
    print("        For NVIDIA GPU support, install the CUDA build of PyTorch from:")
    print("        https://pytorch.org/get-started/locally/")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] The 'ultralytics' package is not installed.")
    print("        Run: pip install -r requirements_real_gpu.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "yolov8s.pt"
DEFAULT_VIDEO = "sample_video.mp4"
WINDOW_NAME = "CPU vs REAL GPU Demo"
METRICS_WINDOW = 30
CHART_PATH = "performance_comparison_real.png"
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu_device():
    """Returns (device_string, human_readable_name) for the best available GPU,
    or (None, None) if no compatible GPU is found."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda:0", f"NVIDIA CUDA - {name}"
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon (Metal/MPS)"
    return None, None


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame, mode, gpu_available, gpu_name, inference_ms, fps, num_objects):
    h, w = frame.shape[:2]
    panel_h = 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    mode_color = (80, 170, 255) if mode == "CPU" else (60, 220, 130)
    label = f"{mode} MODE (REAL)"
    if mode == "GPU" and not gpu_available:
        label = "GPU MODE - NOT AVAILABLE (running on CPU)"
        mode_color = (60, 60, 220)

    cv2.putText(frame, label, (16, 32), FONT, 0.75, mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {inference_ms:6.1f} ms", (16, 62),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (16, 88),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects detected: {num_objects}", (260, 62),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    controls = "[c] CPU   [g] GPU (real)   [q] Quit"
    cv2.putText(frame, controls, (260, 88), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    gpu_status = f"GPU detected: {gpu_name}" if gpu_available else "GPU detected: none (CPU only)"
    cv2.putText(frame, gpu_status, (16, 108), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------

def open_source(source_arg):
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time YOLO inference demo: CPU vs actual GPU hardware."
    )
    parser.add_argument("--source", default=DEFAULT_VIDEO)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()

    gpu_device, gpu_name = detect_gpu_device()
    gpu_available = gpu_device is not None

    print("=== GPU Detection ===")
    if gpu_available:
        print(f"Found compatible GPU: {gpu_name}")
    else:
        print("No compatible GPU found (no NVIDIA CUDA / Apple MPS device detected).")
        print("GPU mode will run on CPU and timing will match CPU mode -")
        print("this is expected behavior, not a bug: there is no simulation here.")
        print("Use cpu_vs_gpu_demo.py for a guaranteed CPU-vs-GPU comparison on any machine.")
    print()

    print("Loading YOLO model (first run downloads weights automatically)...")
    model = YOLO(args.model)

    cap, is_webcam = open_source(args.source)
    loop_video = (not is_webcam) and (not args.no_loop)

    mode = "CPU"
    cpu_fps_hist = deque(maxlen=METRICS_WINDOW)
    gpu_fps_hist = deque(maxlen=METRICS_WINDOW)
    all_cpu_fps, all_gpu_fps = [], []
    all_cpu_ms, all_gpu_ms = [], []

    print("=== AI Inference: CPU vs REAL GPU Demo ===")
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

        active_device = gpu_device if (mode == "GPU" and gpu_available) else "cpu"

        # Sync before/after timing so GPU async execution is measured correctly
        if active_device == "cuda:0":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.predict(frame, conf=args.conf, device=active_device, verbose=False)
        if active_device == "cuda:0":
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - t0

        annotated = results[0].plot()
        num_objects = len(results[0].boxes)

        infer_ms = infer_s * 1000.0
        fps = 1.0 / infer_s if infer_s > 0 else 0.0

        if active_device == "cpu":
            cpu_fps_hist.append(fps)
            all_cpu_fps.append(fps)
            all_cpu_ms.append(infer_ms)
            shown_fps = sum(cpu_fps_hist) / len(cpu_fps_hist)
        else:
            gpu_fps_hist.append(fps)
            all_gpu_fps.append(fps)
            all_gpu_ms.append(infer_ms)
            shown_fps = sum(gpu_fps_hist) / len(gpu_fps_hist)

        annotated = draw_overlay(annotated, mode, gpu_available, gpu_name,
                                  infer_ms, shown_fps, num_objects)

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

    generate_chart(all_cpu_fps, all_gpu_fps, all_cpu_ms, all_gpu_ms, gpu_available, gpu_name)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def generate_chart(cpu_fps, gpu_fps, cpu_ms, gpu_ms, gpu_available, gpu_name):
    if not cpu_fps and not gpu_fps:
        print("No metrics were collected - skipping chart.")
        return

    avg_cpu_fps = sum(cpu_fps) / len(cpu_fps) if cpu_fps else 0
    avg_gpu_fps = sum(gpu_fps) / len(gpu_fps) if gpu_fps else 0
    avg_cpu_ms = sum(cpu_ms) / len(cpu_ms) if cpu_ms else 0
    avg_gpu_ms = sum(gpu_ms) / len(gpu_ms) if gpu_ms else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    title_suffix = f" ({gpu_name})" if gpu_available else " (no GPU detected)"
    fig.suptitle(f"Real-Time AI Inference: CPU vs GPU{title_suffix}",
                 fontsize=13, fontweight="bold")

    labels = ["CPU MODE", "GPU MODE"]
    colors = ["#5AAAFF", "#3CDC82"]

    ax = axes[0]
    bars = ax.bar(labels, [avg_cpu_fps, avg_gpu_fps], color=colors)
    ax.set_title("Average FPS")
    ax.set_ylabel("Frames per second")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.1f}", ha="center", va="bottom", fontweight="bold")

    ax = axes[1]
    bars = ax.bar(labels, [avg_cpu_ms, avg_gpu_ms], color=colors)
    ax.set_title("Average Inference Time")
    ax.set_ylabel("Milliseconds per frame")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.1f} ms", ha="center", va="bottom", fontweight="bold")

    if avg_cpu_fps and avg_gpu_fps and gpu_available:
        speedup = avg_gpu_fps / avg_cpu_fps
        fig.text(0.5, 0.02,
                  f"Measured real GPU speedup vs CPU: ~{speedup:.1f}x faster "
                  "(actual hardware measurement, not simulated)",
                  ha="center", fontsize=9, style="italic", color="#555555")
    elif not gpu_available:
        fig.text(0.5, 0.02,
                  "No compatible GPU was detected on this machine - "
                  "GPU mode ran on CPU, so no speedup is shown.",
                  ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(CHART_PATH, dpi=150)
    print(f"\nSaved performance comparison chart -> {CHART_PATH}")


if __name__ == "__main__":
    main()
