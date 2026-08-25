# AI Inference: CPU vs GPU (Simulated) Demo

**Suggestion 1: Real-Time AI Inference + CPU vs GPU**
*"What happens when we move an AI workload from general-purpose CPU computing to specialized GPU computing?"*

A live, presentation-ready demo built around this flow:

```
INPUT -> Webcam/Video -> YOLO Model -> [ CPU MODE | GPU MODE ]
      -> Metrics -> Performance Comparison -> AI Computing Concept
      -> (NVIDIA / Huawei Ascend / Atlas)
```

It runs real-time YOLOv8 object detection on a video (or webcam feed) and
lets you toggle live between **CPU mode** and **GPU mode**, so an audience
can see - side by side - how specialized accelerators change AI inference
performance.

## Why "simulated" GPU?

To make the demo **fail-proof for live presentations**, it does **not**
require an NVIDIA GPU. Detection always runs on CPU under the hood. In
**GPU mode**, the script overlays realistic simulated CUDA/Tensor Core
timing, calculated by scaling the real, measured CPU inference time down
by a randomized speedup factor (4.5x-7.5x, with small frame-to-frame
jitter) based on published YOLOv8s CPU-vs-GPU benchmark ranges. The
on-screen HUD is labeled `(CUDA Simulated)` and a small footnote is shown
whenever GPU mode is active, so the simulation is transparent rather than
misleading — this is meant to *illustrate the concept and expected
magnitude* of GPU acceleration, not to report a real benchmark.

The bounding boxes and detection quality are identical in both modes;
only the displayed timing/FPS numbers differ.

## Features

- **Input:** captures a video file (or webcam) using OpenCV.
- **YOLO Model:** loads a lightweight YOLOv8s object detection model
  (auto-downloaded on first run via `ultralytics`).
- **CPU vs GPU toggle:** switch live with the keyboard - `c` for CPU,
  `g` for (simulated) GPU.
- **Metrics overlay:** inference time (ms) and FPS drawn directly on the
  video feed, plus a live object count.
- **Performance comparison:** on quit, generates
  `performance_comparison.png` - a bar chart comparing average FPS and
  average inference time across both modes, with the effective simulated
  speedup called out.

## Requirements

- Python 3.9+
- A webcam **or** a sample video file (no GPU required)

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Add a sample video

This repo does not bundle a video file. Either:

1. Drop any short `.mp4` clip into the project root and name it
   `sample_video.mp4` (a street/traffic/office clip with people, cars,
   or everyday objects works best for visible YOLO detections), **or**
2. Point the script at your own file with `--source`, **or**
3. Use your webcam with `--source 0`.

## Usage

```bash
# Uses sample_video.mp4 in the project root by default
python cpu_vs_gpu_demo.py

# Use a specific video file
python cpu_vs_gpu_demo.py --source path/to/video.mp4

# Use a webcam (index 0)
python cpu_vs_gpu_demo.py --source 0

# Optional: adjust confidence threshold, model, or disable video looping
python cpu_vs_gpu_demo.py --conf 0.4 --model yolov8n.pt --no-loop
```

## Controls

| Key       | Action                          |
|-----------|----------------------------------|
| `c`       | Switch to CPU mode                |
| `g`       | Switch to GPU mode (simulated)    |
| `q` / Esc | Quit and generate comparison chart|

## Output

When you quit the demo, `performance_comparison.png` is saved in the
project root, showing:

- Average FPS: CPU mode vs GPU mode (simulated)
- Average inference time (ms): CPU mode vs GPU mode (simulated)
- The effective simulated speedup factor

## Project structure

```
.
├── cpu_vs_gpu_demo.py         # main demo script
├── requirements.txt
├── README.md
├── sample_video.mp4           # add your own (not included)
└── performance_comparison.png # generated after you quit the demo
```

## Notes for presenters

- The model weights (`yolov8s.pt`) download automatically the first time
  you run the script - do this once **before** your live demo so it isn't
  waiting on a download in front of an audience.
- Loop mode (default, for video files) restarts the clip automatically
  when it ends, so the demo can run indefinitely.
- If you want a swap-in visual for "specialized AI hardware" (as in the
  flowchart's NVIDIA / Huawei Ascend / Atlas reference), that's a good
  place to add a slide or talking point right after the on-screen
  performance comparison.

## Troubleshooting

- **Webcam won't open / permission denied:** check OS camera permissions
  for your terminal/IDE, or try a different camera index (`--source 1`).
- **Model download fails:** ensure you have internet access the first
  time you run the script; after that, weights are cached locally.
- **Low FPS on CPU:** try the smaller `yolov8n.pt` model with
  `--model yolov8n.pt`, or reduce input video resolution.
