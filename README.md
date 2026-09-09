# AI Inference: CPU vs GPU (Simulated) Demo

**Suggestion 1: Real-Time AI Inference + CPU vs GPU**
*"What happens when we move an AI workload from general-purpose CPU computing to specialized GPU computing?"*

The recommended UI is a browser dashboard (`python server.py`) built with
TypeScript, React, and Tailwind. The original OpenCV window (`python cpu_vs_gpu_demo.py`)
still works if you want the traditional Python demo.

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

py -m venv venv
venv\Scripts\activate        # Windows PowerShell

pip install -r requirements.txt
```

On Windows, use `py` if the `python` command opens the Microsoft Store instead.
The commands below use the virtual environment's Python directly, so activation
is optional after the environment has been created.

### Add a sample video

This repo does not bundle a video file. Either:

1. Drop any short `.mp4` clip into the project root and name it
   `sample_video.mp4` (a street/traffic/office clip with people, cars,
   or everyday objects works best for visible YOLO detections), **or**
2. Point the script at your own file with `--source`, **or**
3. Use your webcam with `--source 0`.

## Run the web dashboard (recommended)

The web dashboard runs through the Python backend and opens in your browser.
Run these commands from the project folder:

```bash
cd <your-repo-folder>
venv\Scripts\python.exe -m pip install -r requirements.txt

cd web
npm install
npm run build
cd ..

venv\Scripts\python.exe server.py
```

Open http://127.0.0.1:8000

Keep the terminal running while using the dashboard. Press `Ctrl+C` in that
terminal to stop it. The first run may download the YOLO model automatically.

The dashboard can use the webcam, a video uploaded in the browser, or
`sample_video.mp4` in the project folder. A sample video is not included, so
add one yourself if you want to use the Sample button.

During UI development, run the API and Vite together:

```bash
python server.py
# in another terminal
cd web && npm install && npm run dev
```

Then open http://127.0.0.1:5173 (Vite proxies `/api` and `/ws` to the Python server).

## Run the standalone Python demos

Yes. The standalone OpenCV demos are still available and do not require the
web dashboard. Activate the environment or use its Python executable directly.

### Simulated GPU demo

```bash
cd <your-repo-folder>

# Uses sample_video.mp4 in the project root by default
venv\Scripts\python.exe cpu_vs_gpu_demo.py

# Use a specific video file
venv\Scripts\python.exe cpu_vs_gpu_demo.py --source path/to/video.mp4

# Use a webcam (index 0)
venv\Scripts\python.exe cpu_vs_gpu_demo.py --source 0

# Optional: adjust confidence threshold, model, or disable video looping
venv\Scripts\python.exe cpu_vs_gpu_demo.py --conf 0.4 --model yolov8n.pt --no-loop
```

This version always runs detection on the CPU and displays simulated GPU
timing, so it works without a compatible GPU.

### Real GPU demo

Install the additional real-GPU requirements first:

```bash
venv\Scripts\python.exe -m pip install -r requirements_real_gpu.txt
venv\Scripts\python.exe cpu_vs_real_gpu_demo.py --source 0
```

This version uses NVIDIA CUDA or Apple MPS when available. On unsupported
hardware, it falls back to CPU and reports that no compatible GPU was found.

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
├── server.py                  # FastAPI backend for the web dashboard
├── web/                       # TypeScript + React + Tailwind UI
├── cpu_vs_gpu_demo.py         # original OpenCV window demo
├── requirements.txt
├── README.md
├── sample_video.mp4           # add your own (not included)
└── performance_comparison.png # generated after you quit the OpenCV demo
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
