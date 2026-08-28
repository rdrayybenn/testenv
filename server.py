#!/usr/bin/env python3
"""
Web dashboard backend for the CPU vs GPU (simulated) YOLO demo.

Serves the Vite-built UI from web/dist when present, and streams
annotated frames + metrics over a WebSocket.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "[ERROR] The 'ultralytics' package is not installed.\n"
        "        Run:  pip install -r requirements.txt"
    )


ROOT = Path(__file__).resolve().parent
WEB_DIST = ROOT / "web" / "dist"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8s.pt"
DEFAULT_VIDEO = "sample_video.mp4"
GPU_SPEEDUP_MIN = 4.5
GPU_SPEEDUP_MAX = 7.5
GPU_JITTER = 0.06
METRICS_WINDOW = 45
STREAM_MAX_WIDTH = 960


class DemoEngine:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.model_ready = False
        self.model_error: Optional[str] = None

        self.source = str(ROOT / DEFAULT_VIDEO)
        self.conf = 0.35
        self.loop_video = True
        self.mode = "CPU"

        self.running = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

        self.frame_id = 0
        self.jpeg_b64: Optional[str] = None
        self.num_objects = 0
        self.shown_ms = 0.0
        self.shown_fps = 0.0

        self.cpu_fps_hist: deque[float] = deque(maxlen=METRICS_WINDOW)
        self.gpu_fps_hist: deque[float] = deque(maxlen=METRICS_WINDOW)
        self.all_cpu_fps: list[float] = []
        self.all_gpu_fps: list[float] = []
        self.all_cpu_ms: list[float] = []
        self.all_gpu_ms: list[float] = []

        self.cpu_spark: deque[float] = deque(maxlen=40)
        self.gpu_spark: deque[float] = deque(maxlen=40)

    def load_model(self) -> None:
        try:
            self.model = YOLO(MODEL_NAME)
            self.model_ready = True
        except Exception as exc:  # noqa: BLE001
            self.model_error = str(exc)
            self.model_ready = False

    def snapshot(self) -> dict:
        with self._lock:
            cpu_avg_fps = _avg(self.all_cpu_fps)
            gpu_avg_fps = _avg(self.all_gpu_fps)
            cpu_avg_ms = _avg(self.all_cpu_ms)
            gpu_avg_ms = _avg(self.all_gpu_ms)
            speedup = (gpu_avg_fps / cpu_avg_fps) if cpu_avg_fps and gpu_avg_fps else 0.0
            return {
                "running": self.running,
                "mode": self.mode,
                "simulated": self.mode == "GPU",
                "frameId": self.frame_id,
                "jpeg": self.jpeg_b64,
                "objects": self.num_objects,
                "inferenceMs": round(self.shown_ms, 1),
                "fps": round(self.shown_fps, 1),
                "cpuSpark": list(self.cpu_spark),
                "gpuSpark": list(self.gpu_spark),
                "averages": {
                    "cpuFps": round(cpu_avg_fps, 1),
                    "gpuFps": round(gpu_avg_fps, 1),
                    "cpuMs": round(cpu_avg_ms, 1),
                    "gpuMs": round(gpu_avg_ms, 1),
                    "speedup": round(speedup, 2),
                    "cpuFrames": len(self.all_cpu_fps),
                    "gpuFrames": len(self.all_gpu_fps),
                },
            }

    def set_mode(self, mode: str) -> None:
        if mode not in ("CPU", "GPU"):
            return
        with self._lock:
            self.mode = mode

    def set_source(self, source: str, loop: bool = True) -> None:
        self.stop()
        with self._lock:
            self.source = source
            self.loop_video = loop
            self._reset_metrics()

    def reset_metrics(self) -> None:
        with self._lock:
            self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.cpu_fps_hist.clear()
        self.gpu_fps_hist.clear()
        self.all_cpu_fps.clear()
        self.all_gpu_fps.clear()
        self.all_cpu_ms.clear()
        self.all_gpu_ms.clear()
        self.cpu_spark.clear()
        self.gpu_spark.clear()

    def start(self) -> dict:
        if not self.model_ready or self.model is None:
            return {"ok": False, "error": self.model_error or "Model is still loading."}
        source = self.source
        is_webcam = str(source).isdigit()
        if not is_webcam and not os.path.exists(source):
            return {
                "ok": False,
                "error": f"Video source '{source}' was not found. Upload a clip or add sample_video.mp4.",
            }
        if self.running:
            return {"ok": True, "already": True}
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"ok": True}

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        with self._lock:
            self.running = False
            self.jpeg_b64 = None
            self.num_objects = 0
            self.shown_ms = 0.0
            self.shown_fps = 0.0

    def _loop(self) -> None:
        is_webcam = str(self.source).isdigit()
        cap_source = int(self.source) if is_webcam else self.source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            self.running = False
            return

        self.running = True
        loop_video = (not is_webcam) and self.loop_video

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            t0 = time.perf_counter()
            results = self.model.predict(
                frame, conf=self.conf, device="cpu", verbose=False
            )
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

            jpeg_b64 = _encode_frame(annotated)

            if self._stop.is_set():
                break

            with self._lock:
                mode = self.mode
                if mode == "CPU":
                    self.cpu_fps_hist.append(cpu_fps)
                    self.all_cpu_fps.append(cpu_fps)
                    self.all_cpu_ms.append(cpu_ms)
                    self.cpu_spark.append(cpu_fps)
                    shown_ms = cpu_ms
                    shown_fps = sum(self.cpu_fps_hist) / len(self.cpu_fps_hist)
                else:
                    self.gpu_fps_hist.append(gpu_fps)
                    self.all_gpu_fps.append(gpu_fps)
                    self.all_gpu_ms.append(gpu_ms)
                    self.gpu_spark.append(gpu_fps)
                    shown_ms = gpu_ms
                    shown_fps = sum(self.gpu_fps_hist) / len(self.gpu_fps_hist)

                self.frame_id += 1
                self.jpeg_b64 = jpeg_b64
                self.num_objects = int(num_objects)
                self.shown_ms = shown_ms
                self.shown_fps = shown_fps

            self._new_frame.set()
            self._new_frame.clear()

        cap.release()
        self.running = False


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _encode_frame(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > STREAM_MAX_WIDTH:
        scale = STREAM_MAX_WIDTH / w
        frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


class ModeBody(BaseModel):
    mode: str


class SourceBody(BaseModel):
    kind: str = "sample"
    index: str = "0"
    path: Optional[str] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=engine.load_model, daemon=True).start()
    yield
    engine.stop()


engine = DemoEngine()
app = FastAPI(title="CPU vs GPU Inference Lab", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status() -> dict:
    sample = ROOT / DEFAULT_VIDEO
    return {
        "modelReady": engine.model_ready,
        "modelError": engine.model_error,
        "running": engine.running,
        "mode": engine.mode,
        "source": engine.source,
        "hasSampleVideo": sample.exists(),
        "sampleName": DEFAULT_VIDEO,
    }


@app.post("/api/start")
def start() -> dict:
    return engine.start()


@app.post("/api/stop")
def stop() -> dict:
    engine.stop()
    return {"ok": True}


@app.post("/api/mode")
def set_mode(payload: ModeBody) -> dict:
    engine.set_mode(payload.mode.upper())
    return {"ok": True, "mode": engine.mode}


@app.post("/api/reset")
def reset() -> dict:
    engine.reset_metrics()
    return {"ok": True}


@app.post("/api/source")
def set_source(payload: SourceBody) -> dict:
    kind = payload.kind
    if kind == "webcam":
        engine.set_source(str(payload.index), loop=False)
    elif kind == "sample":
        engine.set_source(str(ROOT / DEFAULT_VIDEO), loop=True)
    else:
        if not payload.path:
            return JSONResponse({"ok": False, "error": "Missing path"}, status_code=400)
        engine.set_source(str(payload.path), loop=True)
    return {"ok": True, "source": engine.source}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"source{suffix}"
    contents = await file.read()
    dest.write_bytes(contents)
    engine.set_source(str(dest), loop=True)
    return {"ok": True, "source": str(dest), "name": file.filename}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    last_id = -1
    try:
        while True:
            snap = engine.snapshot()
            if snap["frameId"] != last_id and snap["jpeg"]:
                last_id = snap["frameId"]
                await ws.send_json({"type": "frame", **snap})
            else:
                await ws.send_json({"type": "status", **{k: v for k, v in snap.items() if k != "jpeg"}})
            await asyncio.sleep(0.04)
    except WebSocketDisconnect:
        return


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = WEB_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU vs GPU inference web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
