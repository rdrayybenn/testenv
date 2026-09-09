import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  Cpu,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  ScanEye,
  Upload,
  Video,
  Webcam,
  Zap,
} from "lucide-react";
import type { FramePayload, Mode, ServerStatus } from "./types";

const emptyAverages = {
  cpuFps: 0,
  gpuFps: 0,
  cpuMs: 0,
  gpuMs: 0,
  speedup: 0,
  cpuFrames: 0,
  gpuFrames: 0,
};

function wsUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

export default function App() {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [frame, setFrame] = useState<FramePayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [feedEnabled, setFeedEnabled] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const mode: Mode = frame?.mode ?? status?.mode ?? "CPU";
  const running = frame?.running ?? status?.running ?? false;
  const averages = frame?.averages ?? emptyAverages;

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch("/api/status");
        const data = (await res.json()) as ServerStatus;
        if (!cancelled) {
          setStatus(data);
          setCameraOn(data.running && /^\d+$/.test(data.source));
        }
      } catch {
        if (!cancelled) setError("Cannot reach the Python backend on port 8000.");
      }
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const socket = new WebSocket(wsUrl());
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as FramePayload;
      setFrame((prev) => {
        if (payload.type === "status" && prev?.jpeg && payload.running) {
          return { ...prev, ...payload, jpeg: prev.jpeg };
        }
        return payload;
      });
    };
    socket.onerror = () => setError("Live stream disconnected.");
    return () => socket.close();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "c" || e.key === "C") void setMode("CPU");
      if (e.key === "g" || e.key === "G") void setMode("GPU");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function setMode(next: Mode) {
    await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: next }),
    });
  }

  async function start() {
    setBusy(true);
    setError(null);
    const res = await fetch("/api/start", { method: "POST" });
    const data = await res.json();
    if (!data.ok) setError(data.error ?? "Could not start inference.");
    else setFeedEnabled(true);
    setBusy(false);
  }

  async function stop() {
    setBusy(true);
    await fetch("/api/stop", { method: "POST" });
    setFeedEnabled(false);
    setBusy(false);
  }

  async function resetMetrics() {
    await fetch("/api/reset", { method: "POST" });
  }

  async function useSample() {
    setCameraOn(false);
    setFeedEnabled(false);
    setFrame(null);
    const res = await fetch("/api/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "sample" }),
    });
    const data = await res.json();
    if (!data.ok) {
      setError(data.error ?? "Could not select the sample video.");
    }
  }

  async function useWebcam() {
    if (cameraOn) {
      await stop();
      setCameraOn(false);
      setFrame(null);
      return;
    }

    setFeedEnabled(false);
    setFrame(null);
    await fetch("/api/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "webcam", index: "0" }),
    });
    setCameraOn(true);
  }

  async function onUpload(file: File) {
    const body = new FormData();
    body.append("file", file);
    setBusy(true);
    setCameraOn(false);
    setFeedEnabled(false);
    setFrame(null);
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json();
    if (!data.ok) setError(data.error ?? "Upload failed.");
    setBusy(false);
  }

  const spark = mode === "CPU" ? (frame?.cpuSpark ?? []) : (frame?.gpuSpark ?? []);
  const ready = Boolean(status?.modelReady);

  return (
    <div className="grid-bg min-h-screen">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-6 px-5 py-6 lg:px-8">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] tracking-[0.28em] text-white/40 uppercase">
              Real-time AI inference
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">
              CPU vs GPU Lab
            </h1>
            <p className="mt-1 max-w-xl text-sm text-white/50">
              Same YOLOv8 detections. Switch the accelerator live and watch
              latency and FPS change — GPU numbers are simulated CUDA/Tensor Core
              timing, so this demo runs without an NVIDIA card.
            </p>
          </div>
          <ModeToggle mode={mode} onChange={setMode} />
        </header>

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_380px]">
          <section
            className={`overflow-hidden rounded-3xl bg-[#0c0f16] ${
              mode === "GPU" ? "glow-gpu" : "glow-cpu"
            }`}
          >
            <div className="relative aspect-video bg-black">
              {feedEnabled && frame?.jpeg ? (
                <img
                  src={`data:image/jpeg;base64,${frame.jpeg}`}
                  alt="Annotated inference feed"
                  className="h-full w-full object-contain"
                />
              ) : (
                <EmptyStage ready={ready} hasSample={Boolean(status?.hasSampleVideo)} />
              )}
              <div className="absolute top-4 left-4 flex items-center gap-2">
                <span
                  className={`rounded-full px-3 py-1 font-mono text-xs font-medium ${
                    mode === "GPU"
                      ? "bg-gpu/15 text-gpu"
                      : "bg-cpu/15 text-cpu"
                  }`}
                >
                  {mode} MODE
                </span>
                {mode === "GPU" && (
                  <span className="rounded-full bg-white/8 px-3 py-1 font-mono text-[11px] text-white/60">
                    CUDA simulated
                  </span>
                )}
                {running && (
                  <span className="flex items-center gap-1.5 rounded-full bg-red-500/15 px-3 py-1 font-mono text-[11px] text-red-300">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-400" />
                    LIVE
                  </span>
                )}
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/6 px-4 py-3">
              <div className="flex flex-wrap gap-2">
                <button
                  disabled={busy || !ready}
                  onClick={() => (running ? stop() : start())}
                  className="inline-flex items-center gap-2 rounded-xl bg-white px-3.5 py-2 text-sm font-medium text-black disabled:opacity-40"
                >
                  {running ? <Pause size={16} /> : <Play size={16} />}
                  {running ? "Pause" : "Start inference"}
                </button>
                <button
                  onClick={() => fileRef.current?.click()}
                  className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3.5 py-2 text-sm text-white/80 hover:bg-white/5"
                >
                  <Upload size={16} />
                  Upload clip
                </button>
                <button
                  onClick={useWebcam}
                  aria-pressed={cameraOn}
                  className={`inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-sm transition-colors ${
                    cameraOn
                      ? "border-gpu/50 bg-gpu/15 text-gpu"
                      : "border-white/10 text-white/80 hover:bg-white/5"
                  }`}
                >
                  <Webcam size={16} />
                  {cameraOn ? "Camera on" : "Camera off"}
                </button>
                {status?.hasSampleVideo && (
                  <button
                    onClick={useSample}
                    className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3.5 py-2 text-sm text-white/80 hover:bg-white/5"
                  >
                    <Video size={16} />
                    Sample
                  </button>
                )}
              </div>
              <p className="font-mono text-[11px] text-white/35">
                Shortcuts · C CPU · G GPU
              </p>
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void onUpload(file);
                }}
              />
            </div>
          </section>

          <aside className="flex flex-col gap-4">
            <div className="grid grid-cols-3 gap-3">
              <Metric
                label="FPS"
                value={frame ? frame.fps.toFixed(1) : "—"}
                icon={<Gauge size={14} />}
                accent={mode === "GPU" ? "gpu" : "cpu"}
              />
              <Metric
                label="Inference"
                value={frame ? `${frame.inferenceMs.toFixed(1)}` : "—"}
                unit="ms"
                icon={<Activity size={14} />}
                accent={mode === "GPU" ? "gpu" : "cpu"}
              />
              <Metric
                label="Objects"
                value={frame ? String(frame.objects) : "—"}
                icon={<ScanEye size={14} />}
                accent="neutral"
              />
            </div>

            <div className="rounded-2xl border border-white/8 bg-panel p-4">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-sm font-medium">Live FPS</p>
                <span className="font-mono text-[11px] text-white/40">
                  rolling window
                </span>
              </div>
              <Sparkline values={spark} color={mode === "GPU" ? "#3cdc82" : "#5aaaff"} />
            </div>

            <div className="rounded-2xl border border-white/8 bg-panel p-4">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Session comparison</p>
                  <p className="text-xs text-white/40">
                    Spend time in both modes to fill the bars
                  </p>
                </div>
                <button
                  onClick={resetMetrics}
                  className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-white/50 hover:bg-white/5 hover:text-white"
                >
                  <RotateCcw size={12} />
                  Reset
                </button>
              </div>
              <Bar
                label="Average FPS"
                cpu={averages.cpuFps}
                gpu={averages.gpuFps}
                format={(n) => n.toFixed(1)}
              />
              <Bar
                label="Average inference"
                cpu={averages.cpuMs}
                gpu={averages.gpuMs}
                format={(n) => `${n.toFixed(1)} ms`}
                invert
              />
              <div className="mt-4 flex items-center justify-between rounded-xl bg-white/4 px-3 py-2.5">
                <span className="flex items-center gap-2 text-sm text-white/60">
                  <Zap size={14} className="text-gpu" />
                  Simulated GPU speedup
                </span>
                <span className="font-mono text-lg font-semibold text-gpu">
                  {averages.speedup ? `${averages.speedup.toFixed(2)}×` : "—"}
                </span>
              </div>
              <p className="mt-3 font-mono text-[11px] text-white/30">
                {averages.cpuFrames} CPU frames · {averages.gpuFrames} GPU frames
              </p>
            </div>
          </aside>
        </div>

        {(error || status?.modelError) && (
          <p className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error || status?.modelError}
          </p>
        )}
        {!ready && !status?.modelError && (
          <p className="font-mono text-xs text-white/40">
            Loading YOLOv8 weights… first run downloads the model automatically.
          </p>
        )}
      </div>
    </div>
  );
}

function ModeToggle({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  return (
    <div className="flex rounded-2xl border border-white/10 bg-black/40 p-1">
      <button
        onClick={() => onChange("CPU")}
        className={`inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition ${
          mode === "CPU" ? "bg-cpu text-black" : "text-white/55 hover:text-white"
        }`}
      >
        <Cpu size={16} />
        CPU
      </button>
      <button
        onClick={() => onChange("GPU")}
        className={`inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition ${
          mode === "GPU" ? "bg-gpu text-black" : "text-white/55 hover:text-white"
        }`}
      >
        <Zap size={16} />
        GPU
      </button>
    </div>
  );
}

function Metric({
  label,
  value,
  unit,
  icon,
  accent,
}: {
  label: string;
  value: string;
  unit?: string;
  icon: ReactNode;
  accent: "cpu" | "gpu" | "neutral";
}) {
  const color =
    accent === "cpu" ? "text-cpu" : accent === "gpu" ? "text-gpu" : "text-white";
  return (
    <div className="rounded-2xl border border-white/8 bg-panel px-3 py-3">
      <p className="flex items-center gap-1.5 text-[11px] tracking-wide text-white/40 uppercase">
        {icon}
        {label}
      </p>
      <p className={`mt-1 font-mono text-2xl font-semibold ${color}`}>
        {value}
        {unit && <span className="ml-1 text-xs font-normal text-white/40">{unit}</span>}
      </p>
    </div>
  );
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  const path = useMemo(() => {
    if (values.length < 2) return "";
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const span = Math.max(max - min, 1);
    return values
      .map((v, i) => {
        const x = (i / (values.length - 1)) * 100;
        const y = 36 - ((v - min) / span) * 32;
        return `${i === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ");
  }, [values]);

  return (
    <svg viewBox="0 0 100 40" className="h-20 w-full" preserveAspectRatio="none">
      <path d={path} fill="none" stroke={color} strokeWidth="1.6" />
    </svg>
  );
}

function Bar({
  label,
  cpu,
  gpu,
  format,
  invert = false,
}: {
  label: string;
  cpu: number;
  gpu: number;
  format: (n: number) => string;
  invert?: boolean;
}) {
  const peak = Math.max(cpu, gpu, 1);
  return (
    <div className="mb-4">
      <p className="mb-2 text-xs text-white/45">{label}</p>
      <Row name="CPU" color="bg-cpu" width={(cpu / peak) * 100} value={format(cpu)} />
      <Row name="GPU" color="bg-gpu" width={(gpu / peak) * 100} value={format(gpu)} />
      {invert && cpu > 0 && gpu > 0 && (
        <p className="mt-1 text-[11px] text-white/30">Lower is faster</p>
      )}
    </div>
  );
}

function Row({
  name,
  color,
  width,
  value,
}: {
  name: string;
  color: string;
  width: number;
  value: string;
}) {
  return (
    <div className="mb-1.5 grid grid-cols-[40px_1fr_88px] items-center gap-2">
      <span className="font-mono text-[11px] text-white/45">{name}</span>
      <div className="h-2 overflow-hidden rounded-full bg-white/8">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
      <span className="text-right font-mono text-xs">{value}</span>
    </div>
  );
}

function EmptyStage({ ready, hasSample }: { ready: boolean; hasSample: boolean }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/4">
        <ScanEye className="text-white/50" />
      </div>
      <p className="text-lg font-medium">Ready when you are</p>
      <p className="max-w-md text-sm text-white/45">
        {ready
          ? hasSample
            ? "Start inference on the sample clip, upload your own video, or use a webcam."
            : "Upload a short mp4 (street, office, or traffic works best) or switch to webcam."
          : "The YOLO model is loading. This only takes a moment after the first download."}
      </p>
    </div>
  );
}
