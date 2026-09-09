export type Mode = "CPU" | "GPU";

export type Averages = {
  cpuFps: number;
  gpuFps: number;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
  cpuFrames: number;
  gpuFrames: number;
};

export type FramePayload = {
  type: "frame" | "status";
  running: boolean;
  mode: Mode;
  simulated: boolean;
  frameId: number;
  jpeg?: string | null;
  objects: number;
  inferenceMs: number;
  fps: number;
  cpuSpark: number[];
  gpuSpark: number[];
  averages: Averages;
};

export type ServerStatus = {
  modelReady: boolean;
  modelError: string | null;
  running: boolean;
  mode: Mode;
  source: string;
  hasSampleVideo: boolean;
  sampleName: string;
};
