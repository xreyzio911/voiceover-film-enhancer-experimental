"use client";

import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import JSZip from "jszip";
import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { analyzeFloatSamples, type AudioQcMetrics } from "../lib/audioQc";
import styles from "./VoLeveler.module.css";

const LOUDNESS_PRESETS = {
  "ATSC A/85 (-24 LKFS, -2 dBTP)": { I: "-24", TP: "-2", LRA: "7", suffix: "A85" },
  "EBU R128 (-23 LUFS, -1 dBTP)": { I: "-23", TP: "-1", LRA: "7", suffix: "R128" },
  "Mix-ready only (no loudness normalize)": null,
} as const;

const BREATH_COMPAND = {
  Off: null,
  Light: "compand=attacks=0.2:decays=0.8:points=-90/-90|-60/-66|-40/-40|-20/-20|0/0",
  Medium: "compand=attacks=0.2:decays=0.8:points=-90/-90|-60/-70|-40/-40|-20/-20|0/0",
} as const;

const BREATH_COMPAND_SAFE = {
  Light: "compand=attacks=0.25:decays=1.0:points=-90/-90|-74/-75|-64/-65|-54/-54|-40/-40|-20/-20|0/0",
  Medium: "compand=attacks=0.25:decays=1.0:points=-90/-90|-76/-78|-66/-68|-56/-56|-40/-40|-20/-20|0/0",
} as const;

const FLOOR_GUARD =
  "compand=attacks=0.05:decays=0.2:points=-90/-95|-70/-74|-60/-60|-50/-50|-20/-20|0/0";
const FLOOR_GUARD_STRONG =
  "compand=attacks=0.04:decays=0.18:points=-90/-100|-75/-82|-64/-65|-52/-53|-20/-20|0/0";

const LEVELER_PRESETS = {
  "Minimal (no auto-leveler)": {
    dyna: null,
    compressor: { threshold: "-27dB", ratio: "1.7" },
  },
  Gentle: {
    dyna: { f: 181, g: 5, m: 5 },
    compressor: { threshold: "-26dB", ratio: "2.05" },
  },
  Balanced: {
    dyna: { f: 221, g: 7, m: 7 },
    compressor: { threshold: "-24dB", ratio: "2.25" },
  },
  Firm: {
    dyna: { f: 271, g: 9, m: 9 },
    compressor: { threshold: "-22dB", ratio: "2.45" },
  },
} as const;

const LEVELER_CONSISTENCY = {
  "Minimal (no auto-leveler)": 0.15,
  Gentle: 0.45,
  Balanced: 0.65,
  Firm: 0.85,
} as const;

const SMART_MATCH_PRESETS = {
  Off: { tone: 0, dynamics: 0 },
  Gentle: { tone: 0.45, dynamics: 0.3 },
  Balanced: { tone: 0.7, dynamics: 0.5 },
} as const;

const CORE_BASE_URL = "ffmpeg/ffmpeg-core";
const ANALYSIS_SAMPLE_SECONDS = 180;
const ANALYSIS_SAMPLE_RATE = 16000;
const ENVELOPE_FRAME_MS = 10;
const DISTRIBUTED_ANALYSIS_THRESHOLD_SECONDS = ANALYSIS_SAMPLE_SECONDS + 30;
const DISTRIBUTED_ANALYSIS_WINDOW_SECONDS = 30;
const DISTRIBUTED_ANALYSIS_TARGET_COUNT = 6;
const MIX_SEGMENT_SECONDS = 75;
const MIX_SEGMENT_MIN_DURATION_SECONDS = 105;
const LONG_SPARSE_DURATION_SECONDS = 480;
const LONG_SPARSE_ANALYSIS_WINDOW_SECONDS = 24;
const LONG_SPARSE_ANALYSIS_WINDOW_TARGET_COUNT = 8;
const SPEECH_ALIGNED_SEGMENT_TARGET_SECONDS = 30;
const SPEECH_ALIGNED_SEGMENT_MAX_SECONDS = 42;
const SPEECH_ALIGNED_SEGMENT_PAD_IN_MS = 160;
const SPEECH_ALIGNED_SEGMENT_PAD_OUT_MS = 320;
const SPEECH_ENDING_EXTRA_PAD_OUT_MS = 220;
const SEGMENT_GAIN_MATCH_MIN_DELTA_DB = 0.35;
const SEGMENT_GAIN_MATCH_MAX_DB = 1.8;
const BATCH_MEMORY_GUARD_FILE_THRESHOLD = 8;
const BATCH_MEMORY_GUARD_INTERVAL = 3;
const LIMITER_FILTER = "alimiter=limit=-2dB:level=disabled";
const FATAL_FFMPEG_PATTERN = /memory access out of bounds|runtimeerror/i;
const IMPORTANT_LOG_PATTERN = /error|failed|invalid|aborted|out of bounds/i;

const sanitizeBase = (name: string) =>
  name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9-_]+/g, "_");

const formatBytes = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const toOddInt = (value: number, min: number, max: number) => {
  let rounded = Math.round(clamp(value, min, max));
  if (rounded % 2 === 0) {
    rounded += rounded >= max ? -1 : 1;
  }
  return rounded;
};

const median = (values: number[]) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
};

const fromDb = (db: number) => Math.pow(10, db / 20);

const robustMedian = (values: number[]) => {
  if (values.length === 0) return null;
  const baseMedian = median(values);
  if (baseMedian === null) return null;

  const deviations = values.map((value) => Math.abs(value - baseMedian));
  const mad = median(deviations) ?? 0;
  if (mad <= 1e-6) return baseMedian;

  const scale = 1.4826 * mad;
  const filtered = values.filter((value) => Math.abs(value - baseMedian) / scale <= 2.8);
  return median(filtered.length > 0 ? filtered : values);
};

const downgradeRoomRisk = (risk: RoomRisk): RoomRisk => {
  if (risk === "high") return "medium";
  if (risk === "medium") return "low";
  return "low";
};

const classifyRoomRisk = (roomScore: number): RoomRisk => {
  if (roomScore < 0.33) return "low";
  if (roomScore < 0.58) return "medium";
  return "high";
};

const parseMaybeNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value.trim());
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const formatSigned = (value: number, decimals = 1) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}`;

const shouldRecycleFfmpegForBatch = (completedCount: number, totalCount: number) =>
  totalCount >= BATCH_MEMORY_GUARD_FILE_THRESHOLD &&
  completedCount < totalCount &&
  completedCount % BATCH_MEMORY_GUARD_INTERVAL === 0;

type OutputEntry = {
  name: string;
  url: string;
  size: number;
  kind: "mixready" | "loudness";
  variant: "clean" | "blend";
};

type SilenceSpan = {
  startSec: number;
  endSec: number;
};

type SpeechSpan = {
  startSec: number;
  endSec: number;
};

type RenderSegment = {
  startSec: number;
  endSec: number;
  process: boolean;
  trimInMs: number;
  trimOutMs: number;
  forceEndingProtection?: boolean;
};

type FileAnalysis = {
  inputI: number | null;
  inputLRA: number | null;
  inputTP: number | null;
  inputThresh: number | null;
  lowRms: number | null;
  midRms: number | null;
  highRms: number | null;
  noiseFloorDb: number | null;
  pauseNoiseFloorDb: number | null;
  nearSpeechNoiseFloorDb: number | null;
  speechThresholdDb: number | null;
  noiseContrastDb: number | null;
  dynamicRangeDb: number | null;
  reverbScore: number | null;
  echoScore: number | null;
  roomScore: number | null;
  echoDelayMs: number | null;
  analysisConfidence: number | null;
  drynessScore: number | null;
  instabilityScore: number | null;
  lineSwingScore: number | null;
  sentenceJumpScore: number | null;
  breathSpikeRisk: number | null;
  pauseNoiseRisk: number | null;
  compressionScore: number | null;
  overallRisk: number | null;
  clickScore: number | null;
  speechDutyCyclePct: number | null;
  speechSegmentCount: number | null;
  medianSpeechRunMs: number | null;
  longSilenceCount: number | null;
  onsetOvershootScore: number | null;
  midLineSagScore: number | null;
  endFadeRiskScore: number | null;
  analysisWindowCount: number | null;
  longSparseModeEligible: boolean | null;
};

type BatchReference = {
  lowTilt: number;
  highTilt: number;
  lra: number;
};

type NoiseRisk = "low" | "medium" | "high";
type RoomRisk = "low" | "medium" | "high";

type AdaptiveProfile = {
  highpassHz: number;
  lowMidGainDb: number;
  presenceGainDb: number;
  airGainDb: number;
  emotionalHarshnessCutDb: number;
  topEndHarshnessCutDb: number;
  levelingNeed: number;
  emotionProtection: number;
  compressorRatioOffset: number;
  compressorThresholdOffsetDb: number;
  dynaTrim: number;
  floorGuardFilter: string;
  noiseRisk: NoiseRisk;
  noiseFloorDb: number | null;
  pauseNoiseRisk: number;
  speechThresholdDb: number | null;
  roomRisk: RoomRisk;
  useDenoise: boolean;
  denoiseStrength: number;
  useTailGate: boolean;
  tailGateStrength: number;
  echoNotchCutDb: number;
  instabilityScore: number;
  clickScore: number;
  clickTameStrength: number;
  lineSwingScore: number;
  sentenceJumpScore: number;
  breathSpikeRisk: number;
  breathTameStrength: number;
  lineContinuityRisk: number;
  preserveEndings: boolean;
  onsetTameStrength: number;
  sagRecoveryStrength: number;
  disableDynaThresholdForStability: boolean;
  strictEndingProtection: boolean;
  preferSinglePassContinuity: boolean;
  segmentMatchTargetI: number | null;
  useSpeechAlignedSegmentation: boolean;
  useSpeechPauseSegmentation: boolean;
  segmentTargetSec: number;
  segmentMaxSec: number;
  blendIndoorGain: number;
  blendOutdoorGain: number;
  blendIndoorDelayMs: number;
  blendOutdoorDelayMs: number;
};

type JobEntry = {
  file: File;
  base: string;
  inputName: string;
  mixName: string;
  blendMixName: string;
};

type FailedOptimization = {
  base: string;
  fileName: string;
  reason: string;
};

type AnalysisResult = {
  analysis: FileAnalysis;
  ffmpeg: FFmpeg;
};

type QueueItemStatus = "pending" | "working" | "done" | "error";

type QueueItem = {
  base: string;
  fileName: string;
  index: number;
  status: QueueItemStatus;
  stageLabel: string;
  progress: number;
  detail: string | null;
  updatedAtMs: number;
};

const createEmptyAnalysis = (): FileAnalysis => ({
  inputI: null,
  inputLRA: null,
  inputTP: null,
  inputThresh: null,
  lowRms: null,
  midRms: null,
  highRms: null,
  noiseFloorDb: null,
  pauseNoiseFloorDb: null,
  nearSpeechNoiseFloorDb: null,
  speechThresholdDb: null,
  noiseContrastDb: null,
  dynamicRangeDb: null,
  reverbScore: null,
  echoScore: null,
  roomScore: null,
  echoDelayMs: null,
  analysisConfidence: null,
  drynessScore: null,
  instabilityScore: null,
  lineSwingScore: null,
  sentenceJumpScore: null,
  breathSpikeRisk: null,
  pauseNoiseRisk: null,
  compressionScore: null,
  overallRisk: null,
  clickScore: null,
  speechDutyCyclePct: null,
  speechSegmentCount: null,
  medianSpeechRunMs: null,
  longSilenceCount: null,
  onsetOvershootScore: null,
  midLineSagScore: null,
  endFadeRiskScore: null,
  analysisWindowCount: null,
  longSparseModeEligible: null,
});

export default function VoLeveler() {
  const ffmpegRef = useRef<FFmpeg | null>(null);
  const logBufferRef = useRef<string[]>([]);
  const activeQueueBaseRef = useRef<string | null>(null);
  const activeQueueStageRef = useRef<string>("Queued");
  const activeQueueProgressRef = useRef<number>(-1);

  const [files, setFiles] = useState<File[]>([]);
  const [outputs, setOutputs] = useState<OutputEntry[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("Idle");
  const [loading, setLoading] = useState(false);
  const [zipBusy, setZipBusy] = useState(false);
  const [zipProgress, setZipProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [failedOptimizations, setFailedOptimizations] = useState<FailedOptimization[]>([]);
  const [showFailureWarning, setShowFailureWarning] = useState(false);
  const [queueItems, setQueueItems] = useState<QueueItem[]>([]);

  const [loudnessTarget, setLoudnessTarget] = useState<keyof typeof LOUDNESS_PRESETS>(
    "ATSC A/85 (-24 LKFS, -2 dBTP)"
  );
  const [keepMixReady, setKeepMixReady] = useState(true);
  const [smartMatchMode, setSmartMatchMode] = useState<keyof typeof SMART_MATCH_PRESETS>("Gentle");
  const [eqCleanup, setEqCleanup] = useState(true);
  const [breathControl, setBreathControl] = useState<keyof typeof BREATH_COMPAND>("Light");
  const [leveler, setLeveler] = useState<keyof typeof LEVELER_PRESETS>("Balanced");
  const [roomCleanup, setRoomCleanup] = useState(true);
  const [sceneBlend, setSceneBlend] = useState(true);
  const [softenHarshness, setSoftenHarshness] = useState(true);
  const [noiseGuard, setNoiseGuard] = useState(true);
  const [floorGuard, setFloorGuard] = useState(true);

  const loudnessConfig = useMemo(() => LOUDNESS_PRESETS[loudnessTarget], [loudnessTarget]);
  const smartMatchConfig = useMemo(() => SMART_MATCH_PRESETS[smartMatchMode], [smartMatchMode]);

  useEffect(() => {
    return () => {
      outputs.forEach((output) => URL.revokeObjectURL(output.url));
    };
  }, [outputs]);

  useEffect(() => {
    return () => {
      if (ffmpegRef.current) {
        try {
          ffmpegRef.current.terminate();
        } catch {
          // Ignore terminate failures during unmount.
        }
        ffmpegRef.current = null;
      }
    };
  }, []);

  const appendLog = (message: string) => {
    setLogs((prev) => [...prev.slice(-300), message]);
  };

  const initializeQueueItems = (jobs: JobEntry[]) => {
    const now = Date.now();
    setQueueItems(
      jobs.map((job, index) => ({
        base: job.base,
        fileName: job.file.name,
        index,
        status: "pending",
        stageLabel: "Queued",
        progress: 0,
        detail: null,
        updatedAtMs: now,
      }))
    );
    activeQueueBaseRef.current = null;
    activeQueueStageRef.current = "Queued";
    activeQueueProgressRef.current = -1;
  };

  const updateQueueItem = (
    base: string,
    patch: Partial<Pick<QueueItem, "status" | "stageLabel" | "progress" | "detail">>
  ) => {
    const now = Date.now();
    setQueueItems((prev) =>
      prev.map((item) => (item.base === base ? { ...item, ...patch, updatedAtMs: now } : item))
    );
  };

  const setActiveQueueStage = (base: string, stageLabel: string, detail?: string | null) => {
    activeQueueBaseRef.current = base;
    activeQueueStageRef.current = stageLabel;
    activeQueueProgressRef.current = -1;
    updateQueueItem(base, {
      status: "working",
      stageLabel,
      progress: 0,
      detail: detail ?? null,
    });
  };

  const markQueuePending = (base: string, stageLabel: string, detail?: string | null) => {
    if (activeQueueBaseRef.current === base) {
      activeQueueBaseRef.current = null;
      activeQueueStageRef.current = "Queued";
      activeQueueProgressRef.current = -1;
    }
    updateQueueItem(base, {
      status: "pending",
      stageLabel,
      progress: 0,
      detail: detail ?? null,
    });
  };

  const markQueueDone = (base: string, detail?: string | null) => {
    if (activeQueueBaseRef.current === base) {
      activeQueueBaseRef.current = null;
      activeQueueStageRef.current = "Complete";
      activeQueueProgressRef.current = -1;
    }
    updateQueueItem(base, {
      status: "done",
      stageLabel: "Complete",
      progress: 1,
      detail: detail ?? null,
    });
  };

  const markQueueError = (base: string, detail?: string | null) => {
    if (activeQueueBaseRef.current === base) {
      activeQueueBaseRef.current = null;
      activeQueueStageRef.current = "Error";
      activeQueueProgressRef.current = -1;
    }
    updateQueueItem(base, {
      status: "error",
      stageLabel: "Error",
      progress: 1,
      detail: detail ?? null,
    });
  };

  const toBlobURLSafe = async (url: string, mime: string) => {
    try {
      return await toBlobURL(url, mime);
    } catch {
      return undefined;
    }
  };

  const teardownFfmpeg = () => {
    if (!ffmpegRef.current) return;
    try {
      ffmpegRef.current.terminate();
    } catch {
      // Ignore terminate failures while resetting worker state.
    } finally {
      ffmpegRef.current = null;
      logBufferRef.current = [];
    }
  };

  const hasFatalFfmpegSignal = (text: string) => FATAL_FFMPEG_PATTERN.test(text);

  const shouldResetFfmpegForError = (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    return hasFatalFfmpegSignal(message) || /RuntimeError|memory access out of bounds/i.test(message);
  };

  const refreshFfmpeg = async (reason: string) => {
    appendLog(`Resetting FFmpeg worker (${reason})...`);
    teardownFfmpeg();
    return await ensureFfmpeg();
  };

  const ensureFfmpeg = async () => {
    if (ffmpegRef.current) return ffmpegRef.current;

    const ffmpeg = new FFmpeg();
    ffmpeg.on("log", ({ message }) => {
      if (message.trim() === "Aborted()") {
        // Common during terminate/reset; not a reliable execution failure signal.
        return;
      }
      logBufferRef.current.push(message);
      if (logBufferRef.current.length > 2000) {
        logBufferRef.current = logBufferRef.current.slice(-1200);
      }
      if (IMPORTANT_LOG_PATTERN.test(message)) {
        appendLog(message);
      }
    });

    ffmpeg.on("progress", ({ progress }) => {
      if (progress > 0) {
        setStatus(`Processing ${(progress * 100).toFixed(0)}%`);
        const activeBase = activeQueueBaseRef.current;
        if (activeBase) {
          const clampedProgress = clamp(progress, 0, 1);
          if (Math.abs(clampedProgress - activeQueueProgressRef.current) >= 0.03 || clampedProgress >= 0.995) {
            activeQueueProgressRef.current = clampedProgress;
            updateQueueItem(activeBase, {
              status: "working",
              stageLabel: activeQueueStageRef.current,
              progress: clampedProgress,
            });
          }
        }
      }
    });

    setStatus("Loading FFmpeg core...");
    const coreURL = await toBlobURL(`${CORE_BASE_URL}.js`, "text/javascript");
    const wasmURL = await toBlobURL(`${CORE_BASE_URL}.wasm`, "application/wasm");
    const workerURL = await toBlobURLSafe(`${CORE_BASE_URL}.worker.js`, "text/javascript");

    await ffmpeg.load({
      coreURL,
      wasmURL,
      ...(workerURL ? { workerURL } : {}),
    });

    ffmpegRef.current = ffmpeg;
    setStatus("FFmpeg ready");
    return ffmpeg;
  };

  const resetLogBuffer = () => {
    const snapshot = logBufferRef.current.join("\n");
    logBufferRef.current = [];
    return snapshot;
  };

  const parseLoudnormJson = (text: string) => {
    const matches = text.match(/\{[\s\S]*?\}/g);
    if (!matches || matches.length === 0) return null;
    try {
      return JSON.parse(matches[matches.length - 1]) as Record<string, unknown>;
    } catch {
      return null;
    }
  };

const parseRmsFromAstats = (text: string) => {
    const matches = text.match(/RMS level dB:\s*(-?(?:\d+(?:\.\d+)?|inf))/gi);
    if (!matches || matches.length === 0) return null;
    const raw = matches[matches.length - 1].split(":").at(-1)?.trim().toLowerCase();
    if (!raw) return null;
    if (raw === "-inf" || raw === "inf") return -120;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
};

const parseDurationSeconds = (text: string) => {
  const match = text.match(/Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)/i);
  if (!match) return null;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  const seconds = Number(match[3]);
  if (!Number.isFinite(hours) || !Number.isFinite(minutes) || !Number.isFinite(seconds)) {
    return null;
  }
  return hours * 3600 + minutes * 60 + seconds;
};

const summarizeFailureLog = (text: string) => {
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const important = lines.filter((line) => IMPORTANT_LOG_PATTERN.test(line));
  const selected = (important.length > 0 ? important : lines).slice(-3);
  return selected.join(" | ");
};

const describeError = (error: unknown) => (error instanceof Error ? error.message : String(error));
const summarizeFailureReason = (error: unknown) => {
  const compact = describeError(error).replace(/\s+/g, " ").trim();
  if (compact.length <= 180) return compact;
  return `${compact.slice(0, 177)}...`;
};

  const execOrThrow = async (ffmpeg: FFmpeg, args: string[], context: string) => {
    const exitCode = await ffmpeg.exec(args);
    const snapshot = logBufferRef.current.join("\n");
    if (exitCode !== 0) {
      const summary = summarizeFailureLog(snapshot);
      const exitText = exitCode !== 0 ? ` (exit ${exitCode})` : "";
      throw new Error(`${context} failed${exitText}${summary ? `: ${summary}` : ""}`);
    }
  };

  const readVirtualFileBytes = async (ffmpeg: FFmpeg, name: string) => {
    const data = await ffmpeg.readFile(name);
    return typeof data === "string" ? new TextEncoder().encode(data) : new Uint8Array(data);
  };

  const toFloatSamples = (bytes: Uint8Array) => {
    const usableLength = bytes.byteLength - (bytes.byteLength % 4);
    if (usableLength <= 0) return new Float32Array(0);
    if (bytes.byteOffset % 4 === 0) {
      return new Float32Array(bytes.buffer, bytes.byteOffset, usableLength / 4).slice();
    }
    const aligned = bytes.slice(0, usableLength);
    return new Float32Array(aligned.buffer, aligned.byteOffset, usableLength / 4);
  };

  const emptyEnvelopeMetrics = {
    noiseFloorDb: null,
    pauseNoiseFloorDb: null,
    nearSpeechNoiseFloorDb: null,
    speechThresholdDb: null,
    noiseContrastDb: null,
    dynamicRangeDb: null,
    reverbScore: null,
    echoScore: null,
    roomScore: null,
    echoDelayMs: null,
    analysisConfidence: null,
    drynessScore: null,
    instabilityScore: null,
    lineSwingScore: null,
    sentenceJumpScore: null,
    breathSpikeRisk: null,
    pauseNoiseRisk: null,
    compressionScore: null,
    overallRisk: null,
    clickScore: null,
    speechDutyCyclePct: null,
    speechSegmentCount: null,
    medianSpeechRunMs: null,
    longSilenceCount: null,
    onsetOvershootScore: null,
    midLineSagScore: null,
    endFadeRiskScore: null,
  } satisfies Partial<FileAnalysis>;

  const mapQcMetricsToEnvelopeMetrics = (metrics: AudioQcMetrics) => ({
    noiseFloorDb: metrics.noiseFloorDb,
    pauseNoiseFloorDb: metrics.pauseNoiseFloorDb,
    nearSpeechNoiseFloorDb: metrics.nearSpeechNoiseFloorDb,
    speechThresholdDb: metrics.speechThresholdDb,
    noiseContrastDb: metrics.noiseContrastDb,
    dynamicRangeDb: metrics.dynamicRangeDb,
    reverbScore: metrics.reverbScore,
    echoScore: metrics.echoScore,
    roomScore: metrics.roomScore,
    echoDelayMs: metrics.echoDelayMs,
    analysisConfidence: metrics.analysisConfidence,
    drynessScore: metrics.drynessScore,
    instabilityScore: metrics.instabilityScore,
    lineSwingScore: metrics.lineSwingScore,
    sentenceJumpScore: metrics.sentenceJumpScore,
    breathSpikeRisk: metrics.breathSpikeRisk,
    pauseNoiseRisk: metrics.pauseNoiseRisk,
    compressionScore: metrics.compressionScore,
    overallRisk: metrics.overallRisk,
    clickScore: metrics.clickScore,
    speechDutyCyclePct: metrics.speechDutyCyclePct,
    speechSegmentCount: metrics.speechSegmentCount,
    medianSpeechRunMs: metrics.medianSpeechRunMs,
    longSilenceCount: metrics.longSilenceCount,
    onsetOvershootScore: metrics.onsetOvershootScore,
    midLineSagScore: metrics.midLineSagScore,
    endFadeRiskScore: metrics.endFadeRiskScore,
  });

  const computeEnvelopeMetrics = (samples: Float32Array) => {
    const frameSize = Math.max(1, Math.round((ANALYSIS_SAMPLE_RATE * ENVELOPE_FRAME_MS) / 1000));
    const frameCount = Math.floor(samples.length / frameSize);
    if (frameCount < 20) {
      return emptyEnvelopeMetrics;
    }
    return mapQcMetricsToEnvelopeMetrics(analyzeFloatSamples(samples, ANALYSIS_SAMPLE_RATE, ENVELOPE_FRAME_MS));
  };

  const parseSilencedetectSpans = (text: string, durationSeconds: number | null): SilenceSpan[] => {
    const lines = text.split(/\r?\n/);
    const spans: SilenceSpan[] = [];
    let currentStart: number | null = null;

    for (const rawLine of lines) {
      const line = rawLine.trim();
      const startMatch = line.match(/silence_start:\s*([0-9]+(?:\.[0-9]+)?)/i);
      if (startMatch) {
        const parsed = Number(startMatch[1]);
        if (Number.isFinite(parsed)) currentStart = parsed;
      }
      const endMatch = line.match(/silence_end:\s*([0-9]+(?:\.[0-9]+)?)/i);
      if (endMatch) {
        const end = Number(endMatch[1]);
        if (!Number.isFinite(end)) continue;
        const start = currentStart ?? Math.max(0, end - 0.32);
        spans.push({ startSec: start, endSec: end });
        currentStart = null;
      }
    }

    if (currentStart !== null && durationSeconds !== null) {
      spans.push({ startSec: currentStart, endSec: durationSeconds });
    }

    const clamped = spans
      .map((span) => ({
        startSec: clamp(span.startSec, 0, durationSeconds ?? Math.max(span.endSec, span.startSec)),
        endSec: clamp(span.endSec, 0, durationSeconds ?? Math.max(span.endSec, span.startSec)),
      }))
      .filter((span) => span.endSec - span.startSec >= 0.01)
      .sort((a, b) => a.startSec - b.startSec);

    const merged: SilenceSpan[] = [];
    for (const span of clamped) {
      const last = merged.at(-1);
      if (!last || span.startSec > last.endSec + 0.02) {
        merged.push({ ...span });
      } else {
        last.endSec = Math.max(last.endSec, span.endSec);
      }
    }
    return merged;
  };

  const deriveSpeechSpans = (silenceSpans: SilenceSpan[], durationSeconds: number): SpeechSpan[] => {
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return [];
    const speechSpans: SpeechSpan[] = [];
    let cursor = 0;
    for (const silence of silenceSpans) {
      const start = clamp(silence.startSec, 0, durationSeconds);
      const end = clamp(silence.endSec, 0, durationSeconds);
      if (start > cursor + 0.01) {
        speechSpans.push({ startSec: cursor, endSec: start });
      }
      cursor = Math.max(cursor, end);
    }
    if (cursor < durationSeconds - 0.01) {
      speechSpans.push({ startSec: cursor, endSec: durationSeconds });
    }
    return speechSpans.filter((span) => span.endSec - span.startSec >= 0.06);
  };

  const overlapSeconds = (aStart: number, aEnd: number, bStart: number, bEnd: number) =>
    Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));

  const speechOccupancyPctInWindow = (speechSpans: SpeechSpan[], startSec: number, durationSec: number) => {
    const endSec = startSec + durationSec;
    let speech = 0;
    for (const span of speechSpans) {
      if (span.endSec <= startSec) continue;
      if (span.startSec >= endSec) break;
      speech += overlapSeconds(startSec, endSec, span.startSec, span.endSec);
    }
    return (speech / Math.max(durationSec, 1e-6)) * 100;
  };

  const computeSpeechMapStats = (speechSpans: SpeechSpan[], silenceSpans: SilenceSpan[], durationSeconds: number) => {
    const speechDurationsMs = speechSpans.map((span) => (span.endSec - span.startSec) * 1000);
    const silenceDurationsMs = silenceSpans.map((span) => (span.endSec - span.startSec) * 1000);
    const totalSpeechSeconds = speechSpans.reduce((sum, span) => sum + (span.endSec - span.startSec), 0);
    const speechDutyCyclePct = (totalSpeechSeconds / Math.max(durationSeconds, 1e-6)) * 100;
    const medianSpeechRunMs = median(speechDurationsMs) ?? 0;
    const longSilenceCount = silenceDurationsMs.filter((ms) => ms >= 1500).length;
    return {
      speechDutyCyclePct,
      speechSegmentCount: speechSpans.length,
      medianSpeechRunMs,
      longSilenceCount,
      longSparseModeEligible:
        durationSeconds >= LONG_SPARSE_DURATION_SECONDS &&
        speechDutyCyclePct <= 38 &&
        (longSilenceCount >= 12 || medianSpeechRunMs <= 2600),
    };
  };

  const selectDistributedAnalysisWindowsWithConfig = (
    speechSpans: SpeechSpan[],
    durationSeconds: number,
    windowSec: number,
    targetCount: number
  ) => {
    const maxStart = Math.max(0, durationSeconds - windowSec);

    type Candidate = { startSec: number; occupancyPct: number; centerSec: number; spanLenSec: number };
    const candidates: Candidate[] = [];

    for (const span of speechSpans) {
      const spanLenSec = span.endSec - span.startSec;
      if (spanLenSec <= 0.05) continue;
      const stepSec = spanLenSec > windowSec ? Math.max(10, windowSec * 0.75) : spanLenSec;
      const centerStart = span.startSec + spanLenSec / 2;
      for (let t = span.startSec; t <= span.endSec; t += stepSec) {
        const centerSec = clamp(t, span.startSec, span.endSec);
        const startSec = clamp(centerSec - windowSec / 2, 0, maxStart);
        const occupancyPct = speechOccupancyPctInWindow(speechSpans, startSec, windowSec);
        candidates.push({ startSec, occupancyPct, centerSec, spanLenSec });
      }
      const centeredStart = clamp(centerStart - windowSec / 2, 0, maxStart);
      candidates.push({
        startSec: centeredStart,
        occupancyPct: speechOccupancyPctInWindow(speechSpans, centeredStart, windowSec),
        centerSec: centerStart,
        spanLenSec,
      });
    }

    if (candidates.length === 0) {
      const starts: number[] = [];
      const count = Math.min(targetCount, Math.max(1, Math.ceil(durationSeconds / windowSec)));
      for (let i = 0; i < count; i += 1) {
        const ratio = count === 1 ? 0 : i / (count - 1);
        starts.push(clamp(ratio * maxStart, 0, maxStart));
      }
      return starts.map((startSec) => ({
        startSec,
        durationSec: Math.min(windowSec, durationSeconds - startSec),
        occupancyPct: 0,
      }));
    }

    const buckets = new Map<number, Candidate[]>();
    for (const candidate of candidates) {
      const bucketIndex = clamp(Math.floor((candidate.centerSec / Math.max(durationSeconds, 1e-6)) * targetCount), 0, targetCount - 1);
      const bucket = buckets.get(bucketIndex) ?? [];
      bucket.push(candidate);
      buckets.set(bucketIndex, bucket);
    }

    const selected: Candidate[] = [];
    for (let bucketIndex = 0; bucketIndex < targetCount; bucketIndex += 1) {
      const bucket = buckets.get(bucketIndex);
      if (!bucket || bucket.length === 0) continue;
      bucket.sort((a, b) => b.occupancyPct - a.occupancyPct || a.startSec - b.startSec);
      selected.push(bucket[0]);
    }

    const selectedKeys = new Set(selected.map((item) => item.startSec.toFixed(2)));
    const remaining = [...candidates]
      .filter((candidate) => !selectedKeys.has(candidate.startSec.toFixed(2)))
      .sort((a, b) => b.occupancyPct - a.occupancyPct || a.startSec - b.startSec);

    for (const candidate of remaining) {
      if (selected.length >= targetCount) break;
      const tooClose = selected.some((picked) => Math.abs(picked.startSec - candidate.startSec) < windowSec * 0.35);
      if (tooClose) continue;
      selected.push(candidate);
    }

    const thresholdCandidates = selected.filter((candidate) => candidate.occupancyPct >= 12);
    const finalList = thresholdCandidates.length > 0 ? thresholdCandidates : selected;
    finalList.sort((a, b) => a.startSec - b.startSec);

    return finalList.slice(0, targetCount).map((candidate) => ({
      startSec: candidate.startSec,
      durationSec: Math.min(windowSec, Math.max(1, durationSeconds - candidate.startSec)),
      occupancyPct: candidate.occupancyPct,
    }));
  };

  const selectSpeechAnchoredAnalysisWindow = (
    speechSpans: SpeechSpan[],
    durationSeconds: number,
    windowSec: number
  ): { startSec: number; durationSec: number; occupancyPct: number } | null => {
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0 || speechSpans.length === 0) {
      return null;
    }

    const safeWindowSec = clamp(windowSec, 1, Math.max(1, durationSeconds));
    const maxStart = Math.max(0, durationSeconds - safeWindowSec);

    const candidateStarts = new Set<number>();
    const addCandidate = (startSec: number) => {
      const clampedStart = clamp(startSec, 0, maxStart);
      candidateStarts.add(Number(clampedStart.toFixed(3)));
    };

    for (const span of speechSpans) {
      const spanLen = Math.max(0, span.endSec - span.startSec);
      if (spanLen < 0.05) continue;
      addCandidate(span.startSec - safeWindowSec * 0.12);
      addCandidate(((span.startSec + span.endSec) / 2) - safeWindowSec / 2);
      addCandidate(span.endSec - safeWindowSec * 0.88);
      if (spanLen > safeWindowSec) {
        const stepSec = Math.max(12, safeWindowSec * 0.45);
        for (let t = span.startSec; t <= span.endSec; t += stepSec) {
          addCandidate(t - safeWindowSec / 2);
        }
      }
    }

    if (candidateStarts.size === 0) {
      return null;
    }

    const candidates = [...candidateStarts].map((startSec) => ({
      startSec,
      durationSec: Math.min(safeWindowSec, Math.max(1, durationSeconds - startSec)),
      occupancyPct: speechOccupancyPctInWindow(speechSpans, startSec, safeWindowSec),
      timelineBias: durationSeconds > 0 ? startSec / durationSeconds : 0,
    }));

    const strongCandidate = candidates
      .filter((candidate) => candidate.occupancyPct >= 8)
      .sort((a, b) => b.occupancyPct - a.occupancyPct || a.timelineBias - b.timelineBias)[0];
    if (strongCandidate) {
      return {
        startSec: strongCandidate.startSec,
        durationSec: strongCandidate.durationSec,
        occupancyPct: strongCandidate.occupancyPct,
      };
    }

    const earliestSpeechStart = speechSpans[0]?.startSec ?? 0;
    const fallbackStart = clamp(earliestSpeechStart - safeWindowSec * 0.1, 0, maxStart);
    return {
      startSec: fallbackStart,
      durationSec: Math.min(safeWindowSec, Math.max(1, durationSeconds - fallbackStart)),
      occupancyPct: speechOccupancyPctInWindow(speechSpans, fallbackStart, safeWindowSec),
    };
  };

  const weightedPercentile = (entries: Array<{ value: number; weight: number }>, percent: number) => {
    const usable = entries
      .filter((entry) => Number.isFinite(entry.value) && Number.isFinite(entry.weight) && entry.weight > 0)
      .sort((a, b) => a.value - b.value);
    if (usable.length === 0) return null;
    const totalWeight = usable.reduce((sum, entry) => sum + entry.weight, 0);
    if (totalWeight <= 0) return usable[Math.floor(usable.length / 2)]?.value ?? null;
    const target = (clamp(percent, 0, 100) / 100) * totalWeight;
    let cumulative = 0;
    for (const entry of usable) {
      cumulative += entry.weight;
      if (cumulative >= target) return entry.value;
    }
    return usable.at(-1)?.value ?? null;
  };

  const weightedMetric = (
    analyses: Array<{ analysis: FileAnalysis; weight: number }>,
    getter: (analysis: FileAnalysis) => number | null,
    percent: number
  ) => {
    const entries: Array<{ value: number; weight: number }> = [];
    for (const item of analyses) {
      const value = getter(item.analysis);
      if (value === null || !Number.isFinite(value)) continue;
      entries.push({ value, weight: item.weight });
    }
    return weightedPercentile(entries, percent);
  };

  const runSilenceMapAnalysis = async (
    ffmpeg: FFmpeg,
    inputName: string,
    silenceDb: number,
    durationSeconds: number | null
  ) => {
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        inputName,
        "-af",
        `highpass=f=70,lowpass=f=6000,silencedetect=n=${silenceDb.toFixed(1)}dB:d=0.32`,
        "-f",
        "null",
        "-",
      ],
      "Silence map analysis"
    );
    const logText = resetLogBuffer();
    const silences = parseSilencedetectSpans(logText, durationSeconds);
    const speech = durationSeconds !== null ? deriveSpeechSpans(silences, durationSeconds) : [];
    return { silenceSpans: silences, speechSpans: speech };
  };

  const analyzeFileWindow = async (
    ffmpeg: FFmpeg,
    inputName: string,
    startSeconds: number,
    durationSeconds: number
  ): Promise<FileAnalysis> => {
    const analysis = createEmptyAnalysis();
    const windowStart = Math.max(0, startSeconds);
    const windowDur = Math.max(1, durationSeconds);
    const trimArgs = ["-ss", windowStart.toFixed(3), "-t", windowDur.toFixed(3)];

    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        ...trimArgs,
        "-i",
        inputName,
        "-af",
        "loudnorm=I=-24:TP=-2:LRA=7:print_format=json",
        "-f",
        "null",
        "-",
      ],
      "Smart match loudness analysis"
    );
    const loudData = parseLoudnormJson(resetLogBuffer());
    analysis.inputI = parseMaybeNumber(loudData?.input_i);
    analysis.inputLRA = parseMaybeNumber(loudData?.input_lra);
    analysis.inputTP = parseMaybeNumber(loudData?.input_tp);
    analysis.inputThresh = parseMaybeNumber(loudData?.input_thresh);

    const runRmsAnalysisWindow = async (bandFilter: string) => {
      resetLogBuffer();
      await execOrThrow(
        ffmpeg,
        [
          "-hide_banner",
          "-nostdin",
          "-threads",
          "1",
          ...trimArgs,
          "-i",
          inputName,
          "-af",
          `${bandFilter},astats=metadata=0:reset=0:measure_perchannel=0`,
          "-f",
          "null",
          "-",
        ],
        "RMS analysis"
      );
      return parseRmsFromAstats(resetLogBuffer());
    };

    analysis.lowRms = await runRmsAnalysisWindow("highpass=f=50,lowpass=f=220");
    analysis.midRms = await runRmsAnalysisWindow("highpass=f=300,lowpass=f=2400");
    analysis.highRms = await runRmsAnalysisWindow("highpass=f=2800,lowpass=f=9000");

    const analysisName = `${sanitizeBase(inputName)}_${Math.round(windowStart * 1000)}_${Math.round(windowDur * 1000)}_env.f32`;
    try {
      resetLogBuffer();
      await execOrThrow(
        ffmpeg,
        [
          "-hide_banner",
          "-nostdin",
          "-threads",
          "1",
          "-y",
          ...trimArgs,
          "-i",
          inputName,
          "-ac",
          "1",
          "-ar",
          `${ANALYSIS_SAMPLE_RATE}`,
          "-c:a",
          "pcm_f32le",
          "-f",
          "f32le",
          analysisName,
        ],
        "Envelope analysis render"
      );
      const bytes = await readVirtualFileBytes(ffmpeg, analysisName);
      const envelope = computeEnvelopeMetrics(toFloatSamples(bytes));
      analysis.noiseFloorDb = envelope.noiseFloorDb;
      analysis.pauseNoiseFloorDb = envelope.pauseNoiseFloorDb;
      analysis.nearSpeechNoiseFloorDb = envelope.nearSpeechNoiseFloorDb;
      analysis.speechThresholdDb = envelope.speechThresholdDb;
      analysis.noiseContrastDb = envelope.noiseContrastDb;
      analysis.dynamicRangeDb = envelope.dynamicRangeDb;
      analysis.reverbScore = envelope.reverbScore;
      analysis.echoScore = envelope.echoScore;
      analysis.roomScore = envelope.roomScore;
      analysis.echoDelayMs = envelope.echoDelayMs;
      analysis.analysisConfidence = envelope.analysisConfidence;
      analysis.drynessScore = envelope.drynessScore;
      analysis.instabilityScore = envelope.instabilityScore;
      analysis.lineSwingScore = envelope.lineSwingScore;
      analysis.pauseNoiseRisk = envelope.pauseNoiseRisk;
      analysis.compressionScore = envelope.compressionScore;
      analysis.overallRisk = envelope.overallRisk;
      analysis.clickScore = envelope.clickScore;
      analysis.speechDutyCyclePct = envelope.speechDutyCyclePct;
      analysis.speechSegmentCount = envelope.speechSegmentCount;
      analysis.medianSpeechRunMs = envelope.medianSpeechRunMs;
      analysis.longSilenceCount = envelope.longSilenceCount;
      analysis.onsetOvershootScore = envelope.onsetOvershootScore;
      analysis.midLineSagScore = envelope.midLineSagScore;
      analysis.endFadeRiskScore = envelope.endFadeRiskScore;
    } finally {
      await safeDeleteFile(ffmpeg, analysisName);
    }

    analysis.analysisWindowCount = 1;
    return analysis;
  };

  const aggregateWindowAnalyses = (
    baseAnalysis: FileAnalysis,
    windowAnalyses: Array<{ analysis: FileAnalysis; weight: number }>,
    speechMapStats?: {
      speechDutyCyclePct: number;
      speechSegmentCount: number;
      medianSpeechRunMs: number;
      longSilenceCount: number;
      longSparseModeEligible: boolean;
    }
  ): FileAnalysis => {
    if (windowAnalyses.length === 0) return baseAnalysis;

    const aggregated = createEmptyAnalysis();
    aggregated.inputI = weightedMetric(windowAnalyses, (a) => a.inputI, 50);
    aggregated.inputLRA = weightedMetric(windowAnalyses, (a) => a.inputLRA, 50);
    aggregated.inputTP = weightedMetric(windowAnalyses, (a) => a.inputTP, 50);
    aggregated.inputThresh = weightedMetric(windowAnalyses, (a) => a.inputThresh, 50);
    aggregated.lowRms = weightedMetric(windowAnalyses, (a) => a.lowRms, 50);
    aggregated.midRms = weightedMetric(windowAnalyses, (a) => a.midRms, 50);
    aggregated.highRms = weightedMetric(windowAnalyses, (a) => a.highRms, 50);
    aggregated.noiseFloorDb = weightedMetric(windowAnalyses, (a) => a.noiseFloorDb, 70);
    aggregated.pauseNoiseFloorDb = weightedMetric(windowAnalyses, (a) => a.pauseNoiseFloorDb, 70);
    aggregated.nearSpeechNoiseFloorDb = weightedMetric(windowAnalyses, (a) => a.nearSpeechNoiseFloorDb, 70);
    aggregated.speechThresholdDb = weightedMetric(windowAnalyses, (a) => a.speechThresholdDb, 60);
    aggregated.noiseContrastDb = weightedMetric(windowAnalyses, (a) => a.noiseContrastDb, 50);
    aggregated.dynamicRangeDb = weightedMetric(windowAnalyses, (a) => a.dynamicRangeDb, 50);
    aggregated.reverbScore = weightedMetric(windowAnalyses, (a) => a.reverbScore, 70);
    aggregated.echoScore = weightedMetric(windowAnalyses, (a) => a.echoScore, 70);
    aggregated.roomScore = weightedMetric(windowAnalyses, (a) => a.roomScore, 70);
    aggregated.echoDelayMs = weightedMetric(windowAnalyses, (a) => a.echoDelayMs, 50);
    aggregated.analysisConfidence = weightedMetric(windowAnalyses, (a) => a.analysisConfidence, 50);
    aggregated.drynessScore = weightedMetric(windowAnalyses, (a) => a.drynessScore, 50);
    aggregated.instabilityScore = weightedMetric(windowAnalyses, (a) => a.instabilityScore, 80);
    aggregated.lineSwingScore = weightedMetric(windowAnalyses, (a) => a.lineSwingScore, 82);
    aggregated.sentenceJumpScore = weightedMetric(windowAnalyses, (a) => a.sentenceJumpScore, 85);
    aggregated.breathSpikeRisk = weightedMetric(windowAnalyses, (a) => a.breathSpikeRisk, 85);
    aggregated.pauseNoiseRisk = weightedMetric(windowAnalyses, (a) => a.pauseNoiseRisk, 80);
    aggregated.compressionScore = weightedMetric(windowAnalyses, (a) => a.compressionScore, 70);
    aggregated.overallRisk = weightedMetric(windowAnalyses, (a) => a.overallRisk, 65);
    aggregated.clickScore = weightedMetric(windowAnalyses, (a) => a.clickScore, 70);
    aggregated.onsetOvershootScore = weightedMetric(windowAnalyses, (a) => a.onsetOvershootScore, 85);
    aggregated.midLineSagScore = weightedMetric(windowAnalyses, (a) => a.midLineSagScore, 80);
    aggregated.endFadeRiskScore = weightedMetric(windowAnalyses, (a) => a.endFadeRiskScore, 85);
    aggregated.speechDutyCyclePct =
      speechMapStats?.speechDutyCyclePct ?? weightedMetric(windowAnalyses, (a) => a.speechDutyCyclePct, 50);
    aggregated.speechSegmentCount =
      speechMapStats?.speechSegmentCount ?? weightedMetric(windowAnalyses, (a) => a.speechSegmentCount, 50);
    aggregated.medianSpeechRunMs =
      speechMapStats?.medianSpeechRunMs ?? weightedMetric(windowAnalyses, (a) => a.medianSpeechRunMs, 50);
    aggregated.longSilenceCount =
      speechMapStats?.longSilenceCount ?? weightedMetric(windowAnalyses, (a) => a.longSilenceCount, 50);
    aggregated.analysisWindowCount = windowAnalyses.length;
    aggregated.longSparseModeEligible = speechMapStats?.longSparseModeEligible ?? false;

    for (const key of Object.keys(baseAnalysis) as Array<keyof FileAnalysis>) {
      if (aggregated[key] !== null) continue;
      (aggregated as Record<string, number | boolean | null>)[key] =
        (baseAnalysis as Record<string, number | boolean | null>)[key];
    }

    return aggregated;
  };

  const analyzeFile = async (ffmpeg: FFmpeg, inputName: string): Promise<AnalysisResult> => {
    let durationSeconds: number | null = null;
    let recoveryInputBytes: Uint8Array | null = null;
    const ensureRecoveryInputBytes = async () => {
      if (recoveryInputBytes === null) {
        recoveryInputBytes = await readVirtualFileBytes(ffmpeg, inputName);
      }
      return recoveryInputBytes;
    };
    try {
      durationSeconds = await probeInputDurationSeconds(ffmpeg, inputName);
    } catch {
      durationSeconds = null;
    }

    const baseWindowDuration =
      durationSeconds !== null ? Math.min(ANALYSIS_SAMPLE_SECONDS, Math.max(10, durationSeconds)) : ANALYSIS_SAMPLE_SECONDS;
    let baseWindowStart = 0;
    let coarseSpeechMap:
      | { silenceDb: number; silenceSpans: SilenceSpan[]; speechSpans: SpeechSpan[] }
      | null = null;

    if (durationSeconds !== null && durationSeconds > baseWindowDuration + 1) {
      const coarseSilenceDb = -46;
      try {
        const coarseMap = await runSilenceMapAnalysis(ffmpeg, inputName, coarseSilenceDb, durationSeconds);
        coarseSpeechMap = {
          silenceDb: coarseSilenceDb,
          silenceSpans: coarseMap.silenceSpans,
          speechSpans: coarseMap.speechSpans,
        };
        const bootstrapWindow = selectSpeechAnchoredAnalysisWindow(
          coarseMap.speechSpans,
          durationSeconds,
          baseWindowDuration
        );
        if (bootstrapWindow && bootstrapWindow.occupancyPct > 0.1) {
          baseWindowStart = bootstrapWindow.startSec;
          if (baseWindowStart > 0.5) {
            appendLog(
              `[Analysis] ${sanitizeBase(inputName)}: speech-anchored bootstrap @ ${baseWindowStart.toFixed(
                1
              )}s (${bootstrapWindow.occupancyPct.toFixed(1)}% speech in ${bootstrapWindow.durationSec.toFixed(0)}s window).`
            );
          }
        } else {
          appendLog(
            `[Analysis] ${sanitizeBase(
              inputName
            )}: no reliable speech found in coarse map, using start-window bootstrap.`
          );
        }
      } catch (error) {
        appendLog(
          `[Analysis] Bootstrap speech-map fallback (${sanitizeBase(inputName)}): ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }
    }

    const baseAnalysis = await analyzeFileWindow(ffmpeg, inputName, baseWindowStart, baseWindowDuration);
    baseAnalysis.analysisWindowCount = 1;

    if (durationSeconds === null || durationSeconds < DISTRIBUTED_ANALYSIS_THRESHOLD_SECONDS) {
      baseAnalysis.longSparseModeEligible = false;
      return { analysis: baseAnalysis, ffmpeg };
    }

    const silenceDb = clamp(
      Math.max(baseAnalysis.nearSpeechNoiseFloorDb ?? -90, baseAnalysis.noiseFloorDb ?? -70) + 16,
      -46,
      -30
    );

    try {
      const useCoarseSpeechMap = coarseSpeechMap !== null && Math.abs(coarseSpeechMap.silenceDb - silenceDb) <= 1;
      let silenceMapResult: { silenceSpans: SilenceSpan[]; speechSpans: SpeechSpan[] };
      if (useCoarseSpeechMap && coarseSpeechMap) {
        silenceMapResult = {
          silenceSpans: coarseSpeechMap.silenceSpans,
          speechSpans: coarseSpeechMap.speechSpans,
        };
      } else {
        silenceMapResult = await runSilenceMapAnalysis(ffmpeg, inputName, silenceDb, durationSeconds);
      }
      const { silenceSpans, speechSpans } = silenceMapResult;
      const speechStats = computeSpeechMapStats(speechSpans, silenceSpans, durationSeconds);
      baseAnalysis.speechDutyCyclePct = speechStats.speechDutyCyclePct;
      baseAnalysis.speechSegmentCount = speechStats.speechSegmentCount;
      baseAnalysis.medianSpeechRunMs = speechStats.medianSpeechRunMs;
      baseAnalysis.longSilenceCount = speechStats.longSilenceCount;
      baseAnalysis.longSparseModeEligible = speechStats.longSparseModeEligible;

      const useDistributedCoverage =
        durationSeconds >= DISTRIBUTED_ANALYSIS_THRESHOLD_SECONDS || speechStats.longSparseModeEligible;
      if (!useDistributedCoverage) {
        return { analysis: baseAnalysis, ffmpeg };
      }

      const distributedWindowSec = speechStats.longSparseModeEligible
        ? LONG_SPARSE_ANALYSIS_WINDOW_SECONDS
        : DISTRIBUTED_ANALYSIS_WINDOW_SECONDS;
      const distributedWindowCount = speechStats.longSparseModeEligible
        ? LONG_SPARSE_ANALYSIS_WINDOW_TARGET_COUNT
        : Math.max(
            DISTRIBUTED_ANALYSIS_TARGET_COUNT,
            Math.min(
              LONG_SPARSE_ANALYSIS_WINDOW_TARGET_COUNT,
              Math.ceil(durationSeconds / Math.max(distributedWindowSec * 2.5, 1))
            )
          );
      const windows = selectDistributedAnalysisWindowsWithConfig(
        speechSpans,
        durationSeconds,
        distributedWindowSec,
        distributedWindowCount
      );
      appendLog(
        `[Analysis] ${sanitizeBase(inputName)}: distributed coverage on (speech-duty ${speechStats.speechDutyCyclePct.toFixed(
          1
        )}%, median-run ${(speechStats.medianSpeechRunMs / 1000).toFixed(1)}s, windows ${windows.length}${
          speechStats.longSparseModeEligible ? ", sparse-mode" : ""
        }).`
      );

      const windowAnalyses: Array<{ analysis: FileAnalysis; weight: number }> = [];
      for (const window of windows) {
        let windowCompleted = false;
        for (let attempt = 0; attempt < 2 && !windowCompleted; attempt += 1) {
          try {
            const windowAnalysis = await analyzeFileWindow(ffmpeg, inputName, window.startSec, window.durationSec);
            const speechCoverage = clamp((window.occupancyPct ?? windowAnalysis.speechDutyCyclePct ?? 0) / 100, 0, 1);
            const confidence = clamp(windowAnalysis.analysisConfidence ?? 0.25, 0, 1);
            const roomPenalty = clamp(1 - (windowAnalysis.roomScore ?? 0), 0, 1);
            const weight = clamp(speechCoverage * 0.55 + confidence * 0.35 + roomPenalty * 0.1, 0.05, 1);
            windowAnalyses.push({ analysis: windowAnalysis, weight });
            windowCompleted = true;
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            appendLog(
              `[Analysis] Window fallback (${sanitizeBase(inputName)} @ ${window.startSec.toFixed(1)}s${
                attempt > 0 ? ` retry ${attempt}` : ""
              }): ${errorMessage}`
            );
            if (attempt === 0 && shouldResetFfmpegForError(error)) {
              const inputBytes = await ensureRecoveryInputBytes();
              ffmpeg = await refreshFfmpeg(
                `analysis window retry on ${sanitizeBase(inputName)} @ ${window.startSec.toFixed(1)}s`
              );
              await ffmpeg.writeFile(inputName, inputBytes);
              continue;
            }
            break;
          }
        }
      }

      if (windowAnalyses.length > 0) {
        return {
          analysis: aggregateWindowAnalyses(baseAnalysis, windowAnalyses, speechStats),
          ffmpeg,
        };
      }
    } catch (error) {
      appendLog(
        `[Analysis] Sparse-map fallback (${inputName}): ${error instanceof Error ? error.message : String(error)}`
      );
    }

    return { analysis: baseAnalysis, ffmpeg };
  };

  const buildBatchReference = (analyses: FileAnalysis[]) => {
    const lowTilts: number[] = [];
    const highTilts: number[] = [];
    const lras: number[] = [];

    for (const analysis of analyses) {
      if (analysis.lowRms !== null && analysis.midRms !== null) {
        lowTilts.push(analysis.lowRms - analysis.midRms);
      }
      if (analysis.highRms !== null && analysis.midRms !== null) {
        highTilts.push(analysis.highRms - analysis.midRms);
      }
      if (analysis.inputLRA !== null) {
        lras.push(analysis.inputLRA);
      }
    }

    const lowTilt = robustMedian(lowTilts);
    const highTilt = robustMedian(highTilts);
    const lra = robustMedian(lras);

    if (lowTilt === null && highTilt === null && lra === null) return null;

    return {
      lowTilt: lowTilt ?? -11,
      highTilt: highTilt ?? -13,
      lra: lra ?? 6,
    } satisfies BatchReference;
  };

  const buildAdaptiveProfile = (analysis: FileAnalysis | undefined, reference: BatchReference | null) => {
    const needsAdaptiveProfile =
      smartMatchConfig.tone > 0 || smartMatchConfig.dynamics > 0 || roomCleanup || sceneBlend;
    if (!needsAdaptiveProfile || !analysis) return null;

    const smartToneEnabled = smartMatchConfig.tone > 0;
    const smartDynamicsEnabled = smartMatchConfig.dynamics > 0;

    const referenceLowTilt = reference?.lowTilt ?? -11;
    const referenceHighTilt = reference?.highTilt ?? -13;
    const referenceLra = reference?.lra ?? 6;

    const lowTilt =
      analysis.lowRms !== null && analysis.midRms !== null
        ? analysis.lowRms - analysis.midRms
        : referenceLowTilt;
    const highTilt =
      analysis.highRms !== null && analysis.midRms !== null
        ? analysis.highRms - analysis.midRms
        : referenceHighTilt;
    const lowTiltDiff = lowTilt - referenceLowTilt;
    const highTiltDiff = highTilt - referenceHighTilt;

    const toneFactor = smartToneEnabled ? SMART_MATCH_PRESETS.Gentle.tone : 0;
    const dynamicsFactor = smartDynamicsEnabled ? SMART_MATCH_PRESETS.Gentle.dynamics : 0;

    const highpassHz = Math.round(clamp(80 + lowTiltDiff * 2.2 * toneFactor, 65, 105));
    const lowMidGainDb = clamp(-2 - lowTiltDiff * 0.28 * toneFactor, -3.6, 1.2);

    let presenceGainDb = clamp(-highTiltDiff * 0.45 * toneFactor, -2.2, 1.8);
    let airGainDb = clamp(-highTiltDiff * 0.25 * toneFactor, -1.4, 1.0);

    const lra = analysis.inputLRA ?? referenceLra;
    const lraDiff = lra - referenceLra;
    const compressorRatioOffset = clamp(lraDiff * 0.07 * dynamicsFactor, -0.35, 0.45);
    const compressorThresholdOffsetDb = clamp(lraDiff * 0.6 * dynamicsFactor, -1.5, 1.5);

    const measuredNoiseFloor = Math.max(analysis.noiseFloorDb ?? -70, analysis.nearSpeechNoiseFloorDb ?? -90);
    const measuredSpeechThreshold =
      analysis.speechThresholdDb ?? clamp(measuredNoiseFloor + 10.5, -58, -26);
    let noiseRisk: NoiseRisk = "low";
    if (analysis.noiseFloorDb !== null || analysis.nearSpeechNoiseFloorDb !== null) {
      noiseRisk = measuredNoiseFloor > -52 ? "high" : measuredNoiseFloor > -62 ? "medium" : "low";
    } else if (analysis.inputThresh !== null) {
      // Only use loudnorm threshold when envelope analysis is unavailable.
      noiseRisk = analysis.inputThresh > -33 ? "high" : analysis.inputThresh > -38 ? "medium" : "low";
    }
    if (noiseRisk === "low" && measuredSpeechThreshold > -44) {
      noiseRisk = "medium";
    }
    if (noiseRisk === "medium" && measuredSpeechThreshold > -40) {
      noiseRisk = "high";
    }

    const inputTP = analysis.inputTP ?? -9;
    const hotPeakFactor = clamp((inputTP + 9) / 7, 0, 1);
    const brightFactor = clamp((highTiltDiff + 1.8) / 4.5, 0, 1);
    const dynamicFactor = clamp((lra - 5.5) / 7, 0, 1);
    const emotionProtection = clamp(
      (hotPeakFactor * 0.48 + dynamicFactor * 0.4) * dynamicsFactor,
      0,
      0.82
    );
    const levelingNeed = clamp(
      (dynamicFactor * 0.72 + Math.max(0, lraDiff) / 8) * dynamicsFactor,
      0,
      1
    );
    const emotionalHarshnessCutDb = clamp(
      (hotPeakFactor * 0.95 + brightFactor * 0.7) * toneFactor,
      0,
      1.6
    );
    const topEndHarshnessCutDb = clamp(emotionalHarshnessCutDb * 0.75, 0, 1.2);

    const analysisConfidence = analysis.analysisConfidence ?? 0.25;
    const rawRoomScore = analysis.roomScore ?? 0;
    const confidenceScaledRoom = rawRoomScore * clamp(0.75 + analysisConfidence * 0.25, 0.75, 1);
    let roomRisk = classifyRoomRisk(confidenceScaledRoom);
    if (analysisConfidence < 0.35) {
      roomRisk = downgradeRoomRisk(roomRisk);
    }

    if (noiseRisk !== "low" || roomRisk !== "low") {
      const positiveTrim =
        roomRisk === "high" ? 0.2 : roomRisk === "medium" ? 0.45 : noiseRisk === "high" ? 0.35 : 0.7;
      if (presenceGainDb > 0) presenceGainDb *= positiveTrim;
      if (airGainDb > 0) airGainDb *= positiveTrim;
    }

    const echoScore = analysis.echoScore ?? 0;
    const roomCleanupEnabled =
      roomCleanup && (analysisConfidence >= 0.4 || echoScore >= 0.58 || roomRisk !== "low");
    const useDenoise = false;
    const instabilityScore = clamp(analysis.instabilityScore ?? 0, 0, 1);
    const clickScore = clamp(analysis.clickScore ?? 0, 0, 1);
    const lineSwingScore = clamp(analysis.lineSwingScore ?? 0, 0, 1);
    const sentenceJumpScore = clamp(analysis.sentenceJumpScore ?? 0, 0, 1);
    const breathSpikeRisk = clamp(analysis.breathSpikeRisk ?? 0, 0, 1);
    const pauseNoiseRisk = clamp(
      analysis.pauseNoiseRisk ??
        (analysis.pauseNoiseFloorDb !== null
          ? clamp((analysis.pauseNoiseFloorDb + 62) / 18, 0, 1)
          : noiseRisk === "high"
            ? 0.85
            : noiseRisk === "medium"
              ? 0.45
              : 0.12),
      0,
      1
    );
    const onsetOvershootScore = clamp(analysis.onsetOvershootScore ?? 0, 0, 1);
    const midLineSagScore = clamp(analysis.midLineSagScore ?? 0, 0, 1);
    const endFadeRiskScore = clamp(analysis.endFadeRiskScore ?? 0, 0, 1);
    const hasDistributedCoverage = (analysis.analysisWindowCount ?? 0) > 1;
    const lineContinuityRisk = clamp(
      instabilityScore * 0.24 +
        lineSwingScore * 0.22 +
        sentenceJumpScore * 0.22 +
        onsetOvershootScore * 0.14 +
        midLineSagScore * 0.1 +
        endFadeRiskScore * 0.08,
      0,
      1
    );
    const preserveEndings =
      (noiseRisk === "low" &&
        (endFadeRiskScore >= 0.45 || instabilityScore >= 0.62 || lineSwingScore >= 0.42)) ||
      (midLineSagScore >= 0.52 && echoScore < 0.9);
    const strictEndingProtection =
      preserveEndings &&
      noiseRisk === "low" &&
      (endFadeRiskScore >= 0.78 ||
        lineContinuityRisk >= 0.58 ||
        lineSwingScore >= 0.55 ||
        !!analysis.longSparseModeEligible);
    const onsetTameStrength = clamp(
      Math.max(onsetOvershootScore, breathSpikeRisk * 0.85) * (noiseRisk === "low" ? 1 : 0.72),
      0,
      1
    );
    const breathTameStrength = clamp(
      breathSpikeRisk * (noiseRisk === "high" ? 0.55 : noiseRisk === "medium" ? 0.78 : 1),
      0,
      1
    );
    const sagRecoveryStrength = clamp(midLineSagScore * (noiseRisk === "high" ? 0.5 : 1), 0, 1);
    const disableDynaThresholdForStability =
      noiseRisk === "low" && (preserveEndings || midLineSagScore >= 0.45 || lineSwingScore >= 0.45);
    const preferSinglePassContinuity =
      noiseRisk === "low" && (sentenceJumpScore >= 0.34 || (lineContinuityRisk >= 0.46 && breathSpikeRisk >= 0.42));
    const useSpeechAlignedSegmentation =
      !preferSinglePassContinuity &&
      (!!analysis.longSparseModeEligible || hasDistributedCoverage) &&
      (instabilityScore >= 0.58 ||
        onsetOvershootScore >= 0.45 ||
        midLineSagScore >= 0.45 ||
        breathSpikeRisk >= 0.45 ||
        endFadeRiskScore >= 0.42 ||
        lineSwingScore >= 0.4);
    const useSpeechPauseSegmentation =
      !preferSinglePassContinuity &&
      (pauseNoiseRisk >= 0.34 ||
        lineContinuityRisk >= 0.44 ||
        (hasDistributedCoverage &&
          (pauseNoiseRisk >= 0.24 || lineContinuityRisk >= 0.32 || sentenceJumpScore >= 0.24)) ||
        preserveEndings ||
        (analysis.pauseNoiseFloorDb ?? -120) > -60);

    const forceTailGateForEcho =
      roomCleanupEnabled && roomRisk === "high" && echoScore >= 0.62 && !preserveEndings;
    const useTailGate =
      roomCleanupEnabled &&
      !preserveEndings &&
      (forceTailGateForEcho ||
        (analysisConfidence >= 0.52 && (roomRisk === "high" || (roomRisk === "medium" && echoScore >= 0.5))));
    const denoiseStrength = 0;
    const tailGateStrength = !useTailGate
      ? 0
      : roomRisk === "high"
        ? clamp(0.09 + echoScore * 0.1 + analysisConfidence * 0.06, 0.09, 0.22)
        : clamp(0.06 + echoScore * 0.08, 0.06, 0.14);
    const echoNotchCutDb = roomCleanupEnabled
      ? clamp(
          echoScore * (roomRisk === "high" ? 1.02 : roomRisk === "medium" ? 0.68 : 0.42) +
            (roomRisk === "high" ? 0.12 : 0),
          0,
          1.45
        )
      : 0;

    const baseDynaTrim = noiseGuard ? (noiseRisk === "high" ? 3 : noiseRisk === "medium" ? 2 : 0) : 0;
    const roomDynaTrim = roomRisk === "high" ? 1.4 : roomRisk === "medium" ? 0.7 : 0;
    const instabilityAssist =
      instabilityScore * (noiseRisk === "low" ? 0.9 : noiseRisk === "medium" ? 0.4 : 0.15) +
      lineSwingScore * (noiseRisk === "low" ? 0.55 : 0.28);
    const dynaTrim = Math.max(0, baseDynaTrim + roomDynaTrim - instabilityAssist);
    const clickTameStrength = clamp(
      clickScore * (noiseRisk === "high" ? 1 : noiseRisk === "medium" ? 0.88 : 0.78) * 0.72,
      0,
      1
    );

    const dryness = analysis.drynessScore ?? clamp(1 - confidenceScaledRoom, 0, 1);
    const blendRiskDamp = roomRisk === "high" ? 0.03 : roomRisk === "medium" ? 0.2 : 1;
    const blendEchoDamp = clamp(1 - echoScore * 0.85, 0.08, 1);
    const blendNoiseDamp = noiseRisk === "high" ? 0.22 : noiseRisk === "medium" ? 0.55 : 1;
    const blendInstabilityDamp = instabilityScore >= 0.7 ? 0.65 : 1;
    const blendConfidenceScale = clamp(0.35 + analysisConfidence * 0.65, 0.35, 1);
    const blendBase = clamp(0.022 + dryness * 0.022, 0.016, 0.045);
    let blendAmount = sceneBlend
      ? blendBase * blendRiskDamp * blendConfidenceScale * blendEchoDamp * blendNoiseDamp * blendInstabilityDamp
      : 0;
    if (roomRisk === "high" || echoScore >= 0.72) {
      blendAmount = Math.min(blendAmount, 0.0018);
    }
    const blendIndoorGain = clamp(blendAmount * 0.62, 0, 0.07);
    const blendOutdoorGain = clamp(blendAmount * 0.42, 0, 0.055);
    const blendIndoorDelayMs = Math.round(clamp(24 + (1 - dryness) * 8, 22, 36));
    const blendOutdoorDelayMs = Math.round(clamp(52 + (1 - dryness) * 18, 48, 74));

    return {
      highpassHz,
      lowMidGainDb,
      presenceGainDb,
      airGainDb,
      emotionalHarshnessCutDb,
      topEndHarshnessCutDb,
      levelingNeed,
      emotionProtection,
      compressorRatioOffset,
      compressorThresholdOffsetDb,
      dynaTrim,
      floorGuardFilter: noiseRisk === "high" ? FLOOR_GUARD_STRONG : FLOOR_GUARD,
      noiseRisk,
      noiseFloorDb: measuredNoiseFloor,
      pauseNoiseRisk,
      speechThresholdDb: measuredSpeechThreshold,
      roomRisk,
      useDenoise,
      denoiseStrength,
      useTailGate,
      tailGateStrength,
      echoNotchCutDb,
      instabilityScore,
      clickScore,
      clickTameStrength,
      lineSwingScore,
      sentenceJumpScore,
      breathSpikeRisk,
      breathTameStrength,
      lineContinuityRisk,
      preserveEndings,
      onsetTameStrength,
      sagRecoveryStrength,
      disableDynaThresholdForStability,
      strictEndingProtection,
      preferSinglePassContinuity,
      segmentMatchTargetI: analysis.inputI,
      useSpeechAlignedSegmentation,
      useSpeechPauseSegmentation,
      segmentTargetSec: SPEECH_ALIGNED_SEGMENT_TARGET_SECONDS,
      segmentMaxSec: SPEECH_ALIGNED_SEGMENT_MAX_SECONDS,
      blendIndoorGain,
      blendOutdoorGain,
      blendIndoorDelayMs,
      blendOutdoorDelayMs,
    } satisfies AdaptiveProfile;
  };

  const buildTailGateFilter = (strength: number) => {
    const thresholdDb = clamp(-58 + strength * 3.4, -58, -54.5);
    const threshold = fromDb(thresholdDb);
    const ratio = clamp(1.01 + strength * 0.5, 1.01, 1.22);
    const range = clamp(0.9 - strength * 0.16, 0.72, 0.9);
    const attack = Math.round(clamp(24 - strength * 6, 18, 26));
    const release = Math.round(clamp(760 - strength * 160, 520, 760));
    const makeup = 1;
    return `agate=mode=downward:threshold=${threshold.toFixed(5)}:ratio=${ratio.toFixed(
      2
    )}:range=${range.toFixed(3)}:attack=${attack}:release=${release}:makeup=${makeup.toFixed(
      2
    )}:detection=rms:link=average`;
  };

  const buildClickTamerFilter = (strength: number) => {
    const attack = Math.round(clamp(2 + strength * 4, 2, 6));
    const release = Math.round(clamp(20 + strength * 30, 20, 52));
    const limit = clamp(-3.6 + strength * 0.7, -3.6, -2.8);
    return `alimiter=limit=${limit.toFixed(
      1
    )}dB:attack=${attack}:release=${release}:level=disabled`;
  };

  const buildOnsetTamerFilter = (strength: number) => {
    const thresholdDb = clamp(-19 - strength * 4, -23, -19);
    const ratio = clamp(1.08 + strength * 0.5, 1.08, 1.45);
    const attack = Math.round(clamp(2 + strength * 3, 2, 5));
    const release = Math.round(clamp(32 + strength * 40, 32, 72));
    const mix = clamp(0.18 + strength * 0.22, 0.18, 0.4);
    return `acompressor=threshold=${thresholdDb.toFixed(
      1
    )}dB:ratio=${ratio.toFixed(2)}:attack=${attack}:release=${release}:mix=${mix.toFixed(
      2
    )}:detection=peak`;
  };

  const buildBreathSpikeTamerFilter = (strength: number) => {
    const thresholdDb = clamp(-34 + strength * 6, -34, -28);
    const ratio = clamp(1.18 + strength * 0.82, 1.18, 2.0);
    const attack = Math.round(clamp(1 + strength * 2, 1, 3));
    const release = Math.round(clamp(88 + strength * 80, 88, 168));
    const mix = clamp(0.18 + strength * 0.18, 0.18, 0.36);
    return `acompressor=threshold=${thresholdDb.toFixed(
      1
    )}dB:ratio=${ratio.toFixed(2)}:attack=${attack}:release=${release}:mix=${mix.toFixed(
      2
    )}:detection=peak`;
  };

  const buildAdaptiveNoiseReductionFilter = (noiseRisk: NoiseRisk, noiseFloorDb: number | null) => {
    if (noiseRisk === "low") return null;

    const estimatedFloor = noiseFloorDb ?? (noiseRisk === "high" ? -49 : -55);
    const severeNoise = noiseRisk === "high" && estimatedFloor > -46;

    // Use spectral denoise in-browser for stability and voice-preserving behavior.
    const nf = severeNoise
      ? clamp(estimatedFloor + 3.2, -46, -34)
      : noiseRisk === "high"
        ? clamp(estimatedFloor + 4.5, -50, -37)
        : clamp(estimatedFloor + 5.2, -56, -40);
    const nr = severeNoise ? 12 : noiseRisk === "high" ? 10 : 7;
    const ad = severeNoise ? 0.55 : noiseRisk === "high" ? 0.42 : 0.32;
    const gs = severeNoise ? 12 : noiseRisk === "high" ? 10 : 8;
    return `afftdn=nf=${nf.toFixed(1)}:nr=${nr}:tn=1:ad=${ad.toFixed(2)}:gs=${gs}`;
  };

  type MixRenderOptions = {
    disableRoomCleanup?: boolean;
    disableAdaptiveNoiseReduction?: boolean;
    minimalStabilityChain?: boolean;
    disableLimiter?: boolean;
    segmentBoundaryPadInMs?: number;
    segmentBoundaryPadOutMs?: number;
    trimSegmentPadMs?: number;
    segmentMode?: "fixed" | "speech-aligned" | "speech-pause";
    candidateVariant?: "cinematic-stable" | "continuity-safe" | "pause-safe";
    skipSpeechSegmentation?: boolean;
    forceEndingProtection?: boolean;
  };

  const resolveAdaptiveNoiseReductionFilter = (
    profile: AdaptiveProfile | null,
    options?: MixRenderOptions
  ) => {
    if (!noiseGuard || !profile) return null;
    if (options?.minimalStabilityChain || options?.disableAdaptiveNoiseReduction) return null;
    return buildAdaptiveNoiseReductionFilter(profile.noiseRisk, profile.noiseFloorDb);
  };

  const buildMixFilter = (profile: AdaptiveProfile | null, options?: MixRenderOptions) => {
    const filters: string[] = [];
    const levelerSettings = LEVELER_PRESETS[leveler];
    const consistency = LEVELER_CONSISTENCY[leveler];
    const dyn = levelerSettings.dyna;
    const minimalStabilityChain = options?.minimalStabilityChain === true;
    const candidateVariant = options?.candidateVariant ?? "cinematic-stable";
    const continuitySafeMode = candidateVariant === "continuity-safe";
    const pauseSafeMode = candidateVariant === "pause-safe";
    const roomCleanupEnabled = roomCleanup && !options?.disableRoomCleanup && !minimalStabilityChain;
    const adaptiveNoiseReductionFilter = resolveAdaptiveNoiseReductionFilter(profile, options);
    const useAdaptiveNoiseReduction = adaptiveNoiseReductionFilter !== null;
    const instabilityScore = profile?.instabilityScore ?? 0;
    const clickTameStrength = profile?.clickTameStrength ?? 0;
    const onsetTameStrength = profile?.onsetTameStrength ?? 0;
    const sagRecoveryStrength = profile?.sagRecoveryStrength ?? 0;
    const disableDynaThresholdForStability = profile?.disableDynaThresholdForStability ?? false;
    const lineContinuityRisk = profile?.lineContinuityRisk ?? 0;
    const strictEndingProtection = options?.forceEndingProtection || (profile?.strictEndingProtection ?? false);
    const lineSwingScore = profile?.lineSwingScore ?? 0;
    const sentenceJumpScore = profile?.sentenceJumpScore ?? 0;
    const breathSpikeRisk = profile?.breathSpikeRisk ?? 0;
    const breathTameStrength = profile?.breathTameStrength ?? 0;
    const pauseNoiseRisk = profile?.pauseNoiseRisk ?? 0;
    const useClickTamer =
      !minimalStabilityChain &&
      clickTameStrength >= (continuitySafeMode ? 0.38 : pauseSafeMode ? 0.42 : 0.46);
    const useOnsetTamer =
      !minimalStabilityChain &&
      onsetTameStrength >= (continuitySafeMode ? 0.24 : 0.35);
    const useBreathSpikeTamer =
      !minimalStabilityChain &&
      breathControl !== "Off" &&
      breathTameStrength >= (continuitySafeMode ? 0.18 : 0.24);

    if (eqCleanup) {
      const highpassHz = profile?.highpassHz ?? 80;
      const lowMidGainDb = profile?.lowMidGainDb ?? -2;
      filters.push(`highpass=f=${highpassHz}`);
      filters.push(`equalizer=f=250:width_type=q:width=1.0:g=${lowMidGainDb.toFixed(2)}`);
      if (useAdaptiveNoiseReduction && adaptiveNoiseReductionFilter) {
        // Run denoise before dynamic stages so levelers do not lift room bed/hiss.
        filters.push(adaptiveNoiseReductionFilter);
      }
    }
    if (!eqCleanup && useAdaptiveNoiseReduction && adaptiveNoiseReductionFilter) {
      filters.push(adaptiveNoiseReductionFilter);
    }
    if (useOnsetTamer) {
      filters.push(buildOnsetTamerFilter(onsetTameStrength));
    }
    if (useClickTamer) {
      filters.push(buildClickTamerFilter(clickTameStrength));
    }
    if (useBreathSpikeTamer) {
      filters.push(buildBreathSpikeTamerFilter(breathTameStrength));
    }

    if (dyn) {
      let dynaF: number = dyn.f;
      let dynaG: number = noiseGuard ? Math.max(3, dyn.g - 1) : dyn.g;
      let dynaM: number = noiseGuard ? Math.max(3, dyn.m - 1) : dyn.m;
      let dynaThresholdAmp = 0;

      if (!minimalStabilityChain) {
        if (profile) {
          const adaptiveLift = profile.levelingNeed * 1.8;
          const emotionRelax = profile.emotionProtection * 1.2;
          dynaG = Math.max(3, dynaG + adaptiveLift - profile.dynaTrim - emotionRelax);
          dynaM = Math.max(3, dynaM + adaptiveLift - profile.dynaTrim - emotionRelax);

          if (profile.instabilityScore >= 0.35) {
            if (profile.noiseRisk === "low") {
              // Unstable but clean takes need faster ride response, not a wider/slow window.
              const instabilityNorm = clamp((profile.instabilityScore - 0.35) / 0.65, 0, 1);
              dynaF = Math.round(clamp(dynaF - instabilityNorm * 80, 161, 261));
              dynaG += profile.instabilityScore * 1.8;
              dynaM += profile.instabilityScore * 1.4;
              if (noiseGuard && profile.noiseRisk !== "low") {
                const gateDb = clamp((profile.speechThresholdDb ?? -46) - 8.5, -58, -44);
                dynaThresholdAmp = Math.max(dynaThresholdAmp, fromDb(gateDb));
              }
            } else {
              dynaF = Math.max(dynaF, Math.round(261 + profile.instabilityScore * 90));
            }
          }

          // Noisy takes need slower and lower lift to avoid raising room noise in pauses.
          if (noiseGuard) {
            if (profile.noiseRisk === "high" || (profile.noiseFloorDb ?? -70) > -46) {
              dynaF = Math.max(dynaF, 281);
              dynaG = Math.min(dynaG, 3);
              dynaM = Math.min(dynaM, 3);
              const gateDb = clamp((profile.noiseFloorDb ?? -46) + 7.2, -54, -34);
              dynaThresholdAmp = fromDb(gateDb);
            } else if (profile.noiseRisk === "medium" || (profile.noiseFloorDb ?? -70) > -52) {
              dynaF = Math.max(dynaF, 241);
              dynaG = Math.min(dynaG, 4);
              dynaM = Math.min(dynaM, 5);
              const gateDb = clamp((profile.noiseFloorDb ?? -50) + 6.0, -56, -36);
              dynaThresholdAmp = fromDb(gateDb);
            }
          }

          if (profile.instabilityScore >= 0.62 && profile.noiseRisk !== "high") {
            const instabilityNorm = clamp((profile.instabilityScore - 0.62) / 0.38, 0, 1);
            dynaF = Math.round(clamp(dynaF - instabilityNorm * 48, 201, 261));
            dynaG = Math.min(7.5, dynaG + instabilityNorm * 1.2);
            dynaM = Math.min(9.5, dynaM + instabilityNorm * 1.0);
            if (noiseGuard && profile.noiseRisk !== "low") {
              const speechGateDb = clamp((profile.speechThresholdDb ?? -46) - 8.2, -58, -43);
              dynaThresholdAmp = Math.max(dynaThresholdAmp, fromDb(speechGateDb));
            }
          }

          if (sagRecoveryStrength > 0.12) {
            dynaF = Math.round(
              clamp(
                dynaF - sagRecoveryStrength * (profile.noiseRisk === "high" ? 24 : profile.noiseRisk === "medium" ? 44 : 64),
                181,
                281
              )
            );
            dynaG = Math.min(9.5, dynaG + sagRecoveryStrength * (profile.noiseRisk === "high" ? 0.6 : 1.3));
            dynaM = Math.min(10.5, dynaM + sagRecoveryStrength * (profile.noiseRisk === "high" ? 0.5 : 1.1));
          }

          if (lineSwingScore >= 0.28 && profile.noiseRisk !== "high") {
            const swingNorm = clamp((lineSwingScore - 0.28) / 0.72, 0, 1);
            dynaF = Math.round(clamp(dynaF - swingNorm * (continuitySafeMode ? 72 : 48), 171, 251));
            dynaG = Math.min(10.5, dynaG + swingNorm * (continuitySafeMode ? 1.2 : 0.7));
            dynaM = Math.min(11.5, dynaM + swingNorm * (continuitySafeMode ? 1.0 : 0.55));
            if (continuitySafeMode) {
              dynaThresholdAmp = 0;
            }
          }

          if (sentenceJumpScore >= 0.28 && profile.noiseRisk === "low") {
            const jumpNorm = clamp((sentenceJumpScore - 0.28) / 0.72, 0, 1);
            dynaF = Math.round(clamp(dynaF - jumpNorm * (continuitySafeMode ? 86 : 52), 161, 241));
            dynaG = Math.min(10.5, dynaG + jumpNorm * (continuitySafeMode ? 1.4 : 0.8));
            dynaM = Math.min(11.5, dynaM + jumpNorm * (continuitySafeMode ? 1.2 : 0.6));
            if (continuitySafeMode || profile.preferSinglePassContinuity) {
              dynaThresholdAmp = 0;
            }
          }

          if (breathSpikeRisk >= 0.24) {
            const breathGateDb = clamp(
              (profile.speechThresholdDb ?? -46) - (continuitySafeMode ? 6.2 : 7.4) + breathSpikeRisk * 2.2,
              -56,
              -40
            );
            dynaThresholdAmp = Math.max(dynaThresholdAmp, fromDb(breathGateDb));
            dynaG = Math.min(dynaG, continuitySafeMode ? 6.5 : 5.5);
            dynaM = Math.min(dynaM, continuitySafeMode ? 7.5 : 6.5);
          }

          if (!strictEndingProtection && lineContinuityRisk >= 0.32 && profile.noiseRisk === "low") {
            // Improve phrase continuity without forcing compressor to do gain-riding work.
            const continuityNorm = clamp((lineContinuityRisk - 0.32) / 0.68, 0, 1);
            dynaF = Math.round(clamp(dynaF - continuityNorm * (profile.useSpeechAlignedSegmentation ? 34 : 22), 181, 241));
            dynaG = Math.min(9.5, dynaG + continuityNorm * 0.7);
            dynaM = Math.min(10.5, dynaM + continuityNorm * 0.5);
          }

          if (strictEndingProtection && profile.noiseRisk === "low") {
            // Protect sentence endings and close-gap continuity on sparse dialogue takes.
            dynaF = Math.round(clamp(dynaF - (profile.useSpeechAlignedSegmentation ? 42 : 26), 181, 241));
            dynaG = clamp(dynaG + 0.7, 3, 8.5);
            dynaM = clamp(dynaM + 0.5, 3, 9.5);
            dynaThresholdAmp = 0;
          }

          if (pauseSafeMode && pauseNoiseRisk >= 0.32) {
            const pauseNorm = clamp((pauseNoiseRisk - 0.32) / 0.68, 0, 1);
            dynaF = Math.max(dynaF, Math.round(231 + pauseNorm * 50));
            dynaG = Math.min(dynaG, 5.5 - pauseNorm * 0.8);
            dynaM = Math.min(dynaM, 6.5 - pauseNorm * 0.8);
            const pauseGateDb = clamp((profile.speechThresholdDb ?? -46) - 6.8, -56, -40);
            dynaThresholdAmp = Math.max(dynaThresholdAmp, fromDb(pauseGateDb));
          }
        }
      } else {
        // Stability-safe fallback keeps mandatory leveler but removes adaptive modifiers.
        dynaF = dyn.f;
        dynaG = dyn.g;
        dynaM = dyn.m;
      }

      if (!minimalStabilityChain && disableDynaThresholdForStability && profile?.noiseRisk === "low") {
        dynaThresholdAmp = 0;
      }

      const dynaGInt = toOddInt(dynaG, 3, 301);
      const dynaMValue = toOddInt(dynaM, 3, 301);
      const dynaThreshold =
        dynaThresholdAmp > 0 ? `:t=${clamp(dynaThresholdAmp, fromDb(-60), fromDb(-36)).toFixed(5)}` : "";
      filters.push(`dynaudnorm=f=${dynaF}:g=${dynaGInt}:m=${dynaMValue}${dynaThreshold}`);
    }

    const continuityBreathProtect =
      !minimalStabilityChain &&
      (strictEndingProtection ||
        (profile?.preserveEndings ?? false) ||
        lineContinuityRisk >= 0.52 ||
        lineSwingScore >= 0.42 ||
        breathSpikeRisk >= 0.38 ||
        continuitySafeMode);
    const breath =
      breathControl === "Off"
        ? null
        : pauseSafeMode && pauseNoiseRisk >= 0.38
          ? null
        : continuityBreathProtect
          ? BREATH_COMPAND_SAFE[breathControl === "Medium" ? "Medium" : "Light"]
          : BREATH_COMPAND[breathControl];
    const suppressContinuityGate =
      continuitySafeMode || strictEndingProtection || lineContinuityRisk >= 0.58 || lineSwingScore >= 0.48;
    const roomGateFilter =
      roomCleanupEnabled && !suppressContinuityGate && (profile?.useTailGate ?? false)
        ? buildTailGateFilter(profile?.tailGateStrength ?? 0.12)
        : null;
    const useRoomGate = roomGateFilter !== null;
    const preferFloorGuard =
      floorGuard &&
      (pauseSafeMode ||
        pauseNoiseRisk >= 0.42 ||
        profile?.noiseRisk === "high" ||
        (noiseGuard && profile?.noiseRisk === "medium"));
    const useFloorGuard = !useRoomGate && floorGuard && (breath === null || preferFloorGuard);
    const useBreathCompand = !useRoomGate && breath !== null && !useFloorGuard;

    if (!minimalStabilityChain && useRoomGate && roomGateFilter) {
      filters.push(roomGateFilter);
    }
    if (!minimalStabilityChain && useBreathCompand) {
      filters.push(breath);
    }
    if (!minimalStabilityChain && useFloorGuard) {
      filters.push(profile?.floorGuardFilter ?? FLOOR_GUARD);
    }

    // Merge static harshness softening with smart-match tone offsets to avoid
    // competing EQ moves on the same bands.
    const basePresenceCut = softenHarshness ? -2.0 : 0;
    const baseAirCut = softenHarshness ? -1.1 : 0;
    const harshPresenceCut = profile?.emotionalHarshnessCutDb ?? 0;
    const harshAirCut = profile?.topEndHarshnessCutDb ?? 0;
    const netPresenceGain = clamp(
      basePresenceCut + (profile?.presenceGainDb ?? 0) - harshPresenceCut,
      -4.0,
      0.7
    );
    const netAirGain = clamp(baseAirCut + (profile?.airGainDb ?? 0) - harshAirCut, -2.7, 0.45);

    if (!minimalStabilityChain && Math.abs(netPresenceGain) >= 0.2) {
      filters.push(`equalizer=f=3500:width_type=q:width=1.15:g=${netPresenceGain.toFixed(2)}`);
    }
    if (!minimalStabilityChain && Math.abs(netAirGain) >= 0.2) {
      filters.push(`equalizer=f=8000:width_type=q:width=0.75:g=${netAirGain.toFixed(2)}`);
    }
    if (!minimalStabilityChain && (profile?.topEndHarshnessCutDb ?? 0) >= 0.45) {
      const topShelfCut = clamp(-0.35 - (profile?.topEndHarshnessCutDb ?? 0) * 0.55, -1.1, -0.35);
      filters.push(`equalizer=f=11200:width_type=q:width=0.7:g=${topShelfCut.toFixed(2)}`);
    }
    if (!minimalStabilityChain && roomCleanupEnabled && (profile?.echoNotchCutDb ?? 0) >= 0.25) {
      const echoCut = clamp(profile?.echoNotchCutDb ?? 0, 0.25, 1.25);
      const notch1 = -clamp(echoCut, 0.25, 1.25);
      filters.push(`equalizer=f=2450:width_type=q:width=1.35:g=${notch1.toFixed(2)}`);
      if (echoCut >= 0.55) {
        const notch2 = -clamp(echoCut * 0.62, 0.3, 0.9);
        filters.push(`equalizer=f=1280:width_type=q:width=1.0:g=${notch2.toFixed(2)}`);
      }
      if (echoCut >= 0.9) {
        const notch3 = -clamp(echoCut * 0.45, 0.25, 0.7);
        filters.push(`equalizer=f=3620:width_type=q:width=1.6:g=${notch3.toFixed(2)}`);
      }
    }
    if (!minimalStabilityChain && roomCleanupEnabled && profile?.roomRisk === "high") {
      const roomCutFactor = clamp((profile?.echoNotchCutDb ?? 0.6) / 1.45, 0.25, 1);
      const roomCutLow = -clamp(0.45 + roomCutFactor * 0.55, 0.45, 1.05);
      const roomCutMid = -clamp(0.35 + roomCutFactor * 0.65, 0.35, 1.15);
      filters.push(`equalizer=f=460:width_type=q:width=0.95:g=${roomCutLow.toFixed(2)}`);
      filters.push(`equalizer=f=1650:width_type=q:width=1.2:g=${roomCutMid.toFixed(2)}`);
      if ((profile?.echoNotchCutDb ?? 0) >= 0.95) {
        const roomCutUpperMid = -clamp(0.25 + roomCutFactor * 0.45, 0.25, 0.8);
        filters.push(`equalizer=f=2850:width_type=q:width=1.5:g=${roomCutUpperMid.toFixed(2)}`);
      }
    }

    const thresholdBase = parseFloat(levelerSettings.compressor.threshold.replace("dB", ""));
    const ratioBase = parseFloat(levelerSettings.compressor.ratio);

    let thresholdAdjust = minimalStabilityChain ? 0 : (profile?.compressorThresholdOffsetDb ?? 0);
    let ratioAdjust = minimalStabilityChain ? 0 : (profile?.compressorRatioOffset ?? 0);
    const levelingNeed = minimalStabilityChain ? 0 : (profile?.levelingNeed ?? 0);
    const emotionProtection = minimalStabilityChain ? 0 : (profile?.emotionProtection ?? 0);

    // Keep upstream processors from sounding overcontrolled.
    if (dyn) {
      thresholdAdjust += 0.25;
      ratioAdjust -= 0.08;
    }
    if (useBreathCompand || useFloorGuard) {
      thresholdAdjust += 0.15;
      ratioAdjust -= 0.05;
    }
    if (useRoomGate) {
      thresholdAdjust += 0.22;
      ratioAdjust -= 0.08;
    }
    if (profile?.roomRisk === "high") {
      thresholdAdjust += 0.6;
      ratioAdjust -= 0.24;
    } else if (profile?.roomRisk === "medium") {
      thresholdAdjust += 0.32;
      ratioAdjust -= 0.13;
    }
    const echoPressure = clamp((profile?.echoNotchCutDb ?? 0) / 1.25, 0, 1);
    thresholdAdjust += echoPressure * 0.25;
    ratioAdjust -= echoPressure * 0.12;
    const instabilityCompressorRelax =
      instabilityScore * (profile?.noiseRisk === "high" ? 0.9 : profile?.noiseRisk === "medium" ? 0.75 : 0.65);
    thresholdAdjust += instabilityCompressorRelax * 0.9;
    ratioAdjust -= instabilityCompressorRelax * 0.35;
    thresholdAdjust += lineContinuityRisk * 0.35;
    ratioAdjust -= lineContinuityRisk * 0.14;
    thresholdAdjust += sentenceJumpScore * 0.32;
    ratioAdjust -= sentenceJumpScore * 0.12;
    thresholdAdjust += onsetTameStrength * 0.25;
    ratioAdjust -= onsetTameStrength * 0.12;
    thresholdAdjust += breathTameStrength * 0.18;
    ratioAdjust -= breathTameStrength * 0.08;
    if (strictEndingProtection) {
      thresholdAdjust += 0.45;
      ratioAdjust -= 0.18;
    }
    if (continuitySafeMode) {
      thresholdAdjust += 0.38;
      ratioAdjust -= 0.17;
    }
    if (pauseSafeMode) {
      thresholdAdjust += 0.22 + pauseNoiseRisk * 0.22;
      ratioAdjust -= 0.08 + pauseNoiseRisk * 0.08;
    }

    // Compressor should not create phrase ramps after dynaudnorm. If onset taming is active,
    // make the downstream compressor less grabby on the first word and let it recover faster.
    const continuityAttackBias =
      lineContinuityRisk * 0.2 +
      lineSwingScore * 0.45 +
      sentenceJumpScore * 0.35 +
      sagRecoveryStrength * 0.35 +
      onsetTameStrength * 2.6 +
      (continuitySafeMode ? 1.25 : 0);
    const continuityReleaseBias =
      -lineContinuityRisk * 18 -
      lineSwingScore * 20 -
      sentenceJumpScore * 24 -
      sagRecoveryStrength * 20 -
      onsetTameStrength * 24 -
      (continuitySafeMode ? 18 : 0) +
      (pauseSafeMode ? 10 : 0);

    // Smarter consistency: tighten when needed, but protect emotional peaks.
    const thresholdTighten = consistency * (0.55 + levelingNeed * 0.75);
    const threshold = clamp(
      thresholdBase + thresholdAdjust - thresholdTighten + emotionProtection * 0.65,
      -32.5,
      -17.2
    );
    const ratio = clamp(
      ratioBase + ratioAdjust + consistency * 0.22 + levelingNeed * 0.3 - emotionProtection * 0.35,
      1.55,
      3.0
    );
    const roomRelax = profile?.roomRisk === "high" ? 1 : profile?.roomRisk === "medium" ? 0.45 : 0;
    let attack = Math.round(
      clamp(
        24 -
          consistency * 8 +
          emotionProtection * 8 +
          roomRelax * 4 +
          echoPressure * 2 +
          instabilityCompressorRelax * 3 +
          continuityAttackBias,
        14,
        36
      )
    );
    let release = Math.round(
      clamp(
        170 -
          consistency * 45 +
          emotionProtection * 75 +
          roomRelax * 40 +
          echoPressure * 30 +
          instabilityCompressorRelax * 55 +
          continuityReleaseBias +
          (strictEndingProtection
            ? 95 - onsetTameStrength * 30
            : profile?.preserveEndings
              ? 45 - onsetTameStrength * 12
              : 0),
        95,
        420
      )
    );
    if (!minimalStabilityChain && useOnsetTamer) {
      attack = Math.max(attack, 19);
      release = Math.min(release, strictEndingProtection ? 320 : 285);
    }
    if (!minimalStabilityChain && lineContinuityRisk >= 0.45) {
      release = Math.min(release, strictEndingProtection ? 335 : 295);
    }
    if (!minimalStabilityChain && continuitySafeMode) {
      attack = Math.max(attack, 20);
      release = Math.min(release, strictEndingProtection ? 300 : 255);
    }

    const compMix = clamp(
      0.9 +
        levelingNeed * 0.07 -
        emotionProtection * 0.24 -
        roomRelax * 0.08 -
        echoPressure * 0.04 -
        instabilityCompressorRelax * 0.12 -
        lineContinuityRisk * 0.08 -
        lineSwingScore * 0.08 -
        sentenceJumpScore * 0.07 -
        onsetTameStrength * 0.05 -
        breathTameStrength * 0.04 -
        (strictEndingProtection ? 0.06 : 0) -
        (continuitySafeMode ? 0.06 : 0) -
        (pauseSafeMode ? 0.04 : 0),
      0.58,
      0.95
    );

    filters.push(
      `acompressor=threshold=${threshold.toFixed(1)}dB:ratio=${ratio.toFixed(2)}:attack=${attack}:release=${release}:mix=${compMix.toFixed(2)}:detection=rms`
    );
    if (!options?.disableLimiter) {
      filters.push(LIMITER_FILTER);
    }

    return filters.join(",");
  };

  const buildBlendFilter = (profile: AdaptiveProfile | null) => {
    const indoorGain = profile?.blendIndoorGain ?? 0.015;
    const outdoorGain = profile?.blendOutdoorGain ?? 0.01;
    const indoorDelay = Math.round(profile?.blendIndoorDelayMs ?? 28);
    const outdoorDelay = Math.round(profile?.blendOutdoorDelayMs ?? 58);

    const wetTotal = indoorGain + outdoorGain;
    const dryGain = clamp(1 - wetTotal * 0.55, 0.93, 1);
    const wetGateThreshold = clamp(0.00045 + wetTotal * 0.028, 0.00055, 0.0024);
    const wetGateRatio = clamp(1.16 + wetTotal * 8, 1.16, 1.34);
    const wetGateRange = clamp(0.86 - wetTotal * 2.6, 0.68, 0.86);

    return [
      "asplit=3[dry][ind_src][out_src]",
      `[ind_src]adelay=${indoorDelay}:all=1,highpass=f=280,lowpass=f=4600,volume=${indoorGain.toFixed(
        4
      )}[ind]`,
      `[out_src]adelay=${outdoorDelay}:all=1,highpass=f=220,lowpass=f=3000,volume=${outdoorGain.toFixed(
        4
      )}[out]`,
      `[ind][out]amix=inputs=2:normalize=0,agate=mode=downward:threshold=${wetGateThreshold.toFixed(
        5
      )}:ratio=${wetGateRatio.toFixed(2)}:range=${wetGateRange.toFixed(
        3
      )}:attack=10:release=180:makeup=1.00:detection=rms:link=average[wet]`,
      `[dry]volume=${dryGain.toFixed(4)}[dryv]`,
      "[dryv][wet]amix=inputs=2:normalize=0,alimiter=limit=-2dB:level=disabled",
    ].join(";");
  };

  const writeOutput = async (
    ffmpeg: FFmpeg,
    name: string,
    kind: OutputEntry["kind"],
    variant: OutputEntry["variant"]
  ): Promise<OutputEntry> => {
    const bytes = await readVirtualFileBytes(ffmpeg, name);
    const blob = new Blob([bytes], { type: "audio/wav" });
    return {
      name,
      url: URL.createObjectURL(blob),
      size: blob.size,
      kind,
      variant,
    };
  };

  const analyzeIntegratedLoudness = async (ffmpeg: FFmpeg, inputName: string) => {
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        inputName,
        "-af",
        "loudnorm=I=-24:TP=-2:LRA=7:print_format=json",
        "-f",
        "null",
        "-",
      ],
      "Segment loudness analysis"
    );
    const data = parseLoudnormJson(resetLogBuffer());
    return {
      inputI: parseMaybeNumber(data?.input_i),
      inputTP: parseMaybeNumber(data?.input_tp),
    };
  };

  const maybeMatchSpeechSegmentGain = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    targetI: number | null,
    maxDeltaDb: number
  ) => {
    if (targetI === null || !Number.isFinite(targetI)) {
      return { matched: false, gainDb: 0 };
    }

    const loudness = await analyzeIntegratedLoudness(ffmpeg, inputName);
    if (loudness.inputI === null || !Number.isFinite(loudness.inputI)) {
      return { matched: false, gainDb: 0 };
    }

    let gainDb = clamp(targetI - loudness.inputI, -maxDeltaDb, maxDeltaDb);
    if (loudness.inputTP !== null && Number.isFinite(loudness.inputTP)) {
      const peakGuardGain = clamp(-2.25 - loudness.inputTP, -maxDeltaDb, maxDeltaDb);
      gainDb = Math.min(gainDb, peakGuardGain);
    }
    if (Math.abs(gainDb) < SEGMENT_GAIN_MATCH_MIN_DELTA_DB) {
      return { matched: false, gainDb };
    }

    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
        "-i",
        inputName,
        "-af",
        `volume=${gainDb.toFixed(2)}dB,${LIMITER_FILTER}`,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        outputName,
      ],
      "Segment gain match"
    );

    return { matched: true, gainDb };
  };

  const runMixReady = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    profile: AdaptiveProfile | null,
    options?: MixRenderOptions
  ) => {
    const filterChain = buildMixFilter(profile, options);
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
        "-i",
        inputName,
        "-af",
        filterChain,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        outputName,
      ],
      "Mix-ready render"
    );
  };

  const probeInputDurationSeconds = async (ffmpeg: FFmpeg, inputName: string) => {
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-i",
        inputName,
        "-t",
        "0.1",
        "-f",
        "null",
        "-",
      ],
      "Duration probe"
    );
    const logText = resetLogBuffer();
    return parseDurationSeconds(logText);
  };

  const runMixReadySegmented = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    profile: AdaptiveProfile | null,
    durationSeconds: number,
    options?: MixRenderOptions
  ) => {
    if (durationSeconds < MIX_SEGMENT_MIN_DURATION_SECONDS) {
      throw new Error("Segmented render skipped (input too short).");
    }
    const segmentCount = Math.ceil(durationSeconds / MIX_SEGMENT_SECONDS);
    if (segmentCount < 2) {
      throw new Error("Segmented render skipped (single segment).");
    }

    const filterChain = buildMixFilter(profile, options);
    const tempBase = sanitizeBase(outputName);
    const segmentNames: string[] = [];
    const concatListName = `${tempBase}_segments.txt`;

    try {
      for (let index = 0; index < segmentCount; index += 1) {
        const start = index * MIX_SEGMENT_SECONDS;
        const remaining = durationSeconds - start;
        const span = Math.min(MIX_SEGMENT_SECONDS, Math.max(remaining, 0));
        if (span <= 0.01) break;
        const segmentName = `${tempBase}_seg_${index + 1}.wav`;
        segmentNames.push(segmentName);

        resetLogBuffer();
        await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
        "-ss",
        start.toFixed(3),
            "-t",
            span.toFixed(3),
            "-i",
            inputName,
            "-af",
            filterChain,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            segmentName,
          ],
          `Segment mix-ready render ${index + 1}/${segmentCount}`
        );
      }

      if (segmentNames.length < 2) {
        throw new Error("Segmented render produced insufficient segments.");
      }

      const concatList = `${segmentNames.map((name) => `file '${name}'`).join("\n")}\n`;
      await ffmpeg.writeFile(concatListName, new TextEncoder().encode(concatList));

      resetLogBuffer();
      await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-y",
        "-f",
        "concat",
          "-safe",
          "0",
          "-i",
          concatListName,
          "-c",
          "copy",
          outputName,
        ],
        "Segment concat render"
      );
    } finally {
      await safeDeleteFile(ffmpeg, concatListName);
      for (const segmentName of segmentNames) {
        await safeDeleteFile(ffmpeg, segmentName);
      }
    }
  };

  const buildSpeechAlignedRenderSegments = (
    speechSpans: SpeechSpan[],
    silenceSpans: SilenceSpan[],
    durationSeconds: number,
    profile: AdaptiveProfile | null
  ): RenderSegment[] => {
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return [];
    const targetSec = profile?.segmentTargetSec ?? SPEECH_ALIGNED_SEGMENT_TARGET_SECONDS;
    const maxSec = profile?.segmentMaxSec ?? SPEECH_ALIGNED_SEGMENT_MAX_SECONDS;
    const segments: RenderSegment[] = [];

    const isWithinLongSilence = (timeSec: number) =>
      silenceSpans.some((span) => span.endSec - span.startSec >= 0.5 && timeSec >= span.startSec && timeSec <= span.endSec);

    const speechOccupancy = (startSec: number, endSec: number) => {
      let sum = 0;
      for (const span of speechSpans) {
        if (span.endSec <= startSec) continue;
        if (span.startSec >= endSec) break;
        sum += overlapSeconds(startSec, endSec, span.startSec, span.endSec);
      }
      return sum / Math.max(endSec - startSec, 1e-6);
    };

    let cursor = 0;
    while (cursor < durationSeconds - 0.02) {
      const minCut = Math.min(durationSeconds, cursor + targetSec * 0.55);
      const idealCut = Math.min(durationSeconds, cursor + targetSec);
      const hardCut = Math.min(durationSeconds, cursor + maxSec);

      let cutSec = hardCut;
      let bestScore = Number.POSITIVE_INFINITY;
      for (const silence of silenceSpans) {
        const silenceLen = silence.endSec - silence.startSec;
        if (silenceLen < 0.4) continue;
        const center = (silence.startSec + silence.endSec) / 2;
        if (center < minCut || center > hardCut) continue;
        const score = Math.abs(center - idealCut);
        if (score < bestScore) {
          bestScore = score;
          cutSec = center;
        }
      }

      if (cutSec <= cursor + 0.25) {
        cutSec = Math.min(durationSeconds, cursor + targetSec);
      }

      const startSec = cursor;
      const endSec = Math.max(cursor + 0.1, cutSec);
      const occupancy = speechOccupancy(startSec, endSec);
      const process = occupancy >= 0.01;
      const trimInMs = 0;
      const trimOutMs = 0;
      segments.push({ startSec, endSec, process, trimInMs, trimOutMs });
      cursor = endSec;
    }

    const processedIndexes = segments
      .map((segment, index) => (segment.process ? index : -1))
      .filter((index) => index >= 0);
    const lastProcessedIndex = processedIndexes.length > 0 ? processedIndexes[processedIndexes.length - 1] : -1;

    for (let i = 0; i < segments.length; i += 1) {
      const segment = segments[i];
      if (!segment.process) continue;
      const startOnLongSilence = isWithinLongSilence(segment.startSec);
      const endOnLongSilence = isWithinLongSilence(segment.endSec);
      const strictEndingProtection = profile?.strictEndingProtection ?? false;
      const forceEndingProtection = strictEndingProtection || endOnLongSilence || i === lastProcessedIndex;
      const padInMs = forceEndingProtection
        ? SPEECH_ALIGNED_SEGMENT_PAD_IN_MS + 80
        : SPEECH_ALIGNED_SEGMENT_PAD_IN_MS;
      const padOutMs = forceEndingProtection
        ? SPEECH_ALIGNED_SEGMENT_PAD_OUT_MS + 260
        : SPEECH_ALIGNED_SEGMENT_PAD_OUT_MS;
      segment.forceEndingProtection = forceEndingProtection;
      segment.trimInMs = startOnLongSilence ? 0 : padInMs;
      // Even when cut lands in detected silence, keep a small tail context for conservative
      // look-ahead filters in case the detector marked a low-level word ending as silence.
      segment.trimOutMs = endOnLongSilence
        ? forceEndingProtection
          ? SPEECH_ENDING_EXTRA_PAD_OUT_MS
          : 140
        : padOutMs;
    }

    return segments;
  };

  const buildSilenceSegmentFilter = (profile: AdaptiveProfile | null, options?: MixRenderOptions) => {
    const filters: string[] = [];
    const candidateVariant = options?.candidateVariant ?? "cinematic-stable";
    const pauseSafeMode = candidateVariant === "pause-safe";
    if (eqCleanup) {
      filters.push(`highpass=f=${profile?.highpassHz ?? 80}`);
    }
    if (noiseGuard && profile && (profile.pauseNoiseRisk >= 0.42 || (pauseSafeMode && profile.noiseRisk !== "low"))) {
      const adaptiveNoiseReduction = buildAdaptiveNoiseReductionFilter(profile.noiseRisk, profile.noiseFloorDb);
      if (adaptiveNoiseReduction) {
        filters.push(adaptiveNoiseReduction);
      }
    }
    if (floorGuard && profile && (profile.noiseRisk === "high" || profile.pauseNoiseRisk >= 0.38 || pauseSafeMode)) {
      filters.push(profile.floorGuardFilter);
    }
    if (roomCleanup && profile && profile.roomRisk !== "low" && !profile.preserveEndings) {
      filters.push(buildTailGateFilter((profile.tailGateStrength ?? 0.1) * (pauseSafeMode ? 0.85 : 0.55)));
    }
    if (filters.length === 0) return null;
    return filters.join(",");
  };

  const runMixReadySpeechAlignedSegmented = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    profile: AdaptiveProfile | null,
    durationSeconds: number,
    silenceSpans: SilenceSpan[],
    speechSpans: SpeechSpan[],
    options?: MixRenderOptions
  ) => {
    const segments = buildSpeechAlignedRenderSegments(speechSpans, silenceSpans, durationSeconds, profile);
    if (segments.length < 2) {
      throw new Error("Speech-aligned segmentation skipped (insufficient segments).");
    }

    const segmentMode = options?.segmentMode ?? "speech-aligned";
    const tempBase = sanitizeBase(outputName);
    const concatListName = `${tempBase}_speech_segments.txt`;
    const segmentNames: string[] = [];
    const cleanupNames: string[] = [];
    const enableSegmentGainMatch =
      !!profile &&
      profile.segmentMatchTargetI !== null &&
      (profile.preferSinglePassContinuity ||
        profile.sentenceJumpScore >= 0.28 ||
        profile.breathSpikeRisk >= 0.34 ||
        durationSeconds >= DISTRIBUTED_ANALYSIS_THRESHOLD_SECONDS);

    try {
      for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        const readStart = Math.max(0, segment.startSec - segment.trimInMs / 1000);
        const readEnd = Math.min(durationSeconds, segment.endSec + segment.trimOutMs / 1000);
        const readSpan = Math.max(0.05, readEnd - readStart);
        const trimStartSec = Math.max(0, (segment.startSec - readStart) + ((options?.trimSegmentPadMs ?? 0) / 1000));
        const trimEndSec = Math.max(
          trimStartSec + 0.02,
          (segment.endSec - readStart) - ((options?.trimSegmentPadMs ?? 0) / 1000)
        );
        const segmentName = `${tempBase}_speech_seg_${index + 1}.wav`;
        cleanupNames.push(segmentName);
        const segmentOptions = {
          ...options,
          segmentMode,
          forceEndingProtection: segment.process && (segment.forceEndingProtection ?? false),
        } satisfies MixRenderOptions;
        const processedFilter = buildMixFilter(profile, segmentOptions);
        const silenceFilter = buildSilenceSegmentFilter(profile, segmentOptions);
        const baseFilter = segment.process ? processedFilter : silenceFilter;
        const trimFilter = `atrim=start=${trimStartSec.toFixed(3)}:end=${trimEndSec.toFixed(3)},asetpts=N/SR/TB`;
        const filterChain = baseFilter ? `${baseFilter},${trimFilter}` : trimFilter;

        resetLogBuffer();
        await execOrThrow(
          ffmpeg,
          [
            "-hide_banner",
            "-nostdin",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-y",
            "-ss",
            readStart.toFixed(3),
            "-t",
            readSpan.toFixed(3),
            "-i",
            inputName,
            "-af",
            filterChain,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            segmentName,
          ],
          `Speech-aligned segment render ${index + 1}/${segments.length}`
        );

        let concatName = segmentName;
        if (segment.process && enableSegmentGainMatch) {
          const matchedName = `${tempBase}_speech_seg_${index + 1}_matched.wav`;
          const maxGainDeltaDb =
            profile?.preferSinglePassContinuity || (segment.forceEndingProtection ?? false)
              ? SEGMENT_GAIN_MATCH_MAX_DB
              : 1.25;
          const segmentGainMatch = await maybeMatchSpeechSegmentGain(
            ffmpeg,
            segmentName,
            matchedName,
            profile?.segmentMatchTargetI ?? null,
            maxGainDeltaDb
          );
          if (segmentGainMatch.matched) {
            cleanupNames.push(matchedName);
            concatName = matchedName;
            appendLog(
              `[SegmentMatch] ${sanitizeBase(outputName)} seg ${index + 1}: ${
                segmentGainMatch.gainDb >= 0 ? "+" : ""
              }${segmentGainMatch.gainDb.toFixed(2)} dB`
            );
          }
        }
        segmentNames.push(concatName);
      }

      const concatList = `${segmentNames.map((name) => `file '${name}'`).join("\n")}\n`;
      await ffmpeg.writeFile(concatListName, new TextEncoder().encode(concatList));

      resetLogBuffer();
      await execOrThrow(
        ffmpeg,
        [
          "-hide_banner",
          "-nostdin",
          "-threads",
          "1",
          "-y",
          "-f",
          "concat",
          "-safe",
          "0",
          "-i",
          concatListName,
          "-c",
          "copy",
          outputName,
        ],
        "Speech-aligned segment concat render"
      );

      return segments.length;
    } finally {
      await safeDeleteFile(ffmpeg, concatListName);
      for (const name of cleanupNames) {
        await safeDeleteFile(ffmpeg, name);
      }
    }
  };

  const runBlendMixReady = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    profile: AdaptiveProfile | null
  ) => {
    const filterChain = buildBlendFilter(profile);
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
        "-i",
        inputName,
        "-af",
        filterChain,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        outputName,
      ],
      "Blend mix-ready render"
    );
  };

  const runOnePassLoudnorm = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    cfg: NonNullable<(typeof LOUDNESS_PRESETS)[keyof typeof LOUDNESS_PRESETS]>
  ) => {
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
        "-i",
        inputName,
        "-af",
        `loudnorm=I=${cfg.I}:TP=${cfg.TP}:LRA=${cfg.LRA}:print_format=summary`,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        outputName,
      ],
      "One-pass loudnorm"
    );
  };

  const runLoudnorm = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    cfg: NonNullable<(typeof LOUDNESS_PRESETS)[keyof typeof LOUDNESS_PRESETS]>
  ) => {
    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        inputName,
        "-af",
        `loudnorm=I=${cfg.I}:TP=${cfg.TP}:LRA=${cfg.LRA}:print_format=json`,
        "-f",
        "null",
        "-",
      ],
      "Loudnorm analysis"
    );

    const logText = resetLogBuffer();
    const data = parseLoudnormJson(logText);

    const measuredI = parseMaybeNumber(data?.input_i);
    const measuredTP = parseMaybeNumber(data?.input_tp);
    const measuredLRA = parseMaybeNumber(data?.input_lra);
    const measuredThresh = parseMaybeNumber(data?.input_thresh);
    const offset = parseMaybeNumber(data?.target_offset);

    if (
      measuredI === null ||
      measuredTP === null ||
      measuredLRA === null ||
      measuredThresh === null ||
      offset === null
    ) {
      appendLog("Loudnorm pass1 failed; using one-pass loudnorm.");
      await runOnePassLoudnorm(ffmpeg, inputName, outputName, cfg);
      return;
    }

    resetLogBuffer();
    await execOrThrow(
      ffmpeg,
      [
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
        "-i",
        inputName,
        "-af",
        `loudnorm=I=${cfg.I}:TP=${cfg.TP}:LRA=${cfg.LRA}:measured_I=${measuredI}:measured_TP=${measuredTP}:measured_LRA=${measuredLRA}:measured_thresh=${measuredThresh}:offset=${offset}:linear=true:print_format=summary`,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        outputName,
      ],
      "Loudnorm render"
    );
  };

  const safeDeleteFile = async (ffmpeg: FFmpeg, name: string) => {
    try {
      await ffmpeg.deleteFile(name);
    } catch {
      // Ignore cleanup failures from missing temp files.
    }
  };

  const buildJobs = (inputFiles: File[]) => {
    const seen = new Map<string, number>();
    return inputFiles.map((file, index) => {
      const baseRaw = sanitizeBase(file.name) || `input_${index + 1}`;
      const count = seen.get(baseRaw) ?? 0;
      seen.set(baseRaw, count + 1);
      const base = count === 0 ? baseRaw : `${baseRaw}_${count + 1}`;
      return {
        file,
        base,
        inputName: `${base}_input.wav`,
        mixName: `${base}_mixready.wav`,
        blendMixName: `${base}_blend_mixready.wav`,
      } satisfies JobEntry;
    });
  };

  const writeJobInput = async (ffmpeg: FFmpeg, job: JobEntry) => {
    // Use a fresh buffer every write; FFmpeg worker postMessage can detach transferred ArrayBuffers.
    await ffmpeg.writeFile(job.inputName, await fetchFile(job.file));
  };

  type SpeechRenderPlan = {
    durationSeconds: number;
    silenceSpans: SilenceSpan[];
    speechSpans: SpeechSpan[];
    mode: "speech-aligned" | "speech-pause";
  };

  type CandidateVariant = NonNullable<MixRenderOptions["candidateVariant"]>;

  type CandidateScore = {
    stability: number;
    pause: number;
    compression: number;
    echo: number;
    total: number;
  };

  const formatCandidateVariant = (variant: CandidateVariant) =>
    variant === "cinematic-stable"
      ? "cinematic-stable"
      : variant === "continuity-safe"
        ? "continuity-safe"
        : "pause-safe";

  const buildMixCandidateVariants = (profile: AdaptiveProfile | null): CandidateVariant[] => {
    const variants: CandidateVariant[] = ["cinematic-stable", "continuity-safe"];
    if ((profile?.pauseNoiseRisk ?? 0) >= 0.32 || profile?.noiseRisk === "medium" || profile?.noiseRisk === "high") {
      variants.push("pause-safe");
    }
    return variants;
  };

  const buildCandidateScore = (analysis: FileAnalysis | null): CandidateScore => {
    if (!analysis) {
      return {
        stability: Number.POSITIVE_INFINITY,
        pause: Number.POSITIVE_INFINITY,
        compression: Number.POSITIVE_INFINITY,
        echo: Number.POSITIVE_INFINITY,
        total: Number.POSITIVE_INFINITY,
      };
    }

    const instability = clamp(analysis.instabilityScore ?? 1, 0, 1);
    const lineSwing = clamp(analysis.lineSwingScore ?? 1, 0, 1);
    const sentenceJump = clamp(analysis.sentenceJumpScore ?? 1, 0, 1);
    const breathSpike = clamp(analysis.breathSpikeRisk ?? 1, 0, 1);
    const onset = clamp(analysis.onsetOvershootScore ?? 1, 0, 1);
    const sag = clamp(analysis.midLineSagScore ?? 1, 0, 1);
    const endFade = clamp(analysis.endFadeRiskScore ?? 1, 0, 1);
    const pauseRisk = clamp(analysis.pauseNoiseRisk ?? 1, 0, 1);
    const compression = clamp(analysis.compressionScore ?? 1, 0, 1);
    const echo = clamp(analysis.echoScore ?? 1, 0, 1);

    const stability =
      instability * 0.24 +
      lineSwing * 0.16 +
      sentenceJump * 0.2 +
      breathSpike * 0.12 +
      onset * 0.12 +
      sag * 0.1 +
      endFade * 0.06;
    const pause =
      pauseRisk * 0.58 +
      breathSpike * 0.24 +
      clamp(((analysis.pauseNoiseFloorDb ?? -120) + 62) / 18, 0, 1) * 0.18;
    const total = stability * 1000 + pause * 100 + compression * 10 + echo;
    return { stability, pause, compression, echo, total };
  };

  const compareCandidateScores = (left: CandidateScore, right: CandidateScore) => {
    if (left.stability !== right.stability) return left.stability - right.stability;
    if (left.pause !== right.pause) return left.pause - right.pause;
    if (left.compression !== right.compression) return left.compression - right.compression;
    if (left.echo !== right.echo) return left.echo - right.echo;
    return left.total - right.total;
  };

  const summarizeCandidateScore = (score: CandidateScore) =>
    `stability ${(score.stability * 100).toFixed(0)} / pause ${(score.pause * 100).toFixed(0)} / compression ${(
      score.compression * 100
    ).toFixed(0)} / echo ${(score.echo * 100).toFixed(0)}`;

  const countUsableSpeechPauseBoundaries = (silenceSpans: SilenceSpan[]) => {
    let usableCount = 0;
    let usableSeconds = 0;
    for (const silence of silenceSpans) {
      const silenceLen = silence.endSec - silence.startSec;
      if (silenceLen < 0.22) continue;
      usableCount += 1;
      usableSeconds += silenceLen;
    }
    return { usableCount, usableSeconds };
  };

  const buildSpeechRenderPlan = async (
    ffmpeg: FFmpeg,
    inputName: string,
    profile: AdaptiveProfile | null
  ): Promise<SpeechRenderPlan | null> => {
    if (!profile || (!profile.useSpeechPauseSegmentation && !profile.useSpeechAlignedSegmentation)) {
      return null;
    }

    const durationSeconds = await probeInputDurationSeconds(ffmpeg, inputName);
    if (durationSeconds === null || !Number.isFinite(durationSeconds) || durationSeconds < 20) {
      return null;
    }

    const silenceDb = clamp(
      Math.max(profile.noiseFloorDb ?? -70, profile.speechThresholdDb ?? -48, -70) + 14,
      -48,
      -28
    );
    const silenceMap = await runSilenceMapAnalysis(ffmpeg, inputName, silenceDb, durationSeconds);
    const usable = countUsableSpeechPauseBoundaries(silenceMap.silenceSpans);
    if (silenceMap.speechSpans.length === 0 || usable.usableCount === 0 || usable.usableSeconds < 0.45) {
      return null;
    }

    return {
      durationSeconds,
      silenceSpans: silenceMap.silenceSpans,
      speechSpans: silenceMap.speechSpans,
      mode: profile.useSpeechPauseSegmentation ? "speech-pause" : "speech-aligned",
    };
  };

  const renderMixReadyWithFallbacks = async (
    ffmpeg: FFmpeg,
    job: JobEntry,
    outputName: string,
    profile: AdaptiveProfile | null,
    fileIndex: number,
    totalFiles: number,
    stageLabel: string,
    options?: MixRenderOptions,
    speechRenderPlan?: SpeechRenderPlan | null
  ) => {
    const hasRoomFilters = roomCleanup && !!profile && (profile.useTailGate || profile.echoNotchCutDb >= 0.25);
    const hasAdaptiveNoiseReduction = noiseGuard && !!profile && profile.noiseRisk !== "low";
    const fallbackStrategies: Array<{ label: string; options?: MixRenderOptions }> = [{ label: "primary chain" }];
    if (hasRoomFilters) {
      fallbackStrategies.push({
        label: "room cleanup bypass",
        options: { disableRoomCleanup: true },
      });
    }
    if (hasAdaptiveNoiseReduction) {
      fallbackStrategies.push({
        label: "adaptive-NR bypass",
        options: { disableAdaptiveNoiseReduction: true },
      });
    }
    if (hasRoomFilters && hasAdaptiveNoiseReduction) {
      fallbackStrategies.push({
        label: "room cleanup + adaptive-NR bypass",
        options: { disableRoomCleanup: true, disableAdaptiveNoiseReduction: true },
      });
    }
    fallbackStrategies.push({
      label: "stability-safe chain",
      options: {
        disableRoomCleanup: true,
        disableAdaptiveNoiseReduction: true,
        minimalStabilityChain: true,
      },
    });

    let fallbackApplied: string | null = null;
    let lastMixError: unknown = null;
    let mixRendered = false;
    let inputDurationSeconds: number | null | undefined = speechRenderPlan?.durationSeconds;
    let speechAlignedSegmentCountUsed: number | null = null;

    const ensureInputDuration = async () => {
      if (inputDurationSeconds !== undefined) return inputDurationSeconds;
      try {
        inputDurationSeconds = await probeInputDurationSeconds(ffmpeg, job.inputName);
      } catch {
        inputDurationSeconds = null;
      }
      return inputDurationSeconds;
    };

    for (let strategyIndex = 0; strategyIndex < fallbackStrategies.length; strategyIndex += 1) {
      const strategy = fallbackStrategies[strategyIndex];
      const effectiveOptions = { ...options, ...strategy.options };
      try {
        if (speechRenderPlan && !effectiveOptions.skipSpeechSegmentation) {
          try {
            setStatus(`${stageLabel}: ${job.base} (${fileIndex + 1}/${totalFiles})`);
            setActiveQueueStage(job.base, stageLabel, `File ${fileIndex + 1} of ${totalFiles}`);
            speechAlignedSegmentCountUsed = await runMixReadySpeechAlignedSegmented(
              ffmpeg,
              job.inputName,
              outputName,
              profile,
              speechRenderPlan.durationSeconds,
              speechRenderPlan.silenceSpans,
              speechRenderPlan.speechSpans,
              { ...effectiveOptions, segmentMode: speechRenderPlan.mode }
            );
            mixRendered = true;
            fallbackApplied =
              strategyIndex === 0
                ? `${speechRenderPlan.mode} segmented`
                : `${strategy.label} (${speechRenderPlan.mode} segmented)`;
            appendLog(
              `[Segmented] ${job.base}: ${speechRenderPlan.mode} render used ${speechAlignedSegmentCountUsed} segments (${formatCandidateVariant(
                effectiveOptions.candidateVariant ?? "cinematic-stable"
              )}).`
            );
            break;
          } catch (segError) {
            lastMixError = segError;
            appendLog(
              `[Segmented] ${job.base}: ${speechRenderPlan.mode} ${strategy.label} failed (${describeError(
                segError
              )}), trying single-pass ${strategy.label}.`
            );
            if (shouldResetFfmpegForError(segError)) {
              ffmpeg = await refreshFfmpeg(`${speechRenderPlan.mode} mix fallback on ${job.base}`);
              await writeJobInput(ffmpeg, job);
            }
          }
        }

        setStatus(`${stageLabel}: ${job.base} (${fileIndex + 1}/${totalFiles})`);
        setActiveQueueStage(job.base, stageLabel, `File ${fileIndex + 1} of ${totalFiles}`);
        await runMixReady(ffmpeg, job.inputName, outputName, profile, effectiveOptions);
        mixRendered = true;
        fallbackApplied = strategyIndex === 0 ? null : strategy.label;
        break;
      } catch (error) {
        lastMixError = error;
        const strategyFailureMessage = describeError(error);
        if (shouldResetFfmpegForError(error)) {
          ffmpeg = await refreshFfmpeg(`mix fallback on ${job.base}`);
          await writeJobInput(ffmpeg, job);
        }

        const durationSeconds = await ensureInputDuration();
        const canRunSegmented = durationSeconds !== null && durationSeconds >= MIX_SEGMENT_MIN_DURATION_SECONDS;

        if (canRunSegmented && durationSeconds !== null) {
          try {
            appendLog(
              `[MixFallback] ${job.base}: ${strategy.label} failed (${strategyFailureMessage}), trying segmented ${strategy.label}.`
            );
            await runMixReadySegmented(ffmpeg, job.inputName, outputName, profile, durationSeconds, effectiveOptions);
            mixRendered = true;
            fallbackApplied = strategyIndex === 0 ? "primary chain (segmented)" : `${strategy.label} (segmented)`;
            break;
          } catch (segmentedError) {
            lastMixError = segmentedError;
            if (shouldResetFfmpegForError(segmentedError)) {
              ffmpeg = await refreshFfmpeg(`segmented mix fallback on ${job.base}`);
              await writeJobInput(ffmpeg, job);
            }
          }
        }

        const hasMoreStrategies = strategyIndex < fallbackStrategies.length - 1;
        if (hasMoreStrategies) {
          const finalFailureMessage = describeError(lastMixError);
          appendLog(
            `[MixFallback] ${job.base}: ${strategy.label} failed (${finalFailureMessage}), trying ${
              fallbackStrategies[strategyIndex + 1]?.label
            }.`
          );
        }
      }
    }

    if (!mixRendered) {
      throw lastMixError ?? new Error("Mix-ready render failed.");
    }

    return { ffmpeg, fallbackApplied, speechAlignedSegmentCountUsed };
  };

  const processFiles = async () => {
    if (!files.length) return;
    setLoading(true);
    setOutputs([]);
    setLogs([]);
    setFailedOptimizations([]);
    setShowFailureWarning(false);
    setStatus("Preparing...");

    try {
      let ffmpeg = await ensureFfmpeg();
      const outputEntries: OutputEntry[] = [];
      const jobs = buildJobs(files);
      initializeQueueItems(jobs);
      const analysisByBase = new Map<string, FileAnalysis>();
      let batchReference: BatchReference | null = null;
      const smartMatchEnabled = smartMatchConfig.tone > 0 || smartMatchConfig.dynamics > 0;
      const needsAnalysis = true;

      if (needsAnalysis) {
        appendLog(
          `Deep analysis started for ${jobs.length} file(s) (bootstrap up to ${ANALYSIS_SAMPLE_SECONDS}s, distributed coverage on long takes).`
        );
        const analyses: FileAnalysis[] = [];
        for (let i = 0; i < jobs.length; i += 1) {
          const job = jobs[i];
          setStatus(`Analyze: ${job.base} (${i + 1}/${jobs.length})`);
          setActiveQueueStage(job.base, "Analyze", `Pass ${i + 1} of ${jobs.length}`);
          try {
            await writeJobInput(ffmpeg, job);
            const analysisResult = await analyzeFile(ffmpeg, job.inputName);
            ffmpeg = analysisResult.ffmpeg;
            const analysis = analysisResult.analysis;
            analysisByBase.set(job.base, analysis);
            analyses.push(analysis);
            markQueuePending(job.base, "Ready for render", "Analysis complete");
          } catch (error) {
            appendLog(
              `Analysis fallback (${job.base}): ${error instanceof Error ? error.message : String(error)}`
            );
            const fallback = createEmptyAnalysis();
            analysisByBase.set(job.base, fallback);
            analyses.push(fallback);
            markQueuePending(job.base, "Ready for render", "Analysis fallback used");
            if (shouldResetFfmpegForError(error)) {
              ffmpeg = await refreshFfmpeg(`analysis failure on ${job.base}`);
            }
          } finally {
            await safeDeleteFile(ffmpeg, job.inputName);
          }

          if (shouldRecycleFfmpegForBatch(i + 1, jobs.length)) {
            ffmpeg = await refreshFfmpeg(`analysis memory guard (${i + 1}/${jobs.length})`);
          }
        }

        if (smartMatchEnabled) {
          batchReference = buildBatchReference(analyses);
          if (batchReference) {
            appendLog(
              `Reference tone low/mid ${batchReference.lowTilt.toFixed(1)} dB, high/mid ${batchReference.highTilt.toFixed(1)} dB, LRA ${batchReference.lra.toFixed(1)}.`
            );
          } else {
            appendLog("Reference analysis unavailable; using base processing chain.");
          }
        }
      }

      let hadErrors = false;
      const failedRuns: FailedOptimization[] = [];
      for (let i = 0; i < jobs.length; i += 1) {
        const job = jobs[i];
        let cleanLoudName: string | null = null;
        let blendLoudName: string | null = null;
        let blendRendered = false;

        try {
          await writeJobInput(ffmpeg, job);
          const profile = buildAdaptiveProfile(analysisByBase.get(job.base), batchReference);
          const roomScore = profile ? (analysisByBase.get(job.base)?.roomScore ?? 0) : null;
          const adaptiveNoiseReductionFilter = profile ? resolveAdaptiveNoiseReductionFilter(profile) : null;
          const primaryMixFilterPreview = profile ? buildMixFilter(profile) : "";
          const dynaPreviewMatch = primaryMixFilterPreview.match(/dynaudnorm=([^,]+)/i);
          const dynaPreview = dynaPreviewMatch ? dynaPreviewMatch[1] : "off";
          const adaptiveNoiseReductionLabel =
            adaptiveNoiseReductionFilter === null
              ? "off"
              : adaptiveNoiseReductionFilter.trim().toLowerCase().startsWith("afftdn=")
                ? "on(spectral)"
                : "on(wavelet)";

          if (profile) {
            appendLog(
              `[Adaptive] ${job.base}: HPF ${profile.highpassHz} Hz, low-mid ${formatSigned(
                profile.lowMidGainDb
              )} dB, presence ${formatSigned(profile.presenceGainDb)} dB, room ${profile.roomRisk} (${(
                roomScore ?? 0
              ).toFixed(2)}), noise ${profile.noiseRisk} (${(profile.noiseFloorDb ?? -70).toFixed(
                1
              )} dB; adaptive-NR ${adaptiveNoiseReductionLabel}), instability ${(
                profile.instabilityScore * 100
              ).toFixed(0)}%, line swing ${(profile.lineSwingScore * 100).toFixed(0)}%, sentence jump ${(
                profile.sentenceJumpScore * 100
              ).toFixed(0)}%, breath spike ${(profile.breathSpikeRisk * 100).toFixed(0)}%, pause risk ${(
                profile.pauseNoiseRisk * 100
              ).toFixed(0)}%, onset/sag/end ${((analysisByBase.get(job.base)?.onsetOvershootScore ?? 0) * 100).toFixed(
                0
              )}/${((analysisByBase.get(job.base)?.midLineSagScore ?? 0) * 100).toFixed(0)}/${(
                (analysisByBase.get(job.base)?.endFadeRiskScore ?? 0) * 100
              ).toFixed(0)}%, speech-duty ${(analysisByBase.get(job.base)?.speechDutyCyclePct ?? 0).toFixed(
                1
              )}%, median-run ${(((analysisByBase.get(job.base)?.medianSpeechRunMs ?? 0) as number) / 1000).toFixed(
                1
              )}s, segmentation ${
                profile.preferSinglePassContinuity
                  ? "single-pass continuity"
                  : profile.useSpeechPauseSegmentation
                    ? "speech-pause"
                    : profile.useSpeechAlignedSegmentation
                      ? "speech-aligned"
                      : "off"
              }, clicks ${(
                profile.clickScore * 100
              ).toFixed(0)}%, conf ${(
                analysisByBase.get(job.base)?.analysisConfidence ?? 0
              ).toFixed(2)}, tail-gate ${profile.useTailGate ? "on" : "off"}${
                profile.preserveEndings ? " (endings protect)" : ""
              }${profile.strictEndingProtection ? " [strict]" : ""}, dyna ${dynaPreview}, echo ${
                analysisByBase.get(job.base)?.echoDelayMs ?? 0
              } ms, blend ${
                (profile.blendIndoorGain * 100).toFixed(1)
              }/${(profile.blendOutdoorGain * 100).toFixed(1)}%.`
            );
          }

          let speechRenderPlan: SpeechRenderPlan | null = null;
          try {
            speechRenderPlan = await buildSpeechRenderPlan(ffmpeg, job.inputName, profile);
            if (speechRenderPlan) {
              appendLog(
                `[SegmentPlan] ${job.base}: ${speechRenderPlan.mode} ready with ${
                  speechRenderPlan.speechSpans.length
                } speech spans and ${speechRenderPlan.silenceSpans.length} silence spans.`
              );
            }
          } catch (error) {
            appendLog(
              `[SegmentPlan] ${job.base}: speech/pause plan fallback (${error instanceof Error ? error.message : String(
                error
              )})`
            );
            speechRenderPlan = null;
            if (shouldResetFfmpegForError(error)) {
              ffmpeg = await refreshFfmpeg(`segment-plan failure on ${job.base}`);
              await writeJobInput(ffmpeg, job);
            }
          }

          const candidateVariants = buildMixCandidateVariants(profile);
          let selectedVariant: CandidateVariant | null = null;
          let selectedBytes: Uint8Array | null = null;
          let selectedScore: CandidateScore | null = null;
          let selectedAnalysis: FileAnalysis | null = null;
          let selectedFallbackApplied: string | null = null;

          for (const candidateVariant of candidateVariants) {
            const candidateLabel = formatCandidateVariant(candidateVariant);
            const candidateName = `${job.base}_${candidateLabel.replace(/-/g, "_")}_candidate.wav`;
            const candidateOptions: MixRenderOptions = {
              candidateVariant,
              skipSpeechSegmentation:
                candidateVariant === "continuity-safe" && (profile?.preferSinglePassContinuity ?? false),
            };
            try {
              const renderResult = await renderMixReadyWithFallbacks(
                ffmpeg,
                job,
                candidateName,
                profile,
                i,
                jobs.length,
                `Mix-ready (${candidateLabel})`,
                candidateOptions,
                speechRenderPlan
              );
              ffmpeg = renderResult.ffmpeg;
              const candidateBytes = await readVirtualFileBytes(ffmpeg, candidateName);

              let candidateAnalysis: FileAnalysis | null = null;
              try {
                const analysisResult = await analyzeFile(ffmpeg, candidateName);
                ffmpeg = analysisResult.ffmpeg;
                candidateAnalysis = analysisResult.analysis;
              } catch (error) {
                appendLog(
                  `[CandidateQC] ${job.base}/${candidateLabel}: analysis fallback (${
                    error instanceof Error ? error.message : String(error)
                  }).`
                );
                if (shouldResetFfmpegForError(error)) {
                  ffmpeg = await refreshFfmpeg(`candidate QC on ${job.base}`);
                  await writeJobInput(ffmpeg, job);
                }
              }

              const candidateScore = buildCandidateScore(candidateAnalysis);
              appendLog(
                `[CandidateQC] ${job.base}/${candidateLabel}: ${summarizeCandidateScore(candidateScore)}${
                  renderResult.fallbackApplied ? `, fallback ${renderResult.fallbackApplied}` : ""
                }.`
              );
              const shouldSelect =
                selectedBytes === null ||
                selectedScore === null ||
                compareCandidateScores(candidateScore, selectedScore) < 0;
              if (shouldSelect) {
                selectedVariant = candidateVariant;
                selectedBytes = candidateBytes;
                selectedScore = candidateScore;
                selectedAnalysis = candidateAnalysis;
                selectedFallbackApplied = renderResult.fallbackApplied;
              }

              await safeDeleteFile(ffmpeg, candidateName);
            } catch (error) {
              appendLog(
                `[CandidateQC] ${job.base}/${candidateLabel}: render failed (${error instanceof Error ? error.message : String(
                  error
                )}).`
              );
              if (shouldResetFfmpegForError(error)) {
                ffmpeg = await refreshFfmpeg(`candidate render failure on ${job.base}`);
                await writeJobInput(ffmpeg, job);
              }
            }
          }

          if (!selectedBytes || !selectedVariant) {
            throw new Error("No candidate mix-ready render completed.");
          }

          await ffmpeg.writeFile(job.mixName, selectedBytes);
          appendLog(
            `[CandidateSelect] ${job.base}: kept ${formatCandidateVariant(selectedVariant)}${
              selectedFallbackApplied ? ` (${selectedFallbackApplied})` : ""
            } with ${summarizeCandidateScore(selectedScore ?? buildCandidateScore(selectedAnalysis))}.`
          );

          const mixOutput = await writeOutput(ffmpeg, job.mixName, "mixready", "clean");
          if (keepMixReady || loudnessConfig === null) {
            outputEntries.push(mixOutput);
          }

          if (sceneBlend) {
            const indoorGain = profile?.blendIndoorGain ?? 0;
            const outdoorGain = profile?.blendOutdoorGain ?? 0;
            if (indoorGain + outdoorGain <= 0.0001) {
              appendLog(`[Blend] ${job.base}: bypassed (adaptive blend gain near zero for room/noise safety).`);
            } else {
              try {
                setStatus(`Blend: ${job.base} (${i + 1}/${jobs.length})`);
                setActiveQueueStage(job.base, "Blend", `File ${i + 1} of ${jobs.length}`);
                await runBlendMixReady(ffmpeg, job.mixName, job.blendMixName, profile);
                const blendMixOutput = await writeOutput(ffmpeg, job.blendMixName, "mixready", "blend");
                blendRendered = true;
                if (keepMixReady || loudnessConfig === null) {
                  outputEntries.push(blendMixOutput);
                }
              } catch (error) {
                appendLog(
                  `[Blend] ${job.base}: bypassed (${error instanceof Error ? error.message : String(error)})`
                );
              }
            }
          }

          if (loudnessConfig) {
            cleanLoudName = `${job.base}_${loudnessConfig.suffix}.wav`;
            setStatus(`Loudness clean: ${job.base} (${i + 1}/${jobs.length})`);
            setActiveQueueStage(job.base, "Loudness (clean)", `File ${i + 1} of ${jobs.length}`);
            await runLoudnorm(ffmpeg, job.mixName, cleanLoudName, loudnessConfig);
            const loudOutput = await writeOutput(ffmpeg, cleanLoudName, "loudness", "clean");
            outputEntries.push(loudOutput);

            if (sceneBlend && blendRendered) {
              blendLoudName = `${job.base}_blend_${loudnessConfig.suffix}.wav`;
              setStatus(`Loudness blend: ${job.base} (${i + 1}/${jobs.length})`);
              setActiveQueueStage(job.base, "Loudness (blend)", `File ${i + 1} of ${jobs.length}`);
              await runLoudnorm(ffmpeg, job.blendMixName, blendLoudName, loudnessConfig);
              const blendLoudOutput = await writeOutput(ffmpeg, blendLoudName, "loudness", "blend");
              outputEntries.push(blendLoudOutput);
            }
          }

          markQueueDone(job.base, "Outputs ready");
        } catch (error) {
          hadErrors = true;
          const reason = summarizeFailureReason(error);
          failedRuns.push({
            base: job.base,
            fileName: job.file.name,
            reason,
          });
          appendLog(`Error (${job.base}): ${reason}`);
          markQueueError(job.base, reason);
          if (shouldResetFfmpegForError(error)) {
            ffmpeg = await refreshFfmpeg(`processing failure on ${job.base}`);
          }
        } finally {
          await safeDeleteFile(ffmpeg, job.inputName);
          await safeDeleteFile(ffmpeg, job.mixName);
          await safeDeleteFile(ffmpeg, job.blendMixName);
          if (cleanLoudName) {
            await safeDeleteFile(ffmpeg, cleanLoudName);
          }
          if (blendLoudName) {
            await safeDeleteFile(ffmpeg, blendLoudName);
          }
        }

        if (shouldRecycleFfmpegForBatch(i + 1, jobs.length)) {
          ffmpeg = await refreshFfmpeg(`processing memory guard (${i + 1}/${jobs.length})`);
        }
      }

      setOutputs(outputEntries);
      if (failedRuns.length > 0) {
        setFailedOptimizations(failedRuns);
        setShowFailureWarning(true);
        appendLog(
          `[Warning] ${failedRuns.length} file(s) failed to optimize. Re-submit only the failed files and run again.`
        );
      }
      setStatus(hadErrors ? "Done with warnings" : "Done");
    } catch (err) {
      appendLog(`Error: ${err instanceof Error ? err.message : String(err)}`);
      setStatus("Failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const wavs = Array.from(incoming).filter((file) => file.name.toLowerCase().endsWith(".wav"));
    setFiles((prev) => {
      const merged = [...prev];
      const seen = new Set(prev.map((file) => `${file.name}|${file.size}|${file.lastModified}`));
      for (const file of wavs) {
        const key = `${file.name}|${file.size}|${file.lastModified}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(file);
        }
      }
      return merged;
    });
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    handleFiles(event.dataTransfer.files);
  };

  const downloadOutputsZip = async () => {
    if (outputs.length === 0 || zipBusy) return;

    setZipBusy(true);
    setZipProgress(0);

    try {
      const zip = new JSZip();

      for (let i = 0; i < outputs.length; i += 1) {
        const output = outputs[i];
        const response = await fetch(output.url);
        const blob = await response.blob();
        zip.file(output.name, blob);
        setZipProgress(Math.round(((i + 1) / outputs.length) * 70));
      }

      const zipBlob = await zip.generateAsync(
        {
          type: "blob",
          compression: "DEFLATE",
          compressionOptions: { level: 6 },
        },
        ({ percent }) => {
          setZipProgress(70 + Math.round((percent / 100) * 30));
        }
      );

      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const archiveName = `vo_leveler_outputs_${stamp}.zip`;
      const zipUrl = URL.createObjectURL(zipBlob);
      const link = document.createElement("a");
      link.href = zipUrl;
      link.download = archiveName;
      link.rel = "noopener";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(zipUrl), 30_000);
      appendLog(`ZIP created: ${archiveName} (${formatBytes(zipBlob.size)})`);
    } catch (error) {
      appendLog(`ZIP export failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setZipBusy(false);
      setZipProgress(0);
    }
  };

  const activeQueueItem = useMemo(
    () => queueItems.find((item) => item.status === "working") ?? null,
    [queueItems]
  );
  const queueCounts = useMemo(
    () => ({
      total: queueItems.length,
      done: queueItems.filter((item) => item.status === "done").length,
      error: queueItems.filter((item) => item.status === "error").length,
      working: queueItems.filter((item) => item.status === "working").length,
      pending: queueItems.filter((item) => item.status === "pending").length,
    }),
    [queueItems]
  );

  return (
    <div className={styles.layout}>
      <div className={styles.panel}>
        <div className={styles.card}>
          <div
            className={`${styles.dropzone} ${dragActive ? styles.dropActive : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <div className={styles.dropTitle}>Drop WAV files or pick a folder</div>
            <div className={styles.dropHint}>
              Processing runs locally in the browser. Files never leave this machine.
            </div>
            <div className={styles.dropHint}>New drops are added to the current queue.</div>
            <div className={styles.controls}>
              <label className={styles.button}>
                Select Files
                <input
                  type="file"
                  accept=".wav"
                  multiple
                  hidden
                  onChange={(event) => {
                    handleFiles(event.target.files);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              <label className={`${styles.button} ${styles.buttonSecondary}`}>
                Select Folder
                <input
                  type="file"
                  accept=".wav"
                  multiple
                  hidden
                  // @ts-expect-error webkitdirectory is supported in Chromium-based browsers.
                  webkitdirectory="true"
                  directory="true"
                  onChange={(event) => {
                    handleFiles(event.target.files);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
            </div>
            <div className={styles.fileList}>
              {files.length === 0 && <div className={styles.dropHint}>No files selected.</div>}
              {files.map((file, index) => (
                <div className={styles.fileItem} key={`${file.name}-${file.lastModified}-${index}`}>
                  <div>{file.name}</div>
                  <span>{formatBytes(file.size)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className={styles.card}>
          <div className={styles.optionGrid}>
            <div className={styles.field}>
              <label className={styles.label}>Loudness target</label>
              <select
                className={styles.select}
                value={loudnessTarget}
                onChange={(event) => setLoudnessTarget(event.target.value as keyof typeof LOUDNESS_PRESETS)}
              >
                {Object.keys(LOUDNESS_PRESETS).map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
            </div>
            <div className={styles.field}>
              <label className={styles.label}>Smart voice match</label>
              <select
                className={styles.select}
                value={smartMatchMode}
                onChange={(event) =>
                  setSmartMatchMode(event.target.value as keyof typeof SMART_MATCH_PRESETS)
                }
              >
                {Object.keys(SMART_MATCH_PRESETS).map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
            </div>
            <div className={styles.field}>
              <label className={styles.label}>Leveling strength</label>
              <select
                className={styles.select}
                value={leveler}
                onChange={(event) => setLeveler(event.target.value as keyof typeof LEVELER_PRESETS)}
              >
                {Object.keys(LEVELER_PRESETS).map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
              <div className={styles.label}>
                Balances consistency while keeping performance peaks.
              </div>
            </div>
            <div className={styles.field}>
              <label className={styles.label}>Breath control</label>
              <select
                className={styles.select}
                value={breathControl}
                onChange={(event) => setBreathControl(event.target.value as keyof typeof BREATH_COMPAND)}
              >
                {Object.keys(BREATH_COMPAND).map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className={styles.toggleRow}>
            <div>
              <strong>EQ cleanup</strong>
              <div className={styles.label}>HPF + small low-mid shaping for consistency</div>
            </div>
            <input
              type="checkbox"
              checked={eqCleanup}
              onChange={(event) => setEqCleanup(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Soften harshness</strong>
              <div className={styles.label}>
                Cinematic softening for bright/emotional lines (3.5 kHz + 8 kHz + gentle top-end trim)
              </div>
            </div>
            <input
              type="checkbox"
              checked={softenHarshness}
              onChange={(event) => setSoftenHarshness(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Room cleanup (auto detect)</strong>
              <div className={styles.label}>
                Reduces mild room echo/reverb only when needed.
              </div>
            </div>
            <input
              type="checkbox"
              checked={roomCleanup}
              onChange={(event) => setRoomCleanup(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Scene blend (adaptive subtle)</strong>
              <div className={styles.label}>
                Adds very light mono early reflections so VO sits in-picture without sounding processed.
              </div>
            </div>
            <input
              type="checkbox"
              checked={sceneBlend}
              onChange={(event) => setSceneBlend(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Noise guard</strong>
              <div className={styles.label}>Limits auto-leveler gain to avoid noise lift</div>
            </div>
            <input
              type="checkbox"
              checked={noiseGuard}
              onChange={(event) => setNoiseGuard(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Floor guard</strong>
              <div className={styles.label}>
                Keeps near-silence quiet; auto-prioritized over breath control on noisy tracks
              </div>
            </div>
            <input
              type="checkbox"
              checked={floorGuard}
              onChange={(event) => setFloorGuard(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Keep mix-ready file</strong>
              <div className={styles.label}>Store _mixready.wav alongside loudness exports</div>
            </div>
            <input
              type="checkbox"
              checked={keepMixReady}
              onChange={(event) => setKeepMixReady(event.target.checked)}
              disabled={loudnessConfig === null}
            />
          </div>

          <div className={`${styles.controls} ${styles.sectionTop}`}>
            <button className={styles.button} onClick={processFiles} disabled={loading || files.length === 0}>
              {loading ? "Processing..." : "Run Batch"}
            </button>
            <button
              className={`${styles.button} ${styles.buttonGhost}`}
              onClick={() => {
                setFiles([]);
                setOutputs([]);
                setLogs([]);
                setFailedOptimizations([]);
                setShowFailureWarning(false);
                setQueueItems([]);
                activeQueueBaseRef.current = null;
                activeQueueStageRef.current = "Queued";
                activeQueueProgressRef.current = -1;
                setStatus("Idle");
              }}
              disabled={loading}
            >
              Clear
            </button>
            <div className={styles.progress}>{status}</div>
          </div>

          <div className={styles.footerNote}>
            Processing order is tuned to avoid processor clashes.
          </div>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.queueHeader}>
          <h3>Batch Queue</h3>
          {queueCounts.total > 0 && (
            <div className={styles.queueSummaryBadges}>
              <span className={`${styles.queueCountBadge} ${styles.queueCountNeutral}`}>
                {queueCounts.total} total
              </span>
              <span className={`${styles.queueCountBadge} ${styles.queueCountActive}`}>
                {queueCounts.working} active
              </span>
              <span className={`${styles.queueCountBadge} ${styles.queueCountDone}`}>
                {queueCounts.done} done
              </span>
              <span className={`${styles.queueCountBadge} ${styles.queueCountError}`}>
                {queueCounts.error} failed
              </span>
              <span className={`${styles.queueCountBadge} ${styles.queueCountNeutral}`}>
                {queueCounts.pending} waiting
              </span>
            </div>
          )}
        </div>

        <div className={styles.sectionTop}>
          {queueItems.length === 0 ? (
            <div className={styles.dropHint}>No batch queue yet. Add files and run processing.</div>
          ) : (
            <>
              <div className={styles.queueActiveBar}>
                <div>
                  <strong>{activeQueueItem ? activeQueueItem.fileName : "No active file"}</strong>
                  <div className={styles.label}>
                    {activeQueueItem
                      ? `${activeQueueItem.stageLabel}${activeQueueItem.detail ? ` • ${activeQueueItem.detail}` : ""}`
                      : loading
                        ? "Waiting for next command..."
                        : "Idle"}
                  </div>
                </div>
                <div className={styles.queueActivePercent}>
                  {activeQueueItem ? `${Math.round(activeQueueItem.progress * 100)}%` : "—"}
                </div>
              </div>
              <div className={styles.queueList}>
                {queueItems.map((item) => {
                  const statusClass =
                    item.status === "done"
                      ? styles.queueStatusDone
                      : item.status === "error"
                        ? styles.queueStatusError
                        : item.status === "working"
                          ? styles.queueStatusWorking
                          : styles.queueStatusPending;
                  const progressPercent =
                    item.status === "done" || item.status === "error"
                      ? 100
                      : Math.round(clamp(item.progress, 0, 1) * 100);
                  return (
                    <div
                      key={item.base}
                      className={`${styles.queueItem} ${
                        item.status === "working" ? styles.queueItemActive : ""
                      }`}
                    >
                      <div className={styles.queueRowTop}>
                        <div className={styles.queueTitleWrap}>
                          <span className={styles.queueIndex}>{item.index + 1}</span>
                          <div>
                            <div className={styles.queueFileName}>{item.fileName}</div>
                            <div className={styles.queueStageText}>
                              {item.stageLabel}
                              {item.detail ? ` • ${item.detail}` : ""}
                            </div>
                          </div>
                        </div>
                        <span className={`${styles.queueStatusBadge} ${statusClass}`}>
                          {item.status === "working"
                            ? "Processing"
                            : item.status === "done"
                              ? "Done"
                              : item.status === "error"
                                ? "Failed"
                                : "Queued"}
                        </span>
                      </div>
                      <div className={styles.queueProgressTrack} aria-hidden="true">
                        <div
                          className={`${styles.queueProgressFill} ${statusClass}`}
                          style={{ width: `${progressPercent}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>

      <div className={styles.panel}>
        <div className={styles.card}>
          <h3>Outputs</h3>
          {outputs.length > 0 && (
            <div className={`${styles.controls} ${styles.sectionTop}`}>
              <button
                className={`${styles.button} ${styles.buttonSecondary}`}
                onClick={downloadOutputsZip}
                disabled={zipBusy}
              >
                {zipBusy ? `Building ZIP ${zipProgress}%` : `Download ZIP (${outputs.length})`}
              </button>
            </div>
          )}
          <div className={`${styles.outputList} ${styles.sectionTop}`}>
            {outputs.length === 0 && <div className={styles.dropHint}>No output yet.</div>}
            {outputs.map((output, index) => (
              <div className={styles.outputItem} key={`${output.name}-${output.size}-${index}`}>
                <div>
                  <strong>{output.name}</strong>
                  <div className={styles.label}>{formatBytes(output.size)}</div>
                  <div className={styles.outputMeta}>
                    <span className={styles.outputBadge}>
                      {output.variant === "blend" ? "Blend pass" : "Clean pass"}
                    </span>
                    {output.kind === "mixready" ? (
                      <>
                        <span className={styles.outputBadge}>Mix-ready</span>
                        <span
                          className={styles.outputHint}
                          title={
                            output.variant === "blend"
                              ? "Blend mix-ready: subtle scene glue applied; not loudness-normalized."
                              : "Mix-ready: processed and leveled, but not loudness-normalized. Best for film mix stems."
                          }
                        >
                          What&apos;s this?
                        </span>
                      </>
                    ) : (
                      <>
                        <span className={styles.outputBadge}>Broadcast loudness</span>
                        <span
                          className={styles.outputHint}
                          title={
                            output.variant === "blend"
                              ? "Broadcast loudness + blend: subtle scene glue plus ATSC A/85 or EBU R128 normalization."
                              : "Broadcast loudness: normalized to ATSC A/85 or EBU R128. Use for delivery or QC."
                          }
                        >
                          What&apos;s this?
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <a href={output.url} download={output.name}>
                  Download
                </a>
              </div>
            ))}
          </div>
        </div>
        <div className={styles.card}>
          <h3>Processing Log</h3>
          <div className={`${styles.log} ${styles.sectionTop}`}>
            {logs.length === 0 ? "No logs yet." : logs.join("\n")}
          </div>
          <div className={styles.footerNote}>
            If processing feels slow, run a smaller batch or disable extra features.
          </div>
        </div>
      </div>
      {showFailureWarning && failedOptimizations.length > 0 && (
        <div className={styles.warningOverlay} role="alertdialog" aria-modal="true" aria-labelledby="failed-title">
          <div className={styles.warningCard}>
            <h3 id="failed-title">Some files need re-submission</h3>
            <p className={styles.warningText}>
              Some audio files failed to optimize on this run. Please re-submit only these files and run again.
            </p>
            <div className={styles.warningList}>
              {failedOptimizations.map((item, index) => (
                <div className={styles.warningItem} key={`${item.base}-${index}`}>
                  <strong>{item.fileName}</strong>
                  <span>{item.reason}</span>
                </div>
              ))}
            </div>
            <div className={styles.warningActions}>
              <button className={styles.button} onClick={() => setShowFailureWarning(false)}>
                Understood
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
