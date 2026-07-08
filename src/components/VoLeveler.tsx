"use client";

import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import JSZip from "jszip";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { analyzeFloatSamples, buildSpeechMask, type AudioQcMetrics } from "../lib/audioQc";
import {
  detectAudibilityDropouts,
  frameDbFromFloatSamples,
  type AudibilityDropoutReport,
} from "../lib/audibilityDropout";
import {
  applyKWeighting,
  applyGainCurveToSamples,
  planGainCurve,
  resolvePlannerCalibration,
  speechRunsFromMask,
  tameRenderedConsonantPeaks,
  type SpeechRun as PlannerSpeechRun,
} from "../lib/gainPlanner";
import {
  AUDIBILITY_SAFE_STRATEGY_LABEL,
  PLANNER_TAIL_SAFE_STRATEGY_LABEL,
  buildRenderRiskProfile,
  compareCandidateScores,
  isHealthySegmentedRender,
  resolveNextAudibilityFallbackIndex,
  selectQcUnavailableFallbackCandidate,
  type CandidateRenderMeta,
  type CandidateScore,
  type DegradeReason,
  type RenderPath,
  type RenderRiskProfile,
} from "../lib/renderRecovery";
import {
  DEFAULT_LEARNED_REVIEW_WEIGHTS,
  REVIEW_BUNDLE_SCHEMA_VERSION,
  REVIEW_WEIGHT_STORAGE_KEY,
  autoReviewBundle,
  buildReviewMetricDelta,
  estimateAlignmentMetrics,
  interleavedToMono,
  parseLearnedReviewWeights,
  resolveCorrectiveMaxFilesPerBatch,
  scoreCandidateWithLearnedWeights,
  shouldAttemptCorrectivePassForAssessment,
  toReviewMetricSnapshot,
  type AlignmentMetrics,
  type CandidateRankingBreakdown,
  type LearnedReviewWeights,
  type ReviewBundleManifest,
  type ReviewMetricDelta,
  type ReviewMetricSnapshot,
} from "../lib/reviewLearning";
import {
  computeLogBandSpectrumDb,
  computeSibilanceScore,
  computeToneMatchDeltaDb,
  HOUSE_TONE_BLEND,
  resolveDeEsserBands,
  SPECTRUM_BANDS_HZ,
} from "../lib/spectrum";
import { planBatchLoudnessAlignment } from "../lib/batchLoudnessAlign";
import { decodeWav, encodeWavFloat32 } from "../lib/webAudioRender";
import { queueBrowserDownload, triggerBrowserDownload } from "../lib/downloadBlob";
import { estimateVoZipBytes, planVoZipExportParts } from "../lib/downloadPolicy";
import {
  formatLongFormPartTag,
  planLongFormChunks,
  shouldUseLongFormSafeMode,
} from "../lib/longFormExportPolicy";
import type {
  AudioReviewControls,
  AudioReviewAdaptiveDirectives,
  AudioReviewFileInput,
  AudioReviewMetricSnapshot,
  AudioReviewProfileSnapshot,
  AudioReviewResult,
  AudioReviewSelectedCandidate,
  SourceFirstAudioReviewPlan,
} from "../lib/aiAudioReview";
import {
  DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES,
  buildCorrectiveDirectivesForIssueTags,
  buildSourceFirstAudioReviewPlan,
  hasNonTrivialAdaptiveDirectives,
  mergeAudioReviewAdaptiveDirectives,
  normalizeAdaptiveDirectives,
} from "../lib/aiAudioReview";
import {
  buildEchoRoomCleanupDecision,
  deriveEarlyEchoCancelStrength,
  type RoomCleanupRisk,
} from "../lib/roomCleanupPolicy";
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

/**
 * Cumulative audio (in seconds) the active ffmpeg worker may process before
 * we proactively recycle it. 40 minutes keeps large VO batches safer and
 * triggers a refresh after roughly every 3-4 long episodes — both safer than
 * the file-count guard alone, which can leave the worker grinding through
 * 45+ minutes of dialogue between recycles on long-file batches.
 */
const BATCH_AUDIO_RECYCLE_SECONDS = 2400;
const ANALYSIS_AUDIO_RECYCLE_SECONDS = 1800;
const LONG_FORM_WORKER_REFRESH_INTERVAL = 3;
/**
 * Auto-retry budget per file. We retry twice on recoverable failures
 * (OOM, filter-init errors, watchdog aborts) on a fresh worker. The
 * third failure marks the file permanently failed.
 */
const PER_FILE_MAX_RETRIES = 2;
/**
 * Per-file watchdog budget. We give each file
 *   max(WATCHDOG_BASE_SECONDS, durationSec * WATCHDOG_DURATION_FACTOR + WATCHDOG_BASE_SECONDS)
 * seconds before we terminate the active worker and trigger the retry path.
 * For a 3-min short take that's ~12.9 min budget; for a 15-min long take
 * that's ~62.5 min. Catches genuine hangs, never fires on legitimately
 * slow filter chains.
 */
const WATCHDOG_BASE_SECONDS = 90;

/**
 * Auto-load locally-trained review weights from `localStorage` on app start
 * (and react to cross-tab updates).
 *
 * Currently DISABLED: the prior training runs were on small same-direction
 * datasets (8 reviews / 7 reviews, all winner-preferred, no challenger
 * preferences), which produced ranker weights that don't generalize. The
 * built-in defaults are the safer choice until we have ≥ 30–50 reviews
 * with mixed outcomes.
 *
 * Manual training and import in the QC Lab still work — those just won't
 * auto-apply on the next app load.
 */
const AUTO_LOAD_LOCAL_REVIEW_WEIGHTS = false;
const WATCHDOG_DURATION_FACTOR = 4;
const LIMITER_FILTER = "alimiter=limit=-2dB:level=disabled";
const HEAD_PRIME_ENABLED = true;
const HEAD_PRIME_SECONDS = 1.0;
const HEAD_PRIME_DURATION_TOLERANCE_SECONDS = 0.05;
const POST_RENDER_REVIEW_MAX_REQUESTS = 6;
const MAX_CORRECTIVE_PASSES = 1;
const CORRECTIVE_WIN_MARGIN = 25;
const FINAL_CONSONANT_TAMER_MAX_BYTES = 260 * 1024 * 1024;
const FATAL_FFMPEG_PATTERN = /memory access out of bounds|runtimeerror/i;
const IMPORTANT_LOG_PATTERN = /error|failed|invalid|aborted|out of bounds/i;

/**
 * Sample rate used when the gain planner analyzes the full file.
 * 16 kHz mono is plenty for envelope analysis and keeps memory tiny even on
 * long takes (17-min file ≈ 65 MB Float32). The gain curve is applied via
 * ffmpeg `sendcmd`+`volume`, so we never have to re-decode the full audio
 * at the original rate on the JS side.
 */
const GAIN_PLANNER_ANALYSIS_SAMPLE_RATE = 16000;
const GAIN_PLANNER_FRAME_MS = 10;
const PLANNER_K_WEIGHTING = true;
// Mixed-formant speech fixtures with -4/0/+4 dB high-frequency tilt measured
// +0.15/-0.04/-0.44 dB raw body shift after the raw/K split, so no absolute
// target compensation is needed to stay within the +/-0.5 dB pre-K baseline.
const PLANNER_K_TARGET_OFFSET_DB = 0.0;
/**
 * Memory guard. Even at 16 kHz mono Float32 a 2-hour file would be ~460 MB,
 * which will OOM the WASM heap. Beyond this we fail/retry instead of
 * emitting planner-off legacy output.
 */
const GAIN_PLANNER_MAX_DURATION_SECONDS = 4800; // 80 minutes — fine at 16 kHz mono
const NEURAL_SPEECH_ENHANCEMENT_ENABLED_BY_DEFAULT =
  false;
const sanitizeBase = (name: string) =>
  name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9-_]+/g, "_");

const formatBytes = (bytes: number | null | undefined) => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "unknown size";
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
};

const assertUsableWavBytes = (bytes: Uint8Array, label: string) => {
  const byteLength = bytes.byteLength;
  if (!Number.isFinite(byteLength) || byteLength <= 44) {
    throw new Error(`${label} returned invalid audio (${formatBytes(byteLength)}).`);
  }
  return byteLength;
};

const enhancedOutputName = (name: string) =>
  /\.wav$/i.test(name) ? name.replace(/\.wav$/i, "_enhanced.wav") : `${name}_enhanced.wav`;

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

/**
 * Estimate audio duration in seconds from a WAV file's raw byte size.
 * Worst-case (tightest) guess: 16-bit / 48 kHz / mono ≈ 96 KB per second.
 * Real WAVs are usually larger per second (24-bit, stereo), so this
 * over-estimates duration which is the SAFE direction for memory budgeting:
 * we recycle the worker a little earlier than strictly required.
 *
 * Used pre-analysis to size watchdog timers and recycle decisions without
 * having to call ffmpeg just to probe duration.
 */
const estimateAudioSeconds = (file: File): number => {
  if (!file?.size) return 0;
  return file.size / 96000;
};

const estimatePcmF32MonoWavSeconds = (byteLength: number): number => {
  if (!Number.isFinite(byteLength) || byteLength <= 44) return 0;
  return (byteLength - 44) / (48000 * 4);
};

/**
 * A failure is "recoverable" if a fresh ffmpeg worker would plausibly
 * succeed on the same input. WASM out-of-bounds, filter-init failures,
 * watchdog timeouts all qualify. Format/parse errors don't (the input
 * itself is the issue).
 */
const isRecoverableFailure = (error: unknown): boolean => {
  const msg = error instanceof Error ? error.message : String(error);
  return /memory access out of bounds|RuntimeError|Watchdog timeout|Failed to inject frame|Error initializing filter|terminated|aborted/i.test(
    msg,
  );
};

type OutputEntry = {
  name: string;
  blob: Blob;
  size: number;
  kind: "mixready" | "loudness";
  variant: "clean" | "blend";
  processingFlow: "app" | "app-final-polish" | "app-neural-speech-enhancement-app";
  sourceBase?: string;
  sourceName?: string;
  partIndex?: number;
  partTotal?: number;
};

type ReviewBundleAsset = {
  path: string;
  blob: Blob;
};

type ReviewBundleEntry = {
  bundleId: string;
  manifest: ReviewBundleManifest;
  assets: ReviewBundleAsset[];
};

type DecodedMonoAudio = {
  sampleRate: number;
  channels: number;
  durationSec: number;
  monoSamples: Float32Array;
};

type FfmpegAssetUrls = {
  coreURL: string;
  wasmURL: string;
  workerURL?: string;
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
  bandSpectrumDb: number[] | null;
  sibilanceScore: number | null;
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
  coldOpenDipDb: number | null;
  coldOpenRiskScore: number | null;
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
  endEdgeDipDb: number | null;
  analysisWindowCount: number | null;
  analysisWindowsAttempted: number | null;
  analysisWindowsSucceeded: number | null;
  analysisWindowsDropped: number | null;
  analysisWindowRetryCount: number | null;
  longSparseModeEligible: boolean | null;
};

type BatchReference = {
  lowTilt: number;
  highTilt: number;
  lra: number;
  /** Median long-term band spectrum across the batch, one entry per SPECTRUM_BANDS_HZ band. */
  bandSpectrumDb: number[] | null;
  /** Number of clean-enough files that anchored the reference. */
  anchorCount: number;
};

type NoiseRisk = "low" | "medium" | "high";
type RoomRisk = RoomCleanupRisk;

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
  levelerBias: number;
  dynaTrim: number;
  floorGuardFilter: string;
  noiseRisk: NoiseRisk;
  noiseFloorDb: number | null;
  noiseContrastDb: number | null;
  pauseNoiseRisk: number;
  speechThresholdDb: number | null;
  roomRisk: RoomRisk;
  useDenoise: boolean;
  denoiseStrength: number;
  /** 8-band log-frequency spectrum in dB for this file. Null when not measured. */
  bandSpectrumDb: number[] | null;
  /** Per-band dB delta the tone matcher wants to apply (clamped), or null. */
  toneMatchDeltaDb: number[] | null;
  /** Sibilance score 0..1 — drives the de-esser gate. */
  sibilanceScore: number;
  /** When true, include the cinematic color shelves in the mix. */
  cinematicColorEnabled: boolean;
  useTailGate: boolean;
  tailGateStrength: number;
  echoNotchCutDb: number;
  echoDominantRoom: boolean;
  severeEchoRoom: boolean;
  allowDereverbDuringEndingProtection: boolean;
  allowTailGateDuringEndingProtection: boolean;
  echoDelayMs: number | null;
  earlyEchoCancelStrength: number;
  echoScore: number;
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
  longSparseModeEligible: boolean;
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
  bandSpectrumDb: null,
  sibilanceScore: null,
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
  coldOpenDipDb: null,
  coldOpenRiskScore: null,
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
  endEdgeDipDb: null,
  analysisWindowCount: null,
  analysisWindowsAttempted: null,
  analysisWindowsSucceeded: null,
  analysisWindowsDropped: null,
  analysisWindowRetryCount: null,
  longSparseModeEligible: null,
});

export default function VoLeveler() {
  const ffmpegRef = useRef<FFmpeg | null>(null);
  const ffmpegLoadPromiseRef = useRef<Promise<FFmpeg> | null>(null);
  const ffmpegAssetUrlsRef = useRef<FfmpegAssetUrls | null>(null);
  const logBufferRef = useRef<string[]>([]);
  const activeQueueBaseRef = useRef<string | null>(null);
  const activeQueueStageRef = useRef<string>("Queued");
  const activeQueueProgressRef = useRef<number>(-1);
  const aiReviewCloseRef = useRef<HTMLButtonElement | null>(null);
  const aiReviewTriggerRef = useRef<HTMLButtonElement | null>(null);
  const processingControlsOverrideRef = useRef<AudioReviewControls | null>(null);
  const aiAdaptiveDirectivesOverrideRef = useRef<AudioReviewAdaptiveDirectives | null>(null);
  const sourceFirstCandidateVariantRef = useRef<
    SourceFirstAudioReviewPlan["selectedVariant"] | null
  >(null);
  const sourceFirstPlansByBaseRef = useRef<Map<string, SourceFirstAudioReviewPlan>>(new Map());

  const [files, setFiles] = useState<File[]>([]);
  const [outputs, setOutputs] = useState<OutputEntry[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("Idle");
  const [loading, setLoading] = useState(false);
  const [zipBusy, setZipBusy] = useState(false);
  const [zipProgress, setZipProgress] = useState(0);
  const [downloadQueueBusy, setDownloadQueueBusy] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<string | null>(null);
  const [reviewZipBusy, setReviewZipBusy] = useState(false);
  const [reviewZipProgress, setReviewZipProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [failedOptimizations, setFailedOptimizations] = useState<FailedOptimization[]>([]);
  const [showFailureWarning, setShowFailureWarning] = useState(false);
  const [queueItems, setQueueItems] = useState<QueueItem[]>([]);
  const [reviewBundles, setReviewBundles] = useState<ReviewBundleEntry[]>([]);
  const [aiReviewFiles, setAiReviewFiles] = useState<AudioReviewFileInput[]>([]);
  const [aiReviewOpen, setAiReviewOpen] = useState(false);
  const [aiReviewBusy, setAiReviewBusy] = useState(false);
  const [aiReviewError, setAiReviewError] = useState<string | null>(null);
  const [aiReviewResult, setAiReviewResult] = useState<AudioReviewResult | null>(null);
  const [aiReviewModel, setAiReviewModel] = useState<string | null>(null);
  const [aiAutoPilotEnabled, setAiAutoPilotEnabled] = useState(true);
  const [aiAutoPilotStatus, setAiAutoPilotStatus] = useState<string | null>(null);
  const [learnedReviewWeights, setLearnedReviewWeights] =
    useState<LearnedReviewWeights>(DEFAULT_LEARNED_REVIEW_WEIGHTS);
  const [learnedReviewWeightsSource, setLearnedReviewWeightsSource] =
    useState<"default" | "local-import">("default");

  const [loudnessTarget, setLoudnessTarget] = useState<keyof typeof LOUDNESS_PRESETS>(
    "Mix-ready only (no loudness normalize)"
  );
  // Only applies when a loudness-normalized export is selected; mix-ready mode always emits the mix-ready file.
  const [keepMixReady, setKeepMixReady] = useState(false);
  const [smartMatchMode, setSmartMatchMode] = useState<keyof typeof SMART_MATCH_PRESETS>("Balanced");
  const [eqCleanup, setEqCleanup] = useState(true);
  const [breathControl, setBreathControl] = useState<keyof typeof BREATH_COMPAND>("Medium");
  const [leveler, setLeveler] = useState<keyof typeof LEVELER_PRESETS>("Balanced");
  const [roomCleanup, setRoomCleanup] = useState(true);
  const [sceneBlend, setSceneBlend] = useState(false);
  const [softenHarshness, setSoftenHarshness] = useState(true);
  const [noiseGuard, setNoiseGuard] = useState(true);
  const [floorGuard, setFloorGuard] = useState(true);
  const [cinematicColor, setCinematicColor] = useState(true);
  const [gainPlannerEnabled, setGainPlannerEnabled] = useState(true);
  const [neuralSpeechEnhancementEnabled, setNeuralSpeechEnhancementEnabled] = useState(
    NEURAL_SPEECH_ENHANCEMENT_ENABLED_BY_DEFAULT,
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const loudnessConfig = useMemo(() => LOUDNESS_PRESETS[loudnessTarget], [loudnessTarget]);
  const smartMatchConfig = useMemo(() => SMART_MATCH_PRESETS[smartMatchMode], [smartMatchMode]);
  const audioReviewControls = useMemo<AudioReviewControls>(
    () => ({
      loudnessTarget,
      smartMatchMode,
      leveler,
      breathControl,
      eqCleanup,
      roomCleanup,
      sceneBlend,
      softenHarshness,
      noiseGuard,
      floorGuard,
      cinematicColor,
      gainPlannerEnabled,
      neuralSpeechEnhancementEnabled: false,
    }),
    [
      loudnessTarget,
      smartMatchMode,
      leveler,
      breathControl,
      eqCleanup,
      roomCleanup,
      sceneBlend,
      softenHarshness,
      noiseGuard,
      floorGuard,
      cinematicColor,
      gainPlannerEnabled,
    ],
  );

  useEffect(() => {
    return () => {
      teardownFfmpeg();
      const assetUrls = ffmpegAssetUrlsRef.current;
      if (!assetUrls) return;
      URL.revokeObjectURL(assetUrls.coreURL);
      URL.revokeObjectURL(assetUrls.wasmURL);
      if (assetUrls.workerURL) {
        URL.revokeObjectURL(assetUrls.workerURL);
      }
      ffmpegAssetUrlsRef.current = null;
    };
  }, []);

  const closeAiReview = useCallback(() => {
    setAiReviewOpen(false);
    window.requestAnimationFrame(() => aiReviewTriggerRef.current?.focus());
  }, []);

  useEffect(() => {
    if (!aiReviewOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeAiReview();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.requestAnimationFrame(() => aiReviewCloseRef.current?.focus());
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [aiReviewOpen, closeAiReview]);

  const appendLog = (message: string) => {
    setLogs((prev) => [...prev.slice(-300), message]);
  };

  const loadStoredReviewWeights = () => {
    if (typeof window === "undefined") return;
    if (!AUTO_LOAD_LOCAL_REVIEW_WEIGHTS) {
      setLearnedReviewWeights(DEFAULT_LEARNED_REVIEW_WEIGHTS);
      setLearnedReviewWeightsSource("default");
      return;
    }
    try {
      const stored = window.localStorage.getItem(REVIEW_WEIGHT_STORAGE_KEY);
      if (!stored) {
        setLearnedReviewWeights(DEFAULT_LEARNED_REVIEW_WEIGHTS);
        setLearnedReviewWeightsSource("default");
        return;
      }
      const parsed = parseLearnedReviewWeights(JSON.parse(stored));
      if (!parsed) {
        window.localStorage.removeItem(REVIEW_WEIGHT_STORAGE_KEY);
        setLearnedReviewWeights(DEFAULT_LEARNED_REVIEW_WEIGHTS);
        setLearnedReviewWeightsSource("default");
        return;
      }
      setLearnedReviewWeights(parsed);
      setLearnedReviewWeightsSource("local-import");
    } catch {
      window.localStorage.removeItem(REVIEW_WEIGHT_STORAGE_KEY);
      setLearnedReviewWeights(DEFAULT_LEARNED_REVIEW_WEIGHTS);
      setLearnedReviewWeightsSource("default");
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!AUTO_LOAD_LOCAL_REVIEW_WEIGHTS) {
      // Auto-load disabled — built-in defaults are already in state.
      // No log line so the Processing Log stays clean.
      return;
    }
    try {
      const stored = window.localStorage.getItem(REVIEW_WEIGHT_STORAGE_KEY);
      if (!stored) return;
      const parsed = parseLearnedReviewWeights(JSON.parse(stored));
      if (!parsed) {
        window.localStorage.removeItem(REVIEW_WEIGHT_STORAGE_KEY);
        return;
      }
      setLearnedReviewWeights(parsed);
      setLearnedReviewWeightsSource("local-import");
      setLogs((prev) => [
        ...prev.slice(-300),
        `[ReviewModel] Loaded local review weights: ${parsed.modelName}.`,
      ]);
    } catch {
      window.localStorage.removeItem(REVIEW_WEIGHT_STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!AUTO_LOAD_LOCAL_REVIEW_WEIGHTS) return;
    const handleStorage = (event: StorageEvent) => {
      if (event.key !== null && event.key !== REVIEW_WEIGHT_STORAGE_KEY) return;
      loadStoredReviewWeights();
      setLogs((prev) => [
        ...prev.slice(-300),
        event.newValue
          ? "[ReviewModel] Local review weights updated from another tab."
          : "[ReviewModel] Local review weights reset to built-in default.",
      ]);
    };
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

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

  const applyReviewWeights = (
    weights: LearnedReviewWeights,
    source: "default" | "local-import",
    shouldPersist: boolean,
  ) => {
    setLearnedReviewWeights(weights);
    setLearnedReviewWeightsSource(source);
    if (typeof window === "undefined") return;
    if (shouldPersist) {
      window.localStorage.setItem(REVIEW_WEIGHT_STORAGE_KEY, JSON.stringify(weights));
    } else {
      window.localStorage.removeItem(REVIEW_WEIGHT_STORAGE_KEY);
    }
  };

  const resetReviewWeights = () => {
    applyReviewWeights(DEFAULT_LEARNED_REVIEW_WEIGHTS, "default", false);
    appendLog("[ReviewModel] Reverted to built-in review weights.");
  };

  const importReviewWeights = async (file: File | null) => {
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseLearnedReviewWeights(JSON.parse(text));
      if (!parsed) {
        throw new Error("Unrecognized review-weights.json file.");
      }
      applyReviewWeights(parsed, "local-import", true);
      appendLog(`[ReviewModel] Imported review weights: ${parsed.modelName}.`);
    } catch (error) {
      appendLog(
        `[ReviewModel] Import failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
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

  const resolveFfmpegAssetUrl = (name: string) =>
    typeof window === "undefined" ? `/ffmpeg/${name}` : new URL(`/ffmpeg/${name}`, window.location.origin).toString();

  const ensureFfmpegAssetUrls = async () => {
    if (ffmpegAssetUrlsRef.current) return ffmpegAssetUrlsRef.current;

    let coreURL: string | null = null;
    let wasmURL: string | null = null;
    let workerURL: string | undefined;

    try {
      coreURL = await toBlobURL(resolveFfmpegAssetUrl("ffmpeg-core.js"), "text/javascript");
      wasmURL = await toBlobURL(resolveFfmpegAssetUrl("ffmpeg-core.wasm"), "application/wasm");

      const assetUrls = {
        coreURL,
        wasmURL,
        ...(workerURL ? { workerURL } : {}),
      };
      ffmpegAssetUrlsRef.current = assetUrls;
      return assetUrls;
    } catch (error) {
      if (coreURL) {
        URL.revokeObjectURL(coreURL);
      }
      if (wasmURL) {
        URL.revokeObjectURL(wasmURL);
      }
      if (workerURL) {
        URL.revokeObjectURL(workerURL);
      }
      throw error;
    }
  };

  const attachFfmpegListeners = (ffmpeg: FFmpeg) => {
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
  };

  const createLoadedFfmpeg = async () => {
    const ffmpeg = new FFmpeg();
    attachFfmpegListeners(ffmpeg);

    setStatus("Loading FFmpeg core...");
    const assetUrls = await ensureFfmpegAssetUrls();

    try {
      await ffmpeg.load(assetUrls);
      setStatus("FFmpeg ready");
      return ffmpeg;
    } catch (error) {
      try {
        ffmpeg.terminate();
      } catch {
        // Ignore terminate failures if the fresh worker never fully booted.
      }
      throw error;
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
      ffmpegLoadPromiseRef.current = null;
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
    const previous = ffmpegRef.current;
    const next = await createLoadedFfmpeg().catch((error) => {
      ffmpegRef.current = previous;
      throw error;
    });

    ffmpegRef.current = next;
    if (previous) {
      try {
        previous.terminate();
      } catch {
        // Ignore terminate failures while rotating worker state.
      }
    }
    logBufferRef.current = [];
    return next;
  };

  const ensureFfmpeg = async () => {
    if (ffmpegRef.current) return ffmpegRef.current;
    if (!ffmpegLoadPromiseRef.current) {
      ffmpegLoadPromiseRef.current = createLoadedFfmpeg()
        .then((ffmpeg) => {
          ffmpegRef.current = ffmpeg;
          return ffmpeg;
        })
        .finally(() => {
          ffmpegLoadPromiseRef.current = null;
        });
    }
    return await ffmpegLoadPromiseRef.current;
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

  const cloneBytes = (bytes: Uint8Array) => {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return copy;
  };

  const readVirtualFileBytes = async (ffmpeg: FFmpeg, name: string) => {
    const data = await ffmpeg.readFile(name);
    if (typeof data === "string") {
      return new TextEncoder().encode(data);
    }
    return cloneBytes(data instanceof Uint8Array ? data : new Uint8Array(data));
  };

  const applyFinalConsonantPeakPolish = async (ffmpeg: FFmpeg, name: string, context: string) => {
    let bytes: Uint8Array | null = null;
    try {
      bytes = await readVirtualFileBytes(ffmpeg, name);
      if (bytes.byteLength > FINAL_CONSONANT_TAMER_MAX_BYTES) {
        appendLog(
          `[FinalPeakTamer] ${context}: skipped (${formatBytes(
            bytes.byteLength,
          )} exceeds ${formatBytes(FINAL_CONSONANT_TAMER_MAX_BYTES)} memory guard).`,
        );
        return;
      }

      const decoded = decodeWav(bytes);
      if (decoded.channels !== 1) {
        appendLog(`[FinalPeakTamer] ${context}: skipped (${decoded.channels} channels; mono VO expected).`);
        return;
      }

      const result = tameRenderedConsonantPeaks(decoded.samples, decoded.sampleRate, GAIN_PLANNER_FRAME_MS);
      if (result.stats.tamedFrameCount <= 0) return;

      await ffmpeg.writeFile(name, encodeWavFloat32(result.samples, decoded.sampleRate, decoded.channels));
      appendLog(
        `[FinalPeakTamer] ${context}: touched ${result.stats.tamedFrameCount} frame${
          result.stats.tamedFrameCount === 1 ? "" : "s"
        } (max ${result.stats.maxReductionDb.toFixed(1)} dB).`,
      );
    } catch (error) {
      appendLog(`[FinalPeakTamer] ${context}: skipped (${describeError(error)}).`);
      if (bytes) {
        try {
          await ffmpeg.writeFile(name, bytes);
        } catch {
          // Keep the original render if the polish write fails.
        }
      }
    }
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

  type AudibilityGuardError = Error & { audibilityGuard: true };
  const isAudibilityGuardError = (error: unknown): error is AudibilityGuardError =>
    error instanceof Error && (error as Partial<AudibilityGuardError>).audibilityGuard === true;

  const summarizeAudibilityReport = (report: AudibilityDropoutReport) => {
    const topClusters = report.clusters
      .slice()
      .sort((a, b) => a.minDropDb - b.minDropDb || b.durationSec - a.durationSec)
      .slice(0, 3)
      .map(
        (cluster) =>
          `${cluster.startSec.toFixed(2)}-${cluster.endSec.toFixed(2)}s/${cluster.minDropDb.toFixed(0)}dB`,
      )
      .join(", ");
    return `${report.badSeconds.toFixed(2)}s collapsed speech, ${report.clusterCount} cluster${
      report.clusterCount === 1 ? "" : "s"
    }, worst ${report.worstDropDb.toFixed(0)} dB${topClusters ? ` (${topClusters})` : ""}`;
  };

  const makeAudibilityGuardError = (context: string, report: AudibilityDropoutReport): AudibilityGuardError => {
    const error = new Error(`${context}: ${summarizeAudibilityReport(report)}`) as AudibilityGuardError;
    error.audibilityGuard = true;
    return error;
  };

  const renderAudibilityFrameDb = async (ffmpeg: FFmpeg, inputName: string, context: string) => {
    const rawName = `${sanitizeBase(inputName)}_${sanitizeBase(context)}_audibility.f32`;
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
        "-ac",
        "1",
        "-ar",
        `${AUDIBILITY_GUARD_SAMPLE_RATE}`,
        "-f",
        "f32le",
        rawName,
      ],
      `Audibility guard decode ${context}`,
    );
    try {
      const bytes = await readVirtualFileBytes(ffmpeg, rawName);
      return frameDbFromFloatSamples(toFloatSamples(bytes), AUDIBILITY_GUARD_SAMPLE_RATE, AUDIBILITY_GUARD_FRAME_MS);
    } finally {
      await safeDeleteFile(ffmpeg, rawName);
    }
  };

  const decodeWavToMono = (bytes: Uint8Array): DecodedMonoAudio => {
    const decoded = decodeWav(bytes);
    const frameCount = Math.floor(decoded.samples.length / Math.max(decoded.channels, 1));
    return {
      sampleRate: decoded.sampleRate,
      channels: decoded.channels,
      durationSec: decoded.sampleRate > 0 ? frameCount / decoded.sampleRate : 0,
      monoSamples: interleavedToMono(decoded.samples, decoded.channels),
    };
  };

  const buildFallbackAlignmentMetrics = (
    sourceAudio: DecodedMonoAudio | null,
    candidateAudio: DecodedMonoAudio | null,
  ): AlignmentMetrics => {
    const durationSourceSec = sourceAudio?.durationSec ?? 0;
    const durationCandidateSec = candidateAudio?.durationSec ?? durationSourceSec;
    const durationDeltaSec = durationCandidateSec - durationSourceSec;
    return {
      durationSourceSec,
      durationCandidateSec,
      durationDeltaSec,
      durationDeltaPct: durationSourceSec > 1e-6 ? (durationDeltaSec / durationSourceSec) * 100 : 0,
      estimatedOffsetSec: 0,
      confidence: 0,
    };
  };

  const buildDurationOnlyAlignmentMetrics = (
    sourceDurationSec: number | null,
    candidateDurationSec: number | null,
  ): AlignmentMetrics => {
    const safeSourceDurationSec = sourceDurationSec && Number.isFinite(sourceDurationSec) ? sourceDurationSec : 0;
    const safeCandidateDurationSec =
      candidateDurationSec && Number.isFinite(candidateDurationSec) ? candidateDurationSec : safeSourceDurationSec;
    const durationDeltaSec = safeCandidateDurationSec - safeSourceDurationSec;
    return {
      durationSourceSec: safeSourceDurationSec,
      durationCandidateSec: safeCandidateDurationSec,
      durationDeltaSec,
      durationDeltaPct: safeSourceDurationSec > 1e-6 ? (durationDeltaSec / safeSourceDurationSec) * 100 : 0,
      estimatedOffsetSec: 0,
      confidence: 0,
    };
  };

  const reviewBlobFromBytes = (bytes: Uint8Array) => {
    return new Blob([cloneBytes(bytes)], {
      type: "audio/wav",
    });
  };

  const reviewNumber = (value: number | null | undefined) =>
    typeof value === "number" && Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null;

  const reviewNumberArray = (values: number[] | null | undefined, maxItems = 8) =>
    Array.isArray(values)
      ? values
          .slice(0, maxItems)
          .map(reviewNumber)
          .filter((value): value is number => value !== null)
      : null;

  const isSmartMatchMode = (value: unknown): value is keyof typeof SMART_MATCH_PRESETS =>
    typeof value === "string" && value in SMART_MATCH_PRESETS;

  const isLevelerPreset = (value: unknown): value is keyof typeof LEVELER_PRESETS =>
    typeof value === "string" && value in LEVELER_PRESETS;

  const isBreathControl = (value: unknown): value is keyof typeof BREATH_COMPAND =>
    typeof value === "string" && value in BREATH_COMPAND;

  const getActiveAudioReviewControls = () => processingControlsOverrideRef.current ?? audioReviewControls;

  const getActiveAudioReviewAdaptiveDirectives = () =>
    aiAdaptiveDirectivesOverrideRef.current ?? DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES;

  const getSourceFirstPlanForJob = (job: JobEntry) =>
    sourceFirstPlansByBaseRef.current.get(job.base) ?? sourceFirstPlansByBaseRef.current.get(job.file.name) ?? null;

  const activateSourceFirstPlanForJob = (job: JobEntry) => {
    const plan = getSourceFirstPlanForJob(job);
    processingControlsOverrideRef.current = plan?.controls ?? null;
    aiAdaptiveDirectivesOverrideRef.current = plan?.adaptiveDirectives ?? null;
    sourceFirstCandidateVariantRef.current = plan?.selectedVariant ?? null;
    return plan;
  };

  const getSmartMatchConfigForControls = (controls: AudioReviewControls) =>
    isSmartMatchMode(controls.smartMatchMode) ? SMART_MATCH_PRESETS[controls.smartMatchMode] : smartMatchConfig;

  const getLevelerForControls = (controls: AudioReviewControls): keyof typeof LEVELER_PRESETS =>
    isLevelerPreset(controls.leveler) ? controls.leveler : leveler;

  const getBreathControlForControls = (controls: AudioReviewControls): keyof typeof BREATH_COMPAND =>
    isBreathControl(controls.breathControl) ? controls.breathControl : breathControl;

  const toAudioReviewMetricSnapshot = (analysis: FileAnalysis | null | undefined): AudioReviewMetricSnapshot => ({
    integratedLufs: reviewNumber(analysis?.inputI),
    instabilityScore: reviewNumber(analysis?.instabilityScore),
    lineSwingScore: reviewNumber(analysis?.lineSwingScore),
    sentenceJumpScore: reviewNumber(analysis?.sentenceJumpScore),
    coldOpenDipDb: reviewNumber(analysis?.coldOpenDipDb),
    coldOpenRiskScore: reviewNumber(analysis?.coldOpenRiskScore),
    midLineSagScore: reviewNumber(analysis?.midLineSagScore),
    endFadeRiskScore: reviewNumber(analysis?.endFadeRiskScore),
    endEdgeDipDb: reviewNumber(analysis?.endEdgeDipDb),
    onsetOvershootScore: reviewNumber(analysis?.onsetOvershootScore),
    breathSpikeRisk: reviewNumber(analysis?.breathSpikeRisk),
    pauseNoiseRisk: reviewNumber(analysis?.pauseNoiseRisk),
    compressionScore: reviewNumber(analysis?.compressionScore),
    clickScore: reviewNumber(analysis?.clickScore),
    echoScore: reviewNumber(analysis?.echoScore),
    roomScore: reviewNumber(analysis?.roomScore),
    sibilanceScore: reviewNumber(analysis?.sibilanceScore),
    noiseFloorDb: reviewNumber(analysis?.noiseFloorDb),
    noiseContrastDb: reviewNumber(analysis?.noiseContrastDb),
    dynamicRangeDb: reviewNumber(analysis?.dynamicRangeDb),
    speechDutyCyclePct: reviewNumber(analysis?.speechDutyCyclePct),
    speechSegmentCount: reviewNumber(analysis?.speechSegmentCount),
    bandSpectrumDb: reviewNumberArray(analysis?.bandSpectrumDb, 8),
  });

  const toAudioReviewProfileSnapshot = (profile: AdaptiveProfile | null): AudioReviewProfileSnapshot | null => {
    if (!profile) return null;
    return {
      highpassHz: reviewNumber(profile.highpassHz),
      lowMidGainDb: reviewNumber(profile.lowMidGainDb),
      presenceGainDb: reviewNumber(profile.presenceGainDb),
      airGainDb: reviewNumber(profile.airGainDb),
      emotionalHarshnessCutDb: reviewNumber(profile.emotionalHarshnessCutDb),
      topEndHarshnessCutDb: reviewNumber(profile.topEndHarshnessCutDb),
      levelingNeed: reviewNumber(profile.levelingNeed),
      emotionProtection: reviewNumber(profile.emotionProtection),
      toneMatchDeltaDb: reviewNumberArray(profile.toneMatchDeltaDb, 8),
      noiseRisk: profile.noiseRisk,
      roomRisk: profile.roomRisk,
      lineContinuityRisk: reviewNumber(profile.lineContinuityRisk),
      preserveEndings: profile.preserveEndings,
      preferSinglePassContinuity: profile.preferSinglePassContinuity,
      useSpeechAlignedSegmentation: profile.useSpeechAlignedSegmentation,
      useSpeechPauseSegmentation: profile.useSpeechPauseSegmentation,
      useDenoise: profile.useDenoise,
      denoiseStrength: reviewNumber(profile.denoiseStrength),
      sibilanceScore: reviewNumber(profile.sibilanceScore),
      onsetTameStrength: reviewNumber(profile.onsetTameStrength),
      sagRecoveryStrength: reviewNumber(profile.sagRecoveryStrength),
      breathTameStrength: reviewNumber(profile.breathTameStrength),
      echoNotchCutDb: reviewNumber(profile.echoNotchCutDb),
      useTailGate: profile.useTailGate,
      cinematicColorEnabled: profile.cinematicColorEnabled,
    };
  };

  const buildAudioReviewFileInput = (
    job: JobEntry,
    analysis: FileAnalysis | null | undefined,
    profile: AdaptiveProfile | null,
    durationSeconds: number | null,
    selectedCandidate: AudioReviewSelectedCandidate | null,
  ): AudioReviewFileInput => ({
    fileName: job.file.name,
    base: job.base,
    durationSeconds: reviewNumber(durationSeconds ?? estimateAudioSeconds(job.file)),
    source: toAudioReviewMetricSnapshot(analysis),
    profile: toAudioReviewProfileSnapshot(profile),
    selectedCandidate,
  });

  const buildSelectedCandidateReviewInput = (
    artifact: CandidateReviewArtifact,
    reason: string | null,
    processingFlow: OutputEntry["processingFlow"],
    issueTags: string[] = [],
  ): AudioReviewSelectedCandidate => ({
    variant: artifact.label,
    reason,
    processingFlow,
    qc: toAudioReviewMetricSnapshot(artifact.analysis),
    qcDelta: artifact.qcDelta as Record<string, number | null> | null,
    alignment: {
      durationDeltaSec: reviewNumber(artifact.alignment.durationDeltaSec),
      estimatedOffsetSec: reviewNumber(artifact.alignment.estimatedOffsetSec),
      confidence: reviewNumber(artifact.alignment.confidence),
    },
    issueTags,
    score: {
      stability: reviewNumber(artifact.scoredScore.stability),
      pause: reviewNumber(artifact.scoredScore.pause),
      compression: reviewNumber(artifact.scoredScore.compression),
      echo: reviewNumber(artifact.scoredScore.echo),
      total: reviewNumber(artifact.scoredScore.rankingScore ?? artifact.scoredScore.total),
    },
  });

  const adaptiveDirectiveDeltaFromDefault = (directives: AudioReviewAdaptiveDirectives) =>
    Object.fromEntries(
      (Object.keys(DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES) as Array<keyof AudioReviewAdaptiveDirectives>).map(
        (key) => [key, directives[key] - DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES[key]],
      ),
    ) as Partial<AudioReviewAdaptiveDirectives>;

  const applyAiReviewProfiles = (
    review: AudioReviewResult,
    source: "manual" | "source-auto",
    options: { activateCurrentRun?: boolean } = {},
  ) => {
    const plans = new Map<string, SourceFirstAudioReviewPlan>();
    for (const fileReview of review.perFileProfiles) {
      const plan = buildSourceFirstAudioReviewPlan(fileReview, audioReviewControls);
      plans.set(fileReview.base, plan);
      plans.set(fileReview.fileName, plan);
    }

    if (options.activateCurrentRun) {
      sourceFirstPlansByBaseRef.current = plans;
    }

    const label = source === "source-auto" ? "AI Auto Pilot" : "AI review";
    const statusText = `${label} selected ${review.perFileProfiles.length} per-audio source-first profile${
      review.perFileProfiles.length === 1 ? "" : "s"
    }.`;
    setAiAutoPilotStatus(statusText);
    appendLog(`[AIReview] ${statusText}`);
    return plans;
  };

  const runAiAudioReview = async (
    reviewFiles: AudioReviewFileInput[],
    options: {
      autoApply?: boolean;
      activateCurrentRun?: boolean;
      source?: "manual" | "source-auto" | "post-render";
      reviewStage?: "source" | "post-render";
    } = {},
  ) => {
    if (aiReviewBusy || reviewFiles.length === 0) return null;
    setAiReviewBusy(true);
    setAiReviewError(null);
    setAiAutoPilotStatus(
      options.source === "source-auto" ? "AI Auto Pilot is reviewing source audio before rendering..." : null,
    );
    try {
      const response = await fetch("/api/audio-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          generatedAt: new Date().toISOString(),
          reviewStage: options.reviewStage ?? (options.source === "post-render" ? "post-render" : "source"),
          controls: getActiveAudioReviewControls(),
          files: reviewFiles,
        }),
      });
      const payload = (await response.json().catch(() => null)) as {
        error?: string;
        model?: string;
        review?: AudioReviewResult;
      } | null;
      if (!response.ok || !payload?.review) {
        throw new Error(payload?.error || `AI review failed (HTTP ${response.status}).`);
      }
      setAiReviewResult(payload.review);
      setAiReviewModel(payload.model ?? null);
      appendLog(
        `[AIReview] ${payload.model ?? "Gemini"}: ${payload.review.verdict} (${Math.round(
          payload.review.confidence * 100,
        )}% confidence).`,
      );
      const plans = options.autoApply
        ? applyAiReviewProfiles(payload.review, options.source === "source-auto" ? "source-auto" : "manual", {
            activateCurrentRun: options.activateCurrentRun,
          })
        : null;
      return { review: payload.review, model: payload.model ?? null, plans };
    } catch (error) {
      setAiReviewError(error instanceof Error ? error.message : String(error));
      setAiAutoPilotStatus(
        options.source === "source-auto"
          ? "AI Auto Pilot source review failed; using current deterministic profile."
          : null,
      );
      return null;
    } finally {
      setAiReviewBusy(false);
    }
  };

  /**
   * A completed gain plan. Holds the gain curve + enough metadata to emit a
   * `sendcmd` script over any time window (full file or per-segment). The
   * actual gain application happens inside ffmpeg via `sendcmd`+`volume`, so
   * memory footprint is tiny regardless of audio length.
   */
  type PlannedGain = {
    /** Linear gain, one value per 10 ms frame. */
    gainCurve: Float32Array;
    frameMs: number;
    sampleRate: number;
    durationSec: number;
    /** Target RMS dB the planner aimed every speech run at. */
    targetDb: number;
    /** Depth below edge speech gain that the planner ducks silences to. */
    expanderDepthDb: number;
    /** Diagnostic count of planned speech runs. */
    speechRunCount: number;
    /** Count of runs classified as transient-breath (subset of speechRunCount). */
    breathRunCount: number;
    /** Count of isolated body-speech spike frames locally tamed. */
    speechSpikeFrameCount: number;
    /** Largest localized body-speech spike reduction in dB. */
    speechSpikeMaxReductionDb: number;
    /** Sustained-loud (onomatopoeia / yell) clusters tamed inside body-speech runs. */
    sustainedLoudClusterCount: number;
    /** Largest uniform attenuation applied to a sustained-loud cluster. */
    sustainedLoudMaxReductionDb: number;
    /** Early dialogue runs capped against later dialogue body. */
    earlyRunCapCount: number;
    /** Largest early-dialogue cap in dB. */
    earlyRunMaxReductionDb: number;
    /** Early dialogue runs lifted toward later dialogue body. */
    coldOpenLiftCount: number;
    /** Largest cold-open lift applied in dB. */
    coldOpenLiftMaxDb: number;
    /** Body-speech runs whose soft post-run tails stayed at speech gain. */
    tailRescueRunCount: number;
    /** Total post-run frames held at speech gain for soft spoken tails. */
    tailRescueFrameCount: number;
    /** Longest soft-tail rescue in milliseconds. */
    tailRescueMaxMs: number;
    /** Effective intra-run micro-ride range in dB. Diagnostic. */
    microRideDb: number;
  };

  /**
   * Decode the full `inputName` to 16 kHz mono Float32, compute the frame
   * envelope, and plan the gain curve. Memory peak is ~8 MB per minute at
   * 16 kHz mono. Returns `null` when the input is too short / has no speech.
   */
  const planGainForInput = async (
    ffmpeg: FFmpeg,
    inputName: string,
    profile: AdaptiveProfile | null,
    analysis: FileAnalysis | undefined,
    durationSeconds: number | null,
  ): Promise<PlannedGain | null> => {
    if (!getActiveAudioReviewControls().gainPlannerEnabled) return null;
    if (durationSeconds !== null && durationSeconds > GAIN_PLANNER_MAX_DURATION_SECONDS) {
      throw new Error(
        `duration ${durationSeconds.toFixed(0)}s exceeds planner budget ${GAIN_PLANNER_MAX_DURATION_SECONDS}s`,
      );
    }

    const wavName = `${sanitizeBase(inputName)}_planner_env.wav`;
    try {
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
          "-ar",
          `${GAIN_PLANNER_ANALYSIS_SAMPLE_RATE}`,
          "-ac",
          "1",
          "-c:a",
          "pcm_f32le",
          wavName,
        ],
        "Gain planner envelope decode",
      );
      const bytes = await readVirtualFileBytes(ffmpeg, wavName);
      const decoded = decodeWav(bytes);
      const samples = decoded.samples;
      if (samples.length < GAIN_PLANNER_ANALYSIS_SAMPLE_RATE) return null;

      const frameSamples = Math.max(
        1,
        Math.round((GAIN_PLANNER_ANALYSIS_SAMPLE_RATE * GAIN_PLANNER_FRAME_MS) / 1000),
      );
      const buildFrameDb = (envelopeSamples: Float32Array) => {
        const frameCount = Math.floor(envelopeSamples.length / frameSamples);
        const frameDb = new Array<number>(frameCount);
        for (let f = 0; f < frameCount; f += 1) {
          let sum = 0;
          const start = f * frameSamples;
          for (let i = 0; i < frameSamples; i += 1) {
            const v = envelopeSamples[start + i];
            sum += v * v;
          }
          const rms = Math.sqrt(sum / frameSamples);
          frameDb[f] = rms <= 0 ? -120 : 20 * Math.log10(rms);
        }
        return frameDb;
      };
      const frameDb = buildFrameDb(samples);
      const loudnessFrameDb = PLANNER_K_WEIGHTING
        ? buildFrameDb(applyKWeighting(samples, GAIN_PLANNER_ANALYSIS_SAMPLE_RATE))
        : undefined;

      const analysisNoiseFloorDb =
        profile?.noiseFloorDb ?? analysis?.pauseNoiseFloorDb ?? analysis?.noiseFloorDb ?? -70;
      const analysisSpeechThresholdDb =
        profile?.speechThresholdDb ?? analysis?.speechThresholdDb ?? analysisNoiseFloorDb + 11;
      const plannerCalibration = resolvePlannerCalibration(
        frameDb,
        analysisNoiseFloorDb,
        analysisSpeechThresholdDb,
      );
      const { noiseFloorDb, speechThresholdDb } = plannerCalibration;
      const pauseNoiseRisk = profile?.pauseNoiseRisk ?? analysis?.pauseNoiseRisk ?? 0;
      if (
        Math.abs(noiseFloorDb - analysisNoiseFloorDb) >= 3 ||
        Math.abs(speechThresholdDb - analysisSpeechThresholdDb) >= 3
      ) {
        appendLog(
          `[Planner] ${sanitizeBase(inputName)}: calibration floor ${analysisNoiseFloorDb.toFixed(
            1,
          )}->${noiseFloorDb.toFixed(1)} dB, threshold ${analysisSpeechThresholdDb.toFixed(
            1,
          )}->${speechThresholdDb.toFixed(1)} dB (envelope floor ${plannerCalibration.envelopeNoiseFloorDb.toFixed(1)} dB).`,
        );
      }

      // Blend of the three signals that describe "messiness". The planner's
      // micro-ride widens on messy sources and narrows on clean ones — on a
      // glass-flat source we want the speech body to come out equally flat,
      // not riding up and down by ±1.5 dB because of local envelope noise.
      const instabilityHint = clamp(
        (profile?.instabilityScore ?? analysis?.instabilityScore ?? 0.5) * 0.5 +
          (profile?.lineSwingScore ?? analysis?.lineSwingScore ?? 0.5) * 0.3 +
          (profile?.sentenceJumpScore ?? analysis?.sentenceJumpScore ?? 0.5) * 0.2,
        0,
        1,
      );
      const speechSpikeTaming = clamp(
        instabilityHint * 0.35 +
          (profile?.lineSwingScore ?? analysis?.lineSwingScore ?? 0) * 0.35 +
          (analysis?.onsetOvershootScore ?? 0) * 0.18 +
          (profile?.clickScore ?? analysis?.clickScore ?? 0) * 0.12,
        0,
        1,
      );
      const measuredNoiseContrast = profile?.noiseContrastDb ?? analysis?.noiseContrastDb ?? null;
      const measuredNoiseFloor = profile?.noiseFloorDb ?? analysis?.noiseFloorDb ?? null;
      const measuredPauseNoiseRisk = profile?.pauseNoiseRisk ?? analysis?.pauseNoiseRisk ?? pauseNoiseRisk;
      const cleanBoostHeadroom = clamp(
        clamp(((measuredNoiseContrast ?? 18) - 26) / 14, 0, 1) *
          clamp((-58 - (measuredNoiseFloor ?? -58)) / 18, 0, 1) *
          clamp((0.28 - measuredPauseNoiseRisk) / 0.28, 0, 1),
        0,
        1,
      );
      const sparseSpeech = (analysis?.speechDutyCyclePct ?? 100) < 10 || (analysis?.speechSegmentCount ?? 999) <= 6;
      const plannerMaxGainDb = 14 + cleanBoostHeadroom * (sparseSpeech ? 4 : 2);
      const directives = getActiveAudioReviewAdaptiveDirectives();
      const headGuardBoost = clamp(directives.headGuardBoost, 0, 1);
      const plannerTargetDb = clamp(
        -22 + directives.targetLoudnessBiasDb + (PLANNER_K_WEIGHTING ? PLANNER_K_TARGET_OFFSET_DB : 0),
        -23.5,
        -20.5,
      );

      const mask = buildSpeechMask(frameDb, noiseFloorDb, { frameMs: GAIN_PLANNER_FRAME_MS });
      const speechRuns: PlannerSpeechRun[] = speechRunsFromMask(mask);
      if (speechRuns.length === 0) return null;

      const plan = planGainCurve({
        frameDb,
        loudnessFrameDb,
        speechRuns,
        noiseFloorDb,
        speechThresholdDb,
        pauseNoiseRisk,
        frameMs: GAIN_PLANNER_FRAME_MS,
        samples,
        sampleRate: GAIN_PLANNER_ANALYSIS_SAMPLE_RATE,
        targetDb: plannerTargetDb,
        sourceTargetBlend: 0.1,
        maxGainDb: plannerMaxGainDb,
        coldOpenLiftToleranceDb: 1.5 - headGuardBoost * 0.5,
        coldOpenLiftMaxDb: 5 + headGuardBoost * 1.5,
        peakCeilingDb: -3,
        instabilityHint,
        speechSpikeTaming,
      });

      const inputDuration = samples.length / GAIN_PLANNER_ANALYSIS_SAMPLE_RATE;
      return {
        gainCurve: plan.gainCurve,
        frameMs: GAIN_PLANNER_FRAME_MS,
        sampleRate: GAIN_PLANNER_ANALYSIS_SAMPLE_RATE,
        durationSec: inputDuration,
        targetDb: plan.targetDb,
        expanderDepthDb: plan.expanderDepthDb,
        speechRunCount: plan.runs.length,
        breathRunCount: plan.breathRunCount,
        speechSpikeFrameCount: plan.speechSpikeFrameCount,
        speechSpikeMaxReductionDb: plan.speechSpikeMaxReductionDb,
        sustainedLoudClusterCount: plan.sustainedLoudClusterCount,
        sustainedLoudMaxReductionDb: plan.sustainedLoudMaxReductionDb,
        earlyRunCapCount: plan.earlyRunCapCount,
        earlyRunMaxReductionDb: plan.earlyRunMaxReductionDb,
        coldOpenLiftCount: plan.coldOpenLiftCount,
        coldOpenLiftMaxDb: plan.coldOpenLiftMaxDb,
        tailRescueRunCount: plan.tailRescueRunCount,
        tailRescueFrameCount: plan.tailRescueFrameCount,
        tailRescueMaxMs: plan.tailRescueMaxMs,
        microRideDb: plan.microRideDb,
      };
    } finally {
      await safeDeleteFile(ffmpeg, wavName);
    }
  };

  /**
   * Chunk size for the planner's full-file apply step. 90 seconds at 48 kHz
   * mono Float32 ≈ 17 MB of samples in memory per chunk.
   *
   * Batch episode files (10 ep × ~2 min = 20 min) or 30-min reels push the
   * WASM linear heap hard. For files ≥ 10 min we drop to 60 s per chunk
   * (≈ 11 MB) which has noticeable headroom for the surrounding filter
   * stages without affecting output quality (chunks are re-concatenated
   * with `-c copy`).
   */
  const PLANNER_APPLY_CHUNK_SECONDS_DEFAULT = 90;
  const PLANNER_APPLY_CHUNK_SECONDS_LONG = 60;
  const PLANNER_APPLY_RETRY_CHUNK_SECONDS: Array<number | null> = [null, 60, 30, 15];
  const LONG_FILE_DURATION_SECONDS = 600; // 10 min
  const LONG_CANDIDATE_QC_SAFE_SECONDS = 1200; // 20 min

  /**
   * Overlap between consecutive render chunks/segments, consumed by a
   * `acrossfade=d=<overlap>` so:
   *   - the output length exactly matches the input (no timing drift), AND
   *   - every boundary is smoothed across ~20 ms, eliminating the 1-sample
   *     clicks that a pure `-c copy` hard-concat can leave when the filter
   *     state differs between segments or when two planner chunks meet at a
   *     frame where the gain curve is stepping (speech/silence edge).
   *
   * 20 ms is short enough to be imperceptible as a fade (one phoneme ~ 80 ms)
   * and is matched exactly by the overlap we read from the source, so timing
   * is preserved sample-accurately.
   */
  const CHUNK_CROSSFADE_SECONDS = 0.02;
  /**
   * Sample rate the leveled full-file is produced at. Matches the downstream
   * mix chain so ffmpeg doesn't have to resample again.
   */
  const PLANNER_APPLY_SAMPLE_RATE = 48000;
  const AUDIBILITY_GUARD_SAMPLE_RATE = 16000;
  const AUDIBILITY_GUARD_FRAME_MS = 20;
  const AUDIBILITY_GUARD_MAX_DURATION_SECONDS = 1800;

  /**
   * Decode a time range of `inputName` to mono Float32 at `PLANNER_APPLY_SAMPLE_RATE`,
   * multiply the samples by the slice of the planner's gain curve that covers
   * that range (with smooth inter-frame interpolation), encode the result as
   * a pcm_f32le WAV, and write it to `outputName` in the ffmpeg virtual FS.
   */
  const levelInputRange = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    plan: PlannedGain,
    startSec: number,
    durationSec: number,
  ) => {
    const rawName = `${outputName}.raw.wav`;
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
        startSec.toFixed(3),
        "-t",
        durationSec.toFixed(3),
        "-i",
        inputName,
        "-ac",
        "1",
        "-ar",
        `${PLANNER_APPLY_SAMPLE_RATE}`,
        "-c:a",
        "pcm_f32le",
        rawName,
      ],
      "Planner apply decode",
    );
    try {
      const bytes = await readVirtualFileBytes(ffmpeg, rawName);
      const decoded = decodeWav(bytes);
      const samples = decoded.samples;

      // Slice the gain curve to just the frames covering [startSec, startSec+durationSec).
      const frameStart = Math.max(0, Math.floor((startSec * 1000) / plan.frameMs));
      const frameEnd = Math.min(
        plan.gainCurve.length,
        Math.ceil(((startSec + durationSec) * 1000) / plan.frameMs),
      );
      const gainSlice = frameStart < frameEnd
        ? plan.gainCurve.slice(frameStart, frameEnd)
        : new Float32Array([1]);

      const leveled = applyGainCurveToSamples(
        samples,
        gainSlice,
        PLANNER_APPLY_SAMPLE_RATE,
        decoded.channels, // should always be 1 given our decode args
        plan.frameMs,
      );
      const consonantPolish = tameRenderedConsonantPeaks(
        leveled,
        PLANNER_APPLY_SAMPLE_RATE,
        plan.frameMs,
      );
      const wav = encodeWavFloat32(consonantPolish.samples, PLANNER_APPLY_SAMPLE_RATE, decoded.channels);
      await ffmpeg.writeFile(outputName, wav);
      return consonantPolish.stats;
    } finally {
      await safeDeleteFile(ffmpeg, rawName);
    }
  };

  /**
   * Apply the planner's gain curve to the full input file, producing a
   * leveled pcm_f32le WAV at `outputName` that the rest of the mix chain can
   * feed from. For long inputs the file is processed in chunks so memory
   * never exceeds one chunk's worth of Float32 samples; the chunks are then
   * stitched with a `-c copy` concat (no re-encode).
   */
  const applyPlannerToFullInput = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    plan: PlannedGain,
    chunkSecondsOverride?: number | null,
  ) => {
    const total = plan.durationSec;
    const chunkSeconds =
      chunkSecondsOverride ??
      (total >= LONG_FILE_DURATION_SECONDS
        ? PLANNER_APPLY_CHUNK_SECONDS_LONG
        : PLANNER_APPLY_CHUNK_SECONDS_DEFAULT);
    const consonantPolishStats = { tamedFrameCount: 0, maxReductionDb: 0 };
    const addConsonantPolishStats = (stats: { tamedFrameCount: number; maxReductionDb: number }) => {
      consonantPolishStats.tamedFrameCount += stats.tamedFrameCount;
      consonantPolishStats.maxReductionDb = Math.max(consonantPolishStats.maxReductionDb, stats.maxReductionDb);
    };
    const logConsonantPolishStats = () => {
      if (consonantPolishStats.tamedFrameCount <= 0) return;
      appendLog(
        `[Planner] ${sanitizeBase(inputName)}: full-rate consonant peak tamer touched ${
          consonantPolishStats.tamedFrameCount
        } frame${consonantPolishStats.tamedFrameCount === 1 ? "" : "s"} (max ${consonantPolishStats.maxReductionDb.toFixed(1)} dB).`,
      );
    };
    if (total <= chunkSeconds) {
      addConsonantPolishStats(await levelInputRange(ffmpeg, inputName, outputName, plan, 0, total));
      logConsonantPolishStats();
      return;
    }

    // Plan native (non-overlapping) chunk boundaries, then extend each
    // non-last chunk by CHUNK_CROSSFADE_SECONDS of source material at its
    // END so the downstream `acrossfade=d=CHUNK_CROSSFADE_SECONDS` consumes
    // the duplicate region — sample-accurate timing, click-free seams.
    const nativeStarts: number[] = [];
    const nativeSpans: number[] = [];
    {
      let cursor = 0;
      while (cursor < total - 0.01) {
        const span = Math.min(chunkSeconds, total - cursor);
        nativeStarts.push(cursor);
        nativeSpans.push(span);
        cursor += span;
      }
    }

    const chunkNames: string[] = [];
    try {
      for (let index = 0; index < nativeStarts.length; index += 1) {
        const isLast = index === nativeStarts.length - 1;
        const start = nativeStarts[index];
        const span = isLast
          ? nativeSpans[index]
          : Math.min(nativeSpans[index] + CHUNK_CROSSFADE_SECONDS, total - start);
        const chunkName = `${sanitizeBase(outputName)}_chunk_${index}.wav`;
        addConsonantPolishStats(await levelInputRange(ffmpeg, inputName, chunkName, plan, start, span));
        chunkNames.push(chunkName);
      }

      await runCrossfadeConcat(
        ffmpeg,
        chunkNames,
        outputName,
        CHUNK_CROSSFADE_SECONDS,
        "Planner apply crossfade",
      );
      await logDurationDelta(ffmpeg, "Planner apply crossfade", total, outputName);
      logConsonantPolishStats();
    } finally {
      for (const name of chunkNames) {
        await safeDeleteFile(ffmpeg, name);
      }
    }
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
    coldOpenDipDb: null,
    coldOpenRiskScore: null,
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
    endEdgeDipDb: null,
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
    coldOpenDipDb: metrics.coldOpenDipDb,
    coldOpenRiskScore: metrics.coldOpenRiskScore,
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
    endEdgeDipDb: metrics.endEdgeDipDb,
  });

  const computeEnvelopeMetrics = (samples: Float32Array) => {
    const frameSize = Math.max(1, Math.round((ANALYSIS_SAMPLE_RATE * ENVELOPE_FRAME_MS) / 1000));
    const frameCount = Math.floor(samples.length / frameSize);
    if (frameCount < 20) {
      return { ...emptyEnvelopeMetrics, bandSpectrumDb: null, sibilanceScore: null };
    }
    const base = mapQcMetricsToEnvelopeMetrics(
      analyzeFloatSamples(samples, ANALYSIS_SAMPLE_RATE, ENVELOPE_FRAME_MS),
    );
    // Compute long-term band spectrum + sibilance directly from the samples.
    // This is new data (not present in `emptyEnvelopeMetrics`) so we widen
    // the return type locally.
    let bandSpectrumDb: number[] | null = null;
    let sibilanceScore: number | null = null;
    if (samples.length >= ANALYSIS_SAMPLE_RATE) {
      try {
        bandSpectrumDb = computeLogBandSpectrumDb(samples, ANALYSIS_SAMPLE_RATE);
        sibilanceScore = computeSibilanceScore(bandSpectrumDb);
      } catch {
        bandSpectrumDb = null;
        sibilanceScore = null;
      }
    }
    return { ...base, bandSpectrumDb, sibilanceScore };
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
      analysis.coldOpenDipDb = envelope.coldOpenDipDb;
      analysis.coldOpenRiskScore = envelope.coldOpenRiskScore;
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
      analysis.endEdgeDipDb = envelope.endEdgeDipDb;
      analysis.bandSpectrumDb = envelope.bandSpectrumDb;
      analysis.sibilanceScore = envelope.sibilanceScore;
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
    aggregated.coldOpenDipDb = baseAnalysis.coldOpenDipDb;
    aggregated.coldOpenRiskScore = baseAnalysis.coldOpenRiskScore;
    aggregated.breathSpikeRisk = weightedMetric(windowAnalyses, (a) => a.breathSpikeRisk, 85);
    aggregated.pauseNoiseRisk = weightedMetric(windowAnalyses, (a) => a.pauseNoiseRisk, 80);
    aggregated.compressionScore = weightedMetric(windowAnalyses, (a) => a.compressionScore, 70);
    aggregated.overallRisk = weightedMetric(windowAnalyses, (a) => a.overallRisk, 65);
    aggregated.clickScore = weightedMetric(windowAnalyses, (a) => a.clickScore, 70);
    aggregated.onsetOvershootScore = weightedMetric(windowAnalyses, (a) => a.onsetOvershootScore, 85);
    aggregated.midLineSagScore = weightedMetric(windowAnalyses, (a) => a.midLineSagScore, 80);
    aggregated.endFadeRiskScore = weightedMetric(windowAnalyses, (a) => a.endFadeRiskScore, 85);
    aggregated.endEdgeDipDb = weightedMetric(windowAnalyses, (a) => a.endEdgeDipDb, 85);
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
      (aggregated as Record<string, number | number[] | boolean | null>)[key] =
        (baseAnalysis as Record<string, number | number[] | boolean | null>)[key];
    }

    return aggregated;
  };

  type AnalysisOptions = {
    maxDistributedWindows?: number;
    limitReason?: string;
  };

  function limitAnalysisWindows<T>(windows: T[], maxDistributedWindows?: number): T[] {
    if (!maxDistributedWindows || maxDistributedWindows <= 0 || windows.length <= maxDistributedWindows) {
      return windows;
    }
    if (maxDistributedWindows === 1) return [windows[Math.floor(windows.length / 2)]];
    const selected: T[] = [];
    const last = windows.length - 1;
    for (let index = 0; index < maxDistributedWindows; index += 1) {
      selected.push(windows[Math.round((index / (maxDistributedWindows - 1)) * last)]);
    }
    return selected;
  }

  const analyzeFile = async (
    ffmpeg: FFmpeg,
    inputName: string,
    recoveryInputNames: string[] = [],
    options: AnalysisOptions = {},
  ): Promise<AnalysisResult> => {
    let durationSeconds: number | null = null;
    const recoveryInputs = Array.from(new Set([inputName, ...recoveryInputNames]));
    const recoveryInputBytes = new Map<string, Uint8Array>();
    const ensureRecoveryInputBytes = async (name: string) => {
      const cached = recoveryInputBytes.get(name);
      if (cached) return cached;
      const bytes = await readVirtualFileBytes(ffmpeg, name);
      recoveryInputBytes.set(name, bytes);
      return bytes;
    };
    const restoreRecoveryInputs = async (target: FFmpeg) => {
      for (const name of recoveryInputs) {
        await target.writeFile(name, await ensureRecoveryInputBytes(name));
      }
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
    baseAnalysis.analysisWindowsAttempted = 0;
    baseAnalysis.analysisWindowsSucceeded = 0;
    baseAnalysis.analysisWindowsDropped = 0;
    baseAnalysis.analysisWindowRetryCount = 0;

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
      const capLongSparseWindows = speechStats.longSparseModeEligible && speechStats.speechDutyCyclePct < 6;
      const distributedWindowCount = speechStats.longSparseModeEligible
        ? capLongSparseWindows
          ? 5
          : LONG_SPARSE_ANALYSIS_WINDOW_TARGET_COUNT
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
      const analysisWindows = limitAnalysisWindows(windows, options.maxDistributedWindows);
      appendLog(
        `[Analysis] ${sanitizeBase(inputName)}: distributed coverage on (speech-duty ${speechStats.speechDutyCyclePct.toFixed(
          1
        )}%, median-run ${(speechStats.medianSpeechRunMs / 1000).toFixed(1)}s, windows ${analysisWindows.length}${
          analysisWindows.length < windows.length ? `/${windows.length}` : ""
        }${
          speechStats.longSparseModeEligible ? ", sparse-mode" : ""
        }${options.limitReason ? `, ${options.limitReason}` : ""}).`
      );

      const windowAnalyses: Array<{ analysis: FileAnalysis; weight: number }> = [];
      let windowRetryCount = 0;
      let windowDropCount = 0;
      baseAnalysis.analysisWindowsAttempted = analysisWindows.length;
      for (const window of analysisWindows) {
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
              windowRetryCount += 1;
              ffmpeg = await refreshFfmpeg(
                `analysis window retry on ${sanitizeBase(inputName)} @ ${window.startSec.toFixed(1)}s`
              );
              await restoreRecoveryInputs(ffmpeg);
              continue;
            }
            windowDropCount += 1;
            break;
          }
        }
      }
      baseAnalysis.analysisWindowsSucceeded = windowAnalyses.length;
      baseAnalysis.analysisWindowsDropped = windowDropCount;
      baseAnalysis.analysisWindowRetryCount = windowRetryCount;

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

  const buildBatchReference = (analyses: FileAnalysis[]): BatchReference | null => {
    const lowTilts: number[] = [];
    const highTilts: number[] = [];
    const lras: number[] = [];
    const perBandSamples: number[][] = Array.from({ length: SPECTRUM_BANDS_HZ.length }, () => []);
    const referenceWeight = (analysis: FileAnalysis) => {
      const confidence = clamp(analysis.analysisConfidence ?? 0.35, 0, 1);
      const roomPenalty = clamp((analysis.roomScore ?? 0) * 0.5 + (analysis.echoScore ?? 0) * 0.35, 0, 0.85);
      const noisePenalty = clamp((analysis.pauseNoiseRisk ?? 0) * 0.45, 0, 0.45);
      const compressionPenalty = clamp((analysis.compressionScore ?? 0) * 0.35, 0, 0.35);
      return clamp(confidence - roomPenalty - noisePenalty - compressionPenalty, 0, 1);
    };
    const cleanAnchors = analyses.filter((analysis) => referenceWeight(analysis) >= 0.45);
    const referenceAnalyses = cleanAnchors.length > 0 ? cleanAnchors : analyses;

    for (const analysis of referenceAnalyses) {
      const weight = referenceWeight(analysis);
      if (weight < 0.2) continue;
      const repeats = weight >= 0.75 ? 3 : weight >= 0.45 ? 2 : 1;
      if (analysis.lowRms !== null && analysis.midRms !== null) {
        for (let i = 0; i < repeats; i += 1) lowTilts.push(analysis.lowRms - analysis.midRms);
      }
      if (analysis.highRms !== null && analysis.midRms !== null) {
        for (let i = 0; i < repeats; i += 1) highTilts.push(analysis.highRms - analysis.midRms);
      }
      if (analysis.inputLRA !== null) {
        for (let i = 0; i < repeats; i += 1) lras.push(analysis.inputLRA);
      }
      if (analysis.bandSpectrumDb && analysis.bandSpectrumDb.length === SPECTRUM_BANDS_HZ.length) {
        for (let b = 0; b < SPECTRUM_BANDS_HZ.length; b += 1) {
          const value = analysis.bandSpectrumDb[b];
          if (Number.isFinite(value)) {
            for (let i = 0; i < repeats; i += 1) perBandSamples[b].push(value);
          }
        }
      }
    }

    const lowTilt = robustMedian(lowTilts);
    const highTilt = robustMedian(highTilts);
    const lra = robustMedian(lras);

    const bandMedians: number[] | null = perBandSamples.every((arr) => arr.length > 0)
      ? perBandSamples.map((arr) => robustMedian(arr) ?? 0)
      : null;

    if (lowTilt === null && highTilt === null && lra === null && bandMedians === null) return null;

    return {
      lowTilt: lowTilt ?? -11,
      highTilt: highTilt ?? -13,
      lra: lra ?? 6,
      bandSpectrumDb: bandMedians,
      anchorCount: referenceAnalyses.length,
    };
  };

  const hasMeasuredNoiseProblem = (
    noiseRisk: NoiseRisk,
    noiseFloorDb: number | null,
    noiseContrastDb: number | null,
    pauseNoiseRisk: number,
  ) =>
    pauseNoiseRisk >= 0.28 ||
    (noiseContrastDb !== null && noiseContrastDb < 18) ||
    (noiseFloorDb !== null && noiseFloorDb > -58) ||
    (noiseRisk === "high" && ((noiseFloorDb ?? -70) > -62 || pauseNoiseRisk >= 0.18));

  const buildAdaptiveProfile = (analysis: FileAnalysis | undefined, reference: BatchReference | null) => {
    const controls = getActiveAudioReviewControls();
    const directives = getActiveAudioReviewAdaptiveDirectives();
    const activeSmartMatchConfig = getSmartMatchConfigForControls(controls);
    const needsAdaptiveProfile =
      activeSmartMatchConfig.tone > 0 ||
      activeSmartMatchConfig.dynamics > 0 ||
      controls.roomCleanup ||
      controls.sceneBlend;
    if (!needsAdaptiveProfile || !analysis) return null;

    const smartToneEnabled = activeSmartMatchConfig.tone > 0;
    const smartDynamicsEnabled = activeSmartMatchConfig.dynamics > 0;

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

    const toneFactor = smartToneEnabled ? activeSmartMatchConfig.tone : 0;
    const dynamicsFactor = smartDynamicsEnabled ? activeSmartMatchConfig.dynamics : 0;

    let highpassHz = Math.round(clamp(80 + lowTiltDiff * 2.2 * toneFactor, 65, 105));
    const lowMidGainDb = clamp(-2 - lowTiltDiff * 0.28 * toneFactor + directives.warmthDb, -3.8, 1.6);

    let presenceGainDb = clamp(-highTiltDiff * 0.45 * toneFactor + directives.presenceDb, -2.8, 2.2);
    let airGainDb = clamp(-highTiltDiff * 0.25 * toneFactor + directives.airDb, -2.0, 1.3);

    const lra = analysis.inputLRA ?? referenceLra;
    const lraDiff = lra - referenceLra;
    const compressorRatioOffset = clamp(
      lraDiff * 0.07 * dynamicsFactor + directives.compressionBias * 0.3,
      -0.4,
      0.45,
    );
    const compressorThresholdOffsetDb = clamp(
      lraDiff * 0.6 * dynamicsFactor + directives.compressionBias * 0.8,
      -1.5,
      1.5,
    );
    const levelerBias = clamp(directives.levelerBias, -0.5, 0.5);

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
      (hotPeakFactor * 0.95 + brightFactor * 0.7) * toneFactor + directives.deHarshDb,
      0,
      2.2
    );
    const topEndHarshnessCutDb = clamp(emotionalHarshnessCutDb * 0.75 + directives.deHarshDb * 0.3, 0, 1.8);

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
      controls.roomCleanup && (analysisConfidence >= 0.4 || echoScore >= 0.58 || roomRisk !== "low");
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
      endFadeRiskScore >= 0.38 ||
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
      Math.max(onsetOvershootScore, breathSpikeRisk * 0.85) * (noiseRisk === "low" ? 1 : 0.72) +
        directives.onsetTameBoost,
      0,
      1
    );
    const breathTameStrength = clamp(
      breathSpikeRisk * (noiseRisk === "high" ? 0.55 : noiseRisk === "medium" ? 0.78 : 1) +
        directives.breathTameBoost,
      0,
      1
    );
    const sagRecoveryStrength = clamp(
      midLineSagScore * (noiseRisk === "high" ? 0.5 : 1) + directives.sagRecoveryBoost,
      0,
      1,
    );
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
    const dryness = analysis.drynessScore ?? clamp(1 - confidenceScaledRoom, 0, 1);

    const plannedEchoNotchCutDb = roomCleanupEnabled
      ? clamp(
          echoScore *
            (roomRisk === "high" && echoScore >= 0.68 ? 1.35 : roomRisk === "high" ? 1.02 : roomRisk === "medium" ? 0.68 : 0.42) +
            (roomRisk === "high" ? (echoScore >= 0.72 ? 0.27 : 0.12) : 0) +
            directives.roomCleanupBias * 0.55,
          0,
          1.95
        )
      : 0;
    const echoRoomCleanup = buildEchoRoomCleanupDecision({
      roomCleanupEnabled,
      roomRisk,
      echoScore,
      analysisConfidence,
      preserveEndings,
      pauseNoiseRisk,
      echoNotchCutDb: plannedEchoNotchCutDb,
      endFadeRiskScore,
      lineContinuityRisk,
    });
    if (echoRoomCleanup.severeEchoRoom) {
      highpassHz = Math.max(highpassHz, 88);
    }
    const measuredEchoDelayMs =
      typeof analysis.echoDelayMs === "number" && Number.isFinite(analysis.echoDelayMs)
        ? analysis.echoDelayMs
        : null;
    const earlyEchoCancelStrength = deriveEarlyEchoCancelStrength({
      severeEchoRoom: echoRoomCleanup.severeEchoRoom,
      echoScore,
      drynessScore: dryness,
      echoDelayMs: measuredEchoDelayMs,
      roomCleanupBias: directives.roomCleanupBias,
    });
    const forceTailGateForEcho =
      roomCleanupEnabled &&
      roomRisk === "high" &&
      echoScore >= 0.62 &&
      endFadeRiskScore < 0.36 &&
      lineContinuityRisk < 0.55 &&
      (!preserveEndings || echoRoomCleanup.allowTailGateDuringEndingProtection);
    const useTailGate =
      roomCleanupEnabled &&
      (!preserveEndings || echoRoomCleanup.allowTailGateDuringEndingProtection) &&
      (forceTailGateForEcho ||
        (analysisConfidence >= 0.52 && (roomRisk === "high" || (roomRisk === "medium" && echoScore >= 0.5))));
    // `useDenoise` / `denoiseStrength` are now *derived* from the same signal
    // the NR filter uses, so logs and QC see a meaningful value. The actual
    // filter chain is built in `buildAdaptiveNoiseReductionFilter`.
    const measuredNoiseContrast = analysis.noiseContrastDb ?? null;
    const measuredNoiseNeedsNr = hasMeasuredNoiseProblem(
      noiseRisk,
      measuredNoiseFloor,
      measuredNoiseContrast,
      pauseNoiseRisk,
    );
    const nrContrastPenalty =
      measuredNoiseContrast !== null ? clamp((20 - measuredNoiseContrast) / 12, 0, 1) : 0;
    const nrBandFactor =
      measuredNoiseNeedsNr ? (noiseRisk === "high" ? 1 : noiseRisk === "medium" ? 0.55 : 0.1) : 0;
    const nrRoomFactor = measuredNoiseNeedsNr ? (roomRisk === "high" ? 0.18 : roomRisk === "medium" ? 0.08 : 0) : 0;
    const denoiseStrength = clamp(
      pauseNoiseRisk * 0.45 + nrContrastPenalty * 0.3 + nrBandFactor * 0.2 + nrRoomFactor,
      0,
      1,
    );
    const directedDenoiseStrength = clamp(denoiseStrength + directives.denoiseBias, 0, 1);
    const useDenoise = controls.noiseGuard && (measuredNoiseNeedsNr || directives.denoiseBias > 0.18) && directedDenoiseStrength >= 0.22;
    const tailGateStrength = !useTailGate
      ? 0
      : roomRisk === "high"
        ? clamp(0.09 + echoScore * 0.1 + analysisConfidence * 0.06 + directives.roomCleanupBias * 0.04, 0.09, 0.24)
        : clamp(0.06 + echoScore * 0.08 + directives.roomCleanupBias * 0.03, 0.06, 0.16);
    const echoNotchCutDb = plannedEchoNotchCutDb;

    const baseDynaTrim = controls.noiseGuard ? (noiseRisk === "high" ? 3 : noiseRisk === "medium" ? 2 : 0) : 0;
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

    const blendRiskDamp = roomRisk === "high" ? 0.012 : roomRisk === "medium" ? 0.12 : 1;
    const blendEchoDamp = clamp(1 - echoScore * 0.85, 0.08, 1);
    const blendNoiseDamp = noiseRisk === "high" ? 0.22 : noiseRisk === "medium" ? 0.55 : 1;
    const blendInstabilityDamp = instabilityScore >= 0.7 ? 0.65 : 1;
    const blendConfidenceScale = clamp(0.35 + analysisConfidence * 0.65, 0.35, 1);
    const blendBase = clamp(0.018 + dryness * 0.018, 0.012, 0.036);
    let blendAmount = controls.sceneBlend
      ? blendBase * blendRiskDamp * blendConfidenceScale * blendEchoDamp * blendNoiseDamp * blendInstabilityDamp
      : 0;
    if (roomRisk === "high" || echoScore >= 0.72) {
      blendAmount = Math.min(blendAmount, 0.0012);
    }
    const blendIndoorGain = clamp(blendAmount * 0.62, 0, 0.07);
    const blendOutdoorGain = clamp(blendAmount * 0.42, 0, 0.055);
    const blendIndoorDelayMs = Math.round(clamp(24 + (1 - dryness) * 8, 22, 36));
    const blendOutdoorDelayMs = Math.round(clamp(52 + (1 - dryness) * 18, 48, 74));

    // Compute per-band tone delta vs the batch reference (if we have both).
    const toneMatchDeltaDb =
      reference?.bandSpectrumDb && reference.anchorCount >= 3 && analysis.bandSpectrumDb
        ? computeToneMatchDeltaDb(analysis.bandSpectrumDb, reference.bandSpectrumDb, {
            maxDb: 2.5,
            houseBlend: HOUSE_TONE_BLEND,
            priorityBandCount: 2,
            priorityMaxDb: 3,
          })
        : null;

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
      levelerBias,
      dynaTrim,
      floorGuardFilter: noiseRisk === "high" ? FLOOR_GUARD_STRONG : FLOOR_GUARD,
      noiseRisk,
      noiseFloorDb: measuredNoiseFloor,
      noiseContrastDb: measuredNoiseContrast,
      pauseNoiseRisk,
      speechThresholdDb: measuredSpeechThreshold,
      roomRisk,
      useDenoise,
      denoiseStrength: directedDenoiseStrength,
      bandSpectrumDb: analysis.bandSpectrumDb ?? null,
      toneMatchDeltaDb,
      sibilanceScore: analysis.sibilanceScore ?? 0,
      cinematicColorEnabled: controls.cinematicColor,
      useTailGate,
      tailGateStrength,
      echoNotchCutDb,
      echoDominantRoom: echoRoomCleanup.echoDominantRoom,
      severeEchoRoom: echoRoomCleanup.severeEchoRoom,
      allowDereverbDuringEndingProtection: echoRoomCleanup.allowDereverbDuringEndingProtection,
      allowTailGateDuringEndingProtection: echoRoomCleanup.allowTailGateDuringEndingProtection,
      echoDelayMs: measuredEchoDelayMs,
      earlyEchoCancelStrength,
      echoScore,
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
      longSparseModeEligible: !!analysis.longSparseModeEligible,
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

  const buildTailGateFilter = (strength: number, severeEchoRoom = false) => {
    if (severeEchoRoom) {
      const thresholdDb = clamp(-52 + strength * 42, -52, -41);
      const threshold = fromDb(thresholdDb);
      const ratio = clamp(1.25 + strength * 4.4, 1.25, 2.25);
      const range = clamp(0.9 - strength * 0.8, 0.68, 0.9);
      const attack = Math.round(clamp(18 - strength * 18, 12, 20));
      const release = Math.round(clamp(360 - strength * 420, 210, 380));
      return `agate=mode=downward:threshold=${threshold.toFixed(5)}:ratio=${ratio.toFixed(
        2
      )}:range=${range.toFixed(3)}:attack=${attack}:release=${release}:makeup=1.00:detection=rms:link=average`;
    }

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

  /**
   * Build the adaptive noise-reduction filter chain.
   *
   * Strength is driven by *measured* SNR (`noiseContrastDb`), pause-noise risk,
   * and room risk — not just the coarse `noiseRisk` band. On clean takes we
   * return `null` (no NR) so we never scrub tone off a healthy voice. On bad
   * inputs we chain `afftdn` (spectral subtraction) with an optional
   * `anlmdn` (non-local means) second stage for stubborn hiss.
   *
   * Returns a full filter-chain fragment (comma-joined) or null.
   */
  const buildAdaptiveNoiseReductionFilter = (
    noiseRisk: NoiseRisk,
    noiseFloorDb: number | null,
    noiseContrastDb: number | null,
    pauseNoiseRisk: number,
    roomRisk: RoomRisk,
  ) => {
    // Composite "need" score drives how aggressive NR should be.
    // - pauseNoiseRisk: how noisy the pauses are (0..1).
    // - noiseContrastDb: SNR between speech and pauses. <14 dB = poor.
    // - roomRisk: reverb/echo bed. Used only after real noise evidence exists.
    // - noiseRisk band: coarse backup if metrics missing.
    const measuredNoiseNeedsNr = hasMeasuredNoiseProblem(noiseRisk, noiseFloorDb, noiseContrastDb, pauseNoiseRisk);
    if (!measuredNoiseNeedsNr) return null;

    const contrastPenalty = noiseContrastDb !== null ? clamp((20 - noiseContrastDb) / 12, 0, 1) : 0;
    const bandFactor = noiseRisk === "high" ? 1 : noiseRisk === "medium" ? 0.55 : 0.1;
    const roomFactor = roomRisk === "high" ? 0.18 : roomRisk === "medium" ? 0.08 : 0;
    const needScore = clamp(
      pauseNoiseRisk * 0.45 + contrastPenalty * 0.3 + bandFactor * 0.2 + roomFactor,
      0,
      1,
    );

    // Under ~0.26 the source is already clean; leaving NR off preserves tone.
    if (needScore < 0.26) return null;

    const estimatedFloor = noiseFloorDb ?? (noiseRisk === "high" ? -49 : -55);
    // `nf` anchored a few dB above the measured pause floor so afftdn knows
    // what to treat as noise.
    const nf = clamp(estimatedFloor + 4, -56, -32);
    // `nr` (reduction dB) is capped conservatively to avoid metallic VO.
    const nr = Math.round(clamp(5 + needScore * 12, 5, 17));
    // `ad` (adaptive-decay speed) — stronger needs faster tracking.
    const ad = clamp(0.22 + needScore * 0.32, 0.22, 0.54);
    // `gs` (gain shape) — smoother curve on severe noise.
    const gs = Math.round(clamp(6 + needScore * 6, 6, 12));
    const stages: string[] = [`afftdn=nf=${nf.toFixed(1)}:nr=${nr}:tn=1:ad=${ad.toFixed(2)}:gs=${gs}`];

    // Severe-noise second pass — non-local means removes broadband hiss that
    // spectral subtraction leaves behind, without the metallic artifacts of
    // aggressive `afftdn`.
    const severeNoise =
      pauseNoiseRisk >= 0.68 ||
      (noiseContrastDb !== null && noiseContrastDb < 11) ||
      (noiseRisk === "high" && estimatedFloor > -44);
    if (severeNoise) {
      // Short temporal radius (r=0.006 s) keeps consonants sharp.
      stages.push("anlmdn=s=0.0003:p=0.002:r=0.006");
    }

    return stages.join(",");
  };

  type MixRenderOptions = {
    disableRoomCleanup?: boolean;
    disableAdaptiveNoiseReduction?: boolean;
    minimalStabilityChain?: boolean;
    disableLimiter?: boolean;
    disableSegmentGainMatch?: boolean;
    segmentBoundaryPadInMs?: number;
    segmentBoundaryPadOutMs?: number;
    trimSegmentPadMs?: number;
    segmentLite?: boolean;
    maxProcessedSpeechSegments?: number;
    mergePauseThresholdSec?: number;
    segmentMode?: "fixed" | "speech-aligned" | "speech-pause";
    candidateVariant?: "cinematic-stable" | "continuity-safe" | "pause-safe" | "source-safe";
    sourceSafeChain?: boolean;
    sourcePassthroughChain?: boolean;
    skipSpeechSegmentation?: boolean;
    disableGainPlanner?: boolean;
    disableTailGate?: boolean;
    disableSpikeTamers?: boolean;
    disableHeadPriming?: boolean;
    forceEndingProtection?: boolean;
    /**
     * When true, the input the chain is about to process has already been
     * leveled by the speech-aware gain planner. The downstream `dynaudnorm`
     * is downgraded to a gentle safety pass (tight window, low `g/m`) and
     * `acompressor` is relaxed to glue duty only. This is what actually
     * kills sentence-to-sentence jumps.
     */
    gainPlannerActive?: boolean;
  };

  const resolveAdaptiveNoiseReductionFilter = (
    profile: AdaptiveProfile | null,
    options?: MixRenderOptions
  ) => {
    if (!getActiveAudioReviewControls().noiseGuard || !profile) return null;
    if (options?.minimalStabilityChain || options?.sourceSafeChain || options?.disableAdaptiveNoiseReduction) return null;
    return buildAdaptiveNoiseReductionFilter(
      profile.noiseRisk,
      profile.noiseFloorDb,
      profile.noiseContrastDb,
      profile.pauseNoiseRisk,
      profile.roomRisk,
    );
  };

  const buildMixFilter = (profile: AdaptiveProfile | null, options?: MixRenderOptions) => {
    const controls = getActiveAudioReviewControls();
    const activeLeveler = getLevelerForControls(controls);
    const activeBreathControl = getBreathControlForControls(controls);
    const filters: string[] = [];
    const levelerSettings = LEVELER_PRESETS[activeLeveler];
    const consistency = LEVELER_CONSISTENCY[activeLeveler];
    const minimalStabilityChain = options?.minimalStabilityChain === true;
    const candidateVariant = options?.candidateVariant ?? "cinematic-stable";
    const continuitySafeMode = candidateVariant === "continuity-safe";
    const pauseSafeMode = candidateVariant === "pause-safe";
    const sourceSafeMode = options?.sourceSafeChain === true || candidateVariant === "source-safe";
    const disableTailGate = options?.disableTailGate === true;
    const disableSpikeTamers = options?.disableSpikeTamers === true;
    if (options?.sourcePassthroughChain) {
      return options?.disableLimiter ? "anull" : LIMITER_FILTER;
    }
    const dyn = sourceSafeMode ? null : levelerSettings.dyna;
    const gainPlannerActive = options?.gainPlannerActive === true;
    const roomCleanupEnabled =
      controls.roomCleanup && !options?.disableRoomCleanup && !minimalStabilityChain && !sourceSafeMode;
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
    const levelerBias = minimalStabilityChain ? 0 : clamp(profile?.levelerBias ?? 0, -0.5, 0.5);
    const levelerAdaptationScale = 1 + levelerBias * 0.2;
    const useClickTamer =
      !minimalStabilityChain &&
      !disableSpikeTamers &&
      !sourceSafeMode &&
      clickTameStrength >= (continuitySafeMode ? 0.38 : pauseSafeMode ? 0.42 : 0.46);
    const useOnsetTamer =
      !minimalStabilityChain &&
      !disableSpikeTamers &&
      !sourceSafeMode &&
      onsetTameStrength >= (continuitySafeMode ? 0.24 : 0.35);
    const useBreathSpikeTamer =
      !minimalStabilityChain &&
      !disableSpikeTamers &&
      !sourceSafeMode &&
      activeBreathControl !== "Off" &&
      breathTameStrength >= (continuitySafeMode ? 0.18 : 0.24);

    if (controls.eqCleanup) {
      const highpassHz = profile?.highpassHz ?? 80;
      const lowMidGainDb = sourceSafeMode ? clamp(profile?.lowMidGainDb ?? -0.8, -1.2, 0) : (profile?.lowMidGainDb ?? -2);
      filters.push(`highpass=f=${highpassHz}`);
      filters.push(`equalizer=f=250:width_type=q:width=1.0:g=${lowMidGainDb.toFixed(2)}`);
      if (useAdaptiveNoiseReduction && adaptiveNoiseReductionFilter) {
        // Run denoise before dynamic stages so levelers do not lift room bed/hiss.
        filters.push(adaptiveNoiseReductionFilter);
      }
    }
    if (!controls.eqCleanup && useAdaptiveNoiseReduction && adaptiveNoiseReductionFilter) {
      filters.push(adaptiveNoiseReductionFilter);
    }

    // Subtle cinematic de-reverb. Only fires when the analyzer detects a
    // real room signature (`echoScore` ≥ 0.42 OR roomRisk high), never on
    // clean takes. Uses `anlmdn` at ultra-short temporal radius which
    // smooths the short-time spectral envelope — this attenuates early
    // reflections and room wash without dulling transients. Paired with a
    // narrow notch at the analyzer's detected echo frequency and a gentle
    // HF shelf cut for very reverberant rooms.
    //
    // Opt-out conditions (so we don't fight other stages):
    //  - `minimalStabilityChain` (fallback mode: keep graph cheap).
    //  - `profile.strictEndingProtection` with very high echoScore — a
    //    strict ending-protected take is sparse dialogue, reverb tails
    //    double as ambience; removing them sounds processed.
    const inEchoScore = profile?.echoScore ?? 0;
    const inRoomScore = (profile?.echoNotchCutDb ?? 0) > 0 ? profile?.echoNotchCutDb ?? 0 : 0;
    const roomRiskIsMed = profile?.roomRisk === "medium" || profile?.roomRisk === "high";
    // Gate matches the auto-reviewer's echo flag (echoScore ≥ 0.32) so we
    // don't leave a band of files flagged "echo_roomy" but never treated.
    // Conservative upper-bound still prevents firing on sparse-dialogue
    // strict-ending-protection takes where the tail reverb is performance.
    const endingProtectionBlocksDereverb =
      (profile?.strictEndingProtection ?? false) &&
      inEchoScore >= 0.75 &&
      !(profile?.allowDereverbDuringEndingProtection ?? false);
    const severeEchoRoom = profile?.severeEchoRoom ?? false;
    const dereverbAllowed =
      !minimalStabilityChain &&
      roomCleanupEnabled &&
      !endingProtectionBlocksDereverb &&
      (inEchoScore >= 0.32 || roomRiskIsMed || inRoomScore >= 0.3);
    if (dereverbAllowed) {
      // Strength 0..1, scales with measured echo. Now uses the ACTUAL
      // `echoScore` as the primary driver (was 0.7×echoScore + bonus) so
      // stronger rooms get proportionally stronger treatment. Caps at 1.0.
      const roomStrength = clamp(
        inEchoScore * 1.08 +
          (profile?.roomRisk === "high" ? 0.24 : profile?.roomRisk === "medium" ? 0.13 : 0) +
          ((profile?.echoDominantRoom ?? false) ? 0.08 : 0) +
          (severeEchoRoom ? 0.18 : 0),
        0,
        severeEchoRoom ? 1.2 : 1,
      );
      // `anlmdn` de-reverb strength scales over a wider range (0.00025 →
      // 0.00065). Stronger settings scrub more reflection smear; the
      // extended range catches the 0.5–0.8 echoScore files that were
      // still landing as `echo_roomy` in auto-review.
      const nlmS = (0.00024 + roomStrength * (severeEchoRoom ? 0.00072 : 0.00034)).toFixed(5);
      const nlmP = severeEchoRoom ? "0.003" : "0.002";
      const nlmR = severeEchoRoom ? "0.010" : roomStrength >= 0.72 ? "0.006" : "0.004";
      filters.push(`anlmdn=s=${nlmS}:p=${nlmP}:r=${nlmR}`);

      // Boxy-room notch — now fires at a lower strength threshold (0.25)
      // so it reaches files that were flagged echo_roomy but skipped the
      // notch before. Depth still modest.
      if (roomStrength >= 0.25) {
        const boxyCut = -clamp(0.35 + roomStrength * (severeEchoRoom ? 1.25 : 1.0), 0.35, severeEchoRoom ? 1.75 : 1.35);
        filters.push(`equalizer=f=280:width_type=q:width=1.1:g=${boxyCut.toFixed(2)}`);
      }

      // Mid-range room notch — 1 kHz "honky" band. Fires on stronger rooms.
      if (roomStrength >= 0.45) {
        const midCut = -clamp(0.4 + (roomStrength - 0.45) * (severeEchoRoom ? 1.45 : 1.15), 0.4, severeEchoRoom ? 1.35 : 1.05);
        filters.push(`equalizer=f=1050:width_type=q:width=1.3:g=${midCut.toFixed(2)}`);
      }
      if (severeEchoRoom && roomStrength >= 0.55) {
        const lowerRoomCut = -clamp(0.45 + (roomStrength - 0.45) * 1.45, 0.45, 1.65);
        const upperRoomCut = -clamp(0.3 + (roomStrength - 0.5) * 1.2, 0.3, 1.2);
        filters.push(`equalizer=f=680:width_type=q:width=1.0:g=${lowerRoomCut.toFixed(2)}`);
        filters.push(`equalizer=f=2200:width_type=q:width=1.15:g=${upperRoomCut.toFixed(2)}`);
      }

      // Top-end shelf cut — now fires on medium rooms too (was high only)
      // when echoScore is high, because brightness reflection scatter lives
      // in both medium and high room ratings.
      if (roomStrength >= 0.5) {
        const topShelf = -clamp(0.4 + (roomStrength - 0.5) * 1.0, 0.4, severeEchoRoom ? 1.1 : 1.0);
        filters.push(`equalizer=f=10500:width_type=q:width=0.7:g=${topShelf.toFixed(2)}`);
      }
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

    if (dyn && gainPlannerActive && !minimalStabilityChain) {
      // The planner has already done speech-aware leveling. What the
      // dynaudnorm safety pass does from here depends on how clean the
      // source is:
      //  - Very clean (instabilityBlend < 0.30): BYPASS. Any intra-line
      //    micro-smoothing is either already done by the planner's micro-
      //    ride or is actively adding audible artifacts (the "too
      //    compressed" complaint was tracked to this pass firing on
      //    already-flat material).
      //  - Otherwise: narrow-window safety pass with amplitude threshold
      //    anchored above the pause-noise floor so silences cannot be
      //    lifted.
      const dynaSafetyBlend = clamp(
        (profile?.instabilityScore ?? 0.5) * 0.5 +
          (profile?.lineSwingScore ?? 0.5) * 0.3 +
          (profile?.sentenceJumpScore ?? 0.5) * 0.2,
        0,
        1,
      );
      if (dynaSafetyBlend >= 0.3) {
        const gateDb = clamp((profile?.speechThresholdDb ?? -44) - 6, -56, -38);
        const gateAmp = fromDb(gateDb);
        const safetyF = 161;
        const safetyG = toOddInt(3 * levelerAdaptationScale, 3, 9);
        const safetyM = toOddInt(5 * levelerAdaptationScale, 3, 11);
        filters.push(
          `dynaudnorm=f=${safetyF}:g=${safetyG}:m=${safetyM}:t=${clamp(gateAmp, fromDb(-60), fromDb(-34)).toFixed(5)}`,
        );
      }
    } else if (dyn) {
      let dynaF: number = dyn.f;
      let dynaG: number = controls.noiseGuard ? Math.max(3, dyn.g - 1) : dyn.g;
      let dynaM: number = controls.noiseGuard ? Math.max(3, dyn.m - 1) : dyn.m;
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
              if (controls.noiseGuard && profile.noiseRisk !== "low") {
                const gateDb = clamp((profile.speechThresholdDb ?? -46) - 8.5, -58, -44);
                dynaThresholdAmp = Math.max(dynaThresholdAmp, fromDb(gateDb));
              }
            } else {
              dynaF = Math.max(dynaF, Math.round(261 + profile.instabilityScore * 90));
            }
          }

          // Noisy takes need slower and lower lift to avoid raising room noise in pauses.
          if (controls.noiseGuard) {
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
            if (controls.noiseGuard && profile.noiseRisk !== "low") {
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

      dynaG *= levelerAdaptationScale;
      dynaM *= levelerAdaptationScale;
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
      sourceSafeMode || activeBreathControl === "Off"
        ? null
        : pauseSafeMode && pauseNoiseRisk >= 0.38
          ? null
        : continuityBreathProtect
          ? BREATH_COMPAND_SAFE[activeBreathControl === "Medium" ? "Medium" : "Light"]
          : BREATH_COMPAND[activeBreathControl];
    const allowSevereEchoRoomGate =
      roomCleanupEnabled && (profile?.useTailGate ?? false) && (profile?.allowTailGateDuringEndingProtection ?? false);
    const suppressContinuityGate =
      !allowSevereEchoRoomGate &&
      (continuitySafeMode || strictEndingProtection || lineContinuityRisk >= 0.58 || lineSwingScore >= 0.48);
    const roomGateFilter =
      !disableTailGate && roomCleanupEnabled && !suppressContinuityGate && (profile?.useTailGate ?? false)
        ? buildTailGateFilter(profile?.tailGateStrength ?? 0.12, profile?.severeEchoRoom ?? false)
        : null;
    const useRoomGate = roomGateFilter !== null;
    const endingProtectedDialogue = strictEndingProtection || (profile?.preserveEndings ?? false);
    const preferFloorGuard =
      !sourceSafeMode &&
      controls.floorGuard &&
      (pauseSafeMode ||
        pauseNoiseRisk >= 0.42 ||
        profile?.noiseRisk === "high" ||
        (controls.noiseGuard && profile?.noiseRisk === "medium"));
    const useFloorGuard =
      !sourceSafeMode &&
      !useRoomGate &&
      controls.floorGuard &&
      (breath === null || preferFloorGuard) &&
      !(endingProtectedDialogue && profile?.noiseRisk === "low" && pauseNoiseRisk < 0.36);
    const useBreathCompand =
      !sourceSafeMode &&
      !useRoomGate &&
      breath !== null &&
      !useFloorGuard &&
      !(endingProtectedDialogue && profile?.noiseRisk === "low" && breathTameStrength < 0.45);

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
    const basePresenceCut = controls.softenHarshness ? -2.0 : 0;
    const baseAirCut = controls.softenHarshness ? -1.1 : 0;
    const harshPresenceCut = profile?.emotionalHarshnessCutDb ?? 0;
    const harshAirCut = profile?.topEndHarshnessCutDb ?? 0;
    const netPresenceGain = clamp(
      basePresenceCut + (profile?.presenceGainDb ?? 0) - harshPresenceCut,
      -4.0,
      0.7
    );
    const netAirGain = clamp(baseAirCut + (profile?.airGainDb ?? 0) - harshAirCut, -2.7, 0.45);

    if (!minimalStabilityChain && !sourceSafeMode && Math.abs(netPresenceGain) >= 0.2) {
      filters.push(`equalizer=f=3500:width_type=q:width=1.15:g=${netPresenceGain.toFixed(2)}`);
    }
    if (!minimalStabilityChain && !sourceSafeMode && Math.abs(netAirGain) >= 0.2) {
      filters.push(`equalizer=f=8000:width_type=q:width=0.75:g=${netAirGain.toFixed(2)}`);
    }
    if (!minimalStabilityChain && !sourceSafeMode && (profile?.topEndHarshnessCutDb ?? 0) >= 0.45) {
      const topShelfCut = clamp(-0.35 - (profile?.topEndHarshnessCutDb ?? 0) * 0.55, -1.1, -0.35);
      filters.push(`equalizer=f=11200:width_type=q:width=0.7:g=${topShelfCut.toFixed(2)}`);
    }
    if (!minimalStabilityChain && !sourceSafeMode && roomCleanupEnabled && (profile?.echoNotchCutDb ?? 0) >= 0.25) {
      const severeEchoNotch = profile?.severeEchoRoom ?? false;
      const echoCut = clamp(profile?.echoNotchCutDb ?? 0, 0.25, severeEchoNotch ? 1.65 : 1.25);
      const notch1 = -clamp(echoCut, 0.25, severeEchoNotch ? 1.65 : 1.25);
      filters.push(`equalizer=f=2450:width_type=q:width=1.35:g=${notch1.toFixed(2)}`);
      if (echoCut >= 0.55) {
        const notch2 = -clamp(echoCut * 0.62, 0.3, severeEchoNotch ? 1.05 : 0.9);
        filters.push(`equalizer=f=1280:width_type=q:width=1.0:g=${notch2.toFixed(2)}`);
      }
      if (echoCut >= 0.9) {
        const notch3 = -clamp(echoCut * 0.45, 0.25, severeEchoNotch ? 0.85 : 0.7);
        filters.push(`equalizer=f=3620:width_type=q:width=1.6:g=${notch3.toFixed(2)}`);
      }
    }
    if (!minimalStabilityChain && !sourceSafeMode && roomCleanupEnabled && profile?.roomRisk === "high") {
      const severeEchoRoomEq = profile?.severeEchoRoom ?? false;
      const roomCutFactor = clamp((profile?.echoNotchCutDb ?? 0.6) / (severeEchoRoomEq ? 1.3 : 1.45), 0.25, 1);
      const roomCutLow = -clamp(0.45 + roomCutFactor * (severeEchoRoomEq ? 0.75 : 0.55), 0.45, severeEchoRoomEq ? 1.25 : 1.05);
      const roomCutMid = -clamp(0.35 + roomCutFactor * (severeEchoRoomEq ? 0.87 : 0.65), 0.35, severeEchoRoomEq ? 1.4 : 1.15);
      filters.push(`equalizer=f=460:width_type=q:width=0.95:g=${roomCutLow.toFixed(2)}`);
      filters.push(`equalizer=f=1650:width_type=q:width=1.2:g=${roomCutMid.toFixed(2)}`);
      if ((profile?.echoNotchCutDb ?? 0) >= 0.95) {
        const roomCutUpperMid = -clamp(0.25 + roomCutFactor * (severeEchoRoomEq ? 0.65 : 0.45), 0.25, severeEchoRoomEq ? 1 : 0.8);
        filters.push(`equalizer=f=2850:width_type=q:width=1.5:g=${roomCutUpperMid.toFixed(2)}`);
      }
    }

    // Tone-match EQ: pull this file's long-term spectrum toward the batch median.
    // Only apply the most prominent band deltas so we don't stack many EQs.
    const toneMatchDeltaDb = profile?.toneMatchDeltaDb;
    if (!minimalStabilityChain && !sourceSafeMode && toneMatchDeltaDb && toneMatchDeltaDb.length === SPECTRUM_BANDS_HZ.length) {
      const ranked = toneMatchDeltaDb
        .map((g, i) => ({ g, hz: SPECTRUM_BANDS_HZ[i] }))
        .filter((item) => Math.abs(item.g) >= 0.6)
        .sort((a, b) => Math.abs(b.g) - Math.abs(a.g))
        .slice(0, 3);
      for (const { g, hz } of ranked) {
        // Narrow bells in mids, wider shelves at edges.
        const widthQ = hz <= 250 || hz >= 4000 ? 0.9 : 1.1;
        filters.push(`equalizer=f=${hz}:width_type=q:width=${widthQ}:g=${clamp(g, -3, 3).toFixed(2)}`);
      }
    }

    // Cinematic color — subtle dub-room voicing. Skip when emotionProtection is
    // high so dramatic takes aren't processed flat.
    const cinematicColorOn =
      !minimalStabilityChain &&
      !sourceSafeMode &&
      !!profile?.cinematicColorEnabled &&
      (profile?.emotionProtection ?? 0) < 0.5;
    if (cinematicColorOn) {
      filters.push("equalizer=f=180:width_type=q:width=1.1:g=0.8"); // warmth
      filters.push("equalizer=f=4500:width_type=q:width=1.2:g=0.6"); // intelligibility
      filters.push("equalizer=f=10000:width_type=q:width=0.7:g=-0.5"); // take glassy edge off
    }

    // De-esser — narrow notches at the two main sibilance bands, scaled by
    // the measured sibilance score. We keep this linear (no sidechain graph)
    // to stay inside the `-af` / comma-joined filter chain. Depth caps at
    // -4 dB so we never dull a voice that is only lightly bright.
    const sibilanceScore = profile?.sibilanceScore ?? 0;
    if (!minimalStabilityChain && !sourceSafeMode && sibilanceScore >= 0.4) {
      const depthNorm = clamp((sibilanceScore - 0.4) / 0.6, 0, 1);
      const mainCut = -clamp(1.2 + depthNorm * 2.8, 1.2, 4);
      const secondaryCut = -clamp(0.6 + depthNorm * 1.8, 0.6, 2.4);
      const deEsserBands = resolveDeEsserBands(profile?.bandSpectrumDb ?? []);
      filters.push(`equalizer=f=${deEsserBands.mainHz}:width_type=q:width=1.4:g=${mainCut.toFixed(2)}`);
      filters.push(`equalizer=f=${deEsserBands.secondaryHz}:width_type=q:width=1.2:g=${secondaryCut.toFixed(2)}`);
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
      (0.9 +
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
        (pauseSafeMode ? 0.04 : 0)) *
        levelerAdaptationScale,
      0.58,
      0.95
    );

    if (sourceSafeMode) {
      // Core-safe candidate intentionally leaves dynamics to the planner and final limiter.
    } else if (gainPlannerActive && !disableSpikeTamers) {
      // Planner already normalized sentence-to-sentence level. Downstream
      // compression is now *adaptive*: on clean takes we bypass entirely
      // (planner + de-esser + alimiter are sufficient), on progressively
      // messier takes we scale the glue up. Bypass threshold raised from
      // 0.25 → 0.35 to address the "too_compressed" auto-review tag that
      // fired on ~60 % of files.
      const instabilityBlend = clamp(
        (profile?.instabilityScore ?? 0.5) * 0.5 +
          (profile?.lineSwingScore ?? 0.5) * 0.3 +
          (profile?.sentenceJumpScore ?? 0.5) * 0.2,
        0,
        1,
      );
      if (instabilityBlend < 0.35) {
        // Fully bypass downstream compression. The planner covers leveling,
        // the de-esser covers sibilance, the alimiter covers peaks. Done.
      } else {
        // Scale softly from 0.35..1.0 so the transition from bypass to glue
        // doesn't cliff. Also lighter max mix (0.55 → 0.6) and lighter
        // ratio ceiling (1.7 → 1.6) since the planner has already handled
        // the bulk of dynamics control.
        const norm = (instabilityBlend - 0.35) / 0.65; // 0 at 0.35, 1 at 1.0
        const ratio = clamp(1.3 + norm * 0.3, 1.3, 1.6);
        const mix = clamp(0.15 + norm * 0.45, 0.15, 0.6);
        const threshold = clamp(-22 + (1 - norm) * 2, -22, -18);
        filters.push(
          `acompressor=threshold=${threshold.toFixed(1)}dB:ratio=${ratio.toFixed(2)}:attack=25:release=260:mix=${mix.toFixed(2)}:detection=rms`,
        );
      }
    } else if (!gainPlannerActive) {
      filters.push(
        `acompressor=threshold=${threshold.toFixed(1)}dB:ratio=${ratio.toFixed(2)}:attack=${attack}:release=${release}:mix=${compMix.toFixed(2)}:detection=rms`
      );
    }

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
      "[dryv][wet]amix=inputs=2:duration=first:normalize=0,alimiter=limit=-2dB:level=disabled",
    ].join(";");
  };

  const writeOutput = async (
    ffmpeg: FFmpeg,
    name: string,
    kind: OutputEntry["kind"],
    variant: OutputEntry["variant"],
    processingFlow: OutputEntry["processingFlow"] = "app",
    metadata: Pick<OutputEntry, "sourceBase" | "sourceName" | "partIndex" | "partTotal"> = {},
  ): Promise<OutputEntry> => {
    const bytes = await readVirtualFileBytes(ffmpeg, name);
    const outputName = processingFlow === "app-neural-speech-enhancement-app" ? enhancedOutputName(name) : name;

    const blob = new Blob([bytes], { type: "audio/wav" });
    return {
      name: outputName,
      blob,
      size: blob.size,
      kind,
      variant,
      processingFlow,
      ...metadata,
    };
  };

  const buildFinalPolishProfile = (
    profile: AdaptiveProfile | null,
    directives: AudioReviewAdaptiveDirectives,
  ): AdaptiveProfile | null => {
    if (!profile) return null;

    const polish = clamp(0.28 + directives.finalPolishIntensity * 0.34, 0.28, 0.62);
    const toneScale = clamp(0.34 + polish * 0.38, 0.34, 0.58);
    const stabilityScale = clamp(0.42 + polish * 0.46, 0.42, 0.7);
    const deHarshScale = clamp(0.5 + polish * 0.34, 0.5, 0.72);
    const compressionScale = clamp(0.28 + polish * 0.35, 0.28, 0.5);

    return {
      ...profile,
      lowMidGainDb: clamp(profile.lowMidGainDb * toneScale, -1.35, 0.65),
      presenceGainDb: clamp(profile.presenceGainDb * toneScale, -1.05, 0.75),
      airGainDb: clamp(profile.airGainDb * toneScale, -0.75, 0.55),
      emotionalHarshnessCutDb: clamp(profile.emotionalHarshnessCutDb * deHarshScale, 0, 1.15),
      topEndHarshnessCutDb: clamp(profile.topEndHarshnessCutDb * deHarshScale, 0, 0.85),
      levelingNeed: clamp(profile.levelingNeed * stabilityScale, 0, 0.62),
      emotionProtection: clamp(profile.emotionProtection * stabilityScale, 0, 0.65),
      compressorRatioOffset: clamp(profile.compressorRatioOffset * compressionScale, -0.16, 0.22),
      compressorThresholdOffsetDb: clamp(profile.compressorThresholdOffsetDb * compressionScale, -0.65, 0.75),
      levelerBias: clamp(profile.levelerBias * compressionScale, -0.25, 0.25),
      dynaTrim: clamp(profile.dynaTrim * 0.45, 0, 1.6),
      useDenoise: false,
      denoiseStrength: 0,
      useTailGate: false,
      tailGateStrength: 0,
      echoNotchCutDb: clamp(profile.echoNotchCutDb * 0.55, 0, 0.72),
      clickTameStrength: clamp(profile.clickTameStrength * stabilityScale, 0, 0.5),
      breathTameStrength: clamp(profile.breathTameStrength * stabilityScale, 0, 0.58),
      onsetTameStrength: clamp(profile.onsetTameStrength * stabilityScale, 0, 0.55),
      sagRecoveryStrength: clamp(profile.sagRecoveryStrength * clamp(0.58 + polish * 0.48, 0.58, 0.82), 0, 0.78),
      toneMatchDeltaDb: profile.toneMatchDeltaDb?.map((value) => clamp(value * toneScale, -0.75, 0.75)) ?? null,
    };
  };

  const runFinalAppPolishPass = async (
    ffmpeg: FFmpeg,
    targetName: string,
    profile: AdaptiveProfile | null,
    context: {
      base: string;
      detail: string;
      candidateVariant?: CandidateVariant | null;
      sourceSafeChain?: boolean;
    },
  ) => {
    let activeFfmpeg = ffmpeg;
    const directives = getActiveAudioReviewAdaptiveDirectives();
    const finalPolishProfile = buildFinalPolishProfile(profile, directives);
    const inputBytes = await readVirtualFileBytes(activeFfmpeg, targetName);
    const inputByteLength = assertUsableWavBytes(inputBytes, "App output before final app polish");
    const tempBase = targetName.replace(/\.wav$/i, "");
    const appPassName = `${tempBase}_final_polish_tmp.wav`;
    const sourceSafePolish = context.sourceSafeChain === true || context.candidateVariant === "source-safe";
    const candidateVariant = sourceSafePolish ? "source-safe" : (context.candidateVariant ?? "continuity-safe");

    try {
      setStatus(`Final app polish: ${context.base}`);
      setActiveQueueStage(context.base, "Final app polish", `${context.detail}, subtle single pass`);
      await safeDeleteFile(activeFfmpeg, appPassName);
      appendLog(
        `[FinalPolish] ${context.base}: app output -> final app polish (${formatBytes(inputByteLength)}).`,
      );
      await runMixReady(activeFfmpeg, targetName, appPassName, finalPolishProfile, {
        candidateVariant,
        forceEndingProtection: true,
        skipSpeechSegmentation: true,
        disableSegmentGainMatch: true,
        disableAdaptiveNoiseReduction: true,
        disableRoomCleanup: sourceSafePolish ? true : undefined,
        sourceSafeChain: sourceSafePolish,
      });

      const passBytes = await readVirtualFileBytes(activeFfmpeg, appPassName);
      const passByteLength = assertUsableWavBytes(passBytes, "Final app polish");

      await activeFfmpeg.writeFile(targetName, passBytes);
      appendLog(
        `[FinalPolish] ${context.base}: final app polish applied once (${formatBytes(
          passByteLength,
        )}; delivery uses app + final app polish).`,
      );
      return {
        ffmpeg: activeFfmpeg,
        applied: true,
        polishPasses: 1,
      };
    } catch (error) {
      appendLog(
        `[FinalPolish] ${context.base}: final app polish unavailable; keeping primary app output (${
          error instanceof Error ? error.message : String(error)
        }).`,
      );
      if (shouldResetFfmpegForError(error)) {
        activeFfmpeg = await refreshFfmpeg(`final polish failure on ${context.base}`);
      }
      await activeFfmpeg.writeFile(targetName, inputBytes);
      return {
        ffmpeg: activeFfmpeg,
        applied: false,
        polishPasses: 0,
      };
    } finally {
      await safeDeleteFile(activeFfmpeg, appPassName);
    }
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
    options?: MixRenderOptions,
    plannerLeveledInputName?: string | null,
  ) => {
    // When the caller has already produced a planner-leveled version of the
    // input, we use it verbatim and tell the mix chain that broadband leveling
    // is already done (so it downgrades dynaudnorm to a gentle safety pass).
    const chainInput = plannerLeveledInputName ?? inputName;
    const gainPlannerActive = Boolean(plannerLeveledInputName);
    const filterChain = buildMixFilter(profile, {
      ...options,
      gainPlannerActive,
    });
    const runUnprimed = async () => {
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
          chainInput,
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
        "Mix-ready render",
      );
    };
    await renderWithHeadPriming(ffmpeg, {
      inputName: chainInput,
      outputName,
      filterChain,
      options,
      expectedDurationSec: null,
      contextLabel: `Mix-ready ${sanitizeBase(outputName)}`,
      unprimed: runUnprimed,
    });
  };

  const runMixReadyRange = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    profile: AdaptiveProfile | null,
    startSec: number,
    durationSec: number,
    options?: MixRenderOptions,
  ) => {
    const filterChain = buildMixFilter(profile, {
      ...options,
      gainPlannerActive: false,
    });
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
        startSec.toFixed(3),
        "-t",
        durationSec.toFixed(3),
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
      "Long-form chunk mix-ready render",
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

  const shouldUseHeadPriming = (options: MixRenderOptions | undefined, expectedDurationSec: number | null) =>
    HEAD_PRIME_ENABLED &&
    !options?.minimalStabilityChain &&
    !options?.sourceSafeChain &&
    !options?.disableHeadPriming &&
    (expectedDurationSec === null || expectedDurationSec < GAIN_PLANNER_MAX_DURATION_SECONDS);

  const renderWithHeadPriming = async (
    ffmpeg: FFmpeg,
    input: {
      inputName: string;
      outputName: string;
      filterChain: string | null;
      options?: MixRenderOptions;
      inputReadArgs?: string[];
      expectedDurationSec: number | null;
      contextLabel: string;
      postPrimeFilterChain?: string | null;
      unprimed: () => Promise<void>;
    },
  ) => {
    if (!shouldUseHeadPriming(input.options, input.expectedDurationSec)) {
      await input.unprimed();
      return;
    }

    const primeSeconds = HEAD_PRIME_SECONDS.toFixed(3);
    const baseFilter = input.filterChain?.trim() || "anull";
    const postPrimeFilter = input.postPrimeFilterChain?.trim();
    const filterComplex = [
      `[0:a]atrim=0:${primeSeconds},areverse[headprime]`,
      `[headprime][0:a]concat=n=2:v=0:a=1[primed]`,
      `[primed]${baseFilter},atrim=start=${primeSeconds},asetpts=PTS-STARTPTS${
        postPrimeFilter ? `,${postPrimeFilter}` : ""
      }[out]`,
    ].join(";");

    try {
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
          ...(input.inputReadArgs ?? []),
          "-i",
          input.inputName,
          "-filter_complex",
          filterComplex,
          "-map",
          "[out]",
          "-ar",
          "48000",
          "-ac",
          "1",
          "-c:a",
          "pcm_f32le",
          input.outputName,
        ],
        `${input.contextLabel} head-primed render`,
      );

      const expectedDurationSec =
        input.expectedDurationSec ?? (await probeInputDurationSeconds(ffmpeg, input.inputName));
      const renderedDurationSec = await probeInputDurationSeconds(ffmpeg, input.outputName);
      if (
        expectedDurationSec !== null &&
        renderedDurationSec !== null &&
        Math.abs(renderedDurationSec - expectedDurationSec) > HEAD_PRIME_DURATION_TOLERANCE_SECONDS
      ) {
        throw new Error(
          `duration delta ${(renderedDurationSec - expectedDurationSec).toFixed(3)}s exceeds ${HEAD_PRIME_DURATION_TOLERANCE_SECONDS.toFixed(
            2,
          )}s`,
        );
      }
      appendLog(`[HeadPrime] ${input.contextLabel}: primed first ${HEAD_PRIME_SECONDS.toFixed(1)}s.`);
    } catch (error) {
      appendLog(`[HeadPrime] fallback ${input.contextLabel}: ${describeError(error)}.`);
      await safeDeleteFile(ffmpeg, input.outputName);
      await input.unprimed();
    }
  };

  const runMixReadySegmented = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    profile: AdaptiveProfile | null,
    durationSeconds: number,
    options?: MixRenderOptions,
    plannerLeveledInputName?: string | null,
  ) => {
    if (durationSeconds < MIX_SEGMENT_MIN_DURATION_SECONDS) {
      throw new Error("Segmented render skipped (input too short).");
    }
    const segmentCount = Math.ceil(durationSeconds / MIX_SEGMENT_SECONDS);
    if (segmentCount < 2) {
      throw new Error("Segmented render skipped (single segment).");
    }

    // If the caller pre-leveled the file, every segment reads from that file
    // instead of the original (same time ranges, same timestamps).
    const chainInput = plannerLeveledInputName ?? inputName;
    const gainPlannerActive = Boolean(plannerLeveledInputName);
    const tempBase = sanitizeBase(outputName);
    const segmentNames: string[] = [];
    const filterChain = buildMixFilter(profile, { ...options, gainPlannerActive });

    try {
      for (let index = 0; index < segmentCount; index += 1) {
        const start = index * MIX_SEGMENT_SECONDS;
        const remaining = durationSeconds - start;
        const nativeSpan = Math.min(MIX_SEGMENT_SECONDS, Math.max(remaining, 0));
        if (nativeSpan <= 0.01) break;
        const isLast = index === segmentCount - 1 || start + nativeSpan >= durationSeconds - 0.01;
        // Non-last segments include CHUNK_CROSSFADE_SECONDS of overlap so
        // the downstream acrossfade consumes it without shortening the
        // total output. Per-segment filter-state restart transients get
        // blended across the overlap instead of leaving a sample-level
        // click at the boundary.
        const span = isLast
          ? nativeSpan
          : Math.min(nativeSpan + CHUNK_CROSSFADE_SECONDS, durationSeconds - start);
        const segmentName = `${tempBase}_seg_${index + 1}.wav`;
        segmentNames.push(segmentName);

        const runUnprimed = async () => {
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
              chainInput,
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
            `Segment mix-ready render ${index + 1}/${segmentCount}`,
          );
        };
        if (index === 0) {
          await renderWithHeadPriming(ffmpeg, {
            inputName: chainInput,
            outputName: segmentName,
            filterChain,
            options,
            inputReadArgs: ["-ss", start.toFixed(3), "-t", span.toFixed(3)],
            expectedDurationSec: span,
            contextLabel: `${sanitizeBase(outputName)} segment 1`,
            unprimed: runUnprimed,
          });
        } else {
          await runUnprimed();
        }
      }

      if (segmentNames.length < 2) {
        throw new Error("Segmented render produced insufficient segments.");
      }

      await runCrossfadeConcat(
        ffmpeg,
        segmentNames,
        outputName,
        CHUNK_CROSSFADE_SECONDS,
        "Fixed segment crossfade",
      );
      await logDurationDelta(ffmpeg, "Fixed segmented render", durationSeconds, outputName);
    } finally {
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

  const countProcessedRenderSegments = (segments: RenderSegment[]) => segments.filter((segment) => segment.process).length;

  const mergeRenderSegmentsForRisk = (
    segments: RenderSegment[],
    pauseThresholdSec: number,
    maxProcessedSegments: number
  ) => {
    const merged = segments.map((segment) => ({ ...segment }));
    while (countProcessedRenderSegments(merged) > maxProcessedSegments) {
      let mergeIndex = -1;
      let bestPauseSec = Number.POSITIVE_INFINITY;
      for (let index = 1; index < merged.length - 1; index += 1) {
        const previous = merged[index - 1];
        const current = merged[index];
        const next = merged[index + 1];
        if (current.process || !previous.process || !next.process) continue;
        const pauseSec = current.endSec - current.startSec;
        if (pauseSec > pauseThresholdSec || pauseSec >= bestPauseSec) continue;
        mergeIndex = index;
        bestPauseSec = pauseSec;
      }
      if (mergeIndex < 0) break;

      const previous = merged[mergeIndex - 1];
      const next = merged[mergeIndex + 1];
      merged.splice(mergeIndex - 1, 3, {
        startSec: previous.startSec,
        endSec: next.endSec,
        process: true,
        trimInMs: previous.trimInMs,
        trimOutMs: next.trimOutMs,
        forceEndingProtection: (previous.forceEndingProtection ?? false) || (next.forceEndingProtection ?? false),
      });
    }
    return merged;
  };

  const describeRenderPath = (path: RenderPath) =>
    path === "speech-pause-segmented"
      ? "speech-pause segmented"
      : path === "speech-aligned-segmented"
        ? "speech-aligned segmented"
        : path === "fixed-segmented"
          ? "fixed segmented"
          : path === "single-pass-recovered"
            ? "single-pass recovered"
            : "single-pass";

  const summarizeDegradeReasons = (reasons: DegradeReason[]) => {
    if (reasons.length === 0) return "none";
    return reasons.join("+");
  };

  const summarizeCandidateMeta = (meta: CandidateRenderMeta) =>
    `chain ${meta.strategyLabel}, path ${describeRenderPath(meta.renderPath)}, degraded ${
      meta.degraded ? "yes" : "no"
    } (${summarizeDegradeReasons(meta.degradeReasons)}), windows ${meta.analysisWindowsSucceeded}/${
      meta.analysisWindowsAttempted
    }${
      meta.analysisWindowsDropped > 0 ? ` dropped ${meta.analysisWindowsDropped}` : ""
    }`;

  const isAudibilityProtectedRender = (meta: CandidateRenderMeta | null | undefined) =>
    Boolean(
      meta &&
        (meta.degradeReasons.includes("audibility-dropout-guard") ||
          meta.strategyLabel === PLANNER_TAIL_SAFE_STRATEGY_LABEL ||
          meta.strategyLabel === AUDIBILITY_SAFE_STRATEGY_LABEL),
    );

  const buildSilenceSegmentFilter = (profile: AdaptiveProfile | null, options?: MixRenderOptions) => {
    const controls = getActiveAudioReviewControls();
    const filters: string[] = [];
    const candidateVariant = options?.candidateVariant ?? "cinematic-stable";
    const pauseSafeMode = candidateVariant === "pause-safe";
    const minimalStabilityChain = options?.minimalStabilityChain === true;
    const sourceSafeMode = options?.sourceSafeChain === true || candidateVariant === "source-safe";
    if (controls.eqCleanup) {
      filters.push(`highpass=f=${profile?.highpassHz ?? 80}`);
    }
    if (
      !minimalStabilityChain &&
      !sourceSafeMode &&
      !options?.disableAdaptiveNoiseReduction &&
      controls.noiseGuard &&
      profile &&
      (profile.pauseNoiseRisk >= 0.42 || (pauseSafeMode && profile.noiseRisk !== "low"))
    ) {
      const adaptiveNoiseReduction = buildAdaptiveNoiseReductionFilter(
        profile.noiseRisk,
        profile.noiseFloorDb,
        profile.noiseContrastDb,
        profile.pauseNoiseRisk,
        profile.roomRisk,
      );
      if (adaptiveNoiseReduction) {
        filters.push(adaptiveNoiseReduction);
      }
    }
    if (
      !minimalStabilityChain &&
      !sourceSafeMode &&
      controls.floorGuard &&
      profile &&
      (profile.noiseRisk === "high" || profile.pauseNoiseRisk >= 0.38 || pauseSafeMode)
    ) {
      filters.push(profile.floorGuardFilter);
    }
    if (
      !minimalStabilityChain &&
      !sourceSafeMode &&
      !options?.disableTailGate &&
      !options?.disableRoomCleanup &&
      controls.roomCleanup &&
      profile &&
      profile.roomRisk !== "low" &&
      (!profile.preserveEndings || profile.allowTailGateDuringEndingProtection)
    ) {
      const segmentTailGateScale = profile.severeEchoRoom ? (pauseSafeMode ? 0.75 : 0.9) : pauseSafeMode ? 0.85 : 0.55;
      filters.push(
        buildTailGateFilter(
          (profile.tailGateStrength ?? 0.1) * segmentTailGateScale,
          profile.severeEchoRoom && profile.allowTailGateDuringEndingProtection,
        ),
      );
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
    segments: RenderSegment[],
    options?: MixRenderOptions,
    plannerLeveledInputName?: string | null,
  ) => {
    if (segments.length < 2) {
      throw new Error("Speech-aligned segmentation skipped (insufficient segments).");
    }

    // If the caller pre-leveled the file, every segment reads from that file.
    const chainInput = plannerLeveledInputName ?? inputName;
    const gainPlannerActive = Boolean(plannerLeveledInputName);
    const segmentMode = options?.segmentMode ?? "speech-aligned";
    const tempBase = sanitizeBase(outputName);
    const segmentNames: string[] = [];
    const cleanupNames: string[] = [];
    const enableSegmentGainMatch =
      !!profile &&
      profile.segmentMatchTargetI !== null &&
      !options?.disableSegmentGainMatch &&
      (profile.preferSinglePassContinuity ||
        profile.sentenceJumpScore >= 0.28 ||
        profile.breathSpikeRisk >= 0.34 ||
        durationSeconds >= DISTRIBUTED_ANALYSIS_THRESHOLD_SECONDS);

    try {
      for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        const isLast = index === segments.length - 1;
        const readStart = Math.max(0, segment.startSec - segment.trimInMs / 1000);
        // Non-last segments include CHUNK_CROSSFADE_SECONDS of overlap past
        // their native end so `runCrossfadeConcat` can consume it without
        // shifting timing.
        const boundaryOverlap = isLast ? 0 : CHUNK_CROSSFADE_SECONDS;
        const readEnd = Math.min(
          durationSeconds,
          segment.endSec + segment.trimOutMs / 1000 + boundaryOverlap,
        );
        const readSpan = Math.max(0.05, readEnd - readStart);
        const trimStartSec = Math.max(0, (segment.startSec - readStart) + ((options?.trimSegmentPadMs ?? 0) / 1000));
        const trimEndSec = Math.max(
          trimStartSec + 0.02,
          (segment.endSec - readStart) - ((options?.trimSegmentPadMs ?? 0) / 1000) + boundaryOverlap,
        );
        const segmentName = `${tempBase}_speech_seg_${index + 1}.wav`;
        cleanupNames.push(segmentName);

        const segmentOptions = {
          ...options,
          segmentMode,
          forceEndingProtection: segment.process && (segment.forceEndingProtection ?? false),
          gainPlannerActive,
        } satisfies MixRenderOptions;
        const processedFilter = buildMixFilter(profile, segmentOptions);
        const silenceFilter = buildSilenceSegmentFilter(profile, segmentOptions);
        const baseFilter = segment.process ? processedFilter : silenceFilter;
        const trimFilter = `atrim=start=${trimStartSec.toFixed(3)}:end=${trimEndSec.toFixed(3)},asetpts=N/SR/TB`;
        const filterChain = baseFilter ? `${baseFilter},${trimFilter}` : trimFilter;

        const runUnprimed = async () => {
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
              chainInput,
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
            `Speech-aligned segment render ${index + 1}/${segments.length}`,
          );
        };
        if (index === 0) {
          await renderWithHeadPriming(ffmpeg, {
            inputName: chainInput,
            outputName: segmentName,
            filterChain: baseFilter,
            options: segmentOptions,
            inputReadArgs: ["-ss", readStart.toFixed(3), "-t", readSpan.toFixed(3)],
            expectedDurationSec: Math.max(0.02, trimEndSec - trimStartSec),
            contextLabel: `${sanitizeBase(outputName)} speech segment 1`,
            postPrimeFilterChain: trimFilter,
            unprimed: runUnprimed,
          });
        } else {
          await runUnprimed();
        }

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

      await runCrossfadeConcat(
        ffmpeg,
        segmentNames,
        outputName,
        CHUNK_CROSSFADE_SECONDS,
        "Speech-aligned segment crossfade",
      );
      await logDurationDelta(ffmpeg, `${segmentMode} segmented render`, durationSeconds, outputName);

      return segments.length;
    } finally {
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

  const runStaticLoudnessFallback = async (
    ffmpeg: FFmpeg,
    inputName: string,
    outputName: string,
    cfg: NonNullable<(typeof LOUDNESS_PRESETS)[keyof typeof LOUDNESS_PRESETS]>,
    measuredOffset: number,
  ) => {
    const safeOffset = clamp(measuredOffset, -12, 12);
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
        `volume=${safeOffset.toFixed(3)}dB,alimiter=limit=${cfg.TP}dB:level=disabled`,
        "-ar",
        "48000",
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        outputName,
      ],
      "Static loudness fallback",
    );
  };

  const alignBatchMixReadyOutputs = async (ffmpeg: FFmpeg, outputEntries: OutputEntry[]) => {
    type AlignmentTarget = { entry: OutputEntry; index: number; groupId: string };

    const estimateOutputAudioSeconds = (entry: OutputEntry) => {
      const byteLength = Number.isFinite(entry.size) && entry.size > 0 ? entry.size : entry.blob.size;
      if (!Number.isFinite(byteLength) || byteLength <= 44) return 0;
      return (byteLength - 44) / (48000 * 4);
    };
    const medianNumber = (values: number[]) => {
      const finiteValues = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      if (finiteValues.length === 0) return null;
      const mid = Math.floor(finiteValues.length / 2);
      return finiteValues.length % 2 === 0 ? (finiteValues[mid - 1] + finiteValues[mid]) / 2 : finiteValues[mid];
    };
    const buildGroupId = (entry: OutputEntry) => {
      const sourceKey = entry.sourceBase ?? entry.name.replace(/_(?:part-\d+_)?(?:blend_)?mixready.*$/i, "");
      return `${entry.variant}:${sourceKey}`;
    };

    const targets = outputEntries
      .map((entry, index) => ({ entry, index }))
      .filter(({ entry }) => entry.kind === "mixready")
      .map(({ entry, index }) => ({ entry, index, groupId: buildGroupId(entry) }));
    if (targets.length < 2) {
      appendLog("[BatchAlign] skipped (fewer than 2 mix-ready outputs).");
      return ffmpeg;
    }

    let activeFfmpeg = ffmpeg;
    let cumulativeAudioSec = 0;
    const recycleIfNeeded = async (stage: "measure" | "render") => {
      if (cumulativeAudioSec < BATCH_AUDIO_RECYCLE_SECONDS) return;
      activeFfmpeg = await refreshFfmpeg(
        `batch loudness alignment ${stage} memory guard (${Math.round(cumulativeAudioSec)}s processed)`,
      );
      cumulativeAudioSec = 0;
    };

    const noteProcessedAudio = async (entry: OutputEntry, stage: "measure" | "render") => {
      cumulativeAudioSec += estimateOutputAudioSeconds(entry);
      await recycleIfNeeded(stage);
    };

    try {
      setStatus("Batch loudness alignment");
      const targetsByVariant = new Map<OutputEntry["variant"], AlignmentTarget[]>();
      for (const target of targets) {
        const variantTargets = targetsByVariant.get(target.entry.variant) ?? [];
        variantTargets.push(target);
        targetsByVariant.set(target.entry.variant, variantTargets);
      }

      const stagedEntries: Array<{ index: number; entry: OutputEntry }> = [];
      let alignedCount = 0;
      for (const [variant, variantTargets] of targetsByVariant.entries()) {
        if (variantTargets.length < 2) {
          appendLog(`[BatchAlign] ${variant} skipped (fewer than 2 outputs).`);
          continue;
        }

        const groupedTargets = new Map<string, AlignmentTarget[]>();
        const measurementsByGroup = new Map<string, number[]>();
        const measurementByIndex = new Map<number, number | null>();
        for (const target of variantTargets) {
          const group = groupedTargets.get(target.groupId) ?? [];
          group.push(target);
          groupedTargets.set(target.groupId, group);

          const inputName = `batch_align_${target.index}_measure.wav`;
          try {
            await activeFfmpeg.writeFile(inputName, new Uint8Array(await target.entry.blob.arrayBuffer()));
            try {
              const loudness = await analyzeIntegratedLoudness(activeFfmpeg, inputName);
              measurementByIndex.set(target.index, loudness.inputI);
              if (loudness.inputI !== null && Number.isFinite(loudness.inputI)) {
                const values = measurementsByGroup.get(target.groupId) ?? [];
                values.push(loudness.inputI);
                measurementsByGroup.set(target.groupId, values);
              }
            } catch (error) {
              appendLog(`[BatchAlign] ${target.entry.name}: loudness measure skipped (${describeError(error)}).`);
              measurementByIndex.set(target.index, null);
            }
          } finally {
            await safeDeleteFile(activeFfmpeg, inputName);
          }
          await noteProcessedAudio(target.entry, "measure");
        }

        const groupMeasurements = Array.from(groupedTargets.keys()).map((groupId) => ({
          id: groupId,
          inputI: medianNumber(measurementsByGroup.get(groupId) ?? []),
        }));
        const alignment = planBatchLoudnessAlignment(groupMeasurements);
        if (alignment.anchorLufs === null) {
          appendLog(`[BatchAlign] ${variant} skipped (insufficient loudness measurements).`);
          continue;
        }

        for (const plan of alignment.plans) {
          const groupTargets = groupedTargets.get(plan.id) ?? [];
          if (groupTargets.length === 0) continue;
          const groupLabel = groupTargets[0]?.entry.sourceBase ?? groupTargets[0]?.entry.name ?? plan.id;
          if (!plan.shouldAlign) {
            appendLog(
              `[BatchAlign] ${groupLabel} (${variant}): within ${Math.abs(
                (plan.inputI ?? alignment.anchorLufs) - alignment.anchorLufs,
              ).toFixed(1)} LU of batch anchor ${alignment.anchorLufs.toFixed(1)} LUFS.`,
            );
            continue;
          }

          for (const target of groupTargets) {
            const inputName = `batch_align_${target.index}_in.wav`;
            const outputName = `batch_align_${target.index}_out.wav`;
            let nextEntry = target.entry;
            try {
              await activeFfmpeg.writeFile(inputName, new Uint8Array(await target.entry.blob.arrayBuffer()));
              resetLogBuffer();
              await execOrThrow(
                activeFfmpeg,
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
                  `volume=${plan.offsetDb.toFixed(2)}dB,${LIMITER_FILTER}`,
                  "-ar",
                  "48000",
                  "-ac",
                  "1",
                  "-c:a",
                  "pcm_f32le",
                  outputName,
                ],
                "Batch loudness alignment",
              );
              const expectedDurationSec = estimateOutputAudioSeconds(target.entry);
              const renderedDurationSec = await probeInputDurationSeconds(activeFfmpeg, outputName);
              if (
                expectedDurationSec > 0 &&
                renderedDurationSec !== null &&
                Number.isFinite(renderedDurationSec) &&
                Math.abs(renderedDurationSec - expectedDurationSec) > HEAD_PRIME_DURATION_TOLERANCE_SECONDS
              ) {
                throw new Error(
                  `duration delta ${(renderedDurationSec - expectedDurationSec).toFixed(3)}s exceeds ${HEAD_PRIME_DURATION_TOLERANCE_SECONDS.toFixed(
                    2,
                  )}s`,
                );
              }
              const alignedLoudness = await analyzeIntegratedLoudness(activeFfmpeg, outputName);
              if (alignedLoudness.inputTP !== null && alignedLoudness.inputTP > -1.5) {
                throw new Error(`true peak ${alignedLoudness.inputTP.toFixed(2)} dBTP exceeds -1.5 dBTP gate`);
              }
              const beforeI = measurementByIndex.get(target.index);
              if (
                beforeI !== null &&
                beforeI !== undefined &&
                alignedLoudness.inputI !== null &&
                Number.isFinite(alignedLoudness.inputI)
              ) {
                const movedOppositeDirection =
                  plan.offsetDb > 0
                    ? alignedLoudness.inputI < beforeI - 0.1
                    : alignedLoudness.inputI > beforeI + 0.1;
                if (movedOppositeDirection) {
                  throw new Error(
                    `loudness moved opposite requested offset (${beforeI.toFixed(1)} -> ${alignedLoudness.inputI.toFixed(
                      1,
                    )} LUFS, offset ${formatSigned(plan.offsetDb, 1)} dB)`,
                  );
                }
              }
              const alignedBytes = await readVirtualFileBytes(activeFfmpeg, outputName);
              const alignedBlob = new Blob([alignedBytes], { type: "audio/wav" });
              nextEntry = {
                ...target.entry,
                blob: alignedBlob,
                size: alignedBlob.size,
              };
            } finally {
              await safeDeleteFile(activeFfmpeg, inputName);
              await safeDeleteFile(activeFfmpeg, outputName);
            }
            stagedEntries.push({ index: target.index, entry: nextEntry });
            alignedCount += 1;
            appendLog(
              `[BatchAlign] ${target.entry.name}: ${formatSigned(plan.offsetDb, 1)} dB toward ${variant} batch anchor ${alignment.anchorLufs.toFixed(
                1,
              )} LUFS.`,
            );
            await noteProcessedAudio(nextEntry, "render");
          }
          if (groupTargets.length > 1) {
            appendLog(
              `[BatchAlign] ${groupLabel} (${variant}): used one shared ${formatSigned(
                plan.offsetDb,
                1,
              )} dB offset for ${groupTargets.length} parts.`,
            );
          }
        }
      }

      for (const staged of stagedEntries) {
        outputEntries[staged.index] = staged.entry;
      }

      if (alignedCount === 0) {
        appendLog("[BatchAlign] all mix-ready outputs already within their variant batch anchors.");
      }
      return activeFfmpeg;
    } catch (error) {
      appendLog(`[BatchAlign] skipped after failure; original outputs kept (${describeError(error)}).`);
      if (shouldResetFfmpegForError(error)) {
        return refreshFfmpeg("batch loudness alignment failure");
      }
      return activeFfmpeg;
    }
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

    // Adaptive linear: loudnorm's linear mode tries to apply a single static
    // gain, which is ideal when the source already has the target LRA.
    // When the source has been over-compressed (measured LRA < 4), using
    // linear locks that compression in; switching to dynamic mode lets
    // loudnorm gently re-expand while normalizing.
    const targetLraNum = Number(cfg.LRA);
    const linearMode = measuredLRA !== null && measuredLRA >= 4 && measuredLRA <= targetLraNum + 3;
    if (!linearMode) {
      appendLog(
        `[Loudnorm] ${sanitizeBase(inputName)}: dynamic mode (measured LRA ${measuredLRA.toFixed(1)} outside linear band).`,
      );
    }

    resetLogBuffer();
    try {
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
          `loudnorm=I=${cfg.I}:TP=${cfg.TP}:LRA=${cfg.LRA}:measured_I=${measuredI}:measured_TP=${measuredTP}:measured_LRA=${measuredLRA}:measured_thresh=${measuredThresh}:offset=${offset}:linear=${linearMode ? "true" : "false"}:print_format=summary`,
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
    } catch (error) {
      appendLog(
        `[Loudnorm] ${sanitizeBase(
          inputName,
        )}: render fallback (${error instanceof Error ? error.message : String(error)}); applying measured static gain ${offset.toFixed(
          2,
        )} dB with true-peak limiter.`,
      );
      await runStaticLoudnessFallback(ffmpeg, inputName, outputName, cfg, offset);
    }
  };

  const renderLongFormSafeMode = async (
    ffmpeg: FFmpeg,
    job: JobEntry,
    profile: AdaptiveProfile | null,
    durationSeconds: number,
    fileIndex: number,
    totalFiles: number,
    outputEntries: OutputEntry[],
    silenceSpans: SilenceSpan[] = [],
  ) => {
    const chunks = planLongFormChunks(durationSeconds, undefined, silenceSpans);
    if (chunks.length === 0) {
      throw new Error("Long-form safe mode could not plan export chunks.");
    }
    const stagedOutputs: OutputEntry[] = [];

    const candidateVariant: CandidateVariant =
      sourceFirstCandidateVariantRef.current ??
      (profile?.preferSinglePassContinuity ? "continuity-safe" : "cinematic-stable");
    appendLog(
      `[LongForm] ${job.base}: ${durationSeconds.toFixed(
        0,
      )}s exceeds browser planner budget; using low-stress chunked export (${chunks.length} part${
        chunks.length === 1 ? "" : "s"
      }, silence-aware cuts, one render chain, no full-file planner decode, no duplicate QC candidate renders).`,
    );

    for (const chunk of chunks) {
      const partTag = formatLongFormPartTag(chunk);
      const partLabel = `${chunk.index + 1}/${chunk.total}`;
      const mixChunkName = `${job.base}_${partTag}_mixready.wav`;
      const blendChunkName = `${job.base}_${partTag}_blend_mixready.wav`;
      const loudChunkName = loudnessConfig ? `${job.base}_${partTag}_${loudnessConfig.suffix}.wav` : null;
      const blendLoudChunkName =
        loudnessConfig && sceneBlend ? `${job.base}_${partTag}_blend_${loudnessConfig.suffix}.wav` : null;
      let blendRendered = false;

      try {
        setStatus(`Long-form mix: ${job.base} (${fileIndex + 1}/${totalFiles}, part ${partLabel})`);
        setActiveQueueStage(job.base, "Long-form mix", `File ${fileIndex + 1}/${totalFiles}, part ${partLabel}`);
        await runMixReadyRange(
          ffmpeg,
          job.inputName,
          mixChunkName,
          profile,
          chunk.startSec,
          chunk.durationSec,
          {
            candidateVariant,
            forceEndingProtection: true,
            skipSpeechSegmentation: true,
          },
        );
        const chunkPolish = await runFinalAppPolishPass(ffmpeg, mixChunkName, profile, {
          base: job.base,
          detail: `File ${fileIndex + 1}/${totalFiles}, part ${partLabel}, final app polish`,
          candidateVariant,
        });
        ffmpeg = chunkPolish.ffmpeg;
        const chunkFlow = chunkPolish.applied ? "app-final-polish" : "app";
        await applyFinalConsonantPeakPolish(ffmpeg, mixChunkName, `${job.base} ${partTag}`);
        await assertDurationDeltaWithin(ffmpeg, `Long-form mix ${job.base} ${partTag}`, chunk.durationSec, mixChunkName);
        await assertTruePeakWithin(ffmpeg, `Long-form mix ${job.base} ${partTag}`, mixChunkName);

        if (keepMixReady || loudnessConfig === null) {
          stagedOutputs.push(
            await writeOutput(ffmpeg, mixChunkName, "mixready", "clean", chunkFlow, {
              sourceBase: job.base,
              sourceName: job.file.name,
              partIndex: chunk.index,
              partTotal: chunk.total,
            }),
          );
        }

        if (sceneBlend) {
          const indoorGain = profile?.blendIndoorGain ?? 0;
          const outdoorGain = profile?.blendOutdoorGain ?? 0;
          if (indoorGain + outdoorGain <= 0.0001) {
            appendLog(`[Blend] ${job.base} ${partTag}: bypassed (adaptive blend gain near zero).`);
          } else {
            setStatus(`Long-form blend: ${job.base} (${fileIndex + 1}/${totalFiles}, part ${partLabel})`);
            setActiveQueueStage(job.base, "Long-form blend", `File ${fileIndex + 1}/${totalFiles}, part ${partLabel}`);
            await runBlendMixReady(ffmpeg, mixChunkName, blendChunkName, profile);
            await applyFinalConsonantPeakPolish(ffmpeg, blendChunkName, `${job.base} ${partTag} blend`);
            await assertDurationDeltaWithin(ffmpeg, `Long-form blend ${job.base} ${partTag}`, chunk.durationSec, blendChunkName);
            await assertTruePeakWithin(ffmpeg, `Long-form blend ${job.base} ${partTag}`, blendChunkName);
            blendRendered = true;
            if (keepMixReady || loudnessConfig === null) {
              stagedOutputs.push(
                await writeOutput(ffmpeg, blendChunkName, "mixready", "blend", chunkFlow, {
                  sourceBase: job.base,
                  sourceName: job.file.name,
                  partIndex: chunk.index,
                  partTotal: chunk.total,
                }),
              );
            }
          }
        }

        if (loudnessConfig && loudChunkName) {
          setStatus(`Long-form loudness: ${job.base} (${fileIndex + 1}/${totalFiles}, part ${partLabel})`);
          setActiveQueueStage(job.base, "Long-form loudness", `File ${fileIndex + 1}/${totalFiles}, part ${partLabel}`);
          await runLoudnorm(ffmpeg, mixChunkName, loudChunkName, loudnessConfig);
          await assertDurationDeltaWithin(ffmpeg, `Long-form loudness ${job.base} ${partTag}`, chunk.durationSec, loudChunkName);
          await assertTruePeakWithin(ffmpeg, `Long-form loudness ${job.base} ${partTag}`, loudChunkName);
          stagedOutputs.push(
            await writeOutput(ffmpeg, loudChunkName, "loudness", "clean", chunkFlow, {
              sourceBase: job.base,
              sourceName: job.file.name,
              partIndex: chunk.index,
              partTotal: chunk.total,
            }),
          );

          if (blendRendered && blendLoudChunkName) {
            setStatus(`Long-form blend loudness: ${job.base} (${fileIndex + 1}/${totalFiles}, part ${partLabel})`);
            setActiveQueueStage(
              job.base,
              "Long-form blend loudness",
              `File ${fileIndex + 1}/${totalFiles}, part ${partLabel}`,
            );
            await runLoudnorm(ffmpeg, blendChunkName, blendLoudChunkName, loudnessConfig);
            await assertDurationDeltaWithin(
              ffmpeg,
              `Long-form blend loudness ${job.base} ${partTag}`,
              chunk.durationSec,
              blendLoudChunkName,
            );
            await assertTruePeakWithin(ffmpeg, `Long-form blend loudness ${job.base} ${partTag}`, blendLoudChunkName);
            stagedOutputs.push(
              await writeOutput(ffmpeg, blendLoudChunkName, "loudness", "blend", chunkFlow, {
                sourceBase: job.base,
                sourceName: job.file.name,
                partIndex: chunk.index,
                partTotal: chunk.total,
              }),
            );
          }
        }

        appendLog(
          `[LongForm] ${job.base}: exported ${partTag} (${chunk.startSec.toFixed(1)}s-${(
            chunk.startSec + chunk.durationSec
          ).toFixed(1)}s).`,
        );
      } finally {
        await safeDeleteFile(ffmpeg, mixChunkName);
        await safeDeleteFile(ffmpeg, blendChunkName);
        if (loudChunkName) await safeDeleteFile(ffmpeg, loudChunkName);
        if (blendLoudChunkName) await safeDeleteFile(ffmpeg, blendLoudChunkName);
      }

      if (
        chunk.index < chunks.length - 1 &&
        (chunk.index + 1) % LONG_FORM_WORKER_REFRESH_INTERVAL === 0
      ) {
        ffmpeg = await refreshFfmpeg(`long-form chunk guard on ${job.base} (${partLabel})`);
        await writeJobInput(ffmpeg, job);
      }
    }

    outputEntries.push(...stagedOutputs);
    setOutputs([...outputEntries]);
    appendLog(
      `[LongForm] ${job.base}: complete. Review bundles and duplicate candidate renders were skipped to keep this PC cool and memory-bounded.`,
    );
    return { ffmpeg, chunkCount: chunks.length };
  };

  const safeDeleteFile = async (ffmpeg: FFmpeg, name: string) => {
    try {
      await ffmpeg.deleteFile(name);
    } catch {
      // Ignore cleanup failures from missing temp files.
    }
  };

  /**
   * Stitch `inputNames` into `outputName` using iterative `acrossfade=d=<d>`.
   *
   * CRITICAL: every input except the LAST one must have been rendered with
   * an extra `crossfadeSec` of source material at its end (i.e. rendered
   * span = native_span + crossfadeSec). The crossfade between input i and
   * input i+1 consumes that overlap, so the final output length equals the
   * sum of NATIVE spans — sample-accurate timing relative to the source.
   *
   * This fixes two classes of boundary artifacts that `-c copy` hard concat
   * leaves behind: per-segment filter-state restart transients, and
   * gain-curve step discontinuities in planner chunks at speech/silence
   * edges. At 20 ms the crossfade is well below the audibility threshold
   * for a dub edit (which is ~40 ms) so nothing sounds "smeared".
   */
  const runCrossfadeConcat = async (
    ffmpeg: FFmpeg,
    inputNames: string[],
    outputName: string,
    crossfadeSec: number,
    context: string,
  ) => {
    if (inputNames.length === 0) {
      throw new Error("No segments to concat.");
    }
    if (inputNames.length === 1) {
      const bytes = await readVirtualFileBytes(ffmpeg, inputNames[0]);
      await ffmpeg.writeFile(outputName, bytes);
      return;
    }
    const scratchNames: string[] = [];
    try {
      let accum = inputNames[0];
      for (let i = 1; i < inputNames.length; i += 1) {
        const next = inputNames[i];
        const isFinal = i === inputNames.length - 1;
        const targetName = isFinal ? outputName : `${sanitizeBase(outputName)}_xfade_${i}.wav`;
        if (!isFinal) scratchNames.push(targetName);
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
            accum,
            "-i",
            next,
            "-filter_complex",
            `[0:a][1:a]acrossfade=d=${crossfadeSec.toFixed(3)}:c1=tri:c2=tri`,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            targetName,
          ],
          `${context} [${i}/${inputNames.length - 1}]`,
        );
        accum = targetName;
      }
    } finally {
      for (const scratch of scratchNames) {
        await safeDeleteFile(ffmpeg, scratch);
      }
    }
  };

  const logDurationDelta = async (
    ffmpeg: FFmpeg,
    context: string,
    inputDurationSeconds: number,
    outputName: string,
  ) => {
    if (!Number.isFinite(inputDurationSeconds) || inputDurationSeconds <= 0) return;
    try {
      const outputDurationSeconds = await probeInputDurationSeconds(ffmpeg, outputName);
      if (outputDurationSeconds === null || !Number.isFinite(outputDurationSeconds)) return;
      appendLog(
        `[Duration] ${context}: input ${inputDurationSeconds.toFixed(3)}s -> output ${outputDurationSeconds.toFixed(
          3,
        )}s (delta ${formatSigned(outputDurationSeconds - inputDurationSeconds, 3)}s).`,
      );
    } catch {
      // Ignore probe failures for post-render diagnostics.
    }
  };

  const assertDurationDeltaWithin = async (
    ffmpeg: FFmpeg,
    context: string,
    inputDurationSeconds: number,
    outputName: string,
    maxDeltaSeconds = HEAD_PRIME_DURATION_TOLERANCE_SECONDS,
  ) => {
    if (!Number.isFinite(inputDurationSeconds) || inputDurationSeconds <= 0) return;
    const outputDurationSeconds = await probeInputDurationSeconds(ffmpeg, outputName);
    if (outputDurationSeconds === null || !Number.isFinite(outputDurationSeconds)) {
      throw new Error(`${context}: could not verify output duration`);
    }
    const deltaSeconds = outputDurationSeconds - inputDurationSeconds;
    appendLog(
      `[Duration] ${context}: input ${inputDurationSeconds.toFixed(3)}s -> output ${outputDurationSeconds.toFixed(
        3,
      )}s (delta ${formatSigned(deltaSeconds, 3)}s).`,
    );
    if (Math.abs(deltaSeconds) > maxDeltaSeconds) {
      throw new Error(`${context}: duration delta ${deltaSeconds.toFixed(3)}s exceeds ${maxDeltaSeconds.toFixed(2)}s`);
    }
  };

  const assertTruePeakWithin = async (ffmpeg: FFmpeg, context: string, inputName: string, maxTruePeakDb = -1.5) => {
    const loudness = await analyzeIntegratedLoudness(ffmpeg, inputName);
    if (loudness.inputTP === null || !Number.isFinite(loudness.inputTP)) {
      throw new Error(`${context}: could not verify true peak`);
    }
    if (loudness.inputTP > maxTruePeakDb) {
      throw new Error(`${context}: true peak ${loudness.inputTP.toFixed(2)} dBTP exceeds ${maxTruePeakDb.toFixed(1)} dBTP`);
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
    plannedSegmentCount: number;
    longSparseMode: boolean;
    mode: "speech-aligned" | "speech-pause";
  };

  type CandidateVariant = NonNullable<MixRenderOptions["candidateVariant"]>;

  type CandidateReviewArtifact = {
    variant: CandidateVariant;
    label: string;
    bytes: Uint8Array;
    analysis: FileAnalysis | null;
    meta: CandidateRenderMeta;
    baselineScore: CandidateScore;
    scoredScore: CandidateScore;
    qcSnapshot: ReviewMetricSnapshot | null;
    qcDelta: ReviewMetricDelta | null;
    alignment: AlignmentMetrics;
    ranking: CandidateRankingBreakdown;
    selectionReason: string | null;
  };

  type RenderAttemptResult = {
    ffmpeg: FFmpeg;
    meta: CandidateRenderMeta;
    speechAlignedSegmentCountUsed: number | null;
  };

  type PlannerRenderContext = {
    plan: PlannedGain | null;
    leveledInputName: string | null;
    leveledReady: boolean;
    applyChunkSeconds: number | null;
  };

  const analysisLooksSpeechBearing = (analysis: FileAnalysis | undefined, durationSeconds: number | null) => {
    if ((analysis?.speechSegmentCount ?? 0) > 0) return true;
    if ((analysis?.speechDutyCyclePct ?? 0) >= 0.8) return true;
    return durationSeconds !== null && durationSeconds >= 8 && (analysis?.analysisConfidence ?? 0) >= 0.3;
  };

  const preparePlannerRenderContext = async (
    ffmpeg: FFmpeg,
    job: JobEntry,
    profile: AdaptiveProfile | null,
    analysis: FileAnalysis | undefined,
    durationSeconds: number | null,
  ): Promise<PlannerRenderContext> => {
    const context: PlannerRenderContext = {
      plan: null,
      leveledInputName: null,
      leveledReady: false,
      applyChunkSeconds: null,
    };
    if (!getActiveAudioReviewControls().gainPlannerEnabled) return context;

    const plan = await planGainForInput(ffmpeg, job.inputName, profile, analysis, durationSeconds);
    if (!plan) {
      if (analysisLooksSpeechBearing(analysis, durationSeconds)) {
        throw new Error("speech-aware planner produced no plan on speech-bearing input");
      }
      appendLog(`[Planner] ${job.base}: bypassed (no-op; short or silent input).`);
      return context;
    }

    const breathNote =
      plan.breathRunCount > 0
        ? `, ${plan.breathRunCount} breath/transient run${plan.breathRunCount === 1 ? "" : "s"} tamed`
        : "";
    const speechSpikeNote =
      plan.speechSpikeFrameCount > 0
        ? `, ${plan.speechSpikeFrameCount} speech-spike frame${plan.speechSpikeFrameCount === 1 ? "" : "s"} tamed (max ${plan.speechSpikeMaxReductionDb.toFixed(1)} dB)`
        : "";
    const sustainedLoudNote =
      plan.sustainedLoudClusterCount > 0
        ? `, ${plan.sustainedLoudClusterCount} sustained-loud cluster${plan.sustainedLoudClusterCount === 1 ? "" : "s"} tamed (max ${plan.sustainedLoudMaxReductionDb.toFixed(1)} dB)`
        : "";
    const earlyRunCapNote =
      plan.earlyRunCapCount > 0
        ? `, ${plan.earlyRunCapCount} hot opener${plan.earlyRunCapCount === 1 ? "" : "s"} capped (max ${plan.earlyRunMaxReductionDb.toFixed(1)} dB)`
        : "";
    const coldOpenLiftNote =
      plan.coldOpenLiftCount > 0
        ? `, ${plan.coldOpenLiftCount} quiet opener${plan.coldOpenLiftCount === 1 ? "" : "s"} lifted (max ${plan.coldOpenLiftMaxDb.toFixed(1)} dB)`
        : "";
    const tailRescueNote =
      plan.tailRescueRunCount > 0
        ? `, ${plan.tailRescueRunCount} soft tail${plan.tailRescueRunCount === 1 ? "" : "s"} held (${plan.tailRescueFrameCount} frames, max ${plan.tailRescueMaxMs.toFixed(0)} ms)`
        : "";
    appendLog(
      `[Planner] ${job.base}: leveled ${plan.speechRunCount} speech runs to ${plan.targetDb.toFixed(
        1,
      )} dB (expander ${plan.expanderDepthDb.toFixed(1)} dB, micro-ride +/-${plan.microRideDb.toFixed(
        2,
      )} dB${breathNote}${speechSpikeNote}${sustainedLoudNote}${earlyRunCapNote}${coldOpenLiftNote}${tailRescueNote}).`,
    );

    context.plan = plan;
    context.leveledInputName = `${job.base}_planner_leveled.wav`;
    return context;
  };

  const formatCandidateVariant = (variant: CandidateVariant) =>
    variant === "cinematic-stable"
      ? "cinematic-stable"
      : variant === "continuity-safe"
        ? "continuity-safe"
        : variant === "pause-safe"
          ? "pause-safe"
          : "core-safe";

  const buildMixCandidateVariants = (
    profile: AdaptiveProfile | null,
    durationSeconds: number | null,
  ): CandidateVariant[] => {
    const sourceFirstVariant = sourceFirstCandidateVariantRef.current;
    if (sourceFirstVariant) return [sourceFirstVariant];

    let variant: CandidateVariant = profile?.preferSinglePassContinuity ? "continuity-safe" : "cinematic-stable";
    const longFile = durationSeconds !== null && durationSeconds >= LONG_FILE_DURATION_SECONDS;
    const severeNoise =
      (profile?.pauseNoiseRisk ?? 0) >= 0.55 ||
      profile?.noiseRisk === "high" ||
      ((profile?.pauseNoiseRisk ?? 0) >= 0.4 && profile?.noiseRisk === "medium");
    const longFilePauseCandidate =
      longFile && ((profile?.pauseNoiseRisk ?? 0) >= 0.4 || profile?.noiseRisk === "medium");
    if (severeNoise || longFilePauseCandidate) {
      variant = "pause-safe";
    } else if (
      !longFile &&
      ((profile?.pauseNoiseRisk ?? 0) >= 0.32 ||
        profile?.noiseRisk === "medium" ||
        profile?.noiseRisk === "high")
    ) {
      variant = "pause-safe";
    }

    return [variant];
  };

  const buildCandidateScore = (analysis: FileAnalysis | null): CandidateScore => {
    if (!analysis) {
      return {
        stability: 1,
        pause: 1,
        compression: 1,
        echo: 1,
        total: 1111,
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

  const summarizeCandidateScore = (score: CandidateScore) =>
    `stability ${(score.stability * 100).toFixed(0)} / pause ${(score.pause * 100).toFixed(0)} / compression ${(
      score.compression * 100
    ).toFixed(0)} / echo ${(score.echo * 100).toFixed(0)}${
      typeof score.rankingScore === "number" && Number.isFinite(score.rankingScore)
        ? ` / rank ${score.rankingScore.toFixed(1)}`
        : ""
    }${score.gateReasons && score.gateReasons.length > 0 ? ` / gates ${score.gateReasons.join("+")}` : ""}`;

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

    const plannedSegments = buildSpeechAlignedRenderSegments(
      silenceMap.speechSpans,
      silenceMap.silenceSpans,
      durationSeconds,
      profile
    );

    return {
      durationSeconds,
      silenceSpans: silenceMap.silenceSpans,
      speechSpans: silenceMap.speechSpans,
      plannedSegmentCount: plannedSegments.filter((segment) => segment.process).length,
      longSparseMode: profile?.longSparseModeEligible ?? false,
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
    speechRenderPlan?: SpeechRenderPlan | null,
    analysis?: FileAnalysis | undefined,
    plannerContext?: PlannerRenderContext | null,
  ): Promise<RenderAttemptResult> => {
    const hasRoomFilters =
      getActiveAudioReviewControls().roomCleanup && !!profile && (profile.useTailGate || profile.echoNotchCutDb >= 0.25);
    const hasAdaptiveNoiseReduction = resolveAdaptiveNoiseReductionFilter(profile, options) !== null;
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
    fallbackStrategies.push({
      label: PLANNER_TAIL_SAFE_STRATEGY_LABEL,
      options: {
        candidateVariant: "source-safe",
        sourceSafeChain: true,
        disableRoomCleanup: true,
        disableAdaptiveNoiseReduction: true,
        disableSegmentGainMatch: true,
        skipSpeechSegmentation: true,
        disableHeadPriming: true,
        minimalStabilityChain: true,
        disableTailGate: true,
        disableSpikeTamers: true,
      },
    });
    fallbackStrategies.push({
      label: AUDIBILITY_SAFE_STRATEGY_LABEL,
      options: {
        candidateVariant: "source-safe",
        sourceSafeChain: true,
        disableRoomCleanup: true,
        disableAdaptiveNoiseReduction: true,
        disableSegmentGainMatch: true,
        disableHeadPriming: true,
        skipSpeechSegmentation: true,
        minimalStabilityChain: true,
      },
    });
    fallbackStrategies.push({
      label: "audibility passthrough",
      options: {
        candidateVariant: "source-safe",
        disableRoomCleanup: true,
        disableAdaptiveNoiseReduction: true,
        disableSegmentGainMatch: true,
        disableGainPlanner: true,
        disableHeadPriming: true,
        skipSpeechSegmentation: true,
        sourceSafeChain: true,
        sourcePassthroughChain: true,
      },
    });

    let lastMixError: unknown = null;
    let mixRendered = false;
    let inputDurationSeconds: number | null | undefined = speechRenderPlan?.durationSeconds;
    let speechAlignedSegmentCountUsed: number | null = null;
    let renderMeta: CandidateRenderMeta | null = null;
    let sourceAudibilityFrameDb: number[] | null = null;
    let audibilityGuardTripped = false;

    const ensureInputDuration = async () => {
      if (inputDurationSeconds !== undefined) return inputDurationSeconds;
      try {
        inputDurationSeconds = await probeInputDurationSeconds(ffmpeg, job.inputName);
      } catch {
        inputDurationSeconds = null;
      }
      return inputDurationSeconds;
    };

    const noteAudibilityGuardReason = (reasons: DegradeReason[]) => {
      if (!reasons.includes("audibility-dropout-guard")) {
        reasons.push("audibility-dropout-guard");
      }
    };

    const assertRenderedAudibility = async (strategyLabel: string, renderPath: RenderPath) => {
      const durationSeconds = await ensureInputDuration();
      if (
        durationSeconds === null ||
        !Number.isFinite(durationSeconds) ||
        durationSeconds <= 1 ||
        durationSeconds > AUDIBILITY_GUARD_MAX_DURATION_SECONDS
      ) {
        return;
      }

      try {
        if (!sourceAudibilityFrameDb) {
          sourceAudibilityFrameDb = await renderAudibilityFrameDb(ffmpeg, job.inputName, `${job.base}_source`);
        }
        const renderedFrameDb = await renderAudibilityFrameDb(ffmpeg, outputName, `${job.base}_${strategyLabel}`);
        if (sourceAudibilityFrameDb.length === 0 || renderedFrameDb.length === 0) return;
        const report = detectAudibilityDropouts({
          sourceFrameDb: sourceAudibilityFrameDb,
          renderedFrameDb,
          frameMs: AUDIBILITY_GUARD_FRAME_MS,
        });
        if (!report.severe) return;

        audibilityGuardTripped = true;
        const firstCluster = report.clusters[0] ?? null;
        const clusterNote = firstCluster
          ? `, frames ${firstCluster.startFrame}-${firstCluster.endFrame}`
          : "";
        appendLog(
          `[AudibilityGuard] tripped: strategy=${strategyLabel}, tailRescueRunCount=${
            plan?.tailRescueRunCount ?? "n/a"
          }, tailRescueMaxMs=${plan ? plan.tailRescueMaxMs.toFixed(0) : "n/a"}${clusterNote}.`,
        );
        throw makeAudibilityGuardError(
          `[AudibilityGuard] ${job.base}/${strategyLabel}/${describeRenderPath(renderPath)}`,
          report,
        );
      } catch (error) {
        if (isAudibilityGuardError(error)) throw error;
        appendLog(`[AudibilityGuard] ${job.base}/${strategyLabel}: skipped (${describeError(error)}).`);
      }
    };

    // Candidate renders normally receive a file-level planner context so all
    // variants share the same gain curve. The direct-planning branch is kept
    // defensive for any future caller that has not prepared the context.
    let plan: PlannedGain | null = plannerContext?.plan ?? null;
    if (!plannerContext && getActiveAudioReviewControls().gainPlannerEnabled) {
      try {
        const dur = await ensureInputDuration();
        plan = await planGainForInput(ffmpeg, job.inputName, profile, analysis, dur);
        if (plan) {
          const breathNote = plan.breathRunCount > 0
            ? `, ${plan.breathRunCount} breath/transient run${plan.breathRunCount === 1 ? "" : "s"} tamed`
            : "";
          const speechSpikeNote =
            plan.speechSpikeFrameCount > 0
              ? `, ${plan.speechSpikeFrameCount} speech-spike frame${plan.speechSpikeFrameCount === 1 ? "" : "s"} tamed (max ${plan.speechSpikeMaxReductionDb.toFixed(1)} dB)`
              : "";
          const sustainedLoudNote =
            plan.sustainedLoudClusterCount > 0
              ? `, ${plan.sustainedLoudClusterCount} sustained-loud cluster${plan.sustainedLoudClusterCount === 1 ? "" : "s"} tamed (max ${plan.sustainedLoudMaxReductionDb.toFixed(1)} dB)`
              : "";
          const earlyRunCapNote =
            plan.earlyRunCapCount > 0
              ? `, ${plan.earlyRunCapCount} hot opener${plan.earlyRunCapCount === 1 ? "" : "s"} capped (max ${plan.earlyRunMaxReductionDb.toFixed(1)} dB)`
              : "";
          const coldOpenLiftNote =
            plan.coldOpenLiftCount > 0
              ? `, ${plan.coldOpenLiftCount} quiet opener${plan.coldOpenLiftCount === 1 ? "" : "s"} lifted (max ${plan.coldOpenLiftMaxDb.toFixed(1)} dB)`
              : "";
          const tailRescueNote =
            plan.tailRescueRunCount > 0
              ? `, ${plan.tailRescueRunCount} soft tail${plan.tailRescueRunCount === 1 ? "" : "s"} held (${plan.tailRescueFrameCount} frames, max ${plan.tailRescueMaxMs.toFixed(0)} ms)`
              : "";
          appendLog(
            `[Planner] ${job.base}: leveled ${plan.speechRunCount} speech runs to ${plan.targetDb.toFixed(1)} dB (expander ${plan.expanderDepthDb.toFixed(1)} dB, micro-ride \u00b1${plan.microRideDb.toFixed(2)} dB${breathNote}${speechSpikeNote}${sustainedLoudNote}${earlyRunCapNote}${coldOpenLiftNote}${tailRescueNote}).`,
          );
        } else {
          appendLog(`[Planner] ${job.base}: bypassed (no-op \u2014 short or silent input).`);
        }
      } catch (error) {
        appendLog(
          `[Planner] ${job.base}: failed (${error instanceof Error ? error.message : String(error)}). No legacy fallback emitted.`,
        );
        if (shouldResetFfmpegForError(error)) {
          ffmpeg = await refreshFfmpeg(`planner failure on ${job.base}`);
          await writeJobInput(ffmpeg, job);
        }
        throw error;
      }
    }

    // Pre-level the full file when a plan exists. Cache the rendered planner
    // WAV across candidate variants, and recreate it after an ffmpeg worker
    // refresh because refresh wipes the virtual FS.
    const leveledInputName = plan ? (plannerContext?.leveledInputName ?? `${job.base}_planner_leveled.wav`) : null;
    let leveledReady = plannerContext?.leveledReady ?? false;
    const ensureLeveledInput = async (allowPlanner = true): Promise<string | null> => {
      if (!allowPlanner || !plan || !leveledInputName) return null;
      if (leveledReady) {
        try {
          await ffmpeg.readFile(leveledInputName);
          return leveledInputName;
        } catch {
          leveledReady = false;
          if (plannerContext) plannerContext.leveledReady = false;
        }
      }
      let lastApplyError: unknown = null;
      for (const retryChunkSeconds of PLANNER_APPLY_RETRY_CHUNK_SECONDS) {
        try {
          if (retryChunkSeconds !== null) {
            appendLog(`[Planner] ${job.base}: retrying apply with ${retryChunkSeconds}s chunks.`);
          }
          await applyPlannerToFullInput(ffmpeg, job.inputName, leveledInputName, plan, retryChunkSeconds);
          leveledReady = true;
          if (plannerContext) {
            plannerContext.leveledReady = true;
            plannerContext.applyChunkSeconds = retryChunkSeconds;
          }
          return leveledInputName;
        } catch (error) {
          lastApplyError = error;
          leveledReady = false;
          if (plannerContext) plannerContext.leveledReady = false;
          if (shouldResetFfmpegForError(error)) {
            ffmpeg = await refreshFfmpeg(`planner apply on ${job.base}`);
            await writeJobInput(ffmpeg, job);
          }
        }
      }

      throw new Error(
        `Planner apply failed after chunk retries (${
          lastApplyError instanceof Error ? lastApplyError.message : String(lastApplyError)
        }); skipped to avoid planner-off legacy output.`,
      );
    };

    const buildMeta = (
      strategyLabel: string,
      renderPath: RenderPath,
      degradeReasons: DegradeReason[]
    ): CandidateRenderMeta => {
      const reasons = Array.from(new Set(degradeReasons));
      return {
        strategyLabel,
        renderPath,
        segmentedHealthy:
          (renderPath === "speech-pause-segmented" ||
            renderPath === "speech-aligned-segmented" ||
            renderPath === "fixed-segmented") &&
          !reasons.includes("segment-render-memory-fault") &&
          !reasons.includes("single-pass-recovery"),
        degraded: reasons.length > 0,
        degradeReasons: reasons,
        analysisWindowsAttempted: 0,
        analysisWindowsSucceeded: 0,
        analysisWindowsDropped: 0,
      };
    };

    for (let strategyIndex = 0; strategyIndex < fallbackStrategies.length; strategyIndex += 1) {
      const strategy = fallbackStrategies[strategyIndex];
      const effectiveOptions = { ...options, ...strategy.options };
      const strategyDegradeReasons: DegradeReason[] = [];
      if (audibilityGuardTripped) {
        noteAudibilityGuardReason(strategyDegradeReasons);
      }
      try {
        if (speechRenderPlan && !effectiveOptions.skipSpeechSegmentation) {
          const plannedSegments = buildSpeechAlignedRenderSegments(
            speechRenderPlan.speechSpans,
            speechRenderPlan.silenceSpans,
            speechRenderPlan.durationSeconds,
            profile
          );
          const mergedSegments = mergeRenderSegmentsForRisk(
            plannedSegments,
            0.6,
            18
          );
          const useRoomCleanupForStrategy = hasRoomFilters && !effectiveOptions.disableRoomCleanup;
          const useAdaptiveNoiseReductionForStrategy =
            hasAdaptiveNoiseReduction && !effectiveOptions.disableAdaptiveNoiseReduction;
          let renderRisk: RenderRiskProfile = buildRenderRiskProfile({
            durationSeconds: speechRenderPlan.durationSeconds,
            longSparseMode: speechRenderPlan.longSparseMode,
            plannedSegmentCount: speechRenderPlan.plannedSegmentCount,
            speechSpanCount: speechRenderPlan.speechSpans.length,
            candidateVariant: effectiveOptions.candidateVariant ?? "cinematic-stable",
            useRoomCleanup: useRoomCleanupForStrategy,
            useAdaptiveNoiseReduction: useAdaptiveNoiseReductionForStrategy,
            priorFatalRenderError: false,
            sentenceJumpScore: profile?.sentenceJumpScore ?? 0,
            mergedSegmentCount: countProcessedRenderSegments(mergedSegments),
          });

          if (renderRisk.recycleWorkerBeforeRender) {
            ffmpeg = await refreshFfmpeg(`render risk preflight on ${job.base}/${strategy.label}`);
            await writeJobInput(ffmpeg, job);
          }

          const runFixedSegmentation = async () => {
            setStatus(`${stageLabel}: ${job.base} (${fileIndex + 1}/${totalFiles})`);
            setActiveQueueStage(job.base, stageLabel, `File ${fileIndex + 1} of ${totalFiles}`);
            const leveled = await ensureLeveledInput(!effectiveOptions.disableGainPlanner);
            await runMixReadySegmented(
              ffmpeg,
              job.inputName,
              outputName,
              profile,
              speechRenderPlan.durationSeconds,
              {
                ...effectiveOptions,
                segmentMode: "fixed",
              },
              leveled,
            );
            await assertRenderedAudibility(strategy.label, "fixed-segmented");
            speechAlignedSegmentCountUsed = Math.ceil(speechRenderPlan.durationSeconds / MIX_SEGMENT_SECONDS);
            mixRendered = true;
            renderMeta = buildMeta(strategy.label, "fixed-segmented", strategyDegradeReasons);
            appendLog(
              `[Segmented] ${job.base}: fixed render used ${speechAlignedSegmentCountUsed} segments (${formatCandidateVariant(
                effectiveOptions.candidateVariant ?? "cinematic-stable"
              )}, planner=${leveled ? "on" : "off"}).`
            );
          };

          const runSpeechSegmentation = async (segments: RenderSegment[], segmentLite: boolean) => {
            setStatus(`${stageLabel}: ${job.base} (${fileIndex + 1}/${totalFiles})`);
            setActiveQueueStage(job.base, stageLabel, `File ${fileIndex + 1} of ${totalFiles}`);
            const leveled = await ensureLeveledInput(!effectiveOptions.disableGainPlanner);
            speechAlignedSegmentCountUsed = await runMixReadySpeechAlignedSegmented(
              ffmpeg,
              job.inputName,
              outputName,
              profile,
              speechRenderPlan.durationSeconds,
              segments,
              {
                ...effectiveOptions,
                segmentMode: speechRenderPlan.mode,
                segmentLite,
                disableSegmentGainMatch:
                  effectiveOptions.disableSegmentGainMatch || (segmentLite ? true : renderRisk.disableSegmentGainMatch),
              },
              leveled,
            );
            await assertRenderedAudibility(
              strategy.label,
              speechRenderPlan.mode === "speech-pause" ? "speech-pause-segmented" : "speech-aligned-segmented",
            );
            mixRendered = true;
            renderMeta = buildMeta(
              strategy.label,
              speechRenderPlan.mode === "speech-pause" ? "speech-pause-segmented" : "speech-aligned-segmented",
              strategyDegradeReasons
            );
            appendLog(
              `[Segmented] ${job.base}: ${speechRenderPlan.mode} render used ${speechAlignedSegmentCountUsed} segments (${formatCandidateVariant(
                effectiveOptions.candidateVariant ?? "cinematic-stable"
              )}${segmentLite ? ", lite" : ""}, planner=${leveled ? "on" : "off"}).`
            );
          };

          if (renderRisk.shouldUseFixedSegmentation) {
            appendLog(
              `[Segmented] ${job.base}: high render risk on ${strategy.label}, using fixed segmentation preflight.`
            );
            try {
              await runFixedSegmentation();
              break;
            } catch (fixedError) {
              lastMixError = fixedError;
              if (isAudibilityGuardError(fixedError)) {
                noteAudibilityGuardReason(strategyDegradeReasons);
                throw fixedError;
              }
              if (shouldResetFfmpegForError(fixedError)) {
                strategyDegradeReasons.push("segment-render-memory-fault");
                ffmpeg = await refreshFfmpeg(`fixed-segmented fallback on ${job.base}`);
                await writeJobInput(ffmpeg, job);
              }
            }
          } else {
            const primarySegments =
              renderRisk.level === "high" && countProcessedRenderSegments(mergedSegments) < countProcessedRenderSegments(plannedSegments)
                ? mergedSegments
                : plannedSegments;
            try {
              await runSpeechSegmentation(primarySegments, false);
              break;
            } catch (segError) {
              lastMixError = segError;
              if (isAudibilityGuardError(segError)) {
                noteAudibilityGuardReason(strategyDegradeReasons);
                throw segError;
              }
              appendLog(
                `[Segmented] ${job.base}: ${speechRenderPlan.mode} ${strategy.label} failed (${describeError(
                  segError
                )}), trying segmented-lite ${strategy.label}.`
              );
              if (shouldResetFfmpegForError(segError)) {
                strategyDegradeReasons.push("segment-render-memory-fault");
                ffmpeg = await refreshFfmpeg(`${speechRenderPlan.mode} mix fallback on ${job.base}`);
                await writeJobInput(ffmpeg, job);
              }
              renderRisk = buildRenderRiskProfile({
                durationSeconds: speechRenderPlan.durationSeconds,
                longSparseMode: speechRenderPlan.longSparseMode,
                plannedSegmentCount: speechRenderPlan.plannedSegmentCount,
                speechSpanCount: speechRenderPlan.speechSpans.length,
                candidateVariant: effectiveOptions.candidateVariant ?? "cinematic-stable",
                useRoomCleanup: useRoomCleanupForStrategy,
                useAdaptiveNoiseReduction: useAdaptiveNoiseReductionForStrategy,
                priorFatalRenderError: strategyDegradeReasons.includes("segment-render-memory-fault"),
                sentenceJumpScore: profile?.sentenceJumpScore ?? 0,
                mergedSegmentCount: countProcessedRenderSegments(mergedSegments),
              });
              if (!renderRisk.shouldUseFixedSegmentation) {
                try {
                  await runSpeechSegmentation(mergedSegments, true);
                  break;
                } catch (liteError) {
                  lastMixError = liteError;
                  if (isAudibilityGuardError(liteError)) {
                    noteAudibilityGuardReason(strategyDegradeReasons);
                    throw liteError;
                  }
                  appendLog(
                    `[Segmented] ${job.base}: segmented-lite ${strategy.label} failed (${describeError(
                      liteError
                    )}), trying fixed segmented ${strategy.label}.`
                  );
                  if (shouldResetFfmpegForError(liteError)) {
                    if (!strategyDegradeReasons.includes("segment-render-memory-fault")) {
                      strategyDegradeReasons.push("segment-render-memory-fault");
                    }
                    ffmpeg = await refreshFfmpeg(`segmented-lite fallback on ${job.base}`);
                    await writeJobInput(ffmpeg, job);
                  }
                }
              }

              try {
                await runFixedSegmentation();
                break;
              } catch (fixedError) {
                lastMixError = fixedError;
                if (isAudibilityGuardError(fixedError)) {
                  noteAudibilityGuardReason(strategyDegradeReasons);
                  throw fixedError;
                }
                if (shouldResetFfmpegForError(fixedError)) {
                  if (!strategyDegradeReasons.includes("segment-render-memory-fault")) {
                    strategyDegradeReasons.push("segment-render-memory-fault");
                  }
                  ffmpeg = await refreshFfmpeg(`fixed-segmented fallback on ${job.base}`);
                  await writeJobInput(ffmpeg, job);
                }
              }
            }
          }
        }

        setStatus(`${stageLabel}: ${job.base} (${fileIndex + 1}/${totalFiles})`);
        setActiveQueueStage(job.base, stageLabel, `File ${fileIndex + 1} of ${totalFiles}`);
        const leveledForSinglePass = await ensureLeveledInput(!effectiveOptions.disableGainPlanner);
        await runMixReady(ffmpeg, job.inputName, outputName, profile, effectiveOptions, leveledForSinglePass);
        if (strategyDegradeReasons.includes("segment-render-memory-fault")) {
          strategyDegradeReasons.push("single-pass-recovery");
        }
        await assertRenderedAudibility(
          strategy.label,
          strategyDegradeReasons.includes("single-pass-recovery") ? "single-pass-recovered" : "single-pass",
        );
        appendLog(
          `[SinglePass] ${job.base}: ${strategy.label} (planner=${leveledForSinglePass ? "on" : "off"}).`,
        );
        mixRendered = true;
        renderMeta = buildMeta(
          strategy.label,
          strategyDegradeReasons.includes("single-pass-recovery") ? "single-pass-recovered" : "single-pass",
          strategyDegradeReasons
        );
        break;
      } catch (error) {
        lastMixError = error;
        if (isAudibilityGuardError(error)) {
          noteAudibilityGuardReason(strategyDegradeReasons);
        }
        const strategyFailureMessage = describeError(error);
        if (shouldResetFfmpegForError(error)) {
          ffmpeg = await refreshFfmpeg(`mix fallback on ${job.base}`);
          await writeJobInput(ffmpeg, job);
        }

        const durationSeconds = await ensureInputDuration();
        const canRunSegmented =
          !isAudibilityGuardError(error) &&
          durationSeconds !== null &&
          durationSeconds >= MIX_SEGMENT_MIN_DURATION_SECONDS;

        if (canRunSegmented && durationSeconds !== null) {
          try {
            appendLog(
              `[MixFallback] ${job.base}: ${strategy.label} failed (${strategyFailureMessage}), trying segmented ${strategy.label}.`
            );
            const leveledForSeg = await ensureLeveledInput(!effectiveOptions.disableGainPlanner);
            await runMixReadySegmented(
              ffmpeg,
              job.inputName,
              outputName,
              profile,
              durationSeconds,
              effectiveOptions,
              leveledForSeg,
            );
            await assertRenderedAudibility(strategy.label, "fixed-segmented");
            mixRendered = true;
            renderMeta = buildMeta(strategy.label, "fixed-segmented", strategyDegradeReasons);
            break;
          } catch (segmentedError) {
            lastMixError = segmentedError;
            if (shouldResetFfmpegForError(segmentedError)) {
              strategyDegradeReasons.push("segment-render-memory-fault");
              ffmpeg = await refreshFfmpeg(`segmented mix fallback on ${job.base}`);
              await writeJobInput(ffmpeg, job);
            }
          }
        }

        const hasMoreStrategies = strategyIndex < fallbackStrategies.length - 1;
        if (hasMoreStrategies) {
          if (isAudibilityGuardError(error)) {
            const nextAudibilityFallbackIndex = resolveNextAudibilityFallbackIndex(fallbackStrategies, strategyIndex);
            if (nextAudibilityFallbackIndex !== null && nextAudibilityFallbackIndex > strategyIndex) {
              strategyIndex = nextAudibilityFallbackIndex - 1;
              appendLog(
                `[MixFallback] ${job.base}: audibility guard tripped on ${strategy.label}; jumping to ${
                  fallbackStrategies[nextAudibilityFallbackIndex]?.label
                }.`,
              );
              continue;
            }
          }
          const finalFailureMessage = describeError(lastMixError);
          appendLog(
            `[MixFallback] ${job.base}: ${strategy.label} failed (${finalFailureMessage}), trying ${
              fallbackStrategies[strategyIndex + 1]?.label
            }.`
          );
        }
      }
    }

    if (leveledInputName) {
      await safeDeleteFile(ffmpeg, leveledInputName);
    }

    if (!mixRendered) {
      throw lastMixError ?? new Error("Mix-ready render failed.");
    }

    if (!renderMeta) {
      renderMeta = buildMeta("primary chain", "single-pass", []);
    }

    return { ffmpeg, meta: renderMeta, speechAlignedSegmentCountUsed };
  };

  const processFiles = async () => {
    if (!files.length) return;
    processingControlsOverrideRef.current = null;
    aiAdaptiveDirectivesOverrideRef.current = null;
    sourceFirstCandidateVariantRef.current = null;
    sourceFirstPlansByBaseRef.current = new Map();
    setLoading(true);
    setOutputs([]);
    setReviewBundles([]);
    setLogs([]);
    setFailedOptimizations([]);
    setShowFailureWarning(false);
    setAiReviewFiles([]);
    setAiReviewResult(null);
    setAiReviewError(null);
    setAiReviewModel(null);
    setAiAutoPilotStatus(null);
    setStatus("Preparing...");

    try {
      let ffmpeg = await ensureFfmpeg();
      const outputEntries: OutputEntry[] = [];
      const nextReviewBundles: ReviewBundleEntry[] = [];
      const jobs = buildJobs(files);
      const reviewBundleFinalOutputBlocker = (() => {
        if (sceneBlend) return "scene-blend deliverables are rendered after the per-file review point";
        if (loudnessConfig !== null) return "loudness-target deliverables are rendered after the per-file review point";
        if (jobs.length > 1) return "batch alignment can adjust mix-ready bytes after all files render";
        return null;
      })();
      const canEmitPerFileReviewBundle = reviewBundleFinalOutputBlocker === null;
      initializeQueueItems(jobs);
      const analysisByBase = new Map<string, FileAnalysis>();
      let batchReference: BatchReference | null = null;
      const needsAnalysis = true;

      if (needsAnalysis) {
        appendLog(
          `Deep analysis started for ${jobs.length} file(s) (bootstrap up to ${ANALYSIS_SAMPLE_SECONDS}s, distributed coverage on long takes).`
        );
        const analyses: FileAnalysis[] = [];
        let analysisWorkerCumulativeAudioSec = 0;
        for (let i = 0; i < jobs.length; i += 1) {
          const job = jobs[i];
          const analysisEstDurationSec = estimateAudioSeconds(job.file);
          setStatus(`Analyze: ${job.base} (${i + 1}/${jobs.length})`);
          setActiveQueueStage(job.base, "Analyze", `Pass ${i + 1} of ${jobs.length}`);
          try {
            await writeJobInput(ffmpeg, job);
            const analysisResult = await analyzeFile(ffmpeg, job.inputName);
            ffmpeg = analysisResult.ffmpeg;
            const analysis = analysisResult.analysis;
            analysisByBase.set(job.base, analysis);
            analyses.push(analysis);
            if (analysis.coldOpenDipDb !== null) {
              appendLog(
                `[SourceQC] ${job.base}: cold-open ${analysis.coldOpenDipDb.toFixed(
                  1,
                )} dB (risk ${Math.round((analysis.coldOpenRiskScore ?? 0) * 100)}%).`,
              );
            }
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
              analysisWorkerCumulativeAudioSec = 0;
            }
          } finally {
            await safeDeleteFile(ffmpeg, job.inputName);
          }

          analysisWorkerCumulativeAudioSec += analysisEstDurationSec;
          if (analysisWorkerCumulativeAudioSec >= ANALYSIS_AUDIO_RECYCLE_SECONDS) {
            ffmpeg = await refreshFfmpeg(
              `analysis audio-volume guard (${(analysisWorkerCumulativeAudioSec / 60).toFixed(0)} min scanned since refresh)`,
            );
            analysisWorkerCumulativeAudioSec = 0;
          } else if (shouldRecycleFfmpegForBatch(i + 1, jobs.length)) {
            ffmpeg = await refreshFfmpeg(`analysis memory guard (${i + 1}/${jobs.length})`);
            analysisWorkerCumulativeAudioSec = 0;
          }
        }

        batchReference = buildBatchReference(analyses);
        if (batchReference) {
          appendLog(
            `Reference tone low/mid ${batchReference.lowTilt.toFixed(1)} dB, high/mid ${batchReference.highTilt.toFixed(1)} dB, LRA ${batchReference.lra.toFixed(1)}.`
          );
        } else {
          appendLog("Reference analysis unavailable; using base processing chain.");
        }

        const sourceReviewFiles = jobs.map((job) => {
          const sourceAnalysis = analysisByBase.get(job.base);
          const initialProfile = buildAdaptiveProfile(sourceAnalysis, batchReference);
          return buildAudioReviewFileInput(
            job,
            sourceAnalysis,
            initialProfile,
            estimateAudioSeconds(job.file),
            null,
          );
        });
        setAiReviewFiles(sourceReviewFiles);

        if (aiAutoPilotEnabled && sourceReviewFiles.length > 0) {
          setStatus("AI Auto Pilot: reviewing source");
          const result = await runAiAudioReview(sourceReviewFiles, {
            autoApply: true,
            activateCurrentRun: true,
            source: "source-auto",
          });
          if (result?.plans && result.plans.size > 0) {
            appendLog(`[AIReview] Source-first per-audio pipeline active for ${sourceReviewFiles.length} file(s).`);
          } else {
            appendLog("[AIReview] Source-first review unavailable; continuing with current deterministic controls.");
          }
        }
      }

      let hadErrors = false;
      const failedRuns: FailedOptimization[] = [];
      // Retry tracking: each file gets PER_FILE_MAX_RETRIES attempts on a
      // fresh worker before being marked permanently failed.
      const retryCounts = new Map<string, number>();
      // Cumulative audio (sec) processed by the current ffmpeg worker.
      // Reset on every refresh; drives the duration-aware recycling.
      let workerCumulativeAudioSec = 0;
      let postRenderReviewRequests = 0;
      let correctiveRenderAttempts = 0;
      const correctiveRenderBudget = resolveCorrectiveMaxFilesPerBatch(jobs.length);
      appendLog(
        `[Corrective] batch render budget ${correctiveRenderBudget}/${jobs.length} file(s) (40% cap, minimum 2).`,
      );
      let i = 0;
      while (i < jobs.length) {
        const job = jobs[i];
        const retryAttempt = retryCounts.get(job.base) ?? 0;
        let cleanLoudName: string | null = null;
        let blendLoudName: string | null = null;
        let blendRendered = false;

        // Per-file watchdog. Budget is sized from the file size estimate
        // (post-analysis we have a better number; pre-analysis we use the
        // raw byte count). On budget overrun we terminate the worker —
        // the active exec() rejects, the catch block runs, and the retry
        // path picks up the file with a fresh worker.
        const fileAnalysis = analysisByBase.get(job.base);
        const estDurationSec = Math.max(
          estimateAudioSeconds(job.file),
          fileAnalysis?.medianSpeechRunMs
            ? (fileAnalysis.medianSpeechRunMs / 1000) * Math.max(1, fileAnalysis.speechSegmentCount ?? 1)
            : 0,
        );
        const watchdogBudgetMs =
          (Math.max(WATCHDOG_BASE_SECONDS, estDurationSec * WATCHDOG_DURATION_FACTOR) +
            WATCHDOG_BASE_SECONDS) *
          1000;
        let watchdogFired = false;
        const watchdog = setTimeout(() => {
          watchdogFired = true;
          appendLog(
            `[Watchdog] ${job.base}: aborting after ${(watchdogBudgetMs / 1000).toFixed(0)}s budget exceeded.`,
          );
          try {
            ffmpegRef.current?.terminate();
          } catch {
            // terminate failures are OK — worker may already be dead
          }
        }, watchdogBudgetMs);

        try {
          if (retryAttempt > 0) {
            appendLog(
              `[Retry] ${job.base}: attempt ${retryAttempt + 1}/${PER_FILE_MAX_RETRIES + 1} on fresh worker.`,
            );
            updateQueueItem(job.base, {
              status: "working",
              stageLabel: `Retrying (${retryAttempt + 1}/${PER_FILE_MAX_RETRIES + 1})`,
              progress: 0,
              detail: "Fresh worker, re-running pipeline",
            });
          }
          const sourceFirstPlan = activateSourceFirstPlanForJob(job);
          if (sourceFirstPlan) {
            appendLog(`[AIReview] ${job.base}: ${sourceFirstPlan.summary}`);
          }
          await writeJobInput(ffmpeg, job);
          const profile = buildAdaptiveProfile(fileAnalysis, batchReference);
          const roomScore = profile ? (fileAnalysis?.roomScore ?? 0) : null;
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
              ).toFixed(0)}%, onset/sag/end ${((fileAnalysis?.onsetOvershootScore ?? 0) * 100).toFixed(
                0
              )}/${((fileAnalysis?.midLineSagScore ?? 0) * 100).toFixed(0)}/${(
                (fileAnalysis?.endFadeRiskScore ?? 0) * 100
              ).toFixed(0)}%, speech-duty ${(fileAnalysis?.speechDutyCyclePct ?? 0).toFixed(
                1
              )}%, median-run ${(((fileAnalysis?.medianSpeechRunMs ?? 0) as number) / 1000).toFixed(
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
                fileAnalysis?.analysisConfidence ?? 0
              ).toFixed(2)}, tail-gate ${profile.useTailGate ? "on" : "off"}${
                profile.preserveEndings ? " (endings protect)" : ""
              }${profile.strictEndingProtection ? " [strict]" : ""}, dyna ${dynaPreview}, echo ${
                fileAnalysis?.echoDelayMs ?? 0
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
                } speech spans and ${speechRenderPlan.silenceSpans.length} silence spans (${speechRenderPlan.plannedSegmentCount} planned segments).`
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

          let fileDurationForVariants = speechRenderPlan?.durationSeconds ?? null;
          if (fileDurationForVariants === null) {
            try {
              fileDurationForVariants = await probeInputDurationSeconds(ffmpeg, job.inputName);
            } catch {
              fileDurationForVariants = null;
            }
          }
          const longFormDurationSeconds = fileDurationForVariants ?? estimateAudioSeconds(job.file);
          const longFormSafeMode = shouldUseLongFormSafeMode(fileDurationForVariants, longFormDurationSeconds);

          if (longFormSafeMode) {
            const longFormResult = await renderLongFormSafeMode(
              ffmpeg,
              job,
              profile,
              longFormDurationSeconds,
              i,
              jobs.length,
              outputEntries,
              speechRenderPlan?.silenceSpans ?? [],
            );
            ffmpeg = longFormResult.ffmpeg;
            markQueueDone(job.base, `Long-form outputs ready (${longFormResult.chunkCount} parts)`);
            setOutputs([...outputEntries]);
          } else {
          const sourceQcSnapshot = toReviewMetricSnapshot(fileAnalysis);
          const candidateQcSafeDurationSeconds = fileDurationForVariants ?? longFormDurationSeconds;
          const useLongCandidateMemoryPolicy = candidateQcSafeDurationSeconds >= LONG_CANDIDATE_QC_SAFE_SECONDS;
          let sourceDecodedForReview: DecodedMonoAudio | null = null;
          if (useLongCandidateMemoryPolicy) {
            appendLog(
              `[ReviewBundle] ${job.base}: skipped full-file JS review decode for ${candidateQcSafeDurationSeconds.toFixed(
                0,
              )}s long file to protect memory.`,
            );
          } else {
            try {
              sourceDecodedForReview = decodeWavToMono(new Uint8Array(await job.file.arrayBuffer()));
            } catch (error) {
              appendLog(
                `[ReviewBundle] ${job.base}: source decode fallback (${
                  error instanceof Error ? error.message : String(error)
                }).`
              );
            }
          }
          if (fileDurationForVariants === null) {
            fileDurationForVariants = sourceDecodedForReview?.durationSec ?? null;
          }

          const plannerContext = await preparePlannerRenderContext(
            ffmpeg,
            job,
            profile,
            fileAnalysis,
            fileDurationForVariants,
          );
          const candidateVariants = buildMixCandidateVariants(profile, fileDurationForVariants);
          const isLongFile =
            fileDurationForVariants !== null && fileDurationForVariants >= LONG_FILE_DURATION_SECONDS;
          if (isLongFile) {
            appendLog(
              `[LongFile] ${job.base}: ${fileDurationForVariants!.toFixed(0)}s duration \u2014 using one source-first render variant; reranking is temporarily paused.`,
            );
          }
          let selectedVariant: CandidateVariant | null = null;
          let selectedBytes: Uint8Array | null = null;
          let selectedScore: CandidateScore | null = null;
          let selectedAnalysis: FileAnalysis | null = null;
          let selectedMeta: CandidateRenderMeta | null = null;
          let selectedReason: string | null = null;
          let attemptedCandidates = 0;
          let degradedCandidates = 0;
          let candidateQcMemoryFailed = false;
          const candidateArtifacts: CandidateReviewArtifact[] = [];

          for (let variantIndex = 0; variantIndex < candidateVariants.length; variantIndex += 1) {
            const candidateVariant = candidateVariants[variantIndex];
            // For long files, recycle the ffmpeg worker BEFORE each candidate
            // variant (after the first). This resets the WASM heap, avoiding
            // the slow memory creep that turns a 20-min third candidate into
            // an OOM on otherwise-healthy chains.
            if (isLongFile && variantIndex > 0) {
              ffmpeg = await refreshFfmpeg(`long-file candidate recycle on ${job.base}`);
              await writeJobInput(ffmpeg, job);
            }
            const candidateLabel = formatCandidateVariant(candidateVariant);
            const candidateName = `${job.base}_${candidateLabel.replace(/-/g, "_")}_candidate.wav`;
            const candidateOptions: MixRenderOptions = {
              candidateVariant,
              skipSpeechSegmentation:
                candidateVariant === "source-safe" ||
                (candidateVariant === "continuity-safe" && (profile?.preferSinglePassContinuity ?? false)),
              sourceSafeChain: candidateVariant === "source-safe",
              disableRoomCleanup: candidateVariant === "source-safe" ? true : undefined,
              disableAdaptiveNoiseReduction: candidateVariant === "source-safe" ? true : undefined,
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
                speechRenderPlan,
                fileAnalysis,
                plannerContext,
              );
              ffmpeg = renderResult.ffmpeg;
              attemptedCandidates += 1;
              const candidateBytes = await readVirtualFileBytes(ffmpeg, candidateName);

              let candidateAnalysis: FileAnalysis | null = null;
              let candidateMeta = renderResult.meta;
              let candidateQcUnavailable = false;
              if (useLongCandidateMemoryPolicy && candidateQcMemoryFailed) {
                candidateQcUnavailable = true;
                appendLog(
                  `[CandidateQC] ${job.base}/${candidateLabel}: skipped after previous long-file QC memory failure; using render-metadata fallback.`,
                );
                candidateMeta = {
                  ...candidateMeta,
                  degraded: true,
                  degradeReasons: Array.from(new Set([...candidateMeta.degradeReasons, "qc-unavailable"])),
                };
              } else {
                try {
                  const candidateAnalysisFfmpeg = ffmpeg;
                  const analysisResult = await analyzeFile(ffmpeg, candidateName, [job.inputName]);
                  ffmpeg = analysisResult.ffmpeg;
                  candidateAnalysis = analysisResult.analysis;
                  const degradeReasons = [...candidateMeta.degradeReasons];
                  if ((candidateAnalysis.analysisWindowRetryCount ?? 0) > 0) {
                    degradeReasons.push("analysis-window-retry");
                  }
                  if ((candidateAnalysis.analysisWindowsDropped ?? 0) > 0) {
                    degradeReasons.push("analysis-window-drop");
                  }
                  candidateMeta = {
                    ...candidateMeta,
                    degradeReasons: Array.from(new Set(degradeReasons)),
                    degraded: degradeReasons.length > 0,
                    analysisWindowsAttempted: candidateAnalysis.analysisWindowsAttempted ?? 0,
                    analysisWindowsSucceeded: candidateAnalysis.analysisWindowsSucceeded ?? 0,
                    analysisWindowsDropped: candidateAnalysis.analysisWindowsDropped ?? 0,
                  };
                  if (ffmpeg !== candidateAnalysisFfmpeg) {
                    await writeJobInput(ffmpeg, job);
                  }
                } catch (error) {
                  candidateQcUnavailable = true;
                  const shouldRefreshForQcError = shouldResetFfmpegForError(error);
                  if (useLongCandidateMemoryPolicy && (shouldRefreshForQcError || isRecoverableFailure(error))) {
                    candidateQcMemoryFailed = true;
                  }
                  appendLog(
                    `[CandidateQC] ${job.base}/${candidateLabel}: analysis fallback (${
                      error instanceof Error ? error.message : String(error)
                    }).`
                  );
                  candidateMeta = {
                    ...candidateMeta,
                    degraded: true,
                    degradeReasons: Array.from(
                      new Set([...candidateMeta.degradeReasons, "analysis-window-drop", "qc-unavailable"]),
                    ),
                  };
                  if (shouldRefreshForQcError) {
                    ffmpeg = await refreshFfmpeg(`candidate QC on ${job.base}`);
                    await writeJobInput(ffmpeg, job);
                  }
                }
              }

              const candidateBaselineScore = buildCandidateScore(candidateAnalysis);
              const candidateQcSnapshot = toReviewMetricSnapshot(candidateAnalysis);
              let candidateDecodedForReview: DecodedMonoAudio | null = null;
              if (!useLongCandidateMemoryPolicy) {
                try {
                  candidateDecodedForReview = decodeWavToMono(candidateBytes);
                } catch (error) {
                  appendLog(
                    `[CandidateQC] ${job.base}/${candidateLabel}: alignment fallback (${
                      error instanceof Error ? error.message : String(error)
                    }).`
                  );
                }
              }
              const alignment = useLongCandidateMemoryPolicy
                ? buildDurationOnlyAlignmentMetrics(
                    fileDurationForVariants,
                    estimatePcmF32MonoWavSeconds(candidateBytes.byteLength),
                  )
                : sourceDecodedForReview && candidateDecodedForReview
                  ? estimateAlignmentMetrics(
                      sourceDecodedForReview.monoSamples,
                      sourceDecodedForReview.sampleRate,
                      candidateDecodedForReview.monoSamples,
                      candidateDecodedForReview.sampleRate,
                    )
                  : buildFallbackAlignmentMetrics(sourceDecodedForReview, candidateDecodedForReview);
              const ranking = scoreCandidateWithLearnedWeights({
                baselineScore: candidateBaselineScore,
                candidateQc: candidateQcSnapshot,
                sourceQc: sourceQcSnapshot,
                alignment,
                meta: candidateMeta,
                weights: learnedReviewWeights,
              });
              const candidateQcDelta = buildReviewMetricDelta(sourceQcSnapshot, candidateQcSnapshot);
              const candidateScore: CandidateScore = {
                ...candidateBaselineScore,
                hardGatePenalty: ranking.hardGatePenalty,
                learnedAdjustment: ranking.learnedAdjustment,
                rankingScore: ranking.rankingScore,
                gateReasons: ranking.gateReasons,
              };
              candidateArtifacts.push({
                variant: candidateVariant,
                label: candidateLabel,
                bytes: candidateBytes,
                analysis: candidateAnalysis,
                meta: candidateMeta,
                baselineScore: candidateBaselineScore,
                scoredScore: candidateScore,
                qcSnapshot: candidateQcSnapshot,
                qcDelta: candidateQcDelta,
                alignment,
                ranking,
                selectionReason: null,
              });
              appendLog(
                `[CandidateQC] ${job.base}/${candidateLabel}: ${summarizeCandidateScore(candidateScore)}${
                  candidateAnalysis?.coldOpenDipDb !== null && candidateAnalysis?.coldOpenDipDb !== undefined
                    ? `, cold-open ${candidateAnalysis.coldOpenDipDb.toFixed(1)} dB`
                    : ""
                }, ${summarizeCandidateMeta(
                  candidateMeta
                )}.`
              );
              if (candidateMeta.degraded) {
                degradedCandidates += 1;
              }
              if (!selectedBytes) {
                selectedVariant = candidateVariant;
                selectedBytes = candidateBytes;
                selectedScore = candidateScore;
                selectedAnalysis = candidateAnalysis;
                selectedMeta = candidateMeta;
                selectedReason = "source-first single render; reranking temporarily disabled";
              }

              const candidateHasRenderDegrade = candidateMeta.degradeReasons.some(
                (reason) =>
                  reason !== "analysis-window-retry" &&
                  reason !== "analysis-window-drop" &&
                  reason !== "qc-unavailable",
              );
              await safeDeleteFile(ffmpeg, candidateName);
              if (
                useLongCandidateMemoryPolicy &&
                candidateQcUnavailable &&
                candidateQcMemoryFailed &&
                !candidateHasRenderDegrade
              ) {
                appendLog(
                  `[CandidateQC] ${job.base}/${candidateLabel}: stopping after QC memory failure to keep this PC cool; rendered audio will be selected by fallback if QC is unavailable.`,
                );
                break;
              }
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
            const fallbackSelection = selectQcUnavailableFallbackCandidate(
              candidateArtifacts.map((artifact, index) => ({
                variant: artifact.variant,
                index,
                hasAudio: artifact.bytes.byteLength > 0,
                meta: artifact.meta,
                score: artifact.scoredScore,
              })),
            );
            if (fallbackSelection) {
              const fallbackArtifact = candidateArtifacts[fallbackSelection.candidate.index];
              if (fallbackArtifact) {
                selectedVariant = fallbackArtifact.variant;
                selectedBytes = fallbackArtifact.bytes;
                selectedScore = fallbackArtifact.scoredScore;
                selectedAnalysis = fallbackArtifact.analysis;
                selectedMeta = fallbackArtifact.meta;
                selectedReason = "QC-unavailable fallback because rendered audio exists and all candidate QC failed";
                appendLog(
                  `[CandidateSelect] ${job.base}: kept ${fallbackArtifact.label} via QC-unavailable fallback because rendered audio exists and all candidate QC failed (${fallbackSelection.reason}).`,
                );
              }
            }
          }

          if (!selectedBytes || !selectedVariant) {
            throw new Error("No candidate mix-ready render completed.");
          }

          const selectedArtifact =
            candidateArtifacts.find((artifact) => artifact.variant === selectedVariant) ?? null;
          if (selectedArtifact) {
            selectedArtifact.selectionReason = selectedReason;
          }
          const challengerArtifact =
            [...candidateArtifacts]
              .filter((artifact) => artifact.variant !== selectedVariant)
              .filter((artifact) => !(artifact.scoredScore.gateReasons ?? []).includes("qc-unavailable"))
              .sort((left, right) => compareCandidateScores(left.scoredScore, right.scoredScore))[0] ?? null;
          const selectedSummary =
            attemptedCandidates > 0 && degradedCandidates === attemptedCandidates
              ? "all candidates degraded"
              : selectedMeta?.renderPath === "single-pass-recovered"
                ? "degraded recovered winner"
                : selectedMeta && isHealthySegmentedRender(selectedMeta)
                  ? "healthy segmented winner"
                  : selectedMeta?.renderPath === "single-pass"
                    ? `${formatCandidateVariant(selectedVariant)} single-pass winner`
                    : "best-effort winner";

          await ffmpeg.writeFile(job.mixName, cloneBytes(selectedBytes));
          appendLog(
            `[CandidateSelect] ${job.base}: kept ${formatCandidateVariant(selectedVariant)} via ${summarizeCandidateMeta(
              selectedMeta ?? {
                strategyLabel: "primary chain",
                renderPath: "single-pass",
                segmentedHealthy: false,
                degraded: false,
                degradeReasons: [],
                analysisWindowsAttempted: 0,
                analysisWindowsSucceeded: 0,
                analysisWindowsDropped: 0,
              }
            )} with ${summarizeCandidateScore(selectedScore ?? buildCandidateScore(selectedAnalysis))}${
              selectedReason && selectedReason !== "first completed candidate" && selectedReason !== "better score"
                ? `, ${selectedReason}`
                : ""
            }.`
          );
          appendLog(`[CandidateSummary] ${job.base}: ${selectedSummary}.`);
          const selectedAudibilityProtected = isAudibilityProtectedRender(selectedMeta);
          let outputProcessingFlow: OutputEntry["processingFlow"] = "app";
          if (selectedAudibilityProtected) {
            appendLog(
              `[FinalPolish] ${job.base}: skipped because audibility guard selected a source-safe recovery render.`,
            );
          } else {
            const polishResult = await runFinalAppPolishPass(ffmpeg, job.mixName, profile, {
              base: job.base,
              detail: `File ${i + 1} of ${jobs.length}, final app polish`,
              candidateVariant: selectedVariant,
              sourceSafeChain: selectedVariant === "source-safe",
            });
            ffmpeg = polishResult.ffmpeg;
            outputProcessingFlow = polishResult.applied ? "app-final-polish" : "app";
          }

          const buildArtifactForRenderedMix = async (
            renderedName: string,
            bytes: Uint8Array,
            variant: CandidateVariant,
            label: string,
            meta: CandidateRenderMeta,
            selectionReasonForArtifact: string | null,
          ): Promise<CandidateReviewArtifact> => {
            let renderedAnalysis: FileAnalysis | null = null;
            let renderedMeta = meta;
            try {
              const analysisFfmpeg = ffmpeg;
              const analysisResult = await analyzeFile(ffmpeg, renderedName, [job.inputName]);
              ffmpeg = analysisResult.ffmpeg;
              renderedAnalysis = analysisResult.analysis;
              if (ffmpeg !== analysisFfmpeg) {
                await writeJobInput(ffmpeg, job);
                await ffmpeg.writeFile(renderedName, cloneBytes(bytes));
              }
            } catch (error) {
              appendLog(`[Corrective] ${job.base}/${label}: QC fallback (${describeError(error)}).`);
              renderedMeta = {
                ...renderedMeta,
                degraded: true,
                degradeReasons: Array.from(new Set([...renderedMeta.degradeReasons, "qc-unavailable"])),
              };
              if (shouldResetFfmpegForError(error)) {
                ffmpeg = await refreshFfmpeg(`corrective QC on ${job.base}`);
                await writeJobInput(ffmpeg, job);
                await ffmpeg.writeFile(renderedName, cloneBytes(bytes));
              }
            }

            const baselineScore = buildCandidateScore(renderedAnalysis);
            const qcSnapshot = toReviewMetricSnapshot(renderedAnalysis);
            let renderedDecodedForReview: DecodedMonoAudio | null = null;
            if (!useLongCandidateMemoryPolicy) {
              try {
                renderedDecodedForReview = decodeWavToMono(bytes);
              } catch (error) {
                appendLog(`[Corrective] ${job.base}/${label}: alignment fallback (${describeError(error)}).`);
              }
            }
            const alignment = useLongCandidateMemoryPolicy
              ? buildDurationOnlyAlignmentMetrics(fileDurationForVariants, estimatePcmF32MonoWavSeconds(bytes.byteLength))
              : sourceDecodedForReview && renderedDecodedForReview
                ? estimateAlignmentMetrics(
                    sourceDecodedForReview.monoSamples,
                    sourceDecodedForReview.sampleRate,
                    renderedDecodedForReview.monoSamples,
                    renderedDecodedForReview.sampleRate,
                  )
                : buildFallbackAlignmentMetrics(sourceDecodedForReview, renderedDecodedForReview);
            const ranking = scoreCandidateWithLearnedWeights({
              baselineScore,
              candidateQc: qcSnapshot,
              sourceQc: sourceQcSnapshot,
              alignment,
              meta: renderedMeta,
              weights: learnedReviewWeights,
            });
            const scoredScore: CandidateScore = {
              ...baselineScore,
              hardGatePenalty: ranking.hardGatePenalty,
              learnedAdjustment: ranking.learnedAdjustment,
              rankingScore: ranking.rankingScore,
              gateReasons: ranking.gateReasons,
            };
            return {
              variant,
              label,
              bytes,
              analysis: renderedAnalysis,
              meta: renderedMeta,
              baselineScore,
              scoredScore,
              qcSnapshot,
              qcDelta: buildReviewMetricDelta(sourceQcSnapshot, qcSnapshot),
              alignment,
              ranking,
              selectionReason: selectionReasonForArtifact,
            };
          };

          const buildSingleWinnerManifest = (
            artifact: CandidateReviewArtifact,
            challenger: CandidateReviewArtifact | null,
            reason: string | null,
          ): ReviewBundleManifest => ({
            schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
            bundleId: `${job.base}_post_render_${String(i + 1).padStart(3, "0")}`,
            createdAt: new Date().toISOString(),
            source: {
              fileName: job.file.name,
              audioFile: "source.wav",
              durationSec: sourceDecodedForReview?.durationSec ?? speechRenderPlan?.durationSeconds ?? 0,
              sampleRate: sourceDecodedForReview?.sampleRate ?? 0,
              qc: sourceQcSnapshot,
            },
            decisionContext: {
              jobBase: job.base,
              loudnessTarget,
              selectedVariant: artifact.label,
              selectedReason: reason,
              learnedWeightsName: learnedReviewWeights.modelName,
              learnedWeightsSource: learnedReviewWeightsSource,
              reviewModelType: learnedReviewWeights.modelType,
            },
            candidates: [
              {
                role: "winner",
                audioFile: "winner.wav",
                variantLabel: artifact.label,
                renderMeta: artifact.meta,
                baselineScore: artifact.baselineScore,
                ranking: artifact.ranking,
                qc: artifact.qcSnapshot,
                sourceComparison: {
                  alignment: artifact.alignment,
                  qcDelta: artifact.qcDelta,
                },
                selectionReason: artifact.selectionReason,
              },
              ...(challenger
                ? [
                    {
                      role: "challenger" as const,
                      audioFile: "challenger.wav",
                      variantLabel: challenger.label,
                      renderMeta: challenger.meta,
                      baselineScore: challenger.baselineScore,
                      ranking: challenger.ranking,
                      qc: challenger.qcSnapshot,
                      sourceComparison: {
                        alignment: challenger.alignment,
                        qcDelta: challenger.qcDelta,
                      },
                      selectionReason: challenger.selectionReason,
                    },
                  ]
                : []),
            ],
          });

          if (MAX_CORRECTIVE_PASSES > 0 && selectedVariant && selectedAudibilityProtected) {
            appendLog(`[Corrective] ${job.base}: skipped because audibility guard selected a source-safe recovery render.`);
          }

          if (MAX_CORRECTIVE_PASSES > 0 && selectedVariant && !selectedAudibilityProtected) {
            const selectedPostPolishBytes = await readVirtualFileBytes(ffmpeg, job.mixName);
            const selectedPostPolishArtifact = await buildArtifactForRenderedMix(
              job.mixName,
              selectedPostPolishBytes,
              selectedVariant,
              formatCandidateVariant(selectedVariant),
              selectedMeta ?? {
                strategyLabel: "primary chain",
                renderPath: "single-pass",
                segmentedHealthy: false,
                degraded: false,
                degradeReasons: [],
                analysisWindowsAttempted: 0,
                analysisWindowsSucceeded: 0,
                analysisWindowsDropped: 0,
              },
              selectedReason,
            );
            const postRenderManifest = buildSingleWinnerManifest(selectedPostPolishArtifact, null, selectedReason);
            const postAutoReview = autoReviewBundle(postRenderManifest);
            const hardGateReasons = selectedPostPolishArtifact.ranking.gateReasons;
            const correctiveTriggered = shouldAttemptCorrectivePassForAssessment(
              postAutoReview.selectedAssessment,
              hardGateReasons,
            );
            const correctiveTriggerSummary = `fail ${postAutoReview.selectedAssessment.failCount}, warn ${
              postAutoReview.selectedAssessment.warnCount
            }, tags ${postAutoReview.issueTags.join(", ") || "none"}, gates ${hardGateReasons.join(", ") || "none"}`;
            const shouldTryCorrective =
              candidateQcSafeDurationSeconds < LONG_CANDIDATE_QC_SAFE_SECONDS && correctiveTriggered;

            if (correctiveTriggered && candidateQcSafeDurationSeconds >= LONG_CANDIDATE_QC_SAFE_SECONDS) {
              appendLog(
                `[Corrective] ${job.base}: skipped for long-file QC safety (${correctiveTriggerSummary}).`,
              );
            } else if (shouldTryCorrective && correctiveRenderAttempts >= correctiveRenderBudget) {
              appendLog(
                `[Corrective] ${job.base}: skipped by batch budget ${correctiveRenderAttempts}/${correctiveRenderBudget} (${correctiveTriggerSummary}).`,
              );
            } else if (shouldTryCorrective) {
              appendLog(
                `[Corrective] ${job.base}: trigger accepted (${correctiveTriggerSummary}; budget ${correctiveRenderAttempts}/${correctiveRenderBudget} used).`,
              );
              const baseDirectives = getActiveAudioReviewAdaptiveDirectives();
              let correctiveDirectives = mergeAudioReviewAdaptiveDirectives(
                baseDirectives,
                adaptiveDirectiveDeltaFromDefault(buildCorrectiveDirectivesForIssueTags(postAutoReview.issueTags)),
              );
              if (postRenderReviewRequests < POST_RENDER_REVIEW_MAX_REQUESTS) {
                postRenderReviewRequests += 1;
                const postReviewFile = buildAudioReviewFileInput(
                  job,
                  fileAnalysis,
                  profile,
                  fileDurationForVariants,
                  buildSelectedCandidateReviewInput(
                    selectedPostPolishArtifact,
                    selectedReason,
                    outputProcessingFlow,
                    postAutoReview.issueTags,
                  ),
                );
                const postReview = await runAiAudioReview([postReviewFile], {
                  source: "post-render",
                  reviewStage: "post-render",
                });
                const fileReview = postReview?.review.perFileProfiles.find(
                  (item) => item.base === job.base || item.fileName === job.file.name,
                );
                if (
                  fileReview &&
                  (postReview?.review.verdict === "adjust" || postReview?.review.verdict === "risky") &&
                  hasNonTrivialAdaptiveDirectives(normalizeAdaptiveDirectives(fileReview.adaptiveDirectives), baseDirectives)
                ) {
                  correctiveDirectives = normalizeAdaptiveDirectives(fileReview.adaptiveDirectives);
                }
              } else {
                appendLog(`[Corrective] ${job.base}: Gemini post-render review skipped (batch cap reached).`);
              }

              if (hasNonTrivialAdaptiveDirectives(correctiveDirectives, baseDirectives)) {
                correctiveRenderAttempts += 1;
                appendLog(
                  `[Corrective] ${job.base}: using render budget ${correctiveRenderAttempts}/${correctiveRenderBudget}.`,
                );
                const previousDirectives: AudioReviewAdaptiveDirectives | null = aiAdaptiveDirectivesOverrideRef.current;
                const correctiveName = `${job.base}_corrective_candidate.wav`;
                try {
                  if (candidateQcSafeDurationSeconds >= LONG_FILE_DURATION_SECONDS) {
                    ffmpeg = await refreshFfmpeg(`corrective render preflight on ${job.base}`);
                    await writeJobInput(ffmpeg, job);
                  }
                  aiAdaptiveDirectivesOverrideRef.current = correctiveDirectives;
                  const correctiveProfile = buildAdaptiveProfile(fileAnalysis, batchReference);
                  const correctivePlannerContext = await preparePlannerRenderContext(
                    ffmpeg,
                    job,
                    correctiveProfile,
                    fileAnalysis,
                    fileDurationForVariants,
                  );
                  const correctiveResult = await renderMixReadyWithFallbacks(
                    ffmpeg,
                    job,
                    correctiveName,
                    correctiveProfile,
                    i,
                    jobs.length,
                    "Corrective render",
                    {
                      candidateVariant: selectedVariant,
                      skipSpeechSegmentation:
                        selectedVariant === "source-safe" ||
                        (selectedVariant === "continuity-safe" && (correctiveProfile?.preferSinglePassContinuity ?? false)),
                      sourceSafeChain: selectedVariant === "source-safe",
                      disableRoomCleanup: selectedVariant === "source-safe" ? true : undefined,
                      disableAdaptiveNoiseReduction: selectedVariant === "source-safe" ? true : undefined,
                    },
                    speechRenderPlan,
                    fileAnalysis,
                    correctivePlannerContext,
                  );
                  ffmpeg = correctiveResult.ffmpeg;
                  if (isAudibilityProtectedRender(correctiveResult.meta)) {
                    appendLog(
                      `[FinalPolish] ${job.base}: skipped corrective final app polish because audibility guard selected a source-safe recovery render.`,
                    );
                  } else {
                    const correctivePolish = await runFinalAppPolishPass(ffmpeg, correctiveName, correctiveProfile, {
                      base: job.base,
                      detail: `File ${i + 1} of ${jobs.length}, corrective final app polish`,
                      candidateVariant: selectedVariant,
                      sourceSafeChain: selectedVariant === "source-safe",
                    });
                    ffmpeg = correctivePolish.ffmpeg;
                  }
                  const correctiveBytes = await readVirtualFileBytes(ffmpeg, correctiveName);
                  const correctiveArtifact = await buildArtifactForRenderedMix(
                    correctiveName,
                    correctiveBytes,
                    selectedVariant,
                    `${formatCandidateVariant(selectedVariant)} corrective`,
                    correctiveResult.meta,
                    "bounded corrective pass",
                  );
                  const originalRank = selectedPostPolishArtifact.scoredScore.rankingScore ?? selectedPostPolishArtifact.scoredScore.total;
                  const correctiveRank = correctiveArtifact.scoredScore.rankingScore ?? correctiveArtifact.scoredScore.total;
                  const correctiveWins =
                    correctiveArtifact.ranking.gateReasons.length === 0 &&
                    correctiveRank <= originalRank - CORRECTIVE_WIN_MARGIN;
                  if (correctiveWins) {
                    await ffmpeg.writeFile(job.mixName, cloneBytes(correctiveBytes));
                    selectedBytes = correctiveBytes;
                    selectedAnalysis = correctiveArtifact.analysis;
                    selectedScore = correctiveArtifact.scoredScore;
                    selectedMeta = correctiveArtifact.meta;
                    selectedReason = `corrective pass kept (delta ${(originalRank - correctiveRank).toFixed(1)} ranking points)`;
                    if (canEmitPerFileReviewBundle) {
                      nextReviewBundles.push({
                        bundleId: `${job.base}_corrective_${String(i + 1).padStart(3, "0")}`,
                        manifest: buildSingleWinnerManifest(correctiveArtifact, selectedPostPolishArtifact, selectedReason),
                        assets: [
                          { path: "source.wav", blob: job.file },
                          { path: "winner.wav", blob: reviewBlobFromBytes(correctiveArtifact.bytes) },
                          { path: "challenger.wav", blob: reviewBlobFromBytes(selectedPostPolishArtifact.bytes) },
                        ],
                      });
                    } else {
                      appendLog(
                        `[ReviewBundle] ${job.base}: skipped corrective QC Lab bundle because ${reviewBundleFinalOutputBlocker}; stale winner.wav artifacts are not emitted.`,
                      );
                    }
                    appendLog(
                      `[Corrective] ${job.base}: kept corrective render triggered by ${postAutoReview.issueTags.join(
                        ", ",
                      ) || "auto-review"} (delta ${(originalRank - correctiveRank).toFixed(1)} ranking points).`,
                    );
                  } else {
                    await ffmpeg.writeFile(job.mixName, cloneBytes(selectedPostPolishBytes));
                    appendLog(
                      `[Corrective] ${job.base}: discarded corrective render triggered by ${postAutoReview.issueTags.join(
                        ", ",
                      ) || "auto-review"} (delta ${(originalRank - correctiveRank).toFixed(1)} ranking points; gates ${
                        correctiveArtifact.ranking.gateReasons.join(", ") || "none"
                      }).`,
                    );
                  }
                } catch (error) {
                  appendLog(`[Corrective] ${job.base}: skipped after failure (${describeError(error)}).`);
                  if (shouldResetFfmpegForError(error)) {
                    ffmpeg = await refreshFfmpeg(`corrective failure on ${job.base}`);
                    await writeJobInput(ffmpeg, job);
                  }
                  await ffmpeg.writeFile(job.mixName, cloneBytes(selectedPostPolishBytes));
                } finally {
                  aiAdaptiveDirectivesOverrideRef.current = previousDirectives;
                  await safeDeleteFile(ffmpeg, correctiveName);
                }
              } else {
                appendLog(`[Corrective] ${job.base}: no non-trivial corrective directives for ${postAutoReview.issueTags.join(", ") || "auto-review"}.`);
              }
            } else {
              appendLog(`[Corrective] ${job.base}: no corrective pass needed (${postAutoReview.selectedAssessment.summary}).`);
            }
          }

          await applyFinalConsonantPeakPolish(ffmpeg, job.mixName, job.base);

          const skipReviewBundleForMemory = candidateQcSafeDurationSeconds >= LONG_FILE_DURATION_SECONDS;
          if (skipReviewBundleForMemory) {
            appendLog(
              `[ReviewBundle] ${job.base}: skipped QC Lab review bundle for ${candidateQcSafeDurationSeconds.toFixed(
                0,
              )}s long file to avoid duplicating large audio blobs in browser memory.`,
            );
          } else if (reviewBundleFinalOutputBlocker) {
            appendLog(
              `[ReviewBundle] ${job.base}: skipped QC Lab review bundle because ${reviewBundleFinalOutputBlocker}; stale winner.wav artifacts are not emitted.`,
            );
          } else {
            try {
              const finalBytes = await readVirtualFileBytes(ffmpeg, job.mixName);
              const finalArtifact = await buildArtifactForRenderedMix(
                job.mixName,
                finalBytes,
                selectedVariant,
                `${formatCandidateVariant(selectedVariant)} final`,
                selectedMeta ?? {
                  strategyLabel: "primary chain",
                  renderPath: "single-pass",
                  segmentedHealthy: false,
                  degraded: false,
                  degradeReasons: [],
                  analysisWindowsAttempted: 0,
                  analysisWindowsSucceeded: 0,
                  analysisWindowsDropped: 0,
                },
                selectedReason,
              );
              const bundleId = `${job.base}_review_${String(i + 1).padStart(3, "0")}`;
              const sourceDurationSec =
                sourceDecodedForReview?.durationSec ?? speechRenderPlan?.durationSeconds ?? 0;
              const sourceSampleRate = sourceDecodedForReview?.sampleRate ?? 0;
              const selectedVariantLabel = formatCandidateVariant(selectedVariant);
              const selectedReasonText = selectedReason ?? "first completed candidate";
              const reviewCandidates: Array<{
                role: "winner" | "challenger";
                assetName: string;
                artifact: CandidateReviewArtifact;
              }> = [{ role: "winner", assetName: "winner.wav", artifact: finalArtifact }];
              if (challengerArtifact) {
                reviewCandidates.push({
                  role: "challenger",
                  assetName: "challenger.wav",
                  artifact: challengerArtifact,
                });
              }
              const manifest: ReviewBundleManifest = {
                schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
                bundleId,
                createdAt: new Date().toISOString(),
                source: {
                  fileName: job.file.name,
                  audioFile: "source.wav",
                  durationSec: sourceDurationSec,
                  sampleRate: sourceSampleRate,
                  qc: sourceQcSnapshot,
                },
                decisionContext: {
                  jobBase: job.base,
                  loudnessTarget,
                  selectedVariant: selectedVariantLabel,
                  selectedReason: selectedReasonText,
                  learnedWeightsName: learnedReviewWeights.modelName,
                  learnedWeightsSource: learnedReviewWeightsSource,
                  reviewModelType: learnedReviewWeights.modelType,
                },
                candidates: reviewCandidates.map(({ role, assetName, artifact }) => ({
                  role,
                  audioFile: assetName,
                  variantLabel: artifact.label,
                  renderMeta: artifact.meta,
                  baselineScore: artifact.baselineScore,
                  ranking: artifact.ranking,
                  qc: artifact.qcSnapshot,
                  sourceComparison: {
                    alignment: artifact.alignment,
                    qcDelta: artifact.qcDelta,
                  },
                  selectionReason: artifact.selectionReason,
                })),
              };
              nextReviewBundles.push({
                bundleId,
                manifest,
                assets: [
                  { path: "source.wav", blob: job.file },
                  { path: "winner.wav", blob: reviewBlobFromBytes(finalArtifact.bytes) },
                  ...(challengerArtifact
                    ? [
                        {
                          path: "challenger.wav",
                          blob: reviewBlobFromBytes(challengerArtifact.bytes),
                        },
                      ]
                    : []),
                ],
              });
              appendLog(
                `[ReviewBundle] ${job.base}: captured final ${selectedVariantLabel} winner${
                  challengerArtifact ? ` vs ${challengerArtifact.label}` : ""
                } for QC Lab review.`,
              );
            } catch (error) {
              appendLog(
                `[ReviewBundle] ${job.base}: skipped after final bundle build failed (${describeError(
                  error,
                )}); audio outputs continue.`,
              );
            }
          }

          if (keepMixReady || loudnessConfig === null) {
            const mixOutput = await writeOutput(ffmpeg, job.mixName, "mixready", "clean", outputProcessingFlow, {
              sourceBase: job.base,
              sourceName: job.file.name,
            });
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
                await applyFinalConsonantPeakPolish(ffmpeg, job.blendMixName, `${job.base} blend`);
                blendRendered = true;
                if (keepMixReady || loudnessConfig === null) {
                  const blendMixOutput = await writeOutput(
                    ffmpeg,
                    job.blendMixName,
                    "mixready",
                    "blend",
                  outputProcessingFlow,
                  {
                    sourceBase: job.base,
                    sourceName: job.file.name,
                  },
                );
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
            const loudOutput = await writeOutput(ffmpeg, cleanLoudName, "loudness", "clean", outputProcessingFlow, {
              sourceBase: job.base,
              sourceName: job.file.name,
            });
            outputEntries.push(loudOutput);

            if (sceneBlend && blendRendered) {
              blendLoudName = `${job.base}_blend_${loudnessConfig.suffix}.wav`;
              setStatus(`Loudness blend: ${job.base} (${i + 1}/${jobs.length})`);
              setActiveQueueStage(job.base, "Loudness (blend)", `File ${i + 1} of ${jobs.length}`);
              await runLoudnorm(ffmpeg, job.blendMixName, blendLoudName, loudnessConfig);
              const blendLoudOutput = await writeOutput(
                ffmpeg,
                blendLoudName,
                "loudness",
                "blend",
                outputProcessingFlow,
                {
                  sourceBase: job.base,
                  sourceName: job.file.name,
                },
              );
              outputEntries.push(blendLoudOutput);
            }
          }

          markQueueDone(job.base, "Outputs ready");
          // Live update: push the in-progress outputs to React state so
          // users can see and download completed files even if a later
          // file fails or the browser crashes mid-batch.
          setOutputs([...outputEntries]);
          }
        } catch (error) {
          const reason = summarizeFailureReason(error);
          appendLog(`Error (${job.base}): ${reason}`);

          // Recoverable failure on a file that hasn't exhausted retries:
          // refresh the worker, clean up its virtual FS for this file,
          // and re-process the SAME job (don't increment `i`).
          if (retryAttempt < PER_FILE_MAX_RETRIES && (isRecoverableFailure(error) || watchdogFired)) {
            retryCounts.set(job.base, retryAttempt + 1);
            try {
              await safeDeleteFile(ffmpeg, job.inputName);
              await safeDeleteFile(ffmpeg, job.mixName);
              await safeDeleteFile(ffmpeg, job.blendMixName);
              if (cleanLoudName) await safeDeleteFile(ffmpeg, cleanLoudName);
              if (blendLoudName) await safeDeleteFile(ffmpeg, blendLoudName);
            } catch {
              // best-effort cleanup before refresh; safe to ignore
            }
            ffmpeg = await refreshFfmpeg(`retry-induced refresh on ${job.base}`);
            workerCumulativeAudioSec = 0;
            clearTimeout(watchdog);
            // Skip the recycle check below — we just refreshed.
            continue;
          }

          // Permanent failure (non-recoverable, or out of retries).
          hadErrors = true;
          failedRuns.push({
            base: job.base,
            fileName: job.file.name,
            reason: retryAttempt > 0 ? `${reason} (after ${retryAttempt} retry)` : reason,
          });
          markQueueError(job.base, reason);
          // Live update: also push current outputs after a permanent
          // failure so users can see what made it through up to this point.
          setOutputs([...outputEntries]);
          if (shouldResetFfmpegForError(error)) {
            ffmpeg = await refreshFfmpeg(`processing failure on ${job.base}`);
            workerCumulativeAudioSec = 0;
          }
        } finally {
          clearTimeout(watchdog);
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

        // File is fully done (success or permanent failure). Track its
        // audio for the duration-aware memory guard, advance the cursor.
        workerCumulativeAudioSec += estDurationSec;
        i += 1;

        // Recycle decisions, in priority order:
        //   1. duration-aware: > 60 min of audio on this worker → refresh
        //   2. file-count-aware (existing): every Nth file in long batches
        if (workerCumulativeAudioSec >= BATCH_AUDIO_RECYCLE_SECONDS) {
          ffmpeg = await refreshFfmpeg(
            `audio-volume guard (${(workerCumulativeAudioSec / 60).toFixed(0)} min processed since refresh)`,
          );
          workerCumulativeAudioSec = 0;
        } else if (shouldRecycleFfmpegForBatch(i, jobs.length)) {
          ffmpeg = await refreshFfmpeg(`processing memory guard (${i}/${jobs.length})`);
          workerCumulativeAudioSec = 0;
        }
      }

      if (loudnessConfig === null) {
        ffmpeg = await alignBatchMixReadyOutputs(ffmpeg, outputEntries);
      }

      // Final reconciliation. Live updates inside the loop already pushed
      // outputs as files completed, so this is mostly a no-op now —
      // kept so the array reference matches `outputEntries.length` even
      // if no files succeeded.
      setOutputs([...outputEntries]);
      setReviewBundles(nextReviewBundles);
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
      processingControlsOverrideRef.current = null;
      aiAdaptiveDirectivesOverrideRef.current = null;
      sourceFirstCandidateVariantRef.current = null;
      sourceFirstPlansByBaseRef.current = new Map();
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

  const downloadOutputFile = async (output: OutputEntry) => {
    try {
      if (!(output.blob instanceof Blob)) {
        throw new Error("Output blob missing. Refresh the page and re-run the batch.");
      }
      setDownloadStatus(`Starting ${output.name}`);
      const started = await queueBrowserDownload(output.blob, output.name);
      appendLog(
        `[Download] ${started.fileName}: started (${formatBytes(
          started.size,
        )}); browser link retained for ${Math.round(started.retainMs / 60000)} min.`,
      );
    } catch (error) {
      appendLog(`Download failed (${output.name}): ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setDownloadStatus(null);
    }
  };

  const buildDeliveryManifest = (
    generatedAt: string,
    manifestOutputs = outputs,
    exportPart?: {
      estimatedBytes: number;
      partNumber: number;
      totalParts: number;
    },
  ) => ({
    app: "Shorts Projektt Internal VO Optimizer",
    generatedAt,
    totalFiles: outputs.length,
    exportedFiles: manifestOutputs.length,
    totalOutputBytes,
    estimatedZipBytes,
    exportPart: exportPart ?? null,
    settings: {
      loudnessTarget,
      leveler,
      smartMatchMode,
      eqCleanup,
      softenHarshness,
      cinematicColor,
      breathControl,
      roomCleanup,
      noiseGuard,
      floorGuard,
      sceneBlend,
      keepMixReady,
      gainPlannerEnabled,
      neuralSpeechEnhancement: false,
      reviewReranker: "paused-temporary-single-render",
    },
    files: manifestOutputs.map((output) => ({
      name: output.name,
      kind: output.kind,
      variant: output.variant,
      processingFlow: output.processingFlow,
      sizeBytes: output.size,
    })),
  });

  const downloadOutputsSequentially = async () => {
    if (outputs.length === 0 || downloadQueueBusy) return;

    setDownloadQueueBusy(true);
    setDownloadStatus(`Queueing ${outputs.length} file downloads`);
    appendLog(
      `[Download] Safe queue started for ${outputs.length} file(s), ${formatBytes(
        totalOutputBytes,
      )} total. Keep this tab open until Chrome finishes saving.`,
    );

    try {
      for (let index = 0; index < outputs.length; index += 1) {
        const output = outputs[index];
        if (!(output.blob instanceof Blob)) {
          throw new Error("Output blob missing. Refresh the page and re-run the batch.");
        }
        setDownloadStatus(`Starting ${index + 1}/${outputs.length}: ${output.name}`);
        const started = await queueBrowserDownload(output.blob, output.name);
        appendLog(
          `[Download] ${index + 1}/${outputs.length} ${started.fileName}: started (${formatBytes(
            started.size,
          )}); retained ${Math.round(started.retainMs / 60000)} min.`,
        );
      }
      appendLog(
        `[Download] Safe queue finished starting ${outputs.length} file(s). If Chrome asks to allow multiple downloads, choose Allow.`,
      );
    } catch (error) {
      appendLog(`Safe download queue failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setDownloadQueueBusy(false);
      setDownloadStatus(null);
    }
  };

  const downloadOutputsZip = async () => {
    if (outputs.length === 0 || zipBusy) return;

    setZipBusy(true);
    setZipProgress(0);

    try {
      const generatedAt = new Date().toISOString();
      const stamp = generatedAt.replace(/[:.]/g, "-");
      const parts = planVoZipExportParts(outputs);
      const chunked = parts.length > 1;

      if (chunked) {
        appendLog(
          `[ZIP Policy] Large batch detected (${outputs.length} file(s), ${formatBytes(
            totalOutputBytes,
          )} output). Exporting ${parts.length} smaller ZIP parts instead of one fragile ${formatBytes(
            estimatedZipBytes,
          )} archive.`,
        );
      }

      for (let partIndex = 0; partIndex < parts.length; partIndex += 1) {
        const part = parts[partIndex];
        const zip = new JSZip();

        for (let i = 0; i < part.outputs.length; i += 1) {
          const output = part.outputs[i];
          if (!(output.blob instanceof Blob)) {
            throw new Error("Output blob missing. Refresh the page and re-run the batch.");
          }
          zip.file(output.name, output.blob);
          setZipProgress(
            Math.min(
              95,
              Math.round(((partIndex + (i + 1) / part.outputs.length) / parts.length) * 70),
            ),
          );
        }

        zip.file(
          "delivery_manifest.json",
          JSON.stringify(
            buildDeliveryManifest(generatedAt, part.outputs, {
              estimatedBytes: part.estimatedBytes,
              partNumber: part.partNumber,
              totalParts: part.totalParts,
            }),
            null,
            2,
          ),
        );

        const zipBlob = await zip.generateAsync(
          chunked
            ? {
                type: "blob",
                compression: "STORE",
              }
            : {
                type: "blob",
                compression: "DEFLATE",
                compressionOptions: { level: 6 },
              },
          ({ percent }) => {
            const partProgress = (partIndex + percent / 100) / parts.length;
            setZipProgress(70 + Math.round(partProgress * 30));
          },
        );

        const archiveName =
          parts.length === 1
            ? `vo_leveler_outputs_${stamp}.zip`
            : `vo_leveler_outputs_${stamp}_part-${part.partNumber}-of-${part.totalParts}.zip`;
        const started = await queueBrowserDownload(zipBlob, archiveName);
        appendLog(
          `ZIP created: ${archiveName} (${formatBytes(zipBlob.size)}; retained ${Math.round(
            started.retainMs / 60000,
          )} min)`,
        );
      }
    } catch (error) {
      appendLog(`ZIP export failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setZipBusy(false);
      setZipProgress(0);
    }
  };

  const downloadReviewBundlesZip = async () => {
    if (reviewBundles.length === 0 || reviewZipBusy) return;

    setReviewZipBusy(true);
    setReviewZipProgress(0);

    try {
      const zip = new JSZip();

      for (let index = 0; index < reviewBundles.length; index += 1) {
        const bundle = reviewBundles[index];
        const folder = zip.folder(bundle.bundleId) ?? zip;
        folder.file("manifest.json", JSON.stringify(bundle.manifest, null, 2));
        for (const asset of bundle.assets) {
          folder.file(asset.path, asset.blob);
        }
        setReviewZipProgress(Math.round(((index + 1) / reviewBundles.length) * 70));
      }

      const zipBlob = await zip.generateAsync(
        {
          type: "blob",
          compression: "DEFLATE",
          compressionOptions: { level: 6 },
        },
        ({ percent }) => {
          setReviewZipProgress(70 + Math.round((percent / 100) * 30));
        },
      );

      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const archiveName = `vo_leveler_review_bundles_${stamp}.zip`;
      triggerBrowserDownload(zipBlob, archiveName);
      appendLog(`Review bundle ZIP created: ${archiveName} (${formatBytes(zipBlob.size)})`);
    } catch (error) {
      appendLog(
        `Review bundle export failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      setReviewZipBusy(false);
      setReviewZipProgress(0);
    }
  };

  const activeQueueItem = useMemo(
    () => queueItems.find((item) => item.status === "working") ?? null,
    [queueItems]
  );
  const queueCounts = useMemo(
    () =>
      queueItems.reduce(
        (counts, item) => ({
          ...counts,
          [item.status]: counts[item.status] + 1,
        }),
        {
          total: queueItems.length,
          done: 0,
          error: 0,
          working: 0,
          pending: 0,
        },
      ),
    [queueItems]
  );
  const totalOutputBytes = useMemo(
    () => outputs.reduce((total, output) => total + output.size, 0),
    [outputs],
  );
  const estimatedZipBytes = useMemo(() => estimateVoZipBytes(outputs), [outputs]);
  const zipExportParts = useMemo(() => planVoZipExportParts(outputs), [outputs]);

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
              <strong>Speech-aware leveler</strong>
              <div className={styles.label}>
                Plans a gain curve from the actual sentences before any compressor runs. This is the core
                fix for sudden volume spikes. Classifies breaths and short onomatopoeia runs separately so
                they sit with the performance instead of spiking above it. Handles batch-episode files up
                to 80 minutes (longer files fall back to the legacy leveler).
              </div>
            </div>
            <input
              type="checkbox"
              checked={gainPlannerEnabled}
              onChange={(event) => setGainPlannerEnabled(event.target.checked)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>Cinematic color</strong>
              <div className={styles.label}>
                Subtle dub-room voicing: +0.8 dB @180 Hz warmth, +0.6 dB @4.5 kHz intelligibility, -0.5 dB
                @10 kHz. Automatically bypassed on emotional takes.
              </div>
            </div>
            <input
              type="checkbox"
              checked={cinematicColor}
              onChange={(event) => setCinematicColor(event.target.checked)}
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
          <div className={styles.toggleRow}>
            <div>
              <strong>Neural speech enhancement</strong>
              <div className={styles.label}>
                Temporarily disabled. The active chain is AI review, app render, then one subtle final app polish.
              </div>
            </div>
            <input
              type="checkbox"
              checked={neuralSpeechEnhancementEnabled}
              disabled
              onChange={() => setNeuralSpeechEnhancementEnabled(false)}
            />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <strong>AI Auto Pilot</strong>
              <div className={styles.label}>
                Reviews original source audio first, applies a safe adaptive profile, then renders once through the app and final app touch.
              </div>
              {aiAutoPilotStatus && <div className={styles.label}>{aiAutoPilotStatus}</div>}
            </div>
            <input
              type="checkbox"
              checked={aiAutoPilotEnabled}
              onChange={(event) => setAiAutoPilotEnabled(event.target.checked)}
            />
          </div>

          <button
            type="button"
            className={`${styles.button} ${styles.buttonGhost} ${styles.sectionTop}`}
            onClick={() => setAdvancedOpen((open) => !open)}
            aria-expanded={advancedOpen}
          >
            {advancedOpen ? "Hide advanced options" : "Show advanced options"}
          </button>
          {advancedOpen && (
            <>
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
                    Legacy: presence + air softening for bright/emotional lines. Cinematic color covers
                    most of this; enable both if you want extra softening.
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
                    Legacy: adds very light mono early reflections so VO sits in-picture. Off by default
                    now that cinematic color handles room-sit cues.
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
                  <strong>Keep silences clean (expander + NR)</strong>
                  <div className={styles.label}>
                    Enables the speech-aware expander on the leveler path, and chains measured-SNR
                    spectral NR (afftdn + anlmdn when severe). Previously labeled &quot;Noise guard&quot;.
                  </div>
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
                    Legacy downward curve on the tail. Superseded by the speech-aware expander when the
                    new leveler is on, but still useful as a safety net on very long or very noisy takes.
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={floorGuard}
                  onChange={(event) => setFloorGuard(event.target.checked)}
                />
              </div>
              <div className={styles.reviewModelPanel}>
                <div className={styles.reviewModelHeader}>
                  <div>
                    <strong>Review reranker</strong>
                    <div className={styles.label}>
                      Temporarily paused. The current pipeline renders the AI-selected source-first variant once.
                    </div>
                  </div>
                  <span className={styles.reviewModelBadge}>
                    {learnedReviewWeightsSource === "local-import" ? "Local import" : "Built-in default"}
                  </span>
                </div>
                <div className={styles.reviewModelMeta}>
                  <span>{learnedReviewWeights.modelName}</span>
                  <span>{learnedReviewWeights.modelType}</span>
                </div>
                <div className={`${styles.controls} ${styles.sectionTop}`}>
                  <label className={`${styles.button} ${styles.buttonSecondary}`}>
                    Import weights
                    <input
                      type="file"
                      accept=".json,application/json"
                      hidden
                      disabled
                      onChange={async (event) => {
                        const file = event.target.files?.[0] ?? null;
                        await importReviewWeights(file);
                        event.currentTarget.value = "";
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.buttonGhost}`}
                    onClick={resetReviewWeights}
                    disabled
                  >
                    Reset to default
                  </button>
                </div>
              </div>
            </>
          )}

          <div className={`${styles.controls} ${styles.sectionTop}`}>
            <button className={styles.button} onClick={() => void processFiles()} disabled={loading || files.length === 0}>
              {loading ? "Processing..." : "Run Batch"}
            </button>
            <button
              className={`${styles.button} ${styles.buttonGhost}`}
              onClick={() => {
                setFiles([]);
                setOutputs([]);
                setReviewBundles([]);
                setZipProgress(0);
                setReviewZipProgress(0);
                setLogs([]);
                setFailedOptimizations([]);
                setShowFailureWarning(false);
                setAiReviewFiles([]);
                setAiReviewResult(null);
                setAiReviewError(null);
                setAiReviewModel(null);
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
            <button
              type="button"
              ref={aiReviewTriggerRef}
              className={`${styles.button} ${styles.buttonSecondary}`}
              onClick={() => setAiReviewOpen(true)}
            >
              AI Review
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
          {(outputs.length > 0 || reviewBundles.length > 0) && (
            <div className={`${styles.controls} ${styles.sectionTop}`}>
              {outputs.length > 0 && (
                <button
                  className={`${styles.button} ${styles.buttonSecondary}`}
                  onClick={downloadOutputsZip}
                  disabled={zipBusy || downloadQueueBusy}
                >
                  {zipBusy
                    ? `Building ZIP ${zipProgress}%`
                    : zipExportParts.length > 1
                      ? `Download ZIP Parts (${zipExportParts.length})`
                      : `Download ZIP (${outputs.length})`}
                </button>
              )}
              {outputs.length > 1 && (
                <button
                  className={`${styles.button} ${styles.buttonGhost}`}
                  onClick={downloadOutputsSequentially}
                  disabled={zipBusy || downloadQueueBusy}
                  type="button"
                >
                  {downloadQueueBusy ? "Starting Downloads..." : `Download Files Safely (${outputs.length})`}
                </button>
              )}
              {reviewBundles.length > 0 && (
                <button
                  className={`${styles.button} ${styles.buttonGhost}`}
                  onClick={downloadReviewBundlesZip}
                  disabled={reviewZipBusy}
                >
                  {reviewZipBusy
                    ? `Building Review ZIP ${reviewZipProgress}%`
                    : `Export Review Bundles (${reviewBundles.length})`}
                </button>
              )}
              {outputs.length > 0 && (
                <div className={styles.downloadStatus}>
                  {downloadStatus ??
                    `${formatBytes(totalOutputBytes)} ready${
                      zipExportParts.length > 1
                        ? `; ZIP will export as ${zipExportParts.length} safer parts`
                        : ""
                    }`}
                </div>
              )}
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
                    {output.processingFlow === "app-neural-speech-enhancement-app" && (
                      <span className={styles.outputBadge}>Neural enhancement + app</span>
                    )}
                    {output.processingFlow === "app-final-polish" && (
                      <span className={styles.outputBadge}>Final app polish</span>
                    )}
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
                <button
                  type="button"
                  className={styles.outputDownload}
                  onClick={() => downloadOutputFile(output)}
                  disabled={zipBusy || downloadQueueBusy}
                >
                  Download
                </button>
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
      {aiReviewOpen && (
        <div className={styles.aiReviewOverlay} role="dialog" aria-modal="true" aria-labelledby="ai-review-title">
          <div className={styles.aiReviewModal}>
            <div className={styles.aiReviewHeader}>
              <div>
                <h3 id="ai-review-title">AI Audio Review</h3>
                <div className={styles.label}>
                  {aiReviewFiles.length} file{aiReviewFiles.length === 1 ? "" : "s"} ready
                  {aiReviewModel ? ` - ${aiReviewModel}` : ""}
                </div>
              </div>
              <button
                ref={aiReviewCloseRef}
                type="button"
                className={`${styles.button} ${styles.buttonGhost}`}
                onClick={closeAiReview}
              >
                Close
              </button>
            </div>

            <div className={styles.aiReviewActionBar}>
              <span className={styles.aiReviewTiny}>
                {aiReviewFiles
                  .slice(-3)
                  .map((file) => file.fileName)
                  .join(", ")}
              </span>
              {aiReviewBusy && <span className={styles.aiReviewTiny}>Reviewing source...</span>}
            </div>
            {aiAutoPilotStatus && <div className={styles.aiReviewTiny}>{aiAutoPilotStatus}</div>}

            {aiReviewError && (
              <div className={styles.aiReviewError} aria-live="polite">
                {aiReviewError}
              </div>
            )}

            {aiReviewResult ? (
              <div className={styles.aiReviewBody} aria-live="polite">
                <div className={styles.aiReviewBadges}>
                  <span className={styles.outputBadge}>{aiReviewResult.verdict.toUpperCase()}</span>
                  <span className={styles.outputBadge}>{Math.round(aiReviewResult.confidence * 100)}% confidence</span>
                  <span className={styles.outputBadge}>{aiReviewResult.perFileProfiles.length} per-audio profiles</span>
                </div>

                {aiReviewResult.perFileProfiles.map((fileReview) => (
                  <div className={styles.aiReviewRecommendation} key={`${fileReview.base}-${fileReview.fileName}`}>
                    <div className={styles.aiReviewRecommendationTop}>
                      <strong>{fileReview.fileName}</strong>
                      <span>{fileReview.recommendedProfile.selectedVariant}</span>
                    </div>
                    <div className={styles.aiReviewRecommendationTop}>
                      <span>{fileReview.recommendedProfile.name}</span>
                      <span>{Math.round(fileReview.confidence * 100)}% file confidence</span>
                    </div>
                    <div className={styles.aiReviewProfileGrid}>
                      <span>Match: {fileReview.recommendedProfile.smartMatchMode}</span>
                      <span>Leveler: {fileReview.recommendedProfile.leveler}</span>
                      <span>Breath: {fileReview.recommendedProfile.breathControl}</span>
                      <span>Warmth: {formatSigned(fileReview.adaptiveDirectives.warmthDb, 1)} dB</span>
                      <span>Presence: {formatSigned(fileReview.adaptiveDirectives.presenceDb, 1)} dB</span>
                      <span>Air: {formatSigned(fileReview.adaptiveDirectives.airDb, 1)} dB</span>
                      <span>Sag: {formatSigned(fileReview.adaptiveDirectives.sagRecoveryBoost, 2)}</span>
                      <span>De-harsh: {formatSigned(fileReview.adaptiveDirectives.deHarshDb, 1)} dB</span>
                      <span>Polish: single pass {Math.round(fileReview.adaptiveDirectives.finalPolishIntensity * 100)}%</span>
                      <span>Room: {fileReview.recommendedProfile.roomCleanup ? "on" : "off"}</span>
                      <span>Harshness: {fileReview.recommendedProfile.softenHarshness ? "soften" : "normal"}</span>
                      <span>Color: {fileReview.recommendedProfile.cinematicColor ? "on" : "off"}</span>
                    </div>
                    <p>{fileReview.summary}</p>
                    {fileReview.recommendedProfile.notes && <p>{fileReview.recommendedProfile.notes}</p>}
                  </div>
                ))}

                {aiReviewResult.profileRationale.length > 0 && (
                  <div className={styles.aiReviewSection}>
                    <strong>Profile Rationale</strong>
                    <ul>
                      {aiReviewResult.profileRationale.map((item, index) => (
                        <li key={`${item}-${index}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {aiReviewResult.adjustments.length > 0 && (
                  <div className={styles.aiReviewSection}>
                    <strong>Adjustments</strong>
                    <div className={styles.aiReviewList}>
                      {aiReviewResult.adjustments.map((adjustment, index) => (
                        <div className={styles.aiReviewListItem} key={`${adjustment.control}-${index}`}>
                          <div className={styles.aiReviewListTop}>
                            <span>{adjustment.control}</span>
                            <span>{adjustment.risk}</span>
                          </div>
                          <div>{adjustment.recommendation}</div>
                          <small>{adjustment.why}</small>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {aiReviewResult.findings.length > 0 && (
                  <div className={styles.aiReviewSection}>
                    <strong>Findings</strong>
                    <div className={styles.aiReviewList}>
                      {aiReviewResult.findings.map((finding, index) => (
                        <div className={styles.aiReviewListItem} key={`${finding.fileName}-${finding.issue}-${index}`}>
                          <div className={styles.aiReviewListTop}>
                            <span>{finding.fileName}</span>
                            <span>{finding.severity}</span>
                          </div>
                          <div>{finding.issue}</div>
                          <small>{finding.evidence}</small>
                          <small>{finding.action}</small>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className={styles.aiReviewColumns}>
                  <div className={styles.aiReviewSection}>
                    <strong>Guardrails</strong>
                    <ul>
                      {aiReviewResult.guardrails.map((item, index) => (
                        <li key={`${item}-${index}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className={styles.aiReviewSection}>
                    <strong>Listening Checks</strong>
                    <ul>
                      {aiReviewResult.nextListeningChecks.map((item, index) => (
                        <li key={`${item}-${index}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ) : (
              <div className={styles.aiReviewEmpty}>
                {aiReviewFiles.length === 0
                  ? "Run a WAV batch first. The reviewer uses original source metrics and the pre-render adaptive profile."
                  : "No AI review yet."}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
