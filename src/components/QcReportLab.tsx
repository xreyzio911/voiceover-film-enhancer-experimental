"use client";

import JSZip from "jszip";
import { useEffect, useMemo, useState, type DragEvent } from "react";
import {
  AUDIO_QC_FRAME_MS,
  analyzeFloatSamples,
  analyzeFrameAudio,
  buildFlagsAndRecommendations,
  toDb,
  type AudioQcMetrics,
} from "../lib/audioQc";
import {
  REVIEW_WEIGHT_STORAGE_KEY,
  REVIEW_BUNDLE_SCHEMA_VERSION,
  REVIEW_ISSUE_TAGS,
  autoReviewBundle,
  fitLearnedReviewWeights,
  serializeReviewDecisionJsonl,
  type AutoReviewResult,
  type ReviewBundleManifest,
  type ReviewCandidateRole,
  type ReviewDecisionRecord,
  type ReviewIssueTag,
  type ReviewVerdict,
} from "../lib/reviewLearning";
import { triggerBrowserDownload } from "../lib/downloadBlob";
import styles from "./QcReportLab.module.css";

const QC_STREAMING_WAV_THRESHOLD_BYTES = 64 * 1024 * 1024;
const QC_WAV_STREAM_CHUNK_BYTES = 4 * 1024 * 1024;
const QC_WAV_HEADER_SCAN_BYTES = 8 * 1024 * 1024;

type LabMode = "analyze" | "review";

type QcReport = Omit<AudioQcMetrics, "peakDb" | "clipPct"> & {
  fileName: string;
  fileSize: number;
  status: "ok" | "warning" | "error";
  durationSec: number;
  sampleRate: number;
  peakDb: number;
  clipPct: number;
  flags: string[];
  recommendations: string[];
  error?: string;
};

type QcComparison = {
  key: string;
  before: QcReport;
  after: QcReport;
  deltaRisk: number;
  deltaInstability: number;
  deltaOnsetOvershoot: number;
  deltaMidLineSag: number;
  deltaEndFadeRisk: number;
  deltaCompression: number;
  deltaNoiseFloor: number;
  deltaPauseNoiseRisk: number;
  deltaLineSwing: number;
  deltaNoiseContrast: number;
  deltaClick: number;
  deltaEcho: number;
};

type ParsedWavInfo = {
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  audioFormat: number;
  blockAlign: number;
  dataOffset: number;
  dataBytes: number;
  totalFrames: number;
  durationSec: number;
};

type ReviewDecisionDraft = {
  finalVerdict: ReviewVerdict | null;
  issueTags: ReviewIssueTag[];
  preferredRole: ReviewCandidateRole | null;
  confidence: number | null;
  note: string;
};

type ImportedReviewBundle = {
  manifest: ReviewBundleManifest;
  sourceUrl: string;
  winnerUrl: string;
  challengerUrl: string | null;
};

const formatBytes = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const step = 1024;
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(step)), units.length - 1);
  return `${(bytes / step ** exp).toFixed(1)} ${units[exp]}`;
};

const formatSignedPercent = (value: number) => `${value >= 0 ? "+" : ""}${Math.round(value * 100)}%`;
const formatSignedDb = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(1)} dB`;
const formatSignedSeconds = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)} s`;
const formatPercent = (value: number | null | undefined, digits = 0) =>
  typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
const formatDb = (value: number | null | undefined, digits = 1) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)} dB` : "n/a";
const formatNumber = (value: number | null | undefined, digits = 2) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
const formatSeconds = (value: number | null | undefined, digits = 2) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)} s` : "n/a";
const formatIssueTag = (tag: ReviewIssueTag) =>
  tag
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const normalizeComparisonKey = (fileName: string) => {
  let stem = fileName.replace(/\.[^/.]+$/, "").toLowerCase();
  stem = stem.replace(/\s+/g, "_");
  stem = stem.replace(/[^\w-]+/g, "_");
  stem = stem.replace(/_+/g, "_");
  stem = stem.replace(/^_+|_+$/g, "");
  stem = stem.replace(/_blend_mixready$/, "");
  stem = stem.replace(/_mixready$/, "");
  stem = stem.replace(/_(?:blend_)?(?:a85|r128)$/, "");
  stem = stem.replace(/^before_/, "");
  stem = stem.replace(/^after_/, "");
  return stem;
};

const guessRole = (fileName: string) => {
  const lowered = fileName.toLowerCase();
  if (/(^|[_\s-])(after|optimized|processed)([_\s-]|$)/.test(lowered)) return "after";
  if (/(^|[_\s-])(before|original|raw)([_\s-]|$)/.test(lowered)) return "before";
  if (/(_blend_)?(a85|r128)\.wav$/i.test(lowered) || /_mixready\.wav$/i.test(lowered)) return "after";
  return "unknown";
};

const buildComparisons = (reports: QcReport[]) => {
  const byKey = new Map<string, QcReport[]>();
  for (const report of reports) {
    if (report.status === "error") continue;
    const key = normalizeComparisonKey(report.fileName);
    const list = byKey.get(key) ?? [];
    list.push(report);
    byKey.set(key, list);
  }

  const comparisons: QcComparison[] = [];
  for (const [key, group] of byKey.entries()) {
    if (group.length < 2) continue;
    const beforeCandidates = group.filter((report) => guessRole(report.fileName) !== "after");
    const afterCandidates = group.filter((report) => guessRole(report.fileName) === "after");
    if (beforeCandidates.length === 0 || afterCandidates.length === 0) continue;

    const before =
      [...beforeCandidates].sort((a, b) => a.fileName.length - b.fileName.length || a.fileName.localeCompare(b.fileName))[0];
    const after =
      [...afterCandidates].sort((a, b) => a.fileName.length - b.fileName.length || a.fileName.localeCompare(b.fileName))[0];

    if (!before || !after) continue;

    comparisons.push({
      key,
      before,
      after,
      deltaRisk: after.overallRisk - before.overallRisk,
      deltaInstability: after.instabilityScore - before.instabilityScore,
      deltaOnsetOvershoot: after.onsetOvershootScore - before.onsetOvershootScore,
      deltaMidLineSag: after.midLineSagScore - before.midLineSagScore,
      deltaEndFadeRisk: after.endFadeRiskScore - before.endFadeRiskScore,
      deltaCompression: after.compressionScore - before.compressionScore,
      deltaNoiseFloor: after.pauseNoiseFloorDb - before.pauseNoiseFloorDb,
      deltaPauseNoiseRisk: after.pauseNoiseRisk - before.pauseNoiseRisk,
      deltaLineSwing: after.lineSwingScore - before.lineSwingScore,
      deltaNoiseContrast: after.noiseContrastDb - before.noiseContrastDb,
      deltaClick: after.clickScore - before.clickScore,
      deltaEcho: after.echoScore - before.echoScore,
    });
  }

  return comparisons.sort((a, b) => b.before.overallRisk - a.before.overallRisk);
};

const readFourCC = (view: DataView, offset: number) =>
  String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );

const parseWavHeader = async (file: File): Promise<ParsedWavInfo> => {
  const headerScanBytes = Math.min(file.size, QC_WAV_HEADER_SCAN_BYTES);
  const headerBuffer = await file.slice(0, headerScanBytes).arrayBuffer();
  const view = new DataView(headerBuffer);

  if (view.byteLength < 12) {
    throw new Error("WAV header is too small.");
  }
  const riff = readFourCC(view, 0);
  const wave = readFourCC(view, 8);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error("Unsupported WAV container (expected RIFF/WAVE).");
  }

  let offset = 12;
  let channels: number | null = null;
  let sampleRate: number | null = null;
  let bitsPerSample: number | null = null;
  let audioFormat: number | null = null;
  let blockAlign: number | null = null;
  let dataOffset: number | null = null;
  let dataBytes: number | null = null;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readFourCC(view, offset);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;
    const nextOffset = chunkDataOffset + chunkSize + (chunkSize % 2);

    if (chunkId === "fmt " && chunkDataOffset + Math.min(chunkSize, 40) <= view.byteLength) {
      const rawFormat = view.getUint16(chunkDataOffset, true);
      const parsedChannels = view.getUint16(chunkDataOffset + 2, true);
      const parsedSampleRate = view.getUint32(chunkDataOffset + 4, true);
      const parsedBlockAlign = view.getUint16(chunkDataOffset + 12, true);
      const parsedBitsPerSample = view.getUint16(chunkDataOffset + 14, true);

      let normalizedFormat = rawFormat;
      if (rawFormat === 0xfffe && chunkSize >= 40 && chunkDataOffset + 40 <= view.byteLength) {
        normalizedFormat = view.getUint16(chunkDataOffset + 24, true);
      }

      channels = parsedChannels;
      sampleRate = parsedSampleRate;
      blockAlign = parsedBlockAlign;
      bitsPerSample = parsedBitsPerSample;
      audioFormat = normalizedFormat;
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataBytes = Math.min(chunkSize, Math.max(0, file.size - chunkDataOffset));
      break;
    }

    if (nextOffset <= offset) break;
    offset = nextOffset;
  }

  if (channels === null || sampleRate === null || bitsPerSample === null || audioFormat === null || blockAlign === null) {
    throw new Error("WAV fmt chunk not found or incomplete.");
  }
  if (dataOffset === null || dataBytes === null) {
    throw new Error("WAV data chunk not found in header scan.");
  }
  if (channels <= 0 || sampleRate <= 0 || blockAlign <= 0) {
    throw new Error("Invalid WAV format values.");
  }

  const bytesPerSample = blockAlign / channels;
  if (!Number.isInteger(bytesPerSample) || bytesPerSample <= 0) {
    throw new Error("Unsupported WAV block alignment.");
  }

  const supported =
    (audioFormat === 1 && [8, 16, 24, 32].includes(bitsPerSample)) ||
    (audioFormat === 3 && [32, 64].includes(bitsPerSample));
  if (!supported) {
    throw new Error(`Unsupported WAV sample format (format ${audioFormat}, ${bitsPerSample}-bit).`);
  }

  const totalFrames = Math.floor(dataBytes / blockAlign);
  const durationSec = totalFrames / sampleRate;
  return {
    channels,
    sampleRate,
    bitsPerSample,
    audioFormat,
    blockAlign,
    dataOffset,
    dataBytes,
    totalFrames,
    durationSec,
  };
};

const createWavSampleReader = (view: DataView, audioFormat: number, bitsPerSample: number) => {
  if (audioFormat === 3 && bitsPerSample === 32) {
    return (byteOffset: number) => view.getFloat32(byteOffset, true);
  }
  if (audioFormat === 3 && bitsPerSample === 64) {
    return (byteOffset: number) => view.getFloat64(byteOffset, true);
  }
  if (audioFormat === 1 && bitsPerSample === 8) {
    return (byteOffset: number) => (view.getUint8(byteOffset) - 128) / 128;
  }
  if (audioFormat === 1 && bitsPerSample === 16) {
    return (byteOffset: number) => view.getInt16(byteOffset, true) / 32768;
  }
  if (audioFormat === 1 && bitsPerSample === 24) {
    return (byteOffset: number) => {
      let value =
        view.getUint8(byteOffset) | (view.getUint8(byteOffset + 1) << 8) | (view.getUint8(byteOffset + 2) << 16);
      if (value & 0x800000) value |= ~0xffffff;
      return value / 8388608;
    };
  }
  if (audioFormat === 1 && bitsPerSample === 32) {
    return (byteOffset: number) => view.getInt32(byteOffset, true) / 2147483648;
  }
  throw new Error(`Unsupported WAV reader (format ${audioFormat}, ${bitsPerSample}-bit).`);
};

const decodeToMono = async (file: File, audioContext: AudioContext) => {
  const buffer = await file.arrayBuffer();
  const decoded = await audioContext.decodeAudioData(buffer.slice(0));
  const channels = decoded.numberOfChannels;
  const length = decoded.length;
  const mono = new Float32Array(length);

  for (let channel = 0; channel < channels; channel += 1) {
    const data = decoded.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      mono[i] += data[i] / channels;
    }
  }

  let peak = 0;
  let clipCount = 0;
  for (let i = 0; i < mono.length; i += 1) {
    const abs = Math.abs(mono[i]);
    if (abs > peak) peak = abs;
    if (abs >= 0.995) clipCount += 1;
  }

  return {
    samples: mono,
    sampleRate: decoded.sampleRate,
    durationSec: decoded.duration,
    peakDb: toDb(peak + 1e-12),
    clipPct: (clipCount / Math.max(mono.length, 1)) * 100,
  };
};

const toQcReport = (
  fileName: string,
  fileSize: number,
  sampleRate: number,
  durationSec: number,
  metrics: AudioQcMetrics,
): QcReport => {
  const { flags, recommendations } = buildFlagsAndRecommendations(metrics);
  return {
    fileName,
    fileSize,
    status:
      metrics.overallRisk >= 0.56 ||
      metrics.instabilityScore >= 0.74 ||
      metrics.pauseNoiseRisk >= 0.72 ||
      metrics.clickScore >= 0.66
        ? "warning"
        : "ok",
    durationSec,
    sampleRate,
    ...metrics,
    peakDb: metrics.peakDb ?? -120,
    clipPct: metrics.clipPct ?? 0,
    flags,
    recommendations,
  };
};

const createErrorReport = (
  fileName: string,
  fileSize: number,
  error: string,
  sampleRate = 0,
  durationSec = 0,
): QcReport => ({
  fileName,
  fileSize,
  status: "error",
  overallRisk: 1,
  durationSec,
  sampleRate,
  peakDb: -120,
  clipPct: 0,
  noiseFloorDb: -120,
  pauseNoiseFloorDb: -120,
  nearSpeechNoiseFloorDb: null,
  speechThresholdDb: -120,
  noiseContrastDb: 0,
  speechRatioPct: 0,
  speechDutyCyclePct: 0,
  speechSegmentCount: 0,
  medianSpeechRunMs: 0,
  longSilenceCount: 0,
  dynamicRangeDb: 0,
  instabilityScore: 0,
  onsetOvershootScore: 0,
  midLineSagScore: 0,
  endFadeRiskScore: 0,
  endEdgeDipDb: 0,
  lineSwingScore: 0,
  sentenceJumpScore: 0,
  coldOpenDipDb: 0,
  coldOpenRiskScore: 0,
  breathSpikeRisk: 0,
  pauseNoiseRisk: 0,
  compressionScore: 0,
  clickScore: 0,
  reverbScore: 0,
  echoScore: 0,
  roomScore: 0,
  echoDelayMs: null,
  analysisConfidence: 0,
  drynessScore: 0,
  bandSpectrumDb: null,
  sibilanceScore: 0,
  flags: ["File could not be analyzed."],
  recommendations: ["Check that the file is a valid PCM WAV and retry."],
  error,
});

const analyzeFrameFeatures = (
  fileName: string,
  fileSize: number,
  frameRms: number[],
  framePeak: number[],
  frameDb: number[],
  frameSharpness: number[],
  sampleRate: number,
  durationSec: number,
  peakDb: number,
  clipPct: number,
  sampleSpikeCount: number,
): QcReport => {
  if (frameDb.length < 30) {
    return createErrorReport(fileName, fileSize, "Audio is too short for reliable QC analysis.", sampleRate, durationSec);
  }

  const metrics = analyzeFrameAudio(frameRms, framePeak, frameDb, frameSharpness, {
    sampleRate,
    durationSec,
    frameMs: AUDIO_QC_FRAME_MS,
    peakDb,
    clipPct,
    sampleSpikeCount,
  });
  return toQcReport(fileName, fileSize, sampleRate, durationSec, metrics);
};

const analyzeSamples = (
  fileName: string,
  fileSize: number,
  samples: Float32Array,
  sampleRate: number,
  durationSec: number,
  peakDb: number,
  clipPct: number,
): QcReport => {
  const metrics = analyzeFloatSamples(samples, sampleRate, AUDIO_QC_FRAME_MS);
  return toQcReport(fileName, fileSize, sampleRate, durationSec, {
    ...metrics,
    peakDb,
    clipPct,
  });
};

const analyzePcmWavStreaming = async (file: File): Promise<QcReport> => {
  const wav = await parseWavHeader(file);
  const frameSize = Math.max(1, Math.round((wav.sampleRate * AUDIO_QC_FRAME_MS) / 1000));
  const frameCount = Math.floor(wav.totalFrames / frameSize);

  const frameRms = new Array<number>(Math.max(0, frameCount));
  const frameDb = new Array<number>(Math.max(0, frameCount));
  const framePeak = new Array<number>(Math.max(0, frameCount));
  const frameSharpness = new Array<number>(Math.max(0, frameCount));

  let globalPeak = 0;
  let clipCount = 0;
  let monoSampleCount = 0;
  let sampleSpikeCount = 0;
  const refractorySamples = Math.max(1, Math.round(wav.sampleRate * 0.004));
  let lastSpikeIndex = -refractorySamples;
  let prevMonoSample = 0;
  let hasPrevMonoSample = false;

  let frameWriteIndex = 0;
  let frameFill = 0;
  const frameBuffer = new Float32Array(frameSize);

  const bytesPerFrame = wav.blockAlign;
  const bytesPerChannelSample = wav.blockAlign / wav.channels;

  for (let dataRead = 0; dataRead < wav.dataBytes; ) {
    const remaining = wav.dataBytes - dataRead;
    let chunkBytes = Math.min(QC_WAV_STREAM_CHUNK_BYTES, remaining);
    chunkBytes -= chunkBytes % bytesPerFrame;
    if (chunkBytes <= 0) {
      chunkBytes = Math.min(bytesPerFrame, remaining);
      chunkBytes -= chunkBytes % bytesPerFrame;
      if (chunkBytes <= 0) break;
    }

    const buffer = await file.slice(wav.dataOffset + dataRead, wav.dataOffset + dataRead + chunkBytes).arrayBuffer();
    const view = new DataView(buffer);
    const readSample = createWavSampleReader(view, wav.audioFormat, wav.bitsPerSample);
    const interleavedFrames = Math.floor(view.byteLength / bytesPerFrame);

    for (let frame = 0; frame < interleavedFrames; frame += 1) {
      const baseByteOffset = frame * bytesPerFrame;
      let mono = 0;
      for (let channel = 0; channel < wav.channels; channel += 1) {
        const sampleOffset = baseByteOffset + channel * bytesPerChannelSample;
        mono += readSample(sampleOffset) / wav.channels;
      }

      const abs = Math.abs(mono);
      if (abs > globalPeak) globalPeak = abs;
      if (abs >= 0.995) clipCount += 1;

      if (hasPrevMonoSample) {
        const diff = Math.abs(mono - prevMonoSample);
        if (diff >= 0.09 && abs >= 0.015 && monoSampleCount - lastSpikeIndex >= refractorySamples) {
          sampleSpikeCount += 1;
          lastSpikeIndex = monoSampleCount;
        }
      }
      prevMonoSample = mono;
      hasPrevMonoSample = true;

      if (frameWriteIndex < frameCount) {
        frameBuffer[frameFill] = mono;
        frameFill += 1;
        if (frameFill === frameSize) {
          let sumSquares = 0;
          let peak = 0;
          let sharpEnergy = 0;
          for (let i = 0; i < frameSize; i += 1) {
            const value = frameBuffer[i];
            const frameAbs = Math.abs(value);
            if (frameAbs > peak) peak = frameAbs;
            sumSquares += value * value;
            const prev = i > 0 ? frameBuffer[i - 1] : value;
            const next = i + 1 < frameSize ? frameBuffer[i + 1] : value;
            const spike = value - (prev + next) * 0.5;
            sharpEnergy += spike * spike;
          }
          const rms = Math.sqrt(sumSquares / frameSize);
          frameRms[frameWriteIndex] = rms;
          framePeak[frameWriteIndex] = peak;
          frameDb[frameWriteIndex] = Math.max(-120, toDb(rms + 1e-12));
          frameSharpness[frameWriteIndex] = toDb(Math.sqrt(sharpEnergy / frameSize) + 1e-12);
          frameWriteIndex += 1;
          frameFill = 0;
        }
      }

      monoSampleCount += 1;
    }

    dataRead += interleavedFrames * bytesPerFrame;
    if (interleavedFrames === 0) break;
  }

  const analyzedFrames = Math.min(frameWriteIndex, frameCount);
  const durationSec = wav.durationSec;
  const peakDb = toDb(globalPeak + 1e-12);
  const clipPct = (clipCount / Math.max(monoSampleCount, 1)) * 100;

  return analyzeFrameFeatures(
    file.name,
    file.size,
    frameRms.slice(0, analyzedFrames),
    framePeak.slice(0, analyzedFrames),
    frameDb.slice(0, analyzedFrames),
    frameSharpness.slice(0, analyzedFrames),
    wav.sampleRate,
    durationSec,
    peakDb,
    clipPct,
    sampleSpikeCount,
  );
};

const createEmptyReviewDraft = (): ReviewDecisionDraft => ({
  finalVerdict: null,
  issueTags: [],
  preferredRole: null,
  confidence: null,
  note: "",
});

const revokeImportedReviewBundles = (bundles: ImportedReviewBundle[]) => {
  for (const bundle of bundles) {
    URL.revokeObjectURL(bundle.sourceUrl);
    URL.revokeObjectURL(bundle.winnerUrl);
    if (bundle.challengerUrl) {
      URL.revokeObjectURL(bundle.challengerUrl);
    }
  }
};

const importReviewBundleZip = async (file: File): Promise<ImportedReviewBundle[]> => {
  const zip = await JSZip.loadAsync(await file.arrayBuffer());
  const manifestPaths = Object.keys(zip.files).filter(
    (path) => !zip.files[path]?.dir && /(^|\/)manifest\.json$/i.test(path),
  );

  const bundles: ImportedReviewBundle[] = [];
  for (const manifestPath of manifestPaths) {
    const manifestEntry = zip.file(manifestPath);
    if (!manifestEntry) continue;
    const manifest = JSON.parse(await manifestEntry.async("text")) as ReviewBundleManifest;
    if (manifest.schemaVersion !== REVIEW_BUNDLE_SCHEMA_VERSION) {
      throw new Error(`Unsupported review bundle schema in ${file.name}.`);
    }

    const prefix = manifestPath.slice(0, manifestPath.length - "manifest.json".length);
    const resolveAssetUrl = async (relativePath: string) => {
      const directPath = `${prefix}${relativePath}`;
      const entry = zip.file(directPath) ?? zip.file(relativePath);
      if (!entry) {
        throw new Error(`Missing ${relativePath} in ${file.name}.`);
      }
      const blob = await entry.async("blob");
      return URL.createObjectURL(blob);
    };

    const winner = manifest.candidates.find((candidate) => candidate.role === "winner");
    const challenger = manifest.candidates.find((candidate) => candidate.role === "challenger") ?? null;
    if (!winner) {
      throw new Error(`Review bundle ${manifest.bundleId} is missing a winner candidate.`);
    }

    bundles.push({
      manifest,
      sourceUrl: await resolveAssetUrl(manifest.source.audioFile),
      winnerUrl: await resolveAssetUrl(winner.audioFile),
      challengerUrl: challenger ? await resolveAssetUrl(challenger.audioFile) : null,
    });
  }

  return bundles.sort((left, right) => left.manifest.source.fileName.localeCompare(right.manifest.source.fileName));
};

export default function QcReportLab() {
  const [mode, setMode] = useState<LabMode>("analyze");
  const [files, setFiles] = useState<File[]>([]);
  const [reports, setReports] = useState<QcReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [dragActive, setDragActive] = useState(false);

  const [reviewBundles, setReviewBundles] = useState<ImportedReviewBundle[]>([]);
  const [reviewDecisions, setReviewDecisions] = useState<Record<string, ReviewDecisionDraft>>({});
  const [autoReviewResults, setAutoReviewResults] = useState<Record<string, AutoReviewResult>>({});
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewTrainingBusy, setReviewTrainingBusy] = useState(false);
  const [reviewStatus, setReviewStatus] = useState("Idle");

  const hasWarnings = useMemo(() => reports.some((report) => report.status === "warning"), [reports]);
  const comparisons = useMemo(() => buildComparisons(reports), [reports]);

  useEffect(() => () => revokeImportedReviewBundles(reviewBundles), [reviewBundles]);

  const completedReviewRecords = useMemo(
    () =>
      reviewBundles.flatMap((bundle) => {
        const decision = reviewDecisions[bundle.manifest.bundleId];
        if (!decision?.finalVerdict) return [];
        return [
          {
            schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
            bundleId: bundle.manifest.bundleId,
            reviewedAt: new Date().toISOString(),
            finalVerdict: decision.finalVerdict,
            issueTags: decision.issueTags,
            preferredRole: decision.preferredRole,
            confidence: decision.confidence,
            note: decision.note.trim() || null,
          } satisfies ReviewDecisionRecord,
        ];
      }),
    [reviewBundles, reviewDecisions],
  );

  const reviewSummary = useMemo(
    () => ({
      total: reviewBundles.length,
      completed: completedReviewRecords.length,
      failed: completedReviewRecords.filter((record) => record.finalVerdict === "fail").length,
      challengerWins: completedReviewRecords.filter((record) => record.preferredRole === "challenger").length,
    }),
    [completedReviewRecords, reviewBundles.length],
  );
  const autoReviewSummary = useMemo(
    () => ({
      total: Object.keys(autoReviewResults).length,
      failed: Object.values(autoReviewResults).filter((result) => result.finalVerdict === "fail").length,
      challengerPreferred: Object.values(autoReviewResults).filter((result) => result.preferredRole === "challenger").length,
    }),
    [autoReviewResults],
  );

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

  const runAnalysis = async () => {
    if (files.length === 0 || loading) return;

    setLoading(true);
    setReports([]);
    setStatus("Preparing analysis...");

    let audioContext: AudioContext | null = null;
    const nextReports: QcReport[] = [];

    try {
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        setStatus(`Analyzing ${file.name} (${index + 1}/${files.length})`);
        try {
          let report: QcReport;
          const useStreamingWav = file.size >= QC_STREAMING_WAV_THRESHOLD_BYTES;
          if (useStreamingWav) {
            setStatus(`Analyzing ${file.name} (${index + 1}/${files.length}) • streaming WAV mode`);
            report = await analyzePcmWavStreaming(file);
          } else {
            if (!audioContext) audioContext = new AudioContext();
            const decoded = await decodeToMono(file, audioContext);
            report = analyzeSamples(
              file.name,
              file.size,
              decoded.samples,
              decoded.sampleRate,
              decoded.durationSec,
              decoded.peakDb,
              decoded.clipPct,
            );
          }
          nextReports.push(report);
        } catch (error) {
          nextReports.push(createErrorReport(file.name, file.size, error instanceof Error ? error.message : String(error)));
        }
      }
      setReports(nextReports);
      const hasError = nextReports.some((report) => report.status === "error");
      const hasWarning = nextReports.some((report) => report.status === "warning");
      setStatus(hasError ? "Done with errors" : hasWarning ? "Done with warnings" : "Done");
    } finally {
      if (audioContext) {
        await audioContext.close();
      }
      setLoading(false);
    }
  };

  const downloadReportJson = () => {
    if (reports.length === 0) return;
    const payload = {
      generatedAt: new Date().toISOString(),
      status,
      reports,
    };
    triggerBrowserDownload(
      new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
      `qc_report_${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
    );
  };

  const importReviewBundles = async (incoming: FileList | null) => {
    if (!incoming || reviewLoading) return;
    const zips = Array.from(incoming).filter((file) => file.name.toLowerCase().endsWith(".zip"));
    if (zips.length === 0) {
      setReviewStatus("Select one or more review bundle ZIP files.");
      return;
    }

    setReviewLoading(true);
    setReviewStatus("Importing review bundles...");
    const nextBundles: ImportedReviewBundle[] = [];

    try {
      for (let index = 0; index < zips.length; index += 1) {
        const file = zips[index];
        setReviewStatus(`Importing ${file.name} (${index + 1}/${zips.length})`);
        const imported = await importReviewBundleZip(file);
        nextBundles.push(...imported);
      }

      revokeImportedReviewBundles(reviewBundles);
      setReviewBundles(nextBundles);
      setReviewDecisions(
        Object.fromEntries(nextBundles.map((bundle) => [bundle.manifest.bundleId, createEmptyReviewDraft()])),
      );
      setAutoReviewResults({});
      setReviewStatus(`Loaded ${nextBundles.length} review bundle(s).`);
    } catch (error) {
      revokeImportedReviewBundles(nextBundles);
      setReviewStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setReviewLoading(false);
    }
  };

  const updateReviewDecision = (
    bundleId: string,
    updater: (draft: ReviewDecisionDraft) => ReviewDecisionDraft,
  ) => {
    setReviewDecisions((prev) => ({
      ...prev,
      [bundleId]: updater(prev[bundleId] ?? createEmptyReviewDraft()),
    }));
  };

  const clearReviewBundles = () => {
    revokeImportedReviewBundles(reviewBundles);
    setReviewBundles([]);
    setReviewDecisions({});
    setAutoReviewResults({});
    setReviewStatus("Idle");
  };

  const exportReviewDataset = () => {
    if (completedReviewRecords.length === 0) return;
    const jsonl = serializeReviewDecisionJsonl(completedReviewRecords);
    triggerBrowserDownload(
      new Blob([jsonl], { type: "application/x-ndjson" }),
      `review_labels_${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`,
    );
  };

  const runAutoReviewAll = () => {
    if (reviewBundles.length === 0 || reviewTrainingBusy) return;

    const nextResults: Record<string, AutoReviewResult> = {};
    const nextDrafts: Record<string, ReviewDecisionDraft> = {};

    for (const bundle of reviewBundles) {
      const auto = autoReviewBundle(bundle.manifest);
      nextResults[bundle.manifest.bundleId] = auto;
      nextDrafts[bundle.manifest.bundleId] = {
        finalVerdict: auto.finalVerdict,
        issueTags: auto.issueTags,
        preferredRole: auto.preferredRole,
        confidence: Number(auto.confidence.toFixed(2)),
        note: auto.note,
      };
    }

    setAutoReviewResults(nextResults);
    setReviewDecisions(nextDrafts);
    setReviewStatus(
      `Auto-reviewed ${reviewBundles.length} bundle(s) with detailed technical notes. Review labels are ready for training.`,
    );
  };

  const trainReviewModelInBrowser = async () => {
    if (completedReviewRecords.length === 0 || reviewTrainingBusy) return;

    setReviewTrainingBusy(true);
    setReviewStatus("Training review model from current labels...");

    try {
      const manifests = reviewBundles.map((bundle) => bundle.manifest);
      const { weights, report } = fitLearnedReviewWeights(manifests, completedReviewRecords);

      if (typeof window !== "undefined") {
        window.localStorage.setItem(REVIEW_WEIGHT_STORAGE_KEY, JSON.stringify(weights));
      }

      const zip = new JSZip();
      zip.file("review-weights.json", JSON.stringify(weights, null, 2));
      zip.file("review-training-report.json", JSON.stringify(report, null, 2));

      const zipBlob = await zip.generateAsync({
        type: "blob",
        compression: "DEFLATE",
        compressionOptions: { level: 6 },
      });

      triggerBrowserDownload(
        zipBlob,
        `review_model_${new Date().toISOString().replace(/[:.]/g, "-")}.zip`,
      );
      setReviewStatus(
        `Trained ${weights.modelName} from ${completedReviewRecords.length} review(s), applied it locally, and downloaded the model ZIP.`,
      );
    } catch (error) {
      setReviewStatus(`Training failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setReviewTrainingBusy(false);
    }
  };

  return (
    <div className={styles.layout}>
      <div className={styles.card}>
        <div className={styles.modeSwitch}>
          <button
            type="button"
            className={`${styles.modeButton} ${mode === "analyze" ? styles.modeButtonActive : ""}`}
            onClick={() => setMode("analyze")}
          >
            Analyze + QC
          </button>
          <button
            type="button"
            className={`${styles.modeButton} ${mode === "review" ? styles.modeButtonActive : ""}`}
            onClick={() => setMode("review")}
          >
            Review Mode
          </button>
        </div>
        <p className={styles.footerNote}>
          Analyze mode checks raw WAVs directly. Review mode scores optimizer bundles locally and exports JSONL labels for offline training.
        </p>
      </div>

      {mode === "analyze" ? (
        <>
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
                <div className={styles.dropTitle}>Drop WAV files for QC analysis</div>
                <div className={styles.dropHint}>
                  Analyze + QC report runs locally in this browser. No file is uploaded anywhere.
                </div>
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
                  <button className={styles.buttonSecondary} onClick={runAnalysis} disabled={loading || files.length === 0}>
                    {loading ? "Analyzing..." : "Run Analyze + QC"}
                  </button>
                  <button className={styles.buttonGhost} onClick={downloadReportJson} disabled={reports.length === 0}>
                    Download JSON
                  </button>
                </div>
                <div className={styles.progress}>{status}</div>
                <div className={styles.fileList}>
                  {files.length === 0 && <div className={styles.dropHint}>No files selected.</div>}
                  {files.map((file, index) => (
                    <div className={styles.fileItem} key={`${file.name}-${index}-${file.lastModified}`}>
                      <span>{file.name}</span>
                      <span>{formatBytes(file.size)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className={styles.card}>
              <h3>QC Summary</h3>
              <div className={styles.summaryRow}>
                <span>{reports.length} analyzed file(s)</span>
                <span>{reports.filter((report) => report.status === "error").length} error(s)</span>
              </div>
              <div className={styles.summaryRow}>
                <span>{reports.filter((report) => report.status === "warning").length} warning(s)</span>
                <span>{reports.filter((report) => report.status === "ok").length} pass</span>
              </div>
              <div className={styles.summaryRow}>
                <span>{comparisons.length} before/after pair(s)</span>
                <span>{comparisons.filter((pair) => pair.deltaRisk > 0.05).length} regressed pair(s)</span>
              </div>
              <div className={styles.badges}>
                <span className={styles.badge}>Instability</span>
                <span className={styles.badge}>Onset / Sag / End Fade</span>
                <span className={styles.badge}>Noise lift risk</span>
                <span className={styles.badge}>Clicks + Echo</span>
                <span className={styles.badge}>Compression risk</span>
              </div>
              <p className={styles.footerNote}>
                Use this page as a local diagnostics lab before final production runs.
              </p>
            </div>
          </div>

          <div className={styles.card}>
            <h3>Before vs After Delta</h3>
            {comparisons.length === 0 ? (
              <div className={styles.emptyState}>
                Add before/after files with similar names (for example <code>before_name.wav</code> and{" "}
                <code>name_A85.wav</code>) to auto-generate pair deltas.
              </div>
            ) : (
              <div className={styles.reportList}>
                {comparisons.map((comparison) => {
                  const regressed = comparison.deltaRisk > 0.05;
                  const noiseNotLifted = comparison.deltaNoiseFloor <= 1;
                  const sentenceJumpDown = comparison.after.sentenceJumpScore - comparison.before.sentenceJumpScore <= 0.05;
                  const instabilityDown = comparison.deltaInstability <= 0.02;
                  const truePeakSafe = comparison.after.peakDb <= -1.5;
                  const sibilanceNotWorse =
                    comparison.after.sibilanceScore - comparison.before.sibilanceScore <= 0.05;
                  const checks = [
                    { label: "Volume stability", pass: instabilityDown, detail: `Δ instability ${formatSignedPercent(comparison.deltaInstability)}` },
                    { label: "Sentence-jump", pass: sentenceJumpDown, detail: `Δ ${formatSignedPercent(comparison.after.sentenceJumpScore - comparison.before.sentenceJumpScore)}` },
                    { label: "Noise floor not lifted", pass: noiseNotLifted, detail: `Δ ${formatSignedDb(comparison.deltaNoiseFloor)}` },
                    { label: "True peak ≤ -1.5 dBFS", pass: truePeakSafe, detail: `${comparison.after.peakDb.toFixed(1)} dB` },
                    { label: "Sibilance stable", pass: sibilanceNotWorse, detail: `Δ ${formatSignedPercent(comparison.after.sibilanceScore - comparison.before.sibilanceScore)}` },
                  ];
                  const passCount = checks.filter((check) => check.pass).length;
                  const allPass = passCount === checks.length;

                  return (
                    <div className={styles.reportItem} key={comparison.key}>
                      <div className={styles.reportHeader}>
                        <div>
                          <strong>{comparison.key}</strong>
                          <div className={styles.muted}>
                            {comparison.before.fileName} {"->"} {comparison.after.fileName}
                          </div>
                        </div>
                        <span
                          className={`${styles.statusBadge} ${
                            allPass ? styles.statusOk : regressed ? styles.statusWarning : styles.statusOk
                          }`}
                        >
                          {allPass ? `All checks pass (${passCount}/${checks.length})` : regressed ? "Regression risk" : `${passCount}/${checks.length} checks pass`}
                        </span>
                      </div>

                      <div className={styles.metricGrid}>
                        {checks.map((check) => (
                          <div className={styles.metric} key={check.label}>
                            <span>{check.label}</span>
                            <strong>
                              {check.pass ? "PASS" : "FAIL"} — {check.detail}
                            </strong>
                          </div>
                        ))}
                      </div>

                      <div className={styles.metricGrid}>
                        <div className={styles.metric}>
                          <span>Overall risk delta</span>
                          <strong>{formatSignedPercent(comparison.deltaRisk)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Instability delta</span>
                          <strong>{formatSignedPercent(comparison.deltaInstability)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Onset spike delta</span>
                          <strong>{formatSignedPercent(comparison.deltaOnsetOvershoot)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Mid-line sag delta</span>
                          <strong>{formatSignedPercent(comparison.deltaMidLineSag)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>End-fade delta</span>
                          <strong>{formatSignedPercent(comparison.deltaEndFadeRisk)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Compression delta</span>
                          <strong>{formatSignedPercent(comparison.deltaCompression)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Line swing delta</span>
                          <strong>{formatSignedPercent(comparison.deltaLineSwing)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Pause noise floor delta</span>
                          <strong>{formatSignedDb(comparison.deltaNoiseFloor)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Pause noise risk delta</span>
                          <strong>{formatSignedPercent(comparison.deltaPauseNoiseRisk)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Noise contrast delta</span>
                          <strong>{formatSignedDb(comparison.deltaNoiseContrast)}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Click / Echo delta</span>
                          <strong>
                            {formatSignedPercent(comparison.deltaClick)} / {formatSignedPercent(comparison.deltaEcho)}
                          </strong>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className={styles.card}>
            <h3>Per-file QC Report</h3>
            {reports.length === 0 ? (
              <div className={styles.emptyState}>Run analysis to generate report cards.</div>
            ) : (
              <div className={styles.reportList}>
                {reports.map((report, index) => (
                  <div className={styles.reportItem} key={`${report.fileName}-${index}`}>
                    <div className={styles.reportHeader}>
                      <div>
                        <strong>{report.fileName}</strong>
                        <div className={styles.muted}>{formatBytes(report.fileSize)}</div>
                      </div>
                      <span
                        className={`${styles.statusBadge} ${
                          report.status === "error"
                            ? styles.statusError
                            : report.status === "warning"
                              ? styles.statusWarning
                              : styles.statusOk
                        }`}
                      >
                        {report.status === "error"
                          ? "Error"
                          : report.status === "warning"
                            ? "Needs attention"
                            : "Pass"}
                      </span>
                    </div>

                    {report.status === "error" ? (
                      <div className={styles.errorText}>{report.error ?? "Analysis failed."}</div>
                    ) : (
                      <>
                        <div className={styles.metricGrid}>
                          <div className={styles.metric}>
                            <span>Overall risk</span>
                            <strong>{Math.round(report.overallRisk * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Instability</span>
                            <strong>{Math.round(report.instabilityScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Onset spike risk</span>
                            <strong>{Math.round(report.onsetOvershootScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Mid-line sag risk</span>
                            <strong>{Math.round(report.midLineSagScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>End-fade risk</span>
                            <strong>{Math.round(report.endFadeRiskScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Compression risk</span>
                            <strong>{Math.round(report.compressionScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Line swing risk</span>
                            <strong>{Math.round(report.lineSwingScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Noise floor</span>
                            <strong>{report.noiseFloorDb.toFixed(1)} dB</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Pause noise risk</span>
                            <strong>{Math.round(report.pauseNoiseRisk * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Click score</span>
                            <strong>{Math.round(report.clickScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Echo score</span>
                            <strong>{Math.round(report.echoScore * 100)}%</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Speech range</span>
                            <strong>{report.dynamicRangeDb.toFixed(1)} dB</strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Speech duty / median run</span>
                            <strong>
                              {report.speechDutyCyclePct.toFixed(1)}% / {(report.medianSpeechRunMs / 1000).toFixed(1)}s
                            </strong>
                          </div>
                          <div className={styles.metric}>
                            <span>Peak / clip</span>
                            <strong>
                              {report.peakDb.toFixed(1)} dB / {report.clipPct.toFixed(3)}%
                            </strong>
                          </div>
                        </div>

                        <div className={styles.section}>
                          <div className={styles.sectionTitle}>Flags</div>
                          <ul>
                            {report.flags.map((flag, flagIndex) => (
                              <li key={`${report.fileName}-flag-${flagIndex}`}>{flag}</li>
                            ))}
                          </ul>
                        </div>

                        <div className={styles.section}>
                          <div className={styles.sectionTitle}>Recommendations</div>
                          <ul>
                            {report.recommendations.map((item, recIndex) => (
                              <li key={`${report.fileName}-rec-${recIndex}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {hasWarnings && (
            <div className={styles.warningBanner}>
              QC warnings found. Review flagged files before production export.
            </div>
          )}
        </>
      ) : (
        <>
          <div className={styles.panel}>
            <div className={styles.card}>
              <div className={styles.dropzone}>
                <div className={styles.dropTitle}>Import review bundle ZIPs</div>
                <div className={styles.dropHint}>
                  These bundles come from the optimizer export. Review stays local in this browser and exports JSONL labels for offline training.
                </div>
                <div className={styles.controls}>
                  <label className={styles.button}>
                    Select ZIPs
                    <input
                      type="file"
                      accept=".zip,application/zip"
                      multiple
                      hidden
                      onChange={async (event) => {
                        await importReviewBundles(event.target.files);
                        event.currentTarget.value = "";
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    className={styles.buttonSecondary}
                    onClick={runAutoReviewAll}
                    disabled={reviewBundles.length === 0 || reviewTrainingBusy}
                  >
                    Auto-review all
                  </button>
                  <button
                    type="button"
                    className={styles.buttonSecondary}
                    onClick={exportReviewDataset}
                    disabled={completedReviewRecords.length === 0}
                  >
                    Export JSONL
                  </button>
                  <button
                    type="button"
                    className={styles.buttonSecondary}
                    onClick={trainReviewModelInBrowser}
                    disabled={completedReviewRecords.length === 0 || reviewTrainingBusy}
                  >
                    {reviewTrainingBusy ? "Training..." : "Train + Apply"}
                  </button>
                  <button
                    type="button"
                    className={styles.buttonGhost}
                    onClick={clearReviewBundles}
                    disabled={reviewBundles.length === 0}
                  >
                    Clear
                  </button>
                </div>
                <div className={styles.progress}>
                  {reviewLoading ? "Importing..." : reviewTrainingBusy ? "Training..." : reviewStatus}
                </div>
              </div>
            </div>

            <div className={styles.card}>
              <h3>Review Summary</h3>
              <div className={styles.summaryRow}>
                <span>{reviewSummary.total} bundle(s)</span>
                <span>{reviewSummary.completed} labeled</span>
              </div>
              <div className={styles.summaryRow}>
                <span>{reviewSummary.failed} fail verdict(s)</span>
                <span>{reviewSummary.challengerWins} challenger win(s)</span>
              </div>
              <div className={styles.summaryRow}>
                <span>{autoReviewSummary.total} auto-reviewed</span>
                <span>{autoReviewSummary.failed} auto-fail / {autoReviewSummary.challengerPreferred} challenger preferred</span>
              </div>
              <div className={styles.badges}>
                {REVIEW_ISSUE_TAGS.map((tag) => (
                  <span className={styles.badge} key={tag}>
                    {formatIssueTag(tag)}
                  </span>
                ))}
              </div>
              <p className={styles.footerNote}>
                Required labels are final verdict plus issue tags. A/B winner, confidence, and notes are optional but improve the trainer signal.
                Auto-review all performs a detailed technical pass first. Train + Apply writes `review-weights.json` into this browser&apos;s local storage so the optimizer can use it immediately.
              </p>
            </div>
          </div>

          <div className={styles.card}>
            <h3>Bundle Review Queue</h3>
            {reviewBundles.length === 0 ? (
              <div className={styles.emptyState}>
                Import a review bundle ZIP from the optimizer to start labeling source, winner, and challenger candidates.
              </div>
            ) : (
              <div className={styles.reportList}>
                {reviewBundles.map((bundle) => {
                  const decision = reviewDecisions[bundle.manifest.bundleId] ?? createEmptyReviewDraft();
                  const autoReview = autoReviewResults[bundle.manifest.bundleId] ?? null;
                  const winner = bundle.manifest.candidates.find((candidate) => candidate.role === "winner");
                  const challenger = bundle.manifest.candidates.find((candidate) => candidate.role === "challenger") ?? null;
                  const verdictClass =
                    decision.finalVerdict === "fail"
                      ? styles.statusError
                      : decision.finalVerdict === "pass"
                        ? styles.statusOk
                        : styles.statusWarning;

                  return (
                    <div className={styles.reviewBundleCard} key={bundle.manifest.bundleId}>
                      <div className={styles.reportHeader}>
                        <div>
                          <strong>{bundle.manifest.source.fileName}</strong>
                          <div className={styles.muted}>
                            {bundle.manifest.bundleId} • {bundle.manifest.decisionContext.selectedVariant}
                          </div>
                        </div>
                        <span className={`${styles.statusBadge} ${verdictClass}`}>
                          {decision.finalVerdict ? decision.finalVerdict.toUpperCase() : "Pending review"}
                        </span>
                      </div>

                      <div className={styles.reviewMetaGrid}>
                        <div className={styles.metric}>
                          <span>App decision</span>
                          <strong>{bundle.manifest.decisionContext.selectedReason ?? "n/a"}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Review model</span>
                          <strong>{bundle.manifest.decisionContext.learnedWeightsName}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Weight source</span>
                          <strong>{bundle.manifest.decisionContext.learnedWeightsSource}</strong>
                        </div>
                        <div className={styles.metric}>
                          <span>Source duration / rate</span>
                          <strong>
                            {formatSeconds(bundle.manifest.source.durationSec)} / {bundle.manifest.source.sampleRate || "n/a"} Hz
                          </strong>
                        </div>
                      </div>

                      <div className={styles.reviewAudioGrid}>
                        <div className={styles.audioCard}>
                          <div className={styles.sectionTitle}>Source</div>
                          <audio controls preload="metadata" src={bundle.sourceUrl} className={styles.audioPlayer} />
                        </div>
                        <div className={styles.audioCard}>
                          <div className={styles.sectionTitle}>Winner</div>
                          <audio controls preload="metadata" src={bundle.winnerUrl} className={styles.audioPlayer} />
                        </div>
                        {bundle.challengerUrl && challenger && (
                          <div className={styles.audioCard}>
                            <div className={styles.sectionTitle}>Challenger</div>
                            <audio controls preload="metadata" src={bundle.challengerUrl} className={styles.audioPlayer} />
                          </div>
                        )}
                      </div>

                      <div className={styles.reviewCandidateGrid}>
                        {bundle.manifest.candidates.map((candidate) => (
                          <div className={styles.reviewCandidateCard} key={`${bundle.manifest.bundleId}-${candidate.role}`}>
                            <div className={styles.reviewCandidateHeader}>
                              <strong>{candidate.variantLabel}</strong>
                              <span className={styles.badge}>
                                {candidate.role === "winner" ? "Selected by app" : "Top challenger"}
                              </span>
                            </div>
                            <div className={styles.metricGrid}>
                              <div className={styles.metric}>
                                <span>Render path</span>
                                <strong>{candidate.renderMeta.renderPath}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Healthy segmented</span>
                                <strong>{candidate.renderMeta.segmentedHealthy ? "Yes" : "No"}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Degraded</span>
                                <strong>{candidate.renderMeta.degraded ? "Yes" : "No"}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Baseline total</span>
                                <strong>{formatNumber(candidate.baselineScore.total, 1)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Ranking score</span>
                                <strong>{formatNumber(candidate.ranking.rankingScore, 1)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Hard gate penalty</span>
                                <strong>{formatNumber(candidate.ranking.hardGatePenalty, 1)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Learned adjustment</span>
                                <strong>{formatNumber(candidate.ranking.learnedAdjustment, 1)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Duration delta</span>
                                <strong>{formatSignedSeconds(candidate.sourceComparison.alignment.durationDeltaSec)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Offset / confidence</span>
                                <strong>
                                  {formatSignedSeconds(candidate.sourceComparison.alignment.estimatedOffsetSec)} /{" "}
                                  {formatPercent(candidate.sourceComparison.alignment.confidence)}
                                </strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Overall risk delta</span>
                                <strong>{formatPercent(candidate.sourceComparison.qcDelta?.overallRisk)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>Pause noise delta</span>
                                <strong>{formatPercent(candidate.sourceComparison.qcDelta?.pauseNoiseRisk)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>True peak</span>
                                <strong>{formatDb(candidate.qc?.inputTP)}</strong>
                              </div>
                              <div className={styles.metric}>
                                <span>End-fade risk</span>
                                <strong>{formatPercent(candidate.qc?.endFadeRiskScore)}</strong>
                              </div>
                            </div>
                            {candidate.ranking.gateReasons.length > 0 && (
                              <div className={styles.section}>
                                <div className={styles.sectionTitle}>Gate reasons</div>
                                <div className={styles.tagGrid}>
                                  {candidate.ranking.gateReasons.map((reason) => (
                                    <span className={styles.badge} key={`${candidate.role}-${reason}`}>
                                      {reason}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {candidate.selectionReason && (
                              <div className={styles.section}>
                                <div className={styles.sectionTitle}>Selection reason</div>
                                <div className={styles.muted}>{candidate.selectionReason}</div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>

                      {autoReview && (
                        <div className={styles.autoReviewPanel}>
                          <div className={styles.autoReviewHeader}>
                            <div>
                              <strong>Auto-review Summary</strong>
                              <div className={styles.muted}>{autoReview.executiveSummary}</div>
                            </div>
                            <span className={styles.badge}>
                              {(autoReview.confidence * 100).toFixed(0)}% confidence
                            </span>
                          </div>
                          <div className={styles.autoReviewGrid}>
                            <div className={styles.autoReviewCard}>
                              <div className={styles.sectionTitle}>Selected Output</div>
                              <div className={styles.muted}>{autoReview.selectedAssessment.summary}</div>
                              <div className={styles.findingList}>
                                {autoReview.selectedAssessment.findings.map((finding) => (
                                  <div className={styles.findingItem} key={`${bundle.manifest.bundleId}-sel-${finding.id}`}>
                                    <span
                                      className={`${styles.findingBadge} ${
                                        finding.status === "fail"
                                          ? styles.findingFail
                                          : finding.status === "warn"
                                            ? styles.findingWarn
                                            : styles.findingPass
                                      }`}
                                    >
                                      {finding.status.toUpperCase()}
                                    </span>
                                    <div>
                                      <strong>{finding.label}</strong>
                                      <div className={styles.muted}>{finding.detail}</div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                            {autoReview.challengerAssessment && (
                              <div className={styles.autoReviewCard}>
                                <div className={styles.sectionTitle}>Challenger</div>
                                <div className={styles.muted}>{autoReview.challengerAssessment.summary}</div>
                                <div className={styles.findingList}>
                                  {autoReview.challengerAssessment.findings.map((finding) => (
                                    <div className={styles.findingItem} key={`${bundle.manifest.bundleId}-chal-${finding.id}`}>
                                      <span
                                        className={`${styles.findingBadge} ${
                                          finding.status === "fail"
                                            ? styles.findingFail
                                            : finding.status === "warn"
                                              ? styles.findingWarn
                                              : styles.findingPass
                                        }`}
                                      >
                                        {finding.status.toUpperCase()}
                                      </span>
                                      <div>
                                        <strong>{finding.label}</strong>
                                        <div className={styles.muted}>{finding.detail}</div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      <div className={styles.reviewDecisionGrid}>
                        <div className={styles.field}>
                          <label className={styles.label}>Final verdict</label>
                          <div className={styles.choiceRow}>
                            <button
                              type="button"
                              className={`${styles.choiceButton} ${
                                decision.finalVerdict === "pass" ? styles.choiceButtonActive : ""
                              }`}
                              onClick={() =>
                                updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                  ...draft,
                                  finalVerdict: "pass",
                                }))
                              }
                            >
                              Pass
                            </button>
                            <button
                              type="button"
                              className={`${styles.choiceButton} ${
                                decision.finalVerdict === "fail" ? styles.choiceButtonActive : ""
                              }`}
                              onClick={() =>
                                updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                  ...draft,
                                  finalVerdict: "fail",
                                }))
                              }
                            >
                              Fail
                            </button>
                          </div>
                        </div>

                        <div className={styles.field}>
                          <label className={styles.label}>Preferred A/B output</label>
                          <div className={styles.choiceRow}>
                            <button
                              type="button"
                              className={`${styles.choiceButton} ${
                                decision.preferredRole === null ? styles.choiceButtonActive : ""
                              }`}
                              onClick={() =>
                                updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                  ...draft,
                                  preferredRole: null,
                                }))
                              }
                            >
                              Skip
                            </button>
                            <button
                              type="button"
                              className={`${styles.choiceButton} ${
                                decision.preferredRole === "winner" ? styles.choiceButtonActive : ""
                              }`}
                              onClick={() =>
                                updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                  ...draft,
                                  preferredRole: "winner",
                                }))
                              }
                            >
                              Winner
                            </button>
                            {challenger && (
                              <button
                                type="button"
                                className={`${styles.choiceButton} ${
                                  decision.preferredRole === "challenger" ? styles.choiceButtonActive : ""
                                }`}
                                onClick={() =>
                                  updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                    ...draft,
                                    preferredRole: "challenger",
                                  }))
                                }
                              >
                                Challenger
                              </button>
                            )}
                          </div>
                        </div>

                        <div className={styles.field}>
                          <label className={styles.label}>Confidence</label>
                          <select
                            className={styles.select}
                            value={decision.confidence === null ? "" : String(decision.confidence)}
                            onChange={(event) =>
                              updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                ...draft,
                                confidence: event.target.value ? Number(event.target.value) : null,
                              }))
                            }
                          >
                            <option value="">Not set</option>
                            <option value="0.25">Low</option>
                            <option value="0.5">Medium</option>
                            <option value="0.75">High</option>
                            <option value="1">Very high</option>
                          </select>
                        </div>
                      </div>

                      <div className={styles.section}>
                        <div className={styles.sectionTitle}>Issue tags</div>
                        <div className={styles.tagGrid}>
                          {REVIEW_ISSUE_TAGS.map((tag) => {
                            const active = decision.issueTags.includes(tag);
                            return (
                              <button
                                type="button"
                                key={`${bundle.manifest.bundleId}-${tag}`}
                                className={`${styles.tagButton} ${active ? styles.tagButtonActive : ""}`}
                                onClick={() =>
                                  updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                                    ...draft,
                                    issueTags: draft.issueTags.includes(tag)
                                      ? draft.issueTags.filter((entry) => entry !== tag)
                                      : [...draft.issueTags, tag],
                                  }))
                                }
                              >
                                {formatIssueTag(tag)}
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      <div className={styles.field}>
                        <label className={styles.label}>Reviewer note</label>
                        <textarea
                          className={styles.textarea}
                          value={decision.note}
                          placeholder="Optional note about what you heard."
                          onChange={(event) =>
                            updateReviewDecision(bundle.manifest.bundleId, (draft) => ({
                              ...draft,
                              note: event.target.value,
                            }))
                          }
                        />
                      </div>

                      <div className={styles.reviewFootnote}>
                        Winner: {winner?.variantLabel ?? "n/a"}
                        {challenger ? ` • Challenger: ${challenger.variantLabel}` : ""}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
