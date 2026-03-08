"use client";

import { useMemo, useState, type DragEvent } from "react";
import {
  AUDIO_QC_FRAME_MS,
  analyzeFloatSamples,
  analyzeFrameAudio,
  buildFlagsAndRecommendations,
  toDb,
  type AudioQcMetrics,
} from "../lib/audioQc";
import styles from "./QcReportLab.module.css";

const QC_STREAMING_WAV_THRESHOLD_BYTES = 64 * 1024 * 1024;
const QC_WAV_STREAM_CHUNK_BYTES = 4 * 1024 * 1024;
const QC_WAV_HEADER_SCAN_BYTES = 8 * 1024 * 1024;

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

const formatBytes = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const step = 1024;
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(step)), units.length - 1);
  return `${(bytes / step ** exp).toFixed(1)} ${units[exp]}`;
};

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

const formatSignedPercent = (value: number) => `${value >= 0 ? "+" : ""}${Math.round(value * 100)}%`;
const formatSignedDb = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(1)} dB`;

const readFourCC = (view: DataView, offset: number) =>
  String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3)
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
  metrics: AudioQcMetrics
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
  durationSec = 0
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
  lineSwingScore: 0,
  sentenceJumpScore: 0,
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
  sampleSpikeCount: number
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
  clipPct: number
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
    sampleSpikeCount
  );
};

export default function QcReportLab() {
  const [files, setFiles] = useState<File[]>([]);
  const [reports, setReports] = useState<QcReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [dragActive, setDragActive] = useState(false);

  const hasWarnings = useMemo(() => reports.some((report) => report.status === "warning"), [reports]);
  const comparisons = useMemo(() => buildComparisons(reports), [reports]);

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
              decoded.clipPct
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
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `qc_report_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

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
              return (
                <div className={styles.reportItem} key={comparison.key}>
                  <div className={styles.reportHeader}>
                    <div>
                      <strong>{comparison.key}</strong>
                      <div className={styles.muted}>
                        {comparison.before.fileName} {"->"} {comparison.after.fileName}
                      </div>
                    </div>
                    <span className={`${styles.statusBadge} ${regressed ? styles.statusWarning : styles.statusOk}`}>
                      {regressed ? "Regression risk" : "Improved/Stable"}
                    </span>
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
    </div>
  );
}
