import { spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readSync,
  readdirSync,
  statSync,
} from "node:fs";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import JSZip from "jszip";

type StemName = "BGM" | "VOCAL";
type OutputBitDepth = 16 | 24 | 32;

export type AudioSplitterInput = {
  originalName: string;
  bytes: Uint8Array;
  mimeType?: string;
};

export type AudioSplitterRawStemPaths = {
  vocal: string;
  bgm?: string;
  nonVocal?: string;
  drums?: string;
  bass?: string;
  other?: string;
  guitar?: string;
  piano?: string;
};

export type AudioSplitterEngineContext = {
  originalName: string;
  sanitizedBase: string;
};

export type AudioSplitterEngineBatchItem = AudioSplitterEngineContext & {
  inputIndex: number;
  inputPath: string;
  workDir: string;
  sampleRate: number;
  channels: number;
  durationSeconds: number;
};

export type AudioSplitterEngineBatchResult = {
  inputIndex: number;
  rawStems?: AudioSplitterRawStemPaths;
  error?: string;
};

export type AudioSplitterEngine = {
  name: string;
  split(inputPath: string, workDir: string, context: AudioSplitterEngineContext): Promise<AudioSplitterRawStemPaths>;
  splitBatch?: (
    items: AudioSplitterEngineBatchItem[],
    workDir: string,
    context: { onProgress?: (inputIndex: number, message: string) => void },
  ) => Promise<AudioSplitterEngineBatchResult[]>;
};

export type AudioSplitterOutput = {
  stem: StemName;
  fileName: string;
  path: string;
  sizeBytes: number;
  sampleRate: number;
  channels: number;
  durationSeconds: number;
  qc: AudioSplitterStemQc;
};

export type AudioSplitterStemQc = {
  stem: StemName;
  fileName: string;
  sizeBytes: number;
  sampleRate: number;
  channels: number;
  durationSeconds: number;
  peakDbfs: number | null;
  rmsDbfs: number | null;
  clippedSampleCount: number;
  clippedSamplePct: number;
  silent: boolean;
};

export type AudioSplitterFileReport = {
  inputIndex: number;
  originalName: string;
  sanitizedBase: string | null;
  status: "success" | "failed";
  message: string;
  outputs: string[];
  engine: string;
  sampleRate: number | null;
  channels: number | null;
  durationSeconds: number | null;
  stems: AudioSplitterStemQc[];
  warnings: string[];
};

export type AudioSplitterReport = {
  generatedAt: string;
  engine: string;
  totalFiles: number;
  succeeded: number;
  failed: number;
  files: AudioSplitterFileReport[];
};

export type SplitBatchAudioTracksResult = {
  zip: Buffer;
  zipName: string;
  report: AudioSplitterReport;
};

export type AudioSplitterProgressEvent =
  | {
      type: "file-start";
      inputIndex: number;
      originalName: string;
      sanitizedBase: string;
      message: string;
    }
  | {
      type: "file-progress";
      inputIndex: number;
      originalName: string;
      message: string;
    }
  | {
      type: "file-complete";
      inputIndex: number;
      report: AudioSplitterFileReport;
    }
  | {
      type: "batch-complete";
      report: AudioSplitterReport;
    };

type WavInfo = {
  audioFormat: number;
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  blockAlign: number;
  dataOffset: number;
  dataBytes: number;
  durationSeconds: number;
};

type SplitAudioTrackResult = {
  report: AudioSplitterFileReport;
  outputs: AudioSplitterOutput[];
};

type SplitBatchOptions = {
  engine?: AudioSplitterEngine;
  cleanup?: boolean;
  now?: Date;
  onProgress?: (event: AudioSplitterProgressEvent) => void;
};

const DANGEROUS_FILENAME_CHARS = /[\x00-\x1f<>:"/\\|?*]/g;
const WINDOWS_RESERVED_BASENAME = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;
const DEFAULT_DEMUCS_TIMEOUT_MS = 2 * 60 * 60 * 1000;

const readAscii = (view: DataView, offset: number, length: number) => {
  let value = "";
  for (let i = 0; i < length; i += 1) {
    value += String.fromCharCode(view.getUint8(offset + i));
  }
  return value;
};

const toUint8Array = (bytes: Uint8Array) =>
  new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);

const formatSeconds = (seconds: number | null) =>
  seconds === null || !Number.isFinite(seconds) ? "unknown" : `${seconds.toFixed(3)}s`;

const formatDbfs = (db: number | null) => (db === null || !Number.isFinite(db) ? "n/a" : `${db.toFixed(1)} dBFS`);

export const sanitizeAudioBaseName = (originalName: string, fallbackIndex = 0) => {
  const slashSafeName = originalName.split(/[\\/]/).pop() ?? originalName;
  const withoutExtension = slashSafeName.replace(/\.[^.]*$/, "");
  let safe = withoutExtension.replace(DANGEROUS_FILENAME_CHARS, "_").replace(/[ .]+$/g, "");
  if (WINDOWS_RESERVED_BASENAME.test(safe)) safe = `${safe}_file`;
  if (!safe) safe = `input_${fallbackIndex + 1}`;
  return safe;
};

export const buildAudioSplitterOutputFileNames = (sanitizedBase: string) => ({
  bgm: `${sanitizedBase}_BGM.wav`,
  vocal: `${sanitizedBase}_VOCAL.wav`,
});

const assertSupportedInputWav = (info: WavInfo) => {
  const supportedFormat = info.audioFormat === 1 || info.audioFormat === 3 || info.audioFormat === 0xfffe;
  const supportedBits = [16, 24, 32, 64].includes(info.bitsPerSample);
  if (!supportedFormat || !supportedBits) {
    throw new Error(
      `Unsupported WAV encoding: format ${info.audioFormat}, ${info.bitsPerSample}-bit. Export PCM or float WAV.`,
    );
  }
  if (info.channels < 1 || info.channels > 16) {
    throw new Error(`Unsupported WAV channel count: ${info.channels}.`);
  }
  if (info.sampleRate < 8000 || info.sampleRate > 384000) {
    throw new Error(`Unsupported WAV sample rate: ${info.sampleRate} Hz.`);
  }
  if (info.durationSeconds <= 0) {
    throw new Error("WAV has no audio data.");
  }
};

export const readWavInfoFromBytes = (bytes: Uint8Array): WavInfo => {
  const source = toUint8Array(bytes);
  if (source.byteLength < 44) throw new Error("WAV too small or corrupted.");
  const view = new DataView(source.buffer, source.byteOffset, source.byteLength);

  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") {
    throw new Error("Unsupported file: expected a RIFF/WAVE file.");
  }

  let offset = 12;
  let audioFormat: number | null = null;
  let channels: number | null = null;
  let sampleRate: number | null = null;
  let bitsPerSample: number | null = null;
  let blockAlign: number | null = null;
  let dataOffset = -1;
  let dataBytes = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readAscii(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkStart + chunkSize > view.byteLength) {
      throw new Error(`Corrupted WAV chunk ${chunkId.trim() || "(unknown)"} exceeds file size.`);
    }

    if (chunkId === "fmt ") {
      if (chunkSize < 16) throw new Error("Corrupted WAV fmt chunk.");
      audioFormat = view.getUint16(chunkStart, true);
      channels = view.getUint16(chunkStart + 2, true);
      sampleRate = view.getUint32(chunkStart + 4, true);
      blockAlign = view.getUint16(chunkStart + 12, true);
      bitsPerSample = view.getUint16(chunkStart + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkStart;
      dataBytes = chunkSize;
      break;
    }

    offset = chunkStart + chunkSize + (chunkSize % 2);
  }

  if (
    audioFormat === null ||
    channels === null ||
    sampleRate === null ||
    bitsPerSample === null ||
    blockAlign === null ||
    dataOffset < 0
  ) {
    throw new Error("Corrupted WAV: missing fmt or data chunk.");
  }

  return {
    audioFormat,
    channels,
    sampleRate,
    bitsPerSample,
    blockAlign,
    dataOffset,
    dataBytes,
    durationSeconds: dataBytes / Math.max(sampleRate * blockAlign, 1),
  };
};

const readWavInfoFromFile = (filePath: string): WavInfo => {
  const fd = openSync(filePath, "r");
  try {
    const size = statSync(filePath).size;
    const riff = Buffer.alloc(12);
    if (readSync(fd, riff, 0, 12, 0) !== 12) throw new Error("WAV too small or corrupted.");
    if (riff.toString("ascii", 0, 4) !== "RIFF" || riff.toString("ascii", 8, 12) !== "WAVE") {
      throw new Error("Expected a RIFF/WAVE stem.");
    }

    let offset = 12;
    let audioFormat: number | null = null;
    let channels: number | null = null;
    let sampleRate: number | null = null;
    let bitsPerSample: number | null = null;
    let blockAlign: number | null = null;
    let dataOffset = -1;
    let dataBytes = 0;
    const header = Buffer.alloc(8);

    while (offset + 8 <= size) {
      if (readSync(fd, header, 0, 8, offset) !== 8) break;
      const chunkId = header.toString("ascii", 0, 4);
      const chunkSize = header.readUInt32LE(4);
      const chunkStart = offset + 8;
      if (chunkStart + chunkSize > size) {
        throw new Error(`Corrupted WAV chunk ${chunkId.trim() || "(unknown)"} exceeds file size.`);
      }

      if (chunkId === "fmt ") {
        const fmt = Buffer.alloc(Math.min(chunkSize, 40));
        readSync(fd, fmt, 0, fmt.byteLength, chunkStart);
        if (fmt.byteLength < 16) throw new Error("Corrupted WAV fmt chunk.");
        audioFormat = fmt.readUInt16LE(0);
        channels = fmt.readUInt16LE(2);
        sampleRate = fmt.readUInt32LE(4);
        blockAlign = fmt.readUInt16LE(12);
        bitsPerSample = fmt.readUInt16LE(14);
      } else if (chunkId === "data") {
        dataOffset = chunkStart;
        dataBytes = chunkSize;
        break;
      }

      offset = chunkStart + chunkSize + (chunkSize % 2);
    }

    if (
      audioFormat === null ||
      channels === null ||
      sampleRate === null ||
      bitsPerSample === null ||
      blockAlign === null ||
      dataOffset < 0
    ) {
      throw new Error("Corrupted WAV stem: missing fmt or data chunk.");
    }

    return {
      audioFormat,
      channels,
      sampleRate,
      bitsPerSample,
      blockAlign,
      dataOffset,
      dataBytes,
      durationSeconds: dataBytes / Math.max(sampleRate * blockAlign, 1),
    };
  } finally {
    closeSync(fd);
  }
};

const parseOutputBitDepth = (): OutputBitDepth => {
  const raw = process.env.AUDIO_SPLITTER_OUTPUT_BIT_DEPTH ?? "16";
  const parsed = Number(raw);
  return parsed === 24 || parsed === 32 ? parsed : 16;
};

const decodeWavSamplesFromBytes = (bytes: Uint8Array, info: WavInfo) => {
  const source = toUint8Array(bytes);
  const view = new DataView(source.buffer, source.byteOffset, source.byteLength);
  const bytesPerSample = info.bitsPerSample / 8;
  const totalSamples = Math.floor(info.dataBytes / Math.max(bytesPerSample, 1));
  const samples = new Float32Array(totalSamples);
  const isFloat = info.audioFormat === 3 || (info.audioFormat === 0xfffe && info.bitsPerSample >= 32);

  if (isFloat && info.bitsPerSample === 32) {
    for (let index = 0; index < totalSamples; index += 1) {
      samples[index] = view.getFloat32(info.dataOffset + index * 4, true);
    }
  } else if (isFloat && info.bitsPerSample === 64) {
    for (let index = 0; index < totalSamples; index += 1) {
      samples[index] = view.getFloat64(info.dataOffset + index * 8, true);
    }
  } else if (info.bitsPerSample === 16) {
    for (let index = 0; index < totalSamples; index += 1) {
      samples[index] = view.getInt16(info.dataOffset + index * 2, true) / 0x8000;
    }
  } else if (info.bitsPerSample === 24) {
    for (let index = 0; index < totalSamples; index += 1) {
      const offset = info.dataOffset + index * 3;
      const lo = view.getUint8(offset);
      const mid = view.getUint8(offset + 1);
      const hi = view.getInt8(offset + 2);
      samples[index] = ((hi << 16) | (mid << 8) | lo) / 0x800000;
    }
  } else if (info.bitsPerSample === 32) {
    for (let index = 0; index < totalSamples; index += 1) {
      samples[index] = view.getInt32(info.dataOffset + index * 4, true) / 0x80000000;
    }
  } else {
    throw new Error(`Unsupported WAV sample width ${info.bitsPerSample}-bit.`);
  }

  return samples;
};

const encodeWavFromSamples = (
  samples: Float32Array,
  sampleRate: number,
  channels: number,
  targetBitDepth: OutputBitDepth,
) => {
  const bytesPerSample = targetBitDepth / 8;
  const byteRate = sampleRate * channels * bytesPerSample;
  const blockAlign = channels * bytesPerSample;
  const dataBytes = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, targetBitDepth === 32 ? 3 : 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, targetBitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataBytes, true);

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, Number.isFinite(samples[index]) ? samples[index] : 0));
    const offset = 44 + index * bytesPerSample;
    if (targetBitDepth === 16) {
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    } else if (targetBitDepth === 24) {
      const value = Math.round(sample < 0 ? sample * 0x800000 : sample * 0x7fffff);
      view.setUint8(offset, value & 0xff);
      view.setUint8(offset + 1, (value >> 8) & 0xff);
      view.setUint8(offset + 2, (value >> 16) & 0xff);
    } else {
      view.setFloat32(offset, sample, true);
    }
  }

  return new Uint8Array(buffer);
};

const copyOrEncodeStemAsOutput = async (
  sourcePath: string,
  outputPath: string,
  targetBitDepth: OutputBitDepth,
) => {
  const info = readWavInfoFromFile(sourcePath);
  const sourceIsTargetFloat32 = targetBitDepth === 32 && info.audioFormat === 3 && info.bitsPerSample === 32;
  if (info.bitsPerSample !== targetBitDepth || (targetBitDepth === 32 && !sourceIsTargetFloat32)) {
    const bytes = await readFile(sourcePath);
    await writeFile(
      outputPath,
      Buffer.from(
        encodeWavFromSamples(decodeWavSamplesFromBytes(bytes, info), info.sampleRate, info.channels, targetBitDepth),
      ),
    );
    return;
  }

  if (sourcePath !== outputPath) {
    await copyFile(sourcePath, outputPath);
  }
};

const roundMetric = (value: number, digits: number) => Number(value.toFixed(digits));

const analyzeOutputStemQc = async (
  filePath: string,
  stem: StemName,
  fileName: string,
  sizeBytes: number,
  info: WavInfo,
): Promise<AudioSplitterStemQc> => {
  const samples = decodeWavSamplesFromBytes(await readFile(filePath), info);
  let peak = 0;
  let sumSquares = 0;
  let clippedSampleCount = 0;
  const clipThreshold = 0.999;

  for (const sample of samples) {
    const value = Number.isFinite(sample) ? sample : 0;
    const abs = Math.abs(value);
    if (abs > peak) peak = abs;
    sumSquares += value * value;
    if (abs >= clipThreshold) clippedSampleCount += 1;
  }

  const rms = samples.length > 0 ? Math.sqrt(sumSquares / samples.length) : 0;
  const peakDbfs = peak > 0 ? roundMetric(20 * Math.log10(peak), 2) : null;
  const rmsDbfs = rms > 0 ? roundMetric(20 * Math.log10(rms), 2) : null;

  return {
    stem,
    fileName,
    sizeBytes,
    sampleRate: info.sampleRate,
    channels: info.channels,
    durationSeconds: info.durationSeconds,
    peakDbfs,
    rmsDbfs,
    clippedSampleCount,
    clippedSamplePct: samples.length > 0 ? roundMetric((clippedSampleCount / samples.length) * 100, 5) : 0,
    silent: rmsDbfs === null || rmsDbfs < -75,
  };
};

const copyStemAsOutput = async (
  stem: StemName,
  sourcePath: string,
  outputPath: string,
  fileName: string,
  targetBitDepth: OutputBitDepth,
): Promise<AudioSplitterOutput> => {
  await copyOrEncodeStemAsOutput(sourcePath, outputPath, targetBitDepth);
  const info = readWavInfoFromFile(outputPath);
  const sizeBytes = statSync(outputPath).size;
  const qc = await analyzeOutputStemQc(outputPath, stem, fileName, sizeBytes, info);
  return {
    stem,
    fileName,
    path: outputPath,
    sizeBytes,
    sampleRate: info.sampleRate,
    channels: info.channels,
    durationSeconds: info.durationSeconds,
    qc,
  };
};

const finalizeRawStemOutputs = async (
  rawStems: AudioSplitterRawStemPaths,
  inputInfo: WavInfo,
  reportBase: AudioSplitterFileReport,
  outputDir: string,
  outputNames: ReturnType<typeof buildAudioSplitterOutputFileNames>,
  targetBitDepth: OutputBitDepth,
): Promise<SplitAudioTrackResult> => {
  if (!rawStems.vocal || !existsSync(rawStems.vocal)) {
    throw new Error("Separation engine did not create a VOCAL stem.");
  }

  const vocalOutputPath = path.join(outputDir, "VOCAL.wav");
  const bgmOutputPath = path.join(outputDir, "BGM.wav");
  const bgmSourcePath = rawStems.bgm ?? rawStems.nonVocal;
  if (!bgmSourcePath || !existsSync(bgmSourcePath)) {
    throw new Error("Separation engine did not create a merged BGM stem.");
  }
  const referenceInfo = readWavInfoFromFile(bgmSourcePath);
  const vocalInfo = readWavInfoFromFile(rawStems.vocal);
  if (vocalInfo.sampleRate !== referenceInfo.sampleRate || vocalInfo.channels !== referenceInfo.channels) {
    throw new Error("Generated VOCAL and BGM stems are not format-aligned.");
  }

  const outputs = [
    await copyStemAsOutput("BGM", bgmSourcePath, bgmOutputPath, outputNames.bgm, targetBitDepth),
    await copyStemAsOutput("VOCAL", rawStems.vocal, vocalOutputPath, outputNames.vocal, targetBitDepth),
  ];

  const durationSet = new Set(outputs.map((output) => output.durationSeconds.toFixed(6)));
  if (durationSet.size > 1) {
    throw new Error("Generated stems are not duration-aligned.");
  }

  if (referenceInfo.sampleRate !== inputInfo.sampleRate) {
    reportBase.warnings.push(
      `Output sample rate is ${referenceInfo.sampleRate} Hz because the separation engine produced that rate; original was ${inputInfo.sampleRate} Hz.`,
    );
  }
  for (const output of outputs) {
    if (output.qc.silent) {
      reportBase.warnings.push(`${output.stem} stem appears silent (${formatDbfs(output.qc.rmsDbfs)} RMS).`);
    }
    if (output.qc.clippedSampleCount > 0) {
      reportBase.warnings.push(
        `${output.stem} stem has ${output.qc.clippedSampleCount} near-full-scale sample(s); peak ${formatDbfs(output.qc.peakDbfs)}.`,
      );
    }
  }

  return {
    outputs,
    report: {
      ...reportBase,
      status: "success",
      message: "Split complete.",
      outputs: outputs.map((output) => output.fileName),
      sampleRate: referenceInfo.sampleRate,
      channels: referenceInfo.channels,
      durationSeconds: referenceInfo.durationSeconds,
      stems: outputs.map((output) => output.qc),
    },
  };
};

const splitAudioTrack = async (
  input: AudioSplitterInput,
  inputIndex: number,
  tempRoot: string,
  engine: AudioSplitterEngine,
  targetBitDepth: OutputBitDepth,
): Promise<SplitAudioTrackResult> => {
  const sanitizedBase = sanitizeAudioBaseName(input.originalName, inputIndex);
  const outputNames = buildAudioSplitterOutputFileNames(sanitizedBase);
  const workDir = path.join(tempRoot, `input_${inputIndex + 1}`);
  const sourceDir = path.join(workDir, "source");
  const engineDir = path.join(workDir, "engine");
  const outputDir = path.join(workDir, "outputs");
  await mkdir(sourceDir, { recursive: true });
  await mkdir(engineDir, { recursive: true });
  await mkdir(outputDir, { recursive: true });

  const reportBase: AudioSplitterFileReport = {
    inputIndex,
    originalName: input.originalName,
    sanitizedBase,
    status: "failed",
    message: "",
    outputs: [],
    engine: engine.name,
    sampleRate: null,
    channels: null,
    durationSeconds: null,
    stems: [],
    warnings: [],
  };

  try {
    if (!input.originalName.toLowerCase().endsWith(".wav")) {
      throw new Error("Unsupported file type. Audio Track Splitter accepts .wav files only.");
    }

    const inputInfo = readWavInfoFromBytes(input.bytes);
    assertSupportedInputWav(inputInfo);

    const inputPath = path.join(sourceDir, "input.wav");
    await writeFile(inputPath, Buffer.from(input.bytes));

    const rawStems = await engine.split(inputPath, engineDir, {
      originalName: input.originalName,
      sanitizedBase,
    });
    return await finalizeRawStemOutputs(rawStems, inputInfo, reportBase, outputDir, outputNames, targetBitDepth);
  } catch (error) {
    return {
      outputs: [],
      report: {
        ...reportBase,
        status: "failed",
        message: error instanceof Error ? error.message : String(error),
      },
    };
  }
};

const renderTextReport = (report: AudioSplitterReport) => {
  const lines = [
    "Audio Track Splitter Report",
    `Generated: ${report.generatedAt}`,
    `Engine: ${report.engine}`,
    `Total files: ${report.totalFiles}`,
    `Succeeded: ${report.succeeded}`,
    `Failed: ${report.failed}`,
    "",
  ];

  for (const file of report.files) {
    lines.push(`[${file.status.toUpperCase()}] ${file.originalName}`);
    lines.push(`  Base: ${file.sanitizedBase ?? "n/a"}`);
    lines.push(`  Message: ${file.message}`);
    lines.push(`  Duration: ${formatSeconds(file.durationSeconds)}`);
    lines.push(`  Format: ${file.sampleRate ?? "unknown"} Hz / ${file.channels ?? "unknown"} ch`);
    if (file.outputs.length > 0) {
      lines.push("  Outputs:");
      for (const output of file.outputs) lines.push(`    - ${output}`);
    }
    if (file.stems.length > 0) {
      lines.push("  Stem QC:");
      for (const stem of file.stems) {
        lines.push(
          `    - ${stem.fileName}: peak ${formatDbfs(stem.peakDbfs)}, RMS ${formatDbfs(stem.rmsDbfs)}, clipped ${stem.clippedSampleCount} (${stem.clippedSamplePct.toFixed(5)}%)`,
        );
      }
    }
    if (file.warnings.length > 0) {
      lines.push("  Warnings:");
      for (const warning of file.warnings) lines.push(`    - ${warning}`);
    }
    lines.push("");
  }

  return lines.join("\n");
};

const createZipBundle = async (outputs: AudioSplitterOutput[], report: AudioSplitterReport) => {
  const zip = new JSZip();
  const usedPaths = new Set<string>();

  for (const output of outputs) {
    if (usedPaths.has(output.fileName)) {
      throw new Error(`Duplicate ZIP output path blocked: ${output.fileName}`);
    }
    usedPaths.add(output.fileName);
    zip.file(output.fileName, await readFile(output.path));
  }

  zip.file("split_report.txt", renderTextReport(report));
  zip.file("split_report.json", JSON.stringify(report, null, 2));

  const zipped = await zip.generateAsync({
    type: "uint8array",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
  return Buffer.from(zipped);
};

const buildFailedDuplicateReport = (
  input: AudioSplitterInput,
  inputIndex: number,
  engineName: string,
  sanitizedBase: string,
): AudioSplitterFileReport => ({
  inputIndex,
  originalName: input.originalName,
  sanitizedBase,
  status: "failed",
  message:
    "Duplicate output filenames after sanitization. Rename this source file so ZIP outputs do not overwrite another file.",
  outputs: [],
  engine: engineName,
  sampleRate: null,
  channels: null,
  durationSeconds: null,
  stems: [],
  warnings: [],
});

const buildFailedInputReport = (
  input: AudioSplitterInput,
  inputIndex: number,
  engineName: string,
  sanitizedBase: string,
  message: string,
): AudioSplitterFileReport => ({
  inputIndex,
  originalName: input.originalName,
  sanitizedBase,
  status: "failed",
  message,
  outputs: [],
  engine: engineName,
  sampleRate: null,
  channels: null,
  durationSeconds: null,
  stems: [],
  warnings: [],
});

const buildReportAndZipResult = async (
  inputs: AudioSplitterInput[],
  engineName: string,
  generatedAt: string,
  outputEntries: AudioSplitterOutput[],
  fileReports: AudioSplitterFileReport[],
  onProgress?: (event: AudioSplitterProgressEvent) => void,
) => {
  const report: AudioSplitterReport = {
    generatedAt,
    engine: engineName,
    totalFiles: inputs.length,
    succeeded: fileReports.filter((file) => file.status === "success").length,
    failed: fileReports.filter((file) => file.status === "failed").length,
    files: fileReports.sort((a, b) => a.inputIndex - b.inputIndex),
  };
  const zip = await createZipBundle(outputEntries, report);
  onProgress?.({ type: "batch-complete", report });
  const stamp = generatedAt.replace(/[:.]/g, "-");
  return {
    zip,
    zipName: `audio_track_splitter_${stamp}.zip`,
    report,
  };
};

type PreparedBatchInput = {
  input: AudioSplitterInput;
  inputInfo: WavInfo;
  reportBase: AudioSplitterFileReport;
  outputNames: ReturnType<typeof buildAudioSplitterOutputFileNames>;
  outputDir: string;
  engineItem: AudioSplitterEngineBatchItem;
};

const splitBatchAudioTracksWithPerFileEngine = async (
  inputs: AudioSplitterInput[],
  engine: AudioSplitterEngine,
  generatedAt: string,
  tempRoot: string,
  targetBitDepth: OutputBitDepth,
  options: SplitBatchOptions,
) => {
  const outputEntries: AudioSplitterOutput[] = [];
  const fileReports: AudioSplitterFileReport[] = [];
  const reservedZipNames = new Set<string>();

  for (let index = 0; index < inputs.length; index += 1) {
    const input = inputs[index];
    const sanitizedBase = sanitizeAudioBaseName(input.originalName, index);
    const expectedNames = Object.values(buildAudioSplitterOutputFileNames(sanitizedBase));
    const hasDuplicate = expectedNames.some((name) => reservedZipNames.has(name));
    if (hasDuplicate) {
      const duplicateReport = buildFailedDuplicateReport(input, index, engine.name, sanitizedBase);
      fileReports.push(duplicateReport);
      options.onProgress?.({ type: "file-complete", inputIndex: index, report: duplicateReport });
      continue;
    }
    for (const name of expectedNames) reservedZipNames.add(name);

    options.onProgress?.({
      type: "file-start",
      inputIndex: index,
      originalName: input.originalName,
      sanitizedBase,
      message: `Running ${engine.name}`,
    });
    const result = await splitAudioTrack(input, index, tempRoot, engine, targetBitDepth);
    outputEntries.push(...result.outputs);
    fileReports.push(result.report);
    options.onProgress?.({ type: "file-complete", inputIndex: index, report: result.report });
  }

  return await buildReportAndZipResult(inputs, engine.name, generatedAt, outputEntries, fileReports, options.onProgress);
};

const splitBatchAudioTracksWithBatchEngine = async (
  inputs: AudioSplitterInput[],
  engine: AudioSplitterEngine,
  generatedAt: string,
  tempRoot: string,
  targetBitDepth: OutputBitDepth,
  options: SplitBatchOptions,
) => {
  if (!engine.splitBatch) {
    throw new Error("Batch engine is not configured.");
  }

  const outputEntries: AudioSplitterOutput[] = [];
  const fileReports: AudioSplitterFileReport[] = [];
  const preparedInputs: PreparedBatchInput[] = [];
  const reservedZipNames = new Set<string>();

  for (let index = 0; index < inputs.length; index += 1) {
    const input = inputs[index];
    const sanitizedBase = sanitizeAudioBaseName(input.originalName, index);
    const outputNames = buildAudioSplitterOutputFileNames(sanitizedBase);
    const expectedNames = Object.values(outputNames);
    const hasDuplicate = expectedNames.some((name) => reservedZipNames.has(name));
    if (hasDuplicate) {
      const duplicateReport = buildFailedDuplicateReport(input, index, engine.name, sanitizedBase);
      fileReports.push(duplicateReport);
      options.onProgress?.({ type: "file-complete", inputIndex: index, report: duplicateReport });
      continue;
    }
    for (const name of expectedNames) reservedZipNames.add(name);

    const workDir = path.join(tempRoot, `input_${index + 1}`);
    const sourceDir = path.join(workDir, "source");
    const engineDir = path.join(workDir, "engine");
    const outputDir = path.join(workDir, "outputs");
    const reportBase: AudioSplitterFileReport = {
      inputIndex: index,
      originalName: input.originalName,
      sanitizedBase,
      status: "failed",
      message: "",
      outputs: [],
      engine: engine.name,
      sampleRate: null,
      channels: null,
      durationSeconds: null,
      stems: [],
      warnings: [],
    };

    try {
      if (!input.originalName.toLowerCase().endsWith(".wav")) {
        throw new Error("Unsupported file type. Audio Track Splitter accepts .wav files only.");
      }
      const inputInfo = readWavInfoFromBytes(input.bytes);
      assertSupportedInputWav(inputInfo);
      await mkdir(sourceDir, { recursive: true });
      await mkdir(engineDir, { recursive: true });
      await mkdir(outputDir, { recursive: true });
      const inputPath = path.join(sourceDir, "input.wav");
      await writeFile(inputPath, Buffer.from(input.bytes));

      preparedInputs.push({
        input,
        inputInfo,
        reportBase,
        outputNames,
        outputDir,
        engineItem: {
          inputIndex: index,
          originalName: input.originalName,
          sanitizedBase,
          inputPath,
          workDir: engineDir,
          sampleRate: inputInfo.sampleRate,
          channels: inputInfo.channels,
          durationSeconds: inputInfo.durationSeconds,
        },
      });
    } catch (error) {
      const failedReport = buildFailedInputReport(
        input,
        index,
        engine.name,
        sanitizedBase,
        error instanceof Error ? error.message : String(error),
      );
      fileReports.push(failedReport);
      options.onProgress?.({ type: "file-complete", inputIndex: index, report: failedReport });
    }
  }

  if (preparedInputs.length > 0) {
    const firstPrepared = preparedInputs[0];
    options.onProgress?.({
      type: "file-progress",
      inputIndex: firstPrepared.engineItem.inputIndex,
      originalName: firstPrepared.input.originalName,
      message: `Starting ${engine.name}`,
    });
    const batchResults = await engine.splitBatch(
      preparedInputs.map((prepared) => prepared.engineItem),
      path.join(tempRoot, "batch-engine"),
      {
        onProgress(inputIndex, message) {
          const prepared = preparedInputs.find((item) => item.engineItem.inputIndex === inputIndex);
          options.onProgress?.({
            type: "file-progress",
            inputIndex,
            originalName: prepared?.input.originalName ?? `input ${inputIndex + 1}`,
            message,
          });
        },
      },
    );
    const resultsByIndex = new Map(batchResults.map((result) => [result.inputIndex, result]));

    for (const prepared of preparedInputs) {
      const result = resultsByIndex.get(prepared.engineItem.inputIndex);
      try {
        if (!result) throw new Error("Separation engine did not return a result for this file.");
        if (result.error) throw new Error(result.error);
        if (!result.rawStems) throw new Error("Separation engine did not return stem paths.");
        const finalized = await finalizeRawStemOutputs(
          result.rawStems,
          prepared.inputInfo,
          prepared.reportBase,
          prepared.outputDir,
          prepared.outputNames,
          targetBitDepth,
        );
        outputEntries.push(...finalized.outputs);
        fileReports.push(finalized.report);
        options.onProgress?.({
          type: "file-complete",
          inputIndex: prepared.engineItem.inputIndex,
          report: finalized.report,
        });
      } catch (error) {
        const failedReport = buildFailedInputReport(
          prepared.input,
          prepared.engineItem.inputIndex,
          engine.name,
          prepared.engineItem.sanitizedBase,
          error instanceof Error ? error.message : String(error),
        );
        fileReports.push(failedReport);
        options.onProgress?.({
          type: "file-complete",
          inputIndex: prepared.engineItem.inputIndex,
          report: failedReport,
        });
      }
    }
  }

  return await buildReportAndZipResult(inputs, engine.name, generatedAt, outputEntries, fileReports, options.onProgress);
};

export const splitBatchAudioTracks = async (
  inputs: AudioSplitterInput[],
  options: SplitBatchOptions = {},
): Promise<SplitBatchAudioTracksResult> => {
  const engine = options.engine ?? createDefaultAudioSplitterEngine();
  const generatedAt = (options.now ?? new Date()).toISOString();
  const tempRoot = await mkdtemp(path.join(tmpdir(), "audio-splitter-"));
  const targetBitDepth = parseOutputBitDepth();

  try {
    if (engine.splitBatch) {
      return await splitBatchAudioTracksWithBatchEngine(inputs, engine, generatedAt, tempRoot, targetBitDepth, options);
    }
    return await splitBatchAudioTracksWithPerFileEngine(inputs, engine, generatedAt, tempRoot, targetBitDepth, options);
  } finally {
    if (options.cleanup !== false) {
      await rm(tempRoot, { recursive: true, force: true });
    }
  }
};

const splitCommandLine = (commandLine: string) => {
  const parts: string[] = [];
  let current = "";
  let quote: "\"" | "'" | null = null;

  for (const char of commandLine.trim()) {
    if (quote) {
      if (char === quote) quote = null;
      else current += char;
      continue;
    }
    if (char === "\"" || char === "'") {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) {
        parts.push(current);
        current = "";
      }
      continue;
    }
    current += char;
  }
  if (current) parts.push(current);
  return parts;
};

const resolveProjectRelativePathOption = (value: string | undefined) => {
  if (!value) return undefined;
  if (path.isAbsolute(value)) return value;
  if (value.startsWith(".") || value.includes("/") || value.includes("\\")) {
    return path.resolve(process.cwd(), value);
  }
  return value;
};

const runProcess = async (
  command: string,
  args: string[],
  options: {
    cwd: string;
    timeoutMs: number;
    env?: Record<string, string | undefined>;
    onLine?: (line: string) => void;
  },
) =>
  new Promise<void>((resolve, reject) => {
    const startedAt = Date.now();
    const mirrorOutput =
      process.env.AUDIO_SPLITTER_LOG_ENGINE === "1" ||
      (process.env.AUDIO_SPLITTER_LOG_ENGINE !== "0" && process.env.NODE_ENV !== "production");
    if (mirrorOutput) {
      console.info(`[AudioSplitter] Starting engine: ${command} ${args.join(" ")}`);
    }
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: { ...process.env, ...options.env },
      windowsHide: true,
    });
    let settled = false;
    let lineError: Error | null = null;
    let logTail = "";
    const lineRemainders: Record<"stdout" | "stderr", string> = { stdout: "", stderr: "" };
    const appendLog = (chunk: Buffer, streamName: "stdout" | "stderr") => {
      const text = chunk.toString("utf8");
      logTail = `${logTail}${text}`.slice(-12_000);
      if (options.onLine) {
        const combined = `${lineRemainders[streamName]}${text}`;
        const lines = combined.split(/\r?\n/);
        lineRemainders[streamName] = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            options.onLine(line);
          } catch (error) {
            lineError = error instanceof Error ? error : new Error(String(error));
            child.kill("SIGTERM");
            break;
          }
        }
      }
      if (mirrorOutput) process.stderr.write(chunk);
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill("SIGTERM");
      reject(new Error(`Separation engine timed out after ${Math.round(options.timeoutMs / 1000)}s.`));
    }, options.timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => appendLog(chunk, "stdout"));
    child.stderr.on("data", (chunk: Buffer) => appendLog(chunk, "stderr"));
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(
        new Error(
          `Unable to start separation engine "${command}". Check the configured splitter command and Python environment. ${error.message}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      for (const remainder of Object.values(lineRemainders)) {
        if (!remainder.trim() || lineError) continue;
        try {
          options.onLine?.(remainder);
        } catch (error) {
          lineError = error instanceof Error ? error : new Error(String(error));
        }
      }
      if (lineError) {
        reject(lineError);
        return;
      }
      if (code === 0) {
        if (mirrorOutput) {
          const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(1);
          console.info(`[AudioSplitter] Engine completed in ${elapsedSeconds}s.`);
        }
        resolve();
        return;
      }
      reject(new Error(`Separation engine exited with code ${code}. ${logTail.trim()}`.trim()));
    });
  });

const findStemFile = (rootDir: string, stemName: string): string | null => {
  const wanted = `${stemName.toLowerCase()}.wav`;
  const stack = [rootDir];
  while (stack.length > 0) {
    const current = stack.pop() as string;
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(entryPath);
      } else if (entry.isFile() && entry.name.toLowerCase() === wanted) {
        return entryPath;
      }
    }
  }
  return null;
};

const requireStemFile = (rootDir: string, stemName: string) => {
  const found = findStemFile(rootDir, stemName);
  if (!found) throw new Error(`Demucs did not produce ${stemName}.wav.`);
  return found;
};

const normalizePositiveIntegerOption = (value: string | undefined, fallback: string) => {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return String(Math.max(1, Math.round(parsed)));
};

const normalizePositiveNumberOption = (value: string | undefined, fallback: string) => {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return String(parsed);
};

const WORKER_EVENT_PREFIX = "AUDIO_SPLITTER_EVENT ";
const DEFAULT_AUDIO_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt";

const createAudioSeparatorEngine = (): AudioSplitterEngine => {
  const commandLine =
    process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND ??
    process.env.AUDIO_SPLITTER_PYTHON_COMMAND ??
    "python";
  const [rawCommand, ...commandPrefixArgs] = splitCommandLine(commandLine);
  const command = resolveProjectRelativePathOption(rawCommand);
  const workerPath = path.resolve(
    process.cwd(),
    process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_WORKER ?? path.join("scripts", "audio_separator_worker.py"),
  );
  const model = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL ?? DEFAULT_AUDIO_SEPARATOR_MODEL;
  const modelDir = resolveProjectRelativePathOption(process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL_DIR);
  const device = process.env.AUDIO_SPLITTER_DEVICE ?? process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_DEVICE ?? "cpu";
  const outputFormat = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_OUTPUT_FORMAT ?? "WAV";
  const sampleRate = normalizePositiveIntegerOption(process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_SAMPLE_RATE, "44100");
  const normalization = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_NORMALIZATION ?? "0.98";
  const amplification = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_AMPLIFICATION ?? "0.0";
  const chunkDuration = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_CHUNK_DURATION;
  const mdxcSegmentSize = normalizePositiveIntegerOption(
    process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MDXC_SEGMENT_SIZE,
    "256",
  );
  const mdxcOverlap = normalizePositiveNumberOption(process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MDXC_OVERLAP, "8");
  const mdxcBatchSize = normalizePositiveIntegerOption(
    process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MDXC_BATCH_SIZE,
    "1",
  );
  const useSoundFile = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_SOUNDFILE ?? "1";
  const useAutocast = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_AUTOCAST ?? "0";
  const timeoutMs = Number(process.env.AUDIO_SPLITTER_TIMEOUT_MS ?? DEFAULT_DEMUCS_TIMEOUT_MS);

  const runBatch = async (
    items: AudioSplitterEngineBatchItem[],
    workDir: string,
    context: { onProgress?: (inputIndex: number, message: string) => void },
  ): Promise<AudioSplitterEngineBatchResult[]> => {
    if (!command) throw new Error("AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND is empty.");
    if (!existsSync(workerPath)) throw new Error(`Audio separator worker was not found at ${workerPath}.`);
    await mkdir(workDir, { recursive: true });
    if (modelDir) await mkdir(modelDir, { recursive: true });
    const manifestPath = path.join(workDir, "manifest.json");
    await writeFile(
      manifestPath,
      JSON.stringify(
        {
          items: items.map((item) => ({
            inputIndex: item.inputIndex,
            originalName: item.originalName,
            inputPath: item.inputPath,
            workDir: item.workDir,
            sampleRate: item.sampleRate,
          })),
        },
        null,
        2,
      ),
    );

    const results = new Map<number, AudioSplitterEngineBatchResult>();
    const args = [
      ...commandPrefixArgs,
      workerPath,
      "--manifest",
      manifestPath,
      "--model",
      model,
      "--device",
      device,
      "--output-format",
      outputFormat,
      "--sample-rate",
      sampleRate,
      "--normalization",
      normalization,
      "--amplification",
      amplification,
      "--mdxc-segment-size",
      mdxcSegmentSize,
      "--mdxc-overlap",
      mdxcOverlap,
      "--mdxc-batch-size",
      mdxcBatchSize,
    ];
    if (modelDir) args.push("--model-dir", modelDir);
    if (chunkDuration) args.push("--chunk-duration", normalizePositiveNumberOption(chunkDuration, "30"));
    if (useSoundFile === "1") args.push("--use-soundfile");
    if (useAutocast === "1") args.push("--use-autocast");

    await runProcess(command, args, {
      cwd: workDir,
      timeoutMs: Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_DEMUCS_TIMEOUT_MS,
      env: modelDir ? { AUDIO_SEPARATOR_MODEL_DIR: modelDir } : undefined,
      onLine(line) {
        if (!line.startsWith(WORKER_EVENT_PREFIX)) return;
        const event = JSON.parse(line.slice(WORKER_EVENT_PREFIX.length)) as
          | { type: "file-progress"; inputIndex: number; message: string }
          | { type: "file-complete"; inputIndex: number; vocal: string; bgm: string }
          | { type: "file-error"; inputIndex: number; message: string };

        if (event.type === "file-progress") {
          context.onProgress?.(event.inputIndex, event.message);
        } else if (event.type === "file-complete") {
          results.set(event.inputIndex, {
            inputIndex: event.inputIndex,
            rawStems: {
              vocal: event.vocal,
              bgm: event.bgm,
            },
          });
        } else {
          results.set(event.inputIndex, {
            inputIndex: event.inputIndex,
            error: event.message,
          });
        }
      },
    });

    return items.map(
      (item) =>
        results.get(item.inputIndex) ?? {
          inputIndex: item.inputIndex,
          error: "Audio separator worker exited without returning this file.",
        },
    );
  };

  return {
    name: `audio-separator:${model}`,
    async split(inputPath, workDir, context) {
      const inputInfo = readWavInfoFromFile(inputPath);
      const [result] = await runBatch(
        [
          {
            inputIndex: 0,
            originalName: context.originalName,
            sanitizedBase: context.sanitizedBase,
            inputPath,
            workDir,
            sampleRate: inputInfo.sampleRate,
            channels: inputInfo.channels,
            durationSeconds: inputInfo.durationSeconds,
          },
        ],
        path.join(workDir, "batch"),
        {},
      );
      if (result?.error) throw new Error(result.error);
      if (!result?.rawStems) throw new Error("Audio separator worker did not return stem paths.");
      return result.rawStems;
    },
    splitBatch: runBatch,
  };
};

const createDemucsEngine = (): AudioSplitterEngine => {
  const commandLine = process.env.AUDIO_SPLITTER_DEMUCS_COMMAND ?? "demucs";
  const [rawCommand, ...commandPrefixArgs] = splitCommandLine(commandLine);
  const command = resolveProjectRelativePathOption(rawCommand);
  const model = process.env.AUDIO_SPLITTER_DEMUCS_MODEL ?? "htdemucs";
  const device = process.env.AUDIO_SPLITTER_DEMUCS_DEVICE;
  const segment = normalizePositiveIntegerOption(process.env.AUDIO_SPLITTER_DEMUCS_SEGMENT, "7");
  const shifts = process.env.AUDIO_SPLITTER_DEMUCS_SHIFTS ?? "1";
  const overlap = process.env.AUDIO_SPLITTER_DEMUCS_OVERLAP;
  const jobs = process.env.AUDIO_SPLITTER_DEMUCS_JOBS;
  const clipMode = process.env.AUDIO_SPLITTER_DEMUCS_CLIP_MODE ?? "rescale";
  const targetBitDepth = parseOutputBitDepth();
  const timeoutMs = Number(process.env.AUDIO_SPLITTER_TIMEOUT_MS ?? DEFAULT_DEMUCS_TIMEOUT_MS);

  return {
    name: `demucs:${model}`,
    async split(inputPath, workDir) {
      if (!command) throw new Error("AUDIO_SPLITTER_DEMUCS_COMMAND is empty.");
      mkdirSync(workDir, { recursive: true });
      const args = [
        ...commandPrefixArgs,
        "-n",
        model,
        "-o",
        workDir,
        "--filename",
        "{stem}.{ext}",
        "--two-stems",
        "vocals",
        "--clip-mode",
        clipMode,
      ];
      if (targetBitDepth === 24) args.push("--int24");
      if (targetBitDepth === 32) args.push("--float32");
      if (device) args.push("-d", device);
      if (segment) args.push("--segment", segment);
      if (shifts) args.push("--shifts", shifts);
      if (overlap) args.push("--overlap", overlap);
      if (jobs) args.push("-j", jobs);
      args.push("--", inputPath);

      await runProcess(command, args, {
        cwd: workDir,
        timeoutMs: Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_DEMUCS_TIMEOUT_MS,
        env: { PYTORCH_NO_CUDA_MEMORY_CACHING: "1" },
      });

      return {
        vocal: requireStemFile(workDir, "vocals"),
        nonVocal: requireStemFile(workDir, "no_vocals"),
      };
    },
  };
};

export const createDefaultAudioSplitterEngine = (): AudioSplitterEngine => {
  const engine = (process.env.AUDIO_SPLITTER_ENGINE ?? "audio-separator").toLowerCase();
  if (engine === "demucs") return createDemucsEngine();
  if (engine === "audio-separator" || engine === "audio_separator" || engine === "roformer") {
    return createAudioSeparatorEngine();
  }
  throw new Error(`Unsupported AUDIO_SPLITTER_ENGINE "${engine}". Use "audio-separator" or "demucs".`);
};
