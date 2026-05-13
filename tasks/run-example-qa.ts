import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { analyzeFloatSamples, buildSpeechMask, clamp, median, percentile } from "../src/lib/audioQc.ts";
import {
  applyGainCurveToSamples,
  planGainCurve,
  speechRunsFromMask,
  type SpeechRun,
} from "../src/lib/gainPlanner.ts";
import { readWavInfoFromBytes, splitBatchAudioTracks } from "../src/lib/audioSplitterService.ts";
import { decodeWav } from "../src/lib/webAudioRender.ts";

const FRAME_MS = 10;
const PLANNER_SAMPLE_RATE = 16000;
const VO_EXAMPLE_DIR = "VO example";
const SPLIT_EXAMPLE_DIR = "Audio split example";
const REPORT_JSON = path.join("tasks", "example-qa-report.json");
const REPORT_MD = path.join("tasks", "example-qa-report.md");

type VoQaRow = {
  name: string;
  durationSeconds: number;
  sampleRate: number;
  channels: number;
  speechRuns: number;
  plannerMaxGainDb: number;
  earlyRunCapCount: number;
  earlyRunMaxReductionDb: number;
  inputOverallRisk: number;
  outputOverallRisk: number;
  inputSentenceJumpScore: number;
  outputSentenceJumpScore: number;
  inputPauseNoiseRisk: number;
  outputPauseNoiseRisk: number;
  inputRunSigmaDb: number | null;
  outputRunSigmaDb: number | null;
  inputOpenerOverLaterDb: number | null;
  outputOpenerOverLaterDb: number | null;
  outputPeakDb: number | null;
  outputClipPct: number | null;
};

type SplitPreflightRow = {
  name: string;
  sampleRate: number;
  channels: number;
  durationSeconds: number;
  bitsPerSample: number;
};

const toDb = (value: number) => (value <= 0 ? -120 : 20 * Math.log10(value));
const round = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined || !Number.isFinite(value) ? null : Number(value.toFixed(digits));

const listWavFiles = async (dir: string) =>
  (await readdir(dir))
    .filter((name) => name.toLowerCase().endsWith(".wav"))
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" }));

const monoResample = (samples: Float32Array, sourceRate: number, channels: number, targetRate: number) => {
  const sourceFrames = Math.floor(samples.length / channels);
  const targetFrames = Math.max(1, Math.floor((sourceFrames * targetRate) / sourceRate));
  const out = new Float32Array(targetFrames);
  const rateRatio = sourceRate / targetRate;

  for (let frame = 0; frame < targetFrames; frame += 1) {
    const sourcePos = frame * rateRatio;
    const i0 = Math.min(sourceFrames - 1, Math.floor(sourcePos));
    const i1 = Math.min(sourceFrames - 1, i0 + 1);
    const mix = sourcePos - i0;
    let v0 = 0;
    let v1 = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      v0 += samples[i0 * channels + channel] ?? 0;
      v1 += samples[i1 * channels + channel] ?? 0;
    }
    out[frame] = ((v0 * (1 - mix) + v1 * mix) / channels);
  }

  return out;
};

const frameDbForSamples = (samples: Float32Array, sampleRate: number, frameMs = FRAME_MS) => {
  const frameSamples = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const frameCount = Math.floor(samples.length / frameSamples);
  const frameDb = new Array<number>(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * frameSamples;
    let sum = 0;
    for (let index = 0; index < frameSamples; index += 1) {
      const value = samples[start + index] ?? 0;
      sum += value * value;
    }
    frameDb[frame] = toDb(Math.sqrt(sum / frameSamples));
  }
  return frameDb;
};

const meanRunDb = (frameDb: number[], run: SpeechRun) => {
  const start = Math.max(0, run.startFrame);
  const end = Math.min(frameDb.length, run.endFrame);
  let sumPower = 0;
  for (let frame = start; frame < end; frame += 1) {
    sumPower += Math.pow(10, frameDb[frame] / 10);
  }
  return 10 * Math.log10(sumPower / Math.max(1, end - start) + 1e-30);
};

const standardDeviation = (values: number[]) => {
  if (values.length === 0) return null;
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
};

const speechRunStats = (frameDb: number[], speechRuns: SpeechRun[]) => {
  const runDb = speechRuns
    .filter((run) => run.endFrame - run.startFrame >= 6)
    .map((run) => meanRunDb(frameDb, run))
    .filter((value) => Number.isFinite(value));
  const sigmaDb = standardDeviation(runDb);
  let openerOverLaterDb: number | null = null;
  if (runDb.length >= 5) {
    const early = median(runDb.slice(0, Math.min(3, runDb.length - 2)));
    const later = median(runDb.slice(Math.min(3, runDb.length - 2)));
    if (early !== null && later !== null) openerOverLaterDb = early - later;
  }
  return { sigmaDb, openerOverLaterDb };
};

const runVoQa = async (): Promise<VoQaRow[]> => {
  const names = await listWavFiles(VO_EXAMPLE_DIR);
  const rows: VoQaRow[] = [];

  for (const name of names) {
    console.log(`[VO] Analyzing ${name}`);
    const bytes = await readFile(path.join(VO_EXAMPLE_DIR, name));
    const decoded = decodeWav(bytes);
    const mono = monoResample(decoded.samples, decoded.sampleRate, decoded.channels, PLANNER_SAMPLE_RATE);
    const inputMetrics = analyzeFloatSamples(mono, PLANNER_SAMPLE_RATE, FRAME_MS);
    const inputFrameDb = frameDbForSamples(mono, PLANNER_SAMPLE_RATE, FRAME_MS);
    const noiseFloorDb = inputMetrics.noiseFloorDb ?? inputMetrics.pauseNoiseFloorDb ?? -70;
    const speechThresholdDb = inputMetrics.speechThresholdDb ?? noiseFloorDb + 11;
    const instabilityHint = clamp(
      inputMetrics.instabilityScore * 0.5 +
        inputMetrics.lineSwingScore * 0.3 +
        inputMetrics.sentenceJumpScore * 0.2,
      0,
      1,
    );
    const speechSpikeTaming = clamp(
      instabilityHint * 0.35 +
        inputMetrics.lineSwingScore * 0.35 +
        inputMetrics.onsetOvershootScore * 0.18 +
        inputMetrics.clickScore * 0.12,
      0,
      1,
    );
    const cleanBoostHeadroom = clamp(
      clamp((inputMetrics.noiseContrastDb - 26) / 14, 0, 1) *
        clamp((-58 - inputMetrics.noiseFloorDb) / 18, 0, 1) *
        clamp((0.28 - inputMetrics.pauseNoiseRisk) / 0.28, 0, 1),
      0,
      1,
    );
    const sparseSpeech = inputMetrics.speechDutyCyclePct < 10 || inputMetrics.speechSegmentCount <= 6;
    const plannerMaxGainDb = 14 + cleanBoostHeadroom * (sparseSpeech ? 4 : 2);
    const speechRuns = speechRunsFromMask(buildSpeechMask(inputFrameDb, noiseFloorDb, { frameMs: FRAME_MS }));

    const plan = planGainCurve({
      frameDb: inputFrameDb,
      speechRuns,
      noiseFloorDb,
      speechThresholdDb,
      pauseNoiseRisk: inputMetrics.pauseNoiseRisk,
      frameMs: FRAME_MS,
      samples: mono,
      sampleRate: PLANNER_SAMPLE_RATE,
      targetDb: -22,
      sourceTargetBlend: 0.1,
      maxGainDb: plannerMaxGainDb,
      peakCeilingDb: -3,
      instabilityHint,
      speechSpikeTaming,
    });
    const planned = applyGainCurveToSamples(mono, plan.gainCurve, PLANNER_SAMPLE_RATE, 1, FRAME_MS);
    const outputMetrics = analyzeFloatSamples(planned, PLANNER_SAMPLE_RATE, FRAME_MS);
    const outputFrameDb = frameDbForSamples(planned, PLANNER_SAMPLE_RATE, FRAME_MS);
    const dialogueRuns = plan.runs
      .filter((run) => run.runClass === "body-speech")
      .map((run) => ({ startFrame: run.startFrame, endFrame: run.endFrame }));
    const inputStats = speechRunStats(inputFrameDb, dialogueRuns);
    const outputStats = speechRunStats(outputFrameDb, dialogueRuns);

    rows.push({
      name,
      durationSeconds: round(decoded.samples.length / Math.max(decoded.sampleRate * decoded.channels, 1), 3) ?? 0,
      sampleRate: decoded.sampleRate,
      channels: decoded.channels,
      speechRuns: plan.runs.length,
      plannerMaxGainDb: round(plannerMaxGainDb, 2) ?? plannerMaxGainDb,
      earlyRunCapCount: plan.earlyRunCapCount,
      earlyRunMaxReductionDb: round(plan.earlyRunMaxReductionDb, 2) ?? 0,
      inputOverallRisk: round(inputMetrics.overallRisk, 3) ?? 0,
      outputOverallRisk: round(outputMetrics.overallRisk, 3) ?? 0,
      inputSentenceJumpScore: round(inputMetrics.sentenceJumpScore, 3) ?? 0,
      outputSentenceJumpScore: round(outputMetrics.sentenceJumpScore, 3) ?? 0,
      inputPauseNoiseRisk: round(inputMetrics.pauseNoiseRisk, 3) ?? 0,
      outputPauseNoiseRisk: round(outputMetrics.pauseNoiseRisk, 3) ?? 0,
      inputRunSigmaDb: round(inputStats.sigmaDb, 3),
      outputRunSigmaDb: round(outputStats.sigmaDb, 3),
      inputOpenerOverLaterDb: round(inputStats.openerOverLaterDb, 3),
      outputOpenerOverLaterDb: round(outputStats.openerOverLaterDb, 3),
      outputPeakDb: round(outputMetrics.peakDb, 2),
      outputClipPct: round(outputMetrics.clipPct, 5),
    });
  }

  return rows;
};

const configureCpuSplitterEnv = () => {
  process.env.AUDIO_SPLITTER_ENGINE ??= "audio-separator";
  process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND ??= path.join(
    ".venv-audio-splitter",
    "Scripts",
    "python.exe",
  );
  process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL_DIR ??= ".audio-separator-models";
  process.env.AUDIO_SPLITTER_DEVICE ??= "cpu";
  process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_AUTOCAST ??= "0";
  process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_SOUNDFILE ??= "1";
  process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MDXC_BATCH_SIZE ??= "1";
  process.env.AUDIO_SPLITTER_TIMEOUT_MS ??= String(4 * 60 * 60 * 1000);
};

const runSplitQa = async () => {
  configureCpuSplitterEnv();
  const names = await listWavFiles(SPLIT_EXAMPLE_DIR);
  const preflight: SplitPreflightRow[] = [];
  const inputs = [];

  for (const name of names) {
    const bytes = await readFile(path.join(SPLIT_EXAMPLE_DIR, name));
    const info = readWavInfoFromBytes(bytes);
    preflight.push({
      name,
      sampleRate: info.sampleRate,
      channels: info.channels,
      durationSeconds: round(info.durationSeconds, 3) ?? info.durationSeconds,
      bitsPerSample: info.bitsPerSample,
    });
    inputs.push({ originalName: name, bytes });
  }

  console.log(`[Split] Processing ${inputs.length} files on ${process.env.AUDIO_SPLITTER_DEVICE}`);
  const startedAt = Date.now();
  const result = await splitBatchAudioTracks(inputs, {
    cleanup: true,
    onProgress(event) {
      if (event.type === "file-progress") {
        console.log(`[Split] ${event.originalName}: ${event.message}`);
      } else if (event.type === "file-complete") {
        console.log(`[Split] ${event.report.originalName}: ${event.report.status}`);
      }
    },
  });

  return {
    preflight,
    zipName: result.zipName,
    zipSizeBytes: result.zip.byteLength,
    elapsedSeconds: round((Date.now() - startedAt) / 1000, 1),
    report: result.report,
  };
};

const summarizeVoRows = (rows: VoQaRow[]) => {
  const improvedSigma = rows.filter(
    (row) => row.inputRunSigmaDb !== null && row.outputRunSigmaDb !== null && row.outputRunSigmaDb < row.inputRunSigmaDb,
  ).length;
  const avg = (values: number[]) =>
    values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    files: rows.length,
    improvedRunSigmaFiles: improvedSigma,
    medianOutputRunSigmaDb: round(percentile(rows.flatMap((row) => row.outputRunSigmaDb ?? []), 50), 3),
    averageRiskDelta: round(avg(rows.map((row) => row.outputOverallRisk - row.inputOverallRisk)), 3),
    maxOutputClipPct: round(Math.max(...rows.map((row) => row.outputClipPct ?? 0)), 5),
  };
};

const renderMarkdown = (report: Awaited<ReturnType<typeof buildReport>>) => {
  const lines = [
    "# Example QA Report",
    "",
    `Generated: ${report.generatedAt}`,
    "",
    "## VO examples",
    "",
    `Files: ${report.voSummary.files}`,
    `Improved run-level sigma: ${report.voSummary.improvedRunSigmaFiles}/${report.voSummary.files}`,
    `Median output run sigma: ${report.voSummary.medianOutputRunSigmaDb ?? "n/a"} dB`,
    `Average overall-risk delta: ${report.voSummary.averageRiskDelta ?? "n/a"}`,
    `Max output clip percent: ${report.voSummary.maxOutputClipPct ?? "n/a"}`,
    "",
    "| File | Runs | Sigma in | Sigma out | Sentence jump in | Sentence jump out | Opener out | Peak out |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const row of report.voRows) {
    lines.push(
      `| ${row.name} | ${row.speechRuns} | ${row.inputRunSigmaDb ?? "n/a"} | ${row.outputRunSigmaDb ?? "n/a"} | ${row.inputSentenceJumpScore} | ${row.outputSentenceJumpScore} | ${row.outputOpenerOverLaterDb ?? "n/a"} | ${row.outputPeakDb ?? "n/a"} |`,
    );
  }

  if (report.split) {
    lines.push(
      "",
      "## Audio split examples",
      "",
      `Files: ${report.split.report.succeeded}/${report.split.report.totalFiles} succeeded`,
      `Elapsed: ${report.split.elapsedSeconds}s`,
      `ZIP size: ${report.split.zipSizeBytes} bytes`,
      "",
      "| File | Status | BGM peak | BGM RMS | Vocal peak | Vocal RMS | Warnings |",
      "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    );
    for (const file of report.split.report.files) {
      const bgm = file.stems.find((stem) => stem.stem === "BGM");
      const vocal = file.stems.find((stem) => stem.stem === "VOCAL");
      lines.push(
        `| ${file.originalName} | ${file.status} | ${round(bgm?.peakDbfs, 2) ?? "n/a"} | ${round(bgm?.rmsDbfs, 2) ?? "n/a"} | ${round(vocal?.peakDbfs, 2) ?? "n/a"} | ${round(vocal?.rmsDbfs, 2) ?? "n/a"} | ${file.warnings.join("; ") || ""} |`,
      );
    }
  }

  return `${lines.join("\n")}\n`;
};

const buildReport = async () => {
  const args = new Set(process.argv.slice(2));
  const splitOnly = args.has("--split-only");
  const voOnly = args.has("--vo-only");
  const existing = await readFile(REPORT_JSON, "utf8")
    .then((text) => JSON.parse(text) as { voRows?: VoQaRow[]; split?: Awaited<ReturnType<typeof runSplitQa>> | null })
    .catch(() => null);
  const voRows = splitOnly ? (existing?.voRows ?? []) : await runVoQa();
  const split = voOnly ? (existing?.split ?? null) : await runSplitQa();
  return {
    generatedAt: new Date().toISOString(),
    environment: {
      node: process.version,
      audioSplitterDevice: process.env.AUDIO_SPLITTER_DEVICE ?? null,
      audioSplitterAutocast: process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_AUTOCAST ?? null,
      audioSplitterCommand: process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND ?? null,
    },
    voSummary: summarizeVoRows(voRows),
    voRows,
    split,
  };
};

const main = async () => {
  await mkdir("tasks", { recursive: true });
  const report = await buildReport();
  await writeFile(REPORT_JSON, `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(REPORT_MD, renderMarkdown(report));
  console.log(`Wrote ${REPORT_JSON}`);
  console.log(`Wrote ${REPORT_MD}`);
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
