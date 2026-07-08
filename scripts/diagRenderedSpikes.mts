/**
 * Rendered consonant-spike diagnostic.
 *
 * Compares a mix-ready WAV against its source and reports narrow rendered
 * peaks that sit far above the local speech body. It also runs the same
 * full-rate consonant tamer used by the app in-memory and reports the
 * before/after reduction.
 *
 * Usage:
 *   node --experimental-strip-types --max-old-space-size=6144 scripts/diagRenderedSpikes.mts [source.wav rendered.wav]
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { tameRenderedConsonantPeaks } from "../src/lib/gainPlanner.ts";
import { decodeWav } from "../src/lib/webAudioRender.ts";

const ROOT = path.resolve(import.meta.dirname, "..");
const DEFAULT_SOURCE = path.join(ROOT, "end spiked down", "Lucas Martin_batchvideo1-10.wav");
const DEFAULT_RENDER = path.join(ROOT, "end spiked down", "Lucas_Martin_batchvideo1-10_mixready.wav");
const FRAME_MS = 10;
const LOCAL_WINDOW_MS = 280;
const FAIL_ABSOLUTE_PEAK_DB = -6.5;
const FAIL_VISIBLE_OVER_BODY_DB = 12;
const FAIL_NARROW_OVER_BODY_DB = 17;
const MIN_SPEECH_RMS_DB = -70;
const MIN_CANDIDATE_PEAK_DB = -14;

const [, , sourceArg, renderArg] = process.argv;
const sourcePath = sourceArg ? path.resolve(sourceArg) : DEFAULT_SOURCE;
const renderPath = renderArg ? path.resolve(renderArg) : DEFAULT_RENDER;

type LoadedAudio = {
  name: string;
  samples: Float32Array;
  sampleRate: number;
};

type FrameMetrics = {
  rmsDb: number[];
  peakDb: number[];
  samplesPerFrame: number;
};

type SpikeGroup = {
  startFrame: number;
  endFrame: number;
  frame: number;
  peakDb: number;
  rmsDb: number;
  bodyDb: number;
  peakOverBodyDb: number;
  crestDb: number;
  sourcePeakDb: number | null;
  sourceRmsDb: number | null;
  sourcePeakDeltaDb: number | null;
};

const dbToPower = (db: number) => Math.pow(10, db / 10);

const powerToDb = (power: number) => 10 * Math.log10(power + 1e-30);

const loadMono = (filePath: string): LoadedAudio => {
  const bytes = readFileSync(filePath);
  const decoded = decodeWav(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  if (decoded.channels <= 1) {
    return { name: path.basename(filePath), samples: decoded.samples, sampleRate: decoded.sampleRate };
  }

  const frameCount = Math.floor(decoded.samples.length / decoded.channels);
  const mono = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < decoded.channels; channel += 1) {
      sum += decoded.samples[frame * decoded.channels + channel];
    }
    mono[frame] = sum / decoded.channels;
  }
  return { name: path.basename(filePath), samples: mono, sampleRate: decoded.sampleRate };
};

const measureFrames = (samples: Float32Array, sampleRate: number): FrameMetrics => {
  const samplesPerFrame = Math.max(1, Math.round((sampleRate * FRAME_MS) / 1000));
  const frameCount = Math.ceil(samples.length / samplesPerFrame);
  const rmsDb = new Array<number>(frameCount);
  const peakDb = new Array<number>(frameCount);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * samplesPerFrame;
    const end = Math.min(samples.length, start + samplesPerFrame);
    let power = 0;
    let peak = 0;
    for (let index = start; index < end; index += 1) {
      const value = samples[index];
      power += value * value;
      peak = Math.max(peak, Math.abs(value));
    }
    const rms = Math.sqrt(power / Math.max(1, end - start));
    rmsDb[frame] = rms > 0 ? 20 * Math.log10(rms) : -120;
    peakDb[frame] = peak > 0 ? 20 * Math.log10(peak) : -120;
  }

  return { rmsDb, peakDb, samplesPerFrame };
};

const localBodyDb = (frameDb: number[], centerFrame: number) => {
  const windowFrames = Math.max(1, Math.round(LOCAL_WINDOW_MS / FRAME_MS));
  const values: number[] = [];
  const start = Math.max(0, centerFrame - windowFrames);
  const end = Math.min(frameDb.length, centerFrame + windowFrames + 1);
  for (let frame = start; frame < end; frame += 1) {
    const value = frameDb[frame];
    if (Number.isFinite(value) && value >= -58) values.push(value);
  }
  if (values.length === 0) return frameDb[centerFrame] ?? -120;
  values.sort((left, right) => left - right);
  const trimmed = values.slice(0, Math.max(1, Math.ceil(values.length * 0.72)));
  return trimmed[Math.floor(trimmed.length * 0.6)] ?? values[Math.floor(values.length / 2)] ?? -120;
};

const buildSpikeGroups = (render: FrameMetrics, source: FrameMetrics | null): SpikeGroup[] => {
  const candidateFrames: number[] = [];
  for (let frame = 0; frame < render.rmsDb.length; frame += 1) {
    const peakDb = render.peakDb[frame] ?? -120;
    const rmsDb = render.rmsDb[frame] ?? -120;
    if (peakDb < MIN_CANDIDATE_PEAK_DB || rmsDb < MIN_SPEECH_RMS_DB) continue;

    const bodyDb = localBodyDb(render.rmsDb, frame);
    const peakOverBodyDb = peakDb - bodyDb;
    const crestDb = peakDb - rmsDb;
    const strongVisiblePeak = peakDb >= FAIL_ABSOLUTE_PEAK_DB && peakOverBodyDb >= FAIL_VISIBLE_OVER_BODY_DB;
    const narrowConsonantPeak = peakOverBodyDb >= FAIL_NARROW_OVER_BODY_DB || crestDb >= 18;
    if (strongVisiblePeak || narrowConsonantPeak) candidateFrames.push(frame);
  }

  const groups: SpikeGroup[] = [];
  let index = 0;
  while (index < candidateFrames.length) {
    const startFrame = candidateFrames[index];
    let endFrame = startFrame + 1;
    index += 1;
    while (index < candidateFrames.length && candidateFrames[index] <= endFrame + 1) {
      endFrame = candidateFrames[index] + 1;
      index += 1;
    }

    let best: SpikeGroup | null = null;
    for (let frame = startFrame; frame < endFrame; frame += 1) {
      const peakDb = render.peakDb[frame] ?? -120;
      const rmsDb = render.rmsDb[frame] ?? -120;
      const bodyDb = localBodyDb(render.rmsDb, frame);
      const sourcePeakDb = source?.peakDb[frame] ?? null;
      const sourceRmsDb = source?.rmsDb[frame] ?? null;
      const candidate: SpikeGroup = {
        startFrame,
        endFrame,
        frame,
        peakDb,
        rmsDb,
        bodyDb,
        peakOverBodyDb: peakDb - bodyDb,
        crestDb: peakDb - rmsDb,
        sourcePeakDb,
        sourceRmsDb,
        sourcePeakDeltaDb: sourcePeakDb === null ? null : peakDb - sourcePeakDb,
      };
      if (!best || candidate.peakDb > best.peakDb || candidate.peakOverBodyDb > best.peakOverBodyDb) {
        best = candidate;
      }
    }
    if (best) groups.push(best);
  }

  return groups.sort((left, right) => right.peakDb - left.peakDb);
};

const speechBodyDeltaDb = (before: FrameMetrics, after: FrameMetrics, groups: SpikeGroup[]) => {
  const affected = new Set<number>();
  for (const group of groups) {
    for (let frame = group.startFrame - 3; frame <= group.endFrame + 3; frame += 1) affected.add(frame);
  }

  let beforePower = 0;
  let afterPower = 0;
  let count = 0;
  for (let frame = 0; frame < before.rmsDb.length; frame += 1) {
    const rmsDb = before.rmsDb[frame] ?? -120;
    if (affected.has(frame) || rmsDb < -58 || rmsDb > -12) continue;
    beforePower += dbToPower(rmsDb);
    afterPower += dbToPower(after.rmsDb[frame] ?? -120);
    count += 1;
  }

  return {
    frameCount: count,
    deltaDb: count > 0 ? powerToDb(afterPower / count) - powerToDb(beforePower / count) : 0,
  };
};

const formatDb = (value: number | null) => (value === null ? "n/a" : `${value.toFixed(1)}dB`);

const printGroups = (label: string, groups: SpikeGroup[], limit = 10) => {
  console.log(`[${label}] groups=${groups.length}`);
  for (const group of groups.slice(0, limit)) {
    console.log(
      `  ${((group.frame * FRAME_MS) / 1000).toFixed(2)}s peak=${group.peakDb.toFixed(1)}dB ` +
        `rms=${group.rmsDb.toFixed(1)}dB body=${group.bodyDb.toFixed(1)}dB ` +
        `overBody=${group.peakOverBodyDb.toFixed(1)}dB crest=${group.crestDb.toFixed(1)}dB ` +
        `srcPeak=${formatDb(group.sourcePeakDb)} srcRms=${formatDb(group.sourceRmsDb)} ` +
        `peakDelta=${formatDb(group.sourcePeakDeltaDb)}`,
    );
  }
};

const source = loadMono(sourcePath);
const rendered = loadMono(renderPath);
if (source.sampleRate !== rendered.sampleRate) {
  throw new Error(`Sample-rate mismatch: source ${source.sampleRate}Hz, rendered ${rendered.sampleRate}Hz`);
}

console.log(
  `[Files] source=${source.name} rendered=${rendered.name} sr=${rendered.sampleRate}Hz ` +
    `duration=${(rendered.samples.length / rendered.sampleRate).toFixed(2)}s`,
);

const sourceMetrics = measureFrames(source.samples, source.sampleRate);
const beforeMetrics = measureFrames(rendered.samples, rendered.sampleRate);
const beforeGroups = buildSpikeGroups(beforeMetrics, sourceMetrics);
printGroups("Before", beforeGroups);

const tamed = tameRenderedConsonantPeaks(rendered.samples, rendered.sampleRate, FRAME_MS);
const afterMetrics = measureFrames(tamed.samples, rendered.sampleRate);
const afterGroups = buildSpikeGroups(afterMetrics, sourceMetrics);
const bodyDelta = speechBodyDeltaDb(beforeMetrics, afterMetrics, beforeGroups);
printGroups("AfterTamer", afterGroups);

const worstBefore = beforeGroups[0]?.peakDb ?? -120;
const worstAfter = afterGroups[0]?.peakDb ?? -120;
const failAfter = afterGroups.some(
  (group) =>
    group.peakDb >= FAIL_ABSOLUTE_PEAK_DB &&
    group.peakOverBodyDb >= FAIL_VISIBLE_OVER_BODY_DB &&
    (group.sourcePeakDeltaDb ?? 0) >= 3,
);

console.log(
  `[Tamer] touched=${tamed.stats.tamedFrameCount} maxReduction=${tamed.stats.maxReductionDb.toFixed(1)}dB ` +
    `worstPeak ${worstBefore.toFixed(1)}dB -> ${worstAfter.toFixed(1)}dB ` +
    `speechBodyDelta=${bodyDelta.deltaDb.toFixed(3)}dB over ${bodyDelta.frameCount} frame(s)`,
);

console.log(`[Verdict] ${failAfter ? "FAIL - rendered visible consonant spike remains" : "PASS"}`);
process.exit(failAfter ? 1 : 0);
