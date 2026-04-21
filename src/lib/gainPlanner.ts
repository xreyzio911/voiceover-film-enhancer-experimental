/**
 * Speech-aware gain planner.
 *
 * Produces a per-frame linear gain curve that:
 *  - brings every speech run to a common target RMS (kills sentence-to-sentence jumps),
 *  - gently rides intra-sentence swing with a slew-limited curve (no pumping),
 *  - ducks silences below a noise-risk-driven expander depth (no noise lift),
 *  - guards peaks so the downstream limiter never has to clamp hard (no spikes).
 *
 * This replaces ffmpeg's blind `dynaudnorm` for the core leveling role.
 * Pure JS so it is fully testable in Node and reusable on any Float32Array.
 */

export type SpeechRun = {
  /** inclusive start frame index (10ms-per-frame convention) */
  startFrame: number;
  /** exclusive end frame index */
  endFrame: number;
};

export type GainPlannerInput = {
  /** 10ms-frame RMS in dB (e.g. from analyzeFloatSamples envelope). */
  frameDb: number[];
  /** Speech runs (frame index ranges) already detected by the analyzer. */
  speechRuns: SpeechRun[];
  /** Noise floor of the source in dB (pauseNoiseFloorDb). */
  noiseFloorDb: number;
  /** Speech-vs-noise boundary (speechThresholdDb). */
  speechThresholdDb: number;
  /** 0..1 pause noise risk — drives expander depth. */
  pauseNoiseRisk: number;
  /** Frame duration in ms. Defaults to 10. */
  frameMs?: number;
  /** Target integrated RMS dB for speech runs. Defaults to -22 dBFS RMS (maps roughly to -24 LKFS). */
  targetDb?: number;
  /** Max gain applied to a single run, in dB. Defaults to +12. */
  maxGainDb?: number;
  /** Max attenuation applied to a single run, in dB. Defaults to -12. */
  minGainDb?: number;
  /** Optional Float32 samples + sampleRate. If supplied, peak-guard pass simulates the applied gain. */
  samples?: Float32Array;
  sampleRate?: number;
  /** Ceiling in dBFS for samples after gain is applied (limiter has margin beyond this). Default -3. */
  peakCeilingDb?: number;
  /**
   * 0..1 signal describing how unstable the source is (frame-to-frame RMS
   * deltas inside speech + line swing). On CLEAN takes we want almost no
   * micro-ride so sentences come out glass-flat; on MESSY takes we want the
   * full ±1.5 dB correction. Defaults to 0.5 (midpoint) when unknown.
   */
  instabilityHint?: number;
};

export type GainPlannerOutput = {
  /** One linear gain per frame. Length = frameDb.length. */
  gainCurve: Float32Array;
  /** Per-run diagnostic info. */
  runs: Array<{
    startFrame: number;
    endFrame: number;
    meanDb: number;
    plannedGainDb: number;
    peakReducedDb: number;
  }>;
  /** Computed expander depth in dB used for silences. */
  expanderDepthDb: number;
  /** Target RMS dB that all speech runs were aimed at. */
  targetDb: number;
  /** Effective micro-ride amplitude in dB (peak-to-peak / 2). Diagnostic. */
  microRideDb: number;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const dbToLin = (db: number) => Math.pow(10, db / 20);
const linToDb = (lin: number) => (lin <= 0 ? -120 : 20 * Math.log10(lin));

const median = (values: number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
};

const rmsDbOfSlice = (frameDb: number[], start: number, end: number): number => {
  const a = Math.max(0, start);
  const b = Math.min(frameDb.length, end);
  if (b <= a) return -120;
  // RMS-in-dB over a slice. Each frame is already 20*log10(rms).
  // We want 10*log10(mean(rms^2)) = 10*log10(mean(10^(frameDb/10))).
  let sumPower = 0;
  for (let i = a; i < b; i += 1) {
    sumPower += Math.pow(10, frameDb[i] / 10);
  }
  return 10 * Math.log10(sumPower / (b - a) + 1e-30);
};

/**
 * Apply a one-pole slew limiter to a gain curve in dB, measured in dB/second.
 * Asymmetric: attack and release separately controlled.
 */
const slewLimitDbCurve = (
  dbCurve: Float32Array,
  frameMs: number,
  attackDbPerSec: number,
  releaseDbPerSec: number,
): Float32Array => {
  const out = new Float32Array(dbCurve.length);
  if (dbCurve.length === 0) return out;
  const stepSec = frameMs / 1000;
  const maxUp = attackDbPerSec * stepSec;
  const maxDown = releaseDbPerSec * stepSec;
  out[0] = dbCurve[0];
  for (let i = 1; i < dbCurve.length; i += 1) {
    const prev = out[i - 1];
    const target = dbCurve[i];
    const delta = target - prev;
    if (delta > maxUp) out[i] = prev + maxUp;
    else if (delta < -maxDown) out[i] = prev - maxDown;
    else out[i] = target;
  }
  return out;
};

/**
 * Plan a gain curve for speech-aware leveling.
 */
export const planGainCurve = (input: GainPlannerInput): GainPlannerOutput => {
  const frameMs = input.frameMs ?? 10;
  const targetDbBase = input.targetDb ?? -22;
  const maxGainDb = input.maxGainDb ?? 12;
  const minGainDb = input.minGainDb ?? -12;
  const peakCeilingDb = input.peakCeilingDb ?? -3;

  const frameCount = input.frameDb.length;
  const gainDbCurve = new Float32Array(frameCount);

  // Adaptive micro-ride amplitude. The micro-ride applies small corrective
  // gain inside each speech run so the sliding RMS hugs the target. On a
  // clean take that's already smooth, a large micro-ride introduces its own
  // residual variance that downstream QC flags as "sentence jumps" when the
  // post-processing speech detector splits a sentence at breath gaps. On
  // messy takes, we need the full ride to keep the body level stable.
  //
  // Scale: ±0.4 dB at instabilityHint=0, ±1.5 dB at instabilityHint=1.
  const instabilityHint = Math.max(0, Math.min(1, input.instabilityHint ?? 0.5));
  const microRideDb = 0.4 + instabilityHint * 1.1;

  // 1) Per-run body RMS (trim 12% at each edge to avoid consonant transients).
  const runRmsDb: number[] = [];
  const runMeta: Array<{ startFrame: number; endFrame: number; meanDb: number }> = [];
  for (const run of input.speechRuns) {
    const runFrames = run.endFrame - run.startFrame;
    if (runFrames < 6) continue;
    const trim = Math.max(2, Math.floor(runFrames * 0.12));
    const bodyStart = run.startFrame + trim;
    const bodyEnd = Math.max(bodyStart + 1, run.endFrame - trim);
    const meanDb = rmsDbOfSlice(input.frameDb, bodyStart, bodyEnd);
    if (Number.isFinite(meanDb) && meanDb > -100) {
      runRmsDb.push(meanDb);
      runMeta.push({ startFrame: run.startFrame, endFrame: run.endFrame, meanDb });
    }
  }

  // 2) Target = median of run body RMS, nudged toward targetDbBase.
  // We aim the inter-run level at a stable point: halfway between the median
  // source RMS and the user's target. This avoids over-lifting quiet files
  // (which would bring noise up) and over-squashing loud files.
  // Target = median of run body RMS, nudged toward `targetDbBase`.
  // We weight toward the median to avoid over-lifting quiet takes (which would
  // bring noise up), but keep a meaningful pull toward the loudness target so
  // the file lands near the user's expected operating level after loudnorm.
  let targetDb = targetDbBase;
  if (runRmsDb.length >= 1) {
    const medDb = median(runRmsDb);
    targetDb = clamp(0.55 * medDb + 0.45 * targetDbBase, targetDbBase - 5, targetDbBase + 5);
  }

  // 3) Per-run planned gain (clamped), plus cross-run smoothing so adjacent
  //    runs differing by > 2 dB of planned gain are softened.
  const plannedRunGainDb: number[] = runMeta.map((m) =>
    clamp(targetDb - m.meanDb, minGainDb, maxGainDb),
  );
  for (let i = 1; i < plannedRunGainDb.length; i += 1) {
    const diff = plannedRunGainDb[i] - plannedRunGainDb[i - 1];
    if (Math.abs(diff) > 3) {
      const mid = (plannedRunGainDb[i] + plannedRunGainDb[i - 1]) / 2;
      plannedRunGainDb[i - 1] = plannedRunGainDb[i - 1] + (mid - plannedRunGainDb[i - 1]) * 0.35;
      plannedRunGainDb[i] = plannedRunGainDb[i] + (mid - plannedRunGainDb[i]) * 0.35;
    }
  }

  // 4) Expander depth — deeper duck on noisier files.
  const expanderDepthDb = clamp(12 + input.pauseNoiseRisk * 18, 12, 30);

  // 5) Paint the curve.
  // Default = silence behavior: gain - expanderDepthDb.
  // Inside each speech run, blend from edge to planned body gain with short ramps.
  const silenceGainDefaultDb = -expanderDepthDb;
  for (let i = 0; i < frameCount; i += 1) gainDbCurve[i] = silenceGainDefaultDb;

  const edgeInFrames = Math.max(1, Math.round(80 / frameMs)); // 80 ms attack
  const edgeOutFrames = Math.max(1, Math.round(200 / frameMs)); // 200 ms release

  for (let r = 0; r < runMeta.length; r += 1) {
    const { startFrame, endFrame } = runMeta[r];
    const runLen = endFrame - startFrame;
    const bodyGainDb = plannedRunGainDb[r];

    // Intra-run micro-ride: track a 200ms sliding RMS inside the run and
    // apply small corrective gain (±microRideDb) on top of bodyGainDb.
    const slideFrames = Math.max(4, Math.round(200 / frameMs));
    for (let i = startFrame; i < endFrame; i += 1) {
      const winStart = Math.max(startFrame, i - Math.floor(slideFrames / 2));
      const winEnd = Math.min(endFrame, i + Math.ceil(slideFrames / 2));
      const localDb = rmsDbOfSlice(input.frameDb, winStart, winEnd);
      const microGainDb = clamp(targetDb - (localDb + bodyGainDb), -microRideDb, microRideDb);
      gainDbCurve[i] = bodyGainDb + microGainDb;
    }

    // Edge ramps — blend from silence gain into body gain.
    const rampInLen = Math.min(edgeInFrames, Math.floor(runLen / 2));
    for (let k = 0; k < rampInLen; k += 1) {
      const t = (k + 1) / (rampInLen + 1);
      const silenceEdge = startFrame - 1 >= 0 ? gainDbCurve[startFrame - 1] : silenceGainDefaultDb;
      gainDbCurve[startFrame + k] = silenceEdge + (gainDbCurve[startFrame + k] - silenceEdge) * t;
    }
    const rampOutLen = Math.min(edgeOutFrames, Math.floor(runLen / 2));
    for (let k = 0; k < rampOutLen; k += 1) {
      const t = (k + 1) / (rampOutLen + 1);
      const frame = endFrame - 1 - k;
      const silenceEdge = endFrame < frameCount ? silenceGainDefaultDb : silenceGainDefaultDb;
      gainDbCurve[frame] = silenceEdge + (gainDbCurve[frame] - silenceEdge) * t;
    }
  }

  // 6) Global slew limit — prevents anything faster than 3 dB / 100 ms = 30 dB/sec.
  //    Asymmetric: release slower than attack to avoid audible ducking on plosives.
  const slewed = slewLimitDbCurve(gainDbCurve, frameMs, 30, 12);

  // 7) Peak guard — if samples provided, walk through and reduce any run whose
  //    applied peak would exceed peakCeilingDb.
  const peakReductionDbByRun = new Array<number>(runMeta.length).fill(0);
  if (input.samples && input.sampleRate && input.sampleRate > 0) {
    const samplesPerFrame = Math.max(1, Math.round((input.sampleRate * frameMs) / 1000));
    const ceilingLin = dbToLin(peakCeilingDb);
    for (let r = 0; r < runMeta.length; r += 1) {
      const { startFrame, endFrame } = runMeta[r];
      const sampleStart = startFrame * samplesPerFrame;
      const sampleEnd = Math.min(input.samples.length, endFrame * samplesPerFrame);
      let maxApplied = 0;
      for (let i = sampleStart; i < sampleEnd; i += 1) {
        const frameIdx = Math.min(frameCount - 1, Math.floor(i / samplesPerFrame));
        const gainLin = dbToLin(slewed[frameIdx]);
        const abs = Math.abs(input.samples[i]) * gainLin;
        if (abs > maxApplied) maxApplied = abs;
      }
      if (maxApplied > ceilingLin) {
        const reduceDb = linToDb(ceilingLin / maxApplied); // negative
        peakReductionDbByRun[r] = reduceDb;
        for (let f = startFrame; f < endFrame; f += 1) {
          slewed[f] += reduceDb;
        }
      }
    }
  }

  // 8) Convert to linear.
  const gainCurve = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i += 1) gainCurve[i] = dbToLin(slewed[i]);

  return {
    gainCurve,
    runs: runMeta.map((m, i) => ({
      startFrame: m.startFrame,
      endFrame: m.endFrame,
      meanDb: m.meanDb,
      plannedGainDb: plannedRunGainDb[i] ?? 0,
      peakReducedDb: peakReductionDbByRun[i] ?? 0,
    })),
    expanderDepthDb,
    targetDb,
    microRideDb,
  };
};

/**
 * Apply a per-frame linear gain curve directly to samples.
 * Works in place if `outSamples` is `samples`, otherwise writes into `outSamples`.
 * Linearly interpolates gain between frame midpoints so there are no zipper artifacts.
 */
export const applyGainCurveToSamples = (
  samples: Float32Array,
  gainCurve: Float32Array,
  sampleRate: number,
  channels: number,
  frameMs: number,
  outSamples?: Float32Array,
): Float32Array => {
  const out = outSamples ?? new Float32Array(samples.length);
  const samplesPerFrame = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const framesPerSec = 1000 / frameMs;
  const centerOffset = samplesPerFrame / 2;
  const totalFrames = gainCurve.length;
  const sampleCount = samples.length;
  const frameCountByChannel = Math.floor(sampleCount / channels);

  for (let sIdx = 0; sIdx < frameCountByChannel; sIdx += 1) {
    // position in frame units, offset so gains line up at frame midpoints
    const framePos = (sIdx - centerOffset) / samplesPerFrame;
    const f0 = Math.max(0, Math.min(totalFrames - 1, Math.floor(framePos)));
    const f1 = Math.max(0, Math.min(totalFrames - 1, f0 + 1));
    const mix = Math.max(0, Math.min(1, framePos - f0));
    const gainLin = gainCurve[f0] * (1 - mix) + gainCurve[f1] * mix;
    for (let c = 0; c < channels; c += 1) {
      const i = sIdx * channels + c;
      out[i] = samples[i] * gainLin;
    }
  }
  // unused var kept for API clarity
  void framesPerSec;
  return out;
};

/**
 * Emit a `sendcmd` script that drives ffmpeg's `volume` filter through the
 * planned gain curve for an arbitrary time window.
 *
 * `sendcmd` format: `timestamp command filter arg;`. Timestamps here are
 * **relative** to the sub-stream the script is applied to — for a segmented
 * render using `-ss START -t DUR`, the segment's input timeline starts at 0,
 * so we subtract `windowStartSec` from every keyframe timestamp.
 *
 * Keyframes are decimated: we only emit a new line when the linear gain
 * changes by more than `minDeltaLin` vs the last emitted keyframe. Typical
 * 15-minute file with fast-tracking curve yields ~2-5k lines.
 */
export const emitSendcmdScript = (
  gainCurve: Float32Array,
  frameMs: number,
  windowStartSec: number,
  windowEndSec: number,
  minDeltaLin = 0.015,
): string => {
  const frameSec = frameMs / 1000;
  const startFrame = Math.max(0, Math.floor(windowStartSec / frameSec));
  const endFrame = Math.min(gainCurve.length, Math.ceil(windowEndSec / frameSec));
  if (endFrame <= startFrame) return "";

  const lines: string[] = [];
  let lastEmittedLin = Number.NaN;
  let lastEmittedFrame = -1;
  const emit = (frameIdx: number, lin: number) => {
    const relSec = frameIdx * frameSec - windowStartSec;
    const t = Math.max(0, relSec);
    lines.push(`${t.toFixed(3)} volume volume ${lin.toFixed(5)};`);
    lastEmittedLin = lin;
    lastEmittedFrame = frameIdx;
  };

  // Always emit the first keyframe at t=0 so the filter starts at the right gain.
  emit(startFrame, gainCurve[startFrame]);

  for (let f = startFrame + 1; f < endFrame; f += 1) {
    const lin = gainCurve[f];
    if (Math.abs(lin - lastEmittedLin) >= minDeltaLin) {
      emit(f, lin);
    }
  }
  // Ensure the final gain applies until the end of the window.
  if (lastEmittedFrame !== endFrame - 1) {
    emit(endFrame - 1, gainCurve[endFrame - 1]);
  }

  return lines.join("\n") + "\n";
};

/**
 * Build speech runs from an already-computed speech mask.
 * Exposed for tests; the VoLeveler uses its existing span detection.
 */
export const speechRunsFromMask = (mask: boolean[]): SpeechRun[] => {
  const runs: SpeechRun[] = [];
  let runStart = -1;
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] && runStart < 0) runStart = i;
    if (!mask[i] && runStart >= 0) {
      runs.push({ startFrame: runStart, endFrame: i });
      runStart = -1;
    }
  }
  if (runStart >= 0) runs.push({ startFrame: runStart, endFrame: mask.length });
  return runs;
};
