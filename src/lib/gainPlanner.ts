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
  /**
   * How much the planner may follow the source's typical speech RMS when
   * choosing its target. 0 = fixed house target, 1 = source target.
   */
  sourceTargetBlend?: number;
  /** Max gain applied to a single run, in dB. Defaults to +14. */
  maxGainDb?: number;
  /** Max attenuation applied to a single run, in dB. Defaults to -14. */
  minGainDb?: number;
  /** Optional Float32 samples + sampleRate. If supplied, peak-guard pass simulates the applied gain. */
  samples?: Float32Array;
  sampleRate?: number;
  /** Ceiling in dBFS for samples after gain is applied (limiter has margin beyond this). Default -4. */
  peakCeilingDb?: number;
  /**
   * 0..1 signal describing how unstable the source is (frame-to-frame RMS
   * deltas inside speech + line swing). On CLEAN takes we want almost no
   * micro-ride so sentences come out glass-flat; on MESSY takes we want the
   * full ±1.5 dB correction. Defaults to 0.5 (midpoint) when unknown.
   */
  instabilityHint?: number;
};

/**
 * How a detected speech run is treated by the planner.
 *
 * - `body-speech`: normal dialogue. Targeted to the batch level, full
 *   micro-ride, peak guard at the usual ceiling.
 * - `transient-breath`: a short, high-crest run — a character gasp, laugh,
 *   grunt, or similar onomatopoeic performance beat. Targeted a little
 *   below dialogue so the breath SITS WITH the character rather than
 *   poking above, tight gain clamp (no big swings), no micro-ride (too
 *   short to ride).
 * - `edge-fragment`: too short to process (< 100 ms). Left at body target
 *   unclamped but gets no special handling — too little data to plan on.
 */
export type SpeechRunClass = "body-speech" | "transient-breath" | "edge-fragment";

export type GainPlannerOutput = {
  /** One linear gain per frame. Length = frameDb.length. */
  gainCurve: Float32Array;
  /** Per-run diagnostic info. */
  runs: Array<{
    startFrame: number;
    endFrame: number;
    meanDb: number;
    crestDb: number;
    plannedGainDb: number;
    peakReducedDb: number;
    runClass: SpeechRunClass;
  }>;
  /** Computed expander depth in dB used for silences. */
  expanderDepthDb: number;
  /** Target RMS dB that all speech runs were aimed at. */
  targetDb: number;
  /** Effective micro-ride amplitude in dB (peak-to-peak / 2). Diagnostic. */
  microRideDb: number;
  /** Count of runs classified as transient-breath. Diagnostic. */
  breathRunCount: number;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const dbToLin = (db: number) => Math.pow(10, db / 20);

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
 * Plan a gain curve for speech-aware leveling.
 */
export const planGainCurve = (input: GainPlannerInput): GainPlannerOutput => {
  const frameMs = input.frameMs ?? 10;
  const targetDbBase = input.targetDb ?? -22;
  const sourceTargetBlend = clamp(input.sourceTargetBlend ?? 0.15, 0, 1);
  const maxGainDb = input.maxGainDb ?? 14;
  const minGainDb = input.minGainDb ?? -14;
  // -4 dBFS ceiling gives the downstream `alimiter=limit=-2dB` genuine
  // headroom — peaks that nearly kiss our ceiling leave 2 dB for the
  // limiter to shape transients without clipping.
  const peakCeilingDb = input.peakCeilingDb ?? -4;

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

  // 1) Per-run body RMS + classification.
  //
  // Each detected run is tagged as body-speech / transient-breath / edge.
  // Classification uses run duration and the peak-to-body crest ratio — a
  // sub-400 ms run with > 15 dB crest is almost always a gasp / laugh /
  // grunt / plosive-dominated syllable, not a normal dialogue sentence.
  //
  // When `input.samples` is provided we compute a per-frame peak from the
  // actual waveform (accurate); otherwise we approximate peak ~= body + 12
  // dB which is the typical speech crest factor.
  const runRmsDb: number[] = []; // body-speech runs ONLY — these drive the target
  type RunEntry = {
    startFrame: number;
    endFrame: number;
    meanDb: number;
    peakDb: number;
    crestDb: number;
    runClass: SpeechRunClass;
  };
  const runMeta: RunEntry[] = [];

  const samplesPerFrame = input.sampleRate && input.sampleRate > 0
    ? Math.max(1, Math.round((input.sampleRate * frameMs) / 1000))
    : 0;
  const framePeakDb: number[] | null = input.samples && samplesPerFrame > 0
    ? new Array<number>(frameCount).fill(-120)
    : null;
  if (framePeakDb && input.samples) {
    for (let f = 0; f < frameCount; f += 1) {
      const start = f * samplesPerFrame;
      const end = Math.min(input.samples.length, start + samplesPerFrame);
      let peak = 0;
      for (let i = start; i < end; i += 1) {
        const abs = Math.abs(input.samples[i]);
        if (abs > peak) peak = abs;
      }
      framePeakDb[f] = peak > 0 ? 20 * Math.log10(peak) : -120;
    }
  }

  for (let runIndex = 0; runIndex < input.speechRuns.length; runIndex += 1) {
    const run = input.speechRuns[runIndex];
    const runFrames = run.endFrame - run.startFrame;
    if (runFrames < 6) continue;
    const trim = Math.max(2, Math.floor(runFrames * 0.12));
    const bodyStart = run.startFrame + trim;
    const bodyEnd = Math.max(bodyStart + 1, run.endFrame - trim);
    const meanDb = rmsDbOfSlice(input.frameDb, bodyStart, bodyEnd);
    if (!Number.isFinite(meanDb) || meanDb <= -100) continue;

    // Peak over the ENTIRE run (including edges — that's where plosives
    // live). Fall back to frameDb + 12 dB when samples are unavailable.
    let peakDb = -120;
    if (framePeakDb) {
      for (let f = run.startFrame; f < run.endFrame; f += 1) {
        if (framePeakDb[f] > peakDb) peakDb = framePeakDb[f];
      }
    } else {
      for (let f = run.startFrame; f < run.endFrame; f += 1) {
        if (input.frameDb[f] > peakDb) peakDb = input.frameDb[f];
      }
      peakDb += 12;
    }
    const crestDb = peakDb - meanDb;
    const runLenMs = runFrames * frameMs;
    const previousEndFrame = runIndex > 0 ? input.speechRuns[runIndex - 1].endFrame : 0;
    const nextStartFrame =
      runIndex + 1 < input.speechRuns.length ? input.speechRuns[runIndex + 1].startFrame : frameCount;
    const preGapMs = Math.max(0, run.startFrame - previousEndFrame) * frameMs;
    const postGapMs = Math.max(0, nextStartFrame - run.endFrame) * frameMs;
    const isolatedOrLeadIn = preGapMs >= 70 || postGapMs >= 70;
    const shortHotPerformance =
      runLenMs < 650 &&
      isolatedOrLeadIn &&
      (crestDb >= 13.5 ||
        peakDb >= targetDbBase + 10.5 ||
        (runLenMs < 360 && peakDb >= targetDbBase + 8 && meanDb <= targetDbBase + 1.5));

    let runClass: SpeechRunClass;
    if (shortHotPerformance) {
      runClass = "transient-breath";
    } else if (runLenMs < 100) {
      runClass = "edge-fragment";
    } else if (runLenMs < 400 && crestDb >= 15) {
      runClass = "transient-breath";
    } else {
      runClass = "body-speech";
    }

    runMeta.push({
      startFrame: run.startFrame,
      endFrame: run.endFrame,
      meanDb,
      peakDb,
      crestDb,
      runClass,
    });
    // Only body-speech runs drive the batch target — a single loud gasp
    // must NOT pull the dialogue target level up.
    if (runClass === "body-speech") runRmsDb.push(meanDb);
  }

  // 2) Target = TRIMMED MEAN of run body RMS (drop extreme sentences as
  //    outliers), blended toward targetDbBase. Trimmed mean resists the
  //    single-loud-sentence skew the median had: median tracks the middle
  //    of the distribution, so one very loud line pulled the target up and
  //    left every quiet line under-amplified. Trimmed mean lands in the
  //    actual "typical" level of the take. For files with ≥ 7 runs we
  //    trim 15 % each end (min 1); for shorter takes we don't trim
  //    because the sample is already statistically small.
  let targetDb = targetDbBase;
  if (runRmsDb.length >= 1) {
    const sorted = [...runRmsDb].sort((a, b) => a - b);
    const trimCount =
      sorted.length >= 7 ? Math.max(1, Math.floor(sorted.length * 0.15)) : 0;
    const trimmed = sorted.slice(trimCount, Math.max(trimCount + 1, sorted.length - trimCount));
    const trimmedMean = trimmed.reduce((sum, v) => sum + v, 0) / trimmed.length;
    targetDb = clamp(
      sourceTargetBlend * trimmedMean + (1 - sourceTargetBlend) * targetDbBase,
      targetDbBase - 3,
      targetDbBase + 3
    );
  }

  // 3) Per-run planned gain — class-aware.
  //
  // body-speech: target the batch level, full ±maxGainDb clamp.
  // transient-breath: target 2.5 dB BELOW dialogue so the character beat
  //   sits with the performance instead of poking above it. Tight ±6 dB
  //   clamp so a loud gasp can't be amplified into a scream, and a very
  //   quiet gasp can't be lifted to full dialogue level.
  // edge-fragment: target batch level with tight ±4 dB clamp (not enough
  //   body to plan on, but still contribute to continuity).
  const breathTargetDb = targetDb - 3.2;
  const plannedRunGainDb: number[] = runMeta.map((m) => {
    if (m.runClass === "transient-breath") {
      return clamp(breathTargetDb - m.meanDb, -5, 4);
    }
    if (m.runClass === "edge-fragment") {
      return clamp(targetDb - m.meanDb, -4, 4);
    }
    return clamp(targetDb - m.meanDb, minGainDb, maxGainDb);
  });
  // Cross-run smoothing on adjacent body-speech pairs only (smoothing into
  // breaths would defeat the point of their separate targeting). Single
  // 35 % blend when planned gains differ by > 3 dB.
  for (let i = 1; i < plannedRunGainDb.length; i += 1) {
    const cur = runMeta[i];
    const prev = runMeta[i - 1];
    if (cur.runClass !== "body-speech" || prev.runClass !== "body-speech") continue;
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
  //
  // The old algorithm ramped INSIDE the speech run (first/last 80-200 ms of
  // each sentence got reduced gain), which clipped the first syllable's
  // attack and killed soft word endings (trailing "s", "m", "n" tails). The
  // new algorithm keeps the entire detected run at full body gain and ramps
  // into/out of silence BEFORE/AFTER the run using a cos² equal-power curve.
  //
  //   …silence… ——attack↗ [body at full body gain] release↘—— …silence…
  //               (80 ms in                         (500 ms in
  //                preceding silence)                following silence)
  //
  // Result: ending consonants and soft tails survive. First syllables are
  // not ducked. The expander still ducks sustained silence between runs.
  const silenceGainDefaultDb = -expanderDepthDb;
  for (let i = 0; i < frameCount; i += 1) gainDbCurve[i] = silenceGainDefaultDb;

  // 80 ms attack (short, so speech onset is crisp) and 500 ms release
  // (long, so trailing phonemes survive). Release is equal-power cos² — the
  // perceptual loudness decays linearly, not exponentially, so soft tails
  // don't vanish abruptly.
  const attackFrames = Math.max(1, Math.round(80 / frameMs));
  const releaseFrames = Math.max(1, Math.round(500 / frameMs));

  for (let r = 0; r < runMeta.length; r += 1) {
    const { startFrame, endFrame, runClass, crestDb } = runMeta[r];
    const bodyGainDb = plannedRunGainDb[r];

    // Micro-ride policy per class:
    // - transient-breath / edge-fragment: NO micro-ride. Too short to
    //   benefit and any local amplification raises the transient peak.
    // - body-speech with high crest (≥ 16 dB — consonant-heavy, shouty, or
    //   whispery lines): reduce micro-ride amplitude by 50 % so we don't
    //   amplify the consonant peaks in a body frame that happens to be
    //   locally quiet.
    // - body-speech normal: full micro-ride.
    const runEffectiveMicroRideDb = runClass !== "body-speech"
      ? 0
      : crestDb >= 16
        ? microRideDb * 0.5
        : microRideDb;

    const slideFrames = Math.max(4, Math.round(200 / frameMs));
    for (let i = startFrame; i < endFrame; i += 1) {
      if (runEffectiveMicroRideDb <= 0) {
        gainDbCurve[i] = bodyGainDb;
        continue;
      }
      const winStart = Math.max(startFrame, i - Math.floor(slideFrames / 2));
      const winEnd = Math.min(endFrame, i + Math.ceil(slideFrames / 2));
      const localDb = rmsDbOfSlice(input.frameDb, winStart, winEnd);
      const microGainDb = clamp(
        targetDb - (localDb + bodyGainDb),
        -runEffectiveMicroRideDb,
        runEffectiveMicroRideDb,
      );
      gainDbCurve[i] = bodyGainDb + microGainDb;
    }

    // Attack ramp — lives in the silence BEFORE the run, never inside it.
    // We walk back from startFrame, bounded by the previous run's end so we
    // don't trample that run's release.
    const prevRunEnd = r > 0 ? runMeta[r - 1].endFrame : 0;
    const attackStart = Math.max(startFrame - attackFrames, prevRunEnd, 0);
    const attackLen = startFrame - attackStart;
    const bodyGainAtStart = gainDbCurve[startFrame];
    for (let k = 0; k < attackLen; k += 1) {
      const t = (k + 1) / (attackLen + 1); // 0 → 1 as we approach run start
      const weight = Math.sin((t * Math.PI) / 2) ** 2; // cos² rising
      gainDbCurve[attackStart + k] =
        silenceGainDefaultDb + (bodyGainAtStart - silenceGainDefaultDb) * weight;
    }

    // Release ramp — lives in the silence AFTER the run, never inside it.
    // Bounded by the NEXT run's start so we don't overwrite its attack.
    const nextRunStart = r + 1 < runMeta.length ? runMeta[r + 1].startFrame : frameCount;
    const releaseEnd = Math.min(endFrame + releaseFrames, nextRunStart);
    const releaseLen = releaseEnd - endFrame;
    const bodyGainAtEnd = gainDbCurve[endFrame - 1];
    for (let k = 0; k < releaseLen; k += 1) {
      const t = (k + 1) / (releaseLen + 1); // 0 just after run → 1 deep in silence
      const weight = Math.cos((t * Math.PI) / 2) ** 2; // cos² falling
      gainDbCurve[endFrame + k] =
        silenceGainDefaultDb + (bodyGainAtEnd - silenceGainDefaultDb) * weight;
    }
  }

  // 6) No additional slew limiting. Every transition in `gainDbCurve` is
  //    already explicitly shaped:
  //      - attack: 80 ms cos² rising ramp (sits in preceding silence)
  //      - release: 500 ms cos² falling ramp (sits in following silence)
  //      - intra-run micro-ride: bounded to ±microRideDb over a 200 ms
  //        window = at most ~7 dB/sec slope
  //    A slew limiter here would actively fight those intended curves and
  //    produce under-powered attacks (the first syllable gets ducked
  //    because the slew can't catch up to body gain in 80 ms).
  const slewed = gainDbCurve;

  // 7) LOCALIZED peak guard.
  //
  // Previous approach reduced a WHOLE RUN's gain when any single sample
  // in the run exceeded the ceiling. A single loud plosive therefore cost
  // the entire sentence its body level. The user could hear "spikes"
  // because those plosive-dominated sentences came out with normal peaks
  // but QUIET bodies relative to the rest of the dialogue.
  //
  // New approach: compute the peak per 10 ms frame (we built framePeakDb
  // above when samples were available). For each frame whose applied peak
  // would exceed the ceiling, apply a 50 ms cosine DIP centered on that
  // frame, scaled to the exceedance. This brings that one plosive peak
  // under the ceiling while leaving 99% of the run's frames at full body
  // gain. Result: natural peak-to-body crest is preserved AND nothing
  // clips downstream.
  const peakDipFrames = Math.max(1, Math.round(25 / frameMs)); // 25 ms half-width, must be integer
  const peakReductionDbByRun = new Array<number>(runMeta.length).fill(0);
  const runIndexByFrame = new Array<number>(frameCount).fill(-1);
  for (let r = 0; r < runMeta.length; r += 1) {
    for (let f = runMeta[r].startFrame; f < runMeta[r].endFrame; f += 1) {
      runIndexByFrame[f] = r;
    }
  }
  if (framePeakDb) {
    for (let f = 0; f < frameCount; f += 1) {
      const currentGainDb = slewed[f];
      const runIdx = runIndexByFrame[f];
      const runMetaForFrame = runIdx >= 0 ? runMeta[runIdx] : null;
      const isPerformanceTransient = runMetaForFrame?.runClass === "transient-breath";
      const localPeakCeilingDb = isPerformanceTransient
        ? Math.min(peakCeilingDb, targetDb + 12.5)
        : peakCeilingDb;
      const appliedPeakDb = framePeakDb[f] + currentGainDb;
      if (appliedPeakDb <= localPeakCeilingDb) continue;
      const excessDb = appliedPeakDb - localPeakCeilingDb; // how much over ceiling
      // Apply cosine dip from -peakDipFrames..+peakDipFrames around f.
      for (let k = -peakDipFrames; k <= peakDipFrames; k += 1) {
        const idx = f + k;
        if (idx < 0 || idx >= frameCount) continue;
        if (
          isPerformanceTransient &&
          runMetaForFrame &&
          (idx < runMetaForFrame.startFrame || idx >= runMetaForFrame.endFrame)
        ) {
          continue;
        }
        // Cosine weight peaks at k=0 (full reduction) and falls to 0 at edges.
        const t = Math.abs(k) / (peakDipFrames + 1);
        const weight = Math.cos((t * Math.PI) / 2) ** 2;
        slewed[idx] -= excessDb * weight;
      }
      // Track for diagnostics: the worst dip applied within each run.
      if (runIdx >= 0) {
        peakReductionDbByRun[runIdx] = Math.min(peakReductionDbByRun[runIdx], -excessDb);
      }
    }
  }

  // 8) Convert to linear.
  const gainCurve = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i += 1) gainCurve[i] = dbToLin(slewed[i]);

  const breathRunCount = runMeta.filter((m) => m.runClass === "transient-breath").length;
  return {
    gainCurve,
    runs: runMeta.map((m, i) => ({
      startFrame: m.startFrame,
      endFrame: m.endFrame,
      meanDb: m.meanDb,
      crestDb: m.crestDb,
      plannedGainDb: plannedRunGainDb[i] ?? 0,
      peakReducedDb: peakReductionDbByRun[i] ?? 0,
      runClass: m.runClass,
    })),
    expanderDepthDb,
    targetDb,
    microRideDb,
    breathRunCount,
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
