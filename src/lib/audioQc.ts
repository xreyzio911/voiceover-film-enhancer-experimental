"use client";

export const AUDIO_QC_FRAME_MS = 10;
const AUDIO_QC_FLOOR_DB = -120;
const COLD_OPEN_HEAD_MS = 2500;
const COLD_OPEN_RUN_COUNT = 3;
const COLD_OPEN_WARN_DB = 1.5;
const COLD_OPEN_FLAG_DB = 2.5;
const COLD_OPEN_RISK_FLOOR_DB = 1.0;
const COLD_OPEN_RISK_SPAN_DB = 4.0;
const END_EDGE_TAIL_MS = 150;
const END_EDGE_BODY_MS = 450;
const SOFT_TAIL_HOLD_MS = 180;
const SOFT_TAIL_FLOOR_MARGIN_DB = 14;

export type AudioQcMetrics = {
  peakDb: number | null;
  clipPct: number | null;
  noiseFloorDb: number;
  pauseNoiseFloorDb: number;
  nearSpeechNoiseFloorDb: number | null;
  speechThresholdDb: number;
  noiseContrastDb: number;
  speechRatioPct: number;
  speechDutyCyclePct: number;
  speechSegmentCount: number;
  medianSpeechRunMs: number;
  longSilenceCount: number;
  dynamicRangeDb: number;
  instabilityScore: number;
  onsetOvershootScore: number;
  midLineSagScore: number;
  endFadeRiskScore: number;
  endEdgeDipDb: number;
  lineSwingScore: number;
  sentenceJumpScore: number;
  coldOpenDipDb: number;
  coldOpenRiskScore: number;
  breathSpikeRisk: number;
  pauseNoiseRisk: number;
  compressionScore: number;
  clickScore: number;
  reverbScore: number;
  echoScore: number;
  roomScore: number;
  echoDelayMs: number | null;
  analysisConfidence: number;
  drynessScore: number;
  overallRisk: number;
  /**
   * Long-term band energy in dB. Populated only by `analyzeFloatSamplesWithSpectrum`
   * (the core frame-based analyzer does not see raw samples). Null means not
   * computed for this run.
   */
  bandSpectrumDb: number[] | null;
  /** Sibilance score 0..1 derived from the band spectrum. Zero if spectrum not computed. */
  sibilanceScore: number;
};

export type AudioQcAdvice = {
  flags: string[];
  recommendations: string[];
};

type FrameRun = {
  start: number;
  end: number;
};

type SpeechMaskOptions = {
  frameMs?: number;
  openLiftDb?: number;
  closeLiftDb?: number;
  openConfirmMs?: number;
  minSpeechHoldMs?: number;
  minSpeechRunMs?: number;
  gapBridgeMs?: number;
};

type AudioQcFrameOptions = {
  sampleRate: number;
  durationSec: number;
  frameMs?: number;
  peakDb?: number | null;
  clipPct?: number | null;
  sampleSpikeCount?: number;
};

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export const percentile = (values: number[], percent: number) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (clamp(percent, 0, 100) / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return sorted[lower];
  const weight = rank - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};

export const median = (values: number[]) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
};

export const mean = (values: number[]) => {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

export const toDb = (value: number) => {
  if (value <= 0) return AUDIO_QC_FLOOR_DB;
  return 20 * Math.log10(value);
};

const smoothSeries = (values: number[], radius = 1) =>
  values.map((_, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(values.length - 1, index + radius);
    let sum = 0;
    for (let cursor = start; cursor <= end; cursor += 1) {
      sum += values[cursor];
    }
    return sum / Math.max(end - start + 1, 1);
  });

const finalizeMaskRuns = (
  mask: boolean[],
  minSpeechFrames: number,
  gapBridgeFrames: number
) => {
  let runStart = -1;
  for (let index = 0; index <= mask.length; index += 1) {
    const active = index < mask.length ? mask[index] : false;
    if (active && runStart < 0) {
      runStart = index;
      continue;
    }
    if (active || runStart < 0) continue;
    if (index - runStart < minSpeechFrames) {
      for (let cursor = runStart; cursor < index; cursor += 1) {
        mask[cursor] = false;
      }
    }
    runStart = -1;
  }

  let gapStart = -1;
  for (let index = 0; index <= mask.length; index += 1) {
    const active = index < mask.length ? mask[index] : true;
    if (!active && gapStart < 0) {
      gapStart = index;
      continue;
    }
    if (!active || gapStart < 0) continue;
    const gapFrames = index - gapStart;
    if (
      gapFrames <= gapBridgeFrames &&
      gapStart > 0 &&
      index < mask.length &&
      mask[gapStart - 1] &&
      mask[index]
    ) {
      for (let cursor = gapStart; cursor < index; cursor += 1) {
        mask[cursor] = true;
      }
    }
    gapStart = -1;
  }

  return mask;
};

export const buildSpeechMask = (
  frameDb: number[],
  noiseFloorDb: number,
  options?: SpeechMaskOptions
) => {
  const frameMs = options?.frameMs ?? AUDIO_QC_FRAME_MS;
  const openThresholdDb = clamp(
    noiseFloorDb + (options?.openLiftDb ?? 11.5),
    -58,
    -24
  );
  const closeThresholdDb = clamp(
    noiseFloorDb + (options?.closeLiftDb ?? 8.5),
    -60,
    -26
  );
  const minSpeechHoldFrames = Math.max(
    1,
    Math.round((options?.minSpeechHoldMs ?? 140) / frameMs)
  );
  const softTailHoldFrames = Math.max(
    minSpeechHoldFrames,
    Math.round(SOFT_TAIL_HOLD_MS / frameMs)
  );
  const openConfirmFrames = Math.max(
    1,
    Math.round((options?.openConfirmMs ?? 35) / frameMs)
  );
  const minSpeechRunFrames = Math.max(
    1,
    Math.round((options?.minSpeechRunMs ?? 60) / frameMs)
  );
  const gapBridgeFrames = Math.max(
    0,
    Math.round((options?.gapBridgeMs ?? 90) / frameMs)
  );

  const smoothed = smoothSeries(frameDb, 1);
  const mask = new Array<boolean>(smoothed.length).fill(false);
  let active = false;
  let holdFrames = 0;
  let softTailDecayActive = false;

  for (let index = 0; index < smoothed.length; index += 1) {
    const db = smoothed[index];
    if (!active) {
      if (db >= openThresholdDb) {
        let confirmCount = 0;
        const confirmEnd = Math.min(smoothed.length - 1, index + openConfirmFrames + 1);
        for (let cursor = index; cursor <= confirmEnd; cursor += 1) {
          if (smoothed[cursor] >= closeThresholdDb) {
            confirmCount += 1;
          }
        }
        if (confirmCount < openConfirmFrames) {
          continue;
        }
        active = true;
        holdFrames = minSpeechHoldFrames;
        mask[index] = true;
      }
      continue;
    }

    if (db >= openThresholdDb) {
      holdFrames = minSpeechHoldFrames;
      softTailDecayActive = false;
    } else if (db >= closeThresholdDb) {
      holdFrames = Math.max(holdFrames, Math.ceil(minSpeechHoldFrames * 0.45));
      softTailDecayActive = false;
    } else {
      const softTailFloorDb = noiseFloorDb + SOFT_TAIL_FLOOR_MARGIN_DB;
      const rawDb = frameDb[index] ?? db;
      if (db >= softTailFloorDb && rawDb >= softTailFloorDb) {
        if (!softTailDecayActive) {
          holdFrames = Math.max(holdFrames, softTailHoldFrames);
          softTailDecayActive = true;
        }
        holdFrames -= 1;
      } else {
        softTailDecayActive = false;
        const closeGapDb = closeThresholdDb - db;
        holdFrames -= closeGapDb >= 8 ? 3 : closeGapDb >= 4 ? 2 : 1;
      }
    }

    if (holdFrames < 0) {
      active = false;
      holdFrames = 0;
      softTailDecayActive = false;
      continue;
    }

    mask[index] = true;
  }

  return finalizeMaskRuns(mask, minSpeechRunFrames, gapBridgeFrames);
};

const collectRuns = (mask: boolean[]) => {
  const speechRuns: FrameRun[] = [];
  const silenceRuns: FrameRun[] = [];
  let runStart = 0;
  let runIsSpeech = mask[0] ?? false;

  for (let index = 1; index <= mask.length; index += 1) {
    const currentIsSpeech = index < mask.length ? mask[index] : !runIsSpeech;
    if (index < mask.length && currentIsSpeech === runIsSpeech) continue;
    const runEnd = index;
    if (runEnd > runStart) {
      const run = { start: runStart, end: runEnd };
      if (runIsSpeech) {
        speechRuns.push(run);
      } else {
        silenceRuns.push(run);
      }
    }
    runStart = index;
    runIsSpeech = currentIsSpeech;
  }

  return { speechRuns, silenceRuns };
};

const meanSlice = (values: number[], start: number, end: number) => {
  const safeStart = Math.max(0, start);
  const safeEnd = Math.min(values.length, end);
  if (safeEnd <= safeStart) return null;
  let sum = 0;
  for (let index = safeStart; index < safeEnd; index += 1) {
    sum += values[index];
  }
  return sum / (safeEnd - safeStart);
};

const powerMeanDbSlice = (values: number[], start: number, end: number) => {
  const safeStart = Math.max(0, start);
  const safeEnd = Math.min(values.length, end);
  if (safeEnd <= safeStart) return null;
  let sumPower = 0;
  let count = 0;
  for (let index = safeStart; index < safeEnd; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    sumPower += Math.pow(10, value / 10);
    count += 1;
  }
  if (count === 0) return null;
  return 10 * Math.log10(sumPower / count + 1e-30);
};

const computeLineSwingScore = (
  frameDb: number[],
  speechRuns: FrameRun[]
) => {
  const events: number[] = [];
  for (const run of speechRuns) {
    const runFrames = run.end - run.start;
    if (runFrames < 80) continue;
    const bucketCount = Math.min(5, Math.max(3, Math.floor(runFrames / 18)));
    const bucketMeans: number[] = [];
    for (let bucket = 0; bucket < bucketCount; bucket += 1) {
      const start = run.start + Math.floor((bucket / bucketCount) * runFrames);
      const end = run.start + Math.floor(((bucket + 1) / bucketCount) * runFrames);
      const bucketMean = meanSlice(frameDb, start, Math.max(start + 1, end));
      if (bucketMean !== null) {
        bucketMeans.push(bucketMean);
      }
    }
    if (bucketMeans.length < 3) continue;
    const diffs: number[] = [];
    for (let index = 1; index < bucketMeans.length; index += 1) {
      diffs.push(bucketMeans[index] - bucketMeans[index - 1]);
    }
    const strongDiffs = diffs.filter((value) => Math.abs(value) >= 0.65);
    let signChanges = 0;
    for (let index = 1; index < strongDiffs.length; index += 1) {
      if (strongDiffs[index] * strongDiffs[index - 1] < 0) {
        signChanges += 1;
      }
    }
    const swingRangeDb = Math.max(...bucketMeans) - Math.min(...bucketMeans);
    const eventScore = clamp(
      clamp((swingRangeDb - 3.0) / 4.5, 0, 1) * 0.62 +
        clamp((signChanges - 1) / 2, 0, 1) * 0.38,
      0,
      1
    );
    events.push(eventScore);
  }

  return clamp(percentile(events, 75) ?? 0, 0, 1);
};

const computeSentenceJumpScore = (
  frameDb: number[],
  speechRuns: FrameRun[],
  frameMs: number,
  speechDutyCyclePct: number
) => {
  const sparseMode = speechDutyCyclePct < 6 || speechRuns.length <= 8;
  const bodyMeans: Array<{ run: FrameRun; meanDb: number; medianDb: number }> = [];
  for (const run of speechRuns) {
    const runFrames = run.end - run.start;
    if (runFrames < 55) continue;
    const edgeTrimFrames = Math.max(6, Math.min(18, Math.floor(runFrames * 0.14)));
    const bodyStart = Math.min(run.end - 1, run.start + edgeTrimFrames);
    const bodyEnd = Math.max(bodyStart + 1, run.end - edgeTrimFrames);
    const meanDb = meanSlice(frameDb, bodyStart, bodyEnd) ?? meanSlice(frameDb, run.start, run.end);
    const bodyMedianDb =
      median(frameDb.slice(bodyStart, bodyEnd)) ??
      median(frameDb.slice(run.start, run.end)) ??
      meanDb;
    if (meanDb !== null && bodyMedianDb !== null) {
      bodyMeans.push({ run, meanDb, medianDb: bodyMedianDb });
    }
  }

  const gapFloorFrames = Math.max(1, Math.round((sparseMode ? 80 : 120) / frameMs));
  const jumps: number[] = [];
  for (let index = 1; index < bodyMeans.length; index += 1) {
    const previous = bodyMeans[index - 1];
    const current = bodyMeans[index];
    const gapFrames = Math.max(0, current.run.start - previous.run.end);
    if (gapFrames < gapFloorFrames) continue;
    const bodyJumpDb = Math.abs(
      (sparseMode ? current.medianDb : current.meanDb) - (sparseMode ? previous.medianDb : previous.meanDb)
    );
    const supportJumpDb = Math.abs(current.meanDb - previous.meanDb);
    const jumpDb = sparseMode ? Math.max(bodyJumpDb, supportJumpDb * 0.8) : bodyJumpDb;
    const gapNorm = clamp(((gapFrames * frameMs) - (sparseMode ? 80 : 120)) / (sparseMode ? 700 : 900), 0, 1);
    const jumpNorm = clamp((jumpDb - (sparseMode ? 1.2 : 1.8)) / (sparseMode ? 3.3 : 4.2), 0, 1);
    jumps.push(clamp(jumpNorm * (sparseMode ? 0.9 : 0.82) + gapNorm * (sparseMode ? 0.1 : 0.18), 0, 1));
  }

  return clamp(percentile(jumps, sparseMode ? 80 : 75) ?? 0, 0, 1);
};

const computeEdgeTrimmedRunBodyDb = (frameDb: number[], run: FrameRun) => {
  const runFrames = run.end - run.start;
  if (runFrames < 12) return null;
  const edgeTrimFrames =
    runFrames < 55
      ? Math.max(1, Math.min(4, Math.floor(runFrames * 0.1)))
      : Math.max(6, Math.min(18, Math.floor(runFrames * 0.14)));
  const bodyStart = Math.min(run.end - 1, run.start + edgeTrimFrames);
  const bodyEnd = Math.max(bodyStart + 1, run.end - edgeTrimFrames);
  return powerMeanDbSlice(frameDb, bodyStart, bodyEnd) ?? powerMeanDbSlice(frameDb, run.start, run.end);
};

const computeColdOpenMetrics = (
  frameDb: number[],
  speechRuns: FrameRun[],
  frameMs: number,
) => {
  if (speechRuns.length < 2) {
    return { coldOpenDipDb: 0, coldOpenRiskScore: 0 };
  }

  const maxHeadFrames = Math.max(1, Math.round(COLD_OPEN_HEAD_MS / frameMs));
  const runLimit = Math.min(COLD_OPEN_RUN_COUNT, Math.max(1, speechRuns.length - 1));
  const headBodyDb: number[] = [];
  let remainingHeadFrames = maxHeadFrames;
  let headWindowEndFrame = speechRuns[0].start;

  for (let runIndex = 0; runIndex < runLimit && remainingHeadFrames > 0; runIndex += 1) {
    const run = speechRuns[runIndex];
    const takeFrames = Math.min(run.end - run.start, remainingHeadFrames);
    const bodyDb = computeEdgeTrimmedRunBodyDb(frameDb, { start: run.start, end: run.start + takeFrames });
    if (bodyDb !== null) headBodyDb.push(bodyDb);
    remainingHeadFrames -= takeFrames;
    headWindowEndFrame = run.start + takeFrames;
  }

  if (headBodyDb.length === 0) {
    return { coldOpenDipDb: 0, coldOpenRiskScore: 0 };
  }

  const headSpeechDb = Math.min(...headBodyDb);
  if (!Number.isFinite(headSpeechDb)) {
    return { coldOpenDipDb: 0, coldOpenRiskScore: 0 };
  }

  const bodyMeans: number[] = [];
  for (const run of speechRuns) {
    if (run.start < headWindowEndFrame) continue;
    const bodyDb = computeEdgeTrimmedRunBodyDb(frameDb, run);
    if (bodyDb !== null) bodyMeans.push(bodyDb);
  }

  const bodySpeechDb = median(bodyMeans);
  if (bodySpeechDb === null) {
    return { coldOpenDipDb: 0, coldOpenRiskScore: 0 };
  }

  const coldOpenDipDb = bodySpeechDb - headSpeechDb;
  const coldOpenRiskScore = clamp((coldOpenDipDb - COLD_OPEN_RISK_FLOOR_DB) / COLD_OPEN_RISK_SPAN_DB, 0, 1);
  return { coldOpenDipDb, coldOpenRiskScore };
};

const computeEndEdgeDipDb = (
  frameDb: number[],
  speechRuns: FrameRun[],
  frameMs: number,
) => {
  const tailFrames = Math.max(3, Math.round(END_EDGE_TAIL_MS / frameMs));
  const bodyFrames = Math.max(tailFrames, Math.round(END_EDGE_BODY_MS / frameMs));
  const minRunFrames = tailFrames + Math.max(8, Math.round(120 / frameMs));
  let worstDipDb = 0;

  for (const run of speechRuns) {
    const runFrames = run.end - run.start;
    if (runFrames < minRunFrames) continue;
    const tailStart = Math.max(run.start, run.end - tailFrames);
    const bodyEnd = Math.max(run.start + 1, tailStart);
    const bodyStart = Math.max(run.start, bodyEnd - bodyFrames);
    const bodyDb = powerMeanDbSlice(frameDb, bodyStart, bodyEnd);
    const tailDb = powerMeanDbSlice(frameDb, tailStart, run.end);
    if (bodyDb === null || tailDb === null) continue;
    worstDipDb = Math.max(worstDipDb, bodyDb - tailDb);
  }

  return Math.max(0, worstDipDb);
};

export const buildFlagsAndRecommendations = (
  metrics: Pick<
    AudioQcMetrics,
    | "pauseNoiseRisk"
    | "instabilityScore"
    | "onsetOvershootScore"
    | "midLineSagScore"
    | "endFadeRiskScore"
    | "compressionScore"
    | "clickScore"
    | "echoScore"
    | "clipPct"
    | "lineSwingScore"
    | "sentenceJumpScore"
    | "coldOpenDipDb"
    | "breathSpikeRisk"
  >
): AudioQcAdvice => {
  const flags: string[] = [];
  const recommendations: string[] = [];

  if (metrics.pauseNoiseRisk >= 0.45) {
    flags.push("Noise uplift risk in pauses.");
    recommendations.push("Use stricter pause-noise protection and verify long silences after leveling.");
  }

  if (metrics.instabilityScore >= 0.5) {
    flags.push("Large voice-level jumps detected.");
    recommendations.push("Use instability-safe leveling with less downstream gain-riding.");
  }

  if (metrics.lineSwingScore >= 0.38) {
    flags.push("In-line high-low-high swing detected.");
    recommendations.push("Bias the leveler toward stronger line continuity on this take.");
  }

  if (metrics.sentenceJumpScore >= 0.34) {
    flags.push("Sentence-to-sentence level mismatch detected.");
    recommendations.push("Prefer continuity-safe single-pass leveling when grouped lines do not sit evenly.");
  }

  if (metrics.coldOpenDipDb >= COLD_OPEN_FLAG_DB) {
    flags.push("cold_open_dip: first speech runs sit below the later dialogue body.");
    recommendations.push("Lift and recheck the first few spoken words against the later body level.");
  } else if (metrics.coldOpenDipDb >= COLD_OPEN_WARN_DB) {
    flags.push("Cold-open level dip warning.");
    recommendations.push("Verify that the opening words do not tuck under the later dialogue body.");
  }

  if (metrics.onsetOvershootScore >= 0.38) {
    flags.push("First-word/onset spike risk detected.");
    recommendations.push("Use onset spike taming before auto-leveling on line starts.");
  }

  if (metrics.breathSpikeRisk >= 0.34) {
    flags.push("Breath or inhale spikes are standing above line level.");
    recommendations.push("Apply breath-spike taming ahead of the leveler so inhales are not promoted.");
  }

  if (metrics.midLineSagScore >= 0.36) {
    flags.push("Mid-line level sag detected.");
    recommendations.push("Use faster ride response through the body of the line.");
  }

  if (metrics.endFadeRiskScore >= 0.35) {
    flags.push("Sentence-end audibility loss risk detected.");
    recommendations.push("Protect endings and reduce tail-gating on clean unstable takes.");
  }

  if (metrics.compressionScore >= 0.52) {
    flags.push("Compressed or radio-like tone risk.");
    recommendations.push("Relax compressor ratio and mix to preserve performance movement.");
  }

  if (metrics.clickScore >= 0.2) {
    flags.push("Click/transient artifacts detected.");
    recommendations.push("Run click-taming cleanup and recheck consonant spikes.");
  }

  if (metrics.echoScore >= 0.38) {
    flags.push("Echo/reverb tail risk detected.");
    recommendations.push("Keep room cleanup on and verify tail control plus notch behavior.");
  }

  if ((metrics.clipPct ?? 0) >= 0.02) {
    flags.push("Potential clipped peaks in source.");
    recommendations.push("Ask for a cleaner source if clipped peaks are audible.");
  }

  if (flags.length === 0) {
    flags.push("No major QC risk detected.");
    recommendations.push("Source looks stable for standard optimization.");
  }

  return { flags, recommendations };
};

export const analyzeFrameAudio = (
  frameRms: number[],
  framePeak: number[],
  frameDb: number[],
  frameSharpness: number[],
  options: AudioQcFrameOptions
): AudioQcMetrics => {
  const frameCount = frameDb.length;
  if (frameCount < 30) {
    return {
      peakDb: options.peakDb ?? null,
      clipPct: options.clipPct ?? null,
      noiseFloorDb: AUDIO_QC_FLOOR_DB,
      pauseNoiseFloorDb: AUDIO_QC_FLOOR_DB,
      nearSpeechNoiseFloorDb: null,
      speechThresholdDb: AUDIO_QC_FLOOR_DB,
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
      overallRisk: 0,
      bandSpectrumDb: null,
      sibilanceScore: 0,
    };
  }

  const frameMs = options.frameMs ?? AUDIO_QC_FRAME_MS;
  const baseNoiseFloorDb = percentile(frameDb, 25) ?? -72;
  const speechMask = buildSpeechMask(frameDb, baseNoiseFloorDb, { frameMs });
  const { speechRuns, silenceRuns } = collectRuns(speechMask);

  const speechDb: number[] = [];
  const pauseDb: number[] = [];
  const speechCrest: number[] = [];
  const nearSpeechNoiseDb: number[] = [];
  const nearSpeechPauseFrames: Array<{ db: number; peakDb: number; crestDb: number; sharpnessDb: number }> = [];
  let activeSpeechFrames = 0;
  let nonSpeechFrames = 0;
  let speechClickFrames = 0;
  let pauseClickFrames = 0;

  const speechContextFrames = Math.round(0.35 / (frameMs / 1000));
  for (let frame = 0; frame < frameCount; frame += 1) {
    const crestDb = toDb((framePeak[frame] + 1e-9) / (frameRms[frame] + 1e-9));
    const peakFrameDb = toDb(framePeak[frame] + 1e-12);
    if (speechMask[frame]) {
      activeSpeechFrames += 1;
      speechDb.push(frameDb[frame]);
      speechCrest.push(crestDb);
      if ((crestDb > 24 && peakFrameDb > -8) || (frameSharpness[frame] > -24 && peakFrameDb > -16)) {
        speechClickFrames += 1;
      }
      continue;
    }

    nonSpeechFrames += 1;
    pauseDb.push(frameDb[frame]);
    if ((crestDb > 18 && peakFrameDb > -24) || (frameSharpness[frame] > -28 && peakFrameDb > -30)) {
      pauseClickFrames += 1;
    }

    const from = Math.max(0, frame - speechContextFrames);
    const to = Math.min(frameCount - 1, frame + speechContextFrames);
    for (let cursor = from; cursor <= to; cursor += 1) {
      if (speechMask[cursor]) {
        nearSpeechNoiseDb.push(frameDb[frame]);
        nearSpeechPauseFrames.push({
          db: frameDb[frame],
          peakDb: peakFrameDb,
          crestDb,
          sharpnessDb: frameSharpness[frame],
        });
        break;
      }
    }
  }

  const pauseNoiseFloorDb = clamp(
    percentile(pauseDb.length > 10 ? pauseDb : frameDb, 70) ?? baseNoiseFloorDb,
    -110,
    -28
  );
  const nearSpeechNoiseFloorDb =
    nearSpeechNoiseDb.length > 0 ? clamp(percentile(nearSpeechNoiseDb, 72) ?? -90, -110, -28) : null;
  const noiseFloorDb = clamp(
    Math.max(baseNoiseFloorDb, pauseNoiseFloorDb, nearSpeechNoiseFloorDb ?? AUDIO_QC_FLOOR_DB),
    -110,
    -28
  );
  const speechThresholdDb = clamp(noiseFloorDb + 10.5, -58, -24);
  const speechRatioPct = (speechDb.length / Math.max(frameCount, 1)) * 100;
  const speechDutyCyclePct = (activeSpeechFrames / Math.max(frameCount, 1)) * 100;
  const speechRunMs = speechRuns.map((run) => (run.end - run.start) * frameMs);
  const silenceRunMs = silenceRuns.map((run) => (run.end - run.start) * frameMs);
  const medianSpeechRunMs = median(speechRunMs) ?? 0;
  const longSilenceCount = silenceRunMs.filter((duration) => duration >= 1500).length;

  const p90Speech = percentile(speechDb, 90) ?? -24;
  const p10Speech = percentile(speechDb, 10) ?? p90Speech;
  const p50Speech = percentile(speechDb, 50) ?? p90Speech;
  const dynamicRangeDb = Math.max(0, p90Speech - p10Speech);
  const pauseP80 = percentile(pauseDb, 80) ?? pauseNoiseFloorDb;
  const noiseContrastDb = clamp(p50Speech - pauseP80, 0, 80);

  let instabilityScore = 0;
  if (speechDb.length >= 12) {
    const smoothSpeech = smoothSeries(speechDb, 1);
    const deltas: number[] = [];
    for (let index = 1; index < smoothSpeech.length; index += 1) {
      deltas.push(Math.abs(smoothSpeech[index] - smoothSpeech[index - 1]));
    }
    const p80 = percentile(deltas, 80) ?? 0;
    const p95 = percentile(deltas, 95) ?? 0;
    instabilityScore = clamp(
      clamp((p80 - 1.2) / 3.1, 0, 1) * 0.62 +
        clamp((p95 - 2.2) / 4.6, 0, 1) * 0.38,
      0,
      1
    );
  }

  const onsetEventScores: number[] = [];
  const midSagEventScores: number[] = [];
  const endFadeEventScores: number[] = [];
  const speechFloorGuardDb = speechThresholdDb - 8;
  for (let runIndex = 0; runIndex < speechRuns.length; runIndex += 1) {
    const run = speechRuns[runIndex];
    const runFrames = run.end - run.start;
    const prevEnd = runIndex > 0 ? speechRuns[runIndex - 1].end : 0;
    const nextStart = runIndex + 1 < speechRuns.length ? speechRuns[runIndex + 1].start : frameCount;
    const preSilenceFrames = Math.max(0, run.start - prevEnd);
    const postSilenceFrames = Math.max(0, nextStart - run.end);

    if (preSilenceFrames >= 20 && runFrames >= 70) {
      const onsetDb = meanSlice(frameDb, run.start + 12, Math.min(run.end, run.start + 22));
      const bodyDb = meanSlice(frameDb, run.start + 25, Math.min(run.end, run.start + 70));
      if (onsetDb !== null && bodyDb !== null) {
        onsetEventScores.push(clamp((onsetDb - bodyDb - 2.5) / 4.5, 0, 1));
      }
    }

    if (runFrames >= 90) {
      const third = Math.floor(runFrames / 3);
      const startDb = meanSlice(frameDb, run.start, run.start + third);
      const midDb = meanSlice(frameDb, run.start + third, run.start + third * 2);
      const endDb = meanSlice(frameDb, run.start + third * 2, run.end);
      if (startDb !== null && midDb !== null && endDb !== null) {
        const edgeMean = (startDb + endDb) / 2;
        midSagEventScores.push(clamp((edgeMean - midDb - 1.8) / 4.2, 0, 1));
      }
    }

    if (runFrames >= 70 && postSilenceFrames >= 10) {
      const preTailDb = meanSlice(frameDb, Math.max(run.start, run.end - 50), Math.max(run.start, run.end - 22));
      const tailDb = meanSlice(frameDb, Math.max(run.start, run.end - 22), run.end);
      if (preTailDb !== null && tailDb !== null && tailDb > speechFloorGuardDb) {
        endFadeEventScores.push(clamp((preTailDb - tailDb - 3.2) / 4.8, 0, 1));
      }
    }
  }

  const onsetOvershootScore = clamp(percentile(onsetEventScores, 75) ?? 0, 0, 1);
  const midLineSagScore = clamp(percentile(midSagEventScores, 75) ?? 0, 0, 1);
  const endFadeRiskScore = clamp(percentile(endFadeEventScores, 75) ?? 0, 0, 1);
  const endEdgeDipDb = computeEndEdgeDipDb(frameDb, speechRuns, frameMs);
  const lineSwingScore = computeLineSwingScore(frameDb, speechRuns);
  const sentenceJumpScore = computeSentenceJumpScore(frameDb, speechRuns, frameMs, speechDutyCyclePct);
  const { coldOpenDipDb, coldOpenRiskScore } = computeColdOpenMetrics(frameDb, speechRuns, frameMs);

  const breathSpikeEventScores: number[] = [];
  for (const pauseFrame of nearSpeechPauseFrames) {
    const rmsLift = clamp((pauseFrame.db - (pauseNoiseFloorDb + 9)) / 9.5, 0, 1);
    const peakLift = clamp((pauseFrame.peakDb - (speechThresholdDb - 8.5)) / 9, 0, 1);
    const proximityLift = Math.max(rmsLift, peakLift);
    if (proximityLift <= 0) continue;
    const breathCrest = clamp((pauseFrame.crestDb - 6.5) / 7.5, 0, 1);
    const breathSoftness = 1 - clamp((pauseFrame.sharpnessDb + 32) / 5.5, 0, 1);
    const eventScore = clamp(
      (rmsLift * 0.48 + peakLift * 0.28 + breathCrest * 0.14 + breathSoftness * 0.1) * proximityLift,
      0,
      1
    );
    if (eventScore >= 0.08) {
      breathSpikeEventScores.push(eventScore);
    }
  }
  const breathRunScores: number[] = [];
  const leadInBreathScores: number[] = [];
  for (let runIndex = 0; runIndex < speechRuns.length; runIndex += 1) {
    const run = speechRuns[runIndex];
    const runFrames = run.end - run.start;
    const previousEnd = runIndex > 0 ? speechRuns[runIndex - 1].end : 0;
    const leadWindowFrames = Math.min(Math.max(0, run.start - previousEnd), Math.max(1, Math.round(250 / frameMs)));
    if (leadWindowFrames >= Math.max(3, Math.round(40 / frameMs)) && runFrames >= 28) {
      const followingBodyStart = Math.min(run.end - 1, run.start + Math.max(2, Math.round(40 / frameMs)));
      const followingBodyEnd = Math.min(run.end, run.start + Math.max(6, Math.round(120 / frameMs)));
      const followingMeanDb =
        meanSlice(frameDb, followingBodyStart, Math.max(followingBodyStart + 1, followingBodyEnd)) ??
        meanSlice(frameDb, run.start, Math.min(run.end, run.start + Math.max(8, Math.round(160 / frameMs))));
      if (followingMeanDb !== null) {
        const leadStart = run.start - leadWindowFrames;
        let transientPeakDb = AUDIO_QC_FLOOR_DB;
        let transientFrames = 0;
        for (let frame = leadStart; frame < run.start; frame += 1) {
          const peakDb = toDb((framePeak[frame] ?? 0) + 1e-12);
          const rmsDb = frameDb[frame] ?? AUDIO_QC_FLOOR_DB;
          const exceedFollowing = peakDb - followingMeanDb;
          if (exceedFollowing < 5 || rmsDb >= followingMeanDb - 0.4) continue;
          transientFrames += 1;
          transientPeakDb = Math.max(transientPeakDb, peakDb);
        }
        if (transientFrames > 0) {
          const exceedNorm = clamp((transientPeakDb - followingMeanDb - 5) / 7, 0, 1);
          const transientShortNorm = clamp(
            (Math.max(1, Math.round(120 / frameMs)) - transientFrames) / Math.max(1, Math.round(120 / frameMs)),
            0,
            1
          );
          const proximityNorm = clamp((leadWindowFrames - transientFrames) / Math.max(leadWindowFrames, 1), 0, 1);
          leadInBreathScores.push(clamp(exceedNorm * 0.58 + transientShortNorm * 0.24 + proximityNorm * 0.18, 0, 1));
        }
      }
    }

    if (runFrames < 4 || runFrames > 26) continue;
    const nextRun = runIndex + 1 < speechRuns.length ? speechRuns[runIndex + 1] : null;
    if (!nextRun) continue;
    const nextRunFrames = nextRun.end - nextRun.start;
    const gapAfterFrames = Math.max(0, nextRun.start - run.end);
    if (nextRunFrames < 40 || gapAfterFrames < 2 || gapAfterFrames > 80) continue;

    const runMeanDb = meanSlice(frameDb, run.start, run.end);
    if (runMeanDb === null) continue;
    let runPeakAmp = 0;
    for (let frame = run.start; frame < run.end; frame += 1) {
      runPeakAmp = Math.max(runPeakAmp, framePeak[frame] ?? 0);
    }
    const runPeakDb = toDb(runPeakAmp + 1e-12);
    const shortNorm = clamp((26 - runFrames) / 18, 0, 1);
    const crestNorm = clamp((runPeakDb - runMeanDb - 7) / 8, 0, 1);
    const levelNorm = clamp((runPeakDb - (speechThresholdDb - 6)) / 10, 0, 1);
    const contrastNorm = clamp(((p50Speech ?? -28) - runMeanDb - 1.5) / 8, 0, 1);
    const gapNorm = clamp((80 - gapAfterFrames) / 80, 0, 1);
    breathRunScores.push(
      clamp(shortNorm * 0.25 + crestNorm * 0.28 + levelNorm * 0.22 + contrastNorm * 0.15 + gapNorm * 0.1, 0, 1)
    );
  }
  const breathSpikeRisk = clamp(
    Math.max(
      clamp(percentile(breathSpikeEventScores, 80) ?? 0, 0, 1) * 0.78 +
        clamp((breathSpikeEventScores.length - 2) / 6, 0, 1) * 0.22,
      clamp(percentile(breathRunScores, 75) ?? 0, 0, 1) * 0.82 +
        clamp((breathRunScores.length - 1) / 4, 0, 1) * 0.18,
      clamp(percentile(leadInBreathScores, 80) ?? 0, 0, 1) * 0.86 +
        clamp((leadInBreathScores.length - 1) / 4, 0, 1) * 0.14
    ),
    0,
    1
  );

  const centered = frameRms.map((value) => value - mean(frameRms));
  let bestCorr = 0;
  let bestEchoLagFrames = 0;
  for (let lag = 4; lag <= 20; lag += 1) {
    let num = 0;
    let denA = 0;
    let denB = 0;
    for (let index = 0; index < centered.length - lag; index += 1) {
      const a = centered[index];
      const b = centered[index + lag];
      num += a * b;
      denA += a * a;
      denB += b * b;
    }
    const denom = Math.sqrt(denA * denB) + 1e-12;
    const corr = num / denom;
    if (corr > bestCorr) {
      bestCorr = corr;
      bestEchoLagFrames = lag;
    }
  }

  const tailEventScores: number[] = [];
  let speechRun = 0;
  for (let frame = 0; frame < frameCount - 30; frame += 1) {
    if (speechMask[frame]) {
      speechRun += 1;
      continue;
    }
    if (speechRun >= 6) {
      const early = mean(frameRms.slice(frame + 1, frame + 8));
      const late = mean(frameRms.slice(frame + 14, frame + 28));
      const pauseFloorAmp = Math.pow(10, pauseNoiseFloorDb / 20);
      const lateLiftDb = toDb((late + 1e-9) / (pauseFloorAmp + 1e-9));
      const decayDb = toDb((early + 1e-9) / (late + 1e-9));
      const eventScore = clamp(
        clamp((lateLiftDb - 3.5) / 10.5, 0, 1) * 0.58 +
          clamp((5.5 - decayDb) / 5.5, 0, 1) * 0.42,
        0,
        1
      );
      tailEventScores.push(eventScore);
    }
    speechRun = 0;
  }
  const reverbScore = clamp(tailEventScores.length > 0 ? mean(tailEventScores) : 0, 0, 1);
  const echoCorrelationScore = clamp((bestCorr - 0.22) / 0.22, 0, 1);
  const echoTailSupport = clamp(0.18 + clamp((reverbScore - 0.1) / 0.3, 0, 1) * 0.82, 0.18, 1);
  const echoScore = clamp(
    reverbScore * 0.72 + echoCorrelationScore * 0.28 * echoTailSupport,
    0,
    1
  );

  const speechClickDensity = speechClickFrames / Math.max(activeSpeechFrames, 1);
  const pauseClickDensity = pauseClickFrames / Math.max(nonSpeechFrames, 1);
  const sampleClicksPerMinute =
    ((options.sampleSpikeCount ?? 0) / Math.max(options.durationSec, 1e-6)) * 60;
  const clickScore = clamp(
    pauseClickDensity * 1.45 + speechClickDensity * 0.55 + sampleClicksPerMinute / 30,
    0,
    1
  );

  const crestSpeechDb = speechCrest.length > 0 ? mean(speechCrest) : 12;
  const compressionScore = clamp(
    clamp((10.2 - dynamicRangeDb) / 5.2, 0, 1) * 0.68 +
      clamp((11.5 - crestSpeechDb) / 5.8, 0, 1) * 0.32,
    0,
    1
  );
  const pauseNoiseRisk = clamp(
    clamp((pauseNoiseFloorDb + 62) / 18, 0, 1) * 0.5 +
      clamp((22 - noiseContrastDb) / 14, 0, 1) * 0.35 +
      breathSpikeRisk * 0.15,
    0,
    1
  );
  const roomScore = clamp(reverbScore * 0.62 + echoScore * 0.28 + pauseNoiseRisk * 0.1, 0, 1);

  const speechCoverage = clamp(activeSpeechFrames / Math.max(frameCount * 0.2, 1), 0, 1);
  const eventCoverage = clamp((speechRuns.length + tailEventScores.length) / 10, 0, 1);
  const analysisConfidence = clamp(eventCoverage * 0.65 + speechCoverage * 0.35, 0, 1);
  const drynessScore = clamp(1 - roomScore - pauseNoiseRisk * 0.15, 0, 1);

  const clipScore = clamp((options.clipPct ?? 0) / 0.03, 0, 1);
  const overallRisk = clamp(
    instabilityScore * 0.13 +
      onsetOvershootScore * 0.08 +
      midLineSagScore * 0.08 +
      endFadeRiskScore * 0.08 +
      lineSwingScore * 0.08 +
      sentenceJumpScore * 0.1 +
      breathSpikeRisk * 0.1 +
      compressionScore * 0.14 +
      pauseNoiseRisk * 0.13 +
      clickScore * 0.1 +
      echoScore * 0.08 +
      clipScore * 0.06,
    0,
    1
  );

  return {
    peakDb: options.peakDb ?? null,
    clipPct: options.clipPct ?? null,
    noiseFloorDb,
    pauseNoiseFloorDb,
    nearSpeechNoiseFloorDb,
    speechThresholdDb,
    noiseContrastDb,
    speechRatioPct,
    speechDutyCyclePct,
    speechSegmentCount: speechRuns.length,
    medianSpeechRunMs,
    longSilenceCount,
    dynamicRangeDb,
    instabilityScore,
    onsetOvershootScore,
    midLineSagScore,
    endFadeRiskScore,
    endEdgeDipDb,
    lineSwingScore,
    sentenceJumpScore,
    coldOpenDipDb,
    coldOpenRiskScore,
    breathSpikeRisk,
    pauseNoiseRisk,
    compressionScore,
    clickScore,
    reverbScore,
    echoScore,
    roomScore,
    echoDelayMs: echoScore >= 0.18 && bestEchoLagFrames > 0 ? bestEchoLagFrames * frameMs : null,
    analysisConfidence,
    drynessScore,
    overallRisk,
    // `analyzeFrameAudio` works on pre-computed envelope data and does not see
    // raw samples, so spectrum/sibilance are not available at this stage.
    // `analyzeFloatSamples` fills them in from the raw signal.
    bandSpectrumDb: null,
    sibilanceScore: 0,
  };
};

export const analyzeFloatSamples = (
  samples: Float32Array,
  sampleRate: number,
  frameMs = AUDIO_QC_FRAME_MS
): AudioQcMetrics => {
  const frameSize = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const frameCount = Math.floor(samples.length / frameSize);
  const frameRms = new Array<number>(frameCount);
  const frameDb = new Array<number>(frameCount);
  const framePeak = new Array<number>(frameCount);
  const frameSharpness = new Array<number>(frameCount);

  let globalPeak = 0;
  let clipCount = 0;
  let sampleSpikeCount = 0;
  const refractorySamples = Math.max(1, Math.round(sampleRate * 0.004));
  let lastSpikeIndex = -refractorySamples;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * frameSize;
    let sumSquares = 0;
    let peak = 0;
    let sharpEnergy = 0;
    for (let index = 0; index < frameSize; index += 1) {
      const value = samples[start + index] ?? 0;
      const abs = Math.abs(value);
      if (abs > peak) peak = abs;
      if (abs > globalPeak) globalPeak = abs;
      if (abs >= 0.995) clipCount += 1;
      sumSquares += value * value;

      const prev = index > 0 ? samples[start + index - 1] ?? 0 : value;
      const next = index + 1 < frameSize ? samples[start + index + 1] ?? 0 : value;
      const spike = value - (prev + next) * 0.5;
      sharpEnergy += spike * spike;
    }
    const rms = Math.sqrt(sumSquares / frameSize);
    frameRms[frame] = rms;
    framePeak[frame] = peak;
    frameDb[frame] = Math.max(AUDIO_QC_FLOOR_DB, toDb(rms + 1e-12));
    frameSharpness[frame] = toDb(Math.sqrt(sharpEnergy / frameSize) + 1e-12);
  }

  for (let index = 1; index < samples.length; index += 1) {
    const current = samples[index] ?? 0;
    const previous = samples[index - 1] ?? 0;
    const diff = Math.abs(current - previous);
    if (diff < 0.09) continue;
    if (Math.abs(current) < 0.015) continue;
    if (index - lastSpikeIndex < refractorySamples) continue;
    sampleSpikeCount += 1;
    lastSpikeIndex = index;
  }

  return analyzeFrameAudio(frameRms, framePeak, frameDb, frameSharpness, {
    sampleRate,
    durationSec: samples.length / Math.max(sampleRate, 1),
    frameMs,
    peakDb: toDb(globalPeak + 1e-12),
    clipPct: (clipCount / Math.max(samples.length, 1)) * 100,
    sampleSpikeCount,
  });
};
