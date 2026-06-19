import { buildSpeechMask, percentile, toDb } from "./audioQc.ts";

export type SevereEchoResidualSuppressionOptions = {
  echoDelayMs: number | null;
  strictOpenLiftDb?: number;
  strictCloseLiftDb?: number;
  minSpeechHoldMs?: number;
  gapBridgeMs?: number;
  minSpeechRunMs?: number;
  tailStartFrames?: number;
  tailFullFrames?: number;
  tailDepth?: number;
  tailFloorGain?: number;
  tailAttack?: number;
  tailRelease?: number;
  tapMix?: number;
  tapStride?: number;
  finishOpenLiftDb?: number;
  finishCloseLiftDb?: number;
  finishMinSpeechHoldMs?: number;
  finishGapBridgeMs?: number;
  finishMinSpeechRunMs?: number;
  finishPreviousSpeechRunFrames?: number;
  finishTailStartFrames?: number;
  finishTailFullFrames?: number;
  finishTailDepth?: number;
  finishTailFloorGain?: number;
  finishTailAttack?: number;
  finishTailRelease?: number;
  strictSpeechBodyDropDb?: number;
  finishSpeechBodyDropDb?: number;
};

export type SevereEchoResidualSuppressionResult = {
  samples: Float32Array;
  echoDelaySamples: number;
  echoTapCoeff: number;
  tailFramesSuppressed: number;
  finishingTailFramesSuppressed: number;
};

export const DEFAULT_SEVERE_ECHO_RESIDUAL_SUPPRESSION = {
  strictOpenLiftDb: 25,
  strictCloseLiftDb: 20,
  minSpeechHoldMs: 28,
  gapBridgeMs: 2,
  minSpeechRunMs: 115,
  tailStartFrames: 1,
  tailFullFrames: 5,
  tailDepth: 0.985,
  tailFloorGain: 0.015,
  tailAttack: 0.99,
  tailRelease: 0.68,
  tapMix: 0.65,
  tapStride: 8,
  finishOpenLiftDb: 12,
  finishCloseLiftDb: 9,
  finishMinSpeechHoldMs: 70,
  finishGapBridgeMs: 90,
  finishMinSpeechRunMs: 40,
  finishPreviousSpeechRunFrames: 6,
  finishTailStartFrames: 12,
  finishTailFullFrames: 20,
  finishTailDepth: 0.9,
  finishTailFloorGain: 0.1,
  finishTailAttack: 0.3,
  finishTailRelease: 0.16,
  strictSpeechBodyDropDb: 30,
  finishSpeechBodyDropDb: 16,
} as const;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const computeFrameDb = (samples: Float32Array, sampleRate: number, frameMs: number) => {
  const frameSize = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const frameCount = Math.floor(samples.length / frameSize);
  const frameDb = new Array<number>(frameCount);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * frameSize;
    let sumSquares = 0;
    for (let index = 0; index < frameSize; index += 1) {
      const value = samples[start + index] ?? 0;
      sumSquares += value * value;
    }
    frameDb[frame] = Math.max(-120, toDb(Math.sqrt(sumSquares / frameSize) + 1e-12));
  }

  return { frameSize, frameCount, frameDb };
};

const resolveSpeechMaskFloor = (
  frameDb: number[],
  measuredNoiseFloorDb: number,
  openLiftDb: number,
  speechBodyDropDb: number,
) => {
  const speechBodyDb = percentile(frameDb, 85) ?? measuredNoiseFloorDb;
  if (!Number.isFinite(speechBodyDb)) return measuredNoiseFloorDb;

  const noiseRelativeOpenDb = measuredNoiseFloorDb + openLiftDb;
  const bodyRelativeOpenDb = speechBodyDb - speechBodyDropDb;
  const anchoredOpenDb = Math.max(noiseRelativeOpenDb, bodyRelativeOpenDb);
  return anchoredOpenDb - openLiftDb;
};

export const estimateResidualEchoTap = (
  samples: Float32Array,
  delaySamples: number,
  stride: number = DEFAULT_SEVERE_ECHO_RESIDUAL_SUPPRESSION.tapStride,
) => {
  if (samples.length <= delaySamples || delaySamples <= 0) return 0;

  let numerator = 0;
  let denominator = 0;
  const sampleStride = Math.max(1, Math.round(stride));
  for (let index = delaySamples; index < samples.length; index += sampleStride) {
    const current = samples[index] ?? 0;
    if (Math.abs(current) < 2e-5) continue;
    const delayed = samples[index - delaySamples] ?? 0;
    numerator += delayed * current;
    denominator += delayed * delayed;
  }

  return clamp(numerator / (denominator + 1e-9), -0.36, 0.12);
};

const applySingleDelayBlend = (
  samples: Float32Array,
  delaySamples: number,
  echoTapCoeff: number,
  mix: number,
) => {
  const output = new Float32Array(samples.length);
  output.set(samples.subarray(0, Math.min(delaySamples, samples.length)));
  for (let index = delaySamples; index < samples.length; index += 1) {
    const value = samples[index] - mix * echoTapCoeff * (samples[index - delaySamples] ?? 0);
    output[index] = clamp(value, -1, 1);
  }
  return output;
};

const applyStrictTailSuppression = (
  samples: Float32Array,
  sampleRate: number,
  options: Required<Omit<SevereEchoResidualSuppressionOptions, "echoDelayMs">>,
) => {
  const frameMs = 10;
  const { frameSize, frameCount, frameDb } = computeFrameDb(samples, sampleRate, frameMs);
  const noiseFloorDb = percentile(frameDb, 25) ?? -72;
  const speechMaskFloorDb = resolveSpeechMaskFloor(
    frameDb,
    noiseFloorDb,
    options.strictOpenLiftDb,
    options.strictSpeechBodyDropDb,
  );
  const speechMask = buildSpeechMask(frameDb, speechMaskFloorDb, {
    frameMs,
    openLiftDb: options.strictOpenLiftDb,
    closeLiftDb: options.strictCloseLiftDb,
    minSpeechHoldMs: options.minSpeechHoldMs,
    gapBridgeMs: options.gapBridgeMs,
    minSpeechRunMs: options.minSpeechRunMs,
  });

  const targetGain = new Float32Array(frameCount).fill(1);
  let speechRunFrames = 0;
  let framesSinceSpeech = 9999;
  let previousSpeechRunFrames = 0;
  let tailFramesSuppressed = 0;

  for (let frame = 0; frame < frameCount; frame += 1) {
    if (speechMask[frame]) {
      speechRunFrames += 1;
      framesSinceSpeech = 0;
      continue;
    }

    if (framesSinceSpeech === 0) {
      previousSpeechRunFrames = speechRunFrames;
    }
    speechRunFrames = 0;
    framesSinceSpeech += 1;

    if (previousSpeechRunFrames < 5 || framesSinceSpeech < options.tailStartFrames) continue;

    const progress = clamp(
      (framesSinceSpeech - options.tailStartFrames) /
        Math.max(1, options.tailFullFrames - options.tailStartFrames),
      0,
      1,
    );
    targetGain[frame] = Math.max(options.tailFloorGain, 1 - options.tailDepth * progress);
    if (targetGain[frame] < 0.999) tailFramesSuppressed += 1;
  }

  const smoothGain = new Float32Array(frameCount);
  let gain = 1;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const target = targetGain[frame] ?? 1;
    const coefficient = target < gain ? options.tailAttack : options.tailRelease;
    gain += (target - gain) * coefficient;
    smoothGain[frame] = gain;
  }

  const output = new Float32Array(samples.length);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * frameSize;
    const previousGain = smoothGain[Math.max(0, frame - 1)] ?? smoothGain[frame] ?? 1;
    const currentGain = smoothGain[frame] ?? 1;
    for (let index = 0; index < frameSize; index += 1) {
      const weight = index / frameSize;
      output[start + index] = (samples[start + index] ?? 0) * (previousGain * (1 - weight) + currentGain * weight);
    }
  }
  const trailingGain = smoothGain[frameCount - 1] ?? 1;
  for (let index = frameCount * frameSize; index < samples.length; index += 1) {
    output[index] = (samples[index] ?? 0) * trailingGain;
  }

  return { samples: output, tailFramesSuppressed };
};

const applyFinishingTailSuppression = (
  samples: Float32Array,
  sampleRate: number,
  options: Required<Omit<SevereEchoResidualSuppressionOptions, "echoDelayMs">>,
) => {
  const frameMs = 10;
  const { frameSize, frameCount, frameDb } = computeFrameDb(samples, sampleRate, frameMs);
  const noiseFloorDb = percentile(frameDb, 25) ?? -72;
  const speechMaskFloorDb = resolveSpeechMaskFloor(
    frameDb,
    noiseFloorDb,
    options.finishOpenLiftDb,
    options.finishSpeechBodyDropDb,
  );
  const speechMask = buildSpeechMask(frameDb, speechMaskFloorDb, {
    frameMs,
    openLiftDb: options.finishOpenLiftDb,
    closeLiftDb: options.finishCloseLiftDb,
    minSpeechHoldMs: options.finishMinSpeechHoldMs,
    gapBridgeMs: options.finishGapBridgeMs,
    minSpeechRunMs: options.finishMinSpeechRunMs,
  });

  const targetGain = new Float32Array(frameCount).fill(1);
  const bodyReferenceDb = percentile(frameDb, 85) ?? noiseFloorDb;
  const strongBodyThresholdDb = bodyReferenceDb - 8;
  const decayingTailThresholdDb = bodyReferenceDb - 10;
  const upcomingStrongLookaheadFrames = Math.max(options.finishTailFullFrames, Math.round(260 / frameMs));
  const hasUpcomingStrongBody = (frame: number) => {
    const end = Math.min(frameCount, frame + upcomingStrongLookaheadFrames);
    for (let cursor = frame + 1; cursor < end; cursor += 1) {
      if ((frameDb[cursor] ?? -120) >= strongBodyThresholdDb) return true;
    }
    return false;
  };
  const tailTargetGain = (framesSinceTailStart: number) => {
    const progress = clamp(
      framesSinceTailStart / Math.max(1, options.finishTailFullFrames - options.finishTailStartFrames),
      0,
      1,
    );
    return Math.max(options.finishTailFloorGain, 1 - options.finishTailDepth * progress);
  };
  let speechRunFrames = 0;
  let framesSinceSpeech = 9999;
  let framesSinceStrongBody = 9999;
  let previousSpeechRunFrames = 0;
  let finishingTailFramesSuppressed = 0;

  for (let frame = 0; frame < frameCount; frame += 1) {
    if (speechMask[frame]) {
      speechRunFrames += 1;
      framesSinceSpeech = 0;
      const frameDbValue = frameDb[frame] ?? -120;
      if (frameDbValue >= strongBodyThresholdDb) {
        framesSinceStrongBody = 0;
        continue;
      }
      framesSinceStrongBody += 1;
      const isDecayingTail =
        speechRunFrames >= options.finishPreviousSpeechRunFrames &&
        framesSinceStrongBody >= options.finishTailStartFrames &&
        frameDbValue <= decayingTailThresholdDb &&
        !hasUpcomingStrongBody(frame);
      if (isDecayingTail) {
        targetGain[frame] = Math.min(
          targetGain[frame],
          tailTargetGain(framesSinceStrongBody - options.finishTailStartFrames),
        );
        if (targetGain[frame] < 0.999) finishingTailFramesSuppressed += 1;
      }
      continue;
    }

    if (framesSinceSpeech === 0) {
      previousSpeechRunFrames = speechRunFrames;
    }
    speechRunFrames = 0;
    framesSinceSpeech += 1;
    framesSinceStrongBody += 1;

    if (
      previousSpeechRunFrames < options.finishPreviousSpeechRunFrames ||
      framesSinceSpeech < options.finishTailStartFrames
    ) {
      continue;
    }

    targetGain[frame] = tailTargetGain(framesSinceSpeech - options.finishTailStartFrames);
    if (targetGain[frame] < 0.999) finishingTailFramesSuppressed += 1;
  }

  const smoothGain = new Float32Array(frameCount);
  let gain = 1;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const target = targetGain[frame] ?? 1;
    const coefficient = target < gain ? options.finishTailAttack : options.finishTailRelease;
    gain += (target - gain) * coefficient;
    smoothGain[frame] = gain;
  }

  const output = new Float32Array(samples.length);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * frameSize;
    const previousGain = smoothGain[Math.max(0, frame - 1)] ?? smoothGain[frame] ?? 1;
    const currentGain = smoothGain[frame] ?? 1;
    for (let index = 0; index < frameSize; index += 1) {
      const weight = index / frameSize;
      output[start + index] = (samples[start + index] ?? 0) * (previousGain * (1 - weight) + currentGain * weight);
    }
  }
  const trailingGain = smoothGain[frameCount - 1] ?? 1;
  for (let index = frameCount * frameSize; index < samples.length; index += 1) {
    output[index] = (samples[index] ?? 0) * trailingGain;
  }

  return { samples: output, finishingTailFramesSuppressed };
};

export const suppressSevereEchoResidual = (
  samples: Float32Array,
  sampleRate: number,
  options: SevereEchoResidualSuppressionOptions,
): SevereEchoResidualSuppressionResult => {
  const echoDelayMs = options.echoDelayMs;
  const echoDelaySamples =
    echoDelayMs !== null && Number.isFinite(echoDelayMs)
      ? Math.round((sampleRate * echoDelayMs) / 1000)
      : 0;

  if (samples.length === 0 || sampleRate <= 0 || echoDelaySamples <= 0) {
    return {
      samples: samples.slice(),
      echoDelaySamples: 0,
      echoTapCoeff: 0,
      tailFramesSuppressed: 0,
      finishingTailFramesSuppressed: 0,
    };
  }

  const config = {
    ...DEFAULT_SEVERE_ECHO_RESIDUAL_SUPPRESSION,
    ...options,
  };
  const tailSuppressed = applyStrictTailSuppression(samples, sampleRate, config);
  const echoTapCoeff = estimateResidualEchoTap(
    tailSuppressed.samples,
    echoDelaySamples,
    config.tapStride,
  );
  const output =
    Math.abs(echoTapCoeff) >= 0.03
      ? applySingleDelayBlend(tailSuppressed.samples, echoDelaySamples, echoTapCoeff, config.tapMix)
      : tailSuppressed.samples;
  const finished = applyFinishingTailSuppression(output, sampleRate, config);

  return {
    samples: finished.samples,
    echoDelaySamples,
    echoTapCoeff,
    tailFramesSuppressed: tailSuppressed.tailFramesSuppressed,
    finishingTailFramesSuppressed: finished.finishingTailFramesSuppressed,
  };
};
