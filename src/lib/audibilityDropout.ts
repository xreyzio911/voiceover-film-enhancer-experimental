export type AudibilityDropoutCluster = {
  startFrame: number;
  endFrame: number;
  startSec: number;
  endSec: number;
  durationSec: number;
  minDropDb: number;
  medianDropDb: number;
  maxSourceDb: number;
  minRenderedDb: number;
};

export type AudibilityDropoutReport = {
  severe: boolean;
  sourceSpeechThresholdDb: number;
  badFrameCount: number;
  badSeconds: number;
  clusterCount: number;
  worstDropDb: number;
  clusters: AudibilityDropoutCluster[];
};

export type AudibilityDropoutInput = {
  sourceFrameDb: number[];
  renderedFrameDb: number[];
  frameMs?: number;
  minClusterMs?: number;
  bridgeGapMs?: number;
  missingTailToleranceMs?: number;
  severeBadSeconds?: number;
  severeClusterSeconds?: number;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const median = (values: number[]) => {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return 0;
  const mid = Math.floor(finite.length / 2);
  return finite.length % 2 === 0 ? (finite[mid - 1] + finite[mid]) / 2 : finite[mid];
};

const percentile = (values: number[], pct: number) => {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const index = clamp(Math.round((pct / 100) * (finite.length - 1)), 0, finite.length - 1);
  return finite[index];
};

const estimateSourceSpeechThresholdDb = (sourceFrameDb: number[]) => {
  const finite = sourceFrameDb.filter((db) => Number.isFinite(db));
  const nonDigital = finite.filter((db) => db > -130);
  const quiet = percentile(nonDigital.length > 0 ? nonDigital : finite, 20) ?? -90;
  return clamp(Math.max(quiet + 12, -58), -58, -30);
};

const bridgeMaskGaps = (mask: boolean[], maxGapFrames: number) => {
  if (maxGapFrames <= 0) return mask;
  const bridged = [...mask];
  let index = 0;
  while (index < bridged.length) {
    if (bridged[index]) {
      index += 1;
      continue;
    }
    const gapStart = index;
    while (index < bridged.length && !bridged[index]) index += 1;
    const gapEnd = index;
    if (gapStart > 0 && gapEnd < bridged.length && gapEnd - gapStart <= maxGapFrames) {
      for (let cursor = gapStart; cursor < gapEnd; cursor += 1) bridged[cursor] = true;
    }
  }
  return bridged;
};

const collectClusters = (
  bad: boolean[],
  sourceFrameDb: number[],
  renderedFrameDb: number[],
  frameMs: number,
  minClusterFrames: number,
  gapFrames: number,
) => {
  const clusters: AudibilityDropoutCluster[] = [];
  let index = 0;
  while (index < bad.length) {
    if (!bad[index]) {
      index += 1;
      continue;
    }

    const start = index;
    let end = index + 1;
    let gap = 0;
    index += 1;
    while (index < bad.length) {
      if (bad[index]) {
        end = index + 1;
        gap = 0;
      } else {
        gap += 1;
        if (gap > gapFrames) break;
      }
      index += 1;
    }

    if (end - start < minClusterFrames) continue;

    const drops: number[] = [];
    let maxSourceDb = -240;
    let minRenderedDb = 240;
    for (let frame = start; frame < end; frame += 1) {
      const sourceDb = sourceFrameDb[frame] ?? -240;
      const renderedDb = renderedFrameDb[frame] ?? -240;
      drops.push(renderedDb - sourceDb);
      maxSourceDb = Math.max(maxSourceDb, sourceDb);
      minRenderedDb = Math.min(minRenderedDb, renderedDb);
    }

    clusters.push({
      startFrame: start,
      endFrame: end,
      startSec: (start * frameMs) / 1000,
      endSec: (end * frameMs) / 1000,
      durationSec: ((end - start) * frameMs) / 1000,
      minDropDb: Math.min(...drops),
      medianDropDb: median(drops),
      maxSourceDb,
      minRenderedDb,
    });
  }
  return clusters;
};

export const frameDbFromFloatSamples = (samples: Float32Array, sampleRate: number, frameMs = 20) => {
  const frameSize = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const frameCount = Math.floor(samples.length / frameSize);
  const frames = new Array<number>(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    const offset = frame * frameSize;
    for (let sample = 0; sample < frameSize; sample += 1) {
      const value = samples[offset + sample] ?? 0;
      sum += value * value;
    }
    const rms = Math.sqrt(sum / frameSize);
    frames[frame] = rms > 1e-12 ? 20 * Math.log10(rms) : -240;
  }
  return frames;
};

export const detectAudibilityDropouts = (input: AudibilityDropoutInput): AudibilityDropoutReport => {
  const frameMs = input.frameMs ?? 20;
  const frameCount = input.sourceFrameDb.length;
  const renderedFrameCount = input.renderedFrameDb.length;
  const missingTailToleranceFrames = Math.max(0, Math.round((input.missingTailToleranceMs ?? 60) / frameMs));
  const sourceFrameDb = input.sourceFrameDb.slice(0, frameCount);
  const renderedFrameDb = new Array<number>(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    renderedFrameDb[frame] = input.renderedFrameDb[frame] ?? -240;
  }
  const sourceSpeechThresholdDb = estimateSourceSpeechThresholdDb(sourceFrameDb);
  const speechLikeThresholdDb = Math.min(sourceSpeechThresholdDb, -50);
  const clusteredSpeechMask = bridgeMaskGaps(
    sourceFrameDb.map((db) => db >= speechLikeThresholdDb),
    Math.max(0, Math.round(120 / frameMs)),
  );
  const bad = new Array<boolean>(frameCount).fill(false);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const sourceDb = sourceFrameDb[frame] ?? -240;
    if (sourceDb < speechLikeThresholdDb) continue;
    if (frame >= renderedFrameCount && frame - renderedFrameCount < missingTailToleranceFrames) continue;
    const renderedDb = renderedFrameDb[frame] ?? -240;
    const dropDb = renderedDb - sourceDb;
    const collapsedToSilence = renderedDb <= -68 && sourceDb >= -50;
    const severeRelativeDrop = dropDb <= -24 && renderedDb <= -58;
    const strongSpeechMissing = sourceDb >= -38 && renderedDb <= -54;
    bad[frame] = clusteredSpeechMask[frame] && (collapsedToSilence || severeRelativeDrop || strongSpeechMissing);
  }

  const minClusterFrames = Math.max(1, Math.round((input.minClusterMs ?? 80) / frameMs));
  const clusters = collectClusters(
    bad,
    sourceFrameDb,
    renderedFrameDb,
    frameMs,
    minClusterFrames,
    Math.max(0, Math.round((input.bridgeGapMs ?? 60) / frameMs)),
  );
  const badFrameCount = clusters.reduce((sum, cluster) => sum + (cluster.endFrame - cluster.startFrame), 0);
  const badSeconds = (badFrameCount * frameMs) / 1000;
  const severeBadSeconds = input.severeBadSeconds ?? 0.35;
  const severeClusterSeconds = input.severeClusterSeconds ?? 0.18;
  const worstDropDb = clusters.length > 0 ? Math.min(...clusters.map((cluster) => cluster.minDropDb)) : 0;
  const severe =
    badSeconds >= severeBadSeconds ||
    clusters.some((cluster) => cluster.durationSec >= severeClusterSeconds) ||
    (clusters.length >= 3 && badSeconds >= 0.24);

  return {
    severe,
    sourceSpeechThresholdDb,
    badFrameCount,
    badSeconds,
    clusterCount: clusters.length,
    worstDropDb,
    clusters,
  };
};
