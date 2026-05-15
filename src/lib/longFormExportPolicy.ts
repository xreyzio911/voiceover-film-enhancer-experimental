export const LONG_FORM_SAFE_MODE_DURATION_SECONDS = 4800;
export const LONG_FORM_EXPORT_CHUNK_SECONDS = 900;

export type LongFormChunk = {
  durationSec: number;
  index: number;
  startSec: number;
  total: number;
};

export type LongFormBoundarySpan = {
  endSec: number;
  startSec: number;
};

export const shouldUseLongFormSafeMode = (
  durationSeconds: number | null | undefined,
  estimatedSeconds = 0,
  budgetSeconds = LONG_FORM_SAFE_MODE_DURATION_SECONDS,
) => {
  const effectiveSeconds =
    durationSeconds !== null && durationSeconds !== undefined && Number.isFinite(durationSeconds)
      ? durationSeconds
      : estimatedSeconds;
  return Number.isFinite(effectiveSeconds) && effectiveSeconds > budgetSeconds;
};

export const planLongFormChunks = (
  durationSeconds: number,
  chunkSeconds = LONG_FORM_EXPORT_CHUNK_SECONDS,
  boundarySpans: LongFormBoundarySpan[] = [],
): LongFormChunk[] => {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return [];
  const safeChunkSeconds = Math.max(60, chunkSeconds);
  const chunkStarts: number[] = [];
  let cursor = 0;

  while (durationSeconds - cursor > safeChunkSeconds) {
    const targetCut = cursor + safeChunkSeconds;
    const minCut = Math.min(durationSeconds - 1, Math.max(cursor + 60, targetCut - 90));
    const maxCut = Math.min(durationSeconds - 0.1, targetCut + 90);
    const silenceCut = boundarySpans
      .map((span) => ({
        centerSec: (span.startSec + span.endSec) / 2,
        durationSec: span.endSec - span.startSec,
      }))
      .filter(
        (span) =>
          span.durationSec >= 0.3 &&
          span.centerSec >= minCut &&
          span.centerSec <= maxCut &&
          span.centerSec < durationSeconds - 0.1,
      )
      .sort(
        (left, right) =>
          Math.abs(left.centerSec - targetCut) - Math.abs(right.centerSec - targetCut) ||
          right.durationSec - left.durationSec,
      )[0]?.centerSec;
    const cut = silenceCut ?? Math.min(targetCut, durationSeconds);
    chunkStarts.push(cursor);
    cursor = cut;
  }
  chunkStarts.push(cursor);

  const total = chunkStarts.length;

  return chunkStarts.map((startSec, index) => {
    const nextStart = chunkStarts[index + 1] ?? durationSeconds;
    return {
      durationSec: Math.max(0.1, nextStart - startSec),
      index,
      startSec,
      total,
    };
  });
};

export const formatLongFormPartTag = (chunk: LongFormChunk) => {
  const width = Math.max(2, String(chunk.total).length);
  return `part${String(chunk.index + 1).padStart(width, "0")}-of-${String(chunk.total).padStart(width, "0")}`;
};
