export const VERCEL_FUNCTION_BODY_LIMIT_BYTES = 4.5 * 1024 * 1024;
export const NEURAL_REPAIR_SAFE_FUNCTION_BODY_BYTES = 3.75 * 1024 * 1024;
export const NEURAL_REPAIR_MAX_CHUNK_SECONDS = 20;
export const NEURAL_REPAIR_MIN_CHUNK_SECONDS = 0.75;

export type NeuralRepairChunkPlanEntry = {
  index: number;
  total: number;
  startSec: number;
  endSec: number;
  durationSec: number;
  estimatedBytes: number;
};

export type NeuralRepairTransportPlan = {
  strategy: "direct" | "chunked";
  inputBytes: number;
  durationSeconds: number;
  chunkDurationSeconds: number;
  chunks: NeuralRepairChunkPlanEntry[];
};

const roundTime = (value: number) => Math.round(value * 1000) / 1000;

const assertFinitePositive = (value: number, label: string) => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number.`);
  }
};

export const planNeuralRepairTransport = ({
  inputBytes,
  durationSeconds,
  safeBodyBytes = NEURAL_REPAIR_SAFE_FUNCTION_BODY_BYTES,
}: {
  inputBytes: number;
  durationSeconds: number | null | undefined;
  safeBodyBytes?: number;
}): NeuralRepairTransportPlan => {
  assertFinitePositive(inputBytes, "inputBytes");
  assertFinitePositive(safeBodyBytes, "safeBodyBytes");

  if (inputBytes <= safeBodyBytes) {
    const directDuration = Number.isFinite(durationSeconds ?? Number.NaN) && (durationSeconds ?? 0) > 0
      ? roundTime(durationSeconds as number)
      : 0;
    return {
      strategy: "direct",
      inputBytes,
      durationSeconds: directDuration,
      chunkDurationSeconds: directDuration,
      chunks: [
        {
          index: 0,
          total: 1,
          startSec: 0,
          endSec: directDuration,
          durationSec: directDuration,
          estimatedBytes: inputBytes,
        },
      ],
    };
  }

  if (!Number.isFinite(durationSeconds ?? Number.NaN) || (durationSeconds ?? 0) <= 0) {
    throw new Error("Audio duration is required to chunk an oversized neural repair upload.");
  }

  const safeDurationSeconds = durationSeconds as number;
  const bytesPerSecond = inputBytes / safeDurationSeconds;
  assertFinitePositive(bytesPerSecond, "bytesPerSecond");
  const bodyBudgetSeconds = (safeBodyBytes / bytesPerSecond) * 0.94;
  const chunkDurationSeconds = Math.max(
    NEURAL_REPAIR_MIN_CHUNK_SECONDS,
    Math.min(NEURAL_REPAIR_MAX_CHUNK_SECONDS, bodyBudgetSeconds),
  );
  const chunks: NeuralRepairChunkPlanEntry[] = [];
  let cursor = 0;
  while (cursor < safeDurationSeconds - 0.001) {
    const endSec = Math.min(safeDurationSeconds, cursor + chunkDurationSeconds);
    const durationSec = Math.max(0, endSec - cursor);
    chunks.push({
      index: chunks.length,
      total: 0,
      startSec: roundTime(cursor),
      endSec: roundTime(endSec),
      durationSec: roundTime(durationSec),
      estimatedBytes: Math.ceil(durationSec * bytesPerSecond),
    });
    cursor = endSec;
  }

  const total = chunks.length;
  return {
    strategy: "chunked",
    inputBytes,
    durationSeconds: roundTime(safeDurationSeconds),
    chunkDurationSeconds: roundTime(chunkDurationSeconds),
    chunks: chunks.map((chunk) => ({ ...chunk, total })),
  };
};
