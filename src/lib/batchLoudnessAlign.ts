export const BATCH_ALIGN_TRIGGER_LU = 0.5;
export const BATCH_ALIGN_MAX_DB = 2.0;

export type BatchLoudnessMeasurement = {
  id: string;
  inputI: number | null;
};

export type BatchLoudnessAlignmentPlan = {
  id: string;
  inputI: number | null;
  offsetDb: number;
  shouldAlign: boolean;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const median = (values: number[]) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
};

export const planBatchLoudnessAlignment = (
  measurements: BatchLoudnessMeasurement[],
  options?: {
    triggerLu?: number;
    maxOffsetDb?: number;
  },
): {
  anchorLufs: number | null;
  plans: BatchLoudnessAlignmentPlan[];
} => {
  const triggerLu = options?.triggerLu ?? BATCH_ALIGN_TRIGGER_LU;
  const maxOffsetDb = options?.maxOffsetDb ?? BATCH_ALIGN_MAX_DB;
  const measured = measurements
    .map((measurement) => measurement.inputI)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const anchorLufs = measured.length >= 2 ? median(measured) : null;

  return {
    anchorLufs,
    plans: measurements.map((measurement) => {
      if (anchorLufs === null || typeof measurement.inputI !== "number" || !Number.isFinite(measurement.inputI)) {
        return {
          id: measurement.id,
          inputI: measurement.inputI,
          offsetDb: 0,
          shouldAlign: false,
        };
      }
      const rawOffsetDb = anchorLufs - measurement.inputI;
      const shouldAlign = Math.abs(rawOffsetDb) > triggerLu;
      return {
        id: measurement.id,
        inputI: measurement.inputI,
        offsetDb: shouldAlign ? clamp(rawOffsetDb, -maxOffsetDb, maxOffsetDb) : 0,
        shouldAlign,
      };
    }),
  };
};
