export type NeuralRepairEngine = "clearvoice";

export type NeuralRepairCandidateVariant = "clearvoice-se";

export type NeuralRepairMode = "speech_enhancement";

export type NeuralRepairRequest = {
  variant: NeuralRepairCandidateVariant;
  engine: NeuralRepairEngine;
  mode: NeuralRepairMode;
  model: string;
  description: string;
};

export type NeuralRepairRequestInput = {
  variant?: string | null;
  engine?: string | null;
  mode?: string | null;
  model?: string | null;
};

export type NeuralRepairReport = {
  engine: NeuralRepairEngine;
  mode: NeuralRepairMode;
  model: string;
  inputSampleRate?: number;
  outputSampleRate?: number;
  durationSeconds?: number;
  elapsedSeconds?: number;
  outputSamples?: number;
  outputBytes?: number;
  torchThreads?: number;
  chunksTotal?: number;
  chunksProcessed?: number;
  chunksBypassed?: number;
  speechAware?: boolean;
  speechAwareSpans?: number;
  activeDutyPct?: number;
  activeSeconds?: number;
  activeThresholdDb?: number;
  processedSeconds?: number;
  warnings?: string[];
};

export const CLEARVOICE_SE_VARIANT = "clearvoice-se" as const;

export const CLEARVOICE_SE_REQUEST: NeuralRepairRequest = {
  variant: CLEARVOICE_SE_VARIANT,
  engine: "clearvoice",
  mode: "speech_enhancement",
  model: "MossFormer2_SE_48K",
  description: "Neural speech enhancement",
};

export const isNeuralRepairCandidateVariant = (variant: string): variant is NeuralRepairCandidateVariant =>
  variant === CLEARVOICE_SE_VARIANT;

export const neuralRepairRequestForVariant = (
  variant: NeuralRepairCandidateVariant,
): NeuralRepairRequest => {
  if (variant !== CLEARVOICE_SE_VARIANT) {
    throw new Error(`Unsupported neural repair variant: ${variant}`);
  }
  return CLEARVOICE_SE_REQUEST;
};

export const resolveNeuralRepairRequestInput = (
  input: NeuralRepairRequestInput,
): NeuralRepairRequest | null => {
  const variant = input.variant || CLEARVOICE_SE_VARIANT;
  if (!isNeuralRepairCandidateVariant(variant)) return null;
  return neuralRepairRequestForVariant(variant);
};

export const formatNeuralRepairCandidate = (variant: NeuralRepairCandidateVariant) => {
  if (variant === CLEARVOICE_SE_VARIANT) return "clearvoice-se";
  return variant;
};
