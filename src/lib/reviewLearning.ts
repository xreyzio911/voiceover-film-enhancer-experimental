"use client";

import { BUILTIN_REVIEW_WEIGHTS } from "./defaultReviewWeights.ts";
import type { CandidateRenderMeta, CandidateScore } from "./renderRecovery.ts";

export const REVIEW_BUNDLE_SCHEMA_VERSION = 1 as const;
export const LEARNED_REVIEW_MODEL_TYPE = "linear-review-ranker-v1" as const;
export const REVIEW_WEIGHT_STORAGE_KEY = "vo-leveler-review-weights-v1";

export const REVIEW_ISSUE_TAGS = [
  "timing_shift",
  "level_uneven",
  "cold_open_dip",
  "pause_noise_lift",
  "too_compressed",
  "harsh_sibilance",
  "endings_damaged",
  "echo_roomy",
  "clicks_artifacts",
  "other",
] as const;

export type ReviewIssueTag = (typeof REVIEW_ISSUE_TAGS)[number];
export type ReviewVerdict = "pass" | "fail";
export type ReviewCandidateRole = "winner" | "challenger";

export const HIGH_VALUE_CORRECTIVE_ISSUE_TAGS = [
  "cold_open_dip",
  "endings_damaged",
  "harsh_sibilance",
  "too_compressed",
  "level_uneven",
] as const satisfies readonly ReviewIssueTag[];

const highValueCorrectiveIssueTagSet = new Set<ReviewIssueTag>(HIGH_VALUE_CORRECTIVE_ISSUE_TAGS);

export const REVIEW_FEATURE_KEYS = [
  "baseline_total",
  "baseline_stability",
  "baseline_pause",
  "baseline_compression",
  "baseline_echo",
  "candidate_overallRisk",
  "candidate_instabilityScore",
  "candidate_sentenceJumpScore",
  "candidate_pauseNoiseRisk",
  "candidate_compressionScore",
  "candidate_echoScore",
  "candidate_clickScore",
  "candidate_endFadeRiskScore",
  "candidate_sibilanceScore",
  "delta_overallRisk",
  "delta_instabilityScore",
  "delta_sentenceJumpScore",
  "delta_pauseNoiseRisk",
  "delta_compressionScore",
  "delta_echoScore",
  "delta_clickScore",
  "delta_endFadeRiskScore",
  "delta_sibilanceScore",
  "delta_pauseNoiseFloorDb",
  "delta_noiseContrastDb",
  "alignment_durationDeltaSec",
  "alignment_offsetSec",
  "alignment_offsetConfidence",
  "candidate_peakOverLimitDb",
  "healthy_segmented_bonus",
  "degraded_penalty",
] as const;

export type ReviewFeatureKey = (typeof REVIEW_FEATURE_KEYS)[number];
export type ReviewFeatureMap = Record<ReviewFeatureKey, number>;

export type ReviewMetricSnapshot = {
  inputI: number | null;
  inputTP: number | null;
  inputLRA: number | null;
  noiseFloorDb: number | null;
  pauseNoiseFloorDb: number | null;
  noiseContrastDb: number | null;
  instabilityScore: number | null;
  lineSwingScore: number | null;
  sentenceJumpScore: number | null;
  coldOpenDipDb: number | null;
  coldOpenRiskScore: number | null;
  breathSpikeRisk: number | null;
  pauseNoiseRisk: number | null;
  compressionScore: number | null;
  clickScore: number | null;
  echoScore: number | null;
  roomScore: number | null;
  overallRisk: number | null;
  onsetOvershootScore: number | null;
  midLineSagScore: number | null;
  endFadeRiskScore: number | null;
  endEdgeDipDb: number | null;
  sibilanceScore: number | null;
};

export type ReviewMetricDelta = {
  overallRisk: number | null;
  instabilityScore: number | null;
  sentenceJumpScore: number | null;
  coldOpenDipDb: number | null;
  coldOpenRiskScore: number | null;
  pauseNoiseRisk: number | null;
  compressionScore: number | null;
  clickScore: number | null;
  echoScore: number | null;
  endFadeRiskScore: number | null;
  endEdgeDipDb: number | null;
  sibilanceScore: number | null;
  pauseNoiseFloorDb: number | null;
  noiseContrastDb: number | null;
};

export type AlignmentMetrics = {
  durationSourceSec: number;
  durationCandidateSec: number;
  durationDeltaSec: number;
  durationDeltaPct: number;
  estimatedOffsetSec: number;
  confidence: number;
};

export type CandidateRankingBreakdown = {
  baselineTotal: number;
  hardGatePenalty: number;
  learnedAdjustment: number;
  rankingScore: number;
  gateReasons: string[];
  features: ReviewFeatureMap;
};

export type ReviewBundleCandidate = {
  role: ReviewCandidateRole;
  audioFile: string;
  variantLabel: string;
  renderMeta: CandidateRenderMeta;
  baselineScore: CandidateScore;
  ranking: CandidateRankingBreakdown;
  qc: ReviewMetricSnapshot | null;
  sourceComparison: {
    alignment: AlignmentMetrics;
    qcDelta: ReviewMetricDelta | null;
  };
  selectionReason: string | null;
};

export type ReviewBundleManifest = {
  schemaVersion: typeof REVIEW_BUNDLE_SCHEMA_VERSION;
  bundleId: string;
  createdAt: string;
  source: {
    fileName: string;
    audioFile: string;
    durationSec: number;
    sampleRate: number;
    qc: ReviewMetricSnapshot | null;
  };
  decisionContext: {
    jobBase: string;
    loudnessTarget: string;
    selectedVariant: string;
    selectedReason: string | null;
    learnedWeightsName: string;
    learnedWeightsSource: "default" | "local-import";
    reviewModelType: typeof LEARNED_REVIEW_MODEL_TYPE;
  };
  candidates: ReviewBundleCandidate[];
};

export type ReviewDecisionRecord = {
  schemaVersion: typeof REVIEW_BUNDLE_SCHEMA_VERSION;
  bundleId: string;
  reviewedAt: string;
  finalVerdict: ReviewVerdict;
  issueTags: ReviewIssueTag[];
  preferredRole: ReviewCandidateRole | null;
  confidence: number | null;
  note: string | null;
};

export type LearnedReviewWeights = {
  schemaVersion: typeof REVIEW_BUNDLE_SCHEMA_VERSION;
  modelType: typeof LEARNED_REVIEW_MODEL_TYPE;
  modelName: string;
  createdAt: string;
  sourceSummary: string;
  intercept: number;
  gateThresholds: {
    maxDurationDeltaSec: number;
    maxOffsetSec: number;
    minOffsetConfidence: number;
    maxPeakDb: number;
    maxEndFadeRisk: number;
    maxOverallRiskDelta: number;
    maxInstabilityDelta: number;
    maxSentenceJumpDelta: number;
    maxPauseNoiseRiskDelta: number;
    maxCompressionDelta: number;
    maxEchoDelta: number;
    maxClickDelta: number;
    maxSibilanceDelta: number;
    maxPauseNoiseFloorLiftDb: number;
    maxNoiseContrastLossDb: number;
  };
  penaltyWeights: {
    durationMismatch: number;
    offsetMismatch: number;
    peakViolation: number;
    endingDamage: number;
    sourceRegression: number;
  };
  featureWeights: ReviewFeatureMap;
};

export type ReviewTrainingReport = {
  manifestCount: number;
  decisionCount: number;
  reviewedBundleCount: number;
  pairwiseExampleCount: number;
  pairwiseAccuracy: number | null;
  issueTagCounts: Record<ReviewIssueTag, number>;
  challengerWins: number;
  failedReviews: number;
  weightsModelName: string;
};

export type AutoReviewCheckStatus = "pass" | "warn" | "fail";
export type AutoReviewCheckSeverity = "critical" | "major" | "minor";

export type AutoReviewCheck = {
  id: string;
  label: string;
  status: AutoReviewCheckStatus;
  severity: AutoReviewCheckSeverity;
  detail: string;
};

export type AutoReviewCandidateAssessment = {
  role: ReviewCandidateRole;
  variantLabel: string;
  technicalRiskScore: number;
  passCount: number;
  warnCount: number;
  failCount: number;
  issueTags: ReviewIssueTag[];
  findings: AutoReviewCheck[];
  summary: string;
};

export type AutoReviewResult = {
  bundleId: string;
  finalVerdict: ReviewVerdict;
  issueTags: ReviewIssueTag[];
  preferredRole: ReviewCandidateRole | null;
  confidence: number;
  note: string;
  executiveSummary: string;
  selectedAssessment: AutoReviewCandidateAssessment;
  challengerAssessment: AutoReviewCandidateAssessment | null;
};

export const shouldAttemptCorrectivePassForAssessment = (
  assessment: Pick<AutoReviewCandidateAssessment, "failCount" | "warnCount" | "issueTags">,
  gateReasons: readonly string[] = [],
) =>
  gateReasons.length > 0 ||
  assessment.failCount > 0 ||
  assessment.warnCount >= 2 ||
  (assessment.warnCount === 1 &&
    assessment.issueTags.some((tag) => highValueCorrectiveIssueTagSet.has(tag)));

export const resolveCorrectiveMaxFilesPerBatch = (fileCount: number) => {
  const safeFileCount = Number.isFinite(fileCount) ? Math.max(0, fileCount) : 0;
  return Math.max(2, Math.ceil(safeFileCount * 0.4));
};

type MetricSource = Partial<ReviewMetricSnapshot> | null | undefined;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const safeNumber = (value: number | null | undefined, fallback = 0) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;
const hasFiniteNumber = (value: number | null | undefined): value is number =>
  typeof value === "number" && Number.isFinite(value);
const QC_UNAVAILABLE_HARD_GATE_PENALTY = 100000;
const diffNullable = (next: number | null | undefined, previous: number | null | undefined) =>
  typeof next === "number" && Number.isFinite(next) && typeof previous === "number" && Number.isFinite(previous)
    ? next - previous
    : null;

const createZeroFeatureMap = (): ReviewFeatureMap =>
  Object.fromEntries(REVIEW_FEATURE_KEYS.map((key) => [key, 0])) as ReviewFeatureMap;

const formatSignedSeconds = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)} s`;
const formatSignedDb = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(1)} dB`;
const formatPercent = (value: number | null | undefined, digits = 0) =>
  hasFiniteNumber(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
const formatDb = (value: number | null | undefined, digits = 1) =>
  hasFiniteNumber(value) ? `${value.toFixed(digits)} dB` : "n/a";
const formatNumber = (value: number | null | undefined, digits = 2) =>
  hasFiniteNumber(value) ? value.toFixed(digits) : "n/a";

export const DEFAULT_LEARNED_REVIEW_WEIGHTS: LearnedReviewWeights =
  BUILTIN_REVIEW_WEIGHTS as unknown as LearnedReviewWeights;

export const toReviewMetricSnapshot = (source: MetricSource): ReviewMetricSnapshot | null => {
  if (!source) return null;
  return {
    inputI: source.inputI ?? null,
    inputTP: source.inputTP ?? null,
    inputLRA: source.inputLRA ?? null,
    noiseFloorDb: source.noiseFloorDb ?? null,
    pauseNoiseFloorDb: source.pauseNoiseFloorDb ?? null,
    noiseContrastDb: source.noiseContrastDb ?? null,
    instabilityScore: source.instabilityScore ?? null,
    lineSwingScore: source.lineSwingScore ?? null,
    sentenceJumpScore: source.sentenceJumpScore ?? null,
    coldOpenDipDb: source.coldOpenDipDb ?? null,
    coldOpenRiskScore: source.coldOpenRiskScore ?? null,
    breathSpikeRisk: source.breathSpikeRisk ?? null,
    pauseNoiseRisk: source.pauseNoiseRisk ?? null,
    compressionScore: source.compressionScore ?? null,
    clickScore: source.clickScore ?? null,
    echoScore: source.echoScore ?? null,
    roomScore: source.roomScore ?? null,
    overallRisk: source.overallRisk ?? null,
    onsetOvershootScore: source.onsetOvershootScore ?? null,
    midLineSagScore: source.midLineSagScore ?? null,
    endFadeRiskScore: source.endFadeRiskScore ?? null,
    endEdgeDipDb: source.endEdgeDipDb ?? null,
    sibilanceScore: source.sibilanceScore ?? null,
  };
};

export const buildReviewMetricDelta = (
  source: ReviewMetricSnapshot | null,
  candidate: ReviewMetricSnapshot | null,
): ReviewMetricDelta | null => {
  if (!source || !candidate) return null;
  return {
    overallRisk: diffNullable(candidate.overallRisk, source.overallRisk),
    instabilityScore: diffNullable(candidate.instabilityScore, source.instabilityScore),
    sentenceJumpScore: diffNullable(candidate.sentenceJumpScore, source.sentenceJumpScore),
    coldOpenDipDb: diffNullable(candidate.coldOpenDipDb, source.coldOpenDipDb),
    coldOpenRiskScore: diffNullable(candidate.coldOpenRiskScore, source.coldOpenRiskScore),
    pauseNoiseRisk: diffNullable(candidate.pauseNoiseRisk, source.pauseNoiseRisk),
    compressionScore: diffNullable(candidate.compressionScore, source.compressionScore),
    clickScore: diffNullable(candidate.clickScore, source.clickScore),
    echoScore: diffNullable(candidate.echoScore, source.echoScore),
    endFadeRiskScore: diffNullable(candidate.endFadeRiskScore, source.endFadeRiskScore),
    endEdgeDipDb: diffNullable(candidate.endEdgeDipDb, source.endEdgeDipDb),
    sibilanceScore: diffNullable(candidate.sibilanceScore, source.sibilanceScore),
    pauseNoiseFloorDb: diffNullable(candidate.pauseNoiseFloorDb, source.pauseNoiseFloorDb),
    noiseContrastDb: diffNullable(candidate.noiseContrastDb, source.noiseContrastDb),
  };
};

export const interleavedToMono = (samples: Float32Array, channels: number) => {
  const safeChannels = Math.max(1, channels);
  if (safeChannels === 1) return samples.slice();
  const frameCount = Math.floor(samples.length / safeChannels);
  const mono = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < safeChannels; channel += 1) {
      sum += samples[frame * safeChannels + channel] ?? 0;
    }
    mono[frame] = sum / safeChannels;
  }
  return mono;
};

const buildEnvelope = (
  samples: Float32Array,
  sampleRate: number,
  targetRate = 100,
  maxWindowSeconds = 20,
) => {
  if (samples.length === 0 || sampleRate <= 0) return { values: new Float32Array(0), rate: targetRate };
  const step = Math.max(1, Math.round(sampleRate / targetRate));
  const maxSamples = Math.min(samples.length, Math.max(step, Math.round(sampleRate * maxWindowSeconds)));
  const buckets = Math.max(1, Math.floor(maxSamples / step));
  const out = new Float32Array(buckets);
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    let sum = 0;
    const start = bucket * step;
    const end = Math.min(maxSamples, start + step);
    for (let index = start; index < end; index += 1) {
      sum += Math.abs(samples[index] ?? 0);
    }
    out[bucket] = sum / Math.max(1, end - start);
  }
  return { values: out, rate: targetRate };
};

export const estimateAlignmentMetrics = (
  sourceMono: Float32Array,
  sourceRate: number,
  candidateMono: Float32Array,
  candidateRate: number,
): AlignmentMetrics => {
  const durationSourceSec = sourceRate > 0 ? sourceMono.length / sourceRate : 0;
  const durationCandidateSec = candidateRate > 0 ? candidateMono.length / candidateRate : 0;
  const durationDeltaSec = durationCandidateSec - durationSourceSec;
  const durationDeltaPct =
    durationSourceSec > 1e-6 ? (durationDeltaSec / durationSourceSec) * 100 : 0;

  const sourceEnv = buildEnvelope(sourceMono, sourceRate);
  const candidateEnv = buildEnvelope(candidateMono, candidateRate);
  if (sourceEnv.values.length === 0 || candidateEnv.values.length === 0) {
    return {
      durationSourceSec,
      durationCandidateSec,
      durationDeltaSec,
      durationDeltaPct,
      estimatedOffsetSec: 0,
      confidence: 0,
    };
  }

  const envRate = Math.min(sourceEnv.rate, candidateEnv.rate);
  const sourceValues = sourceEnv.values;
  const candidateValues = candidateEnv.values;
  const maxLagFrames = Math.min(Math.round(envRate * 2), Math.max(4, Math.floor(sourceValues.length / 2)));
  let bestLag = 0;
  let bestScore = Number.NEGATIVE_INFINITY;
  let secondBest = Number.NEGATIVE_INFINITY;

  for (let lag = -maxLagFrames; lag <= maxLagFrames; lag += 1) {
    let dot = 0;
    let sourceEnergy = 0;
    let candidateEnergy = 0;
    let overlap = 0;
    for (let index = 0; index < sourceValues.length; index += 1) {
      const candidateIndex = index + lag;
      if (candidateIndex < 0 || candidateIndex >= candidateValues.length) continue;
      const left = sourceValues[index];
      const right = candidateValues[candidateIndex];
      dot += left * right;
      sourceEnergy += left * left;
      candidateEnergy += right * right;
      overlap += 1;
    }
    if (overlap < envRate) continue;
    const score = dot / Math.max(Math.sqrt(sourceEnergy * candidateEnergy), 1e-9);
    if (score > bestScore) {
      secondBest = bestScore;
      bestScore = score;
      bestLag = lag;
    } else if (score > secondBest) {
      secondBest = score;
    }
  }

  const confidence = clamp((bestScore - Math.max(0, secondBest)) * 3 + bestScore * 0.35, 0, 1);
  return {
    durationSourceSec,
    durationCandidateSec,
    durationDeltaSec,
    durationDeltaPct,
    estimatedOffsetSec: bestLag / Math.max(envRate, 1),
    confidence,
  };
};

export const buildReviewFeatureMap = (input: {
  baselineScore: CandidateScore;
  candidateQc: ReviewMetricSnapshot | null;
  qcDelta: ReviewMetricDelta | null;
  alignment: AlignmentMetrics;
  meta: CandidateRenderMeta;
  peakLimitDb: number;
}): ReviewFeatureMap => {
  const features = createZeroFeatureMap();
  features.baseline_total = safeNumber(input.baselineScore.total);
  features.baseline_stability = safeNumber(input.baselineScore.stability);
  features.baseline_pause = safeNumber(input.baselineScore.pause);
  features.baseline_compression = safeNumber(input.baselineScore.compression);
  features.baseline_echo = safeNumber(input.baselineScore.echo);

  const qc = input.candidateQc;
  features.candidate_overallRisk = safeNumber(qc?.overallRisk);
  features.candidate_instabilityScore = safeNumber(qc?.instabilityScore);
  features.candidate_sentenceJumpScore = safeNumber(qc?.sentenceJumpScore);
  features.candidate_pauseNoiseRisk = safeNumber(qc?.pauseNoiseRisk);
  features.candidate_compressionScore = safeNumber(qc?.compressionScore);
  features.candidate_echoScore = safeNumber(qc?.echoScore);
  features.candidate_clickScore = safeNumber(qc?.clickScore);
  features.candidate_endFadeRiskScore = safeNumber(qc?.endFadeRiskScore);
  features.candidate_sibilanceScore = safeNumber(qc?.sibilanceScore);

  const delta = input.qcDelta;
  features.delta_overallRisk = safeNumber(delta?.overallRisk);
  features.delta_instabilityScore = safeNumber(delta?.instabilityScore);
  features.delta_sentenceJumpScore = safeNumber(delta?.sentenceJumpScore);
  features.delta_pauseNoiseRisk = safeNumber(delta?.pauseNoiseRisk);
  features.delta_compressionScore = safeNumber(delta?.compressionScore);
  features.delta_echoScore = safeNumber(delta?.echoScore);
  features.delta_clickScore = safeNumber(delta?.clickScore);
  features.delta_endFadeRiskScore = safeNumber(delta?.endFadeRiskScore);
  features.delta_sibilanceScore = safeNumber(delta?.sibilanceScore);
  features.delta_pauseNoiseFloorDb = safeNumber(delta?.pauseNoiseFloorDb);
  features.delta_noiseContrastDb = safeNumber(delta?.noiseContrastDb);

  features.alignment_durationDeltaSec = Math.abs(input.alignment.durationDeltaSec);
  features.alignment_offsetSec = Math.abs(input.alignment.estimatedOffsetSec);
  features.alignment_offsetConfidence = safeNumber(input.alignment.confidence);
  features.candidate_peakOverLimitDb = Math.max(
    0,
    safeNumber(qc?.inputTP, -120) - input.peakLimitDb,
  );
  features.healthy_segmented_bonus = input.meta.segmentedHealthy ? 1 : 0;
  features.degraded_penalty = input.meta.degraded ? 1 : 0;
  return features;
};

export const scoreCandidateWithLearnedWeights = (input: {
  baselineScore: CandidateScore;
  candidateQc: ReviewMetricSnapshot | null;
  sourceQc: ReviewMetricSnapshot | null;
  alignment: AlignmentMetrics;
  meta: CandidateRenderMeta;
  weights?: LearnedReviewWeights | null;
}) => {
  const weights = mergeLearnedReviewWeights(input.weights ?? DEFAULT_LEARNED_REVIEW_WEIGHTS);
  const qcDelta = buildReviewMetricDelta(input.sourceQc, input.candidateQc);
  const features = buildReviewFeatureMap({
    baselineScore: input.baselineScore,
    candidateQc: input.candidateQc,
    qcDelta,
    alignment: input.alignment,
    meta: input.meta,
    peakLimitDb: weights.gateThresholds.maxPeakDb,
  });

  let hardGatePenalty = 0;
  const gateReasons: string[] = [];
  if (!input.candidateQc) {
    hardGatePenalty += QC_UNAVAILABLE_HARD_GATE_PENALTY;
    gateReasons.push("qc-unavailable");
  }

  const absDurationDelta = Math.abs(input.alignment.durationDeltaSec);
  if (absDurationDelta > weights.gateThresholds.maxDurationDeltaSec) {
    hardGatePenalty +=
      (absDurationDelta - weights.gateThresholds.maxDurationDeltaSec) *
      weights.penaltyWeights.durationMismatch;
    gateReasons.push("duration-mismatch");
  }

  const absOffset = Math.abs(input.alignment.estimatedOffsetSec);
  if (
    absOffset > weights.gateThresholds.maxOffsetSec &&
    input.alignment.confidence >= weights.gateThresholds.minOffsetConfidence
  ) {
    hardGatePenalty +=
      (absOffset - weights.gateThresholds.maxOffsetSec) *
      weights.penaltyWeights.offsetMismatch *
      clamp(input.alignment.confidence, 0.25, 1);
    gateReasons.push("timing-offset");
  }

  const peakOver = features.candidate_peakOverLimitDb;
  if (peakOver > 0) {
    hardGatePenalty += peakOver * weights.penaltyWeights.peakViolation;
    gateReasons.push("peak-violation");
  }

  const candidateEndFadeRisk = safeNumber(input.candidateQc?.endFadeRiskScore);
  const sourceEndFadeRisk = safeNumber(input.sourceQc?.endFadeRiskScore);
  const endingDelta = safeNumber(qcDelta?.endFadeRiskScore);
  const endingOver = candidateEndFadeRisk - weights.gateThresholds.maxEndFadeRisk;
  const createdEndingDamage =
    endingOver > 0 &&
    (sourceEndFadeRisk <= weights.gateThresholds.maxEndFadeRisk - 0.08 || endingDelta >= 0.05);
  if (createdEndingDamage) {
    hardGatePenalty += endingOver * weights.penaltyWeights.endingDamage;
    gateReasons.push("ending-damage");
  }

  const candidateEndEdgeDipDb = safeNumber(input.candidateQc?.endEdgeDipDb);
  const sourceEndEdgeDipDb = safeNumber(input.sourceQc?.endEdgeDipDb);
  const endEdgeDipDeltaDb = safeNumber(qcDelta?.endEdgeDipDb);
  const hasSourceEndEdgeDip = hasFiniteNumber(input.sourceQc?.endEdgeDipDb);
  const hasEndEdgeDipDelta = hasFiniteNumber(qcDelta?.endEdgeDipDb);
  const endEdgeDipIsNewOrWorse =
    !hasSourceEndEdgeDip ||
    sourceEndEdgeDipDb < 3.5 ||
    (hasEndEdgeDipDelta && endEdgeDipDeltaDb >= 1);
  const createdSevereEndEdgeDip =
    endEdgeDipIsNewOrWorse &&
    (candidateEndEdgeDipDb >= 6 ||
      (candidateEndEdgeDipDb >= 4.5 && (!hasEndEdgeDipDelta || endEdgeDipDeltaDb >= 1)));
  if (createdSevereEndEdgeDip) {
    hardGatePenalty +=
      Math.max(1, candidateEndEdgeDipDb - 3.5, endEdgeDipDeltaDb) *
      weights.penaltyWeights.endingDamage;
    gateReasons.push("end-edge-dip");
  }

  const sourceRegression =
    Math.max(0, safeNumber(qcDelta?.overallRisk) - weights.gateThresholds.maxOverallRiskDelta) / 0.08 +
    Math.max(0, safeNumber(qcDelta?.instabilityScore) - weights.gateThresholds.maxInstabilityDelta) / 0.06 +
    Math.max(0, safeNumber(qcDelta?.sentenceJumpScore) - weights.gateThresholds.maxSentenceJumpDelta) / 0.05 +
    Math.max(0, safeNumber(qcDelta?.pauseNoiseRisk) - weights.gateThresholds.maxPauseNoiseRiskDelta) / 0.05 +
    Math.max(0, safeNumber(qcDelta?.compressionScore) - weights.gateThresholds.maxCompressionDelta) / 0.06 +
    Math.max(0, safeNumber(qcDelta?.echoScore) - weights.gateThresholds.maxEchoDelta) / 0.05 +
    Math.max(0, safeNumber(qcDelta?.clickScore) - weights.gateThresholds.maxClickDelta) / 0.05 +
    Math.max(0, safeNumber(qcDelta?.sibilanceScore) - weights.gateThresholds.maxSibilanceDelta) / 0.05 +
    Math.max(0, safeNumber(qcDelta?.pauseNoiseFloorDb) - weights.gateThresholds.maxPauseNoiseFloorLiftDb) / 1.0 +
    Math.max(0, -safeNumber(qcDelta?.noiseContrastDb) - weights.gateThresholds.maxNoiseContrastLossDb) / 2.0;
  if (sourceRegression > 0) {
    hardGatePenalty += sourceRegression * weights.penaltyWeights.sourceRegression;
    gateReasons.push("source-regression");
  }

  let learnedAdjustment = weights.intercept;
  for (const featureKey of REVIEW_FEATURE_KEYS) {
    learnedAdjustment += features[featureKey] * weights.featureWeights[featureKey];
  }

  const rankingScore = input.baselineScore.total + hardGatePenalty + learnedAdjustment;
  return {
    baselineTotal: input.baselineScore.total,
    hardGatePenalty,
    learnedAdjustment,
    rankingScore,
    gateReasons,
    features,
  } satisfies CandidateRankingBreakdown;
};

export const mergeLearnedReviewWeights = (input: Partial<LearnedReviewWeights> | LearnedReviewWeights) => {
  const mergedFeatureWeights = createZeroFeatureMap();
  for (const featureKey of REVIEW_FEATURE_KEYS) {
    mergedFeatureWeights[featureKey] =
      safeNumber(input.featureWeights?.[featureKey], DEFAULT_LEARNED_REVIEW_WEIGHTS.featureWeights[featureKey]);
  }

  return {
    ...DEFAULT_LEARNED_REVIEW_WEIGHTS,
    ...input,
    gateThresholds: {
      ...DEFAULT_LEARNED_REVIEW_WEIGHTS.gateThresholds,
      ...input.gateThresholds,
    },
    penaltyWeights: {
      ...DEFAULT_LEARNED_REVIEW_WEIGHTS.penaltyWeights,
      ...input.penaltyWeights,
    },
    featureWeights: mergedFeatureWeights,
    schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
    modelType: LEARNED_REVIEW_MODEL_TYPE,
    modelName:
      typeof input.modelName === "string" && input.modelName.trim()
        ? input.modelName.trim()
        : LEARNED_REVIEW_MODEL_TYPE,
    createdAt:
      typeof input.createdAt === "string" && input.createdAt.trim()
        ? input.createdAt
        : DEFAULT_LEARNED_REVIEW_WEIGHTS.createdAt,
    sourceSummary:
      typeof input.sourceSummary === "string" && input.sourceSummary.trim()
        ? input.sourceSummary.trim()
        : DEFAULT_LEARNED_REVIEW_WEIGHTS.sourceSummary,
    intercept: safeNumber(input.intercept),
  } satisfies LearnedReviewWeights;
};

export const parseLearnedReviewWeights = (value: unknown): LearnedReviewWeights | null => {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<LearnedReviewWeights>;
  if (candidate.modelType !== LEARNED_REVIEW_MODEL_TYPE) return null;
  if (candidate.schemaVersion !== REVIEW_BUNDLE_SCHEMA_VERSION) return null;
  return mergeLearnedReviewWeights(candidate);
};

export const serializeReviewDecisionJsonl = (records: ReviewDecisionRecord[]) =>
  records.map((record) => JSON.stringify(record)).join("\n") + (records.length > 0 ? "\n" : "");

export const parseReviewDecisionJsonl = (text: string) =>
  text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ReviewDecisionRecord);

const pushAssessmentCheck = (
  findings: AutoReviewCheck[],
  issueTags: Set<ReviewIssueTag>,
  input: {
    id: string;
    label: string;
    detail: string;
    warn: boolean;
    fail: boolean;
    tag?: ReviewIssueTag;
    failSeverity?: AutoReviewCheckSeverity;
    warnSeverity?: AutoReviewCheckSeverity;
  },
) => {
  const status: AutoReviewCheckStatus = input.fail ? "fail" : input.warn ? "warn" : "pass";
  const severity =
    status === "fail"
      ? input.failSeverity ?? "major"
      : status === "warn"
        ? input.warnSeverity ?? "minor"
        : "minor";
  findings.push({
    id: input.id,
    label: input.label,
    status,
    severity,
    detail: input.detail,
  });
  if (input.tag && status !== "pass") {
    issueTags.add(input.tag);
  }
};

const summarizeAssessment = (assessment: AutoReviewCandidateAssessment) => {
  const issueLead =
    assessment.issueTags.length > 0
      ? `Issues: ${assessment.issueTags.join(", ")}.`
      : "No significant technical defects detected.";
  return `${assessment.variantLabel}: ${assessment.failCount} fail, ${assessment.warnCount} warn, ${assessment.passCount} pass checks. ${issueLead}`;
};

const summarizePreferredRoleReason = (
  selected: AutoReviewCandidateAssessment,
  challenger: AutoReviewCandidateAssessment | null,
  preferredRole: ReviewCandidateRole | null,
) => {
  if (!challenger || !preferredRole) {
    return "No challenger comparison was available.";
  }
  if (preferredRole === selected.role) {
    return `The selected output remains the lower-risk technical choice (${selected.technicalRiskScore.toFixed(
      2,
    )} vs ${challenger.technicalRiskScore.toFixed(2)} risk score).`;
  }
  return `The challenger is technically cleaner than the selected output (${challenger.technicalRiskScore.toFixed(
    2,
  )} vs ${selected.technicalRiskScore.toFixed(2)} risk score).`;
};

const buildAssessmentRiskScore = (assessment: AutoReviewCandidateAssessment, ranking: CandidateRankingBreakdown) => {
  const findingRisk = assessment.findings.reduce((total, finding) => {
    if (finding.status === "fail") {
      return total + (finding.severity === "critical" ? 1.2 : 0.8);
    }
    if (finding.status === "warn") {
      return total + (finding.severity === "major" ? 0.35 : 0.18);
    }
    return total;
  }, 0);
  return (
    findingRisk +
    Math.min(1.5, ranking.hardGatePenalty / 1500) +
    Math.max(0, ranking.learnedAdjustment / 250)
  );
};

const buildCandidateAssessment = (
  manifest: ReviewBundleManifest,
  candidate: ReviewBundleCandidate,
): AutoReviewCandidateAssessment => {
  const findings: AutoReviewCheck[] = [];
  const issueTags = new Set<ReviewIssueTag>();
  const qc = candidate.qc;
  const sourceQc = manifest.source.qc;
  const delta = candidate.sourceComparison.qcDelta;
  const alignment = candidate.sourceComparison.alignment;
  const absDurationDelta = Math.abs(alignment.durationDeltaSec);
  const absOffset = Math.abs(alignment.estimatedOffsetSec);

  pushAssessmentCheck(findings, issueTags, {
    id: "timing-integrity",
    label: "Timing Integrity",
    tag: "timing_shift",
    fail: absDurationDelta > 0.05 || (absOffset > 0.08 && alignment.confidence >= 0.35),
    warn:
      absDurationDelta > 0.02 ||
      (absOffset > 0.04 && alignment.confidence >= 0.2) ||
      (absOffset > 0.1 && alignment.confidence < 0.2),
    failSeverity: "critical",
    warnSeverity: "major",
    detail: `Duration delta ${formatSignedSeconds(alignment.durationDeltaSec)}, estimated offset ${formatSignedSeconds(
      alignment.estimatedOffsetSec,
    )}, confidence ${formatPercent(alignment.confidence)}.`,
  });

  const instabilityDelta = safeNumber(delta?.instabilityScore);
  const sentenceJumpDelta = safeNumber(delta?.sentenceJumpScore);
  const lineSwing = safeNumber(qc?.lineSwingScore);
  pushAssessmentCheck(findings, issueTags, {
    id: "level-continuity",
    label: "Dialogue Level Continuity",
    tag: "level_uneven",
    fail:
      safeNumber(qc?.instabilityScore) >= 0.42 ||
      safeNumber(qc?.sentenceJumpScore) >= 0.34 ||
      instabilityDelta >= 0.09 ||
      sentenceJumpDelta >= 0.08,
    warn:
      safeNumber(qc?.instabilityScore) >= 0.28 ||
      safeNumber(qc?.sentenceJumpScore) >= 0.22 ||
      lineSwing >= 0.24 ||
      instabilityDelta >= 0.04 ||
      sentenceJumpDelta >= 0.04,
    failSeverity: "major",
    warnSeverity: "major",
    detail: `Instability ${formatPercent(qc?.instabilityScore)}, line swing ${formatPercent(
      qc?.lineSwingScore,
    )}, sentence jump ${formatPercent(qc?.sentenceJumpScore)}; delta ${formatPercent(
      delta?.instabilityScore,
    )} / ${formatPercent(delta?.sentenceJumpScore)}.`,
  });

  const coldOpenDipDb = safeNumber(qc?.coldOpenDipDb);
  const coldOpenDipDeltaDb = safeNumber(delta?.coldOpenDipDb);
  pushAssessmentCheck(findings, issueTags, {
    id: "cold-open",
    label: "Cold-Open Level Dip",
    tag: "cold_open_dip",
    fail: coldOpenDipDb >= 3,
    warn: coldOpenDipDb >= 1.5 || coldOpenDipDeltaDb >= 0.75,
    failSeverity: "major",
    warnSeverity: "major",
    detail: `Cold-open dip ${formatDb(qc?.coldOpenDipDb)} (${formatSignedDb(
      coldOpenDipDeltaDb,
    )} vs source), risk ${formatPercent(qc?.coldOpenRiskScore)}.`,
  });

  const pauseFloorLift = safeNumber(delta?.pauseNoiseFloorDb);
  pushAssessmentCheck(findings, issueTags, {
    id: "pause-bed",
    label: "Pause Bed And Noise Floor",
    tag: "pause_noise_lift",
    fail:
      pauseFloorLift >= 2 ||
      safeNumber(delta?.pauseNoiseRisk) >= 0.08 ||
      safeNumber(qc?.pauseNoiseRisk) >= 0.45,
    warn:
      pauseFloorLift >= 0.75 ||
      safeNumber(delta?.pauseNoiseRisk) >= 0.03 ||
      safeNumber(qc?.pauseNoiseRisk) >= 0.28,
    failSeverity: "major",
    warnSeverity: "major",
    detail: `Pause floor ${formatDb(qc?.pauseNoiseFloorDb)} (${formatSignedDb(
      safeNumber(delta?.pauseNoiseFloorDb),
    )} vs source), pause-noise risk ${formatPercent(qc?.pauseNoiseRisk)} (delta ${formatPercent(
      delta?.pauseNoiseRisk,
    )}).`,
  });

  const lraDelta = diffNullable(qc?.inputLRA, sourceQc?.inputLRA);
  pushAssessmentCheck(findings, issueTags, {
    id: "dynamic-control",
    label: "Dynamic Control And Headroom",
    tag: "too_compressed",
    fail:
      safeNumber(qc?.compressionScore) >= 0.48 ||
      safeNumber(qc?.inputTP, -120) > -1.5 ||
      safeNumber(delta?.compressionScore) >= 0.09,
    warn:
      safeNumber(qc?.compressionScore) >= 0.32 ||
      safeNumber(qc?.inputTP, -120) > -2 ||
      safeNumber(delta?.compressionScore) >= 0.04 ||
      Math.abs(safeNumber(lraDelta)) >= 1.8,
    failSeverity: "major",
    warnSeverity: "major",
    detail: `True peak ${formatDb(qc?.inputTP)}, compression risk ${formatPercent(
      qc?.compressionScore,
    )}, delta ${formatPercent(delta?.compressionScore)}, LRA ${formatNumber(qc?.inputLRA, 1)} LU${
      hasFiniteNumber(lraDelta) ? ` (Δ ${lraDelta >= 0 ? "+" : ""}${lraDelta.toFixed(1)} LU)` : ""
    }.`,
  });

  pushAssessmentCheck(findings, issueTags, {
    id: "endings",
    label: "Ending Protection",
    tag: "endings_damaged",
    fail:
      safeNumber(qc?.endFadeRiskScore) >= 0.62 ||
      safeNumber(delta?.endFadeRiskScore) >= 0.12,
    warn:
      safeNumber(qc?.endFadeRiskScore) >= 0.38 ||
      safeNumber(delta?.endFadeRiskScore) >= 0.05,
    failSeverity: "critical",
    warnSeverity: "major",
    detail: `End-fade risk ${formatPercent(qc?.endFadeRiskScore)} (delta ${formatPercent(
      delta?.endFadeRiskScore,
    )}), onset ${formatPercent(qc?.onsetOvershootScore)}, mid-line sag ${formatPercent(
      qc?.midLineSagScore,
    )}.`,
  });

  const endEdgeDipDb = safeNumber(qc?.endEdgeDipDb);
  const endEdgeDipDeltaDb = safeNumber(delta?.endEdgeDipDb);
  const hasEndEdgeDipDelta = hasFiniteNumber(delta?.endEdgeDipDb);
  const endEdgeDipIsNewOrWorse = !hasEndEdgeDipDelta || endEdgeDipDeltaDb >= 1;
  const severeEndEdgeDip =
    endEdgeDipIsNewOrWorse &&
    (endEdgeDipDb >= 6 || (endEdgeDipDb >= 4.5 && (!hasEndEdgeDipDelta || endEdgeDipDeltaDb >= 1)));
  pushAssessmentCheck(findings, issueTags, {
    id: "end-edge-dip",
    label: "End-Edge Level Dip",
    tag: "endings_damaged",
    fail: severeEndEdgeDip,
    warn:
      endEdgeDipIsNewOrWorse &&
      (endEdgeDipDb >= 4 || (endEdgeDipDb >= 2.5 && (!hasEndEdgeDipDelta || endEdgeDipDeltaDb >= 1))),
    failSeverity: "major",
    warnSeverity: "major",
    detail: `End-edge dip ${formatDb(qc?.endEdgeDipDb)} (${formatSignedDb(
      endEdgeDipDeltaDb,
    )} vs source).`,
  });

  pushAssessmentCheck(findings, issueTags, {
    id: "sibilance",
    label: "Top-End Harshness And Sibilance",
    tag: "harsh_sibilance",
    fail:
      safeNumber(qc?.sibilanceScore) >= 0.42 ||
      safeNumber(delta?.sibilanceScore) >= 0.08,
    warn:
      safeNumber(qc?.sibilanceScore) >= 0.26 ||
      safeNumber(delta?.sibilanceScore) >= 0.03,
    failSeverity: "major",
    warnSeverity: "minor",
    detail: `Sibilance score ${formatPercent(qc?.sibilanceScore)} (delta ${formatPercent(
      delta?.sibilanceScore,
    )}).`,
  });

  pushAssessmentCheck(findings, issueTags, {
    id: "echo-room",
    label: "Room Imprint And Echo",
    tag: "echo_roomy",
    fail:
      safeNumber(qc?.echoScore) >= 0.32 ||
      safeNumber(delta?.echoScore) >= 0.08 ||
      safeNumber(qc?.roomScore) >= 0.48,
    warn:
      safeNumber(qc?.echoScore) >= 0.18 ||
      safeNumber(delta?.echoScore) >= 0.03 ||
      safeNumber(qc?.roomScore) >= 0.28,
    failSeverity: "major",
    warnSeverity: "minor",
    detail: `Echo ${formatPercent(qc?.echoScore)} (delta ${formatPercent(delta?.echoScore)}), room score ${formatPercent(
      qc?.roomScore,
    )}.`,
  });

  pushAssessmentCheck(findings, issueTags, {
    id: "clicks-artifacts",
    label: "Transient Artifacts And Clicks",
    tag: "clicks_artifacts",
    fail:
      safeNumber(qc?.clickScore) >= 0.18 ||
      safeNumber(delta?.clickScore) >= 0.08,
    warn:
      safeNumber(qc?.clickScore) >= 0.08 ||
      safeNumber(delta?.clickScore) >= 0.03,
    failSeverity: "major",
    warnSeverity: "minor",
    detail: `Click score ${formatPercent(qc?.clickScore)} (delta ${formatPercent(
      delta?.clickScore,
    )}).`,
  });

  pushAssessmentCheck(findings, issueTags, {
    id: "noise-contrast",
    label: "Speech To Noise Separation",
    tag: "other",
    fail:
      safeNumber(qc?.overallRisk) >= 0.56 ||
      safeNumber(delta?.overallRisk) >= 0.12 ||
      safeNumber(delta?.noiseContrastDb) <= -4,
    warn:
      safeNumber(qc?.overallRisk) >= 0.36 ||
      safeNumber(delta?.overallRisk) >= 0.05 ||
      safeNumber(delta?.noiseContrastDb) <= -1.5,
    failSeverity: "major",
    warnSeverity: "minor",
    detail: `Overall risk ${formatPercent(qc?.overallRisk)} (delta ${formatPercent(
      delta?.overallRisk,
    )}), noise contrast ${formatDb(qc?.noiseContrastDb)}${
      hasFiniteNumber(delta?.noiseContrastDb) ? ` (${formatSignedDb(delta.noiseContrastDb)} vs source)` : ""
    }.`,
  });

  pushAssessmentCheck(findings, issueTags, {
    id: "render-robustness",
    label: "Render Robustness",
    tag: "other",
    fail: candidate.renderMeta.degraded && candidate.ranking.gateReasons.length >= 2,
    warn: candidate.renderMeta.degraded || candidate.ranking.gateReasons.length > 0,
    failSeverity: "major",
    warnSeverity: "minor",
    detail: `Path ${candidate.renderMeta.renderPath}, degraded ${candidate.renderMeta.degraded ? "yes" : "no"}, gates ${
      candidate.ranking.gateReasons.length > 0 ? candidate.ranking.gateReasons.join(", ") : "none"
    }.`,
  });

  const passCount = findings.filter((finding) => finding.status === "pass").length;
  const warnCount = findings.filter((finding) => finding.status === "warn").length;
  const failCount = findings.filter((finding) => finding.status === "fail").length;
  const assessment: AutoReviewCandidateAssessment = {
    role: candidate.role,
    variantLabel: candidate.variantLabel,
    technicalRiskScore: 0,
    passCount,
    warnCount,
    failCount,
    issueTags: Array.from(issueTags),
    findings,
    summary: "",
  };
  assessment.technicalRiskScore = buildAssessmentRiskScore(assessment, candidate.ranking);
  assessment.summary = summarizeAssessment(assessment);
  return assessment;
};

export const autoReviewBundle = (manifest: ReviewBundleManifest): AutoReviewResult => {
  const selected = findCandidateRole(manifest, "winner");
  if (!selected) {
    throw new Error(`Bundle ${manifest.bundleId} is missing the selected winner candidate.`);
  }
  const challenger = findCandidateRole(manifest, "challenger");
  const selectedAssessment = buildCandidateAssessment(manifest, selected);
  const challengerAssessment = challenger ? buildCandidateAssessment(manifest, challenger) : null;

  const selectedFailsCritical = selectedAssessment.findings.some(
    (finding) => finding.status === "fail" && finding.severity === "critical",
  );
  const finalVerdict: ReviewVerdict =
    selectedFailsCritical ||
    selectedAssessment.failCount >= 2 ||
    selectedAssessment.technicalRiskScore >= 2.1
      ? "fail"
      : "pass";

  let preferredRole: ReviewCandidateRole | null = challengerAssessment ? "winner" : null;
  if (challengerAssessment) {
    const riskGap = selectedAssessment.technicalRiskScore - challengerAssessment.technicalRiskScore;
    if (
      riskGap >= 0.22 ||
      (finalVerdict === "fail" && challengerAssessment.technicalRiskScore + 0.05 < selectedAssessment.technicalRiskScore)
    ) {
      preferredRole = "challenger";
    }
  }

  const comparisonGap = challengerAssessment
    ? Math.abs(selectedAssessment.technicalRiskScore - challengerAssessment.technicalRiskScore)
    : 0.14;
  const confidence = clamp(
    0.45 +
      comparisonGap * 0.22 +
      (finalVerdict === "fail" ? 0.14 : 0) +
      (selectedFailsCritical ? 0.12 : 0) +
      (challengerAssessment && preferredRole === "challenger" ? 0.08 : 0),
    0.3,
    0.97,
  );

  const executiveSummary = `Selected output ${finalVerdict.toUpperCase()} with ${selectedAssessment.failCount} fail and ${selectedAssessment.warnCount} warn checks. ${summarizePreferredRoleReason(
    selectedAssessment,
    challengerAssessment,
    preferredRole,
  )}`;

  const buildFindingsBlock = (heading: string, assessment: AutoReviewCandidateAssessment) =>
    [
      `${heading}: ${assessment.summary}`,
      ...assessment.findings.map(
        (finding) =>
          `- ${finding.status.toUpperCase()} | ${finding.label}: ${finding.detail}`,
      ),
    ].join("\n");

  const note = [
    "Automated engineering review generated from source/output alignment, QC deltas, and render diagnostics.",
    `Selected output verdict: ${finalVerdict.toUpperCase()}.`,
    `A/B preference: ${preferredRole ?? "skip"} (confidence ${(confidence * 100).toFixed(0)}%).`,
    executiveSummary,
    buildFindingsBlock("Selected output assessment", selectedAssessment),
    challengerAssessment ? buildFindingsBlock("Challenger assessment", challengerAssessment) : null,
  ]
    .filter(Boolean)
    .join("\n\n");

  return {
    bundleId: manifest.bundleId,
    finalVerdict,
    issueTags: selectedAssessment.issueTags,
    preferredRole,
    confidence,
    note,
    executiveSummary,
    selectedAssessment,
    challengerAssessment,
  };
};

const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

const featureVectorFromMap = (featureMap: ReviewFeatureMap) =>
  REVIEW_FEATURE_KEYS.map((key) => featureMap[key]);

const mapFromFeatureVector = (values: number[]) => {
  const out = createZeroFeatureMap();
  for (let index = 0; index < REVIEW_FEATURE_KEYS.length; index += 1) {
    out[REVIEW_FEATURE_KEYS[index]] = values[index] ?? 0;
  }
  return out;
};

const findCandidateRole = (manifest: ReviewBundleManifest, role: ReviewCandidateRole) =>
  manifest.candidates.find((candidate) => candidate.role === role) ?? null;

export const evaluateLearnedReviewWeights = (
  manifests: ReviewBundleManifest[],
  decisions: ReviewDecisionRecord[],
  weights: LearnedReviewWeights,
): ReviewTrainingReport => {
  const manifestById = new Map(manifests.map((manifest) => [manifest.bundleId, manifest]));
  const issueTagCounts = Object.fromEntries(REVIEW_ISSUE_TAGS.map((tag) => [tag, 0])) as Record<
    ReviewIssueTag,
    number
  >;
  let pairwiseExamples = 0;
  let pairwiseCorrect = 0;
  let challengerWins = 0;
  let failedReviews = 0;
  let reviewedBundleCount = 0;

  for (const decision of decisions) {
    const manifest = manifestById.get(decision.bundleId);
    if (!manifest) continue;
    reviewedBundleCount += 1;
    if (decision.finalVerdict === "fail") {
      failedReviews += 1;
    }
    for (const tag of decision.issueTags) {
      issueTagCounts[tag] += 1;
    }

    if (!decision.preferredRole) continue;
    const preferred = findCandidateRole(manifest, decision.preferredRole);
    const other = findCandidateRole(
      manifest,
      decision.preferredRole === "winner" ? "challenger" : "winner",
    );
    if (!preferred || !other) continue;

    pairwiseExamples += 1;
    if (decision.preferredRole === "challenger") challengerWins += 1;
    const preferredScore = scoreCandidateWithLearnedWeights({
      baselineScore: preferred.baselineScore,
      candidateQc: preferred.qc,
      sourceQc: manifest.source.qc,
      alignment: preferred.sourceComparison.alignment,
      meta: preferred.renderMeta,
      weights,
    }).rankingScore;
    const otherScore = scoreCandidateWithLearnedWeights({
      baselineScore: other.baselineScore,
      candidateQc: other.qc,
      sourceQc: manifest.source.qc,
      alignment: other.sourceComparison.alignment,
      meta: other.renderMeta,
      weights,
    }).rankingScore;
    if (preferredScore < otherScore) {
      pairwiseCorrect += 1;
    }
  }

  return {
    manifestCount: manifests.length,
    decisionCount: decisions.length,
    reviewedBundleCount,
    pairwiseExampleCount: pairwiseExamples,
    pairwiseAccuracy: pairwiseExamples > 0 ? pairwiseCorrect / pairwiseExamples : null,
    issueTagCounts,
    challengerWins,
    failedReviews,
    weightsModelName: weights.modelName,
  };
};

export const fitLearnedReviewWeights = (
  manifests: ReviewBundleManifest[],
  decisions: ReviewDecisionRecord[],
): { weights: LearnedReviewWeights; report: ReviewTrainingReport } => {
  const weights = mergeLearnedReviewWeights(DEFAULT_LEARNED_REVIEW_WEIGHTS);
  const featureWeights = featureVectorFromMap(weights.featureWeights);
  const manifestById = new Map(manifests.map((manifest) => [manifest.bundleId, manifest]));
  const pairwiseExamples: number[][] = [];
  let reviewedCount = 0;
  const failTagCounts = Object.fromEntries(REVIEW_ISSUE_TAGS.map((tag) => [tag, 0])) as Record<
    ReviewIssueTag,
    number
  >;
  let failCount = 0;

  for (const decision of decisions) {
    const manifest = manifestById.get(decision.bundleId);
    if (!manifest) continue;
    reviewedCount += 1;
    if (decision.finalVerdict === "fail") {
      failCount += 1;
      for (const tag of decision.issueTags) {
        failTagCounts[tag] += 1;
      }
    }
    if (!decision.preferredRole) continue;
    const preferred = findCandidateRole(manifest, decision.preferredRole);
    const other = findCandidateRole(
      manifest,
      decision.preferredRole === "winner" ? "challenger" : "winner",
    );
    if (!preferred || !other) continue;
    const preferredFeatures = featureVectorFromMap(preferred.ranking.features);
    const otherFeatures = featureVectorFromMap(other.ranking.features);
    pairwiseExamples.push(otherFeatures.map((value, index) => value - preferredFeatures[index]));
  }

  const learningRate = 0.08;
  const iterations = 240;
  const regularization = 0.0015;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const gradient = new Array<number>(featureWeights.length).fill(0);
    for (const example of pairwiseExamples) {
      let dot = 0;
      for (let index = 0; index < featureWeights.length; index += 1) {
        dot += featureWeights[index] * example[index];
      }
      const prediction = sigmoid(dot);
      const error = prediction - 1;
      for (let index = 0; index < featureWeights.length; index += 1) {
        gradient[index] += error * example[index];
      }
    }
    for (let index = 0; index < featureWeights.length; index += 1) {
      const average = pairwiseExamples.length > 0 ? gradient[index] / pairwiseExamples.length : 0;
      featureWeights[index] -= learningRate * (average + regularization * featureWeights[index]);
    }
  }

  weights.featureWeights = mapFromFeatureVector(featureWeights);
  weights.modelName = `review-trained-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  weights.createdAt = new Date().toISOString();
  weights.sourceSummary = `Trained from ${reviewedCount} review(s) and ${pairwiseExamples.length} pairwise preference example(s).`;

  if (failCount > 0) {
    const tagRate = (tag: ReviewIssueTag) => failTagCounts[tag] / failCount;
    weights.penaltyWeights.durationMismatch += tagRate("timing_shift") * 1100;
    weights.penaltyWeights.offsetMismatch += tagRate("timing_shift") * 900;
    weights.penaltyWeights.endingDamage += tagRate("endings_damaged") * 550;
    weights.penaltyWeights.peakViolation += tagRate("too_compressed") * 180;
    weights.featureWeights.candidate_instabilityScore += tagRate("level_uneven") * 8;
    weights.featureWeights.candidate_sentenceJumpScore += tagRate("level_uneven") * 9;
    weights.featureWeights.candidate_pauseNoiseRisk += tagRate("pause_noise_lift") * 8;
    weights.featureWeights.candidate_compressionScore += tagRate("too_compressed") * 8;
    weights.featureWeights.candidate_sibilanceScore += tagRate("harsh_sibilance") * 7;
    weights.featureWeights.candidate_echoScore += tagRate("echo_roomy") * 8;
    weights.featureWeights.candidate_clickScore += tagRate("clicks_artifacts") * 8;
    weights.featureWeights.delta_endFadeRiskScore += tagRate("endings_damaged") * 8;
  }

  const report = evaluateLearnedReviewWeights(manifests, decisions, weights);
  return { weights, report };
};
