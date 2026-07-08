"use client";

export type RenderPath =
  | "speech-pause-segmented"
  | "speech-aligned-segmented"
  | "fixed-segmented"
  | "single-pass"
  | "single-pass-recovered";

export type DegradeReason =
  | "segment-render-memory-fault"
  | "analysis-window-retry"
  | "analysis-window-drop"
  | "qc-unavailable"
  | "planner-required"
  | "planner-apply-failed"
  | "audibility-dropout-guard"
  | "single-pass-recovery";

export type CandidateScore = {
  stability: number;
  pause: number;
  compression: number;
  echo: number;
  total: number;
  hardGatePenalty?: number;
  learnedAdjustment?: number;
  rankingScore?: number;
  gateReasons?: string[];
};

export type CandidateRenderMeta = {
  strategyLabel: string;
  renderPath: RenderPath;
  segmentedHealthy: boolean;
  degraded: boolean;
  degradeReasons: DegradeReason[];
  analysisWindowsAttempted: number;
  analysisWindowsSucceeded: number;
  analysisWindowsDropped: number;
};

export type RenderRiskProfile = {
  level: "normal" | "high";
  durationSeconds: number;
  longSparseMode: boolean;
  plannedSegmentCount: number;
  speechSpanCount: number;
  candidateVariant: "cinematic-stable" | "continuity-safe" | "pause-safe" | "source-safe";
  useRoomCleanup: boolean;
  useAdaptiveNoiseReduction: boolean;
  priorFatalRenderError: boolean;
  targetProcessedSegmentCount: number;
  mergePauseThresholdSec: number;
  disableSegmentGainMatch: boolean;
  recycleWorkerBeforeRender: boolean;
  shouldUseFixedSegmentation: boolean;
};

export type QcUnavailableFallbackCandidate<TVariant extends string = string> = {
  variant: TVariant;
  index: number;
  hasAudio: boolean;
  meta: CandidateRenderMeta;
  score: CandidateScore;
};

export type QcUnavailableFallbackSelection<TVariant extends string = string> = {
  candidate: QcUnavailableFallbackCandidate<TVariant>;
  reason: string;
};

type RenderRiskInput = {
  durationSeconds: number;
  longSparseMode: boolean;
  plannedSegmentCount: number;
  speechSpanCount: number;
  candidateVariant: "cinematic-stable" | "continuity-safe" | "pause-safe" | "source-safe";
  useRoomCleanup: boolean;
  useAdaptiveNoiseReduction: boolean;
  priorFatalRenderError: boolean;
  sentenceJumpScore: number;
  mergedSegmentCount?: number | null;
};

const HEALTHY_SEGMENTED_STABILITY_DELTA = 0.03;
const HEALTHY_SEGMENTED_PAUSE_DELTA = 0.03;
const HEALTHY_SEGMENTED_COMPRESSION_DELTA = 0.05;
const QC_ONLY_DEGRADE_REASONS = new Set<DegradeReason>([
  "analysis-window-retry",
  "analysis-window-drop",
  "qc-unavailable",
]);

const roundedScore = (value: number) => Math.round(value * 100);

export const isHealthySegmentedRender = (meta: CandidateRenderMeta) =>
  meta.segmentedHealthy &&
  (meta.renderPath === "speech-pause-segmented" ||
    meta.renderPath === "speech-aligned-segmented" ||
    meta.renderPath === "fixed-segmented");

const isRecoveredSinglePass = (meta: CandidateRenderMeta) => meta.renderPath === "single-pass-recovered";

const withinHealthySegmentedTolerance = (challenger: CandidateScore, recovered: CandidateScore) =>
  challenger.stability <= recovered.stability + HEALTHY_SEGMENTED_STABILITY_DELTA &&
  challenger.pause <= recovered.pause + HEALTHY_SEGMENTED_PAUSE_DELTA &&
  challenger.compression <= recovered.compression + HEALTHY_SEGMENTED_COMPRESSION_DELTA;

const materiallyBetterThanHealthySegmented = (recovered: CandidateScore, healthy: CandidateScore) =>
  recovered.stability + HEALTHY_SEGMENTED_STABILITY_DELTA < healthy.stability &&
  recovered.pause + HEALTHY_SEGMENTED_PAUSE_DELTA < healthy.pause &&
  recovered.compression + HEALTHY_SEGMENTED_COMPRESSION_DELTA < healthy.compression;

const hasUnselectableGate = (score: CandidateScore) =>
  (score.gateReasons ?? []).some((reason) =>
    reason === "qc-unavailable" ||
    reason === "planner-required" ||
    reason === "planner-apply-failed"
  );

const hasPlannerGate = (score: CandidateScore) =>
  (score.gateReasons ?? []).some((reason) => reason === "planner-required" || reason === "planner-apply-failed");

const hasQcUnavailableGate = (score: CandidateScore) =>
  (score.gateReasons ?? []).includes("qc-unavailable");

const hasRenderDegradeReason = (meta: CandidateRenderMeta) =>
  meta.degradeReasons.some((reason) => !QC_ONLY_DEGRADE_REASONS.has(reason));

const fallbackVariantPriority = (variant: string) => {
  if (variant === "cinematic-stable") return 0;
  if (variant === "continuity-safe") return 1;
  if (variant === "pause-safe") return 2;
  if (variant === "source-safe" || variant === "core-safe") return 3;
  return 4;
};

const fallbackRenderPathPriority = (meta: CandidateRenderMeta) => {
  if (meta.renderPath === "fixed-segmented") return 0;
  if (meta.renderPath === "speech-pause-segmented" || meta.renderPath === "speech-aligned-segmented") return 1;
  if (meta.renderPath === "single-pass") return 2;
  return 3;
};

const compareQcUnavailableFallbackCandidates = <TVariant extends string>(
  left: QcUnavailableFallbackCandidate<TVariant>,
  right: QcUnavailableFallbackCandidate<TVariant>,
) => {
  const leftRenderDegraded = hasRenderDegradeReason(left.meta);
  const rightRenderDegraded = hasRenderDegradeReason(right.meta);
  if (leftRenderDegraded !== rightRenderDegraded) return leftRenderDegraded ? 1 : -1;

  const leftHealthy = isHealthySegmentedRender(left.meta);
  const rightHealthy = isHealthySegmentedRender(right.meta);
  if (leftHealthy !== rightHealthy) return leftHealthy ? -1 : 1;

  const leftRecovered = isRecoveredSinglePass(left.meta);
  const rightRecovered = isRecoveredSinglePass(right.meta);
  if (leftRecovered !== rightRecovered) return leftRecovered ? 1 : -1;

  const variantDelta = fallbackVariantPriority(left.variant) - fallbackVariantPriority(right.variant);
  if (variantDelta !== 0) return variantDelta;

  const pathDelta = fallbackRenderPathPriority(left.meta) - fallbackRenderPathPriority(right.meta);
  if (pathDelta !== 0) return pathDelta;

  const scoreDelta = compareCandidateScores(left.score, right.score);
  if (scoreDelta !== 0) return scoreDelta;

  return left.index - right.index;
};

export const selectQcUnavailableFallbackCandidate = <TVariant extends string>(
  candidates: QcUnavailableFallbackCandidate<TVariant>[],
): QcUnavailableFallbackSelection<TVariant> | null => {
  const eligible = candidates.filter(
    (candidate) => candidate.hasAudio && hasQcUnavailableGate(candidate.score) && !hasPlannerGate(candidate.score),
  );
  if (eligible.length === 0) return null;

  const [candidate] = [...eligible].sort(compareQcUnavailableFallbackCandidates);
  const reason = hasRenderDegradeReason(candidate.meta)
    ? "QC-unavailable fallback with render degradation because no cleaner rendered candidate was available"
    : isHealthySegmentedRender(candidate.meta)
      ? "QC-unavailable fallback from healthy rendered segmented audio"
      : "QC-unavailable fallback from rendered audio";
  return { candidate, reason };
};

export const buildRenderRiskProfile = (input: RenderRiskInput): RenderRiskProfile => {
  const highRisk =
    input.plannedSegmentCount >= 24 ||
    (input.durationSeconds >= 480 && input.longSparseMode) ||
    (input.speechSpanCount >= 18 && (input.useRoomCleanup || input.useAdaptiveNoiseReduction)) ||
    input.priorFatalRenderError;
  const mergedSegmentCount = input.mergedSegmentCount ?? input.plannedSegmentCount;
  return {
    level: highRisk ? "high" : "normal",
    durationSeconds: input.durationSeconds,
    longSparseMode: input.longSparseMode,
    plannedSegmentCount: input.plannedSegmentCount,
    speechSpanCount: input.speechSpanCount,
    candidateVariant: input.candidateVariant,
    useRoomCleanup: input.useRoomCleanup,
    useAdaptiveNoiseReduction: input.useAdaptiveNoiseReduction,
    priorFatalRenderError: input.priorFatalRenderError,
    targetProcessedSegmentCount: 18,
    mergePauseThresholdSec: 0.6,
    disableSegmentGainMatch: highRisk && input.sentenceJumpScore < 0.4,
    recycleWorkerBeforeRender: highRisk,
    shouldUseFixedSegmentation: highRisk && mergedSegmentCount > 18,
  };
};

export const compareCandidateScores = (left: CandidateScore, right: CandidateScore) => {
  if (
    typeof left.rankingScore === "number" &&
    Number.isFinite(left.rankingScore) &&
    typeof right.rankingScore === "number" &&
    Number.isFinite(right.rankingScore) &&
    left.rankingScore !== right.rankingScore
  ) {
    return left.rankingScore - right.rankingScore;
  }
  if (left.stability !== right.stability) return left.stability - right.stability;
  if (left.pause !== right.pause) return left.pause - right.pause;
  if (left.compression !== right.compression) return left.compression - right.compression;
  if (left.echo !== right.echo) return left.echo - right.echo;
  return left.total - right.total;
};

export const explainCandidateDelta = (winner: CandidateScore, loser: CandidateScore) => {
  if (
    typeof winner.rankingScore === "number" &&
    typeof loser.rankingScore === "number" &&
    winner.rankingScore !== loser.rankingScore
  ) {
    return "winner by learned ranking";
  }
  if (winner.stability !== loser.stability) return "winner by raw stability delta";
  if (winner.pause !== loser.pause) return "winner by raw pause delta";
  if (winner.compression !== loser.compression) return "winner by raw compression delta";
  if (winner.echo !== loser.echo) return "winner by raw echo delta";
  return "winner by raw total delta";
};

export const shouldPreferCandidate = (
  challengerScore: CandidateScore,
  challengerMeta: CandidateRenderMeta,
  currentScore: CandidateScore | null,
  currentMeta: CandidateRenderMeta | null
) => {
  if (hasUnselectableGate(challengerScore)) {
    return { select: false, reason: "candidate unavailable for selection" };
  }

  if (currentScore === null || currentMeta === null) {
    return { select: true, reason: "first completed candidate" };
  }

  const challengerHealthySegmented = isHealthySegmentedRender(challengerMeta);
  const currentHealthySegmented = isHealthySegmentedRender(currentMeta);
  const challengerRecovered = isRecoveredSinglePass(challengerMeta);
  const currentRecovered = isRecoveredSinglePass(currentMeta);

  if (
    challengerHealthySegmented &&
    currentRecovered &&
    withinHealthySegmentedTolerance(challengerScore, currentScore)
  ) {
    return { select: true, reason: "prefer healthy segmented" };
  }

  if (
    challengerRecovered &&
    currentHealthySegmented &&
    !materiallyBetterThanHealthySegmented(challengerScore, currentScore)
  ) {
    return { select: false, reason: "protected healthy segmented" };
  }

  const compare = compareCandidateScores(challengerScore, currentScore);
  if (compare < 0) {
    const roundedTie =
      roundedScore(challengerScore.stability) === roundedScore(currentScore.stability) &&
      roundedScore(challengerScore.pause) === roundedScore(currentScore.pause) &&
      roundedScore(challengerScore.compression) === roundedScore(currentScore.compression) &&
      roundedScore(challengerScore.echo) === roundedScore(currentScore.echo);
    return {
      select: true,
      reason: roundedTie ? explainCandidateDelta(challengerScore, currentScore) : "better score",
    };
  }

  return { select: false, reason: "kept current winner" };
};
