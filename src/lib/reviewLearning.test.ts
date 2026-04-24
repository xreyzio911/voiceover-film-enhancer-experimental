import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_LEARNED_REVIEW_WEIGHTS,
  REVIEW_BUNDLE_SCHEMA_VERSION,
  autoReviewBundle,
  buildReviewMetricDelta,
  estimateAlignmentMetrics,
  fitLearnedReviewWeights,
  parseLearnedReviewWeights,
  parseReviewDecisionJsonl,
  scoreCandidateWithLearnedWeights,
  serializeReviewDecisionJsonl,
  toReviewMetricSnapshot,
  type AlignmentMetrics,
  type LearnedReviewWeights,
  type ReviewBundleManifest,
  type ReviewDecisionRecord,
} from "./reviewLearning.ts";
import type { CandidateRenderMeta, CandidateScore } from "./renderRecovery.ts";

const buildMeta = (overrides: Partial<CandidateRenderMeta> = {}): CandidateRenderMeta => ({
  strategyLabel: "primary chain",
  renderPath: "speech-aligned-segmented",
  segmentedHealthy: true,
  degraded: false,
  degradeReasons: [],
  analysisWindowsAttempted: 4,
  analysisWindowsSucceeded: 4,
  analysisWindowsDropped: 0,
  ...overrides,
});

const buildScore = (overrides: Partial<CandidateScore> = {}): CandidateScore => {
  const score = {
    stability: 0.22,
    pause: 0.11,
    compression: 0.08,
    echo: 0.05,
    total: 0,
    ...overrides,
  };
  score.total = score.stability * 1000 + score.pause * 100 + score.compression * 10 + score.echo;
  return score;
};

const buildAlignment = (overrides: Partial<AlignmentMetrics> = {}): AlignmentMetrics => ({
  durationSourceSec: 10,
  durationCandidateSec: 10,
  durationDeltaSec: 0,
  durationDeltaPct: 0,
  estimatedOffsetSec: 0,
  confidence: 0.92,
  ...overrides,
});

const sourceQc = toReviewMetricSnapshot({
  inputTP: -2.4,
  overallRisk: 0.18,
  instabilityScore: 0.16,
  sentenceJumpScore: 0.12,
  pauseNoiseRisk: 0.14,
  compressionScore: 0.1,
  clickScore: 0.06,
  echoScore: 0.08,
  endFadeRiskScore: 0.07,
  sibilanceScore: 0.09,
  pauseNoiseFloorDb: -72,
  noiseContrastDb: 28,
});

const buildManifest = (bundleId: string): ReviewBundleManifest => {
  const winnerQc = toReviewMetricSnapshot({
    inputTP: -2.1,
    overallRisk: 0.28,
    instabilityScore: 0.24,
    sentenceJumpScore: 0.22,
    pauseNoiseRisk: 0.26,
    compressionScore: 0.18,
    clickScore: 0.08,
    echoScore: 0.09,
    endFadeRiskScore: 0.1,
    sibilanceScore: 0.12,
    pauseNoiseFloorDb: -69,
    noiseContrastDb: 26,
  });
  const challengerQc = toReviewMetricSnapshot({
    inputTP: -2.3,
    overallRisk: 0.2,
    instabilityScore: 0.17,
    sentenceJumpScore: 0.15,
    pauseNoiseRisk: 0.13,
    compressionScore: 0.1,
    clickScore: 0.05,
    echoScore: 0.06,
    endFadeRiskScore: 0.06,
    sibilanceScore: 0.08,
    pauseNoiseFloorDb: -72.5,
    noiseContrastDb: 29,
  });
  const winnerScore = buildScore({ stability: 0.26, pause: 0.18, compression: 0.11, echo: 0.08 });
  const challengerScore = buildScore({ stability: 0.18, pause: 0.1, compression: 0.07, echo: 0.05 });
  const winnerAlignment = buildAlignment({ estimatedOffsetSec: 0.01 });
  const challengerAlignment = buildAlignment({ estimatedOffsetSec: 0.002 });

  const winnerRanking = scoreCandidateWithLearnedWeights({
    baselineScore: winnerScore,
    candidateQc: winnerQc,
    sourceQc,
    alignment: winnerAlignment,
    meta: buildMeta(),
  });
  const challengerRanking = scoreCandidateWithLearnedWeights({
    baselineScore: challengerScore,
    candidateQc: challengerQc,
    sourceQc,
    alignment: challengerAlignment,
    meta: buildMeta(),
  });

  return {
    schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
    bundleId,
    createdAt: "2026-04-22T00:00:00.000Z",
    source: {
      fileName: `${bundleId}.wav`,
      audioFile: "source.wav",
      durationSec: 10,
      sampleRate: 48000,
      qc: sourceQc,
    },
    decisionContext: {
      jobBase: bundleId,
      loudnessTarget: "ATSC A/85 (-24 LKFS, -2 dBTP)",
      selectedVariant: "cinematic-stable",
      selectedReason: "better score",
      learnedWeightsName: DEFAULT_LEARNED_REVIEW_WEIGHTS.modelName,
      learnedWeightsSource: "default",
      reviewModelType: DEFAULT_LEARNED_REVIEW_WEIGHTS.modelType,
    },
    candidates: [
      {
        role: "winner",
        audioFile: "winner.wav",
        variantLabel: "cinematic-stable",
        renderMeta: buildMeta(),
        baselineScore: winnerScore,
        ranking: winnerRanking,
        qc: winnerQc,
        sourceComparison: {
          alignment: winnerAlignment,
          qcDelta: buildReviewMetricDelta(sourceQc, winnerQc),
        },
        selectionReason: "better score",
      },
      {
        role: "challenger",
        audioFile: "challenger.wav",
        variantLabel: "continuity-safe",
        renderMeta: buildMeta({ strategyLabel: "fallback" }),
        baselineScore: challengerScore,
        ranking: challengerRanking,
        qc: challengerQc,
        sourceComparison: {
          alignment: challengerAlignment,
          qcDelta: buildReviewMetricDelta(sourceQc, challengerQc),
        },
        selectionReason: null,
      },
    ],
  };
};

test("parseLearnedReviewWeights merges valid input and rejects invalid payloads", () => {
  const parsed = parseLearnedReviewWeights({
    schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
    modelType: DEFAULT_LEARNED_REVIEW_WEIGHTS.modelType,
    modelName: "custom-review",
    createdAt: "2026-04-22T00:00:00.000Z",
    sourceSummary: "custom",
    intercept: -1.5,
    gateThresholds: {
      maxDurationDeltaSec: 0.03,
    },
    penaltyWeights: {
      peakViolation: 123,
    },
    featureWeights: {
      ...DEFAULT_LEARNED_REVIEW_WEIGHTS.featureWeights,
      candidate_echoScore: 99,
    },
  } satisfies Partial<LearnedReviewWeights>);

  assert.ok(parsed);
  assert.equal(parsed.modelName, "custom-review");
  assert.equal(parsed.gateThresholds.maxDurationDeltaSec, 0.03);
  assert.equal(parsed.penaltyWeights.peakViolation, 123);
  assert.equal(parsed.featureWeights.candidate_echoScore, 99);
  assert.equal(parseLearnedReviewWeights({ schemaVersion: 999 }), null);
});

test("estimateAlignmentMetrics detects coarse source-output offset", () => {
  const sampleRate = 1000;
  const source = new Float32Array(sampleRate * 3);
  const candidate = new Float32Array(sampleRate * 3);

  for (let index = 400; index < 700; index += 1) source[index] = 0.8;
  for (let index = 1200; index < 1500; index += 1) source[index] = 0.6;
  for (let index = 600; index < 900; index += 1) candidate[index] = 0.8;
  for (let index = 1400; index < 1700; index += 1) candidate[index] = 0.6;

  const alignment = estimateAlignmentMetrics(source, sampleRate, candidate, sampleRate);

  assert.ok(Math.abs(alignment.estimatedOffsetSec) >= 0.18);
  assert.ok(Math.abs(alignment.estimatedOffsetSec) <= 0.24);
  assert.ok(alignment.confidence > 0.25);
});

test("scoreCandidateWithLearnedWeights applies hard gates for timing, peak, and endings", () => {
  const ranking = scoreCandidateWithLearnedWeights({
    baselineScore: buildScore(),
    candidateQc: toReviewMetricSnapshot({
      inputTP: -0.8,
      endFadeRiskScore: 0.92,
      overallRisk: 0.5,
      instabilityScore: 0.42,
      sentenceJumpScore: 0.31,
      pauseNoiseRisk: 0.28,
      compressionScore: 0.24,
      clickScore: 0.1,
      echoScore: 0.09,
      sibilanceScore: 0.08,
      pauseNoiseFloorDb: -63,
      noiseContrastDb: 20,
    }),
    sourceQc,
    alignment: buildAlignment({
      durationCandidateSec: 10.24,
      durationDeltaSec: 0.24,
      durationDeltaPct: 2.4,
      estimatedOffsetSec: 0.16,
      confidence: 0.8,
    }),
    meta: buildMeta({ degraded: true, degradeReasons: ["analysis-window-drop"] }),
  });

  assert.ok(ranking.hardGatePenalty > 0);
  assert.ok(ranking.gateReasons.includes("duration-mismatch"));
  assert.ok(ranking.gateReasons.includes("timing-offset"));
  assert.ok(ranking.gateReasons.includes("peak-violation"));
  assert.ok(ranking.gateReasons.includes("ending-damage"));
  assert.ok(ranking.gateReasons.includes("source-regression"));
  assert.ok(ranking.rankingScore > ranking.baselineTotal);
});

test("scoreCandidateWithLearnedWeights hard-penalizes source regression without timing faults", () => {
  const ranking = scoreCandidateWithLearnedWeights({
    baselineScore: buildScore({ stability: 0.16, pause: 0.1, compression: 0.08, echo: 0.06 }),
    candidateQc: toReviewMetricSnapshot({
      inputTP: -2.4,
      overallRisk: 0.34,
      instabilityScore: 0.27,
      sentenceJumpScore: 0.22,
      pauseNoiseRisk: 0.24,
      compressionScore: 0.19,
      clickScore: 0.14,
      echoScore: 0.16,
      endFadeRiskScore: 0.12,
      sibilanceScore: 0.18,
      pauseNoiseFloorDb: -69.5,
      noiseContrastDb: 24,
    }),
    sourceQc,
    alignment: buildAlignment(),
    meta: buildMeta(),
  });

  assert.ok(ranking.gateReasons.includes("source-regression"));
  assert.ok(!ranking.gateReasons.includes("duration-mismatch"));
  assert.ok(!ranking.gateReasons.includes("timing-offset"));
  assert.ok(ranking.hardGatePenalty > 0);
});

test("review decision JSONL round-trips", () => {
  const records: ReviewDecisionRecord[] = [
    {
      schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
      bundleId: "bundle-1",
      reviewedAt: "2026-04-22T00:00:00.000Z",
      finalVerdict: "fail",
      issueTags: ["timing_shift", "other"],
      preferredRole: "challenger",
      confidence: 0.75,
      note: "timing moved",
    },
  ];

  const roundTrip = parseReviewDecisionJsonl(serializeReviewDecisionJsonl(records));
  assert.deepEqual(roundTrip, records);
});

test("fitLearnedReviewWeights is deterministic for the same labeled manifests", () => {
  const manifests = [buildManifest("bundle-a"), buildManifest("bundle-b")];
  const decisions: ReviewDecisionRecord[] = [
    {
      schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
      bundleId: "bundle-a",
      reviewedAt: "2026-04-22T00:00:00.000Z",
      finalVerdict: "pass",
      issueTags: [],
      preferredRole: "challenger",
      confidence: 1,
      note: null,
    },
    {
      schemaVersion: REVIEW_BUNDLE_SCHEMA_VERSION,
      bundleId: "bundle-b",
      reviewedAt: "2026-04-22T00:00:00.000Z",
      finalVerdict: "fail",
      issueTags: ["timing_shift", "level_uneven"],
      preferredRole: "challenger",
      confidence: 0.75,
      note: null,
    },
  ];

  const first = fitLearnedReviewWeights(manifests, decisions);
  const second = fitLearnedReviewWeights(manifests, decisions);

  assert.deepEqual(first.weights.featureWeights, second.weights.featureWeights);
  assert.deepEqual(first.weights.penaltyWeights, second.weights.penaltyWeights);
  assert.equal(first.report.pairwiseExampleCount, 2);
  assert.ok((first.report.pairwiseAccuracy ?? 0) >= 0.5);
});

test("autoReviewBundle flags technical defects and prefers the cleaner challenger", () => {
  const manifest = buildManifest("bundle-auto");
  const winner = manifest.candidates.find((candidate) => candidate.role === "winner");
  const challenger = manifest.candidates.find((candidate) => candidate.role === "challenger");
  assert.ok(winner);
  assert.ok(challenger);

  if (!winner || !challenger) {
    throw new Error("Missing test candidates.");
  }

  winner.qc = toReviewMetricSnapshot({
    ...winner.qc,
    inputTP: -0.9,
    pauseNoiseRisk: 0.48,
    compressionScore: 0.55,
    endFadeRiskScore: 0.71,
    sibilanceScore: 0.29,
    echoScore: 0.21,
    clickScore: 0.12,
    overallRisk: 0.6,
  });
  winner.sourceComparison.alignment = buildAlignment({
    durationCandidateSec: 10.11,
    durationDeltaSec: 0.11,
    durationDeltaPct: 1.1,
    estimatedOffsetSec: 0.12,
    confidence: 0.86,
  });
  winner.sourceComparison.qcDelta = buildReviewMetricDelta(
    manifest.source.qc,
    winner.qc,
  );
  winner.ranking = scoreCandidateWithLearnedWeights({
    baselineScore: winner.baselineScore,
    candidateQc: winner.qc,
    sourceQc: manifest.source.qc,
    alignment: winner.sourceComparison.alignment,
    meta: winner.renderMeta,
  });

  challenger.qc = toReviewMetricSnapshot({
    ...challenger.qc,
    inputTP: -2.4,
    pauseNoiseRisk: 0.14,
    compressionScore: 0.11,
    endFadeRiskScore: 0.08,
    sibilanceScore: 0.09,
    echoScore: 0.07,
    clickScore: 0.04,
    overallRisk: 0.18,
  });
  challenger.sourceComparison.alignment = buildAlignment({
    estimatedOffsetSec: 0.01,
    confidence: 0.9,
  });
  challenger.sourceComparison.qcDelta = buildReviewMetricDelta(
    manifest.source.qc,
    challenger.qc,
  );
  challenger.ranking = scoreCandidateWithLearnedWeights({
    baselineScore: challenger.baselineScore,
    candidateQc: challenger.qc,
    sourceQc: manifest.source.qc,
    alignment: challenger.sourceComparison.alignment,
    meta: challenger.renderMeta,
  });

  const auto = autoReviewBundle(manifest);

  assert.equal(auto.finalVerdict, "fail");
  assert.equal(auto.preferredRole, "challenger");
  assert.ok(auto.confidence >= 0.6);
  assert.ok(auto.issueTags.includes("timing_shift"));
  assert.ok(auto.issueTags.includes("pause_noise_lift"));
  assert.ok(auto.issueTags.includes("too_compressed"));
  assert.ok(auto.issueTags.includes("endings_damaged"));
  assert.match(auto.note, /Selected output verdict: FAIL/);
});
