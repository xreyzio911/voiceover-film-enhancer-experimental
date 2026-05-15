import assert from "node:assert/strict";
import test from "node:test";
import {
  buildRenderRiskProfile,
  compareCandidateScores,
  selectQcUnavailableFallbackCandidate,
  shouldPreferCandidate,
  type CandidateRenderMeta,
  type CandidateScore,
} from "./renderRecovery.ts";

const buildMeta = (overrides: Partial<CandidateRenderMeta> = {}): CandidateRenderMeta => ({
  strategyLabel: "primary chain",
  renderPath: "speech-pause-segmented",
  segmentedHealthy: true,
  degraded: false,
  degradeReasons: [],
  analysisWindowsAttempted: 5,
  analysisWindowsSucceeded: 5,
  analysisWindowsDropped: 0,
  ...overrides,
});

const buildScore = (overrides: Partial<CandidateScore> = {}): CandidateScore => {
  const score = {
    stability: 0.84,
    pause: 0.31,
    compression: 0.2,
    echo: 0.64,
    total: 0,
    ...overrides,
  };
  score.total = score.stability * 1000 + score.pause * 100 + score.compression * 10 + score.echo;
  return score;
};

test("render risk goes high for long sparse high-segment files", () => {
  const risk = buildRenderRiskProfile({
    durationSeconds: 812,
    longSparseMode: true,
    plannedSegmentCount: 29,
    speechSpanCount: 25,
    candidateVariant: "cinematic-stable",
    useRoomCleanup: true,
    useAdaptiveNoiseReduction: false,
    priorFatalRenderError: false,
    sentenceJumpScore: 0.18,
    mergedSegmentCount: 21,
  });

  assert.equal(risk.level, "high");
  assert.equal(risk.recycleWorkerBeforeRender, true);
  assert.equal(risk.disableSegmentGainMatch, true);
  assert.equal(risk.shouldUseFixedSegmentation, true);
});

test("render risk keeps segment gain matching only for strong sentence jumps", () => {
  const risk = buildRenderRiskProfile({
    durationSeconds: 605,
    longSparseMode: true,
    plannedSegmentCount: 26,
    speechSpanCount: 18,
    candidateVariant: "continuity-safe",
    useRoomCleanup: false,
    useAdaptiveNoiseReduction: false,
    priorFatalRenderError: false,
    sentenceJumpScore: 0.44,
    mergedSegmentCount: 16,
  });

  assert.equal(risk.level, "high");
  assert.equal(risk.disableSegmentGainMatch, false);
  assert.equal(risk.shouldUseFixedSegmentation, false);
});

test("healthy segmented candidate can beat degraded recovered candidate when close", () => {
  const currentScore = buildScore({ stability: 0.84, pause: 0.31, compression: 0.2, echo: 0.7 });
  const currentMeta = buildMeta({
    renderPath: "single-pass-recovered",
    segmentedHealthy: false,
    degraded: true,
    degradeReasons: ["segment-render-memory-fault", "single-pass-recovery"],
  });
  const challengerScore = buildScore({ stability: 0.86, pause: 0.33, compression: 0.24, echo: 0.62 });
  const challengerMeta = buildMeta();

  const decision = shouldPreferCandidate(challengerScore, challengerMeta, currentScore, currentMeta);

  assert.equal(decision.select, true);
  assert.equal(decision.reason, "prefer healthy segmented");
});

test("recovered single-pass candidate must materially beat healthy segmented winner", () => {
  const currentScore = buildScore({ stability: 0.84, pause: 0.31, compression: 0.2, echo: 0.64 });
  const currentMeta = buildMeta();
  const challengerScore = buildScore({ stability: 0.83, pause: 0.3, compression: 0.19, echo: 0.55 });
  const challengerMeta = buildMeta({
    renderPath: "single-pass-recovered",
    segmentedHealthy: false,
    degraded: true,
    degradeReasons: ["single-pass-recovery"],
  });

  const decision = shouldPreferCandidate(challengerScore, challengerMeta, currentScore, currentMeta);

  assert.equal(decision.select, false);
  assert.equal(decision.reason, "protected healthy segmented");
});

test("raw score deltas still choose a winner when rounded summaries look tied", () => {
  const currentScore = buildScore({ stability: 0.8404, pause: 0.3104, compression: 0.2004, echo: 0.6404 });
  const challengerScore = buildScore({ stability: 0.8402, pause: 0.3104, compression: 0.2004, echo: 0.6404 });

  const decision = shouldPreferCandidate(challengerScore, buildMeta(), currentScore, buildMeta());

  assert.equal(compareCandidateScores(challengerScore, currentScore) < 0, true);
  assert.equal(decision.select, true);
  assert.equal(decision.reason, "winner by raw stability delta");
});

test("rankingScore overrides raw totals when learned reranking is available", () => {
  const currentScore = buildScore({
    stability: 0.8404,
    pause: 0.3104,
    compression: 0.2004,
    echo: 0.6404,
    rankingScore: 320.4,
  });
  const challengerScore = buildScore({
    stability: 0.8404,
    pause: 0.3104,
    compression: 0.2004,
    echo: 0.6404,
    rankingScore: 210.2,
  });

  const decision = shouldPreferCandidate(challengerScore, buildMeta(), currentScore, buildMeta());

  assert.equal(compareCandidateScores(challengerScore, currentScore) < 0, true);
  assert.equal(decision.select, true);
  assert.equal(decision.reason, "winner by learned ranking");
});

test("unavailable QC candidate cannot become the first winner", () => {
  const challengerScore = buildScore({
    stability: 0.1,
    pause: 0.1,
    compression: 0.1,
    echo: 0.1,
    rankingScore: 1,
    gateReasons: ["qc-unavailable"],
  });

  const decision = shouldPreferCandidate(
    challengerScore,
    buildMeta({ degraded: true, degradeReasons: ["qc-unavailable"] }),
    null,
    null,
  );

  assert.equal(decision.select, false);
  assert.equal(decision.reason, "candidate unavailable for selection");
});

test("qc-unavailable fallback chooses rendered cinematic candidate when all QC failed", () => {
  const cinematicScore = buildScore({ gateReasons: ["qc-unavailable"] });
  const continuityScore = buildScore({ gateReasons: ["qc-unavailable"] });

  const selection = selectQcUnavailableFallbackCandidate([
    {
      variant: "cinematic-stable",
      index: 0,
      hasAudio: true,
      meta: buildMeta({ degraded: true, degradeReasons: ["analysis-window-drop", "qc-unavailable"] }),
      score: cinematicScore,
    },
    {
      variant: "continuity-safe",
      index: 1,
      hasAudio: true,
      meta: buildMeta({ degraded: true, degradeReasons: ["analysis-window-drop", "qc-unavailable"] }),
      score: continuityScore,
    },
  ]);

  assert.equal(selection?.candidate.variant, "cinematic-stable");
  assert.match(selection?.reason ?? "", /healthy rendered segmented audio/);
});

test("qc-unavailable fallback prefers cleaner render metadata over variant priority", () => {
  const selection = selectQcUnavailableFallbackCandidate([
    {
      variant: "cinematic-stable",
      index: 0,
      hasAudio: true,
      meta: buildMeta({
        renderPath: "single-pass-recovered",
        segmentedHealthy: false,
        degraded: true,
        degradeReasons: ["segment-render-memory-fault", "single-pass-recovery", "qc-unavailable"],
      }),
      score: buildScore({ gateReasons: ["qc-unavailable"] }),
    },
    {
      variant: "continuity-safe",
      index: 1,
      hasAudio: true,
      meta: buildMeta({ degraded: true, degradeReasons: ["analysis-window-drop", "qc-unavailable"] }),
      score: buildScore({ gateReasons: ["qc-unavailable"] }),
    },
  ]);

  assert.equal(selection?.candidate.variant, "continuity-safe");
});

test("qc-unavailable fallback rejects candidates without rendered audio or with planner gates", () => {
  const selection = selectQcUnavailableFallbackCandidate([
    {
      variant: "cinematic-stable",
      index: 0,
      hasAudio: false,
      meta: buildMeta({ degraded: true, degradeReasons: ["qc-unavailable"] }),
      score: buildScore({ gateReasons: ["qc-unavailable"] }),
    },
    {
      variant: "continuity-safe",
      index: 1,
      hasAudio: true,
      meta: buildMeta({ degraded: true, degradeReasons: ["planner-apply-failed", "qc-unavailable"] }),
      score: buildScore({ gateReasons: ["planner-apply-failed", "qc-unavailable"] }),
    },
  ]);

  assert.equal(selection, null);
});
