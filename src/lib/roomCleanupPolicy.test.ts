import assert from "node:assert/strict";
import test from "node:test";
import { buildEchoRoomCleanupDecision, deriveEarlyEchoCancelStrength } from "./roomCleanupPolicy.ts";

test("severe echo room can clean despite ending protection", () => {
  const decision = buildEchoRoomCleanupDecision({
    roomCleanupEnabled: true,
    roomRisk: "high",
    echoScore: 0.766,
    analysisConfidence: 1,
    preserveEndings: true,
    pauseNoiseRisk: 0.134,
  });

  assert.equal(decision.echoDominantRoom, true);
  assert.equal(decision.severeEchoRoom, true);
  assert.equal(decision.allowDereverbDuringEndingProtection, true);
  assert.equal(decision.allowTailGateDuringEndingProtection, true);
});

test("normal ending protection still blocks room cleanup exceptions", () => {
  const decision = buildEchoRoomCleanupDecision({
    roomCleanupEnabled: true,
    roomRisk: "medium",
    echoScore: 0.42,
    analysisConfidence: 0.9,
    preserveEndings: true,
    pauseNoiseRisk: 0.1,
  });

  assert.equal(decision.echoDominantRoom, false);
  assert.equal(decision.severeEchoRoom, false);
  assert.equal(decision.allowDereverbDuringEndingProtection, false);
  assert.equal(decision.allowTailGateDuringEndingProtection, false);
});

test("high pause noise prevents ending-protected tail gate", () => {
  const decision = buildEchoRoomCleanupDecision({
    roomCleanupEnabled: true,
    roomRisk: "high",
    echoScore: 0.8,
    analysisConfidence: 1,
    preserveEndings: true,
    pauseNoiseRisk: 0.72,
  });

  assert.equal(decision.severeEchoRoom, true);
  assert.equal(decision.allowDereverbDuringEndingProtection, true);
  assert.equal(decision.allowTailGateDuringEndingProtection, false);
});

test("fragile endings still allow dereverb but block tail gating", () => {
  const decision = buildEchoRoomCleanupDecision({
    roomCleanupEnabled: true,
    roomRisk: "high",
    echoScore: 0.78,
    analysisConfidence: 0.92,
    preserveEndings: true,
    pauseNoiseRisk: 0.18,
    echoNotchCutDb: 1.1,
    endFadeRiskScore: 0.72,
    lineContinuityRisk: 0.61,
  });

  assert.equal(decision.echoDominantRoom, true);
  assert.equal(decision.severeEchoRoom, true);
  assert.equal(decision.allowDereverbDuringEndingProtection, true);
  assert.equal(decision.allowTailGateDuringEndingProtection, false);
});

test("subtle high-room echo qualifies for ending-safe dereverb", () => {
  const decision = buildEchoRoomCleanupDecision({
    roomCleanupEnabled: true,
    roomRisk: "high",
    echoScore: 0.58,
    analysisConfidence: 0.83,
    preserveEndings: true,
    pauseNoiseRisk: 0.16,
    echoNotchCutDb: 0.92,
    endFadeRiskScore: 0.18,
    lineContinuityRisk: 0.24,
  });

  assert.equal(decision.echoDominantRoom, true);
  assert.equal(decision.severeEchoRoom, false);
  assert.equal(decision.allowDereverbDuringEndingProtection, true);
  assert.equal(decision.allowTailGateDuringEndingProtection, false);
});

test("early echo cancellation is bounded to severe rooms with usable short delay", () => {
  assert.equal(
    deriveEarlyEchoCancelStrength({
      severeEchoRoom: true,
      echoScore: 0.766,
      drynessScore: 0.333,
      echoDelayMs: 40,
    }),
    0.4,
  );

  assert.equal(
    deriveEarlyEchoCancelStrength({
      severeEchoRoom: true,
      echoScore: 0.9,
      drynessScore: 0.1,
      echoDelayMs: 140,
    }),
    0,
  );

  assert.equal(
    deriveEarlyEchoCancelStrength({
      severeEchoRoom: false,
      echoScore: 0.9,
      drynessScore: 0.1,
      echoDelayMs: 40,
    }),
    0,
  );
});
