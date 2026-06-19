export type RoomCleanupRisk = "low" | "medium" | "high";

export type EchoRoomCleanupInput = {
  roomCleanupEnabled: boolean;
  roomRisk: RoomCleanupRisk;
  echoScore: number;
  analysisConfidence: number;
  preserveEndings: boolean;
  pauseNoiseRisk: number;
  echoNotchCutDb?: number;
  endFadeRiskScore?: number;
  lineContinuityRisk?: number;
};

export type EchoRoomCleanupDecision = {
  echoDominantRoom: boolean;
  severeEchoRoom: boolean;
  allowDereverbDuringEndingProtection: boolean;
  allowTailGateDuringEndingProtection: boolean;
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export const buildEchoRoomCleanupDecision = (input: EchoRoomCleanupInput): EchoRoomCleanupDecision => {
  const echoScore = clamp01(input.echoScore);
  const analysisConfidence = clamp01(input.analysisConfidence);
  const pauseNoiseRisk = clamp01(input.pauseNoiseRisk);
  const echoNotchCutDb = Math.max(0, input.echoNotchCutDb ?? 0);
  const endFadeRiskScore = clamp01(input.endFadeRiskScore ?? 0);
  const lineContinuityRisk = clamp01(input.lineContinuityRisk ?? 0);
  const endingFragility = Math.max(endFadeRiskScore, lineContinuityRisk * 0.85);
  const roomCleanupReady = input.roomCleanupEnabled && input.roomRisk !== "low" && analysisConfidence >= 0.5;

  const subtleHighRoomEcho = input.roomRisk === "high" && echoScore >= 0.56 && echoNotchCutDb >= 0.72;
  const subtleMediumRoomEcho = input.roomRisk === "medium" && echoScore >= 0.6 && echoNotchCutDb >= 0.82;
  const echoDominantRoom =
    roomCleanupReady &&
    (echoScore >= 0.68 || subtleHighRoomEcho || subtleMediumRoomEcho);

  const severeEchoRoom =
    roomCleanupReady &&
    input.roomRisk === "high" &&
    (echoScore >= 0.72 || (echoScore >= 0.64 && echoNotchCutDb >= 0.98));

  return {
    echoDominantRoom,
    severeEchoRoom,
    allowDereverbDuringEndingProtection: severeEchoRoom || (echoDominantRoom && echoScore >= 0.56),
    allowTailGateDuringEndingProtection:
      input.preserveEndings && severeEchoRoom && pauseNoiseRisk <= 0.42 && endingFragility < 0.36,
  };
};

export const deriveEarlyEchoCancelStrength = (input: {
  severeEchoRoom: boolean;
  echoScore: number;
  drynessScore: number;
  echoDelayMs: number | null;
  roomCleanupBias?: number;
}) => {
  const echoDelayMs = input.echoDelayMs;
  const delayUsable =
    input.severeEchoRoom &&
    echoDelayMs !== null &&
    Number.isFinite(echoDelayMs) &&
    echoDelayMs >= 24 &&
    echoDelayMs <= 90;
  if (!delayUsable) return 0;

  return clamp(
    0.14 +
      clamp01(input.echoScore) * 0.22 +
      (1 - clamp01(input.drynessScore)) * 0.14 +
      clamp(input.roomCleanupBias ?? 0, -0.3, 0.6) * 0.04,
    0.16,
    0.4,
  );
};
