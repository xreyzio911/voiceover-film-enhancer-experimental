export const DEFAULT_GEMINI_AUDIO_REVIEW_MODEL = "gemini-3.1-flash-lite";
export const MAX_AUDIO_REVIEW_FILES = 12;
export const AUDIO_REVIEW_FILES_PER_GEMINI_REQUEST = 4;

export type AudioReviewControls = {
  loudnessTarget: string;
  smartMatchMode: string;
  leveler: string;
  breathControl: string;
  eqCleanup: boolean;
  roomCleanup: boolean;
  sceneBlend: boolean;
  softenHarshness: boolean;
  noiseGuard: boolean;
  floorGuard: boolean;
  cinematicColor: boolean;
  gainPlannerEnabled: boolean;
  neuralSpeechEnhancementEnabled: boolean;
};

export type AudioReviewMetricSnapshot = {
  instabilityScore: number | null;
  lineSwingScore: number | null;
  sentenceJumpScore: number | null;
  midLineSagScore: number | null;
  endFadeRiskScore: number | null;
  onsetOvershootScore: number | null;
  breathSpikeRisk: number | null;
  pauseNoiseRisk: number | null;
  compressionScore: number | null;
  clickScore: number | null;
  echoScore: number | null;
  roomScore: number | null;
  sibilanceScore: number | null;
  noiseFloorDb: number | null;
  noiseContrastDb: number | null;
  dynamicRangeDb: number | null;
  speechDutyCyclePct: number | null;
  speechSegmentCount: number | null;
};

export type AudioReviewProfileSnapshot = {
  highpassHz: number | null;
  lowMidGainDb: number | null;
  presenceGainDb: number | null;
  airGainDb: number | null;
  emotionalHarshnessCutDb: number | null;
  topEndHarshnessCutDb: number | null;
  levelingNeed: number | null;
  emotionProtection: number | null;
  toneMatchDeltaDb: number[] | null;
  noiseRisk: string;
  roomRisk: string;
  lineContinuityRisk: number | null;
  preserveEndings: boolean;
  preferSinglePassContinuity: boolean;
  useSpeechAlignedSegmentation: boolean;
  useSpeechPauseSegmentation: boolean;
  useDenoise: boolean;
  denoiseStrength: number | null;
  sibilanceScore: number | null;
  onsetTameStrength: number | null;
  sagRecoveryStrength: number | null;
  breathTameStrength: number | null;
  echoNotchCutDb: number | null;
  useTailGate: boolean;
  cinematicColorEnabled: boolean;
};

export type AudioReviewSelectedCandidate = {
  variant: string;
  reason: string | null;
  processingFlow: string | null;
  score: {
    stability: number | null;
    pause: number | null;
    compression: number | null;
    echo: number | null;
    total: number | null;
  } | null;
};

export type AudioReviewFileInput = {
  fileName: string;
  base: string;
  durationSeconds: number | null;
  source: AudioReviewMetricSnapshot;
  profile: AudioReviewProfileSnapshot | null;
  selectedCandidate: AudioReviewSelectedCandidate | null;
};

export type AudioReviewRequestPayload = {
  generatedAt: string;
  controls: AudioReviewControls;
  files: AudioReviewFileInput[];
};

export type AudioReviewFinding = {
  fileName: string;
  issue: string;
  severity: "low" | "medium" | "high";
  evidence: string;
  action: string;
};

export type AudioReviewAdjustment = {
  control: string;
  recommendation: string;
  why: string;
  risk: string;
};

export type AudioReviewAdaptiveDirectives = {
  warmthDb: number;
  presenceDb: number;
  airDb: number;
  deHarshDb: number;
  sagRecoveryBoost: number;
  onsetTameBoost: number;
  breathTameBoost: number;
  denoiseBias: number;
  roomCleanupBias: number;
  compressionBias: number;
  finalPolishIntensity: number;
};

export type AudioReviewRecommendedProfile = {
  name: string;
  selectedVariant: "cinematic-stable" | "continuity-safe" | "pause-safe" | "source-safe";
  smartMatchMode: "Off" | "Gentle" | "Balanced";
  leveler: "Minimal (no auto-leveler)" | "Gentle" | "Balanced" | "Firm";
  breathControl: "Off" | "Light" | "Medium";
  neuralSpeechEnhancement: "off";
  roomCleanup: boolean;
  softenHarshness: boolean;
  cinematicColor: boolean;
  notes: string;
};

export type AudioReviewPerFileProfile = {
  fileName: string;
  base: string;
  confidence: number;
  summary: string;
  recommendedProfile: AudioReviewRecommendedProfile;
  adaptiveDirectives: AudioReviewAdaptiveDirectives;
  profileRationale: string[];
  guardrails: string[];
  nextListeningChecks: string[];
};

export type AudioReviewResult = {
  verdict: "ready" | "adjust" | "risky";
  confidence: number;
  summary: string;
  perFileProfiles: AudioReviewPerFileProfile[];
  adjustments: AudioReviewAdjustment[];
  findings: AudioReviewFinding[];
  profileRationale: string[];
  guardrails: string[];
  nextListeningChecks: string[];
};

type AudioReviewVerdict = AudioReviewResult["verdict"];

export type AudioReviewControlPatch = Partial<
  Pick<
    AudioReviewControls,
    | "smartMatchMode"
    | "leveler"
    | "breathControl"
    | "neuralSpeechEnhancementEnabled"
    | "roomCleanup"
    | "softenHarshness"
    | "cinematicColor"
  >
>;

export type AudioReviewControlPatchResult = {
  controls: AudioReviewControlPatch;
  changedKeys: Array<keyof AudioReviewControlPatch>;
  summary: string;
};

export type SourceFirstAudioReviewPlan = {
  controls: AudioReviewControls;
  changedKeys: Array<keyof AudioReviewControlPatch>;
  selectedVariant: AudioReviewRecommendedProfile["selectedVariant"];
  adaptiveDirectives: AudioReviewAdaptiveDirectives;
  fileName: string;
  base: string;
  summary: string;
};

type GeminiAudioReviewBody = {
  systemInstruction: {
    parts: Array<{ text: string }>;
  };
  contents: Array<{
    role: "user";
    parts: Array<{ text: string }>;
  }>;
  generationConfig: {
    thinkingConfig: {
      thinkingLevel: "low";
    };
    responseMimeType: "application/json";
    responseSchema: Record<string, unknown>;
    maxOutputTokens: number;
  };
};

export type GeminiAudioReviewRequest = {
  model: string;
  body: GeminiAudioReviewBody;
};

const metricKeys: Array<keyof AudioReviewMetricSnapshot> = [
  "instabilityScore",
  "lineSwingScore",
  "sentenceJumpScore",
  "midLineSagScore",
  "endFadeRiskScore",
  "onsetOvershootScore",
  "breathSpikeRisk",
  "pauseNoiseRisk",
  "compressionScore",
  "clickScore",
  "echoScore",
  "roomScore",
  "sibilanceScore",
  "noiseFloorDb",
  "noiseContrastDb",
  "dynamicRangeDb",
  "speechDutyCyclePct",
  "speechSegmentCount",
];

const AUDIO_REVIEW_RECOMMENDED_PROFILE_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string" },
    selectedVariant: {
      type: "string",
      enum: ["cinematic-stable", "continuity-safe", "pause-safe", "source-safe"],
    },
    smartMatchMode: { type: "string", enum: ["Off", "Gentle", "Balanced"] },
    leveler: {
      type: "string",
      enum: ["Minimal (no auto-leveler)", "Gentle", "Balanced", "Firm"],
    },
    breathControl: { type: "string", enum: ["Off", "Light", "Medium"] },
    neuralSpeechEnhancement: { type: "string", enum: ["off"] },
    roomCleanup: { type: "boolean" },
    softenHarshness: { type: "boolean" },
    cinematicColor: { type: "boolean" },
    notes: { type: "string" },
  },
  required: [
    "name",
    "selectedVariant",
    "smartMatchMode",
    "leveler",
    "breathControl",
    "neuralSpeechEnhancement",
    "roomCleanup",
    "softenHarshness",
    "cinematicColor",
    "notes",
  ],
} as const;

const AUDIO_REVIEW_ADAPTIVE_DIRECTIVES_SCHEMA = {
  type: "object",
  properties: {
    warmthDb: { type: "number", minimum: -1.2, maximum: 1.2 },
    presenceDb: { type: "number", minimum: -1.2, maximum: 1.2 },
    airDb: { type: "number", minimum: -0.9, maximum: 0.9 },
    deHarshDb: { type: "number", minimum: 0, maximum: 1.2 },
    sagRecoveryBoost: { type: "number", minimum: -0.1, maximum: 0.45 },
    onsetTameBoost: { type: "number", minimum: -0.1, maximum: 0.45 },
    breathTameBoost: { type: "number", minimum: -0.1, maximum: 0.45 },
    denoiseBias: { type: "number", minimum: -0.1, maximum: 0.45 },
    roomCleanupBias: { type: "number", minimum: -0.1, maximum: 0.45 },
    compressionBias: { type: "number", minimum: -0.45, maximum: 0.45 },
    finalPolishIntensity: { type: "number", minimum: 0, maximum: 1 },
  },
  required: [
    "warmthDb",
    "presenceDb",
    "airDb",
    "deHarshDb",
    "sagRecoveryBoost",
    "onsetTameBoost",
    "breathTameBoost",
    "denoiseBias",
    "roomCleanupBias",
    "compressionBias",
    "finalPolishIntensity",
  ],
} as const;

const AUDIO_REVIEW_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    verdict: { type: "string", enum: ["ready", "adjust", "risky"] },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    summary: { type: "string" },
    perFileProfiles: {
      type: "array",
      items: {
        type: "object",
        properties: {
          fileName: { type: "string" },
          base: { type: "string" },
          confidence: { type: "number", minimum: 0, maximum: 1 },
          summary: { type: "string" },
          recommendedProfile: AUDIO_REVIEW_RECOMMENDED_PROFILE_SCHEMA,
          adaptiveDirectives: AUDIO_REVIEW_ADAPTIVE_DIRECTIVES_SCHEMA,
          profileRationale: { type: "array", items: { type: "string" } },
          guardrails: { type: "array", items: { type: "string" } },
          nextListeningChecks: { type: "array", items: { type: "string" } },
        },
        required: [
          "fileName",
          "base",
          "confidence",
          "summary",
          "recommendedProfile",
          "adaptiveDirectives",
          "profileRationale",
          "guardrails",
          "nextListeningChecks",
        ],
      },
    },
    adjustments: {
      type: "array",
      items: {
        type: "object",
        properties: {
          control: { type: "string" },
          recommendation: { type: "string" },
          why: { type: "string" },
          risk: { type: "string" },
        },
        required: ["control", "recommendation", "why", "risk"],
      },
    },
    findings: {
      type: "array",
      items: {
        type: "object",
        properties: {
          fileName: { type: "string" },
          issue: { type: "string" },
          severity: { type: "string", enum: ["low", "medium", "high"] },
          evidence: { type: "string" },
          action: { type: "string" },
        },
        required: ["fileName", "issue", "severity", "evidence", "action"],
      },
    },
    profileRationale: { type: "array", items: { type: "string" } },
    guardrails: { type: "array", items: { type: "string" } },
    nextListeningChecks: { type: "array", items: { type: "string" } },
  },
  required: [
    "verdict",
    "confidence",
    "summary",
    "perFileProfiles",
    "adjustments",
    "findings",
    "profileRationale",
    "guardrails",
    "nextListeningChecks",
  ],
} as const;

export const AUDIO_REVIEW_SYSTEM_PROMPT = `
You are a senior dialogue editor, post-production VO mastering engineer, and QA reviewer for a browser-based VO leveling app.

You are not an audio enhancer. You are a control-plane judge: read the app's objective metrics and recommend the safest profile for the next render or review pass.

The app pipeline you are reviewing:
1. Browser FFmpeg analyzes each WAV with loudness stats, spectral bands, speech-envelope QC, sibilance, noise, room, echo, compression, and line-continuity metrics.
2. The app builds an adaptive profile per file: high-pass, low-mid warmth, presence/air correction, harshness cuts, de-esser depth, room cleanup, measured-SNR noise reduction, segmentation, tone matching against the batch reference, and cinematic color.
3. The speech-aware gain planner runs before downstream dynamics. It normalizes speech runs toward a shared house target, micro-rides only where needed, ducks pauses, protects sentence endings, and locally tames plosives or body-relative speech spikes.
4. Candidate variants are rendered and QC-scored: cinematic-stable, continuity-safe, pause-safe, and source-safe. Hard gates prefer stable volume, clean pauses, low compression artifacts, and controlled echo.
5. Neural speech enhancement is temporarily disabled. Do not ask for ClearVoice, neural repair, neural cleanup, remote worker changes, neural strength changes, or a neural retry. The active path is source-first AI review, one app render, and one subtle final app polish.
6. Loudness normalization is a delivery step after mix-ready processing; it must not hide profile mistakes.

Primary quality target: actors recorded with different microphones and settings should converge toward the same rich, smooth, balanced, crystal-clear, cinematic house tone while preserving actor identity and performance intent.

Known problem to hunt: rare mid-sentence shallow-volume dips that recover later, plus occasional harsh or sharp VO. Prioritize line continuity, mid-line sag, sentence jump, sibilance, over-compression, and bright presence/air risks. Recommend subtle changes only; never propose extreme EQ, heavy compression, or fully replacing the actor's voice.

You must return one perFileProfiles entry for every input file. Each file needs its own recommendedProfile and adaptiveDirectives. Do not collapse the batch into one shared profile; use the batch only as a house-tone reference. Use adaptiveDirectives as bounded micro-intents for the adaptive DSP: warmth/presence/air bias, de-harshing, sag recovery, onset/breath taming, denoise/room bias, compression bias, and single-pass final polish intensity. These are expert nudges, not raw filter graphs.

Review depth requirement: do not give generic feedback. For each per-file recommendation, include concrete metric evidence, the exact app-side control or adaptiveDirective to use, the expected audible change, and the acceptance check. Guardrails must be phrased as things the app can enforce or the editor can verify; never write guardrails that imply changing neural enhancement internals or reranking multiple candidates.

Return only valid JSON matching the schema. Keep each string concise and concrete. If objective metrics are insufficient, say what to listen for rather than inventing certainty.
`.trim();

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const cleanString = (value: unknown, fallback = "", maxLength = 280) => {
  if (typeof value !== "string") return fallback;
  const cleaned = value.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim();
  return cleaned.slice(0, maxLength) || fallback;
};

const finiteOrNull = (value: unknown) => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  if (!Number.isFinite(numeric)) return null;
  return Math.round(numeric * 1000) / 1000;
};

const finiteNumberArray = (value: unknown, maxItems = 16) => {
  if (!Array.isArray(value)) return null;
  const normalized = value.map(finiteOrNull).filter((item): item is number => item !== null).slice(0, maxItems);
  return normalized.length > 0 ? normalized : null;
};

const uniqueStrings = (values: string[], maxItems = 8) => {
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const value of values) {
    const cleaned = cleanString(value, "", 180);
    if (!cleaned || seen.has(cleaned)) continue;
    seen.add(cleaned);
    deduped.push(cleaned);
    if (deduped.length >= maxItems) break;
  }
  return deduped;
};

export const DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES: AudioReviewAdaptiveDirectives = {
  warmthDb: 0,
  presenceDb: 0,
  airDb: 0,
  deHarshDb: 0,
  sagRecoveryBoost: 0,
  onsetTameBoost: 0,
  breathTameBoost: 0,
  denoiseBias: 0,
  roomCleanupBias: 0,
  compressionBias: 0,
  finalPolishIntensity: 0.5,
};

const boolOrFalse = (value: unknown) => value === true;

const stringEnum = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T => {
  if (typeof value === "string" && allowed.includes(value as T)) return value as T;
  return fallback;
};

const boundedNumber = (value: unknown, fallback: number, min: number, max: number) =>
  clamp(finiteOrNull(value) ?? fallback, min, max);

const normalizeAdaptiveDirectives = (value: unknown): AudioReviewAdaptiveDirectives => {
  const record = isRecord(value) ? value : {};
  return {
    warmthDb: boundedNumber(record.warmthDb, DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.warmthDb, -1.2, 1.2),
    presenceDb: boundedNumber(record.presenceDb, DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.presenceDb, -1.2, 1.2),
    airDb: boundedNumber(record.airDb, DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.airDb, -0.9, 0.9),
    deHarshDb: boundedNumber(record.deHarshDb, DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.deHarshDb, 0, 1.2),
    sagRecoveryBoost: boundedNumber(
      record.sagRecoveryBoost,
      DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.sagRecoveryBoost,
      -0.1,
      0.45,
    ),
    onsetTameBoost: boundedNumber(
      record.onsetTameBoost,
      DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.onsetTameBoost,
      -0.1,
      0.45,
    ),
    breathTameBoost: boundedNumber(
      record.breathTameBoost,
      DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.breathTameBoost,
      -0.1,
      0.45,
    ),
    denoiseBias: boundedNumber(record.denoiseBias, DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.denoiseBias, -0.1, 0.45),
    roomCleanupBias: boundedNumber(
      record.roomCleanupBias,
      DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.roomCleanupBias,
      -0.1,
      0.45,
    ),
    compressionBias: boundedNumber(
      record.compressionBias,
      DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.compressionBias,
      -0.45,
      0.45,
    ),
    finalPolishIntensity: boundedNumber(
      record.finalPolishIntensity,
      DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES.finalPolishIntensity,
      0,
      1,
    ),
  };
};

const normalizeControls = (value: unknown): AudioReviewControls | null => {
  if (!isRecord(value)) return null;
  return {
    loudnessTarget: cleanString(value.loudnessTarget, "Mix-ready only (no loudness normalize)", 120),
    smartMatchMode: cleanString(value.smartMatchMode, "Balanced", 80),
    leveler: cleanString(value.leveler, "Balanced", 80),
    breathControl: cleanString(value.breathControl, "Medium", 80),
    eqCleanup: boolOrFalse(value.eqCleanup),
    roomCleanup: boolOrFalse(value.roomCleanup),
    sceneBlend: boolOrFalse(value.sceneBlend),
    softenHarshness: boolOrFalse(value.softenHarshness),
    noiseGuard: boolOrFalse(value.noiseGuard),
    floorGuard: boolOrFalse(value.floorGuard),
    cinematicColor: boolOrFalse(value.cinematicColor),
    gainPlannerEnabled: boolOrFalse(value.gainPlannerEnabled),
    neuralSpeechEnhancementEnabled: boolOrFalse(value.neuralSpeechEnhancementEnabled),
  };
};

const normalizeMetrics = (value: unknown): AudioReviewMetricSnapshot => {
  const record = isRecord(value) ? value : {};
  return metricKeys.reduce((acc, key) => {
    acc[key] = finiteOrNull(record[key]);
    return acc;
  }, {} as AudioReviewMetricSnapshot);
};

const normalizeProfile = (value: unknown): AudioReviewProfileSnapshot | null => {
  if (!isRecord(value)) return null;
  return {
    highpassHz: finiteOrNull(value.highpassHz),
    lowMidGainDb: finiteOrNull(value.lowMidGainDb),
    presenceGainDb: finiteOrNull(value.presenceGainDb),
    airGainDb: finiteOrNull(value.airGainDb),
    emotionalHarshnessCutDb: finiteOrNull(value.emotionalHarshnessCutDb),
    topEndHarshnessCutDb: finiteOrNull(value.topEndHarshnessCutDb),
    levelingNeed: finiteOrNull(value.levelingNeed),
    emotionProtection: finiteOrNull(value.emotionProtection),
    toneMatchDeltaDb: finiteNumberArray(value.toneMatchDeltaDb, 8),
    noiseRisk: cleanString(value.noiseRisk, "unknown", 40),
    roomRisk: cleanString(value.roomRisk, "unknown", 40),
    lineContinuityRisk: finiteOrNull(value.lineContinuityRisk),
    preserveEndings: boolOrFalse(value.preserveEndings),
    preferSinglePassContinuity: boolOrFalse(value.preferSinglePassContinuity),
    useSpeechAlignedSegmentation: boolOrFalse(value.useSpeechAlignedSegmentation),
    useSpeechPauseSegmentation: boolOrFalse(value.useSpeechPauseSegmentation),
    useDenoise: boolOrFalse(value.useDenoise),
    denoiseStrength: finiteOrNull(value.denoiseStrength),
    sibilanceScore: finiteOrNull(value.sibilanceScore),
    onsetTameStrength: finiteOrNull(value.onsetTameStrength),
    sagRecoveryStrength: finiteOrNull(value.sagRecoveryStrength),
    breathTameStrength: finiteOrNull(value.breathTameStrength),
    echoNotchCutDb: finiteOrNull(value.echoNotchCutDb),
    useTailGate: boolOrFalse(value.useTailGate),
    cinematicColorEnabled: boolOrFalse(value.cinematicColorEnabled),
  };
};

const normalizeSelectedCandidate = (value: unknown): AudioReviewSelectedCandidate | null => {
  if (!isRecord(value)) return null;
  const score = isRecord(value.score)
    ? {
        stability: finiteOrNull(value.score.stability),
        pause: finiteOrNull(value.score.pause),
        compression: finiteOrNull(value.score.compression),
        echo: finiteOrNull(value.score.echo),
        total: finiteOrNull(value.score.total),
      }
    : null;
  return {
    variant: cleanString(value.variant, "unknown", 60),
    reason: cleanString(value.reason, "", 180) || null,
    processingFlow: cleanString(value.processingFlow, "", 80) || null,
    score,
  };
};

export const normalizeAudioReviewRequest = (value: unknown): AudioReviewRequestPayload | null => {
  if (!isRecord(value)) return null;
  const controls = normalizeControls(value.controls);
  if (!controls || !Array.isArray(value.files)) return null;

  const files = value.files
    .slice(0, MAX_AUDIO_REVIEW_FILES)
    .map((file): AudioReviewFileInput | null => {
      if (!isRecord(file)) return null;
      const fileName = cleanString(file.fileName, "", 180);
      const base = cleanString(file.base, fileName, 160);
      if (!fileName || !base) return null;
      return {
        fileName,
        base,
        durationSeconds: finiteOrNull(file.durationSeconds),
        source: normalizeMetrics(file.source),
        profile: normalizeProfile(file.profile),
        selectedCandidate: normalizeSelectedCandidate(file.selectedCandidate),
      };
    })
    .filter((file): file is AudioReviewFileInput => file !== null);

  if (files.length === 0) return null;

  return {
    generatedAt: cleanString(value.generatedAt, new Date().toISOString(), 80),
    controls,
    files,
  };
};

export const splitAudioReviewPayloadForGemini = (
  payload: AudioReviewRequestPayload,
): AudioReviewRequestPayload[] => {
  const normalized = normalizeAudioReviewRequest(payload);
  if (!normalized) {
    throw new Error("Invalid audio review payload.");
  }

  const chunks: AudioReviewRequestPayload[] = [];
  for (let index = 0; index < normalized.files.length; index += AUDIO_REVIEW_FILES_PER_GEMINI_REQUEST) {
    chunks.push({
      generatedAt: normalized.generatedAt,
      controls: normalized.controls,
      files: normalized.files.slice(index, index + AUDIO_REVIEW_FILES_PER_GEMINI_REQUEST),
    });
  }
  return chunks;
};

export const buildAudioReviewUserPrompt = (payload: AudioReviewRequestPayload) => {
  const normalized = normalizeAudioReviewRequest(payload);
  if (!normalized) {
    throw new Error("Invalid audio review payload.");
  }

  return [
    "Review this source-first VO batch before rendering and recommend the safest processing profile.",
    "",
    "Required source-first pipeline: per-audio AI review -> per-file adaptive profile -> one app pass -> one subtle final app polish -> result.",
    "App pipeline reminder: source analysis metrics -> per-file AI profile selection -> adaptive profile -> speech-aware gain planner -> one AI-selected render variant -> one subtle final app polish -> loudness delivery.",
    "Neural speech enhancement is temporarily off. Always set neuralSpeechEnhancement to off. Do not recommend ClearVoice, neural repair, neural worker setup, neural bypass logic, neural strength changes, or neural retry behavior.",
    "Candidate reranking is temporarily off. Pick exactly one selectedVariant from source evidence before render; do not ask for challenger renders, learned reranking, review bundles, or post-render winner selection.",
    "Return exactly one perFileProfiles item per input file, preserving each fileName and base exactly as provided. Every file can choose a different selectedVariant, controls, and adaptiveDirectives.",
    "House tone target: use the batch reference and profile toneMatchDeltaDb to pull different mics toward the same rich, smooth, balanced bass/treble profile without extreme EQ or voice identity shifts.",
    "Main failure priorities: mid-sentence shallow-volume dips, line swing, mid-line sag, sentence jumps, harsh sibilance, over-bright presence/air, over-compression, and room/echo that makes actors sound unmatched.",
    "Adaptive safeguards to respect: sagRecoveryStrength, preserveEndings, onsetTameStrength, breathTameStrength, echoNotchCutDb, denoiseStrength, and useTailGate are subtle app controls; do not recommend stacking all of them unless the metrics justify it.",
    "perFileProfiles[].adaptiveDirectives are bounded expert nudges beyond presets. Keep them subtle: prefer +/-0.2 to +/-0.5 moves, reserve larger moves for strong evidence, and make finalPolishIntensity a single-pass finishing strength, not an iteration request.",
    "Detail rules: every summary, finding, adjustment, guardrail, and listening check should say why it matters using source metrics or profile fields, what app-side action to take, and what audible/QC result should prove it worked.",
    "Recommendation rules: choose the profile and one selectedVariant from source evidence before render, keep parameter changes subtle, preserve actor identity, prefer objective gates over taste, and call out listening checks when the metrics cannot prove a problem.",
    "The selectedCandidate field may be null because this review happens before the app render. Do not require finished output metrics to make a source-first recommendation.",
    "",
    "Current batch payload:",
    JSON.stringify(normalized, null, 2),
  ].join("\n");
};

export const buildGeminiAudioReviewRequest = (
  payload: AudioReviewRequestPayload,
  model = DEFAULT_GEMINI_AUDIO_REVIEW_MODEL,
): GeminiAudioReviewRequest => ({
  model,
  body: {
    systemInstruction: {
      parts: [{ text: AUDIO_REVIEW_SYSTEM_PROMPT }],
    },
    contents: [
      {
        role: "user",
        parts: [{ text: buildAudioReviewUserPrompt(payload) }],
      },
    ],
    generationConfig: {
      thinkingConfig: {
        thinkingLevel: "low",
      },
      responseMimeType: "application/json",
      responseSchema: AUDIO_REVIEW_RESPONSE_SCHEMA,
      maxOutputTokens: 6400,
    },
  },
});

const extractJsonText = (text: string) => {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenced) return fenced[1].trim();
  const objectStart = trimmed.indexOf("{");
  const objectEnd = trimmed.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    return trimmed.slice(objectStart, objectEnd + 1);
  }
  return trimmed;
};

const repairCommonGeminiJsonText = (jsonText: string) =>
  jsonText
    .replace(/,\s*([}\]])/g, "$1")
    .replace(/"\s*[\r\n]+\s*"/g, "\",\n\"")
    .replace(/}\s*[\r\n]+\s*{/g, "},\n{")
    .replace(/]\s*[\r\n]+\s*"/g, "],\n\"")
    .replace(/}\s*[\r\n]+\s*"/g, "},\n\"")
    .replace(/(true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[\r\n]+\s*"/g, "$1,\n\"");

const parseGeminiJsonObject = (text: string) => {
  const jsonText = extractJsonText(text);
  try {
    return JSON.parse(jsonText) as unknown;
  } catch (firstError) {
    const repairedText = repairCommonGeminiJsonText(jsonText);
    if (repairedText !== jsonText) {
      try {
        return JSON.parse(repairedText) as unknown;
      } catch {
        // Keep the first parser location because it points at the provider's original response.
      }
    }
    throw firstError;
  }
};

const stringArray = (value: unknown, maxItems = 8, maxLength = 180) =>
  Array.isArray(value)
    ? value
        .map((item) => cleanString(item, "", maxLength))
        .filter(Boolean)
        .slice(0, maxItems)
    : [];

export const buildAudioReviewControlPatch = (
  profile: AudioReviewRecommendedProfile,
  currentControls: AudioReviewControls,
): AudioReviewControlPatchResult => {
  const requested: AudioReviewControlPatch = {
    smartMatchMode: profile.smartMatchMode,
    leveler: profile.leveler,
    breathControl: profile.breathControl,
    neuralSpeechEnhancementEnabled: false,
    roomCleanup: profile.roomCleanup,
    softenHarshness: profile.softenHarshness,
    cinematicColor: profile.cinematicColor,
  };

  const controls: AudioReviewControlPatch = {};
  const changedKeys: Array<keyof AudioReviewControlPatch> = [];
  for (const key of Object.keys(requested) as Array<keyof AudioReviewControlPatch>) {
    const value = requested[key];
    if (value !== undefined && currentControls[key] !== value) {
      controls[key] = value as never;
      changedKeys.push(key);
    }
  }

  return {
    controls,
    changedKeys,
    summary:
      changedKeys.length === 0
        ? "AI profile already matches current controls."
        : `AI profile changes ${changedKeys.length} control${changedKeys.length === 1 ? "" : "s"}.`,
  };
};

export const buildSourceFirstAudioReviewPlan = (
  fileReview: AudioReviewPerFileProfile,
  currentControls: AudioReviewControls,
): SourceFirstAudioReviewPlan => {
  const patch = buildAudioReviewControlPatch(fileReview.recommendedProfile, currentControls);
  return {
    controls: {
      ...currentControls,
      ...patch.controls,
      neuralSpeechEnhancementEnabled: false,
    },
    changedKeys: patch.changedKeys,
    selectedVariant: fileReview.recommendedProfile.selectedVariant,
    adaptiveDirectives: fileReview.adaptiveDirectives,
    fileName: fileReview.fileName,
    base: fileReview.base,
    summary:
      patch.changedKeys.length === 0
        ? `${fileReview.fileName}: source-first AI profile already matches current controls; render ${fileReview.recommendedProfile.selectedVariant}.`
        : `Source-first AI profile changes ${patch.changedKeys.length} control${
            patch.changedKeys.length === 1 ? "" : "s"
          } for ${fileReview.fileName}; render ${fileReview.recommendedProfile.selectedVariant}.`,
  };
};

const normalizeRecommendedProfile = (value: unknown): AudioReviewRecommendedProfile => {
  const profile = isRecord(value) ? value : {};
  return {
    name: cleanString(profile.name, "Balanced VO Profile", 100),
    selectedVariant: stringEnum(
      profile.selectedVariant,
      ["cinematic-stable", "continuity-safe", "pause-safe", "source-safe"] as const,
      "continuity-safe",
    ),
    smartMatchMode: stringEnum(profile.smartMatchMode, ["Off", "Gentle", "Balanced"] as const, "Balanced"),
    leveler: stringEnum(
      profile.leveler,
      ["Minimal (no auto-leveler)", "Gentle", "Balanced", "Firm"] as const,
      "Balanced",
    ),
    breathControl: stringEnum(profile.breathControl, ["Off", "Light", "Medium"] as const, "Medium"),
    neuralSpeechEnhancement: "off",
    roomCleanup: boolOrFalse(profile.roomCleanup),
    softenHarshness: boolOrFalse(profile.softenHarshness),
    cinematicColor: boolOrFalse(profile.cinematicColor),
    notes: cleanString(profile.notes, "", 360),
  };
};

const verdictRank: Record<AudioReviewVerdict, number> = {
  ready: 0,
  adjust: 1,
  risky: 2,
};

const worstVerdict = (results: AudioReviewResult[]): AudioReviewVerdict =>
  results.reduce<AudioReviewVerdict>(
    (worst, result) => (verdictRank[result.verdict] > verdictRank[worst] ? result.verdict : worst),
    "ready",
  );

export const mergeChunkedAudioReviewResults = (
  results: AudioReviewResult[],
  payload: AudioReviewRequestPayload,
): AudioReviewResult => {
  const normalized = normalizeAudioReviewRequest(payload);
  if (!normalized) {
    throw new Error("Invalid audio review payload.");
  }
  if (results.length === 0) {
    throw new Error("No Gemini audio review results to merge.");
  }

  const profilesByKey = new Map<string, AudioReviewPerFileProfile>();
  for (const result of results) {
    for (const profile of result.perFileProfiles) {
      if (profile.base) profilesByKey.set(profile.base, profile);
      if (profile.fileName) profilesByKey.set(profile.fileName, profile);
    }
  }

  const perFileProfiles = normalized.files
    .map((file) => profilesByKey.get(file.base) ?? profilesByKey.get(file.fileName) ?? null)
    .filter((profile): profile is AudioReviewPerFileProfile => profile !== null)
    .slice(0, MAX_AUDIO_REVIEW_FILES);
  const averageConfidence =
    results.reduce((sum, result) => sum + result.confidence, 0) / Math.max(1, results.length);

  return {
    verdict: worstVerdict(results),
    confidence: clamp(Math.round(averageConfidence * 1000) / 1000, 0, 1),
    summary:
      results.length === 1
        ? results[0].summary
        : `Gemini reviewed ${perFileProfiles.length} file${perFileProfiles.length === 1 ? "" : "s"} across ${
            results.length
          } bounded batch request${results.length === 1 ? "" : "s"}. Worst verdict: ${worstVerdict(results)}.`,
    perFileProfiles,
    adjustments: results.flatMap((result) => result.adjustments).slice(0, 12),
    findings: results.flatMap((result) => result.findings).slice(0, 16),
    profileRationale: uniqueStrings(results.flatMap((result) => result.profileRationale)),
    guardrails: uniqueStrings(results.flatMap((result) => result.guardrails)),
    nextListeningChecks: uniqueStrings(results.flatMap((result) => result.nextListeningChecks)),
  };
};

export const parseGeminiAudioReviewText = (text: string): AudioReviewResult => {
  const parsed = parseGeminiJsonObject(text);
  if (!isRecord(parsed)) {
    throw new Error("Gemini review response was not an object.");
  }

  const perFileProfiles = Array.isArray(parsed.perFileProfiles) ? parsed.perFileProfiles : [];
  const adjustments = Array.isArray(parsed.adjustments) ? parsed.adjustments : [];
  const findings = Array.isArray(parsed.findings) ? parsed.findings : [];

  return {
    verdict: stringEnum(parsed.verdict, ["ready", "adjust", "risky"] as const, "adjust"),
    confidence: clamp(finiteOrNull(parsed.confidence) ?? 0, 0, 1),
    summary: cleanString(parsed.summary, "Review completed.", 420),
    perFileProfiles: perFileProfiles
      .filter(isRecord)
      .map((item, index) => ({
        fileName: cleanString(item.fileName, `file-${index + 1}.wav`, 160),
        base: cleanString(item.base, "", 120),
        confidence: clamp(finiteOrNull(item.confidence) ?? finiteOrNull(parsed.confidence) ?? 0, 0, 1),
        summary: cleanString(item.summary, "Per-file review completed.", 320),
        recommendedProfile: normalizeRecommendedProfile(item.recommendedProfile),
        adaptiveDirectives: normalizeAdaptiveDirectives(item.adaptiveDirectives),
        profileRationale: stringArray(item.profileRationale, 6, 180),
        guardrails: stringArray(item.guardrails, 6, 180),
        nextListeningChecks: stringArray(item.nextListeningChecks, 6, 180),
      }))
      .slice(0, 12),
    adjustments: adjustments
      .filter(isRecord)
      .map((item) => ({
        control: cleanString(item.control, "Control", 80),
        recommendation: cleanString(item.recommendation, "", 180),
        why: cleanString(item.why, "", 220),
        risk: cleanString(item.risk, "Low", 80),
      }))
      .slice(0, 8),
    findings: findings
      .filter(isRecord)
      .map((item) => ({
        fileName: cleanString(item.fileName, "Unknown file", 160),
        issue: cleanString(item.issue, "Audio issue", 120),
        severity: stringEnum(item.severity, ["low", "medium", "high"] as const, "medium"),
        evidence: cleanString(item.evidence, "", 220),
        action: cleanString(item.action, "", 220),
      }))
      .slice(0, 10),
    profileRationale: stringArray(parsed.profileRationale),
    guardrails: stringArray(parsed.guardrails),
    nextListeningChecks: stringArray(parsed.nextListeningChecks),
  };
};
