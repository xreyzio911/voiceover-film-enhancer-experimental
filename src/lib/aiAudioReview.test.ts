import assert from "node:assert/strict";
import test from "node:test";
import {
  AUDIO_REVIEW_FILES_PER_GEMINI_REQUEST,
  buildAudioReviewControlPatch,
  buildCorrectiveDirectivesForIssueTags,
  buildSourceFirstAudioReviewPlan,
  buildAudioReviewUserPrompt,
  buildGeminiAudioReviewRequest,
  hasNonTrivialAdaptiveDirectives,
  mergeChunkedAudioReviewResults,
  normalizeAdaptiveDirectives,
  normalizeAudioReviewRequest,
  parseGeminiAudioReviewText,
  splitAudioReviewPayloadForGemini,
} from "./aiAudioReview.ts";

const baseControls = {
  loudnessTarget: "Mix-ready only (no loudness normalize)",
  smartMatchMode: "Balanced",
  leveler: "Balanced",
  breathControl: "Medium",
  eqCleanup: true,
  roomCleanup: true,
  sceneBlend: false,
  softenHarshness: true,
  noiseGuard: true,
  floorGuard: true,
  cinematicColor: true,
  gainPlannerEnabled: true,
  neuralSpeechEnhancementEnabled: false,
};

const reviewPayload = {
  generatedAt: "2026-06-19T00:00:00.000Z",
  reviewStage: "source" as const,
  controls: baseControls,
  files: [
    {
      fileName: "actor-a.wav",
      base: "actor_a",
      durationSeconds: 42.5,
      source: {
        integratedLufs: -23.4,
        instabilityScore: 0.28,
        lineSwingScore: 0.61,
        sentenceJumpScore: 0.37,
        midLineSagScore: 0.52,
        endFadeRiskScore: 0.18,
        onsetOvershootScore: 0.24,
        breathSpikeRisk: 0.31,
        sibilanceScore: 0.72,
        compressionScore: 0.34,
        clickScore: 0.06,
        pauseNoiseRisk: 0.22,
        echoScore: 0.18,
        roomScore: 0.2,
        noiseFloorDb: -68,
        noiseContrastDb: 26,
        dynamicRangeDb: 13.4,
        speechDutyCyclePct: 48,
        speechSegmentCount: 12,
        bandSpectrumDb: [-30, -24, -22, -22, -23, -22, -20, -24],
      },
      profile: {
        highpassHz: 78,
        lowMidGainDb: -1.2,
        presenceGainDb: -0.4,
        airGainDb: -0.25,
        emotionalHarshnessCutDb: 0.9,
        topEndHarshnessCutDb: 0.68,
        levelingNeed: 0.36,
        emotionProtection: 0.22,
        toneMatchDeltaDb: [0.4, -0.7, 0, 0.2, -0.9, -0.3, 0, 0.5],
        noiseRisk: "low",
        roomRisk: "low",
        lineContinuityRisk: 0.54,
        preserveEndings: true,
        preferSinglePassContinuity: true,
        useSpeechAlignedSegmentation: false,
        useSpeechPauseSegmentation: false,
        useDenoise: false,
        denoiseStrength: 0.08,
        sibilanceScore: 0.72,
        onsetTameStrength: 0.2,
        sagRecoveryStrength: 0.52,
        breathTameStrength: 0.35,
        echoNotchCutDb: 0.45,
        useTailGate: false,
        cinematicColorEnabled: true,
      },
      selectedCandidate: {
        variant: "continuity-safe",
        reason: "line continuity risk",
        processingFlow: "app-final-polish",
        score: {
          stability: 0.22,
          pause: 0.08,
          compression: 0.2,
          echo: 0.05,
          total: 229,
        },
      },
    },
  ],
};

test("audio review prompt teaches Gemini the app pipeline and failure priorities", () => {
  const prompt = buildAudioReviewUserPrompt(reviewPayload);

  assert.match(prompt, /source-first/i);
  assert.match(prompt, /before rendering/i);
  assert.match(prompt, /per-audio AI review -> per-file adaptive profile -> one app pass -> one subtle final app polish/i);
  assert.match(prompt, /neural speech enhancement is temporarily off/i);
  assert.match(prompt, /Always set neuralSpeechEnhancement to off/i);
  assert.match(prompt, /Candidate reranking is temporarily off/i);
  assert.match(prompt, /one selectedVariant from source evidence/i);
  assert.match(prompt, /one perFileProfiles item per input file/i);
  assert.match(prompt, /adaptiveDirectives/i);
  assert.match(prompt, /Detail rules/i);
  assert.match(prompt, /source metrics or profile fields/i);
  assert.match(prompt, /audible\/QC result/i);
  assert.match(prompt, /speech-aware gain planner/i);
  assert.match(prompt, /one AI-selected render variant/i);
  assert.match(prompt, /mid-sentence shallow-volume/i);
  assert.match(prompt, /harshness/i);
  assert.match(prompt, /house tone/i);
  assert.match(prompt, /toneMatchDeltaDb/i);
  assert.match(prompt, /sagRecoveryStrength/i);
  assert.match(prompt, /actor-a\.wav/);
  assert.match(prompt, /continuity-safe/);
});

test("Gemini request uses Flash-Lite with low thinking and JSON schema output", () => {
  const request = buildGeminiAudioReviewRequest(reviewPayload);

  assert.equal(request.model, "gemini-3.1-flash-lite");
  assert.equal(request.body.generationConfig.thinkingConfig.thinkingLevel, "low");
  assert.equal(request.body.generationConfig.responseMimeType, "application/json");
  assert.equal(request.body.generationConfig.maxOutputTokens, 6400);
  assert.equal(request.body.generationConfig.responseSchema.type, "object");
  const schema = request.body.generationConfig.responseSchema as { properties?: Record<string, unknown> };
  assert.ok(schema.properties?.perFileProfiles);
  assert.match(request.body.systemInstruction.parts[0].text, /VO mastering/i);
  assert.match(request.body.systemInstruction.parts[0].text, /source-first AI review, one app render, and one subtle final app polish/i);
  assert.match(request.body.systemInstruction.parts[0].text, /never write guardrails that imply changing neural enhancement internals/i);
});

test("normalizes review payloads and bounds batch size", () => {
  const files = Array.from({ length: 16 }, (_, index) => ({
    ...reviewPayload.files[0],
    fileName: `actor-${index}.wav`,
    source: {
      ...reviewPayload.files[0].source,
      lineSwingScore: Number.NaN,
    },
  }));

  const normalized = normalizeAudioReviewRequest({
    generatedAt: reviewPayload.generatedAt,
    controls: baseControls,
    files,
  });

  assert.ok(normalized);
  assert.equal(normalized.files.length, 12);
  assert.equal(normalized.files[0].source.lineSwingScore, null);
  assert.equal(normalized.files[0].fileName, "actor-0.wav");
  assert.deepEqual(normalized.files[0].profile?.toneMatchDeltaDb, [0.4, -0.7, 0, 0.2, -0.9, -0.3, 0, 0.5]);
  assert.equal(normalized.files[0].profile?.sagRecoveryStrength, 0.52);
});

test("splits 10+ file audio review payloads into bounded Gemini requests", () => {
  const files = Array.from({ length: 12 }, (_, index) => ({
    ...reviewPayload.files[0],
    fileName: `actor-${String(index + 1).padStart(2, "0")}.wav`,
    base: `actor_${String(index + 1).padStart(2, "0")}`,
  }));
  const normalized = normalizeAudioReviewRequest({
    generatedAt: reviewPayload.generatedAt,
    controls: baseControls,
    files,
  });

  assert.ok(normalized);
  const chunks = splitAudioReviewPayloadForGemini(normalized);

  assert.equal(AUDIO_REVIEW_FILES_PER_GEMINI_REQUEST, 4);
  assert.deepEqual(
    chunks.map((chunk) => chunk.files.length),
    [4, 4, 4],
  );
  assert.equal(chunks[0].files[0].fileName, "actor-01.wav");
  assert.equal(chunks[2].files[3].fileName, "actor-12.wav");
  assert.deepEqual(chunks[1].controls, normalized.controls);
  assert.equal(chunks[2].generatedAt, reviewPayload.generatedAt);
});

test("merges chunked Gemini review results in original file order with worst verdict", () => {
  const files = Array.from({ length: 10 }, (_, index) => ({
    ...reviewPayload.files[0],
    fileName: `actor-${String(index + 1).padStart(2, "0")}.wav`,
    base: `actor_${String(index + 1).padStart(2, "0")}`,
  }));
  const payload = normalizeAudioReviewRequest({
    generatedAt: reviewPayload.generatedAt,
    controls: baseControls,
    files,
  });

  assert.ok(payload);
  const profileFor = (fileIndex: number, confidence: number) => ({
    fileName: `actor-${String(fileIndex).padStart(2, "0")}.wav`,
    base: `actor_${String(fileIndex).padStart(2, "0")}`,
    confidence,
    summary: `Actor ${fileIndex} reviewed.`,
    recommendedProfile: {
      name: "Continuity-Safe Cinematic",
      selectedVariant: "continuity-safe" as const,
      smartMatchMode: "Balanced" as const,
      leveler: "Gentle" as const,
      breathControl: "Medium" as const,
      neuralSpeechEnhancement: "off" as const,
      roomCleanup: true,
      softenHarshness: true,
      cinematicColor: true,
      notes: "Keep correction subtle.",
    },
    adaptiveDirectives: {
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
        targetLoudnessBiasDb: 0,
        levelerBias: 0,
        headGuardBoost: 0,
      },
    profileRationale: [`Actor ${fileIndex} rationale.`],
    guardrails: [`Actor ${fileIndex} guardrail.`],
    nextListeningChecks: [`Actor ${fileIndex} check.`],
  });

  const merged = mergeChunkedAudioReviewResults(
    [
      {
        verdict: "ready",
        confidence: 0.88,
        summary: "First chunk ready.",
        perFileProfiles: [profileFor(1, 0.91), profileFor(2, 0.89), profileFor(3, 0.87), profileFor(4, 0.86)],
        adjustments: [],
        findings: [],
        profileRationale: ["First chunk rationale."],
        guardrails: ["First chunk guardrail."],
        nextListeningChecks: ["First chunk check."],
      },
      {
        verdict: "risky",
        confidence: 0.72,
        summary: "Last chunk risky.",
        perFileProfiles: [profileFor(9, 0.71), profileFor(10, 0.69)],
        adjustments: [],
        findings: [
          {
            fileName: "actor-10.wav",
            issue: "Echo risk",
            severity: "high",
            evidence: "echoScore 0.48",
            action: "Use room cleanup and verify tail",
          },
        ],
        profileRationale: ["Last chunk rationale."],
        guardrails: ["Last chunk guardrail."],
        nextListeningChecks: ["Last chunk check."],
      },
      {
        verdict: "adjust",
        confidence: 0.8,
        summary: "Middle chunk adjust.",
        perFileProfiles: [profileFor(5, 0.85), profileFor(6, 0.83), profileFor(7, 0.81), profileFor(8, 0.79)],
        adjustments: [
          {
            control: "Room cleanup",
            recommendation: "Keep enabled",
            why: "Medium room risk",
            risk: "Low",
          },
        ],
        findings: [],
        profileRationale: ["Middle chunk rationale."],
        guardrails: ["Middle chunk guardrail."],
        nextListeningChecks: ["Middle chunk check."],
      },
    ],
    payload,
  );

  assert.equal(merged.verdict, "risky");
  assert.equal(merged.confidence, 0.8);
  assert.deepEqual(
    merged.perFileProfiles.map((profile) => profile.fileName),
    files.map((file) => file.fileName),
  );
  assert.equal(merged.adjustments.length, 1);
  assert.equal(merged.findings.length, 1);
  assert.match(merged.summary, /10 file/);
  assert.deepEqual(merged.profileRationale, [
    "First chunk rationale.",
    "Last chunk rationale.",
    "Middle chunk rationale.",
  ]);
});

test("parses Gemini JSON review and clamps unsafe confidence values", () => {
  const parsed = parseGeminiAudioReviewText(`
    {
      "verdict": "adjust",
      "confidence": 1.7,
      "summary": "Continuity-safe profile is preferred.",
      "perFileProfiles": [
        {
          "fileName": "actor-a.wav",
          "base": "actor_a",
          "confidence": 1.4,
          "summary": "Actor A needs continuity-safe polish.",
          "recommendedProfile": {
            "name": "Continuity-Safe Cinematic",
            "selectedVariant": "continuity-safe",
            "smartMatchMode": "Balanced",
            "leveler": "Balanced",
            "breathControl": "Medium",
            "neuralSpeechEnhancement": "off",
            "roomCleanup": true,
            "softenHarshness": true,
            "cinematicColor": true,
            "notes": "Keep changes subtle."
          },
          "adaptiveDirectives": {
            "warmthDb": 1.9,
            "presenceDb": -1.6,
            "airDb": -1.4,
            "deHarshDb": 1.8,
            "sagRecoveryBoost": 1.2,
            "onsetTameBoost": -0.5,
            "breathTameBoost": 0.7,
            "denoiseBias": 0.8,
            "roomCleanupBias": 1.4,
            "compressionBias": -1.5,
            "finalPolishIntensity": 1.3,
            "targetLoudnessBiasDb": 2.5,
            "levelerBias": -1.1,
            "headGuardBoost": 1.4
          },
          "profileRationale": ["Line continuity dominates."],
          "guardrails": ["Reject if harshness increases."],
          "nextListeningChecks": ["Listen for shallow mid-sentence dips."]
        }
      ],
      "adjustments": [
        {
          "control": "Soften harshness",
          "recommendation": "Keep enabled",
          "why": "High sibilance score",
          "risk": "Low"
        }
      ],
      "findings": [
        {
          "fileName": "actor-a.wav",
          "issue": "Mid-line sag",
          "severity": "high",
          "evidence": "midLineSagScore 0.52",
          "action": "Use continuity-safe candidate"
        }
      ],
      "profileRationale": ["Line continuity dominates."],
      "guardrails": ["Reject if harshness increases."],
      "nextListeningChecks": ["Listen for shallow mid-sentence dips."]
    }
  `);

  assert.equal(parsed.verdict, "adjust");
  assert.equal(parsed.confidence, 1);
  assert.equal(parsed.perFileProfiles.length, 1);
  assert.equal(parsed.perFileProfiles[0].fileName, "actor-a.wav");
  assert.equal(parsed.perFileProfiles[0].base, "actor_a");
  assert.equal(parsed.perFileProfiles[0].confidence, 1);
  assert.equal(parsed.perFileProfiles[0].recommendedProfile.selectedVariant, "continuity-safe");
  assert.equal(parsed.perFileProfiles[0].recommendedProfile.neuralSpeechEnhancement, "off");
  assert.deepEqual(parsed.perFileProfiles[0].adaptiveDirectives, {
    warmthDb: 1.8,
    presenceDb: -1.6,
    airDb: -1.2,
    deHarshDb: 1.8,
    sagRecoveryBoost: 0.7,
    onsetTameBoost: -0.15,
    breathTameBoost: 0.7,
    denoiseBias: 0.7,
    roomCleanupBias: 0.7,
    compressionBias: -0.6,
    finalPolishIntensity: 1,
    targetLoudnessBiasDb: 1.5,
    levelerBias: -0.5,
    headGuardBoost: 1,
  });
  assert.equal(parsed.adjustments.length, 1);
  assert.equal(parsed.findings[0].severity, "high");
});

test("parses Gemini review with a missing comma between adjacent array strings", () => {
  const parsed = parseGeminiAudioReviewText(`
    {
      "verdict": "adjust",
      "confidence": 0.82,
      "summary": "Use continuity-safe processing.",
      "perFileProfiles": [
        {
          "fileName": "actor-a.wav",
          "base": "actor_a",
          "confidence": 0.82,
          "summary": "Actor A needs source-first continuity protection.",
          "recommendedProfile": {
            "name": "Continuity-Safe Cinematic",
            "selectedVariant": "continuity-safe",
            "smartMatchMode": "Balanced",
            "leveler": "Gentle",
            "breathControl": "Medium",
            "neuralSpeechEnhancement": "off",
            "roomCleanup": true,
            "softenHarshness": true,
            "cinematicColor": true,
            "notes": "Keep the app-side correction subtle."
          },
          "adaptiveDirectives": {
            "warmthDb": 0.2,
            "presenceDb": -0.1,
            "airDb": 0,
            "deHarshDb": 0.25,
            "sagRecoveryBoost": 0.3,
            "onsetTameBoost": 0.1,
            "breathTameBoost": 0.15,
            "denoiseBias": 0.1,
            "roomCleanupBias": 0.2,
            "compressionBias": -0.15,
            "finalPolishIntensity": 0.7,
            "targetLoudnessBiasDb": 0,
            "levelerBias": 0,
            "headGuardBoost": 0
          },
          "profileRationale": [
            "midLineSagScore 0.52 needs continuity-safe recovery"
            "preserveEndings is already active so keep tail protection"
          ],
          "guardrails": ["Reject if sentence endings dull."],
          "nextListeningChecks": ["Listen for the final word tail."]
        }
      ],
      "adjustments": [],
      "findings": [],
      "profileRationale": ["Source metrics show line-continuity risk."],
      "guardrails": ["Render once and verify endings."],
      "nextListeningChecks": ["Check the final word tail."]
    }
  `);

  assert.deepEqual(parsed.perFileProfiles[0].profileRationale, [
    "midLineSagScore 0.52 needs continuity-safe recovery",
    "preserveEndings is already active so keep tail protection",
  ]);
  assert.equal(parsed.perFileProfiles[0].recommendedProfile.selectedVariant, "continuity-safe");
});

test("builds a bounded automatic control patch from Gemini profile recommendations", () => {
  const review = parseGeminiAudioReviewText(`
    {
      "verdict": "adjust",
      "confidence": 0.84,
      "summary": "Use a safer profile.",
      "perFileProfiles": [
        {
          "fileName": "actor-a.wav",
          "base": "actor_a",
          "confidence": 0.84,
          "summary": "Use a safer profile.",
          "recommendedProfile": {
            "name": "Continuity-Safe Cinematic",
            "selectedVariant": "continuity-safe",
            "smartMatchMode": "Balanced",
            "leveler": "Gentle",
            "breathControl": "Light",
            "neuralSpeechEnhancement": "off",
            "roomCleanup": true,
            "softenHarshness": true,
            "cinematicColor": true,
            "notes": "Subtle correction only."
          },
          "adaptiveDirectives": {
            "warmthDb": 0,
            "presenceDb": 0,
            "airDb": 0,
            "deHarshDb": 0,
            "sagRecoveryBoost": 0,
            "onsetTameBoost": 0,
            "breathTameBoost": 0,
            "denoiseBias": 0,
            "roomCleanupBias": 0,
            "compressionBias": 0,
            "finalPolishIntensity": 0.5,
            "targetLoudnessBiasDb": 0,
            "levelerBias": 0,
            "headGuardBoost": 0
          },
          "profileRationale": ["Line continuity risk is high."],
          "guardrails": ["Rerun once only."],
          "nextListeningChecks": ["Listen for sag recovery."]
        }
      ],
      "adjustments": [],
      "findings": [],
      "profileRationale": ["Line continuity risk is high."],
      "guardrails": ["Rerun once only."],
      "nextListeningChecks": ["Listen for sag recovery."]
    }
  `);

  const patch = buildAudioReviewControlPatch(review.perFileProfiles[0].recommendedProfile, {
    ...baseControls,
    smartMatchMode: "Gentle",
    leveler: "Firm",
    breathControl: "Medium",
    neuralSpeechEnhancementEnabled: false,
    roomCleanup: false,
    softenHarshness: false,
    cinematicColor: false,
  });

  assert.deepEqual(patch.controls, {
    smartMatchMode: "Balanced",
    leveler: "Gentle",
    breathControl: "Light",
    roomCleanup: true,
    softenHarshness: true,
    cinematicColor: true,
  });
  assert.deepEqual(patch.changedKeys, [
    "smartMatchMode",
    "leveler",
    "breathControl",
    "roomCleanup",
    "softenHarshness",
    "cinematicColor",
  ]);
  assert.match(patch.summary, /6 control/);
});

test("builds separate source-first render plans from per-file Gemini recommendations", () => {
  const review = parseGeminiAudioReviewText(`
    {
      "verdict": "adjust",
      "confidence": 0.91,
      "summary": "Review source first and render once per audio.",
      "perFileProfiles": [
        {
          "fileName": "actor-a.wav",
          "base": "actor_a",
          "confidence": 0.91,
          "summary": "Pause noise dominates actor A.",
          "recommendedProfile": {
            "name": "Source-First Pause Safe",
            "selectedVariant": "pause-safe",
            "smartMatchMode": "Balanced",
            "leveler": "Gentle",
            "breathControl": "Light",
            "neuralSpeechEnhancement": "off",
            "roomCleanup": false,
            "softenHarshness": true,
            "cinematicColor": true,
            "notes": "No rerender loop."
          },
          "adaptiveDirectives": {
            "warmthDb": 0.4,
            "presenceDb": -0.3,
            "airDb": -0.2,
            "deHarshDb": 0.35,
            "sagRecoveryBoost": 0.25,
            "onsetTameBoost": 0.15,
            "breathTameBoost": 0.2,
            "denoiseBias": 0.1,
            "roomCleanupBias": 0.2,
            "compressionBias": -0.2,
            "finalPolishIntensity": 0.85,
            "targetLoudnessBiasDb": 0.2,
            "levelerBias": 0.1,
            "headGuardBoost": 0.4
          },
          "profileRationale": ["Pause noise dominates the source."],
          "guardrails": ["Render the selected variant once."],
          "nextListeningChecks": ["Check room bed between lines."]
        },
        {
          "fileName": "actor-b.wav",
          "base": "actor_b",
          "confidence": 0.78,
          "summary": "Actor B needs continuity and warmth.",
          "recommendedProfile": {
            "name": "Source-First Continuity Warm",
            "selectedVariant": "continuity-safe",
            "smartMatchMode": "Gentle",
            "leveler": "Firm",
            "breathControl": "Medium",
            "neuralSpeechEnhancement": "off",
            "roomCleanup": true,
            "softenHarshness": false,
            "cinematicColor": true,
            "notes": "Different file, different profile."
          },
          "adaptiveDirectives": {
            "warmthDb": 0.7,
            "presenceDb": 0.1,
            "airDb": 0.3,
            "deHarshDb": 0.1,
            "sagRecoveryBoost": 0.35,
            "onsetTameBoost": 0.05,
            "breathTameBoost": 0.1,
            "denoiseBias": 0,
            "roomCleanupBias": 0.3,
            "compressionBias": 0.12,
            "finalPolishIntensity": 0.6,
            "targetLoudnessBiasDb": -0.1,
            "levelerBias": 0.2,
            "headGuardBoost": 0
          },
          "profileRationale": ["Line continuity dominates."],
          "guardrails": ["Keep warmth subtle."],
          "nextListeningChecks": ["Check mid-sentence body."]
        }
      ],
      "adjustments": [],
      "findings": [],
      "profileRationale": ["Pause noise dominates the source."],
      "guardrails": ["Render the selected variant once."],
      "nextListeningChecks": ["Check room bed between lines."]
    }
  `);

  const plan = buildSourceFirstAudioReviewPlan(review.perFileProfiles[0], {
    ...baseControls,
    smartMatchMode: "Gentle",
    leveler: "Firm",
    breathControl: "Medium",
    neuralSpeechEnhancementEnabled: false,
    roomCleanup: true,
    softenHarshness: false,
    cinematicColor: false,
  });

  const secondPlan = buildSourceFirstAudioReviewPlan(review.perFileProfiles[1], baseControls);

  assert.equal(plan.selectedVariant, "pause-safe");
  assert.equal(plan.fileName, "actor-a.wav");
  assert.equal(plan.base, "actor_a");
  assert.deepEqual(plan.controls, {
    ...baseControls,
    smartMatchMode: "Balanced",
    leveler: "Gentle",
    breathControl: "Light",
    neuralSpeechEnhancementEnabled: false,
    roomCleanup: false,
    softenHarshness: true,
    cinematicColor: true,
  });
  assert.deepEqual(plan.changedKeys, [
    "smartMatchMode",
    "leveler",
    "breathControl",
    "roomCleanup",
    "softenHarshness",
    "cinematicColor",
  ]);
  assert.deepEqual(plan.adaptiveDirectives, {
    warmthDb: 0.4,
    presenceDb: -0.3,
    airDb: -0.2,
    deHarshDb: 0.35,
    sagRecoveryBoost: 0.25,
    onsetTameBoost: 0.15,
    breathTameBoost: 0.2,
    denoiseBias: 0.1,
    roomCleanupBias: 0.2,
    compressionBias: -0.2,
    finalPolishIntensity: 0.85,
    targetLoudnessBiasDb: 0.2,
    levelerBias: 0.1,
    headGuardBoost: 0.4,
  });
  assert.match(plan.summary, /source-first/i);
  assert.equal(secondPlan.selectedVariant, "continuity-safe");
  assert.equal(secondPlan.fileName, "actor-b.wav");
  assert.equal(secondPlan.base, "actor_b");
  assert.equal(secondPlan.adaptiveDirectives.warmthDb, 0.7);
  assert.equal(secondPlan.adaptiveDirectives.finalPolishIntensity, 0.6);
  assert.notDeepEqual(secondPlan.adaptiveDirectives, plan.adaptiveDirectives);
});

test("post-render prompt uses selected candidate evidence and permits one bounded corrective pass", () => {
  const prompt = buildAudioReviewUserPrompt({
    ...reviewPayload,
    reviewStage: "post-render",
    files: [
      {
        ...reviewPayload.files[0],
        selectedCandidate: {
          variant: "cinematic-stable",
          reason: "selected render",
          processingFlow: "app-final-polish",
          qc: {
            ...reviewPayload.files[0].source,
            integratedLufs: -23.8,
            coldOpenDipDb: 3.2,
            coldOpenRiskScore: 0.55,
            bandSpectrumDb: [-30, -24, -22, -22, -23, -22, -20, -24],
          },
          qcDelta: { coldOpenDipDb: 1.8, sibilanceScore: 0.1 },
          alignment: { durationDeltaSec: 0.01, estimatedOffsetSec: 0.02, confidence: 0.7 },
          issueTags: ["cold_open_dip"],
          score: { stability: 0.2, pause: 0.1, compression: 0.12, echo: 0.05, total: 170 },
        },
      },
    ],
  });

  assert.match(prompt, /already-rendered VO batch/i);
  assert.match(prompt, /one bounded corrective pass/i);
  assert.match(prompt, /absolute final corrective settings/i);
  assert.match(prompt, /not deltas/i);
  assert.doesNotMatch(prompt, /corrective deltas/i);
  assert.match(prompt, /selectedCandidate field contains rendered output QC/i);
  assert.match(prompt, /headGuardBoost/i);
});

test("post-render Gemini adaptive directives normalize as absolute final values", () => {
  const directives = normalizeAdaptiveDirectives({
    finalPolishIntensity: 0.25,
    headGuardBoost: 0.2,
    compressionBias: -0.15,
  });

  assert.equal(directives.finalPolishIntensity, 0.25);
  assert.equal(directives.headGuardBoost, 0.2);
  assert.equal(directives.compressionBias, -0.15);
});

test("deterministic corrective directive map produces non-trivial bounded deltas", () => {
  const directives = buildCorrectiveDirectivesForIssueTags([
    "cold_open_dip",
    "harsh_sibilance",
    "too_compressed",
  ]);

  assert.equal(directives.headGuardBoost, 0.5);
  assert.equal(directives.deHarshDb, 0.6);
  assert.equal(directives.compressionBias, -0.35);
  assert.equal(directives.finalPolishIntensity, 0.3);
  assert.equal(directives.levelerBias, -0.25);
  assert.equal(hasNonTrivialAdaptiveDirectives(directives), true);
});

test("deterministic ending corrections reduce final polish instead of increasing it", () => {
  const directives = buildCorrectiveDirectivesForIssueTags(["endings_damaged"]);

  assert.equal(directives.compressionBias, -0.2);
  assert.equal(directives.finalPolishIntensity, 0.25);
});
