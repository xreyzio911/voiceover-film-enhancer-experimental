# Prompt for Opus 4.8 — Execute the Cinematic Voice Upgrade

Copy everything below the line into a fresh Opus 4.8 agent session opened at the repo root
(`e:\Work Files\Audio volume fixer\experimental\vo-leveler-web-experimental`).

---

You are a senior audio-DSP engineer and full-stack developer executing a pre-approved implementation plan for a production VO-leveling web app. You have deep experience with broadcast loudness (BS.1770 / EBU R128), dialogue mixing for film, ffmpeg filter graphs, and TypeScript/React. Work autonomously, verify everything, and do not ask for hand-holding.

## Mission

Execute `implementation-plan.md` (repo root) completely, in phase order (0 → 1 → 2 → 3 → 4). The plan is the contract: it fixes the cold-open level dip ("first few words dip down, then spike up, then stabilize"), adds perceptual loudness + batch voice unification, and turns the AI review into a bounded closed loop with real authority over output quality. Read the whole plan before writing any code.

## Context you must load first (in this order)

1. `implementation-plan.md` — the contract. Every task, constant name, threshold, and exit criterion is specified there.
2. `agent.md` — operating rules for this repo (plan mode, surgical changes, verification-before-done, update `tasks/lessons.md` after corrections).
3. `tasks/lessons.md` — past mistakes; do not repeat them.
4. The load-bearing source files, end to end, before editing them:
   - `src/lib/gainPlanner.ts` (~940 lines) — speech-aware gain planner. Key anchors: run classification L280–296, per-run gains L380–409, `earlyRunCap` L444–476, expander-floor init L495–496, attack/release ramps L502–565, peak/spike guards L579–796, `applyGainCurveToSamples` L836–867.
   - `src/lib/audioQc.ts` — QC metrics. `buildSpeechMask` L164–242, `analyzeFrameAudio` ~L455, `analyzeFloatSamples` ~L874, flags L367–452.
   - `src/lib/aiAudioReview.ts` — Gemini review plumbing. Directive schema L273–301, system/user prompts ~L640–691, control patch L739–771, source-first plan L773–796, normalization/clamps L465–510.
   - `src/lib/reviewLearning.ts` — learned ranker + deterministic auto-review. `scoreCandidateWithLearnedWeights` L481–575, `buildCandidateAssessment` L708–899, `autoReviewBundle` L934–1014.
   - `src/lib/spectrum.ts` — 8-band log spectrum (bands at L13), `computeSibilanceScore` L121, `computeToneMatchDeltaDb` L135.
   - `src/components/VoLeveler.tsx` (~7600 lines, the orchestrator) — read at least: constants L74–210, `buildAudioReviewFileInput` L1281, `runAiAudioReview` L1321, `planGainForInput` L1416–1557, planner apply + chunking L1560–1740, `buildAdaptiveProfile` L2603–2963, tamer builders L2966–3030, `buildMixFilter` L3135–3796, `buildFinalPolishProfile`/`runFinalAppPolishPass` L3849–3955, `analyzeIntegratedLoudness` ~L3957, `runMixReady` ~L4039, `runLoudnorm` ~L4682, `buildMixCandidateVariants` L5149, `processFiles` L5716 (source review call L5814, candidate loop L6048–6250, selection L6211, review bundles L6329–6412, polish call L6413, output/export from L6421).
5. `package.json` — scripts: `npm run lint`, `npm run build`, `npm run test:audio-qc` (Node test runner with `--experimental-strip-types`; new test files must be appended to that script).

## Architecture facts you must respect (do not rediscover them wrong)

- All audio processing is client-side ffmpeg.wasm (single-threaded, WASM heap is the scarce resource). There is **no** OfflineAudioContext path; `webAudioRender.ts` is only WAV encode/decode.
- The gain planner is the level authority (JS gain curve applied to samples at 48 kHz); `dynaudnorm` is only a gated safety pass when the planner is active (`gainPlannerActive`, VoLeveler.tsx L3283–3311). Do not let any new stage re-introduce blind auto-leveling.
- Renders are per-file sequential with worker recycling (`BATCH_AUDIO_RECYCLE_SECONDS = 2400`, `PER_FILE_MAX_RETRIES = 2`, watchdog `max(90, dur×4)+90` s). Long files: ≥600 s single variant, ≥1200 s no full-file JS decode for QC, ≥4800 s long-form safe mode (900 s chunks). Your additions must respect all of these budgets.
- Hard quality gates that must never regress: duration delta ≤ 0.05 s, alignment offset ≤ 0.08 s, true peak ≤ −1.5 dB (ranker gate) with final `alimiter=limit=-2dB`, ending-damage limits, source-regression limits (see `defaultReviewWeights.ts`).
- AI review today is source-first only: one pre-render Gemini call (`gemini-3.1-flash-lite` via `/api/audio-review`, 4 files/request, 18 requests/10 min rate limit), returning per-file control patches + small clamped `adaptiveDirectives`. Verdict/findings are currently display-only. The plan changes this deliberately and boundedly — implement exactly the loop it specifies (one corrective pass, ranker-gated adoption, capped request budget).
- Neural speech enhancement is intentionally disabled everywhere. Keep it that way.
- **Do not touch the Audio Splitter** (`AudioTrackSplitter.tsx`, `src/app/api/audio-splitter/**`, `audioSplitterJobs.ts`, `audioSplitterService.ts`, python splitter scripts).

## Execution protocol

1. Maintain `tasks/todo.md` with checkable items per phase; mark them as you go (repo convention).
2. Implement **phase by phase**. After each phase: run `npm run lint && npm run build && npm run test:audio-qc`; all green before moving on. Commit-sized, reviewable diffs per phase (do not commit unless asked; keep the working tree clean and organized).
3. Write tests **with** (or before) each behavior change, not after the whole plan:
   - Phase 0: `coldOpenDipDb` synthetic-envelope tests in `audioQc.test.ts`; WARN/FAIL threshold tests in `reviewLearning.test.ts`.
   - Phase 1: quiet-opener lift, hot-opener cap regression, speech-at-frame-0, lift bounds — in `gainPlanner.test.ts`.
   - Phase 2: K-weighting equal-loudness test; `batchLoudnessAlign.test.ts` (median anchor, ±2 dB clamp, 0.5 LU trigger, single-file no-op); house-blend tone tests; `resolveDeEsserBands` tests in a spectrum test (create `spectrum.test.ts` if none exists and register it in `test:audio-qc`).
   - Phase 3: directive schema/clamp tests, corrective-mapping-table tests, post-render plan-building tests in `aiAudioReview.test.ts`.
4. DSP correctness rules:
   - Every new gain move must be expressed in dB with named constants, clamped, and logged (`appendLog`) with enough detail to audit from the in-app log panel.
   - The head-priming render (Phase 1.2) must preserve output duration exactly: verify sample counts; on mismatch or any render error, automatically fall back to the un-primed render and log `[HeadPrime] fallback`. Never fail a file because of priming.
   - K-weighting coefficients must be derived for the actual sample rate (16 kHz analysis) via bilinear redesign of the ITU-R BS.1770 pre-filter stages — do not paste 48 kHz textbook coefficients. Include a comment with the derivation source.
   - The corrective re-render may be adopted **only** when `scoreCandidateWithLearnedWeights` prefers it by the plan's margin AND all hard gates pass; otherwise keep the original and log why. One corrective pass max, never for files ≥ 1200 s.
5. Failure handling: any new stage fails closed to current behavior (log + continue). No new stage may throw out of `processFiles`' per-file try/catch in a way that changes retry semantics.
6. If you discover the plan conflicts with reality (line numbers drifted, a helper already exists, a constraint was missed), prefer reality: implement the plan's **intent**, note the deviation explicitly in your final summary, and keep the plan's bounds (clamps, caps, gates) intact. Update `implementation-plan.md` checkboxes/notes if you make a justified deviation.
7. Keep changes surgical: no drive-by refactors of `VoLeveler.tsx`, no formatting churn, no renaming existing symbols, match existing code style (verbose named constants + explanatory comments for DSP decisions).

## Definition of done

- All plan phases implemented; every exit criterion in `implementation-plan.md` demonstrably met.
- `npm run lint`, `npm run build`, `npm run test:audio-qc` all pass; list the test counts in your summary.
- New QC metric (`coldOpenDipDb`) visible in logs for source and candidates; corrective-pass logs demonstrate trigger, adoption, and rejection paths (unit-level demonstration acceptable where a real Gemini key is unavailable).
- A final summary containing: per-phase change list with file paths, new constants + values table, verification evidence (test output, lint/build status, fusion gate verdicts), known limitations, and suggested listening checks for the team (which files to A/B and what to listen for at file heads, sibilants, pauses, and batch-to-batch loudness).
- `SUMMARY.md` updated with a dated section describing the upgrade; `tasks/lessons.md` updated if anything required a correction mid-way.

## Non-goals (hard boundaries)

- No Audio Splitter changes. No neural-enhancement enablement. No new runtime dependencies. No UI redesign beyond queue-stage/log text. No changes to export formats or loudness presets. No multi-iteration AI loops beyond the single corrective pass. No reranking beyond the 2-candidate original-vs-corrective comparison.

Begin by reading `implementation-plan.md` end to end, then produce your `tasks/todo.md` breakdown, then start Phase 0.
