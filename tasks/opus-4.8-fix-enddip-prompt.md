# Prompt for Opus 4.8 — Fix the End-of-Sentence Dip Regression + Post-Review Hardening

Copy everything below the line into a fresh Opus 4.8 agent session opened at the repo root
(`e:\Work Files\Audio volume fixer\experimental\vo-leveler-web-experimental`).

---

You are a senior audio-DSP engineer and full-stack developer continuing work on a production VO-leveling web app (browser ffmpeg.wasm, Next.js/TypeScript). A previous session implemented `implementation-plan.md` (repo root): cold-open dip fix, BS.1770 K-weighted planner loudness, head priming, batch loudness alignment, house-tone blend, adaptive de-esser, and a bounded post-render AI corrective loop. An expert code review of that work found **one user-confirmed audio regression and four concrete defects**. Your mission is to fix them, in priority order, without reintroducing the cold-open dip and without regressing any existing gate or test.

Current verification state: `npm run lint` passes, `npm run build` passes, `npm run test:audio-qc` is 122 pass / 0 fail / 1 skipped. Keep it that way after every task.

## Context to load first

1. `implementation-plan.md` and `tasks/opus-4.8-implementation-prompt.md` — what was built and why. `SUMMARY.md` (top dated section) — the change log.
2. `agent.md` and `tasks/lessons.md` — repo operating rules (surgical changes, verify before done, update lessons after corrections).
3. Read fully before editing: `src/lib/gainPlanner.ts` (~990 lines) and the `planGainForInput` region of `src/components/VoLeveler.tsx` (~L1490–1640). Read targeted regions of `src/lib/audioQc.ts` (`computeColdOpenMetrics` ~L391–446), `src/lib/aiAudioReview.ts` (`CORRECTIVE_DIRECTIVE_MAP` ~L656, post-render prompt ~L754–791), and `VoLeveler.tsx` (`alignBatchMixReadyOutputs` ~L4915–5020, corrective-pass block ~L6858–7095, `shouldTryCorrective` ~L6883–6888).

## PRIORITY 1 — Fix the end-of-sentence dip (user-confirmed regression)

**Symptom (reported from real renders):** the old cold-open dip is gone, but now sentences play stable through start and middle, then **spike down at the very edge of the end of the sentence**.

**Root cause — K-weighting domain mismatch.** `planGainForInput` (`VoLeveler.tsx` L1526–1546) now computes the planner envelope `frameDb` from **K-weighted** samples (`applyKWeighting`: +4 dB high-shelf above ~1.68 kHz, HPF ~38 Hz), while everything downstream of that envelope was designed and tuned for the **raw** RMS domain:

- **The body-relative spike guard** (`planGainCurve` step 7, `gainPlanner.ts` ~L757–870; `allowedRmsSpikeDb` at L790, `clearlyHot` at L841). `bodyLevelDb` (K-weighted run mean + gain) and `appliedFrameDb` (K-weighted frame + gain) sit in the K domain, but `framePeakDb` is built from **raw** `input.samples`. K-weighting boosts 2–8 kHz — exactly where sentence-final consonants live ("s", "sh", "ts", "ch"). A trailing sibilant that used to read ~2 dB *under* the run body now reads 1–2 dB *over* it; `clearlyHot` fires at only 1 dB RMS excess; the guard then applies its ±40 ms cosine dip (up to `5.5 + speechSpikeTaming × 6` dB) **at the last frames of the run**. That is the reported artifact, exactly.
- **The speech mask** (`buildSpeechMask(frameDb, noiseFloorDb, …)`, `VoLeveler.tsx` L1591) receives K-weighted frames but `noiseFloorDb` measured by the un-weighted QC analysis — open/close thresholds are mis-calibrated, so run end boundaries drift and the 500 ms release ramp (toward −expanderDepth) can begin while the actor is still finishing the word.
- **Classification crest** (`crestDb = peakDb − meanDb`) mixes raw peak with K-weighted mean (~1–1.5 dB low), skewing transient-breath / sustained-high-crest decisions, and the **post-clamp residual pass** (step 7b) reads applied bodies ~1–1.5 dB hotter so uniform run cuts fire more often.
- **Absolute level shift:** targeting −22 in the K domain ≈ −23…−23.5 raw, so outputs render ~1 dB quieter than pre-change builds.

**Required fix — domain separation (do NOT simply revert K-weighting):**

1. In `planGainForInput`, compute **two** envelopes from the decoded 16 kHz samples: `frameDb` (raw, exactly as before the K change) and `loudnessFrameDb` (K-weighted). Pass the **raw** `frameDb` to `buildSpeechMask` and as the planner's main `frameDb` input. Pass `loudnessFrameDb` as a new optional `GainPlannerInput.loudnessFrameDb` (same length; validate and fall back to `frameDb` when absent/mismatched).
2. In `planGainCurve`, use the **K-weighted** envelope ONLY for loudness targeting math:
   - per-run body `meanDb` used to compute `plannedRunGainDb` (target − mean), the trimmed-mean `targetDb` blend, the early-run cap / cold-open lift anchor comparisons, and the micro-ride `localDb` (so the ride target stays consistent with the gain target).
   - Everything else stays in the **raw** domain: speech-mask-derived run boundaries, `framePeakDb`, `crestDb` and `hotFrameRatio` classification (compute a parallel raw run mean for these), `shortHotPerformance`, the absolute peak ceiling, the body-relative spike guard (`bodyLevelDb`, `appliedFrameDb`, `localContrastDb`, `isHotFrame` — all raw; keep using the raw run mean as the guard's body baseline), the post-clamp residual pass, expander depth, attack/release ramps.
   - Keep the struct/field additions minimal: one extra `rawMeanDb` (or `loudnessMeanDb`, pick one naming direction) per run entry is expected; do not fork the whole pipeline.
3. **Recalibrate the absolute target:** after the split, measure on the existing synthetic test fixtures how far K-domain targeting shifts output level vs the pre-K baseline, and compensate with a documented constant (e.g. `PLANNER_K_TARGET_OFFSET_DB` applied to the −22 target) so output speech level stays within ±0.5 dB of the pre-K-weighting builds. State the measured offset in a code comment and in your summary.
4. Keep `PLANNER_K_WEIGHTING` as the single flag; `false` must restore the exact pre-K behavior (raw everywhere).

**Required regression tests (`src/lib/gainPlanner.test.ts`):**

- **End-edge dip repro:** build a synthetic sentence run whose final ~150 ms is a "sibilant" — same raw frame RMS as the body but +3.5 dB in the `loudnessFrameDb` domain (you can synthesize the two envelopes directly). With samples provided (so `framePeakDb` exists) and default taming, assert the applied gain over the last 200 ms of the run stays within 1 dB of the run body gain (no localized dip). Before your fix, this test must fail; verify that, then fix, then confirm it passes.
- **Loudness-targeting still perceptual:** keep/extend the existing K-weighting equal-loudness test — a 100 Hz-heavy voice and a 3 kHz-forward voice with equal K-weighted loudness get planned gains within 0.5 dB.
- **Cold-open regression guard:** all existing cold-open tests (quiet opener lift, hot opener cap, speech at frame 0) must still pass unchanged. The reported cold-open fix works in production — do not alter its behavior.
- **Mask stability:** a fixture proving speech-run boundaries with the raw envelope are identical to pre-K-change boundaries (mask receives raw frames again).

**Also verify:** the rendered-output QC "Endings" check (`endFadeRiskScore` delta gate in `reviewLearning.ts` `buildCandidateAssessment`) would catch this artifact class. If it demonstrably cannot see an 80 ms edge dip, add a small `endEdgeDipDb` metric to `audioQc.ts` mirroring `computeColdOpenMetrics` (per-run tail body vs last-150 ms level, worst case across runs), wire it into snapshots/assessment as WARN-only, with unit tests. Keep this addition minimal; skip it if `endFadeRiskScore` provably covers the case (show evidence either way in your summary).

## PRIORITY 2 — Batch loudness alignment will OOM on real batches (latent blocker)

`alignBatchMixReadyOutputs` (`VoLeveler.tsx` ~L4915) writes **every** clean mix-ready output into the in-memory WASM FS during the measure loop (`batch_align_<i>_in.wav`) and deletes them only in the final `finally`. The team's normal batch is 20–30 minute files at 48 kHz float32 mono (~330–345 MB each); ten files ≈ 3.4 GB of MEMFS → guaranteed OOM, plus the apply loop adds an output copy per file.

Restructure to **one file at a time**: for each target — write input → measure integrated loudness → delete input — collecting measurements; compute the plan; then for each file needing alignment — write input → render `volume=<offset>dB,alimiter` → read output bytes → replace the output entry blob → delete both temp files — before touching the next file. Track cumulative audio seconds through the pass and recycle the worker with the existing `refreshFfmpeg` when it exceeds `BATCH_AUDIO_RECYCLE_SECONDS` (the FS is empty between files by construction, so recycling is safe). Keep the existing fail-closed behavior (any error → log `[BatchAlign]` and keep the un-aligned outputs). Add/adjust `batchLoudnessAlign.test.ts` only if the pure-planning function changes (it should not need to).

## PRIORITY 3 — Corrective pass triggers far too eagerly (batch-time cost)

`shouldTryCorrective` (`VoLeveler.tsx` ~L6883) fires on `failCount > 0 || warnCount > 0 || gateReasons.length > 0`. Auto-review WARNs are common on ordinary healthy files, so most files get a full second render + polish + QC — nearly doubling batch time for 20–30 min episodes.

Tighten to: any FAIL, **or** ≥ 2 WARNs, **or** a single WARN whose issue tag is in a high-value set (`cold_open_dip`, `harsh_sibilance`, `too_compressed`, `level_uneven`), **or** hard-gate reasons present. Add a per-batch corrective budget: at most `CORRECTIVE_MAX_FILES_PER_BATCH = max(2, ceil(jobs.length × 0.4))` corrective renders, consumed worst-first is not required (sequential processing) but log when the budget blocks a pass. Named constants + log lines for every skip reason. Update no tests unless one asserts the old trigger.

## PRIORITY 4 — Cold-open metric bias inflates `coldOpenDipDb` (false WARNs feed Priority 3)

`computeColdOpenMetrics` (`audioQc.ts` ~L391) compares an **untrimmed** head window (includes run onsets and inter-syllable low frames of the first 2.5 s) against **edge-trimmed** later run bodies — a structurally positive bias of ~1–2 dB on clean files. Make the measurement symmetric: compute the head as the power-mean of the **edge-trimmed bodies of the first `COLD_OPEN_RUN_COUNT` runs** (same 12–14 % trim rule as the body side; keep a minimum-frames guard and return zeros when insufficient). Re-tune `COLD_OPEN_WARN_DB` / `COLD_OPEN_FLAG_DB` / risk floor only if the existing unit-test fixtures show the scale changed; update `audioQc.test.ts` fixtures/expectations accordingly and keep the `reviewLearning.ts` WARN/FAIL deltas coherent with the new scale.

## PRIORITY 5 — Post-render directive stacking ambiguity

In the corrective block (`VoLeveler.tsx` ~L6900–6930), Gemini's post-render `adaptiveDirectives` are converted via `adaptiveDirectiveDeltaFromDefault(...)` (delta from DEFAULT) and merged **onto the active base directives**. If the source-first pass already set e.g. `deHarshDb 0.6` and post-render Gemini answers 0.6 meaning "use 0.6", the merge yields 1.2 — silent over-processing. Fix by making intent explicit: in the post-render user prompt (`aiAudioReview.ts` `buildAudioReviewUserPrompt`, post-render branch), state that `adaptiveDirectives` must be **absolute final values** (the payload already contains the active profile snapshot for reference), and in the corrective block replace the delta-from-default merge with `normalizeAdaptiveDirectives(fileReview.adaptiveDirectives)` used directly as the corrective directives. Keep the deterministic `CORRECTIVE_DIRECTIVE_MAP` path as-is (those are genuine deltas on top of the active base). Add/extend an `aiAudioReview.test.ts` case pinning the semantics of both paths.

## Hard boundaries (unchanged from the original mission)

- Do not touch the Audio Splitter (`AudioTrackSplitter.tsx`, `src/app/api/audio-splitter/**`, `audioSplitterJobs.ts`, `audioSplitterService.ts`, python splitter scripts).
- Neural speech enhancement stays disabled everywhere.
- Never regress: duration delta ≤ 0.05 s, alignment offset ≤ 0.08 s, true-peak/limiter gates (`alimiter=limit=-2dB`), ending-damage and source-regression gates, long-file memory budgets (≥ 600 s single variant, ≥ 1200 s no full-file JS decode, ≥ 4800 s long-form safe mode), worker recycling and watchdog semantics.
- Every new stage fails closed to current behavior; no new throw may escape `processFiles`' per-file try/catch in a way that changes retry semantics.
- Surgical diffs only: no drive-by refactors, no renames, no formatting churn; match existing style (named constants + explanatory DSP comments). No new dependencies.

## Execution protocol

1. Maintain `tasks/todo.md` with checkable items per priority; mark as you go.
2. Work in priority order; after each priority run `npm run lint && npm run build && npm run test:audio-qc` — all green before moving on.
3. For Priority 1, follow test-first: write the end-edge dip repro, watch it fail on current code, then fix.
4. Optional external gates if available: the Fusion sidecar can be copied verbatim from `E:\Work Files\Audio volume fixer\vo-leveler-web` (`scripts/fusionReview.mjs` + `fusion:review` npm script). Run `--preset implementation --git-diff` after Priority 1–2 and `--preset patch --git-diff` at the end. Pipe stdin closed (PowerShell: `$null | npm run fusion:review -- ...`) or it hangs. If `OPENROUTER_API_KEY` is missing or out of credits (402), note it and continue — local tests remain the mandatory gate.
5. Update `SUMMARY.md` with a dated section and `tasks/lessons.md` with the K-weighting domain-mismatch lesson (pattern: when changing the measurement domain of a shared envelope, audit every consumer's calibration — thresholds tuned in one domain silently break in another).

## Definition of done

- All five priorities implemented; the end-edge dip repro test fails on pre-fix code and passes after; all cold-open tests still pass; lint/build/tests green (report counts).
- Final summary must include: per-priority change list with file paths; the measured K-domain target offset and how you compensated; evidence the batch-align pass now holds at most one file in MEMFS at a time; the new corrective trigger rules and budget; before/after semantics of post-render directives; and a listening checklist for the team — specifically: sentence endings with sibilants ("s/sh/ch" finals) must hold level to the last phoneme, file heads must still open at body level, and batch outputs must still land within ±0.5 LU of each other.
