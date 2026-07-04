# VO Leveler — Cinematic Voice Upgrade Implementation Plan

Date: 2026-07-02
Repo: `vo-leveler-web-experimental`
Goal: unify batch VO (20–30 min files, mixed mic quality per actor) into a consistent, crystal-clear cinematic voice; fix the cold-open level dip; make the AI review a real quality-control loop instead of a cosmetic pre-render nudge — **without degrading current output quality in any way**.

---

## 0. Ground rules (read first)

1. **Do not touch the Audio Splitter.** No changes to `src/components/AudioTrackSplitter.tsx`, `src/app/api/audio-splitter/**`, `src/lib/audioSplitterJobs.ts`, `src/lib/audioSplitterService.ts`, `scripts/audio_separator_worker.py`, `scripts/setup-audio-splitter.ps1`.
2. **Neural speech enhancement stays disabled.** Do not wire `requestNeuralRepair` into the render path. Keep `NEURAL_SPEECH_ENHANCEMENT_ENABLED_BY_DEFAULT = false` and the "always off" rules in `aiAudioReview.ts`.
3. **Never regress the existing hard gates**: output duration delta ≤ 0.05 s, offset ≤ 0.08 s, true peak ≤ −1.5 dB (learned-gate) / limiter at −2 dB, no ending damage. Every new stage must fail closed to current behavior (log + fallback, never crash the batch).
4. **All tuning constants introduced by this plan live in named constants** (grouped near the top of the owning module) so any single feature can be reverted by flipping one constant.
5. After each phase: `npm run lint`, `npm run build`, `npm run test:audio-qc` must pass.
6. Follow `agent.md` (surgical changes, verify before done, update `tasks/lessons.md` after corrections).

Current signal chain (for orientation):

```
WAV → analysis (loudnorm stats + 16 kHz envelope QC + log-band spectrum)
    → buildAdaptiveProfile (per file, batch reference + AI directives)
    → gain planner (JS, per-10ms-frame gain curve, target −22 dBFS RMS)
    → ffmpeg mix chain (HPF → EQ → NR → dereverb → onset/click/breath tamers
      → dynaudnorm safety → gate/compand → presence/air EQ → tone match
      → cinematic color → de-esser → glue compressor → alimiter −2 dB)
    → final app polish pass (runMixReady again, scaled-down profile)
    → optional loudnorm (ATSC/EBU) → WAV export (48 kHz mono pcm_f32le)
```

---

## Phase 0 — Measurement first: cold-open QC metric + baseline capture

**Why:** we fix what we can measure. The reported artifact ("first few words dip down, then spike up, then stabilize") must become a numeric QC signal before we change the planner, so we can prove the fix and gate regressions forever.

### 0.1 Add `coldOpenDipDb` + `coldOpenRiskScore` to `src/lib/audioQc.ts`

- In `analyzeFrameAudio` (audioQc.ts, ~L455) / `analyzeFloatSamples` (~L874), after speech runs are collected:
  - `headSpeechDb` = power-mean frame dB of speech frames inside the first `COLD_OPEN_HEAD_MS = 2500` ms of the **first** speech run onward (first 2–3 runs, capped at 2.5 s of speech frames).
  - `bodySpeechDb` = median per-run body RMS of all body runs after the head window (reuse the run RMS logic that already exists for `sentenceJumpScore`).
  - `coldOpenDipDb = bodySpeechDb − headSpeechDb` (positive = head is quieter than body).
  - `coldOpenRiskScore = clamp((coldOpenDipDb − 1.0) / 4.0, 0, 1)` (0 at ≤1 dB, 1.0 at ≥5 dB).
- Add both fields to `AudioQcMetrics`, to `toReviewMetricSnapshot` in `VoLeveler.tsx`, and to `AudioReviewMetricSnapshot` in `src/lib/aiAudioReview.ts` so Gemini sees them.
- Add advisory flag in `buildFlagsAndRecommendations` (audioQc.ts ~L367): warn at `coldOpenDipDb ≥ 1.5`, flag `cold_open_dip` at `≥ 2.5`.
- Add a check in `buildCandidateAssessment` + `autoReviewBundle` (`src/lib/reviewLearning.ts` ~L708/~L934): WARN when rendered `coldOpenDipDb ≥ 1.5` **and** worse than source by ≥ 0.75 dB; FAIL when rendered `≥ 3.0` and worse than source by ≥ 1.5 dB. (Compare against source so naturally quiet openings by performance are not punished.)

### 0.2 Tests

- `src/lib/audioQc.test.ts`: synthetic envelope where head speech sits 4 dB under body → expect `coldOpenDipDb ≈ 4`, risk ≈ 0.75; flat file → ≈ 0.
- Extend `src/lib/reviewLearning.test.ts` with the new WARN/FAIL thresholds.

**Exit criteria:** tests pass; running a real batch logs the new metric per file (source and candidate QC lines in the existing `[CandidateQC]` log).

---

## Phase 1 — Fix the cold-open dip at its sources

Three verified mechanisms in the current code produce exactly "first words down → up → stable". Fix all three.

### 1.1 Gain planner: opener classification + cold-open lift (`src/lib/gainPlanner.ts`)

**Root cause A — opener misclassification.** The first dialogue line is often short and hot-crested ("Hey!", "Well…"). Runs < 400 ms with crest ≥ 15 dB, or < 650 ms isolated (`shortHotPerformance`, L280–296), are classed `transient-breath`: target = `targetDb − 3.2`, boost clamp **+4 dB max** (L379–385). `edge-fragment` (< 100 ms) is clamped **±4 dB**. On a quiet mic needing +8…+14 dB, the opener lands 4–10 dB under target while the next full sentence gets the full boost — the exact reported artifact.

Fixes:
- Add run metadata `isColdOpen`: any run whose `startFrame * frameMs ≤ COLD_OPEN_WINDOW_MS = 2500` **or** which is among the first `COLD_OPEN_RUN_COUNT = 3` detected runs.
- For cold-open runs, require stronger evidence before downgrading class: only classify `transient-breath` when samples exist AND `crestDb ≥ 15` AND `peakDb ≥ targetDb + 8`. Otherwise treat as `body-speech` (or `edge-fragment` only when < 100 ms).
- For cold-open `transient-breath` / `edge-fragment` runs, widen the **positive** clamp to `min(maxGainDb, targetClassGain)` so quiet openers can be lifted to their class target. Keep attenuation clamps unchanged (hot openers must still come down — do not break `earlyRunCapCount` behavior).

**Root cause B — one-sided opener guard.** `earlyRunCap` (L444–476) caps early runs that are **hotter** than the later anchor, but nothing lifts early runs that are **quieter**. Add the symmetric guard:
- After `earlyRunCap`, compute the same `laterAnchorDb` (median applied body of later body runs, ≥ 5 body runs required).
- For the first 3 body runs: if `appliedBodyDb < laterAnchorDb − COLD_OPEN_LIFT_TOLERANCE_DB (1.5)`, lift `plannedRunGainDb` by `min(deficit − tolerance, COLD_OPEN_LIFT_MAX_DB = 5)`; respect `maxGainDb` and let the existing peak guard (step 7) re-clamp peaks afterwards (it runs later, so ordering is already correct).
- Emit diagnostics `coldOpenLiftCount` / `coldOpenLiftMaxDb` in `GainPlannerOutput` (mirror the `earlyRunCap*` fields) and log them from `planGainForInput` in `VoLeveler.tsx` (~L1537–1553).

**Root cause C — attack ramp swallows the first syllable.** The whole curve initializes at the expander floor (−12…−30 dB, L495–496) and the 80 ms cos² attack lives in the silence before each run (L540–552). At file start, mask open-confirm (35 ms, `audioQc.ts` L184–215) plus smoothing means the true first syllable frames can sit *before* the detected `startFrame` — at expander gain — and then ride the ramp up.

Fixes (first run only):
- Soft-onset capture: walk back from run 0's `startFrame` up to `COLD_OPEN_ONSET_BACKTRACK_MS = 60` ms while `frameDb[f] ≥ openThresholdDb − 6`; extend the run start to include those frames at body gain.
- Early-completing attack: for run 0, make the attack ramp **finish** `COLD_OPEN_ATTACK_LEAD_MS = 40` ms before the (extended) run start instead of exactly at it, so gain is already at body level when the first phoneme lands. If the file starts in speech (`attackLen == 0`), force full body gain from frame 0.

### 1.2 Prime stateful ffmpeg filters at file start (`src/components/VoLeveler.tsx`)

**Root cause D — filter warm-up.** `dynaudnorm` (safety pass `f=161:g=3:m=5` at L3302–3311, or full presets `f=181–281` at L3312–3452), `acompressor`, `agate`/`compand`, and `afftdn` all start with empty windows/envelopes at t=0. dynaudnorm's centered gaussian window means the first ~0.4–0.8 s is gained conservatively, then rises — and the **final polish pass re-runs the chain, warming up a second time** (`runFinalAppPolishPass` L3886–3955). This compounds with 1.1 on messy files where the safety pass engages (`dynaSafetyBlend ≥ 0.3`).

Fix — mirrored pre-roll priming (classic filter-priming technique, exact-duration safe):
- New helper `renderWithHeadPriming` used only for the **first** rendered range of a file (single-pass render, segment 0 of segmented paths, and the final polish pass — polish is single-pass by definition):
  1. Build `in_primed.wav`: decode the first `HEAD_PRIME_SECONDS = 1.0` s, `areverse` it, and concat before the input (pcm f32 concat of two WAVs; reuse the existing `runCrossfadeConcat`-adjacent concat plumbing but with plain `concat` demuxer, no crossfade — the seam is at the mirror point and never audible because it is trimmed away).
  2. Run the existing filter chain on `in_primed.wav`, appending `,atrim=start=1.0,asetpts=PTS-STARTPTS` (exact `HEAD_PRIME_SECONDS`) as the **last** chain elements after the limiter.
  3. Verify output sample count matches the un-primed expectation (duration hard gate 0.05 s); on any failure, log `[HeadPrime] fallback` and re-render without priming (fail closed).
- Gate with `HEAD_PRIME_ENABLED = true` constant. Skip when `minimalStabilityChain` (recovery paths must stay maximally simple) and in long-form safe mode (≥ 4800 s files).

### 1.3 Tests + proof

- `src/lib/gainPlanner.test.ts` new cases:
  - Quiet opener: first run body −34 dB, later runs −26 dB, target −22 → first run applied body within 1.5 dB of later anchor (previously ~6+ dB under when misclassified).
  - Short hot opener on a **loud** file still gets capped (regression test for `earlyRunCap`).
  - Speech starting at frame 0 → gain at frame 0 equals body gain (no ramp-from-floor).
  - Cold-open lift never exceeds +5 dB and never fires with < 5 body runs.
- Manual verification (document in PR/summary): process one previously-affected file; compare waveform head + rendered `coldOpenDipDb` before/after (expect ≥ 2 dB improvement on affected files, ≤ 0.3 dB change on clean files).

**Exit criteria:** unit tests green; `coldOpenDipDb` (Phase 0) on rendered output ≤ 1.5 dB on the affected sample files; no duration/peak gate regressions.

---

## Phase 2 — Perceptual loudness + batch unification ("one crystal-clear voice")

### 2.1 K-weighted planner loudness (BS.1770) — `src/lib/gainPlanner.ts` + `VoLeveler.tsx`

Today the planner targets −22 dBFS **plain RMS** per 10 ms frame (`planGainForInput` L1519–1534). Plain RMS over-weights low-frequency energy, so boomy mics read hotter than they sound and bright mics read quieter — a direct cause of "same numbers, different perceived loudness" across actors.

- In `planGainForInput` (VoLeveler.tsx L1460–1475), before computing `frameDb`, run the decoded 16 kHz samples through a K-weighting pre-filter (two biquads: stage-1 high-shelf ≈ +4 dB above ~1.68 kHz, stage-2 high-pass ≈ 38 Hz), with coefficients computed for 16 kHz via the standard bilinear redesign (implement `applyKWeighting(samples, sampleRate)` in `gainPlanner.ts`; the ITU coefficient derivation for arbitrary sample rates is well documented — do not hardcode 48 kHz coefficients).
- Keep the target at −22: K-weighted frame energy at −22 maps ≈ −23…−24 LUFS speech-gated, matching the existing house target comment (L34).
- Feature flag `PLANNER_K_WEIGHTING = true`. Validate on synthetic fixtures: a 100 Hz-heavy voice and a 3 kHz-forward voice with equal K-weighted loudness must receive gains within 0.5 dB of each other (new unit test), while under plain RMS they diverge by several dB.

### 2.2 Batch output loudness alignment — new `src/lib/batchLoudnessAlign.ts`

Per-file leveling already converges near the house target, but residual ±1–2 LU spread across actors remains audible in a batch. Add a final cross-file alignment pass (only when `loudnessTarget` is "Mix-ready only" — the ATSC/EBU loudnorm paths already normalize):

- After all jobs render (end of the render `while` loop in `processFiles`, before ZIP/output finalization), measure integrated loudness of each mix-ready output with the existing `analyzeIntegratedLoudness` (VoLeveler.tsx ~L3957).
- Compute the batch anchor: median integrated I across files (skip failed files).
- For each file deviating > `BATCH_ALIGN_TRIGGER_LU = 0.5` from the anchor: apply a static `volume=<offset>dB` (clamped to ± `BATCH_ALIGN_MAX_DB = 2.0`) followed by `alimiter=limit=-2dB:level=disabled`, re-export. Static gain + limiter cannot pump or change dynamics — this is the "do no harm" way to unify.
- Pure-function core (`planBatchLoudnessAlignment(measurements) → offsets`) lives in `src/lib/batchLoudnessAlign.ts` with unit tests (`batchLoudnessAlign.test.ts`, added to `test:audio-qc` script in `package.json`): median anchoring, clamping, single-file batch no-op, missing-measurement skip.
- Log `[BatchAlign] <file>: −1.2 dB toward batch anchor −22.8 LUFS` per file; skip and log when all files already within trigger.

### 2.3 House-tone hardening — `src/lib/spectrum.ts` + `buildAdaptiveProfile`

The batch reference is a pure median of the batch (`buildBatchReference`). If the whole batch skews (e.g. all boomy), the target skews with it. Blend the batch median with a fixed cinematic VO reference curve:

- Add `CINEMATIC_VO_REFERENCE_DB` in `spectrum.ts`: an 8-value target for the existing `SPECTRUM_BANDS_HZ` (60, 120, 250, 500, 1k, 2k, 4k, 8k) encoding the proven VO shape — controlled lows, full 120–250 body, present 2–4 kHz, aired but not brittle 8 kHz. Derive initial values from the median band spectrum of the batch's *cleanest* files (the existing reference already selects clean files) shifted by the cinematic color intents (+0.8 @ 180, +0.6 @ 4.5k, −0.5 @ 10k already in the chain — keep consistent, do not double-apply).
- `computeToneMatchDeltaDb` (spectrum.ts L135): accept a `houseBlend` parameter; effective reference = `0.65 × batchMedian + 0.35 × house` (constant `HOUSE_TONE_BLEND = 0.35`). Raise `maxDb` from 2.5 → 3.0 **only** for the two largest-magnitude bands, keep 2.5 for the rest (identity-preserving).
- Unit tests: all-boomy batch pulls toward house curve; single-file batch works; clamps hold.

### 2.4 Adaptive de-esser placement — `buildMixFilter`

The de-esser notches are fixed at 6500/9000 Hz. Sibilance center varies by actor (5.5–9 kHz). Use the measured band spectrum:
- From `bandSpectrumDb`, compare 4 kHz vs 8 kHz band energy (relative to 1–2 kHz body, same math as `computeSibilanceScore`, spectrum.ts L121–128). Map the ratio to notch centers: 4k-dominant → 5800/8200 Hz; 8k-dominant → 7200/9800 Hz; balanced → keep 6500/9000.
- Depths and the `sibilanceScore ≥ 0.4` gate stay exactly as today. Constant-table `DE_ESSER_PLACEMENTS` + unit-testable helper `resolveDeEsserBands(bandSpectrumDb)` (pure function in `spectrum.ts`).

**Exit criteria:** new tests green; batch of 3+ heterogeneous files exports within ±0.5 LU of anchor; tone-match deltas logged with house blend; no QC hard-gate regressions.

---

## Phase 3 — Make the AI review a closed loop with real authority

Today: one **pre-render** Gemini call (`processFiles` L5814–5826) → per-file control patch + tiny directives (schema clamps ±0.45, further scaled, e.g. `compressionBias × 0.16` in `buildAdaptiveProfile` L2643–2651) → single render → verdict/findings displayed only. No post-render review, no retry, reranking disabled (L6211–6218).

### 3.1 Post-render review with real rendered evidence

- After candidate selection + final polish (after `runFinalAppPolishPass` call at ~L6413), build the `selectedCandidate` snapshot that the schema already supports but currently always receives `null` (`buildAudioReviewFileInput` L1281–1287, `AudioReviewSelectedCandidate` type): rendered QC snapshot, QC delta vs source (`buildReviewMetricDelta` output already exists at L6181), `coldOpenDipDb`, alignment metrics, selected variant + render path.
- Collect these per file during the batch; after the render loop, run **one** post-render Gemini review for the files that need it (batched 4 files/request as today, `AUDIO_REVIEW_FILES_PER_GEMINI_REQUEST`):
  - Trigger per file: deterministic auto-review says WARN/FAIL (`autoReviewBundle` checks, reviewLearning.ts L708–1014 — run it inline on the in-memory snapshots; it is pure), **or** any QC hard-flag (cold-open, sibilance, echo, compression thresholds above).
  - Respect the route rate limit (18 req / 10 min): cap post-render review at `POST_RENDER_REVIEW_MAX_REQUESTS = 6` per batch; prioritize worst files by auto-review severity; log skips.
- New prompt variant in `aiAudioReview.ts` (`source: "post-render"`): "you are hearing the rendered result via metrics; diagnose remaining issues and output complete bounded final `adaptiveDirectives` values for a single corrective pass" — reuse the existing response schema (verdict, perFileProfiles, directives). Explicitly instruct decisive values when evidence is strong; remove the blanket "prefer ±0.2–0.5" cap sentence from the corrective prompt. Deterministic issue-tag fallbacks remain delta-on-base; AI-returned post-render directives are absolute final settings.

### 3.2 One bounded corrective re-render (this is where impact comes from)

- New constant `MAX_CORRECTIVE_PASSES = 1`.
- A file qualifies for a corrective pass when post-render review verdict is `adjust`/`risky` **with** non-trivial directives, or (Gemini unavailable) when deterministic auto-review FAILs with a mapped issue tag.
- Deterministic fallback mapping table `CORRECTIVE_DIRECTIVE_MAP` in `aiAudioReview.ts` — issue tag → bounded directive delta, e.g.:
  - `cold_open_dip` → `headGuardBoost +0.5` (new directive, below)
  - `sibilant_harsh` → `deHarshDb +0.6`
  - `overcompressed` → `compressionBias −0.35`, `finalPolishIntensity −0.2`
  - `pause_noise` → `denoiseBias +0.3`
  - `echo_roomy` → `roomCleanupBias +0.35`
  - `unstable_levels` → `levelerBias +0.35` (new directive, below)
- Re-render the file once with merged directives (Gemini's if available, else mapped), same variant unless the review recommends a different `selectedVariant`.
- **Winner selection between original and corrective render** re-uses the existing learned ranker + hard gates: `scoreCandidateWithLearnedWeights` (reviewLearning.ts L481–575). The corrective render must win by a margin (`CORRECTIVE_WIN_MARGIN = 25` ranking points) *and* pass hard gates, else keep the original (do-no-harm guarantee). This also un-pauses the reranker in a bounded, memory-safe way: exactly 2 candidates, only for flagged files.
- Memory guards: skip corrective pass for files ≥ `LONG_CANDIDATE_QC_SAFE_SECONDS` (1200 s) — QC of a second candidate is already skipped there today; recycle the worker before the corrective render for files ≥ 600 s (same policy as the candidate loop L6054–6057).
- Log everything: `[Corrective] <file>: triggered by <tags>, directives <json>, result kept/discarded (Δscore …)`. Store both renders in the QC-Lab review bundle (winner + challenger slots already exist, L6337–6399) so every corrective pass becomes future training data for the learned ranker.

### 3.3 Expand directive authority (bounded but meaningful)

In `aiAudioReview.ts` schema (L273–301) + `normalizeAdaptiveDirectives` + `buildAdaptiveProfile` merges:

- Widen: `warmthDb`/`presenceDb` ±1.2 → **±1.8**; `airDb` ±0.9 → **±1.2**; `deHarshDb` 0–1.2 → **0–1.8**; `sagRecoveryBoost`/`onsetTameBoost`/`breathTameBoost`/`denoiseBias`/`roomCleanupBias` −0.1–0.45 → **−0.15–0.7**; `compressionBias` ±0.45 → **±0.6** with merge scale `× 0.16 → × 0.3` (ratio) and `× 0.5 → × 0.8` (threshold). Keep (and where needed slightly widen) the final profile clamps in `buildAdaptiveProfile` (e.g. `lowMidGainDb` −3.8..1.3 → −3.8..1.6, `presenceGainDb` −2.4..1.8 → −2.8..2.2, `airGainDb` −1.8..1.0 → −2.0..1.3) so the widened directives are actually representable but still safe.
- New directives:
  - `targetLoudnessBiasDb` (−1.5…+1.5): added to the planner `targetDb` (−22) in `planGainForInput` — lets the AI push a whispery or shouty file toward the batch center.
  - `levelerBias` (−0.5…+0.5): scales dynaudnorm `g`/`m` adaptation and glue-compressor `mix` in `buildMixFilter` (±20% at extremes).
  - `headGuardBoost` (0…1): scales the Phase-1 cold-open lift tolerance/max (`tolerance − 0.5 dB`, `max +1.5 dB` at 1.0).
- Update: schema, normalization clamps + defaults (`DEFAULT_AUDIO_REVIEW_ADAPTIVE_DIRECTIVES`), the source-first prompt text (document each directive's exact effect and bounds so the model uses them deliberately), and `aiAudioReview.test.ts` clamp tests.

### 3.4 Give the model better eyes

Extend `AudioReviewMetricSnapshot` (aiAudioReview.ts L21–40) with: `coldOpenDipDb`, `integratedLufs` (from analysis `inputI`), `bandSpectrumDb` (8 numbers), `sibilanceScore`, and (post-render only) the QC delta object. Keep the payload bounded (numbers only, no arrays beyond the 8-band spectrum).

**Exit criteria:** post-render loop runs on flagged files and demonstrably changes output (log shows corrective render kept on at least one degraded test file and discarded when not better); verdict is no longer cosmetic; all review tests green; rate-limit ceiling respected.

---

## Phase 4 — Verification, gates, and documentation

1. **Full suite:** `npm run lint`, `npm run build`, `npm run test:audio-qc` (now includes `batchLoudnessAlign.test.ts` and the new planner/QC cases).
2. **Corpus spot-check (manual, documented in the final summary):** process 3+ real files (one clean, one noisy/echoey, one with the cold-open artifact). Confirm: `coldOpenDipDb` improvement, batch outputs within ±0.5 LU, no new QC flags vs the pre-change build, corrective pass triggering/skipping correctly.
3. **Docs:** update `SUMMARY.md` with a dated section; add lessons to `tasks/lessons.md` if anything bit you.

---

## Explicit non-goals

- No UI redesign (the only acceptable UI change: surface corrective-pass status in the existing queue stage text/log panel).
- No neural enhancement enablement, no new cloud providers, no new dependencies unless unavoidable (K-weighting biquads are ~30 lines of plain math — no dependency).
- No changes to export format (48 kHz mono pcm_f32le mix-ready), loudness presets, or the long-form safe mode chunker beyond what Phase 1.2 explicitly gates off.
- No reranking of >2 candidates; no multi-iteration AI loops (exactly one corrective pass).

## Risk register

| Risk | Mitigation |
| --- | --- |
| Head priming changes duration/alignment | Exact `atrim=start=<preSec>` + sample-count assertion + fail-closed re-render without priming |
| Cold-open lift boosts a genuinely soft artistic opener | Lift requires ≥5 body runs, ≥1.5 dB deficit vs later anchor, caps at +5 dB, and peak guard re-clamps after |
| K-weighting shifts overall level vs today | Feature flag + equal-loudness unit tests + keep −22 target (validated ≈ −23/−24 LUFS mapping) |
| Corrective pass makes output worse | Ranker + hard gates must prefer it by margin, else original kept; both stored in review bundle |
| Gemini rate limit / cost | Post-render review capped at 6 requests/batch, triggered only by deterministic flags |
| Long-file memory (20–30 min files) | Corrective pass skipped ≥1200 s; worker recycle before corrective render ≥600 s; priming skipped in long-form safe mode |

## Suggested execution order

Phase 0 → 1 → 2 → 3 → 4. Phases 0+1 ship together (the metric proves the fix). Phase 2 and Phase 3 are independent of each other and can be separate commits; both depend on Phase 0's metric plumbing.
