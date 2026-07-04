# Session Summary (2026-02-08)

## Project Scope Completed
This session focused on making the VO leveling app:
- more consistent across different mics/recordings,
- safer against overprocessing,
- more robust for larger batches,
- easier to use for internal team workflows.

Primary implementation file was `src/components/VoLeveler.tsx`.

## Major Audio Pipeline Upgrades

### 1) Smart per-file analysis + batch tone matching
- Added per-file analysis before processing:
  - loudness metrics (`input_i`, `input_lra`, `input_tp`, `input_thresh`) via `loudnorm`,
  - tonal band RMS (low/mid/high) via `astats`.
- Added batch reference profile (median-based) to align multiple actors toward a common tonal center.
- Added adaptive per-file profile generation:
  - HPF and low-mid cleanup offsets,
  - presence/air adjustments,
  - noise-risk-aware behavior,
  - dynamic control offsets.

### 2) Smart Voice Match control
- Added `Smart voice match` mode in UI (`Off`, `Gentle`, `Balanced`).
- Default kept conservative to avoid "AI overprocessed" sound.

### 3) Non-conflicting DSP chain
- Removed clashes between overlapping processes:
  - merged static harshness EQ + smart-match EQ into a single net move,
  - prevented breath compand and floor guard from stacking unnecessarily.
- Added logic to prioritize floor guard on noisy tracks.
- Rebalanced compressor behavior so upstream processors do not cause extra "radio-style" compression.

### 4) Cinematic softening improvements
- Tuned harshness handling for emotional loud lines:
  - stronger but smoother upper-mid/top-end softening,
  - optional extra top-end trim only when needed.
- Reduced compand aggressiveness and softened floor-guard curves.
- Slowed compression timing and changed limiter behavior for more natural voice feel.

### 5) Smarter consistency while preserving emotion
- Implemented adaptive "emotion protection":
  - new profile signals: `levelingNeed` + `emotionProtection`,
  - leveling tightens when dynamics are uneven,
  - compression/leveling relaxes when emotional peak behavior is detected.
- Added adaptive compressor threshold/ratio/attack/release/mix logic to hold average loudness while keeping performance dynamics.

## Robustness and Stability Fixes

### 1) FFmpeg runtime hardening
- Added command-level exit checking and better error summaries.
- Added worker reset path only for real fatal runtime conditions.
- Removed false-failure behavior caused by treating log text `Aborted()` as a hard failure.
- Added log buffer capping to reduce browser memory pressure.

### 2) dynaudnorm validity fixes
- Fixed invalid `dynaudnorm` ranges that previously triggered processing failures.
- Ensured `m` is always odd (as required by `dynaudnorm`) using a guard function.
- Updated preset `m` values to odd values to avoid "filter size is invalid" warnings.

### 3) Output/file lifecycle safety
- Added safer temp file cleanup and duplicate-safe output naming.
- Added output object URL cleanup to prevent memory leaks.

## UX and Workflow Improvements

### 1) Bulk download UX
- Replaced "download all files one-by-one" behavior with ZIP export.
- Added ZIP generation progress and a single downloadable archive output.
- Added `jszip` dependency.

### 2) File intake behavior
- Dropzone/file-picker now stacks new files instead of replacing previous selections.
- Duplicate file detection added (`name + size + lastModified`).
- Input reset added so selecting files one-by-one repeatedly works reliably.

### 3) UI copy updates
- Updated hero/feature copy to match adaptive tone-matching behavior.
- Updated control helper text to reflect current DSP logic.

## Files Updated
- `src/components/VoLeveler.tsx` (main implementation changes)
- `src/app/page.tsx` (hero text update)
- `eslint.config.mjs` (ignore `public/ffmpeg/**` third-party bundle)
- `package.json` / `package-lock.json` (added `jszip`)

## Build/Quality Status
Throughout the session, changes were repeatedly validated with:
- `npm run lint`
- `npm run build`

Latest state at handoff: both pass successfully.

## Product/Deployment Guidance Discussed (No Code Applied)
- For remote team usage, simple hosting is preferred over per-user local setup.
- Compute requirements are low because audio processing is browser-side (ffmpeg.wasm).
- SSO is optional but recommended for lower auth maintenance.
- If using SSO, access can be restricted to one account or a strict allowlist.
- Username/password auth is possible but has higher maintenance/security overhead than SSO.

---

# Session Summary (2026-07-02)

## Cinematic VO Upgrade

- Added cold-open QC metrics (`coldOpenDipDb`, `coldOpenRiskScore`) across source analysis, candidate QC logs, AI review snapshots, and auto-review WARN/FAIL checks.
- Fixed cold-open planner behavior:
  - opener speech is protected from breath/fragment misclassification unless strong sample-backed transient evidence exists,
  - quiet first dialogue runs can lift toward the later dialogue anchor within bounded caps,
  - first-run attack ramps complete before the detected opener so the first phoneme does not start at the expander floor.
- Added exact-duration head priming for stateful ffmpeg chains on single-pass renders, segment zero, and final app polish, with duration verification and unprimed fallback.
- Added K-weighted planner frame energy so boomy and bright voices converge by perceived loudness instead of plain RMS.
- Added batch mix-ready loudness alignment over completed clean outputs, median-anchored and capped at +/-2 dB, skipped for loudness-normalized delivery paths.
- Hardened tone matching with a fixed cinematic VO house-curve blend and adaptive de-esser band placement from measured spectrum.
- Expanded AI adaptive directives with bounded `targetLoudnessBiasDb`, `levelerBias`, and `headGuardBoost`, plus wider safe ranges for tone, de-harshing, noise/room, and compression nudges.
- Added post-render review support and exactly one bounded corrective render for flagged files. Corrective output is kept only if it passes hard gates and beats the original by the learned-ranker margin; otherwise the original is retained.

## Guardrails Preserved

- Audio Splitter files were not modified.
- Neural speech enhancement remains disabled by default and constrained to the existing off path.
- Corrective and head-prime stages fail closed to the existing render behavior.

## Verification

- Phase gates were run after each implementation phase:
  - `npm run lint`
  - `npm run build`
  - `npm run test:audio-qc`
- The audio-QC suite now includes `batchLoudnessAlign.test.ts` and `spectrum.test.ts`; latest counted run passed 122 tests with 1 skipped worker smoke test.
- Manual listening/corpus spot-checks were not run in this pass because no specific real audio corpus was selected for the implementation run.

---

# Session Summary (2026-07-02)

## Cinematic VO Review Fixes

- Priority 1: Split gain-planner raw and K-weighted domains so speech masks, run classification, peak guards, and ending guards stay on raw envelope data while target/gain/micro-ride use optional K-weighted loudness. Added WARN-only `endEdgeDipDb` coverage for short final-phoneme dips that `endFadeRiskScore` did not isolate.
- Priority 2: Reworked batch mix-ready loudness alignment so MEMFS holds only the current file during measurement and only the current input/output pair during render. Each file is deleted before moving to the next, and the worker recycles through `refreshFfmpeg` after cumulative processed audio crosses `BATCH_AUDIO_RECYCLE_SECONDS`.
- Priority 3: Tightened corrective triggers to any FAIL, two or more WARNs, one high-value WARN (`cold_open_dip`, `harsh_sibilance`, `too_compressed`, `level_uneven`), or hard gates. Added `resolveCorrectiveMaxFilesPerBatch(jobs.length) = max(2, ceil(jobs.length * 0.4))` plus budget/skip logs.
- Priority 4: Made `coldOpenDipDb` measure the first `COLD_OPEN_RUN_COUNT` run bodies with the same edge-trimmed body logic used for later dialogue, removing edge-only cold-open false positives without retuning thresholds.
- Priority 5: Updated post-render Gemini guidance so `adaptiveDirectives` are absolute final corrective values, not deltas. The deterministic `CORRECTIVE_DIRECTIVE_MAP` remains delta-on-base.

## Constants And Evidence

- `PLANNER_K_TARGET_OFFSET_DB = 0.0`: mixed-formant speech fixtures with -4/0/+4 dB high-frequency tilt measured +0.15/-0.04/-0.44 dB raw body shift after the raw/K split, within the +/-0.5 dB pre-K baseline.
- `BATCH_AUDIO_RECYCLE_SECONDS = 2400`: batch align now writes, measures, deletes, and optionally recycles per file before planning; render alignment writes input, renders output, reads/replaces the `OutputEntry`, deletes both temp files, then moves to the next file.
- `CORRECTIVE_MAX_FILES_PER_BATCH = max(2, ceil(jobs.length * 0.4))`: exhausted-budget files skip before Gemini post-render review or corrective render, with trigger details logged.
- Corrective adoption remains ranker-gated: one corrective render max, hard gates must pass, and the corrective must beat the original by `CORRECTIVE_WIN_MARGIN`.

## Verification

- Per-priority gates were run after each fix: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- Latest counted run before final closeout: `npm run test:audio-qc` passed 131 tests total, 130 pass, 1 skipped optional worker smoke test.

## Listening Checklist

- A/B the first three spoken runs for each actor: confirm the opening no longer dips down, spikes, then stabilizes.
- Check bright/sibilant line endings: final consonants should stay audible without a new end-edge dip or over-bright tail.
- Check boomy versus bright actors in the same batch: perceived dialogue body should converge without flattening performance dynamics.
- Check batches with three or more flagged files: corrective renders should appear only for clear FAIL/high-value WARN/hard-gate cases and stop at the logged budget.
- Check any Gemini corrective pass: returned directive values should behave as final absolute settings, while deterministic fallback corrections remain additive deltas from the active base.

---

# Session Summary (2026-07-03)

## Oracle Review Closeout

- Ran the requested first Oracle review over the uncommitted VO upgrade context, applied the release-blocking and high-priority recommendations, then ran the requested second Oracle review.
- Second Oracle blockers addressed:
  - Review bundles now fail closed for blend, loudness, and multi-file batch-alignment paths so stale `winner.wav` artifacts are not emitted.
  - Scene-blend `amix` now uses dry-track duration as the output authority to keep strict duration gates stable.
  - Sparse clean-take spike shaping was split from the residual loud-cluster safety floor so explicit zeroes cannot bypass safety.
  - Severe end-edge dips now hard-gate ranking, fail auto-review, and trigger corrective eligibility through `endings_damaged`.
  - Long-form chunk outputs are staged locally and committed to the UI only after all required parts and variants pass duration and true-peak gates.

## Final Verification

- `npm run test:audio-qc` passed 140 tests total: 139 pass, 1 optional worker smoke skipped.
- `npm run lint` passed.
- `npm run build` passed.
- `git diff --check` passed with only LF-to-CRLF working-copy warnings.
