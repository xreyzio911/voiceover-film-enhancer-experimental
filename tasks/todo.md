# VO Cinematic Voice Upgrade Review Fixes

## Priority Checklist

- [x] Read the review prompt, original implementation plan/prompt, latest summary, `agent.md`, `tasks/lessons.md`, `package.json`, and required VO code regions.
- [x] Collect uncommitted VO upgrade context and run first Oracle review with the implementation/test context.
- [x] Priority 1: Add the end-edge dip reproducer, split raw vs K-weighted planner domains, calibrate target offset, and verify.
- [x] Run priority 1 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Priority 2: Make batch loudness alignment process one file at a time with safe worker recycling.
- [x] Run priority 2 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Priority 3: Tighten corrective triggers, add high-value WARN rules and per-batch corrective budget.
- [x] Run priority 3 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Priority 4: Remove cold-open metric bias with symmetric edge-trimmed head/body measurement.
- [x] Run priority 4 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Priority 5: Make post-render Gemini directives absolute final values while preserving deterministic delta mapping.
- [x] Run priority 5 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Run second Oracle review after fixes and apply remaining blockers: delivered review-bundle guard, blend duration authority, sparse spike floor split, severe end-edge gates, and atomic long-form output commit.
- [x] Update `SUMMARY.md` and `tasks/lessons.md`; run final verification and record evidence.

# VO Cinematic Voice Upgrade

## Phase Checklist

- [x] Read `implementation-plan.md`, `agent.md`, `tasks/lessons.md`, package scripts, and load-bearing VO files/ranges.
- [x] Phase 0: Add `coldOpenDipDb` / `coldOpenRiskScore`, AI/review snapshot plumbing, WARN/FAIL review checks, and tests.
- [x] Run phase 0 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Phase 1: Fix cold-open planner classification/lift/ramp behavior and add head-primed rendering fallback.
- [x] Run phase 1 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Phase 2: Add K-weighted planner loudness, batch loudness alignment, house-tone blend, adaptive de-esser helper, and tests.
- [x] Run phase 2 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Phase 3: Add post-render AI review, one bounded corrective pass, widened directives, deterministic fallback mapping, and tests.
- [x] Run phase 3 verification: `npm run lint`, `npm run build`, `npm run test:audio-qc`.
- [x] Phase 4: Run full verification, update `SUMMARY.md`, document verification evidence and listening checks.

# Audio Track Splitter Plan

## Checklist

- [x] Inspect repository structure, upload flow, export logic, and tests.
- [x] Add an isolated `audioSplitterService` with batch processing, validation, report generation, and cleanup-friendly file outputs.
- [x] Add `POST /api/audio-splitter` for authenticated multipart WAV uploads and ZIP export.
- [x] Add an "Audio Track Splitter" tool UI that supports multiple WAV files, queue status, success/failure state, and ZIP download.
- [x] Add focused service tests for single/multiple files, filename rules, unsupported/corrupted files, partial batch failure, ZIP filenames, and aligned durations.
- [x] Update setup docs for the local separation engine.
- [x] Run verification and record results.

## Cleaner Splitter Upgrade

- [x] Add an `audio-separator` RoFormer batch worker that loads the model once per batch.
- [x] Keep Demucs as a configurable fallback engine.
- [x] Export only direct model `BGM`/`VOCAL` WAV stems and default them to 16-bit PCM.
- [x] Surface real-time per-file worker progress through the existing job polling API.
- [x] Update tests for two-stem output, bit depth, batch worker progress, and optional worker smoke coverage.
- [x] Update setup docs and environment examples for the Python 3.12/CUDA audio engine.
- [x] Run splitter tests, lint, and build.

## Assumptions

- The app remains a Next.js app with browser-side VO leveling untouched.
- The splitter backend runs in a Node runtime with filesystem access and a local Demucs CLI installation available for real separation.
- Demucs is used as the first practical local/open-source engine; the drama-specific SFX split is isolated as a replaceable heuristic because Demucs does not directly output an SFX stem.

## Review Notes

- `npm run test:audio-qc` passed (70 tests).
- `npm run lint` passed.
- `npm run build` passed.
- Local dev server started at `http://localhost:3000`.
- Cleaner Splitter upgrade: `npm run test:audio-qc` passed (71 pass, 1 optional worker smoke skipped).
- Cleaner Splitter upgrade: `npm run lint` passed.
- Cleaner Splitter upgrade: `npm run build` passed.
- Cleaner Splitter upgrade: local dev server is running at `http://localhost:3000`; HTTP GET `/` returned 200.
- Local audio engine setup: `.venv-audio-splitter` installed with Python 3.11, CUDA Torch 2.11.0+cu128, `audio-separator`, and `imageio-ffmpeg`.
- Local audio engine setup: RoFormer worker smoke test passed with CUDA and the downloaded model.
