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
