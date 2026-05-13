# Lessons

## 2026-04-28 - Next Dev Upload Limit

- What went wrong: The Audio Track Splitter route accepted multipart uploads, but Next's proxy body clone limit stayed at the 10 MB default, so WAV batches were truncated before `request.formData()`.
- Rule to prevent it: For any new upload route intended for large media, check the framework-level request body/proxy limit in addition to the route handler logic.
- How to verify next time: Run or simulate an upload larger than the default limit and confirm the route sees a complete multipart body; run `next build` after config changes.

## 2026-04-28 - Demucs Runtime Setup On Windows

- What went wrong: Demucs installed successfully through pip, but `demucs.exe` was placed in a user Scripts folder outside PATH; the Python 3.13 environment also had mismatched `torch` and `torchaudio` versions.
- Rule to prevent it: After adding a CLI-backed ML feature, verify the exact configured command from the app environment, not just package installation success.
- How to verify next time: Run `python.exe -m demucs --help` and import-check `torch`/`torchaudio` before testing the app upload path.

## 2026-04-28 - Demucs Segment Argument

- What went wrong: The app defaulted `AUDIO_SPLITTER_DEMUCS_SEGMENT` to `7.8`, but the installed Demucs CLI expects `--segment` as an integer and rejects decimal values. Changing it to `8` parsed but exceeded HTDemucs' trained maximum of `7.8`.
- Rule to prevent it: Validate and normalize external CLI option values to the exact type accepted by the installed tool and model.
- How to verify next time: Run the configured Demucs command with every configured option on a tiny WAV before processing real media.

## 2026-04-28 - Splitter Progress And Runtime

- What went wrong: A single long POST meant the frontend could not observe backend per-file progress, so the queue showed file 1 while the terminal was already on later files.
- Rule to prevent it: Long-running media jobs need an explicit job/status API or stream; don't rely on a single request/response for multi-file ML work.
- How to verify next time: Run a multi-file batch and confirm the UI active file changes before the ZIP is ready.

## 2026-04-28 - SFX Stem Quality And Output Size

- What went wrong: Deriving SFX from music stems produced a waveform too similar to BGM, and 32-bit float WAV outputs were unnecessarily large.
- Rule to prevent it: If a requested stem is not supported by the selected model, prefer a simpler truthful output contract over heuristic stems that look precise but are not useful.
- How to verify next time: Inspect output count, waveform differences, and bit depth/file size on a real sample before presenting stem separation as production-ready.

## 2026-04-28 - Relative ML Runtime Paths

- What went wrong: The splitter stored `.venv-audio-splitter\Scripts\python.exe` as a relative command, but Node spawned it from each temporary job directory, causing `ENOENT`.
- Rule to prevent it: Resolve local runtime commands, model directories, and worker paths from the project root before passing them to `spawn`.
- How to verify next time: Run the configured backend path through the real API, not only a direct shell command from the repo root.

## 2026-04-28 - Audio Separator Model Cache Directory

- What went wrong: `.env` pointed `AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL_DIR` to `.audio-separator-models`, but that directory did not exist, so `audio-separator` failed before loading the cached model.
- Rule to prevent it: Create configured model/cache directories in setup scripts and again at runtime before initializing ML libraries.
- How to verify next time: Smoke test with the same model directory configured in `.env`, not only the library's default user cache.
