# VO Batch Leveler Web

Browser-based voice-over processing app built with Next.js and FFmpeg WASM.

## Authentication

Google SSO is enabled and access is restricted to:

- `shortsprojektt@gmail.com`
- `reyhanputraph@gmail.com`

The allowlist is defined in `src/lib/authAllowlist.ts`.

## Environment Variables

Copy `.env.example` to `.env.local` and fill values:

```bash
AUTH_SECRET=replace-with-a-long-random-string
NEXTAUTH_URL=http://localhost:3000
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## Google OAuth Setup

In Google Cloud Console, configure OAuth client:

- Authorized JavaScript origins:
  - `http://localhost:3000`
  - your production URL (for example `https://your-app.vercel.app`)
- Authorized redirect URIs:
  - `http://localhost:3000/api/auth/callback/google`
  - `https://your-app.vercel.app/api/auth/callback/google`

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Audio Track Splitter Setup

The Audio Track Splitter UI posts WAV batches to `POST /api/audio-splitter`. The default backend engine is `audio-separator` with a BS-RoFormer vocal/instrumental model. It exports only direct model stems:

- `originalName_BGM.wav`
- `originalName_VOCAL.wav`

The BGM stem is the model's direct `Instrumental` output. There is no SFX extraction pass, EQ, denoise, high/low-pass filter, or smoothing filter. The app re-encodes output WAVs to 16-bit PCM by default to reduce ZIP size.

Every splitter ZIP includes `split_report.txt` and `split_report.json` with per-stem peak, RMS, clipping, duration, sample-rate, and warning metadata. VO Optimizer ZIP exports include `delivery_manifest.json` so internal deliveries can be audited after download.

For Windows/local setup, install Python 3.12 and run:

```bash
powershell -ExecutionPolicy Bypass -File scripts/setup-audio-splitter.ps1
```

Then set:

```bash
AUDIO_SPLITTER_ENGINE=audio-separator
AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND=.venv-audio-splitter\Scripts\python.exe
AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL=model_bs_roformer_ep_317_sdr_12.9755.ckpt
AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL_DIR=.audio-separator-models
AUDIO_SPLITTER_DEVICE=cpu
AUDIO_SPLITTER_OUTPUT_BIT_DEPTH=16
AUDIO_SPLITTER_AUDIO_SEPARATOR_SAMPLE_RATE=44100
AUDIO_SPLITTER_AUDIO_SEPARATOR_NORMALIZATION=0.98
AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_AUTOCAST=0
```

Relative splitter paths such as `.venv-audio-splitter\Scripts\python.exe` and `.audio-separator-models` are resolved from the project root, not from the temporary job folder.

The first real split may download the RoFormer model. The default app configuration uses CPU so non-GPU deployments do not try to allocate CUDA. CPU is slower on drama-length files. Verify the audio environment:

```bash
.venv-audio-splitter\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Useful knobs:

- `AUDIO_SPLITTER_AUDIO_SEPARATOR_CHUNK_DURATION=30` to keep memory bounded on long tracks.
- `AUDIO_SPLITTER_AUDIO_SEPARATOR_SAMPLE_RATE=44100` keeps RoFormer on its model rate; the ZIP report warns when this differs from the source WAV.
- `AUDIO_SPLITTER_AUDIO_SEPARATOR_NORMALIZATION=0.98` leaves light sample-peak headroom without adding compression or filtering.
- `AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_AUTOCAST=0` keeps the default CPU-safe. Set it to `1` only with an explicit CUDA deployment.
- `AUDIO_SPLITTER_OUTPUT_BIT_DEPTH=16`; use `24` only if larger delivery files are acceptable.
- `AUDIO_SPLITTER_TIMEOUT_MS=7200000` for very long batches.

Fallback engine:

```bash
AUDIO_SPLITTER_ENGINE=demucs
AUDIO_SPLITTER_DEMUCS_COMMAND="python.exe -m demucs"
AUDIO_SPLITTER_DEMUCS_MODEL=htdemucs
AUDIO_SPLITTER_DEMUCS_DEVICE=cpu
AUDIO_SPLITTER_DEMUCS_SEGMENT=7
AUDIO_SPLITTER_DEMUCS_SHIFTS=1
```

Run this feature on a Node host with filesystem access and enough CPU/GPU memory for local ML separation. Serverless deployments are not a good fit for long drama tracks.

## Build

```bash
npm run lint
npm run build
```
