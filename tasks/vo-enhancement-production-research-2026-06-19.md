# Production VO Enhancement Research - 2026-06-19

Generated: 2026-06-19

Scope: find production-grade features from the requested repos and other current sources that can make this VO app sound clearer, less echoey, more stable in volume, more cinematic, and more consistent across actors, microphones, and rooms.

## Executive Summary

The production-ready direction is not a generative audio model or an audio-restoration system. It is a controlled speech enhancement/mastering candidate stage wrapped by the app's existing deterministic gain planner, room cleanup, cinematic color, and QC gates.

The current worktree already has a useful server-side neural enhancement path: a `neural-repair` route, a ClearVoice/MossFormer2 48 kHz worker, and a UI toggle that runs `app -> neural speech enhancement -> app` before export. The internal route name still says "repair", but the product direction should be enhancement/mastering only. That path needs production hardening: model prewarm or a worker service, streaming/chunk backpressure for large files, per-engine health checks, pre/post QC acceptance gates, and real corpus A/B validation.

Highest-confidence production candidates:

1. **Local ClearVoice/MossFormer2_SE_48K engine** as the default neural speech enhancement candidate. Use it for background-noise cleanup and clarity, then let the app do final leveling, tone, limiting, and QC.
2. **Auphonic or Cleanvoice API** as deployment-safe production fallbacks when the user accepts upload/cost tradeoffs. Auphonic is stronger for level/loudness consistency; Cleanvoice is stronger as one-call studio enhancement and editing, but filler/silence cuts should be disabled by default for acted VO.
3. **DeepFilterNet** only as a lightweight local 48 kHz denoise prepass. It is not the main echo-removal or cinematic solution.
4. **ElevenLabs Voice Isolator** only as a severe-noise isolation fallback. It is not a leveling, cinematic tone, or full mastering solution.
5. **NVIDIA Maxine Audio Effects SDK** only as an enterprise/self-hosted GPU option for hard room echo and dereverb. It is not a default public deployment path unless the app owns a compatible GPU backend and licensing/access is confirmed.

Reject for direct app feature work:

- `stability-ai/stable-audio-tools`: useful for conditional/generative audio research and training, not VO cleanup.
- `facebookresearch/audiocraft`: useful for generative audio, MusicGen, AudioGen, EnCodec, and watermarking research, not VO cleanup/mastering.
- `pytorch/audio`: useful as part of a PyTorch backend stack, but not a product feature. It is in maintenance phase, so avoid new dependencies on removed/deprecated audio I/O.

## Current App Evidence

Relevant current surfaces:

- `src/components/VoLeveler.tsx` imports `requestNeuralRepair` and `CLEARVOICE_SE_REQUEST`, defines neural enhancement defaults, tracks `processingFlow`, and labels enhanced outputs.
- `src/components/VoLeveler.tsx` currently runs `runNeuralSpeechEnhancementAppPass`, which sends the app output to the internal neural enhancement route, writes the enhanced WAV back into FFmpeg, then runs a final app pass.
- `src/app/api/neural-repair/route.ts` exposes a node runtime route with auth/localhost allow, engine gating, max byte limit, timeout, worker self-test, temporary files, and worker stdout/stderr report parsing. The route name is legacy/internal wording; user-facing copy should not promise restoration.
- `scripts/neural_repair_worker.py` runs `ClearVoice(task=args.mode, model_names=[args.model])`, resamples to 48 kHz, chunks long files with context, bypasses silent chunks, and emits a JSON report.
- `src/lib/gainPlanner.ts` is the deterministic level authority: it targets speech runs, prevents sentence jumps, rides intra-sentence swing, ducks pauses, and peak-guards spikes.
- `src/lib/audioQc.ts` already measures line swing, sentence jump, pause noise, reverb, echo, room score, compression, and clipping, which are the right signals for accepting or rejecting neural candidates.

Important production gap: the current route reads uploaded audio with `arrayBuffer()` and the client reads the whole virtual WAV before upload. That is acceptable for a prototype but should be changed or bounded for long-form production. The route also spawns a Python process per request, which is simpler but slow and fragile compared with a warmed worker/service.

## Current Processing Pipeline Analysis

This is the current VO optimizer pipeline as implemented in the active worktree:

1. **Input and analysis.** The browser-side app loads WAV files into FFmpeg, extracts loudness/true-peak/LRA, analyzes speech-envelope windows, and builds `FileAnalysis` metrics such as `instabilityScore`, `lineSwingScore`, `sentenceJumpScore`, `midLineSagScore`, `endFadeRiskScore`, `sibilanceScore`, `noiseFloorDb`, `noiseContrastDb`, `echoScore`, `roomScore`, `compressionScore`, and speech coverage counts.
2. **Batch reference.** `buildBatchReference` computes robust median low tilt, high tilt, LRA, and per-band spectrum across clean-enough files. This is the app's house-tone anchor for actors recorded on different microphones or settings.
3. **Adaptive profile.** `buildAdaptiveProfile` converts each file's analysis plus the batch reference into conservative processing decisions: high-pass frequency, low-mid warmth, presence/air correction, harshness cuts, compression offsets, denoise strength, room/echo cleanup, de-esser depth, tone-match EQ deltas, onset/breath/click taming, sag recovery, ending protection, and segmentation strategy.
4. **Speech-aware gain planning.** When enabled, the gain planner runs before downstream dynamics. It targets speech runs rather than silence, protects endings, controls body-relative plosives/spikes, and avoids letting `dynaudnorm` become the main level rider. This is the core defense against rare mid-sentence shallow-volume dips and sentence-to-sentence jumps.
5. **Filter-chain rendering.** `buildMixFilter` assembles only the needed FFmpeg filters for the selected candidate: high-pass/EQ, measured-SNR denoise, subtle dereverb/notches, onset/click/breath tamers, gated or bypassed `dynaudnorm`, floor or tail gating, harshness softening, tone-match EQ, cinematic color, de-essing, and final glue compression. The values are clamped and intentionally small so the result does not sound flattened or artificial.
6. **Candidate strategy.** The app renders and ranks `cinematic-stable`, `continuity-safe`, `pause-safe`, and `source-safe` candidates. QC/learned review scores prefer stable dialogue, clean pauses, low compression artifacts, and controlled echo. Risky candidates can be rejected or replaced by safer fallbacks.
7. **Neural enhancement sandwich.** When enabled, the app runs `app -> neural speech enhancement -> app`. The neural pass is a candidate cleanup stage, not the final authority. The final app pass re-applies continuity-safe deterministic leveling and tone control so neural output cannot bypass the house-tone and stability rules.
8. **AI review and adaptive directive plane.** `/api/audio-review` sends bounded source metrics, pre-render adaptive profile snapshots, tone-match deltas, and current controls to Gemini 3.1 Flash-Lite with low thinking and JSON-schema output. It does not send the Gemini key to the browser and it does not use the model as an audio enhancer. The model must return one `perFileProfiles` entry for every input file, each with its own safe profile, preferred render variant, guardrails, listening checks, and bounded `adaptiveDirectives` for warmth, presence, air, de-harshing, sag recovery, onset/breath taming, denoise/room bias, compression bias, and single-pass final-polish intensity. `AI Auto Pilot` now runs source-first per audio: original source review -> per-file adaptive profile -> app render -> mandatory neural VO -> subtle single final app polish -> result.
9. **Delivery.** Mix-ready WAV is the primary product output. Loudness normalization is optional and runs after the mix-ready pass so it cannot mask profile mistakes.

How this addresses the stated problems:

- **Different mics/settings.** Batch median spectrum and LRA create a house reference; tone-match deltas make only the strongest few band corrections, capped to avoid extreme EQ.
- **Mid-sentence shallow-volume dips.** The pipeline now exposes `midLineSagScore`, `sagRecoveryStrength`, `lineContinuityRisk`, ending protection, and continuity-safe candidate selection to both deterministic rendering and Gemini review.
- **Harsh VO.** Sibilance, high tilt, hot peaks, and room brightness drive subtle harshness cuts, de-esser depth, and top-end shelf control. Gemini sees these fields and can recommend keeping or relaxing `softenHarshness` based on objective evidence.
- **Rich, smooth, cinematic tone.** Cinematic color adds small warmth/intelligibility shelves only when emotion protection allows it, while compression is relaxed when other stages are already controlling dynamics.

Complexity scan notes from `python C:\Users\reyha\.codex\skills\complexity-optimizer\scripts\analyze_complexity.py src --format markdown`:

- `src/app/api/audio-review/route.ts` scanner hits are low-risk leads. The route makes one Gemini call per review request; the reported nested loop is bounded extraction of candidate text parts from one provider response.
- `src/components/VoLeveler.tsx` remains the real maintainability hotspot because it owns analysis, profile building, rendering, queueing, AI review state, and output delivery in one large component. The current change keeps behavior localized, but a future cleanup should extract profile snapshots, modal state, and rendering policy helpers into focused modules.
- `src/components/QcReportLab.tsx` and `src/components/AudioTrackSplitter.tsx` still show many nested-loop leads outside this goal's hot path. They should be reviewed separately before broad refactors.

## Requested Repo Decisions

| Repo | Production Fit | Decision | Reason |
| --- | --- | --- | --- |
| `stability-ai/stable-audio-tools` | Low | Reject for VO enhancement | The repo is built around training/fine-tuning/inference of conditional generative audio models and PyTorch Lightning training wrappers, not deterministic speech cleanup or mastering. Source: [stable-audio-tools README](https://github.com/stability-ai/stable-audio-tools). |
| `pytorch/audio` | Medium as dependency, low as feature | Use only indirectly | TorchAudio has moved into maintenance phase and removed/deprecated user-facing features in the 2.8/2.9 transition. Do not build a product feature around its old I/O APIs. Use `soundfile`, `librosa`, `TorchCodec`, or the model package's own loader where possible. Source: [pytorch/audio README](https://github.com/pytorch/audio), [TorchAudio release notes](https://github.com/pytorch/audio/releases). |
| `facebookresearch/audiocraft` | Low | Reject for VO cleanup | AudioCraft is for audio generation research: MusicGen, AudioGen, EnCodec, MAGNeT, JASCO, etc. It is not a speech dereverb/leveling/mastering toolkit, and the README still anchors installation around Python 3.9/PyTorch 2.1. Source: [AudioCraft README](https://github.com/facebookresearch/audiocraft). |

## Useful Open-Source Candidates

### 1. ClearerVoice-Studio / MossFormer2_SE_48K

Decision: **keep and harden as the default local neural speech-enhancement engine.**

Why it fits:

- ClearerVoice-Studio is explicitly a speech processing toolkit with speech enhancement and related speech-processing tasks. For this app, use only the 48 kHz speech-enhancement path; do not use speech super-resolution, target speaker extraction, or any reconstructive mode. Source: [ClearerVoice-Studio README](https://github.com/modelscope/ClearerVoice-Studio).
- It has a 48 kHz speech enhancement model, `MossFormer2_SE_48K`, with Apache-2.0 model card and sample `ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])` usage. Source: [MossFormer2_SE_48K model card](https://huggingface.co/alibabasglab/MossFormer2_SE_48K/blob/main/README.md).
- It matches the app's desired flow: speech-enhancement candidate first, deterministic app pass last.

Production concerns:

- Heavy Python/PyTorch stack. Keep it server-side only.
- Needs model cache, warm worker, GPU/CPU capacity planning, timeout/backpressure, and version pinning.
- It is a speech enhancer, not a guaranteed full dereverb engine. Keep app room cleanup and consider Maxine for true room echo.

App feature to build:

- `clearvoice-se-48k` engine under `neural-repair`.
- Accept only when post-QC proves improvement or no regression.
- Store report fields: engine, model, sample rates, duration, elapsed seconds, chunk counts, pre/post QC metrics, accept/reject reason.

### 2. Resemble Enhance

Decision: **secondary benchmark or fallback, not first production engine.**

Why it is useful:

- It has separate denoise and enhancement stages and is speech-specific. Source: [resemble-enhance README](https://github.com/resemble-ai/resemble-enhance).
- It is useful as a comparison model for perceived clarity and denoise/enhance staging.

Why it is not first:

- The public README targets folder/CLI usage and 44.1 kHz speech enhancement.
- It overlaps with ClearVoice but is less naturally aligned with the app's 48 kHz speech-enhancement worker.

### 3. DeepFilterNet

Decision: **optional lightweight denoise prepass.**

Why it fits:

- DeepFilterNet is a low-complexity full-band 48 kHz speech enhancement framework, with a precompiled `deep-filter` binary path. Source: [DeepFilterNet README](https://github.com/rikorose/deepfilternet).

Best use:

- Fast local denoise for mild broadband noise before the app's main pass.
- CPU-friendly fallback when ClearVoice/Maxine are unavailable.

Limit:

- Treat as denoise, not the main no-echo/cinematic/mastering feature.

## Production SDK/API Candidates

### 1. NVIDIA Maxine Audio Effects SDK

Decision: **enterprise-only premium engine for hard echo and dereverb, not the default deploy path.**

Why it fits:

- NVIDIA documents denoise, room echo removal/dereverb, and denoise plus dereverb. Those are the relevant modes for this app. Ignore audio super-resolution, voice font, and any reconstructive/voice-changing feature unless a separate corpus test proves actor identity is preserved. Source: [NVIDIA Maxine AFX docs](https://docs.nvidia.com/maxine/afx/2.0.0/index.html).
- NVIDIA documents Linux support for systems with at least 10 GB RAM and NVIDIA GPUs with Tensor Cores, such as T4/A10/L4/H100-class GPUs. Source: [NVIDIA Maxine Linux get started](https://docs.nvidia.com/maxine/afx/latest/LinuxAFXSDK/GetStartedOnLinux.html).
- The docs distinguish Windows client integration and Linux server/datacenter deployments. That means users can access Maxine only if this app calls a server-side GPU worker we operate; users cannot just access it from a normal browser/client deployment.

Best app feature:

- Add `maxine-denoise-dereverb` only as a premium server-side engine behind explicit env flags, deployment health checks, and a confirmed GPU/licensing path.
- Trigger when `echoScore`, `reverbScore`, or `roomScore` crosses threshold.

Tradeoffs:

- Requires NVIDIA GPU deployment, SDK/licensing review, and NGC/package access.
- Not a browser/WASM feature.
- Not suitable as the default path for normal hosted deployments like CPU-only servers or serverless platforms.

### 2. Auphonic API

Decision: **strongest cloud fallback for cross-actor consistency and loudness compliance.**

Why it fits:

- Auphonic's adaptive leveler equalizes speakers, avoids amplifying noise/breath/silence, and applies compression/limiting for a balanced mix. It also supports LUFS/true-peak broadcast targets. Source: [Auphonic singletrack algorithms](https://auphonic.com/help/algorithms/singletrack.html).
- Auphonic also documents AI speech isolation and static denoiser paths, including reverb/static-noise removal. Source: [Auphonic singletrack algorithms](https://auphonic.com/help/algorithms/singletrack.html).
- Auphonic has a REST API surface for automated production workflows. Source: [Auphonic API reference](https://eu1.auphonic.com/help/redoc.html).

Best app feature:

- Optional cloud `auphonic-master` job for users who prioritize batch consistency over local-only processing.
- Use as a benchmark against the app's deterministic leveler on a corpus of inconsistent actor mics.

Tradeoffs:

- Cloud upload/privacy/cost.
- Avoid overprocessing if the app has already applied strong leveling.

### 3. Cleanvoice API

Decision: **production cloud candidate for one-call studio enhancement, but disable edit cuts by default.**

Why it fits:

- Cleanvoice advertises Python, JS, and REST APIs; audio/video in/out; studio sound with loudness normalization, level balancing, EQ, dereverb, noise removal, room tone removal, and breath control. Source: [Cleanvoice developers](https://cleanvoice.ai/developers/).
- It has production-oriented claims around EU hosting, GDPR, DPA/SLA, no training on user data, and optional zero retention. Source: [Cleanvoice developers](https://cleanvoice.ai/developers/).

Best app feature:

- `cleanvoice-studio-enhance` cloud engine for users who want a managed enhancement fallback.
- Keep filler-word, dead-air, stutter, and mouth-sound editing off unless the user explicitly chooses an editing workflow, because acted VO timing and performance should not be cut automatically.

### 4. ElevenLabs Voice Isolator

Decision: **severe-noise isolation fallback only.**

Why it fits:

- Voice Isolator is documented as an API for transforming noisy recordings into clean speech, with audio/video file support up to 500 MB and 1 hour. Source: [ElevenLabs Voice Isolator docs](https://elevenlabs.io/docs/overview/capabilities/voice-isolator), [Audio Isolation endpoint](https://elevenlabs.io/docs/api-reference/audio-isolation/convert).

Best app feature:

- `elevenlabs-isolate` cloud candidate for noisy, music/ambience-contaminated takes when local cleanup fails.

Limits:

- It is not a final mastering engine. It does not solve cross-actor volume matching or cinematic tone by itself.

### 5. Dolby.io Enhance

Decision: **watchlist/cloud benchmark, not primary until current Media API availability is confirmed with account access.**

Why:

- Dolby's Node client docs still show media upload, enhance job start, polling, and download examples. Source: [Dolby.io REST APIs client docs](https://api-references.dolby.io/dolbyio-rest-apis-client-node/).
- Search-visible docs around current Dolby developer pages emphasize OptiView/streaming more than Media Enhance. Confirm account-level availability before committing engineering time.

### 6. Krisp SDK

Decision: **real-time/voice-agent candidate, not batch mastering default.**

Why:

- Krisp documents voice isolation, noise cancellation, background voice cancellation, and de-reverberation SDKs across platforms. Source: [Krisp developer hub](https://sdk-docs.krisp.ai/).
- It is strongest for real-time capture and communications, while this app is offline VO batch mastering.

### 7. Adobe Enhance Speech, Supertone Clear, Hush, Accentize dxRevive, Waves Clarity

Decision: **use as external listening benchmarks, not direct first integrations.**

Why:

- Adobe Enhance Speech is strong for one-click cleanup but the public product page is not a normal developer API surface. Source: [Adobe Enhance Speech](https://podcast.adobe.com/en/enhancespeech).
- Supertone Clear, Hush, Accentize dxRevive, and Waves Clarity are credible professional/manual cleanup references, but many of them are restoration-oriented plugin or desktop-app workflows rather than clean web-app API integrations. Use them only as listening references, not as the product direction. Sources: [Supertone Clear](https://www.supertone.ai/en/clear), [Hush](https://hush.audio/products/hush), [Accentize dxRevive](https://www.accentize.com/product/dxrevive/), [Waves Clarity Vx Pro](https://www.waves.com/plugins/clarity-vx-pro).

## OpenAI/Gemini Use

Decision: **use OpenAI or Gemini as an AI review/control plane, not as the audio enhancer.**

Current OpenAI and Gemini audio APIs are strong for understanding, transcription, TTS, realtime voice agents, and spoken responses. They are not currently a production-grade "same actor in, cleaner same actor out" VO mastering endpoint.

Useful roles:

- AI mastering reviewer: sample before/after windows and return structured ratings for echo, room tone, plosives, sibilance, clipping risk, volume jumps, emotional naturalness, and whether the actor still sounds like the same person.
- Preset/candidate/directive selector: combine objective metrics from `audioQc.ts` with AI comments to choose `cinematic-stable`, `continuity-safe`, `pause-safe`, or `source-safe`, keep mandatory neural enhancement active, and apply bounded adaptive directives plus one subtle final-polish pass.
- Transcript-aware phrase protection: use transcription/diarization/timestamps to protect word endings, line starts, pauses, and acted breaths from overprocessing.
- Batch consistency judge: compare each actor/mic take against a project "house tone" profile and recommend subtle EQ/level corrections.
- Regression reviewer: run after rendering and reject outputs that sound overprocessed, warbly, too bright, too thin, clipped, or inconsistent with the batch.

OpenAI fit:

- OpenAI audio docs frame the core audio tasks as speech-to-text, text-to-speech, speech-to-speech, and speech translation. Source: [OpenAI audio and speech guide](https://platform.openai.com/docs/guides/audio).
- OpenAI speech-to-text supports transcription/translation endpoints and GPT-4o transcription models including diarization. Source: [OpenAI speech-to-text guide](https://platform.openai.com/docs/guides/speech-to-text).
- OpenAI TTS turns text into generated speech with built-in voices. It should not be used to "enhance" actor VO unless the product intentionally replaces the actor with AI speech, which is not this app's goal. Source: [OpenAI text-to-speech guide](https://platform.openai.com/docs/guides/text-to-speech).

Gemini fit:

- Gemini audio understanding can analyze audio and generate text responses, including transcription/translation, emotion detection, segment analysis, and timestamps. Source: [Gemini audio understanding](https://ai.google.dev/gemini-api/docs/audio).
- Gemini Live API is for low-latency voice/vision interactions and returns text/audio responses; it is better for voice agents than offline VO mastering. Source: [Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api).
- Gemini TTS transforms text into generated single-speaker or multi-speaker audio. That is useful for generated narration workflows, not for enhancing original actor files. Source: [Gemini TTS guide](https://ai.google.dev/gemini-api/docs/speech-generation).

API key guidance:

- Keep provider keys server-side only. Do not put OpenAI or Gemini keys in the browser.
- Multiple keys can be useful for separating dev/prod, provider fallback, spend caps, and observability. Do not rotate keys to bypass rate limits; request higher quotas or use provider batch/flex tiers instead.
- Store model outputs as review metadata, not as the final authority. The app's deterministic metrics and listening corpus should decide whether an output is accepted.

## Recommended Product Feature

Build a **Production VO Enhancement Engine** behind the existing internal `neural-repair` surface.

User-facing behavior:

- One toggle: `Neural speech enhancement`.
- One advanced selector later: `Local ClearVoice`, `NVIDIA Dereverb`, `Cloud Mastering`, `Severe Noise Isolator`.
- The default remains conservative: app pass, neural candidate, final app pass, QC acceptance.
- If the neural result fails gates, output the deterministic app result and report why the neural candidate was rejected.

Internal engine contract:

```ts
type NeuralRepairEngineId =
  | "clearvoice-se-48k"
  | "maxine-denoise-dereverb"
  | "deepfilter-denoise"
  | "auphonic-master"
  | "cleanvoice-studio"
  | "elevenlabs-isolate";
```

Processing flow:

1. Decode/analyze input with existing app analysis.
2. Render deterministic app baseline.
3. Run one neural candidate only when enabled or triggered by QC risk.
4. Run final app pass on neural output using continuity-safe settings and ending protection.
5. Run pre/post QC.
6. Accept neural result only if it improves the target problem and does not regress volume, clipping, compression, duration, or tone.
7. Write manifest with engine, model, settings, QC deltas, elapsed time, and accept/reject status.

## Acceptance Gates

Minimum objective gates before a neural candidate can replace the app baseline:

- Duration drift: <= 75 ms or <= 0.1 percent, whichever is larger.
- Peak/clipping: no clipping increase; true peak stays within export target.
- Volume consistency: `sentenceJumpScore`, `lineSwingScore`, and `instabilityScore` must not worsen beyond a small tolerance.
- Room cleanup: if triggered by echo/reverb, `echoScore`, `reverbScore`, or `roomScore` must improve.
- Pause cleanliness: `pauseNoiseRisk` must not worsen.
- Dynamics: `compressionScore` must not cross the app's compressed/radio-like threshold.
- Timbre: long-term spectral tilt should stay within the app's house-tone tolerance so actors still sound like themselves.
- Speech integrity: no obvious word truncation, syllable warble, consonant smearing, or voice identity shift in listening review.

Recommended corpus gate:

- At least 20 real files across different actors, mics, rooms, languages/accents, emotional takes, breaths/plosives, and long batch files.
- Compare current app, ClearVoice, Maxine if available, Auphonic/Cleanvoice cloud if acceptable, and one manual cleanup reference export from a tool like Supertone Clear or Waves Clarity.
- Keep deterministic app output as fallback unless a candidate wins on both objective metrics and listening review.

## Implementation Priorities

1. Harden current ClearVoice worker.
   - Add pinned Python environment docs or script.
   - Pre-download/model-cache checks.
   - Worker self-test in deployment health.
   - Fail closed to app-only output.
   - Replace whole-file request path for long files or enforce shorter bounded chunks.

2. Add QC acceptance/rejection.
   - Use current `audioQc.ts` metrics as the gate.
   - Store pre/post metrics in output manifest.
   - Show "enhanced accepted" or "enhanced rejected" badge.

3. Add production engine abstraction.
   - Keep current ClearVoice as one engine, not hardcoded policy.
   - Add Maxine engine only after SDK/licensing/GPU deployment is confirmed; otherwise keep it out of the default product path.
   - Add cloud engine only behind explicit server-side env vars and privacy notice.

4. Add batch house-tone matching.
   - Build a project-level reference from median speech spectrum, loudness, and dynamics.
   - Apply subtle final EQ and level targets across all files after enhancement.
   - This directly addresses different actors using different mics/settings.

5. Measure before claiming "better".
   - No model should become default for all users until it passes the corpus gate.
   - Neural enhancement can sound impressive on one file and damage another.

## Final Recommendation

The most useful production-grade feature is:

**A gated VO enhancement/mastering candidate system, defaulting to ClearVoice/MossFormer2_SE_48K locally for cleanup clarity, with Auphonic/Cleanvoice as deployable cloud fallbacks and Maxine reserved only for a future self-hosted GPU/enterprise tier, always wrapped by the app's deterministic final pass and QC acceptance.**

Do not spend implementation time on `stable-audio-tools` or AudioCraft for this VO quality goal. Use PyTorch/TorchAudio only as backend plumbing required by selected models, not as the app-level feature.
