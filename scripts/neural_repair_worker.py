#!/usr/bin/env python3
import argparse
import faulthandler
import importlib.util
import json
import os
import time
from pathlib import Path

faulthandler.enable()


def module_available(name):
    return importlib.util.find_spec(name) is not None


def emit(payload):
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def load_audio_soundfile(path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype="float32"), int(sr)


def write_audio_soundfile(path, audio, sr):
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio, dtype="float32")
    if audio.size == 0:
        raise RuntimeError("speech enhancement produced zero samples")
    sf.write(path, audio, int(sr), subtype="FLOAT")


def fit_audio_length(audio, target_len):
    import numpy as np

    audio = np.asarray(audio, dtype="float32").reshape(-1)
    if target_len <= 0:
        return np.zeros(0, dtype="float32")
    if audio.shape[0] == target_len:
        return audio
    if audio.shape[0] > target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - audio.shape[0]), mode="constant")


def extract_mono_output(output):
    import numpy as np

    output = np.asarray(output, dtype="float32")
    if output.ndim == 3:
        return output[0, 0, :]
    if output.ndim == 2:
        return output[0, :]
    return output.reshape(-1)


def clearvoice_process_chunk(model, audio):
    import numpy as np
    import torch

    audio = np.asarray(audio, dtype="float32").reshape(-1)
    if audio.shape[0] == 0:
        return audio
    with torch.inference_mode():
        output = model(np.reshape(audio, [1, audio.shape[0]]), False)
    return fit_audio_length(extract_mono_output(output), audio.shape[0])


def configure_torch_threads():
    try:
        import torch

        raw_threads = os.environ.get("VO_NEURAL_REPAIR_TORCH_THREADS", "2")
        threads = max(1, min(int(raw_threads), max(os.cpu_count() or 2, 1)))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        return threads
    except Exception:
        return None


def clearvoice_should_bypass(audio, silence_db):
    import numpy as np

    if audio.shape[0] == 0:
        return True
    peak = float(np.max(np.abs(audio)))
    return peak < 10 ** (silence_db / 20)


def merge_sample_spans(spans, merge_gap_samples):
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1] + merge_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(start), int(end)) for start, end in merged if end > start]


def detect_active_speech_spans(audio, sr, silence_db, pad_ms, merge_gap_ms, min_span_ms):
    import numpy as np

    audio = np.asarray(audio, dtype="float32").reshape(-1)
    if audio.shape[0] == 0:
        return [], {"activeDutyPct": 0.0, "activeSeconds": 0.0, "activeThresholdDb": silence_db}

    frame_samples = max(int(sr * 0.025), 1)
    hop_samples = max(int(sr * 0.010), 1)
    if audio.shape[0] <= frame_samples:
        peak = float(np.max(np.abs(audio)))
        if peak >= 10 ** (silence_db / 20):
            return [(0, audio.shape[0])], {
                "activeDutyPct": 100.0,
                "activeSeconds": float(audio.shape[0] / max(sr, 1)),
                "activeThresholdDb": silence_db,
            }
        return [], {"activeDutyPct": 0.0, "activeSeconds": 0.0, "activeThresholdDb": silence_db}

    starts = np.arange(0, audio.shape[0] - frame_samples + 1, hop_samples, dtype=np.int64)
    rms_values = np.empty(starts.shape[0], dtype="float32")
    peak_values = np.empty(starts.shape[0], dtype="float32")
    for index, start in enumerate(starts):
        frame = audio[start : start + frame_samples]
        rms_values[index] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        peak_values[index] = float(np.max(np.abs(frame)))

    rms_db = 20 * np.log10(np.maximum(rms_values, 1e-8))
    peak_db = 20 * np.log10(np.maximum(peak_values, 1e-8))
    noise_floor_db = float(np.percentile(rms_db, 20))
    active_threshold_db = max(noise_floor_db + 8.0, silence_db)
    transient_threshold_db = max(active_threshold_db + 7.0, -48.0)
    active = np.logical_or(rms_db >= active_threshold_db, peak_db >= transient_threshold_db)

    spans = []
    pad_samples = max(int(sr * max(pad_ms, 0) / 1000), 0)
    min_span_samples = max(int(sr * max(min_span_ms, 0) / 1000), 1)
    active_start = None
    for index, is_active in enumerate(active):
        frame_start = int(starts[index])
        frame_end = min(frame_start + frame_samples, audio.shape[0])
        if is_active and active_start is None:
            active_start = frame_start
        elif not is_active and active_start is not None:
            if frame_end - active_start >= min_span_samples:
                spans.append((max(0, active_start - pad_samples), min(audio.shape[0], frame_end + pad_samples)))
            active_start = None
    if active_start is not None:
        spans.append((max(0, active_start - pad_samples), audio.shape[0]))

    spans = merge_sample_spans(spans, int(sr * max(merge_gap_ms, 0) / 1000))
    active_samples = sum(end - start for start, end in spans)
    return spans, {
        "activeDutyPct": float((active_samples / max(audio.shape[0], 1)) * 100),
        "activeSeconds": float(active_samples / max(sr, 1)),
        "activeThresholdDb": float(active_threshold_db),
    }


def write_repaired_block(output, original, block_start, block, sr, fade_ms):
    import numpy as np

    block = np.asarray(block, dtype="float32").reshape(-1)
    block_end = min(block_start + block.shape[0], output.shape[0])
    if block_end <= block_start:
        return
    block = block[: block_end - block_start]
    fade_samples = min(int(sr * max(fade_ms, 0) / 1000), block.shape[0] // 2)
    if fade_samples <= 0:
        output[block_start:block_end] = block
        return

    middle_start = block_start + fade_samples
    middle_end = block_end - fade_samples
    if middle_start < middle_end:
        output[middle_start:middle_end] = block[fade_samples : block.shape[0] - fade_samples]

    if block_start == 0:
        output[block_start:middle_start] = block[:fade_samples]
    else:
        alpha = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype="float32")
        output[block_start:middle_start] = (
            original[block_start:middle_start] * (1.0 - alpha) + block[:fade_samples] * alpha
        )

    if block_end == output.shape[0]:
        output[middle_end:block_end] = block[block.shape[0] - fade_samples :]
    else:
        alpha = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype="float32")
        output[middle_end:block_end] = (
            block[block.shape[0] - fade_samples :] * alpha + original[middle_end:block_end] * (1.0 - alpha)
        )


def clearvoice_process_speech_aware(model, audio, sr, args):
    import numpy as np

    audio = np.asarray(audio, dtype="float32").reshape(-1)
    spans, speech_report = detect_active_speech_spans(
        audio,
        sr,
        args.clearvoice_silence_db,
        args.speech_aware_pad_ms,
        args.speech_aware_merge_gap_ms,
        args.speech_aware_min_span_ms,
    )
    duration_seconds = audio.shape[0] / max(sr, 1)
    if (
        not args.speech_aware
        or duration_seconds < args.speech_aware_min_duration
        or not spans
        or speech_report["activeDutyPct"] >= args.speech_aware_max_duty_pct
    ):
        output, report = clearvoice_process_longform(
            model,
            audio,
            sr,
            args.clearvoice_chunk_seconds,
            args.clearvoice_context_ms,
            args.clearvoice_silence_db,
        )
        return output, {
            **report,
            **speech_report,
            "speechAware": False,
            "speechAwareSpans": len(spans),
        }

    output = audio.copy()
    chunk_samples = max(int(sr * max(args.clearvoice_chunk_seconds, 1.0)), int(sr))
    context_samples = max(0, min(int(sr * max(args.clearvoice_context_ms, 0) / 1000), chunk_samples // 4))
    total = 0
    processed = 0
    bypassed = 0
    processed_samples = 0

    for span_start, span_end in spans:
        cursor = span_start
        while cursor < span_end:
            block_end = min(cursor + chunk_samples, span_end)
            read_start = max(0, cursor - context_samples)
            read_end = min(audio.shape[0], block_end + context_samples)
            chunk = audio[read_start:read_end]
            total += 1
            if clearvoice_should_bypass(chunk, args.clearvoice_silence_db):
                bypassed += 1
            else:
                repaired = clearvoice_process_chunk(model, chunk)
                crop_start = cursor - read_start
                crop_end = crop_start + (block_end - cursor)
                block = fit_audio_length(repaired[crop_start:crop_end], block_end - cursor)
                write_repaired_block(output, audio, cursor, block, sr, args.speech_aware_fade_ms)
                processed += 1
                processed_samples += block.shape[0]
            cursor = block_end

    return output, {
        "chunksTotal": total,
        "chunksProcessed": processed,
        "chunksBypassed": bypassed,
        "speechAware": True,
        "speechAwareSpans": len(spans),
        "activeDutyPct": speech_report["activeDutyPct"],
        "activeSeconds": speech_report["activeSeconds"],
        "activeThresholdDb": speech_report["activeThresholdDb"],
        "processedSeconds": float(processed_samples / max(sr, 1)),
    }


def clearvoice_process_longform(model, audio, sr, chunk_seconds, context_ms, silence_db):
    import numpy as np

    chunk_samples = max(int(sr * max(chunk_seconds, 1.0)), int(sr))
    context_samples = max(0, min(int(sr * max(context_ms, 0) / 1000), chunk_samples // 4))
    output = np.zeros_like(audio, dtype="float32")
    total = 0
    processed = 0
    bypassed = 0

    for block_start in range(0, audio.shape[0], chunk_samples):
        block_end = min(block_start + chunk_samples, audio.shape[0])
        read_start = max(0, block_start - context_samples)
        read_end = min(audio.shape[0], block_end + context_samples)
        chunk = audio[read_start:read_end]
        total += 1

        if clearvoice_should_bypass(chunk, silence_db):
            block = audio[block_start:block_end]
            bypassed += 1
        else:
            repaired = clearvoice_process_chunk(model, chunk)
            crop_start = block_start - read_start
            crop_end = crop_start + (block_end - block_start)
            block = fit_audio_length(repaired[crop_start:crop_end], block_end - block_start)
            processed += 1

        output[block_start:block_end] = block

    return output, {"chunksTotal": total, "chunksProcessed": processed, "chunksBypassed": bypassed}


def resample_numpy(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    if not module_available("librosa"):
        raise RuntimeError("librosa is required to resample speech enhancement inputs to 48 kHz")
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype("float32")


def run_clearvoice(args):
    from clearvoice import ClearVoice

    start = time.perf_counter()
    torchThreads = configure_torch_threads()
    input_audio, input_sr = load_audio_soundfile(args.input)
    target_sr = 48000
    audio = resample_numpy(input_audio, input_sr, target_sr)

    model = ClearVoice(task=args.mode, model_names=[args.model])
    output_audio, chunk_report = clearvoice_process_speech_aware(
        model,
        audio,
        target_sr,
        args,
    )
    if output_audio.size == 0:
        raise RuntimeError("ClearVoice produced zero samples")

    write_audio_soundfile(args.output, output_audio, target_sr)
    output_bytes = Path(args.output).stat().st_size
    if output_bytes <= 44:
        raise RuntimeError(f"ClearVoice produced invalid WAV output ({output_bytes} bytes)")
    elapsed = time.perf_counter() - start
    return {
        "engine": "clearvoice",
        "mode": args.mode,
        "model": args.model,
        "inputSampleRate": input_sr,
        "outputSampleRate": target_sr,
        "durationSeconds": float(len(input_audio) / max(input_sr, 1)),
        "elapsedSeconds": elapsed,
        "outputSamples": int(output_audio.shape[0]),
        "outputBytes": int(output_bytes),
        **({"torchThreads": torchThreads} if torchThreads is not None else {}),
        **chunk_report,
    }


def self_test():
    emit(
        {
            "ok": True,
            "modules": {
                "clearvoice": module_available("clearvoice"),
                "soundfile": module_available("soundfile"),
                "librosa": module_available("librosa"),
                "torch": module_available("torch"),
            },
        }
    )


def main():
    parser = argparse.ArgumentParser(description="VO neural repair worker")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--engine", choices=["clearvoice"])
    parser.add_argument("--mode", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clearvoice-chunk-seconds", type=float, default=8.0)
    parser.add_argument("--clearvoice-context-ms", type=float, default=250.0)
    parser.add_argument("--clearvoice-silence-db", type=float, default=-70.0)
    parser.add_argument("--speech-aware", action="store_true")
    parser.add_argument("--speech-aware-min-duration", type=float, default=90.0)
    parser.add_argument("--speech-aware-max-duty-pct", type=float, default=62.0)
    parser.add_argument("--speech-aware-pad-ms", type=float, default=360.0)
    parser.add_argument("--speech-aware-merge-gap-ms", type=float, default=900.0)
    parser.add_argument("--speech-aware-min-span-ms", type=float, default=140.0)
    parser.add_argument("--speech-aware-fade-ms", type=float, default=80.0)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if not args.input or not args.output or not args.engine or not args.mode or not args.model:
        raise RuntimeError("input, output, engine, mode, and model are required")
    if not Path(args.input).exists():
        raise RuntimeError(f"input file does not exist: {args.input}")

    report = run_clearvoice(args)

    emit(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit({"ok": False, "error": str(exc)})
        raise SystemExit(2)
