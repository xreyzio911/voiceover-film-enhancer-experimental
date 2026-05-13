#!/usr/bin/env python3
"""Batch audio-separator worker for the Next.js Audio Track Splitter.

The Node service owns validation, ZIP packaging, and final WAV bit depth. This
worker keeps the separation model loaded while processing one batch and emits
machine-readable progress events for the polling UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path


EVENT_PREFIX = "AUDIO_SPLITTER_EVENT "


def emit(event: dict) -> None:
    print(f"{EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}", flush=True)


def positive_float(value: str | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def find_stem_file(work_dir: Path, tokens: tuple[str, ...], extension: str) -> str | None:
    wanted_extension = f".{extension.lower().lstrip('.')}"
    matches: list[Path] = []
    for candidate in work_dir.rglob(f"*{wanted_extension}"):
        name = candidate.name.lower()
        if any(token in name for token in tokens):
            matches.append(candidate)
    if not matches:
        return None
    matches.sort(key=lambda item: (len(item.name), item.name.lower()))
    return str(matches[0])


def find_stem_path(paths: list[str], work_dir: Path, tokens: tuple[str, ...], extension: str) -> str | None:
    wanted_extension = f".{extension.lower().lstrip('.')}"
    matches: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = work_dir / candidate
        name = candidate.name.lower()
        if candidate.suffix.lower() != wanted_extension:
            continue
        if any(token in name for token in tokens):
            matches.append(candidate)
    if not matches:
        return None
    matches.sort(key=lambda item: (len(item.name), item.name.lower()))
    return str(matches[0])


def ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
    except Exception:
        return

    source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not source.exists():
        return

    shim_dir = Path.home() / ".cache" / "audio-splitter-ffmpeg"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "ffmpeg.exe"
    if not shim.exists() or shim.stat().st_size != source.stat().st_size:
        shutil.copy2(source, shim)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audio-separator on a batch manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "directml"))
    parser.add_argument("--output-format", default="WAV")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--normalization", type=float, default=1.0)
    parser.add_argument("--amplification", type=float, default=0.0)
    parser.add_argument("--chunk-duration", type=positive_float)
    parser.add_argument("--mdxc-segment-size", type=int, default=256)
    parser.add_argument("--mdxc-overlap", type=float, default=8.0)
    parser.add_argument("--mdxc-batch-size", type=int, default=1)
    parser.add_argument("--use-soundfile", action="store_true")
    parser.add_argument("--use-autocast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        # Empty string leaves torch.cuda.is_available() true with zero devices
        # on some Windows/CUDA builds, which later crashes RoFormer while it
        # probes flash-attention device properties. -1 reliably hides CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.model_dir:
        os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = args.model_dir

    ensure_ffmpeg_on_path()

    try:
        import torch
        from audio_separator.separator import Separator
    except Exception as exc:
        emit(
            {
                "type": "startup-error",
                "message": (
                    "Unable to import audio-separator. Install the audio engine environment first. "
                    f"{exc}"
                ),
            }
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 2

    with open(args.manifest, "r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    items = manifest.get("items", [])
    if not items:
        emit({"type": "startup-error", "message": "Batch manifest did not contain any items."})
        return 2

    use_autocast = bool(args.use_autocast and args.device != "cpu" and torch.cuda.is_available())
    model_dir = args.model_dir or os.path.join(Path.home(), ".cache", "audio-separator-models")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    try:
        emit({"type": "file-progress", "inputIndex": items[0]["inputIndex"], "message": "Loading RoFormer model"})
        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=model_dir,
            output_dir=str(Path(items[0]["workDir"])),
            output_format=args.output_format,
            normalization_threshold=args.normalization,
            amplification_threshold=args.amplification,
            sample_rate=args.sample_rate,
            use_soundfile=args.use_soundfile,
            use_autocast=use_autocast,
            use_directml=args.device == "directml",
            chunk_duration=args.chunk_duration,
            mdxc_params={
                "segment_size": args.mdxc_segment_size,
                "override_model_segment_size": False,
                "batch_size": args.mdxc_batch_size,
                "overlap": args.mdxc_overlap,
                "pitch_shift": 0,
            },
        )
        separator.load_model(args.model)
    except Exception as exc:
        message = f"Unable to load audio-separator model {args.model}: {exc}"
        for item in items:
            emit({"type": "file-error", "inputIndex": item["inputIndex"], "message": message})
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 0

    for item in items:
        input_index = item["inputIndex"]
        work_dir = Path(item["workDir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            separator.output_dir = str(work_dir)
            separator.sample_rate = args.sample_rate
            if getattr(separator, "model_instance", None) is not None:
                separator.model_instance.output_dir = str(work_dir)
            emit({"type": "file-progress", "inputIndex": input_index, "message": "Separating VOCAL and BGM"})
            output_files = separator.separate(
                item["inputPath"],
                custom_output_names={
                    "Vocals": "vocals",
                    "Instrumental": "instrumental",
                },
            )
            emit({"type": "file-progress", "inputIndex": input_index, "message": "Writing separated WAV stems"})
            vocal = find_stem_path(output_files, work_dir, ("vocal",), args.output_format) or find_stem_file(
                work_dir,
                ("vocal",),
                args.output_format,
            )
            bgm = find_stem_path(
                output_files,
                work_dir,
                ("instrumental", "no_vocals", "no vocals", "inst"),
                args.output_format,
            ) or find_stem_file(
                work_dir,
                ("instrumental", "no_vocals", "no vocals", "inst"),
                args.output_format,
            )
            if not vocal or not bgm:
                raise RuntimeError("audio-separator did not produce both Vocals and Instrumental stems.")
            emit({"type": "file-complete", "inputIndex": input_index, "vocal": vocal, "bgm": bgm})
        except Exception as exc:
            emit({"type": "file-error", "inputIndex": input_index, "message": str(exc)})
            print(traceback.format_exc(), file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
