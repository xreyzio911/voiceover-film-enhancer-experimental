#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from neural_repair_worker import configure_torch_threads, module_available, run_clearvoice

MIN_WAV_HEADER_BYTES = 44


def bool_env(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def number_header(headers, key, fallback):
    value = headers.get(key)
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def json_bytes(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def auth_token_from(headers):
    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return headers.get("x-vo-neural-worker-token", "").strip()


def expected_token():
    return os.environ.get("VO_NEURAL_REPAIR_WORKER_TOKEN") or os.environ.get("VO_NEURAL_REPAIR_REMOTE_TOKEN") or ""


def self_test_payload():
    return {
        "ok": True,
        "modules": {
            "clearvoice": module_available("clearvoice"),
            "soundfile": module_available("soundfile"),
            "librosa": module_available("librosa"),
            "torch": module_available("torch"),
        },
    }


def request_args(headers, input_path, output_path):
    speech_aware = bool_env(headers.get("x-vo-neural-speech-aware"), True)
    return Namespace(
        input=str(input_path),
        output=str(output_path),
        engine=headers.get("x-vo-neural-engine", "clearvoice"),
        mode=headers.get("x-vo-neural-mode", "speech_enhancement"),
        model=headers.get("x-vo-neural-model", "MossFormer2_SE_48K"),
        device=headers.get("x-vo-neural-device", os.environ.get("VO_NEURAL_REPAIR_DEVICE", "cpu")),
        clearvoice_chunk_seconds=number_header(headers, "x-vo-neural-clearvoice-chunk-seconds", 8.0),
        clearvoice_context_ms=number_header(headers, "x-vo-neural-clearvoice-context-ms", 250.0),
        clearvoice_silence_db=number_header(headers, "x-vo-neural-clearvoice-silence-db", -70.0),
        speech_aware=speech_aware,
        speech_aware_min_duration=number_header(headers, "x-vo-neural-speech-aware-min-seconds", 90.0),
        speech_aware_max_duty_pct=number_header(headers, "x-vo-neural-speech-aware-max-duty-pct", 62.0),
        speech_aware_pad_ms=number_header(headers, "x-vo-neural-speech-aware-pad-ms", 360.0),
        speech_aware_merge_gap_ms=number_header(headers, "x-vo-neural-speech-aware-merge-gap-ms", 900.0),
        speech_aware_min_span_ms=number_header(headers, "x-vo-neural-speech-aware-min-span-ms", 140.0),
        speech_aware_fade_ms=number_header(headers, "x-vo-neural-speech-aware-fade-ms", 80.0),
    )


class NeuralRepairHandler(BaseHTTPRequestHandler):
    server_version = "VONeuralRepairHTTP/1.0"

    def send_json(self, status, payload):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json(status, {"ok": False, "error": message})

    def is_authorized(self):
        token = expected_token()
        return not token or auth_token_from(self.headers) == token

    def do_GET(self):
        if not self.is_authorized():
            self.send_error_json(401, "Unauthorized")
            return
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if query.get("selfTest") == ["1"] or parsed.path in {"/health", "/self-test"}:
            self.send_json(200, self_test_payload())
            return
        self.send_json(200, {"ok": True, "service": "vo-neural-repair-worker"})

    def do_POST(self):
        if not self.is_authorized():
            self.send_error_json(401, "Unauthorized")
            return
        if self.headers.get("x-vo-neural-engine", "clearvoice") != "clearvoice":
            self.send_error_json(400, "Unsupported neural repair engine.")
            return
        try:
            content_length = int(self.headers.get("content-length", "0"))
        except ValueError:
            self.send_error_json(400, "Invalid content-length.")
            return
        max_bytes = int(float(os.environ.get("VO_NEURAL_REPAIR_MAX_AUDIO_MB", "350")) * 1024 * 1024)
        if content_length <= MIN_WAV_HEADER_BYTES:
            self.send_error_json(400, "Expected a WAV audio body.")
            return
        if content_length > max_bytes:
            self.send_error_json(413, f"Audio file too large ({content_length} bytes). Max is {max_bytes} bytes.")
            return

        with tempfile.TemporaryDirectory(prefix="vo-neural-repair-http-") as temp_root:
            temp_root_path = Path(temp_root)
            input_path = temp_root_path / "input.wav"
            output_path = temp_root_path / "output.wav"
            input_path.write_bytes(self.rfile.read(content_length))

            try:
                report = run_clearvoice(request_args(self.headers, input_path, output_path))
            except Exception as exc:
                self.send_error_json(502, str(exc))
                return

            output_bytes = output_path.read_bytes()
            if len(output_bytes) <= MIN_WAV_HEADER_BYTES:
                self.send_error_json(502, f"Neural worker produced invalid audio ({len(output_bytes)} bytes).")
                return

            report_header = quote(json.dumps(report, separators=(",", ":")), safe="")
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(output_bytes)))
            self.send_header("x-vo-neural-report", report_header)
            self.end_headers()
            self.wfile.write(output_bytes)

    def log_message(self, format, *args):
        if bool_env(os.environ.get("VO_NEURAL_REPAIR_HTTP_LOG"), False):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Persistent VO neural repair HTTP worker")
    parser.add_argument("--host", default=os.environ.get("VO_NEURAL_REPAIR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VO_NEURAL_REPAIR_PORT", "8787")))
    args = parser.parse_args()

    configure_torch_threads()
    server = HTTPServer((args.host, args.port), NeuralRepairHandler)
    print(json.dumps({"ok": True, "listening": f"http://{args.host}:{args.port}"}, separators=(",", ":")), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
