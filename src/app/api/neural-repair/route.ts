import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { NextResponse, type NextRequest } from "next/server";
import { getServerAuthSession } from "@/auth";
import { isAllowedEmail } from "@/lib/authAllowlist";
import { isLocalHost } from "@/lib/isLocalHost";
import {
  resolveNeuralRepairRequestInput,
  type NeuralRepairEngine,
  type NeuralRepairRequest,
  type NeuralRepairReport,
} from "@/lib/neuralRepairPolicy";
import { resolveNeuralRepairRuntimeTarget, type NeuralRepairRuntimeTarget } from "@/lib/neuralRepairRuntime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type NeuralRepairErrorCode = "config" | "auth" | "bad_request" | "worker";

const DEFAULT_MAX_AUDIO_BYTES = 350 * 1024 * 1024;
const DEFAULT_TIMEOUT_SECONDS = 45 * 60;
const WORKER_STDIO_LIMIT = 64 * 1024;
const MIN_WAV_HEADER_BYTES = 44;

const jsonError = (
  error: string,
  status: number,
  code: NeuralRepairErrorCode,
  extra?: Record<string, unknown>,
) => NextResponse.json({ error, code, ...extra }, { status });

const boolEnv = (value: string | undefined) => value === "1" || value === "true" || value === "on";

const maxAudioBytes = () => {
  const parsed = Number(process.env.VO_NEURAL_REPAIR_MAX_AUDIO_MB);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed * 1024 * 1024) : DEFAULT_MAX_AUDIO_BYTES;
};

const timeoutMs = () => {
  const parsed = Number(process.env.VO_NEURAL_REPAIR_TIMEOUT_SECONDS);
  const seconds = Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT_SECONDS;
  return Math.floor(seconds * 1000);
};

const numericEnv = (key: string, fallback: number) => {
  const parsed = Number(process.env[key]);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const isUploadFile = (value: FormDataEntryValue | null): value is File =>
  typeof value === "object" &&
  value !== null &&
  "arrayBuffer" in value &&
  typeof (value as File).arrayBuffer === "function";

const authorizeRequest = async (request: NextRequest) => {
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host");
  const localMode = isLocalHost(host);
  const session = localMode ? null : await getServerAuthSession();
  return localMode || isAllowedEmail(session?.user?.email);
};

const runtimeTarget = () =>
  resolveNeuralRepairRuntimeTarget({
    commandLine: process.env.VO_NEURAL_REPAIR_COMMAND,
    remoteUrl: process.env.VO_NEURAL_REPAIR_REMOTE_URL,
    remoteToken: process.env.VO_NEURAL_REPAIR_REMOTE_TOKEN,
    cwd: process.cwd(),
    platform: process.platform,
    isVercel: process.env.VERCEL === "1",
    exists: existsSync,
  });

const enabledEngines = (): NeuralRepairEngine[] =>
  (process.env.VO_NEURAL_REPAIR_ENGINE || "clearvoice")
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter((value): value is NeuralRepairEngine => value === "clearvoice");

const readTextField = (formData: FormData, key: string) => {
  const value = formData.get(key);
  return typeof value === "string" ? value : "";
};

const resolveMultipartRequest = (formData: FormData): NeuralRepairRequest | null => {
  return resolveNeuralRepairRequestInput({
    variant: readTextField(formData, "variant"),
    engine: readTextField(formData, "engine"),
    mode: readTextField(formData, "mode"),
    model: readTextField(formData, "model"),
  });
};

const resolveHeaderRequest = (request: NextRequest): NeuralRepairRequest | null => {
  const header = (key: string) => request.headers.get(`x-vo-neural-${key}`);
  return resolveNeuralRepairRequestInput({
    variant: header("variant"),
    engine: header("engine"),
    mode: header("mode"),
    model: header("model"),
  });
};

const requestContentLength = (request: NextRequest) => {
  const raw = request.headers.get("content-length");
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
};

const isBinaryAudioContentType = (contentType: string) => {
  const normalized = contentType.split(";")[0].trim().toLowerCase();
  return (
    normalized === "application/octet-stream" ||
    normalized === "audio/wav" ||
    normalized === "audio/x-wav" ||
    normalized === "audio/wave" ||
    normalized === "audio/vnd.wave"
  );
};

type UploadPayload = {
  audioBytes: Buffer;
  repairRequest: NeuralRepairRequest | null;
};

const readUploadPayload = async (request: NextRequest): Promise<UploadPayload | NextResponse> => {
  const contentType = request.headers.get("content-type") ?? "";
  if (contentType.toLowerCase().startsWith("multipart/form-data")) {
    let formData: FormData;
    try {
      formData = await request.formData();
    } catch {
      return jsonError("Expected multipart/form-data with an audio file.", 400, "bad_request");
    }

    const entry = formData.get("audio");
    if (!isUploadFile(entry)) {
      return jsonError("Missing audio file.", 400, "bad_request");
    }
    return {
      audioBytes: Buffer.from(await entry.arrayBuffer()),
      repairRequest: resolveMultipartRequest(formData),
    };
  }

  if (!isBinaryAudioContentType(contentType)) {
    return jsonError("Expected audio/wav binary body or multipart/form-data with an audio file.", 400, "bad_request");
  }

  return {
    audioBytes: Buffer.from(await request.arrayBuffer()),
    repairRequest: resolveHeaderRequest(request),
  };
};

const runLocalWorker = async (
  target: Extract<NeuralRepairRuntimeTarget, { kind: "local" }>,
  args: string[],
  timeout: number,
) => {
  return await new Promise<{ exitCode: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(target.command, [...target.args, ...args], {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`neural repair worker timed out after ${(timeout / 1000).toFixed(0)}s`));
    }, timeout);
    child.stdout.on("data", (chunk: Buffer) => {
      stdout = (stdout + chunk.toString("utf8")).slice(-WORKER_STDIO_LIMIT);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString("utf8")).slice(-WORKER_STDIO_LIMIT);
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      const nodeError = error as NodeJS.ErrnoException;
      reject(
        new Error(
          nodeError.code === "ENOENT"
            ? `neural repair worker command not found: ${target.command}. Set VO_NEURAL_REPAIR_COMMAND to a Python runtime with ClearVoice installed, or provision .venv-neural for this deployment.`
            : error.message,
        ),
      );
    });
    child.on("close", (exitCode) => {
      clearTimeout(timer);
      resolve({ exitCode, stdout, stderr });
    });
  });
};

const remoteAuthHeaders = (target: Extract<NeuralRepairRuntimeTarget, { kind: "remote" }>) => {
  const headers: Record<string, string> = {};
  if (target.token) {
    headers.Authorization = `Bearer ${target.token}`;
    headers["x-vo-neural-worker-token"] = target.token;
  }
  return headers;
};

const remoteSelfTestUrl = (target: Extract<NeuralRepairRuntimeTarget, { kind: "remote" }>) => {
  const url = new URL(target.url);
  url.searchParams.set("selfTest", "1");
  return url;
};

const readRemoteError = async (response: Response) => {
  const text = (await response.text()).trim();
  if (!text) return `HTTP ${response.status}`;
  try {
    const parsed = JSON.parse(text) as { error?: unknown };
    if (typeof parsed.error === "string") return parsed.error;
  } catch {
    // Keep the raw text below.
  }
  return text.slice(0, 1000);
};

const parseReportHeader = (value: string | null): NeuralRepairReport | null => {
  if (!value) return null;
  try {
    const parsed = JSON.parse(decodeURIComponent(value)) as NeuralRepairReport;
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
};

const runRemoteSelfTest = async (
  target: Extract<NeuralRepairRuntimeTarget, { kind: "remote" }>,
  timeout: number,
) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(remoteSelfTestUrl(target), {
      headers: remoteAuthHeaders(target),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`remote neural worker self-test failed: ${await readRemoteError(response)}`);
    }
    return (await response.json()) as NeuralRepairReport | Record<string, unknown>;
  } finally {
    clearTimeout(timer);
  }
};

const remoteRepairHeaders = (
  target: Extract<NeuralRepairRuntimeTarget, { kind: "remote" }>,
  repairRequest: NeuralRepairRequest,
) => ({
  ...remoteAuthHeaders(target),
  "Content-Type": "audio/wav",
  "x-vo-neural-variant": repairRequest.variant,
  "x-vo-neural-engine": repairRequest.engine,
  "x-vo-neural-mode": repairRequest.mode,
  "x-vo-neural-model": repairRequest.model,
  "x-vo-neural-device": process.env.VO_NEURAL_REPAIR_DEVICE || "cpu",
  "x-vo-neural-speech-aware": boolEnv(process.env.VO_NEURAL_REPAIR_SPEECH_AWARE ?? "on") ? "on" : "off",
  "x-vo-neural-speech-aware-min-seconds": String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_MIN_SECONDS", 90)),
  "x-vo-neural-speech-aware-max-duty-pct": String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_MAX_DUTY_PCT", 62)),
  "x-vo-neural-speech-aware-pad-ms": String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_PAD_MS", 360)),
  "x-vo-neural-speech-aware-merge-gap-ms": String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_MERGE_GAP_MS", 900)),
  "x-vo-neural-speech-aware-fade-ms": String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_FADE_MS", 80)),
});

const runRemoteRepair = async (
  target: Extract<NeuralRepairRuntimeTarget, { kind: "remote" }>,
  audioBytes: Buffer,
  repairRequest: NeuralRepairRequest,
  timeout: number,
) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(target.url, {
      method: "POST",
      headers: remoteRepairHeaders(target, repairRequest),
      body: new Uint8Array(audioBytes),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`remote neural worker failed: ${await readRemoteError(response)}`);
    }
    const outputBytes = Buffer.from(await response.arrayBuffer());
    if (outputBytes.byteLength <= MIN_WAV_HEADER_BYTES) {
      throw new Error(`remote neural worker produced invalid audio (${outputBytes.byteLength} bytes).`);
    }
    return {
      outputBytes,
      report: parseReportHeader(response.headers.get("x-vo-neural-report")),
    };
  } finally {
    clearTimeout(timer);
  }
};

const parseWorkerReport = (stdout: string, stderr: string): NeuralRepairReport | null => {
  const lines = `${stdout}\n${stderr}`
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const parsed = JSON.parse(lines[index]) as NeuralRepairReport;
      if (parsed && typeof parsed === "object") return parsed;
    } catch {
      // Keep searching earlier lines; model libraries can print logs.
    }
  }
  return null;
};

const workerPath = () => path.join(process.cwd(), "scripts", "neural_repair_worker.py");

export async function GET(request: NextRequest) {
  if (!(await authorizeRequest(request))) {
    return jsonError("Unauthorized", 401, "auth");
  }

  const enabled = boolEnv(process.env.VO_NEURAL_REPAIR_ENABLED);
  const engines = enabledEngines();
  const target = runtimeTarget();
  if (request.nextUrl.searchParams.get("selfTest") === "1") {
    try {
      if (target.kind === "unavailable") {
        return jsonError(`Neural repair self-test unavailable: ${target.reason}`, 503, "config");
      }
      if (target.kind === "remote") {
        const report = await runRemoteSelfTest(target, 30_000);
        return NextResponse.json(
          {
            enabled,
            engines,
            target: target.kind,
            selfTest: report,
            exitCode: 0,
          },
          { headers: { "Cache-Control": "no-store" } },
        );
      }
      const result = await runLocalWorker(target, [workerPath(), "--self-test"], 30_000);
      const report = parseWorkerReport(result.stdout, result.stderr);
      return NextResponse.json(
        {
          enabled,
          engines,
          target: target.kind,
          command: target.command,
          commandSource: target.source,
          selfTest: report,
          exitCode: result.exitCode,
        },
        { headers: { "Cache-Control": "no-store" } },
      );
    } catch (error) {
      return jsonError(
        `Neural repair self-test failed: ${error instanceof Error ? error.message : String(error)}`,
        503,
        "worker",
      );
    }
  }

  return NextResponse.json(
    {
      enabled,
      engines,
      target: target.kind,
      ...(target.kind === "local" ? { command: target.command, commandSource: target.source } : {}),
      ...(target.kind === "unavailable" ? { unavailableReason: target.reason } : {}),
      maxAudioBytes: maxAudioBytes(),
      timeoutSeconds: timeoutMs() / 1000,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}

export async function POST(request: NextRequest) {
  if (!(await authorizeRequest(request))) {
    return jsonError("Unauthorized", 401, "auth");
  }
  if (!boolEnv(process.env.VO_NEURAL_REPAIR_ENABLED)) {
    return jsonError("Neural speech enhancement is disabled. Set VO_NEURAL_REPAIR_ENABLED=on on the server.", 503, "config");
  }
  const target = runtimeTarget();
  if (target.kind === "unavailable") {
    return jsonError(`Neural speech enhancement is not available on this server: ${target.reason}`, 503, "config");
  }

  const maxBytes = maxAudioBytes();
  const contentLength = requestContentLength(request);
  if (contentLength !== null && contentLength > maxBytes) {
    return jsonError(`Audio file too large (${contentLength} bytes). Max is ${maxBytes} bytes.`, 413, "bad_request");
  }

  const upload = await readUploadPayload(request);
  if (upload instanceof NextResponse) {
    return upload;
  }

  const { audioBytes, repairRequest } = upload;
  if (audioBytes.byteLength === 0) {
    return jsonError("Empty audio file.", 400, "bad_request");
  }
  if (audioBytes.byteLength > maxBytes) {
    return jsonError(
      `Audio file too large (${audioBytes.byteLength} bytes). Max is ${maxBytes} bytes.`,
      413,
      "bad_request",
    );
  }

  if (!repairRequest) {
    return jsonError("Unsupported neural speech enhancement request.", 400, "bad_request");
  }
  if (!enabledEngines().includes(repairRequest.engine)) {
    return jsonError(`Neural repair engine is not enabled: ${repairRequest.engine}`, 400, "bad_request");
  }

  if (target.kind === "remote") {
    try {
      const result = await runRemoteRepair(target, audioBytes, repairRequest, timeoutMs());
      const headerReport = result.report ? encodeURIComponent(JSON.stringify(result.report)) : "";
      return new NextResponse(result.outputBytes, {
        status: 200,
        headers: {
          "Content-Type": "audio/wav",
          "Cache-Control": "no-store",
          ...(headerReport ? { "x-vo-neural-report": headerReport } : {}),
        },
      });
    } catch (error) {
      return jsonError(
        `Neural repair failed: ${error instanceof Error ? error.message : String(error)}`,
        502,
        "worker",
      );
    }
  }

  const tempRoot = await mkdtemp(path.join(tmpdir(), "vo-neural-repair-"));
  const inputPath = path.join(tempRoot, "input.wav");
  const outputPath = path.join(tempRoot, "output.wav");
  try {
    await writeFile(inputPath, audioBytes);
    const result = await runLocalWorker(
      target,
      [
        workerPath(),
        "--input",
        inputPath,
        "--output",
        outputPath,
        "--engine",
        repairRequest.engine,
        "--mode",
        repairRequest.mode,
        "--model",
        repairRequest.model,
        "--device",
        process.env.VO_NEURAL_REPAIR_DEVICE || "cpu",
        ...(boolEnv(process.env.VO_NEURAL_REPAIR_SPEECH_AWARE ?? "on")
          ? [
              "--speech-aware",
              "--speech-aware-min-duration",
              String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_MIN_SECONDS", 90)),
              "--speech-aware-max-duty-pct",
              String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_MAX_DUTY_PCT", 62)),
              "--speech-aware-pad-ms",
              String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_PAD_MS", 360)),
              "--speech-aware-merge-gap-ms",
              String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_MERGE_GAP_MS", 900)),
              "--speech-aware-fade-ms",
              String(numericEnv("VO_NEURAL_REPAIR_SPEECH_AWARE_FADE_MS", 80)),
            ]
          : []),
      ],
      timeoutMs(),
    );
    const report = parseWorkerReport(result.stdout, result.stderr);
    if (result.exitCode !== 0) {
      const detail =
        report && "error" in report && typeof (report as { error?: unknown }).error === "string"
          ? (report as { error: string }).error
          : (result.stderr || result.stdout || "Worker exited without a diagnostic.").slice(-1000);
      return jsonError(`Neural repair worker failed: ${detail}`, 502, "worker");
    }

    const outputBytes = await readFile(outputPath);
    if (outputBytes.byteLength <= MIN_WAV_HEADER_BYTES) {
      return jsonError(
        `Neural repair worker produced invalid audio (${outputBytes.byteLength} bytes).`,
        502,
        "worker",
      );
    }
    const headerReport = report ? encodeURIComponent(JSON.stringify(report)) : "";
    return new NextResponse(outputBytes, {
      status: 200,
      headers: {
        "Content-Type": "audio/wav",
        "Cache-Control": "no-store",
        ...(headerReport ? { "x-vo-neural-report": headerReport } : {}),
      },
    });
  } catch (error) {
    return jsonError(
      `Neural repair failed: ${error instanceof Error ? error.message : String(error)}`,
      502,
      "worker",
    );
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
}
