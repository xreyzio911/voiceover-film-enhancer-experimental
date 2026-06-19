import type { NeuralRepairReport, NeuralRepairRequest } from "./neuralRepairPolicy";

export class NeuralRepairError extends Error {
  status: number;
  code: string | null;

  constructor(message: string, status: number, code: string | null = null) {
    super(message);
    this.name = "NeuralRepairError";
    this.status = status;
    this.code = code;
  }
}

export type NeuralRepairResult = {
  bytes: Uint8Array;
  report: NeuralRepairReport | null;
  contentType: string;
};

export type NeuralRepairHealth = {
  enabled: boolean;
  engines: string[];
  target: string | null;
  selfTest: { ok?: boolean; [key: string]: unknown } | null;
  exitCode: number | null;
};

const MIN_WAV_HEADER_BYTES = 44;

const parseReportHeader = (value: string | null): NeuralRepairReport | null => {
  if (!value) return null;
  try {
    return JSON.parse(decodeURIComponent(value)) as NeuralRepairReport;
  } catch {
    return null;
  }
};

const responseErrorMessage = async (response: Response) => {
  const text = await response.text().catch(() => "");
  if (!text) return `Neural repair failed (HTTP ${response.status}).`;
  try {
    const payload = JSON.parse(text) as { error?: unknown; code?: unknown };
    return {
      message:
        typeof payload.error === "string"
          ? payload.error
          : `Neural repair failed (HTTP ${response.status}).`,
      code: typeof payload.code === "string" ? payload.code : null,
    };
  } catch {
    return text.slice(0, 300);
  }
};

const neuralRepairHeaders = (request: NeuralRepairRequest, fileName: string, contentType: string) => ({
  "Content-Type": contentType || "audio/wav",
  "x-vo-neural-variant": request.variant,
  "x-vo-neural-engine": request.engine,
  "x-vo-neural-mode": request.mode,
  "x-vo-neural-model": request.model,
  "x-vo-neural-file-name": encodeURIComponent(fileName),
});

const audioBytesBody = (audio: Uint8Array): ArrayBuffer => {
  const { buffer, byteOffset, byteLength } = audio;
  if (buffer instanceof ArrayBuffer && byteOffset === 0 && byteLength === buffer.byteLength) {
    return buffer;
  }
  const copy = new ArrayBuffer(byteLength);
  new Uint8Array(copy).set(audio);
  return copy;
};

const stringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
};

export const requestNeuralRepairHealth = async (): Promise<NeuralRepairHealth> => {
  let response: Response;
  try {
    response = await fetch("/api/neural-repair?selfTest=1", {
      method: "GET",
      cache: "no-store",
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new NeuralRepairError(
      `Neural repair self-test could not reach /api/neural-repair (${detail}).`,
      0,
      "network",
    );
  }

  if (!response.ok) {
    const error = await responseErrorMessage(response);
    if (typeof error === "string") {
      throw new NeuralRepairError(error, response.status);
    }
    throw new NeuralRepairError(error.message, response.status, error.code);
  }

  const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null;
  if (!payload || typeof payload !== "object") {
    throw new NeuralRepairError("Neural repair self-test returned invalid JSON.", response.status, "invalid-json");
  }

  const selfTest =
    payload.selfTest && typeof payload.selfTest === "object"
      ? (payload.selfTest as { ok?: boolean; [key: string]: unknown })
      : null;

  return {
    enabled: payload.enabled === true,
    engines: stringArray(payload.engines),
    target: typeof payload.target === "string" ? payload.target : null,
    selfTest,
    exitCode: typeof payload.exitCode === "number" ? payload.exitCode : null,
  };
};

export const requestNeuralRepair = async (
  audio: Blob | Uint8Array,
  request: NeuralRepairRequest,
  fileName = "input.wav",
): Promise<NeuralRepairResult> => {
  const body = audio instanceof Blob ? audio : audioBytesBody(audio);
  const contentType = audio instanceof Blob && audio.type ? audio.type : "audio/wav";

  let response: Response;
  try {
    response = await fetch("/api/neural-repair", {
      method: "POST",
      headers: neuralRepairHeaders(request, fileName, contentType),
      body,
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new NeuralRepairError(
      `Neural repair request could not reach /api/neural-repair (${detail}).`,
      0,
      "network",
    );
  }

  if (!response.ok) {
    const error = await responseErrorMessage(response);
    if (typeof error === "string") {
      throw new NeuralRepairError(error, response.status);
    }
    throw new NeuralRepairError(error.message, response.status, error.code);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength <= MIN_WAV_HEADER_BYTES) {
    throw new NeuralRepairError(
      `Neural repair returned invalid audio (${bytes.byteLength} bytes).`,
      response.status,
    );
  }

  return {
    bytes,
    report: parseReportHeader(response.headers.get("x-vo-neural-report")),
    contentType: response.headers.get("content-type") ?? "audio/wav",
  };
};
