import { NextResponse, type NextRequest } from "next/server";
import { getServerAuthSession } from "@/auth";
import {
  DEFAULT_GEMINI_AUDIO_REVIEW_MODEL,
  buildGeminiAudioReviewRequest,
  normalizeAudioReviewRequest,
  parseGeminiAudioReviewText,
} from "@/lib/aiAudioReview";
import { isAllowedEmail } from "@/lib/authAllowlist";
import { isLocalHost } from "@/lib/isLocalHost";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type AudioReviewErrorCode = "auth" | "bad_request" | "config" | "rate_limit" | "provider";

const RATE_LIMIT_WINDOW_MS = 10 * 60 * 1000;
const RATE_LIMIT_MAX = 18;
const DEFAULT_TIMEOUT_MS = 45_000;

const rateLimitState = new Map<string, { count: number; resetAtMs: number }>();

const jsonError = (error: string, status: number, code: AudioReviewErrorCode) =>
  NextResponse.json({ error, code }, { status, headers: { "Cache-Control": "no-store" } });

const authorizeRequest = async (request: NextRequest) => {
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host");
  const localMode = isLocalHost(host);
  const session = localMode ? null : await getServerAuthSession();
  return localMode || isAllowedEmail(session?.user?.email);
};

const clientKey = (request: NextRequest) => {
  const forwardedFor = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host") ?? "unknown";
  return `${forwardedFor || host}:audio-review`;
};

const consumeRateLimit = (key: string) => {
  const now = Date.now();
  const current = rateLimitState.get(key);
  if (!current || current.resetAtMs <= now) {
    rateLimitState.set(key, { count: 1, resetAtMs: now + RATE_LIMIT_WINDOW_MS });
    return true;
  }
  if (current.count >= RATE_LIMIT_MAX) {
    return false;
  }
  rateLimitState.set(key, { ...current, count: current.count + 1 });
  return true;
};

const timeoutMs = () => {
  const parsed = Number(process.env.GEMINI_AUDIO_REVIEW_TIMEOUT_MS);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : DEFAULT_TIMEOUT_MS;
};

const reviewModel = () =>
  process.env.GEMINI_AUDIO_REVIEW_MODEL ||
  process.env.GEMINI_FLASH_LITE_MODEL ||
  DEFAULT_GEMINI_AUDIO_REVIEW_MODEL;

const geminiApiKey = () => process.env.GEMINI_API_KEY || process.env.GOOGLE_GEMINI_API_KEY || "";

const extractGeminiText = (payload: unknown) => {
  const candidates = (payload as { candidates?: Array<{ content?: { parts?: Array<{ text?: unknown }> } }> })
    ?.candidates;
  if (!Array.isArray(candidates)) return "";
  return candidates
    .flatMap((candidate) => candidate.content?.parts ?? [])
    .map((part) => (typeof part.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n")
    .trim();
};

const providerErrorMessage = async (response: Response) => {
  const text = await response.text().catch(() => "");
  if (!text) return `Gemini review failed (HTTP ${response.status}).`;
  try {
    const parsed = JSON.parse(text) as { error?: { message?: unknown } };
    const message = parsed.error && typeof parsed.error.message === "string" ? parsed.error.message : "";
    return message ? `Gemini review failed: ${message}` : `Gemini review failed (HTTP ${response.status}).`;
  } catch {
    return `Gemini review failed (HTTP ${response.status}).`;
  }
};

export async function POST(request: NextRequest) {
  if (!(await authorizeRequest(request))) {
    return jsonError("Unauthorized", 401, "auth");
  }
  if (!consumeRateLimit(clientKey(request))) {
    return jsonError("Too many AI review requests. Wait a few minutes and try again.", 429, "rate_limit");
  }

  const apiKey = geminiApiKey();
  if (!apiKey) {
    return jsonError("Gemini API key is not configured.", 503, "config");
  }

  let rawBody: unknown;
  try {
    rawBody = await request.json();
  } catch {
    return jsonError("Expected a JSON audio review payload.", 400, "bad_request");
  }

  const payload = normalizeAudioReviewRequest(rawBody);
  if (!payload) {
    return jsonError("Invalid audio review payload.", 400, "bad_request");
  }

  const model = reviewModel();
  const geminiRequest = buildGeminiAudioReviewRequest(payload, model);
  const modelPath = model.startsWith("models/") ? model.slice("models/".length) : model;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs());

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelPath)}:generateContent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
        },
        body: JSON.stringify(geminiRequest.body),
        signal: controller.signal,
      },
    );

    if (!response.ok) {
      return jsonError(await providerErrorMessage(response), response.status >= 500 ? 502 : response.status, "provider");
    }

    const responseJson = (await response.json()) as unknown;
    const text = extractGeminiText(responseJson);
    if (!text) {
      return jsonError("Gemini review returned an empty response.", 502, "provider");
    }

    const review = parseGeminiAudioReviewText(text);
    const reviewedKeys = new Set(
      review.perFileProfiles.flatMap((file) => [file.base, file.fileName].filter(Boolean)),
    );
    const missingFiles = payload.files.filter(
      (file) => !reviewedKeys.has(file.base) && !reviewedKeys.has(file.fileName),
    );
    if (missingFiles.length > 0) {
      return jsonError(
        `Gemini review did not return per-file profiles for: ${missingFiles
          .slice(0, 4)
          .map((file) => file.fileName)
          .join(", ")}${missingFiles.length > 4 ? ", ..." : ""}`,
        502,
        "provider",
      );
    }
    return NextResponse.json(
      { model: geminiRequest.model, review },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    const message =
      error instanceof Error && error.name === "AbortError"
        ? "Gemini review timed out."
        : `Gemini review failed: ${error instanceof Error ? error.message : String(error)}`;
    return jsonError(message, 502, "provider");
  } finally {
    clearTimeout(timer);
  }
}
