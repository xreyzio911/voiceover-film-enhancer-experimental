import assert from "node:assert/strict";
import test from "node:test";
import { CLEARVOICE_SE_REQUEST } from "./neuralRepairPolicy.ts";
import { NeuralRepairError, requestNeuralRepair, requestNeuralRepairHealth } from "./neuralRepairClient.ts";

test("neural repair client uploads raw wav bytes without multipart framing", async () => {
  const originalFetch = globalThis.fetch;
  const input = new Uint8Array([1, 2, 3, 4]);
  let captured: RequestInit | undefined;

  try {
    globalThis.fetch = (async (url: RequestInfo | URL, init?: RequestInit) => {
      assert.equal(url, "/api/neural-repair");
      captured = init;
      return new Response(new Uint8Array(64).fill(7), {
        status: 200,
        headers: {
          "content-type": "audio/wav",
          "x-vo-neural-report": encodeURIComponent(
            JSON.stringify({
              engine: "clearvoice",
              mode: "speech_enhancement",
              model: "MossFormer2_SE_48K",
              elapsedSeconds: 1.25,
            }),
          ),
        },
      });
    }) as typeof fetch;

    const result = await requestNeuralRepair(input, CLEARVOICE_SE_REQUEST, "lini.wav");
    const headers = captured?.headers as Record<string, string>;

    assert.equal(captured?.method, "POST");
    assert.equal(captured?.body, input.buffer);
    assert.equal(headers["Content-Type"], "audio/wav");
    assert.equal(headers["x-vo-neural-variant"], "clearvoice-se");
    assert.equal(headers["x-vo-neural-engine"], "clearvoice");
    assert.equal(headers["x-vo-neural-mode"], "speech_enhancement");
    assert.equal(headers["x-vo-neural-model"], "MossFormer2_SE_48K");
    assert.equal(headers["x-vo-neural-file-name"], "lini.wav");
    assert.equal(result.bytes.byteLength, 64);
    assert.equal(result.report?.elapsedSeconds, 1.25);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("neural repair client rejects headerless, zero-byte, or header-only success responses", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = (async () => new Response(new Uint8Array(0), { status: 200 })) as typeof fetch;

    await assert.rejects(
      () => requestNeuralRepair(new Uint8Array([1]), CLEARVOICE_SE_REQUEST, "lini.wav"),
      /invalid audio \(0 bytes\)/i,
    );

    globalThis.fetch = (async () => new Response(new Uint8Array(44), { status: 200 })) as typeof fetch;
    await assert.rejects(
      () => requestNeuralRepair(new Uint8Array([1]), CLEARVOICE_SE_REQUEST, "lini.wav"),
      /invalid audio \(44 bytes\)/i,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("neural repair client reports network fetch failures with a stable code", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = (async () => {
      throw new TypeError("Failed to fetch");
    }) as typeof fetch;

    await assert.rejects(
      () => requestNeuralRepair(new Uint8Array([1]), CLEARVOICE_SE_REQUEST, "lini.wav"),
      (error) => {
        assert.equal(error instanceof NeuralRepairError, true);
        assert.equal((error as NeuralRepairError).status, 0);
        assert.equal((error as NeuralRepairError).code, "network");
        assert.match((error as Error).message, /could not reach \/api\/neural-repair/i);
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("neural repair client self-tests the server worker once with cache disabled", async () => {
  const originalFetch = globalThis.fetch;
  let captured: RequestInit | undefined;

  try {
    globalThis.fetch = (async (url: RequestInfo | URL, init?: RequestInit) => {
      assert.equal(url, "/api/neural-repair?selfTest=1");
      captured = init;
      return Response.json({
        enabled: true,
        engines: ["clearvoice"],
        target: "remote",
        selfTest: { ok: true },
        exitCode: 0,
      });
    }) as typeof fetch;

    const result = await requestNeuralRepairHealth();

    assert.equal(captured?.method, "GET");
    assert.equal(captured?.cache, "no-store");
    assert.equal(result.enabled, true);
    assert.equal(result.target, "remote");
    assert.equal(result.selfTest?.ok, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("neural repair client turns self-test config failures into stable errors", async () => {
  const originalFetch = globalThis.fetch;

  try {
    globalThis.fetch = (async () =>
      Response.json(
        {
          error: "Neural repair self-test unavailable: configure VO_NEURAL_REPAIR_REMOTE_URL.",
          code: "config",
        },
        { status: 503 },
      )) as typeof fetch;

    await assert.rejects(
      () => requestNeuralRepairHealth(),
      (error) => {
        assert.equal(error instanceof NeuralRepairError, true);
        assert.equal((error as NeuralRepairError).status, 503);
        assert.equal((error as NeuralRepairError).code, "config");
        assert.match((error as Error).message, /VO_NEURAL_REPAIR_REMOTE_URL/i);
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
