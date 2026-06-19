import assert from "node:assert/strict";
import test from "node:test";
import {
  NEURAL_REPAIR_SAFE_FUNCTION_BODY_BYTES,
  planNeuralRepairTransport,
} from "./neuralRepairChunkPolicy.ts";

test("keeps small neural repair uploads on the direct function path", () => {
  const plan = planNeuralRepairTransport({
    inputBytes: 2 * 1024 * 1024,
    durationSeconds: 11,
  });

  assert.equal(plan.strategy, "direct");
  assert.equal(plan.chunks.length, 1);
  assert.equal(plan.chunks[0].startSec, 0);
  assert.equal(plan.chunks[0].durationSec, 11);
});

test("splits a deployed Vercel neural repair upload into safe request-sized chunks", () => {
  const plan = planNeuralRepairTransport({
    inputBytes: Math.round(199.2 * 1024 * 1024),
    durationSeconds: 1087.982,
  });

  assert.equal(plan.strategy, "chunked");
  assert.ok(plan.chunks.length >= 50);
  assert.equal(plan.chunks[0].startSec, 0);
  assert.equal(plan.chunks.at(-1)?.endSec, 1087.982);
  assert.ok(plan.chunkDurationSeconds <= 20);
  assert.ok(
    plan.chunks.every((chunk) => chunk.estimatedBytes <= NEURAL_REPAIR_SAFE_FUNCTION_BODY_BYTES),
    "every planned chunk should stay under the Vercel-safe function body budget",
  );
});

test("rejects chunk planning when duration is missing for an oversized neural upload", () => {
  assert.throws(
    () =>
      planNeuralRepairTransport({
        inputBytes: 199 * 1024 * 1024,
        durationSeconds: null,
      }),
    /duration is required/i,
  );
});
