import assert from "node:assert/strict";
import test from "node:test";
import { planBatchLoudnessAlignment } from "./batchLoudnessAlign.ts";

test("batch loudness alignment anchors to the median and clamps offsets", () => {
  const result = planBatchLoudnessAlignment([
    { id: "quiet", inputI: -25.2 },
    { id: "anchor", inputI: -22.8 },
    { id: "loud", inputI: -20.2 },
  ]);

  assert.equal(result.anchorLufs, -22.8);
  assert.deepEqual(
    result.plans.map((plan) => ({ id: plan.id, offsetDb: plan.offsetDb })),
    [
      { id: "quiet", offsetDb: 2 },
      { id: "anchor", offsetDb: 0 },
      { id: "loud", offsetDb: -2 },
    ],
  );
});

test("batch loudness alignment no-ops single-file and in-threshold batches", () => {
  assert.deepEqual(planBatchLoudnessAlignment([{ id: "solo", inputI: -22.4 }]).plans, [
    { id: "solo", inputI: -22.4, offsetDb: 0, shouldAlign: false },
  ]);

  const result = planBatchLoudnessAlignment([
    { id: "a", inputI: -22.7 },
    { id: "b", inputI: -22.3 },
    { id: "missing", inputI: null },
  ]);

  assert.equal(result.anchorLufs, -22.5);
  assert.ok(result.plans.every((plan) => !plan.shouldAlign));
});
