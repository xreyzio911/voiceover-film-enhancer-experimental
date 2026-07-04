import assert from "node:assert/strict";
import test from "node:test";
import {
  CINEMATIC_VO_REFERENCE_DB,
  computeToneMatchDeltaDb,
  resolveDeEsserBands,
  resolveSpectrumFrameBudget,
} from "./spectrum.ts";

test("house-tone blend pulls an all-boomy batch toward cinematic VO shape", () => {
  const fileDb = [-14, -13, -14, -18, -24, -27, -31, -35];
  const boomyBatchReference = [-15, -14, -15, -18, -24, -28, -32, -36];
  const delta = computeToneMatchDeltaDb(fileDb, boomyBatchReference, {
    houseBlend: 0.35,
    houseReferenceDb: CINEMATIC_VO_REFERENCE_DB,
  });

  assert.ok(delta[0] < -1.2, `60 Hz should be pulled down, got ${delta[0].toFixed(2)} dB`);
  assert.ok(delta[1] < -0.6, `120 Hz should be controlled, got ${delta[1].toFixed(2)} dB`);
  assert.ok(delta[6] > 1.2, `4 kHz presence should be restored, got ${delta[6].toFixed(2)} dB`);
  assert.ok(Math.max(...delta.map(Math.abs)) <= 3);
});

test("tone-match priority bands can use the wider 3 dB cap while other bands stay at 2.5 dB", () => {
  const delta = computeToneMatchDeltaDb(
    [-8, -18, -18, -18, -18, -18, -32, -18],
    [-18, -18, -18, -18, -18, -18, -18, -18],
    { maxDb: 2.5, priorityMaxDb: 3, priorityBandCount: 2 },
  );

  assert.equal(delta[0], -3);
  assert.equal(delta[6], 3);
  assert.ok(delta.every((value, index) => index === 0 || index === 6 || Math.abs(value) <= 2.5));
});

test("adaptive de-esser placement follows measured sibilance center", () => {
  assert.deepEqual(resolveDeEsserBands([-40, -40, -35, -34, -32, -31, -22, -31]), {
    mainHz: 5800,
    secondaryHz: 8200,
  });
  assert.deepEqual(resolveDeEsserBands([-40, -40, -35, -34, -32, -31, -31, -21]), {
    mainHz: 7200,
    secondaryHz: 9800,
  });
  assert.deepEqual(resolveDeEsserBands([-40, -40, -35, -34, -32, -31, -25, -24]), {
    mainHz: 6500,
    secondaryHz: 9000,
  });
});

test("spectrum analysis caps long-file frame visits", () => {
  const sampleRate = 16000;
  const thirtyMinutes = sampleRate * 60 * 30;
  const budget = resolveSpectrumFrameBudget(thirtyMinutes, sampleRate, { maxFrames: 1600 });

  assert.ok(budget.totalFrames > 80000, `fixture should represent a long file, got ${budget.totalFrames} frames`);
  assert.ok(budget.frameStride > 1, `long files should stride frames, got stride ${budget.frameStride}`);
  assert.ok(budget.framesToVisit <= 1600, `frame visits should stay capped, got ${budget.framesToVisit}`);
});
