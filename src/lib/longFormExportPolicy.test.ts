import assert from "node:assert/strict";
import test from "node:test";
import {
  LONG_FORM_EXPORT_CHUNK_SECONDS,
  formatLongFormPartTag,
  planLongFormChunks,
  shouldUseLongFormSafeMode,
} from "./longFormExportPolicy.ts";

test("long-form safe mode starts only beyond the planner budget", () => {
  assert.equal(shouldUseLongFormSafeMode(4800), false);
  assert.equal(shouldUseLongFormSafeMode(4800.1), true);
  assert.equal(shouldUseLongFormSafeMode(null, 7800), true);
});

test("two-hour recordings are split into bounded sequential export chunks", () => {
  const chunks = planLongFormChunks(7883);

  assert.equal(chunks.length, Math.ceil(7883 / LONG_FORM_EXPORT_CHUNK_SECONDS));
  assert.equal(chunks[0].startSec, 0);
  assert.equal(chunks[0].durationSec, LONG_FORM_EXPORT_CHUNK_SECONDS);
  assert.ok(chunks.at(-1)!.durationSec <= LONG_FORM_EXPORT_CHUNK_SECONDS);
  assert.equal(
    Math.round(chunks.reduce((total, chunk) => total + chunk.durationSec, 0)),
    7883,
  );
});

test("long-form part tags sort correctly in file managers", () => {
  const chunks = planLongFormChunks(7883);

  assert.equal(formatLongFormPartTag(chunks[0]), "part01-of-09");
  assert.equal(formatLongFormPartTag(chunks[8]), "part09-of-09");
});

test("long-form chunks prefer nearby silence boundaries", () => {
  const chunks = planLongFormChunks(1900, 900, [{ startSec: 884, endSec: 886 }]);

  assert.equal(chunks.length, 3);
  assert.equal(chunks[1].startSec, 885);
  assert.equal(Math.round(chunks.reduce((total, chunk) => total + chunk.durationSec, 0)), 1900);
});
