import assert from "node:assert/strict";
import test from "node:test";
import {
  VO_ZIP_CHUNK_TARGET_BYTES,
  estimateVoZipBytes,
  planVoZipExportParts,
  shouldChunkVoZip,
} from "./downloadPolicy.ts";
import { getDownloadQueueDelayMs, getDownloadUrlRetainMs, sanitizeDownloadFileName } from "./downloadBlob.ts";

const MB = 1024 * 1024;

test("VO ZIP policy keeps small batches as one archive", () => {
  const outputs = [
    { name: "line_01_A85.wav", size: 12 * MB },
    { name: "line_02_A85.wav", size: 14 * MB },
    { name: "line_03_A85.wav", size: 10 * MB },
  ];

  assert.equal(shouldChunkVoZip(outputs), false);
  const parts = planVoZipExportParts(outputs);
  assert.equal(parts.length, 1);
  assert.equal(parts[0].outputs.length, 3);
});

test("VO ZIP policy chunks 20 long WAV outputs into safer parts", () => {
  const outputs = Array.from({ length: 22 }, (_, index) => ({
    name: `long_take_${String(index + 1).padStart(2, "0")}_A85.wav`,
    size: 72 * MB,
  }));

  assert.equal(shouldChunkVoZip(outputs), true);
  const parts = planVoZipExportParts(outputs);

  assert.ok(parts.length >= 4);
  assert.equal(parts.flatMap((part) => part.outputs).length, outputs.length);
  for (const part of parts) {
    assert.equal(part.totalParts, parts.length);
    assert.ok(part.outputs.length <= 8);
    assert.ok(part.estimatedBytes <= VO_ZIP_CHUNK_TARGET_BYTES + 72 * MB);
  }
});

test("download helper keeps large blob URLs alive much longer than small files", () => {
  const smallRetain = getDownloadUrlRetainMs(10 * MB);
  const largeRetain = getDownloadUrlRetainMs(180 * MB);
  const hugeRetain = getDownloadUrlRetainMs(900 * MB);

  assert.ok(largeRetain > smallRetain);
  assert.ok(hugeRetain > largeRetain);
  assert.ok(getDownloadQueueDelayMs(900 * MB) > getDownloadQueueDelayMs(10 * MB));
});

test("download filename sanitizer preserves basename and removes unsafe characters", () => {
  assert.equal(sanitizeDownloadFileName('C:\\temp\\EP:89*Lucas?.wav'), "EP_89_Lucas_.wav");
  assert.match(sanitizeDownloadFileName("   "), /^download_/);
});

test("ZIP byte estimate includes all output payloads", () => {
  const outputs = [
    { name: "a.wav", size: 5 * MB },
    { name: "b.wav", size: 6 * MB },
  ];

  assert.ok(estimateVoZipBytes(outputs) > 11 * MB);
});
