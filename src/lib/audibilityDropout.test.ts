import assert from "node:assert/strict";
import test from "node:test";
import { detectAudibilityDropouts } from "./audibilityDropout.ts";

const makeFrames = (length: number, db: number) => new Array<number>(length).fill(db);

test("detects rendered speech that collapses to inaudible digital silence", () => {
  const sourceFrameDb = makeFrames(260, -90);
  const renderedFrameDb = makeFrames(260, -92);
  for (let frame = 40; frame < 170; frame += 1) {
    sourceFrameDb[frame] = -28;
    renderedFrameDb[frame] = -21;
  }
  for (let frame = 94; frame < 122; frame += 1) {
    renderedFrameDb[frame] = -140;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.equal(report.clusterCount, 1);
  assert.ok(report.badSeconds >= 0.5, `expected at least 0.5s of collapsed speech, got ${report.badSeconds}s`);
  assert.ok(report.worstDropDb <= -90, `expected severe drop, got ${report.worstDropDb} dB`);
});

test("does not flag real pauses or normal mastering gain differences", () => {
  const sourceFrameDb = makeFrames(220, -92);
  const renderedFrameDb = makeFrames(220, -120);
  for (let frame = 30; frame < 92; frame += 1) {
    sourceFrameDb[frame] = -30;
    renderedFrameDb[frame] = -22;
  }
  for (let frame = 130; frame < 188; frame += 1) {
    sourceFrameDb[frame] = -33;
    renderedFrameDb[frame] = -42;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, false);
  assert.equal(report.clusterCount, 0);
});

test("does not flag a uniform rendered level shift without local collapse", () => {
  const sourceFrameDb = makeFrames(260, -90);
  const renderedFrameDb = makeFrames(260, -92);
  for (let frame = 30; frame < 220; frame += 1) {
    sourceFrameDb[frame] = -39;
    renderedFrameDb[frame] = -63;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, false);
  assert.equal(report.clusterCount, 0);
});

test("does not flag repeated real source pauses rendered as silence", () => {
  const sourceFrameDb = makeFrames(700, -92);
  const renderedFrameDb = makeFrames(700, -120);
  for (let frame = 20; frame < 680; frame += 1) {
    const inPause = frame % 70 >= 42 && frame % 70 < 48;
    sourceFrameDb[frame] = inPause ? -92 : -34;
    renderedFrameDb[frame] = inPause ? -120 : -27;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, false);
  assert.equal(report.clusterCount, 0);
});

test("ignores isolated one-frame render mismatches after cluster filtering", () => {
  const sourceFrameDb = makeFrames(900, -88);
  const renderedFrameDb = makeFrames(900, -94);
  for (let frame = 10; frame < 880; frame += 1) {
    sourceFrameDb[frame] = -32;
    renderedFrameDb[frame] = -25;
  }
  for (let frame = 80; frame < 850; frame += 120) {
    renderedFrameDb[frame] = -130;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, false);
  assert.equal(report.badSeconds, 0);
  assert.equal(report.clusterCount, 0);
});

test("aggregates repeated sub-sentence dropouts into a severe render failure", () => {
  const sourceFrameDb = makeFrames(500, -90);
  const renderedFrameDb = makeFrames(500, -95);
  for (let frame = 20; frame < 460; frame += 1) {
    sourceFrameDb[frame] = frame % 80 < 55 ? -31 : -82;
    renderedFrameDb[frame] = sourceFrameDb[frame] + 6;
  }
  for (const start of [70, 190, 310, 430]) {
    for (let frame = start; frame < start + 8; frame += 1) {
      sourceFrameDb[frame] = -29;
      renderedFrameDb[frame] = -125;
    }
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.equal(report.clusterCount, 4);
  assert.ok(report.badSeconds >= 0.6, `expected repeated dropouts to accumulate, got ${report.badSeconds}s`);
});

test("keeps a single 200 ms collapsed phrase fragment severe", () => {
  const sourceFrameDb = makeFrames(7000, -92);
  const renderedFrameDb = makeFrames(7000, -94);
  for (let frame = 5600; frame < 6050; frame += 1) {
    sourceFrameDb[frame] = -31;
    renderedFrameDb[frame] = -24;
  }
  for (let frame = 5957; frame < 5967; frame += 1) {
    renderedFrameDb[frame] = -68;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.equal(report.clusterCount, 1);
  assert.equal(report.badSeconds, 0.2);
});

test("detects rendered files truncated before active source speech ends", () => {
  const sourceFrameDb = makeFrames(200, -90);
  const renderedFrameDb = makeFrames(188, -84);
  for (let frame = 20; frame < 200; frame += 1) {
    sourceFrameDb[frame] = -31;
    if (frame < renderedFrameDb.length) renderedFrameDb[frame] = -24;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.ok(report.clusters.some((cluster) => cluster.startSec >= 3.8));
});

test("detects continuous soft speech that collapses below audibility", () => {
  const sourceFrameDb = makeFrames(420, -92);
  const renderedFrameDb = makeFrames(420, -94);
  for (let frame = 20; frame < 390; frame += 1) {
    sourceFrameDb[frame] = -45;
    renderedFrameDb[frame] = -39;
  }
  for (let frame = 120; frame < 165; frame += 1) {
    renderedFrameDb[frame] = -120;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.equal(report.clusterCount, 1);
  assert.ok(report.badSeconds >= 0.8, `expected collapsed soft speech to trip, got ${report.badSeconds}s`);
});

test("detects end-edge gradual dips that erase the last phoneme", () => {
  const sourceFrameDb = makeFrames(260, -90);
  const renderedFrameDb = makeFrames(260, -92);
  for (let frame = 40; frame < 180; frame += 1) {
    sourceFrameDb[frame] = -30;
    renderedFrameDb[frame] = -30;
  }
  for (let frame = 150; frame < 180; frame += 1) {
    const progress = (frame - 150) / 29;
    renderedFrameDb[frame] = -30 + (-65 + 30) * progress;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.equal(report.clusterCount, 1);
});

test("detects all-soft speech when the adaptive threshold would otherwise rise too high", () => {
  const sourceFrameDb = makeFrames(360, -40);
  const renderedFrameDb = makeFrames(360, -34);
  for (let frame = 110; frame < 160; frame += 1) {
    renderedFrameDb[frame] = -125;
  }

  const report = detectAudibilityDropouts({
    sourceFrameDb,
    renderedFrameDb,
    frameMs: 20,
  });

  assert.equal(report.severe, true);
  assert.equal(report.clusterCount, 1);
  assert.ok(report.sourceSpeechThresholdDb >= -31);
  assert.ok(report.badSeconds >= 1, `expected all-soft speech collapse to trip, got ${report.badSeconds}s`);
});
