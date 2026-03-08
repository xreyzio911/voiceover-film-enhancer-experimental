import assert from "node:assert/strict";
import test from "node:test";
import { analyzeFrameAudio, buildSpeechMask } from "./audioQc.ts";

const FRAME_MS = 10;
const SAMPLE_RATE = 16000;

type Section = {
  frames: number;
  fromDb: number;
  toDb?: number;
  peakLiftDb?: number;
  sharpnessDb?: number;
};

const ampFromDb = (db: number) => Math.pow(10, db / 20);

const analyzeSections = (sections: Section[]) => {
  const frameDb: number[] = [];
  const frameRms: number[] = [];
  const framePeak: number[] = [];
  const frameSharpness: number[] = [];

  for (const section of sections) {
    for (let index = 0; index < section.frames; index += 1) {
      const progress = section.frames <= 1 ? 0 : index / (section.frames - 1);
      const db = section.fromDb + ((section.toDb ?? section.fromDb) - section.fromDb) * progress;
      const rms = ampFromDb(db);
      const peak = Math.min(0.98, rms * ampFromDb(section.peakLiftDb ?? 10));

      frameDb.push(db);
      frameRms.push(rms);
      framePeak.push(peak);
      frameSharpness.push(section.sharpnessDb ?? -60);
    }
  }

  return analyzeFrameAudio(frameRms, framePeak, frameDb, frameSharpness, {
    sampleRate: SAMPLE_RATE,
    durationSec: (frameDb.length * FRAME_MS) / 1000,
    frameMs: FRAME_MS,
    peakDb: null,
    clipPct: 0,
    sampleSpikeCount: 0,
  });
};

test("buildSpeechMask keeps soft endings active and bridges short gaps", () => {
  const frameDb = [
    ...new Array(60).fill(-74),
    ...new Array(20).fill(-29),
    ...new Array(8).fill(-62),
    ...new Array(6).fill(-66),
    ...new Array(18).fill(-30),
    ...new Array(20).fill(-74),
  ];

  const mask = buildSpeechMask(frameDb, -74, { frameMs: FRAME_MS });

  assert.equal(mask.slice(80, 88).every(Boolean), true);
  assert.equal(mask.slice(88, 94).every(Boolean), true);
  assert.equal(mask.slice(0, 50).some(Boolean), false);
});

test("protected steady endings avoid fade and pause-noise flags", () => {
  const protectedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -31 },
    { frames: 5, fromDb: -75 },
    { frames: 90, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);

  assert.ok(protectedTail.onsetOvershootScore < 0.1);
  assert.ok(protectedTail.endFadeRiskScore < 0.1);
  assert.ok(protectedTail.pauseNoiseRisk < 0.1);
});

test("onset spike scoring rises for hot first-word entries", () => {
  const steady = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 180, fromDb: -28 },
    { frames: 100, fromDb: -75 },
  ]);
  const onsetHeavy = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 24, fromDb: -20 },
    { frames: 156, fromDb: -28 },
    { frames: 100, fromDb: -75 },
  ]);

  assert.ok(onsetHeavy.onsetOvershootScore > steady.onsetOvershootScore + 0.35);
});

test("mid-line sag scoring rises when the center of the line collapses", () => {
  const protectedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -31 },
    { frames: 5, fromDb: -75 },
    { frames: 90, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);
  const sagging = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 40, fromDb: -30, toDb: -28 },
    { frames: 50, fromDb: -28 },
    { frames: 70, fromDb: -38 },
    { frames: 50, fromDb: -28 },
    { frames: 40, fromDb: -28, toDb: -30 },
    { frames: 100, fromDb: -75 },
  ]);

  assert.ok(sagging.midLineSagScore > protectedTail.midLineSagScore + 0.3);
});

test("end-fade scoring rises when the last word drops away before silence", () => {
  const protectedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -31 },
    { frames: 5, fromDb: -75 },
    { frames: 90, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);
  const fadedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -42 },
    { frames: 100, fromDb: -75 },
  ]);

  assert.ok(fadedTail.endFadeRiskScore > protectedTail.endFadeRiskScore + 0.6);
});

test("pause-noise scoring rises when long silences stay lifted", () => {
  const quietPauses = analyzeSections([
    { frames: 120, fromDb: -80 },
    { frames: 180, fromDb: -28 },
    { frames: 160, fromDb: -80 },
  ]);
  const noisyPauses = analyzeSections([
    { frames: 120, fromDb: -50 },
    { frames: 180, fromDb: -30 },
    { frames: 160, fromDb: -50 },
  ]);

  assert.ok(noisyPauses.pauseNoiseRisk > quietPauses.pauseNoiseRisk + 0.3);
});

test("breath-spike scoring rises for isolated inhale bursts before speech", () => {
  const protectedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -31 },
    { frames: 5, fromDb: -75 },
    { frames: 90, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);
  const breathy = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 8, fromDb: -47, peakLiftDb: 15, sharpnessDb: -45 },
    { frames: 20, fromDb: -75 },
    { frames: 25, fromDb: -32, toDb: -28 },
    { frames: 120, fromDb: -28 },
    { frames: 20, fromDb: -28, toDb: -31 },
    { frames: 20, fromDb: -75 },
    { frames: 10, fromDb: -46, peakLiftDb: 14, sharpnessDb: -44 },
    { frames: 18, fromDb: -75 },
    { frames: 70, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);

  assert.ok(breathy.breathSpikeRisk > protectedTail.breathSpikeRisk + 0.5);
});

test("sentence-jump scoring rises when grouped lines land at different body levels", () => {
  const protectedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -31 },
    { frames: 5, fromDb: -75 },
    { frames: 90, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);
  const jumpy = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 120, fromDb: -28 },
    { frames: 20, fromDb: -28, toDb: -31 },
    { frames: 28, fromDb: -75 },
    { frames: 28, fromDb: -38, toDb: -33 },
    { frames: 90, fromDb: -33 },
    { frames: 20, fromDb: -33, toDb: -35 },
  ]);

  assert.ok(jumpy.sentenceJumpScore > protectedTail.sentenceJumpScore + 0.45);
});

test("line swing scoring rises for high-low-high speech contours", () => {
  const protectedTail = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 30, fromDb: -32, toDb: -28 },
    { frames: 140, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -31 },
    { frames: 5, fromDb: -75 },
    { frames: 90, fromDb: -29 },
    { frames: 20, fromDb: -29, toDb: -31 },
  ]);
  const swingy = analyzeSections([
    { frames: 120, fromDb: -75 },
    { frames: 25, fromDb: -32, toDb: -28 },
    { frames: 35, fromDb: -28 },
    { frames: 35, fromDb: -35 },
    { frames: 35, fromDb: -27 },
    { frames: 35, fromDb: -34 },
    { frames: 35, fromDb: -28 },
    { frames: 25, fromDb: -28, toDb: -32 },
    { frames: 100, fromDb: -75 },
  ]);

  assert.ok(swingy.lineSwingScore > protectedTail.lineSwingScore + 0.05);
});
