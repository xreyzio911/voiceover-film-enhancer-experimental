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

test("buildSpeechMask holds quiet trailing speech below close threshold for at least 150 ms", () => {
  const tailStart = 160;
  const frameDb = [
    ...new Array(60).fill(-82),
    ...new Array(100).fill(-28),
    ...new Array(20).fill(-65),
    ...new Array(20).fill(-82),
  ];

  const mask = buildSpeechMask(frameDb, -82, { frameMs: FRAME_MS });

  assert.equal(mask.slice(tailStart, tailStart + 15).every(Boolean), true);
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

test("end-edge dip metric catches a short final-phoneme level drop", () => {
  const steadyEnding = analyzeSections([
    { frames: 100, fromDb: -78 },
    { frames: 160, fromDb: -27 },
    { frames: 30, fromDb: -78 },
  ]);
  const dippedEnding = analyzeSections([
    { frames: 100, fromDb: -78 },
    { frames: 145, fromDb: -27 },
    { frames: 15, fromDb: -33 },
    { frames: 30, fromDb: -78 },
  ]);

  assert.ok(
    steadyEnding.endEdgeDipDb < 2.0,
    `steady ending should stay below warning level, got ${steadyEnding.endEdgeDipDb.toFixed(1)} dB`,
  );
  assert.ok(dippedEnding.endEdgeDipDb > 4.5, `dipped ending should expose a short edge dip, got ${dippedEnding.endEdgeDipDb.toFixed(1)} dB`);
});

test("end-edge dip metric keeps damaged speech tails in the measured tail", () => {
  const damagedTail = analyzeSections([
    { frames: 100, fromDb: -78 },
    { frames: 145, fromDb: -27 },
    { frames: 15, fromDb: -39 },
    { frames: 30, fromDb: -78 },
  ]);

  assert.ok(
    damagedTail.endEdgeDipDb > 9,
    `damaged speech tail should not be trimmed out before scoring, got ${damagedTail.endEdgeDipDb.toFixed(1)} dB`,
  );
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

test("echo scoring does not treat clean rhythmic speech correlation as room echo", () => {
  const rhythmicSpeech: Section[] = [];
  for (let index = 0; index < 50; index += 1) {
    rhythmicSpeech.push({ frames: 4, fromDb: -30 }, { frames: 4, fromDb: -78 });
  }

  const metrics = analyzeSections(rhythmicSpeech);

  assert.ok(metrics.reverbScore < 0.05);
  assert.ok(metrics.echoScore < 0.08);
  assert.equal(metrics.echoDelayMs, null);
});

test("echo scoring still flags real room tails with supported short-lag correlation", () => {
  const roomy = analyzeSections([
    { frames: 120, fromDb: -78 },
    { frames: 120, fromDb: -30 },
    { frames: 18, fromDb: -35, toDb: -42 },
    { frames: 22, fromDb: -42, toDb: -48 },
    { frames: 70, fromDb: -78 },
    { frames: 110, fromDb: -30 },
    { frames: 18, fromDb: -35, toDb: -42 },
    { frames: 22, fromDb: -42, toDb: -48 },
    { frames: 80, fromDb: -78 },
  ]);

  assert.ok(roomy.reverbScore > 0.35);
  assert.ok(roomy.echoScore >= 0.38);
  assert.notEqual(roomy.echoDelayMs, null);
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

test("cold-open scoring measures quiet heads against later dialogue body", () => {
  const dipped = analyzeSections([
    { frames: 100, fromDb: -78 },
    { frames: 110, fromDb: -34 },
    { frames: 24, fromDb: -78 },
    { frames: 110, fromDb: -34 },
    { frames: 28, fromDb: -78 },
    { frames: 150, fromDb: -30 },
    { frames: 32, fromDb: -78 },
    { frames: 150, fromDb: -30 },
    { frames: 100, fromDb: -78 },
  ]);

  assert.ok(
    Math.abs(dipped.coldOpenDipDb - 4) < 0.35,
    `expected ~4 dB cold-open dip, got ${dipped.coldOpenDipDb}`,
  );
  assert.ok(
    Math.abs(dipped.coldOpenRiskScore - 0.75) < 0.12,
    `expected ~0.75 cold-open risk, got ${dipped.coldOpenRiskScore}`,
  );
});

test("cold-open scoring stays near zero for flat dialogue heads", () => {
  const flat = analyzeSections([
    { frames: 100, fromDb: -78 },
    { frames: 110, fromDb: -30 },
    { frames: 24, fromDb: -78 },
    { frames: 110, fromDb: -30 },
    { frames: 28, fromDb: -78 },
    { frames: 150, fromDb: -30 },
    { frames: 32, fromDb: -78 },
    { frames: 150, fromDb: -30 },
    { frames: 100, fromDb: -78 },
  ]);

  assert.ok(Math.abs(flat.coldOpenDipDb) < 0.35, `expected flat cold-open dip, got ${flat.coldOpenDipDb}`);
  assert.equal(flat.coldOpenRiskScore, 0);
});

test("cold-open scoring trims first-run edges before comparing to the later body", () => {
  const edgedHeads = analyzeSections([
    { frames: 100, fromDb: -78 },
    { frames: 10, fromDb: -42 },
    { frames: 35, fromDb: -30 },
    { frames: 10, fromDb: -42 },
    { frames: 24, fromDb: -78 },
    { frames: 10, fromDb: -42 },
    { frames: 35, fromDb: -30 },
    { frames: 10, fromDb: -42 },
    { frames: 24, fromDb: -78 },
    { frames: 10, fromDb: -42 },
    { frames: 35, fromDb: -30 },
    { frames: 10, fromDb: -42 },
    { frames: 32, fromDb: -78 },
    { frames: 150, fromDb: -30 },
    { frames: 100, fromDb: -78 },
  ]);

  assert.ok(
    edgedHeads.coldOpenDipDb < 1,
    `edge-only first-run dips should not trigger a cold-open warning, got ${edgedHeads.coldOpenDipDb}`,
  );
  assert.equal(edgedHeads.coldOpenRiskScore, 0);
});

test("cold-open scoring includes short opening words", () => {
  const shortOpener = analyzeSections([
    { frames: 80, fromDb: -78 },
    { frames: 28, fromDb: -34 },
    { frames: 18, fromDb: -78 },
    { frames: 100, fromDb: -27 },
    { frames: 20, fromDb: -78 },
    { frames: 100, fromDb: -27 },
    { frames: 80, fromDb: -78 },
  ]);

  assert.ok(
    shortOpener.coldOpenDipDb > 5,
    `short opening words should contribute to cold-open scoring, got ${shortOpener.coldOpenDipDb.toFixed(1)} dB`,
  );
});

test("sparse sentence-jump scoring still rises when isolated lines land at different levels", () => {
  const steadySparse = analyzeSections([
    { frames: 200, fromDb: -78 },
    { frames: 80, fromDb: -30 },
    { frames: 180, fromDb: -78 },
    { frames: 76, fromDb: -30 },
    { frames: 220, fromDb: -78 },
    { frames: 72, fromDb: -30 },
    { frames: 180, fromDb: -78 },
  ]);
  const jumpySparse = analyzeSections([
    { frames: 200, fromDb: -78 },
    { frames: 80, fromDb: -29 },
    { frames: 180, fromDb: -78 },
    { frames: 76, fromDb: -36 },
    { frames: 220, fromDb: -78 },
    { frames: 72, fromDb: -31 },
    { frames: 180, fromDb: -78 },
  ]);

  assert.ok(jumpySparse.sentenceJumpScore > steadySparse.sentenceJumpScore + 0.28);
});

test("lead-in breath-spike scoring rises for short pre-word transients above following speech", () => {
  const steady = analyzeSections([
    { frames: 180, fromDb: -78 },
    { frames: 24, fromDb: -78 },
    { frames: 24, fromDb: -31 },
    { frames: 80, fromDb: -29 },
    { frames: 200, fromDb: -78 },
  ]);
  const leadBurst = analyzeSections([
    { frames: 180, fromDb: -78 },
    { frames: 6, fromDb: -48, peakLiftDb: 28, sharpnessDb: -50 },
    { frames: 24, fromDb: -78 },
    { frames: 24, fromDb: -31 },
    { frames: 80, fromDb: -29 },
    { frames: 200, fromDb: -78 },
  ]);

  assert.ok(leadBurst.breathSpikeRisk > steady.breathSpikeRisk + 0.18);
});

test("click scoring ignores normal high-crest speech consonants", () => {
  const cleanConsonants = analyzeSections([
    { frames: 120, fromDb: -78 },
    { frames: 180, fromDb: -29, peakLiftDb: 22, sharpnessDb: -30 },
    { frames: 100, fromDb: -78 },
  ]);

  assert.ok(cleanConsonants.clickScore < 0.08);
});

test("click scoring still rises for repeated isolated non-speech clicks", () => {
  const clickBursts: Section[] = [{ frames: 60, fromDb: -78 }];
  for (let index = 0; index < 8; index += 1) {
    clickBursts.push({ frames: 4, fromDb: -56, peakLiftDb: 35, sharpnessDb: -20 });
    clickBursts.push({ frames: 8, fromDb: -78 });
  }
  clickBursts.push({ frames: 120, fromDb: -29 });

  const clicky = analyzeSections(clickBursts);

  assert.ok(clicky.clickScore > 0.12);
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
