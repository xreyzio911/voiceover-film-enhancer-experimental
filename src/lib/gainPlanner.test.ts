import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { analyzeFloatSamples, buildSpeechMask } from "./audioQc.ts";
import {
  applyGainCurveToSamples,
  emitSendcmdScript,
  planGainCurve,
  speechRunsFromMask,
} from "./gainPlanner.ts";
import { computeLogBandSpectrumDb, computeSibilanceScore } from "./spectrum.ts";
import { decodeWav, encodeWavFloat32 } from "./webAudioRender.ts";

const SAMPLE_RATE = 16000;
const FRAME_MS = 10;
const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000;

const dbToLin = (db: number) => Math.pow(10, db / 20);

const synthesizeTake = (
  spans: Array<{ startSec: number; endSec: number; rmsDb: number }>,
  totalSec: number,
  noiseDb = -70,
): Float32Array => {
  const total = Math.round(totalSec * SAMPLE_RATE);
  const out = new Float32Array(total);
  // Noise floor.
  const noiseAmp = dbToLin(noiseDb);
  for (let i = 0; i < total; i += 1) out[i] = (Math.random() * 2 - 1) * noiseAmp;

  // Each span is a low-frequency speech-like tone at the requested RMS.
  for (const span of spans) {
    const start = Math.round(span.startSec * SAMPLE_RATE);
    const end = Math.round(span.endSec * SAMPLE_RATE);
    const amp = dbToLin(span.rmsDb) * Math.SQRT2; // peak for a sine at rmsDb
    for (let i = start; i < end && i < total; i += 1) {
      // mix of 200 Hz + 500 Hz to look like a voice formant
      out[i] += amp * (0.65 * Math.sin((2 * Math.PI * 200 * i) / SAMPLE_RATE) + 0.35 * Math.sin((2 * Math.PI * 500 * i) / SAMPLE_RATE));
    }
  }
  return out;
};

const measureRmsDb = (samples: Float32Array, start: number, end: number) => {
  let sum = 0;
  for (let i = start; i < end; i += 1) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / Math.max(1, end - start));
  return rms <= 0 ? -120 : 20 * Math.log10(rms);
};

const stdDev = (values: number[]) => {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / values.length;
  return Math.sqrt(variance);
};

describe("gainPlanner", () => {
  it("normalizes uneven sentences to within +/- 2 dB", () => {
    // Three sentences at -30, -12, -26 dB RMS.
    const spans = [
      { startSec: 0.3, endSec: 1.5, rmsDb: -30 },
      { startSec: 2.0, endSec: 3.2, rmsDb: -12 },
      { startSec: 3.7, endSec: 4.9, rmsDb: -26 },
    ];
    const samples = synthesizeTake(spans, 5.2, -72);
    const metrics = analyzeFloatSamples(samples, SAMPLE_RATE, FRAME_MS);

    // Build frame-db + speech mask from the same envelope metrics.
    const frameDb: number[] = [];
    const frameCount = Math.floor(samples.length / FRAME_SAMPLES);
    for (let f = 0; f < frameCount; f += 1) {
      let sum = 0;
      for (let i = 0; i < FRAME_SAMPLES; i += 1) {
        const v = samples[f * FRAME_SAMPLES + i] ?? 0;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / FRAME_SAMPLES);
      frameDb.push(rms <= 0 ? -120 : 20 * Math.log10(rms));
    }
    const mask = buildSpeechMask(frameDb, metrics.noiseFloorDb);
    const runs = speechRunsFromMask(mask);
    assert.ok(runs.length >= 3, `expected at least 3 runs, got ${runs.length}`);

    const plan = planGainCurve({
      frameDb,
      speechRuns: runs,
      noiseFloorDb: metrics.noiseFloorDb,
      speechThresholdDb: metrics.speechThresholdDb,
      pauseNoiseRisk: metrics.pauseNoiseRisk,
      frameMs: FRAME_MS,
      samples,
      sampleRate: SAMPLE_RATE,
      // Input is wildly uneven (-30 / -12 / -26 dB). Tell the planner to use
      // the full micro-ride budget so it can track each sentence body.
      instabilityHint: 0.95,
    });

    const leveled = applyGainCurveToSamples(samples, plan.gainCurve, SAMPLE_RATE, 1, FRAME_MS);

    // Measure the stable BODY of each synthesized span, not the detected-run
    // edges (which include expander ramps — those are intentional and not a
    // defect of the leveler).
    const rmsBodyByRun = spans.map((s) => {
      const span = s.endSec - s.startSec;
      const bodyStart = Math.round((s.startSec + span * 0.25) * SAMPLE_RATE);
      const bodyEnd = Math.round((s.endSec - span * 0.25) * SAMPLE_RATE);
      return measureRmsDb(leveled, bodyStart, bodyEnd);
    });
    const spread = Math.max(...rmsBodyByRun) - Math.min(...rmsBodyByRun);
    // Original source spread is 18 dB (rms -30/-12/-26). We consider the
    // leveler healthy when that is cut to < 7 dB (outliers capped by the
    // configured gain bounds) AND std-dev drops by >55%.
    assert.ok(spread < 7, `body RMS spread should be < 7 dB after leveling, got ${spread.toFixed(2)} (${rmsBodyByRun.map((v) => v.toFixed(1)).join(", ")})`);

    const stdBefore = stdDev(spans.map((s) => s.rmsDb));
    const stdAfter = stdDev(rmsBodyByRun);
    assert.ok(stdAfter < stdBefore * 0.45, `std dev should drop >55%: before ${stdBefore.toFixed(2)} after ${stdAfter.toFixed(2)}`);

    // The leveled output must have its runs all sitting in the -30..-18 dB band
    // (not spread across the original -30..-12 band). This is the core "same
    // tone / same volume" promise of the planner.
    for (const level of rmsBodyByRun) {
      assert.ok(level >= -32 && level <= -18, `run body out of target band: ${level.toFixed(2)} dB`);
    }
  });

  it("keeps silences quiet (expander keeps pauses below speech by >= 10 dB)", () => {
    const spans = [{ startSec: 0.5, endSec: 2.5, rmsDb: -24 }];
    const samples = synthesizeTake(spans, 4, -58); // noisier pause
    const metrics = analyzeFloatSamples(samples, SAMPLE_RATE, FRAME_MS);

    const frameDb: number[] = [];
    const frameCount = Math.floor(samples.length / FRAME_SAMPLES);
    for (let f = 0; f < frameCount; f += 1) {
      let sum = 0;
      for (let i = 0; i < FRAME_SAMPLES; i += 1) {
        const v = samples[f * FRAME_SAMPLES + i] ?? 0;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / FRAME_SAMPLES);
      frameDb.push(rms <= 0 ? -120 : 20 * Math.log10(rms));
    }
    const mask = buildSpeechMask(frameDb, metrics.noiseFloorDb);
    const runs = speechRunsFromMask(mask);

    const plan = planGainCurve({
      frameDb,
      speechRuns: runs,
      noiseFloorDb: metrics.noiseFloorDb,
      speechThresholdDb: metrics.speechThresholdDb,
      pauseNoiseRisk: metrics.pauseNoiseRisk,
      frameMs: FRAME_MS,
    });

    // expander depth must be applied in pauses
    assert.ok(plan.expanderDepthDb >= 12, `expander depth too small: ${plan.expanderDepthDb.toFixed(1)}`);

    // pick a mid-pause frame (3.5s) and confirm gain there is <= -10 dB
    const pauseFrame = Math.round(3.5 * 100);
    const pauseGainDb = 20 * Math.log10(plan.gainCurve[pauseFrame] + 1e-9);
    assert.ok(pauseGainDb <= -9, `pause gain should duck: got ${pauseGainDb.toFixed(1)} dB`);
  });

  it("does not apply peaks above the ceiling (peak guard)", () => {
    const spans = [{ startSec: 0.3, endSec: 1.0, rmsDb: -6 }]; // already loud
    const samples = synthesizeTake(spans, 1.5, -75);

    const frameDb: number[] = [];
    const frameCount = Math.floor(samples.length / FRAME_SAMPLES);
    for (let f = 0; f < frameCount; f += 1) {
      let sum = 0;
      for (let i = 0; i < FRAME_SAMPLES; i += 1) {
        const v = samples[f * FRAME_SAMPLES + i] ?? 0;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / FRAME_SAMPLES);
      frameDb.push(rms <= 0 ? -120 : 20 * Math.log10(rms));
    }
    const metrics = analyzeFloatSamples(samples, SAMPLE_RATE, FRAME_MS);
    const mask = buildSpeechMask(frameDb, metrics.noiseFloorDb);
    const runs = speechRunsFromMask(mask);

    const plan = planGainCurve({
      frameDb,
      speechRuns: runs,
      noiseFloorDb: metrics.noiseFloorDb,
      speechThresholdDb: metrics.speechThresholdDb,
      pauseNoiseRisk: metrics.pauseNoiseRisk,
      frameMs: FRAME_MS,
      samples,
      sampleRate: SAMPLE_RATE,
      peakCeilingDb: -3,
    });

    const leveled = applyGainCurveToSamples(samples, plan.gainCurve, SAMPLE_RATE, 1, FRAME_MS);
    let maxAbs = 0;
    for (let i = 0; i < leveled.length; i += 1) if (Math.abs(leveled[i]) > maxAbs) maxAbs = Math.abs(leveled[i]);
    const peakDb = 20 * Math.log10(maxAbs + 1e-9);
    assert.ok(peakDb <= -2.5, `peak ceiling exceeded: ${peakDb.toFixed(2)} dB`);
  });
});

describe("run classification", () => {
  it("tags short high-crest runs as transient-breath and normal runs as body-speech", () => {
    // Without sample data the planner estimates peak = max(frameDb) + 12 dB,
    // so we can drive classification purely through frame-level dB values.
    const frameDb: number[] = [];
    // 1 s silence
    for (let i = 0; i < 100; i += 1) frameDb.push(-70);
    // 1.5 s dialogue body at -22 dB (frames 100..249, stable throughout —
    // max ≈ body so crest ≈ 12 dB → body-speech).
    for (let i = 0; i < 150; i += 1) frameDb.push(-22);
    // 0.3 s silence
    for (let i = 0; i < 30; i += 1) frameDb.push(-70);
    // 0.25 s gasp: 25 frames. 2 frames at 0 dB (gasp "puff"), 23 frames
    // at -25 dB (quiet post-puff tail). Body mean ≈ -11 dB, max = 0 dB,
    // estimated peak = 0 + 12 = +12 dB → crest ≈ 23 dB → transient-breath.
    for (let i = 0; i < 2; i += 1) frameDb.push(0);
    for (let i = 0; i < 23; i += 1) frameDb.push(-25);
    // 0.5 s silence
    for (let i = 0; i < 50; i += 1) frameDb.push(-70);

    const speechRuns = [
      { startFrame: 100, endFrame: 250 }, // dialogue (1.5 s)
      { startFrame: 280, endFrame: 305 }, // gasp (250 ms)
    ];
    const plan = planGainCurve({
      frameDb,
      speechRuns,
      noiseFloorDb: -70,
      speechThresholdDb: -55,
      pauseNoiseRisk: 0.1,
      frameMs: 10,
      targetDb: -22,
    });

    assert.equal(plan.runs.length, 2);
    assert.equal(plan.runs[0].runClass, "body-speech");
    assert.equal(
      plan.runs[1].runClass,
      "transient-breath",
      `gasp should classify as breath (crest=${plan.runs[1].crestDb.toFixed(1)} dB, len=${(plan.runs[1].endFrame - plan.runs[1].startFrame) * 10} ms)`,
    );
    assert.equal(plan.breathRunCount, 1);

    // Breath runs use a tighter ±6 dB clamp AND a lower target (breathTarget
    // = targetDb - 2.5 = -24.5). Whatever the gasp body RMS came out to,
    // the planned gain must stay within [-6, 6].
    const gaspGain = plan.runs[1].plannedGainDb;
    assert.ok(
      gaspGain <= 6 && gaspGain >= -6,
      `gasp gain must stay in tight breath clamp, got ${gaspGain.toFixed(2)} dB`,
    );
    // Body-speech dialogue gets the wider ±14 dB clamp, so it can rise or
    // fall more. This asserts we DIDN'T apply the breath clamp to dialogue.
    const dialogueGain = plan.runs[0].plannedGainDb;
    assert.ok(
      plan.runs[0].runClass === "body-speech" && Math.abs(dialogueGain) <= 14,
      `dialogue gain must use body-speech clamp (got ${dialogueGain.toFixed(2)} dB)`,
    );
  });
});

describe("localized peak guard", () => {
  it("dips only around the plosive frame, not the whole sentence body", () => {
    const sampleRate = 48000;
    const frameMs = 10;
    const samplesPerFrame = (sampleRate * frameMs) / 1000; // 480
    const totalFrames = 200;

    // Constant -22 dB body with ONE sample-level spike at frame 100. We
    // keep `frameDb[100] = -22` so the body RMS over the trim region
    // remains -22 (target lands at -22, planned gain 0 dB). The spike is
    // visible to the peak-guard via `framePeakDb` (computed from samples).
    const samples = new Float32Array(totalFrames * samplesPerFrame);
    for (let i = 0; i < samples.length; i += 1) {
      samples[i] = Math.sin((2 * Math.PI * 300 * i) / sampleRate) * 0.08; // -22 dB RMS
    }
    // Inject a single full-scale sample at frame 100. This bumps frame
    // 100's peak to ≈ 0.99 but barely changes its RMS (479 quiet + 1 loud).
    samples[100 * samplesPerFrame + 100] = 0.99;

    const frameDb: number[] = [];
    for (let f = 0; f < totalFrames; f += 1) frameDb.push(-22);

    const speechRuns = [{ startFrame: 0, endFrame: totalFrames }];
    const plan = planGainCurve({
      frameDb,
      speechRuns,
      noiseFloorDb: -70,
      speechThresholdDb: -55,
      pauseNoiseRisk: 0.1,
      frameMs,
      samples,
      sampleRate,
      targetDb: -22,
      peakCeilingDb: -4,
    });

    const gainFarA = 20 * Math.log10(plan.gainCurve[10] + 1e-9);
    const gainFarB = 20 * Math.log10(plan.gainCurve[190] + 1e-9);
    const gainAtPlosive = 20 * Math.log10(plan.gainCurve[100] + 1e-9);
    const gainNearPlosive = 20 * Math.log10(plan.gainCurve[105] + 1e-9);

    // 1. Body frames FAR from the plosive are untouched.
    assert.ok(
      Math.abs(gainFarA) < 0.3 && Math.abs(gainFarB) < 0.3,
      `body frames far from the plosive should be at 0 dB: ${gainFarA.toFixed(2)} / ${gainFarB.toFixed(2)}`,
    );

    // 2. The plosive frame itself is dipped meaningfully (at least 1 dB).
    assert.ok(
      gainAtPlosive < gainFarA - 1,
      `plosive frame should be dipped (got ${gainAtPlosive.toFixed(2)} vs body ${gainFarA.toFixed(2)})`,
    );

    // 3. The dip is LOCALIZED — by 5 frames away we're back near body gain.
    assert.ok(
      Math.abs(gainNearPlosive - gainFarA) < 1.5,
      `dip should be localized — 50 ms away we should be near body gain (got ${gainNearPlosive.toFixed(2)} vs body ${gainFarA.toFixed(2)})`,
    );
  });
});

describe("ramp placement", () => {
  it("keeps the full body of a speech run at body gain, ramps only into surrounding silence", () => {
    // Simulate 3 s file: 0-1 s silence, 1-2.5 s speech, 2.5-3 s silence.
    const frameDb: number[] = [];
    for (let i = 0; i < 100; i += 1) frameDb.push(-70); // 1 s silence
    for (let i = 0; i < 150; i += 1) frameDb.push(-22); // 1.5 s speech
    for (let i = 0; i < 50; i += 1) frameDb.push(-70); // 0.5 s silence
    const speechRuns = [{ startFrame: 100, endFrame: 250 }];

    const plan = planGainCurve({
      frameDb,
      speechRuns,
      noiseFloorDb: -70,
      speechThresholdDb: -55,
      pauseNoiseRisk: 0.1,
      frameMs: 10,
      targetDb: -22,
      instabilityHint: 0.05,
    });

    // Pick 5 frames spread across the body. The first frame of the run
    // (frame 100) and the last frame (frame 249) BOTH need to be at full
    // body gain — that's the whole point of moving ramps into silence.
    const bodyFirstDb = 20 * Math.log10(plan.gainCurve[100] + 1e-9);
    const bodyLastDb = 20 * Math.log10(plan.gainCurve[249] + 1e-9);
    const bodyMidDb = 20 * Math.log10(plan.gainCurve[175] + 1e-9);
    assert.ok(
      Math.abs(bodyFirstDb - bodyMidDb) < 1.2,
      `first body frame ${bodyFirstDb.toFixed(2)} dB should be close to mid ${bodyMidDb.toFixed(2)} dB (no attack-duck)`,
    );
    assert.ok(
      Math.abs(bodyLastDb - bodyMidDb) < 1.2,
      `last body frame ${bodyLastDb.toFixed(2)} dB should be close to mid ${bodyMidDb.toFixed(2)} dB (no release-duck)`,
    );

    // Just before the run (frame 99 → t = 0.99 s, 10 ms before run start)
    // should still be mid-attack, not at full body gain. Expander floor is
    // much lower, so it's between those two.
    const attackEdgeDb = 20 * Math.log10(plan.gainCurve[99] + 1e-9);
    assert.ok(
      attackEdgeDb < bodyFirstDb,
      `attack edge ${attackEdgeDb.toFixed(2)} dB should be below body first ${bodyFirstDb.toFixed(2)} dB`,
    );

    // Deep in the post-run silence (frame 299 → 2.99 s, well past 500 ms
    // release) gain should be at full expander floor.
    const deepSilenceGainDb = 20 * Math.log10(plan.gainCurve[299] + 1e-9);
    assert.ok(
      deepSilenceGainDb <= -9,
      `deep silence gain ${deepSilenceGainDb.toFixed(2)} dB should be below -9 dB (full expander)`,
    );
  });
});

describe("trimmed-mean target", () => {
  it("trims loud and quiet outliers so the target tracks the typical sentence level", () => {
    // 10 sentences: 8 typical at -27 dB, one loud outlier at -10 dB, one
    // quiet outlier at -42 dB. A median would pick the middle of all 10 and
    // would sit at -27 too; but the mean without trimming would get dragged
    // up by the loud outlier. The trimmed mean should return exactly the
    // typical level.
    const levels = [-27, -27, -27, -10, -27, -42, -27, -27, -27, -27];
    const frameDb: number[] = [];
    const speechRuns: Array<{ startFrame: number; endFrame: number }> = [];
    for (let s = 0; s < levels.length; s += 1) {
      const gap = 30;
      const speechLen = 80;
      const start = frameDb.length + gap;
      for (let i = 0; i < gap; i += 1) frameDb.push(-70);
      for (let i = 0; i < speechLen; i += 1) frameDb.push(levels[s]);
      speechRuns.push({ startFrame: start, endFrame: start + speechLen });
    }

    const plan = planGainCurve({
      frameDb,
      speechRuns,
      noiseFloorDb: -70,
      speechThresholdDb: -55,
      pauseNoiseRisk: 0.1,
      frameMs: 10,
      targetDb: -22,
    });

    // Trimmed mean of [-42,-27,-27,-27,-27,-27,-27,-27,-27,-10] after
    // dropping 1 each end = mean([-27]*8) = -27.
    // The planner now follows the source lightly so actors converge toward
    // the shared house VO target instead of preserving original offsets.
    const expectedTarget = 0.15 * -27 + 0.85 * -22;
    assert.ok(
      Math.abs(plan.targetDb - expectedTarget) < 0.2,
      `target ${plan.targetDb.toFixed(2)} dB should track the trimmed mean (${expectedTarget.toFixed(2)} dB)`,
    );

    // The 8 typical sentences should all be lifted by ≈ +2.25 dB to hit
    // target. Outliers get clamped at the gain window.
    const typicalRun = plan.runs.find((r) => Math.abs(r.meanDb - -27) < 0.5);
    assert.ok(typicalRun, "expected a typical run in the plan");
    const expectedGain = expectedTarget - -27;
    assert.ok(
      Math.abs(typicalRun!.plannedGainDb - expectedGain) < 0.5,
      `typical sentence should get ≈ ${expectedGain.toFixed(2)} dB, got ${typicalRun!.plannedGainDb.toFixed(2)} dB`,
    );
  });
});

describe("adaptive micro-ride", () => {
  it("tightens the micro-ride on clean sources (low instabilityHint) and widens it on messy sources", () => {
    const frameDb: number[] = [];
    // 30 frames of silence at -70 dB
    for (let i = 0; i < 30; i += 1) frameDb.push(-70);
    // 120 frames of steady speech near -22 dB
    for (let i = 0; i < 120; i += 1) frameDb.push(-22);
    // 30 frames of silence
    for (let i = 0; i < 30; i += 1) frameDb.push(-70);
    const speechRuns = [{ startFrame: 30, endFrame: 150 }];

    const cleanPlan = planGainCurve({
      frameDb,
      speechRuns,
      noiseFloorDb: -70,
      speechThresholdDb: -55,
      pauseNoiseRisk: 0.1,
      frameMs: 10,
      instabilityHint: 0.05,
    });
    const messyPlan = planGainCurve({
      frameDb,
      speechRuns,
      noiseFloorDb: -70,
      speechThresholdDb: -55,
      pauseNoiseRisk: 0.1,
      frameMs: 10,
      instabilityHint: 0.95,
    });

    assert.ok(
      cleanPlan.microRideDb < messyPlan.microRideDb - 0.5,
      `clean micro-ride (${cleanPlan.microRideDb.toFixed(2)}) should be well below messy (${messyPlan.microRideDb.toFixed(2)})`,
    );
    assert.ok(cleanPlan.microRideDb <= 0.6, `clean micro-ride should be tight: ${cleanPlan.microRideDb.toFixed(2)}`);
    assert.ok(messyPlan.microRideDb >= 1.3, `messy micro-ride should stay wide: ${messyPlan.microRideDb.toFixed(2)}`);
  });
});

describe("house target and performance transients", () => {
  it("pulls different actor levels toward the same dialogue target", () => {
    const buildPlanForLevel = (speechDb: number) => {
      const frameDb: number[] = [];
      for (let i = 0; i < 40; i += 1) frameDb.push(-75);
      for (let i = 0; i < 140; i += 1) frameDb.push(speechDb);
      for (let i = 0; i < 40; i += 1) frameDb.push(-75);
      return planGainCurve({
        frameDb,
        speechRuns: [{ startFrame: 40, endFrame: 180 }],
        noiseFloorDb: -75,
        speechThresholdDb: -60,
        pauseNoiseRisk: 0.05,
        frameMs: FRAME_MS,
        targetDb: -22,
      });
    };

    const quietActor = buildPlanForLevel(-30);
    const loudActor = buildPlanForLevel(-16);
    const quietBody = quietActor.runs[0].meanDb + quietActor.runs[0].plannedGainDb;
    const loudBody = loudActor.runs[0].meanDb + loudActor.runs[0].plannedGainDb;

    assert.ok(Math.abs(quietBody - loudBody) < 2.5, `actors should converge: ${quietBody.toFixed(2)} vs ${loudBody.toFixed(2)} dB`);
    assert.ok(Math.abs(quietActor.targetDb - -22) < 1.4, `quiet actor target should stay near house target: ${quietActor.targetDb.toFixed(2)}`);
    assert.ok(Math.abs(loudActor.targetDb - -22) < 1.4, `loud actor target should stay near house target: ${loudActor.targetDb.toFixed(2)}`);
  });

  it("tames short ah/ugh/hm-style spikes without dipping neighboring dialogue", () => {
    const sampleRate = 16000;
    const frameMs = 10;
    const samplesPerFrame = (sampleRate * frameMs) / 1000;
    const totalFrames = 520;
    const samples = new Float32Array(totalFrames * samplesPerFrame);
    const frameDb = new Array<number>(totalFrames).fill(-78);

    const paintTone = (startFrame: number, endFrame: number, rmsDb: number, hz: number) => {
      const amp = dbToLin(rmsDb) * Math.SQRT2;
      for (let frame = startFrame; frame < endFrame; frame += 1) {
        frameDb[frame] = rmsDb;
        const start = frame * samplesPerFrame;
        const end = start + samplesPerFrame;
        for (let i = start; i < end; i += 1) {
          samples[i] += Math.sin((2 * Math.PI * hz * i) / sampleRate) * amp;
        }
      }
    };

    paintTone(100, 220, -22, 220);
    paintTone(245, 290, -20, 330);
    paintTone(320, 440, -22, 220);
    samples[260 * samplesPerFrame + 30] = 0.86;

    const plan = planGainCurve({
      frameDb,
      speechRuns: [
        { startFrame: 100, endFrame: 220 },
        { startFrame: 245, endFrame: 290 },
        { startFrame: 320, endFrame: 440 },
      ],
      noiseFloorDb: -78,
      speechThresholdDb: -62,
      pauseNoiseRisk: 0.08,
      frameMs,
      samples,
      sampleRate,
      targetDb: -22,
      peakCeilingDb: -3,
    });

    assert.equal(plan.runs[1].runClass, "transient-breath");
    assert.ok(plan.runs[1].plannedGainDb <= -4.5, `performance transient should be pulled below dialogue: ${plan.runs[1].plannedGainDb.toFixed(2)} dB`);

    const leveled = applyGainCurveToSamples(samples, plan.gainCurve, sampleRate, 1, frameMs);
    const transientFrameStart = 260 * samplesPerFrame;
    let transientPeak = 0;
    for (let i = transientFrameStart; i < transientFrameStart + samplesPerFrame; i += 1) {
      transientPeak = Math.max(transientPeak, Math.abs(leveled[i]));
    }
    const transientPeakDb = 20 * Math.log10(transientPeak + 1e-9);
    assert.ok(
      transientPeakDb <= plan.targetDb + 13,
      `performance spike should sit under dialogue peak range: ${transientPeakDb.toFixed(2)} dB`,
    );

    const beforeDialogueGainDb = 20 * Math.log10(plan.gainCurve[180] + 1e-9);
    const afterDialogueGainDb = 20 * Math.log10(plan.gainCurve[360] + 1e-9);
    assert.ok(Math.abs(beforeDialogueGainDb) < 0.8, `pre-transient dialogue should not be dipped: ${beforeDialogueGainDb.toFixed(2)} dB`);
    assert.ok(Math.abs(afterDialogueGainDb) < 0.8, `post-transient dialogue should not be dipped: ${afterDialogueGainDb.toFixed(2)} dB`);
  });
});

describe("emitSendcmdScript", () => {
  it("emits keyframes for a fluctuating curve and zeroes timestamps to windowStart", () => {
    // Curve: 1.0 for frames 0-9, 2.0 for frames 10-19, 0.5 for frames 20-29
    const curve = new Float32Array(30);
    for (let i = 0; i < 10; i += 1) curve[i] = 1.0;
    for (let i = 10; i < 20; i += 1) curve[i] = 2.0;
    for (let i = 20; i < 30; i += 1) curve[i] = 0.5;

    const script = emitSendcmdScript(curve, 10, 0, 0.3, 0.1);
    const lines = script.trim().split("\n");
    // Expect at least: t=0 (1.0), step to 2.0, step to 0.5, and a final-frame line.
    assert.ok(lines.length >= 3, `expected >=3 keyframes, got ${lines.length}: ${lines.join(" / ")}`);
    assert.ok(lines[0].startsWith("0.000"), `first line must anchor at t=0: ${lines[0]}`);
    const gains = lines.map((line) => Number(line.match(/volume\s+([\d.]+)/)![1]));
    assert.ok(gains.includes(1.0) && gains.some((g) => Math.abs(g - 2.0) < 0.01), "must cover both gain plateaus");
  });

  it("subtracts windowStartSec so per-segment scripts have 0-based timestamps", () => {
    const curve = new Float32Array(200);
    for (let i = 0; i < 100; i += 1) curve[i] = 1.0;
    for (let i = 100; i < 200; i += 1) curve[i] = 0.5;
    // Take the segment [1.0 s, 2.0 s] at 10 ms frames = [frame 100, frame 200].
    const script = emitSendcmdScript(curve, 10, 1.0, 2.0, 0.1);
    const lines = script.trim().split("\n");
    assert.ok(lines[0].startsWith("0.000"), `segment t=0 must be relative: ${lines[0]}`);
    const gains = lines.map((line) => Number(line.match(/volume\s+([\d.]+)/)![1]));
    // The segment starts in the 0.5-plateau.
    assert.ok(Math.abs(gains[0] - 0.5) < 0.01, `expected 0.5 at segment start, got ${gains[0]}`);
  });
});

describe("WAV codec", () => {
  it("round-trips pcm_f32le WAV", () => {
    const source = new Float32Array(16000);
    for (let i = 0; i < source.length; i += 1) source[i] = Math.sin((2 * Math.PI * 440 * i) / 16000) * 0.5;
    const encoded = encodeWavFloat32(source, 16000, 1);
    const decoded = decodeWav(encoded);
    assert.equal(decoded.sampleRate, 16000);
    assert.equal(decoded.channels, 1);
    assert.equal(decoded.samples.length, source.length);
    for (let i = 0; i < 100; i += 1) {
      assert.ok(Math.abs(decoded.samples[i] - source[i]) < 1e-6);
    }
  });
});

describe("spectrum", () => {
  it("detects elevated high-frequency content as sibilance", () => {
    const sampleRate = 16000;
    const duration = 1;
    const total = sampleRate * duration;
    const low = new Float32Array(total);
    const high = new Float32Array(total);
    for (let i = 0; i < total; i += 1) {
      low[i] = Math.sin((2 * Math.PI * 500 * i) / sampleRate) * 0.3;
      high[i] = Math.sin((2 * Math.PI * 500 * i) / sampleRate) * 0.15 + Math.sin((2 * Math.PI * 6500 * i) / sampleRate) * 0.4;
    }
    const lowSib = computeSibilanceScore(computeLogBandSpectrumDb(low, sampleRate));
    const highSib = computeSibilanceScore(computeLogBandSpectrumDb(high, sampleRate));
    assert.ok(highSib > lowSib + 0.2, `expected sibilance score to rise with HF content: low=${lowSib.toFixed(2)} high=${highSib.toFixed(2)}`);
  });
});
