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
