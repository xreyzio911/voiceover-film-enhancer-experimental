import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  estimateResidualEchoTap,
  suppressSevereEchoResidual,
} from "./echoResidualSuppression.ts";

const rms = (samples: Float32Array, start: number, end: number) => {
  let sumSquares = 0;
  for (let index = start; index < end; index += 1) {
    const value = samples[index] ?? 0;
    sumSquares += value * value;
  }
  return Math.sqrt(sumSquares / Math.max(end - start, 1));
};

const makeEchoTailFixture = (sampleRate: number) => {
  const samples = new Float32Array(sampleRate * 2);
  const speechStart = Math.round(sampleRate * 0.1);
  const speechEnd = Math.round(sampleRate * 0.45);
  const tailEnd = Math.round(sampleRate * 0.82);
  for (let index = speechStart; index < speechEnd; index += 1) {
    samples[index] = Math.sin((index / sampleRate) * Math.PI * 2 * 220) * 0.42;
  }
  for (let index = speechEnd; index < tailEnd; index += 1) {
    const progress = (index - speechEnd) / Math.max(tailEnd - speechEnd, 1);
    samples[index] += Math.sin((index / sampleRate) * Math.PI * 2 * 220) * 0.18 * (1 - progress);
  }
  return samples;
};

describe("severe echo residual suppression", () => {
  it("suppresses post-speech residual tails while preserving speech body", () => {
    const sampleRate = 16000;
    const source = makeEchoTailFixture(sampleRate);
    const result = suppressSevereEchoResidual(source, sampleRate, { echoDelayMs: 40 });

    const speechBefore = rms(source, Math.round(sampleRate * 0.2), Math.round(sampleRate * 0.3));
    const speechAfter = rms(result.samples, Math.round(sampleRate * 0.2), Math.round(sampleRate * 0.3));
    const tailBefore = rms(source, Math.round(sampleRate * 0.56), Math.round(sampleRate * 0.7));
    const tailAfter = rms(result.samples, Math.round(sampleRate * 0.56), Math.round(sampleRate * 0.7));

    assert.equal(result.samples.length, source.length);
    assert.ok(result.tailFramesSuppressed > 0);
    assert.ok(tailAfter < tailBefore * 0.99);
    assert.ok(speechAfter > speechBefore * 0.9);
  });

  it("reduces residual tails even when processed pauses are digital black", () => {
    const sampleRate = 16000;
    const source = new Float32Array(sampleRate * 2);
    const speechStart = Math.round(sampleRate * 0.1);
    const speechEnd = Math.round(sampleRate * 0.5);
    const tailEnd = Math.round(sampleRate * 0.82);

    for (let index = speechStart; index < speechEnd; index += 1) {
      source[index] = Math.sin((index / sampleRate) * Math.PI * 2 * 210) * 0.24;
    }
    for (let index = speechEnd; index < tailEnd; index += 1) {
      const progress = (index - speechEnd) / Math.max(tailEnd - speechEnd, 1);
      source[index] = Math.sin((index / sampleRate) * Math.PI * 2 * 210) * 0.055 * (1 - progress);
    }

    const result = suppressSevereEchoResidual(source, sampleRate, { echoDelayMs: 40 });

    const speechBefore = rms(source, Math.round(sampleRate * 0.22), Math.round(sampleRate * 0.34));
    const speechAfter = rms(result.samples, Math.round(sampleRate * 0.22), Math.round(sampleRate * 0.34));
    const tailBefore = rms(source, Math.round(sampleRate * 0.58), Math.round(sampleRate * 0.72));
    const tailAfter = rms(result.samples, Math.round(sampleRate * 0.58), Math.round(sampleRate * 0.72));

    assert.ok(result.tailFramesSuppressed > 0);
    assert.ok(result.finishingTailFramesSuppressed > 0);
    assert.ok(tailAfter < tailBefore * 0.82);
    assert.ok(speechAfter > speechBefore * 0.8);
  });

  it("preserves a soft final consonant before suppressing the room tail", () => {
    const sampleRate = 16000;
    const source = new Float32Array(sampleRate * 2);
    const speechStart = Math.round(sampleRate * 0.1);
    const speechEnd = Math.round(sampleRate * 0.5);
    const consonantEnd = Math.round(sampleRate * 0.59);
    const tailEnd = Math.round(sampleRate * 0.88);

    for (let index = speechStart; index < speechEnd; index += 1) {
      source[index] = Math.sin((index / sampleRate) * Math.PI * 2 * 220) * 0.22;
    }
    for (let index = speechEnd; index < consonantEnd; index += 1) {
      source[index] += Math.sin((index / sampleRate) * Math.PI * 2 * 3600) * 0.0048;
    }
    for (let index = consonantEnd; index < tailEnd; index += 1) {
      const progress = (index - consonantEnd) / Math.max(tailEnd - consonantEnd, 1);
      source[index] += Math.sin((index / sampleRate) * Math.PI * 2 * 220) * 0.045 * (1 - progress);
    }

    const result = suppressSevereEchoResidual(source, sampleRate, { echoDelayMs: 40 });

    const consonantBefore = rms(source, speechEnd, consonantEnd);
    const consonantAfter = rms(result.samples, speechEnd, consonantEnd);
    const tailBefore = rms(source, Math.round(sampleRate * 0.68), Math.round(sampleRate * 0.82));
    const tailAfter = rms(result.samples, Math.round(sampleRate * 0.68), Math.round(sampleRate * 0.82));

    assert.ok(
      consonantAfter > consonantBefore * 0.72,
      `final consonant should not be gated like echo: ${consonantAfter} vs ${consonantBefore}`,
    );
    assert.ok(tailAfter < tailBefore * 0.9, `room tail should still reduce: ${tailAfter} vs ${tailBefore}`);
  });

  it("bounds single-delay tap estimation", () => {
    const samples = new Float32Array(2000);
    for (let index = 80; index < samples.length; index += 1) {
      samples[index] = 4 * (samples[index - 80] || 0) + Math.sin(index * 0.05) * 0.1;
    }

    assert.equal(estimateResidualEchoTap(samples, 80), 0.12);
  });

  it("returns a same-length copy when echo delay is unavailable", () => {
    const samples = new Float32Array([0, 0.1, -0.1, 0]);
    const result = suppressSevereEchoResidual(samples, 16000, { echoDelayMs: null });

    assert.notEqual(result.samples, samples);
    assert.deepEqual(Array.from(result.samples), Array.from(samples));
    assert.equal(result.echoDelaySamples, 0);
    assert.equal(result.echoTapCoeff, 0);
  });
});
