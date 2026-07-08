/**
 * Quiet-speech destruction / passthrough diagnostic.
 *
 * Compares a rendered mix-ready WAV against its source and reports:
 *   1. Collapsed-speech clusters (source speech that the render erased/crushed)
 *   2. Per-speech-run level deltas with ERASED / CRUSHED markers
 *   3. Whether the render actually leveled anything (passthrough detection)
 *   4. Global time shift estimate between render and source
 *   5. A planner reproduction on the source (expander depth, mask coverage)
 *
 * Usage:
 *   node --experimental-strip-types --max-old-space-size=6144 scripts/diagQuietSpeech.mts [source.wav rendered.wav]
 *
 * With no args it runs against the known-bad pair in "end spiked down/".
 * Exit code 1 when the render fails a check (collapse, passthrough), 0 when clean.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { decodeWav } from "../src/lib/webAudioRender.ts";
import { analyzeFloatSamples, buildSpeechMask } from "../src/lib/audioQc.ts";
import { applyKWeighting, planGainCurve, speechRunsFromMask, type SpeechRun } from "../src/lib/gainPlanner.ts";
import { detectAudibilityDropouts, frameDbFromFloatSamples } from "../src/lib/audibilityDropout.ts";

const ROOT = path.resolve(import.meta.dirname, "..");
const DEFAULT_SRC = path.join(ROOT, "end spiked down", "Lucas Martin_batchvideo1-10.wav");
const DEFAULT_OUT = path.join(ROOT, "end spiked down", "Lucas_Martin_batchvideo1-10_mixready.wav");
const [, , srcArg, outArg] = process.argv;
const SRC = srcArg ? path.resolve(srcArg) : DEFAULT_SRC;
const OUT = outArg ? path.resolve(outArg) : DEFAULT_OUT;

const FRAME_MS = 10;
const GUARD_MS = 20;

const loadMono = (file: string) => {
  const bytes = readFileSync(file);
  const decoded = decodeWav(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  let mono = decoded.samples;
  if (decoded.channels > 1) {
    const frames = Math.floor(decoded.samples.length / decoded.channels);
    mono = new Float32Array(frames);
    for (let i = 0; i < frames; i += 1) {
      let sum = 0;
      for (let c = 0; c < decoded.channels; c += 1) sum += decoded.samples[i * decoded.channels + c];
      mono[i] = sum / decoded.channels;
    }
  }
  console.log(
    `${path.basename(file)}: sr=${decoded.sampleRate} ch=${decoded.channels} dur=${(mono.length / decoded.sampleRate).toFixed(1)}s`,
  );
  return { mono, sampleRate: decoded.sampleRate };
};

const rmsDbOfSlice = (frameDb: number[], a: number, b: number) => {
  let sum = 0;
  let n = 0;
  for (let i = Math.max(0, a); i < Math.min(frameDb.length, b); i += 1) {
    sum += Math.pow(10, frameDb[i] / 10);
    n += 1;
  }
  return n > 0 ? 10 * Math.log10(sum / n + 1e-30) : -240;
};

const src = loadMono(SRC);
const out = loadMono(OUT);
const SR = src.sampleRate;
let failed = false;

// ---------- 1) Source QC + planner reproduction ----------
const metrics = analyzeFloatSamples(src.mono, SR, FRAME_MS);
const segSilenceDb = Math.min(-28, Math.max(-48, Math.max(metrics.noiseFloorDb, metrics.speechThresholdDb, -70) + 14));
console.log(
  `[QC] noiseFloor=${metrics.noiseFloorDb.toFixed(1)} pauseFloor=${metrics.pauseNoiseFloorDb.toFixed(1)} ` +
    `speechThresh=${metrics.speechThresholdDb.toFixed(1)} pauseNoiseRisk=${metrics.pauseNoiseRisk.toFixed(2)} ` +
    `speechDuty=${metrics.speechDutyCyclePct.toFixed(1)}% | app silencedetect threshold would be ${segSilenceDb.toFixed(1)} dB`,
);

const srcFrame10 = frameDbFromFloatSamples(src.mono, SR, FRAME_MS);
const mask = buildSpeechMask(srcFrame10, metrics.noiseFloorDb, { frameMs: FRAME_MS });
const speechRuns: SpeechRun[] = speechRunsFromMask(mask);
const plan = planGainCurve({
  frameDb: srcFrame10,
  loudnessFrameDb: frameDbFromFloatSamples(applyKWeighting(src.mono, SR), SR, FRAME_MS),
  speechRuns,
  noiseFloorDb: metrics.noiseFloorDb,
  speechThresholdDb: metrics.speechThresholdDb,
  pauseNoiseRisk: metrics.pauseNoiseRisk,
  frameMs: FRAME_MS,
  samples: src.mono,
  sampleRate: SR,
  targetDb: -22,
  sourceTargetBlend: 0.1,
  maxGainDb: 14,
  peakCeilingDb: -3,
  instabilityHint: Math.min(
    1,
    metrics.instabilityScore * 0.5 + metrics.lineSwingScore * 0.3 + metrics.sentenceJumpScore * 0.2,
  ),
});
console.log(
  `[PlannerRepro] runs=${plan.runs.length} target=${plan.targetDb.toFixed(1)}dB expanderDepth=${plan.expanderDepthDb.toFixed(1)}dB tailRescues=${plan.tailRescueRunCount}`,
);

// ---------- 2) Collapse clusters (audibility guard on the pair) ----------
const srcFrame20 = frameDbFromFloatSamples(src.mono, SR, GUARD_MS);
const outFrame20 = frameDbFromFloatSamples(out.mono, SR, GUARD_MS);
const report = detectAudibilityDropouts({ sourceFrameDb: srcFrame20, renderedFrameDb: outFrame20, frameMs: GUARD_MS });
console.log(
  `[Collapse] severe=${report.severe} badSeconds=${report.badSeconds.toFixed(2)} clusters=${report.clusterCount} worstDrop=${report.worstDropDb.toFixed(1)}dB`,
);
for (const c of report.clusters
  .slice()
  .sort((a, b) => a.minDropDb - b.minDropDb)
  .slice(0, 10)) {
  const f10a = Math.floor((c.startSec * 1000) / FRAME_MS);
  const f10b = Math.ceil((c.endSec * 1000) / FRAME_MS);
  let inMask = 0;
  let minG = Infinity;
  for (let f = f10a; f < f10b; f += 1) {
    if (mask[f]) inMask += 1;
    const g = 20 * Math.log10(plan.gainCurve[f] ?? 1);
    if (g < minG) minG = g;
  }
  console.log(
    `  ${c.startSec.toFixed(2)}-${c.endSec.toFixed(2)}s srcMax=${c.maxSourceDb.toFixed(1)} rendered=${c.minRenderedDb.toFixed(1)} ` +
      `drop=${c.minDropDb.toFixed(1)}dB plannerGainMin=${minG.toFixed(1)}dB inMask=${inMask}/${f10b - f10a}`,
  );
}
if (report.severe) failed = true;

// ---------- 3) Per-run delta table ----------
const outFrame10 = frameDbFromFloatSamples(out.mono, SR, FRAME_MS);
let erased = 0;
let crushed = 0;
for (let i = 0; i < speechRuns.length; i += 1) {
  const r = speechRuns[i];
  const srcDb = rmsDbOfSlice(srcFrame10, r.startFrame, r.endFrame);
  const outDb = rmsDbOfSlice(outFrame10, r.startFrame, r.endFrame);
  const delta = outDb - srcDb;
  if (outDb <= -100) {
    erased += 1;
    console.log(
      `  [ERASED]  run#${i} ${(r.startFrame / 100).toFixed(2)}-${(r.endFrame / 100).toFixed(2)}s src=${srcDb.toFixed(1)} out=${outDb.toFixed(1)}`,
    );
  } else if (delta <= -12) {
    crushed += 1;
    console.log(
      `  [CRUSHED] run#${i} ${(r.startFrame / 100).toFixed(2)}-${(r.endFrame / 100).toFixed(2)}s src=${srcDb.toFixed(1)} out=${outDb.toFixed(1)} delta=${delta.toFixed(1)}`,
    );
  }
}
console.log(`[Runs] total=${speechRuns.length} erased=${erased} crushed=${crushed}`);
if (erased + crushed > 0) failed = true;

// ---------- 4) Passthrough detection ----------
const deltas: number[] = [];
const quietDeltas: number[] = [];
for (let i = 0; i < speechRuns.length; i += 1) {
  const r = speechRuns[i];
  const srcDb = rmsDbOfSlice(srcFrame10, r.startFrame, r.endFrame);
  const outDb = rmsDbOfSlice(outFrame10, r.startFrame, r.endFrame);
  if (outDb <= -100) continue;
  deltas.push(outDb - srcDb);
  if (srcDb <= -26) quietDeltas.push(outDb - srcDb);
}
deltas.sort((a, b) => a - b);
quietDeltas.sort((a, b) => a - b);
const median = (arr: number[]) => (arr.length > 0 ? arr[Math.floor(arr.length / 2)] : NaN);
const medAll = median(deltas);
const medQuiet = median(quietDeltas);
// A leveling render must materially change the speech-body distribution. It
// may lift quiet lines, pull hot lines down, or do both; a true passthrough
// leaves both the raw-quiet subset and the full run set nearly unchanged.
const quietChanged = Number.isFinite(medQuiet) && Math.abs(medQuiet) >= 1.5;
const globalChanged = Number.isFinite(medAll) && Math.abs(medAll) >= 1.5;
const passthrough = !quietChanged && !globalChanged;
console.log(
  `[Leveling] run-delta median all=${medAll.toFixed(1)}dB quietRuns(<=-26dB src)=${medQuiet.toFixed(1)}dB -> ${
    passthrough ? "PASSTHROUGH (no leveling!)" : "leveled"
  }`,
);
if (passthrough) failed = true;

// ---------- 5) Global time shift ----------
const findShift = (centerSec: number) => {
  const win = Math.round(SR * 1.0);
  const t0 = Math.round(centerSec * SR - win / 2);
  if (t0 < 0 || t0 + win > src.mono.length) return null;
  const template = src.mono.subarray(t0, t0 + win);
  let tEnergy = 0;
  for (let i = 0; i < template.length; i += 1) tEnergy += template[i] * template[i];
  if (tEnergy < 1e-6) return null;
  let bestCorr = 0;
  let bestShift = NaN;
  const radius = Math.round(SR * 1.5);
  for (let shift = -radius; shift <= radius; shift += 48) {
    const o0 = t0 + shift;
    if (o0 < 0 || o0 + win > out.mono.length) continue;
    let dot = 0;
    let oEnergy = 0;
    for (let i = 0; i < win; i += 16) {
      dot += template[i] * out.mono[o0 + i];
      oEnergy += out.mono[o0 + i] * out.mono[o0 + i];
    }
    const corr = dot / Math.sqrt((tEnergy / 16) * Math.max(oEnergy, 1e-12));
    if (Math.abs(corr) > Math.abs(bestCorr)) {
      bestCorr = corr;
      bestShift = shift / SR;
    }
  }
  return { shiftSec: bestShift, corr: bestCorr };
};
const shiftProbes: number[] = [];
for (const r of plan.runs.filter((x) => x.meanDb > -30).slice(0, 8)) {
  const probe = findShift(((r.startFrame + r.endFrame) / 2) / 100);
  if (probe && Math.abs(probe.corr) >= 0.15 && Number.isFinite(probe.shiftSec)) shiftProbes.push(probe.shiftSec);
}
shiftProbes.sort((a, b) => a - b);
const medShift = median(shiftProbes);
console.log(
  `[Shift] median rendered-vs-source shift=${Number.isFinite(medShift) ? (medShift * 1000).toFixed(0) : "n/a"}ms over ${shiftProbes.length} probes`,
);
if (Number.isFinite(medShift) && Math.abs(medShift) > 0.03) failed = true;

console.log(`\n[Verdict] ${failed ? "FAIL — render destroys or ignores quiet speech" : "PASS"}`);
process.exit(failed ? 1 : 0);
