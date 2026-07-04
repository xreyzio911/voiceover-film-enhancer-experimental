/**
 * Lightweight long-term spectrum analyzer for tone matching across VO files.
 *
 * Computes the average energy in 8 log-spaced bands across the entire file.
 * Used by the batch-reference tone matcher to align every file toward a
 * common timbre, and (optionally) by the cinematic-color EQ curve.
 *
 * Intentionally hand-rolled DFT per band (Goertzel-style grouped into a
 * single pass over short hamming-windowed frames) — no FFT dep, works in
 * Node tests, and avoids allocating per-frame spectrum buffers.
 */

export const SPECTRUM_BANDS_HZ = [60, 120, 250, 500, 1000, 2000, 4000, 8000] as const;
export type SpectrumBandsHz = typeof SPECTRUM_BANDS_HZ;
export const HOUSE_TONE_BLEND = 0.35;
export const CINEMATIC_VO_REFERENCE_DB = [-28, -23, -21, -22, -23, -21.5, -20.5, -24] as const;

export const DE_ESSER_PLACEMENTS = {
  lowCenter: { mainHz: 5800, secondaryHz: 8200 },
  balanced: { mainHz: 6500, secondaryHz: 9000 },
  highCenter: { mainHz: 7200, secondaryHz: 9800 },
} as const;

/** Width of each band in octaves. 1.0 = one octave wide, centered on band Hz. */
const DEFAULT_BAND_WIDTH_OCT = 0.7;
const FRAME_MS = 20;
const HOP_MS = 20;
export const DEFAULT_MAX_SPECTRUM_FRAMES = 1600;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

type BandFilter = {
  centerHz: number;
  lowHz: number;
  highHz: number;
  /** Goertzel-like summed energy across a coarse grid inside the band. */
  grid: Array<{ cosOmega: number; sinOmega: number }>;
};

const buildBandFilters = (sampleRate: number, bands: readonly number[], widthOct: number): BandFilter[] => {
  const nyquist = sampleRate / 2;
  return bands.map((centerHz) => {
    const lowHz = Math.max(20, centerHz * Math.pow(2, -widthOct / 2));
    const highHz = Math.min(nyquist - 10, centerHz * Math.pow(2, widthOct / 2));
    const gridPoints = 5;
    const grid: Array<{ cosOmega: number; sinOmega: number }> = [];
    for (let i = 0; i < gridPoints; i += 1) {
      const t = i / (gridPoints - 1);
      const hz = lowHz + (highHz - lowHz) * t;
      const omega = (2 * Math.PI * hz) / sampleRate;
      grid.push({ cosOmega: Math.cos(omega), sinOmega: Math.sin(omega) });
    }
    return { centerHz, lowHz, highHz, grid };
  });
};

export const resolveSpectrumFrameBudget = (
  sampleCount: number,
  sampleRate: number,
  options?: { maxFrames?: number; frameStride?: number },
) => {
  const frameSize = Math.max(32, Math.round((sampleRate * FRAME_MS) / 1000));
  const hopSize = Math.max(1, Math.round((sampleRate * HOP_MS) / 1000));
  const totalFrames = sampleCount >= frameSize ? Math.floor((sampleCount - frameSize) / hopSize) + 1 : 0;
  const maxFrames = Math.max(1, Math.floor(options?.maxFrames ?? DEFAULT_MAX_SPECTRUM_FRAMES));
  const frameStride =
    typeof options?.frameStride === "number" && Number.isFinite(options.frameStride) && options.frameStride > 0
      ? Math.max(1, Math.floor(options.frameStride))
      : Math.max(1, Math.ceil(totalFrames / maxFrames));
  const framesToVisit = totalFrames === 0 ? 0 : Math.ceil(totalFrames / frameStride);
  return { frameSize, hopSize, totalFrames, frameStride, framesToVisit };
};

/**
 * Compute per-band average energy (dB) across the entire signal.
 * Returns one number per band in SPECTRUM_BANDS_HZ order.
 */
export const computeLogBandSpectrumDb = (
  samples: Float32Array,
  sampleRate: number,
  options?: { bands?: readonly number[]; widthOct?: number; maxFrames?: number; frameStride?: number },
): number[] => {
  const bandsHz = options?.bands ?? SPECTRUM_BANDS_HZ;
  const widthOct = options?.widthOct ?? DEFAULT_BAND_WIDTH_OCT;
  const filters = buildBandFilters(sampleRate, bandsHz, widthOct);
  const { frameSize, hopSize, totalFrames, frameStride } = resolveSpectrumFrameBudget(samples.length, sampleRate, options);

  // Hann window for modest leakage suppression.
  const window = new Float32Array(frameSize);
  for (let i = 0; i < frameSize; i += 1) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frameSize - 1)));
  }

  const bandSumPower = new Array<number>(filters.length).fill(0);
  let framesCounted = 0;

  for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += frameStride) {
    const frameStart = frameIndex * hopSize;
    // Skip near-silent frames so background hiss doesn't dominate the median.
    let rms = 0;
    for (let i = 0; i < frameSize; i += 1) {
      const v = samples[frameStart + i];
      rms += v * v;
    }
    rms = Math.sqrt(rms / frameSize);
    if (rms < 1e-4) continue;

    framesCounted += 1;

    for (let b = 0; b < filters.length; b += 1) {
      let totalPower = 0;
      for (const { cosOmega, sinOmega } of filters[b].grid) {
        let re = 0;
        let im = 0;
        // DFT at this single frequency over the windowed frame.
        // Iterative rotation avoids Math.cos/sin per sample.
        let c = 1;
        let s = 0;
        for (let i = 0; i < frameSize; i += 1) {
          const v = samples[frameStart + i] * window[i];
          re += v * c;
          im += v * s;
          const nc = c * cosOmega - s * sinOmega;
          const ns = s * cosOmega + c * sinOmega;
          c = nc;
          s = ns;
        }
        totalPower += (re * re + im * im) / (frameSize * frameSize);
      }
      bandSumPower[b] += totalPower / filters[b].grid.length;
    }
  }

  if (framesCounted === 0) return bandsHz.map(() => -120);

  return bandSumPower.map((p) => {
    const avg = p / framesCounted;
    return 10 * Math.log10(avg + 1e-30);
  });
};

/**
 * Compute a simple sibilance score from band energy ratios.
 * High-frequency (≥4 kHz) peaks relative to 1–2 kHz body indicate harsh sibilance.
 * Returns 0..1.
 */
export const computeSibilanceScore = (bandDb: number[]): number => {
  // bands: [60, 120, 250, 500, 1000, 2000, 4000, 8000]
  // body: mean(1k, 2k) ; sibilance: mean(4k, 8k)
  if (bandDb.length < 8) return 0;
  const body = (bandDb[4] + bandDb[5]) / 2;
  const sib = (bandDb[6] + bandDb[7]) / 2;
  const ratio = sib - body; // positive => bright/sibilant
  return clamp((ratio + 1) / 9, 0, 1); // 0 at ratio=-1dB, 1 at ratio=+8dB
};

export const resolveDeEsserBands = (bandDb: number[]) => {
  if (bandDb.length < 8) return DE_ESSER_PLACEMENTS.balanced;
  const body = (bandDb[4] + bandDb[5]) / 2;
  const lowSibilance = bandDb[6] - body;
  const highSibilance = bandDb[7] - body;
  const centerBiasDb = highSibilance - lowSibilance;
  if (centerBiasDb >= 2) return DE_ESSER_PLACEMENTS.highCenter;
  if (centerBiasDb <= -2) return DE_ESSER_PLACEMENTS.lowCenter;
  return DE_ESSER_PLACEMENTS.balanced;
};

/**
 * Build a per-band corrective EQ delta (in dB) that nudges `fileDb` toward `referenceDb`.
 * Output is one delta per band, clamped to +/- maxDb.
 */
export const computeToneMatchDeltaDb = (
  fileDb: number[],
  referenceDb: number[],
  optionsOrMaxDb:
    | number
    | {
        maxDb?: number;
        houseBlend?: number;
        houseReferenceDb?: readonly number[];
        priorityBandCount?: number;
        priorityMaxDb?: number;
      } = 3,
): number[] => {
  const n = Math.min(fileDb.length, referenceDb.length);
  const options = typeof optionsOrMaxDb === "number" ? { maxDb: optionsOrMaxDb } : optionsOrMaxDb;
  const maxDb = options.maxDb ?? 3;
  const houseBlend = clamp(options.houseBlend ?? 0, 0, 1);
  const houseReferenceDb = options.houseReferenceDb ?? CINEMATIC_VO_REFERENCE_DB;
  const priorityBandCount = Math.max(0, Math.floor(options.priorityBandCount ?? (houseBlend > 0 ? 2 : 0)));
  const priorityMaxDb = options.priorityMaxDb ?? Math.max(maxDb, 3);
  const effectiveReferenceDb = referenceDb.slice(0, n).map((value, index) => {
    const houseValue = houseReferenceDb[index];
    return typeof houseValue === "number" && Number.isFinite(houseValue)
      ? value * (1 - houseBlend) + houseValue * houseBlend
      : value;
  });
  // Normalize out the mean difference (that's a loudness offset, not tone).
  const meanFile = fileDb.slice(0, n).reduce((a, b) => a + b, 0) / n;
  const meanRef = effectiveReferenceDb.reduce((a, b) => a + b, 0) / n;
  const offset = meanRef - meanFile;
  const raw = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    raw[i] = effectiveReferenceDb[i] - fileDb[i] - offset;
  }
  const priorityIndexes = new Set(
    raw
      .map((value, index) => ({ value: Math.abs(value), index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, priorityBandCount)
      .map((item) => item.index),
  );
  return raw.map((value, index) => {
    const cap = priorityIndexes.has(index) ? priorityMaxDb : maxDb;
    return clamp(value, -cap, cap);
  });
};
