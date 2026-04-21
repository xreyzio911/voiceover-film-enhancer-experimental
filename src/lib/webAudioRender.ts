/**
 * WAV codec helpers for the gain-planner pre-render step.
 *
 * Intentionally does NOT use OfflineAudioContext — pure DataView is faster,
 * avoids browser-only APIs (so tests run in Node), and has no allocation
 * surprises from browser AudioBuffer conversion.
 *
 * Supported formats (read): pcm_s16le, pcm_s24le, pcm_s32le, pcm_f32le, pcm_f64le.
 * Supported format (write): pcm_f32le (best fidelity for downstream ffmpeg stages).
 */

export type DecodedWav = {
  samples: Float32Array;
  sampleRate: number;
  channels: number;
};

const readString = (view: DataView, offset: number, length: number) => {
  let s = "";
  for (let i = 0; i < length; i += 1) {
    s += String.fromCharCode(view.getUint8(offset + i));
  }
  return s;
};

/**
 * Parse a RIFF/WAVE byte buffer into interleaved Float32 samples.
 */
export const decodeWav = (buffer: ArrayBufferLike | Uint8Array): DecodedWav => {
  const ab = buffer instanceof Uint8Array ? buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) : buffer;
  const view = new DataView(ab as ArrayBuffer);

  if (view.byteLength < 44) throw new Error("WAV too small");
  if (readString(view, 0, 4) !== "RIFF") throw new Error("Not a RIFF file");
  if (readString(view, 8, 4) !== "WAVE") throw new Error("Not a WAVE file");

  let offset = 12;
  let fmt: { audioFormat: number; channels: number; sampleRate: number; bitsPerSample: number } | null = null;
  let dataOffset = -1;
  let dataLength = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readString(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkId === "fmt ") {
      const audioFormat = view.getUint16(chunkStart, true);
      const channels = view.getUint16(chunkStart + 2, true);
      const sampleRate = view.getUint32(chunkStart + 4, true);
      const bitsPerSample = view.getUint16(chunkStart + 14, true);
      fmt = { audioFormat, channels, sampleRate, bitsPerSample };
    } else if (chunkId === "data") {
      dataOffset = chunkStart;
      dataLength = chunkSize;
      break;
    }
    offset = chunkStart + chunkSize + (chunkSize % 2);
  }

  if (!fmt || dataOffset < 0) throw new Error("WAV missing fmt or data chunk");

  const { audioFormat, channels, sampleRate, bitsPerSample } = fmt;
  // Format codes: 1 = PCM int, 3 = IEEE float, 0xFFFE = extensible.
  const isFloat = audioFormat === 3 || (audioFormat === 0xfffe && bitsPerSample >= 32);
  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(dataLength / bytesPerSample);
  const out = new Float32Array(totalSamples);

  if (isFloat && bitsPerSample === 32) {
    for (let i = 0; i < totalSamples; i += 1) {
      out[i] = view.getFloat32(dataOffset + i * 4, true);
    }
  } else if (isFloat && bitsPerSample === 64) {
    for (let i = 0; i < totalSamples; i += 1) {
      out[i] = view.getFloat64(dataOffset + i * 8, true);
    }
  } else if (bitsPerSample === 16) {
    for (let i = 0; i < totalSamples; i += 1) {
      out[i] = view.getInt16(dataOffset + i * 2, true) / 0x8000;
    }
  } else if (bitsPerSample === 24) {
    for (let i = 0; i < totalSamples; i += 1) {
      const o = dataOffset + i * 3;
      const lo = view.getUint8(o);
      const mid = view.getUint8(o + 1);
      const hi = view.getInt8(o + 2);
      const v = (hi << 16) | (mid << 8) | lo;
      out[i] = v / 0x800000;
    }
  } else if (bitsPerSample === 32) {
    for (let i = 0; i < totalSamples; i += 1) {
      out[i] = view.getInt32(dataOffset + i * 4, true) / 0x80000000;
    }
  } else {
    throw new Error(`Unsupported WAV sample width ${bitsPerSample}-bit fmt=${audioFormat}`);
  }

  return { samples: out, sampleRate, channels };
};

/**
 * Encode interleaved Float32 samples as pcm_f32le WAV.
 */
export const encodeWavFloat32 = (samples: Float32Array, sampleRate: number, channels: number): Uint8Array => {
  const byteRate = sampleRate * channels * 4;
  const blockAlign = channels * 4;
  const dataBytes = samples.length * 4;
  const totalSize = 44 + dataBytes;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i += 1) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, totalSize - 8, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 3, true); // IEEE float
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 32, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataBytes, true);

  for (let i = 0; i < samples.length; i += 1) {
    view.setFloat32(44 + i * 4, samples[i], true);
  }

  return new Uint8Array(buffer);
};
