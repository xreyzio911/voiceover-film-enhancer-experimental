import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import JSZip from "jszip";
import {
  readWavInfoFromBytes,
  splitBatchAudioTracks,
  type AudioSplitterEngine,
  type AudioSplitterEngineBatchItem,
  type AudioSplitterInput,
} from "./audioSplitterService.ts";
import { decodeWav, encodeWavFloat32 } from "./webAudioRender.ts";

const makeWavBytes = (frames = 800, sampleRate = 8000, channels = 1) => {
  const samples = new Float32Array(frames * channels);
  for (let frame = 0; frame < frames; frame += 1) {
    const value = Math.sin((frame / sampleRate) * Math.PI * 2 * 440) * 0.2;
    for (let channel = 0; channel < channels; channel += 1) {
      samples[frame * channels + channel] = value;
    }
  }
  return encodeWavFloat32(samples, sampleRate, channels);
};

const scaleSamples = (samples: Float32Array, gain: number) => {
  const scaled = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    scaled[i] = samples[i] * gain;
  }
  return scaled;
};

const makeInput = (originalName: string, bytes = makeWavBytes()): AudioSplitterInput => ({
  originalName,
  bytes,
});

const makeFakeEngine = (failNames = new Set<string>()): AudioSplitterEngine => ({
  name: "fake-splitter",
  async split(inputPath, workDir, context) {
    if (failNames.has(context.originalName)) {
      throw new Error("Injected engine failure.");
    }

    await mkdir(workDir, { recursive: true });
    const decoded = decodeWav(await readFile(inputPath));
    const vocal = path.join(workDir, "vocals.wav");
    const nonVocal = path.join(workDir, "no_vocals.wav");

    await writeFile(vocal, encodeWavFloat32(scaleSamples(decoded.samples, 0.5), decoded.sampleRate, decoded.channels));
    await writeFile(nonVocal, encodeWavFloat32(scaleSamples(decoded.samples, 0.5), decoded.sampleRate, decoded.channels));

    return { vocal, nonVocal };
  },
});

const loadZip = async (buffer: Buffer) => JSZip.loadAsync(buffer);

const readZipWav = async (zip: JSZip, name: string) => {
  const file = zip.file(name);
  assert.ok(file, `Expected ${name} in zip`);
  return decodeWav(await file.async("uint8array"));
};

const readZipWavInfo = async (zip: JSZip, name: string) => {
  const file = zip.file(name);
  assert.ok(file, `Expected ${name} in zip`);
  return readWavInfoFromBytes(await file.async("uint8array"));
};

test("creates two aligned stems for a single WAV and preserves filename text", async () => {
  const result = await splitBatchAudioTracks([makeInput("episode 01_scene-03.wav")], {
    engine: makeFakeEngine(),
    now: new Date("2026-04-28T00:00:00.000Z"),
  });
  const zip = await loadZip(result.zip);

  const bgm = await readZipWav(zip, "episode 01_scene-03_BGM.wav");
  const vocal = await readZipWav(zip, "episode 01_scene-03_VOCAL.wav");

  assert.equal(result.report.succeeded, 1);
  assert.equal(result.report.failed, 0);
  assert.equal(bgm.sampleRate, 8000);
  assert.equal(vocal.sampleRate, 8000);
  assert.equal(bgm.samples.length, vocal.samples.length);
  assert.equal(zip.file("episode 01_scene-03_SFX.wav"), null);
  assert.equal((await readZipWavInfo(zip, "episode 01_scene-03_BGM.wav")).bitsPerSample, 16);
  assert.equal((await readZipWavInfo(zip, "episode 01_scene-03_VOCAL.wav")).bitsPerSample, 16);
  assert.equal(result.report.files[0].stems.length, 2);
  assert.deepEqual(
    result.report.files[0].stems.map((stem) => stem.stem).sort(),
    ["BGM", "VOCAL"],
  );
  assert.ok(
    result.report.files[0].stems.every((stem) => stem.peakDbfs !== null && stem.rmsDbfs !== null && !stem.silent),
    "successful stems should include usable peak/RMS QC",
  );
  assert.ok(zip.file("split_report.txt"));
  assert.ok(zip.file("split_report.json"));
});

test("processes multiple files, long filenames, and dangerous character sanitization", async () => {
  const longBase = `episode ${"very-long-scene-name-".repeat(8)}final`;
  const result = await splitBatchAudioTracks(
    [makeInput("scene: 01?.wav"), makeInput(`${longBase}.wav`, makeWavBytes(1200, 16000, 2))],
    { engine: makeFakeEngine() },
  );
  const zip = await loadZip(result.zip);
  const names = Object.keys(zip.files);

  assert.equal(result.report.succeeded, 2);
  assert.ok(names.some((name) => name.endsWith("_BGM.wav") && !name.includes(":") && !name.includes("?")));
  assert.ok(zip.file(`${longBase}_VOCAL.wav`));
});

test("continues a batch when files are corrupted, unsupported, or fail separation", async () => {
  const result = await splitBatchAudioTracks(
    [
      makeInput("good.wav"),
      makeInput("corrupted.wav", new Uint8Array([1, 2, 3, 4])),
      makeInput("unsupported.mp3", makeWavBytes()),
      makeInput("engine-fail.wav"),
    ],
    { engine: makeFakeEngine(new Set(["engine-fail.wav"])) },
  );
  const zip = await loadZip(result.zip);

  assert.equal(result.report.succeeded, 1);
  assert.equal(result.report.failed, 3);
  assert.ok(zip.file("good_BGM.wav"));
  assert.ok(!zip.file("corrupted_BGM.wav"));
  assert.match(result.report.files[1].message, /too small|corrupted/i);
  assert.match(result.report.files[2].message, /\.wav files only/i);
  assert.match(result.report.files[3].message, /Injected engine failure/i);
});

test("blocks duplicate output filenames instead of overwriting zip entries", async () => {
  const result = await splitBatchAudioTracks([makeInput("dupe.wav"), makeInput("dupe.wav")], {
    engine: makeFakeEngine(),
  });
  const zip = await loadZip(result.zip);
  const outputNames = Object.keys(zip.files).filter((name) => name.endsWith(".wav"));

  assert.equal(result.report.succeeded, 1);
  assert.equal(result.report.failed, 1);
  assert.equal(outputNames.length, 2);
  assert.match(result.report.files[1].message, /Duplicate output filenames/i);
});

test("uses a batch engine once and emits per-file progress", async () => {
  let batchCalls = 0;
  const progress: string[] = [];
  const batchEngine: AudioSplitterEngine = {
    name: "fake-batch-splitter",
    async split() {
      throw new Error("Per-file split should not run when splitBatch exists.");
    },
    async splitBatch(items: AudioSplitterEngineBatchItem[], _workDir, context) {
      batchCalls += 1;
      const results = [];
      for (const item of items) {
        context.onProgress?.(item.inputIndex, `Separating ${item.originalName}`);
        await mkdir(item.workDir, { recursive: true });
        const decoded = decodeWav(await readFile(item.inputPath));
        const vocal = path.join(item.workDir, "vocals.wav");
        const bgm = path.join(item.workDir, "instrumental.wav");
        await writeFile(vocal, encodeWavFloat32(scaleSamples(decoded.samples, 0.4), decoded.sampleRate, decoded.channels));
        await writeFile(bgm, encodeWavFloat32(scaleSamples(decoded.samples, 0.6), decoded.sampleRate, decoded.channels));
        results.push({ inputIndex: item.inputIndex, rawStems: { vocal, bgm } });
      }
      return results;
    },
  };

  const result = await splitBatchAudioTracks([makeInput("first.wav"), makeInput("second file.wav")], {
    engine: batchEngine,
    onProgress(event) {
      if (event.type === "file-progress") progress.push(event.message);
    },
  });
  const zip = await loadZip(result.zip);

  assert.equal(batchCalls, 1);
  assert.equal(result.report.succeeded, 2);
  assert.deepEqual(progress, [
    "Starting fake-batch-splitter",
    "Separating first.wav",
    "Separating second file.wav",
  ]);
  assert.ok(zip.file("first_BGM.wav"));
  assert.ok(zip.file("second file_VOCAL.wav"));
  assert.equal(zip.file("first_SFX.wav"), null);
});

test(
  "audio-separator worker smoke test",
  { skip: process.env.RUN_AUDIO_SEPARATOR_WORKER_SMOKE !== "1" },
  async () => {
    const root = path.join(tmpdir(), `audio-separator-worker-${Date.now()}`);
    await mkdir(root, { recursive: true });
    const inputPath = path.join(root, "input.wav");
    const workDir = path.join(root, "engine");
    const manifestPath = path.join(root, "manifest.json");
    await mkdir(workDir, { recursive: true });
    await writeFile(inputPath, Buffer.from(makeWavBytes(529200, 44100, 1)));
    await writeFile(
      manifestPath,
      JSON.stringify({
        items: [
          {
            inputIndex: 0,
            originalName: "smoke.wav",
            inputPath,
            workDir,
            sampleRate: 44100,
          },
        ],
      }),
    );

    const workerPath = path.resolve("scripts/audio_separator_worker.py");
    const python = process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND ?? "python";
    const child = spawn(python, [
      workerPath,
      "--manifest",
      manifestPath,
      "--model",
      process.env.AUDIO_SPLITTER_AUDIO_SEPARATOR_MODEL ?? "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
      "--device",
      process.env.AUDIO_SPLITTER_DEVICE ?? "auto",
      "--output-format",
      "WAV",
      "--normalization",
      "1.0",
      "--amplification",
      "0.0",
    ]);

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      output += chunk.toString("utf8");
    });
    const code = await new Promise<number | null>((resolve) => child.on("close", resolve));
    assert.equal(code, 0, output);
    assert.match(output, /"type": "file-complete"|"type":"file-complete"/);
  },
);
