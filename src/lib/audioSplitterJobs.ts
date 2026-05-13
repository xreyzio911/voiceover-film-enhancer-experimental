import {
  splitBatchAudioTracks,
  type AudioSplitterFileReport,
  type AudioSplitterInput,
  type AudioSplitterReport,
  type AudioSplitterStemQc,
} from "./audioSplitterService";

export type AudioSplitterJobFile = {
  inputIndex: number;
  originalName: string;
  sizeBytes: number;
  status: "pending" | "working" | "done" | "error";
  detail: string;
  outputs: string[];
  durationSeconds: number | null;
  sampleRate: number | null;
  channels: number | null;
  stems: AudioSplitterStemQc[];
  warnings: string[];
  updatedAt: string;
};

export type AudioSplitterJobSnapshot = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  createdAt: string;
  updatedAt: string;
  engine: string | null;
  totalFiles: number;
  succeeded: number;
  failed: number;
  activeIndex: number | null;
  message: string;
  zipName: string | null;
  files: AudioSplitterJobFile[];
  report: AudioSplitterReport | null;
};

type AudioSplitterJobRecord = AudioSplitterJobSnapshot & {
  zip: Buffer | null;
};

const JOB_TTL_MS = 60 * 60 * 1000;

const globalJobs = globalThis as typeof globalThis & {
  __audioSplitterJobs?: Map<string, AudioSplitterJobRecord>;
};

const jobs = globalJobs.__audioSplitterJobs ?? new Map<string, AudioSplitterJobRecord>();
globalJobs.__audioSplitterJobs = jobs;

const nowIso = () => new Date().toISOString();

const toSnapshot = (job: AudioSplitterJobRecord): AudioSplitterJobSnapshot => ({
  id: job.id,
  status: job.status,
  createdAt: job.createdAt,
  updatedAt: job.updatedAt,
  engine: job.engine,
  totalFiles: job.totalFiles,
  succeeded: job.succeeded,
  failed: job.failed,
  activeIndex: job.activeIndex,
  message: job.message,
  zipName: job.zipName,
  files: job.files.map((file) => ({
    ...file,
    outputs: [...file.outputs],
    stems: file.stems.map((stem) => ({ ...stem })),
    warnings: [...file.warnings],
  })),
  report: job.report,
});

const patchJob = (job: AudioSplitterJobRecord, patch: Partial<AudioSplitterJobRecord>) => {
  Object.assign(job, patch, { updatedAt: nowIso() });
};

const applyReportToFile = (job: AudioSplitterJobRecord, report: AudioSplitterFileReport) => {
  const file = job.files[report.inputIndex];
  if (!file) return;
  file.status = report.status === "success" ? "done" : "error";
  file.detail =
    report.status === "success"
      ? `${report.outputs.length} stems ready`
      : report.message;
  file.outputs = [...report.outputs];
  file.durationSeconds = report.durationSeconds;
  file.sampleRate = report.sampleRate;
  file.channels = report.channels;
  file.stems = report.stems.map((stem) => ({ ...stem }));
  file.warnings = [...report.warnings];
  file.updatedAt = nowIso();
};

export const startAudioSplitterJob = (inputs: AudioSplitterInput[]) => {
  cleanupExpiredAudioSplitterJobs();

  const timestamp = nowIso();
  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const job: AudioSplitterJobRecord = {
    id,
    status: "queued",
    createdAt: timestamp,
    updatedAt: timestamp,
    engine: null,
    totalFiles: inputs.length,
    succeeded: 0,
    failed: 0,
    activeIndex: null,
    message: "Queued",
    zipName: null,
    zip: null,
    report: null,
    files: inputs.map((input, inputIndex) => ({
      inputIndex,
      originalName: input.originalName,
      sizeBytes: input.bytes.byteLength,
      status: "pending",
      detail: "Queued",
      outputs: [],
      durationSeconds: null,
      sampleRate: null,
      channels: null,
      stems: [],
      warnings: [],
      updatedAt: timestamp,
    })),
  };

  jobs.set(id, job);

  void splitBatchAudioTracks(inputs, {
    onProgress(event) {
      if (event.type === "file-start") {
        const file = job.files[event.inputIndex];
        if (file) {
          file.status = "working";
          file.detail = event.message;
          file.updatedAt = nowIso();
        }
        patchJob(job, {
          status: "running",
          activeIndex: event.inputIndex,
          message: `Processing ${event.originalName}`,
        });
      } else if (event.type === "file-progress") {
        const file = job.files[event.inputIndex];
        if (file) {
          file.status = "working";
          file.detail = event.message;
          file.updatedAt = nowIso();
        }
        patchJob(job, {
          status: "running",
          activeIndex: event.inputIndex,
          message: event.message,
        });
      } else if (event.type === "file-complete") {
        applyReportToFile(job, event.report);
        patchJob(job, {
          succeeded: job.files.filter((file) => file.status === "done").length,
          failed: job.files.filter((file) => file.status === "error").length,
          activeIndex: null,
          message: event.report.status === "success" ? "File complete" : "File failed; continuing batch",
        });
      } else {
        patchJob(job, {
          report: event.report,
          engine: event.report.engine,
          succeeded: event.report.succeeded,
          failed: event.report.failed,
          message: "Packaging ZIP",
        });
      }
    },
  })
    .then((result) => {
      patchJob(job, {
        status: "done",
        engine: result.report.engine,
        succeeded: result.report.succeeded,
        failed: result.report.failed,
        activeIndex: null,
        message: result.report.failed > 0 ? "Done with warnings" : "Done",
        zip: result.zip,
        zipName: result.zipName,
        report: result.report,
      });
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      patchJob(job, {
        status: "failed",
        activeIndex: null,
        message,
      });
    });

  return toSnapshot(job);
};

export const getAudioSplitterJob = (jobId: string) => {
  const job = jobs.get(jobId);
  return job ? toSnapshot(job) : null;
};

export const getAudioSplitterJobDownload = (jobId: string) => {
  const job = jobs.get(jobId);
  if (!job || !job.zip || !job.zipName) return null;
  return { zip: job.zip, zipName: job.zipName };
};

export const cleanupExpiredAudioSplitterJobs = () => {
  const cutoff = Date.now() - JOB_TTL_MS;
  for (const [jobId, job] of jobs) {
    if (Date.parse(job.updatedAt) < cutoff) jobs.delete(jobId);
  }
};
