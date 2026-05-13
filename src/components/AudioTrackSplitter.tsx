"use client";

import JSZip from "jszip";
import { useMemo, useState, type DragEvent } from "react";
import { triggerBrowserDownload } from "../lib/downloadBlob";
import styles from "./AudioTrackSplitter.module.css";

type SplitterFileStatus = "pending" | "working" | "done" | "error";

type SplitterQueueItem = {
  id: string;
  fileName: string;
  size: number;
  status: SplitterFileStatus;
  detail: string;
};

type SplitterStemQc = {
  stem: "BGM" | "VOCAL";
  fileName: string;
  sizeBytes: number;
  sampleRate: number;
  channels: number;
  durationSeconds: number;
  peakDbfs: number | null;
  rmsDbfs: number | null;
  clippedSampleCount: number;
  clippedSamplePct: number;
  silent: boolean;
};

type SplitterReportFile = {
  inputIndex: number;
  originalName: string;
  status: "success" | "failed";
  message: string;
  outputs: string[];
  sampleRate: number | null;
  channels: number | null;
  durationSeconds: number | null;
  stems: SplitterStemQc[];
  warnings: string[];
};

type SplitterReport = {
  generatedAt: string;
  engine: string;
  totalFiles: number;
  succeeded: number;
  failed: number;
  files: SplitterReportFile[];
};

type SplitterJobFile = {
  inputIndex: number;
  originalName: string;
  sizeBytes: number;
  status: SplitterFileStatus;
  detail: string;
  outputs: string[];
  durationSeconds: number | null;
  sampleRate: number | null;
  channels: number | null;
  stems: SplitterStemQc[];
  warnings: string[];
};

type SplitterJob = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  message: string;
  zipName: string | null;
  files: SplitterJobFile[];
  report: SplitterReport | null;
};

const formatBytes = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
};

const formatDuration = (seconds: number | null) =>
  seconds === null || !Number.isFinite(seconds) ? "Duration unknown" : `${seconds.toFixed(3)}s`;

const formatDbfs = (db: number | null) =>
  db === null || !Number.isFinite(db) ? "n/a" : `${db.toFixed(1)} dBFS`;

const queueIdForFile = (file: File, index: number) => `${file.name}|${file.size}|${file.lastModified}|${index}`;

const getDownloadName = (response: Response) => {
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = disposition.match(/filename="([^"]+)"/i);
  if (match?.[1]) return match[1];
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `audio_track_splitter_${stamp}.zip`;
};

const readReportFromZip = async (blob: Blob) => {
  const zip = await JSZip.loadAsync(blob);
  const reportFile = zip.file("split_report.json");
  if (!reportFile) return null;
  return JSON.parse(await reportFile.async("string")) as SplitterReport;
};

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function AudioTrackSplitter() {
  const [files, setFiles] = useState<File[]>([]);
  const [queueItems, setQueueItems] = useState<SplitterQueueItem[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<SplitterReport | null>(null);
  const [zipBlob, setZipBlob] = useState<Blob | null>(null);
  const [zipName, setZipName] = useState("audio_track_splitter.zip");

  const queueCounts = useMemo(
    () => ({
      total: queueItems.length,
      done: queueItems.filter((item) => item.status === "done").length,
      error: queueItems.filter((item) => item.status === "error").length,
      working: queueItems.filter((item) => item.status === "working").length,
      pending: queueItems.filter((item) => item.status === "pending").length,
    }),
    [queueItems],
  );

  const resetResults = () => {
    setQueueItems([]);
    setZipBlob(null);
    setReport(null);
    setError(null);
    setStatus("Idle");
  };

  const handleFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const allFiles = Array.from(incoming);
    const wavs = allFiles.filter((file) => file.name.toLowerCase().endsWith(".wav"));
    const rejected = allFiles.length - wavs.length;
    if (rejected > 0) {
      setError(`${rejected} unsupported file(s) ignored. Audio Track Splitter accepts .wav files only.`);
    } else {
      setError(null);
    }
    setFiles((prev) => {
      const merged = [...prev];
      const seen = new Set(prev.map((file) => `${file.name}|${file.size}|${file.lastModified}`));
      for (const file of wavs) {
        const key = `${file.name}|${file.size}|${file.lastModified}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(file);
        }
      }
      return merged;
    });
    setQueueItems([]);
    setZipBlob(null);
    setReport(null);
    setStatus("Idle");
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    handleFiles(event.dataTransfer.files);
  };

  const clearAll = () => {
    setFiles([]);
    resetResults();
  };

  const applyJobSnapshot = (job: SplitterJob) => {
    setStatus(job.message);
    if (job.report) setReport(job.report);
    setQueueItems((prev) =>
      job.files.map((file, index) => {
        const previous = prev[index];
        return {
          id: previous?.id ?? `${file.originalName}|${file.sizeBytes}|${index}`,
          fileName: file.originalName,
          size: file.sizeBytes,
          status: file.status,
          detail: file.detail,
        };
      }),
    );
  };

  const runSplitter = async () => {
    if (files.length === 0 || busy) return;
    const batchFiles = [...files];
    setBusy(true);
    setError(null);
    setReport(null);
    setZipBlob(null);
    setStatus("Uploading batch...");
    setQueueItems(
      batchFiles.map((file, index) => ({
        id: queueIdForFile(file, index),
        fileName: file.name,
        size: file.size,
        status: index === 0 ? "working" : "pending",
        detail: index === 0 ? "Uploading to splitter" : "Queued",
      })),
    );

    try {
      const formData = new FormData();
      for (const file of batchFiles) {
        formData.append("files", file, file.name);
      }

      const startResponse = await fetch("/api/audio-splitter", {
        method: "POST",
        body: formData,
      });

      if (!startResponse.ok) {
        const contentType = startResponse.headers.get("content-type") ?? "";
        if (contentType.includes("application/json")) {
          const payload = (await startResponse.json()) as { error?: string };
          throw new Error(payload.error ?? "Audio splitter request failed.");
        }
        throw new Error(`Audio splitter request failed with status ${startResponse.status}.`);
      }

      let job = (await startResponse.json()) as SplitterJob;
      applyJobSnapshot(job);

      while (job.status === "queued" || job.status === "running") {
        await wait(1000);
        const statusResponse = await fetch(`/api/audio-splitter/${job.id}`, {
          method: "GET",
          cache: "no-store",
        });
        if (!statusResponse.ok) {
          throw new Error(`Audio splitter status failed with status ${statusResponse.status}.`);
        }
        job = (await statusResponse.json()) as SplitterJob;
        applyJobSnapshot(job);
      }

      if (job.status === "failed") {
        throw new Error(job.message);
      }

      setStatus("Downloading ZIP...");
      const zipResponse = await fetch(`/api/audio-splitter/${job.id}/download`, {
        method: "GET",
        cache: "no-store",
      });
      if (!zipResponse.ok) {
        throw new Error(`Audio splitter download failed with status ${zipResponse.status}.`);
      }

      const blob = await zipResponse.blob();
      const downloadName = getDownloadName(zipResponse);
      const parsedReport = job.report ?? (await readReportFromZip(blob));
      setZipBlob(blob);
      setZipName(downloadName);
      setReport(parsedReport);

      if (parsedReport) {
        const reportByIndex = new Map(parsedReport.files.map((file) => [file.inputIndex, file]));
        setQueueItems((prev) =>
          prev.map((item, index) => {
            const fileReport = reportByIndex.get(index);
            if (!fileReport) {
              return { ...item, status: "error", detail: "Missing from split report" };
            }
            return {
              ...item,
              status: fileReport.status === "success" ? "done" : "error",
              detail:
                fileReport.status === "success"
                  ? `${fileReport.outputs.length} stems ready, ${formatDuration(fileReport.durationSeconds)}`
                  : fileReport.message,
            };
          }),
        );
        setStatus(parsedReport.failed > 0 ? "Done with warnings" : "Done");
      } else {
        setQueueItems((prev) =>
          prev.map((item) => ({
            ...item,
            status: "done",
            detail: "ZIP ready",
          })),
        );
        setStatus("Done");
      }
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Failed");
      setQueueItems((prev) =>
        prev.map((item) =>
          item.status === "done"
            ? item
            : {
                ...item,
                status: "error",
                detail: message,
              },
        ),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={styles.layout}>
      <div className={styles.panel}>
        <div className={styles.card}>
          <div
            className={`${styles.dropzone} ${dragActive ? styles.dropActive : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <div className={styles.dropTitle}>Drop WAV tracks for stem splitting</div>
            <div className={styles.dropHint}>
              Batch upload shorts drama mixes and export clean BGM and VOCAL stems in one ZIP.
            </div>
            <div className={styles.controls}>
              <label className={styles.button}>
                Select WAV Files
                <input
                  type="file"
                  accept=".wav"
                  multiple
                  hidden
                  onChange={(event) => {
                    handleFiles(event.target.files);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              <button
                type="button"
                className={`${styles.button} ${styles.buttonGhost}`}
                onClick={clearAll}
                disabled={busy || files.length === 0}
              >
                Clear
              </button>
            </div>
            <div className={styles.fileList}>
              {files.length === 0 && <div className={styles.dropHint}>No files selected.</div>}
              {files.map((file, index) => (
                <div className={styles.fileItem} key={queueIdForFile(file, index)}>
                  <div>{file.name}</div>
                  <span>{formatBytes(file.size)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className={styles.card}>
          <h3>Splitter Export</h3>
          <div className={styles.suffixGrid}>
            <span>originalName_BGM.wav</span>
            <span>originalName_VOCAL.wav</span>
          </div>
          <div className={`${styles.controls} ${styles.sectionTop}`}>
            <button type="button" className={styles.button} onClick={runSplitter} disabled={busy || files.length === 0}>
              {busy ? "Splitting..." : "Run Audio Splitter"}
            </button>
            {zipBlob && (
              <button
                type="button"
                className={`${styles.button} ${styles.buttonSecondary}`}
                onClick={() => triggerBrowserDownload(zipBlob, zipName)}
              >
                Download ZIP
              </button>
            )}
            <div className={styles.progress}>{status}</div>
          </div>
          <div className={styles.footerNote}>
            Real separation runs on the configured local RoFormer engine. First run may download the model;
            CUDA acceleration is strongly recommended for drama-length tracks. The ZIP always includes split_report.txt.
          </div>
          {error && <div className={styles.errorBox}>{error}</div>}
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.queueHeader}>
          <h3>Splitter Queue</h3>
          {queueCounts.total > 0 && (
            <div className={styles.queueSummaryBadges}>
              <span>{queueCounts.total} total</span>
              <span>{queueCounts.working} active</span>
              <span>{queueCounts.done} done</span>
              <span>{queueCounts.error} failed</span>
              <span>{queueCounts.pending} waiting</span>
            </div>
          )}
        </div>
        <div className={styles.queueList}>
          {queueItems.length === 0 ? (
            <div className={styles.dropHint}>No splitter queue yet. Add WAV files and run the splitter.</div>
          ) : (
            queueItems.map((item, index) => {
              const statusClass =
                item.status === "done"
                  ? styles.statusDone
                  : item.status === "error"
                    ? styles.statusError
                    : item.status === "working"
                      ? styles.statusWorking
                      : styles.statusPending;
              return (
                <div className={`${styles.queueItem} ${statusClass}`} key={item.id}>
                  <div className={styles.queueTitleWrap}>
                    <span className={styles.queueIndex}>{index + 1}</span>
                    <div>
                      <div className={styles.queueFileName}>{item.fileName}</div>
                      <div className={styles.queueMeta}>
                        {formatBytes(item.size)} - {item.detail}
                      </div>
                    </div>
                  </div>
                  <span className={styles.statusBadge}>
                    {item.status === "working"
                      ? "Processing"
                      : item.status === "done"
                        ? "Done"
                        : item.status === "error"
                          ? "Failed"
                          : "Queued"}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {report && (
        <div className={styles.card}>
          <div className={styles.queueHeader}>
            <h3>Split Report</h3>
            <div className={styles.reportMeta}>
              {report.engine} - {report.succeeded} succeeded / {report.failed} failed
            </div>
          </div>
          <div className={styles.reportList}>
            {report.files.map((file) => (
              <div className={styles.reportItem} key={`${file.inputIndex}-${file.originalName}`}>
                <div>
                  <strong>{file.originalName}</strong>
                  <div className={styles.queueMeta}>
                    {file.status === "success"
                      ? `${file.outputs.join(", ")} - ${file.sampleRate ?? "unknown"} Hz`
                      : file.message}
                  </div>
                  {file.stems.length > 0 && (
                    <div className={styles.stemQcList}>
                      {file.stems.map((stem) => (
                        <span key={`${file.inputIndex}-${stem.stem}`}>
                          {stem.stem}: peak {formatDbfs(stem.peakDbfs)}, RMS {formatDbfs(stem.rmsDbfs)}
                          {stem.clippedSampleCount > 0 ? `, clipped ${stem.clippedSampleCount}` : ""}
                        </span>
                      ))}
                    </div>
                  )}
                  {file.warnings.length > 0 && (
                    <div className={styles.warningList}>
                      {file.warnings.map((warning) => (
                        <span key={warning}>{warning}</span>
                      ))}
                    </div>
                  )}
                </div>
                <span className={`${styles.statusBadge} ${file.status === "success" ? styles.statusDone : styles.statusError}`}>
                  {file.status === "success" ? "Done" : "Failed"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
