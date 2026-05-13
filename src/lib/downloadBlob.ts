const EXTENSION_CONTENT_TYPES: Record<string, string> = {
  json: "application/json",
  txt: "text/plain",
  wav: "audio/wav",
  zip: "application/zip",
};

const MB = 1024 * 1024;
const SMALL_DOWNLOAD_RETAIN_MS = 10 * 60 * 1000;
const LARGE_DOWNLOAD_RETAIN_MS = 45 * 60 * 1000;
const HUGE_DOWNLOAD_RETAIN_MS = 2 * 60 * 60 * 1000;

type ActiveDownloadLease = {
  link: HTMLAnchorElement;
  timeoutId: number;
  url: string;
};

type BrowserDownloadOptions = {
  delayMs?: number;
  retainMs?: number;
};

export type BrowserDownloadStart = {
  fileName: string;
  retainMs: number;
  size: number;
};

const activeDownloadLeases = new Set<ActiveDownloadLease>();
let cleanupRegistered = false;
let downloadQueue: Promise<void> = Promise.resolve();

export const sanitizeDownloadFileName = (fileName: string) => {
  const base = (fileName.split(/[\\/]/).pop() ?? fileName).trim();
  const safe = base
    .replace(/[\x00-\x1f<>:"/\\|?*]/g, "_")
    .replace(/[ .]+$/g, "");
  return safe || `download_${new Date().toISOString().replace(/[:.]/g, "-")}`;
};

export const inferContentType = (fileName: string) => {
  const extension = fileName.split(".").pop()?.toLowerCase();
  return extension ? EXTENSION_CONTENT_TYPES[extension] ?? "application/octet-stream" : "application/octet-stream";
};

export const getDownloadUrlRetainMs = (bytes: number) => {
  if (bytes >= 512 * MB) return HUGE_DOWNLOAD_RETAIN_MS;
  if (bytes >= 128 * MB) return LARGE_DOWNLOAD_RETAIN_MS;
  return SMALL_DOWNLOAD_RETAIN_MS;
};

export const getDownloadQueueDelayMs = (bytes: number) => {
  if (bytes >= 512 * MB) return 2500;
  if (bytes >= 128 * MB) return 1600;
  return 900;
};

const cleanupDownloadLease = (lease: ActiveDownloadLease) => {
  if (!activeDownloadLeases.delete(lease)) return;
  window.clearTimeout(lease.timeoutId);
  lease.link.remove();
  URL.revokeObjectURL(lease.url);
};

const cleanupAllDownloadLeases = () => {
  for (const lease of Array.from(activeDownloadLeases)) {
    cleanupDownloadLease(lease);
  }
};

const ensureDownloadCleanupRegistered = () => {
  if (cleanupRegistered || typeof window === "undefined") return;
  cleanupRegistered = true;
  window.addEventListener("pagehide", cleanupAllDownloadLeases, { once: true });
  window.addEventListener("beforeunload", cleanupAllDownloadLeases, { once: true });
};

const wrapBlobWithDownloadName = (blob: Blob, safeName: string) => {
  if (typeof File === "undefined") return blob;
  if (blob instanceof File && blob.name === safeName && blob.type) return blob;
  return new File([blob], safeName, {
    type: blob.type || inferContentType(safeName),
    lastModified: Date.now(),
  });
};

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export const triggerBrowserDownload = (
  blob: Blob,
  fileName: string,
  options: BrowserDownloadOptions = {},
): BrowserDownloadStart => {
  ensureDownloadCleanupRegistered();
  const safeName = sanitizeDownloadFileName(fileName);
  const namedBlob = wrapBlobWithDownloadName(blob, safeName);
  const url = URL.createObjectURL(namedBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = safeName;
  link.setAttribute("download", safeName);
  link.rel = "noopener";
  link.style.display = "none";

  document.body.appendChild(link);
  link.click();

  const retainMs = options.retainMs ?? getDownloadUrlRetainMs(blob.size);
  const lease: ActiveDownloadLease = {
    link,
    timeoutId: 0,
    url,
  };
  lease.timeoutId = window.setTimeout(() => cleanupDownloadLease(lease), retainMs);
  activeDownloadLeases.add(lease);

  return {
    fileName: safeName,
    retainMs,
    size: blob.size,
  };
};

export const queueBrowserDownload = (
  blob: Blob,
  fileName: string,
  options: BrowserDownloadOptions = {},
): Promise<BrowserDownloadStart> => {
  const run = async () => {
    const started = triggerBrowserDownload(blob, fileName, options);
    await wait(options.delayMs ?? getDownloadQueueDelayMs(blob.size));
    return started;
  };
  const next = downloadQueue.catch(() => undefined).then(run);
  downloadQueue = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
};
