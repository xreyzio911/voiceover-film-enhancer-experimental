const EXTENSION_CONTENT_TYPES: Record<string, string> = {
  json: "application/json",
  txt: "text/plain",
  wav: "audio/wav",
  zip: "application/zip",
};

const sanitizeDownloadFileName = (fileName: string) => {
  const base = (fileName.split(/[\\/]/).pop() ?? fileName).trim();
  const safe = base
    .replace(/[\x00-\x1f<>:"/\\|?*]/g, "_")
    .replace(/[ .]+$/g, "");
  return safe || `download_${new Date().toISOString().replace(/[:.]/g, "-")}`;
};

const inferContentType = (fileName: string) => {
  const extension = fileName.split(".").pop()?.toLowerCase();
  return extension ? EXTENSION_CONTENT_TYPES[extension] ?? "application/octet-stream" : "application/octet-stream";
};

export const triggerBrowserDownload = (blob: Blob, fileName: string) => {
  const safeName = sanitizeDownloadFileName(fileName);
  const namedBlob =
    blob instanceof File && blob.name === safeName
      ? blob
      : new File([blob], safeName, {
          type: blob.type || inferContentType(safeName),
          lastModified: Date.now(),
        });
  const url = URL.createObjectURL(namedBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = safeName;
  link.setAttribute("download", safeName);
  link.rel = "noopener";
  link.style.display = "none";

  document.body.appendChild(link);
  link.click();

  // Keep the anchor and object URL alive while Chromium starts large blob
  // downloads. Removing the node immediately can make Chrome fall back to the
  // blob URL UUID as the saved filename.
  setTimeout(() => {
    link.remove();
    URL.revokeObjectURL(url);
  }, 120_000);
};
