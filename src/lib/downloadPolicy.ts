const MB = 1024 * 1024;

export const VO_ZIP_CHUNK_TARGET_BYTES = 512 * MB;
export const VO_ZIP_LARGE_BATCH_BYTES = 768 * MB;
export const VO_ZIP_LARGE_BATCH_FILE_COUNT = 16;
export const VO_ZIP_MAX_FILES_PER_CHUNK = 8;

export type DownloadPolicyOutput = {
  name: string;
  size: number;
};

export type VoZipExportPart<T extends DownloadPolicyOutput> = {
  estimatedBytes: number;
  outputs: T[];
  partNumber: number;
  totalParts: number;
};

export const estimateVoZipBytes = (outputs: DownloadPolicyOutput[]) => {
  const payloadBytes = outputs.reduce((total, output) => total + Math.max(0, output.size || 0), 0);
  const zipDirectoryBytes = outputs.reduce((total, output) => total + 128 + output.name.length * 2, 4096);
  return payloadBytes + zipDirectoryBytes;
};

export const shouldChunkVoZip = (outputs: DownloadPolicyOutput[]) =>
  outputs.length > VO_ZIP_LARGE_BATCH_FILE_COUNT || estimateVoZipBytes(outputs) >= VO_ZIP_LARGE_BATCH_BYTES;

export const planVoZipExportParts = <T extends DownloadPolicyOutput>(outputs: T[]): VoZipExportPart<T>[] => {
  if (outputs.length === 0) return [];
  if (!shouldChunkVoZip(outputs)) {
    return [
      {
        estimatedBytes: estimateVoZipBytes(outputs),
        outputs,
        partNumber: 1,
        totalParts: 1,
      },
    ];
  }

  const parts: Array<Omit<VoZipExportPart<T>, "partNumber" | "totalParts">> = [];
  let currentOutputs: T[] = [];
  let currentBytes = 4096;

  const flush = () => {
    if (currentOutputs.length === 0) return;
    parts.push({
      estimatedBytes: estimateVoZipBytes(currentOutputs),
      outputs: currentOutputs,
    });
    currentOutputs = [];
    currentBytes = 4096;
  };

  for (const output of outputs) {
    const entryBytes = estimateVoZipBytes([output]);
    const wouldExceedSize =
      currentOutputs.length > 0 && currentBytes + entryBytes > VO_ZIP_CHUNK_TARGET_BYTES;
    const wouldExceedCount = currentOutputs.length >= VO_ZIP_MAX_FILES_PER_CHUNK;

    if (wouldExceedSize || wouldExceedCount) {
      flush();
    }

    currentOutputs.push(output);
    currentBytes += entryBytes;
  }

  flush();

  return parts.map((part, index) => ({
    ...part,
    partNumber: index + 1,
    totalParts: parts.length,
  }));
};

export const describeVoZipPolicy = (outputs: DownloadPolicyOutput[]) => {
  const parts = planVoZipExportParts(outputs);
  const estimatedBytes = estimateVoZipBytes(outputs);
  if (parts.length <= 1) {
    return `Standard ZIP export (${outputs.length} file${outputs.length === 1 ? "" : "s"}).`;
  }
  return `Large batch ZIP export: ${outputs.length} files, ${parts.length} parts, estimated ${estimatedBytes} bytes.`;
};
