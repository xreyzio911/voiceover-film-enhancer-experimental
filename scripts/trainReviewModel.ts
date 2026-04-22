import JSZip from "jszip";
import { promises as fs } from "node:fs";
import path from "node:path";
import {
  fitLearnedReviewWeights,
  parseReviewDecisionJsonl,
  type ReviewBundleManifest,
} from "../src/lib/reviewLearning.ts";

type CliArgs = {
  bundlesPath: string;
  labelsPath: string;
  outDir: string;
};

const usage = () =>
  [
    "Usage:",
    "  npm run train:review-model -- --bundles <review-bundles.zip|dir> --labels <labels.jsonl> [--outDir <dir>]",
  ].join("\n");

const parseArgs = (argv: string[]): CliArgs => {
  let bundlesPath = "";
  let labelsPath = "";
  let outDir = path.resolve(process.cwd(), "review-learning-output");

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--bundles") {
      bundlesPath = argv[index + 1] ?? "";
      index += 1;
    } else if (arg === "--labels") {
      labelsPath = argv[index + 1] ?? "";
      index += 1;
    } else if (arg === "--outDir") {
      outDir = path.resolve(process.cwd(), argv[index + 1] ?? "");
      index += 1;
    }
  }

  if (!bundlesPath || !labelsPath) {
    throw new Error(usage());
  }

  return {
    bundlesPath: path.resolve(process.cwd(), bundlesPath),
    labelsPath: path.resolve(process.cwd(), labelsPath),
    outDir,
  };
};

const readManifestFile = async (filePath: string) =>
  JSON.parse(await fs.readFile(filePath, "utf8")) as ReviewBundleManifest;

const collectManifestPaths = async (rootDir: string): Promise<string[]> => {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  const manifests: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      manifests.push(...(await collectManifestPaths(fullPath)));
      continue;
    }
    if (entry.isFile() && entry.name === "manifest.json") {
      manifests.push(fullPath);
    }
  }

  return manifests;
};

const loadManifestsFromZip = async (zipPath: string) => {
  const zip = await JSZip.loadAsync(await fs.readFile(zipPath));
  const manifestPaths = Object.keys(zip.files).filter(
    (entryPath) => !zip.files[entryPath]?.dir && /(^|\/)manifest\.json$/i.test(entryPath),
  );

  const manifests: ReviewBundleManifest[] = [];
  for (const manifestPath of manifestPaths) {
    const entry = zip.file(manifestPath);
    if (!entry) continue;
    manifests.push(JSON.parse(await entry.async("text")) as ReviewBundleManifest);
  }
  return manifests;
};

const loadManifests = async (sourcePath: string) => {
  const stat = await fs.stat(sourcePath);
  if (stat.isDirectory()) {
    const manifestPaths = await collectManifestPaths(sourcePath);
    const manifests = await Promise.all(manifestPaths.map((manifestPath) => readManifestFile(manifestPath)));
    return manifests;
  }
  if (sourcePath.toLowerCase().endsWith(".zip")) {
    return await loadManifestsFromZip(sourcePath);
  }
  return [await readManifestFile(sourcePath)];
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  const manifests = await loadManifests(args.bundlesPath);
  const labelsText = await fs.readFile(args.labelsPath, "utf8");
  const decisions = parseReviewDecisionJsonl(labelsText);

  if (manifests.length === 0) {
    throw new Error("No review bundle manifests found.");
  }
  if (decisions.length === 0) {
    throw new Error("No review decisions found in labels JSONL.");
  }

  const { weights, report } = fitLearnedReviewWeights(manifests, decisions);
  await fs.mkdir(args.outDir, { recursive: true });

  const weightsPath = path.join(args.outDir, "review-weights.json");
  const reportPath = path.join(args.outDir, "review-training-report.json");
  await fs.writeFile(weightsPath, JSON.stringify(weights, null, 2));
  await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

  console.log(
    [
      `Trained review model: ${weights.modelName}`,
      `Manifests: ${manifests.length}`,
      `Decisions: ${decisions.length}`,
      `Pairwise accuracy: ${report.pairwiseAccuracy === null ? "n/a" : `${(report.pairwiseAccuracy * 100).toFixed(1)}%`}`,
      `Weights: ${weightsPath}`,
      `Report: ${reportPath}`,
    ].join("\n"),
  );
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
