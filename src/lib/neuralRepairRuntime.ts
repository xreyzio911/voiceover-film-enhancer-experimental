import path from "node:path";

export type NeuralRepairWorkerCommandSource =
  | "configured-command"
  | "configured-path"
  | "project-linux-venv"
  | "project-windows-venv"
  | "runtime-python";

export type ResolvedNeuralRepairWorkerCommand = {
  command: string;
  args: string[];
  source: NeuralRepairWorkerCommandSource;
  configuredCommand: string;
};

type ResolveNeuralRepairWorkerCommandOptions = {
  commandLine?: string;
  cwd?: string;
  platform?: NodeJS.Platform;
  exists?: (candidate: string) => boolean;
};

export const splitCommandLine = (commandLine: string | undefined) => {
  const parts: string[] = [];
  let current = "";
  let quote: "\"" | "'" | null = null;

  for (const char of (commandLine ?? "").trim()) {
    if (quote) {
      if (char === quote) quote = null;
      else current += char;
      continue;
    }
    if (char === "\"" || char === "'") {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) {
        parts.push(current);
        current = "";
      }
      continue;
    }
    current += char;
  }
  if (current) parts.push(current);
  return parts;
};

const isPathLikeCommand = (value: string) => value.startsWith(".") || value.includes("/") || value.includes("\\");

const normalizePathSeparators = (value: string) => value.replace(/[\\/]+/g, path.sep);

export const resolveProjectRelativePathOption = (value: string | undefined, cwd = process.cwd()) => {
  if (!value) return undefined;
  if (!isPathLikeCommand(value)) return value;

  const normalized = normalizePathSeparators(value);
  return path.isAbsolute(normalized) ? normalized : path.resolve(cwd, normalized);
};

const isWindowsNeuralVenvCommand = (value: string) =>
  value.replace(/\\/g, "/").toLowerCase().includes(".venv-neural/scripts/python");

const existingProjectPython = (
  candidate: string,
  source: NeuralRepairWorkerCommandSource,
  cwd: string,
  exists: (candidatePath: string) => boolean,
): Pick<ResolvedNeuralRepairWorkerCommand, "command" | "source"> | null => {
  const command = resolveProjectRelativePathOption(candidate, cwd);
  if (command && exists(command)) {
    return { command, source };
  }
  return null;
};

const runtimePythonCommand = (platform: NodeJS.Platform) => (platform === "win32" ? "python.exe" : "python3");

export const resolveNeuralRepairWorkerCommand = ({
  commandLine,
  cwd = process.cwd(),
  platform = process.platform,
  exists = () => false,
}: ResolveNeuralRepairWorkerCommandOptions = {}): ResolvedNeuralRepairWorkerCommand => {
  const [configuredCommand = "", ...args] = splitCommandLine(commandLine);
  const configuredIsWindowsVenv = configuredCommand ? isWindowsNeuralVenvCommand(configuredCommand) : false;
  const configuredIsIncompatible = platform !== "win32" && configuredIsWindowsVenv;

  if (configuredCommand && !configuredIsIncompatible) {
    if (!isPathLikeCommand(configuredCommand)) {
      return {
        command: configuredCommand,
        args,
        source: "configured-command",
        configuredCommand,
      };
    }

    const resolved = resolveProjectRelativePathOption(configuredCommand, cwd);
    if (resolved && exists(resolved)) {
      return {
        command: resolved,
        args,
        source: "configured-path",
        configuredCommand,
      };
    }
  }

  const projectVenv =
    platform === "win32"
      ? existingProjectPython(".venv-neural\\Scripts\\python.exe", "project-windows-venv", cwd, exists)
      : existingProjectPython(".venv-neural/bin/python", "project-linux-venv", cwd, exists);

  if (projectVenv) {
    return {
      ...projectVenv,
      args,
      configuredCommand,
    };
  }

  return {
    command: runtimePythonCommand(platform),
    args,
    source: "runtime-python",
    configuredCommand,
  };
};
