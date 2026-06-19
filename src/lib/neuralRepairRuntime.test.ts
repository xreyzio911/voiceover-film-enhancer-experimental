import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import {
  resolveNeuralRepairRuntimeTarget,
  resolveNeuralRepairWorkerCommand,
  resolveProjectRelativePathOption,
  splitCommandLine,
} from "./neuralRepairRuntime.ts";

test("splits quoted neural worker commands without dropping module arguments", () => {
  assert.deepEqual(splitCommandLine('"python executable" -m clearvoice_worker --flag'), [
    "python executable",
    "-m",
    "clearvoice_worker",
    "--flag",
  ]);
});

test("resolves project-relative neural worker paths from the repository root", () => {
  const cwd = path.resolve("runtime-root");

  assert.equal(
    resolveProjectRelativePathOption(".venv-neural\\Scripts\\python.exe", cwd),
    path.join(cwd, ".venv-neural", "Scripts", "python.exe"),
  );
  assert.equal(resolveProjectRelativePathOption("python3", cwd), "python3");
});

test("maps a Windows neural venv command to the Linux venv command when deployed", () => {
  const cwd = path.resolve("runtime-root");
  const linuxPython = path.join(cwd, ".venv-neural", "bin", "python");

  const resolved = resolveNeuralRepairWorkerCommand({
    commandLine: ".venv-neural\\Scripts\\python.exe",
    cwd,
    platform: "linux",
    exists: (candidate) => candidate === linuxPython,
  });

  assert.equal(resolved.command, linuxPython);
  assert.deepEqual(resolved.args, []);
  assert.equal(resolved.source, "project-linux-venv");
});

test("falls back to python3 instead of spawning a Windows venv path on Linux", () => {
  const resolved = resolveNeuralRepairWorkerCommand({
    commandLine: ".venv-neural\\Scripts\\python.exe",
    cwd: path.resolve("runtime-root"),
    platform: "linux",
    exists: () => false,
  });

  assert.equal(resolved.command, "python3");
  assert.deepEqual(resolved.args, []);
  assert.equal(resolved.source, "runtime-python");
});

test("keeps a valid Windows neural venv command on Windows", () => {
  const cwd = path.resolve("runtime-root");
  const windowsPython = path.join(cwd, ".venv-neural", "Scripts", "python.exe");

  const resolved = resolveNeuralRepairWorkerCommand({
    commandLine: ".venv-neural\\Scripts\\python.exe",
    cwd,
    platform: "win32",
    exists: (candidate) => candidate === windowsPython,
  });

  assert.equal(resolved.command, windowsPython);
  assert.deepEqual(resolved.args, []);
  assert.equal(resolved.source, "configured-path");
});

test("uses a remote neural worker when one is configured", () => {
  const resolved = resolveNeuralRepairRuntimeTarget({
    commandLine: ".venv-neural\\Scripts\\python.exe",
    remoteUrl: "https://worker.example.test/neural",
    remoteToken: "secret-token",
    cwd: path.resolve("runtime-root"),
    platform: "linux",
    isVercel: true,
    exists: () => false,
  });

  assert.equal(resolved.kind, "remote");
  assert.equal(resolved.url, "https://worker.example.test/neural");
  assert.equal(resolved.token, "secret-token");
});

test("does not fall back to python3 inside the Vercel Node runtime", () => {
  const resolved = resolveNeuralRepairRuntimeTarget({
    commandLine: ".venv-neural\\Scripts\\python.exe",
    cwd: path.resolve("runtime-root"),
    platform: "linux",
    isVercel: true,
    exists: () => false,
  });

  assert.equal(resolved.kind, "unavailable");
  assert.match(resolved.reason, /remote ClearVoice worker/i);
});
