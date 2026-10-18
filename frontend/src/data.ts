import type { BenchmarkSnapshot, ImportedLog, Kernel, MatrixSize } from "./types";

const measured: Record<MatrixSize, { engine: number; naive: number; latency: number }> = {
  64: { engine: 18.423, naive: 2.306, latency: 0.0285 },
  128: { engine: 26.645, naive: 2.531, latency: 0.1574 },
  256: { engine: 25.448, naive: 2.204, latency: 1.3185 },
  512: { engine: 25.855, naive: 1.863, latency: 10.3825 },
};

const kernelMultiplier: Record<Kernel, number> = {
  scalar: 0.083,
  "neon-4x4": 1,
  avx512: 1.42,
};

export const sizes: MatrixSize[] = [64, 128, 256, 512];

export const getSnapshot = (size: MatrixSize, kernel: Kernel): BenchmarkSnapshot => {
  const base = measured[size];
  const gflops = Number((base.engine * kernelMultiplier[kernel]).toFixed(2));
  const speedup = Number((gflops / base.naive).toFixed(2));

  return {
    size,
    kernel,
    gflops,
    naiveGflops: base.naive,
    speedup,
    latencyMs: Number((base.latency / kernelMultiplier[kernel]).toFixed(4)),
    threads: 4,
    l1Hit: kernel === "scalar" ? 74.8 : kernel === "avx512" ? 98.1 : 96.8,
    l2Hit: kernel === "scalar" ? 61.3 : kernel === "avx512" ? 92.7 : 89.2,
    bandwidth: Number((kernel === "scalar" ? 18.4 : kernel === "avx512" ? 112.8 : 78.6).toFixed(1)),
    tile: { mc: 64, kc: 128, nc: 128 },
    source: "sample",
  };
};

export const normalizeKernel = (kernel?: string): Kernel => {
  if (kernel === "avx512" || kernel === "avx512f") return "avx512";
  if (kernel === "neon" || kernel === "neon-4x4") return "neon-4x4";
  return "scalar";
};

const nearestSize = (value: number): MatrixSize => {
  const match = sizes.find((size) => size === value);
  if (match) return match;
  return sizes.reduce((closest, size) =>
    Math.abs(size - value) < Math.abs(closest - value) ? size : closest,
  64);
};

export const snapshotFromImportedLog = (log: ImportedLog): BenchmarkSnapshot | null => {
  const rows = log.results ?? [];
  const engineRows = rows.filter((row) => row.variant === "engine" && row.shape?.m);
  const row = engineRows.at(-1) ?? rows.at(-1);
  if (!row?.shape?.m || !row.gflops) return null;

  const size = nearestSize(row.shape.m);
  const kernel = normalizeKernel(row.kernel ?? log.environment?.info?.kernel);
  const baseline = rows.find(
    (candidate) =>
      candidate.variant === "naive" &&
      candidate.shape?.m === row.shape?.m &&
      candidate.shape?.n === row.shape?.n &&
      candidate.shape?.k === row.shape?.k,
  );
  const snapshot = getSnapshot(size, kernel);
  const naiveGflops = baseline?.gflops ?? snapshot.naiveGflops;
  const gflops = row.gflops;

  return {
    ...snapshot,
    gflops: Number(gflops.toFixed(2)),
    naiveGflops: Number(naiveGflops.toFixed(2)),
    speedup: Number((row.speedup ?? gflops / naiveGflops).toFixed(2)),
    latencyMs: Number(((row.median_ns ?? snapshot.latencyMs * 1e6) / 1e6).toFixed(4)),
    threads: row.threads ?? snapshot.threads,
    source: "imported",
  };
};

export const formatNumber = (value: number, digits = 1): string =>
  value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
