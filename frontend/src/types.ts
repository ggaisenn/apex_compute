export type Page = "home" | "execution" | "simd" | "threads";
export type Theme = "dark" | "light";
export type Kernel = "scalar" | "neon-4x4" | "avx512";
export type RunMode = "engine" | "naive";
export type MatrixSize = 64 | 128 | 256 | 512;

export interface BenchmarkSnapshot {
  size: MatrixSize;
  kernel: Kernel;
  gflops: number;
  naiveGflops: number;
  speedup: number;
  latencyMs: number;
  threads: number;
  l1Hit: number;
  l2Hit: number;
  bandwidth: number;
  tile: { mc: number; kc: number; nc: number };
  source: "sample" | "imported";
}

export interface ImportedLog {
  schema?: { name?: string; version?: number };
  environment?: {
    info?: {
      kernel?: string;
    };
  };
  results?: Array<{
    shape?: { m?: number; n?: number; k?: number };
    variant?: string;
    kernel?: string;
    threads?: number;
    gflops?: number;
    median_ns?: number;
    speedup?: number;
  }>;
}
