# APEX Compute benchmark report: apex-bench-arm64.json

## Environment

- Runner: `apex-runner`
- Architecture: `arm64`
- Operating system: `macos`

## Results

| M × N × K | Variant | Kernel | Threads requested / used | Latency (ms) | GFLOPS | Speedup |
|---:|:---|:---|---:|---:|---:|---:|
| 64 × 64 × 64 | naive | naive | 1 / 1 | 0.1453 | 3.607 | 1.000× |
| 64 × 64 × 64 | engine | scalar | 1 / 1 | 0.1627 | 3.223 | 0.893× |
| 64 × 64 × 64 | naive | naive | 1 / 1 | 0.1498 | 3.501 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 1 / 1 | 0.0198 | 26.546 | 7.583× |
| 128 × 128 × 128 | naive | naive | 1 / 1 | 1.6920 | 2.479 | 1.000× |
| 128 × 128 × 128 | engine | scalar | 1 / 1 | 1.7319 | 2.422 | 0.977× |
| 128 × 128 × 128 | naive | naive | 1 / 1 | 1.5824 | 2.651 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 1 / 1 | 0.1629 | 25.742 | 9.712× |
| 256 × 256 × 256 | naive | naive | 1 / 1 | 16.3972 | 2.046 | 1.000× |
| 256 × 256 × 256 | engine | scalar | 1 / 1 | 15.7115 | 2.136 | 1.044× |
| 256 × 256 × 256 | naive | naive | 1 / 1 | 15.6925 | 2.138 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 1 / 1 | 1.3579 | 24.711 | 11.557× |
| 512 × 512 × 512 | naive | naive | 1 / 1 | 151.8610 | 1.768 | 1.000× |
| 512 × 512 × 512 | engine | scalar | 1 / 1 | 145.3688 | 1.847 | 1.045× |
| 512 × 512 × 512 | naive | naive | 1 / 1 | 145.0819 | 1.850 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 1 / 1 | 10.7174 | 25.047 | 13.537× |

Speedup is computed against the immediately preceding naive run for the same shape and kernel-selection batch. The engine scalar path is between 0.893× and 1.045× of naive on these cases; NEON is between 7.583× and 13.537×. Timings and throughput values are unchanged from the recorded JSON.
