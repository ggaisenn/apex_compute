# APEX Compute benchmark report: local-benchmark.json

## Environment

- Runner: `apex-runner`
- Architecture: `arm64`
- Operating system: `macos`
- SIMD features: NEON `True`, AVX-512 `False`
- Cache sizes (L1d / L2 / L3): `65536` / `4194304` / `8388608` bytes
- Runs: `1` launch(es), `2` warm-up(s), `5` repetitions per launch; cache state `warm`

## Results

| M × N × K | Variant | Kernel | Threads requested / used | Latency (ms) | GFLOPS | Speedup |
|---:|:---|:---|---:|---:|---:|---:|
| 64 × 64 × 64 | naive | naive | 1 / 1 | 0.1388 | 3.778 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 1 / 1 | 0.0194 | 27.060 | 7.163× |
| 64 × 64 × 64 | naive | naive | 2 / 1 | 0.1388 | 3.778 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 2 / 1 | 0.0193 | 27.178 | 7.195× |
| 64 × 64 × 64 | naive | naive | 4 / 1 | 0.1388 | 3.779 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 4 / 1 | 0.0193 | 27.119 | 7.177× |
| 128 × 128 × 128 | naive | naive | 1 / 1 | 1.5238 | 2.753 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 1 / 1 | 0.1518 | 27.624 | 10.036× |
| 128 × 128 × 128 | naive | naive | 2 / 1 | 1.5296 | 2.742 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 2 / 1 | 0.1521 | 27.571 | 10.055× |
| 128 × 128 × 128 | naive | naive | 4 / 1 | 1.5226 | 2.755 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 4 / 1 | 0.1518 | 27.624 | 10.028× |
| 256 × 256 × 256 | naive | naive | 1 / 1 | 16.2092 | 2.070 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 1 / 1 | 1.3704 | 24.485 | 11.828× |
| 256 × 256 × 256 | naive | naive | 2 / 1 | 16.1445 | 2.078 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 2 / 1 | 1.3571 | 24.725 | 11.896× |
| 256 × 256 × 256 | naive | naive | 4 / 1 | 16.4813 | 2.036 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 4 / 1 | 1.4047 | 23.886 | 11.733× |
| 512 × 512 × 512 | naive | naive | 1 / 1 | 144.3144 | 1.860 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 1 / 1 | 10.3234 | 26.003 | 13.979× |
| 512 × 512 × 512 | naive | naive | 2 / 1 | 143.6656 | 1.868 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 2 / 1 | 10.6032 | 25.316 | 13.549× |
| 512 × 512 × 512 | naive | naive | 4 / 1 | 154.0384 | 1.743 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 4 / 1 | 58.3410 | 4.601 | 2.640× |

Speedup compares each engine run with its corresponding naive run for the same shape and requested kernel. Legacy logs without requested-kernel metadata use the most recent preceding naive run for that shape.

Generated from the recorded runner JSON output; timings are unchanged.
