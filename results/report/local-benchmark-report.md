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
| 64 × 64 × 64 | naive | naive | 1 / 1 | 0.2273 | 2.306 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 1 / 1 | 0.0285 | 18.423 | 7.988× |
| 64 × 64 × 64 | naive | naive | 2 / 1 | 0.1876 | 2.795 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 2 / 1 | 0.0272 | 19.299 | 6.905× |
| 64 × 64 × 64 | naive | naive | 4 / 1 | 0.1727 | 3.036 | 1.000× |
| 64 × 64 × 64 | engine | neon-4x4 | 4 / 1 | 0.0226 | 23.173 | 7.634× |
| 128 × 128 × 128 | naive | naive | 1 / 1 | 1.6571 | 2.531 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 1 / 1 | 0.1574 | 26.645 | 10.527× |
| 128 × 128 × 128 | naive | naive | 2 / 1 | 1.7205 | 2.438 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 2 / 1 | 0.1573 | 26.666 | 10.939× |
| 128 × 128 × 128 | naive | naive | 4 / 1 | 1.5501 | 2.706 | 1.000× |
| 128 × 128 × 128 | engine | neon-4x4 | 4 / 1 | 0.1519 | 27.609 | 10.203× |
| 256 × 256 × 256 | naive | naive | 1 / 1 | 15.2235 | 2.204 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 1 / 1 | 1.3185 | 25.448 | 11.546× |
| 256 × 256 × 256 | naive | naive | 2 / 1 | 15.2902 | 2.195 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 2 / 1 | 1.3276 | 25.274 | 11.517× |
| 256 × 256 × 256 | naive | naive | 4 / 1 | 15.1613 | 2.213 | 1.000× |
| 256 × 256 × 256 | engine | neon-4x4 | 4 / 1 | 1.3870 | 24.193 | 10.931× |
| 512 × 512 × 512 | naive | naive | 1 / 1 | 144.1058 | 1.863 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 1 / 1 | 10.3825 | 25.855 | 13.880× |
| 512 × 512 × 512 | naive | naive | 2 / 1 | 147.2707 | 1.823 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 2 / 1 | 10.6465 | 25.214 | 13.833× |
| 512 × 512 × 512 | naive | naive | 4 / 1 | 145.8603 | 1.840 | 1.000× |
| 512 × 512 × 512 | engine | neon-4x4 | 4 / 1 | 10.9170 | 24.589 | 13.361× |

Speedup compares each engine run with its corresponding naive run for the same shape and requested kernel. Legacy logs without requested-kernel metadata use the most recent preceding naive run for that shape.

Generated from the recorded runner JSON output; timings are unchanged.
