# APEX-COMPUTE

A runtime engine and model harness built to accelerate dense matrix multiplication (GEMM) on commodity CPUs via cache tiling, panel packing, and explicit SIMD vectorization (ARM NEON & AVX-512).

## Why this project

Dense matrix multiplication is a major building block in transformer inference. The attention and feed-forward parts of transformer models perform large matrix operations; their throughput depends in part on how efficiently the hardware moves and multiplies those values. APEX-COMPUTE focuses on that systems layer: aligned storage, cache-aware tiling, packed panels, SIMD micro-kernels, and parallel execution. It is a GEMM engine and benchmark project; it does not itself load or serve a language model.

## Architecture and technology

| Area | What it does | Main technologies |
| --- | --- | --- |
| Memory and platform | 64-byte aligned buffers, row-padded matrices, CPU/cache detection, timing and status codes | C++20, standard library and native OS queries |
| Kernels | Scalar reference, packed-panel kernels, NEON and AVX-512 micro-kernels | C++20, ARM NEON and Intel intrinsics |
| Orchestration | Tile planning, blocked GEMM scheduling, dispatch and worker execution | C++20, pthreads / platform threads |
| Runner and public entry points | JSON `info`, `plan`, `run` and `selftest` process contract | C++20, `apex-runner` |
| Validation and benchmarking | Differential checks, tolerance policy, timings and reports | Python standard library and C++ runner |
| Optional companion | Natural-language intent, plan validation, safe local tool execution and grounded explanation | Python 3.9+, `urllib`, `json`, `subprocess`, Gemini API |

The normal C++ build has no third-party runtime library dependency. CMake, a C++20 compiler and Ninja are build tools; Python 3 is needed for the validation, benchmark and companion command-line clients. The companion is optional and is not required to compile or run the C++ engine.

## Quick start

### Requirements

- Git
- CMake 3.21 or newer
- Ninja
- A C++20 compiler: GCC or Clang on Linux, Apple Clang on macOS, or MSVC on Windows
- Python 3 for the Python command-line clients

### Clone and build

```sh
git clone https://github.com/ggaisenn/apex_compute.git
cd apex_compute
cmake --preset release
cmake --build --preset release
```

The `release` preset builds `apex-runner` and `apex-validate` under `build/release/`. The Python clients discover the built runner from CMake's generated runner-location metadata. You can also pass its path with `--runner` or set `APEX_RUNNER`.

### Run the tools

Run a quick differential correctness check:

```sh
python3 apex-validate.py --sizes 16,31,32,33 --kernels auto,scalar --threads 1
```

Run a benchmark and save JSON results, a Markdown table and SVG charts:

```sh
python3 apex-bench.py \
  --sizes 64,128,256,512 \
  --threads 1,2,4 \
  --kernels auto \
  --variants naive,engine \
  --warmup 2 --reps 5 \
  --json-out results/local-benchmark.json \
  --report-out results/report
```

The command prints per-case latency and GFLOPS. The report directory receives a Markdown summary and SVG throughput and speedup charts. For a runner handshake and environment record:

```sh
python3 apex-bench.py --environment
```

The runner also supports direct `info`, `plan`, `run` and `selftest` modes. See the wrappers' built-in help for all supported options:

```sh
python3 apex-validate.py --help
python3 apex-bench.py --help
```

### Build presets

```sh
cmake --preset debug && cmake --build --preset debug
cmake --preset release && cmake --build --preset release
cmake --preset portable && cmake --build --preset portable
cmake --preset asan && cmake --build --preset asan
```

`portable` forces the scalar implementation. `asan` enables AddressSanitizer and UndefinedBehaviorSanitizer where supported. All presets enforce warnings as errors (`/W4 /WX` on MSVC; `-Wall -Wextra -Wpedantic -Werror` on GCC and Clang).

## Performance snapshot

The charts below were generated from a local Release build on macOS arm64 using the auto-selected `neon-4x4` engine kernel. Measurements used warm cache, seed 1, two warm-up runs and five timed repetitions. The runner reports architecture and OS but does not report the exact Mac model, so these are a reproducible sample from that host rather than a universal performance claim.

![Throughput for the naive and engine variants at one requested thread, measured in GFLOPS](docs/assets/readme-throughput-arm64-macos.svg)

![Engine speedup over the naive baseline at one requested thread](docs/assets/readme-speedup-arm64-macos.svg)

| Square size | Naive GFLOPS | Engine GFLOPS | Engine median latency | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3.483 | 8.842 | 0.0593 ms | 2.54× |
| 128 | 1.680 | 20.418 | 0.2054 ms | 12.16× |
| 256 | 2.185 | 25.172 | 1.3330 ms | 11.52× |

The values are medians from the benchmark run on this checkout. Re-run the command above on the target host before comparing results: compiler, CPU model, power mode, thermal state, OS scheduling, selected kernel and matrix sizes all affect throughput. A requested thread count can differ from the count reported as used, especially for sizes too small to benefit from parallel work.


### Track alignment

```text
User request
    │
    ▼
Gemma 4 through Gemini API ── proposes structured plan
    │
    ▼
Strict schema + operation allow-list + range/work caps
    │                         (reject unsupported fields/actions)
    ▼
Code-built argv ── user confirmation ── local Python tool process
                                           │
                                           ▼
                                    apex-runner / GEMM
                                           │
                                           ▼
                                    JSON result → facts table
                                           │
                                           ▼
                                    optional Gemma explanation
```

### Using Companion Offline Interpreter

Use the companion's recorded offline interpreter for a local run:

```sh
python3 -m companion --offline --yes "validate edge cases"
python3 -m companion --offline --explain --yes "benchmark a small square multiply"
```

The first command uses the checked-in interpretation fixture and runs the real local validator. The second uses a checked-in interpretation and explanation fixture while running the local benchmark. To replay recorded results without a built runner, run:

```sh
python3 -m companion --offline --replay-results --yes "benchmark a small square multiply"
```



## License

APEX-COMPUTE is distributed under the [Apache License, Version 2.0](LICENSE). The repository includes the complete license text.
