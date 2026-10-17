#pragma once

#include "../core/memory/matrix.hpp"
#include "../core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace apex::gemm {

using BenchmarkOperation = Status (*)(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    float alpha,
    float beta,
    void* context) noexcept;

[[nodiscard]] Status naive_baseline_benchmark_operation(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    float alpha,
    float beta,
    void* context) noexcept;

struct BenchmarkVariant {
    const char* name = nullptr;
    BenchmarkOperation operation = nullptr;
    void* context = nullptr;
};

struct BenchmarkConfig {
    std::size_t warmup_iterations = 2;
    std::size_t repetitions = 10;
    float alpha = 1.0F;
    float beta = 0.0F;
};

struct BenchmarkVariantResult {
    std::string name;
    std::vector<std::uint64_t> durations_ns;
    double median_ns = 0.0;
    double minimum_ns = 0.0;
    double maximum_ns = 0.0;
    double gflops = 0.0;
    double speedup = 0.0;
    double checksum = 0.0;
};

struct BenchmarkReport {
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    std::size_t warmup_iterations = 0;
    std::size_t repetitions = 0;
    double latency_ns_per_multiplication = 0.0;
    BenchmarkVariantResult naive_baseline;
    BenchmarkVariantResult candidate;
};

class BenchmarkEngine {
public:
    [[nodiscard]] static Status run(
        MatrixView a,
        MatrixView b,
        MatrixView c,
        BenchmarkVariant naive_baseline,
        BenchmarkVariant candidate,
        const BenchmarkConfig& config,
        BenchmarkReport* report);

    [[nodiscard]] static Status write_terminal_table(
        const BenchmarkReport& report,
        std::ostream& output);

    [[nodiscard]] static Status write_json(
        const BenchmarkReport& report,
        std::ostream& output);
};

}  // namespace apex::gemm
