#include "bench_engine.hpp"

#include "../core/gemm/naive_baseline.hpp"
#include "../core/platform/timer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <new>
#include <ostream>
#include <sstream>
#include <utility>

namespace apex::gemm {
namespace {

[[nodiscard]] bool checked_element_count(
    std::size_t rows,
    std::size_t columns,
    std::size_t* count) noexcept {
    if (count == nullptr ||
        (columns != 0 && rows > static_cast<std::size_t>(-1) / columns)) {
        return false;
    }
    *count = rows * columns;
    return true;
}

void save_output(MatrixView matrix, std::vector<float>* saved) {
    std::size_t count = 0;
    if (saved == nullptr || !checked_element_count(matrix.rows(), matrix.columns(), &count)) {
        throw std::bad_alloc();
    }
    saved->resize(count);
    std::size_t index = 0;
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            (*saved)[index++] = matrix(row, column);
        }
    }
}

void restore_output(MatrixView matrix, const std::vector<float>& saved) noexcept {
    std::size_t index = 0;
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            matrix(row, column) = saved[index++];
        }
    }
}

[[nodiscard]] double consume_output(MatrixView matrix) noexcept {
    double checksum = 0.0;
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            checksum += static_cast<double>(matrix(row, column));
        }
    }
    return checksum;
}

[[nodiscard]] double median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() % 2U) != 0U) {
        return static_cast<double>(values[middle]);
    }
    return (static_cast<double>(values[middle - 1]) +
            static_cast<double>(values[middle])) / 2.0;
}

[[nodiscard]] Status time_variant(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    const std::vector<float>& pristine_output,
    BenchmarkVariant variant,
    const BenchmarkConfig& config,
    std::size_t warmups,
    std::size_t repetitions,
    BenchmarkVariantResult* result) {
    if (result == nullptr || variant.name == nullptr || variant.operation == nullptr) {
        return Status::InvalidArgument;
    }

    BenchmarkVariantResult replacement;
    replacement.name = variant.name;
    replacement.durations_ns.reserve(repetitions);
    for (std::size_t index = 0; index < warmups; ++index) {
        restore_output(c, pristine_output);
        const Status status = variant.operation(
            a, b, c, config.alpha, config.beta, variant.context);
        if (status != Status::Success) {
            return status;
        }
    }

    for (std::size_t index = 0; index < repetitions; ++index) {
        restore_output(c, pristine_output);
        const Timer::TimePoint start = Timer::now();
        const Status status = variant.operation(
            a, b, c, config.alpha, config.beta, variant.context);
        const Timer::TimePoint end = Timer::now();
        if (status != Status::Success) {
            return status;
        }
        const auto elapsed = Timer::elapsed_between(start, end).count();
        replacement.durations_ns.push_back(
            elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0U);
        replacement.checksum = consume_output(c);
    }

    const auto bounds = std::minmax_element(
        replacement.durations_ns.begin(), replacement.durations_ns.end());
    replacement.minimum_ns = static_cast<double>(*bounds.first);
    replacement.maximum_ns = static_cast<double>(*bounds.second);
    replacement.median_ns = median(replacement.durations_ns);
    *result = std::move(replacement);
    return Status::Success;
}

[[nodiscard]] double nominal_gflops(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    double median_ns) noexcept {
    if (median_ns <= 0.0) {
        return 0.0;
    }
    const double operations = 2.0 * static_cast<double>(m) *
        static_cast<double>(n) * static_cast<double>(k);
    return operations / median_ns;
}

[[nodiscard]] std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (character < 0x20U || character > 0x7eU) {
                    escaped << "\\u00" << std::hex << std::setw(2)
                            << std::setfill('0') << static_cast<unsigned int>(character)
                            << std::dec;
                } else {
                    escaped << static_cast<char>(character);
                }
                break;
        }
    }
    escaped << '"';
    return escaped.str();
}

void write_variant_json(
    const BenchmarkVariantResult& result,
    std::ostream& output) {
    output << "{\"name\":" << json_escape(result.name)
           << ",\"durations_ns\":[";
    for (std::size_t index = 0; index < result.durations_ns.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << result.durations_ns[index];
    }
    output << "],\"median_ns\":" << result.median_ns
           << ",\"minimum_ns\":" << result.minimum_ns
           << ",\"maximum_ns\":" << result.maximum_ns
           << ",\"latency_ns_per_multiplication\":" << result.median_ns
           << ",\"gflops\":" << result.gflops
           << ",\"speedup\":" << result.speedup
           << ",\"checksum\":" << result.checksum << '}';
}

}  // namespace

Status naive_baseline_benchmark_operation(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    float alpha,
    float beta,
    void* context) noexcept {
    (void)context;
    return naive_baseline(a, b, c, alpha, beta);
}

Status BenchmarkEngine::run(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    BenchmarkVariant naive_baseline,
    BenchmarkVariant candidate,
    const BenchmarkConfig& config,
    BenchmarkReport* report) {
    if (report == nullptr || config.repetitions == 0 ||
        naive_baseline.name == nullptr || naive_baseline.operation == nullptr ||
        candidate.name == nullptr || candidate.operation == nullptr) {
        return Status::InvalidArgument;
    }
    const Status validation_status = detail::validate_gemm_views(a, b, c);
    if (validation_status != Status::Success) {
        return validation_status;
    }
    if (c.rows() == 0 || c.columns() == 0 || a.columns() == 0) {
        return Status::EmptyMatrix;
    }

    std::vector<float> pristine_output;
    try {
        save_output(c, &pristine_output);

        BenchmarkReport replacement;
        replacement.m = a.rows();
        replacement.n = b.columns();
        replacement.k = a.columns();
        replacement.warmup_iterations = config.warmup_iterations;
        replacement.repetitions = config.repetitions;

        Status status = time_variant(
            a, b, c, pristine_output, naive_baseline, config,
            config.warmup_iterations, config.repetitions,
            &replacement.naive_baseline);
        if (status != Status::Success) {
            restore_output(c, pristine_output);
            return status;
        }
        status = time_variant(
            a, b, c, pristine_output, candidate, config,
            config.warmup_iterations, config.repetitions,
            &replacement.candidate);
        restore_output(c, pristine_output);
        if (status != Status::Success) {
            return status;
        }

        replacement.naive_baseline.gflops = nominal_gflops(
            replacement.m, replacement.n, replacement.k,
            replacement.naive_baseline.median_ns);
        replacement.naive_baseline.speedup = 1.0;
        replacement.candidate.gflops = nominal_gflops(
            replacement.m, replacement.n, replacement.k,
            replacement.candidate.median_ns);
        replacement.candidate.speedup =
            replacement.candidate.median_ns > 0.0
            ? replacement.naive_baseline.median_ns / replacement.candidate.median_ns
            : 0.0;
        replacement.latency_ns_per_multiplication =
            replacement.candidate.median_ns;
        *report = std::move(replacement);
        return Status::Success;
    } catch (...) {
        std::size_t expected_elements = 0;
        if (checked_element_count(c.rows(), c.columns(), &expected_elements) &&
            pristine_output.size() == expected_elements) {
            restore_output(c, pristine_output);
        }
        return Status::AllocationFailed;
    }
}

Status BenchmarkEngine::write_terminal_table(
    const BenchmarkReport& report,
    std::ostream& output) {
    output << "GEMM benchmark (M=" << report.m << ", N=" << report.n
           << ", K=" << report.k << ")\n"
           << "Warm-up: " << report.warmup_iterations
           << "  Repetitions: " << report.repetitions << "\n\n"
           << std::left << std::setw(24) << "Variant"
           << std::right << std::setw(16) << "Latency (us)"
           << std::setw(14) << "GFLOPS"
           << std::setw(12) << "Speedup" << '\n'
           << std::string(66, '-') << '\n';

    output << std::left << std::setw(24) << report.naive_baseline.name
           << std::right << std::fixed << std::setprecision(3)
           << std::setw(16) << report.naive_baseline.median_ns / 1000.0
           << std::setw(14) << report.naive_baseline.gflops
           << std::setw(12) << report.naive_baseline.speedup << '\n';
    output << std::left << std::setw(24) << report.candidate.name
           << std::right << std::fixed << std::setprecision(3)
           << std::setw(16) << report.candidate.median_ns / 1000.0
           << std::setw(14) << report.candidate.gflops
           << std::setw(12) << report.candidate.speedup << '\n';
    return output ? Status::Success : Status::InternalError;
}

Status BenchmarkEngine::write_json(
    const BenchmarkReport& report,
    std::ostream& output) {
    output << std::setprecision(17)
           << "{\"schema\":{\"name\":\"apex.bench.result\",\"version\":1}"
           << ",\"shape\":{\"m\":" << report.m
           << ",\"n\":" << report.n << ",\"k\":" << report.k << '}'
           << ",\"methodology\":{\"warmup_runs\":"
           << report.warmup_iterations << ",\"repetitions\":"
           << report.repetitions << '}'
           << ",\"results\":[";
    write_variant_json(report.naive_baseline, output);
    output << ',';
    write_variant_json(report.candidate, output);
    output << "],\"summary\":{\"latency_ns_per_multiplication\":"
           << report.latency_ns_per_multiplication
           << ",\"speedup_over_naive\":" << report.candidate.speedup
           << ",\"best_gflops\":"
           << std::max(report.naive_baseline.gflops, report.candidate.gflops)
           << "}}\n";
    return output ? Status::Success : Status::InternalError;
}

}  // namespace apex::gemm
