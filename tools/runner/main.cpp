#include "../../core/gemm/kernel_dispatcher.hpp"
#include "../../core/gemm/naive_baseline.hpp"
#include "../../core/gemm/scalar_reference.hpp"
#include "../../core/memory/matrix_fill.hpp"
#include "../../core/orchestration/gemm_engine.hpp"
#include "../../core/orchestration/tiling_planner.hpp"
#include "../../core/platform/timer.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using apex::gemm::Status;

struct Options {
    std::string mode = "info";
    std::string variant = "engine";
    std::string kernel = "auto";
    std::size_t m = 128;
    std::size_t n = 128;
    std::size_t k = 128;
    std::size_t threads = 1;
    std::size_t warmup = 2;
    std::size_t reps = 5;
    std::size_t mc = 0;
    std::size_t kc = 0;
    std::size_t nc = 0;
    std::uint64_t seed = 1;
    float alpha = 1.0F;
    float beta = 0.0F;
    std::string json_out;
};

[[nodiscard]] bool parse_size(std::string_view text, std::size_t* output) {
    if (output == nullptr || text.empty()) return false;
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return false;
    *output = value;
    return true;
}

[[nodiscard]] bool parse_u64(std::string_view text, std::uint64_t* output) {
    if (output == nullptr || text.empty()) return false;
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return false;
    *output = value;
    return true;
}

[[nodiscard]] bool parse_float(const char* text, float* output) {
    if (text == nullptr || output == nullptr) return false;
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) return false;
    *output = value;
    return true;
}

[[nodiscard]] bool parse_options(int argc, char** argv, Options* options) {
    if (options == nullptr) return false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view key(argv[index]);
        if (key == "--help" || key == "-h") {
            std::cout << "apex-runner --mode info|plan|run|selftest [options]\n"
                      << "  --variant engine|naive|reference --kernel auto|scalar|neon|avx512\n"
                      << "  --m N --n N --k N --threads N --warmup N --reps N --seed N\n"
                      << "  --alpha X --beta X --mc N --kc N --nc N --json-out FILE\n";
            std::exit(0);
        }
        if (index + 1 >= argc) return false;
        const char* value = argv[++index];
        std::size_t size = 0;
        if (key == "--mode") options->mode = value;
        else if (key == "--variant") options->variant = value;
        else if (key == "--kernel") options->kernel = value;
        else if (key == "--json-out") options->json_out = value;
        else if (key == "--m" || key == "--n" || key == "--k" ||
                 key == "--threads" || key == "--warmup" || key == "--reps" ||
                 key == "--mc" || key == "--kc" || key == "--nc") {
            if (!parse_size(value, &size)) return false;
            if (key == "--m") options->m = size;
            else if (key == "--n") options->n = size;
            else if (key == "--k") options->k = size;
            else if (key == "--threads") options->threads = size;
            else if (key == "--warmup") options->warmup = size;
            else if (key == "--reps") options->reps = size;
            else if (key == "--mc") options->mc = size;
            else if (key == "--kc") options->kc = size;
            else options->nc = size;
        } else if (key == "--seed") {
            if (!parse_u64(value, &options->seed)) return false;
        } else if (key == "--alpha") {
            if (!parse_float(value, &options->alpha)) return false;
        } else if (key == "--beta") {
            if (!parse_float(value, &options->beta)) return false;
        } else {
            return false;
        }
    }
    const bool valid_kernel = options->kernel == "auto" || options->kernel == "scalar" ||
        options->kernel == "neon" || options->kernel == "avx512";
    return valid_kernel && (options->mode == "info" || options->mode == "plan" ||
        options->mode == "run" || options->mode == "selftest");
}

[[nodiscard]] apex::gemm::KernelPreference preference(const std::string& name) {
    if (name == "scalar") return apex::gemm::KernelPreference::Scalar;
    if (name == "neon") return apex::gemm::KernelPreference::Neon;
    if (name == "avx512") return apex::gemm::KernelPreference::Avx512;
    return apex::gemm::KernelPreference::Automatic;
}

[[nodiscard]] Status select_kernel(
    const Options& options, apex::gemm::KernelDescriptor* descriptor) {
    return apex::gemm::KernelDispatcher::select(
        apex::gemm::KernelSelectionOptions{preference(options.kernel)}, descriptor);
}

[[nodiscard]] Status make_plan(
    const Options& options,
    const apex::gemm::KernelDescriptor& kernel,
    apex::gemm::TilingPlan* plan) {
    apex::gemm::TilingPlannerOptions planner_options;
    planner_options.tile = {kernel.mr, kernel.nr, kernel.kr};
    planner_options.thread_count = options.threads;
    planner_options.overrides = {options.mc, options.kc, options.nc};
    apex::gemm::TilingPlanner planner;
    Status status = apex::gemm::TilingPlanner::create(planner_options, &planner);
    if (status != Status::Success) return status;
    return planner.plan(options.m, options.k, options.n, plan);
}

[[nodiscard]] std::uint64_t checksum(apex::gemm::MatrixView matrix) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(matrix(row, column));
            for (unsigned shift = 0; shift < 32; shift += 8) {
                hash ^= static_cast<std::uint8_t>(bits >> shift);
                hash *= UINT64_C(1099511628211);
            }
        }
    }
    return hash;
}

[[nodiscard]] const char* architecture_name() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

[[nodiscard]] const char* os_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

void write_info(std::ostream& out) {
    const auto& cpu = apex::gemm::CpuFeatures::current();
    apex::gemm::KernelDescriptor selected{};
    const Status status = apex::gemm::KernelDispatcher::select({}, &selected);
    const auto cache = [](apex::gemm::CacheSize size) {
        return size.bytes;
    };
    out << "{\"schema\":{\"name\":\"apex.runner.info\",\"version\":1}"
        << ",\"contract\":{\"major\":1,\"minor\":0}"
        << ",\"runner\":\"apex-runner\",\"library_version\":\"0.1.0\""
        << ",\"architecture\":\"" << architecture_name() << "\""
        << ",\"os\":\"" << os_name() << "\""
        << ",\"features\":{\"neon\":" << (cpu.neon() ? "true" : "false")
        << ",\"avx2\":" << (cpu.avx2() ? "true" : "false")
        << ",\"avx512f\":" << (cpu.avx512f() ? "true" : "false")
        << ",\"fma\":" << (cpu.fma() ? "true" : "false") << '}'
        << ",\"cache_bytes\":{\"l1d\":" << cache(cpu.l1d_cache())
        << ",\"l2\":" << cache(cpu.l2_cache())
        << ",\"l3\":" << cache(cpu.l3_cache())
        << ",\"line\":" << cache(cpu.cache_line()) << '}'
        << ",\"kernel\":\""
        << (status == Status::Success ? selected.name : "unavailable") << "\""
        << ",\"variants\":[\"engine\",\"naive\",\"reference\"]} \n";
}

void write_plan(
    std::ostream& out,
    const Options& options,
    const apex::gemm::KernelDescriptor& kernel,
    const apex::gemm::TilingPlan& plan) {
    out << "{\"schema\":{\"name\":\"apex.runner.plan\",\"version\":1}"
        << ",\"status\":\"Ok\",\"kernel\":\"" << kernel.name << "\""
        << ",\"tile\":{\"mr\":" << kernel.mr << ",\"nr\":" << kernel.nr << '}'
        << ",\"shape\":{\"m\":" << options.m << ",\"n\":" << options.n
        << ",\"k\":" << options.k << '}'
        << ",\"blocks\":{\"mc\":" << plan.mc << ",\"kc\":" << plan.kc
        << ",\"nc\":" << plan.nc << '}'
        << ",\"threads_requested\":" << options.threads
        << ",\"threads_used\":1}";
}

[[nodiscard]] Status run_gemm(
    const Options& options,
    apex::gemm::MatrixView a,
    apex::gemm::MatrixView b,
    apex::gemm::MatrixView c,
    const apex::gemm::KernelDescriptor& kernel,
    const apex::gemm::TilingPlan& plan,
    apex::gemm::GemmWorkspace* workspace) {
    if (options.variant == "naive") {
        return apex::gemm::naive_baseline(a, b, c, options.alpha, options.beta);
    }
    if (options.variant == "reference") {
        return apex::gemm::scalar_reference(a, b, c, options.alpha, options.beta);
    }
    return apex::gemm::GemmEngine::execute(
        a, b, c, plan, {kernel.mr, kernel.nr, kernel.kr}, workspace,
        kernel.function, options.alpha, options.beta);
}

void restore_output(
    apex::gemm::MatrixView output,
    apex::gemm::MatrixView initial_output) noexcept {
    for (std::size_t row = 0; row < output.rows(); ++row) {
        for (std::size_t column = 0; column < output.columns(); ++column) {
            output(row, column) = initial_output(row, column);
        }
    }
}

[[nodiscard]] Status write_run(
    std::ostream& out,
    const Options& options,
    const apex::gemm::KernelDescriptor& kernel,
    const apex::gemm::TilingPlan& plan) {
    apex::gemm::Matrix a;
    apex::gemm::Matrix b;
    apex::gemm::Matrix c;
    apex::gemm::Matrix initial_c;
    Status status = apex::gemm::Matrix::create(options.m, options.k, &a);
    if (status != Status::Success) return status;
    status = apex::gemm::Matrix::create(options.k, options.n, &b);
    if (status != Status::Success) return status;
    status = apex::gemm::Matrix::create(options.m, options.n, &c);
    if (status != Status::Success) return status;
    status = apex::gemm::Matrix::create(options.m, options.n, &initial_c);
    if (status != Status::Success) return status;
    (void)apex::gemm::fill_random(a, options.seed);
    (void)apex::gemm::fill_random(b, options.seed + 1U);
    (void)apex::gemm::fill_random(initial_c, options.seed + 2U);
    apex::gemm::GemmWorkspace workspace;
    if (options.variant == "engine") {
        status = workspace.reserve(plan, {kernel.mr, kernel.nr, kernel.kr});
        if (status != Status::Success) return status;
    }

    for (std::size_t index = 0; index < options.warmup; ++index) {
        restore_output(c.view(), initial_c.view());
        status = run_gemm(options, a.view(), b.view(), c.view(), kernel, plan, &workspace);
        if (status != Status::Success) return status;
    }

    std::vector<std::uint64_t> durations;
    try {
        durations.reserve(options.reps);
    } catch (...) {
        return Status::AllocationFailed;
    }
    for (std::size_t index = 0; index < options.reps; ++index) {
        restore_output(c.view(), initial_c.view());
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto start = apex::gemm::Timer::now();
        status = run_gemm(options, a.view(), b.view(), c.view(), kernel, plan, &workspace);
        const auto end = apex::gemm::Timer::now();
        std::atomic_signal_fence(std::memory_order_seq_cst);
        if (status != Status::Success) return status;
        durations.push_back(static_cast<std::uint64_t>(
            apex::gemm::Timer::elapsed_between(start, end).count()));
    }

    double sum = 0.0;
    double absolute_sum = 0.0;
    double reference_absolute_sum = 0.0;
    double maximum_error = 0.0;
    double maximum_element_scale = 0.0;
    apex::gemm::Matrix reference;
    status = apex::gemm::Matrix::create(options.m, options.n, &reference);
    if (status != Status::Success) return status;
    for (std::size_t row = 0; row < reference.rows(); ++row) {
        for (std::size_t column = 0; column < reference.columns(); ++column) {
            reference.view()(row, column) = initial_c.view()(row, column);
        }
    }
    status = apex::gemm::scalar_reference(
        a.view(), b.view(), reference.view(), options.alpha, options.beta);
    if (status != Status::Success) return status;
    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t column = 0; column < c.columns(); ++column) {
            const double value = static_cast<double>(c.view()(row, column));
            sum += value;
            absolute_sum += std::abs(value);
            reference_absolute_sum += std::abs(
                static_cast<double>(reference.view()(row, column)));
            double element_scale = std::abs(static_cast<double>(options.beta)) *
                std::abs(static_cast<double>(initial_c.view()(row, column)));
            for (std::size_t inner = 0; inner < options.k; ++inner) {
                element_scale += std::abs(static_cast<double>(options.alpha)) *
                    std::abs(static_cast<double>(a.view()(row, inner)) *
                             static_cast<double>(b.view()(inner, column)));
            }
            maximum_element_scale = std::max(maximum_element_scale, element_scale);
            maximum_error = std::max(maximum_error,
                std::abs(value - static_cast<double>(reference.view()(row, column))));
        }
    }
    std::sort(durations.begin(), durations.end());
    const double minimum_ns = durations.empty() ? 0.0 : static_cast<double>(durations.front());
    const double maximum_ns = durations.empty() ? 0.0 : static_cast<double>(durations.back());
    const double median_ns = durations.empty() ? 0.0 :
        durations.size() % 2U == 0U
        ? (static_cast<double>(durations[durations.size() / 2U - 1U]) +
           static_cast<double>(durations[durations.size() / 2U])) / 2.0
        : static_cast<double>(durations[durations.size() / 2U]);
    const double gflops = median_ns > 0.0
        ? 2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
            static_cast<double>(options.k) / median_ns
        : 0.0;
    out << "{\"schema\":{\"name\":\"apex.runner.result\",\"version\":1}"
        << ",\"contract\":{\"major\":1,\"minor\":0}"
        << ",\"status\":\"Ok\",\"variant\":\"" << options.variant << "\""
        << ",\"kernel_used\":\""
        << (options.variant == "engine" ? kernel.name : options.variant) << "\""
        << ",\"shape\":{\"m\":" << options.m << ",\"n\":" << options.n
        << ",\"k\":" << options.k << '}'
        << ",\"blocks\":{\"mc\":" << plan.mc << ",\"kc\":" << plan.kc
        << ",\"nc\":" << plan.nc << '}'
        << ",\"threads_requested\":" << options.threads
        << ",\"threads_used\":1"
        << ",\"warmup_runs\":" << options.warmup << ",\"reps\":" << options.reps
        << ",\"durations_ns\":[";
    for (std::size_t index = 0; index < durations.size(); ++index) {
        if (index != 0) out << ',';
        out << durations[index];
    }
    out << "],\"minimum_ns\":" << minimum_ns
        << ",\"median_ns\":" << median_ns
        << ",\"maximum_ns\":" << maximum_ns
        << ",\"gflops\":" << gflops
        << ",\"c_hash\":" << checksum(c.view())
        << ",\"c_sum\":" << sum << ",\"c_abs_sum\":" << absolute_sum
        << ",\"reference_abs_sum\":" << reference_absolute_sum
        << ",\"max_abs_error\":" << maximum_error
        << ",\"max_element_scale\":" << maximum_element_scale << '}';
    return Status::Success;
}

void write_selftest(std::ostream& out) {
    apex::gemm::Matrix first;
    apex::gemm::Matrix second;
    const Status a = apex::gemm::Matrix::create(4, 4, &first);
    const Status b = apex::gemm::Matrix::create(4, 4, &second);
    const Status fill_a = a == Status::Success
        ? apex::gemm::fill_random(first, 42) : a;
    const Status fill_b = b == Status::Success
        ? apex::gemm::fill_random(second, 42) : b;
    const bool fill_ok = fill_a == Status::Success && fill_b == Status::Success &&
        checksum(first.view()) == checksum(second.view());
    const bool timer_ok = apex::gemm::Timer::resolution_nanoseconds() > 0;
    out << "{\"schema\":{\"name\":\"apex.runner.selftest\",\"version\":1}"
        << ",\"status\":\"" << (fill_ok && timer_ok ? "Ok" : "InternalError")
        << "\",\"checks\":[{\"name\":\"fill_reproducibility\",\"passed\":"
        << (fill_ok ? "true" : "false") << "},{\"name\":\"timer_resolution\",\"passed\":"
        << (timer_ok ? "true" : "false") << ",\"resolution_ns\":"
        << apex::gemm::Timer::resolution_nanoseconds() << "}]}";
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        std::cerr << "invalid command line; use --help for usage\n";
        return 2;
    }
    std::ofstream json_file;
    std::ostream* output = &std::cout;
    if (!options.json_out.empty()) {
        json_file.open(options.json_out, std::ios::out | std::ios::trunc);
        if (!json_file) {
            std::cerr << "could not open JSON output file: " << options.json_out << '\n';
            return 3;
        }
        output = &json_file;
    }

    if (options.mode == "info") {
        write_info(*output);
        return *output ? 0 : 4;
    }
    if (options.mode == "selftest") {
        write_selftest(*output);
        return *output ? 0 : 4;
    }

    apex::gemm::KernelDescriptor kernel{};
    Status status = select_kernel(options, &kernel);
    if (status != Status::Success) {
        *output << "{\"status\":\"" << apex::gemm::status_name(status) << "\"}\n";
        return 0;
    }
    apex::gemm::TilingPlan plan;
    status = make_plan(options, kernel, &plan);
    if (status != Status::Success) {
        *output << "{\"status\":\"" << apex::gemm::status_name(status) << "\"}\n";
        return 0;
    }
    if (options.mode == "plan") {
        write_plan(*output, options, kernel, plan);
        *output << '\n';
        return *output ? 0 : 4;
    }
    if (options.reps == 0 || (options.variant != "engine" && options.variant != "naive" &&
        options.variant != "reference")) {
        std::cerr << "run requires a supported --variant and --reps greater than zero\n";
        return 2;
    }
    status = write_run(*output, options, kernel, plan);
    if (status != Status::Success) {
        *output << "{\"status\":\"" << apex::gemm::status_name(status) << "\"}\n";
    }
    *output << '\n';
    return *output ? 0 : 4;
}
