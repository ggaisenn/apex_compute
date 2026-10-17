#include "../core/gemm/naive_baseline.hpp"
#include "../core/gemm/scalar_reference.hpp"
#include "../core/memory/matrix_fill.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <limits>

namespace {

struct Shape {
    std::size_t rows;
    std::size_t inner;
    std::size_t columns;
};

enum class InputPattern {
    random,
    zero_left,
    zero_right,
    identity_left,
    identity_right,
    mixed_magnitude,
};

using GemmKernel = apex::gemm::Status (*)(
    apex::gemm::MatrixView,
    apex::gemm::MatrixView,
    apex::gemm::MatrixView,
    float,
    float) noexcept;

struct KernelEntry {
    const char* name;
    GemmKernel execute;
};

struct ValidationSummary {
    std::size_t checks = 0;
    std::size_t failures = 0;
    double worst_error = 0.0;
    double worst_ratio = 0.0;
};

constexpr std::array<Shape, 11> kShapes{{
    {1, 1, 1},
    {2, 3, 4},
    {3, 5, 2},
    {5, 7, 11},
    {8, 3, 1},
    {1, 9, 6},
    {13, 5, 17},
    {17, 19, 13},
    {3, 3, 4},
    {4, 3, 3},
    {5, 5, 5},
}};

constexpr std::array<Shape, 3> kPropertyShapes{{
    {3, 3, 4},
    {4, 3, 3},
    {1, 1, 1},
}};

constexpr std::array<float, 5> kScales{{0.0F, 1.0F, -1.0F, 0.5F, -0.75F}};
constexpr std::array<int, 7> kMagnitudeExponents{{-64, -50, -30, 0, 30, 50, 60}};

constexpr KernelEntry kScalarKernel{"scalar-baseline", apex::gemm::naive_baseline};
constexpr std::array<KernelEntry, 0> kVectorKernels{};

// These fallback tile sizes are replaced by a runner plan query when available.
constexpr std::size_t kFallbackTileM = 32;
constexpr std::size_t kFallbackTileK = 32;
constexpr std::size_t kFallbackTileN = 32;

constexpr std::array<Shape, 6> kTileBoundaryShapes{{
    {kFallbackTileM - 1, 7, 7},
    {kFallbackTileM + 1, 7, 7},
    {7, kFallbackTileK - 1, 7},
    {7, kFallbackTileK + 1, 7},
    {7, 7, kFallbackTileN - 1},
    {7, 7, kFallbackTileN + 1},
}};

[[nodiscard]] bool copy_matrix(
    apex::gemm::MatrixView source,
    apex::gemm::MatrixView destination) noexcept {
    if (source.validate() != apex::gemm::Status::Success ||
        destination.validate() != apex::gemm::Status::Success ||
        source.rows() != destination.rows() ||
        source.columns() != destination.columns()) {
        return false;
    }

    for (std::size_t row = 0; row < source.rows(); ++row) {
        for (std::size_t column = 0; column < source.columns(); ++column) {
            destination(row, column) = source(row, column);
        }
    }
    return true;
}

[[nodiscard]] double element_scale(
    apex::gemm::MatrixView a,
    apex::gemm::MatrixView b,
    apex::gemm::MatrixView initial_c,
    std::size_t row,
    std::size_t column,
    float alpha,
    float beta) noexcept {
    double scale = 0.0;
    if (alpha != 0.0F) {
        double absolute_products = 0.0;
        for (std::size_t inner = 0; inner < a.columns(); ++inner) {
            absolute_products += std::abs(
                static_cast<double>(a(row, inner)) *
                static_cast<double>(b(inner, column)));
        }
        scale += std::abs(static_cast<double>(alpha)) * absolute_products;
    }
    if (beta != 0.0F) {
        scale += std::abs(static_cast<double>(beta)) *
            std::abs(static_cast<double>(initial_c(row, column)));
    }
    return scale;
}

[[nodiscard]] double element_tolerance(
    std::size_t inner_dimension,
    double scale) noexcept {
    constexpr double unit_roundoff = 0x1p-24;
    const double operation_count =
        2.0 * static_cast<double>(inner_dimension) + 3.0;
    const double scaled_roundoff = operation_count * unit_roundoff;
    if (scaled_roundoff >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double gamma = scaled_roundoff / (1.0 - scaled_roundoff);
    // Two bounds cover legal single-precision reductions with different orders;
    // fused multiply-add uses fewer roundings and remains within this limit.
    return 2.0 * gamma * scale +
        static_cast<double>(std::numeric_limits<float>::min());
}

void record_failure(ValidationSummary* summary) noexcept {
    ++summary->failures;
}

void expect_status(
    const char* check_name,
    apex::gemm::Status expected,
    apex::gemm::Status actual,
    ValidationSummary* summary) noexcept {
    ++summary->checks;
    if (actual != expected) {
        record_failure(summary);
        std::fprintf(stderr, "FAIL input-check=%s expected=%s actual=%s\n",
            check_name, apex::gemm::status_name(expected),
            apex::gemm::status_name(actual));
    }
}

[[nodiscard]] bool run_case(
    Shape shape,
    float alpha,
    float beta,
    bool poison_ignored_inputs,
    bool poison_ignored_output,
    int magnitude_exponent,
    InputPattern pattern,
    std::uint64_t seed,
    ValidationSummary* summary,
    const KernelEntry& kernel = kScalarKernel) noexcept {
    apex::gemm::Matrix a;
    apex::gemm::Matrix b;
    apex::gemm::Matrix initial_c;
    apex::gemm::Matrix scalar_result;
    apex::gemm::Matrix candidate;
    apex::gemm::Matrix reference;

    using apex::gemm::Matrix;
    using apex::gemm::Status;
    if (Matrix::create(shape.rows, shape.inner, &a) != Status::Success ||
        Matrix::create(shape.inner, shape.columns, &b) != Status::Success ||
        Matrix::create(shape.rows, shape.columns, &initial_c) != Status::Success ||
        Matrix::create(shape.rows, shape.columns, &scalar_result) != Status::Success ||
        Matrix::create(shape.rows, shape.columns, &candidate) != Status::Success ||
        Matrix::create(shape.rows, shape.columns, &reference) != Status::Success) {
        record_failure(summary);
        std::fprintf(stderr, "FAIL allocation for shape %zux%zux%zu\n",
            shape.rows, shape.inner, shape.columns);
        return false;
    }

    if (apex::gemm::fill_random(a, seed) != Status::Success ||
        apex::gemm::fill_random(b, seed + 1U) != Status::Success ||
        apex::gemm::fill_random(initial_c, seed + 2U) != Status::Success) {
        record_failure(summary);
        std::fprintf(stderr, "FAIL input generation for shape %zux%zux%zu\n",
            shape.rows, shape.inner, shape.columns);
        return false;
    }

    switch (pattern) {
        case InputPattern::random:
            break;
        case InputPattern::zero_left:
            if (apex::gemm::fill_zeros(a) != Status::Success) {
                record_failure(summary);
                return false;
            }
            break;
        case InputPattern::zero_right:
            if (apex::gemm::fill_zeros(b) != Status::Success) {
                record_failure(summary);
                return false;
            }
            break;
        case InputPattern::identity_left:
            if (shape.rows != shape.inner ||
                apex::gemm::fill_identity(a) != Status::Success) {
                record_failure(summary);
                return false;
            }
            break;
        case InputPattern::identity_right:
            if (shape.inner != shape.columns ||
                apex::gemm::fill_identity(b) != Status::Success) {
                record_failure(summary);
                return false;
            }
            break;
        case InputPattern::mixed_magnitude:
            for (std::size_t row = 0; row < a.rows(); ++row) {
                for (std::size_t column = 0; column < a.columns(); ++column) {
                    const int exponent = static_cast<int>((row * 3U + column * 5U) % 13U) - 6;
                    a.view()(row, column) =
                        std::scalbn(a.view()(row, column), exponent);
                }
            }
            for (std::size_t row = 0; row < b.rows(); ++row) {
                for (std::size_t column = 0; column < b.columns(); ++column) {
                    const int exponent = static_cast<int>((row * 7U + column * 2U) % 13U) - 6;
                    b.view()(row, column) =
                        std::scalbn(b.view()(row, column), exponent);
                }
            }
            for (std::size_t row = 0; row < initial_c.rows(); ++row) {
                for (std::size_t column = 0; column < initial_c.columns(); ++column) {
                    const int exponent = static_cast<int>((row * 5U + column * 3U) % 25U) - 12;
                    initial_c.view()(row, column) =
                        std::scalbn(initial_c.view()(row, column), exponent);
                }
            }
            break;
    }

    const int output_exponent = 2 * magnitude_exponent;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t column = 0; column < a.columns(); ++column) {
            a.view()(row, column) =
                std::scalbn(a.view()(row, column), magnitude_exponent);
        }
    }
    for (std::size_t row = 0; row < b.rows(); ++row) {
        for (std::size_t column = 0; column < b.columns(); ++column) {
            b.view()(row, column) =
                std::scalbn(b.view()(row, column), magnitude_exponent);
        }
    }
    for (std::size_t row = 0; row < initial_c.rows(); ++row) {
        for (std::size_t column = 0; column < initial_c.columns(); ++column) {
            initial_c.view()(row, column) =
                std::scalbn(initial_c.view()(row, column), output_exponent);
        }
    }

    if (poison_ignored_inputs) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t row = 0; row < a.rows(); ++row) {
            for (std::size_t column = 0; column < a.columns(); ++column) {
                a.view()(row, column) = nan;
            }
        }
        for (std::size_t row = 0; row < b.rows(); ++row) {
            for (std::size_t column = 0; column < b.columns(); ++column) {
                b.view()(row, column) = nan;
            }
        }
    }

    if (!copy_matrix(initial_c.view(), scalar_result.view()) ||
        !copy_matrix(initial_c.view(), candidate.view()) ||
        !copy_matrix(initial_c.view(), reference.view())) {
        record_failure(summary);
        std::fprintf(stderr, "FAIL output initialization for shape %zux%zux%zu\n",
            shape.rows, shape.inner, shape.columns);
        return false;
    }

    if (poison_ignored_output) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t row = 0; row < candidate.rows(); ++row) {
            for (std::size_t column = 0; column < candidate.columns(); ++column) {
                scalar_result.view()(row, column) = nan;
                candidate.view()(row, column) = nan;
                reference.view()(row, column) = nan;
            }
        }
    }

    const Status reference_status = apex::gemm::scalar_reference(
        a.view(), b.view(), reference.view(), alpha, beta);
    const Status scalar_status = apex::gemm::naive_baseline(
        a.view(), b.view(), scalar_result.view(), alpha, beta);
    const Status candidate_status = kernel.execute(
        a.view(), b.view(), candidate.view(), alpha, beta);
    ++summary->checks;

    if (reference_status != Status::Success || scalar_status != Status::Success ||
        candidate_status != Status::Success) {
        record_failure(summary);
        std::fprintf(stderr,
            "FAIL kernel=%s shape=%zux%zux%zu alpha=%g beta=%g "
            "reference=%s scalar=%s candidate=%s\n",
            kernel.name,
            shape.rows, shape.inner, shape.columns, static_cast<double>(alpha),
            static_cast<double>(beta), apex::gemm::status_name(reference_status),
            apex::gemm::status_name(scalar_status),
            apex::gemm::status_name(candidate_status));
        return false;
    }

    bool passed = true;
    for (std::size_t row = 0; row < shape.rows; ++row) {
        for (std::size_t column = 0; column < shape.columns; ++column) {
            const double expected = static_cast<double>(reference.view()(row, column));
            const double actual = static_cast<double>(candidate.view()(row, column));
            const double error = std::abs(actual - expected);
            const double scalar_error = std::abs(
                actual - static_cast<double>(scalar_result.view()(row, column)));
            const double scale = element_scale(
                a.view(), b.view(), initial_c.view(), row, column, alpha,
                poison_ignored_output ? 0.0F : beta);
            const double tolerance = element_tolerance(shape.inner, scale);

            if (std::isfinite(error) && error > summary->worst_error) {
                summary->worst_error = error;
            }
            const double ratio = tolerance > 0.0 ? error / tolerance : error;
            if (std::isfinite(ratio) && ratio > summary->worst_ratio) {
                summary->worst_ratio = ratio;
            }

            if (!std::isfinite(expected) || !std::isfinite(actual) ||
                !std::isfinite(tolerance) || error > tolerance ||
                scalar_error > tolerance) {
                passed = false;
                std::fprintf(stderr,
                    "FAIL kernel=%s shape=%zux%zux%zu row=%zu column=%zu "
                    "exp2=%d alpha=%g beta=%g expected=%.9g actual=%.9g "
                    "error=%.3e scalar-error=%.3e tol=%.3e\n",
                    kernel.name, shape.rows, shape.inner, shape.columns, row, column,
                    magnitude_exponent, static_cast<double>(alpha),
                    static_cast<double>(beta), expected, actual, error, scalar_error,
                    tolerance);
                break;
            }
        }
        if (!passed) {
            break;
        }
    }

    if (!passed) {
        record_failure(summary);
    }

    if (passed && alpha == 1.0F && beta == 0.0F) {
        for (std::size_t row = 0; row < shape.rows && passed; ++row) {
            for (std::size_t column = 0; column < shape.columns; ++column) {
                float property_value = 0.0F;
                bool has_property = true;
                if (pattern == InputPattern::zero_left ||
                    pattern == InputPattern::zero_right) {
                    property_value = 0.0F;
                } else if (pattern == InputPattern::identity_left) {
                    property_value = b.view()(row, column);
                } else if (pattern == InputPattern::identity_right) {
                    property_value = a.view()(row, column);
                } else {
                    has_property = false;
                }
                if (has_property && candidate.view()(row, column) != property_value) {
                    passed = false;
                    ++summary->failures;
                    std::fprintf(stderr,
                        "FAIL property pattern=%d shape=%zux%zux%zu row=%zu "
                        "column=%zu expected=%.9g actual=%.9g\n",
                        static_cast<int>(pattern), shape.rows, shape.inner,
                        shape.columns, row, column,
                        static_cast<double>(property_value),
                        static_cast<double>(candidate.view()(row, column)));
                    break;
                }
            }
        }
    }
    return passed;
}

void run_malformed_input_cases(ValidationSummary* summary) noexcept {
    using apex::gemm::MatrixView;
    using apex::gemm::Status;

    float storage_a[8]{};
    float storage_b[8]{};
    float storage_c[8]{};
    MatrixView a;
    MatrixView b;
    MatrixView c;
    MatrixView invalid;

    expect_status("null matrix data", Status::NullData,
        MatrixView::create(nullptr, 1, 1, 1, 1, &invalid), summary);
    expect_status("null view output", Status::InvalidArgument,
        MatrixView::create(storage_a, 1, 1, 1, 1, nullptr), summary);
    expect_status("short leading dimension", Status::InvalidLeadingDimension,
        MatrixView::create(storage_a, 2, 2, 1, 8, &invalid), summary);
    expect_status("short backing buffer", Status::BufferTooSmall,
        MatrixView::create(storage_a, 2, 2, 2, 3, &invalid), summary);
    expect_status("size arithmetic overflow", Status::SizeOverflow,
        MatrixView::create(storage_a, std::numeric_limits<std::size_t>::max(),
            2, 2, std::numeric_limits<std::size_t>::max(), &invalid), summary);
    expect_status("empty null-data view", Status::Success,
        MatrixView::create(nullptr, 0, 4, 4, 0, &invalid), summary);

    expect_status("valid A descriptor", Status::Success,
        MatrixView::create(storage_a, 2, 2, 2, 8, &a), summary);
    expect_status("valid mismatched B descriptor", Status::Success,
        MatrixView::create(storage_b, 3, 1, 1, 8, &b), summary);
    expect_status("valid mismatched C descriptor", Status::Success,
        MatrixView::create(storage_c, 2, 1, 1, 8, &c), summary);
    expect_status("baseline dimension status", Status::InvalidDimension,
        apex::gemm::naive_baseline(a, b, c), summary);
    expect_status("reference dimension status", Status::InvalidDimension,
        apex::gemm::scalar_reference(a, b, c), summary);
}

[[nodiscard]] apex::gemm::Status injected_corruption_kernel(
    apex::gemm::MatrixView a,
    apex::gemm::MatrixView b,
    apex::gemm::MatrixView c,
    float alpha,
    float beta) noexcept {
    const apex::gemm::Status status =
        apex::gemm::naive_baseline(a, b, c, alpha, beta);
    if (status == apex::gemm::Status::Success && c.rows() != 0 &&
        c.columns() != 0) {
        c(0, 0) += 1.0F;
    }
    return status;
}

void run_fault_sensitivity_case(ValidationSummary* summary) noexcept {
    apex::gemm::Matrix a;
    apex::gemm::Matrix b;
    apex::gemm::Matrix candidate;
    apex::gemm::Matrix reference;
    using apex::gemm::Matrix;
    using apex::gemm::Status;
    if (Matrix::create(1, 1, &a) != Status::Success ||
        Matrix::create(1, 1, &b) != Status::Success ||
        Matrix::create(1, 1, &candidate) != Status::Success ||
        Matrix::create(1, 1, &reference) != Status::Success) {
        record_failure(summary);
        std::fprintf(stderr, "FAIL fault-sensitivity allocation\n");
        return;
    }

    a.view()(0, 0) = 2.0F;
    b.view()(0, 0) = 3.0F;
    candidate.view()(0, 0) = 0.0F;
    reference.view()(0, 0) = 0.0F;

    const Status expected_status = apex::gemm::scalar_reference(
        a.view(), b.view(), reference.view(), 1.0F, 0.0F);
    const Status injected_status = injected_corruption_kernel(
        a.view(), b.view(), candidate.view(), 1.0F, 0.0F);
    const double error = std::abs(
        static_cast<double>(candidate.view()(0, 0)) -
        static_cast<double>(reference.view()(0, 0)));
    const double tolerance = element_tolerance(
        1, std::abs(static_cast<double>(a.view()(0, 0)) *
            static_cast<double>(b.view()(0, 0))));
    ++summary->checks;

    if (expected_status != Status::Success || injected_status != Status::Success ||
        error <= tolerance) {
        record_failure(summary);
        std::fprintf(stderr,
            "FAIL fault-sensitivity expected-status=%s injected-status=%s "
            "error=%.3e tol=%.3e\n",
            apex::gemm::status_name(expected_status),
            apex::gemm::status_name(injected_status), error, tolerance);
    } else {
        std::printf("PASS fault-sensitivity detected injected output corruption\n");
    }
}

}  // namespace

int main() {
    ValidationSummary summary;
    for (const Shape shape : kShapes) {
        for (const float alpha : kScales) {
            for (const float beta : kScales) {
                const bool poison_inputs = alpha == 0.0F;
                const bool poison_output = beta == 0.0F;
                for (std::size_t regime = 0; regime < kMagnitudeExponents.size(); ++regime) {
                    (void)run_case(
                        shape, alpha, beta, poison_inputs, poison_output,
                        kMagnitudeExponents[regime], InputPattern::random,
                        static_cast<std::uint64_t>(shape.rows * 131U +
                            shape.inner * 17U + shape.columns + regime),
                        &summary);
                }
            }
        }
    }

    for (const Shape shape : kTileBoundaryShapes) {
        (void)run_case(shape, 1.0F, 0.0F, false, false, 0,
            InputPattern::random,
            static_cast<std::uint64_t>(shape.rows * 131U +
                shape.inner * 17U + shape.columns),
            &summary);
    }

    if (kVectorKernels.empty()) {
        std::printf("SKIP vector-kernel cross-check: no vectorized kernel is registered\n");
    }
    for (const KernelEntry& kernel : kVectorKernels) {
        for (const Shape shape : kShapes) {
            for (const float alpha : kScales) {
                for (const float beta : kScales) {
                    for (const int exponent : kMagnitudeExponents) {
                        (void)run_case(shape, alpha, beta, alpha == 0.0F,
                            beta == 0.0F, exponent, InputPattern::random,
                            static_cast<std::uint64_t>(shape.rows * 131U +
                                shape.inner * 17U + shape.columns),
                            &summary, kernel);
                    }
                }
            }
        }
    }

    (void)run_case(kPropertyShapes[0], 1.0F, 0.0F, false, false, 0,
        InputPattern::zero_left, static_cast<std::uint64_t>(101), &summary);
    (void)run_case(kPropertyShapes[1], 1.0F, 0.0F, false, false, 0,
        InputPattern::zero_right, static_cast<std::uint64_t>(102), &summary);
    (void)run_case(kPropertyShapes[0], 1.0F, 0.0F, false, false, 0,
        InputPattern::identity_left, static_cast<std::uint64_t>(103), &summary);
    (void)run_case(kPropertyShapes[1], 1.0F, 0.0F, false, false, 0,
        InputPattern::identity_right, static_cast<std::uint64_t>(104), &summary);
    (void)run_case(kPropertyShapes[2], 1.0F, 0.0F, false, false, 0,
        InputPattern::zero_left, static_cast<std::uint64_t>(105), &summary);
    (void)run_case({5, 7, 11}, 0.5F, -1.0F, false, false, 0,
        InputPattern::mixed_magnitude, static_cast<std::uint64_t>(106), &summary);

    run_malformed_input_cases(&summary);
    run_fault_sensitivity_case(&summary);

    std::printf(
        "Validation: %zu checks, %zu failures, worst error %.3e, worst ratio %.3e\n",
        summary.checks, summary.failures, summary.worst_error, summary.worst_ratio);
    return summary.failures == 0 ? 0 : 1;
}
