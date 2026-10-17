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

struct ValidationSummary {
    std::size_t checks = 0;
    std::size_t failures = 0;
    double worst_error = 0.0;
    double worst_ratio = 0.0;
};

constexpr std::array<Shape, 8> kShapes{{
    {1, 1, 1},
    {2, 3, 4},
    {3, 5, 2},
    {5, 7, 11},
    {8, 3, 1},
    {1, 9, 6},
    {13, 5, 17},
    {17, 19, 13},
}};

constexpr std::array<float, 5> kScales{{0.0F, 1.0F, -1.0F, 0.5F, -0.75F}};

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
    const double operation_count = static_cast<double>(inner_dimension) + 3.0;
    const double scaled_roundoff = operation_count * unit_roundoff;
    if (scaled_roundoff >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double gamma = scaled_roundoff / (1.0 - scaled_roundoff);
    return gamma * scale + static_cast<double>(std::numeric_limits<float>::min());
}

void record_failure(ValidationSummary* summary) noexcept {
    ++summary->failures;
}

[[nodiscard]] bool run_case(
    Shape shape,
    float alpha,
    float beta,
    bool poison_ignored_inputs,
    bool poison_ignored_output,
    std::uint64_t seed,
    ValidationSummary* summary) noexcept {
    apex::gemm::Matrix a;
    apex::gemm::Matrix b;
    apex::gemm::Matrix initial_c;
    apex::gemm::Matrix candidate;
    apex::gemm::Matrix reference;

    using apex::gemm::Matrix;
    using apex::gemm::Status;
    if (Matrix::create(shape.rows, shape.inner, &a) != Status::Success ||
        Matrix::create(shape.inner, shape.columns, &b) != Status::Success ||
        Matrix::create(shape.rows, shape.columns, &initial_c) != Status::Success ||
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

    if (!copy_matrix(initial_c.view(), candidate.view()) ||
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
                candidate.view()(row, column) = nan;
                reference.view()(row, column) = nan;
            }
        }
    }

    const Status reference_status = apex::gemm::scalar_reference(
        a.view(), b.view(), reference.view(), alpha, beta);
    const Status candidate_status = apex::gemm::naive_baseline(
        a.view(), b.view(), candidate.view(), alpha, beta);
    ++summary->checks;

    if (reference_status != Status::Success || candidate_status != Status::Success) {
        record_failure(summary);
        std::fprintf(stderr,
            "FAIL status shape=%zux%zux%zu alpha=%g beta=%g reference=%s candidate=%s\n",
            shape.rows, shape.inner, shape.columns, static_cast<double>(alpha),
            static_cast<double>(beta), apex::gemm::status_name(reference_status),
            apex::gemm::status_name(candidate_status));
        return false;
    }

    bool passed = true;
    for (std::size_t row = 0; row < shape.rows; ++row) {
        for (std::size_t column = 0; column < shape.columns; ++column) {
            const double expected = static_cast<double>(reference.view()(row, column));
            const double actual = static_cast<double>(candidate.view()(row, column));
            const double error = std::abs(actual - expected);
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

            if (!std::isfinite(expected) || !std::isfinite(actual) || error > tolerance) {
                passed = false;
                std::fprintf(stderr,
                    "FAIL element shape=%zux%zux%zu row=%zu column=%zu "
                    "alpha=%g beta=%g expected=%.9g actual=%.9g error=%.3e tol=%.3e\n",
                    shape.rows, shape.inner, shape.columns, row, column,
                    static_cast<double>(alpha), static_cast<double>(beta), expected,
                    actual, error, tolerance);
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
    return passed;
}

}  // namespace

int main() {
    ValidationSummary summary;
    for (const Shape shape : kShapes) {
        for (const float alpha : kScales) {
            for (const float beta : kScales) {
                const bool poison_inputs = alpha == 0.0F;
                const bool poison_output = beta == 0.0F;
                (void)run_case(
                    shape, alpha, beta, poison_inputs, poison_output,
                    static_cast<std::uint64_t>(shape.rows * 131U +
                        shape.inner * 17U + shape.columns),
                    &summary);
            }
        }
    }

    std::printf(
        "Validation: %zu checks, %zu failures, worst error %.3e, worst ratio %.3e\n",
        summary.checks, summary.failures, summary.worst_error, summary.worst_ratio);
    return summary.failures == 0 ? 0 : 1;
}
