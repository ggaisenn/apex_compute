#pragma once

#include "../memory/matrix.hpp"

#include <cstddef>

namespace apex::gemm {

namespace detail {

[[nodiscard]] inline Status validate_gemm_views(
    MatrixView a,
    MatrixView b,
    MatrixView c) noexcept {
    const Status a_status = a.validate();
    if (a_status != Status::Success) {
        return a_status;
    }

    const Status b_status = b.validate();
    if (b_status != Status::Success) {
        return b_status;
    }

    const Status c_status = c.validate();
    if (c_status != Status::Success) {
        return c_status;
    }

    if (a.columns() != b.rows() || c.rows() != a.rows() ||
        c.columns() != b.columns()) {
        return Status::InvalidDimension;
    }

    return Status::Success;
}

}  // namespace detail

[[nodiscard]] inline Status naive_baseline(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    float alpha = 1.0F,
    float beta = 0.0F) noexcept {
    const Status validation_status = detail::validate_gemm_views(a, b, c);
    if (validation_status != Status::Success) {
        return validation_status;
    }

    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t column = 0; column < c.columns(); ++column) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < a.columns(); ++inner) {
                sum += a(row, inner) * b(inner, column);
            }

            // Avoid reading C when beta is zero, including for uninitialized output.
            if (beta == 0.0F) {
                c(row, column) = alpha * sum;
            } else {
                c(row, column) = alpha * sum + beta * c(row, column);
            }
        }
    }

    return Status::Success;
}

}  // namespace apex::gemm
