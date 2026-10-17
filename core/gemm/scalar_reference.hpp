#pragma once

#include "naive_baseline.hpp"

namespace apex::gemm {

[[nodiscard]] inline Status scalar_reference(
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
            double sum = 0.0;
            for (std::size_t inner = 0; inner < a.columns(); ++inner) {
                sum += static_cast<double>(a(row, inner)) *
                    static_cast<double>(b(inner, column));
            }

            double result = static_cast<double>(alpha) * sum;
            // Avoid reading C when beta is zero, including for uninitialized output.
            if (beta != 0.0F) {
                result += static_cast<double>(beta) *
                    static_cast<double>(c(row, column));
            }
            c(row, column) = static_cast<float>(result);
        }
    }

    return Status::Success;
}

}  // namespace apex::gemm
