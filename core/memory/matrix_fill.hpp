#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <cstdint>

namespace apex::gemm {

[[nodiscard]] inline MatrixStatus fill_zeros(MatrixView matrix) noexcept {
    const MatrixStatus status = matrix.validate();
    if (status != MatrixStatus::success) {
        return status;
    }
    if (matrix.rows() == 0 || matrix.columns() == 0) {
        return MatrixStatus::empty_matrix;
    }

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            matrix(row, column) = 0.0F;
        }
    }
    return MatrixStatus::success;
}

[[nodiscard]] inline MatrixStatus fill_constant(
    MatrixView matrix,
    float value) noexcept {
    const MatrixStatus status = matrix.validate();
    if (status != MatrixStatus::success) {
        return status;
    }
    if (matrix.rows() == 0 || matrix.columns() == 0) {
        return MatrixStatus::empty_matrix;
    }

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            matrix(row, column) = value;
        }
    }
    return MatrixStatus::success;
}

[[nodiscard]] inline MatrixStatus fill_identity(MatrixView matrix) noexcept {
    const MatrixStatus status = fill_zeros(matrix);
    if (status != MatrixStatus::success) {
        return status;
    }

    const std::size_t diagonal_size =
        matrix.rows() < matrix.columns() ? matrix.rows() : matrix.columns();
    for (std::size_t index = 0; index < diagonal_size; ++index) {
        matrix(index, index) = 1.0F;
    }
    return MatrixStatus::success;
}

[[nodiscard]] inline MatrixStatus fill_random(
    MatrixView matrix,
    std::uint64_t seed) noexcept {
    const MatrixStatus status = matrix.validate();
    if (status != MatrixStatus::success) {
        return status;
    }
    if (matrix.rows() == 0 || matrix.columns() == 0) {
        return MatrixStatus::empty_matrix;
    }

    std::uint64_t state = seed;
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            state += UINT64_C(0x9E3779B97F4A7C15);
            std::uint64_t value = state;
            value = (value ^ (value >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
            value = (value ^ (value >> 27U)) * UINT64_C(0x94D049BB133111EB);
            value ^= value >> 31U;

            const std::uint32_t sample = static_cast<std::uint32_t>(value >> 40U);
            matrix(row, column) = static_cast<float>(sample) * 0x1p-23F - 1.0F;
        }
    }
    return MatrixStatus::success;
}

[[nodiscard]] inline MatrixStatus fill_zeros(Matrix& matrix) noexcept {
    return fill_zeros(matrix.view());
}

[[nodiscard]] inline MatrixStatus fill_constant(Matrix& matrix, float value) noexcept {
    return fill_constant(matrix.view(), value);
}

[[nodiscard]] inline MatrixStatus fill_identity(Matrix& matrix) noexcept {
    return fill_identity(matrix.view());
}

[[nodiscard]] inline MatrixStatus fill_random(
    Matrix& matrix,
    std::uint64_t seed) noexcept {
    return fill_random(matrix.view(), seed);
}

}  // namespace apex::gemm
