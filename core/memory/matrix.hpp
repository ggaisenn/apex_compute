#pragma once

#include "aligned_buffer.hpp"
#include "../status.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>

namespace apex::gemm {

using MatrixStatus = Status;

enum class MatrixStorageOrder {
    row_major,
};

namespace detail {

[[nodiscard]] inline MatrixStatus validate_matrix_descriptor(
    const float* data,
    std::size_t rows,
    std::size_t columns,
    std::size_t leading_dimension,
    std::size_t buffer_elements) noexcept {
    if (leading_dimension < columns) {
        return MatrixStatus::invalid_leading_dimension;
    }
    if (rows != 0 && columns != 0 && data == nullptr) {
        return MatrixStatus::invalid_argument;
    }

    std::size_t required_elements = 0;
    if (rows != 0 && columns != 0) {
        const std::size_t preceding_rows = rows - 1;
        if (preceding_rows >
            (std::numeric_limits<std::size_t>::max() - columns) / leading_dimension) {
            return MatrixStatus::size_overflow;
        }
        required_elements = preceding_rows * leading_dimension + columns;
    }
    if (buffer_elements < required_elements) {
        return MatrixStatus::buffer_too_small;
    }
    return MatrixStatus::success;
}

}  // namespace detail

class Matrix;

class MatrixView {
public:
    MatrixView() noexcept = default;

    [[nodiscard]] static MatrixStatus create(
        float* data,
        std::size_t rows,
        std::size_t columns,
        std::size_t leading_dimension,
        std::size_t buffer_elements,
        MatrixView* output) noexcept {
        if (output == nullptr) {
            return MatrixStatus::invalid_argument;
        }

        const MatrixStatus validation_status = detail::validate_matrix_descriptor(
            data, rows, columns, leading_dimension, buffer_elements);
        if (validation_status != MatrixStatus::success) {
            return validation_status;
        }

        *output = MatrixView(
            data, rows, columns, leading_dimension, buffer_elements);
        return MatrixStatus::success;
    }

    [[nodiscard]] MatrixStatus validate() const noexcept {
        MatrixView validated;
        return create(
            data_, rows_, columns_, leading_dimension_, buffer_elements_, &validated);
    }

    [[nodiscard]] float* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t leading_dimension() const noexcept {
        return leading_dimension_;
    }
    [[nodiscard]] std::size_t buffer_elements() const noexcept {
        return buffer_elements_;
    }
    [[nodiscard]] MatrixStorageOrder storage_order() const noexcept {
        return MatrixStorageOrder::row_major;
    }

    [[nodiscard]] float& operator()(std::size_t row, std::size_t column) const noexcept {
        return data_[row * leading_dimension_ + column];
    }

private:
    friend class Matrix;

    MatrixView(
        float* data,
        std::size_t rows,
        std::size_t columns,
        std::size_t leading_dimension,
        std::size_t buffer_elements) noexcept
        : data_(data),
          rows_(rows),
          columns_(columns),
          leading_dimension_(leading_dimension),
          buffer_elements_(buffer_elements) {}

    float* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t columns_ = 0;
    std::size_t leading_dimension_ = 0;
    std::size_t buffer_elements_ = 0;
};

class Matrix {
public:
    Matrix() noexcept = default;

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    [[nodiscard]] static MatrixStatus create(
        std::size_t rows,
        std::size_t columns,
        Matrix* output) noexcept {
        if (output == nullptr) {
            return MatrixStatus::invalid_argument;
        }

        AlignedBuffer buffer;
        std::size_t leading_dimension = 0;
        const AllocationStatus allocation_status =
            AlignedAllocator::allocate_matrix<float>(
                rows, columns, &buffer, &leading_dimension);
        if (allocation_status != AllocationStatus::success) {
            return allocation_status;
        }

        if (buffer.size_bytes() != 0) {
            std::memset(buffer.data(), 0, buffer.size_bytes());
        }

        Matrix replacement;
        replacement.buffer_ = std::move(buffer);
        replacement.rows_ = rows;
        replacement.columns_ = columns;
        replacement.leading_dimension_ = leading_dimension;
        *output = std::move(replacement);
        return MatrixStatus::success;
    }

    [[nodiscard]] MatrixStatus validate() const noexcept {
        return detail::validate_matrix_descriptor(
            data(), rows_, columns_, leading_dimension_,
            buffer_.size_bytes() / sizeof(float));
    }

    [[nodiscard]] MatrixView view() noexcept {
        return MatrixView(
            data(), rows_, columns_, leading_dimension_,
            buffer_.size_bytes() / sizeof(float));
    }

    [[nodiscard]] float* data() noexcept {
        return static_cast<float*>(buffer_.data());
    }
    [[nodiscard]] const float* data() const noexcept {
        return static_cast<const float*>(buffer_.data());
    }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t leading_dimension() const noexcept {
        return leading_dimension_;
    }
    [[nodiscard]] MatrixStorageOrder storage_order() const noexcept {
        return MatrixStorageOrder::row_major;
    }
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return buffer_.size_bytes();
    }

private:
    AlignedBuffer buffer_;
    std::size_t rows_ = 0;
    std::size_t columns_ = 0;
    std::size_t leading_dimension_ = 0;
};

}  // namespace apex::gemm
