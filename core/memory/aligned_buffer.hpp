#pragma once

#include "../status.hpp"

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <stdlib.h>
#endif

namespace apex::gemm {

inline constexpr std::size_t kCacheLineAlignment = 64;

using AllocationStatus = Status;

class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;

    ~AlignedBuffer() noexcept { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept { swap(other); }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }
    [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

    void reset() noexcept {
        if (data_ != nullptr) {
            free_aligned(data_);
            data_ = nullptr;
            size_bytes_ = 0;
        }
    }

private:
    friend class AlignedAllocator;

    explicit AlignedBuffer(void* data, std::size_t size_bytes) noexcept
        : data_(data), size_bytes_(size_bytes) {}

    static void free_aligned(void* pointer) noexcept {
#if defined(_WIN32)
        _aligned_free(pointer);
#else
        std::free(pointer);
#endif
    }

    void swap(AlignedBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_bytes_, other.size_bytes_);
    }

    void* data_ = nullptr;
    std::size_t size_bytes_ = 0;
};

class AlignedAllocator {
public:
    static constexpr std::size_t alignment = kCacheLineAlignment;

    // A zero-byte request succeeds with an empty owner and performs no allocation.
    [[nodiscard]] static AllocationStatus allocate(
        std::size_t size_bytes,
        AlignedBuffer* output) noexcept {
        if (output == nullptr) {
            return AllocationStatus::invalid_argument;
        }

        if (size_bytes == 0) {
            output->reset();
            return AllocationStatus::success;
        }

        std::size_t allocation_size = size_bytes;
#if !defined(_WIN32)
        const std::size_t remainder = allocation_size % alignment;
        if (remainder != 0) {
            const std::size_t padding = alignment - remainder;
            if (allocation_size > std::numeric_limits<std::size_t>::max() - padding) {
                return AllocationStatus::size_overflow;
            }
            allocation_size += padding;
        }
#endif

        void* memory = nullptr;
#if defined(_WIN32)
        memory = _aligned_malloc(allocation_size, alignment);
        if (memory == nullptr) {
            return AllocationStatus::allocation_failure;
        }
#elif defined(__APPLE__)
        memory = std::aligned_alloc(alignment, allocation_size);
        if (memory == nullptr) {
            return AllocationStatus::allocation_failure;
        }
#else
        const int result = ::posix_memalign(&memory, alignment, allocation_size);
        if (result != 0) {
            return AllocationStatus::allocation_failure;
        }
#endif

        AlignedBuffer replacement(memory, size_bytes);
        *output = std::move(replacement);
        return AllocationStatus::success;
    }

    // Computes a row stride whose byte size is a multiple of the cache-line alignment.
    template <typename Element>
    [[nodiscard]] static AllocationStatus padded_row_stride(
        std::size_t columns,
        std::size_t* output_stride) noexcept {
        static_assert(
            alignment % sizeof(Element) == 0,
            "Element size must divide the cache-line alignment.");
        if (output_stride == nullptr) {
            return AllocationStatus::invalid_argument;
        }
        if (columns == 0) {
            *output_stride = 0;
            return AllocationStatus::success;
        }

        constexpr std::size_t elements_per_line = alignment / sizeof(Element);
        const std::size_t remainder = columns % elements_per_line;
        if (remainder == 0) {
            *output_stride = columns;
            return AllocationStatus::success;
        }

        const std::size_t padding = elements_per_line - remainder;
        if (columns > std::numeric_limits<std::size_t>::max() - padding) {
            return AllocationStatus::size_overflow;
        }
        *output_stride = columns + padding;
        return AllocationStatus::success;
    }

    template <typename Element>
    [[nodiscard]] static AllocationStatus allocate_matrix(
        std::size_t rows,
        std::size_t columns,
        AlignedBuffer* output,
        std::size_t* output_stride) noexcept {
        if (output == nullptr || output_stride == nullptr) {
            return AllocationStatus::invalid_argument;
        }

        if (rows == 0 || columns == 0) {
            output->reset();
            *output_stride = columns;
            return AllocationStatus::success;
        }

        std::size_t stride = 0;
        const AllocationStatus stride_status =
            padded_row_stride<Element>(columns, &stride);
        if (stride_status != AllocationStatus::success) {
            return stride_status;
        }

        if (rows > std::numeric_limits<std::size_t>::max() / stride) {
            return AllocationStatus::size_overflow;
        }
        const std::size_t element_count = rows * stride;
        if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
            return AllocationStatus::size_overflow;
        }

        const AllocationStatus allocation_status =
            allocate(element_count * sizeof(Element), output);
        if (allocation_status == AllocationStatus::success) {
            *output_stride = stride;
        }
        return allocation_status;
    }
};

}  // namespace apex::gemm
