#pragma once

#include "../memory/aligned_buffer.hpp"
#include "../status.hpp"

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace apex::gemm {

// Owns the two packing areas used by one execution thread.
class alignas(kCacheLineAlignment) ScratchBuffers {
public:
    ScratchBuffers() noexcept = default;
    ScratchBuffers(const ScratchBuffers&) = delete;
    ScratchBuffers& operator=(const ScratchBuffers&) = delete;
    ScratchBuffers(ScratchBuffers&&) noexcept = default;
    ScratchBuffers& operator=(ScratchBuffers&&) noexcept = default;

    [[nodiscard]] Status reserve(
        std::size_t packed_a_bytes,
        std::size_t packed_b_bytes) noexcept {
        Status status = reserve_one(packed_a_bytes, &packed_a_);
        if (status != Status::Success) {
            return status;
        }
        return reserve_one(packed_b_bytes, &packed_b_);
    }

    [[nodiscard]] void* packed_a_data() noexcept { return packed_a_.data(); }
    [[nodiscard]] const void* packed_a_data() const noexcept {
        return packed_a_.data();
    }
    [[nodiscard]] void* packed_b_data() noexcept { return packed_b_.data(); }
    [[nodiscard]] const void* packed_b_data() const noexcept {
        return packed_b_.data();
    }
    [[nodiscard]] std::size_t packed_a_capacity_bytes() const noexcept {
        return packed_a_.size_bytes();
    }
    [[nodiscard]] std::size_t packed_b_capacity_bytes() const noexcept {
        return packed_b_.size_bytes();
    }

private:
    [[nodiscard]] static Status reserve_one(
        std::size_t requested_bytes,
        AlignedBuffer* buffer) noexcept {
        if (buffer == nullptr) {
            return Status::InvalidArgument;
        }
        if (requested_bytes <= buffer->size_bytes()) {
            return Status::Success;
        }
        return AlignedAllocator::allocate(requested_bytes, buffer);
    }

    AlignedBuffer packed_a_;
    AlignedBuffer packed_b_;
};

// Buffer slots are indexed by stable worker id. Call prepare() before dispatch,
// then each worker accesses only its own slot without synchronization.
class ScratchBufferManager {
public:
    ScratchBufferManager() noexcept = default;
    ~ScratchBufferManager() = default;

    ScratchBufferManager(const ScratchBufferManager&) = delete;
    ScratchBufferManager& operator=(const ScratchBufferManager&) = delete;
    ScratchBufferManager(ScratchBufferManager&&) noexcept = default;
    ScratchBufferManager& operator=(ScratchBufferManager&&) noexcept = default;

    // Grows the pool outside the compute region. Existing buffers are retained
    // when worker count or packing requirements increase.
    [[nodiscard]] Status prepare(
        std::size_t worker_count,
        std::size_t packed_a_bytes,
        std::size_t packed_b_bytes) noexcept {
        if (worker_count == 0) {
            return Status::InvalidArgument;
        }

        if (worker_count > slot_capacity_) {
            std::unique_ptr<ScratchBuffers[]> replacement(
                new (std::nothrow) ScratchBuffers[worker_count]);
            if (replacement == nullptr) {
                return Status::AllocationFailed;
            }
            for (std::size_t index = 0; index < slot_capacity_; ++index) {
                replacement[index] = std::move(slots_[index]);
            }
            slots_ = std::move(replacement);
            slot_capacity_ = worker_count;
        }

        for (std::size_t index = 0; index < worker_count; ++index) {
            const Status status = slots_[index].reserve(
                packed_a_bytes, packed_b_bytes);
            if (status != Status::Success) {
                return status;
            }
        }
        worker_count_ = worker_count;
        return Status::Success;
    }

    [[nodiscard]] ScratchBuffers* for_worker(
        std::size_t worker_index) noexcept {
        return worker_index < worker_count_ ? &slots_[worker_index] : nullptr;
    }

    [[nodiscard]] const ScratchBuffers* for_worker(
        std::size_t worker_index) const noexcept {
        return worker_index < worker_count_ ? &slots_[worker_index] : nullptr;
    }

    [[nodiscard]] std::size_t worker_capacity() const noexcept {
        return slot_capacity_;
    }

    // Concurrent callers that cannot use a pool worker can reuse scratch owned
    // by their calling thread, avoiding contention on the shared manager.
    [[nodiscard]] static Status prepare_local(
        std::size_t packed_a_bytes,
        std::size_t packed_b_bytes,
        ScratchBuffers** output) noexcept {
        if (output == nullptr) {
            return Status::InvalidArgument;
        }
        thread_local ScratchBuffers local_buffers;
        const Status status = local_buffers.reserve(
            packed_a_bytes, packed_b_bytes);
        if (status == Status::Success) {
            *output = &local_buffers;
        }
        return status;
    }

private:
    std::unique_ptr<ScratchBuffers[]> slots_;
    std::size_t slot_capacity_ = 0;
    std::size_t worker_count_ = 0;
};

}  // namespace apex::gemm
