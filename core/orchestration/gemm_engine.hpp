#pragma once

#include "../memory/aligned_buffer.hpp"
#include "../memory/matrix.hpp"
#include "../../kernels/packing.hpp"
#include "tiling_planner.hpp"

#include <cstddef>

namespace apex::gemm {

using MicroKernelFunction = Status (*)(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

class GemmWorkspace {
public:
    GemmWorkspace() noexcept = default;
    GemmWorkspace(const GemmWorkspace&) = delete;
    GemmWorkspace& operator=(const GemmWorkspace&) = delete;
    GemmWorkspace(GemmWorkspace&&) noexcept = default;
    GemmWorkspace& operator=(GemmWorkspace&&) noexcept = default;

    [[nodiscard]] Status reserve(
        const TilingPlan& plan,
        PackingTileShape tile) noexcept;

private:
    friend class GemmEngine;

    AlignedBuffer packed_a_;
    AlignedBuffer packed_b_;
    AlignedBuffer edge_tile_;
};

class GemmEngine {
public:
    [[nodiscard]] static Status execute(
        MatrixView a,
        MatrixView b,
        MatrixView c,
        const TilingPlan& plan,
        PackingTileShape tile,
        GemmWorkspace* workspace,
        MicroKernelFunction micro_kernel = nullptr,
        float alpha = 1.0F,
        float beta = 0.0F) noexcept;
};

// Portable packed-panel kernel used when no specialized kernel is supplied.
[[nodiscard]] Status scalar_packed_microkernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

}  // namespace apex::gemm
