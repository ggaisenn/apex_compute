#pragma once

#include "../memory/aligned_buffer.hpp"
#include "../memory/matrix.hpp"
#include "../../kernels/packing.hpp"
#include "../../kernels/kernel_contract.hpp"
#include "../../kernels/scalar/gemm_scalar_kernel.hpp"
#include "tiling_planner.hpp"

#include <cstddef>

namespace apex::gemm {

using MicroKernelFunction = KernelFunction;

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

}  // namespace apex::gemm
