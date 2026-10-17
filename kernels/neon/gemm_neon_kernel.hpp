#pragma once

#include "../../core/status.hpp"

#include <cstddef>

namespace apex::gemm {

using NeonMicroKernelFunction = Status (*)(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

struct NeonKernelDescriptor {
    const char* name;
    std::size_t mr;
    std::size_t nr;
    std::size_t kr;
    NeonMicroKernelFunction function;
};

[[nodiscard]] Status gemm_neon_4x4_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

[[nodiscard]] Status gemm_neon_8x8_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

// Selects the matching fixed tile or reports UnsupportedKernel.
[[nodiscard]] Status gemm_neon_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

[[nodiscard]] const NeonKernelDescriptor* neon_kernel_descriptors(
    std::size_t* count) noexcept;

}  // namespace apex::gemm
