#pragma once

#include "../../core/status.hpp"

#include <cstddef>

namespace apex::gemm {

using Avx512MicroKernelFunction = Status (*)(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

struct Avx512KernelDescriptor {
    const char* name;
    std::size_t mr;
    std::size_t nr;
    std::size_t kr;
    Avx512MicroKernelFunction function;
};

[[nodiscard]] Status gemm_avx512_16x16_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

[[nodiscard]] const Avx512KernelDescriptor* avx512_kernel_descriptors(
    std::size_t* count) noexcept;

}  // namespace apex::gemm
