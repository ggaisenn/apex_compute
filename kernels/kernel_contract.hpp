#pragma once

#include "../core/status.hpp"

#include <cstddef>

namespace apex::gemm {

using KernelFunction = Status (*)(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept;

enum class KernelKind {
    Scalar,
    Neon,
    Avx512,
};

struct KernelDescriptor {
    const char* name;
    KernelKind kind;
    std::size_t mr;
    std::size_t nr;
    std::size_t kr;
    KernelFunction function;
};

}  // namespace apex::gemm
