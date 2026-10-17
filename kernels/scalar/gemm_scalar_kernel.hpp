#pragma once

#include "../kernel_contract.hpp"

namespace apex::gemm {

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
