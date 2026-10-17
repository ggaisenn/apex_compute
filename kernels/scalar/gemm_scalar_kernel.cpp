#include "gemm_scalar_kernel.hpp"

namespace apex::gemm {

Status scalar_packed_microkernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept {
    if (mr == 0 || nr == 0 || ldc < nr) {
        return Status::InvalidArgument;
    }
    if (k != 0 && (packed_a == nullptr || packed_b == nullptr)) {
        return Status::NullData;
    }
    if (c == nullptr) {
        return Status::NullData;
    }

    for (std::size_t row = 0; row < mr; ++row) {
        for (std::size_t column = 0; column < nr; ++column) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < k; ++inner) {
                sum += packed_a[inner * mr + row] *
                    packed_b[inner * nr + column];
            }
            float& output = c[row * ldc + column];
            if (beta == 0.0F) {
                output = alpha * sum;
            } else {
                output = alpha * sum + beta * output;
            }
        }
    }
    return Status::Success;
}

}  // namespace apex::gemm
