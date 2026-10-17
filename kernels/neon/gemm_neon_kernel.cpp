#include "gemm_neon_kernel.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace apex::gemm {
namespace {

[[nodiscard]] Status validate_call(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    std::size_t expected_mr,
    std::size_t expected_nr) noexcept {
    if (mr != expected_mr || nr != expected_nr || ldc < nr) {
        return Status::InvalidArgument;
    }
    if (c == nullptr || (k != 0 && (packed_a == nullptr || packed_b == nullptr))) {
        return Status::NullData;
    }
    return Status::Success;
}

#if defined(__aarch64__) || defined(_M_ARM64)
[[nodiscard]] float32x4_t apply_scale(
    float32x4_t accumulator,
    float* output,
    float alpha,
    float beta) noexcept {
    if (beta == 0.0F) {
        return vmulq_n_f32(accumulator, alpha);
    }
    const float32x4_t current = vld1q_f32(output);
    if (beta == 1.0F) {
        return vfmaq_n_f32(current, accumulator, alpha);
    }
    return vfmaq_n_f32(
        vmulq_n_f32(current, beta), accumulator, alpha);
}

[[nodiscard]] Status run_4x4(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    float alpha,
    float beta) noexcept {
    float32x4_t row0 = vdupq_n_f32(0.0F);
    float32x4_t row1 = vdupq_n_f32(0.0F);
    float32x4_t row2 = vdupq_n_f32(0.0F);
    float32x4_t row3 = vdupq_n_f32(0.0F);

    for (std::size_t inner = 0; inner < k; ++inner) {
        const float* a_step = packed_a + inner * 4;
        const float32x4_t b_values = vld1q_f32(packed_b + inner * 4);
        row0 = vfmaq_f32(row0, b_values, vdupq_n_f32(a_step[0]));
        row1 = vfmaq_f32(row1, b_values, vdupq_n_f32(a_step[1]));
        row2 = vfmaq_f32(row2, b_values, vdupq_n_f32(a_step[2]));
        row3 = vfmaq_f32(row3, b_values, vdupq_n_f32(a_step[3]));
    }

    vst1q_f32(c, apply_scale(row0, c, alpha, beta));
    vst1q_f32(c + ldc, apply_scale(row1, c + ldc, alpha, beta));
    vst1q_f32(c + 2 * ldc, apply_scale(row2, c + 2 * ldc, alpha, beta));
    vst1q_f32(c + 3 * ldc, apply_scale(row3, c + 3 * ldc, alpha, beta));
    return Status::Success;
}

[[nodiscard]] Status run_8x8(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    float alpha,
    float beta) noexcept {
    float32x4_t r00 = vdupq_n_f32(0.0F);
    float32x4_t r01 = vdupq_n_f32(0.0F);
    float32x4_t r10 = vdupq_n_f32(0.0F);
    float32x4_t r11 = vdupq_n_f32(0.0F);
    float32x4_t r20 = vdupq_n_f32(0.0F);
    float32x4_t r21 = vdupq_n_f32(0.0F);
    float32x4_t r30 = vdupq_n_f32(0.0F);
    float32x4_t r31 = vdupq_n_f32(0.0F);
    float32x4_t r40 = vdupq_n_f32(0.0F);
    float32x4_t r41 = vdupq_n_f32(0.0F);
    float32x4_t r50 = vdupq_n_f32(0.0F);
    float32x4_t r51 = vdupq_n_f32(0.0F);
    float32x4_t r60 = vdupq_n_f32(0.0F);
    float32x4_t r61 = vdupq_n_f32(0.0F);
    float32x4_t r70 = vdupq_n_f32(0.0F);
    float32x4_t r71 = vdupq_n_f32(0.0F);

    for (std::size_t inner = 0; inner < k; ++inner) {
        const float* a_step = packed_a + inner * 8;
        const float* b_step = packed_b + inner * 8;
        const float32x4_t b0 = vld1q_f32(b_step);
        const float32x4_t b1 = vld1q_f32(b_step + 4);
        const float32x4_t a0 = vdupq_n_f32(a_step[0]);
        const float32x4_t a1 = vdupq_n_f32(a_step[1]);
        const float32x4_t a2 = vdupq_n_f32(a_step[2]);
        const float32x4_t a3 = vdupq_n_f32(a_step[3]);
        const float32x4_t a4 = vdupq_n_f32(a_step[4]);
        const float32x4_t a5 = vdupq_n_f32(a_step[5]);
        const float32x4_t a6 = vdupq_n_f32(a_step[6]);
        const float32x4_t a7 = vdupq_n_f32(a_step[7]);

        r00 = vfmaq_f32(r00, b0, a0); r01 = vfmaq_f32(r01, b1, a0);
        r10 = vfmaq_f32(r10, b0, a1); r11 = vfmaq_f32(r11, b1, a1);
        r20 = vfmaq_f32(r20, b0, a2); r21 = vfmaq_f32(r21, b1, a2);
        r30 = vfmaq_f32(r30, b0, a3); r31 = vfmaq_f32(r31, b1, a3);
        r40 = vfmaq_f32(r40, b0, a4); r41 = vfmaq_f32(r41, b1, a4);
        r50 = vfmaq_f32(r50, b0, a5); r51 = vfmaq_f32(r51, b1, a5);
        r60 = vfmaq_f32(r60, b0, a6); r61 = vfmaq_f32(r61, b1, a6);
        r70 = vfmaq_f32(r70, b0, a7); r71 = vfmaq_f32(r71, b1, a7);
    }

    vst1q_f32(c, apply_scale(r00, c, alpha, beta));
    vst1q_f32(c + 4, apply_scale(r01, c + 4, alpha, beta));
    vst1q_f32(c + ldc, apply_scale(r10, c + ldc, alpha, beta));
    vst1q_f32(c + ldc + 4, apply_scale(r11, c + ldc + 4, alpha, beta));
    vst1q_f32(c + 2 * ldc, apply_scale(r20, c + 2 * ldc, alpha, beta));
    vst1q_f32(c + 2 * ldc + 4, apply_scale(r21, c + 2 * ldc + 4, alpha, beta));
    vst1q_f32(c + 3 * ldc, apply_scale(r30, c + 3 * ldc, alpha, beta));
    vst1q_f32(c + 3 * ldc + 4, apply_scale(r31, c + 3 * ldc + 4, alpha, beta));
    vst1q_f32(c + 4 * ldc, apply_scale(r40, c + 4 * ldc, alpha, beta));
    vst1q_f32(c + 4 * ldc + 4, apply_scale(r41, c + 4 * ldc + 4, alpha, beta));
    vst1q_f32(c + 5 * ldc, apply_scale(r50, c + 5 * ldc, alpha, beta));
    vst1q_f32(c + 5 * ldc + 4, apply_scale(r51, c + 5 * ldc + 4, alpha, beta));
    vst1q_f32(c + 6 * ldc, apply_scale(r60, c + 6 * ldc, alpha, beta));
    vst1q_f32(c + 6 * ldc + 4, apply_scale(r61, c + 6 * ldc + 4, alpha, beta));
    vst1q_f32(c + 7 * ldc, apply_scale(r70, c + 7 * ldc, alpha, beta));
    vst1q_f32(c + 7 * ldc + 4, apply_scale(r71, c + 7 * ldc + 4, alpha, beta));
    return Status::Success;
}
#endif

}  // namespace

Status gemm_neon_4x4_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept {
    const Status validation = validate_call(
        k, packed_a, packed_b, c, ldc, mr, nr, 4, 4);
    if (validation != Status::Success) {
        return validation;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return run_4x4(k, packed_a, packed_b, c, ldc, alpha, beta);
#else
    (void)alpha;
    (void)beta;
    return Status::UnsupportedKernel;
#endif
}

Status gemm_neon_8x8_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept {
    const Status validation = validate_call(
        k, packed_a, packed_b, c, ldc, mr, nr, 8, 8);
    if (validation != Status::Success) {
        return validation;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return run_8x8(k, packed_a, packed_b, c, ldc, alpha, beta);
#else
    (void)alpha;
    (void)beta;
    return Status::UnsupportedKernel;
#endif
}

Status gemm_neon_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept {
    if (mr == 4 && nr == 4) {
        return gemm_neon_4x4_kernel(
            k, packed_a, packed_b, c, ldc, mr, nr, alpha, beta);
    }
    if (mr == 8 && nr == 8) {
        return gemm_neon_8x8_kernel(
            k, packed_a, packed_b, c, ldc, mr, nr, alpha, beta);
    }
    return Status::UnsupportedKernel;
}

const NeonKernelDescriptor* neon_kernel_descriptors(
    std::size_t* count) noexcept {
    if (count == nullptr) {
        return nullptr;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    static constexpr NeonKernelDescriptor descriptors[] = {
        {"neon-4x4", 4, 4, 1, gemm_neon_4x4_kernel},
        {"neon-8x8", 8, 8, 1, gemm_neon_8x8_kernel},
    };
    *count = sizeof(descriptors) / sizeof(descriptors[0]);
    return descriptors;
#else
    *count = 0;
    return nullptr;
#endif
}

}  // namespace apex::gemm
