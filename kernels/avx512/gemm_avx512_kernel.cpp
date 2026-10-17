#include "gemm_avx512_kernel.hpp"

#if (defined(__AVX512F__) || defined(_M_AVX512F)) && !defined(APEX_FORCE_SCALAR)
#include <immintrin.h>
#endif

namespace apex::gemm {

#if (defined(__AVX512F__) || defined(_M_AVX512F)) && !defined(APEX_FORCE_SCALAR)
namespace {

void store_row(
    __m512 accumulator,
    float* output,
    __m512 alpha_values,
    __m512 beta_values,
    float beta) noexcept {
    __m512 result;
    if (beta == 0.0F) {
        result = _mm512_mul_ps(accumulator, alpha_values);
    } else {
        const __m512 current = _mm512_loadu_ps(output);
        if (beta == 1.0F) {
            result = _mm512_fmadd_ps(accumulator, alpha_values, current);
        } else {
            result = _mm512_fmadd_ps(
                accumulator, alpha_values,
                _mm512_mul_ps(current, beta_values));
        }
    }
    _mm512_storeu_ps(output, result);
}

}  // namespace
#endif

Status gemm_avx512_16x16_kernel(
    std::size_t k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr,
    float alpha,
    float beta) noexcept {
    if (mr != 16 || nr != 16 || ldc < nr) {
        return Status::InvalidArgument;
    }
    if (c == nullptr || (k != 0 && (packed_a == nullptr || packed_b == nullptr))) {
        return Status::NullData;
    }

#if (defined(__AVX512F__) || defined(_M_AVX512F)) && !defined(APEX_FORCE_SCALAR)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    __m512 acc8 = _mm512_setzero_ps();
    __m512 acc9 = _mm512_setzero_ps();
    __m512 acc10 = _mm512_setzero_ps();
    __m512 acc11 = _mm512_setzero_ps();
    __m512 acc12 = _mm512_setzero_ps();
    __m512 acc13 = _mm512_setzero_ps();
    __m512 acc14 = _mm512_setzero_ps();
    __m512 acc15 = _mm512_setzero_ps();

    for (std::size_t inner = 0; inner < k; ++inner) {
        const float* a_step = packed_a + inner * 16;
        const __m512 b_values = _mm512_loadu_ps(packed_b + inner * 16);
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[0]), b_values, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[1]), b_values, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[2]), b_values, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[3]), b_values, acc3);
        acc4 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[4]), b_values, acc4);
        acc5 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[5]), b_values, acc5);
        acc6 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[6]), b_values, acc6);
        acc7 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[7]), b_values, acc7);
        acc8 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[8]), b_values, acc8);
        acc9 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[9]), b_values, acc9);
        acc10 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[10]), b_values, acc10);
        acc11 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[11]), b_values, acc11);
        acc12 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[12]), b_values, acc12);
        acc13 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[13]), b_values, acc13);
        acc14 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[14]), b_values, acc14);
        acc15 = _mm512_fmadd_ps(_mm512_set1_ps(a_step[15]), b_values, acc15);
    }

    const __m512 alpha_values = _mm512_set1_ps(alpha);
    const __m512 beta_values = _mm512_set1_ps(beta);
    store_row(acc0, c, alpha_values, beta_values, beta);
    store_row(acc1, c + ldc, alpha_values, beta_values, beta);
    store_row(acc2, c + 2 * ldc, alpha_values, beta_values, beta);
    store_row(acc3, c + 3 * ldc, alpha_values, beta_values, beta);
    store_row(acc4, c + 4 * ldc, alpha_values, beta_values, beta);
    store_row(acc5, c + 5 * ldc, alpha_values, beta_values, beta);
    store_row(acc6, c + 6 * ldc, alpha_values, beta_values, beta);
    store_row(acc7, c + 7 * ldc, alpha_values, beta_values, beta);
    store_row(acc8, c + 8 * ldc, alpha_values, beta_values, beta);
    store_row(acc9, c + 9 * ldc, alpha_values, beta_values, beta);
    store_row(acc10, c + 10 * ldc, alpha_values, beta_values, beta);
    store_row(acc11, c + 11 * ldc, alpha_values, beta_values, beta);
    store_row(acc12, c + 12 * ldc, alpha_values, beta_values, beta);
    store_row(acc13, c + 13 * ldc, alpha_values, beta_values, beta);
    store_row(acc14, c + 14 * ldc, alpha_values, beta_values, beta);
    store_row(acc15, c + 15 * ldc, alpha_values, beta_values, beta);
    return Status::Success;
#else
    (void)alpha;
    (void)beta;
    return Status::UnsupportedKernel;
#endif
}

const Avx512KernelDescriptor* avx512_kernel_descriptors(
    std::size_t* count) noexcept {
    if (count == nullptr) {
        return nullptr;
    }
#if (defined(__AVX512F__) || defined(_M_AVX512F)) && !defined(APEX_FORCE_SCALAR)
    static constexpr Avx512KernelDescriptor descriptors[] = {
        {"avx512-16x16", 16, 16, 1, gemm_avx512_16x16_kernel},
    };
    *count = sizeof(descriptors) / sizeof(descriptors[0]);
    return descriptors;
#else
    *count = 0;
    return nullptr;
#endif
}

}  // namespace apex::gemm
