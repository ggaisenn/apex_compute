#include "kernel_dispatcher.hpp"

#include "../../kernels/avx512/gemm_avx512_kernel.hpp"
#include "../../kernels/neon/gemm_neon_kernel.hpp"
#include "../../kernels/scalar/gemm_scalar_kernel.hpp"

namespace apex::gemm {
namespace {

#if (defined(APEX_KERNEL_OVERRIDE_SCALAR) + defined(APEX_KERNEL_OVERRIDE_NEON) + \
     defined(APEX_KERNEL_OVERRIDE_AVX512)) > 1
#error "Select no more than one APEX kernel compile-time override."
#endif

[[nodiscard]] KernelPreference effective_preference(
    KernelPreference requested) noexcept {
#if defined(APEX_FORCE_SCALAR) || defined(APEX_KERNEL_OVERRIDE_SCALAR)
    (void)requested;
    return KernelPreference::Scalar;
#elif defined(APEX_KERNEL_OVERRIDE_NEON)
    (void)requested;
    return KernelPreference::Neon;
#elif defined(APEX_KERNEL_OVERRIDE_AVX512)
    (void)requested;
    return KernelPreference::Avx512;
#else
    return requested;
#endif
}

[[nodiscard]] bool find_isa_descriptor(
    KernelKind kind,
    KernelPreference preference,
    const CpuFeatures& cpu,
    KernelDescriptor* result) noexcept {
    if (result == nullptr) {
        return false;
    }

    if (kind == KernelKind::Avx512) {
        if (!cpu.avx512f()) {
            return false;
        }
        std::size_t count = 0;
        const Avx512KernelDescriptor* descriptors = avx512_kernel_descriptors(&count);
        for (std::size_t index = 0; descriptors != nullptr && index < count; ++index) {
            const Avx512KernelDescriptor& descriptor = descriptors[index];
            if (preference == KernelPreference::Automatic ||
                preference == KernelPreference::Avx512) {
                *result = KernelDescriptor{
                    descriptor.name, KernelKind::Avx512,
                    descriptor.mr, descriptor.nr, descriptor.kr, descriptor.function};
                return true;
            }
        }
        return false;
    }

    if (kind == KernelKind::Neon) {
        if (!cpu.neon()) {
            return false;
        }
        std::size_t count = 0;
        const NeonKernelDescriptor* descriptors = neon_kernel_descriptors(&count);
        for (std::size_t index = 0; descriptors != nullptr && index < count; ++index) {
            const NeonKernelDescriptor& descriptor = descriptors[index];
            if (preference == KernelPreference::Automatic ||
                preference == KernelPreference::Neon) {
                *result = KernelDescriptor{
                    descriptor.name, KernelKind::Neon,
                    descriptor.mr, descriptor.nr, descriptor.kr, descriptor.function};
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] KernelDescriptor scalar_descriptor() noexcept {
    return KernelDescriptor{
        "scalar", KernelKind::Scalar, 4, 4, 1, scalar_packed_microkernel};
}

}  // namespace

Status KernelDispatcher::select(
    const KernelSelectionOptions& options,
    const CpuFeatures& cpu,
    KernelDescriptor* output) noexcept {
    if (output == nullptr) {
        return Status::InvalidArgument;
    }

    const KernelPreference preference = effective_preference(options.preference);
    KernelDescriptor replacement{};
    switch (preference) {
        case KernelPreference::Scalar:
            replacement = scalar_descriptor();
            break;
        case KernelPreference::Neon:
            if (!find_isa_descriptor(
                    KernelKind::Neon, preference, cpu, &replacement)) {
                return Status::UnsupportedKernel;
            }
            break;
        case KernelPreference::Avx512:
            if (!find_isa_descriptor(
                    KernelKind::Avx512, preference, cpu, &replacement)) {
                return Status::UnsupportedKernel;
            }
            break;
        case KernelPreference::Automatic:
            if (find_isa_descriptor(
                    KernelKind::Avx512, preference, cpu, &replacement) ||
                find_isa_descriptor(
                    KernelKind::Neon, preference, cpu, &replacement)) {
                break;
            }
            replacement = scalar_descriptor();
            break;
        default:
            return Status::InvalidArgument;
    }

    *output = replacement;
    return Status::Success;
}

Status KernelDispatcher::select(
    const KernelSelectionOptions& options,
    KernelDescriptor* output) noexcept {
    return select(options, CpuFeatures::current(), output);
}

}  // namespace apex::gemm
