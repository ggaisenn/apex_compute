#pragma once

#include "../../core/platform/cpu_features.hpp"
#include "../../kernels/kernel_contract.hpp"

namespace apex::gemm {

enum class KernelPreference {
    Automatic,
    Scalar,
    Neon,
    Avx512,
};

struct KernelSelectionOptions {
    KernelPreference preference = KernelPreference::Automatic;
};

class KernelDispatcher {
public:
    [[nodiscard]] static Status select(
        const KernelSelectionOptions& options,
        const CpuFeatures& cpu,
        KernelDescriptor* output) noexcept;

    [[nodiscard]] static Status select(
        const KernelSelectionOptions& options,
        KernelDescriptor* output) noexcept;
};

}  // namespace apex::gemm
