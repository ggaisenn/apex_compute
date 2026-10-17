#pragma once

#include "../platform/cpu_features.hpp"
#include "../status.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace apex::gemm {

struct KernelTileShape {
    std::size_t mr = 4;
    std::size_t nr = 4;
    std::size_t kr = 1;
};

struct BlockSizeOverrides {
    // A zero value selects a cache-derived block size.
    std::size_t mc = 0;
    std::size_t kc = 0;
    std::size_t nc = 0;
};

struct TilingPlannerOptions {
    KernelTileShape tile{};
    BlockSizeOverrides overrides{};
    std::size_t thread_count = 1;
    std::uint32_t cache_utilization_percent = 50;
};

struct TilingCacheGeometry {
    CacheSize l1d{};
    CacheSize l2{};
    CacheSize l3{};
};

struct TilingPlan {
    std::size_t mc = 0;
    std::size_t kc = 0;
    std::size_t nc = 0;
    std::size_t l1d_target_bytes = 0;
    std::size_t l2_target_bytes = 0;
    std::size_t l3_per_thread_target_bytes = 0;
    CacheSize l1d_cache{};
    CacheSize l2_cache{};
    CacheSize l3_cache{};
};

class TilingPlanner {
public:
    static constexpr std::size_t kMaximumManualBlockSize = 1U << 20U;

    TilingPlanner() noexcept = default;

    [[nodiscard]] static Status create(
        const TilingPlannerOptions& options,
        const TilingCacheGeometry& cache_geometry,
        TilingPlanner* output) noexcept {
        if (output == nullptr) {
            return Status::InvalidArgument;
        }
        if (!valid_options(options) || cache_geometry.l1d.bytes == 0 ||
            cache_geometry.l2.bytes == 0 || cache_geometry.l3.bytes == 0) {
            return Status::InvalidConfig;
        }

        const std::size_t l1_budget = cache_budget(
            cache_geometry.l1d.bytes, options.cache_utilization_percent);
        const std::size_t l2_budget = cache_budget(
            cache_geometry.l2.bytes, options.cache_utilization_percent);
        const std::size_t l3_budget = cache_budget(
            cache_geometry.l3.bytes, options.cache_utilization_percent) /
            options.thread_count;

        const std::size_t b_micro_panel_bytes =
            options.tile.nr * sizeof(float);
        if (b_micro_panel_bytes / sizeof(float) != options.tile.nr) {
            return Status::SizeOverflow;
        }

        // Keep the packed B micro-panel (kc by nr floats) inside the L1 budget.
        std::size_t kc = round_down_to_multiple(
            l1_budget / b_micro_panel_bytes, options.tile.kr);
        if (kc < options.tile.kr) {
            kc = options.tile.kr;
        }
        if (options.overrides.kc != 0 &&
            !round_up_to_multiple(options.overrides.kc, options.tile.kr, &kc)) {
            return Status::SizeOverflow;
        }
        if (kc > kMaximumManualBlockSize && options.overrides.kc != 0) {
            return Status::InvalidConfig;
        }

        const std::size_t a_column_bytes = kc * sizeof(float);
        if (a_column_bytes / sizeof(float) != kc) {
            return Status::SizeOverflow;
        }
        // Keep the packed A panel (mc by kc floats) inside the L2 budget.
        std::size_t mc = round_down_to_multiple(
            l2_budget / a_column_bytes, options.tile.mr);
        if (mc < options.tile.mr) {
            mc = options.tile.mr;
        }
        if (options.overrides.mc != 0 &&
            !round_up_to_multiple(options.overrides.mc, options.tile.mr, &mc)) {
            return Status::SizeOverflow;
        }
        if (mc > kMaximumManualBlockSize && options.overrides.mc != 0) {
            return Status::InvalidConfig;
        }

        // The packed B block uses a per-thread share of the last-level cache.
        std::size_t nc = round_down_to_multiple(
            l3_budget / a_column_bytes, options.tile.nr);
        if (nc < options.tile.nr) {
            nc = options.tile.nr;
        }
        if (options.overrides.nc != 0 &&
            !round_up_to_multiple(options.overrides.nc, options.tile.nr, &nc)) {
            return Status::SizeOverflow;
        }
        if (nc > kMaximumManualBlockSize && options.overrides.nc != 0) {
            return Status::InvalidConfig;
        }

        TilingPlanner replacement;
        replacement.options_ = options;
        replacement.cache_geometry_ = cache_geometry;
        replacement.base_mc_ = mc;
        replacement.base_kc_ = kc;
        replacement.base_nc_ = nc;
        replacement.l1_budget_ = l1_budget;
        replacement.l2_budget_ = l2_budget;
        replacement.l3_budget_per_thread_ = l3_budget;
        *output = replacement;
        return Status::Success;
    }

    [[nodiscard]] static Status create(
        const TilingPlannerOptions& options,
        TilingPlanner* output) noexcept {
        const CpuFeatures& cpu = CpuFeatures::current();
        const TilingCacheGeometry cache_geometry{
            cpu.l1d_cache(), cpu.l2_cache(), cpu.l3_cache()};
        return create(options, cache_geometry, output);
    }

    [[nodiscard]] Status plan(
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        TilingPlan* output) const noexcept {
        if (output == nullptr) {
            return Status::InvalidArgument;
        }
        if (rows == 0 || inner == 0 || columns == 0) {
            return Status::EmptyMatrix;
        }
        if (base_mc_ == 0 || base_kc_ == 0 || base_nc_ == 0) {
            return Status::InvalidConfig;
        }

        TilingPlan replacement;
        replacement.mc = rows < base_mc_ ? rows : base_mc_;
        replacement.kc = inner < base_kc_ ? inner : base_kc_;
        replacement.nc = columns < base_nc_ ? columns : base_nc_;
        replacement.l1d_target_bytes = l1_budget_;
        replacement.l2_target_bytes = l2_budget_;
        replacement.l3_per_thread_target_bytes = l3_budget_per_thread_;
        replacement.l1d_cache = cache_geometry_.l1d;
        replacement.l2_cache = cache_geometry_.l2;
        replacement.l3_cache = cache_geometry_.l3;
        *output = replacement;
        return Status::Success;
    }

    [[nodiscard]] std::size_t base_mc() const noexcept { return base_mc_; }
    [[nodiscard]] std::size_t base_kc() const noexcept { return base_kc_; }
    [[nodiscard]] std::size_t base_nc() const noexcept { return base_nc_; }

private:
    [[nodiscard]] static bool valid_options(
        const TilingPlannerOptions& options) noexcept {
        return options.tile.mr != 0 && options.tile.nr != 0 &&
            options.tile.kr != 0 && options.thread_count != 0 &&
            options.tile.mr <= kMaximumManualBlockSize &&
            options.tile.nr <= kMaximumManualBlockSize &&
            options.tile.kr <= kMaximumManualBlockSize &&
            options.cache_utilization_percent != 0 &&
            options.cache_utilization_percent <= 100 &&
            options.overrides.mc <= kMaximumManualBlockSize &&
            options.overrides.kc <= kMaximumManualBlockSize &&
            options.overrides.nc <= kMaximumManualBlockSize;
    }

    [[nodiscard]] static std::size_t cache_budget(
        std::size_t cache_bytes,
        std::uint32_t utilization_percent) noexcept {
        const std::size_t percent = utilization_percent;
        return (cache_bytes / 100U) * percent +
            ((cache_bytes % 100U) * percent) / 100U;
    }

    [[nodiscard]] static std::size_t round_down_to_multiple(
        std::size_t value,
        std::size_t multiple) noexcept {
        return value - (value % multiple);
    }

    [[nodiscard]] static bool round_up_to_multiple(
        std::size_t value,
        std::size_t multiple,
        std::size_t* output) noexcept {
        if (output == nullptr || multiple == 0) {
            return false;
        }
        const std::size_t remainder = value % multiple;
        if (remainder == 0) {
            *output = value;
            return true;
        }
        const std::size_t increment = multiple - remainder;
        if (value > std::numeric_limits<std::size_t>::max() - increment) {
            return false;
        }
        *output = value + increment;
        return true;
    }

    TilingPlannerOptions options_{};
    TilingCacheGeometry cache_geometry_{};
    std::size_t base_mc_ = 0;
    std::size_t base_kc_ = 0;
    std::size_t base_nc_ = 0;
    std::size_t l1_budget_ = 0;
    std::size_t l2_budget_ = 0;
    std::size_t l3_budget_per_thread_ = 0;
};

}  // namespace apex::gemm
