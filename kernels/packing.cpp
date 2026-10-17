#include "packing.hpp"

#include <cstdint>
#include <limits>

namespace apex::gemm {
namespace {

constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kCacheLineElements = kCacheLineBytes / kFloatBytes;

[[nodiscard]] bool checked_multiply(
    std::size_t left,
    std::size_t right,
    std::size_t* result) noexcept {
    if (result == nullptr ||
        (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

[[nodiscard]] bool checked_round_up(
    std::size_t value,
    std::size_t multiple,
    std::size_t* result) noexcept {
    if (result == nullptr || multiple == 0) {
        return false;
    }
    const std::size_t remainder = value % multiple;
    if (remainder == 0) {
        *result = value;
        return true;
    }
    const std::size_t increment = multiple - remainder;
    if (value > std::numeric_limits<std::size_t>::max() - increment) {
        return false;
    }
    *result = value + increment;
    return true;
}

[[nodiscard]] Status calculate_layout(
    std::size_t panel_extent,
    std::size_t k,
    std::size_t panel_width,
    std::size_t kr,
    PackedPanelLayout* layout) noexcept {
    if (layout == nullptr || panel_width == 0 || kr == 0) {
        return Status::InvalidArgument;
    }

    PackedPanelLayout result;
    if (panel_extent == 0 || k == 0) {
        *layout = result;
        return Status::Success;
    }

    std::size_t padded_extent = 0;
    if (!checked_round_up(panel_extent, panel_width, &padded_extent) ||
        !checked_round_up(k, kr, &result.padded_k)) {
        return Status::SizeOverflow;
    }
    result.panel_count = padded_extent / panel_width;

    std::size_t panel_elements = 0;
    if (!checked_multiply(result.padded_k, panel_width, &panel_elements) ||
        !checked_round_up(panel_elements, kCacheLineElements,
                          &result.panel_stride_elements) ||
        !checked_multiply(result.panel_count, result.panel_stride_elements,
                          &result.required_elements)) {
        return Status::SizeOverflow;
    }

    *layout = result;
    return Status::Success;
}

[[nodiscard]] Status validate_destination(
    float* destination,
    std::size_t capacity,
    std::size_t required) noexcept {
    if (required == 0) {
        return Status::Success;
    }
    if (destination == nullptr) {
        return Status::NullData;
    }
    if (capacity < required) {
        return Status::BufferTooSmall;
    }
    if (reinterpret_cast<std::uintptr_t>(destination) % kCacheLineBytes != 0) {
        return Status::InvalidArgument;
    }
    return Status::Success;
}

[[nodiscard]] bool range_fits(
    std::size_t origin,
    std::size_t extent,
    std::size_t bound) noexcept {
    return origin <= bound && extent <= bound - origin;
}

}  // namespace

Status packed_a_layout(
    std::size_t rows,
    std::size_t k,
    PackingTileShape tile,
    PackedPanelLayout* layout) noexcept {
    return calculate_layout(rows, k, tile.mr, tile.kr, layout);
}

Status packed_b_layout(
    std::size_t k,
    std::size_t columns,
    PackingTileShape tile,
    PackedPanelLayout* layout) noexcept {
    return calculate_layout(columns, k, tile.nr, tile.kr, layout);
}

Status pack_a(
    MatrixView source,
    std::size_t row_origin,
    std::size_t inner_origin,
    std::size_t rows,
    std::size_t k,
    PackingTileShape tile,
    float* destination,
    std::size_t destination_elements) noexcept {
    const Status source_status = source.validate();
    if (source_status != Status::Success) {
        return source_status;
    }
    if (tile.mr == 0 || tile.kr == 0) {
        return Status::InvalidArgument;
    }
    if (!range_fits(row_origin, rows, source.rows()) ||
        !range_fits(inner_origin, k, source.columns())) {
        return Status::InvalidDimension;
    }

    PackedPanelLayout layout;
    const Status layout_status = packed_a_layout(rows, k, tile, &layout);
    if (layout_status != Status::Success) {
        return layout_status;
    }
    const Status destination_status = validate_destination(
        destination, destination_elements, layout.required_elements);
    if (destination_status != Status::Success || layout.required_elements == 0) {
        return destination_status;
    }

    for (std::size_t panel = 0; panel < layout.panel_count; ++panel) {
        const std::size_t panel_base = panel * layout.panel_stride_elements;
        for (std::size_t packed_k = 0; packed_k < layout.padded_k; ++packed_k) {
            for (std::size_t lane = 0; lane < tile.mr; ++lane) {
                const std::size_t output_index = panel_base + packed_k * tile.mr + lane;
                if (packed_k < k && panel * tile.mr + lane < rows) {
                    destination[output_index] = source(
                        row_origin + panel * tile.mr + lane,
                        inner_origin + packed_k);
                } else {
                    destination[output_index] = 0.0F;
                }
            }
        }
        const std::size_t data_end = panel_base + layout.padded_k * tile.mr;
        const std::size_t panel_end = panel_base + layout.panel_stride_elements;
        for (std::size_t index = data_end; index < panel_end; ++index) {
            destination[index] = 0.0F;
        }
    }
    return Status::Success;
}

Status pack_b(
    MatrixView source,
    std::size_t inner_origin,
    std::size_t column_origin,
    std::size_t k,
    std::size_t columns,
    PackingTileShape tile,
    float* destination,
    std::size_t destination_elements) noexcept {
    const Status source_status = source.validate();
    if (source_status != Status::Success) {
        return source_status;
    }
    if (tile.nr == 0 || tile.kr == 0) {
        return Status::InvalidArgument;
    }
    if (!range_fits(inner_origin, k, source.rows()) ||
        !range_fits(column_origin, columns, source.columns())) {
        return Status::InvalidDimension;
    }

    PackedPanelLayout layout;
    const Status layout_status = packed_b_layout(k, columns, tile, &layout);
    if (layout_status != Status::Success) {
        return layout_status;
    }
    const Status destination_status = validate_destination(
        destination, destination_elements, layout.required_elements);
    if (destination_status != Status::Success || layout.required_elements == 0) {
        return destination_status;
    }

    for (std::size_t panel = 0; panel < layout.panel_count; ++panel) {
        const std::size_t panel_base = panel * layout.panel_stride_elements;
        for (std::size_t packed_k = 0; packed_k < layout.padded_k; ++packed_k) {
            for (std::size_t lane = 0; lane < tile.nr; ++lane) {
                const std::size_t output_index = panel_base + packed_k * tile.nr + lane;
                if (packed_k < k && panel * tile.nr + lane < columns) {
                    destination[output_index] = source(
                        inner_origin + packed_k,
                        column_origin + panel * tile.nr + lane);
                } else {
                    destination[output_index] = 0.0F;
                }
            }
        }
        const std::size_t data_end = panel_base + layout.padded_k * tile.nr;
        const std::size_t panel_end = panel_base + layout.panel_stride_elements;
        for (std::size_t index = data_end; index < panel_end; ++index) {
            destination[index] = 0.0F;
        }
    }
    return Status::Success;
}

}  // namespace apex::gemm
