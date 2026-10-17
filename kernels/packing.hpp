#pragma once

#include "../core/memory/matrix.hpp"
#include "../core/status.hpp"

#include <cstddef>

namespace apex::gemm {

// Tile dimensions shared by the panel packers and the selected micro-kernel.
struct PackingTileShape {
    std::size_t mr = 4;
    std::size_t nr = 4;
    std::size_t kr = 1;
};

struct PackedPanelLayout {
    std::size_t panel_count = 0;
    std::size_t padded_k = 0;
    std::size_t panel_stride_elements = 0;
    std::size_t required_elements = 0;
};

// A is stored as mr-row micro-panels; each K position contains mr adjacent
// values. B is stored as nr-column panels grouped into one macro-panel.
[[nodiscard]] Status packed_a_layout(
    std::size_t rows,
    std::size_t k,
    PackingTileShape tile,
    PackedPanelLayout* layout) noexcept;

[[nodiscard]] Status packed_b_layout(
    std::size_t k,
    std::size_t columns,
    PackingTileShape tile,
    PackedPanelLayout* layout) noexcept;

[[nodiscard]] Status pack_a(
    MatrixView source,
    std::size_t row_origin,
    std::size_t inner_origin,
    std::size_t rows,
    std::size_t k,
    PackingTileShape tile,
    float* destination,
    std::size_t destination_elements) noexcept;

[[nodiscard]] Status pack_b(
    MatrixView source,
    std::size_t inner_origin,
    std::size_t column_origin,
    std::size_t k,
    std::size_t columns,
    PackingTileShape tile,
    float* destination,
    std::size_t destination_elements) noexcept;

}  // namespace apex::gemm
