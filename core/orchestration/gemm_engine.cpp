#include "gemm_engine.hpp"

#include "../gemm/naive_baseline.hpp"

#include <limits>
#include <utility>

namespace apex::gemm {
namespace {

[[nodiscard]] bool checked_multiply(
    std::size_t left,
    std::size_t right,
    std::size_t* output) noexcept {
    if (output == nullptr ||
        (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

[[nodiscard]] Status ensure_capacity(
    std::size_t elements,
    AlignedBuffer* buffer) noexcept {
    if (buffer == nullptr) {
        return Status::InvalidArgument;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return Status::SizeOverflow;
    }
    const std::size_t required_bytes = elements * sizeof(float);
    if (buffer->size_bytes() >= required_bytes) {
        return Status::Success;
    }

    AlignedBuffer replacement;
    const Status allocation_status =
        AlignedAllocator::allocate(required_bytes, &replacement);
    if (allocation_status != Status::Success) {
        return allocation_status;
    }
    *buffer = std::move(replacement);
    return Status::Success;
}

[[nodiscard]] std::size_t minimum(std::size_t left, std::size_t right) noexcept {
    return left < right ? left : right;
}

void scale_output(MatrixView c, float beta) noexcept {
    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t column = 0; column < c.columns(); ++column) {
            c(row, column) = beta == 0.0F ? 0.0F : beta * c(row, column);
        }
    }
}

}  // namespace

Status GemmWorkspace::reserve(
    const TilingPlan& plan,
    PackingTileShape tile) noexcept {
    if (plan.mc == 0 || plan.kc == 0 || plan.nc == 0 ||
        tile.mr == 0 || tile.nr == 0 || tile.kr == 0) {
        return Status::InvalidConfig;
    }

    PackedPanelLayout a_layout;
    Status status = packed_a_layout(plan.mc, plan.kc, tile, &a_layout);
    if (status != Status::Success) {
        return status;
    }
    PackedPanelLayout b_layout;
    status = packed_b_layout(plan.kc, plan.nc, tile, &b_layout);
    if (status != Status::Success) {
        return status;
    }
    std::size_t edge_elements = 0;
    if (!checked_multiply(tile.mr, tile.nr, &edge_elements)) {
        return Status::SizeOverflow;
    }

    status = ensure_capacity(a_layout.required_elements, &packed_a_);
    if (status != Status::Success) {
        return status;
    }
    status = ensure_capacity(b_layout.required_elements, &packed_b_);
    if (status != Status::Success) {
        return status;
    }
    return ensure_capacity(edge_elements, &edge_tile_);
}

Status GemmEngine::execute(
    MatrixView a,
    MatrixView b,
    MatrixView c,
    const TilingPlan& plan,
    PackingTileShape tile,
    GemmWorkspace* workspace,
    MicroKernelFunction micro_kernel,
    float alpha,
    float beta) noexcept {
    const Status validation_status = detail::validate_gemm_views(a, b, c);
    if (validation_status != Status::Success) {
        return validation_status;
    }
    if (c.rows() == 0 || c.columns() == 0) {
        return Status::Success;
    }
    if (alpha == 0.0F || a.columns() == 0) {
        scale_output(c, beta);
        return Status::Success;
    }
    if (workspace == nullptr) {
        return Status::InvalidArgument;
    }
    if (plan.mc == 0 || plan.kc == 0 || plan.nc == 0 ||
        tile.mr == 0 || tile.nr == 0 || tile.kr == 0) {
        return Status::InvalidConfig;
    }

    const Status reserve_status = workspace->reserve(plan, tile);
    if (reserve_status != Status::Success) {
        return reserve_status;
    }
    if (micro_kernel == nullptr) {
        micro_kernel = scalar_packed_microkernel;
    }

    const std::size_t m = c.rows();
    const std::size_t n = c.columns();
    const std::size_t k = a.columns();
    for (std::size_t column_origin = 0; column_origin < n;) {
        const std::size_t nc = minimum(plan.nc, n - column_origin);
        for (std::size_t inner_origin = 0; inner_origin < k;) {
            const std::size_t kc = minimum(plan.kc, k - inner_origin);
            const Status pack_b_status = pack_b(
                b, inner_origin, column_origin, kc, nc, tile,
                static_cast<float*>(workspace->packed_b_.data()),
                workspace->packed_b_.size_bytes() / sizeof(float));
            if (pack_b_status != Status::Success) {
                return pack_b_status;
            }

            const float block_beta = inner_origin == 0 ? beta : 1.0F;
            for (std::size_t row_origin = 0; row_origin < m;) {
                const std::size_t mc = minimum(plan.mc, m - row_origin);
                const Status pack_a_status = pack_a(
                    a, row_origin, inner_origin, mc, kc, tile,
                    static_cast<float*>(workspace->packed_a_.data()),
                    workspace->packed_a_.size_bytes() / sizeof(float));
                if (pack_a_status != Status::Success) {
                    return pack_a_status;
                }

                PackedPanelLayout a_layout;
                PackedPanelLayout b_layout;
                Status status = packed_a_layout(mc, kc, tile, &a_layout);
                if (status != Status::Success) {
                    return status;
                }
                status = packed_b_layout(kc, nc, tile, &b_layout);
                if (status != Status::Success) {
                    return status;
                }

                // Keep each packed B micro-panel hot while streaming A panels.
                for (std::size_t b_panel = 0; b_panel < b_layout.panel_count; ++b_panel) {
                    const float* packed_b =
                        static_cast<const float*>(workspace->packed_b_.data()) +
                        b_panel * b_layout.panel_stride_elements;
                    for (std::size_t a_panel = 0;
                         a_panel < a_layout.panel_count; ++a_panel) {
                        const float* packed_a =
                            static_cast<const float*>(workspace->packed_a_.data()) +
                            a_panel * a_layout.panel_stride_elements;
                        const std::size_t tile_row = row_origin + a_panel * tile.mr;
                        const std::size_t tile_column =
                            column_origin + b_panel * tile.nr;
                        const std::size_t valid_rows = minimum(tile.mr, m - tile_row);
                        const std::size_t valid_columns = minimum(tile.nr, n - tile_column);
                        float* output = nullptr;
                        std::size_t output_stride = c.leading_dimension();

                        if (valid_rows == tile.mr && valid_columns == tile.nr) {
                            output = &c(tile_row, tile_column);
                        } else {
                            output = static_cast<float*>(workspace->edge_tile_.data());
                            output_stride = tile.nr;
                            for (std::size_t row = 0; row < tile.mr; ++row) {
                                for (std::size_t column = 0; column < tile.nr; ++column) {
                                    if (row < valid_rows && column < valid_columns) {
                                        output[row * output_stride + column] =
                                            block_beta == 0.0F
                                            ? 0.0F
                                            : c(tile_row + row, tile_column + column);
                                    } else {
                                        output[row * output_stride + column] = 0.0F;
                                    }
                                }
                            }
                        }

                        status = micro_kernel(
                            a_layout.padded_k, packed_a, packed_b, output, output_stride,
                            tile.mr, tile.nr, alpha, block_beta);
                        if (status != Status::Success) {
                            return status;
                        }

                        if (valid_rows != tile.mr || valid_columns != tile.nr) {
                            for (std::size_t row = 0; row < valid_rows; ++row) {
                                for (std::size_t column = 0;
                                     column < valid_columns; ++column) {
                                    c(tile_row + row, tile_column + column) =
                                        output[row * output_stride + column];
                                }
                            }
                        }
                    }
                }
                row_origin += mc;
            }
            inner_origin += kc;
        }
        column_origin += nc;
    }
    return Status::Success;
}

}  // namespace apex::gemm
