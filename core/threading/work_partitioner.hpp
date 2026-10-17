#pragma once

#include "../status.hpp"

#include <cstddef>
#include <limits>

namespace apex::gemm {

struct WorkPartitionRequest {
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t mr = 0;
    std::size_t nr = 0;
    std::size_t worker_count = 1;
    std::size_t cache_line_bytes = 64;
};

struct WorkRegion {
    std::size_t worker_index = 0;
    std::size_t row_begin = 0;
    std::size_t row_end = 0;
    std::size_t column_begin = 0;
    std::size_t column_end = 0;
    std::size_t row_tile_begin = 0;
    std::size_t row_tile_end = 0;
    std::size_t column_tile_begin = 0;
    std::size_t column_tile_end = 0;
};

struct WorkPartition {
    std::size_t worker_count = 0;
    std::size_t row_partitions = 0;
    std::size_t column_partitions = 0;
};

class WorkPartitioner {
public:
    [[nodiscard]] static Status partition(
        const WorkPartitionRequest& request,
        WorkRegion* regions,
        std::size_t region_capacity,
        WorkPartition* output) noexcept {
        if (output == nullptr) {
            return Status::InvalidArgument;
        }
        if (request.rows == 0 || request.columns == 0) {
            return Status::EmptyMatrix;
        }
        if (request.mr == 0 || request.nr == 0 || request.worker_count == 0) {
            return Status::InvalidConfig;
        }

        const std::size_t row_tile_count = ceil_div(request.rows, request.mr);
        const std::size_t column_tile_count = ceil_div(request.columns, request.nr);
        const std::size_t cache_line_bytes =
            request.cache_line_bytes > 64 ? request.cache_line_bytes : 64;
        const std::size_t line_elements = ceil_div(
            cache_line_bytes, sizeof(float));
        const std::size_t alignment_elements =
            least_common_multiple(request.nr, line_elements);
        if (alignment_elements == 0) {
            return Status::SizeOverflow;
        }
        const std::size_t column_tiles_per_group = alignment_elements / request.nr;
        const std::size_t row_groups = row_tile_count;
        const std::size_t column_groups = ceil_div(
            column_tile_count, column_tiles_per_group);

        std::size_t row_partitions = 0;
        std::size_t column_partitions = 0;
        std::size_t active_workers = 0;
        choose_grid(
            request, row_groups, column_groups,
            &row_partitions, &column_partitions, &active_workers);
        if (active_workers == 0) {
            return Status::InvalidConfig;
        }
        if (regions == nullptr) {
            return Status::NullData;
        }
        if (region_capacity < active_workers) {
            return Status::BufferTooSmall;
        }

        std::size_t worker = 0;
        // Grid cells have disjoint tile ranges; cache-line aligned column
        // boundaries also keep neighboring workers from sharing output lines.
        for (std::size_t row_part = 0; row_part < row_partitions; ++row_part) {
            const std::size_t row_group_begin = balanced_begin(
                row_groups, row_partitions, row_part);
            const std::size_t row_group_end = balanced_begin(
                row_groups, row_partitions, row_part + 1);
            const std::size_t first_row_tile = row_group_begin;
            const std::size_t last_row_tile = row_group_end;
            const std::size_t row_begin = first_row_tile * request.mr;
            const std::size_t row_end = clipped_product(
                last_row_tile, request.mr, request.rows);

            for (std::size_t column_part = 0;
                 column_part < column_partitions; ++column_part) {
                const std::size_t column_group_begin = balanced_begin(
                    column_groups, column_partitions, column_part);
                const std::size_t column_group_end = balanced_begin(
                    column_groups, column_partitions, column_part + 1);
                const std::size_t first_column_tile = clipped_product(
                    column_group_begin, column_tiles_per_group,
                    column_tile_count);
                const std::size_t last_column_tile = minimum(
                    clipped_product(
                        column_group_end, column_tiles_per_group,
                        column_tile_count),
                    column_tile_count);
                const std::size_t column_begin =
                    first_column_tile * request.nr;
                const std::size_t column_end = clipped_product(
                    last_column_tile, request.nr, request.columns);

                regions[worker] = WorkRegion{
                    worker,
                    row_begin,
                    row_end,
                    column_begin,
                    column_end,
                    first_row_tile,
                    last_row_tile,
                    first_column_tile,
                    last_column_tile};
                ++worker;
            }
        }

        *output = WorkPartition{
            active_workers, row_partitions, column_partitions};
        return Status::Success;
    }

private:
    [[nodiscard]] static std::size_t ceil_div(
        std::size_t value,
        std::size_t divisor) noexcept {
        return value / divisor + static_cast<std::size_t>(value % divisor != 0);
    }

    [[nodiscard]] static std::size_t gcd(
        std::size_t left,
        std::size_t right) noexcept {
        while (right != 0) {
            const std::size_t remainder = left % right;
            left = right;
            right = remainder;
        }
        return left;
    }

    [[nodiscard]] static std::size_t least_common_multiple(
        std::size_t left,
        std::size_t right) noexcept {
        const std::size_t divisor = gcd(left, right);
        if (divisor == 0 || left / divisor >
                std::numeric_limits<std::size_t>::max() / right) {
            return 0;
        }
        return (left / divisor) * right;
    }

    [[nodiscard]] static std::size_t balanced_begin(
        std::size_t items,
        std::size_t partitions,
        std::size_t index) noexcept {
        const std::size_t base = items / partitions;
        const std::size_t remainder = items % partitions;
        return index * base + minimum(index, remainder);
    }

    [[nodiscard]] static std::size_t clipped_product(
        std::size_t left,
        std::size_t right,
        std::size_t limit) noexcept {
        if (left != 0 && right > limit / left) {
            return limit;
        }
        const std::size_t product = left * right;
        return product < limit ? product : limit;
    }

    [[nodiscard]] static std::size_t minimum(
        std::size_t left,
        std::size_t right) noexcept {
        return left < right ? left : right;
    }

    static void choose_grid(
        const WorkPartitionRequest& request,
        std::size_t row_groups,
        std::size_t column_groups,
        std::size_t* row_partitions,
        std::size_t* column_partitions,
        std::size_t* active_workers) noexcept {
        if (row_partitions == nullptr || column_partitions == nullptr ||
            active_workers == nullptr) {
            return;
        }

        std::size_t best_workers = 0;
        long double best_aspect = std::numeric_limits<long double>::max();
        const std::size_t max_rows = minimum(row_groups, request.worker_count);
        for (std::size_t rows = 1; rows <= max_rows; ++rows) {
            const std::size_t columns = minimum(
                column_groups, request.worker_count / rows);
            if (columns == 0) {
                continue;
            }
            const std::size_t workers = rows * columns;
            const long double region_height =
                static_cast<long double>(request.rows) / rows;
            const long double region_width =
                static_cast<long double>(request.columns) / columns;
            const long double aspect = region_height > region_width
                ? region_height / region_width
                : region_width / region_height;
            if (workers > best_workers ||
                (workers == best_workers && aspect < best_aspect)) {
                best_workers = workers;
                best_aspect = aspect;
                *row_partitions = rows;
                *column_partitions = columns;
            }
        }
        *active_workers = best_workers;
    }
};

}  // namespace apex::gemm
