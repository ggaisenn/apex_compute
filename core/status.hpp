#pragma once

#include <cstdint>

namespace apex::gemm {

enum class Status : std::uint8_t {
    Success = 0,
    InvalidDimension = 1,
    AllocationFailed = 2,
    UnsupportedKernel = 3,
    InvalidArgument = 4,
    InvalidLeadingDimension = 5,
    SizeOverflow = 6,
    NullData = 7,
    OutputOverlap = 8,
    InvalidConfig = 9,
    Unsupported = 10,
    InternalError = 11,
    BufferTooSmall = 12,
    EmptyMatrix = 13,
    KernelUnavailable = 14,
    success = Success,
    invalid_argument = InvalidArgument,
    allocation_failure = AllocationFailed,
    invalid_leading_dimension = InvalidLeadingDimension,
    size_overflow = SizeOverflow,
    buffer_too_small = BufferTooSmall,
    empty_matrix = EmptyMatrix,
};

[[nodiscard]] constexpr const char* status_name(Status status) noexcept {
    switch (status) {
        case Status::Success:
            return "Ok";
        case Status::InvalidDimension:
            return "InvalidDimension";
        case Status::AllocationFailed:
            return "AllocationFailed";
        case Status::UnsupportedKernel:
            return "UnsupportedKernel";
        case Status::InvalidArgument:
            return "InvalidArgument";
        case Status::InvalidLeadingDimension:
            return "InvalidLeadingDimension";
        case Status::SizeOverflow:
            return "SizeOverflow";
        case Status::NullData:
            return "NullData";
        case Status::OutputOverlap:
            return "OutputOverlap";
        case Status::InvalidConfig:
            return "InvalidConfig";
        case Status::Unsupported:
            return "Unsupported";
        case Status::InternalError:
            return "InternalError";
        case Status::BufferTooSmall:
            return "BufferTooSmall";
        case Status::EmptyMatrix:
            return "EmptyMatrix";
        case Status::KernelUnavailable:
            return "KernelUnavailable";
    }
    return "UnknownStatus";
}

}  // namespace apex::gemm
