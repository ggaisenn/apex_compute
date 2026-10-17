#pragma once

#include <chrono>
#include <cstdint>
#include <type_traits>

namespace apex::gemm {

class Timer {
public:
    using Clock = std::conditional_t<
        std::chrono::high_resolution_clock::is_steady,
        std::chrono::high_resolution_clock,
        std::chrono::steady_clock>;
    using TimePoint = Clock::time_point;
    using Nanoseconds = std::chrono::nanoseconds;

    Timer() noexcept : start_time_(Clock::now()) {}

    void reset() noexcept { start_time_ = Clock::now(); }

    [[nodiscard]] Nanoseconds elapsed() const noexcept {
        return elapsed_between(start_time_, Clock::now());
    }

    [[nodiscard]] std::uint64_t elapsed_nanoseconds() const noexcept {
        const auto count = elapsed().count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    [[nodiscard]] static TimePoint now() noexcept { return Clock::now(); }

    [[nodiscard]] static Nanoseconds elapsed_between(
        TimePoint start,
        TimePoint end) noexcept {
        const Nanoseconds duration =
            std::chrono::duration_cast<Nanoseconds>(end - start);
        return duration.count() > 0 ? duration : Nanoseconds::zero();
    }

    [[nodiscard]] static std::uint64_t resolution_nanoseconds() noexcept {
        const Nanoseconds resolution =
            std::chrono::duration_cast<Nanoseconds>(Clock::duration{1});
        return resolution.count() > 0
                   ? static_cast<std::uint64_t>(resolution.count())
                   : 1U;
    }

    [[nodiscard]] static constexpr bool is_monotonic() noexcept {
        return Clock::is_steady;
    }

private:
    TimePoint start_time_;
};

}  // namespace apex::gemm
