#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER) && \
    (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#if defined(__arm__) && !defined(__aarch64__)
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif
#endif

namespace apex::gemm {

enum class FeatureValueSource {
    detected,
    fallback,
};

struct CacheSize {
    std::size_t bytes;
    FeatureValueSource source;
};

class CpuFeatures {
public:
    // Retain this reference after setup so kernel checks are plain cached reads.
    [[nodiscard]] static const CpuFeatures& current() noexcept {
        static const CpuFeatures features;
        return features;
    }

    [[nodiscard]] bool neon() const noexcept { return neon_; }
    [[nodiscard]] bool avx2() const noexcept { return avx2_; }
    [[nodiscard]] bool avx512f() const noexcept { return avx512f_; }
    [[nodiscard]] bool fma() const noexcept { return fma_; }
    [[nodiscard]] CacheSize l1d_cache() const noexcept { return l1d_cache_; }
    [[nodiscard]] CacheSize l2_cache() const noexcept { return l2_cache_; }
    [[nodiscard]] CacheSize l3_cache() const noexcept { return l3_cache_; }
    [[nodiscard]] CacheSize cache_line() const noexcept { return cache_line_; }

private:
    CpuFeatures() noexcept {
        detect_instruction_sets();
        detect_cache_geometry();
    }

    static constexpr std::size_t kDefaultL1dBytes = 32U * 1024U;
    static constexpr std::size_t kDefaultL2Bytes = 256U * 1024U;
    static constexpr std::size_t kDefaultL3Bytes = 8U * 1024U * 1024U;
    static constexpr std::size_t kDefaultCacheLineBytes = 64U;

    static void set_detected(CacheSize* value, std::uint64_t bytes) noexcept {
        if (value != nullptr && bytes != 0 &&
            bytes <= static_cast<std::uint64_t>(static_cast<std::size_t>(-1))) {
            value->bytes = static_cast<std::size_t>(bytes);
            value->source = FeatureValueSource::detected;
        }
    }

    void detect_instruction_sets() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
        neon_ = true;
#elif defined(__arm__) && defined(__linux__) && defined(HWCAP_NEON)
        neon_ = (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int registers[4] = {};
        __cpuid(registers, 0);
        const int maximum_leaf = registers[0];
        if (maximum_leaf < 1) {
            return;
        }

        __cpuidex(registers, 1, 0);
        const unsigned int ecx = static_cast<unsigned int>(registers[2]);
        const bool cpu_fma = (ecx & (1U << 12U)) != 0;
        const bool osxsave = (ecx & (1U << 27U)) != 0;
        const bool cpu_avx = (ecx & (1U << 28U)) != 0;
        if (!osxsave || !cpu_avx) {
            return;
        }

        const unsigned __int64 xcr0 = _xgetbv(0);
        const bool ymm_state = (xcr0 & 0x6U) == 0x6U;
        fma_ = cpu_fma && ymm_state;
        if (!ymm_state || maximum_leaf < 7) {
            return;
        }

        __cpuidex(registers, 7, 0);
        const unsigned int ebx = static_cast<unsigned int>(registers[1]);
        avx2_ = (ebx & (1U << 5U)) != 0;
        const bool cpu_avx512f = (ebx & (1U << 16U)) != 0;
        const unsigned __int64 avx512_state_mask = 0xE6U;
        avx512f_ = cpu_avx512f &&
                   avx2_ && fma_ &&
                   (xcr0 & avx512_state_mask) == avx512_state_mask;
#elif defined(__i386__) || defined(__x86_64__)
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        const unsigned int maximum_leaf = __get_cpuid_max(0, nullptr);
        if (maximum_leaf < 1 ||
            !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            return;
        }

        const bool cpu_fma = (ecx & bit_FMA) != 0;
        const bool osxsave = (ecx & bit_OSXSAVE) != 0;
        const bool cpu_avx = (ecx & bit_AVX) != 0;
        if (!osxsave || !cpu_avx) {
            return;
        }

        std::uint32_t xcr0_low = 0;
        std::uint32_t xcr0_high = 0;
        __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
        const std::uint64_t xcr0 =
            (static_cast<std::uint64_t>(xcr0_high) << 32U) | xcr0_low;
        const bool ymm_state = (xcr0 & 0x6U) == 0x6U;
        fma_ = cpu_fma && ymm_state;
        if (!ymm_state || maximum_leaf < 7 ||
            !__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return;
        }

        avx2_ = (ebx & bit_AVX2) != 0;
        const bool cpu_avx512f = (ebx & (1U << 16U)) != 0;
        constexpr std::uint64_t avx512_state_mask = 0xE6U;
        avx512f_ = cpu_avx512f &&
                   avx2_ && fma_ &&
                   (xcr0 & avx512_state_mask) == avx512_state_mask;
#endif
    }

    void detect_cache_geometry() noexcept {
        l1d_cache_ = {kDefaultL1dBytes, FeatureValueSource::fallback};
        l2_cache_ = {kDefaultL2Bytes, FeatureValueSource::fallback};
        l3_cache_ = {kDefaultL3Bytes, FeatureValueSource::fallback};
        cache_line_ = {kDefaultCacheLineBytes, FeatureValueSource::fallback};

#if defined(__APPLE__)
        set_detected(&l1d_cache_, query_sysctl_size("hw.l1dcachesize"));
        set_detected(&l2_cache_, query_sysctl_size("hw.l2cachesize"));
        set_detected(&l3_cache_, query_sysctl_size("hw.l3cachesize"));
        set_detected(&cache_line_, query_sysctl_size("hw.cachelinesize"));
#elif defined(__linux__)
#ifdef _SC_LEVEL1_DCACHE_SIZE
        set_detected(&l1d_cache_, query_sysconf(_SC_LEVEL1_DCACHE_SIZE));
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
        set_detected(&l2_cache_, query_sysconf(_SC_LEVEL2_CACHE_SIZE));
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
        set_detected(&l3_cache_, query_sysconf(_SC_LEVEL3_CACHE_SIZE));
#endif
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
        set_detected(&cache_line_, query_sysconf(_SC_LEVEL1_DCACHE_LINESIZE));
#endif
        detect_linux_sysfs_caches();
#endif

        detect_x86_cpuid_caches();
    }

#if defined(__APPLE__)
    [[nodiscard]] static std::uint64_t query_sysctl_size(
        const char* name) noexcept {
        std::uint64_t value = 0;
        std::size_t value_size = sizeof(value);
        if (sysctlbyname(name, &value, &value_size, nullptr, 0) != 0) {
            return 0;
        }
        if (value_size == sizeof(std::uint32_t)) {
            return static_cast<std::uint32_t>(value);
        }
        return value_size == sizeof(value) ? value : 0;
    }
#endif

#if defined(__linux__)
    [[nodiscard]] static std::uint64_t query_sysconf(int name) noexcept {
        const long value = sysconf(name);
        return value > 0 ? static_cast<std::uint64_t>(value) : 0;
    }

    [[nodiscard]] static std::uint64_t parse_cache_size(
        const char* text) noexcept {
        if (text == nullptr || text[0] == '\0') {
            return 0;
        }
        std::uint64_t number = 0;
        std::size_t index = 0;
        for (; text[index] != '\0'; ++index) {
            const char character = text[index];
            if (character < '0' || character > '9') {
                break;
            }
            const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
            if (number > (static_cast<std::uint64_t>(-1) - digit) / 10U) {
                return 0;
            }
            number = number * 10U + digit;
        }
        if (index == 0) {
            return 0;
        }

        std::uint64_t multiplier = 1;
        if (text[index] != '\0') {
            const char suffix = text[index];
            if (suffix == 'K' || suffix == 'k') {
                multiplier = 1024U;
            } else if (suffix == 'M' || suffix == 'm') {
                multiplier = 1024U * 1024U;
            } else if (suffix == 'G' || suffix == 'g') {
                multiplier = 1024U * 1024U * 1024U;
            }
        }
        return number <= static_cast<std::uint64_t>(-1) / multiplier
                   ? number * multiplier
                   : 0;
    }

    void detect_linux_sysfs_caches() noexcept {
        for (unsigned int index = 0; index < 16; ++index) {
            char base[128] = {};
            char path[160] = {};
            if (std::snprintf(
                    base, sizeof(base),
                    "/sys/devices/system/cpu/cpu0/cache/index%u/", index) < 0) {
                continue;
            }

            unsigned int level = 0;
            char type[32] = {};
            char size_text[32] = {};
            std::snprintf(path, sizeof(path), "%slevel", base);
            std::FILE* file = std::fopen(path, "r");
            if (file == nullptr || std::fscanf(file, "%u", &level) != 1) {
                if (file != nullptr) {
                    std::fclose(file);
                }
                continue;
            }
            std::fclose(file);

            std::snprintf(path, sizeof(path), "%stype", base);
            file = std::fopen(path, "r");
            if (file == nullptr || std::fscanf(file, "%31s", type) != 1) {
                if (file != nullptr) {
                    std::fclose(file);
                }
                continue;
            }
            std::fclose(file);

            std::snprintf(path, sizeof(path), "%ssize", base);
            file = std::fopen(path, "r");
            if (file == nullptr || std::fscanf(file, "%31s", size_text) != 1) {
                if (file != nullptr) {
                    std::fclose(file);
                }
                continue;
            }
            std::fclose(file);

            CacheSize* target = nullptr;
            if (level == 1 &&
                (std::strcmp(type, "Data") == 0 ||
                 std::strcmp(type, "Unified") == 0)) {
                target = &l1d_cache_;
            } else if (level == 2 &&
                       (std::strcmp(type, "Data") == 0 ||
                        std::strcmp(type, "Unified") == 0)) {
                target = &l2_cache_;
            } else if (level == 3 &&
                       (std::strcmp(type, "Unified") == 0 ||
                        std::strcmp(type, "Data") == 0)) {
                target = &l3_cache_;
            }
            if (target != nullptr && target->source == FeatureValueSource::fallback) {
                set_detected(target, parse_cache_size(size_text));
            }
            if (level == 1 && target != nullptr) {
                std::snprintf(path, sizeof(path), "%scoherency_line_size", base);
                file = std::fopen(path, "r");
                unsigned long long line_size = 0;
                if (file != nullptr && std::fscanf(file, "%llu", &line_size) == 1 &&
                    cache_line_.source == FeatureValueSource::fallback) {
                    set_detected(&cache_line_, static_cast<std::uint64_t>(line_size));
                }
                if (file != nullptr) {
                    std::fclose(file);
                }
            }
        }
    }
#endif

    void detect_x86_cpuid_caches() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int registers[4] = {};
        __cpuid(registers, 0);
        if (registers[0] < 4) {
            return;
        }
        for (int subleaf = 0; subleaf < 16; ++subleaf) {
            __cpuidex(registers, 4, subleaf);
            const unsigned int eax = static_cast<unsigned int>(registers[0]);
            const unsigned int ebx = static_cast<unsigned int>(registers[1]);
            const unsigned int ecx = static_cast<unsigned int>(registers[2]);
            const unsigned int cache_type = eax & 0x1FU;
            if (cache_type == 0) {
                break;
            }
            const std::uint64_t line_size = (ebx & 0xFFFU) + 1U;
            const std::uint64_t partitions = ((ebx >> 12U) & 0x3FFU) + 1U;
            const std::uint64_t ways = ((ebx >> 22U) & 0x3FFU) + 1U;
            const std::uint64_t sets = static_cast<std::uint64_t>(ecx) + 1U;
            const std::uint64_t size = cache_size_bytes(line_size, partitions, ways, sets);
            const unsigned int level = (eax >> 5U) & 0x7U;
            assign_x86_cache(level, cache_type, size, line_size);
        }
#elif defined(__i386__) || defined(__x86_64__)
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        if (__get_cpuid_max(0, nullptr) < 4) {
            return;
        }
        for (unsigned int subleaf = 0; subleaf < 16; ++subleaf) {
            if (!__get_cpuid_count(4, subleaf, &eax, &ebx, &ecx, &edx)) {
                break;
            }
            const unsigned int cache_type = eax & 0x1FU;
            if (cache_type == 0) {
                break;
            }
            const std::uint64_t line_size = (ebx & 0xFFFU) + 1U;
            const std::uint64_t partitions = ((ebx >> 12U) & 0x3FFU) + 1U;
            const std::uint64_t ways = ((ebx >> 22U) & 0x3FFU) + 1U;
            const std::uint64_t sets = static_cast<std::uint64_t>(ecx) + 1U;
            const std::uint64_t size = cache_size_bytes(line_size, partitions, ways, sets);
            const unsigned int level = (eax >> 5U) & 0x7U;
            assign_x86_cache(level, cache_type, size, line_size);
        }
#endif
    }

    [[nodiscard]] static std::uint64_t cache_size_bytes(
        std::uint64_t line_size,
        std::uint64_t partitions,
        std::uint64_t ways,
        std::uint64_t sets) noexcept {
        constexpr std::uint64_t maximum = static_cast<std::uint64_t>(-1);
        if (partitions != 0 && line_size > maximum / partitions) {
            return 0;
        }
        std::uint64_t size = line_size * partitions;
        if (ways != 0 && size > maximum / ways) {
            return 0;
        }
        size *= ways;
        if (sets != 0 && size > maximum / sets) {
            return 0;
        }
        return size * sets;
    }

    void assign_x86_cache(
        unsigned int level,
        unsigned int cache_type,
        std::uint64_t size,
        std::uint64_t line_size) noexcept {
        if (level == 1 && cache_type == 1) {
            if (l1d_cache_.source == FeatureValueSource::fallback) {
                set_detected(&l1d_cache_, size);
            }
            if (cache_line_.source == FeatureValueSource::fallback) {
                set_detected(&cache_line_, line_size);
            }
        } else if (level == 2 && (cache_type == 1 || cache_type == 3)) {
            if (l2_cache_.source == FeatureValueSource::fallback) {
                set_detected(&l2_cache_, size);
            }
        } else if (level == 3 && (cache_type == 1 || cache_type == 3)) {
            if (l3_cache_.source == FeatureValueSource::fallback) {
                set_detected(&l3_cache_, size);
            }
        }
    }

    bool neon_ = false;
    bool avx2_ = false;
    bool avx512f_ = false;
    bool fma_ = false;
    CacheSize l1d_cache_{kDefaultL1dBytes, FeatureValueSource::fallback};
    CacheSize l2_cache_{kDefaultL2Bytes, FeatureValueSource::fallback};
    CacheSize l3_cache_{kDefaultL3Bytes, FeatureValueSource::fallback};
    CacheSize cache_line_{kDefaultCacheLineBytes, FeatureValueSource::fallback};
};

}  // namespace apex::gemm
