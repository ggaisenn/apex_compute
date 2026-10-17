cmake_minimum_required(VERSION 3.21)

project(apex_compute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(APEX_FORCE_SCALAR "Build only the portable scalar kernel" OFF)
option(APEX_ENABLE_SANITIZERS "Enable supported address and undefined behavior sanitizers" OFF)

if(MSVC)
    add_compile_options(/W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
else()
    message(FATAL_ERROR
        "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}'. Use MSVC, GCC, or Clang.")
endif()

if(APEX_FORCE_SCALAR)
    add_compile_definitions(APEX_FORCE_SCALAR=1)
endif()

if(APEX_ENABLE_SANITIZERS)
    if(MSVC)
        add_compile_options(/fsanitize=address)
        add_link_options(/fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    else()
        message(FATAL_ERROR
            "Sanitizers are not configured for '${CMAKE_CXX_COMPILER_ID}'.")
    endif()
endif()

# The core will use only the C++ standard library; keep third-party runtime
# dependencies out of the project targets.

add_executable(apex-validate
    apex-validate/validate_engine.cpp
    kernels/avx512/gemm_avx512_kernel.cpp
    core/orchestration/gemm_engine.cpp
    kernels/neon/gemm_neon_kernel.cpp
    kernels/packing.cpp)
target_include_directories(apex-validate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT APEX_FORCE_SCALAR AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    if(MSVC)
        set_source_files_properties(
            kernels/avx512/gemm_avx512_kernel.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set_source_files_properties(
            kernels/avx512/gemm_avx512_kernel.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f")
    endif()
endif()

add_library(apex-bench-engine STATIC apex-bench/bench_engine.cpp)
target_include_directories(apex-bench-engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
