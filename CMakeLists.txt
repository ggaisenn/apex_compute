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
    core/orchestration/gemm_engine.cpp
    kernels/packing.cpp)
target_include_directories(apex-validate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(apex-bench-engine STATIC apex-bench/bench_engine.cpp)
target_include_directories(apex-bench-engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
