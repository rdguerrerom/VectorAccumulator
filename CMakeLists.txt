cmake_minimum_required(VERSION 3.15)
project(VectorAccumulator VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include FetchContent for dependencies
include(FetchContent)

# Set FetchContent base directory
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps)

# Declare VCL2 dependency
FetchContent_Declare(
    vcl2
    GIT_REPOSITORY https://github.com/vectorclass/version2.git
    GIT_TAG master
)

# Make VCL2 available
FetchContent_MakeAvailable(vcl2)

# Create VCL2 interface library
FetchContent_GetProperties(vcl2)
if(vcl2_POPULATED)
    add_library(vcl2_headers INTERFACE)
    target_include_directories(vcl2_headers INTERFACE
        $<BUILD_INTERFACE:${vcl2_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/vcl2>
    )
endif()

# Find required packages
find_package(Threads REQUIRED)

# Platform-specific compiler flags
if(APPLE)
    add_definitions(-D_DARWIN_C_SOURCE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ARCH_FLAGS "-mcpu=apple-m1")
    else()
        set(ARCH_FLAGS "-march=native")
    endif()
else()
    set(ARCH_FLAGS "-march=native")
endif()

# Define the library
add_library(vector_accumulator
    lib/VectorAccumulator.cpp
)

# Set header include directory structure
target_include_directories(vector_accumulator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Link dependencies
target_link_libraries(vector_accumulator
    PUBLIC
        Threads::Threads
        vcl2_headers
)

# Enhanced compiler flags with platform-specific optimizations
target_compile_options(vector_accumulator
    PRIVATE
        $<$<CONFIG:Debug>:
            -g
            -O0
            -Wall
            -Wextra
            -Wpedantic
            -fno-omit-frame-pointer
        >
        $<$<CONFIG:Release>:
            -O3
            ${ARCH_FLAGS}
            -DNDEBUG
            -ffast-math
            -ftree-vectorize
            -fstrict-aliasing
        >
)

# Build examples
if(BUILD_EXAMPLES)
    add_executable(vector_accumulator_example
        bin/examples/main.cpp
    )
    target_link_libraries(vector_accumulator_example 
        PRIVATE 
            vector_accumulator
    )
    set_target_properties(vector_accumulator_example 
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples
    )
endif()

# Build tests
if(BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(vector_accumulator_tests
        bin/tests/TestVectorAccumulator.cpp
    )
    target_link_libraries(vector_accumulator_tests
        PRIVATE
            vector_accumulator
            GTest::gtest_main
    )
    set_target_properties(vector_accumulator_tests 
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )
    
    include(GoogleTest)
    gtest_discover_tests(vector_accumulator_tests)
endif()

# Build benchmarks
if(BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_MakeAvailable(benchmark)

    add_executable(vector_accumulator_benchmarks
        bin/benchmarks/PerformanceVectorAccumulator.cpp
    )
    target_link_libraries(vector_accumulator_benchmarks
        PRIVATE
            vector_accumulator
            benchmark::benchmark
            benchmark::benchmark_main
    )
    set_target_properties(vector_accumulator_benchmarks 
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
    )
endif()

# Installation
install(TARGETS vector_accumulator
    EXPORT vector-accumulator-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/VectorAccumulator
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

if(vcl2_POPULATED)
    install(DIRECTORY ${vcl2_SOURCE_DIR}/
        DESTINATION include/vcl2
        FILES_MATCHING PATTERN "*.h"
    )
endif()

install(EXPORT vector-accumulator-targets
    FILE VectorAccumulatorTargets.cmake
    NAMESPACE VectorAccumulator::
    DESTINATION lib/cmake/VectorAccumulator
)

# Generate and install package configuration files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/VectorAccumulatorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VectorAccumulatorConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/VectorAccumulatorConfig.cmake"
    INSTALL_DESTINATION lib/cmake/VectorAccumulator
)

install(FILES
    "${CMAKE_BINARY_DIR}/VectorAccumulatorConfig.cmake"
    "${CMAKE_BINARY_DIR}/VectorAccumulatorConfigVersion.cmake"
    DESTINATION lib/cmake/VectorAccumulator
)
