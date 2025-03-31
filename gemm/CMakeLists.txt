cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_gemm CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set build type if not set
# Set optimization flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O3")
else()
    string(APPEND CMAKE_CUDA_FLAGS " -G")
    string(APPEND CMAKE_CUDA_FLAGS " -g")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /MDd")
endif()

# Set default CUDA architecture if not specified
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code=arch=compute_86,code=[compute_86,sm_86]")

# Create executable
add_executable(gemm_test
    test.cu
    sgemm_naive.cuh
    sgemm_shared.cuh
    sgemm_1D_blocktiling.cuh
    sgemm_2D_blocktiling.cuh
    sgemm_warptilling.cuh
    utility.cuh
)

# Link CUDA libraries
target_link_libraries(gemm_test
    PRIVATE
    cublas
)

# Set CUDA specific compiler options
set_target_properties(gemm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
