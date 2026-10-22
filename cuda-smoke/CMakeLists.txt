cmake_minimum_required(VERSION 3.24)
project(dynaverse_cuda_smoke LANGUAGES CXX CUDA)

# CUDA_ARCH comes from env/local.env or env/h200.env via CMAKE_CUDA_ARCHITECTURES.
# If not provided, fail loudly rather than guessing.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set (source env/local.env or env/h200.env)")
endif()
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_STANDARD 17)
add_executable(vector_add vector_add.cu)
