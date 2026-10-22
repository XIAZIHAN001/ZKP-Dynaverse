cmake_minimum_required(VERSION 3.24)
project(dynaverse_icicle_smoke LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set from env/local.env or env/h200.env")
endif()
message(STATUS "Building icicle-smoke for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_STANDARD 17)

set(ICICLE_INCLUDE_DIR "/opt/src/icicle-src/icicle/include" CACHE PATH "ICICLE v2 include directory")
set(ICICLE_LIB_DIR "/opt/icicle/arch${CMAKE_CUDA_ARCHITECTURES}" CACHE PATH "ICICLE static libraries for target architecture")

if(NOT EXISTS "${ICICLE_INCLUDE_DIR}")
    message(FATAL_ERROR "ICICLE include dir not found: ${ICICLE_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ICICLE_LIB_DIR}/libingo_curve_bn254.a")
    message(FATAL_ERROR "ICICLE static libs not found at: ${ICICLE_LIB_DIR}")
endif()

add_executable(roundtrip_msm roundtrip_msm.cu)
target_include_directories(roundtrip_msm PRIVATE "${ICICLE_INCLUDE_DIR}")
target_link_libraries(roundtrip_msm PRIVATE
    "${ICICLE_LIB_DIR}/libingo_curve_bn254.a"
    "${ICICLE_LIB_DIR}/libingo_field_bn254.a"
    cuda
    cudart
)
