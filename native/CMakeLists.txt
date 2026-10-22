cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES explicitly (120 local, 90 H200)")
endif()
project(dynaverse_native LANGUAGES CXX CUDA)
set(ICICLE_INCLUDE_DIR "/opt/src/icicle-src/icicle/include" CACHE PATH "ICICLE v2.8.0 headers")
if(NOT EXISTS "${ICICLE_INCLUDE_DIR}/api/bn254.h")
    message(FATAL_ERROR "Missing pinned ICICLE headers: ${ICICLE_INCLUDE_DIR}")
endif()
# Compile the same pinned MSM source with host allocation bookkeeping. This
# narrowly scoped resource repair changes neither kernels nor their arithmetic.
# Refuse a different upstream revision instead of silently rewriting new code.
set(ICICLE_MSM_SOURCE "${ICICLE_INCLUDE_DIR}/../src/msm/msm.cu")
file(SHA256 "${ICICLE_MSM_SOURCE}" msm_hash)
if(NOT msm_hash STREQUAL "2136687e6d886d1d91f939db9626eb7471945c28d0b54baf3e5bad5e3087948d")
    message(FATAL_ERROR "ICICLE MSM source differs from pinned v2.8.0; review the allocation repair")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ICICLE_MSM_SOURCE}")
file(READ "${ICICLE_MSM_SOURCE}" msm_source)
string(REPLACE "CHK_INIT_IF_RETURN();" "CHK_INIT_IF_RETURN();\n    DynaverseMsmAllocations resources;" msm_source "${msm_source}")
string(REPLACE "DynaverseMsmAllocations resources;" "DynaverseMsmResources side_resources;\n    DynaverseMsmAllocations resources;" msm_source "${msm_source}")
string(REPLACE "cudaMallocAsync(" "resources.allocate(" msm_source "${msm_source}")
string(REPLACE "cudaFreeAsync(" "resources.release(" msm_source "${msm_source}")
# RunLengthEncode writes only the actual run count, while the pinned exclusive
# scan consumes total_nof_buckets + 1 entries. Define its unused tail before
# encoding. This is host-side initialization; all arithmetic kernels stay intact.
string(REPLACE
    "CHK_IF_RETURN(resources.allocate(&bucket_sizes, sizeof(unsigned) * (total_nof_buckets + 1), stream));"
    "CHK_IF_RETURN(resources.allocate(&bucket_sizes, sizeof(unsigned) * (total_nof_buckets + 1), stream));\n      CHK_IF_RETURN(cudaMemsetAsync(bucket_sizes, 0, sizeof(unsigned) * (total_nof_buckets + 1), stream));"
    msm_source "${msm_source}")
string(REPLACE "cudaEventCreateWithFlags(" "side_resources.create_event(" msm_source "${msm_source}")
string(REPLACE "cudaEventDestroy(" "side_resources.destroy_event(" msm_source "${msm_source}")
string(REPLACE "cudaStreamCreate(" "side_resources.create_stream(" msm_source "${msm_source}")
string(REPLACE "cudaStreamDestroy(" "side_resources.destroy_stream(" msm_source "${msm_source}")
# Each pinned CHK_LAST return has a CHK_INIT owner. Cleanup failures must
# become CUDA errors instead of being swallowed by a noexcept destructor.
string(REPLACE "return CHK_LAST();" "CHK_IF_RETURN(side_resources.finish());\n      return CHK_LAST();" msm_source "${msm_source}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/icicle_msm_with_cleanup.cuh" "${msm_source}")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_msm_timeline.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_msm_timeline.py"
    "${CMAKE_CURRENT_BINARY_DIR}/icicle_msm_with_cleanup.cuh"
    "${CMAKE_CURRENT_BINARY_DIR}/icicle_msm_with_timeline.cuh"
    COMMAND_ERROR_IS_FATAL ANY)
set(ICICLE_NTT_SOURCE "${ICICLE_INCLUDE_DIR}/../src/ntt/ntt.cu")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${ICICLE_NTT_SOURCE}" "${ICICLE_INCLUDE_DIR}/../src/ntt/kernel_ntt.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_ntt_owner.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_ntt_owner.py"
    "${ICICLE_NTT_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/icicle_ntt_with_owner.cuh"
    COMMAND_ERROR_IS_FATAL ANY)
add_library(dynaverse_icicle STATIC src/icicle_msm.cu src/icicle_msm_g2.cu src/icicle_ntt.cu src/icicle_resident.cu src/icicle_g2_validation.cu src/icicle_bn254_tower.cu src/icicle_bn254_lines.cu src/icicle_bn254_miller.cu)
set_source_files_properties(src/icicle_g2_validation.cu PROPERTIES COMPILE_OPTIONS "--ptxas-options=-v")
set_source_files_properties(src/icicle_ntt.cu PROPERTIES COMPILE_DEFINITIONS "FIELD_ID=1;CURVE_ID=1")
target_include_directories(dynaverse_icicle PRIVATE include "${ICICLE_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(dynaverse_icicle PROPERTIES CUDA_STANDARD 17)
option(DYNAVERSE_BUILD_TEST_FIXTURES "Build CUDA allocation-failure LD_PRELOAD fixture" OFF)
if(DYNAVERSE_BUILD_TEST_FIXTURES)
    add_library(fail_cuda_alloc SHARED tests/fail_cuda_alloc.cpp)
    target_include_directories(fail_cuda_alloc PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(fail_cuda_alloc PRIVATE dl)
endif()
