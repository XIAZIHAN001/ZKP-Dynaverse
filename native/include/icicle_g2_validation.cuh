#pragma once
#include "dynaverse_gpu.h"
#include "dynaverse_g2_experiments.h"
#include <cuda_runtime.h>

// Device buffers are borrowed in the owner's context and ordered on its stream.
cudaError_t dvgpu_g2_validate_device(const uint8_t* wire,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,cudaStream_t stream);
cudaError_t dvgpu_g2_validate_device_events(const uint8_t* wire,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,cudaStream_t stream,cudaEvent_t parse_done,cudaEvent_t subgroup_done);
cudaError_t dvgpu_g2_corrupt_test_point(uint8_t* wire,size_t index,cudaStream_t stream);
cudaError_t dvgpu_g2_validate_configured(const uint8_t* wire,uint8_t* layout_scratch,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,cudaStream_t stream,const DvgpuG2KernelConfig& config,
    bool prechecked,cudaEvent_t parse_done,cudaEvent_t repack_done,cudaEvent_t subgroup_done);
cudaError_t dvgpu_g2_resources_run(uint32_t algorithm,const DvgpuG2KernelConfig& config,DvgpuG2KernelResources& output);
// Standalone host probe, using the same kernels. Throws only inside C++.
void dvgpu_g2_inspect_run(const uint8_t* wire,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,DvgpuG2ValidationStats& stats,
    cudaStream_t stream,const DvgpuG2KernelConfig* config=nullptr,DvgpuG2ExperimentStats* experiment=nullptr);
// Diagnostic only: n<=129 homogeneous192 inputs, three images per point in
// homogeneous192 output (576*n bytes). Never used for production validation.
void dvgpu_g2_psi_probe_run(const uint8_t* points,size_t n,uint8_t* output,cudaStream_t stream);
