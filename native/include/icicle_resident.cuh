#pragma once
#include "dynaverse_gpu.h"
#include "icicle_ntt_owner.cuh"
#include "icicle_msm_allocations.cuh"
struct DvgpuResidentState;
DvgpuResidentState* dvgpu_resident_create(uint64_t limit, uint64_t journal,
    bn254::scalar_t root, DvgpuNttState* ntt);
void dvgpu_resident_destroy(DvgpuResidentState*, bool current) noexcept;
uint64_t dvgpu_resident_execute(DvgpuResidentState*, const DvgpuResidentOp&);
uint64_t dvgpu_resident_execute_profiled(DvgpuResidentState*,const DvgpuResidentOp&,DvgpuResidentProfile*);
void dvgpu_resident_publish(DvgpuResidentState*, bool seal) noexcept;
DvgpuResidentStats dvgpu_resident_get_stats(const DvgpuResidentState*) noexcept;
DvgpuResidentDiagnostics dvgpu_resident_get_diagnostics(const DvgpuResidentState*) noexcept;
// Constant-size whole-owner accounting also used by legacy host-I/O entries.
void dvgpu_resident_preflight_external(DvgpuResidentState*, uint64_t pending, uint64_t scratch);
void dvgpu_resident_set_external(DvgpuResidentState*, uint64_t retained);
DvgpuTemporaryAllocationStats* dvgpu_resident_temporary(DvgpuResidentState*);
extern "C" cudaError_t dvgpu_g1_msm_run(const bn254::scalar_t*, const bn254::affine_t*,
    int length, msm::MSMConfig*, bn254::projective_t*);
extern "C" cudaError_t dvgpu_g2_msm_run(const bn254::scalar_t*, const bn254::g2_affine_t*,
    int length, msm::MSMConfig*, bn254::g2_projective_t*);
