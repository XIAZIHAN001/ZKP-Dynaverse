#include "icicle_resident.cuh"
#include "icicle_msm_allocations.cuh"
#include "icicle_msm_resources.cuh"
#include "icicle_msm_with_cleanup.cuh"

// One pinned template instantiation serves both borrowed-host foreground MSM
// and resident-scalar rebuild blocks. The resource repair is shared with G1.
extern "C" cudaError_t dvgpu_g2_msm_run(const bn254::scalar_t* scalars,
    const bn254::g2_affine_t* points, int length, msm::MSMConfig* config,
    bn254::g2_projective_t* output) {
    return msm::msm<bn254::scalar_t, bn254::g2_affine_t, bn254::g2_projective_t>(
        scalars, points, length, *config, output);
}
