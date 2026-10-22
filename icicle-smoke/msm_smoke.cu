// Minimal ICICLE v2.8.0 BN254 smoke test: GPU MSM with mathematical identity check.
// points[i] = (i+1)*G, scalars[i] = (i+1); MSM must equal (sum_{i=1..N} i^2)*G.
#include "api/bn254.h"
#include "msm/msm.cuh"
#include <cstdio>

using namespace bn254;

int main() {
    const unsigned N = 1024;
    printf("ICICLE BN254 MSM smoke test (N=%u)\n", N);

    static scalar_t scalars[N];
    static affine_t points[N];
    for (unsigned i = 0; i < N; i++) {
        scalars[i] = scalar_t::from((uint32_t)(i + 1));
        points[i] = projective_t::to_affine(projective_t::generator() * scalar_t::from((uint32_t)(i + 1)));
    }

    scalar_t sum_sq = scalar_t::zero();
    for (unsigned i = 0; i < N; i++) {
        scalar_t s = scalar_t::from((uint32_t)(i + 1));
        sum_sq = sum_sq + s * s;
    }
    affine_t expected = projective_t::to_affine(projective_t::generator() * sum_sq);

    projective_t result;
    msm::MSMConfig cfg = msm::default_msm_config();
    cfg.is_async = false;
    cudaError_t err = bn254_msm_cuda(scalars, points, N, cfg, &result);
    if (err != cudaSuccess) {
        printf("MSM error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    affine_t got = projective_t::to_affine(result);

    if (got == expected) {
        printf("MSM PASS: GPU result == (sum i^2)*G\n");
        return 0;
    }
    printf("MSM FAIL: mismatch\n");
    return 1;
}
