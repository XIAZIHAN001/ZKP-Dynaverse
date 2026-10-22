#pragma once
#include "icicle_bn254_lines.cuh"
#include "dynaverse_bn254_miller.h"

// D-twist line embedding and Miller schedule adapted from ark-ec 0.4.2
// src/models/bn/mod.rs (MIT OR Apache-2.0); source/licenses in this phase's
// docs/reference. General dense multiplication deliberately reuses the
// accepted tower arithmetic without cyclotomic or sparse-only assumptions.
namespace dvgpu_bn254_miller {
using namespace dvgpu_bn254_tower;
__device__ inline Fq12 evaluate(const dvgpu_bn254_lines::Line& line,const Fq& px,const Fq& py) {
    const Fq2 a{line.c0.real*py,line.c0.imaginary*py};
    const Fq2 b{line.c1.real*px,line.c1.imaginary*px};
    // arkworks D twist mul_by_034: positions 0,3,4 in the six-Fq2 layout.
    return {{a,Fq2::zero(),Fq2::zero()},{b,line.c2,Fq2::zero()}};
}
static_assert(DVGPU_MILLER_LINES==DVGPU_LINE_COUNT,"Miller line schedule changed");
} // namespace dvgpu_bn254_miller
