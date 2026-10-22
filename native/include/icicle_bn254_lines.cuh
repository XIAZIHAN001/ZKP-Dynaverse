#pragma once
#include "icicle_bn254_lines_constants.cuh"

// Adapted from ark-ec 0.4.2 src/models/bn/g2.rs (MIT OR Apache-2.0).
// Locked source and both licenses: docs/bn254-lines-source-20261021/reference/.
// These are HOMOGENEOUS coordinates x=X/Z,y=Y/Z, not Jacobian coordinates.
namespace dvgpu_bn254_lines {
using dvgpu_bn254_tower::add;
using dvgpu_bn254_tower::subtract;
using dvgpu_bn254_tower::multiply;
using dvgpu_bn254_tower::square;
using dvgpu_bn254_tower::negate_canonical;

struct Point {Fq2 x,y,z;};
struct Line {Fq2 c0,c1,c2;};

__device__ inline Fq2 negate(const Fq2& a) {
    return {negate_canonical(a.real),negate_canonical(a.imaginary)};
}
__device__ inline Fq2 triple(const Fq2& a) {return add(add(a,a),a);}
__device__ inline Fq2 half(const Fq2& a) {
    const Fq factor=two_inverse();return {a.real*factor,a.imaginary*factor};
}
__device__ inline Line double_line(Point& r) {
    const Fq2 a=half(multiply(r.x,r.y));
    const Fq2 b=square(r.y),c=square(r.z);
    const Fq2 e=multiply(curve_b(),triple(c)),f=triple(e);
    const Fq2 g=half(add(b,f));
    const Fq2 h=subtract(square(add(r.y,r.z)),add(b,c));
    const Fq2 i=subtract(e,b),j=square(r.x);
    r={multiply(a,subtract(b,f)),subtract(square(g),triple(square(e))),multiply(b,h)};
    // BN254's D twist: (-h,3j,i), not the M-twist coefficient permutation.
    return {negate(h),triple(j),i};
}
__device__ inline Line add_line(Point& r,const Point& q) {
    const Fq2 theta=subtract(r.y,multiply(q.y,r.z));
    const Fq2 lambda=subtract(r.x,multiply(q.x,r.z));
    const Fq2 c=square(theta),d=square(lambda),e=multiply(lambda,d);
    const Fq2 f=multiply(r.z,c),g=multiply(r.x,d);
    const Fq2 h=subtract(add(e,f),add(g,g));
    const Point next{multiply(lambda,h),
        subtract(multiply(theta,subtract(g,h)),multiply(e,r.y)),multiply(r.z,e)};
    const Fq2 j=subtract(multiply(theta,q.x),multiply(lambda,q.y));
    r=next;
    return {lambda,negate(theta),j};
}
__device__ inline Point characteristic(const Point& q) {
    // This function takes affine q (z=1). Canonical conjugation keeps zeros
    // valid for the next image; fixed ICICLE Field::neg(0) is not canonical.
    return {multiply(dvgpu_bn254_tower::frobenius_once(q.x),q_x()),
        multiply(dvgpu_bn254_tower::frobenius_once(q.y),q_y()),Fq2::one()};
}
} // namespace dvgpu_bn254_lines
