#pragma once
#include "icicle_bn254_tower_frobenius.cuh"

// General BN254 extension arithmetic. This is not a Miller loop or a pairing
// implementation. Fq/Fq2 operations reuse the pinned ICICLE implementation.
namespace dvgpu_bn254_tower {

static_assert(sizeof(Fq) == 32 && sizeof(Fq2) == 64, "BN254 field layout changed");
static_assert(bn254::fq_config::nonresidue == 1 && bn254::fq_config::nonresidue_is_negative,
    "Fq2 must use u^2 = -1");
static_assert(bn254::fq_config::modulus.limbs[0] == 0xd87cfd47u &&
    bn254::fq_config::modulus.limbs[1] == 0x3c208c16u &&
    bn254::fq_config::modulus.limbs[2] == 0x6871ca8du &&
    bn254::fq_config::modulus.limbs[3] == 0x97816a91u &&
    bn254::fq_config::modulus.limbs[4] == 0x8181585du &&
    bn254::fq_config::modulus.limbs[5] == 0xb85045b6u &&
    bn254::fq_config::modulus.limbs[6] == 0xe131a029u &&
    bn254::fq_config::modulus.limbs[7] == 0x30644e72u,
    "Frobenius constants require the pinned BN254 base modulus");

struct Fq6 {
    Fq2 c0, c1, c2;
    static __device__ Fq6 zero() { return {Fq2::zero(), Fq2::zero(), Fq2::zero()}; }
    static __device__ Fq6 one() { return {Fq2::one(), Fq2::zero(), Fq2::zero()}; }
};
struct Fq12 {
    Fq6 c0, c1;
    static __device__ Fq12 zero() { return {Fq6::zero(), Fq6::zero()}; }
    static __device__ Fq12 one() { return {Fq6::one(), Fq6::zero()}; }
};
static_assert(sizeof(Fq6) == 192 && sizeof(Fq12) == 384, "BN254 tower layout changed");

__device__ inline Fq negate_canonical(const Fq& value) {
    // Pinned Field::neg(0) is the raw modulus, which is not canonical input.
    return value == Fq::zero() ? Fq::zero() : Fq::neg(value);
}
__device__ inline bool is_zero(const Fq2& a) { return a == Fq2::zero(); }
__device__ inline Fq2 add(const Fq2& a, const Fq2& b) { return a + b; }
__device__ inline Fq2 subtract(const Fq2& a, const Fq2& b) { return a - b; }
__device__ inline Fq2 multiply(const Fq2& a, const Fq2& b) { return a * b; }
__device__ inline Fq2 square(const Fq2& a) { return Fq2::sqr(a); }
__device__ inline Fq2 inverse(const Fq2& a) {
    // Caller rejects zero. Norm is a0^2+a1^2 for u^2=-1.
    const Fq denominator = Fq::inverse(Fq::sqr(a.real) + Fq::sqr(a.imaginary));
    return {a.real * denominator, negate_canonical(a.imaginary) * denominator};
}
__device__ inline Fq2 multiply_xi(const Fq2& a) {
    const Fq nine{storage<8>{9,0,0,0,0,0,0,0}};
    return a * Fq2{nine, Fq::one()};
}
__device__ inline Fq2 frobenius_once(const Fq2& a) {
    return {a.real, negate_canonical(a.imaginary)};
}

__device__ inline bool is_zero(const Fq6& a) {
    return is_zero(a.c0) && is_zero(a.c1) && is_zero(a.c2);
}
__device__ inline Fq6 add(const Fq6& a, const Fq6& b) {
    return {add(a.c0,b.c0), add(a.c1,b.c1), add(a.c2,b.c2)};
}
__device__ inline Fq6 subtract(const Fq6& a, const Fq6& b) {
    return {subtract(a.c0,b.c0), subtract(a.c1,b.c1), subtract(a.c2,b.c2)};
}
__device__ inline Fq6 multiply_v(const Fq6& a) {
    // v^3=xi=9+u, so multiplication by v rotates the three coefficients.
    return {multiply_xi(a.c2), a.c0, a.c1};
}
__device__ inline Fq6 multiply(const Fq6& a, const Fq6& b) {
    const Fq2 t0 = multiply(a.c0,b.c0);
    const Fq2 t1 = multiply(a.c1,b.c1);
    const Fq2 t2 = multiply(a.c2,b.c2);
    const Fq2 cross12 = subtract(subtract(multiply(add(a.c1,a.c2),add(b.c1,b.c2)),t1),t2);
    const Fq2 cross01 = subtract(subtract(multiply(add(a.c0,a.c1),add(b.c0,b.c1)),t0),t1);
    const Fq2 cross02 = subtract(subtract(multiply(add(a.c0,a.c2),add(b.c0,b.c2)),t0),t2);
    return {add(t0,multiply_xi(cross12)), add(cross01,multiply_xi(t2)), add(cross02,t1)};
}
__device__ inline Fq6 square(const Fq6& a) {
    // General multiplication, without a cyclotomic-subgroup assumption.
    return multiply(a,a);
}
__device__ inline Fq6 inverse(const Fq6& a) {
    const Fq2 t0 = subtract(square(a.c0),multiply_xi(multiply(a.c1,a.c2)));
    const Fq2 t1 = subtract(multiply_xi(square(a.c2)),multiply(a.c0,a.c1));
    const Fq2 t2 = subtract(square(a.c1),multiply(a.c0,a.c2));
    const Fq2 denominator = inverse(add(multiply(a.c0,t0),
        multiply_xi(add(multiply(a.c2,t1),multiply(a.c1,t2)))));
    return {multiply(t0,denominator), multiply(t1,denominator), multiply(t2,denominator)};
}
__device__ inline Fq6 frobenius_once(const Fq6& a) {
    return {frobenius_once(a.c0),
        multiply(frobenius_once(a.c1),frobenius_fq6_c1()),
        multiply(frobenius_once(a.c2),frobenius_fq6_c2())};
}

__device__ inline bool is_zero(const Fq12& a) { return is_zero(a.c0) && is_zero(a.c1); }
__device__ inline Fq12 add(const Fq12& a, const Fq12& b) {
    return {add(a.c0,b.c0), add(a.c1,b.c1)};
}
__device__ inline Fq12 subtract(const Fq12& a, const Fq12& b) {
    return {subtract(a.c0,b.c0), subtract(a.c1,b.c1)};
}
__device__ inline Fq12 multiply(const Fq12& a, const Fq12& b) {
    const Fq6 t0 = multiply(a.c0,b.c0);
    const Fq6 t1 = multiply(a.c1,b.c1);
    return {add(t0,multiply_v(t1)),
        subtract(subtract(multiply(add(a.c0,a.c1),add(b.c0,b.c1)),t0),t1)};
}
__device__ inline Fq12 square(const Fq12& a) { return multiply(a,a); }
__device__ inline Fq12 inverse(const Fq12& a) {
    const Fq6 denominator = inverse(subtract(square(a.c0),multiply_v(square(a.c1))));
    return {multiply(a.c0,denominator),
        subtract(Fq6::zero(),multiply(a.c1,denominator))};
}
__device__ inline Fq12 frobenius_once(const Fq12& a) {
    const Fq6 high = frobenius_once(a.c1);
    const Fq2 coefficient = frobenius_fq12_c1();
    return {frobenius_once(a.c0),
        {multiply(high.c0,coefficient), multiply(high.c1,coefficient), multiply(high.c2,coefficient)}};
}
template<typename T>
__device__ inline T frobenius(T value, uint32_t power) {
    // The public probe bounds power to 0..12. Iteration avoids an independently
    // generated table for every exponent and preserves each tower automorphism.
    for (uint32_t i=0; i<power; ++i) value = frobenius_once(value);
    return value;
}
} // namespace dvgpu_bn254_tower
