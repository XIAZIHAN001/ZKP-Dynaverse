// Generated source only; no product build/test has run.
// ark-bn254 0.4.0 configuration (MIT OR Apache-2.0); canonical NOT Montgomery.
#pragma once
#include "icicle_bn254_tower.cuh"
#include "dynaverse_bn254_lines.h"
namespace dvgpu_bn254_lines {
using dvgpu_bn254_tower::Fq; using dvgpu_bn254_tower::Fq2;
static constexpr unsigned LOOP_DIGITS=65;
__host__ __device__ constexpr int loop_digit(unsigned index) {
    constexpr int8_t digits[65]={0,0,0,1,0,1,0,-1,0,0,1,-1,0,0,1,0,0,1,1,0,-1,0,0,1,0,-1,0,0,0,0,1,1,1,0,0,-1,0,0,1,0,0,0,0,0,-1,0,0,1,1,0,0,-1,0,0,0,1,1,0,-1,0,0,1,0,1,1};
    return digits[index];
}
constexpr unsigned line_count() {
    unsigned count=2;
    for (unsigned i=0;i<LOOP_DIGITS-1;++i) count+=1+(loop_digit(i)!=0);
    return count;
}
static_assert(line_count()==DVGPU_LINE_COUNT,"BN254 line slot/NAF shape mismatch");
__device__ inline Fq two_inverse() { return Fq{storage<8>{0x6c3e7ea4u,0x9e10460bu,0xb438e546u,0xcbc0b548u,0x40c0ac2eu,0xdc2822dbu,0x7098d014u,0x18322739u}}; }
__device__ inline Fq2 curve_b() { return Fq2{Fq{storage<8>{0x24a138e5u,0x3267e6dcu,0x59dbefa3u,0xb5b4c5e5u,0x1be06ac3u,0x81be1899u,0xceb8aaaeu,0x2b149d40u}},Fq{storage<8>{0x85c315d2u,0xe4a2bd06u,0xe52d1852u,0xa74fa084u,0xeed8fdf4u,0xcd2cafadu,0x3af0fed4u,0x009713b0u}}}; }
__device__ inline Fq2 q_x() { return Fq2{Fq{storage<8>{0x176f553du,0x99e39557u,0xc2c3330cu,0xb78cc310u,0xf559b143u,0x4c0bec3cu,0x4f7911f7u,0x2fb34798u}},Fq{storage<8>{0x640fcba2u,0x1665d51cu,0x0b7c9dceu,0x32ae2a1du,0xd75a0794u,0x4ba4cc8bu,0x61ebae20u,0x16c9e550u}}}; }
__device__ inline Fq2 q_y() { return Fq2{Fq{storage<8>{0x71a0135au,0xdc540146u,0xa9c95998u,0xdbaae0edu,0xb6e2f9b9u,0xdc5ec698u,0x489af5dcu,0x063cf305u}},Fq{storage<8>{0x2623b0e3u,0x82d37f63u,0x8fa25bd2u,0x21807dc9u,0xec796f2bu,0x0704b5a7u,0xac41049au,0x07c03cbcu}}}; }
} // namespace dvgpu_bn254_lines
