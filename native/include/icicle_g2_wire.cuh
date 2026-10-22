#pragma once
#include "api/bn254.h"
#include <cstdint>
#include <cstddef>
// Shared frozen parser for separate and fused checks. All representations use
// canonical LE fields. SoA stores32 wire-limb planes; flags are still byte127.
namespace dvgpu_g2_wire {
using P=bn254::g2_projective_t;using E=decltype(P::zero().x);using F=decltype(P::zero().x.real);
__device__ inline bool canonical(const F& value) {
    constexpr uint32_t q[8]={0xd87cfd47,0x3c208c16,0x6871ca8d,0x97816a91,0x8181585d,0xb85045b6,0xe131a029,0x30644e72};
    for(int i=7;i>=0;--i) {if(value.limbs_storage.limbs[i]<q[i]) return true;if(value.limbs_storage.limbs[i]>q[i]) return false;}return false;
}
__device__ inline uint8_t byte(const uint8_t* data,size_t n,size_t index,unsigned offset,bool soa) {
    if(!soa) return data[index*128+offset];
    return uint8_t(reinterpret_cast<const uint32_t*>(data)[size_t(offset/4)*n+index]>>(8*(offset%4)));
}
__device__ inline uint32_t parse(const uint8_t* data,P& point,size_t n=1,size_t index=0,bool soa=false) {
    bool all_zero=true;for(unsigned i=0;i<128;++i) all_zero&=byte(data,n,index,i,soa)==0;
    if(all_zero) {point=P::zero();return 0;}
    const unsigned flags=byte(data,n,index,127,soa)&0xc0;if(flags==0xc0) return 1;
    F fields[4];
    for(unsigned j=0;j<4;++j) {
        for(unsigned k=0;k<8;++k) {
            uint32_t value=0;for(unsigned b=0;b<4;++b) {
                const unsigned offset=j*32+k*4+b;const uint32_t raw=byte(data,n,index,offset,soa);
                value|=(offset==127?(raw&0x3f):raw)<<(8*b);
            }fields[j].limbs_storage.limbs[k]=value;
        }if(!canonical(fields[j])) return 1;
    }
    if(flags==0x40) {point=P::zero();return 0;}
    point=P{E{fields[0],fields[1]},E{fields[2],fields[3]},E::one()};return 0;
}
__device__ inline uint32_t check(const uint8_t* data) {
    P point;const auto status=parse(data,point);return status?status:(P::is_on_curve(point)?0:2);
}
}
