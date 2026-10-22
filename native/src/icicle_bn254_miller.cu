#include "icicle_bn254_miller.cuh"
#include "icicle_bn254_miller_probe.cuh"
#include "icicle_g2_wire.cuh"
#include "icicle_g2_validation.cuh"
#include "icicle_msm_allocations.cuh"
#include <stdexcept>

namespace {
using namespace dvgpu_bn254_miller;
using dvgpu_bn254_lines::Point;
using dvgpu_bn254_lines::Line;

// G1 has the same frozen uncompressed flags as G2, in the final y byte.
// Keep canonical comparison shared with G2 and retain the full Rust decoder
// as an independent oracle. h(G1)=1 is asserted in the Rust diagnostic module.
__device__ uint32_t read_g1(const uint8_t* bytes,bn254::projective_t& p) {
    bool zero=true;for(unsigned i=0;i<64;++i) zero&=bytes[i]==0;
    if(zero){p=bn254::projective_t::zero();return DVGPU_MILLER_INFINITY;}
    const unsigned flags=bytes[63]&0xc0;if(flags==0xc0)return 1;
    Fq fields[2];
    for(unsigned j=0;j<2;++j){
        for(unsigned k=0;k<8;++k){
            uint32_t word=0;
            for(unsigned b=0;b<4;++b){
                const unsigned offset=j*32+k*4+b;
                word|=uint32_t(offset==63?(bytes[offset]&0x3f):bytes[offset])<<(8*b);
            }
            fields[j].limbs_storage.limbs[k]=word;
        }
        if(!dvgpu_g2_wire::canonical(fields[j]))return 1;
    }
    if(flags==0x40){p=bn254::projective_t::zero();return DVGPU_MILLER_INFINITY;}
    p={fields[0],fields[1],Fq::one()};return bn254::projective_t::is_on_curve(p)?0:2;
}
__device__ void write_fq(const Fq& value,uint8_t* bytes){
    for(unsigned i=0;i<8;++i){
        const uint32_t word=value.limbs_storage.limbs[i];
        for(unsigned b=0;b<4;++b)bytes[4*i+b]=uint8_t(word>>(8*b));
    }
}
__device__ void write_fq12(const Fq12& value,uint8_t* bytes){
    const Fq2 fields[6]={value.c0.c0,value.c0.c1,value.c0.c2,value.c1.c0,value.c1.c1,value.c1.c2};
    for(unsigned i=0;i<6;++i){write_fq(fields[i].real,bytes+64*i);write_fq(fields[i].imaginary,bytes+64*i+32);}
}
__device__ void consume(const Line& line,const bn254::projective_t& p,Fq12& f,uint8_t* out,unsigned& index){
    const Fq12 evaluated=evaluate(line,p.x,p.y);
    f=multiply(f,evaluated);
    write_fq12(evaluated,out+(1+index)*DVGPU_MILLER_FQ12_BYTES);
    write_fq12(f,out+(1+DVGPU_MILLER_LINES+index)*DVGPU_MILLER_FQ12_BYTES);
    ++index;
}
__global__ void accumulate(const uint8_t* wire,uint8_t* out,uint32_t* metadata){
    // Exactly one pair, single owning thread; no cross-pair product or barrier.
    if(blockIdx.x || threadIdx.x)return;
    bn254::projective_t p;metadata[1]=read_g1(wire,p);
    bn254::g2_projective_t parsed;
    if(metadata[2]==0){
        dvgpu_g2_wire::parse(wire+64,parsed);
        if(bn254::g2_projective_t::is_zero(parsed))metadata[2]=DVGPU_MILLER_INFINITY;
    }
    // Validation is complete for BOTH points before applying pair omission.
    if((metadata[1]!=0 && metadata[1]!=DVGPU_MILLER_INFINITY) ||
       (metadata[2]!=0 && metadata[2]!=DVGPU_MILLER_INFINITY))return;
    Fq12 f=Fq12::one();
    if(metadata[1]==DVGPU_MILLER_INFINITY || metadata[2]==DVGPU_MILLER_INFINITY){write_fq12(f,out);return;}
    const Point q{parsed.x,parsed.y,Fq2::one()},negative{q.x,dvgpu_bn254_lines::negate(q.y),Fq2::one()};
    Point r=q;unsigned written=0;
    for(int i=int(dvgpu_bn254_lines::LOOP_DIGITS)-2;i>=0;--i){
        // Skip the initial square as in the locked arkworks Miller loop.
        if(i!=int(dvgpu_bn254_lines::LOOP_DIGITS)-2)f=square(f);
        consume(dvgpu_bn254_lines::double_line(r),p,f,out,written);
        const int bit=dvgpu_bn254_lines::loop_digit(unsigned(i));
        if(bit==1)consume(dvgpu_bn254_lines::add_line(r,q),p,f,out,written);
        else if(bit==-1)consume(dvgpu_bn254_lines::add_line(r,negative),p,f,out,written);
    }
    // Fixed positive BN254 X: no negative-X inversion. Both characteristic
    // tails are consumed without another square; the second image is negated.
    const Point q1=dvgpu_bn254_lines::characteristic(q);Point q2=dvgpu_bn254_lines::characteristic(q1);
    q2.y=dvgpu_bn254_lines::negate(q2.y);
    consume(dvgpu_bn254_lines::add_line(r,q1),p,f,out,written);
    consume(dvgpu_bn254_lines::add_line(r,q2),p,f,out,written);
    write_fq12(f,out);metadata[0]=written;
}
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
}

void dvgpu_bn254_miller_run(const uint8_t* p,const uint8_t* q,uint8_t* pending,uint32_t* metadata,cudaStream_t stream){
    DynaverseMsmAllocations allocations;
    uint8_t *input=nullptr,*output=nullptr;uint32_t* device_metadata=nullptr;DvgpuG2ValidationReport* report=nullptr;
    check(allocations.allocate(&input,192,stream));
    check(allocations.allocate(&output,DVGPU_MILLER_OUTPUT_BYTES,stream));
    check(allocations.allocate(&device_metadata,3*sizeof(uint32_t),stream));
    check(allocations.allocate(&report,sizeof(DvgpuG2ValidationReport),stream));
    check(cudaMemcpyAsync(input,p,64,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(input+64,q,128,cudaMemcpyHostToDevice,stream));
    check(cudaMemsetAsync(output,0,DVGPU_MILLER_OUTPUT_BYTES,stream));
    check(cudaMemsetAsync(device_metadata,0,3*sizeof(uint32_t),stream));
    check(dvgpu_g2_validate_device(input+64,1,1,device_metadata+2,report,stream));
    accumulate<<<1,1,0,stream>>>(input,output,device_metadata);check(cudaGetLastError());
    check(cudaMemcpyAsync(pending,output,DVGPU_MILLER_OUTPUT_BYTES,cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(metadata,device_metadata,3*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    check(allocations.release(input,stream));check(allocations.release(output,stream));
    check(allocations.release(device_metadata,stream));check(allocations.release(report,stream));
    check(cudaStreamSynchronize(stream));
    const uint32_t ps=metadata[1],qs=metadata[2];
    if((ps!=0 && ps!=1 && ps!=2 && ps!=DVGPU_MILLER_INFINITY) ||
       (qs!=0 && qs!=1 && qs!=2 && qs!=4 && qs!=DVGPU_MILLER_INFINITY) ||
       metadata[0]!=(ps==0 && qs==0?DVGPU_MILLER_LINES:0))
        throw std::runtime_error("inconsistent BN254 Miller kernel result");
}
