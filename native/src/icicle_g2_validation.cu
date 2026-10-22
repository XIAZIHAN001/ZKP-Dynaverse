#include "api/bn254.h"
#include "icicle_g2_validation.cuh"
#include "icicle_msm_allocations.cuh"
#include "icicle_g2_wire.cuh"
#include <stdexcept>
#include <vector>

namespace {
using P=bn254::g2_projective_t;
using E=decltype(P::zero().x);
using F=decltype(P::zero().x.real);
#include "icicle_g2_constants.cuh"
static_assert(sizeof(F)==32 && sizeof(E)==64 && sizeof(P)==192,"G2 layout changed");
static_assert(sizeof(DvgpuG2ValidationReport)==32 && sizeof(DvgpuG2ValidationStats)==64,"G2 ABI changed");
static_assert(sizeof(DvgpuG2KernelConfig)==16 && sizeof(DvgpuG2KernelResources)==56 && sizeof(DvgpuG2ExperimentStats)==24,"G2 experiment ABI changed");

using dvgpu_g2_wire::canonical;using dvgpu_g2_wire::parse;

// General integer binary chain: r is NOT an Fr field element.
__device__ P full_order(const P& point) {
    constexpr uint32_t r[8]={0xf0000001,0x43e1f593,0x79b97091,0x2833e848,0x8181585d,0xb85045b6,0xe131a029,0x30644e72};
    P result=P::zero();
    for (int bit=253;bit>=0;--bit) {
        result=P::dbl(result);
        if ((r[bit/32]>>(bit%32))&1) result=result+point;
    }
    return result;
}

__device__ E conjugate(const E& value) {
    // Pinned Field::neg(0) returns the raw modulus. Keep zero canonical,
    // especially for homogeneous identity Z=0 and the next image's parser.
    return E{value.real,value.imaginary==F::zero()?F::zero():F::neg(value.imaginary)};
}
__device__ P psi(const P& point) {
    return P{conjugate(point.x)*psi_cx(),conjugate(point.y)*psi_cy(),conjugate(point.z)};
}
__device__ bool equal(const P& a,const P& b) {
    const bool az=P::is_zero(a),bz=P::is_zero(b);
    if (az || bz) return az && bz;
    // ICICLE's raw projective == would consider (0,0,0) equal to any point.
    if (a.z==E::zero() || b.z==E::zero()) return false;
    return a.x*b.z==b.x*a.z && a.y*b.z==b.y*a.z;
}
template<unsigned BITS,unsigned WORDS>
__device__ P integer_mul(const P& point,const uint32_t (&scalar)[WORDS]) {
    P result=P::zero();
    for (int bit=BITS-1;bit>=0;--bit) {
        result=P::dbl(result);if ((scalar[bit/32]>>(bit%32))&1) result=result+point;
    }
    return result;
}
template<unsigned ALGORITHM,bool COMPACT=false>
__device__ bool in_subgroup(const P& point) {
    if constexpr (ALGORITHM==1) return P::is_zero(full_order(point));
    if constexpr (ALGORITHM==2) {
        constexpr uint32_t lambda[4]={DVGPU_G2_LAMBDA_WORDS};
        return equal(psi(point),integer_mul<127>(point,lambda));
    }
    if constexpr (ALGORITHM==3) {
        constexpr uint32_t seed[2]={DVGPU_G2_SEED_WORDS};
        if constexpr(COMPACT) {
            P temp=integer_mul<63>(point,seed);P sum=temp+point;
            temp=psi(temp);sum=sum+temp;temp=psi(temp);sum=sum+temp;
            return equal(sum,P::dbl(psi(temp)));
        }
        const P a=integer_mul<63>(point,seed),b=psi(a),c=psi(b);
        return equal(a+point+b+c,P::dbl(psi(c)));
    }
    return false;
}

__global__ void parse_curve(const uint8_t* wire,size_t n,uint32_t* masks) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=n) return;
    P point;const uint32_t status=parse(wire+i*128,point);
    masks[i]=status?status:(P::is_on_curve(point)?0:2);
}
__global__ void corrupt_test_point(uint8_t* wire,size_t index) {wire[index*128]^=1;}
template<unsigned ALGORITHM,bool COMPACT=false>
__global__ void subgroup(const uint8_t* wire,size_t n,bool soa,uint32_t* masks) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=n || masks[i]) return;
    P point;parse(wire,point,n,i,soa);
    if (P::is_zero(point)) return;
    masks[i]=in_subgroup<ALGORITHM,COMPACT>(point)?0:4;
}
__global__ void repack(const uint32_t* wire,uint32_t* planes,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n*32) planes[i]=wire[(i%n)*32+i/n];
}
__global__ void psi_probe(const P* points,size_t n,P* output,uint32_t* masks) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if (i>=n) return;
    P point=points[i];
    const F fields[6]={point.x.real,point.x.imaginary,point.y.real,point.y.imaginary,point.z.real,point.z.imaginary};
    bool valid=true;for (const auto& field:fields) valid&=canonical(field);
    valid=valid && P::is_on_curve(point);
    masks[i]=valid?0:1;
    if (!valid) {for (unsigned j=0;j<3;++j) output[3*i+j]=P::zero();return;}
    for (unsigned j=0;j<3;++j) {point=psi(point);output[3*i+j]=point;}
}
__global__ void reduce_status(const uint32_t* masks,size_t n,uint32_t algorithm,DvgpuG2ValidationReport* report) {
    __shared__ uint64_t counts[128],first[128];__shared__ uint32_t errors[128];
    const unsigned lane=threadIdx.x;
    uint64_t count=0,index=UINT64_MAX;uint32_t mask=0;
    for (size_t i=lane;i<n;i+=128) if (masks[i]) {++count;mask|=masks[i];if (i<index) index=i;}
    counts[lane]=count;first[lane]=index;errors[lane]=mask;__syncthreads();
    for (unsigned width=64;width;width/=2) {
        if (lane<width) {
            counts[lane]+=counts[lane+width];errors[lane]|=errors[lane+width];
            if (first[lane+width]<first[lane]) first[lane]=first[lane+width];
        }
        __syncthreads();
    }
    if (!lane) *report={n,counts[0],first[0],errors[0],algorithm};
}
cudaError_t launch(const uint8_t* wire,size_t n,uint32_t algorithm,uint32_t* masks,
    DvgpuG2ValidationReport* report,cudaStream_t stream,cudaEvent_t* events,
    const DvgpuG2KernelConfig& config=DvgpuG2KernelConfig{128,0,0,0},uint8_t* scratch=nullptr,
    bool prechecked=false,cudaEvent_t repack_done=nullptr) {
    if (!n || n>4096 || algorithm<1 || algorithm>3 || !wire || !masks || !report) return cudaErrorInvalidValue;
    if((config.threads!=64 && config.threads!=128 && config.threads!=256) || config.schedule>1 || config.layout>1 || config.fused>1
        || (config.layout && !scratch)) return cudaErrorInvalidValue;
    const auto threads=config.threads;
    if(!prechecked) parse_curve<<<(n+threads-1)/threads,threads,0,stream>>>(wire,n,masks);
    auto status=cudaGetLastError();if (status!=cudaSuccess) return status;
    if (events && (status=cudaEventRecord(events[2],stream))!=cudaSuccess) return status;
    const uint8_t* input=wire;
    if(config.layout) {
        repack<<<(n*32+127)/128,128,0,stream>>>(reinterpret_cast<const uint32_t*>(wire),reinterpret_cast<uint32_t*>(scratch),n);
        status=cudaGetLastError();if(status!=cudaSuccess) return status;input=scratch;
    }
    if(repack_done && (status=cudaEventRecord(repack_done,stream))!=cudaSuccess) return status;
    if (algorithm==1) subgroup<1><<<(n+threads-1)/threads,threads,0,stream>>>(input,n,config.layout!=0,masks);
    if (algorithm==2) subgroup<2><<<(n+threads-1)/threads,threads,0,stream>>>(input,n,config.layout!=0,masks);
    if (algorithm==3 && !config.schedule) subgroup<3><<<(n+threads-1)/threads,threads,0,stream>>>(input,n,config.layout!=0,masks);
    if (algorithm==3 && config.schedule) subgroup<3,true><<<(n+threads-1)/threads,threads,0,stream>>>(input,n,config.layout!=0,masks);
    status=cudaGetLastError();if (status!=cudaSuccess) return status;
    if (events && (status=cudaEventRecord(events[3],stream))!=cudaSuccess) return status;
    reduce_status<<<1,128,0,stream>>>(masks,n,algorithm,report);
    return cudaGetLastError();
}
void check(cudaError_t status) {if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));}
struct Events {
    cudaEvent_t values[7]{};
    Events() {try {for (auto& e:values) check(cudaEventCreate(&e));}catch (...) {for (auto e:values) if(e) cudaEventDestroy(e);throw;}}
    ~Events() {for (auto e:values) if(e) cudaEventDestroy(e);}
    void mark(int index,cudaStream_t stream) {check(cudaEventRecord(values[index],stream));}
    uint64_t ns(int index) {float ms=0;check(cudaEventElapsedTime(&ms,values[index],values[index+1]));return uint64_t(double(ms)*1000000.0);}
    uint64_t between(int a,int b) {float ms=0;check(cudaEventElapsedTime(&ms,values[a],values[b]));return uint64_t(double(ms)*1000000.0);}
    void finish() {for (auto& e:values) {check(cudaEventDestroy(e));e=nullptr;}}
};
}

cudaError_t dvgpu_g2_validate_device(const uint8_t* wire,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,cudaStream_t stream) {
    return launch(wire,n,algorithm,masks,report,stream,nullptr);
}
cudaError_t dvgpu_g2_validate_device_events(const uint8_t* wire,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,cudaStream_t stream,cudaEvent_t parsed,cudaEvent_t subgroup) {
    cudaEvent_t events[6]={nullptr,nullptr,parsed,subgroup,nullptr,nullptr};
    return launch(wire,n,algorithm,masks,report,stream,events);
}
cudaError_t dvgpu_g2_corrupt_test_point(uint8_t* wire,size_t index,cudaStream_t stream) {
    corrupt_test_point<<<1,1,0,stream>>>(wire,index);return cudaGetLastError();
}
cudaError_t dvgpu_g2_validate_configured(const uint8_t* wire,uint8_t* scratch,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,cudaStream_t stream,const DvgpuG2KernelConfig& config,
    bool prechecked,cudaEvent_t parsed,cudaEvent_t repacked,cudaEvent_t subgroup_done) {
    cudaEvent_t events[6]={nullptr,nullptr,parsed,subgroup_done,nullptr,nullptr};
    return launch(wire,n,algorithm,masks,report,stream,events,config,scratch,prechecked,repacked);
}
cudaError_t dvgpu_g2_resources_run(uint32_t algorithm,const DvgpuG2KernelConfig& config,DvgpuG2KernelResources& output) {
    cudaFuncAttributes attrs{};int blocks=0,device=0,maximum=0;cudaError_t status;
    const void* kernel=algorithm==1?reinterpret_cast<const void*>(subgroup<1>):(algorithm==2?
        reinterpret_cast<const void*>(subgroup<2>):(config.schedule?reinterpret_cast<const void*>(subgroup<3,true>):reinterpret_cast<const void*>(subgroup<3>)));
    if((status=cudaFuncGetAttributes(&attrs,kernel))!=cudaSuccess) return status;
    if((status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,config.threads,0))!=cudaSuccess) return status;
    if((status=cudaGetDevice(&device))!=cudaSuccess) return status;
    if((status=cudaDeviceGetAttribute(&maximum,cudaDevAttrMaxThreadsPerMultiProcessor,device))!=cudaSuccess) return status;
    output={uint32_t(attrs.numRegs),config.threads,uint32_t(blocks),uint32_t(maximum),attrs.localSizeBytes,attrs.sharedSizeBytes,0,0,0,0};return cudaSuccess;
}
void dvgpu_g2_inspect_run(const uint8_t* wire,size_t n,uint32_t algorithm,
    uint32_t* masks,DvgpuG2ValidationReport* report,DvgpuG2ValidationStats& stats,cudaStream_t stream,
    const DvgpuG2KernelConfig* selected,DvgpuG2ExperimentStats* experiment) {
    DynaverseMsmAllocations allocations;uint8_t* bytes=nullptr;
    uint8_t* scratch=nullptr;const DvgpuG2KernelConfig config=selected?*selected:DvgpuG2KernelConfig{128,0,0,0};
    uint32_t* status=nullptr;DvgpuG2ValidationReport* device_report=nullptr;
    check(allocations.allocate(&bytes,n*128,stream));
    check(allocations.allocate(&status,n*4,stream));
    check(allocations.allocate(&device_report,sizeof(*device_report),stream));
    if(config.layout) check(allocations.allocate(&scratch,n*128,stream));
    Events events;events.mark(0,stream);
    check(cudaMemcpyAsync(bytes,wire,n*128,cudaMemcpyHostToDevice,stream));events.mark(1,stream);
    check(launch(bytes,n,algorithm,status,device_report,stream,events.values,config,scratch,false,events.values[6]));events.mark(4,stream);
    check(cudaMemcpyAsync(masks,status,n*4,cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(report,device_report,sizeof(*report),cudaMemcpyDeviceToHost,stream));events.mark(5,stream);
    check(allocations.release(bytes,stream));check(allocations.release(status,stream));check(allocations.release(device_report,stream));
    if(scratch) check(allocations.release(scratch,stream));
    check(cudaStreamSynchronize(stream));
    stats.h2d_device_ns=events.ns(0);stats.parse_curve_device_ns=events.ns(1);
    stats.subgroup_device_ns=events.ns(2);stats.reduce_device_ns=events.ns(3);stats.d2h_device_ns=events.ns(4);
    if(experiment) {experiment->repack_device_ns=events.between(2,6);experiment->subgroup_kernel_ns=events.between(6,3);
        experiment->logical_repack_bytes=config.layout?n*256:0;}
    events.finish();
}

void dvgpu_g2_psi_probe_run(const uint8_t* input,size_t n,uint8_t* output,cudaStream_t stream) {
    // Reverse destruction order: drain/free device resources before destroying
    // the host buffer borrowed by an outstanding D2H, including on exceptions.
    std::vector<uint32_t> status(n);
    DynaverseMsmAllocations allocations;P *points=nullptr,*images=nullptr;uint32_t* masks=nullptr;
    check(allocations.allocate(&points,n*sizeof(P),stream));check(allocations.allocate(&images,3*n*sizeof(P),stream));
    check(allocations.allocate(&masks,n*4,stream));
    check(cudaMemcpyAsync(points,input,n*sizeof(P),cudaMemcpyHostToDevice,stream));
    psi_probe<<<(n+127)/128,128,0,stream>>>(points,n,images,masks);check(cudaGetLastError());
    check(cudaMemcpyAsync(output,images,3*n*sizeof(P),cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(status.data(),masks,n*4,cudaMemcpyDeviceToHost,stream));
    check(allocations.release(points,stream));check(allocations.release(images,stream));check(allocations.release(masks,stream));
    check(cudaStreamSynchronize(stream));
    for (auto mask:status) if (mask) throw std::invalid_argument("invalid homogeneous psi input");
}
