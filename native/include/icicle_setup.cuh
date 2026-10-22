#pragma once
#include "icicle_msm_allocations.cuh"
#include "icicle_g2_validation.cuh"
#include "icicle_g2_wire.cuh"

#include <stdexcept>
#include <memory>

constexpr unsigned DVGPU_SETUP_WINDOWS=(bn254::scalar_t::NBITS+3)/4;
constexpr unsigned DVGPU_SETUP_TABLE_SIZE=DVGPU_SETUP_WINDOWS*16;
inline void dvgpu_setup_check(cudaError_t s) {
    if (s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct DvgpuSetupEvents {
    cudaEvent_t marks[6]{};
    cudaEvent_t d2h_start=nullptr;
    explicit DvgpuSetupEvents(bool separate_d2h=false) {
        try { for (auto& e:marks) dvgpu_setup_check(cudaEventCreate(&e));
            if (separate_d2h) dvgpu_setup_check(cudaEventCreate(&d2h_start)); }
        catch (...) { for (auto e:marks) if (e) cudaEventDestroy(e);if(d2h_start) cudaEventDestroy(d2h_start);throw; }
    }
    ~DvgpuSetupEvents() { for (auto e:marks) if(e) cudaEventDestroy(e);if(d2h_start) cudaEventDestroy(d2h_start); }
    void mark(int i,cudaStream_t stream) { dvgpu_setup_check(cudaEventRecord(marks[i],stream)); }
    void collect(DvgpuSetupStats& stats) {
        for (int i=0;i<5;++i) {
            float ms=0; dvgpu_setup_check(cudaEventElapsedTime(&ms,i==4 && d2h_start?d2h_start:marks[i],marks[i+1]));
            stats.device_ns[i]+=uint64_t(double(ms)*1000000.0);
        }
    }
    void finish() {
        for(auto& e:marks) {dvgpu_setup_check(cudaEventDestroy(e));e=nullptr;}
        if(d2h_start) {dvgpu_setup_check(cudaEventDestroy(d2h_start));d2h_start=nullptr;}
    }
};
struct DvgpuG2CheckedRun {
    uint32_t algorithm;DvgpuG2ValidationReport* report;DvgpuG2ValidationStats* stats;
    int64_t corrupt_index=-1;
    const DvgpuG2KernelConfig* config=nullptr;
    DvgpuG2ExperimentStats* experiment=nullptr;
};

// One thread per public window; all later SRS columns reuse these multiples.
template<class Projective>
__global__ void dvgpu_setup_make_table(Projective* table) {
    const unsigned w=blockIdx.x*blockDim.x+threadIdx.x;
    if (w>=DVGPU_SETUP_WINDOWS) return;
    bn254::scalar_t power=bn254::scalar_t::zero();
    power.limbs_storage.limbs[(w*4)/32]=uint32_t(1)<<((w*4)%32);
    const Projective base=power*Projective::generator();
    Projective point=Projective::zero();
    for (unsigned d=0;d<16;++d) { table[w*16+d]=point; point=point+base; }
}
template<class Projective>
__global__ void dvgpu_setup_generate(const bn254::scalar_t* scalars,
    const Projective* table,Projective* output,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=n) return;
    Projective point=Projective::zero();
    if (!table) point=scalars[i]*Projective::generator();
    else for (unsigned w=0;w<DVGPU_SETUP_WINDOWS;++w) {
        const unsigned d=scalars[i].get_scalar_digit(w,4);
        if (d) point=point+table[w*16+d];
    }
    output[i]=point;
}
template<class Projective,class Affine,bool FUSED=false>
__global__ void dvgpu_setup_normalize_each(const Projective* input,Affine* output,size_t n,uint32_t* masks=nullptr,int64_t corrupt=-1) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<n) {
        output[i]=Projective::to_affine(input[i]);
        if constexpr(FUSED) {
            auto* bytes=reinterpret_cast<uint8_t*>(output+i);if(int64_t(i)==corrupt) bytes[0]^=1;
            masks[i]=dvgpu_g2_wire::check(bytes);
        }
    }
}

// O(tile) field operations, log2(tile) synchronization and one inversion.
// ICICLE 2.8 homogeneous coordinates: (x/z,y/z). Zero leaves contribute one.
template<class Projective,class Affine,unsigned TILE=128,bool FUSED=false>
__global__ void dvgpu_setup_normalize_batch(const Projective* input,Affine* output,size_t n,uint32_t* masks=nullptr,int64_t corrupt=-1) {
    using Field=decltype(input[0].z);
    __shared__ Field tree[2*TILE];
    const unsigned t=threadIdx.x;
    const size_t i=size_t(blockIdx.x)*TILE+t;
    Field denominator=Field::one();
    bool zero=true;
    if (i<n) { zero=input[i].z==Field::zero(); if (!zero) denominator=input[i].z; }
    tree[TILE+t]=denominator;
    __syncthreads();
    for (unsigned width=TILE/2;width;width/=2) {
        if (t<width) tree[width+t]=tree[2*(width+t)]*tree[2*(width+t)+1];
        __syncthreads();
    }
    if (t==0) tree[1]=Field::inverse(tree[1]);
    __syncthreads();
    for (unsigned width=1;width<TILE;width*=2) {
        if (t<width) {
            const unsigned node=width+t;
            const Field left=tree[2*node],right=tree[2*node+1],inverse=tree[node];
            tree[2*node]=inverse*right; tree[2*node+1]=inverse*left;
        }
        __syncthreads();
    }
    if (i<n) {
        output[i]=zero?Affine::zero():Affine{input[i].x*tree[TILE+t],input[i].y*tree[TILE+t]};
        if constexpr(FUSED) {
            auto* bytes=reinterpret_cast<uint8_t*>(output+i);if(int64_t(i)==corrupt) bytes[0]^=1;
            masks[i]=dvgpu_g2_wire::check(bytes);
        }
    }
}

struct DvgpuSetupState {
    bn254::projective_t* g1=nullptr;
    bn254::g2_projective_t* g2=nullptr;
    int mode=2;
    DvgpuSetupStats stats{};
    uint64_t pending(bool second) const {
        if (!mode || (second?g2!=nullptr:g1!=nullptr)) return 0;
        return DVGPU_SETUP_TABLE_SIZE*(second?sizeof(bn254::g2_projective_t):sizeof(bn254::projective_t));
    }
    void release(bool current) noexcept {
        if (current) { if (g1) cudaFree(g1); if (g2) cudaFree(g2); }
        g1=nullptr; g2=nullptr;
    }
};

template<class Projective,class Affine>
bool dvgpu_fixed_generator_run(DvgpuSetupState& state,Projective*& retained,
    const uint8_t* input,size_t n,uint8_t* output,cudaStream_t stream,bool g2,DvgpuG2CheckedRun* checked=nullptr) {
    DynaverseMsmAllocations allocations;
    bn254::scalar_t* scalars=nullptr;
    Projective *points=nullptr,*pending=nullptr;
    Affine* affine=nullptr;
    uint32_t* masks=nullptr;DvgpuG2ValidationReport* device_report=nullptr;
    uint8_t* layout_scratch=nullptr;
    const DvgpuG2KernelConfig config=checked && checked->config?*checked->config:DvgpuG2KernelConfig{128,0,0,0};
    const bool fused=checked && config.fused && sizeof(Affine)==128;
    dvgpu_setup_check(allocations.allocate(&scalars,n*sizeof(*scalars),stream));
    dvgpu_setup_check(allocations.allocate(&points,n*sizeof(*points),stream));
    dvgpu_setup_check(allocations.allocate(&affine,n*sizeof(*affine),stream));
    const bool build=state.mode && !retained;
    if (build) dvgpu_setup_check(allocations.allocate(&pending,DVGPU_SETUP_TABLE_SIZE*sizeof(Projective),stream));
    std::unique_ptr<DvgpuSetupEvents> validation;
    if(checked) {
        dvgpu_setup_check(allocations.allocate(&masks,n*sizeof(*masks),stream));
        dvgpu_setup_check(allocations.allocate(&device_report,sizeof(*device_report),stream));
        if(config.layout) dvgpu_setup_check(allocations.allocate(&layout_scratch,n*128,stream));
        validation=std::make_unique<DvgpuSetupEvents>();
    }
    DvgpuSetupEvents events(checked!=nullptr);
    events.mark(0,stream);
    if (build) {
        dvgpu_setup_make_table<<<(DVGPU_SETUP_WINDOWS+31)/32,32,0,stream>>>(pending);
        dvgpu_setup_check(cudaGetLastError());
    }
    events.mark(1,stream);
    dvgpu_setup_check(cudaMemcpyAsync(scalars,input,n*sizeof(*scalars),cudaMemcpyHostToDevice,stream));
    events.mark(2,stream);
    const auto* table=state.mode?(build?pending:retained):nullptr;
    dvgpu_setup_generate<<<(n+127)/128,128,0,stream>>>(scalars,table,points,n);
    dvgpu_setup_check(cudaGetLastError());
    events.mark(3,stream);
    if(fused) validation->mark(1,stream);
    if constexpr(sizeof(Affine)==128) {
        if(fused) {
            if(state.mode==2) dvgpu_setup_normalize_batch<Projective,Affine,128,true><<<(n+127)/128,128,0,stream>>>(points,affine,n,masks,checked->corrupt_index);
            else dvgpu_setup_normalize_each<Projective,Affine,true><<<(n+127)/128,128,0,stream>>>(points,affine,n,masks,checked->corrupt_index);
        }
    }
    if(!fused) {
        if (state.mode==2) dvgpu_setup_normalize_batch<Projective,Affine><<<(n+127)/128,128,0,stream>>>(points,affine,n);
        else dvgpu_setup_normalize_each<Projective,Affine><<<(n+127)/128,128,0,stream>>>(points,affine,n);
    }
    dvgpu_setup_check(cudaGetLastError());
    events.mark(4,stream);
    if(checked) {
        if(!fused && checked->corrupt_index>=0) dvgpu_setup_check(dvgpu_g2_corrupt_test_point(
            reinterpret_cast<uint8_t*>(affine),size_t(checked->corrupt_index),stream));
        if(!fused) validation->mark(1,stream);
        dvgpu_setup_check(dvgpu_g2_validate_configured(reinterpret_cast<const uint8_t*>(affine),layout_scratch,n,
            checked->algorithm,masks,device_report,stream,config,fused,validation->marks[2],validation->marks[0],validation->marks[3]));
        validation->mark(4,stream);
        dvgpu_setup_check(cudaEventRecord(events.d2h_start,stream));
    }
    dvgpu_setup_check(cudaMemcpyAsync(output,affine,n*sizeof(*affine),cudaMemcpyDeviceToHost,stream));
    if(checked) {
        dvgpu_setup_check(cudaMemcpyAsync(checked->report,device_report,sizeof(*device_report),cudaMemcpyDeviceToHost,stream));
        validation->mark(5,stream);
    }
    events.mark(5,stream);
    dvgpu_setup_check(cudaStreamSynchronize(stream));
    events.collect(state.stats);
    bool valid=!checked || !checked->report->invalid_points;
    if(checked) {
        uint64_t* values[]={&checked->stats->parse_curve_device_ns,&checked->stats->subgroup_device_ns,
            &checked->stats->reduce_device_ns,&checked->stats->d2h_device_ns};
        for(int i=1;i<=4;++i) {float ms=0;dvgpu_setup_check(cudaEventElapsedTime(&ms,validation->marks[i],validation->marks[i+1]));
            *values[i-1]+=uint64_t(double(ms)*1000000.0);}
        if(checked->experiment) {
            float ms=0;dvgpu_setup_check(cudaEventElapsedTime(&ms,validation->marks[2],validation->marks[0]));
            checked->experiment->repack_device_ns+=uint64_t(double(ms)*1000000.0);
            dvgpu_setup_check(cudaEventElapsedTime(&ms,validation->marks[0],validation->marks[3]));
            checked->experiment->subgroup_kernel_ns+=uint64_t(double(ms)*1000000.0);
            checked->experiment->logical_repack_bytes+=config.layout?n*256:0;
        }
    }
    if (build && valid) {
        retained=pending; allocations.detach(pending);
        ++state.stats.table_builds[g2]; state.stats.table_bytes+=DVGPU_SETUP_TABLE_SIZE*sizeof(Projective);
    }
    if(build && !valid) dvgpu_setup_check(allocations.release(pending,stream));
    ++state.stats.calls[g2]; state.stats.points[g2]+=n;
    dvgpu_setup_check(allocations.release(scalars,stream));
    dvgpu_setup_check(allocations.release(points,stream));
    dvgpu_setup_check(allocations.release(affine,stream));
    if(checked) {dvgpu_setup_check(allocations.release(masks,stream));dvgpu_setup_check(allocations.release(device_report,stream));validation->finish();}
    if(layout_scratch) dvgpu_setup_check(allocations.release(layout_scratch,stream));
    events.finish();return valid;
}
