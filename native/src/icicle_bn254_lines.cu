#include "icicle_bn254_lines.cuh"
#include "icicle_bn254_lines_probe.cuh"
#include "icicle_g2_validation.cuh"
#include "icicle_g2_wire.cuh"
#include "icicle_msm_allocations.cuh"
#include <stdexcept>

namespace {
using namespace dvgpu_bn254_lines;
static_assert(sizeof(Fq)==32 && sizeof(Fq2)==64 && sizeof(Line)==192,"line layout changed");

__device__ void write_fq(const Fq& value,uint8_t* bytes) {
    for (unsigned limb=0;limb<8;++limb) {
        const uint32_t word=value.limbs_storage.limbs[limb];
        for (unsigned byte=0;byte<4;++byte) bytes[4*limb+byte]=uint8_t(word>>(8*byte));
    }
}
__device__ void write_line(const Line& line,uint8_t* bytes) {
    const Fq2 values[3]={line.c0,line.c1,line.c2};
    for (unsigned i=0;i<3;++i) {
        write_fq(values[i].real,bytes+i*64);write_fq(values[i].imaginary,bytes+i*64+32);
    }
}
__global__ void prepare(const uint8_t* wire,size_t n,uint8_t* output,uint32_t* counts,uint32_t* statuses) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=n || statuses[index]) return;
    bn254::g2_projective_t parsed;
    // The retained validator already checked canonical encoding, curve and
    // full integer subgroup order in the same stream over this exact buffer.
    dvgpu_g2_wire::parse(wire+128*index,parsed);
    if (bn254::g2_projective_t::is_zero(parsed)) {statuses[index]=DVGPU_LINE_INFINITY;return;}
    const Point q{parsed.x,parsed.y,Fq2::one()},negative{q.x,negate(q.y),Fq2::one()};
    Point r=q;unsigned written=0;uint8_t* slot=output+index*DVGPU_LINE_ITEM_BYTES;
    // Fixed BN254 positive x, signed NAF digits least-significant first.
    // Skip the leading one, then emit each doubling before its optional add.
    for (int i=int(LOOP_DIGITS)-2;i>=0;--i) {
        write_line(double_line(r),slot+DVGPU_LINE_BYTES*written++);
        const int bit=loop_digit(unsigned(i));
        if (bit==1) write_line(add_line(r,q),slot+DVGPU_LINE_BYTES*written++);
        else if (bit==-1) write_line(add_line(r,negative),slot+DVGPU_LINE_BYTES*written++);
    }
    const Point q1=characteristic(q);Point q2=characteristic(q1);
    // X_IS_NEGATIVE=false, so r.y is not negated. The second q image IS negated.
    q2.y=negate(q2.y);
    write_line(add_line(r,q1),slot+DVGPU_LINE_BYTES*written++);
    write_line(add_line(r,q2),slot+DVGPU_LINE_BYTES*written++);
    counts[index]=written;
}
void check(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

void dvgpu_bn254_lines_run(const uint8_t* points,size_t count,uint8_t* pending,
    uint32_t* pending_counts,uint32_t* pending_statuses,cudaStream_t stream) {
    const size_t input_bytes=count*128,output_bytes=count*DVGPU_LINE_ITEM_BYTES;
    const size_t status_bytes=count*sizeof(uint32_t);
    DynaverseMsmAllocations allocations;
    uint8_t *device_input=nullptr,*device_output=nullptr;
    uint32_t *device_counts=nullptr,*device_statuses=nullptr;
    DvgpuG2ValidationReport* report=nullptr;
    check(allocations.allocate(&device_input,input_bytes,stream));
    check(allocations.allocate(&device_output,output_bytes,stream));
    check(allocations.allocate(&device_counts,status_bytes,stream));
    check(allocations.allocate(&device_statuses,status_bytes,stream));
    check(allocations.allocate(&report,sizeof(DvgpuG2ValidationReport),stream));
    check(cudaMemcpyAsync(device_input,points,input_bytes,cudaMemcpyHostToDevice,stream));
    check(cudaMemsetAsync(device_output,0,output_bytes,stream));
    check(cudaMemsetAsync(device_counts,0,status_bytes,stream));
    // Reuse the established full-order implementation, not a new validator or
    // a subgroup assumption hidden in the coefficient generator.
    check(dvgpu_g2_validate_device(device_input,count,1,device_statuses,report,stream));
    prepare<<<unsigned((count+31)/32),32,0,stream>>>(device_input,count,device_output,device_counts,device_statuses);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(pending,device_output,output_bytes,cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(pending_counts,device_counts,status_bytes,cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(pending_statuses,device_statuses,status_bytes,cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    check(allocations.release(device_input,stream));check(allocations.release(device_output,stream));
    check(allocations.release(device_counts,stream));check(allocations.release(device_statuses,stream));
    check(allocations.release(report,stream));check(cudaStreamSynchronize(stream));
    // Fail closed before the owner publishes any host output if the internal
    // fixed loop shape/status contract was violated. Execution errors retire it.
    for (size_t i=0;i<count;++i) {
        const uint32_t status=pending_statuses[i];
        if ((status!=0 && status!=1 && status!=2 && status!=4 && status!=DVGPU_LINE_INFINITY) ||
            pending_counts[i]!=(status==0?DVGPU_LINE_COUNT:0))
            throw std::runtime_error("inconsistent BN254 line kernel result");
    }
}
