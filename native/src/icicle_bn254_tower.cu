#include "icicle_bn254_tower.cuh"
#include "icicle_bn254_tower_probe.cuh"
#include "icicle_g2_wire.cuh"
#include "icicle_msm_allocations.cuh"
#include <stdexcept>

namespace {
using namespace dvgpu_bn254_tower;

__device__ bool read_fq(const uint8_t* bytes, Fq& value) {
    for (unsigned limb=0; limb<8; ++limb) {
        uint32_t word = 0;
        for (unsigned byte=0; byte<4; ++byte) word |= uint32_t(bytes[4*limb+byte]) << (8*byte);
        value.limbs_storage.limbs[limb] = word;
    }
    // Field wire has no curve flags and no reduction of noncanonical input.
    return dvgpu_g2_wire::canonical(value);
}
__device__ void write_fq(const Fq& value, uint8_t* bytes) {
    for (unsigned limb=0; limb<8; ++limb) {
        const uint32_t word = value.limbs_storage.limbs[limb];
        for (unsigned byte=0; byte<4; ++byte) bytes[4*limb+byte] = uint8_t(word >> (8*byte));
    }
}

template<typename T> struct Wire;
template<> struct Wire<Fq2> {
    static constexpr size_t width = 64;
    static __device__ bool read(const uint8_t* bytes, Fq2& value) {
        const bool c0 = read_fq(bytes,value.real);
        const bool c1 = read_fq(bytes+32,value.imaginary);
        return c0 && c1;
    }
    static __device__ void write(const Fq2& value, uint8_t* bytes) {
        write_fq(value.real,bytes); write_fq(value.imaginary,bytes+32);
    }
};
template<> struct Wire<Fq6> {
    static constexpr size_t width = 192;
    static __device__ bool read(const uint8_t* bytes, Fq6& value) {
        const bool c0 = Wire<Fq2>::read(bytes,value.c0);
        const bool c1 = Wire<Fq2>::read(bytes+64,value.c1);
        const bool c2 = Wire<Fq2>::read(bytes+128,value.c2);
        return c0 && c1 && c2;
    }
    static __device__ void write(const Fq6& value, uint8_t* bytes) {
        Wire<Fq2>::write(value.c0,bytes); Wire<Fq2>::write(value.c1,bytes+64);
        Wire<Fq2>::write(value.c2,bytes+128);
    }
};
template<> struct Wire<Fq12> {
    static constexpr size_t width = 384;
    static __device__ bool read(const uint8_t* bytes, Fq12& value) {
        const bool c0 = Wire<Fq6>::read(bytes,value.c0);
        const bool c1 = Wire<Fq6>::read(bytes+192,value.c1);
        return c0 && c1;
    }
    static __device__ void write(const Fq12& value, uint8_t* bytes) {
        Wire<Fq6>::write(value.c0,bytes); Wire<Fq6>::write(value.c1,bytes+192);
    }
};

template<typename T>
__global__ void inspect(const uint8_t* left, const uint8_t* right, size_t count,
    uint32_t operation, uint32_t power, uint8_t* output, uint32_t* statuses) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= count) return;
    T a = T::zero(), b = T::zero(), value = T::zero();
    const size_t offset = index*Wire<T>::width;
    const bool binary = operation <= DVGPU_TOWER_MULTIPLY;
    const bool valid_left = Wire<T>::read(left+offset,a);
    const bool valid_right = !binary || Wire<T>::read(right+offset,b);
    uint32_t status = DVGPU_TOWER_OK;
    if (!valid_left || !valid_right) {
        status = DVGPU_TOWER_NON_CANONICAL;
    } else if (operation == DVGPU_TOWER_INVERSE && is_zero(a)) {
        status = DVGPU_TOWER_ZERO_INVERSE;
    } else {
        switch (operation) {
        case DVGPU_TOWER_ADD: value = add(a,b); break;
        case DVGPU_TOWER_SUBTRACT: value = subtract(a,b); break;
        case DVGPU_TOWER_MULTIPLY: value = multiply(a,b); break;
        case DVGPU_TOWER_SQUARE: value = square(a); break;
        case DVGPU_TOWER_INVERSE: value = inverse(a); break;
        case DVGPU_TOWER_FROBENIUS: value = frobenius(a,power); break;
        }
    }
    // Invalid items never expose an uninitialized or partially parsed value.
    Wire<T>::write(value,output+offset);
    statuses[index] = status;
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

void dvgpu_bn254_tower_run(const uint8_t* left, const uint8_t* right, size_t count,
    uint32_t degree, uint32_t operation, uint32_t power,
    uint8_t* pending_output, uint32_t* pending_statuses, cudaStream_t stream) {
    const size_t bytes = count*degree*32;
    const bool binary = operation <= DVGPU_TOWER_MULTIPLY;
    DynaverseMsmAllocations allocations;
    uint8_t *device_left=nullptr, *device_right=nullptr, *device_output=nullptr;
    uint32_t* device_statuses=nullptr;
    check(allocations.allocate(&device_left,bytes,stream));
    if (binary) check(allocations.allocate(&device_right,bytes,stream));
    check(allocations.allocate(&device_output,bytes,stream));
    check(allocations.allocate(&device_statuses,count*sizeof(uint32_t),stream));
    check(cudaMemcpyAsync(device_left,left,bytes,cudaMemcpyHostToDevice,stream));
    if (binary) check(cudaMemcpyAsync(device_right,right,bytes,cudaMemcpyHostToDevice,stream));
    // One thread owns one bounded field item; no task aggregation or hidden
    // global prepared state. Register/local-memory tuning is a later stage.
    const unsigned blocks = unsigned((count+31)/32);
    switch (degree) {
    case DVGPU_TOWER_FQ2:
        inspect<Fq2><<<blocks,32,0,stream>>>(device_left,device_right,count,operation,power,device_output,device_statuses);
        break;
    case DVGPU_TOWER_FQ6:
        inspect<Fq6><<<blocks,32,0,stream>>>(device_left,device_right,count,operation,power,device_output,device_statuses);
        break;
    case DVGPU_TOWER_FQ12:
        inspect<Fq12><<<blocks,32,0,stream>>>(device_left,device_right,count,operation,power,device_output,device_statuses);
        break;
    default: throw std::invalid_argument("invalid BN254 tower degree");
    }
    check(cudaGetLastError());
    check(cudaMemcpyAsync(pending_output,device_output,bytes,cudaMemcpyDeviceToHost,stream));
    check(cudaMemcpyAsync(pending_statuses,device_statuses,count*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    check(allocations.release(device_left,stream));
    if (binary) check(allocations.release(device_right,stream));
    check(allocations.release(device_output,stream));
    check(allocations.release(device_statuses,stream));
    check(cudaStreamSynchronize(stream));
}
