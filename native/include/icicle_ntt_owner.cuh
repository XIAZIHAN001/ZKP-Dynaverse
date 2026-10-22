#pragma once
#include "api/bn254.h"

// Internal C++ boundary. All operations, including destruction, run while
// the owning isolated context is current. A state belongs to exactly one owner.
struct DvgpuNttState;
DvgpuNttState* dvgpu_ntt_create();
void dvgpu_ntt_destroy(DvgpuNttState* state, bool context_current) noexcept;
cudaError_t dvgpu_ntt_run(DvgpuNttState* state, bn254::scalar_t root,
    const bn254::scalar_t* input, int length, bool inverse,
    device_context::DeviceContext& context, bn254::scalar_t* output);
cudaError_t dvgpu_ntt_prepare(DvgpuNttState* state, bn254::scalar_t root,
    device_context::DeviceContext& context);
cudaError_t dvgpu_ntt_rows(DvgpuNttState* state, const bn254::scalar_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::scalar_t* output);
// Pinned curve parameters have internal linkage; C linkage keeps identical
// projective layouts callable across translation units without type-name drift.
extern "C" cudaError_t dvgpu_ecntt_rows(DvgpuNttState* state, const bn254::projective_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::projective_t* output);
// Packet-only variants enqueue device-to-device rows on the owner stream.
// The enclosing resident call must synchronize before publishing success.
cudaError_t dvgpu_ntt_rows_async(DvgpuNttState* state, const bn254::scalar_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::scalar_t* output);
extern "C" cudaError_t dvgpu_ecntt_rows_async(DvgpuNttState* state, const bn254::projective_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::projective_t* output);
cudaError_t dvgpu_ntt_four_step_root(DvgpuNttState* state, int length, bool inverse,
    bn254::scalar_t* output);
