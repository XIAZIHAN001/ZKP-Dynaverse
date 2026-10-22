#pragma once
#include "dynaverse_bn254_tower.h"
#include <cuda_runtime.h>

// Internal synchronous run helper; caller owns current context, stream, budget
// and unpublished host staging. No pointer is retained on success or failure.
void dvgpu_bn254_tower_run(const uint8_t* left, const uint8_t* right, size_t count,
    uint32_t degree, uint32_t operation, uint32_t power,
    uint8_t* pending_output, uint32_t* pending_statuses, cudaStream_t stream);
