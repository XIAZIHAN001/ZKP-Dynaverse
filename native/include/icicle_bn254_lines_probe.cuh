#pragma once
#include "dynaverse_bn254_lines.h"
#include <cuda_runtime.h>

// Borrowed exact-size host buffers, staged by the owner thunk. Throws inside
// C++ only; all allocations and asynchronous work are complete before return.
void dvgpu_bn254_lines_run(const uint8_t* points,size_t count,uint8_t* pending,
    uint32_t* pending_counts,uint32_t* pending_statuses,cudaStream_t stream);
