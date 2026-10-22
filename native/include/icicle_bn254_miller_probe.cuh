#pragma once
#include "dynaverse_bn254_miller.h"
#include <cuda_runtime.h>
void dvgpu_bn254_miller_run(const uint8_t* p,const uint8_t* q,uint8_t* pending,
    uint32_t* metadata,cudaStream_t stream);
