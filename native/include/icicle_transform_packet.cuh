#pragma once
#include "dynaverse_gpu.h"

// Additive opcode in the existing resident operation layout. a=input,
// b=scratch, c=output; length is the complete FFT size, flags is inverse.
// All three buffers are distinct and owned by the configured resident owner.
enum { DVGPU_RESIDENT_TRANSFORM_PACKET = 17, DVGPU_MAX_TRANSFORM_PACKET_SIZE = 4096 };
