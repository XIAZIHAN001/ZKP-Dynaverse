#pragma once
#include <stddef.h>
#include <stdint.h>

enum { DVGPU_TOWER_MAX_ITEMS = 64 };
enum { DVGPU_TOWER_FQ2 = 2, DVGPU_TOWER_FQ6 = 6, DVGPU_TOWER_FQ12 = 12 };
enum {
    DVGPU_TOWER_ADD = 1, DVGPU_TOWER_SUBTRACT = 2, DVGPU_TOWER_MULTIPLY = 3,
    DVGPU_TOWER_SQUARE = 4, DVGPU_TOWER_INVERSE = 5, DVGPU_TOWER_FROBENIUS = 6
};
enum {
    DVGPU_TOWER_OK = 0, DVGPU_TOWER_NON_CANONICAL = 1, DVGPU_TOWER_ZERO_INVERSE = 2
};

#ifdef __cplusplus
extern "C" {
#endif
// Explicit arithmetic diagnostic, never a point/protocol validity certificate.
// Canonical non-Montgomery LE32 Fq coefficients, AoS order:
// Fq2=(c0,c1), Fq6=(c0,c1,c2) of Fq2, Fq12=(c0,c1) of Fq6.
// left_bytes must be a multiple of degree*32, at most 64 items. Add/sub/mul
// require equally long right input; unary operations require right_bytes=0.
// Frobenius power is 0..12; other operations require power=0. Output has exactly
// left_bytes bytes, statuses_count equals item count. Rejected items have zero
// output and a nonzero per-item status. Invalid call arguments return -1.
// Host output/status storage must not overlap. Only successful synchronization
// and cleanup publish either host output; execution failure retires the owner.
// All pointers are borrowed synchronously; no pointer survives this call.
int dvgpu_bn254_tower_inspect(void* owner, uint32_t degree, uint32_t operation, uint32_t power,
    const uint8_t* left, size_t left_bytes, const uint8_t* right, size_t right_bytes,
    uint8_t* output, size_t output_bytes, uint32_t* statuses, size_t statuses_count,
    char* error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
