#pragma once
#include <stddef.h>
#include <stdint.h>

enum {
    DVGPU_LINE_MAX_ITEMS=8, DVGPU_LINE_COUNT=91, DVGPU_LINE_BYTES=192,
    DVGPU_LINE_ITEM_BYTES=DVGPU_LINE_COUNT*DVGPU_LINE_BYTES,
    DVGPU_LINE_INFINITY=8
};

#ifdef __cplusplus
extern "C" {
#endif
// Diagnostic only. Input is the retained canonical G2 uncompressed LE128 wire,
// including its all-zero/flagged infinity convention. Device validation uses
// the retained parser, curve test and full INTEGER subgroup order (algorithm 1).
// Each point owns 91 triples of Fq2 (3*64 canonical non-Montgomery LE bytes),
// in ark-ec 0.4.2 G2Prepared order. count=91/status=0 for finite valid points;
// count=0/status=8 for infinity; count=0/status=1,2,4 for invalid items.
// Every unused/rejected coefficient byte is zero. No pairing is evaluated.
// Shapes are checked before CUDA. Nonempty pointers must cover declared lengths;
// the three output regions and error buffer must be mutually disjoint. Results
// publish only after synchronization and resource/stream cleanup succeeds.
int dvgpu_bn254_lines_inspect(void* owner,const uint8_t* points,size_t points_bytes,
    uint8_t* output,size_t output_bytes,uint32_t* counts,size_t counts_count,
    uint32_t* statuses,size_t statuses_count,char* error,size_t error_capacity);
#ifdef __cplusplus
}
#endif
