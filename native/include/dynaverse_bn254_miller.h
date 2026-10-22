#pragma once
#include <stddef.h>
#include <stdint.h>
enum {
    DVGPU_MILLER_LINES=91, DVGPU_MILLER_FQ12_BYTES=384,
    DVGPU_MILLER_OUTPUT_BYTES=(1+2*DVGPU_MILLER_LINES)*DVGPU_MILLER_FQ12_BYTES,
    DVGPU_MILLER_METADATA_COUNT=3, DVGPU_MILLER_INFINITY=8
};
#ifdef __cplusplus
extern "C" {
#endif
// Diagnostic only: exactly one canonical uncompressed G1 LE64 and G2 LE128.
// Retains legacy all-zero, flagged infinity and uncompressed sign semantics.
// Checks both points even if either is infinity. G1 has cofactor one; G2 uses
// retained full INTEGER r subgroup check. metadata = {line count,G1 mask,G2 mask}.
// Masks: 0 finite; 1 encoding; 2 curve; G2-only 4 subgroup; 8 accepted infinity.
// output: final Fq12, 91 evaluated Fq12 lines, 91 post-line Fq12 accumulators.
// Each Fq12 is nested c0/c1 (Fq6 c0/c1/c2, Fq2 c0/c1), canonical LE32 Fq.
// Finite accepted pair: count=91. Valid infinity pair: count=0, final=one,
// traces zero. Any rejected point: count=0, all output bytes zero.
// Shape/pointer/quota errors leave both destinations unchanged. Execution or
// cleanup failures retire the owner and publish neither destination. Borrowed
// input/output/error regions must be disjoint and cover their declared lengths.
// No final exponentiation, multi-pair product or protocol validity certificate.
int dvgpu_bn254_miller_inspect(void* owner,const uint8_t* p,size_t p_bytes,
    const uint8_t* q,size_t q_bytes,uint8_t* output,size_t output_bytes,
    uint32_t* metadata,size_t metadata_count,char* error,size_t error_capacity);
#ifdef __cplusplus
}
#endif
