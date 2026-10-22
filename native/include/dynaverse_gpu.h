#pragma once
#include <stddef.h>
#include <stdint.h>
enum { DVGPU_MAX_MSM_SIZE = 1 << 20 };
enum { DVGPU_MAX_NTT_SIZE = 1 << 20 };
// Four-step matrices combine small row NTTs with global twiddles/transposes.
// A 2^20 circuit needs a 2n matrix; this does not enlarge the row domain.
enum { DVGPU_MAX_TRANSFORM_MATRIX_SIZE = 1 << 21 };
enum { DVGPU_MAX_MSM_BATCH_SIZE = 64 };
enum { DVGPU_MAX_LARGE_BUCKET_FACTOR = 100 };

#ifdef __cplusplus
extern "C" {
#endif
// Owns an isolated CUDA context on visible device 0. No process-wide reset.
// Zero = success; errors are NUL-terminated when capacity > 0.
// The caller serializes use of one owner and destroys it after all calls finish.
int dvgpu_backend_create(void** backend, char* error, size_t error_capacity);
// Explicit per-owner threshold multiplier: 0 disables large buckets; 1..100
// selects ICICLE's existing path. Auto c, full scalar bits and big triangle
// remain fixed. Invalid values are rejected before CUDA initialization.
int dvgpu_backend_create_with_large_bucket_factor(uint32_t factor, void** backend,
                                                  char* error, size_t error_capacity);
void dvgpu_backend_destroy(void* backend);
// Stable hardware identity captured from visible device 0 during creation.
int dvgpu_backend_device_identity(void* backend, char* output, size_t output_capacity,
                                  char* error, size_t error_capacity);
// Synchronous borrowed buffers; no pointers survive this call. The caller
// supplies length validated canonical BN254 points/scalars. Points = 64-byte
// x||y LE (zero = infinity); scalars = 32-byte LE. result has 64 writable bytes.
// C++ exceptions never cross the ABI. Result is written only on success.
// An execution failure retires this context; use a new backend to recover.
int dvgpu_bn254_msm(void* backend, const uint8_t* points, const uint8_t* scalars, size_t length,
                    uint8_t* result, char* error, size_t error_capacity);
// G2 canonical order = x.c0 || x.c1 || y.c0 || y.c1 (each Fq LE32);
// infinity is all zero. The borrowed-host synchronous contract matches G1.
int dvgpu_bn254_g2_msm(void* backend, const uint8_t* points, const uint8_t* scalars,
    size_t length, uint8_t* result, char* error, size_t error_capacity);
// Natural-order BN254 Fr transform; inverse includes 1/length normalization.
// Values/result are length * 32 canonical LE bytes. root is the canonical
// primitive 2^20-th root used by the CPU reference (same across this owner).
// Domain storage is lazy and belongs to this isolated context. Synchronous;
// output is published only on success. Length must be a power of two <= 2^20.
int dvgpu_bn254_ntt(void* backend, const uint8_t* values, size_t length, int inverse,
                   const uint8_t* root, uint8_t* result, char* error, size_t error_capacity);
// One actual ICICLE batch. Distinct point/scalar sets are contiguous, each
// padded to length; points_size = length * count. count <=64 and total <=2^20.
// Owner input buffers are reused, never caller pointers. All-or-error output
// is count * 64 bytes, published after checked synchronous completion.
int dvgpu_bn254_msm_batch(void* backend, const uint8_t* points, const uint8_t* scalars,
                         size_t length, size_t count, uint8_t* results,
                         char* error, size_t error_capacity);
typedef struct DvgpuBatchStats {
    uint64_t icicle_calls;
    uint64_t buffer_grows;
    uint64_t max_batch_size;
    uint64_t input_capacity;
} DvgpuBatchStats;
// Setup ablations: ordinary multiplication / table / table + batch normalize.
typedef struct DvgpuSetupStats {
    uint64_t table_builds[2], calls[2], points[2], table_bytes;
    // Ordered stream event ranges: table, H2D, generate, normalize, D2H.
    uint64_t device_ns[5], native_wall_ns;
} DvgpuSetupStats;
int dvgpu_setup_mode(void* backend, int mode);
int dvgpu_setup_stats(void* backend, DvgpuSetupStats* stats);
// Standalone research validator: execution success is distinct from acceptance.
// Does not construct typed points or publish SRS. n<=4096; algorithms 1/2/3.
typedef struct DvgpuG2ValidationReport {
    uint64_t checked_points, invalid_points, first_invalid;
    uint32_t failure_mask, algorithm;
} DvgpuG2ValidationReport;
typedef struct DvgpuG2ValidationStats {
    uint64_t parse_curve_device_ns, subgroup_device_ns, reduce_device_ns;
    uint64_t h2d_device_ns, d2h_device_ns, native_wall_ns;
    uint64_t checked_points, accepted_points;
} DvgpuG2ValidationStats;
int dvgpu_bn254_g2_validation_inspect(void* owner,const uint8_t* wire,size_t n,
    uint32_t algorithm,uint32_t* point_masks,DvgpuG2ValidationReport* report,
    char* error,size_t error_capacity);
int dvgpu_g2_validation_stats(void* owner,DvgpuG2ValidationStats* stats);
typedef struct DvgpuG2ValidationTraffic {
    uint64_t scalar_h2d_bytes,point_h2d_bytes,point_d2h_bytes,status_d2h_bytes;
} DvgpuG2ValidationTraffic;
int dvgpu_g2_validation_traffic(void* owner,DvgpuG2ValidationTraffic* stats);
// Validates the existing producer affine buffer. Return -5 publishes only a
// rejection report; output remains untouched. CUDA errors retire the owner.
int dvgpu_bn254_fixed_base_g2_checked(void* owner,const uint8_t* scalars,size_t n,
    uint32_t algorithm,uint8_t* output,DvgpuG2ValidationReport* report,char* error,size_t error_capacity);
// Diagnostics only: actual attempted ICICLE batch calls; successful persistent
// input growth; maximum submitted count and retained point/scalar capacity.
int dvgpu_backend_batch_stats(void* backend, DvgpuBatchStats* stats);
typedef struct DvgpuMemoryInfo { uint64_t free_bytes; uint64_t total_bytes; } DvgpuMemoryInfo;
// Diagnostic device-wide availability in this owner's current CUDA context.
// Includes other GPU users; this is not a per-owner allocation counter.
int dvgpu_backend_memory_info(void* backend, DvgpuMemoryInfo* info,
                              char* error, size_t error_capacity);

// Additive resident ABI. All integer fields are fixed width; borrowed canonical
// host bytes are used only for the synchronous upload/download/MSM call.
// General G1 buffers are projective96. ALLOC_BASES (16) uploads initialized
// immutable affine64 MSM bases (internal kind2); only READ/MSM_G1 accept them.
// Generic write/copy/reverse/transpose/transform/scale reject packed handles.
// Boundary points are x||y64; owner-tagged handles cannot cross owners.
// Transactions save each element on first write only, including overlaps;
// begin/swap do not scan whole buffers. Commit clears touched marker ranges.
enum DvgpuResidentOpcode {
    DVGPU_RESIDENT_ALLOC = 1, DVGPU_RESIDENT_LENGTH, DVGPU_RESIDENT_KIND,
    DVGPU_RESIDENT_BEGIN, DVGPU_RESIDENT_ROLLBACK, DVGPU_RESIDENT_WRITE,
    DVGPU_RESIDENT_READ, DVGPU_RESIDENT_COPY, DVGPU_RESIDENT_REVERSE,
    DVGPU_RESIDENT_TRANSPOSE, DVGPU_RESIDENT_TRANSFORM, DVGPU_RESIDENT_ZIP,
    DVGPU_RESIDENT_SCALE, DVGPU_RESIDENT_MSM_G1, DVGPU_RESIDENT_MSM_G2,
    DVGPU_RESIDENT_ALLOC_BASES
};
// READ kind1 flags=1 opts into device normalization for <=256 projective
// points, using the preallocated owner workspace. Larger reads fall back;
// packed affine bases remain direct copies. Other READ flags are rejected.
enum { DVGPU_READ_DEVICE_AFFINE = 1, DVGPU_READ_DEVICE_AFFINE_MAX = 256 };
typedef struct DvgpuResidentOp {
    uint32_t opcode, kind, flags, reserved;
    uint64_t a, b, c;
    uint64_t a_offset, b_offset, c_offset, length, rows, cols;
    const uint8_t* input;
    uint8_t* output;
} DvgpuResidentOp;
typedef struct DvgpuResidentStats {
    uint64_t h2d_bytes, d2h_bytes, bytes_allocated;
    uint64_t fr_transform_elements, g1_transform_elements;
    uint64_t msm_g1_points, msm_g2_points, calls;
} DvgpuResidentStats;
typedef struct DvgpuResidentDiagnostics {
    uint64_t resident_allocations, temporary_allocations, temporary_peak_bytes;
    uint64_t journal_saved_bytes, journal_saved_ranges, journal_peak_bytes;
    uint64_t device_operation_ns, device_operations;
    uint64_t physical_h2d_bytes, physical_d2h_bytes, explicit_d2d_bytes;
    uint64_t journal_metadata_capacity_bytes;
} DvgpuResidentDiagnostics;
int dvgpu_resident_configure(void* backend, uint64_t memory_limit_bytes,
    uint64_t journal_bytes, const uint8_t* root, char* error, size_t error_capacity);
int dvgpu_resident_call(void* backend, const DvgpuResidentOp* operation,
    uint64_t* result, char* error, size_t error_capacity);
// Both operations publish only host metadata and cannot fail or allocate.
void dvgpu_resident_seal(void* backend);
void dvgpu_resident_commit(void* backend);
int dvgpu_resident_stats(void* backend, DvgpuResidentStats* stats);
int dvgpu_resident_diagnostics(void* backend, DvgpuResidentDiagnostics* stats);
// Separate diagnostic ABI; existing resident layouts remain unchanged.
// event_mask:1 journal,2 operation,4 total. Event spans are stream envelopes,
// not pure kernel busy time and never additive to host clocks. wait_host_ns
// counts explicit bridge waits; ICICLE internal waits remain in operation_host.
typedef struct DvgpuResidentProfile {
    uint64_t native_wall_ns, journal_host_ns, conversion_host_ns;
    uint64_t operation_host_ns, wait_host_ns;
    uint64_t event_total_ns, event_journal_ns, event_operation_ns, event_mask;
    uint64_t h2d_bytes, d2h_bytes, d2d_bytes;
} DvgpuResidentProfile;
int dvgpu_resident_call_profiled(void* backend,const DvgpuResidentOp* operation,
    uint64_t* result,DvgpuResidentProfile* profile,char* error,size_t error_capacity);
// Optional host-clock diagnostic spans, in nanoseconds. icicle_ns includes
// ICICLE allocation, transfers, kernels and synchronization, not kernel alone.
typedef struct DvgpuMsmTimings {
    uint64_t conversion_ns;
    uint64_t icicle_ns;
    uint64_t output_ns;
} DvgpuMsmTimings;
int dvgpu_bn254_msm_profiled(void* backend, const uint8_t* points, const uint8_t* scalars,
                            size_t length, uint8_t* result, DvgpuMsmTimings* timings,
                            char* error, size_t error_capacity);
// Separate diagnostic ABI. Wall and event arrays are different clock domains,
// not additive costs. Phase order is defined in icicle_msm_timeline.cuh and
// mirrored by Rust MsmTimeline. label is borrowed only during this call.
enum { DVGPU_TIMELINE_PHASES = 9, DVGPU_TIMELINE_KERNELS = 10 };
typedef struct DvgpuMsmTimeline {
    uint64_t wall_ns[DVGPU_TIMELINE_PHASES];
    uint64_t event_ns[DVGPU_TIMELINE_PHASES];
    uint64_t calls[DVGPU_TIMELINE_PHASES];
    uint64_t event_calls[DVGPU_TIMELINE_PHASES];
    uint64_t bytes[DVGPU_TIMELINE_PHASES];
    uint64_t kernel_ns[DVGPU_TIMELINE_KERNELS];
    uint64_t kernel_calls[DVGPU_TIMELINE_KERNELS];
    uint64_t instrumentation_inclusive_ns;
    uint64_t instrumentation_total_ns;
    uint64_t c;
    uint64_t bitsize;
    uint64_t windows;
    uint64_t buckets;
} DvgpuMsmTimeline;
int dvgpu_bn254_msm_timeline(void* backend, const uint8_t* points, const uint8_t* scalars,
                            size_t length, uint8_t* result, const char* label,
                            DvgpuMsmTimings* timings, DvgpuMsmTimeline* timeline,
                            char* error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
