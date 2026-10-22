#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct DvgpuG2KernelConfig {uint32_t threads,schedule,layout,fused;} DvgpuG2KernelConfig;
typedef struct DvgpuG2ExperimentStats {uint64_t repack_device_ns,subgroup_kernel_ns,logical_repack_bytes;} DvgpuG2ExperimentStats;
typedef struct DvgpuG2KernelResources {
    uint32_t registers,block_threads,active_blocks,max_threads_per_sm;
    uint64_t local_bytes,shared_bytes;
    uint32_t normalize_registers,normalize_active_blocks;
    uint64_t normalize_local_bytes,normalize_shared_bytes;
} DvgpuG2KernelResources;
int dvgpu_g2_kernel_config_set(void* owner,const DvgpuG2KernelConfig* config);
int dvgpu_g2_kernel_config_get(void* owner,DvgpuG2KernelConfig* config);
int dvgpu_g2_kernel_resources(void* owner,uint32_t algorithm,DvgpuG2KernelResources* output);
int dvgpu_g2_experiment_stats(void* owner,DvgpuG2ExperimentStats* output);
#ifdef __cplusplus
}
#endif
