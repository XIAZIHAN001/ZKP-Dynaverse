#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstdint>

// A serial resident submission binds optional diagnostics. Ordinary legacy
// calls do not read clocks or update counters. The bound record outlives all
// allocation guards in that synchronous submission.
struct DvgpuTemporaryAllocationStats {
    uint64_t calls = 0, live_bytes = 0, peak_bytes = 0;
    uint64_t limit_bytes = UINT64_MAX;
};
inline thread_local DvgpuTemporaryAllocationStats* dvgpu_temporary_allocations = nullptr;

// Host resource bookkeeping around the pinned ICICLE implementation. Kernel
// bodies and arithmetic are unchanged. Early CUDA error returns still unwind
// this guard, unlike ICICLE v2.8's raw allocation locals.
class DynaverseMsmAllocations {
    std::array<void*, 64> live_{};
    std::array<size_t, 64> sizes_{};
    DvgpuTemporaryAllocationStats* stats_ = dvgpu_temporary_allocations;
public:
    DynaverseMsmAllocations() = default;
    DynaverseMsmAllocations(const DynaverseMsmAllocations&) = delete;
    DynaverseMsmAllocations& operator=(const DynaverseMsmAllocations&) = delete;

    template <typename T>
    cudaError_t allocate(T** pointer, size_t bytes, cudaStream_t stream) {
        if (stats_ && (stats_->live_bytes > stats_->limit_bytes ||
            bytes > stats_->limit_bytes - stats_->live_bytes)) return cudaErrorMemoryAllocation;
        for (size_t i = 0; i < live_.size(); ++i) {
            auto& slot = live_[i];
            if (slot) continue;
            const auto status = cudaMallocAsync(reinterpret_cast<void**>(pointer), bytes, stream);
            if (status == cudaSuccess) {
                slot = *pointer;
                sizes_[i] = bytes;
                if (stats_) {
                    ++stats_->calls;
                    stats_->live_bytes += bytes;
                    if (stats_->live_bytes > stats_->peak_bytes) stats_->peak_bytes = stats_->live_bytes;
                }
            }
            return status;
        }
        // Bounded bookkeeping cannot itself allocate or throw after a device
        // allocation. ICICLE has fewer than 64 simultaneously live buffers.
        return cudaErrorMemoryAllocation;
    }

    cudaError_t release(void* pointer, cudaStream_t stream) {
        const auto status = cudaFreeAsync(pointer, stream);
        if (status == cudaSuccess) {
            for (size_t i = 0; i < live_.size(); ++i) {
                if (live_[i] == pointer) {
                    if (stats_) stats_->live_bytes -= sizes_[i];
                    live_[i] = nullptr; sizes_[i] = 0; break;
                }
            }
        }
        return status;
    }

    // Transfer a persistent allocation to the enclosing owner. Used only by
    // the NTT twiddle generator once it has published the internal table.
    void detach(void* pointer) noexcept {
        for (size_t i = 0; i < live_.size(); ++i) {
            if (live_[i] == pointer) {
                if (stats_) stats_->live_bytes -= sizes_[i];
                live_[i] = nullptr; sizes_[i] = 0; break;
            }
        }
    }

    ~DynaverseMsmAllocations() {
        bool pending = false;
        for (auto pointer : live_) { if (pointer) { pending = true; break; } }
        if (!pending) return;
        // An allocation can be referenced from ICICLE's internal side streams.
        // Synchronize this backend's isolated context before freeing leftovers.
        // cudaFree accepts async allocations after their users complete.
        cudaDeviceSynchronize();
        for (size_t i = 0; i < live_.size(); ++i) {
            if (live_[i]) {
                cudaFree(live_[i]);
                if (stats_) stats_->live_bytes -= sizes_[i];
            }
        }
    }
};
