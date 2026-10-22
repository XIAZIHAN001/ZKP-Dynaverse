#pragma once
#include <cuda_runtime.h>
#include <array>

// Own dependency events/side streams created by pinned ICICLE host code.
// Declared before allocation ownership, so async buffers unwind first.
class DynaverseMsmResources {
    std::array<cudaEvent_t, 32> events_{};
    std::array<cudaStream_t, 8> streams_{};
public:
    DynaverseMsmResources() = default;
    DynaverseMsmResources(const DynaverseMsmResources&) = delete;
    DynaverseMsmResources& operator=(const DynaverseMsmResources&) = delete;

    cudaError_t create_event(cudaEvent_t* event, unsigned flags) {
        *event = nullptr;
        for (auto& slot : events_) {
            if (slot) continue;
            const auto status = cudaEventCreateWithFlags(event, flags);
            if (status == cudaSuccess) slot = *event;
            return status;
        }
        return cudaErrorMemoryAllocation;
    }
    cudaError_t destroy_event(cudaEvent_t event) {
        const auto status = cudaEventDestroy(event);
        if (status == cudaSuccess) {
            for (auto& slot : events_) { if (slot == event) { slot = nullptr; break; } }
        }
        return status;
    }
    cudaError_t create_stream(cudaStream_t* stream) {
        *stream = nullptr;
        for (auto& slot : streams_) {
            if (slot) continue;
            const auto status = cudaStreamCreate(stream);
            if (status == cudaSuccess) slot = *stream;
            return status;
        }
        return cudaErrorMemoryAllocation;
    }
    cudaError_t destroy_stream(cudaStream_t stream) {
        const auto status = cudaStreamDestroy(stream);
        if (status == cudaSuccess) {
            for (auto& slot : streams_) { if (slot == stream) { slot = nullptr; break; } }
        }
        return status;
    }
    cudaError_t finish() noexcept {
        bool pending = false;
        for (auto event : events_) { if (event) pending = true; }
        for (auto stream : streams_) { if (stream) pending = true; }
        if (!pending) return cudaSuccess;
        // Drain users on all streams in this owner's isolated context. This
        // is also safe on early ICICLE returns with side-stream work pending.
        const auto synchronized = cudaDeviceSynchronize();
        if (synchronized != cudaSuccess) return synchronized;
        cudaError_t first_error = cudaSuccess;
        for (auto& event : events_) {
            if (!event) continue;
            const auto status = cudaEventDestroy(event);
            if (status == cudaSuccess) event = nullptr;
            else if (first_error == cudaSuccess) first_error = status;
        }
        for (auto& stream : streams_) {
            if (!stream) continue;
            const auto status = cudaStreamDestroy(stream);
            if (status == cudaSuccess) stream = nullptr;
            else if (first_error == cudaSuccess) first_error = status;
        }
        return first_error;
    }
    // The success path checks finish's error. During unwinding, retry only
    // handles not successfully destroyed; the enclosing owner then retires.
    ~DynaverseMsmResources() { finish(); }
};
