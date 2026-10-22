#pragma once
#include "dynaverse_gpu.h"
#include "icicle_msm_allocations.cuh"
#include <nvtx3/nvToolsExt.h>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

// Diagnostics only: no hook is called from the ordinary ICICLE instantiation.
enum DvgpuTimelinePhase {
    AllocationPreparation, PointH2D, ScalarH2D, KernelSubmission,
    Synchronization, OutputD2H, RemainingOverhead, MetadataD2H, Free
};
inline constexpr const char* dvgpu_phase_names[] = {
    "allocation_preparation", "point_h2d", "scalar_h2d", "kernel_submission",
    "synchronization", "output_d2h", "remaining_overhead", "metadata_d2h", "free"
};
inline void dvgpu_event_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
class DvgpuNvtxRange {
    bool open_ = true;
public:
    explicit DvgpuNvtxRange(const char* label) { nvtxRangePushA(label); }
    void end() noexcept { if (open_) { nvtxRangePop(); open_ = false; } }
    ~DvgpuNvtxRange() { end(); }
};

class DvgpuTimelineRecorder {
    using Clock = std::chrono::steady_clock;
    struct Pair { cudaEvent_t start = nullptr, end = nullptr; int phase = 0, kernel = -1; };
    // Large-bucket reduction adds a variable number of measured scopes.
    // Allocate stable storage before creating CUDA handles, so growth failures
    // leave every previously acquired (including partial) pair owned here.
    std::deque<Pair> events_;
    uint64_t control_ns_ = 0;
    DvgpuMsmTimeline& output_;
    static uint64_t ns(Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
public:
    cudaStream_t main_stream;
    explicit DvgpuTimelineRecorder(DvgpuMsmTimeline& output, cudaStream_t stream)
        : output_(output), main_stream(stream) {}
    DvgpuTimelineRecorder(const DvgpuTimelineRecorder&) = delete;
    DvgpuTimelineRecorder& operator=(const DvgpuTimelineRecorder&) = delete;

    void configure(unsigned bitsize, unsigned c, unsigned precompute, unsigned batch) {
        output_.bitsize = bitsize;
        output_.c = c;
        output_.windows = (((bitsize + c - 1) / c - 1) / precompute) + 1;
        output_.buckets = ((output_.windows << c) - output_.windows) * batch;
    }

    template <typename Action>
    cudaError_t measure(int phase, cudaStream_t stream, size_t bytes, bool use_events,
                        const char* label, int kernel, Action action) {
        const auto before = Clock::now();
        DvgpuNvtxRange range(label);
        Pair* pair = nullptr;
        if (use_events) {
            events_.emplace_back();
            pair = &events_.back(); // Own even a partially created pair on error.
            pair->phase = phase;
            pair->kernel = kernel;
            ++output_.event_calls[phase];
            dvgpu_event_check(cudaEventCreateWithFlags(&pair->start, cudaEventDefault));
            dvgpu_event_check(cudaEventCreateWithFlags(&pair->end, cudaEventDefault));
            dvgpu_event_check(cudaEventRecord(pair->start, stream));
        }
        const auto started = Clock::now();
        control_ns_ += ns(before, started);
        const auto status = action();
        const auto finished = Clock::now();
        output_.wall_ns[phase] += ns(started, finished);
        ++output_.calls[phase];
        output_.bytes[phase] += bytes;
        if (kernel >= 0) ++output_.kernel_calls[kernel];
        if (pair && status == cudaSuccess) dvgpu_event_check(cudaEventRecord(pair->end, stream));
        range.end();
        control_ns_ += ns(finished, Clock::now());
        return status;
    }

    void finish(uint64_t inclusive_ns) {
        output_.instrumentation_inclusive_ns = control_ns_;
        uint64_t accounted = control_ns_;
        for (int i = 0; i < DVGPU_TIMELINE_PHASES; ++i) accounted += output_.wall_ns[i];
        if (accounted > inclusive_ns) throw std::runtime_error("timeline wall scopes overlap");
        output_.wall_ns[RemainingOverhead] = inclusive_ns - accounted;
        const auto started = Clock::now();
        // Original ICICLE and bridge synchronization have completed. No new
        // phase barrier or cudaEventSynchronize is inserted by diagnostics.
        for (auto& pair : events_) {
            float milliseconds = 0;
            dvgpu_event_check(cudaEventElapsedTime(&milliseconds, pair.start, pair.end));
            const auto elapsed = uint64_t(std::llround(double(milliseconds) * 1000000.0));
            output_.event_ns[pair.phase] += elapsed;
            if (pair.kernel >= 0) output_.kernel_ns[pair.kernel] += elapsed;
            dvgpu_event_check(cudaEventDestroy(pair.start));
            pair.start = nullptr;
            dvgpu_event_check(cudaEventDestroy(pair.end));
            pair.end = nullptr;
        }
        control_ns_ += ns(started, Clock::now());
        output_.instrumentation_total_ns = control_ns_;
    }

    ~DvgpuTimelineRecorder() {
        // On an error the ICICLE allocation guard and bridge stream guard
        // drain borrowed work, and the backend context is then retired.
        for (auto& pair : events_) {
            if (pair.start) cudaEventDestroy(pair.start);
            if (pair.end) cudaEventDestroy(pair.end);
        }
    }
};

// Scope-local dispatch lets extracted host orchestration use small wrappers
// without modifying the upstream config/ABI or retaining pointers afterward.
inline thread_local DvgpuTimelineRecorder* dvgpu_active_timeline = nullptr;
class DvgpuTimelineBinding {
public:
    explicit DvgpuTimelineBinding(DvgpuTimelineRecorder& recorder) {
        if (dvgpu_active_timeline) throw std::runtime_error("nested MSM timeline");
        dvgpu_active_timeline = &recorder;
    }
    ~DvgpuTimelineBinding() { dvgpu_active_timeline = nullptr; }
};
inline DvgpuTimelineRecorder& dvgpu_timeline() {
    if (!dvgpu_active_timeline) throw std::runtime_error("missing MSM timeline owner");
    return *dvgpu_active_timeline;
}

class DynaverseTimelineAllocations {
    DynaverseMsmAllocations allocations_;
public:
    template <typename T>
    cudaError_t allocate(T** pointer, size_t bytes, cudaStream_t stream) {
        return dvgpu_timeline().measure(AllocationPreparation, stream, bytes, true,
            "DVGPU/allocation", -1, [&] { return allocations_.allocate(pointer, bytes, stream); });
    }
    cudaError_t release(void* pointer, cudaStream_t stream) {
        return dvgpu_timeline().measure(Free, stream, 0, true,
            "DVGPU/free", -1, [&] { return allocations_.release(pointer, stream); });
    }
};
inline cudaError_t dvgpu_timeline_copy(void* destination, const void* source, size_t bytes,
                                      cudaMemcpyKind kind, cudaStream_t stream) {
    auto& recorder = dvgpu_timeline();
    int phase;
    if (kind == cudaMemcpyHostToDevice) phase = stream == recorder.main_stream ? ScalarH2D : PointH2D;
    else if (kind == cudaMemcpyDeviceToHost) phase = bytes == 4 ? MetadataD2H : OutputD2H;
    else throw std::runtime_error("unexpected timeline memcpy kind");
    return recorder.measure(phase, stream, bytes, true, dvgpu_phase_names[phase], -1,
        [&] { return cudaMemcpyAsync(destination, source, bytes, kind, stream); });
}
inline cudaError_t dvgpu_timeline_synchronize(cudaStream_t stream) {
    return dvgpu_timeline().measure(Synchronization, stream, 0, false, "DVGPU/synchronize", -1,
        [&] { return cudaStreamSynchronize(stream); });
}
inline cudaError_t dvgpu_timeline_wait_event(cudaStream_t stream, cudaEvent_t event) {
    return dvgpu_timeline().measure(Synchronization, stream, 0, false, "DVGPU/stream_wait_event", -1,
        [&] { return cudaStreamWaitEvent(stream, event); });
}
inline cudaError_t dvgpu_timeline_stream_create(DynaverseMsmResources& resources, cudaStream_t* stream) {
    return dvgpu_timeline().measure(AllocationPreparation, nullptr, 0, false, "DVGPU/stream_create", -1,
        [&] { return resources.create_stream(stream); });
}
inline cudaError_t dvgpu_timeline_stream_destroy(DynaverseMsmResources& resources, cudaStream_t stream) {
    return dvgpu_timeline().measure(AllocationPreparation, stream, 0, false, "DVGPU/stream_destroy", -1,
        [&] { return resources.destroy_stream(stream); });
}
inline cudaError_t dvgpu_timeline_memset(void* pointer, int value, size_t bytes) {
    return dvgpu_timeline().measure(AllocationPreparation, nullptr, bytes, true, "DVGPU/memset", -1,
        [&] { return cudaMemset(pointer, value, bytes); });
}
inline cudaError_t dvgpu_timeline_memset_async(void* pointer, int value, size_t bytes, cudaStream_t stream) {
    return dvgpu_timeline().measure(AllocationPreparation, stream, bytes, true, "DVGPU/memset_async", -1,
        [&] { return cudaMemsetAsync(pointer, value, bytes, stream); });
}
template <typename Action>
void dvgpu_timeline_kernel(const char* name, int kernel, cudaStream_t stream, Action action) {
    dvgpu_event_check(dvgpu_timeline().measure(KernelSubmission, stream, 0, true, name, kernel,
        [&] { action(); return cudaSuccess; }));
}
template <typename Action>
cudaError_t dvgpu_timeline_cub(const char* name, int kernel, cudaStream_t stream, bool execute, Action action) {
    const auto phase = execute ? KernelSubmission : AllocationPreparation;
    return dvgpu_timeline().measure(phase, stream, 0, execute, name, execute ? kernel : -1, action);
}
