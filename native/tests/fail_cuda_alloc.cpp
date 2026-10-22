// Test-only LD_PRELOAD fixture. Fail exactly one real ICICLE allocation after
// earlier allocations succeed; no hook enters the production bridge or kernel.
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <array>
#include <mutex>

static std::mutex allocation_mutex;
static std::array<void*, 512> live_allocations{};
static void record_allocation(void* pointer) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    for (auto& slot : live_allocations) { if (!slot) { slot = pointer; return; } }
    std::abort();
}
static void record_free(void* pointer) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    for (auto& slot : live_allocations) { if (slot == pointer) { slot = nullptr; return; } }
}
extern "C" cudaError_t cudaFree(void* pointer) {
    using Free = cudaError_t (*)(void*);
    static auto real_free = reinterpret_cast<Free>(dlsym(RTLD_NEXT, "cudaFree"));
    const auto status = real_free ? real_free(pointer) : cudaErrorUnknown;
    if (status == cudaSuccess) record_free(pointer);
    return status;
}
extern "C" cudaError_t cudaFreeAsync(void* pointer, cudaStream_t stream) {
    using Free = cudaError_t (*)(void*, cudaStream_t);
    static auto real_free = reinterpret_cast<Free>(dlsym(RTLD_NEXT, "cudaFreeAsync"));
    const auto status = real_free ? real_free(pointer, stream) : cudaErrorUnknown;
    if (status == cudaSuccess) record_free(pointer);
    return status;
}
__attribute__((destructor)) static void verify_allocation_cleanup() {
    if (std::getenv("DVGPU_TEST_CHECK_ALLOCATIONS")) {
        unsigned live = 0;
        for (auto pointer : live_allocations) { if (pointer) ++live; }
        std::fprintf(stderr, "CUDA_ALLOCATION_BALANCE live=%u\n", live);
        if (live) std::abort();
    }
}

static std::atomic<int> live_events{0};
static std::atomic<unsigned> timing_events{0};
// Route the default timing-event API through the same fixture accounting.
extern "C" cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event,0);
}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    using Create = cudaError_t (*)(cudaEvent_t*, unsigned);
    static auto real_create = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "cudaEventCreateWithFlags"));
    static std::atomic<unsigned> calls{0};
    const char* configured = std::getenv("DVGPU_TEST_FAIL_EVENT_AT");
    const unsigned fail_at = configured ? std::strtoul(configured, nullptr, 10) : 0;
    const unsigned call = ++calls;
    if (fail_at && call == fail_at) {
        *event = nullptr;
        std::fprintf(stderr, "INJECTED_CUDA_EVENT_FAILURE call=%u\n", call);
        return cudaErrorMemoryAllocation;
    }
    if (!real_create) return cudaErrorUnknown;
    const auto status = real_create(event, flags);
    if (status == cudaSuccess) {
        ++live_events;
        if (!(flags & cudaEventDisableTiming)) ++timing_events;
    }
    return status;
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
    using Destroy = cudaError_t (*)(cudaEvent_t);
    static auto real_destroy = reinterpret_cast<Destroy>(dlsym(RTLD_NEXT, "cudaEventDestroy"));
    static std::atomic<unsigned> calls{0};
    const char* configured = std::getenv("DVGPU_TEST_FAIL_EVENT_DESTROY_AT");
    const unsigned fail_at = configured ? std::strtoul(configured, nullptr, 10) : 0;
    if (fail_at && ++calls == fail_at) {
        std::fprintf(stderr, "INJECTED_CUDA_EVENT_DESTROY_FAILURE call=%u\n", fail_at);
        return cudaErrorInvalidValue;
    }
    if (!real_destroy) return cudaErrorUnknown;
    const auto status = real_destroy(event);
    if (status == cudaSuccess) --live_events;
    return status;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    using Record = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    static auto real_record = reinterpret_cast<Record>(dlsym(RTLD_NEXT, "cudaEventRecord"));
    static std::atomic<unsigned> calls{0};
    const char* configured = std::getenv("DVGPU_TEST_FAIL_RECORD_AT");
    const unsigned fail_at = configured ? std::strtoul(configured, nullptr, 10) : 0;
    if (fail_at && ++calls == fail_at) {
        std::fprintf(stderr, "INJECTED_CUDA_RECORD_FAILURE call=%u\n", fail_at);
        return cudaErrorInvalidValue;
    }
    return real_record ? real_record(event, stream) : cudaErrorUnknown;
}

static std::atomic<int> live_streams{0};
extern "C" cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    using Create = cudaError_t (*)(cudaStream_t*);
    static auto real_create = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "cudaStreamCreate"));
    static std::atomic<unsigned> calls{0};
    const char* configured = std::getenv("DVGPU_TEST_FAIL_STREAM_AT");
    const unsigned fail_at = configured ? std::strtoul(configured, nullptr, 10) : 0;
    if (fail_at && ++calls == fail_at) {
        *stream = nullptr;
        std::fprintf(stderr, "INJECTED_CUDA_STREAM_FAILURE call=%u\n", fail_at);
        return cudaErrorMemoryAllocation;
    }
    if (!real_create) return cudaErrorUnknown;
    const auto status = real_create(stream);
    if (status == cudaSuccess) ++live_streams;
    return status;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    using Destroy = cudaError_t (*)(cudaStream_t);
    static auto real_destroy = reinterpret_cast<Destroy>(dlsym(RTLD_NEXT, "cudaStreamDestroy"));
    static std::atomic<unsigned> calls{0};
    const char* configured = std::getenv("DVGPU_TEST_FAIL_STREAM_DESTROY_AT");
    const unsigned fail_at = configured ? std::strtoul(configured, nullptr, 10) : 0;
    if (fail_at && ++calls == fail_at) {
        std::fprintf(stderr, "INJECTED_CUDA_STREAM_DESTROY_FAILURE call=%u\n", fail_at);
        return cudaErrorInvalidValue;
    }
    if (!real_destroy) return cudaErrorUnknown;
    const auto status = real_destroy(stream);
    if (status == cudaSuccess) --live_streams;
    return status;
}
__attribute__((destructor)) static void verify_stream_cleanup() {
    if (std::getenv("DVGPU_TEST_CHECK_STREAMS")) {
        std::fprintf(stderr, "CUDA_STREAM_BALANCE live=%d\n", live_streams.load());
        if (live_streams.load() != 0) std::abort();
    }
}
__attribute__((destructor)) static void verify_event_cleanup() {
    if (std::getenv("DVGPU_TEST_CHECK_EVENTS")) {
        std::fprintf(stderr, "CUDA_EVENT_BALANCE live=%d\n", live_events.load());
        if (live_events.load() != 0) std::abort();
    }
    if (std::getenv("DVGPU_TEST_EXPECT_NO_TIMING_EVENTS")) {
        std::fprintf(stderr, "CUDA_TIMING_EVENT_COUNT count=%u\n", timing_events.load());
        if (timing_events.load() != 0) std::abort();
    }
}

extern "C" cudaError_t cudaMallocAsync(void** pointer, size_t bytes, cudaStream_t stream) {
    using Allocate = cudaError_t (*)(void**, size_t, cudaStream_t);
    static auto real_allocate = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "cudaMallocAsync"));
    static std::atomic<unsigned> calls{0};
    const char* configured = std::getenv("DVGPU_TEST_FAIL_ALLOC_AT");
    const unsigned fail_at = configured ? std::strtoul(configured, nullptr, 10) : 0;
    const unsigned call = ++calls;
    if (fail_at && call == fail_at) {
        *pointer = nullptr;
        std::fprintf(stderr, "INJECTED_CUDA_ALLOCATION_FAILURE call=%u bytes=%zu\n", call, bytes);
        return cudaErrorMemoryAllocation;
    }
    if (!real_allocate) return cudaErrorUnknown;
    const auto status = real_allocate(pointer, bytes, stream);
    if (status == cudaSuccess) record_allocation(*pointer);
    return status;
}
