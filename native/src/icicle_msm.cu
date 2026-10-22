#include "dynaverse_gpu.h"
#include "api/bn254.h"
#include "msm/msm.cuh"
#include "icicle_msm_allocations.cuh"
#include "icicle_msm_resources.cuh"
#include "icicle_msm_with_cleanup.cuh"
#include "icicle_msm_timeline.cuh"
#include "icicle_msm_with_timeline.cuh"
#include "icicle_ntt_owner.cuh"
#include "icicle_resident.cuh"
#include "icicle_transform_packet.cuh"
#include "icicle_setup.cuh"
#include "icicle_g2_validation.cuh"
#include "icicle_bn254_tower_probe.cuh"
#include "icicle_bn254_lines_probe.cuh"
#include "icicle_bn254_miller_probe.cuh"
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bn254;
static_assert(sizeof(scalar_t) == 32 && sizeof(affine_t) == 64,
              "ICICLE BN254 field/point layout changed");

extern "C" cudaError_t dvgpu_g1_msm_run(const scalar_t* scalars,
    const affine_t* points, int length, msm::MSMConfig* config, projective_t* output) {
    return msm::msm<scalar_t, affine_t, projective_t>(scalars, points, length, *config, output);
}

namespace {
int fail(int status, const char* message, char* error, size_t capacity) {
    if (error && capacity) {
        const size_t count = std::min(capacity - 1, std::strlen(message));
        std::memcpy(error, message, count);
        error[count] = '\0';
    }
    return status;
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_driver(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* message = "CUDA driver error";
        cuGetErrorString(status, &message);
        throw std::runtime_error(message);
    }
}
struct BatchBuffers {
    scalar_t* scalars = nullptr;
    affine_t* points = nullptr;
    size_t capacity = 0;
    std::vector<scalar_t> host_scalars;
    std::vector<affine_t> host_points;

    void reserve(size_t length, cudaStream_t stream, DvgpuBatchStats& stats) {
        host_scalars.resize(length);
        host_points.resize(length);
        if (length <= capacity) return;
        DynaverseMsmAllocations pending;
        scalar_t* next_scalars = nullptr;
        affine_t* next_points = nullptr;
        check(pending.allocate(&next_scalars, length * sizeof(scalar_t), stream));
        check(pending.allocate(&next_points, length * sizeof(affine_t), stream));
        // Earlier calls completed before reusing this owner. Clear each old
        // pointer immediately after its successful free, including on errors.
        if (scalars) { check(cudaFreeAsync(scalars, stream)); scalars = nullptr; }
        if (points) { check(cudaFreeAsync(points, stream)); points = nullptr; }
        scalars = next_scalars;
        points = next_points;
        pending.detach(scalars);
        pending.detach(points);
        capacity = length;
        ++stats.buffer_grows;
        stats.input_capacity = capacity;
    }
    void release_device() noexcept {
        cudaDeviceSynchronize();
        if (scalars) cudaFree(scalars);
        if (points) cudaFree(points);
        scalars = nullptr;
        points = nullptr;
    }
};
struct Backend {
    CUcontext context = nullptr;
    uint32_t large_bucket_factor = 0;
    DvgpuNttState* ntt = nullptr;
    DvgpuResidentState* resident = nullptr;
    DvgpuResidentStats retired_resident_stats{};
    DvgpuResidentDiagnostics retired_resident_diagnostics{};
    std::string device_identity;
    BatchBuffers* batch = nullptr;
    DvgpuBatchStats batch_stats{};
    DvgpuSetupState setup;
    DvgpuG2ValidationStats g2_validation{};
    DvgpuG2ValidationTraffic g2_traffic{};
    DvgpuG2KernelConfig g2_kernel{128,0,0,0};
    DvgpuG2ExperimentStats g2_experiment{};
    int64_t g2_test_corrupt_next=-1; // Only explicit test ABI can set this.
    uint64_t external_bytes() const {
        return (batch ? batch->capacity*96 : 0)+setup.stats.table_bytes;
    }
    void retire() noexcept {
        if (context) {
            // The allocation guard frees async buffers first. Context teardown
            // also retires any inaccessible CUDA state after
            // an error, without resetting other libraries' device state.
            const auto doomed = context;
            context = nullptr;
            // Domain pointers belong to this context, including managed
            // twiddles. Release them before retiring the context itself.
            if (ntt || batch || resident || setup.g1 || setup.g2) {
                const bool current = cuCtxPushCurrent(doomed) == CUDA_SUCCESS;
                setup.release(current);
                if (resident) {
                    retired_resident_stats = dvgpu_resident_get_stats(resident);
                    retired_resident_diagnostics = dvgpu_resident_get_diagnostics(resident);
                    dvgpu_resident_destroy(resident, current); resident = nullptr;
                }
                if (ntt) dvgpu_ntt_destroy(ntt, current);
                if (batch) {
                    if (current) batch->release_device();
                    delete batch;
                    batch = nullptr;
                }
                if (current) {
                    CUcontext popped;
                    cuCtxPopCurrent(&popped);
                }
                ntt = nullptr;
            }
            cuCtxDestroy(doomed);
        }
    }
    ~Backend() { retire(); }
};
struct ContextScope {
    explicit ContextScope(CUcontext context) { check_driver(cuCtxPushCurrent(context)); }
    ~ContextScope() { CUcontext popped; cuCtxPopCurrent(&popped); }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
};
// Legacy users keep their previous behavior. A resident-configured owner
// shares one persistent/scratch cap across foreground and rebuilding work.
struct ForegroundBudget {
    DvgpuTemporaryAllocationStats* previous;
    explicit ForegroundBudget(Backend& backend, uint64_t pending, uint64_t scratch)
        : previous(dvgpu_temporary_allocations) {
        dvgpu_resident_preflight_external(backend.resident, pending, scratch);
        dvgpu_temporary_allocations = dvgpu_resident_temporary(backend.resident);
    }
    ~ForegroundBudget() { dvgpu_temporary_allocations = previous; }
};
// Explicit stream ownership. On either success or error, outstanding work is
// completed before borrowed host vectors leave scope.
struct Stream {
    cudaStream_t value = nullptr;
    // ICICLE v2.8's bucket reduction also launches initialization kernels on
    // the legacy default stream (msm.cu). A blocking stream preserves that
    // dependency; cudaStreamNonBlocking races with those initializations.
    Stream() { check(cudaStreamCreate(&value)); }
    void finish() {
        check(cudaStreamSynchronize(value));
        check(cudaStreamDestroy(value));
        value = nullptr;
    }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
uint32_t read_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
void write_u32(uint8_t* p, uint32_t value) {
    for (int j = 0; j < 4; ++j) p[j] = uint8_t(value >> (8 * j));
}
}

extern "C" int dvgpu_backend_create_with_large_bucket_factor(uint32_t factor, void** output, char* error, size_t capacity) {
    if (!output) return fail(-1, "null backend output", error, capacity);
    *output = nullptr;
    if (factor > DVGPU_MAX_LARGE_BUCKET_FACTOR)
        return fail(-1, "large bucket factor must be within 0..=100", error, capacity);
    try {
        check_driver(cuInit(0));
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if (!count) throw std::runtime_error("no visible CUDA device");
        // Map visible runtime device 0 to the corresponding driver device;
        // CUDA_VISIBLE_DEVICES can change the runtime ordinal ordering.
        char pci_bus_id[32];
        check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        CUdevice device;
        check_driver(cuDeviceGetByPCIBusId(&device, pci_bus_id));
        auto backend = std::make_unique<Backend>();
        backend->large_bucket_factor = factor;
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        backend->device_identity = std::string(properties.name) + "/sm_" +
            std::to_string(properties.major) + std::to_string(properties.minor);
        check_driver(cuCtxCreate(&backend->context, CU_CTX_SCHED_AUTO, device));
        CUcontext popped;
        check_driver(cuCtxPopCurrent(&popped));
        {
            ContextScope scope(backend->context);
            check(cudaFree(nullptr));
        }
        *output = backend.release();
        return 0;
    } catch (const std::exception& e) { return fail(-2, e.what(), error, capacity); }
      catch (...) { return fail(-3, "unknown native exception", error, capacity); }
}

extern "C" int dvgpu_backend_create(void** output, char* error, size_t capacity) {
    return dvgpu_backend_create_with_large_bucket_factor(0, output, error, capacity);
}

extern "C" void dvgpu_backend_destroy(void* backend) {
    delete static_cast<Backend*>(backend);
}

extern "C" int dvgpu_backend_device_identity(void* opaque, char* output, size_t output_capacity,
    char* error, size_t capacity) {
    const auto* backend = static_cast<const Backend*>(opaque);
    if (!backend || !output || output_capacity <= backend->device_identity.size())
        return fail(-1, "invalid device identity output", error, capacity);
    std::memcpy(output, backend->device_identity.c_str(), backend->device_identity.size() + 1);
    return 0;
}

extern "C" int dvgpu_backend_batch_stats(void* opaque, DvgpuBatchStats* stats) {
    const auto* backend = static_cast<const Backend*>(opaque);
    if (!backend || !stats) return -1;
    *stats = backend->batch_stats;
    return 0;
}

extern "C" int dvgpu_backend_memory_info(void* opaque, DvgpuMemoryInfo* info, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context)
        return fail(-4, "backend context retired after an error; create a new backend", error, capacity);
    if (!info) return fail(-1, "null memory info output", error, capacity);
    try {
        ContextScope scope(backend->context);
        size_t free_bytes, total_bytes;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        *info = {uint64_t(free_bytes), uint64_t(total_bytes)};
        return 0;
    } catch (const std::exception& e) {
        backend->retire();
        return fail(-2, e.what(), error, capacity);
    } catch (...) {
        backend->retire();
        return fail(-3, "unknown native exception", error, capacity);
    }
}

extern "C" int dvgpu_resident_configure(void* opaque, uint64_t limit, uint64_t journal,
    const uint8_t* root_bytes, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4, "backend context retired", error, capacity);
    if (!root_bytes || backend->resident) return fail(-1, "resident configuration already set or root absent", error, capacity);
    try {
        ContextScope scope(backend->context);
        scalar_t root;
        for (int j = 0; j < 8; ++j) root.limbs_storage.limbs[j] = read_u32(root_bytes + 4 * j);
        scalar_t power = root;
        for (int i = 0; i < 19; ++i) power = scalar_t::sqr(power);
        if (power == scalar_t::one() || !(scalar_t::sqr(power) == scalar_t::one()))
            return fail(-1, "resident NTT root must have order 2^20", error, capacity);
        if (!backend->ntt) backend->ntt = dvgpu_ntt_create();
        // Validate the whole candidate before publishing it. A quota refusal
        // must permit the legacy owner to retry with a larger configuration.
        auto candidate = std::unique_ptr<DvgpuResidentState, void(*)(DvgpuResidentState*)>(
            dvgpu_resident_create(limit, journal, root, backend->ntt),
            [](DvgpuResidentState* state) { dvgpu_resident_destroy(state, true); });
        dvgpu_resident_set_external(candidate.get(),backend->external_bytes());
        dvgpu_resident_preflight_external(candidate.get(), 0, 64ull << 20);
        backend->resident = candidate.release();
        return 0;
    } catch (const std::invalid_argument& e) { return fail(-1, e.what(), error, capacity); }
      catch (const std::exception& e) { backend->retire(); return fail(-2, e.what(), error, capacity); }
      catch (...) { backend->retire(); return fail(-3, "unknown resident configuration exception", error, capacity); }
}
extern "C" int dvgpu_resident_call(void* opaque, const DvgpuResidentOp* operation,
    uint64_t* result, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4, "backend context retired", error, capacity);
    if (!operation || !result || !backend->resident) return fail(-1, "invalid resident call", error, capacity);
    try {
        ContextScope scope(backend->context);
        *result = dvgpu_resident_execute(backend->resident, *operation);
        return 0;
    } catch (const std::invalid_argument& e) { return fail(-1, e.what(), error, capacity); }
      catch (const std::exception& e) { backend->retire(); return fail(-2, e.what(), error, capacity); }
      catch (...) { backend->retire(); return fail(-3, "unknown resident execution exception", error, capacity); }
}
extern "C" int dvgpu_resident_call_profiled(void* opaque,const DvgpuResidentOp* operation,
    uint64_t* result,DvgpuResidentProfile* profile,char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    if (!operation || !result || !profile || !backend->resident)
        return fail(-1,"invalid profiled resident call",error,capacity);
    if (operation->opcode!=DVGPU_RESIDENT_WRITE && operation->opcode!=DVGPU_RESIDENT_READ &&
        operation->opcode!=DVGPU_RESIDENT_TRANSPOSE && operation->opcode!=DVGPU_RESIDENT_TRANSFORM &&
        operation->opcode!=DVGPU_RESIDENT_TRANSFORM_PACKET)
        return fail(-1,"unsupported profiled resident operation",error,capacity);
    *profile={};
    try {
        const auto started=std::chrono::steady_clock::now();
        ContextScope scope(backend->context);
        *result=dvgpu_resident_execute_profiled(backend->resident,*operation,profile);
        profile->native_wall_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        return 0;
    } catch (const std::invalid_argument& e) {return fail(-1,e.what(),error,capacity);}
      catch (const std::exception& e) {backend->retire();return fail(-2,e.what(),error,capacity);}
      catch (...) {backend->retire();return fail(-3,"unknown profiled resident exception",error,capacity);}
}
extern "C" void dvgpu_resident_seal(void* opaque) {
    auto* backend = static_cast<Backend*>(opaque);
    if (backend) dvgpu_resident_publish(backend->resident, true);
}
extern "C" void dvgpu_resident_commit(void* opaque) {
    auto* backend = static_cast<Backend*>(opaque);
    if (backend) dvgpu_resident_publish(backend->resident, false);
}
extern "C" int dvgpu_resident_stats(void* opaque, DvgpuResidentStats* output) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !output) return -1;
    *output = backend->resident ? dvgpu_resident_get_stats(backend->resident) : backend->retired_resident_stats; return 0;
}
extern "C" int dvgpu_resident_diagnostics(void* opaque, DvgpuResidentDiagnostics* output) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !output) return -1;
    *output = backend->resident ? dvgpu_resident_get_diagnostics(backend->resident) : backend->retired_resident_diagnostics; return 0;
}

namespace {
using Clock = std::chrono::steady_clock;
uint64_t elapsed_ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
// Compile out all diagnostic clock reads from the ordinary ABI entry point.
template <bool Profile, bool Timeline = false>
int msm_impl(void* opaque, const uint8_t* point_bytes, const uint8_t* scalar_bytes,
             size_t length, uint8_t* output, DvgpuMsmTimings* timings,
             DvgpuMsmTimeline* timeline, const char* label, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context)
        return fail(-4, "backend context retired after an error; create a new backend", error, capacity);
    if (!output || length > DVGPU_MAX_MSM_SIZE || (length && (!point_bytes || !scalar_bytes)))
        return fail(-1, "invalid MSM buffers or length", error, capacity);
    if (length == 0) { std::memset(output, 0, 64); return 0; }
    try {
        ContextScope scope(backend->context);
        ForegroundBudget quota(*backend, 0, (64ull << 20) + uint64_t(length) * 1024);
        Clock::time_point started{};
        if constexpr (Profile) started = Clock::now();
        std::vector<scalar_t> scalars(length);
        std::vector<affine_t> points(length);
        for (size_t i = 0; i < length; ++i) {
            for (int j = 0; j < 8; ++j) {
                scalars[i].limbs_storage.limbs[j] = read_u32(scalar_bytes + 32 * i + 4 * j);
                points[i].x.limbs_storage.limbs[j] = read_u32(point_bytes + 64 * i + 4 * j);
                points[i].y.limbs_storage.limbs[j] = read_u32(point_bytes + 64 * i + 32 + 4 * j);
            }
        }
        if constexpr (Profile) timings->conversion_ns = elapsed_ns(started);
        // Destroy this after config/results but before points/scalars.
        Stream stream;
        device_context::DeviceContext context{stream.value, 0, nullptr};
        auto config = msm::default_msm_config(context);
        config.are_scalars_on_device = false;
        config.are_points_on_device = false;
        config.are_scalars_montgomery_form = false;
        config.are_points_montgomery_form = false;
        config.are_results_on_device = false;
        config.is_async = false;
        // Preserve conservative scalar bits/automatic c and the big-triangle
        // reduction. Only the explicit owner option enables large buckets.
        config.is_big_triangle = true;
        config.large_bucket_factor = backend->large_bucket_factor;
        projective_t result;
        // Instantiate ICICLE's pinned MSM with allocation cleanup. This is the
        // implementation behind bn254_msm_cuda, with its kernel bodies intact.
        if constexpr (Timeline) {
            DvgpuNvtxRange call_range(label);
            DvgpuTimelineRecorder recorder(*timeline, stream.value);
            DvgpuTimelineBinding binding(recorder);
            started = Clock::now();
            check(msm::timeline_msm<scalar_t, affine_t, projective_t>(
                scalars.data(), points.data(), int(length), config, &result));
            check(dvgpu_timeline_synchronize(stream.value));
            timings->icicle_ns = elapsed_ns(started);
            recorder.finish(timings->icicle_ns);
        } else {
            if constexpr (Profile) started = Clock::now();
            check(msm::msm<scalar_t, affine_t, projective_t>(
                scalars.data(), points.data(), int(length), config, &result));
            check(cudaStreamSynchronize(stream.value));
            if constexpr (Profile) timings->icicle_ns = elapsed_ns(started);
        }
        // Check main-stream cleanup before publishing a result. An error here
        // retires this owner just like a failed ICICLE operation.
        stream.finish();
        if constexpr (Profile) started = Clock::now();
        const affine_t affine = projective_t::to_affine(result);
        for (int j = 0; j < 8; ++j) {
            write_u32(output + 4 * j, affine.x.limbs_storage.limbs[j]);
            write_u32(output + 32 + 4 * j, affine.y.limbs_storage.limbs[j]);
        }
        if constexpr (Profile) timings->output_ns = elapsed_ns(started);
        return 0;
    } catch (const std::invalid_argument& e) {
        return fail(-1, e.what(), error, capacity);
    } catch (const std::exception& e) {
        backend->retire();
        return fail(-2, e.what(), error, capacity);
    } catch (...) {
        backend->retire();
        return fail(-3, "unknown native exception", error, capacity);
    }
}
}

extern "C" int dvgpu_bn254_msm(void* opaque, const uint8_t* point_bytes, const uint8_t* scalar_bytes,
                               size_t length, uint8_t* output, char* error, size_t capacity) {
    return msm_impl<false>(opaque, point_bytes, scalar_bytes, length, output, nullptr, nullptr, nullptr, error, capacity);
}

extern "C" int dvgpu_bn254_g2_msm(void* opaque, const uint8_t* point_bytes,
    const uint8_t* scalar_bytes, size_t length, uint8_t* output, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4, "backend context retired", error, capacity);
    if (!output || length > DVGPU_MAX_MSM_SIZE || (length && (!point_bytes || !scalar_bytes)))
        return fail(-1, "invalid G2 MSM buffers or length", error, capacity);
    if (!length) { std::memset(output, 0, 128); return 0; }
    try {
        ContextScope scope(backend->context);
        ForegroundBudget quota(*backend, 0, (64ull << 20) + uint64_t(length) * 1024);
        static_assert(sizeof(g2_affine_t) == 128, "G2 canonical component layout changed");
        std::vector<g2_affine_t> points(length);
        std::vector<scalar_t> scalars(length);
        std::memcpy(points.data(), point_bytes, length * 128);
        std::memcpy(scalars.data(), scalar_bytes, length * 32);
        Stream stream;
        device_context::DeviceContext context{stream.value, 0, nullptr};
        auto config = msm::default_msm_config(context);
        config.are_scalars_on_device = false; config.are_points_on_device = false;
        config.are_scalars_montgomery_form = false; config.are_points_montgomery_form = false;
        config.are_results_on_device = false; config.is_async = false;
        config.is_big_triangle = true; config.large_bucket_factor = backend->large_bucket_factor;
        g2_projective_t result;
        check(dvgpu_g2_msm_run(scalars.data(), points.data(), int(length), &config, &result));
        stream.finish();
        const auto affine = g2_projective_t::to_affine(result);
        std::memcpy(output, &affine, 128);
        return 0;
    } catch (const std::invalid_argument& e) { return fail(-1, e.what(), error, capacity); }
      catch (const std::exception& e) { backend->retire(); return fail(-2, e.what(), error, capacity); }
      catch (...) { backend->retire(); return fail(-3, "unknown G2 MSM exception", error, capacity); }
}

extern "C" int dvgpu_reserve_batch(void* opaque,size_t points,char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    if (points>DVGPU_MAX_MSM_SIZE) return fail(-1,"invalid batch reserve capacity",error,capacity);
    try {
        ContextScope scope(backend->context);
        const uint64_t old=backend->batch ? backend->batch->capacity : 0;
        ForegroundBudget quota(*backend,points>old ? uint64_t(points)*96 : 0,
            (64ull<<20)+uint64_t(points)*1024);
        if (!backend->batch) backend->batch=new BatchBuffers;
        Stream stream;
        backend->batch->reserve(points,stream.value,backend->batch_stats);
        dvgpu_resident_set_external(backend->resident,backend->external_bytes());
        stream.finish();
        return 0;
    } catch (const std::invalid_argument& e) { return fail(-1,e.what(),error,capacity); }
      catch (const std::exception& e) { backend->retire(); return fail(-2,e.what(),error,capacity); }
      catch (...) { backend->retire(); return fail(-3,"unknown batch reservation exception",error,capacity); }
}

extern "C" int dvgpu_bn254_fixed_base(void* opaque, const uint8_t* scalar_bytes,
    size_t length, int g2, uint8_t* output, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4, "backend context retired", error, capacity);
    if (length > 4096 || (g2 != 0 && g2 != 1) || (length && (!scalar_bytes || !output)))
        return fail(-1, "invalid fixed-base buffers or chunk length", error, capacity);
    if (!length) return 0;
    try {
        ContextScope scope(backend->context);
        const auto started=std::chrono::steady_clock::now();
        ForegroundBudget quota(*backend, backend->setup.pending(g2),
            (64ull << 20) + uint64_t(length)*512);
        Stream stream;
        static_assert(sizeof(g2_affine_t)==128, "G2 canonical layout changed");
        if (g2) dvgpu_fixed_generator_run<g2_projective_t,g2_affine_t>(
            backend->setup,backend->setup.g2,scalar_bytes,length,output,stream.value,true);
        else dvgpu_fixed_generator_run<projective_t,affine_t>(
            backend->setup,backend->setup.g1,scalar_bytes,length,output,stream.value,false);
        stream.finish();
        dvgpu_resident_set_external(backend->resident,backend->external_bytes());
        backend->setup.stats.native_wall_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        if(g2) {backend->g2_traffic.scalar_h2d_bytes+=length*32;backend->g2_traffic.point_d2h_bytes+=length*128;}
        return 0;
    } catch (const std::invalid_argument& e) { return fail(-1,e.what(),error,capacity); }
      catch (const std::exception& e) { backend->retire(); return fail(-2,e.what(),error,capacity); }
      catch (...) { backend->retire(); return fail(-3,"unknown fixed-base exception",error,capacity); }
}

extern "C" int dvgpu_setup_mode(void* opaque,int mode) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !backend->context || mode<0 || mode>2) return -1;
    backend->setup.mode=mode; return 0;
}
extern "C" int dvgpu_setup_stats(void* opaque,DvgpuSetupStats* output) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !output) return -1;
    *output=backend->setup.stats; return 0;
}

extern "C" int dvgpu_g2_validation_stats(void* opaque,DvgpuG2ValidationStats* output) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !output) return -1;
    *output=backend->g2_validation; return 0;
}
extern "C" int dvgpu_g2_validation_traffic(void* opaque,DvgpuG2ValidationTraffic* output) {
    auto* backend=static_cast<Backend*>(opaque);if(!backend || !output) return -1;
    *output=backend->g2_traffic;return 0;
}
extern "C" int dvgpu_g2_kernel_config_set(void* opaque,const DvgpuG2KernelConfig* config) {
    auto* backend=static_cast<Backend*>(opaque);if(!backend || !backend->context) return -4;
    if(!config || (config->threads!=64 && config->threads!=128 && config->threads!=256)
        || config->schedule>1 || config->layout>1 || config->fused>1) return -1;
    backend->g2_kernel=*config;return 0;
}
extern "C" int dvgpu_g2_kernel_config_get(void* opaque,DvgpuG2KernelConfig* output) {
    auto* backend=static_cast<Backend*>(opaque);if(!backend || !output) return -1;
    *output=backend->g2_kernel;return 0;
}
extern "C" int dvgpu_g2_experiment_stats(void* opaque,DvgpuG2ExperimentStats* output) {
    auto* backend=static_cast<Backend*>(opaque);if(!backend || !output) return -1;
    *output=backend->g2_experiment;return 0;
}
extern "C" int dvgpu_g2_kernel_resources(void* opaque,uint32_t algorithm,DvgpuG2KernelResources* output) {
    auto* backend=static_cast<Backend*>(opaque);if(!backend || !backend->context) return -4;
    if(!output || algorithm<1 || algorithm>3) return -1;
    try {
        ContextScope context(backend->context);DvgpuG2KernelResources result{};
        check(dvgpu_g2_resources_run(algorithm,backend->g2_kernel,result));
        const bool fused=backend->g2_kernel.fused!=0;
        const void* kernel;
        if(backend->setup.mode==2) kernel=fused?reinterpret_cast<const void*>(dvgpu_setup_normalize_batch<g2_projective_t,g2_affine_t,128,true>)
            :reinterpret_cast<const void*>(dvgpu_setup_normalize_batch<g2_projective_t,g2_affine_t>);
        else kernel=fused?reinterpret_cast<const void*>(dvgpu_setup_normalize_each<g2_projective_t,g2_affine_t,true>)
            :reinterpret_cast<const void*>(dvgpu_setup_normalize_each<g2_projective_t,g2_affine_t>);
        cudaFuncAttributes attrs{};int blocks=0;check(cudaFuncGetAttributes(&attrs,kernel));
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0));
        result.normalize_registers=attrs.numRegs;result.normalize_active_blocks=blocks;
        result.normalize_local_bytes=attrs.localSizeBytes;result.normalize_shared_bytes=attrs.sharedSizeBytes;
        *output=result;return 0;
    } catch(const std::exception&) {backend->retire();return -2;}
      catch(...) {backend->retire();return -3;}
}
// Explicit test ABI, never an environment-variable production switch.
extern "C" int dvgpu_g2_test_corrupt_next(void* opaque,uint64_t index) {
    auto* backend=static_cast<Backend*>(opaque);
    if(!backend || !backend->context || index>=4096) return -1;
    backend->g2_test_corrupt_next=int64_t(index);return 0;
}
extern "C" int dvgpu_bn254_fixed_base_g2_checked(void* opaque,const uint8_t* input,
    size_t n,uint32_t algorithm,uint8_t* output,DvgpuG2ValidationReport* report,char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if(!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    if(n>4096 || algorithm<1 || algorithm>3 || !report || (n && (!input || !output)))
        return fail(-1,"invalid checked G2 buffers or algorithm",error,capacity);
    if(!n) {*report={0,0,UINT64_MAX,0,algorithm};return 0;}
    const int64_t corrupt=backend->g2_test_corrupt_next;backend->g2_test_corrupt_next=-1;
    if(corrupt>=int64_t(n)) return fail(-1,"test corruption index outside chunk",error,capacity);
    try {
        const auto started=std::chrono::steady_clock::now();ContextScope context(backend->context);
        ForegroundBudget quota(*backend,backend->setup.pending(true),(64ull<<20)+uint64_t(n)*768);
        // Both host destinations outlive stream/resource guards during unwinding.
        std::vector<uint8_t> pending(n*128);DvgpuG2ValidationReport result{};DvgpuG2ValidationStats stats{};
        DvgpuG2ExperimentStats experiment{};
        DvgpuG2CheckedRun checked{algorithm,&result,&stats,corrupt,&backend->g2_kernel,&experiment};Stream stream;
        const bool valid=dvgpu_fixed_generator_run<g2_projective_t,g2_affine_t>(
            backend->setup,backend->setup.g2,input,n,pending.data(),stream.value,true,&checked);
        stream.finish();dvgpu_resident_set_external(backend->resident,backend->external_bytes());
        const uint64_t elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
        backend->setup.stats.native_wall_ns+=elapsed;
        auto& total=backend->g2_validation;
        total.parse_curve_device_ns+=stats.parse_curve_device_ns;total.subgroup_device_ns+=stats.subgroup_device_ns;
        total.reduce_device_ns+=stats.reduce_device_ns;total.d2h_device_ns+=stats.d2h_device_ns;
        total.native_wall_ns+=elapsed;total.checked_points+=n;
        backend->g2_experiment.repack_device_ns+=experiment.repack_device_ns;
        backend->g2_experiment.subgroup_kernel_ns+=experiment.subgroup_kernel_ns;
        backend->g2_experiment.logical_repack_bytes+=experiment.logical_repack_bytes;
        auto& traffic=backend->g2_traffic;traffic.scalar_h2d_bytes+=n*32;traffic.point_d2h_bytes+=n*128;traffic.status_d2h_bytes+=sizeof(result);
        *report=result;
        if(!valid) return fail(-5,"G2 content validation rejected",error,capacity);
        total.accepted_points+=n;std::copy(pending.begin(),pending.end(),output);return 0;
    } catch(const std::invalid_argument& e) {return fail(-1,e.what(),error,capacity);}
      catch(const std::exception& e) {backend->retire();return fail(-2,e.what(),error,capacity);}
      catch(...) {backend->retire();return fail(-3,"unknown checked G2 exception",error,capacity);}
}
extern "C" int dvgpu_bn254_g2_validation_inspect(void* opaque,const uint8_t* wire,
    size_t n,uint32_t algorithm,uint32_t* masks,DvgpuG2ValidationReport* report,
    char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    if (n>4096 || algorithm<1 || algorithm>3 || !report || (n && (!wire || !masks)))
        return fail(-1,"invalid G2 validation buffers, algorithm or chunk length",error,capacity);
    if (!n) { *report={0,0,UINT64_MAX,0,algorithm}; return 0; }
    try {
        const auto started=std::chrono::steady_clock::now();
        ContextScope context(backend->context);
        ForegroundBudget quota(*backend,0,(64ull<<20)+uint64_t(n)*512);
        std::vector<uint32_t> pending(n);
        DvgpuG2ValidationReport result{}; DvgpuG2ValidationStats stats{};
        DvgpuG2ExperimentStats experiment{};
        Stream stream;
        dvgpu_g2_inspect_run(wire,n,algorithm,pending.data(),&result,stats,stream.value,&backend->g2_kernel,&experiment);
        stream.finish();
        stats.native_wall_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        auto& total=backend->g2_validation;
        total.parse_curve_device_ns+=stats.parse_curve_device_ns;
        total.subgroup_device_ns+=stats.subgroup_device_ns;
        total.reduce_device_ns+=stats.reduce_device_ns;
        total.h2d_device_ns+=stats.h2d_device_ns; total.d2h_device_ns+=stats.d2h_device_ns;
        total.native_wall_ns+=stats.native_wall_ns;total.checked_points+=n;
        backend->g2_experiment.repack_device_ns+=experiment.repack_device_ns;
        backend->g2_experiment.subgroup_kernel_ns+=experiment.subgroup_kernel_ns;
        backend->g2_experiment.logical_repack_bytes+=experiment.logical_repack_bytes;
        backend->g2_traffic.point_h2d_bytes+=n*128;backend->g2_traffic.status_d2h_bytes+=n*4+sizeof(result);
        // Standalone inspection never increments accepted_points or SRS counts.
        std::copy(pending.begin(),pending.end(),masks); *report=result; return 0;
    } catch (const std::invalid_argument& e) { return fail(-1,e.what(),error,capacity); }
      catch (const std::exception& e) { backend->retire(); return fail(-2,e.what(),error,capacity); }
      catch (...) { backend->retire(); return fail(-3,"unknown G2 validation exception",error,capacity); }
}

extern "C" int dvgpu_bn254_tower_inspect(void* opaque, uint32_t degree, uint32_t operation, uint32_t power,
    const uint8_t* left, size_t left_bytes, const uint8_t* right, size_t right_bytes,
    uint8_t* output, size_t output_bytes, uint32_t* statuses, size_t statuses_count,
    char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    if ((degree != DVGPU_TOWER_FQ2 && degree != DVGPU_TOWER_FQ6 && degree != DVGPU_TOWER_FQ12) ||
        operation < DVGPU_TOWER_ADD || operation > DVGPU_TOWER_FROBENIUS ||
        (operation == DVGPU_TOWER_FROBENIUS ? power > 12 : power != 0))
        return fail(-1,"invalid BN254 tower field, operation or power",error,capacity);
    const size_t width = size_t(degree)*32;
    const size_t count = left_bytes/width;
    const bool binary = operation <= DVGPU_TOWER_MULTIPLY;
    if (left_bytes%width || count > DVGPU_TOWER_MAX_ITEMS || output_bytes != left_bytes ||
        statuses_count != count || (binary ? right_bytes != left_bytes : right_bytes != 0) ||
        (count && (!left || !output || !statuses || (binary && !right))))
        return fail(-1,"invalid BN254 tower buffer lengths",error,capacity);
    if (!count) return 0;
    try {
        ContextScope context(backend->context);
        // Bounded transient buffers share the resident owner's existing cap.
        // Include CUDA allocator overhead conservatively, before any upload.
        ForegroundBudget quota(*backend,0,(64ull<<20)+3ull*left_bytes+count*sizeof(uint32_t));
        std::vector<uint8_t> pending(left_bytes);
        std::vector<uint32_t> pending_statuses(count);
        Stream stream;
        dvgpu_bn254_tower_run(left,right,count,degree,operation,power,
            pending.data(),pending_statuses.data(),stream.value);
        stream.finish();
        std::copy(pending.begin(),pending.end(),output);
        std::copy(pending_statuses.begin(),pending_statuses.end(),statuses);
        return 0;
    } catch (const std::invalid_argument& e) {return fail(-1,e.what(),error,capacity);}
      catch (const std::exception& e) {backend->retire();return fail(-2,e.what(),error,capacity);}
      catch (...) {backend->retire();return fail(-3,"unknown BN254 tower exception",error,capacity);}
}

extern "C" int dvgpu_bn254_lines_inspect(void* opaque,const uint8_t* points,size_t points_bytes,
    uint8_t* output,size_t output_bytes,uint32_t* counts,size_t counts_count,
    uint32_t* statuses,size_t statuses_count,char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    const size_t count=points_bytes/128;
    if (points_bytes%128 || count>DVGPU_LINE_MAX_ITEMS)
        return fail(-1,"invalid BN254 line input length",error,capacity);
    if (output_bytes!=count*DVGPU_LINE_ITEM_BYTES || counts_count!=count || statuses_count!=count ||
        (count && (!points || !output || !counts || !statuses)))
        return fail(-1,"invalid BN254 line output buffers",error,capacity);
    if (!count) return 0;
    try {
        ContextScope context(backend->context);
        ForegroundBudget quota(*backend,0,(64ull<<20)+points_bytes+output_bytes+
            2ull*count*sizeof(uint32_t)+sizeof(DvgpuG2ValidationReport));
        std::vector<uint8_t> pending(output_bytes);
        std::vector<uint32_t> pending_counts(count),pending_statuses(count);
        Stream stream;
        dvgpu_bn254_lines_run(points,count,pending.data(),pending_counts.data(),pending_statuses.data(),stream.value);
        stream.finish();
        std::copy(pending.begin(),pending.end(),output);
        std::copy(pending_counts.begin(),pending_counts.end(),counts);
        std::copy(pending_statuses.begin(),pending_statuses.end(),statuses);
        return 0;
    } catch (const std::invalid_argument& e) {return fail(-1,e.what(),error,capacity);}
      catch (const std::exception& e) {backend->retire();return fail(-2,e.what(),error,capacity);}
      catch (...) {backend->retire();return fail(-3,"unknown BN254 line exception",error,capacity);}
}

extern "C" int dvgpu_bn254_miller_inspect(void* opaque,const uint8_t* p,size_t p_bytes,
    const uint8_t* q,size_t q_bytes,uint8_t* output,size_t output_bytes,
    uint32_t* metadata,size_t metadata_count,char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if(!backend || !backend->context)return fail(-4,"backend context retired",error,capacity);
    if(p_bytes!=64 || q_bytes!=128 || output_bytes!=DVGPU_MILLER_OUTPUT_BYTES ||
       metadata_count!=DVGPU_MILLER_METADATA_COUNT || !p || !q || !output || !metadata)
        return fail(-1,"invalid single-pair BN254 Miller buffers",error,capacity);
    try {
        ContextScope context(backend->context);
        ForegroundBudget quota(*backend,0,(64ull<<20)+192+DVGPU_MILLER_OUTPUT_BYTES+
            DVGPU_MILLER_METADATA_COUNT*sizeof(uint32_t)+sizeof(DvgpuG2ValidationReport));
        // Host staging outlives Stream/allocations during all error unwinding.
        std::vector<uint8_t> pending(output_bytes);
        uint32_t pending_metadata[DVGPU_MILLER_METADATA_COUNT]{};
        Stream stream;
        dvgpu_bn254_miller_run(p,q,pending.data(),pending_metadata,stream.value);
        stream.finish();
        std::copy(pending.begin(),pending.end(),output);
        std::copy(pending_metadata,pending_metadata+DVGPU_MILLER_METADATA_COUNT,metadata);
        return 0;
    } catch(const std::invalid_argument& e){return fail(-1,e.what(),error,capacity);}
      catch(const std::exception& e){backend->retire();return fail(-2,e.what(),error,capacity);}
      catch(...){backend->retire();return fail(-3,"unknown BN254 Miller exception",error,capacity);}
}

// Bounded mathematical diagnostic entry; not part of SRS publication.
extern "C" int dvgpu_g2_psi_probe(void* opaque,const uint8_t* points,size_t n,
    uint8_t* output,char* error,size_t capacity) {
    auto* backend=static_cast<Backend*>(opaque);
    if (!backend || !backend->context) return fail(-4,"backend context retired",error,capacity);
    if (n>129 || (n && (!points || !output))) return fail(-1,"invalid psi probe buffers",error,capacity);
    if (!n) return 0;
    try {
        ContextScope context(backend->context);
        ForegroundBudget quota(*backend,0,(64ull<<20)+n*1024);
        std::vector<uint8_t> pending(n*576); Stream stream;
        dvgpu_g2_psi_probe_run(points,n,pending.data(),stream.value);stream.finish();
        std::copy(pending.begin(),pending.end(),output);return 0;
    } catch (const std::invalid_argument& e) {return fail(-1,e.what(),error,capacity);}
      catch (const std::exception& e) {backend->retire();return fail(-2,e.what(),error,capacity);}
      catch (...) {backend->retire();return fail(-3,"unknown psi probe exception",error,capacity);}
}

extern "C" int dvgpu_bn254_ntt(void* opaque, const uint8_t* value_bytes, size_t length, int inverse,
    const uint8_t* root_bytes, uint8_t* output, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context)
        return fail(-4, "backend context retired after an error; create a new backend", error, capacity);
    if (!value_bytes || !output || !root_bytes || !length || length > DVGPU_MAX_NTT_SIZE ||
        (length & (length - 1)) || (inverse != 0 && inverse != 1))
        return fail(-1, "invalid NTT buffers, direction or length", error, capacity);
    try {
        ContextScope scope(backend->context);
        // The singleton transform is identity; no kernel with log_size=0.
        if (length == 1) { std::memcpy(output, value_bytes, 32); return 0; }
        ForegroundBudget quota(*backend, 0, (64ull << 20) + uint64_t(length) * 256);
        std::vector<scalar_t> values(length), result(length);
        scalar_t root;
        for (int j = 0; j < 8; ++j) root.limbs_storage.limbs[j] = read_u32(root_bytes + 4 * j);
        scalar_t power = root;
        for (int i = 0; i < 19; ++i) power = scalar_t::sqr(power);
        if (power == scalar_t::one() || !(scalar_t::sqr(power) == scalar_t::one()))
            return fail(-1, "NTT root must have order 2^20", error, capacity);
        for (size_t i = 0; i < length; ++i)
            for (int j = 0; j < 8; ++j)
                values[i].limbs_storage.limbs[j] = read_u32(value_bytes + 32 * i + 4 * j);
        Stream stream;
        device_context::DeviceContext context{stream.value, 0, nullptr};
        if (!backend->ntt) backend->ntt = dvgpu_ntt_create();
        check(dvgpu_ntt_run(backend->ntt, root, values.data(), int(length), inverse != 0, context, result.data()));
        stream.finish();
        for (size_t i = 0; i < length; ++i)
            for (int j = 0; j < 8; ++j)
                write_u32(output + 32 * i + 4 * j, result[i].limbs_storage.limbs[j]);
        return 0;
    } catch (const std::invalid_argument& e) {
        return fail(-1, e.what(), error, capacity);
    } catch (const std::exception& e) {
        backend->retire();
        return fail(-2, e.what(), error, capacity);
    } catch (...) {
        backend->retire();
        return fail(-3, "unknown native exception", error, capacity);
    }
}

extern "C" int dvgpu_bn254_msm_batch(void* opaque, const uint8_t* point_bytes, const uint8_t* scalar_bytes,
    size_t length, size_t count, uint8_t* output, char* error, size_t capacity) {
    auto* backend = static_cast<Backend*>(opaque);
    if (!backend || !backend->context)
        return fail(-4, "backend context retired after an error; create a new backend", error, capacity);
    if (!point_bytes || !scalar_bytes || !output || !length || !count ||
        count > DVGPU_MAX_MSM_BATCH_SIZE || length > DVGPU_MAX_MSM_SIZE / count)
        return fail(-1, "invalid batch buffers or bounded work size", error, capacity);
    try {
        ContextScope scope(backend->context);
        const uint64_t total_points = uint64_t(length) * count;
        const uint64_t old_capacity = backend->batch ? backend->batch->capacity : 0;
        const uint64_t pending = total_points > old_capacity ? total_points * 96 : 0;
        ForegroundBudget quota(*backend, pending, (64ull << 20) + total_points * 1024);
        if (!backend->batch) backend->batch = new BatchBuffers;
        auto& buffers = *backend->batch;
        // Dynamic host output must outlive stream synchronization on errors.
        std::vector<projective_t> results(count);
        Stream stream;
        const size_t total = length * count;
        buffers.reserve(total, stream.value, backend->batch_stats);
        dvgpu_resident_set_external(backend->resident, backend->external_bytes());
        // Pending allocations were detached into persistent owner storage;
        // now shrink the same scratch budget to account for that publication.
        dvgpu_resident_temporary(backend->resident);
        for (size_t i = 0; i < total; ++i)
            for (int j = 0; j < 8; ++j) {
                buffers.host_scalars[i].limbs_storage.limbs[j] = read_u32(scalar_bytes + 32 * i + 4 * j);
                buffers.host_points[i].x.limbs_storage.limbs[j] = read_u32(point_bytes + 64 * i + 4 * j);
                buffers.host_points[i].y.limbs_storage.limbs[j] = read_u32(point_bytes + 64 * i + 32 + 4 * j);
            }
        check(cudaMemcpyAsync(buffers.scalars, buffers.host_scalars.data(), total * sizeof(scalar_t),
            cudaMemcpyHostToDevice, stream.value));
        check(cudaMemcpyAsync(buffers.points, buffers.host_points.data(), total * sizeof(affine_t),
            cudaMemcpyHostToDevice, stream.value));
        device_context::DeviceContext context{stream.value, 0, nullptr};
        auto config = msm::default_msm_config(context);
        config.points_size = int(total);
        config.batch_size = int(count);
        config.are_scalars_on_device = true;
        config.are_points_on_device = true;
        config.are_scalars_montgomery_form = false;
        config.are_points_montgomery_form = false;
        config.are_results_on_device = false;
        config.is_async = false;
        config.is_big_triangle = true;
        config.large_bucket_factor = backend->large_bucket_factor;
        ++backend->batch_stats.icicle_calls;
        backend->batch_stats.max_batch_size = std::max<uint64_t>(backend->batch_stats.max_batch_size, count);
        check(msm::msm<scalar_t, affine_t, projective_t>(buffers.scalars, buffers.points, int(length), config, results.data()));
        stream.finish();
        uint8_t encoded[DVGPU_MAX_MSM_BATCH_SIZE * 64];
        for (size_t i = 0; i < count; ++i) {
            const auto affine = projective_t::to_affine(results[i]);
            for (int j = 0; j < 8; ++j) {
                write_u32(encoded + i * 64 + 4 * j, affine.x.limbs_storage.limbs[j]);
                write_u32(encoded + i * 64 + 32 + 4 * j, affine.y.limbs_storage.limbs[j]);
            }
        }
        std::memcpy(output, encoded, count * 64);
        return 0;
    } catch (const std::invalid_argument& e) {
        return fail(-1, e.what(), error, capacity);
    } catch (const std::exception& e) {
        backend->retire();
        return fail(-2, e.what(), error, capacity);
    } catch (...) {
        backend->retire();
        return fail(-3, "unknown native exception", error, capacity);
    }
}

extern "C" int dvgpu_bn254_msm_profiled(void* opaque, const uint8_t* point_bytes, const uint8_t* scalar_bytes,
                                       size_t length, uint8_t* output, DvgpuMsmTimings* timings,
                                       char* error, size_t capacity) {
    if (!timings) return fail(-1, "null diagnostic timings", error, capacity);
    *timings = {};
    return msm_impl<true>(opaque, point_bytes, scalar_bytes, length, output, timings, nullptr, nullptr, error, capacity);
}

extern "C" int dvgpu_bn254_msm_timeline(void* opaque, const uint8_t* point_bytes, const uint8_t* scalar_bytes,
                                       size_t length, uint8_t* output, const char* label, DvgpuMsmTimings* timings,
                                       DvgpuMsmTimeline* timeline, char* error, size_t capacity) {
    if (!timings || !timeline || !label) return fail(-1, "null timeline diagnostics", error, capacity);
    *timings = {};
    *timeline = {};
    return msm_impl<true, true>(opaque, point_bytes, scalar_bytes, length, output, timings, timeline, label, error, capacity);
}
