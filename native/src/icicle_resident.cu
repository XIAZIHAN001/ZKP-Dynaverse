#include "icicle_resident.cuh"
#include "icicle_transform_packet.cuh"
#include "icicle_msm_allocations.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace bn254;
namespace {
using ProfileClock=std::chrono::steady_clock;
uint64_t profile_elapsed(ProfileClock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(ProfileClock::now()-start).count();
}
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
uint64_t checked_product(uint64_t a, uint64_t b) {
    require(!a || b <= UINT64_MAX / a, "resident size overflow");
    return a * b;
}
bool overlaps(uint64_t a, uint64_t b, uint64_t n) {
    return n && (a < b ? b - a < n : a - b < n);
}
template <typename E> __global__ void zero_elements(E* output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = E::zero();
}
template <typename E> __global__ void reverse_elements(const E* input, E* output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[n - 1 - i];
}
__device__ scalar_t scalar_power(scalar_t base, size_t exponent) {
    scalar_t output = scalar_t::one();
    while (exponent) {
        if (exponent & 1) output = output * base;
        exponent >>= 1;
        if (exponent) base = scalar_t::sqr(base);
    }
    return output;
}
template <typename E> __global__ void transpose_elements(const E* input, E* output,
    size_t rows, size_t cols, size_t start, size_t n, bool twiddle, scalar_t root) {
    const size_t offset = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= n) return;
    const size_t i = start + offset, row = i / cols, col = i % cols;
    E value = input[i];
    if (twiddle) value = scalar_power(root, row * col) * value;
    output[col * rows + row] = value;
}
__global__ void zip_fields(const scalar_t* left, const scalar_t* right,
    scalar_t* output, size_t n, unsigned op) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = op == 0 ? left[i] + right[i] :
        (op == 1 ? left[i] - right[i] : left[i] * right[i]);
}
__global__ void scale_elements(const projective_t* points, const scalar_t* scalars,
    projective_t* output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = scalars[i] * points[i];
}
__global__ void normalize_elements(const projective_t* points, affine_t* output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = projective_t::to_affine(points[i]);
}
constexpr uint64_t DOMAIN_BYTES = (uint64_t(DVGPU_MAX_NTT_SIZE) + 1 + 6 + 64) * sizeof(scalar_t);
constexpr uint64_t WORKSPACE_MARGIN = 64ull << 20;
std::atomic<uint32_t> next_owner{1};
struct Buffer {
    void* pointer = nullptr;
    uint64_t length = 0;
    uint32_t kind = 0;
    bool writable = false;
    std::vector<uint8_t> saved; // eight first-write flags per host byte
    size_t width() const { return kind == 0 ? sizeof(scalar_t) :
        (kind == 2 ? sizeof(affine_t) : sizeof(projective_t)); }
    bool marked(size_t i) const { return (saved[i/8] & uint8_t(1u<<(i%8))) != 0; }
    void mark(size_t i,bool set) {
        const uint8_t bit=uint8_t(1u<<(i%8));
        if (set) saved[i/8] |= bit; else saved[i/8] &= uint8_t(~bit);
    }
    uint8_t* bytes(size_t offset = 0) const {
        return static_cast<uint8_t*>(pointer) + offset * width();
    }
};
struct JournalRange { uint64_t buffer, offset, length, arena; };
}

struct DvgpuResidentState {
    uint64_t limit, journal_capacity, journal_used = 0, max_length = 0, external_bytes = 0;
    uint32_t owner = next_owner.fetch_add(1);
    bool sealed = false, transaction = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t started = nullptr, finished = nullptr;
    cudaEvent_t profile_events[4]{};
    DvgpuResidentProfile* profile=nullptr; // stack borrow only within synchronous diagnostic entry
    ProfileClock::time_point operation_started{};
    bool operation_active=false;
    uint8_t* journal = nullptr;
    affine_t* affine_workspace = nullptr;
    size_t affine_capacity = 0;
    DvgpuNttState* ntt;
    std::vector<Buffer> buffers;
    std::vector<JournalRange> undo;
    DvgpuResidentStats stats{};
    DvgpuResidentDiagnostics diagnostics{};
    DvgpuTemporaryAllocationStats temporary{};

    DvgpuResidentState(uint64_t memory_limit, uint64_t journal_bytes, DvgpuNttState* domain)
        : limit(memory_limit), journal_capacity(journal_bytes), ntt(domain) {
        require(owner != 0, "resident owner tag exhausted");
        require(limit >= DOMAIN_BYTES + WORKSPACE_MARGIN &&
            journal_capacity <= limit - DOMAIN_BYTES - WORKSPACE_MARGIN,
            "resident quota cannot contain the fixed domain, journal and workspace reserve");
        buffers.reserve(1024);
        // Every saved range contains at least one 32-byte Fr, so byte capacity
        // also bounds range count. Allocate metadata once, before service.
        undo.reserve(std::max<uint64_t>(1, journal_capacity / sizeof(scalar_t)));
        diagnostics.journal_metadata_capacity_bytes = undo.capacity() * sizeof(JournalRange);
    }
    Buffer& buffer(uint64_t id) {
        require((id >> 32) == owner, "resident handle belongs to another owner");
        const uint64_t slot = id & 0xffffffffu;
        require(slot && slot <= buffers.size(), "unknown resident handle");
        return buffers[slot - 1];
    }
    Buffer& range(uint64_t id, uint64_t offset, uint64_t n, int kind = -1, bool write = false) {
        auto& value = buffer(id);
        require(kind < 0 || value.kind == unsigned(kind), "resident buffer kind mismatch");
        require(offset <= value.length && n <= value.length - offset, "resident range out of bounds");
        require(!write || !sealed || value.writable, "resident buffer is immutable after seal");
        return value;
    }
    void preflight_pair(uint64_t first, uint64_t second, uint64_t n) {
        if (!transaction) return;
        uint64_t bytes = 0, ranges = 0;
        // Both buffers were validated as distinct before this scan. Account
        // only first writes, against remaining space including prior commands.
        for (const auto id : {first, second}) {
            auto& value = buffer(id);
            uint64_t entries = 0;
            bool prior = false;
            for (uint64_t i = 0; i < n; ++i) {
                const bool fresh = !value.marked(i);
                if (fresh) { ++entries; if (!prior) ++ranges; }
                prior = fresh;
            }
            bytes += checked_product(entries, value.width());
        }
        require(bytes <= journal_capacity - journal_used, "resident packet journal exhausted");
        require(ranges <= undo.capacity() - undo.size(), "resident packet range limit exceeded");
    }
    void save(uint64_t id, uint64_t offset, uint64_t n) {
        if (!transaction || !n) return;
        const auto entered=profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (profile && !(profile->event_mask&1)) {
            checked(cudaEventRecord(profile_events[0],stream));profile->event_mask|=1;
        }
        auto& value = buffer(id);
        const uint64_t end=offset+n;
        uint64_t entries=0,ranges=0;
        bool prior=false;
        // Preflight the complete save before queueing anything. Repeated
        // writes retain the original value, not an intermediate checkpoint.
        for (uint64_t i=offset;i<end;++i) {
            const bool fresh=!value.marked(i);
            if (fresh) {++entries;if (!prior) ++ranges;}
            prior=fresh;
        }
        const auto bytes=checked_product(entries,value.width());
        require(bytes<=journal_capacity-journal_used,"resident transaction journal exhausted");
        require(ranges<=undo.capacity()-undo.size(),"resident transaction range limit exceeded");
        uint64_t i=offset;
        while (i<end) {
            if (value.marked(i)) {++i;continue;}
            const uint64_t first=i;
            while (i<end && !value.marked(i)) ++i;
            const auto count=i-first, size=checked_product(count,value.width());
            checked(cudaMemcpyAsync(journal+journal_used,value.bytes(first),size,
                cudaMemcpyDeviceToDevice,stream));
            undo.push_back({id,first,count,journal_used});
            for (uint64_t j=first;j<i;++j) value.mark(j,true);
            journal_used+=size;
            diagnostics.journal_saved_bytes+=size;diagnostics.explicit_d2d_bytes+=size;
            ++diagnostics.journal_saved_ranges;
        }
        diagnostics.journal_peak_bytes=std::max(diagnostics.journal_peak_bytes,journal_used);
        if (profile) {
            checked(cudaEventRecord(profile_events[1],stream));
            profile->journal_host_ns+=profile_elapsed(entered);
        }
    }

    void operation_begin() {
        if (!profile) return;
        operation_started=ProfileClock::now();operation_active=true;
        checked(cudaEventRecord(profile_events[2],stream));profile->event_mask|=2;
    }
    void operation_end() {
        if (!profile || !operation_active) return;
        checked(cudaEventRecord(profile_events[3],stream));
        profile->operation_host_ns+=profile_elapsed(operation_started);operation_active=false;
    }
    void wait() {
        const auto entered=profile ? ProfileClock::now() : ProfileClock::time_point{};
        const auto result=cudaStreamSynchronize(stream);
        if (profile) profile->wait_host_ns+=profile_elapsed(entered);
        checked(result);
    }

    void quota(uint64_t extra, uint64_t next_max) {
        const uint64_t workspace = WORKSPACE_MARGIN + checked_product(next_max, 1024);
        const auto allocated = stats.bytes_allocated + external_bytes;
        require(allocated <= limit && extra <= limit - allocated &&
            workspace <= limit - allocated - extra,
            "resident peak memory quota exceeded before allocation");
        size_t free_bytes = 0, total_bytes = 0;
        checked(cudaMemGetInfo(&free_bytes, &total_bytes));
        require(extra <= free_bytes && workspace <= free_bytes - extra,
            "insufficient device memory for resident buffers and bounded workspace");
    }
    void complete() {
        operation_end();
        checked(cudaGetLastError());
        checked(cudaEventRecord(finished, stream));
        wait();
        float ms = 0;
        checked(cudaEventElapsedTime(&ms, started, finished));
        diagnostics.device_operation_ns += uint64_t(double(ms) * 1000000.0);
        ++diagnostics.device_operations;
        if (profile) {
            profile->event_total_ns=uint64_t(double(ms)*1000000.0);profile->event_mask|=4;
            if (profile->event_mask&1) {
                checked(cudaEventElapsedTime(&ms,profile_events[0],profile_events[1]));
                profile->event_journal_ns=uint64_t(double(ms)*1000000.0);
            }
            if (profile->event_mask&2) {
                checked(cudaEventElapsedTime(&ms,profile_events[2],profile_events[3]));
                profile->event_operation_ns=uint64_t(double(ms)*1000000.0);
            }
        }
    }
};

DvgpuResidentState* dvgpu_resident_create(uint64_t limit, uint64_t journal,
    scalar_t root, DvgpuNttState* ntt) {
    auto state = std::make_unique<DvgpuResidentState>(limit, journal, ntt);
    try {
        // Blocking application stream: pinned MSM also initializes buckets on
        // the legacy default stream. Never replace with cudaStreamNonBlocking.
        checked(cudaStreamCreate(&state->stream));
        checked(cudaEventCreate(&state->started));
        checked(cudaEventCreate(&state->finished));
        for (auto& event:state->profile_events) checked(cudaEventCreate(&event));
        device_context::DeviceContext context{state->stream, 0, nullptr};
        checked(dvgpu_ntt_prepare(ntt, root, context));
        state->stats.bytes_allocated = DOMAIN_BYTES;
        state->diagnostics.resident_allocations = 7;
        state->quota(journal, 0);
        if (journal) {
            checked(cudaMalloc(&state->journal, journal));
            state->stats.bytes_allocated += journal;
            ++state->diagnostics.resident_allocations;
        }
        checked(cudaStreamSynchronize(state->stream));
        return state.release();
    } catch (...) {
        dvgpu_resident_destroy(state.release(), true);
        throw;
    }
}
void dvgpu_resident_destroy(DvgpuResidentState* state, bool current) noexcept {
    if (!state) return;
    if (current) {
        cudaDeviceSynchronize();
        for (auto& value : state->buffers) if (value.pointer) cudaFree(value.pointer);
        if (state->journal) cudaFree(state->journal);
        if (state->affine_workspace) cudaFree(state->affine_workspace);
        if (state->started) cudaEventDestroy(state->started);
        if (state->finished) cudaEventDestroy(state->finished);
        for (auto event:state->profile_events) if (event) cudaEventDestroy(event);
        if (state->stream) cudaStreamDestroy(state->stream);
    }
    delete state;
}
void dvgpu_resident_publish(DvgpuResidentState* state, bool seal) noexcept {
    if (!state) return;
    if (seal) state->sealed = true;
    else {
        // Clear only first-write ranges touched by this transaction. No
        // buffer-sized clearing at begin, commit, rollback or anchor switch.
        for (const auto& range:state->undo) {
            auto& value=state->buffer(range.buffer);
            for (uint64_t i=range.offset;i<range.offset+range.length;++i) value.mark(i,false);
        }
        state->undo.clear();state->journal_used=0;state->transaction=false;
    }
}
DvgpuResidentStats dvgpu_resident_get_stats(const DvgpuResidentState* state) noexcept {
    auto result = state ? state->stats : DvgpuResidentStats{};
    if (state) result.bytes_allocated += state->external_bytes;
    return result;
}
DvgpuResidentDiagnostics dvgpu_resident_get_diagnostics(const DvgpuResidentState* state) noexcept {
    auto result = state ? state->diagnostics : DvgpuResidentDiagnostics{};
    if (state) {
        result.temporary_allocations = state->temporary.calls;
        result.temporary_peak_bytes = state->temporary.peak_bytes;
    }
    return result;
}

void dvgpu_resident_preflight_external(DvgpuResidentState* s, uint64_t pending, uint64_t scratch) {
    if (!s) return;
    const uint64_t allocated = s->stats.bytes_allocated + s->external_bytes;
    require(allocated <= s->limit && pending <= s->limit - allocated &&
        scratch <= s->limit - allocated - pending, "foreground owner memory quota exceeded before allocation");
    size_t available = 0, total = 0;
    checked(cudaMemGetInfo(&available, &total));
    require(pending <= available && scratch <= available - pending,
        "insufficient device memory for foreground buffers and workspace");
}
void dvgpu_resident_set_external(DvgpuResidentState* s, uint64_t retained) {
    if (!s) return;
    require(s->stats.bytes_allocated <= s->limit && retained <= s->limit - s->stats.bytes_allocated,
        "foreground retained memory exceeds owner quota");
    s->external_bytes = retained;
}
DvgpuTemporaryAllocationStats* dvgpu_resident_temporary(DvgpuResidentState* s) {
    if (!s) return nullptr;
    s->temporary.limit_bytes = s->limit - s->stats.bytes_allocated - s->external_bytes;
    return &s->temporary;
}

namespace {
struct AllocationBinding {
    DvgpuTemporaryAllocationStats* previous;
    explicit AllocationBinding(DvgpuTemporaryAllocationStats& value)
        : previous(dvgpu_temporary_allocations) { dvgpu_temporary_allocations = &value; }
    ~AllocationBinding() { dvgpu_temporary_allocations = previous; }
};
void alias(uint64_t src, uint64_t src_offset, uint64_t dst, uint64_t dst_offset,
    uint64_t n, bool exact_allowed) {
    require(src != dst || !overlaps(src_offset, dst_offset, n) ||
        (exact_allowed && src_offset == dst_offset), "overlapping resident ranges");
}
template <typename G> void publish_point(const G& point, uint8_t* output) {
    const auto affine = G::to_affine(point);
    std::memcpy(output, &affine, sizeof(affine));
}
}

uint64_t dvgpu_resident_execute(DvgpuResidentState* s, const DvgpuResidentOp& op) {
    require(s, "resident backend has not been configured");
    ++s->stats.calls;
    const size_t n = op.length;
    if (op.opcode == DVGPU_RESIDENT_LENGTH) return s->buffer(op.a).length;
    if (op.opcode == DVGPU_RESIDENT_KIND) return s->buffer(op.a).kind;
    if (op.opcode == DVGPU_RESIDENT_BEGIN) {
        require(!s->transaction, "resident transaction already open");
        s->transaction = true; s->journal_used = 0; s->undo.clear(); return 0;
    }
    checked(cudaEventRecord(s->started, s->stream));
    device_context::DeviceContext context{s->stream, 0, nullptr};
    s->temporary.limit_bytes = s->limit - s->stats.bytes_allocated - s->external_bytes;
    AllocationBinding binding(s->temporary);
    if (op.opcode == DVGPU_RESIDENT_ALLOC || op.opcode == DVGPU_RESIDENT_ALLOC_BASES) {
        const bool packed=op.opcode==DVGPU_RESIDENT_ALLOC_BASES;
        require(!s->sealed && !s->transaction, "resident allocation after seal or transaction start");
        require(op.kind <= 1 && s->buffers.size() < s->buffers.capacity(), "resident kind or buffer limit invalid");
        require(!packed || !n || op.input,"null affine basis input");
        Buffer value{nullptr, op.length, packed ? 2u : op.kind, !packed && op.flags != 0};
        const uint64_t bytes = checked_product(n, value.width());
        const size_t next_affine = !packed && op.kind == 1 ? std::max(n, s->affine_capacity) : s->affine_capacity;
        const uint64_t affine_extra = checked_product(next_affine - s->affine_capacity, sizeof(affine_t));
        s->quota(bytes + (next_affine > s->affine_capacity ? next_affine * sizeof(affine_t) : 0), std::max(s->max_length, op.length));
        value.saved.resize(n/8+(n%8!=0),0);
        if (bytes) {
            checked(cudaMalloc(&value.pointer, bytes));
            ++s->diagnostics.resident_allocations;
        }
        // Publish the allocation into the owner immediately; a later CUDA
        // failure retires the owner and its destructor releases this pointer.
        s->buffers.push_back(std::move(value));
        s->stats.bytes_allocated += bytes;
        if (next_affine > s->affine_capacity) {
            affine_t* next = nullptr;
            checked(cudaMalloc(&next, next_affine * sizeof(affine_t)));
            if (s->affine_workspace) checked(cudaFree(s->affine_workspace));
            s->affine_workspace = next; s->affine_capacity = next_affine;
            s->stats.bytes_allocated += affine_extra;
            ++s->diagnostics.resident_allocations;
        }
        s->max_length = std::max(s->max_length, op.length);
        if (packed && n) {
            checked(cudaMemcpyAsync(value.pointer,op.input,n*sizeof(affine_t),cudaMemcpyHostToDevice,s->stream));
            s->stats.h2d_bytes+=n*64;s->diagnostics.physical_h2d_bytes+=n*64;
        }
        if (!packed && n && op.kind == 0) zero_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(static_cast<scalar_t*>(value.pointer), n);
        if (!packed && n && op.kind == 1) zero_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(static_cast<projective_t*>(value.pointer), n);
        s->complete();
        return (uint64_t(s->owner) << 32) | s->buffers.size();
    }
    if (op.opcode == DVGPU_RESIDENT_ROLLBACK) {
        require(s->transaction, "no resident transaction open");
        for (auto it = s->undo.rbegin(); it != s->undo.rend(); ++it) {
            auto& value = s->buffer(it->buffer);
            checked(cudaMemcpyAsync(value.bytes(it->offset), s->journal + it->arena,
                it->length * value.width(), cudaMemcpyDeviceToDevice, s->stream));
            s->diagnostics.explicit_d2d_bytes += it->length * value.width();
        }
        s->complete(); dvgpu_resident_publish(s, false); return 0;
    }
    if (op.opcode == DVGPU_RESIDENT_WRITE || op.opcode == DVGPU_RESIDENT_READ) {
        const bool write = op.opcode == DVGPU_RESIDENT_WRITE;
        auto& value = s->range(op.a, op.a_offset, n, -1, write);
        require(value.kind==op.kind || (value.kind==2 && op.kind==1),"resident buffer kind mismatch");
        require(!write || value.kind!=2,"packed MSM bases are immutable");
        require(write || (op.flags <= DVGPU_READ_DEVICE_AFFINE &&
            (!op.flags || op.kind == 1)), "invalid resident read flags");
        require(!n || (write ? op.input != nullptr : op.output != nullptr), "null resident host buffer");
        if (!n) return 0;
        const bool device_affine = !write && value.kind == 1 &&
            op.flags == DVGPU_READ_DEVICE_AFFINE && n <= DVGPU_READ_DEVICE_AFFINE_MAX;
        require(!device_affine || (s->affine_workspace && n <= s->affine_capacity),
            "resident affine read workspace unavailable");
        if (write) s->save(op.a, op.a_offset, n);
        if (value.kind == 0 || value.kind == 2) {
            s->operation_begin();
            checked(cudaMemcpyAsync(write ? value.bytes(op.a_offset) : op.output,
                write ? op.input : value.bytes(op.a_offset), n * value.width(),
                write ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, s->stream));
            s->operation_end();
        } else if (device_affine) {
            // Read-only scratch reuse: no journal entry or allocation after
            // seal. The same serial stream orders normalization, copy and
            // completion before the borrowed canonical output is decoded.
            s->operation_begin();
            normalize_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(
                reinterpret_cast<const projective_t*>(value.bytes(op.a_offset)),
                s->affine_workspace, n);
            checked(cudaGetLastError());
            checked(cudaMemcpyAsync(op.output, s->affine_workspace, n * sizeof(affine_t),
                cudaMemcpyDeviceToHost, s->stream));
            s->operation_end();
        } else {
            // Bounded conversion only: no full-buffer host mirror survives.
            auto converted=s->profile ? ProfileClock::now() : ProfileClock::time_point{};
            std::vector<projective_t> host(n);
            if (write) {
                for (size_t i = 0; i < n; ++i) {
                    affine_t point; std::memcpy(&point, op.input + i * 64, 64);
                    host[i] = projective_t::from_affine(point);
                }
                if (s->profile) s->profile->conversion_host_ns+=profile_elapsed(converted);
                s->operation_begin();
                checked(cudaMemcpyAsync(value.bytes(op.a_offset), host.data(), n * sizeof(projective_t), cudaMemcpyHostToDevice, s->stream));
                s->operation_end();s->wait();
            } else {
                if (s->profile) s->profile->conversion_host_ns+=profile_elapsed(converted);
                s->operation_begin();
                checked(cudaMemcpyAsync(host.data(), value.bytes(op.a_offset), n * sizeof(projective_t), cudaMemcpyDeviceToHost, s->stream));
                s->operation_end();s->wait();
                if (s->profile) converted=ProfileClock::now();
                for (size_t i = 0; i < n; ++i) publish_point(host[i], op.output + i * 64);
                if (s->profile) s->profile->conversion_host_ns+=profile_elapsed(converted);
            }
        }
        if (write) s->stats.h2d_bytes += n * (op.kind == 0 ? 32 : 64);
        else s->stats.d2h_bytes += n * (op.kind == 0 ? 32 : 64);
        if (write) s->diagnostics.physical_h2d_bytes += n * value.width();
        else s->diagnostics.physical_d2h_bytes += n * (device_affine ? sizeof(affine_t) : value.width());
    } else if (op.opcode == DVGPU_RESIDENT_COPY || op.opcode == DVGPU_RESIDENT_REVERSE) {
        auto& src = s->range(op.a, op.a_offset, n);
        require(src.kind<=1,"packed MSM bases cannot be copied or reversed");
        auto& dst = s->range(op.c, op.c_offset, n, src.kind, true);
        if (op.opcode == DVGPU_RESIDENT_REVERSE) require(src.kind == op.kind, "reverse kind mismatch");
        alias(op.a, op.a_offset, op.c, op.c_offset, n,
            op.opcode == DVGPU_RESIDENT_COPY || n <= 1);
        if (!n || (op.opcode == DVGPU_RESIDENT_COPY && op.a == op.c && op.a_offset == op.c_offset)) return 0;
        s->save(op.c, op.c_offset, n);
        if (op.opcode == DVGPU_RESIDENT_COPY) {
            checked(cudaMemcpyAsync(dst.bytes(op.c_offset), src.bytes(op.a_offset), n * src.width(), cudaMemcpyDeviceToDevice, s->stream));
            s->diagnostics.explicit_d2d_bytes += n * src.width();
        }
        else if (src.kind == 0) reverse_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(reinterpret_cast<scalar_t*>(src.bytes(op.a_offset)), reinterpret_cast<scalar_t*>(dst.bytes(op.c_offset)), n);
        else reverse_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(reinterpret_cast<projective_t*>(src.bytes(op.a_offset)), reinterpret_cast<projective_t*>(dst.bytes(op.c_offset)), n);
    } else if (op.opcode == DVGPU_RESIDENT_TRANSFORM_PACKET) {
        require(op.kind <= 1 && op.flags <= 1 && n && !(n & (n - 1)) &&
            n <= DVGPU_MAX_TRANSFORM_PACKET_SIZE && op.a != op.b &&
            op.a != op.c && op.b != op.c, "invalid resident packet shape, kind or alias");
        auto& src = s->range(op.a, 0, n, op.kind);
        auto& scratch = s->range(op.b, 0, n, op.kind, true);
        auto& dst = s->range(op.c, 0, n, op.kind, true);
        scalar_t root;
        checked(dvgpu_ntt_four_step_root(s->ntt, int(n), bool(op.flags), &root));
        s->preflight_pair(op.b, op.c, n);
        s->save(op.b, 0, n);
        s->save(op.c, 0, n);
        unsigned logarithm = 0;
        for (size_t size = n; size > 1; size >>= 1) ++logarithm;
        const size_t a = size_t(1) << (logarithm / 2), b = n / a;
        s->operation_begin();
        // Same five phases, layouts and inverse scaling as FourStepPlan.
        // No borrowed host data survives and complete() below joins the stream.
        if (op.kind == 0) {
            const auto* input = static_cast<scalar_t*>(src.pointer);
            auto* temporary = static_cast<scalar_t*>(scratch.pointer);
            auto* output = static_cast<scalar_t*>(dst.pointer);
            transpose_elements<<<(n+127)/128,128,0,s->stream>>>(input,output,a,b,0,n,false,root);
            checked(dvgpu_ntt_rows_async(s->ntt,output,int(a),int(b),bool(op.flags),context,temporary));
            transpose_elements<<<(n+127)/128,128,0,s->stream>>>(temporary,output,b,a,0,n,true,root);
            checked(dvgpu_ntt_rows_async(s->ntt,output,int(b),int(a),bool(op.flags),context,temporary));
            transpose_elements<<<(n+127)/128,128,0,s->stream>>>(temporary,output,a,b,0,n,false,root);
            s->stats.fr_transform_elements += 2*n;
        } else {
            const auto* input = static_cast<projective_t*>(src.pointer);
            auto* temporary = static_cast<projective_t*>(scratch.pointer);
            auto* output = static_cast<projective_t*>(dst.pointer);
            transpose_elements<<<(n+127)/128,128,0,s->stream>>>(input,output,a,b,0,n,false,root);
            checked(dvgpu_ecntt_rows_async(s->ntt,output,int(a),int(b),bool(op.flags),context,temporary));
            transpose_elements<<<(n+127)/128,128,0,s->stream>>>(temporary,output,b,a,0,n,true,root);
            checked(dvgpu_ecntt_rows_async(s->ntt,output,int(b),int(a),bool(op.flags),context,temporary));
            transpose_elements<<<(n+127)/128,128,0,s->stream>>>(temporary,output,a,b,0,n,false,root);
            s->stats.g1_transform_elements += 2*n;
        }
        // Explicit bridge traffic, including row-length-one copies. NTT's
        // internal workspace traffic is not represented by this counter.
        s->diagnostics.explicit_d2d_bytes += ((a==1)+(b==1))*n*src.width();
    } else if (op.opcode == DVGPU_RESIDENT_TRANSPOSE) {
        require(op.kind<=1,"packed MSM bases cannot be transposed");
        const size_t rows = op.rows, cols = op.cols, start = op.a_offset;
        const auto total = checked_product(rows, cols);
        require(rows && cols && total <= DVGPU_MAX_TRANSFORM_MATRIX_SIZE && start <= total && n <= total - start && op.a != op.c, "invalid resident transpose shape or alias");
        auto& src = s->range(op.a, 0, total, op.kind);
        auto& dst = s->range(op.c, 0, total, op.kind, true);
        if (!n) return 0;
        scalar_t root = scalar_t::one();
        if (op.flags) { require(op.input, "null transpose root"); std::memcpy(&root, op.input, 32); }
        // Transposed outputs are contiguous within each column. Journal only
        // those touched ranges; never clone either complete candidate buffer.
        const size_t end = start + n, first = start / cols, last = (end - 1) / cols;
        const size_t count_cols = std::min(cols, n);
        for (size_t step = 0; step < count_cols; ++step) {
            const size_t col = (start % cols + step) % cols;
            const size_t lo = first + (col < start % cols);
            const size_t hi = last - (col > (end - 1) % cols);
            if (lo <= hi && hi < rows) s->save(op.c, col * rows + lo, hi - lo + 1);
        }
        s->operation_begin();
        if (op.kind == 0) transpose_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(static_cast<scalar_t*>(src.pointer), static_cast<scalar_t*>(dst.pointer), rows, cols, start, n, bool(op.flags), root);
        else transpose_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(static_cast<projective_t*>(src.pointer), static_cast<projective_t*>(dst.pointer), rows, cols, start, n, bool(op.flags), root);
    } else if (op.opcode == DVGPU_RESIDENT_TRANSFORM) {
        require(op.kind<=1,"packed MSM bases cannot be transformed");
        const size_t row_len = op.cols, count = op.rows;
        const size_t offset = checked_product(op.a_offset, row_len), total = checked_product(row_len, count);
        require(row_len && !(row_len & (row_len - 1)) && row_len <= DVGPU_MAX_NTT_SIZE && count <= INT_MAX && total <= DVGPU_MAX_TRANSFORM_MATRIX_SIZE && op.a != op.c, "invalid resident transform shape or alias");
        auto& src = s->range(op.a, offset, total, op.kind);
        auto& dst = s->range(op.c, offset, total, op.kind, true);
        if (!total) return 0;
        s->save(op.c, offset, total);
        s->operation_begin();
        if (op.kind == 0) {
            checked(dvgpu_ntt_rows(s->ntt, reinterpret_cast<scalar_t*>(src.bytes(offset)), int(row_len), int(count), bool(op.flags), context, reinterpret_cast<scalar_t*>(dst.bytes(offset))));
            s->stats.fr_transform_elements += total;
        } else {
            checked(dvgpu_ecntt_rows(s->ntt, reinterpret_cast<projective_t*>(src.bytes(offset)), int(row_len), int(count), bool(op.flags), context, reinterpret_cast<projective_t*>(dst.bytes(offset))));
            s->stats.g1_transform_elements += total;
        }
    } else if (op.opcode == DVGPU_RESIDENT_ZIP || op.opcode == DVGPU_RESIDENT_SCALE) {
        const bool scale = op.opcode == DVGPU_RESIDENT_SCALE;
        auto& left = s->range(op.a, op.a_offset, n, scale ? 1 : 0);
        auto& right = s->range(op.b, op.b_offset, n, 0);
        auto& dst = s->range(op.c, op.c_offset, n, scale ? 1 : 0, true);
        require(scale || op.flags <= 2, "invalid resident field binary operation");
        alias(op.a, op.a_offset, op.c, op.c_offset, n, true);
        alias(op.b, op.b_offset, op.c, op.c_offset, n, true);
        if (!n) return 0;
        s->save(op.c, op.c_offset, n);
        if (scale) scale_elements<<<(n + 127) / 128, 128, 0, s->stream>>>(reinterpret_cast<projective_t*>(left.bytes(op.a_offset)), reinterpret_cast<scalar_t*>(right.bytes(op.b_offset)), reinterpret_cast<projective_t*>(dst.bytes(op.c_offset)), n);
        else zip_fields<<<(n + 127) / 128, 128, 0, s->stream>>>(reinterpret_cast<scalar_t*>(left.bytes(op.a_offset)), reinterpret_cast<scalar_t*>(right.bytes(op.b_offset)), reinterpret_cast<scalar_t*>(dst.bytes(op.c_offset)), n, op.flags);
    } else if (op.opcode == DVGPU_RESIDENT_MSM_G1 || op.opcode == DVGPU_RESIDENT_MSM_G2) {
        auto& scalars = s->range(op.b, op.b_offset, n, 0);
        require(n <= DVGPU_MAX_MSM_SIZE && op.output, "invalid resident MSM length or output");
        auto config = msm::default_msm_config(context);
        config.are_scalars_on_device = true;
        config.are_points_on_device = op.opcode == DVGPU_RESIDENT_MSM_G1;
        config.are_scalars_montgomery_form = false;
        config.are_points_montgomery_form = false;
        config.are_results_on_device = false;
        config.is_async = false; config.is_big_triangle = true; config.large_bucket_factor = 0;
        if (op.opcode == DVGPU_RESIDENT_MSM_G1) {
            auto& points = s->range(op.a, op.a_offset, n);
            require(points.kind==1 || points.kind==2,"resident MSM point kind mismatch");
            require(points.kind==2 || n<=s->affine_capacity,"resident affine workspace exceeded");
            projective_t result = projective_t::zero();
            if (n) {
                const affine_t* bases=reinterpret_cast<affine_t*>(points.bytes(op.a_offset));
                if (points.kind==1) {
                    normalize_elements<<<(n+127)/128,128,0,s->stream>>>(
                        reinterpret_cast<projective_t*>(points.bytes(op.a_offset)),s->affine_workspace,n);
                    bases=s->affine_workspace;
                }
                checked(dvgpu_g1_msm_run(reinterpret_cast<scalar_t*>(scalars.bytes(op.b_offset)),bases,int(n),&config,&result));
            }
            checked(cudaStreamSynchronize(s->stream)); publish_point(result, op.output);
            s->stats.msm_g1_points += n; s->stats.d2h_bytes += 64;
            if (n) s->diagnostics.physical_d2h_bytes += sizeof(projective_t);
        } else {
            require(!n || op.input, "null resident G2 bases");
            static_assert(sizeof(g2_affine_t) == 128, "G2 Fq2 layout changed");
            std::vector<g2_affine_t> points(n);
            if (n) std::memcpy(points.data(), op.input, n * 128);
            g2_projective_t result = g2_projective_t::zero();
            if (n) checked(dvgpu_g2_msm_run(reinterpret_cast<scalar_t*>(scalars.bytes(op.b_offset)), points.data(), int(n), &config, &result));
            checked(cudaStreamSynchronize(s->stream)); publish_point(result, op.output);
            s->stats.msm_g2_points += n; s->stats.h2d_bytes += n * 128; s->stats.d2h_bytes += 128;
            s->diagnostics.physical_h2d_bytes += n * sizeof(g2_affine_t);
            if (n) s->diagnostics.physical_d2h_bytes += sizeof(g2_projective_t);
        }
    } else throw std::invalid_argument("unknown resident opcode");
    s->complete();
    return 0;
}

uint64_t dvgpu_resident_execute_profiled(DvgpuResidentState* state,const DvgpuResidentOp& op,DvgpuResidentProfile* profile) {
    require(state && profile && !state->profile,"invalid or nested resident profile");
    static_assert(sizeof(DvgpuResidentProfile)==12*sizeof(uint64_t),"resident profile ABI changed");
    // Clear the stack borrow on every exception before the outer owner can retire.
    struct ProfileGuard {
        DvgpuResidentState* state;
        ~ProfileGuard() {state->profile=nullptr;state->operation_active=false;}
    } guard{state};
    *profile={};state->profile=profile;state->operation_active=false;
    const auto before=state->diagnostics;
    const auto result=dvgpu_resident_execute(state,op);
    profile->h2d_bytes=state->diagnostics.physical_h2d_bytes-before.physical_h2d_bytes;
    profile->d2h_bytes=state->diagnostics.physical_d2h_bytes-before.physical_d2h_bytes;
    profile->d2d_bytes=state->diagnostics.explicit_d2d_bytes-before.explicit_d2d_bytes;
    return result;
}
