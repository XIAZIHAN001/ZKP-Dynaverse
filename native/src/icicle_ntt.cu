#include "icicle_ntt_twiddles_with_cleanup.cuh"
#include "icicle_ntt_with_owner.cuh"

struct DvgpuNttState {
    dvgpu_ntt::Domain<bn254::scalar_t> domain;
    bn254::scalar_t root = bn254::scalar_t::zero();
    bool has_root = false;

    void release_device() noexcept {
        // Also release partially initialized domains. ICICLE's upstream
        // release_domain nulls the managed pointer without freeing it.
        cudaDeviceSynchronize();
        bn254::scalar_t* pointers[] = {
            domain.twiddles, domain.internal_twiddles, domain.basic_twiddles,
            domain.fast_external_twiddles, domain.fast_internal_twiddles,
            domain.fast_basic_twiddles, domain.fast_external_twiddles_inv,
            domain.fast_internal_twiddles_inv, domain.fast_basic_twiddles_inv};
        for (auto pointer : pointers) { if (pointer) cudaFree(pointer); }
    }
};

namespace {
struct DomainBinding {
    dvgpu_ntt::Domain<bn254::scalar_t>* previous;
    explicit DomainBinding(DvgpuNttState& state)
        : previous(dvgpu_ntt::bound_domain<bn254::scalar_t>) {
        dvgpu_ntt::bound_domain<bn254::scalar_t> = &state.domain;
    }
    ~DomainBinding() { dvgpu_ntt::bound_domain<bn254::scalar_t> = previous; }
    DomainBinding(const DomainBinding&) = delete;
    DomainBinding& operator=(const DomainBinding&) = delete;
};
}

DvgpuNttState* dvgpu_ntt_create() { return new DvgpuNttState; }
void dvgpu_ntt_destroy(DvgpuNttState* state, bool context_current) noexcept {
    // Always free host state, even when the failed context cannot be pushed.
    // Never free this owner's device pointers in an unrelated current context.
    if (context_current) state->release_device();
    delete state;
}

cudaError_t dvgpu_ntt_run(DvgpuNttState* state, bn254::scalar_t root,
    const bn254::scalar_t* input, int length, bool inverse,
    device_context::DeviceContext& context, bn254::scalar_t* output) {
    auto status = dvgpu_ntt_prepare(state, root, context);
    if (status != cudaSuccess) return status;
    DomainBinding binding(*state);
    auto config = dvgpu_ntt::default_ntt_config<bn254::scalar_t>(context);
    return dvgpu_ntt::ntt<bn254::scalar_t, bn254::scalar_t>(input, length,
        inverse ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, config, output);
}

cudaError_t dvgpu_ntt_prepare(DvgpuNttState* state, bn254::scalar_t root,
    device_context::DeviceContext& context) {
    DomainBinding binding(*state);
    if (state->has_root && !(root == state->root)) return cudaErrorInvalidValue;
    if (!state->has_root) {
        const auto status = dvgpu_ntt::init_domain(root, context, false);
        if (status != cudaSuccess) return status;
        state->root = root;
        state->has_root = true;
    }
    return cudaSuccess;
}

template <typename E>
cudaError_t resident_rows(DvgpuNttState* state, const E* input, int length,
    int count, bool inverse, device_context::DeviceContext& context, E* output,
    bool async = false) {
    DomainBinding binding(*state);
    auto config = dvgpu_ntt::default_ntt_config<bn254::scalar_t>(context);
    config.batch_size = count;
    config.columns_batch = false;
    config.ordering = ntt::Ordering::kNN;
    config.ntt_algorithm = ntt::NttAlgorithm::Radix2;
    config.are_inputs_on_device = true;
    config.are_outputs_on_device = true;
    config.is_async = async;
    if (length == 1) return cudaMemcpyAsync(output, input,
        size_t(count) * sizeof(E), cudaMemcpyDeviceToDevice, context.stream);
    return dvgpu_ntt::ntt<bn254::scalar_t, E>(input, length,
        inverse ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, config, output);
}
cudaError_t dvgpu_ntt_rows(DvgpuNttState* state, const bn254::scalar_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::scalar_t* output) {
    return resident_rows(state, input, length, count, inverse, context, output);
}
extern "C" cudaError_t dvgpu_ecntt_rows(DvgpuNttState* state, const bn254::projective_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::projective_t* output) {
    return resident_rows(state, input, length, count, inverse, context, output);
}
cudaError_t dvgpu_ntt_rows_async(DvgpuNttState* state, const bn254::scalar_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::scalar_t* output) {
    return resident_rows(state, input, length, count, inverse, context, output, true);
}
extern "C" cudaError_t dvgpu_ecntt_rows_async(DvgpuNttState* state, const bn254::projective_t* input,
    int length, int count, bool inverse, device_context::DeviceContext& context,
    bn254::projective_t* output) {
    return resident_rows(state, input, length, count, inverse, context, output, true);
}
cudaError_t dvgpu_ntt_four_step_root(DvgpuNttState* state, int length, bool inverse,
    bn254::scalar_t* output) {
    constexpr unsigned order = 1u << 20;
    if (!state || !state->has_root || !output || length <= 0 ||
        (length & (length - 1)) || unsigned(length) > order) return cudaErrorInvalidValue;
    // Configuration validated the exact order of this owner's root. Derive
    // its subgroup root without reading managed twiddles or accepting a second
    // unrelated caller root. Inverse row normalization remains ICICLE's job.
    unsigned exponent = inverse ? order - order / unsigned(length) : order / unsigned(length);
    auto value = bn254::scalar_t::one(), base = state->root;
    while (exponent) {
        if (exponent & 1) value = value * base;
        exponent >>= 1;
        if (exponent) base = bn254::scalar_t::sqr(base);
    }
    *output = value;
    return cudaSuccess;
}
