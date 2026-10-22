// Real-CUDA ownership regression. Expected points are independent group math.
#include "dynaverse_gpu.h"
#include "api/bn254.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace bn254;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    void* owner = nullptr;
    try {
        const unsigned factor = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 0;
        constexpr size_t n = 1024;
        static_assert(sizeof(affine_t) == 64 && sizeof(scalar_t) == 32);
        const auto generator = projective_t::to_affine(projective_t::generator());
        std::vector<affine_t> points(n, generator);
        std::vector<scalar_t> scalars(n, scalar_t::from(3));
        affine_t output;
        char error[1024]{};
        auto create = [&] {
#ifdef DVGPU_TEST_CONFIGURED
            return dvgpu_backend_create_with_large_bucket_factor(factor, &owner, error, sizeof(error));
#else
            // Compile this same regression against the old ABI for the RED run.
            require(factor == 0, "legacy RED probe only supports factor zero");
            return dvgpu_backend_create(&owner, error, sizeof(error));
#endif
        };
        if (std::getenv("DVGPU_PROBE_EXPECT_CONFIG_REJECTION")) {
            require(create() != 0, "invalid factor was accepted");
            require(owner == nullptr, "invalid factor created a CUDA owner");
            require(std::strstr(error, "factor") != nullptr, "factor was not rejected before CUDA initialization");
            std::puts("NATIVE_CONFIG_REJECTION_OK");
            return 0;
        }
        require(create() == 0, error);
        const bool failure = std::getenv("DVGPU_PROBE_EXPECT_FAILURE") != nullptr;
        const char* path = std::getenv("DVGPU_PROBE_PATH");
        for (int iteration = 0; iteration < 4; ++iteration) {
            DvgpuMsmTimings timings{};
            DvgpuMsmTimeline timeline{};
            const auto* p = reinterpret_cast<const uint8_t*>(points.data());
            const auto* s = reinterpret_cast<const uint8_t*>(scalars.data());
            auto* out = reinterpret_cast<uint8_t*>(&output);
            int status;
            if (path && std::strcmp(path, "timeline") == 0)
                status = dvgpu_bn254_msm_timeline(owner, p, s, n, out, "RESOURCE_PROBE", &timings, &timeline, error, sizeof(error));
            else if (path && std::strcmp(path, "profile") == 0)
                status = dvgpu_bn254_msm_profiled(owner, p, s, n, out, &timings, error, sizeof(error));
            else
                status = dvgpu_bn254_msm(owner, p, s, n, out, error, sizeof(error));
            if (iteration == 0 && failure) {
                require(status != 0, "injected failure was not reached or propagated");
                std::fprintf(stderr, "EXPECTED_NATIVE_FAILURE %s\n", error);
                require(dvgpu_bn254_msm(owner, p, s, n, out, error, sizeof(error)) != 0,
                    "retired context accepted another call");
                require(std::strstr(error, "retired") != nullptr, "owner was not retired");
                dvgpu_backend_destroy(owner);
                owner = nullptr;
                require(create() == 0, error);
                continue;
            }
            require(status == 0, error);
            const auto expected = projective_t::to_affine(projective_t::generator() * scalar_t::from(3072));
            require(output == expected, "1024 repeated scalar MSM differs from 3072G");
        }
        dvgpu_backend_destroy(owner);
        owner = nullptr;
        std::printf("RESOURCE_PROBE_MATH_AND_RECOVERY_OK factor=%u path=%s\n", factor, path ? path : "ordinary");
        return 0;
    } catch (const std::exception& e) {
        dvgpu_backend_destroy(owner);
        std::fprintf(stderr, "RESOURCE_PROBE_FAILED %s\n", e.what());
        return 1;
    }
}
