#include "dynaverse_gpu.h"
#include <array>
#include <cassert>

// Compile before implementing the ABI; then run through the Rust tests for
// independent arkworks expectations rather than a native round trip alone.
int main() {
    void* owner = nullptr;
    char error[256]{};
    assert(dvgpu_backend_create(&owner, error, sizeof(error)) == 0);
    std::array<uint8_t, 32> root{};
    DvgpuResidentOp operation{};
    DvgpuResidentStats stats{};
    operation.opcode = DVGPU_RESIDENT_ALLOC;
    operation.length = 16;
    uint64_t result = 0;
    assert(dvgpu_resident_call(owner, &operation, &result, error, sizeof(error)) != 0);
    assert(dvgpu_resident_stats(owner, &stats) == 0);
    assert(stats.bytes_allocated == 0);
    dvgpu_backend_destroy(owner);
}
