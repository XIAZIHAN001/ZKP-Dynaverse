#pragma once
#include <stdint.h>
// Explicit test ABI. One-shot corruption of the next nonempty checked G2 call,
// after normalize and before validation; index<4096. No production env switch
// or safe public Rust setter. Caller uses only its own serialized test owner.
extern "C" int dvgpu_g2_test_corrupt_next(void* owner,uint64_t index);
