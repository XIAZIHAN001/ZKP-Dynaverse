"""Pinned ICICLE NTT host ownership repair; arithmetic and kernels stay intact."""
import hashlib
from pathlib import Path
import sys

source_path, output_path = map(Path, sys.argv[1:])
source = source_path.read_text(encoding="utf-8")
expected = "70a69470f6d244474821be24354f0d2d8da0b72003154ce8e0e184c273f6d6d1"
if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
    raise SystemExit("NTT source differs from pinned v2.8.0; review the owner repair")

def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise SystemExit(f"expected exactly one pinned fragment: {old!r}")
    source = source.replace(old, new)

# Keep the upstream config/enums: the official mixed-radix library's symbols
# include ntt::Ordering. Isolate only this copy's functions and domain storage.
replace_once("namespace ntt {", "namespace dvgpu_ntt {")
replace_once("  public:\n    template <typename U>\n    friend cudaError_t init_domain",
             "  public:\n    friend struct ::DvgpuNttState;\n    template <typename U>\n    friend cudaError_t init_domain")
replace_once("static inline Domain<S> domains_for_devices[device_context::MAX_DEVICES] = {};",
             "static inline thread_local Domain<S>* bound_domain = nullptr;")
if source.count("domains_for_devices<S>[") != 4:
    raise SystemExit("unexpected pinned domain access count")
source = source.replace("domains_for_devices<S>[ctx.device_id]", "*bound_domain<S>")
source = source.replace("domains_for_devices<S>[config.ctx.device_id]", "*bound_domain<S>")
source = source.replace("ntt::radix2_ntt(", "dvgpu_ntt::radix2_ntt(")
source = source.replace("mxntt::generate_external_twiddles_generic(",
                        "dvgpu_twiddles::generate_external_twiddles_generic(")

# WSL advertises concurrentManagedAccess=0. A host read of managed twiddles
# can fault while a different owner executes on GPU, even after this owner's
# stream is synchronized. Compute the identical small lookup table from the
# root, without touching managed memory. Transform kernels stay intact.
replace_once("        for (int i = 0; i < domain.max_log_size; ++i) {\n"
             "          const int index = (int)pow(2, i);\n"
             "          domain.coset_index[domain.twiddles[index]] = index;\n"
             "        }",
             "        S host_power = primitive_root;\n"
             "        for (int i = 0; i < domain.max_log_size; ++i) {\n"
             "          const int index = (int)pow(2, i);\n"
             "          domain.coset_index[host_power] = index;\n"
             "          host_power = S::sqr(host_power);\n"
             "        }")

# Track temporary transform buffers, never the domain's persistent allocations.
start = source.index("  template <typename S, typename E>\n  cudaError_t ntt(")
end = source.index("  template <typename S>\n  NTTConfig<S> default_ntt_config", start)
body = source[start:end]
if body.count("CHK_INIT_IF_RETURN();") != 1:
    raise SystemExit("unexpected pinned NTT allocation scope")
body = body.replace("CHK_INIT_IF_RETURN();", "CHK_INIT_IF_RETURN();\n    DynaverseMsmAllocations resources;")
body = body.replace("cudaMallocAsync(", "resources.allocate(")
body = body.replace("cudaFreeAsync(", "resources.release(")
source = source[:start] + body + source[end:]

prefix = """#include "icicle_ntt_owner.cuh"
#include "icicle_msm_allocations.cuh"
namespace dvgpu_ntt {
using ntt::NTTConfig;
using ntt::NTTDir;
using ntt::Ordering;
using ntt::NttAlgorithm;
template <typename S> cudaError_t init_domain(S, device_context::DeviceContext&, bool);
template <typename S> cudaError_t release_domain(device_context::DeviceContext&);
template <typename S> S get_root_of_unity(uint64_t);
template <typename S> S get_root_of_unity(uint64_t, device_context::DeviceContext&);
template <typename S> S get_root_of_unity_from_domain(uint64_t, device_context::DeviceContext&);
template <typename S, typename E>
cudaError_t ntt(const E*, int, NTTDir, NTTConfig<S>&, E*);
}
"""
# Only the generic domain generator needs its raw temporary allocation repair.
# Retain its four pinned kernel bodies; actual mixed-radix transforms still
# use the official archive. Refuse any upstream drift before extraction.
twiddle_path = source_path.with_name("kernel_ntt.cu")
if hashlib.sha256(twiddle_path.read_bytes()).hexdigest() != "73d09c94a85105524ed813588c922273a33e8a3263642f5ca2b6330fa6266cc3":
    raise SystemExit("mixed NTT source differs from pinned v2.8.0")
twiddles = twiddle_path.read_text(encoding="utf-8")
begin = twiddles.index("  template <typename S>\n  __global__ void generate_base_table(")
end = twiddles.index("  // Fast-twiddles: 2N twiddles", begin)
twiddles = twiddles[begin:end]
twiddles = twiddles.replace("CHK_INIT_IF_RETURN();", "CHK_INIT_IF_RETURN();\n    DynaverseMsmAllocations resources;")
# basic_twiddles is assigned directly into the domain, so is already owned.
twiddles = twiddles.replace("cudaMallocAsync(&w", "resources.allocate(&w")
twiddles = twiddles.replace("cudaFreeAsync(", "resources.release(")
twiddles = twiddles.replace("    cudaStreamSynchronize(stream);", "    CHK_IF_RETURN(cudaStreamSynchronize(stream));")
twiddles = twiddles.replace("    internal_twiddles = w6_table;", "    internal_twiddles = w6_table;\n    resources.detach(w6_table);")
twiddle_output = output_path.with_name("icicle_ntt_twiddles_with_cleanup.cuh")
twiddle_output.write_text('#include "icicle_ntt_owner.cuh"\n#include "icicle_msm_allocations.cuh"\n'
                         + "namespace dvgpu_twiddles {\n" + twiddles + "}\n", encoding="utf-8", newline="\n")
output_path.write_text(prefix + source, encoding="utf-8", newline="\n")
