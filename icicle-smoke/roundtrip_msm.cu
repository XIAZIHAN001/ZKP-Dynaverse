// CPU->GPU->CPU roundtrip harness (day 2 closure).
//
// Protocol (all little-endian, canonical integers, NOT Montgomery):
//   input file:  u32 n || (n * 32B scalars) || (n * 64B affine points, x||y)
//   output file: u32 1 || 64B affine result (= sum_i s_i * P_i)
//
// This binary is intentionally a dumb executor: it loads bytes, calls the
// ICICLE v2.8.0 C ABI (bn254_msm_cuda) with Montgomery flags OFF (inputs are
// canonical), and writes back the affine result in canonical LE. All
// mathematical expectations live in the Rust test, so the CPU oracle stays
// single-sourced in arkworks.
//
// Verified against ICICLE v2.8.0 headers:
//   * bn254 fq/fr have limbs_count == 8, and storage<N>::limbs is uint32_t[N],
//     so a field element is 8 x u32 == 32 bytes, loaded as a plain LE limb run.
//   * field internals are canonical, not Montgomery (params_gen defines
//     one = {1}), so no value conversion happens on either side of the wire.
//   * Affine<FF> has exactly {x, y} -- there is NO `infinity` flag. Infinity is
//     represented as Affine::zero() == {0, 0}, and Field::inverse(zero) returns
//     zero, so Projective::to_affine(Projective::zero()) == {0, 0}. That is
//     bit-identical to the frozen codec's all-zero 64B infinity, so the generic
//     limb dump below already handles the point at infinity with no branch.
#include "api/bn254.h"
#include "msm/msm.cuh"
#include "gpu-utils/device_context.cuh"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace bn254;

// A bn254 field element is 8 little-endian u32 limbs (32 bytes).
static constexpr int FIELD_LIMBS = 8;
static constexpr int FIELD_BYTES = 32;
static constexpr int POINT_BYTES = 64;

static void bytes_to_limbs(const uint8_t* src, uint32_t* dst) {
  for (int i = 0; i < FIELD_LIMBS; i++) {
    uint32_t v = 0;
    for (int b = 0; b < 4; b++) v |= (uint32_t)src[i * 4 + b] << (8 * b);
    dst[i] = v;
  }
}

static void limbs_to_bytes(const uint32_t* src, uint8_t* dst) {
  for (int i = 0; i < FIELD_LIMBS; i++) {
    uint32_t v = src[i];
    for (int b = 0; b < 4; b++) dst[i * 4 + b] = (uint8_t)(v >> (8 * b));
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
    return 2;
  }

  // Static guards: if ICICLE ever changes limb width or struct layout, fail at
  // compile time rather than silently writing garbage across the boundary.
  static_assert(sizeof(scalar_t) == FIELD_BYTES, "scalar_t must be 32 bytes");
  static_assert(sizeof(affine_t) == POINT_BYTES, "affine_t must be 64 bytes (x||y)");

  FILE* fin = fopen(argv[1], "rb");
  if (!fin) { perror("open input"); return 2; }
  uint32_t n = 0;
  if (fread(&n, 4, 1, fin) != 1) { fprintf(stderr, "read n\n"); fclose(fin); return 2; }
  if (n == 0 || n > (1u << 20)) { fprintf(stderr, "bad n=%u\n", n); fclose(fin); return 2; }

  std::vector<uint8_t> sbytes((size_t)n * FIELD_BYTES);
  std::vector<uint8_t> pbytes((size_t)n * POINT_BYTES);
  if (fread(sbytes.data(), FIELD_BYTES, n, fin) != n) {
    fprintf(stderr, "read scalars\n"); fclose(fin); return 2;
  }
  if (fread(pbytes.data(), POINT_BYTES, n, fin) != n) {
    fprintf(stderr, "read points\n"); fclose(fin); return 2;
  }
  fclose(fin);

  std::vector<scalar_t> scalars(n);
  std::vector<affine_t> points(n);
  for (uint32_t i = 0; i < n; i++) {
    bytes_to_limbs(&sbytes[(size_t)i * FIELD_BYTES], scalars[i].limbs_storage.limbs);
    bytes_to_limbs(&pbytes[(size_t)i * POINT_BYTES + 0], points[i].x.limbs_storage.limbs);
    bytes_to_limbs(&pbytes[(size_t)i * POINT_BYTES + FIELD_BYTES], points[i].y.limbs_storage.limbs);
  }

  // GPU MSM, canonical (non-Montgomery) host inputs, synchronous.
  // NOTE: the MSMConfig doc comments claim the montgomery flags default to
  // true, but default_msm_config() actually sets them false. We set every
  // flag we depend on explicitly so the behaviour does not hinge on that.
  msm::MSMConfig cfg = msm::default_msm_config();
  cfg.are_scalars_on_device = false;
  cfg.are_scalars_montgomery_form = false;
  cfg.are_points_on_device = false;
  cfg.are_points_montgomery_form = false;
  cfg.are_results_on_device = false;
  cfg.is_async = false;

  projective_t result;
  cudaError_t err = bn254_msm_cuda(scalars.data(), points.data(), (int)n, cfg, &result);
  if (err != cudaSuccess) {
    fprintf(stderr, "msm error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  affine_t affine = projective_t::to_affine(result);

  // Infinity needs no special case: to_affine of the projective zero yields
  // {0, 0}, which is the frozen codec's all-zero encoding (see header note).
  uint8_t out[POINT_BYTES];
  limbs_to_bytes(affine.x.limbs_storage.limbs, &out[0]);
  limbs_to_bytes(affine.y.limbs_storage.limbs, &out[FIELD_BYTES]);

  FILE* fout = fopen(argv[2], "wb");
  if (!fout) { perror("open output"); return 2; }
  uint32_t one = 1;
  bool ok = fwrite(&one, 4, 1, fout) == 1 && fwrite(out, POINT_BYTES, 1, fout) == 1;
  if (fclose(fout) != 0 || !ok) { fprintf(stderr, "write output\n"); return 2; }
  printf("ROUNDTRIP_MSM_OK n=%u\n", n);
  return 0;
}
