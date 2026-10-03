/* vikshep_example.c - the stable C API (include/vikshep.h) end to end:
 * version, scattering with a size query, r2, fingerprint, provenance
 * manifest, an external backend registered through vikshep_backend.h, and
 * the conformance self-test.
 *
 * Build (after `cargo build --release -p vikshep-capi`):
 *   cc -std=c99 -Wall -Wextra -Werror -pedantic -Iinclude \
 *      examples/c/vikshep_example.c target/release/libvikshep_capi.a \
 *      <native libraries printed by rustc> -o vikshep_example
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Samvardhan Singh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vikshep.h"

#define CHECK(call)                                                        \
  do {                                                                     \
    int32_t s_ = (call);                                                   \
    if (s_ != VKSP_OK) {                                                   \
      char msg_[256];                                                      \
      size_t n_ = 0;                                                       \
      vksp_last_error(msg_, sizeof msg_, &n_);                             \
      fprintf(stderr, "%s:%d: %s -> %s (%s)\n", __FILE__, __LINE__, #call, \
              vksp_status_string(s_), msg_);                               \
      return 1;                                                            \
    }                                                                      \
  } while (0)

/* An external backend: forwards every kernel to the CPU reference table and
 * counts the calls. A CUDA or Metal backend fills the same table with its
 * own kernels (docs/external_backends.md). */
typedef struct {
  VkspBackendV1 inner;
  unsigned long calls;
} Forwarding;

static const char *fwd_name(void *ctx) {
  (void)ctx;
  return "forwarding-cpu";
}
static uint32_t fwd_numerics_version(void *ctx) {
  Forwarding *f = (Forwarding *)ctx;
  return f->inner.numerics_version(f->inner.ctx);
}
static VkspCapabilities fwd_capabilities(void *ctx) {
  Forwarding *f = (Forwarding *)ctx;
  return f->inner.capabilities(f->inner.ctx);
}
static int32_t fwd_fft(void *ctx, VkspComplex32 *data, size_t batch, VkspCanvas canvas,
                       const VkspComplex32 *tw_cols, const VkspComplex32 *tw_rows) {
  Forwarding *f = (Forwarding *)ctx;
  f->calls++;
  return f->inner.fft(f->inner.ctx, data, batch, canvas, tw_cols, tw_rows);
}
static int32_t fwd_ifft(void *ctx, VkspComplex32 *data, size_t batch, VkspCanvas canvas,
                        const VkspComplex32 *tw_cols, const VkspComplex32 *tw_rows) {
  Forwarding *f = (Forwarding *)ctx;
  f->calls++;
  return f->inner.ifft(f->inner.ctx, data, batch, canvas, tw_cols, tw_rows);
}
static int32_t fwd_mul(void *ctx, VkspComplex32 *data, size_t batch, VkspCanvas canvas,
                       const float *bank, size_t n_filters, const uint32_t *index) {
  Forwarding *f = (Forwarding *)ctx;
  f->calls++;
  return f->inner.mul_real_filter(f->inner.ctx, data, batch, canvas, bank, n_filters, index);
}
static int32_t fwd_modulus(void *ctx, VkspComplex32 *data, size_t len) {
  Forwarding *f = (Forwarding *)ctx;
  f->calls++;
  return f->inner.modulus(f->inner.ctx, data, len);
}
static int32_t fwd_subsample(void *ctx, const VkspComplex32 *input, size_t batch,
                             VkspCanvas canvas, VkspSubsample spec, float *out) {
  Forwarding *f = (Forwarding *)ctx;
  f->calls++;
  return f->inner.subsample(f->inner.ctx, input, batch, canvas, spec, out);
}
static unsigned long *g_calls = NULL;
static void fwd_destroy(void *ctx) {
  Forwarding *f = (Forwarding *)ctx;
  printf("forwarding-cpu destroyed after %lu kernel calls\n", f->calls);
  g_calls = NULL;
  f->inner.destroy(f->inner.ctx);
  free(f);
}

static int register_forwarding(void) {
  Forwarding *f = (Forwarding *)calloc(1, sizeof *f);
  if (f == NULL) return 1;
  f->inner = vksp_cpu_backend_v1();
  VkspBackendV1 t;
  memset(&t, 0, sizeof t);
  t.abi_version = VKSP_ABI_VERSION;
  t.ctx = f;
  t.name = fwd_name;
  t.numerics_version = fwd_numerics_version;
  t.capabilities = fwd_capabilities;
  t.fft = fwd_fft;
  t.ifft = fwd_ifft;
  t.mul_real_filter = fwd_mul;
  t.modulus = fwd_modulus;
  t.subsample = fwd_subsample;
  t.destroy = fwd_destroy;
  if (vksp_register_backend(&t) != VKSP_OK) {
    f->inner.destroy(f->inner.ctx); /* on failure the caller keeps ownership */
    free(f);
    return 1;
  }
  g_calls = &f->calls;
  return 0;
}

int main(void) {
  VkspVersionInfo v;
  memset(&v, 0, sizeof v);
  v.struct_size = sizeof v;
  CHECK(vksp_version_info(&v));
  printf("vikshep-compute %s (%s): api %u, numerics_version %u, tier2_version %u\n",
         v.package_version, v.platform, v.api_version, v.numerics_version, v.tier2_version);

  /* 1-D, 256 samples, J = 4, Q = 1, zero padding. */
  VkspScatterConfig cfg;
  CHECK(vksp_config_1d(256, 4, 1, VKSP_PAD_ZERO, &cfg));
  size_t signal_len = 0, n_paths = 0, out_len = 0, n_r2 = 0;
  CHECK(vksp_scatter_info(&cfg, &signal_len, &n_paths, &out_len, &n_r2));
  printf("signal_len %zu, paths %zu, samples per path %zu, r2 pairs %zu\n", signal_len,
         n_paths, out_len, n_r2);

  enum { BATCH = 2 };
  float x[BATCH * 256];
  for (int i = 0; i < BATCH * 256; i++) x[i] = (float)((i * 37) % 101) / 101.0f - 0.5f;
  char in_oid[VKSP_OID_LEN + 1];
  CHECK(vksp_oid(x, sizeof x, in_oid));

  size_t n = 0;
  CHECK(vksp_scatter(&cfg, NULL, x, BATCH * 256, NULL, 0, &n, NULL)); /* size query */
  float *s = (float *)malloc(n * sizeof *s);
  if (s == NULL) return 1;
  char s_oid[VKSP_OID_LEN + 1];
  CHECK(vksp_scatter(&cfg, NULL, x, BATCH * 256, s, n, &n, s_oid));
  printf("input %s -> coefficients %s (%zu floats)\n", in_oid, s_oid, n);

  size_t nr = 0, nf = 0;
  CHECK(vksp_r2(&cfg, s, n, NULL, 0, &nr, NULL));
  float *r = (float *)malloc(nr * sizeof *r);
  double fp[BATCH * 64];
  char r_oid[VKSP_OID_LEN + 1], fp_oid[VKSP_OID_LEN + 1];
  if (r == NULL) return 1;
  CHECK(vksp_r2(&cfg, s, n, r, nr, &nr, r_oid));
  CHECK(vksp_fingerprint(&cfg, s, n, fp, BATCH * 64, &nf, fp_oid));
  printf("r2 %s (%zu floats), fingerprint %s (%zu doubles)\n", r_oid, nr, fp_oid, nf);

  size_t mlen = 0;
  char hash[VKSP_SHA3_HEX_LEN + 1];
  CHECK(vksp_provenance_manifest(&cfg, BATCH, in_oid, s_oid, NULL, NULL, VKSP_EXECUTOR_LOCAL,
                                 0, 0, NULL, 0, &mlen, NULL));
  char *manifest = (char *)malloc(mlen + 1);
  if (manifest == NULL) return 1;
  CHECK(vksp_provenance_manifest(&cfg, BATCH, in_oid, s_oid, NULL, NULL, VKSP_EXECUTOR_LOCAL,
                                 0, 0, manifest, mlen + 1, &mlen, hash));
  printf("manifest_hash %s\n%s\n", hash, manifest);

  /* The same computation through an external backend. */
  if (register_forwarding() != 0) {
    fprintf(stderr, "registration failed\n");
    return 1;
  }
  char e_oid[VKSP_OID_LEN + 1];
  CHECK(vksp_scatter(&cfg, "forwarding-cpu", x, BATCH * 256, s, n, &n, e_oid));
  printf("forwarding-cpu -> coefficients %s after %lu kernel calls\n", e_oid,
         g_calls ? *g_calls : 0UL);
  if (strcmp(e_oid, s_oid) != 0) {
    fprintf(stderr, "external backend differs from the CPU reference\n");
    return 1;
  }

  size_t cases = 0, failed = 0;
  CHECK(vksp_conformance_selftest("quick", NULL, &cases, &failed, NULL, 0, NULL));
  printf("self-test (cpu): %zu cases, %zu failed\n", cases, failed);
  CHECK(vksp_conformance_selftest("quick", "forwarding-cpu", &cases, &failed, NULL, 0, NULL));
  printf("self-test (forwarding-cpu): %zu cases, %zu failed\n", cases, failed);
  CHECK(vksp_unregister_backend("forwarding-cpu"));

  free(manifest);
  free(r);
  free(s);
  printf("OK\n");
  return 0;
}
