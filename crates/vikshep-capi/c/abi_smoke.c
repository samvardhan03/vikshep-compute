/* C smoke test of include/vikshep_backend.h: drives the CPU reference
 * through its C table and checks a few exact results. */
#include <stdio.h>
#include <string.h>

#include "vikshep_backend.h"

static int failures = 0;

static void expect(int cond, const char *what) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", what);
    failures++;
  }
}

int main(void) {
  VkspBackendV1 b = vksp_cpu_backend_v1();
  expect(b.abi_version == VKSP_ABI_VERSION, "ABI version");
  expect(strcmp(b.name(b.ctx), "cpu") == 0, "name");
  expect(b.numerics_version(b.ctx) == 2, "numerics version");

  /* FFT of an impulse is all ones; modulus of (3, -4) is 5. */
  VkspComplex32 d[4] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
  VkspCanvas c = {1, 4};
  VkspComplex32 tw[2] = {{1, 0}, {0, -1}};
  expect(b.fft(b.ctx, d, 1, c, tw, NULL) == VKSP_OK, "fft status");
  for (int i = 0; i < 4; i++) {
    expect(d[i].re == 1.0f && d[i].im == 0.0f, "fft of impulse");
  }
  VkspComplex32 m[1] = {{3, -4}};
  expect(b.modulus(b.ctx, m, 1) == VKSP_OK && m[0].re == 5.0f && m[0].im == 0.0f, "modulus");

  /* Filter multiplication and subsampling. */
  float bank[4] = {2, 2, 2, 2};
  uint32_t index[1] = {0};
  expect(b.mul_real_filter(b.ctx, d, 1, c, bank, 1, index) == VKSP_OK && d[2].re == 2.0f,
         "mul_real_filter");
  VkspSubsample spec = {2, 0, 1, 1, 1};
  float out[1] = {0};
  expect(b.subsample(b.ctx, d, 1, c, spec, out) == VKSP_OK && out[0] == 2.0f, "subsample");

  /* Invalid arguments are rejected, not executed. */
  expect(b.fft(b.ctx, d, 1, c, NULL, NULL) == VKSP_INVALID_ARGUMENT, "invalid twiddles");

  b.destroy(b.ctx);
  if (failures == 0) {
    printf("abi_smoke: OK\n");
  }
  return failures == 0 ? 0 : 1;
}
