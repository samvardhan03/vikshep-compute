// vikshep_example.cpp - the stable C API (include/vikshep.h) from C++:
// small RAII-free wrappers that turn status codes into exceptions and use
// the size-query protocol to fill std::vector and std::string.
//
// Build (after `cargo build --release -p vikshep-capi`), on one line:
//   c++ -std=c++17 -Wall -Wextra -Werror -pedantic -Iinclude
//       examples/cpp/vikshep_example.cpp target/release/libvikshep_capi.a
//       <native libraries printed by rustc> -o vikshep_example_cpp
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Samvardhan Singh

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "vikshep.h"

namespace vksp {

[[noreturn]] void raise(int32_t status) {
  size_t n = 0;
  vksp_last_error(nullptr, 0, &n);
  std::string msg(n, '\0');
  vksp_last_error(msg.data(), n + 1, &n);
  throw std::runtime_error(std::string(vksp_status_string(status)) + ": " + msg);
}

void check(int32_t status) {
  if (status != VKSP_OK) raise(status);
}

struct Tensor {
  std::vector<float> values;
  std::string oid;
};

Tensor scatter(const VkspScatterConfig& cfg, const std::vector<float>& x,
               const char* backend = nullptr) {
  size_t n = 0;
  check(vksp_scatter(&cfg, backend, x.data(), x.size(), nullptr, 0, &n, nullptr));
  Tensor t{std::vector<float>(n), std::string(VKSP_OID_LEN, '\0')};
  check(vksp_scatter(&cfg, backend, x.data(), x.size(), t.values.data(), n, &n, t.oid.data()));
  return t;
}

Tensor r2(const VkspScatterConfig& cfg, const Tensor& s) {
  size_t n = 0;
  check(vksp_r2(&cfg, s.values.data(), s.values.size(), nullptr, 0, &n, nullptr));
  Tensor t{std::vector<float>(n), std::string(VKSP_OID_LEN, '\0')};
  check(vksp_r2(&cfg, s.values.data(), s.values.size(), t.values.data(), n, &n, t.oid.data()));
  return t;
}

std::string oid_of(const std::vector<float>& x) {
  std::string o(VKSP_OID_LEN, '\0');
  check(vksp_oid(x.data(), x.size() * sizeof(float), o.data()));
  return o;
}

std::string manifest(const VkspScatterConfig& cfg, size_t batch, const std::string& in,
                     const std::string& out, std::string* hash) {
  size_t n = 0;
  check(vksp_provenance_manifest(&cfg, batch, in.c_str(), out.c_str(), nullptr, nullptr,
                                 VKSP_EXECUTOR_LOCAL, 0, 0, nullptr, 0, &n, nullptr));
  std::string json(n, '\0');
  hash->assign(VKSP_SHA3_HEX_LEN, '\0');
  check(vksp_provenance_manifest(&cfg, batch, in.c_str(), out.c_str(), nullptr, nullptr,
                                 VKSP_EXECUTOR_LOCAL, 0, 0, json.data(), n + 1, &n,
                                 hash->data()));
  return json;
}

}  // namespace vksp

int main() {
  try {
    // 2-D, 32x32, J = 3, L = 4, circular padding, trivial group.
    VkspScatterConfig cfg;
    vksp::check(vksp_config_2d(32, 32, 3, 4, VKSP_PAD_CIRCULAR, VKSP_PAD_CIRCULAR,
                               VKSP_GROUP_TRIVIAL, &cfg));
    std::vector<float> image(32 * 32);
    for (size_t i = 0; i < image.size(); i++) {
      image[i] = static_cast<float>((i * 13) % 17) / 17.0f;
    }
    vksp::Tensor s = vksp::scatter(cfg, image);
    vksp::Tensor ratios = vksp::r2(cfg, s);
    std::string in_oid = vksp::oid_of(image);
    std::string hash;
    std::string json = vksp::manifest(cfg, 1, in_oid, s.oid, &hash);
    std::printf("coefficients %s (%zu floats), r2 %s (%zu floats)\n", s.oid.c_str(),
                s.values.size(), ratios.oid.c_str(), ratios.values.size());
    std::printf("manifest_hash %s\n", hash.c_str());

    // An error surfaces as an exception with the library's message.
    try {
      vksp::scatter(cfg, image, "no-such-backend");
      return 1;
    } catch (const std::runtime_error& e) {
      std::printf("expected error: %s\n", e.what());
    }

    size_t cases = 0, failed = 0;
    vksp::check(vksp_conformance_selftest("quick", nullptr, &cases, &failed, nullptr, 0,
                                          nullptr));
    std::printf("self-test: %zu cases, %zu failed\nOK\n", cases, failed);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
