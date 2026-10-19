// Serial float32 MGARD API, matching the benchmark's pinned CLI options.
#include "compress_x.hpp"
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
static thread_local std::string last_error;
static mgard_x::Config config() {
  mgard_x::Config c;
  c.dev_type = mgard_x::device_type::SERIAL;
  c.log_level = mgard_x::log::ERR;
  c.fuse_decompose_quantize = c.fuse_dequantize_recompose = true;
  c.decomposition = mgard_x::decomposition_type::MultiDim;
  c.num_local_refactoring_level = 1;
  c.num_global_refactoring_level = 0;
  c.projection_mode = mgard_x::compression_projection_mode_type::Auto;
  c.domain_decomposition = mgard_x::domain_decomposition_type::MaxDim;
  c.cpu_mode = mgard_x::cpu_parallelization_mode::INTRA_BLOCK;
  c.domain_decomposition_dim = 0;
  c.estimate_outlier_ratio = 1.0;
  c.auto_pin_host_buffers = true;
  c.max_memory_footprint = std::numeric_limits<mgard_x::SIZE>::max();
  c.huff_dict_size = 8192;
  c.adjust_shape = false;
  // Release per-call allocations instead of retaining shape-dependent caches
  // across fields. This also matches the old fresh-process lifetime.
  c.auto_cache_release = true;
  c.lossless = mgard_x::lossless_type::Huffman_Zstd;
  return c;
}
extern "C" {
const char* cb_error() { return last_error.c_str(); }
void cb_free(void* p) { std::free(p); }
int cb_compress(const float* input, int ndim, const uint64_t* dims,
                double bound, int mode, int level, int target,
                void** output, size_t* size) {
  *output = nullptr;
  try {
    if (ndim != 2 && ndim != 3) throw std::runtime_error("Expected 2D or 3D input");
    std::vector<mgard_x::SIZE> shape(dims, dims + ndim);
    auto status = mgard_x::compress(ndim, mgard_x::data_type::Float, shape, bound,
        std::numeric_limits<double>::infinity(), mgard_x::error_bound_type::ABS,
        input, *output, *size, config(), false);
    if (status != mgard_x::compress_status_type::Success)
      throw std::runtime_error("MGARD compression failed");
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
int cb_decompress(void* input, size_t size, float* output) {
  try {
    void* destination = output;
    auto status = mgard_x::decompress(input, size, destination, config(), true);
    if (status != mgard_x::compress_status_type::Success)
      throw std::runtime_error("MGARD decompression failed");
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
}
