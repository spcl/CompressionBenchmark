// C ABI around the same float32 API and configuration used by the HPEZ CLI.
#include "QoZ/api/sz.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

static thread_local std::string last_error;

extern "C" {
const char *hp_error() { return last_error.c_str(); }
void hp_free_stream(char *p) { delete[] p; }
void hp_free_field(float *p) { delete[] p; }

int hp_compress(const float *input, int ndim, const size_t *dims, double bound,
                int level, int target, char **output, size_t *size) {
    *output = nullptr;
    *size = 0;
    try {
        if (ndim != 2 && ndim != 3) throw std::runtime_error("Expected 2D or 3D input");
        if (!std::isfinite(bound) || bound <= 0 || level < 0 || level > 4 ||
                target < 0 || target > 3)
            throw std::runtime_error("Invalid HPEZ bound, level or tuning target");
        QoZ::Config conf;
        conf.setDims(dims, dims + ndim);
        conf.errorBoundMode = QoZ::EB_ABS;
        conf.absErrorBound = bound;
        conf.QoZ = level;
        conf.tuningTarget = target;
        *output = SZ_compress<float>(conf, input, *size);
        if (!*output || !*size) throw std::runtime_error("HPEZ encoding failed");
        return 0;
    } catch (const std::exception &e) {
        last_error = e.what();
        return 1;
    }
}

int hp_decompress(char *input, size_t size, float **output, size_t *count) {
    *output = nullptr;
    *count = 0;
    try {
        // The upstream decoder loads a private Config copy, not its reference argument.
        // Read the same trailer here to report the actual decoded element count.
        if (size < sizeof(int)) throw std::runtime_error("Truncated HPEZ stream");
        int config_size;
        std::memcpy(&config_size, input + size - sizeof(int), sizeof(int));
        if (config_size <= 0 || static_cast<size_t>(config_size) > size - sizeof(int))
            throw std::runtime_error("Invalid HPEZ configuration size");
        QoZ::Config conf;
        const auto *config_data = reinterpret_cast<const QoZ::uchar *>(
            input + size - sizeof(int) - config_size);
        conf.load(config_data);
        SZ_decompress<float>(conf, input, size, *output);
        *count = conf.num;
        if (!*output) throw std::runtime_error("HPEZ decoding failed");
        return 0;
    } catch (const std::exception &e) {
        last_error = e.what();
        return 1;
    }
}
}
