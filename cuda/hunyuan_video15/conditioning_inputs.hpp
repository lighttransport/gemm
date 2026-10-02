#ifndef PIXAL3D_HV15_CONDITIONING_INPUTS_HPP
#define PIXAL3D_HV15_CONDITIONING_INPUTS_HPP
#include <cmath>
#include <fstream>
#include "core/tensor.hpp"
inline sd::Tensor<float> hv15_siglip_pixels;
inline bool hv15_load_siglip_pixels(const char *path) {
    hv15_siglip_pixels = {};
    if (!path || !*path) return true;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    constexpr size_t count = 384 * 384 * 3;
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(float))) return false;
    file.seekg(0);
    sd::Tensor<float> pixels({384, 384, 3, 1});
    file.read(reinterpret_cast<char *>(pixels.data()), count * sizeof(float));
    if (!file) return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(pixels.data()[i]) || std::fabs(pixels.data()[i]) > 1.000001f) return false;
    hv15_siglip_pixels = std::move(pixels);
    return true;
}
#endif
