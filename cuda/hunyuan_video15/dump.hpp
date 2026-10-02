#ifndef PIXAL3D_HV15_DUMP_HPP
#define PIXAL3D_HV15_DUMP_HPP
#include <cstdlib>
#include <fstream>
#include <set>
#include "core/tensor.hpp"
/* Diagnostics only. First occurrence by default; latent_final is overwritten. */
inline void hv15_dump(const char *name, const sd::Tensor<float>& tensor, bool overwrite = false) {
    const char *dir = std::getenv("HV15_DUMP_DIR");
    if (!dir || !*dir || tensor.empty()) return;
    static std::set<std::string> written;
    std::string path = std::string(dir) + "/" + name;
    if (!overwrite && written.count(path)) return;
    std::ofstream data(path + ".f32", std::ios::binary);
    std::ofstream shape(path + ".shape.json");
    if (!data || !shape) throw std::runtime_error("cannot write HV15_DUMP_DIR");
    const auto& dims = tensor.shape();
    size_t count = 1;
    shape << '[';
    for (size_t i = 0; i < dims.size(); ++i) {
        count *= static_cast<size_t>(dims[i]);
        if (i) shape << ',';
        shape << dims[dims.size() - i - 1];
    }
    shape << "]\n";
    data.write(reinterpret_cast<const char *>(tensor.data()), count * sizeof(float));
    if (!data || !shape) throw std::runtime_error("failed writing native tensor dump");
    written.insert(path);
}
inline void hv15_dump_sigmas(const std::vector<float>& sigmas) {
    const char *dir=std::getenv("HV15_DUMP_DIR");
    if (!dir || !*dir) return;
    std::ofstream file(std::string(dir)+"/sigmas.json");
    if (!file) throw std::runtime_error("cannot write denoising schedule");
    file.precision(9); file << '[';
    for (size_t i=0;i<sigmas.size();++i) {
        if (!std::isfinite(sigmas[i])) throw std::runtime_error("non-finite denoising schedule");
        if (i) file << ',';
        file << sigmas[i];
    }
    file << "]\n";
    if (!file) throw std::runtime_error("failed writing denoising schedule");
}
inline void hv15_override_noise(sd::Tensor<float>& tensor) {
    const char *path = std::getenv("HV15_NOISE_F32");
    if (!path || !*path) return;
    size_t count = 1;
    for (auto dim : tensor.shape()) count *= static_cast<size_t>(dim);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
        throw std::runtime_error("HV15_NOISE_F32 has the wrong byte count");
    file.seekg(0);
    file.read(reinterpret_cast<char *>(tensor.data()), count * sizeof(float));
    if (!file) throw std::runtime_error("failed reading HV15_NOISE_F32");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(tensor.data()[i])) throw std::runtime_error("non-finite reference noise");
}
#endif
