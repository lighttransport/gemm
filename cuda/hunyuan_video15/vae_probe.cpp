/* Bounded 128x128 one/five-frame VAE probe; raw F32 inputs are RGB in [0,1]. */
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include "stable-diffusion.h"
#include "model/vae/hunyuan_vae.hpp"
#include "core/hv15_dump.hpp"

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: %s VAE.safetensors RGB.f32 DUMP_DIR [1|5] [--portrait]\n", argv[0]);
        return 2;
    }
    const bool portrait=argc==6 && std::string(argv[5])=="--portrait";
    if (argc==6 && !portrait) return 2;
    sd_set_log_callback([](sd_log_level_t, const char *text, void *) { std::fprintf(stderr, "%s", text); }, nullptr);
    int frames = argc >= 5 && std::string(argv[4]) == "5" ? 5 : 1;
    if (argc >= 5 && std::string(argv[4]) != "1" && std::string(argv[4]) != "5") return 2;
    sd::Tensor<float> pixels({portrait ? 480 : 128, portrait ? 848 : 128, frames, 3, 1});
    std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(pixels.numel() * sizeof(float))) return 2;
    file.seekg(0);
    file.read(reinterpret_cast<char *>(pixels.data()), pixels.numel() * sizeof(float));
    if (!file) return 2;
    for (float v : pixels.values()) if (!std::isfinite(v) || v < 0 || v > 1) return 2;
    if (setenv("HV15_DUMP_DIR", argv[3], 1)) return 2;
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 1;
    int result = 1;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(argv[1], "first_stage_model.", VERSION_HUNYUAN_VIDEO)) return 1;
        auto storage = loader.get_tensor_storage_map();
        auto manager = std::make_shared<ModelManager>();
        if (!manager->set_loader(std::move(loader))) return 1;
        Hunyuan::HunyuanVideoVAERunner vae(backend, storage, "first_stage_model", false,
                                         VERSION_HUNYUAN_VIDEO, manager);
        std::map<std::string, ggml_tensor *> tensors;
        vae.get_param_tensors(tensors);
        if (manager->register_param_tensors(ModelComponent::VAE, std::move(tensors),
                ModelManager::ResidencyMode::ParamBackend, backend, backend)) {
            sd_tiling_params_t tiling = {};
            if (portrait) {
                tiling.enabled=true;
                tiling.tile_size_w=tiling.tile_size_h=128;
                tiling.target_overlap=0.25f;
            }
            auto encoded = vae.encode(8, pixels, tiling);
            if (!encoded.empty()) {
                hv15_dump("vae_encoded", vae.vae_to_diffusion_latents(encoded));
                auto decoded = vae.decode(8, encoded, tiling, true);
                if (!decoded.empty()) {
                    hv15_dump("vae_decoded", decoded);
                    result = 0;
                }
            }
        }
        manager->unregister_param_tensors(ModelComponent::VAE);
    }
    ggml_backend_free(backend);
    return result;
}
