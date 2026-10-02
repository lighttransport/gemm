/* Decode saved 480p diffusion latents, preserving the full temporal context. */
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include "stable-diffusion.h"
#include "model/vae/hunyuan_vae.hpp"
#include "core/hv15_dump.hpp"

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s VAE.safetensors LATENT_480x848.f32 NEW_OUT_DIR [1|5|81]\n", argv[0]);
        return 2;
    }
    int frames = 81;
    if (argc == 5) {
        const std::string count(argv[4]);
        if (count != "1" && count != "5" && count != "81") return 2;
        frames = std::stoi(count);
    }
    const int latent_frames = (frames-1)/4+1;
    sd_set_log_callback([](sd_log_level_t, const char *text, void *) { std::fprintf(stderr, "%s", text); }, nullptr);
    sd::Tensor<float> latent({30, 53, latent_frames, 32, 1});
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(latent.numel()*sizeof(float))) return 2;
    input.seekg(0);
    input.read(reinterpret_cast<char *>(latent.data()), latent.numel()*sizeof(float));
    if (!input) return 2;
    for (float value : latent.values()) if (!std::isfinite(value)) return 2;
    if (setenv("HV15_DUMP_DIR", argv[3], 1)) return 2;
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!backend || !cpu) return 1;
    int result = 1;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(argv[1], "first_stage_model.", VERSION_HUNYUAN_VIDEO)) return 1;
        auto storage = loader.get_tensor_storage_map();
        auto manager = std::make_shared<ModelManager>();
        if (!manager->set_loader(std::move(loader))) return 1;
        manager->set_enable_mmap(true);
        manager->prepare_file_io();
        Hunyuan::HunyuanVideoVAERunner vae(backend, storage, "first_stage_model", true,
                                         VERSION_HUNYUAN_VIDEO, manager);
        vae.set_max_graph_vram_bytes(13ULL*1024*1024*1024);
        std::map<std::string, ggml_tensor *> tensors;
        vae.get_param_tensors(tensors);
        if (manager->register_param_tensors(ModelComponent::VAE, std::move(tensors),
                ModelManager::ResidencyMode::ParamBackend, backend, cpu)) {
            sd_tiling_params_t tiling = {};
            tiling.enabled = true;
            tiling.tile_size_w = 128;
            tiling.tile_size_h = 128;
            tiling.target_overlap = 0.25f;
            auto pixels = vae.decode(8, vae.diffusion_to_vae_latents(latent), tiling, true);
            if (!pixels.empty() && pixels.shape() == std::vector<int64_t>({480,848,frames,3,1})) {
                for (float value : pixels.values()) if (!std::isfinite(value) || value<0.f || value>1.f) return 1;
                hv15_dump("vae_decoded", pixels);
                const size_t area = 480*848, plane = area*frames;
                std::vector<unsigned char> frame(area*3);
                result = 0;
                for (int t=0; t<frames; ++t) {
                    for (size_t p=0; p<area; ++p) for (int c=0; c<3; ++c)
                        frame[p*3+c] = static_cast<unsigned char>(pixels.data()[p+t*area+c*plane]*255.f);
                    std::string path = std::string(argv[3])+"/frame_";
                    char suffix[32]; std::snprintf(suffix,sizeof(suffix),"%05d.ppm",t); path+=suffix;
                    std::ofstream file(path,std::ios::binary);
                    file << "P6\n480 848\n255\n";
                    file.write(reinterpret_cast<const char *>(frame.data()),frame.size());
                    if (!file) { result=1; break; }
                }
            }
            vae.runner_end();
        }
        manager->unregister_param_tensors(ModelComponent::VAE);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return result;
}
