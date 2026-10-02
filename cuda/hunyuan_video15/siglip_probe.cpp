#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include "conditioning/conditioner.hpp"
#include "core/hv15_conditioning_inputs.hpp"
#include "core/hv15_dump.hpp"
#include "ggml-backend.h"
#include "model_manager.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s VISION.safetensors PIXELS.f32 OUTPUT.f32\n", argv[0]);
        return 2;
    }
    sd_set_log_callback([](sd_log_level_t, const char *text, void *) { std::fprintf(stderr, "%s", text); }, nullptr);
    if (!hv15_load_siglip_pixels(argv[2])) return 2;
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) { std::fprintf(stderr, "CUDA0 unavailable\n"); return 1; }
    int result = 1;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(argv[1], "clip_vision.", VERSION_HUNYUAN_VIDEO)) return 1;
        auto manager = std::make_shared<ModelManager>();
        auto storage = loader.get_tensor_storage_map();
        if (!manager->set_loader(std::move(loader))) return 1;
        FrozenCLIPVisionEmbedder vision(backend, storage, manager);
        if (!vision.siglip) { std::fprintf(stderr, "not a SigLIP vision checkpoint\n"); return 1; }
        std::map<std::string, ggml_tensor *> tensors;
        vision.get_param_tensors(tensors);
        if (manager->register_param_tensors(ModelComponent::CLIPVision, std::move(tensors),
                                           ModelManager::ResidencyMode::ParamBackend, backend, backend)) {
            auto graph = [&]() {
                auto gf = vision.build_graph(hv15_siglip_pixels, false, -1);
                for (int i=0; i<ggml_graph_n_nodes(gf); ++i) {
                    auto node = ggml_graph_node(gf,i);
                    if (node->op == GGML_OP_IM2COL)
                        std::fprintf(stderr,"SIGLIP_PATCH_ACTIVATIONS %s\n",ggml_type_name(node->type));
                }
                return gf;
            };
            HV15CudaF32Scope precision(backend,true);
            auto computed = vision.GGMLRunner::compute(graph,8,true);
            if (computed && !computed->empty()) {
                auto& output = *computed;
                size_t count = 1;
                for (auto dim : output.shape()) count *= static_cast<size_t>(dim);
                std::ofstream file(argv[3], std::ios::binary);
                file.write(reinterpret_cast<const char *>(output.data()), count * sizeof(float));
                if (file) {
                    std::printf("SigLIP output elements=%zu\n", count);
                    result = count == 729 * 1152 ? 0 : 1;
                }
            }
        }
        manager->unregister_param_tensors(ModelComponent::CLIPVision);
        if (result) std::fprintf(stderr, "probe load/compute/output failed\n");
    }
    ggml_backend_free(backend);
    return result;
}
