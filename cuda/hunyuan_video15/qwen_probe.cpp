/* Bounded text-conditioning probe for HunyuanVideo's selected Qwen layer. */
#include <cstdio>
#include <cstring>
#include <memory>
#include "conditioning/conditioner.hpp"
#include "model_manager.h"
#include "core/hv15_dump.hpp"

int main(int argc, char **argv) {
    if (argc != 5 || std::strlen(argv[4]) > 256) {
        std::fprintf(stderr, "usage: %s QWEN.safetensors TOKENIZER.json DUMP_DIR PROMPT(max256bytes)\n", argv[0]);
        return 2;
    }
    sd_set_log_callback([](sd_log_level_t, const char *text, void *) { std::fprintf(stderr, "%s", text); }, nullptr);
    if (setenv("HV15_DUMP_DIR", argv[3], 1)) return 2;
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!backend || !cpu) return 1;
    int result = 1;
    {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(argv[1], "text_encoders.llm.", VERSION_HUNYUAN_VIDEO)) return 1;
        auto storage = loader.get_tensor_storage_map();
        auto manager = std::make_shared<ModelManager>();
        if (!manager->set_loader(std::move(loader))) return 1;
        manager->set_enable_mmap(true);
        manager->prepare_file_io();
        LLMEmbedder encoder(backend, storage, VERSION_HUNYUAN_VIDEO, "", false,
                            manager, TokenizerConfig(argv[2]));
        encoder.set_max_graph_vram_bytes(13ULL * 1024 * 1024 * 1024);
        encoder.set_flash_attention_enabled(true);
        std::map<std::string, ggml_tensor *> tensors;
        encoder.get_param_tensors(tensors);
        if (manager->register_param_tensors(ModelComponent::Conditioner, std::move(tensors),
                ModelManager::ResidencyMode::ParamBackend, backend, cpu)) {
            ConditionerParams params;
            params.text = argv[4];
            params.clip_skip = 2;
            auto condition = encoder.get_learned_condition(8, params);
            if (!condition.c_crossattn.empty()) {
                hv15_dump("qwen_hidden", condition.c_crossattn);
                result = 0;
            }
            encoder.runner_end();
        }
        manager->unregister_param_tensors(ModelComponent::Conditioner);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return result;
}
