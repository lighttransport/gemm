#include "llama-model-loader.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s FIRST_GGUF_SHARD\n", argv[0]);
        return 2;
    }

    try {
        std::vector<std::string> splits;
        llama_model_loader loader(
            nullptr, nullptr, nullptr, argv[1], splits, nullptr,
            LLAMA_LOAD_MODE_NONE, true, true, false, nullptr, nullptr);

        std::printf("LLAMA_GLM53F_LOADER arch=%s tensors=%d bytes=%zu files=%zu\n",
                    loader.get_arch_name().c_str(), loader.n_tensors,
                    loader.n_bytes, loader.files.size());
        if (loader.get_arch_name() != "glm5next" || loader.n_tensors != 1412 ||
            loader.n_bytes != 108710550904ull || loader.files.size() != 4) {
            std::fprintf(stderr, "LLAMA_GLM53F_LOADER FAIL unexpected metadata\n");
            return 1;
        }
        std::printf("LLAMA_GLM53F_LOADER PASS split_index_and_bounds=ok\n");
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "LLAMA_GLM53F_LOADER FAIL exception=%s\n", ex.what());
        return 1;
    }
}
