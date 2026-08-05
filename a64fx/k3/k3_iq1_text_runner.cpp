/* Minimal distributed IQ1 text smoke runner.
 *
 * The K3-native runner consumes the older safetensor layout.  This wrapper
 * uses the same A64FX CPU backend and the GGUF Kimi-K3 graph in llama.cpp,
 * registering one RPC CPU backend per node so the complete IQ1 model is
 * distributed across the allocation.  It is intentionally greedy and
 * writes plain decoded text for batch verification.
 */
#include "llama.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * p) {
    std::fprintf(stderr, "usage: %s MODEL RPC_LIST OUTPUT PROMPT [TOKENS]\n", p);
}

static bool register_rpc(const std::string & list) {
    ggml_backend_reg_t rpc = ggml_backend_reg_by_name("RPC");
    if (!rpc) {
        std::fprintf(stderr, "K3_IQ1_RPC_ERROR backend RPC is unavailable\n");
        return false;
    }
    using add_fn = ggml_backend_reg_t (*)(const char *);
    add_fn add = reinterpret_cast<add_fn>(
        ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_add_server"));
    if (!add) {
        std::fprintf(stderr, "K3_IQ1_RPC_ERROR add-server entry point is unavailable\n");
        return false;
    }
    size_t begin = 0;
    int count = 0;
    while (begin < list.size()) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) end = list.size();
        std::string endpoint = list.substr(begin, end - begin);
        if (endpoint.empty()) return false;
        ggml_backend_reg_t reg = add(endpoint.c_str());
        if (!reg) {
            std::fprintf(stderr, "K3_IQ1_RPC_ERROR endpoint=%s\n", endpoint.c_str());
            return false;
        }
        ggml_backend_register(reg);
        ++count;
        begin = end + 1;
    }
    std::fprintf(stderr, "K3_IQ1_RPC endpoints=%d\n", count);
    return count > 0;
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const std::string rpc_list = argv[2];
    const char * output_path = argv[3];
    const char * prompt = argv[4];
    int n_predict = argc == 6 ? std::atoi(argv[5]) : 64;
    if (n_predict < 1 || n_predict > 512) return 2;

    ggml_backend_load_all();
    if (!register_rpc(rpc_list)) return 3;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "K3_IQ1_LOAD_ERROR model=%s\n", model_path);
        return 4;
    }
    if (std::getenv("K3_IQ1_VOCAB_ONLY")) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::fprintf(stderr, "K3_IQ1_VOCAB_PASS vocab=%d\n",
                     llama_vocab_n_tokens(vocab));
        llama_model_free(model);
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_prompt = -llama_tokenize(vocab, prompt, std::strlen(prompt),
                                   nullptr, 0, true, true);
    if (n_prompt <= 0) {
        llama_model_free(model);
        return 5;
    }
    std::vector<llama_token> prompt_tokens((size_t)n_prompt);
    if (llama_tokenize(vocab, prompt, std::strlen(prompt), prompt_tokens.data(),
                       prompt_tokens.size(), true, true) < 0) {
        llama_model_free(model);
        return 5;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n_prompt + n_predict + 8);
    cp.n_batch = (uint32_t)n_prompt;
    cp.n_ubatch = (uint32_t)n_prompt;
    cp.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 6;
    }
    llama_sampler * sampler = llama_sampler_init_greedy();
    if (!sampler) {
        llama_free(ctx);
        llama_model_free(model);
        return 6;
    }
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    FILE * out = std::fopen(output_path, "w");
    if (!out) return 7;
    std::fprintf(out, "K3_IQ1_TEXT_V1 prompt_tokens=%d requested=%d\n", n_prompt, n_predict);
    std::fprintf(out, "prompt: %s\nresponse: ", prompt);
    if (llama_decode(ctx, batch)) return 8;
    int generated = 0;
    for (; generated < n_predict; ++generated) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        char piece[4096];
        int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (n < 0) return 9;
        std::fwrite(piece, 1, (size_t)n, out);
        std::fflush(out);
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch)) return 8;
    }
    std::fprintf(out, "\nstatus=PASS generated_tokens=%d\n", generated);
    std::fclose(out);
    std::fprintf(stderr, "K3_IQ1_TEXT_PASS prompt_tokens=%d generated_tokens=%d output=%s\n",
                 n_prompt, generated, output_path);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
