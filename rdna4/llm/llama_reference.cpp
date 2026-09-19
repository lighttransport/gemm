/* Reproducible HIP oracle; link only the build made by build_llama_reference.sh. */
#include "llama.h"
#include "ggml-backend.h"
#include "generation_trace.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <cctype>
#include <filesystem>

struct tensor_trace {
    const char *directory = nullptr;
    int layer = 0;
    int sequence = 0;
    bool failed = false;
};

static bool dump_tensor(ggml_tensor *t, bool ask, void *opaque) {
    auto &trace = *static_cast<tensor_trace *>(opaque);
    const char *raw_name = ggml_get_name(t);
    std::string name = raw_name ? raw_name : "";
    const std::string suffix = "-" + std::to_string(trace.layer);
    bool selected = name == "model.input_embed" ||
        (name.size() >= suffix.size() && name.compare(name.size()-suffix.size(), suffix.size(), suffix) == 0);
    if (ask) return selected;
    if (!selected || !t->data || !ggml_nbytes(t)) return true;
    for (char &c : name) if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-') c = '_';
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%04d-%s.bin", trace.directory, trace.sequence++, name.c_str());
    if (n < 0 || n >= (int)sizeof(path)) { trace.failed = true; return false; }
    std::vector<unsigned char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    FILE *f = fopen(path, "wb");
    if (!f) { trace.failed = true; return false; }
    bool ok = fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    ok = fclose(f) == 0 && ok;
    FILE *meta = fopen((std::string(path) + ".json").c_str(), "w");
    if (!meta) { trace.failed = true; return false; }
    fprintf(meta, "{\"type\":%d,\"shape\":[%lld,%lld,%lld,%lld],\"strides\":[%zu,%zu,%zu,%zu]}\n",
        (int)t->type, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    ok = fclose(meta) == 0 && ok;
    trace.failed |= !ok;
    return ok;
}

static double milliseconds() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s MODEL PROMPT [--trace-prefix PATH] [--decode N] [--repeat N] [--temp T] [--seed S] [--ubatch N]\n", argv[0]);
        return 2;
    }
    const char *prefix = nullptr;
    int generate = 256, repeats = 1, chunk = 512, capacity = 8192, top_k = 20;
    float temp = 0, top_p = .95f, min_p = 0, repetition = 1, frequency = 0, presence = 0;
    int penalty_last_n = 64;
    uint32_t seed = 42;
    tensor_trace tensors;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 == argc) return 2;
        const char *value = argv[++i];
        if (arg == "--trace-prefix") prefix = value;
        else if (arg == "--dump-tensors") tensors.directory = value;
        else if (arg == "--dump-layer") tensors.layer = atoi(value);
        else if (arg == "--decode") generate = atoi(value);
        else if (arg == "--repeat") repeats = atoi(value);
        else if (arg == "--ubatch") chunk = atoi(value);
        else if (arg == "--ctx") capacity = atoi(value);
        else if (arg == "--temp") temp = strtof(value, nullptr);
        else if (arg == "--seed") seed = (uint32_t)strtoul(value, nullptr, 10);
        else if (arg == "--top-k") top_k = atoi(value);
        else if (arg == "--top-p") top_p = strtof(value, nullptr);
        else if (arg == "--min-p") min_p = strtof(value, nullptr);
        else if (arg == "--repeat-penalty") repetition = strtof(value, nullptr);
        else if (arg == "--frequency-penalty") frequency = strtof(value, nullptr);
        else if (arg == "--presence-penalty") presence = strtof(value, nullptr);
        else if (arg == "--penalty-last-n") penalty_last_n = atoi(value);
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); return 2; }
    }
    if (chunk < 1 || capacity < 1 || generate < 0 || repeats < 1) return 2;
    if (tensors.directory) {
        std::filesystem::create_directories(tensors.directory);
        if (!std::filesystem::is_empty(tensors.directory)) {
            fprintf(stderr, "Tensor capture directory must be empty: %s\n", tensors.directory);
            return 2;
        }
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) return 2;
    std::string prompt{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    llama_backend_init();
    auto *device = ggml_backend_dev_by_name("ROCm0");
    if (!device) { fprintf(stderr, "ROCm0 unavailable\n"); return 3; }
    ggml_backend_dev_t devices[] = {device, nullptr};
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.devices = devices;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 4;
    auto *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int count = -llama_tokenize(vocab, prompt.data(), (int)prompt.size(), nullptr, 0, true, true);
    if (count <= 0 || count > capacity) return 5;
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), (int)prompt.size(), tokens.data(), count, true, true) != count) return 5;
    if (prefix) {
        FILE *f = fopen((std::string(prefix) + ".prompt.tokens").c_str(), "wb");
        if (!f) return 6;
        for (int token : tokens) fprintf(f, "%d\n", token);
        if (fclose(f)) return 6;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = capacity;
    cp.n_batch = chunk;
    cp.n_ubatch = chunk;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    if (tensors.directory) { cp.cb_eval = dump_tensor; cp.cb_eval_user_data = &tensors; }
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 7;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        llama_memory_clear(llama_get_memory(ctx), true);
        auto *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(n_vocab, penalty_last_n,
            repetition, frequency, presence));
        if (temp == 0) llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        else {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 0));
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 0));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
        }
        for (int token : tokens) llama_sampler_accept(sampler, token);
        hllm_generation_trace trace{};
        if (hllm_trace_open(&trace, prefix, repeat)) return 8;
        double begin = milliseconds();
        for (int offset = 0; offset < count; offset += chunk) {
            auto batch = llama_batch_get_one(tokens.data() + offset, std::min(chunk, count - offset));
            if (llama_decode(ctx, batch) || tensors.failed) return 9;
        }
        llama_synchronize(ctx);
        double prefill_ms = milliseconds() - begin;
        fprintf(stderr, "PREFILL repeat=%d tokens=%d chunk=%d ms=%.6f tok_s=%.6f\n", repeat, count, chunk, prefill_ms, 1000 * count / prefill_ms);
        std::vector<llama_token_data> candidates(n_vocab);
        int selected = 0, emitted = 0;
        const char *finish = "length";
        printf("=== Generated text ===\n");
        begin = milliseconds();
        for (int step = 0; step < generate && count + step < capacity; ++step) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) return 10;
            for (int i = 0; i < n_vocab; ++i) candidates[i] = {i, logits[i], 0};
            llama_token_data_array a{candidates.data(), candidates.size(), -1, false};
            llama_sampler_apply(sampler, &a);
            if (a.selected < 0) return 11;
            llama_token token = a.data[a.selected].id;
            if (hllm_trace_token(&trace, token, logits, n_vocab)) return 12;
            ++selected;
            if (llama_vocab_is_eog(vocab, token)) { finish = "eos"; break; }
            std::vector<char> piece(256);
            int n = llama_token_to_piece(vocab, token, piece.data(), (int)piece.size(), 0, false);
            if (n < 0) {
                piece.resize(-n);
                n = llama_token_to_piece(vocab, token, piece.data(), (int)piece.size(), 0, false);
            }
            if (n < 0) return 13;
            fwrite(piece.data(), 1, n, stdout);
            if (trace.bytes && fwrite(piece.data(), 1, n, trace.bytes) != (size_t)n) return 14;
            ++emitted;
            llama_sampler_accept(sampler, token);
            if (step + 1 == generate || count + step + 1 == capacity) break;
            auto batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, batch) || tensors.failed) return 15;
        }
        double decode_ms = milliseconds() - begin;
        printf("\n=== end ===\n");
        if (fflush(stdout) || hllm_trace_close(&trace)) return 16;
        fprintf(stderr, "GENERATION finish=%s selected=%d emitted=%d synthetic=0\n", finish, selected, emitted);
        fprintf(stderr, "DECODE repeat=%d ms=%.6f tok_s=%.6f\n", repeat, decode_ms, emitted * 1000 / decode_ms);
        llama_sampler_free(sampler);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "Result: PASS\n");
}
