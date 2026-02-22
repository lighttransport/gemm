// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Standalone test: GPU LLM inference with Vulkan compute shaders.
// Loads a GGUF model, generates text, optionally compares with CPU reference.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>

extern "C" {
#include "../common/gguf_loader.h"
#include "../common/ggml_dequant.h"
#include "../common/bpe_tokenizer.h"
#include "../common/transformer.h"
}

#include "vulkan_llm_runner.hh"

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <model.gguf> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>      GGUF model file\n");
    fprintf(stderr, "  --prompt <text>     Input prompt (default: 'Hello')\n");
    fprintf(stderr, "  --max-gen <N>       Max tokens to generate (default: 50)\n");
    fprintf(stderr, "  --device <id>       Vulkan device (default: 0)\n");
    fprintf(stderr, "  --compare           Compare GPU vs CPU output per-layer\n");
    fprintf(stderr, "  -t, --threads <N>   CPU threads for comparison (default: 1)\n");
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "Hello";
    int max_gen = 50, device_id = 0, n_threads = 1;
    bool compare = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--max-gen") == 0 && i+1 < argc) max_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i+1 < argc) device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--compare") == 0) compare = true;
        else if ((strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) && i+1 < argc)
            n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model required\n");
        print_usage(argv[0]); return 1;
    }

    // Load GGUF
    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    gguf_context *gguf = gguf_open(model_path.c_str(), 1);
    if (!gguf) { fprintf(stderr, "Failed to open model\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) { fprintf(stderr, "Failed to load vocab\n"); return 1; }

    // GPU model
    VulkanLLMRunner gpu;
    if (!gpu.initialize(device_id, true)) {
        fprintf(stderr, "GPU init failed: %s\n", gpu.getLastError().c_str());
        return 1;
    }
    if (!gpu.loadWeights(gguf)) {
        fprintf(stderr, "GPU weight load failed: %s\n", gpu.getLastError().c_str());
        return 1;
    }

    // CPU model (for comparison)
    transformer_model *cpu_model = nullptr;
    if (compare) {
        cpu_model = transformer_load(gguf, 1024);
        if (!cpu_model) { fprintf(stderr, "Failed to load CPU model\n"); return 1; }
        if (n_threads > 1) transformer_set_threads(cpu_model, n_threads);
    }

    // Tokenize prompt
    int32_t tokens[256];
    int n_tokens = bpe_tokenize(vocab, prompt.c_str(), -1, tokens, 256);
    fprintf(stderr, "Prompt: \"%s\" -> %d tokens\n", prompt.c_str(), n_tokens);
    fprintf(stderr, "Tokens:");
    for (int i = 0; i < n_tokens; i++) {
        const char *ts = bpe_token_to_str(vocab, tokens[i]);
        fprintf(stderr, " %d(\"%s\")", tokens[i], ts ? ts : "?");
    }
    fprintf(stderr, "\n");

    // Prefill
    fprintf(stderr, "\n=== Prefill ===\n");
    clock_t t0 = clock();
    for (int i = 0; i < n_tokens; i++) {
        bool last = (i == n_tokens - 1);
        float *gpu_logits = gpu.forward(tokens[i], i, last);

        if (compare && cpu_model) {
            float *cpu_logits;
            if (last) {
                cpu_logits = transformer_forward_logits(cpu_model, tokens[i], i);
            } else {
                transformer_forward(cpu_model, tokens[i], i);
                cpu_logits = nullptr;
            }

            if (last && cpu_logits && gpu_logits) {
                // Compare logits
                float max_diff = 0.0f;
                double sum_diff = 0.0;
                for (int j = 0; j < gpu.n_vocab(); j++) {
                    float d = fabsf(gpu_logits[j] - cpu_logits[j]);
                    if (d > max_diff) max_diff = d;
                    sum_diff += d;
                }
                fprintf(stderr, "  Logits comparison: max_diff=%.6f avg_diff=%.6f\n",
                        max_diff, sum_diff / gpu.n_vocab());

                // Compare argmax
                int gpu_argmax = 0, cpu_argmax = 0;
                for (int j = 1; j < gpu.n_vocab(); j++) {
                    if (gpu_logits[j] > gpu_logits[gpu_argmax]) gpu_argmax = j;
                    if (cpu_logits[j] > cpu_logits[cpu_argmax]) cpu_argmax = j;
                }
                fprintf(stderr, "  GPU argmax=%d CPU argmax=%d %s\n",
                        gpu_argmax, cpu_argmax, gpu_argmax == cpu_argmax ? "MATCH" : "MISMATCH");
            }
        }
    }
    clock_t t1 = clock();
    fprintf(stderr, "Prefill: %.3f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    // Generate
    fprintf(stderr, "\n=== Generation ===\n\n");
    int pos = n_tokens;
    int32_t next_token = -1;
    clock_t gen_t0 = clock();
    int gen_count = 0;

    for (int g = 0; g < max_gen; g++) {
        float *logits;
        if (g == 0) {
            // Use logits from last prefill step
            logits = gpu.forward(tokens[n_tokens - 1], n_tokens - 1, true);
            // Actually we already computed this above; re-compute for simplicity
        } else {
            logits = gpu.forward(next_token, pos, true);
            pos++;
        }
        if (!logits) break;

        // Greedy argmax
        next_token = 0;
        for (int j = 1; j < gpu.n_vocab(); j++) {
            if (logits[j] > logits[next_token]) next_token = j;
        }

        if (next_token == bpe_eos_id(vocab) || next_token == bpe_eot_id(vocab)) {
            fprintf(stderr, "  [EOS at step %d]\n", g);
            break;
        }

        const char *tok_str = bpe_token_to_str(vocab, next_token);
        if (tok_str) {
            int dec_len;
            char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
            fwrite(decoded, 1, dec_len, stdout);
            fflush(stdout);
            free(decoded);
        }

        if (g == 0) pos++;
        gen_count++;
    }
    printf("\n");
    clock_t gen_t1 = clock();
    double gen_secs = (double)(gen_t1 - gen_t0) / CLOCKS_PER_SEC;
    if (gen_count > 0) {
        double tok_per_sec = gen_count / gen_secs;
        // Model size estimate: n_layers * (4*n_embd^2 + 3*n_ff*n_embd) * bytes_per_param
        // For F16: bytes_per_param = 2
        double params_per_layer = 4.0 * gpu.n_embd() * gpu.n_embd() +
                                   3.0 * 6144 * gpu.n_embd();  // approximate
        double total_bytes = gpu.n_layers() * params_per_layer * 2.0;
        double gflops = total_bytes * tok_per_sec / 1e9;
        fprintf(stderr, "Decode: %d tokens in %.3f s = %.1f tok/s (%.1f GB/s effective bandwidth)\n",
                gen_count, gen_secs, tok_per_sec, gflops);
    }

    // Cleanup
    if (cpu_model) transformer_free(cpu_model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
