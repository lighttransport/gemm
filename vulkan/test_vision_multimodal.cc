// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Multimodal inference: Full GPU pipeline (Vulkan vision encoder + Vulkan LLM)
// Loads an image, encodes with GPU, generates text description with GPU.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>

#define PROFILER_IMPLEMENTATION
extern "C" {
#include "../common/profiler.h"
#include "../common/gguf_loader.h"
#include "../common/ggml_dequant.h"
#include "../common/bpe_tokenizer.h"
#include "../common/transformer.h"
#include "../common/vision_encoder.h"
}

#include "vulkan_vision_encoder.hh"
#include "vulkan_llm_runner.hh"
#include "deps/stb_image.h"
#include "deps/stb_image_resize2.h"

static const char *gguf_get_kv_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return nullptr;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return nullptr;
    return g->kv[idx].value.str.str;
}

static void build_multimodal_chat_prompt(const gguf_context *gguf_main,
                                          const std::string &user_prompt,
                                          std::string &text_before,
                                          std::string &text_after) {
    const char *tmpl = gguf_get_kv_string(gguf_main, "tokenizer.chat_template");
    const bool has_chatml = tmpl &&
                            strstr(tmpl, "<|im_start|>") &&
                            strstr(tmpl, "<|im_end|>");
    const bool has_vision = tmpl &&
                            strstr(tmpl, "<|vision_start|>") &&
                            strstr(tmpl, "<|vision_end|>");

    if (has_chatml && has_vision) {
        text_before = "<|im_start|>user\n<|vision_start|>";
        text_after = "<|vision_end|>" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
        fprintf(stderr, "Chat template: tokenizer.chat_template detected (chatml+vision)\n");
        return;
    }

    text_before = "<|im_start|>user\n<|vision_start|>";
    text_after = "<|vision_end|>" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (tmpl) {
        fprintf(stderr, "Chat template: unsupported template format, using chatml fallback\n");
    } else {
        fprintf(stderr, "Chat template: missing tokenizer.chat_template, using chatml fallback\n");
    }
}

// Generate a simple test pattern
static uint8_t *generate_test_image(int width, int height, const char *pattern) {
    uint8_t *img = (uint8_t *)malloc(width * height * 3);

    if (strcmp(pattern, "checkerboard") == 0) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int checker = ((x / 32) + (y / 32)) & 1;
                uint8_t val = checker ? 240 : 15;
                int idx = (y * width + x) * 3;
                img[idx] = val; img[idx+1] = val; img[idx+2] = val;
            }
        }
    } else if (strcmp(pattern, "gradient") == 0) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                img[idx]   = (uint8_t)(255 * x / width);        // R gradient left->right
                img[idx+1] = (uint8_t)(255 * y / height);       // G gradient top->bottom
                img[idx+2] = (uint8_t)(255 * (width-x) / width); // B gradient right->left
            }
        }
    } else if (strcmp(pattern, "red") == 0) {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                img[idx] = 220; img[idx+1] = 30; img[idx+2] = 30;
            }
    } else if (strcmp(pattern, "circles") == 0) {
        memset(img, 240, width * height * 3); // white bg
        int cx = width/2, cy = height/2;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                int dx = x - cx, dy = y - cy;
                int r2 = dx*dx + dy*dy;
                int idx = (y * width + x) * 3;
                if (r2 < (width/6)*(width/6)) {
                    img[idx] = 30; img[idx+1] = 30; img[idx+2] = 200; // blue circle
                } else if (r2 < (width/3)*(width/3)) {
                    img[idx] = 30; img[idx+1] = 180; img[idx+2] = 30; // green ring
                }
            }
    } else {
        // default: solid gray
        memset(img, 128, width * height * 3);
    }
    return img;
}

// Smart resize matching llama.cpp's Qwen-VL preprocessing:
// - Preserves aspect ratio
// - Aligns dimensions to multiples of align_size (patch_size * spatial_merge)
// - Constrains total pixels between min_pixels and max_pixels
static void calc_size_preserved_ratio(int in_w, int in_h, int align_size,
                                       int min_pixels, int max_pixels,
                                       int *out_w, int *out_h) {
    auto round_by = [align_size](float x) { return (int)roundf(x / align_size) * align_size; };
    auto ceil_by  = [align_size](float x) { return (int)ceilf(x / align_size) * align_size; };
    auto floor_by = [align_size](float x) { return (int)floorf(x / align_size) * align_size; };

    int h_bar = std::max(align_size, round_by((float)in_h));
    int w_bar = std::max(align_size, round_by((float)in_w));

    if (h_bar * w_bar > max_pixels) {
        float beta = sqrtf((float)(in_h * in_w) / max_pixels);
        h_bar = std::max(align_size, floor_by(in_h / beta));
        w_bar = std::max(align_size, floor_by(in_w / beta));
    } else if (h_bar * w_bar < min_pixels) {
        float beta = sqrtf((float)min_pixels / (in_h * in_w));
        h_bar = ceil_by(in_h * beta);
        w_bar = ceil_by(in_w * beta);
    }
    *out_w = w_bar;
    *out_h = h_bar;
}

// Bilinear resize matching llama.cpp's implementation exactly
static uint8_t *resize_bilinear(const uint8_t *src, int src_w, int src_h,
                                 int dst_w, int dst_h) {
    uint8_t *dst = (uint8_t *)malloc(dst_w * dst_h * 3);
    float x_ratio = (float)(src_w - 1) / dst_w;
    float y_ratio = (float)(src_h - 1) / dst_h;

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = (int)px;
            int y_floor = (int)py;
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = (1.0f - x_lerp) * src[3 * (y_floor * src_w + x_floor) + c]
                          +         x_lerp  * src[3 * (y_floor * src_w + (x_floor + 1)) + c];
                float bot = (1.0f - x_lerp) * src[3 * ((y_floor + 1) * src_w + x_floor) + c]
                          +         x_lerp  * src[3 * ((y_floor + 1) * src_w + (x_floor + 1)) + c];
                dst[3 * (y * dst_w + x) + c] = (uint8_t)((1.0f - y_lerp) * top + y_lerp * bot);
            }
        }
    }
    return dst;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <model.gguf> --mmproj <mmproj.gguf> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>      LLM model GGUF\n");
    fprintf(stderr, "  --mmproj <path>     Vision mmproj GGUF\n");
    fprintf(stderr, "  --image <path>      Input image (JPG, PNG, BMP, TGA, etc.)\n");
    fprintf(stderr, "  --pattern <name>    Test pattern: checkerboard, gradient, red, circles (default: gradient)\n");
    fprintf(stderr, "  --prompt <text>     User prompt (default: 'Describe what you see in this image in detail.')\n");
    fprintf(stderr, "  --image-size <N>    Max image size (0=auto, default: 0). Caps smart resize.\n");
    fprintf(stderr, "  --max-gen <N>       Max tokens to generate (default: 100)\n");
    fprintf(stderr, "  --attn <mode>       Attention: cpu, naive, flash (default: flash)\n");
    fprintf(stderr, "  --device <id>       Vulkan device (default: 0)\n");
    fprintf(stderr, "  --gpu-llm           Use Vulkan GPU for LLM (default: CPU)\n");
    fprintf(stderr, "  -t, --threads <N>   CPU threads for LLM (default: 1)\n");
}

int main(int argc, char **argv) {
    std::string model_path, mmproj_path, image_path, pattern = "gradient";
    std::string prompt = "Describe what you see in this image in detail.";
    int n_phys_cores = (int)std::thread::hardware_concurrency() / 2;
    if (n_phys_cores < 1) n_phys_cores = 1;
    int image_size = 0, max_gen = 100, device_id = 0, n_threads = n_phys_cores;  // 0 = auto (smart resize)
    bool gpu_llm = false, cpu_vision = false;
    VulkanVisionEncoder::AttentionMode attn_mode = VulkanVisionEncoder::ATTN_FLASH_GPU;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--mmproj") == 0 && i+1 < argc) mmproj_path = argv[++i];
        else if (strcmp(argv[i], "--image") == 0 && i+1 < argc) image_path = argv[++i];
        else if (strcmp(argv[i], "--pattern") == 0 && i+1 < argc) pattern = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--image-size") == 0 && i+1 < argc) image_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-gen") == 0 && i+1 < argc) max_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i+1 < argc) device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-llm") == 0) gpu_llm = true;
        else if (strcmp(argv[i], "--cpu-vision") == 0) cpu_vision = true;
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--attn") == 0 && i+1 < argc) {
            i++;
            if (strcmp(argv[i], "cpu") == 0) attn_mode = VulkanVisionEncoder::ATTN_CPU;
            else if (strcmp(argv[i], "naive") == 0) attn_mode = VulkanVisionEncoder::ATTN_NAIVE_GPU;
            else attn_mode = VulkanVisionEncoder::ATTN_FLASH_GPU;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || mmproj_path.empty()) {
        fprintf(stderr, "Error: --model and --mmproj required\n");
        print_usage(argv[0]); return 1;
    }

    // Load main model
    fprintf(stderr, "Loading LLM: %s\n", model_path.c_str());
    gguf_context *gguf_main = gguf_open(model_path.c_str(), 1);
    if (!gguf_main) { fprintf(stderr, "Failed to open model\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_main);
    if (!vocab) { fprintf(stderr, "Failed to load vocab\n"); return 1; }
    fprintf(stderr, "Vocab loaded\n");

    // CPU LLM (always loaded for fallback and n_vocab)
    int max_seq_len = 1024;
    transformer_model *model = transformer_load(gguf_main, max_seq_len);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    if (n_threads > 1) transformer_set_threads(model, n_threads);

    // GPU LLM (optional)
    VulkanLLMRunner *gpu_llm_runner = nullptr;
    if (gpu_llm) {
        fprintf(stderr, "\n=== GPU LLM Setup ===\n");
        gpu_llm_runner = new VulkanLLMRunner();
        if (!gpu_llm_runner->initialize(device_id, true)) {
            fprintf(stderr, "GPU LLM init failed: %s\n", gpu_llm_runner->getLastError().c_str());
            return 1;
        }
        if (!gpu_llm_runner->loadWeights(gguf_main)) {
            fprintf(stderr, "GPU LLM weight load failed: %s\n", gpu_llm_runner->getLastError().c_str());
            return 1;
        }
    }

    // Load vision encoder (for normalization params)
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path.c_str());
    gguf_context *gguf_mmproj = gguf_open(mmproj_path.c_str(), 1);
    if (!gguf_mmproj) { fprintf(stderr, "Failed to open mmproj\n"); return 1; }

    vision_model *vm = vision_load(gguf_mmproj);
    if (!vm) { fprintf(stderr, "Failed to load vision model\n"); return 1; }

    // Load or generate image
    uint8_t *img_rgb = NULL;
    int img_w = image_size, img_h = image_size;

    if (!image_path.empty()) {
        int channels;
        img_rgb = stbi_load(image_path.c_str(), &img_w, &img_h, &channels, 3);
        if (!img_rgb) {
            fprintf(stderr, "Failed to load image: %s (%s)\n",
                    image_path.c_str(), stbi_failure_reason());
            return 1;
        }
        fprintf(stderr, "Loaded image: %dx%d (%d channels)\n", img_w, img_h, channels);

        // Smart resize: preserve aspect ratio, align to patch_size * spatial_merge
        // Matches llama.cpp Qwen-VL preprocessing exactly
        {
            int align_size = vm->patch_size * vm->spatial_merge;
            // min/max tokens: 8..4096, each token covers align_size^2 pixels
            int min_pixels = 8 * align_size * align_size;     // 8192
            int max_pixels = 4096 * align_size * align_size;  // 4194304
            // If user specified --image-size, cap max_pixels accordingly
            if (image_size > 0) {
                int user_max_pixels = image_size * image_size;
                if (user_max_pixels < max_pixels) max_pixels = user_max_pixels;
            }

            int target_w, target_h;
            calc_size_preserved_ratio(img_w, img_h, align_size, min_pixels, max_pixels, &target_w, &target_h);

            if (img_w != target_w || img_h != target_h) {
                fprintf(stderr, "Smart resize %dx%d -> %dx%d (aspect-preserving, align=%d)\n",
                        img_w, img_h, target_w, target_h, align_size);
                uint8_t *resized = resize_bilinear(img_rgb, img_w, img_h, target_w, target_h);
                stbi_image_free(img_rgb);
                img_rgb = resized;
                img_w = target_w;
                img_h = target_h;
            }
        }
    } else {
        if (image_size <= 0) image_size = 192;  // default for test patterns
        img_w = img_h = image_size;
        fprintf(stderr, "Generating '%s' test image (%dx%d)...\n", pattern.c_str(), img_w, img_h);
        img_rgb = generate_test_image(img_w, img_h, pattern.c_str());
    }

    // Normalize
    fprintf(stderr, "Normalizing image...\n");
    float *img_norm = vision_normalize_image(vm, img_rgb, img_w, img_h);
    free(img_rgb);

    // GPU vision encode
    fprintf(stderr, "\n=== GPU Vision Encoding ===\n");
    VulkanVisionEncoder gpu_enc;
    if (!gpu_enc.initialize(device_id, true)) {
        fprintf(stderr, "GPU init failed: %s\n", gpu_enc.getLastError().c_str());
        return 1;
    }
    gpu_enc.setAttentionMode(attn_mode);

    if (!gpu_enc.loadWeights(gguf_mmproj)) {
        fprintf(stderr, "GPU weight load failed: %s\n", gpu_enc.getLastError().c_str());
        return 1;
    }

    clock_t t0 = clock();
    float *vision_embd = gpu_enc.encode(img_norm, img_w, img_h);
    clock_t t1 = clock();
    free(img_norm);

    if (!vision_embd) {
        fprintf(stderr, "Vision encoding failed: %s\n", gpu_enc.getLastError().c_str());
        return 1;
    }

    int patches_w = img_w / vm->patch_size;
    int patches_h = img_h / vm->patch_size;
    int n_vision_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
    int proj_dim = vm->proj_dim;
    int n_ds = vm->n_deepstack;
    int embd_stride = proj_dim * (1 + n_ds);  // total dim per token (main + deepstack)
    fprintf(stderr, "Vision encoding: %d tokens x %d dim (%d main + %d deepstack) (%.3f s)\n",
            n_vision_tokens, embd_stride, proj_dim, n_ds, (double)(t1 - t0) / CLOCKS_PER_SEC);

    // Debug: dump first vision token embedding values
    fprintf(stderr, "  GPU vision embd[0][0..7]:");
    for (int j = 0; j < 8 && j < proj_dim; j++)
        fprintf(stderr, " %.4f", vision_embd[j]);
    fprintf(stderr, "\n");
    // Compute norm of first token
    {
        float norm = 0;
        for (int j = 0; j < proj_dim; j++) norm += vision_embd[j] * vision_embd[j];
        fprintf(stderr, "  GPU vision embd[0] norm=%.4f\n", sqrtf(norm));
    }

    // CPU vision encode for comparison (disabled for performance — enable with --cpu-vision)
    if (cpu_vision) {
        int channels;
        int cpu_w, cpu_h;
        uint8_t *img2 = stbi_load(image_path.c_str(), &cpu_w, &cpu_h, &channels, 3);
        if (img2 && (cpu_w != img_w || cpu_h != img_h)) {
            uint8_t *resized = resize_bilinear(img2, cpu_w, cpu_h, img_w, img_h);
            stbi_image_free(img2); img2 = resized;
        }
        if (img2) {
            float *cpu_norm = vision_normalize_image(vm, img2, img_w, img_h);
            free(img2);
            float *cpu_embd = vision_encode(vm, cpu_norm, img_w, img_h, n_threads);
            free(cpu_norm);
            if (cpu_embd) {
                fprintf(stderr, "  CPU vision embd[0][0..7]:");
                for (int j = 0; j < 8; j++) fprintf(stderr, " %.4f", cpu_embd[j]);
                fprintf(stderr, "\n");
                float norm = 0;
                for (int j = 0; j < proj_dim; j++) norm += cpu_embd[j] * cpu_embd[j];
                fprintf(stderr, "  CPU vision embd[0] norm=%.4f\n", sqrtf(norm));
                float max_diff = 0;
                for (int j = 0; j < n_vision_tokens * embd_stride; j++) {
                    float d = fabsf(cpu_embd[j] - vision_embd[j]);
                    if (d > max_diff) max_diff = d;
                }
                fprintf(stderr, "  GPU vs CPU max diff: %.6f\n", max_diff);
                free(cpu_embd);
            }
        }
    }

    // Build token sequence from GGUF chat template when supported.
    std::string text_before, text_after;
    build_multimodal_chat_prompt(gguf_main, prompt, text_before, text_after);

    int32_t tokens_before[64], tokens_after[128];
    int n_before = bpe_tokenize(vocab, text_before.c_str(), -1, tokens_before, 64);
    int n_after  = bpe_tokenize(vocab, text_after.c_str(), -1, tokens_after, 128);

    int total_prompt_len = n_before + n_vision_tokens + n_after;
    fprintf(stderr, "\nPrompt: %d tokens (%d text + %d vision + %d text)\n",
            total_prompt_len, n_before, n_vision_tokens, n_after);
    fprintf(stderr, "User prompt: \"%s\"\n", prompt.c_str());

    // Compute merged grid dimensions for M-RoPE vision positions
    int merged_w = patches_w / vm->spatial_merge;
    int merged_h = patches_h / vm->spatial_merge;

    // Prefill
    fprintf(stderr, "\n=== Prefill (%s LLM) ===\n", gpu_llm ? "GPU" : "CPU");
    // Keep KV cache index separate from M-RoPE logical position progression.
    int cache_pos = 0;
    int rope_pos = 0;
    int n_vocab_actual = gpu_llm ? gpu_llm_runner->n_vocab() : model->n_vocab;
    float *last_prefill_logits = nullptr;

    // Compute merged grid dimensions for M-RoPE vision positions
    // (already computed above but needed for batch building)

    if (!gpu_llm && model) {
        // === BATCHED CPU PREFILL ===
        struct timespec wall_t0, wall_t1;
        clock_gettime(CLOCK_MONOTONIC, &wall_t0);
        t0 = clock();
        int total_N = n_before + n_vision_tokens + n_after;
        fprintf(stderr, "  Batched prefill: %d tokens total\n", total_N);

        // Allocate embedding buffer [total_N, embd_stride_or_n_embd]
        int batch_embd_stride = (embd_stride > model->n_embd) ? embd_stride : model->n_embd;
        float *all_embds = new float[(size_t)total_N * batch_embd_stride]();
        int *all_cache_pos = new int[total_N];
        int *all_pos_t = new int[total_N];
        int *all_pos_h = new int[total_N];
        int *all_pos_w = new int[total_N];
        const float **all_ds_embds = new const float*[total_N]();

        int idx = 0;

        // Text before: dequant token embeddings
        for (int i = 0; i < n_before; i++) {
            {
                size_t rb = dequant_row_size(model->token_embd.type, model->n_embd);
                const void *rd = (const uint8_t*)model->token_embd.data + (size_t)tokens_before[i] * rb;
                dequant_row(model->token_embd.type, rd, all_embds + (size_t)idx * batch_embd_stride, model->n_embd);
            }
            all_cache_pos[idx] = cache_pos;
            all_pos_t[idx] = rope_pos;
            all_pos_h[idx] = rope_pos;
            all_pos_w[idx] = rope_pos;
            all_ds_embds[idx] = nullptr;
            cache_pos++; rope_pos++; idx++;
        }

        // Vision tokens with M-RoPE 3D positions
        model->ds_embd_stride = embd_stride;
        const int image_pos0 = rope_pos;
        for (int i = 0; i < n_vision_tokens; i++) {
            float *embd_i = vision_embd + (size_t)i * embd_stride;
            // Copy full embedding (including deepstack) to batch buffer
            memcpy(all_embds + (size_t)idx * batch_embd_stride, embd_i, model->n_embd * sizeof(float));
            all_cache_pos[idx] = cache_pos;
            int y = i / merged_w;
            int x = i % merged_w;
            all_pos_t[idx] = image_pos0;
            all_pos_h[idx] = image_pos0 + y;
            all_pos_w[idx] = image_pos0 + x;
            // Point deepstack to the vision embedding (which has stride embd_stride)
            all_ds_embds[idx] = (embd_stride > model->n_embd) ? embd_i : nullptr;
            cache_pos++; idx++;
        }
        rope_pos += std::max(merged_w, merged_h);

        // Text after
        for (int i = 0; i < n_after; i++) {
            {
                size_t rb = dequant_row_size(model->token_embd.type, model->n_embd);
                const void *rd = (const uint8_t*)model->token_embd.data + (size_t)tokens_after[i] * rb;
                dequant_row(model->token_embd.type, rd, all_embds + (size_t)idx * batch_embd_stride, model->n_embd);
            }
            all_cache_pos[idx] = cache_pos;
            all_pos_t[idx] = rope_pos;
            all_pos_h[idx] = rope_pos;
            all_pos_w[idx] = rope_pos;
            all_ds_embds[idx] = nullptr;
            cache_pos++; rope_pos++; idx++;
        }

        transformer_batch batch;
        batch.embds = all_embds;
        batch.N = total_N;
        batch.embd_stride = batch_embd_stride;
        batch.cache_pos = all_cache_pos;
        batch.pos_t = all_pos_t;
        batch.pos_h = all_pos_h;
        batch.pos_w = all_pos_w;
        batch.ds_embds = all_ds_embds;
        batch.ds_embd_stride = embd_stride;

        last_prefill_logits = transformer_forward_batch_logits(model, &batch);

        delete[] all_embds;
        delete[] all_cache_pos;
        delete[] all_pos_t;
        delete[] all_pos_h;
        delete[] all_pos_w;
        delete[] all_ds_embds;
        delete[] vision_embd;

        t1 = clock();
        clock_gettime(CLOCK_MONOTONIC, &wall_t1);
        double wall_s = (wall_t1.tv_sec - wall_t0.tv_sec) + (wall_t1.tv_nsec - wall_t0.tv_nsec) / 1e9;
        fprintf(stderr, "  Batched prefill done: %.1f s wall (%.1f s CPU)\n", wall_s, (double)(t1 - t0) / CLOCKS_PER_SEC);
    } else if (gpu_llm) {
        // === GPU PREFILL ===
        // Try batched coopmat GEMM first (requires Q8_0 weights), fall back to per-token
        bool batch_ok = false;
        {
            int total_N = n_before + n_vision_tokens + n_after;
            if (gpu_llm_runner->enablePrefill(total_N)) {
                struct timespec wall_t0, wall_t1;
                clock_gettime(CLOCK_MONOTONIC, &wall_t0);
                t0 = clock();

                fprintf(stderr, "  GPU batched prefill: %d tokens total\n", total_N);

                int n_embd_for_batch = gpu_llm_runner->n_embd();
                std::vector<float> all_embds((size_t)total_N * n_embd_for_batch, 0.0f);
                std::vector<int> all_cache_pos(total_N);
                std::vector<int> all_pos_t(total_N), all_pos_h(total_N), all_pos_w(total_N);

                int idx = 0;

                for (int i = 0; i < n_before; i++) {
                    size_t rb = dequant_row_size(model->token_embd.type, model->n_embd);
                    const void *rd = (const uint8_t*)model->token_embd.data + (size_t)tokens_before[i] * rb;
                    dequant_row(model->token_embd.type, rd, all_embds.data() + (size_t)idx * n_embd_for_batch, model->n_embd);
                    all_cache_pos[idx] = cache_pos;
                    all_pos_t[idx] = rope_pos;
                    all_pos_h[idx] = rope_pos;
                    all_pos_w[idx] = rope_pos;
                    cache_pos++; rope_pos++; idx++;
                }

                const int image_pos0 = rope_pos;
                for (int i = 0; i < n_vision_tokens; i++) {
                    float *embd_i = vision_embd + (size_t)i * embd_stride;
                    memcpy(all_embds.data() + (size_t)idx * n_embd_for_batch, embd_i, n_embd_for_batch * sizeof(float));
                    all_cache_pos[idx] = cache_pos;
                    int y = i / merged_w;
                    int x = i % merged_w;
                    all_pos_t[idx] = image_pos0;
                    all_pos_h[idx] = image_pos0 + y;
                    all_pos_w[idx] = image_pos0 + x;
                    cache_pos++; idx++;
                }
                rope_pos += std::max(merged_w, merged_h);

                for (int i = 0; i < n_after; i++) {
                    size_t rb = dequant_row_size(model->token_embd.type, model->n_embd);
                    const void *rd = (const uint8_t*)model->token_embd.data + (size_t)tokens_after[i] * rb;
                    dequant_row(model->token_embd.type, rd, all_embds.data() + (size_t)idx * n_embd_for_batch, model->n_embd);
                    all_cache_pos[idx] = cache_pos;
                    all_pos_t[idx] = rope_pos;
                    all_pos_h[idx] = rope_pos;
                    all_pos_w[idx] = rope_pos;
                    cache_pos++; rope_pos++; idx++;
                }

                last_prefill_logits = gpu_llm_runner->prefillEmbds(
                    all_embds.data(), total_N, n_embd_for_batch,
                    all_cache_pos.data(), all_pos_t.data(), all_pos_h.data(), all_pos_w.data());

                t1 = clock();
                clock_gettime(CLOCK_MONOTONIC, &wall_t1);
                double wall_s = (wall_t1.tv_sec - wall_t0.tv_sec) + (wall_t1.tv_nsec - wall_t0.tv_nsec) / 1e9;

                if (last_prefill_logits) {
                    fprintf(stderr, "  GPU batched prefill done: %.1f s wall (%.1f s CPU)\n", wall_s, (double)(t1 - t0) / CLOCKS_PER_SEC);
                    batch_ok = true;
                    delete[] vision_embd;
                } else {
                    fprintf(stderr, "  GPU batch prefill not available (%s), falling back to per-token\n",
                            gpu_llm_runner->getLastError().c_str());
                    // Reset positions for per-token fallback
                    cache_pos = 0;
                    rope_pos = 0;
                }
            }
        }

        if (!batch_ok) {
            // Per-token GPU fallback (F16/F32 weights)
            // vision_embd is still valid (not deleted in batch path on failure)
            for (int i = 0; i < n_before; i++) {
                gpu_llm_runner->forward(tokens_before[i], cache_pos, rope_pos, rope_pos, rope_pos, false);
                cache_pos++; rope_pos++;
            }
            fprintf(stderr, "  Text prefix: %d tokens done\n", n_before);

            gpu_llm_runner->setDeepstackStride(embd_stride);
            fprintf(stderr, "  Vision: %d tokens (merged grid %dx%d, M-RoPE, deepstack=%d)...\n",
                    n_vision_tokens, merged_w, merged_h, n_ds);
            t0 = clock();
            const int image_pos0 = rope_pos;
            for (int i = 0; i < n_vision_tokens; i++) {
                float *embd_i = vision_embd + i * embd_stride;
                int y = i / merged_w;
                int x = i % merged_w;
                gpu_llm_runner->forwardEmbd(embd_i, cache_pos, image_pos0, image_pos0 + y, image_pos0 + x, false);
                cache_pos++;
            }
            rope_pos += std::max(merged_w, merged_h);
            t1 = clock();
            fprintf(stderr, "  Vision prefill: %.1f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

            delete[] vision_embd;

            for (int i = 0; i < n_after; i++) {
                bool last = (i == n_after - 1);
                last_prefill_logits = gpu_llm_runner->forward(tokens_after[i], cache_pos, rope_pos, rope_pos, rope_pos, last);
                cache_pos++; rope_pos++;
            }
            fprintf(stderr, "  Text suffix: %d tokens done\n", n_after);
        }
    } else {
        // === Single-token CPU path ===
        for (int i = 0; i < n_before; i++) {
            transformer_forward_pos(model, tokens_before[i], cache_pos, rope_pos, rope_pos, rope_pos);
            cache_pos++; rope_pos++;
        }
        fprintf(stderr, "  Text prefix: %d tokens done\n", n_before);

        model->ds_embd_stride = embd_stride;
        fprintf(stderr, "  Vision: %d tokens (merged grid %dx%d, M-RoPE, deepstack=%d)...\n",
                n_vision_tokens, merged_w, merged_h, n_ds);
        t0 = clock();
        const int image_pos0 = rope_pos;
        for (int i = 0; i < n_vision_tokens; i++) {
            float *embd_i = vision_embd + i * embd_stride;
            int y = i / merged_w;
            int x = i % merged_w;
            transformer_forward_embd_pos(model, embd_i, cache_pos, image_pos0, image_pos0 + y, image_pos0 + x);
            cache_pos++;
        }
        rope_pos += std::max(merged_w, merged_h);
        t1 = clock();
        fprintf(stderr, "  Vision prefill: %.1f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

        delete[] vision_embd;

        for (int i = 0; i < n_after; i++) {
            bool last = (i == n_after - 1);
            if (last) {
                last_prefill_logits = transformer_forward_logits_pos(model, tokens_after[i], cache_pos, rope_pos, rope_pos, rope_pos);
            } else {
                transformer_forward_pos(model, tokens_after[i], cache_pos, rope_pos, rope_pos, rope_pos);
            }
            cache_pos++; rope_pos++;
        }
        fprintf(stderr, "  Text suffix: %d tokens done\n", n_after);
    }

    // Generation
    fprintf(stderr, "\n=== Generation (%s LLM) ===\n\n", gpu_llm ? "GPU" : "CPU");

    int32_t next_token = -1;
    for (int g = 0; g < max_gen; g++) {
        float *logits;
        if (g == 0) {
            logits = last_prefill_logits;
        } else {
            if (gpu_llm) {
                logits = gpu_llm_runner->forward(next_token, cache_pos, rope_pos, rope_pos, rope_pos, true);
            } else {
                logits = transformer_forward_logits_pos(model, next_token, cache_pos, rope_pos, rope_pos, rope_pos);
            }
            cache_pos++;
            rope_pos++;
        }
        if (!logits) break;

        // Greedy argmax
        next_token = 0;
        for (int j = 1; j < n_vocab_actual; j++) {
            if (logits[j] > logits[next_token]) next_token = j;
        }

        if (next_token == bpe_eos_id(vocab) || next_token == bpe_eot_id(vocab)) {
            fprintf(stderr, "  [EOS token %d at gen step %d]\n", next_token, g);
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

        if (g == 0) {
            cache_pos++;
            rope_pos++;
        }
    }
    printf("\n");

    // Print profiling summary
    prof_summary();

    // Cleanup
    delete gpu_llm_runner;
    vision_free(vm);
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf_mmproj);
    gguf_close(gguf_main);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
