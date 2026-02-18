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
    int image_size = 0, max_gen = 100, device_id = 0, n_threads = 1;  // 0 = auto (smart resize)
    bool gpu_llm = false;
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
            int align_size = 32; // patch_size(16) * spatial_merge(2)
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

    // CPU vision encode for comparison
    {
        // Re-load image for CPU test (use same smart resize)
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
            float *cpu_embd = vision_encode(vm, cpu_norm, img_w, img_h);
            free(cpu_norm);
            if (cpu_embd) {
                fprintf(stderr, "  CPU vision embd[0][0..7]:");
                for (int j = 0; j < 8; j++) fprintf(stderr, " %.4f", cpu_embd[j]);
                fprintf(stderr, "\n");
                float norm = 0;
                for (int j = 0; j < proj_dim; j++) norm += cpu_embd[j] * cpu_embd[j];
                fprintf(stderr, "  CPU vision embd[0] norm=%.4f\n", sqrtf(norm));
                // Compare (full embedding including deepstack features)
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

    // Build token sequence
    // <|im_start|>user\n<|vision_start|>[vision]<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n
    std::string text_before = "<|im_start|>user\n<|vision_start|>";
    std::string text_after = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

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
    int pos = 0;
    int n_vocab_actual = gpu_llm ? gpu_llm_runner->n_vocab() : model->n_vocab;

    // Text before vision: all 3 RoPE dims use the same position
    for (int i = 0; i < n_before; i++) {
        if (gpu_llm) {
            gpu_llm_runner->forward(tokens_before[i], pos, pos, pos, pos, false);
        } else {
            transformer_forward_pos(model, tokens_before[i], pos, pos, pos, pos);
        }
        pos++;
    }
    fprintf(stderr, "  Text prefix: %d tokens done\n", n_before);

    // Vision tokens with M-RoPE 3D positions
    // Set deepstack embedding stride so LLM injects deepstack slices at early layers
    if (model) {
        model->ds_embd_stride = embd_stride;
    }
    if (gpu_llm_runner) {
        gpu_llm_runner->setDeepstackStride(embd_stride);
    }
    fprintf(stderr, "  Vision: %d tokens (merged grid %dx%d, M-RoPE, deepstack=%d)...\n",
            n_vision_tokens, merged_w, merged_h, n_ds);
    t0 = clock();
    for (int i = 0; i < n_vision_tokens; i++) {
        float *embd_i = vision_embd + i * embd_stride;
        int mrope_h = i / merged_w;
        int mrope_w = i % merged_w;
        if (gpu_llm) {
            gpu_llm_runner->forwardEmbd(embd_i, pos, 0, mrope_h, mrope_w, false);
        } else {
            transformer_forward_embd_pos(model, embd_i, pos, 0, mrope_h, mrope_w);
        }
        pos++;
    }
    t1 = clock();
    fprintf(stderr, "  Vision prefill: %.1f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    delete[] vision_embd;

    // Text after vision
    for (int i = 0; i < n_after; i++) {
        bool last = (i == n_after - 1);
        if (gpu_llm) {
            gpu_llm_runner->forward(tokens_after[i], pos, pos, pos, pos, last);
        } else {
            if (last) {
                transformer_forward_logits_pos(model, tokens_after[i], pos, pos, pos, pos);
            } else {
                transformer_forward_pos(model, tokens_after[i], pos, pos, pos, pos);
            }
        }
        pos++;
    }
    fprintf(stderr, "  Text suffix: %d tokens done\n", n_after);

    // Generation
    fprintf(stderr, "\n=== Generation (%s LLM) ===\n\n", gpu_llm ? "GPU" : "CPU");

    int32_t next_token = -1;
    for (int g = 0; g < max_gen; g++) {
        float *logits;
        if (g == 0) {
            if (gpu_llm) {
                // Logits already computed in last prefill step
                logits = gpu_llm_runner->forward(tokens_after[n_after - 1], pos - 1, pos - 1, pos - 1, pos - 1, true);
                // Actually we already called forward for the last token above with compute_logits=true,
                // so the logits are from that call. But we re-ran with same cache_pos which overwrites.
                // This is fine since it's the same token at the same position.
            } else {
                logits = model->logits;
            }
        } else {
            if (gpu_llm) {
                logits = gpu_llm_runner->forward(next_token, pos, true);
            } else {
                logits = transformer_forward_logits(model, next_token, pos);
            }
            pos++;
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

        if (g == 0) pos++;
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
