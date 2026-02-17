// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Multimodal inference: Vulkan GPU vision encoder + CPU LLM
// Loads an image, encodes with GPU, generates text description.
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
#include "../common/vision_encoder.h"
}

#include "vulkan_vision_encoder.hh"

// Simple PPM image loader (P6 binary format)
static uint8_t *load_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(f); return NULL;
    }

    // Skip comments
    int ch;
    while ((ch = fgetc(f)) == '#' || ch == ' ' || ch == '\n') {
        if (ch == '#') while (fgetc(f) != '\n');
    }
    ungetc(ch, f);

    int maxval;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) {
        fclose(f); return NULL;
    }
    fgetc(f); // consume newline

    size_t size = (size_t)(*w) * (*h) * 3;
    uint8_t *data = (uint8_t *)malloc(size);
    if (fread(data, 1, size, f) != size) {
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    return data;
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

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <model.gguf> --mmproj <mmproj.gguf> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>      LLM model GGUF\n");
    fprintf(stderr, "  --mmproj <path>     Vision mmproj GGUF\n");
    fprintf(stderr, "  --image <path.ppm>  Input image (PPM P6 format)\n");
    fprintf(stderr, "  --pattern <name>    Test pattern: checkerboard, gradient, red, circles (default: gradient)\n");
    fprintf(stderr, "  --prompt <text>     User prompt (default: 'Describe what you see in this image in detail.')\n");
    fprintf(stderr, "  --image-size <N>    Image size (default: 192)\n");
    fprintf(stderr, "  --max-gen <N>       Max tokens to generate (default: 100)\n");
    fprintf(stderr, "  --attn <mode>       Attention: cpu, naive, flash (default: flash)\n");
    fprintf(stderr, "  --device <id>       Vulkan device (default: 0)\n");
}

int main(int argc, char **argv) {
    std::string model_path, mmproj_path, image_path, pattern = "gradient";
    std::string prompt = "Describe what you see in this image in detail.";
    int image_size = 192, max_gen = 100, device_id = 0;
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

    int max_seq_len = 1024;
    transformer_model *model = transformer_load(gguf_main, max_seq_len);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

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
        img_rgb = load_ppm(image_path.c_str(), &img_w, &img_h);
        if (!img_rgb) {
            fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
            return 1;
        }
        // Resize to image_size if needed (simple nearest-neighbor)
        if (img_w != image_size || img_h != image_size) {
            fprintf(stderr, "Warning: image is %dx%d, need %dx%d. Using nearest-neighbor resize.\n",
                    img_w, img_h, image_size, image_size);
            uint8_t *resized = (uint8_t *)malloc(image_size * image_size * 3);
            for (int y = 0; y < image_size; y++)
                for (int x = 0; x < image_size; x++) {
                    int sx = x * img_w / image_size;
                    int sy = y * img_h / image_size;
                    int si = (sy * img_w + sx) * 3;
                    int di = (y * image_size + x) * 3;
                    resized[di] = img_rgb[si];
                    resized[di+1] = img_rgb[si+1];
                    resized[di+2] = img_rgb[si+2];
                }
            free(img_rgb);
            img_rgb = resized;
            img_w = img_h = image_size;
        }
    } else {
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
    fprintf(stderr, "Vision encoding: %d tokens x %d dim (%.3f s)\n",
            n_vision_tokens, proj_dim, (double)(t1 - t0) / CLOCKS_PER_SEC);

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

    // Prefill
    fprintf(stderr, "\n=== Prefill ===\n");
    int pos = 0;

    // Text before vision
    for (int i = 0; i < n_before; i++) {
        transformer_forward(model, tokens_before[i], pos);
        pos++;
    }
    fprintf(stderr, "  Text prefix: %d tokens done\n", n_before);

    // Vision tokens
    fprintf(stderr, "  Vision: %d tokens...\n", n_vision_tokens);
    t0 = clock();
    for (int i = 0; i < n_vision_tokens; i++) {
        float *embd_i = vision_embd + i * proj_dim;
        transformer_forward_embd(model, embd_i, pos);
        pos++;
    }
    t1 = clock();
    fprintf(stderr, "  Vision prefill: %.1f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    delete[] vision_embd;

    // Text after vision
    for (int i = 0; i < n_after; i++) {
        if (i == n_after - 1) {
            transformer_forward_logits(model, tokens_after[i], pos);
        } else {
            transformer_forward(model, tokens_after[i], pos);
        }
        pos++;
    }
    fprintf(stderr, "  Text suffix: %d tokens done\n", n_after);

    // Generation
    fprintf(stderr, "\n=== Generation ===\n\n");

    int32_t next_token = -1;
    for (int g = 0; g < max_gen; g++) {
        float *logits;
        if (g == 0) {
            logits = model->logits;
        } else {
            logits = transformer_forward_logits(model, next_token, pos);
            pos++;
        }
        if (!logits) break;

        // Greedy argmax
        next_token = 0;
        for (int j = 1; j < model->n_vocab; j++) {
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

    // Cleanup
    vision_free(vm);
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf_mmproj);
    gguf_close(gguf_main);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
