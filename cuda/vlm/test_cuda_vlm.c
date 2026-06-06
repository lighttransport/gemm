/*
 * test_cuda_vlm.c - End-to-end VLM inference: CUDA vision encoder + CUDA LLM
 *
 * Usage:
 *   ./test_cuda_vlm <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen]
 *                   [-n max_tokens] [--resize dynamic|fit] [--budget N]
 *
 * Positional args after model+mmproj are classified by shape (mirrors
 * cpu/vlm/test_vision so the sidecar's OursQwenVlmBackend can shell-out
 * to either binary with the same argv):
 *   - extension .png/.jpg/.jpeg/.bmp/.ppm  -> image path
 *   - starts with a digit                  -> max_gen token cap
 *   - anything else                        -> user prompt
 *
 * If no image is passed a synthetic checkerboard is used (smoke-test
 * parity with test_vision).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>

/* stb_image for JPEG loading */
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* SafeTensors loader needed by cuda_llm_runner helper paths */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

/* Dequantization */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

/* BPE tokenizer */
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

/* transformer.h for qtensor type */
#include "../../common/transformer.h"

/* CPU vision model for normalize_image + metadata */
#define VISION_ENCODER_IMPLEMENTATION
#include "../../common/vision_encoder.h"

/* CUDA runners */
#include "cuda_vision_encoder.h"
#include "../llm/cuda_llm_runner.h"
/* Standalone llama.cpp vision encoder binary (avoids symbol conflicts) */
#include <sys/wait.h>
#include <unistd.h>

static int arg_looks_like_image(const char *s) {
    if (!s) return 0;
    int n = (int)strlen(s);
    if (n < 4) return 0;
    const char *ext = s + n - 4;
    const char *ext5 = (n >= 5) ? s + n - 5 : NULL;
    if (strcasecmp(ext, ".png") == 0) return 1;
    if (strcasecmp(ext, ".jpg") == 0) return 1;
    if (strcasecmp(ext, ".bmp") == 0) return 1;
    if (strcasecmp(ext, ".ppm") == 0) return 1;
    if (ext5 && strcasecmp(ext5, ".jpeg") == 0) return 1;
    return 0;
}

static unsigned char *generate_checkerboard(int width, int height, int cell_size) {
    unsigned char *img = (unsigned char *)malloc((size_t)width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int checker = ((x / cell_size) + (y / cell_size)) & 1;
            unsigned char val = checker ? 255 : 0;
            int idx = (y * width + x) * 3;
            img[idx + 0] = val;
            img[idx + 1] = val;
            img[idx + 2] = val;
        }
    }
    return img;
}

/* Cubic kernel coefficient for bicubic interpolation (a = -0.5) */
static float cubic_kernel(float x) {
    x = fabsf(x);
    if (x < 1.0f) return (1.5f*x - 2.5f)*x*x + 1.0f;
    if (x < 2.0f) return ((-0.5f*x + 2.5f)*x - 4.0f)*x + 2.0f;
    return 0.0f;
}

/* Bicubic resize with antialiasing, matching PyTorch's BICUBIC antialias=True */
static unsigned char *bicubic_resize(const unsigned char *src, int sw, int sh,
                                      int dw, int dh) {
    unsigned char *dst = (unsigned char *)malloc(dw * dh * 3);
    float x_scale = (float)sw / dw;
    float y_scale = (float)sh / dh;
    for (int dy = 0; dy < dh; dy++) {
        float cy = ((float)dy + 0.5f) * y_scale - 0.5f;  /* align_corners=False */
        int y0 = (int)floorf(cy);
        for (int dx = 0; dx < dw; dx++) {
            float cx = ((float)dx + 0.5f) * x_scale - 0.5f;
            int x0 = (int)floorf(cx);
            float v[3] = {0,0,0};
            float total_w = 0;
            for (int ky = -1; ky <= 2; ky++) {
                int sy = y0 + ky;
                if (sy < 0) sy = 0;
                if (sy >= sh) sy = sh - 1;
                float wy = cubic_kernel(ky - (cy - y0));
                for (int kx = -1; kx <= 2; kx++) {
                    int sx = x0 + kx;
                    if (sx < 0) sx = 0;
                    if (sx >= sw) sx = sw - 1;
                    float w = wy * cubic_kernel(kx - (cx - x0));
                    int idx = (sy * sw + sx) * 3;
                    for (int c = 0; c < 3; c++) v[c] += src[idx + c] * w;
                    total_w += w;
                }
            }
            if (total_w > 0) {
                for (int c = 0; c < 3; c++) {
                    float val = v[c] / total_w + 0.5f;
                    if (val < 0) val = 0;
                    if (val > 255) val = 255;
                    dst[(dy * dw + dx) * 3 + c] = (unsigned char)val;
                }
            }
        }
    }
    return dst;
}

/* Simple bilinear resize */
static unsigned char *bilinear_resize(const unsigned char *src, int sw, int sh,
                                       int dw, int dh) {
    unsigned char *dst = (unsigned char *)malloc(dw * dh * 3);
    for (int dy = 0; dy < dh; dy++) {
        float fy = (sh > 1) ? (float)dy * (sh - 1) / (dh - 1) : 0;
        int y0 = (int)fy, y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int dx = 0; dx < dw; dx++) {
            float fx = (sw > 1) ? (float)dx * (sw - 1) / (dw - 1) : 0;
            int x0 = (int)fx, x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float v = src[(y0*sw+x0)*3+c] * (1-wy)*(1-wx)
                        + src[(y0*sw+x1)*3+c] * (1-wy)*wx
                        + src[(y1*sw+x0)*3+c] * wy*(1-wx)
                        + src[(y1*sw+x1)*3+c] * wy*wx;
                dst[(dy*dw+dx)*3+c] = (unsigned char)(v + 0.5f);
            }
        }
    }
    return dst;
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* llama.cpp-style dynamic resolution: preserve aspect ratio within pixel budget */
static void calc_size_preserved_ratio(int orig_w, int orig_h, int align_size,
                                       int min_pixels, int max_pixels,
                                       int *out_w, int *out_h) {
    /* Round to nearest align_size */
    int w = (int)((orig_w + align_size / 2) / align_size) * align_size;
    int h = (int)((orig_h + align_size / 2) / align_size) * align_size;
    if (w < align_size) w = align_size;
    if (h < align_size) h = align_size;

    if ((long long)w * h > max_pixels) {
        /* Scale down: use floor rounding to stay under budget */
        float beta = sqrtf((float)orig_w * orig_h / max_pixels);
        w = (int)(orig_w / beta / align_size) * align_size;  /* floor */
        h = (int)(orig_h / beta / align_size) * align_size;  /* floor */
        if (w < align_size) w = align_size;
        if (h < align_size) h = align_size;
    } else if ((long long)w * h < min_pixels) {
        /* Scale up: use ceil rounding to meet minimum */
        float beta = sqrtf((float)min_pixels / (orig_w * orig_h));
        w = ((int)ceilf((float)(orig_w * beta) / align_size)) * align_size;
        h = ((int)ceilf((float)(orig_h * beta) / align_size)) * align_size;
    }

    *out_w = w;
    *out_h = h;
}

static int32_t argmax_i32(const float *x, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

typedef struct {
    int32_t eos_id;
    int32_t eot_id;
    int32_t think_start_id;
    int32_t think_end_id;
    int in_thought;
} qwen_decode_state;

typedef enum {
    RB_IDLE,
    RB_COUNTING,
    RB_FORCING,
    RB_DONE
} rb_state;

typedef struct {
    rb_state state;
    int budget;
    int count;
    int32_t start_tokens[16];
    int n_start;
    int start_matched;
    int32_t end_tokens[16];
    int n_end;
    int end_matched;
    int force_idx;
} reasoning_budget;

static const char *gguf_get_kv_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

static int32_t find_exact_token_id(const bpe_vocab *vocab, const char *text) {
    if (!vocab || !text) return -1;
    return bpe_hm_get(&vocab->token_to_id, text, (int)strlen(text));
}

static qwen_decode_state init_decode_state(const bpe_vocab *vocab) {
    qwen_decode_state st;
    st.eos_id = bpe_eos_id(vocab);
    st.eot_id = bpe_eot_id(vocab);
    st.think_start_id = find_exact_token_id(vocab, "<think>");
    st.think_end_id = find_exact_token_id(vocab, "</think>");
    st.in_thought = 0;
    return st;
}

static reasoning_budget rb_init(const bpe_vocab *vocab, int budget) {
    reasoning_budget rb;
    memset(&rb, 0, sizeof(rb));
    rb.state = RB_IDLE;
    rb.budget = budget;
    rb.n_start = bpe_tokenize(vocab, "<think>\n", -1, rb.start_tokens, 16);
    if (rb.n_start < 0) rb.n_start = 0;
    rb.n_end = bpe_tokenize(vocab, "</think>\n\n", -1, rb.end_tokens, 16);
    if (rb.n_end < 0) rb.n_end = 0;
    return rb;
}

static void rb_observe(reasoning_budget *rb, int32_t token) {
    switch (rb->state) {
    case RB_IDLE:
        if (rb->n_start > 0 && token == rb->start_tokens[rb->start_matched]) {
            rb->start_matched++;
            if (rb->start_matched >= rb->n_start) {
                rb->state = RB_COUNTING;
                rb->count = 0;
                rb->start_matched = 0;
            }
        } else {
            rb->start_matched = 0;
            if (rb->n_start > 0 && token == rb->start_tokens[0]) rb->start_matched = 1;
        }
        break;
    case RB_COUNTING:
        rb->count++;
        if (rb->n_end > 0 && token == rb->end_tokens[rb->end_matched]) {
            rb->end_matched++;
            if (rb->end_matched >= rb->n_end) {
                rb->state = RB_DONE;
                rb->end_matched = 0;
            }
        } else {
            rb->end_matched = 0;
            if (rb->n_end > 0 && token == rb->end_tokens[0]) rb->end_matched = 1;
        }
        if (rb->state == RB_COUNTING && rb->count >= rb->budget) {
            rb->state = RB_FORCING;
            rb->force_idx = 0;
        }
        break;
    case RB_FORCING:
    case RB_DONE:
        break;
    }
}

static int rb_needs_forcing(const reasoning_budget *rb) {
    return rb->state == RB_FORCING;
}

static int32_t rb_force_next(reasoning_budget *rb) {
    if (rb->state != RB_FORCING) return -1;
    if (rb->force_idx >= rb->n_end) {
        rb->state = RB_DONE;
        return -1;
    }
    {
        int32_t tok = rb->end_tokens[rb->force_idx++];
        if (rb->force_idx >= rb->n_end) rb->state = RB_DONE;
        return tok;
    }
}

static void emit_visible_token(const bpe_vocab *vocab, qwen_decode_state *st, int32_t token) {
    if (token == st->eos_id || token == st->eot_id) return;
    if (token == st->think_start_id) {
        st->in_thought = 1;
        return;
    }
    if (token == st->think_end_id) {
        st->in_thought = 0;
        return;
    }
    if (st->in_thought) return;

    {
        const char *tok_str = bpe_token_to_str(vocab, token);
        if (!tok_str) return;
        int dec_len = 0;
        char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
        if (decoded && dec_len > 0) {
            fwrite(decoded, 1, (size_t)dec_len, stdout);
            fflush(stdout);
        }
        free(decoded);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen] [-n N] [--resize dynamic|fit|bicubic] [--budget N] [--bf16] [--vision-engine cuda|llama]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = NULL;
    const char *user_prompt = "Describe the image briefly.";
    int max_gen = 128;
    int reasoning_budget_tokens = 32;
    int resize_mode = 0;  /* 0 = dynamic (default, matches llama.cpp), 1 = fit (simple) */
    int use_bf16 = 0;
    int vision_engine = 0; /* 0 = CUDA (default), 1 = llama.cpp (reference) */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) max_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bf16") == 0) use_bf16 = 1;
        else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) reasoning_budget_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) user_prompt = argv[++i];
        else if (strcmp(argv[i], "--vision-engine") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "cuda") == 0) vision_engine = 0;
            else if (strcmp(argv[i+1], "llama") == 0) vision_engine = 1;
            else if (strcmp(argv[i+1], "llamacpp") == 0) vision_engine = 1;
            else { fprintf(stderr, "Unknown vision engine: %s (use 'cuda' or 'llama')\n", argv[i+1]); return 1; }
            i++;
        }
        else if (strcmp(argv[i], "--resize") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "fit") == 0) resize_mode = 1;
            else if (strcmp(argv[i], "dynamic") == 0) resize_mode = 0;
            else if (strcmp(argv[i], "bicubic") == 0) resize_mode = 2;
            else { fprintf(stderr, "Unknown resize mode: %s (use 'dynamic'|'fit'|'bicubic')\n", argv[i]); return 1; }
        }
        else if (arg_looks_like_image(argv[i])) image_path = argv[i];
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            int v = atoi(argv[i]);
            if (v > 0) max_gen = v;
        }
        else user_prompt = argv[i];
    }

    fprintf(stderr, "=== CUDA VLM Pipeline ===\n");
    fprintf(stderr, "LLM model:   %s\n", model_path);
    fprintf(stderr, "Vision model: %s\n", mmproj_path);
    fprintf(stderr, "Image:       %s\n", image_path ? image_path : "<synthetic checkerboard>");
    fprintf(stderr, "Prompt:      %s\n", user_prompt);
    fprintf(stderr, "Max gen:     %d\n", max_gen);
    fprintf(stderr, "Budget:      %d\n\n", reasoning_budget_tokens);

    /* ---- 1. Load image ---- */
    int img_w, img_h, img_c;
    unsigned char *img_data = NULL;
    if (image_path) {
        fprintf(stderr, "Loading image...\n");
        img_data = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
        if (!img_data) {
            fprintf(stderr, "Failed to load image: %s\n", image_path);
            return 1;
        }
        fprintf(stderr, "Image: %dx%d (%d channels)\n", img_w, img_h, img_c);
    } else {
        img_w = img_h = 448;
        img_c = 3;
        img_data = generate_checkerboard(img_w, img_h, 32);
        fprintf(stderr, "Using synthetic checkerboard: %dx%d\n", img_w, img_h);
    }

    /* ---- 2. Load mmproj GGUF ---- */
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path);
    gguf_context *gguf_mmproj = gguf_open(mmproj_path, 0);
    if (!gguf_mmproj) { fprintf(stderr, "Failed to open mmproj GGUF\n"); return 1; }

    /* Load CPU vision model for metadata (mean/std, patch_size, spatial_merge) */
    vision_model *vm = vision_load(gguf_mmproj);
    if (!vm) { fprintf(stderr, "Failed to load vision model metadata\n"); return 1; }

    /* ---- 3. Resize image ---- */
    int ps = vm->patch_size;
    int tile = ps * vm->spatial_merge;  /* typically 32 for Qwen3.5-VL */
    int new_w, new_h;

    int dyn_max_pixels = 4194304;  /* also passed to encoder */
    if (resize_mode == 0 || resize_mode == 2) {
        /* Dynamic mode: llama.cpp-style preserved ratio (mode 2 = bicubic, same calc) */
        int min_pixels = 8192;
        /* Try to read from GGUF metadata */
        int idx;
        idx = gguf_find_key(gguf_mmproj, "clip.vision.image_min_pixels");
        if (idx >= 0) min_pixels = (int)gguf_mmproj->kv[idx].value.u32;
        idx = gguf_find_key(gguf_mmproj, "clip.vision.image_max_pixels");
        if (idx >= 0) dyn_max_pixels = (int)gguf_mmproj->kv[idx].value.u32;

        calc_size_preserved_ratio(img_w, img_h, tile, min_pixels, dyn_max_pixels, &new_w, &new_h);
        fprintf(stderr, "Dynamic resize: %dx%d -> %dx%d (min_px=%d, max_px=%d)\n",
                img_w, img_h, new_w, new_h, min_pixels, dyn_max_pixels);
    } else {
        /* Fit mode: align to tile, scale down if exceeds max patches */
        int max_patches = vm->n_patches;
        new_w = (img_w / tile) * tile;
        new_h = (img_h / tile) * tile;
        if (new_w < tile) new_w = tile;
        if (new_h < tile) new_h = tile;

        int n_patches = (new_w / ps) * (new_h / ps);
        if (n_patches > max_patches) {
            float scale = sqrtf((float)max_patches / n_patches);
            new_w = ((int)(new_w * scale) / tile) * tile;
            new_h = ((int)(new_h * scale) / tile) * tile;
            if (new_w < tile) new_w = tile;
            if (new_h < tile) new_h = tile;
        }
        fprintf(stderr, "Fit resize: %dx%d -> %dx%d\n", img_w, img_h, new_w, new_h);
    }

    /* Bicubic (matches PyTorch) or bilinear resize */
    unsigned char *resized = NULL;
    if (new_w != img_w || new_h != img_h) {
        if (resize_mode == 2)
            resized = bicubic_resize(img_data, img_w, img_h, new_w, new_h);
        else
            resized = bilinear_resize(img_data, img_w, img_h, new_w, new_h);
        stbi_image_free(img_data);
        img_data = resized;
        img_w = new_w;
        img_h = new_h;
    }

    /* ---- 4. Normalize image ---- */
    fprintf(stderr, "Normalizing image...\n");
    float *rgb_norm = vision_normalize_image(vm, img_data, img_w, img_h);
    if (resized) free(resized); else stbi_image_free(img_data);
    img_data = NULL;

    /* ---- 5. Vision encode (CUDA or llama.cpp reference) ---- */
    int n_vision_tokens = 0, total_embd = 0, proj_dim = 0, expected_proj_dim = 0;
    float *vision_embd = NULL;
    double t_vis = 0, t0 = 0, t1 = 0;

    if (vision_engine == 0) {
        /* CUDA vision encoder */
        fprintf(stderr, "Initializing CUDA vision encoder%s...\n", use_bf16 ? " (BF16)" : "");
        if (use_bf16) setenv("VLM_BF16", "1", 1);
        cuda_vision_runner *cvis = cuda_vision_init(0, 1, 0);
        if (!cvis) { fprintf(stderr, "CUDA vision init failed\n"); return 1; }
        cuda_vision_set_max_pixels(cvis, dyn_max_pixels);
        if (cuda_vision_load_weights(cvis, gguf_mmproj) != 0) {
            fprintf(stderr, "CUDA vision weight load failed\n"); return 1;
        }
        fprintf(stderr, "Encoding image (%dx%d)...\n", img_w, img_h);
        double tvs0 = get_time_ms();
        vision_embd = cuda_vision_encode(cvis, rgb_norm, img_w, img_h);
        t_vis = get_time_ms() - tvs0;

        if (!vision_embd) { fprintf(stderr, "Vision encoding failed\n"); return 1; }

        int patches_w = img_w / vm->patch_size;
        int patches_h = img_h / vm->patch_size;
        n_vision_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
        total_embd = cuda_vision_total_embd(cvis);
        proj_dim = cuda_vision_proj_dim(cvis);
        fprintf(stderr, "Vision encoding: %d tokens x %d dim (proj=%d, %.1f ms)\n",
                n_vision_tokens, total_embd, proj_dim, t_vis);

        /* Free CUDA vision encoder (no longer needed after encoding) */
        fprintf(stderr, "Freeing vision encoder to reclaim VRAM...\n");
        cuda_vision_free(cvis);
    } else {
        /* llama.cpp vision encoder (standalone binary, avoids symbol conflicts).
         * The binary is built as llamacpp_vision_standalone and takes:
         *   <mmproj> <image> <out_w> <out_h> <output.bin>
         * It writes the embeddings to a file we then read back. */
        const char *standalone_bin = "./llamacpp_vision_standalone";
        char tmp_path[256];
        snprintf(tmp_path, sizeof(tmp_path), "/tmp/llamacpp_vision_%d.bin", getpid());

        fprintf(stderr, "Initializing llama.cpp vision encoder (standalone)...\n");

        /* Get output dimensions by encoding a small test image first.
         * Actually, we need n_tokens and proj_dim. Let the standalone compute them.
         * We'll estimate n_tokens from patch geometry. */
        int ps = vm->patch_size;
        int sm = vm->spatial_merge;
        int patches_w = img_w / ps;
        int patches_h = img_h / ps;
        proj_dim = vm->proj_dim;  /* from vision_model metadata */
        n_vision_tokens = (patches_w / sm) * (patches_h / sm);
        total_embd = proj_dim;  /* no deepstack in llama's clip */
        fprintf(stderr, "llamacpp vision: ~%d tokens x %d dim\n", n_vision_tokens, proj_dim);

        /* Launch standalone vision encoder */
        pid_t pid = fork();
        if (pid == 0) {
            /* Child: exec the standalone binary */
            char w_str[32], h_str[32];
            snprintf(w_str, sizeof(w_str), "%d", img_w);
            snprintf(h_str, sizeof(h_str), "%d", img_h);
            execlp(standalone_bin, standalone_bin,
                   mmproj_path, image_path,
                   w_str, h_str, tmp_path, (char *)NULL);
            /* If execlp fails: */
            fprintf(stderr, "Failed to exec %s\n", standalone_bin);
            _exit(1);
        } else if (pid < 0) {
            fprintf(stderr, "fork failed\n");
            return 1;
        }

        /* Parent: wait for child */
        int status;
        double tvs0 = get_time_ms();
        waitpid(pid, &status, 0);
        t_vis = get_time_ms() - tvs0;

        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            /* Read output file */
            FILE *f = fopen(tmp_path, "rb");
            if (!f) { fprintf(stderr, "Cannot read %s\n", tmp_path); return 1; }
            int32_t hdr[4];
            if (fread(hdr, sizeof(int32_t), 4, f) != 4) { fclose(f); return 1; }
            int n_tok_file = hdr[0], dim_file = hdr[1];
            fprintf(stderr, "llamacpp vision: got %d tokens x %d dim\n", n_tok_file, dim_file);

            /* Preserve full llama output including deepstack features. */
            expected_proj_dim = proj_dim;  /* save for n_embd check below */
            if (dim_file != total_embd) {
                fprintf(stderr, "  Note: file has %dx%d (expected %dx%d)\n",
                        n_tok_file, dim_file, n_vision_tokens, total_embd);
            }
            n_vision_tokens = n_tok_file;
            proj_dim = dim_file;  /* full dim including deepstack for stride */
            int prev_total = total_embd;
            total_embd = dim_file;
            n_vision_tokens = n_tok_file;
            total_embd = dim_file;
            proj_dim = dim_file;

            vision_embd = (float *)malloc((size_t)n_vision_tokens * total_embd * sizeof(float));
            size_t nread = fread(vision_embd, sizeof(float),
                                 (size_t)n_vision_tokens * total_embd, f);
            fclose(f);
            unlink(tmp_path);  /* clean up temp file */
            if ((int)nread != n_vision_tokens * total_embd) {
                fprintf(stderr, "Short read\n");
                free(vision_embd);
                return 1;
            }
            /* If llama outputs include deepstack features (dim > expected_proj_dim),
             * slice to only the main projection for the LLM. */
            if (dim_file > expected_proj_dim && expected_proj_dim > 0) {
                float *sliced = (float *)malloc((size_t)n_vision_tokens * expected_proj_dim * sizeof(float));
                for (int t = 0; t < n_vision_tokens; t++)
                    for (int d = 0; d < expected_proj_dim; d++)
                        sliced[t * expected_proj_dim + d] = vision_embd[t * dim_file + d];
                free(vision_embd);
                vision_embd = sliced;
                total_embd = expected_proj_dim;
                proj_dim = expected_proj_dim;
                fprintf(stderr, "  Sliced to main projection: %d x %d\n", n_vision_tokens, proj_dim);
            }
            fprintf(stderr, "Vision encoding: %.1f ms\n", t_vis);
        } else {
            fprintf(stderr, "llamacpp vision failed (status=%d)\n", status);
            return 1;
        }
    }
    free(rgb_norm);
    gguf_close(gguf_mmproj);
    gguf_mmproj = NULL;

    /* Print vision embedding stats */
    {
        float vmin = vision_embd[0], vmax = vision_embd[0], vnorm = 0;
        for (int i = 0; i < n_vision_tokens * total_embd; i++) {
            if (vision_embd[i] < vmin) vmin = vision_embd[i];
            if (vision_embd[i] > vmax) vmax = vision_embd[i];
            vnorm += vision_embd[i] * vision_embd[i];
        }
        fprintf(stderr, "Vision embeddings: min=%.4f max=%.4f L2=%.4f\n",
                vmin, vmax, sqrtf(vnorm));
    }
    /* Dump vision embeddings for comparison if VLM_DUMP_VISION_EMBD is set */
    {
        const char *dump_path = getenv("VLM_DUMP_VISION_EMBD");
        if (dump_path && dump_path[0]) {
            FILE *f = fopen(dump_path, "wb");
            if (f) {
                int32_t hdr[4] = { n_vision_tokens, total_embd, 0, 0 };
                fwrite(hdr, sizeof(int32_t), 4, f);
                fwrite(vision_embd, sizeof(float), (size_t)n_vision_tokens * total_embd, f);
                fclose(f);
                fprintf(stderr, "Dumped vision embeddings to %s\n", dump_path);
            }
        }
    }

    /* ---- 6. Load LLM model ---- */
    fprintf(stderr, "\nLoading LLM model: %s\n", model_path);
    gguf_context *gguf_llm = gguf_open(model_path, 1);
    if (!gguf_llm) { fprintf(stderr, "Failed to open LLM GGUF\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_llm);
    if (!vocab) { fprintf(stderr, "Failed to load vocab\n"); return 1; }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    /* Need enough seq_len for prompt + vision tokens + generation */
    int max_seq_len = n_vision_tokens + 256 + max_gen;
    if (max_seq_len < 1024) max_seq_len = 1024;

    fprintf(stderr, "Initializing CUDA LLM runner (max_seq_len=%d)...\n", max_seq_len);
    cuda_llm_runner *llm = cuda_llm_init(0, 1);
    if (!llm) { fprintf(stderr, "CUDA LLM init failed\n"); return 1; }
    if (cuda_llm_load_weights(llm, gguf_llm, max_seq_len) != 0) {
        fprintf(stderr, "CUDA LLM weight load failed\n");
        return 1;
    }

    int n_embd = cuda_llm_n_embd(llm);
    int n_vocab = cuda_llm_n_vocab(llm);
    fprintf(stderr, "LLM: n_embd=%d n_vocab=%d n_layers=%d\n",
            n_embd, n_vocab, cuda_llm_n_layers(llm));

    /* Verify vision main projection dim matches LLM n_embd */
    if (expected_proj_dim > 0 && expected_proj_dim != n_embd) {
        fprintf(stderr, "ERROR: vision proj_dim=%d != LLM n_embd=%d\n", expected_proj_dim, n_embd);
        return 1;
    }

    /* ---- 7. Build multimodal prompt ---- */
    char text_before[256];
    char text_after[16384];
    {
        const char *tmpl = gguf_get_kv_string(gguf_llm, "tokenizer.chat_template");
        (void)tmpl;  /* always use chatml+vision format */
        snprintf(text_before, sizeof(text_before), "<|im_start|>user\n<|vision_start|>");
        snprintf(text_after, sizeof(text_after),
                 "<|vision_end|>%s<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n",
                 user_prompt);
    }

    int32_t tokens_before[64];
    int32_t *tokens_after = (int32_t *)malloc(sizeof(int32_t) * 4096);
    int n_before = bpe_tokenize(vocab, text_before, -1, tokens_before, 64);
    int n_after  = bpe_tokenize(vocab, text_after, -1, tokens_after, 4096);

    fprintf(stderr, "\nTokens before vision (%d):", n_before);
    for (int i = 0; i < n_before; i++)
        fprintf(stderr, " %d", tokens_before[i]);
    fprintf(stderr, "\nTokens after vision (%d):", n_after);
    for (int i = 0; i < n_after; i++)
        fprintf(stderr, " %d", tokens_after[i]);
    fprintf(stderr, "\n");

    int total_prompt_len = n_before + n_vision_tokens + n_after;
    fprintf(stderr, "\nTotal prompt: %d tokens (%d text + %d vision + %d text)\n",
            total_prompt_len, n_before, n_vision_tokens, n_after);

    /* ---- 8. Prefill ---- */
    fprintf(stderr, "\n=== Prefill ===\n");
    int pos = 0;

    /* Text tokens before vision */
    t0 = get_time_ms();
    if (!cuda_llm_prefill(llm, tokens_before, NULL, 0, n_before, pos)) {
        fprintf(stderr, "ERROR: text-before prefill failed\n");
        return 1;
    }
    pos += n_before;
    t1 = get_time_ms();
    fprintf(stderr, "  Text before: %d tokens, %.1f ms\n", n_before, t1 - t0);

    /* Vision tokens */
    t0 = get_time_ms();
    if (!cuda_llm_prefill(llm, NULL, vision_embd, total_embd, n_vision_tokens, pos)) {
        fprintf(stderr, "ERROR: vision prefill failed\n");
        return 1;
    }
    pos += n_vision_tokens;
    t1 = get_time_ms();
    fprintf(stderr, "  Vision prefill: %d tokens, %.1f ms (%.2f ms/token)\n",
            n_vision_tokens, t1 - t0, (t1 - t0) / n_vision_tokens);

    /* Text tokens after vision (last one returns logits) */
    float *logits = NULL;
    t0 = get_time_ms();
    logits = cuda_llm_prefill_logits(llm, tokens_after, NULL, 0, n_after, pos);
    pos += n_after;
    t1 = get_time_ms();
    fprintf(stderr, "  Text after: %d tokens, %.1f ms\n", n_after, t1 - t0);

    if (!logits) {
        fprintf(stderr, "ERROR: forward_logits returned NULL\n");
        return 1;
    }

    /* Debug: check first token prediction */
    {
        int32_t first = argmax_i32(logits, n_vocab);
        qwen_decode_state tmp_state = init_decode_state(vocab);
        int32_t eos = tmp_state.eos_id, eot = tmp_state.eot_id;
        const char *first_str = bpe_token_to_str(vocab, first);
        fprintf(stderr, "  [First token: %d '%s'  (EOS=%d EOT=%d)]\n", first, first_str ? first_str : "?", eos, eot);
        if (first == eos) fprintf(stderr, "  [WARNING: EOS immediately! Model does not understand vision embeddings]\n");
        /* Also check top-5 candidates */
        int32_t top5[5]; float top5v[5] = {-1e30f};
        for (int j = 0; j < n_vocab; j++) {
            for (int k = 0; k < 5; k++) {
                if (logits[j] > top5v[k]) {
                    for (int k2 = 4; k2 > k; k2--) { top5[k2] = top5[k2-1]; top5v[k2] = top5v[k2-1]; }
                    top5[k] = j; top5v[k] = logits[j]; break;
                }
            }
        }
        fprintf(stderr, "  [Top-5 tokens:");
        for (int k = 0; k < 5; k++) {
            const char *s = bpe_token_to_str(vocab, top5[k]);
            fprintf(stderr, " %d='%s'(%.2f)", top5[k], s ? s : "?", top5v[k]);
        }
        fprintf(stderr, "]\n");
    }

    /* ---- 9. Generate ---- */
    fprintf(stderr, "\n=== Generation ===\n");
    qwen_decode_state decode_state = init_decode_state(vocab);
    reasoning_budget rb = rb_init(vocab, reasoning_budget_tokens);
    int32_t next_token = argmax_i32(logits, n_vocab);
    double gen_t0 = get_time_ms();

    for (int g = 0; g < max_gen; g++) {
        emit_visible_token(vocab, &decode_state, next_token);
        rb_observe(&rb, next_token);

        if (next_token == decode_state.eos_id || next_token == decode_state.eot_id) {
            fprintf(stderr, "\n  [EOS/EOT token %d at step %d]\n", next_token, g);
            break;
        }

        /* Next step */
        logits = cuda_llm_forward_logits(llm, next_token, pos);
        pos++;
        if (!logits) break;
        if (rb_needs_forcing(&rb)) {
            int32_t forced = rb_force_next(&rb);
            if (forced >= 0) {
                next_token = forced;
                continue;
            }
        }
        next_token = argmax_i32(logits, n_vocab);
    }
    double gen_t1 = get_time_ms();
    printf("\n");
    fprintf(stderr, "[thought: %d/%d tokens used]\n", rb.count, rb.budget);
    fprintf(stderr, "  [Decode: %d tokens in %.0f ms (%.2f ms/tok, %.1f tok/s)]\n",
            max_gen, gen_t1 - gen_t0,
            (gen_t1 - gen_t0) / (max_gen > 0 ? max_gen : 1),
            max_gen / ((gen_t1 - gen_t0) / 1000.0));

    /* ---- 10. Cleanup ---- */
    fprintf(stderr, "\nCleaning up...\n");
    free(vision_embd);
    free(tokens_after);
    vision_free(vm);
    cuda_llm_free(llm);
    bpe_vocab_free(vocab);
    gguf_close(gguf_llm);

    fprintf(stderr, "Done.\n");
    return 0;
}
