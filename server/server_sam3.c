/*
 * server_sam3.c - SAM 3 CPU segmentation wrapper for diffusion-server.
 *
 * Compiled as a separate TU so the sam3 runner's _IMPLEMENTATION macros
 * (safetensors, ggml_dequant, cpu_compute, stb_image_resize2) don't collide
 * with server.c's own definitions.
 *
 * Exposes server_sam3_cpu_segment(): decode base64 image, tokenize phrase,
 * run the full sam3 pipeline, encode surviving masks (up to out_cap) as
 * single-channel PNGs via stb_image_write.
 *
 * SPDX-License-Identifier: MIT
 */

#define _POSIX_C_SOURCE 200809L

/* server.c already carries STB_IMAGE_WRITE_IMPLEMENTATION, so include the
 * header here for declarations only. STB_IMAGE_IMPLEMENTATION (the decoder)
 * is unique to this TU since server.c does not decode input images. */
#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"
#include "../common/stb_image_write.h"

#include "../cpu/sam3/sam3_runner.h"
#include "../cpu/sam3/sam3_clip_bpe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Per-checkpoint context cache ----
 * The sam3 weights load (32 ViT blocks + 24 text blocks + DETR + heads)
 * takes seconds-to-minutes of disk + dequant + preprocessing per request.
 * Cache one ctx keyed by ckpt path + thread count so repeated requests
 * reuse the loaded model. Caller serializes via g_infer_mu, so no extra
 * locking needed here. */
static sam3_ctx *g_cached_ctx        = NULL;
static char     *g_cached_ckpt       = NULL;
static int       g_cached_threads    = -1;

static sam3_ctx *get_or_create_ctx(const char *ckpt_path, int threads,
                                    char *err_buf, size_t err_cap,
                                    int *out_was_cached) {
    *out_was_cached = 0;
    if (g_cached_ctx && g_cached_ckpt &&
        strcmp(g_cached_ckpt, ckpt_path) == 0 &&
        g_cached_threads == threads) {
        *out_was_cached = 1;
        return g_cached_ctx;
    }
    if (g_cached_ctx) {
        sam3_destroy(g_cached_ctx);
        g_cached_ctx = NULL;
    }
    free(g_cached_ckpt);
    g_cached_ckpt = NULL;

    sam3_config cfg = { .ckpt_path = ckpt_path, .image_size = 1008, .num_threads = threads };
    double t0 = now_ms();
    fprintf(stderr, "sam3: loading checkpoint %s (one-time, will be cached) ...\n", ckpt_path);
    sam3_ctx *ctx = sam3_create(&cfg);
    if (!ctx) {
        snprintf(err_buf, err_cap, "sam3_create failed (ckpt=%s)", ckpt_path);
        return NULL;
    }
    fprintf(stderr, "sam3: ckpt loaded in %.0f ms (cached for subsequent requests)\n", now_ms() - t0);
    g_cached_ctx     = ctx;
    g_cached_ckpt    = strdup(ckpt_path);
    g_cached_threads = threads;
    return ctx;
}

/* ---- base64 decode ---- */
static int b64v(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return 26 + c - 'a';
    if (c >= '0' && c <= '9') return 52 + c - '0';
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
static uint8_t *b64_decode(const char *s, size_t in_len, size_t *out_len) {
    /* skip whitespace, compute usable len */
    size_t n = 0;
    for (size_t i = 0; i < in_len; i++) {
        char c = s[i];
        if (c == ' ' || c == '\r' || c == '\n' || c == '\t') continue;
        if (c == '=') break;
        if (b64v((unsigned char)c) < 0) return NULL;
        n++;
    }
    size_t pad = (4 - (n % 4)) % 4;
    size_t out_cap = ((n + pad) / 4) * 3;
    uint8_t *out = (uint8_t *)malloc(out_cap + 4);
    if (!out) return NULL;
    size_t oi = 0;
    int buf[4]; int bc = 0;
    for (size_t i = 0; i < in_len && oi < out_cap; i++) {
        char c = s[i];
        if (c == ' ' || c == '\r' || c == '\n' || c == '\t') continue;
        if (c == '=') break;
        int v = b64v((unsigned char)c);
        if (v < 0) { free(out); return NULL; }
        buf[bc++] = v;
        if (bc == 4) {
            out[oi++] = (uint8_t)((buf[0] << 2) | (buf[1] >> 4));
            if (oi < out_cap) out[oi++] = (uint8_t)(((buf[1] & 0xF) << 4) | (buf[2] >> 2));
            if (oi < out_cap) out[oi++] = (uint8_t)(((buf[2] & 0x3) << 6) | buf[3]);
            bc = 0;
        }
    }
    if (bc == 2) out[oi++] = (uint8_t)((buf[0] << 2) | (buf[1] >> 4));
    else if (bc == 3) {
        out[oi++] = (uint8_t)((buf[0] << 2) | (buf[1] >> 4));
        out[oi++] = (uint8_t)(((buf[1] & 0xF) << 4) | (buf[2] >> 2));
    }
    *out_len = oi;
    return out;
}

/* stb write-to-memory callback */
typedef struct { uint8_t *data; int len; int cap; } mbuf;
static void mbuf_write(void *ctx, void *d, int n) {
    mbuf *b = (mbuf *)ctx;
    if (b->len + n > b->cap) {
        int nc = b->cap ? b->cap * 2 : 4096;
        while (nc < b->len + n) nc *= 2;
        uint8_t *p = (uint8_t *)realloc(b->data, (size_t)nc);
        if (!p) return;
        b->data = p;
        b->cap = nc;
    }
    memcpy(b->data + b->len, d, (size_t)n);
    b->len += n;
}

typedef struct {
    uint8_t *png;   /* caller-free */
    int      png_len;
    int      width;
    int      height;
    float    score;
    float    box[4]; /* xyxy in source-image pixels */
} server_sam3_mask;

/* End-to-end sam3 CPU segmentation. All input paths required:
 *   ckpt_path  - detector_model safetensors
 *   vocab/merges - CLIP BPE tokenizer files
 *   img_bytes/img_len - compressed image (png/jpg/webp supported by stb_image)
 *   phrase     - text prompt, e.g. "cat"
 *   score_thr/mask_thr - HF defaults 0.3 / 0.5
 *   out_cap    - max masks to return
 *   threads    - 0 = auto
 *   out_masks  - caller allocates array of out_cap entries
 *   out_n      - [out] masks produced (0..out_cap)
 *
 * Returns 0 on success, non-zero on failure (err_buf populated).
 */
int server_sam3_cpu_segment(const char *ckpt_path,
                             const char *vocab_path,
                             const char *merges_path,
                             const uint8_t *img_bytes, size_t img_len,
                             const char *phrase,
                             float score_thr, float mask_thr,
                             int threads,
                             server_sam3_mask *out_masks, int out_cap,
                             int *out_n,
                             char *err_buf, size_t err_cap);

int server_sam3_cpu_segment(const char *ckpt_path,
                             const char *vocab_path,
                             const char *merges_path,
                             const uint8_t *img_bytes, size_t img_len,
                             const char *phrase,
                             float score_thr, float mask_thr,
                             int threads,
                             server_sam3_mask *out_masks, int out_cap,
                             int *out_n,
                             char *err_buf, size_t err_cap) {
    *out_n = 0;
    if (!ckpt_path || !*ckpt_path) {
        snprintf(err_buf, err_cap, "sam3 ckpt path not set");
        return 1;
    }
    if (!vocab_path || !merges_path) {
        snprintf(err_buf, err_cap, "sam3 vocab/merges paths not set");
        return 1;
    }
    if (!phrase || !*phrase) {
        snprintf(err_buf, err_cap, "phrase (inputs.text) required");
        return 1;
    }
    if (!img_bytes || img_len == 0) {
        snprintf(err_buf, err_cap, "image bytes missing");
        return 1;
    }
    /* Validate weight/tokenizer files upfront so an invalid path fails
     * fast instead of wedging inside the model loader. */
    if (access(ckpt_path, R_OK) != 0) {
        snprintf(err_buf, err_cap, "sam3 ckpt not readable: %s", ckpt_path);
        return 1;
    }
    if (access(vocab_path, R_OK) != 0) {
        snprintf(err_buf, err_cap, "sam3 vocab not readable: %s", vocab_path);
        return 1;
    }
    if (access(merges_path, R_OK) != 0) {
        snprintf(err_buf, err_cap, "sam3 merges not readable: %s", merges_path);
        return 1;
    }

    int W, H, C;
    unsigned char *rgb = stbi_load_from_memory(img_bytes, (int)img_len, &W, &H, &C, 3);
    if (!rgb) {
        snprintf(err_buf, err_cap, "failed to decode input image");
        return 2;
    }

    sam3_clip_bpe *tok = sam3_clip_bpe_load(vocab_path, merges_path);
    if (!tok) {
        stbi_image_free(rgb);
        snprintf(err_buf, err_cap, "failed to load sam3 BPE tokenizer (%s + %s)",
                 vocab_path, merges_path);
        return 3;
    }
    int32_t ids[32] = {0}, mask[32] = {0};
    int nv = sam3_clip_bpe_encode(tok, phrase, 32, ids, mask);
    sam3_clip_bpe_free(tok);
    if (nv < 0) {
        stbi_image_free(rgb);
        snprintf(err_buf, err_cap, "sam3 BPE encode failed for phrase '%s'", phrase);
        return 4;
    }

    int was_cached = 0;
    sam3_ctx *ctx = get_or_create_ctx(ckpt_path, threads, err_buf, err_cap, &was_cached);
    if (!ctx) {
        stbi_image_free(rgb);
        return 5;
    }
    if (was_cached) fprintf(stderr, "sam3: reusing cached ctx\n");

    int rc = 0;
    double t_overall = now_ms();
    #define STEP(call, code, msg) do { \
        double _t0 = now_ms(); \
        fprintf(stderr, "sam3: %-16s ...\n", (msg)); fflush(stderr); \
        if ((call)) { rc = (code); \
            snprintf(err_buf, err_cap, "sam3: %s", (msg)); goto done; } \
        fprintf(stderr, "sam3: %-16s done in %.0f ms\n", (msg), now_ms() - _t0); \
        fflush(stderr); \
    } while (0)
    STEP(sam3_set_image(ctx, rgb, H, W),                              10, "set_image");
    STEP(sam3_run_vit(ctx, 31),                                       11, "run_vit (32 blks)");
    STEP(sam3_run_fpn(ctx),                                           12, "run_fpn");
    STEP(sam3_set_input_ids(ctx, ids, mask),                          13, "set_input_ids");
    STEP(sam3_run_text(ctx),                                          14, "run_text");
    STEP(sam3_run_detr_enc(ctx),                                      15, "run_detr_enc");
    STEP(sam3_run_detr_dec(ctx),                                      16, "run_detr_dec");
    STEP(sam3_run_dot_score(ctx),                                     17, "run_dot_score");
    STEP(sam3_run_mask_dec(ctx),                                      18, "run_mask_dec");
    STEP(sam3_run_postprocess(ctx, H, W, score_thr, mask_thr),        19, "run_postprocess");
    #undef STEP
    fprintf(stderr, "sam3: pipeline total %.0f ms\n", now_ms() - t_overall);

    int nk = 0, oh = 0, ow = 0;
    const float   *scores = sam3_get_final_scores(ctx, &nk);
    const float   *boxes  = sam3_get_final_boxes(ctx, &nk);
    const uint8_t *masks  = sam3_get_final_masks(ctx, &nk, &oh, &ow);
    int emit = nk < out_cap ? nk : out_cap;
    for (int i = 0; i < emit; i++) {
        /* binary mask (0/1 → 0/255), encode as grayscale PNG */
        const uint8_t *m = masks + (size_t)i * (size_t)oh * (size_t)ow;
        uint8_t *g = (uint8_t *)malloc((size_t)oh * (size_t)ow);
        if (!g) { rc = 30; snprintf(err_buf, err_cap, "oom building mask"); goto done; }
        for (int p = 0; p < oh * ow; p++) g[p] = m[p] ? 255 : 0;
        mbuf mb = {0};
        int ok = stbi_write_png_to_func(mbuf_write, &mb, ow, oh, 1, g, ow);
        free(g);
        if (!ok || !mb.data) {
            free(mb.data);
            rc = 31; snprintf(err_buf, err_cap, "mask PNG encode failed"); goto done;
        }
        out_masks[i].png     = mb.data;
        out_masks[i].png_len = mb.len;
        out_masks[i].width   = ow;
        out_masks[i].height  = oh;
        out_masks[i].score   = scores[i];
        out_masks[i].box[0]  = boxes[i*4+0];
        out_masks[i].box[1]  = boxes[i*4+1];
        out_masks[i].box[2]  = boxes[i*4+2];
        out_masks[i].box[3]  = boxes[i*4+3];
    }
    *out_n = emit;

done:
    /* ctx is cached — do NOT destroy. Released at process exit. */
    stbi_image_free(rgb);
    return rc;
}

void server_sam3_free_mask(server_sam3_mask *m) {
    if (!m) return;
    free(m->png);
    m->png = NULL;
    m->png_len = 0;
}
