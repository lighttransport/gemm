/*
 * server_sam3_cuda.c - SAM 3 CUDA segmentation wrapper for diffusion-server.
 *
 * Mirrors server_sam3.c but dispatches through cuda/sam3/cuda_sam3_runner.
 * The CUDA runner uses NVRTC via cuew (no nvcc) and runs the same pipeline
 * stages as the CPU runner; pre/post-processing on the host matches.
 *
 * Compiled as a separate TU; reuses the BPE tokenizer and stb_image decoder
 * from server_sam3.c (linked via shared symbols if both TUs are present, or
 * directly when only the CUDA backend is enabled — see the local helpers
 * below for the standalone case).
 *
 * SPDX-License-Identifier: MIT
 */

#define _POSIX_C_SOURCE 200809L

#include "../cuda/sam3/cuda_sam3_runner.h"
#include "../cpu/sam3/sam3_clip_bpe.h"
#include "../common/stb_image.h"
#include "../common/stb_image_write.h"
#include "image_decode.h"
#include "server_sam3.h"  /* server_sam3_mask + server_sam3_free_mask */

/* CUDA_RUNNER_COMMON helpers are static and live inside cuda_sam3_runner.c. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double cuda_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Per-checkpoint CUDA context cache ----
 * cuda_sam3_create runs NVRTC compilation (seconds for first call) and
 * uploads ~3 GB of weights to the GPU. Cache the ctx across requests.
 * Caller serializes via g_infer_mu so no extra locking needed. */
static cuda_sam3_ctx *g_cuda_ctx        = NULL;
static char          *g_cuda_ckpt       = NULL;
static int            g_cuda_device_ord = -1;
static char          *g_cuda_precision  = NULL;

static cuda_sam3_ctx *cuda_get_or_create_ctx(const char *ckpt_path, int device_ord,
                                              const char *precision,
                                              char *err_buf, size_t err_cap,
                                              int *out_was_cached) {
    *out_was_cached = 0;
    const char *prec = (precision && *precision) ? precision : "fp16";
    if (g_cuda_ctx && g_cuda_ckpt &&
        strcmp(g_cuda_ckpt, ckpt_path) == 0 &&
        g_cuda_device_ord == device_ord &&
        g_cuda_precision && strcmp(g_cuda_precision, prec) == 0) {
        *out_was_cached = 1;
        return g_cuda_ctx;
    }
    if (g_cuda_ctx) {
        cuda_sam3_destroy(g_cuda_ctx);
        g_cuda_ctx = NULL;
    }
    free(g_cuda_ckpt);
    g_cuda_ckpt = NULL;
    free(g_cuda_precision);
    g_cuda_precision = NULL;

    cuda_sam3_config cfg = {
        .ckpt_path      = ckpt_path,
        .image_size     = 1008,
        .device_ordinal = device_ord,
        .verbose        = 1,
        .precision      = prec,
    };
    double t0 = cuda_now_ms();
    fprintf(stderr, "sam3-cuda: NVRTC compile + weight upload to device %d (one-time, will be cached) ...\n",
            device_ord);
    cuda_sam3_ctx *ctx = cuda_sam3_create(&cfg);
    if (!ctx) {
        snprintf(err_buf, err_cap, "cuda_sam3_create failed (ckpt=%s, device=%d)",
                 ckpt_path, device_ord);
        return NULL;
    }
    fprintf(stderr, "sam3-cuda: ctx ready in %.0f ms (cached)\n", cuda_now_ms() - t0);
    g_cuda_ctx        = ctx;
    g_cuda_ckpt       = strdup(ckpt_path);
    g_cuda_device_ord = device_ord;
    g_cuda_precision  = strdup(prec);
    return ctx;
}

/* PNG-to-memory writer (stb_image_write). Implementation lives in server.c. */
typedef struct { uint8_t *data; int len; int cap; } cu_mbuf;
static void cu_mbuf_write(void *ctx, void *d, int n) {
    cu_mbuf *b = (cu_mbuf *)ctx;
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

int server_sam3_cuda_segment(const char *ckpt_path,
                              const char *vocab_path,
                              const char *merges_path,
                              const uint8_t *img_bytes, size_t img_len,
                              const char *phrase,
                              float score_thr, float mask_thr,
                              int device_ordinal,
                              const char *precision,
                              server_sam3_mask *out_masks, int out_cap,
                              int *out_n,
                              char *err_buf, size_t err_cap);

int server_sam3_cuda_segment(const char *ckpt_path,
                              const char *vocab_path,
                              const char *merges_path,
                              const uint8_t *img_bytes, size_t img_len,
                              const char *phrase,
                              float score_thr, float mask_thr,
                              int device_ordinal,
                              const char *precision,
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
    /* Validate file paths upfront so an invalid ckpt/vocab/merges fails
     * fast instead of first spending 15-30 s on NVRTC compile inside
     * cuda_sam3_create. Skipped when the ctx is already cached. */
    const char *prec_eff = (precision && *precision) ? precision : "fp16";
    int _cached_match = (g_cuda_ctx && g_cuda_ckpt &&
                         strcmp(g_cuda_ckpt, ckpt_path) == 0 &&
                         g_cuda_device_ord == device_ordinal &&
                         g_cuda_precision && strcmp(g_cuda_precision, prec_eff) == 0);
    if (!_cached_match && access(ckpt_path, R_OK) != 0) {
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

    int W, H;
    unsigned char *rgb = server_decode_image_rgb(img_bytes, img_len, &W, &H);
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
    int32_t ids[32] = {0}, mask_ids[32] = {0};
    int nv = sam3_clip_bpe_encode(tok, phrase, 32, ids, mask_ids);
    sam3_clip_bpe_free(tok);
    if (nv < 0) {
        stbi_image_free(rgb);
        snprintf(err_buf, err_cap, "sam3 BPE encode failed for phrase '%s'", phrase);
        return 4;
    }

    int was_cached = 0;
    cuda_sam3_ctx *ctx = cuda_get_or_create_ctx(ckpt_path, device_ordinal, prec_eff,
                                                 err_buf, err_cap, &was_cached);
    if (!ctx) {
        stbi_image_free(rgb);
        return 5;
    }
    if (was_cached) fprintf(stderr, "sam3-cuda: reusing cached ctx\n");

    int rc = 0;
    double t_overall = cuda_now_ms();
    #define STEP(call, code, msg) do { \
        double _t0 = cuda_now_ms(); \
        fprintf(stderr, "sam3-cuda: %-16s ...\n", (msg)); fflush(stderr); \
        if ((call)) { rc = (code); \
            snprintf(err_buf, err_cap, "sam3-cuda: %s", (msg)); goto done; } \
        fprintf(stderr, "sam3-cuda: %-16s done in %.0f ms\n", (msg), cuda_now_ms() - _t0); \
        fflush(stderr); \
    } while (0)
    STEP(cuda_sam3_set_image(ctx, rgb, H, W),                          10, "set_image");
    STEP(cuda_sam3_run_vit(ctx, 31),                                   11, "run_vit (32 blks)");
    STEP(cuda_sam3_run_fpn(ctx),                                       12, "run_fpn");
    STEP(cuda_sam3_set_input_ids(ctx, ids, mask_ids),                  13, "set_input_ids");
    STEP(cuda_sam3_run_text(ctx),                                      14, "run_text");
    STEP(cuda_sam3_run_detr_enc(ctx),                                  15, "run_detr_enc");
    STEP(cuda_sam3_run_detr_dec(ctx),                                  16, "run_detr_dec");
    STEP(cuda_sam3_run_dot_score(ctx),                                 17, "run_dot_score");
    STEP(cuda_sam3_run_mask_dec(ctx),                                  18, "run_mask_dec");
    STEP(cuda_sam3_run_postprocess(ctx, H, W, score_thr, mask_thr),    19, "run_postprocess");
    #undef STEP
    fprintf(stderr, "sam3-cuda: pipeline total %.0f ms\n", cuda_now_ms() - t_overall);

    int nk = 0, oh = 0, ow = 0;
    const float   *scores = cuda_sam3_get_final_scores(ctx, &nk);
    const float   *boxes  = cuda_sam3_get_final_boxes(ctx, &nk);
    const uint8_t *masks  = cuda_sam3_get_final_masks(ctx, &nk, &oh, &ow);
    int emit = nk < out_cap ? nk : out_cap;
    for (int i = 0; i < emit; i++) {
        const uint8_t *m = masks + (size_t)i * (size_t)oh * (size_t)ow;
        uint8_t *g = (uint8_t *)malloc((size_t)oh * (size_t)ow);
        if (!g) { rc = 30; snprintf(err_buf, err_cap, "oom building mask"); goto done; }
        for (int p = 0; p < oh * ow; p++) g[p] = m[p] ? 255 : 0;
        cu_mbuf mb = {0};
        int ok = stbi_write_png_to_func(cu_mbuf_write, &mb, ow, oh, 1, g, ow);
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
