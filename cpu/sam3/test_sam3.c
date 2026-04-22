/* CLI entry for SAM 3 CPU runner — end-to-end text-prompted segmentation.
 *
 * Usage: test_sam3 <ckpt> <image> --phrase "cat" [-o mask.npy]
 *                  [--vocab vocab.json] [--merges merges.txt]
 *                  [--score-thr 0.3] [--mask-thr 0.5] [-t N]
 */
#include "sam3_runner.h"
#include "sam3_clip_bpe.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void write_npy_u8(const char *path, const uint8_t *data,
                          int n, int h, int w) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    char hdr[256];
    int L = snprintf(hdr, sizeof(hdr),
        "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d, %d), }",
        n, h, w);
    while ((L + 10) % 16 != 0) hdr[L++] = ' ';
    hdr[L++] = '\n';
    uint8_t magic[10] = { 0x93, 'N','U','M','P','Y', 1, 0, 0, 0 };
    uint16_t hl = (uint16_t)L;
    memcpy(magic + 8, &hl, 2);
    fwrite(magic, 1, 10, f);
    fwrite(hdr, 1, L, f);
    fwrite(data, 1, (size_t)n * h * w, f);
    fclose(f);
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s <ckpt> <image> --phrase \"cat\" [-o mask.npy]\n"
        "       [--vocab /mnt/disk1/models/clip-bpe/vocab.json]\n"
        "       [--merges /mnt/disk1/models/clip-bpe/merges.txt]\n"
        "       [--score-thr 0.3] [--mask-thr 0.5] [-t N]\n", p);
}

int main(int argc, char **argv) {
    if (argc < 5) { usage(argv[0]); return 1; }
    const char *ckpt = argv[1];
    const char *img_path = argv[2];
    const char *phrase = NULL;
    const char *out_path = "mask.npy";
    const char *vocab = "/mnt/disk1/models/clip-bpe/vocab.json";
    const char *merges = "/mnt/disk1/models/clip-bpe/merges.txt";
    float score_thr = 0.3f, mask_thr = 0.5f;
    int threads = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--phrase") && i+1 < argc) phrase = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--vocab") && i+1 < argc) vocab = argv[++i];
        else if (!strcmp(argv[i], "--merges") && i+1 < argc) merges = argv[++i];
        else if (!strcmp(argv[i], "--score-thr") && i+1 < argc) score_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--mask-thr") && i+1 < argc) mask_thr  = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) threads = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (!phrase) { fprintf(stderr, "--phrase required\n"); return 1; }

    int H, W, C;
    unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) { fprintf(stderr, "failed to load %s\n", img_path); return 2; }

    sam3_clip_bpe *tok = sam3_clip_bpe_load(vocab, merges);
    if (!tok) { fprintf(stderr, "tokenizer load failed\n"); return 3; }
    int32_t ids[32], mask[32];
    int nv = sam3_clip_bpe_encode(tok, phrase, 32, ids, mask);
    fprintf(stderr, "tokenized '%s' -> %d tokens\n", phrase, nv);
    sam3_clip_bpe_free(tok);

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008, .num_threads = threads };
    struct timespec _t0, _t1;
    clock_gettime(CLOCK_MONOTONIC, &_t0);
    sam3_ctx *ctx = sam3_create(&cfg);
    clock_gettime(CLOCK_MONOTONIC, &_t1);
    fprintf(stderr, "[timing] sam3_create (weight load+dequant): %.2f s\n",
            (_t1.tv_sec - _t0.tv_sec) + (_t1.tv_nsec - _t0.tv_nsec) / 1e9);
    if (!ctx) return 4;
    clock_gettime(CLOCK_MONOTONIC, &_t0);

    #define TIME_STAGE(label, call) do { \
        struct timespec _s0, _s1; \
        clock_gettime(CLOCK_MONOTONIC, &_s0); \
        if ((call)) return 99; \
        clock_gettime(CLOCK_MONOTONIC, &_s1); \
        fprintf(stderr, "[stage] %-20s %.2f s\n", (label), \
            (_s1.tv_sec - _s0.tv_sec) + (_s1.tv_nsec - _s0.tv_nsec) / 1e9); \
    } while (0)
    TIME_STAGE("set_image",       sam3_set_image(ctx, rgb, H, W));
    TIME_STAGE("run_vit (32 blk)", sam3_run_vit(ctx, 31));
    TIME_STAGE("run_fpn",          sam3_run_fpn(ctx));
    TIME_STAGE("set_input_ids",    sam3_set_input_ids(ctx, ids, mask));
    TIME_STAGE("run_text",         sam3_run_text(ctx));
    TIME_STAGE("run_detr_enc",     sam3_run_detr_enc(ctx));
    TIME_STAGE("run_detr_dec",     sam3_run_detr_dec(ctx));
    TIME_STAGE("run_dot_score",    sam3_run_dot_score(ctx));
    TIME_STAGE("run_mask_dec",     sam3_run_mask_dec(ctx));
    TIME_STAGE("run_postprocess",  sam3_run_postprocess(ctx, H, W, score_thr, mask_thr));
    clock_gettime(CLOCK_MONOTONIC, &_t1);
    fprintf(stderr, "[timing] inference total: %.2f s\n",
            (_t1.tv_sec - _t0.tv_sec) + (_t1.tv_nsec - _t0.tv_nsec) / 1e9);

    int nk, oh, ow;
    const float   *scores = sam3_get_final_scores(ctx, &nk);
    const float   *boxes  = sam3_get_final_boxes(ctx, &nk);
    const uint8_t *masks  = sam3_get_final_masks(ctx, &nk, &oh, &ow);
    fprintf(stderr, "kept %d masks @ %dx%d\n", nk, oh, ow);
    for (int i = 0; i < nk && i < 10; i++) {
        fprintf(stderr, "  [%d] score=%.4f box=(%.1f, %.1f, %.1f, %.1f)\n",
                i, scores[i], boxes[i*4], boxes[i*4+1],
                boxes[i*4+2], boxes[i*4+3]);
    }
    if (nk > 0 && masks) {
        write_npy_u8(out_path, masks, nk, oh, ow);
        fprintf(stderr, "wrote %s (%d, %d, %d)\n", out_path, nk, oh, ow);
    }

    sam3_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
