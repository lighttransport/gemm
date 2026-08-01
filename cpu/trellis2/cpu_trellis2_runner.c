/*
 * cpu_trellis2_runner.c - implementation of the thin CPU TRELLIS.2 runner.
 *
 * Carries the same IMPLEMENTATION macros as test_trellis2.c (minus STB_IMAGE:
 * the runner receives raw RGB bytes, it never decodes an image file), then
 * exposes an opaque-handle API that loads weights once and reuses them.
 *
 * Pipeline mirrors test_trellis2.c mode_full():
 *   DINOv3 encode -> Stage-1 flow sampling -> structure decode -> occupancy.
 */

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define DINOV3_IMPLEMENTATION
#include "../../common/dinov3.h"

#define T2DIT_IMPLEMENTATION
#include "../../common/trellis2_dit.h"

#define T2_STAGE1_IMPLEMENTATION
#include "../../common/trellis2_stage1.h"

#define T2_SS_DEC_IMPLEMENTATION
#include "../../common/trellis2_ss_decoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cpu_trellis2_runner.h"

struct cpu_trellis2_runner {
    dinov3_model *dm;
    t2_stage1    *s;
    t2_ss_dec    *dec;
    int           n_threads;
    int           verbose;
};

cpu_trellis2_runner *cpu_trellis2_init(int n_threads, int verbose) {
    cpu_trellis2_runner *r = (cpu_trellis2_runner *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->n_threads = (n_threads > 0) ? n_threads : 4;
    r->verbose = verbose;
    return r;
}

int cpu_trellis2_load_weights(cpu_trellis2_runner *r,
                              const char *dinov3_path,
                              const char *stage1_path,
                              const char *decoder_path) {
    if (!r) return 1;
    if (r->verbose)
        fprintf(stderr, "[cpu_trellis2] loading DINOv3 %s\n", dinov3_path);
    r->dm = dinov3_load_safetensors(dinov3_path);
    if (!r->dm) { fprintf(stderr, "[cpu_trellis2] DINOv3 load failed\n"); return 1; }

    if (r->verbose)
        fprintf(stderr, "[cpu_trellis2] loading Stage1 %s\n", stage1_path);
    r->s = t2_stage1_load(stage1_path);
    if (!r->s) { fprintf(stderr, "[cpu_trellis2] Stage1 load failed\n"); return 1; }

    if (r->verbose)
        fprintf(stderr, "[cpu_trellis2] loading decoder %s\n", decoder_path);
    r->dec = t2_ss_dec_load(decoder_path);
    if (!r->dec) { fprintf(stderr, "[cpu_trellis2] decoder load failed\n"); return 1; }

    return 0;
}

float *cpu_trellis2_predict(cpu_trellis2_runner *r,
                            const uint8_t *rgb, int w, int h,
                            uint32_t seed) {
    if (!r || !r->dm || !r->s || !r->dec || !rgb) return NULL;

    /* Step 1: DINOv3 encode */
    dinov3_result dr = dinov3_encode(r->dm, rgb, w, h, r->n_threads);
    if (!dr.features) {
        fprintf(stderr, "[cpu_trellis2] DINOv3 encode failed\n");
        return NULL;
    }

    /* Step 2: Stage-1 flow sampling -> latent [n_tokens * in_channels] */
    float *latent = t2_stage1_sample(r->s, dr.features, dr.n_tokens,
                                     r->n_threads, (uint64_t)seed);
    dinov3_result_free(&dr);
    if (!latent) {
        fprintf(stderr, "[cpu_trellis2] Stage1 sampling failed\n");
        return NULL;
    }

    /* Step 3: structure decode -> occupancy [64*64*64] (malloc'd) */
    float *occupancy = t2_ss_dec_forward(r->dec, latent, r->n_threads);
    free(latent);
    if (!occupancy) {
        fprintf(stderr, "[cpu_trellis2] structure decode failed\n");
        return NULL;
    }
    return occupancy;
}

void cpu_trellis2_free_buffer(void *p) {
    free(p);
}

void cpu_trellis2_free(cpu_trellis2_runner *r) {
    if (!r) return;
    if (r->dec) t2_ss_dec_free(r->dec);
    if (r->s)   t2_stage1_free(r->s);
    if (r->dm)  dinov3_free(r->dm);
    free(r);
}
