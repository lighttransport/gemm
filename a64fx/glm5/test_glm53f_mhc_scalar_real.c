/* Compare every real-weight mHC site with the independent scalar oracle. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "glm53f_mhc_sve.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LAYERS = 45 };

static void *a256(size_t bytes) {
    void *p = NULL;
    return posix_memalign(&p, 256, bytes) ? NULL : p;
}

static void *load_exact(glm53f_st_context *st, const char *name, size_t bytes) {
    const st_tensor_info *t = glm53f_st_find(st, name, NULL);
    void *p = a256(bytes);
    if (!t || t->nbytes != bytes || !p || glm53f_st_read(st, name, 0, p, bytes)) {
        fprintf(stderr, "load failed: %s expected=%zu got=%zu\n", name, bytes,
                t ? t->nbytes : 0);
        free(p);
        return NULL;
    }
    return p;
}

static void error_stats(const float *got, const float *ref, size_t n,
                        double *rel_l2, double *max_abs) {
    double d2 = 0.0, r2 = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)got[i] - ref[i];
        d2 += d * d;
        r2 += (double)ref[i] * ref[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    *rel_l2 = sqrt(d2 / (r2 + 1e-30));
    *max_abs = mx;
}

int main(int argc, char **argv) {
    glm53f_st_context *st;
    float *input, *sublayer, *scalar_streams, *sve_streams;
    float *scalar_collapsed, *scalar_post, *scalar_combine, *scalar_norm;
    glm53f_mhc_scratch *sve;
    int failed = 0;
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    st = glm53f_st_open(argv[1]);
    input = a256((size_t)GLM53F_MHC_FLAT * sizeof(float));
    sublayer = a256((size_t)GLM53F_MHC_WIDTH * sizeof(float));
    scalar_streams = a256((size_t)GLM53F_MHC_FLAT * sizeof(float));
    sve_streams = a256((size_t)GLM53F_MHC_FLAT * sizeof(float));
    scalar_collapsed = a256((size_t)GLM53F_MHC_WIDTH * sizeof(float));
    scalar_post = a256(GLM53F_MHC_STREAMS * sizeof(float));
    scalar_combine = a256(GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS * sizeof(float));
    scalar_norm = a256((size_t)GLM53F_MHC_WIDTH * sizeof(float));
    sve = a256(sizeof(*sve));
    if (!st || !input || !sublayer || !scalar_streams || !sve_streams ||
        !scalar_collapsed || !scalar_post || !scalar_combine || !scalar_norm || !sve)
        return 2;
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
        input[i] = (float)(((i * 29 + 7) % 257) - 128) / 128.0f;
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i)
        sublayer[i] = (float)(((i * 17 + 3) % 251) - 125) / 1000.0f;

    for (int layer = 0; layer < LAYERS; ++layer) {
        for (int site_index = 0; site_index < 2; ++site_index) {
            const char *site_name = site_index ? "ffn" : "attn";
            const char *norm_suffix = site_index ? "post_attention_layernorm.weight" :
                                                   "input_layernorm.weight";
            char name[256];
            snprintf(name, sizeof(name), "model.language_model.layers.%d.hc_%s_fn",
                     layer, site_name);
            uint16_t *fn = load_exact(st, name,
                (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * sizeof(uint16_t));
            snprintf(name, sizeof(name), "model.language_model.layers.%d.hc_%s_base",
                     layer, site_name);
            float *base = load_exact(st, name, GLM53F_MHC_MIX * sizeof(float));
            snprintf(name, sizeof(name), "model.language_model.layers.%d.hc_%s_scale",
                     layer, site_name);
            float *scale = load_exact(st, name, 3 * sizeof(float));
            snprintf(name, sizeof(name), "model.language_model.layers.%d.%s",
                     layer, norm_suffix);
            uint16_t *norm = load_exact(st, name,
                GLM53F_MHC_WIDTH * sizeof(uint16_t));
            if (!fn || !base || !scale || !norm) return 2;

            glm53f_mhc_site site = {fn, base, scale};
            memcpy(scalar_streams, input, (size_t)GLM53F_MHC_FLAT * sizeof(float));
            memcpy(sve_streams, input, (size_t)GLM53F_MHC_FLAT * sizeof(float));
            glm53f_mhc_pre(scalar_collapsed, scalar_post, scalar_combine,
                scalar_streams, fn, base, scale, GLM53F_MHC_STREAMS,
                GLM53F_MHC_WIDTH, 20, 1e-5f, 1e-6f);
            glm53f_rmsnorm_bf16(scalar_norm, scalar_collapsed, norm,
                                GLM53F_MHC_WIDTH, 1e-5f);
            glm53f_mhc_pre_sve(sve, sve_streams, &site, norm);

            double pre_rel, pre_max, norm_rel, norm_max, post_rel, post_max;
            error_stats(sve->collapsed, scalar_collapsed, GLM53F_MHC_WIDTH,
                        &pre_rel, &pre_max);
            error_stats(sve->normalized, scalar_norm, GLM53F_MHC_WIDTH,
                        &norm_rel, &norm_max);
            glm53f_mhc_post(scalar_streams, input, sublayer, scalar_post,
                            scalar_combine, GLM53F_MHC_STREAMS, GLM53F_MHC_WIDTH);
            glm53f_mhc_post_sve(sve_streams, sublayer, sve);
            error_stats(sve_streams, scalar_streams, GLM53F_MHC_FLAT,
                        &post_rel, &post_max);
            int finite = 1;
            for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
                finite &= isfinite(sve_streams[i]);
            int ok = finite && pre_rel < 2e-5 && norm_rel < 2e-5 && post_rel < 2e-5;
            printf("GLM53F_MHC_SCALAR layer=%d site=%s collapsed_rel_l2=%.9g "
                   "collapsed_max_abs=%.9g norm_rel_l2=%.9g norm_max_abs=%.9g "
                   "post_rel_l2=%.9g post_max_abs=%.9g finite=%s %s\n",
                   layer, site_name, pre_rel, pre_max, norm_rel, norm_max,
                   post_rel, post_max, finite ? "YES" : "NO", ok ? "PASS" : "FAIL");
            failed |= !ok;
            free(norm); free(scale); free(base); free(fn);
        }
    }
    glm53f_st_close(st);
    free(sve); free(scalar_norm); free(scalar_combine); free(scalar_post);
    free(scalar_collapsed); free(sve_streams); free(scalar_streams);
    free(sublayer); free(input);
    return failed ? 1 : 0;
}
