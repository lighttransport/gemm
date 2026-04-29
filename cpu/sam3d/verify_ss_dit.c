/*
 * verify_ss_dit — Sparse-Structure Flow DiT numerics check.
 *
 * Smoke-tests the weight loader; if --refdir is given AND it contains
 * ss_dit_in_{shape,6drotation_normalized,translation,scale,translation_scale}.npy
 * + ss_dit_cond.npy + ss_dit_t.npy (+ optional ss_dit_d.npy), runs one
 * forward pass and diffs each output modality against ss_dit_out_*.npy.
 *
 * Generate the ref via:
 *   python ref/sam3d/dump_ss_dit_io.py --image ... --mask ...
 *      --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml
 *      --pointmap /tmp/sam3d_ref/pointmap.npy --outdir /tmp/sam3d_ref
 */
#include "sam3d_runner.h"
#include "sam3d_ss_flow_dit.h"
#include "verify_npy.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *load_modality(const char *refdir, const char *name, int *out_n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/ss_dit_%s.npy", refdir, name);
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    void *d = verify_read_npy(path, &nd, dims, &is_f32);
    if (!d) {
        fprintf(stderr, "[verify_ss_dit] missing %s\n", path);
        return NULL;
    }
    if (!is_f32) {
        fprintf(stderr, "[verify_ss_dit] %s is not float32\n", path);
        free(d);
        return NULL;
    }
    int n = 1;
    for (int i = 0; i < nd; i++) n *= dims[i];
    if (out_n) *out_n = n;
    return (float *)d;
}

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *refdir = NULL, *sft_dir = NULL;
    int verbose = 0, n_threads = 1;
    float threshold = 5e-2f;  /* bf16 inference-path floor */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ckpt")    && i+1 < argc) ckpt    = argv[++i];
        else if (!strcmp(argv[i], "--refdir")  && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc)       n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))                     verbose = 1;
        else if (!strcmp(argv[i], "--use-ref-inputs"))       { /* implied */ }
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!ckpt) {
        fprintf(stderr, "Usage: %s --ckpt <pipeline.yaml> [--safetensors-dir <dir>] "
                        "[--refdir <dir>] [--threshold F] [-t N] [-v]\n", argv[0]);
        return 2;
    }

    sam3d_config cfg = {
        .pipeline_yaml = ckpt,
        .safetensors_dir = sft_dir,
        .seed = 42,
        .verbose = verbose,
    };
    sam3d_ctx *ctx = sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_create failed\n"); return 5; }

    char path[1280];
    snprintf(path, sizeof(path), "%s/sam3d_ss_dit.safetensors",
             sam3d_safetensors_dir(ctx));
    fprintf(stderr, "[verify_ss_dit] loading %s\n", path);

    sam3d_ss_flow_dit_model *m = sam3d_ss_flow_dit_load_safetensors(path);
    if (!m) {
        fprintf(stderr, "[verify_ss_dit] loader failed\n");
        sam3d_destroy(ctx);
        return 3;
    }

    fprintf(stderr, "[verify_ss_dit] OK: n_blocks=%d dim=%d heads=%d head_dim=%d "
                    "cond=%d shortcut=%s\n",
            m->n_blocks, m->dim, m->n_heads, m->head_dim, m->cond_channels,
            m->is_shortcut ? "yes" : "no");

    if (!refdir) {
        fprintf(stderr, "[verify_ss_dit] no --refdir; loader smoke-test only\n");
        sam3d_ss_flow_dit_free(m);
        sam3d_destroy(ctx);
        return 0;
    }

    /* Load ref inputs per modality. */
    static const char *lat_names[SAM3D_SS_DIT_N_LATENTS] = {
        "shape", "6drotation_normalized", "translation", "scale", "translation_scale"
    };
    float *inputs[SAM3D_SS_DIT_N_LATENTS]  = {0};
    float *outputs[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *refouts[SAM3D_SS_DIT_N_LATENTS] = {0};
    int    in_n[SAM3D_SS_DIT_N_LATENTS]    = {0};
    int    out_n[SAM3D_SS_DIT_N_LATENTS]   = {0};
    int rc = 0;

    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        char name[128];
        snprintf(name, sizeof(name), "in_%s", lat_names[i]);
        inputs[i] = load_modality(refdir, name, &in_n[i]);
        if (!inputs[i]) { rc = 4; goto cleanup; }

        snprintf(name, sizeof(name), "out_%s", lat_names[i]);
        refouts[i] = load_modality(refdir, name, &out_n[i]);
        if (!refouts[i]) { rc = 4; goto cleanup; }

        outputs[i] = (float *)calloc((size_t)out_n[i], sizeof(float));
    }

    /* cond, t, d */
    int cond_n = 0;
    float *cond = load_modality(refdir, "cond", &cond_n);
    if (!cond) { rc = 4; goto cleanup; }
    int n_cond_tokens = cond_n / m->cond_channels;

    int t_n = 0;
    float *t_buf = load_modality(refdir, "t", &t_n);
    if (!t_buf) { rc = 4; free(cond); goto cleanup; }
    float t = t_buf[0];
    free(t_buf);

    float d = 0.0f;
    if (m->is_shortcut) {
        int d_n = 0;
        float *d_buf = load_modality(refdir, "d", &d_n);
        if (d_buf) { d = d_buf[0]; free(d_buf); }
    }

    fprintf(stderr, "[verify_ss_dit] running forward: t=%g d=%g n_cond=%d\n",
            (double)t, (double)d, n_cond_tokens);

    if (sam3d_ss_flow_dit_forward(m,
                                  (const float *const *)inputs,
                                  outputs,
                                  cond, n_cond_tokens,
                                  t, d, n_threads) != 0) {
        fprintf(stderr, "[verify_ss_dit] forward failed\n");
        free(cond);
        rc = 5;
        goto cleanup;
    }
    free(cond);

    int fail = 0;
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        double mean = 0.0;
        float mx = verify_max_abs(outputs[i], refouts[i], out_n[i], &mean);
        int ok = (mx < threshold) && (mean < threshold * 1e-2);
        fprintf(stderr, "[verify_ss_dit] %-22s  max_abs=%.6e  mean_abs=%.6e  %s\n",
                lat_names[i], (double)mx, mean, ok ? "OK" : "FAIL");
        if (!ok) fail = 1;
    }
    if (fail) rc = 1;

cleanup:
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(inputs[i]); free(outputs[i]); free(refouts[i]);
    }
    sam3d_ss_flow_dit_free(m);
    sam3d_destroy(ctx);
    return rc;
}
