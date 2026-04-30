/*
 * verify_ss_dit (CUDA) — diff a single SS Flow DiT forward call against
 * /tmp/sam3d_ref/ss_dit_out_*.npy. Inputs (5 modalities + cond + t + d)
 * are loaded directly from refs and passed via
 * hip_sam3d_debug_ss_dit_forward, isolating per-call DiT drift from
 * upstream encoder/fuser drift.
 *
 * Usage:
 *   verify_ss_dit --safetensors-dir DIR --refdir /tmp/sam3d_ref
 *                 [--threshold F] [-v]
 */

#include "hip_sam3d_runner.h"
#include "../../common/npy_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *load_modality_f32(const char *refdir, const char *fragment,
                                int *out_n)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/ss_dit_%s.npy", refdir, fragment);
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    void *d = npy_load(path, &nd, dims, &is_f32);
    if (!d) {
        fprintf(stderr, "[verify_ss_dit.cuda] missing %s\n", path);
        return NULL;
    }
    if (!is_f32) {
        fprintf(stderr, "[verify_ss_dit.cuda] %s is not float32\n", path);
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
    const char *sft_dir = NULL, *yaml = NULL, *refdir = NULL;
    int verbose = 0;
    float threshold = 5e-2f;  /* bf16 inference-path floor */

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir   = argv[++i];
        else if (!strcmp(a, "--pipeline-yaml")   && i+1 < argc) yaml      = argv[++i];
        else if (!strcmp(a, "--refdir")          && i+1 < argc) refdir    = argv[++i];
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                              verbose   = 1;
        else if (!strcmp(a, "--use-ref-inputs"))                { /* implied */ }
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if ((!sft_dir && !yaml) || !refdir) {
        fprintf(stderr,
                "Usage: %s (--safetensors-dir DIR | --pipeline-yaml YAML) "
                "--refdir DIR [--threshold F] [-v]\n", argv[0]);
        return 2;
    }

    hip_sam3d_config cfg = {0};
    cfg.safetensors_dir = sft_dir;
    cfg.pipeline_yaml   = yaml;
    cfg.verbose         = verbose;
    cfg.precision       = "fp16";
    cfg.seed            = 42;
    hip_sam3d_ctx *ctx = hip_sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "hip_sam3d_create failed\n"); return 5; }

    int n_blocks = 0, dim = 0, cond_ch = 0, is_shortcut = 0;
    if (hip_sam3d_ss_dit_info(ctx, &n_blocks, &dim, &cond_ch, &is_shortcut) != 0) {
        fprintf(stderr, "hip_sam3d_ss_dit_info failed\n");
        hip_sam3d_destroy(ctx); return 3;
    }
    fprintf(stderr, "[verify_ss_dit.cuda] OK: n_blocks=%d dim=%d cond=%d shortcut=%s\n",
            n_blocks, dim, cond_ch, is_shortcut ? "yes" : "no");

    static const char *names[5] = {
        "shape", "6drotation_normalized", "translation", "scale", "translation_scale"
    };
    int n_lat = hip_sam3d_ss_dit_n_latents();
    if (n_lat != 5) {
        fprintf(stderr, "unexpected n_lat=%d\n", n_lat);
        hip_sam3d_destroy(ctx); return 3;
    }

    float *inputs [5] = {0};
    float *outputs[5] = {0};
    float *refouts[5] = {0};
    int    in_n   [5] = {0};
    int    out_n  [5] = {0};
    int rc = 0;

    for (int i = 0; i < n_lat; i++) {
        char frag[64];
        snprintf(frag, sizeof(frag), "in_%s",  names[i]);
        inputs[i]  = load_modality_f32(refdir, frag, &in_n[i]);
        snprintf(frag, sizeof(frag), "out_%s", names[i]);
        refouts[i] = load_modality_f32(refdir, frag, &out_n[i]);
        if (!inputs[i] || !refouts[i]) { rc = 4; goto cleanup; }
        outputs[i] = (float *)calloc((size_t)out_n[i], sizeof(float));
        if (!outputs[i]) { rc = 5; goto cleanup; }
        int expect = hip_sam3d_ss_dit_lat_elts(i);
        if (in_n[i] != expect || out_n[i] != expect) {
            fprintf(stderr, "[verify_ss_dit.cuda] %s: size mismatch in=%d out=%d expect=%d\n",
                    names[i], in_n[i], out_n[i], expect);
            rc = 4; goto cleanup;
        }
    }

    int cond_n = 0;
    float *cond = load_modality_f32(refdir, "cond", &cond_n);
    if (!cond) { rc = 4; goto cleanup; }
    int n_cond_tokens = cond_n / cond_ch;
    if (n_cond_tokens * cond_ch != cond_n) {
        fprintf(stderr, "[verify_ss_dit.cuda] cond size %d not multiple of channels %d\n",
                cond_n, cond_ch);
        free(cond); rc = 4; goto cleanup;
    }

    int t_n = 0;
    float *t_buf = load_modality_f32(refdir, "t", &t_n);
    if (!t_buf) { free(cond); rc = 4; goto cleanup; }
    float t = t_buf[0];
    free(t_buf);

    float d = 0.0f;
    if (is_shortcut) {
        int d_n = 0;
        float *d_buf = load_modality_f32(refdir, "d", &d_n);
        if (d_buf) { d = d_buf[0]; free(d_buf); }
    }

    fprintf(stderr, "[verify_ss_dit.cuda] forward: t=%g d=%g n_cond=%d\n",
            (double)t, (double)d, n_cond_tokens);

    if (hip_sam3d_debug_ss_dit_forward(ctx,
                                        (const float *const *)inputs,
                                        outputs,
                                        cond, n_cond_tokens,
                                        t, d) != 0) {
        fprintf(stderr, "hip_sam3d_debug_ss_dit_forward failed\n");
        free(cond); rc = 6; goto cleanup;
    }
    free(cond);

    int fail = 0;
    for (int i = 0; i < n_lat; i++) {
        double mean_abs = 0.0;
        float mx = npy_max_abs_f32(outputs[i], refouts[i], out_n[i], &mean_abs);
        int ok = (mx < threshold) && (mean_abs < threshold * 1e-2);
        fprintf(stderr, "[verify_ss_dit.cuda] %-22s  max_abs=%.6e  mean_abs=%.6e  %s\n",
                names[i], (double)mx, mean_abs, ok ? "OK" : "FAIL");
        if (!ok) fail = 1;
    }
    if (fail) rc = 1;

cleanup:
    for (int i = 0; i < 5; i++) {
        free(inputs[i]); free(outputs[i]); free(refouts[i]);
    }
    hip_sam3d_destroy(ctx);
    return rc;
}
