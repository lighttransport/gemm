/*
 * test_hip_tex_dit — TRELLIS.2 Tex DiT HIP/RDNA4 verify harness.
 *
 * Mirrors test_hip_slat_dit but for the stage-3 texture DiT (in_ch=64,
 * out_ch=32). Loads:
 *   <dump_dir>/10b_tex_dit_step_x_t.npy        [N, 64]    f32
 *   <dump_dir>/10b_tex_dit_step_coords.npy     [N, 4]     i32
 *   <dump_dir>/10b_tex_dit_step_t.npy          [1]        f32
 *   <dump_dir>/10b_tex_dit_step_cond.npy       [1, M, C]  f32
 *   <dump_dir>/10b_tex_dit_step_velocity.npy   [N, 32]    f32 (reference)
 *
 * Pass criterion: cosine >= 0.999 vs PyT-ROCm reference.
 *
 * Usage:
 *   test_hip_tex_dit <tex_dit.safetensors> <dump_dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "hip_trellis2_runner.h"
#include "../../common/npy_io.h"

static void *load_npy_or_die(const char *dir, const char *name,
                             int *ndim, int *dims, int *is_f32) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    void *p = npy_load(path, ndim, dims, is_f32);
    if (!p) {
        fprintf(stderr, "[test_hip_tex_dit] missing/unreadable: %s\n", path);
        exit(2);
    }
    return p;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <safetensors> <dump_dir>\n", argv[0]);
        return 2;
    }
    const char *st_path = argv[1];
    const char *dir     = argv[2];

    fprintf(stderr, "[test_hip_tex_dit] safetensors: %s\n", st_path);
    fprintf(stderr, "[test_hip_tex_dit] dump_dir   : %s\n", dir);

    hip_trellis2_runner *r = hip_trellis2_init(0, 1);
    if (!r) { fprintf(stderr, "[test_hip_tex_dit] runner init failed\n"); return 3; }

    if (hip_trellis2_load_tex_dit(r, st_path) != 0) {
        fprintf(stderr, "[test_hip_tex_dit] load failed\n");
        hip_trellis2_free(r);
        return 3;
    }

    int nd, dims[8], is_f32;

    float *x_t = (float *)load_npy_or_die(dir, "10b_tex_dit_step_x_t.npy",
                                          &nd, dims, &is_f32);
    if (!is_f32 || nd != 2 || dims[1] != 64) {
        fprintf(stderr, "[test_hip_tex_dit] bad x_t shape (expect [N,64])\n");
        return 4;
    }
    int N = dims[0];

    int32_t *coords = (int32_t *)load_npy_or_die(dir,
        "10b_tex_dit_step_coords.npy", &nd, dims, &is_f32);
    if (is_f32 || nd != 2 || dims[0] != N || dims[1] != 4) {
        fprintf(stderr, "[test_hip_tex_dit] bad coords shape\n");
        return 4;
    }

    float *t_arr = (float *)load_npy_or_die(dir, "10b_tex_dit_step_t.npy",
                                            &nd, dims, &is_f32);
    float t_val = t_arr[0];
    free(t_arr);

    float *cond_raw = (float *)load_npy_or_die(dir,
        "10b_tex_dit_step_cond.npy", &nd, dims, &is_f32);
    int n_cond, cond_C;
    const float *cond;
    if (nd == 3) {
        if (dims[0] != 1) {
            fprintf(stderr, "[test_hip_tex_dit] cond batch != 1 (got %d)\n", dims[0]);
            return 4;
        }
        n_cond = dims[1]; cond_C = dims[2]; cond = cond_raw;
    } else if (nd == 2) {
        n_cond = dims[0]; cond_C = dims[1]; cond = cond_raw;
    } else {
        fprintf(stderr, "[test_hip_tex_dit] bad cond ndim=%d\n", nd);
        return 4;
    }
    if (cond_C != 1024) {
        fprintf(stderr, "[test_hip_tex_dit] cond_dim mismatch: file=%d expect=1024\n", cond_C);
        return 4;
    }

    float *ref = (float *)load_npy_or_die(dir,
        "10b_tex_dit_step_velocity.npy", &nd, dims, &is_f32);
    if (!is_f32 || nd != 2 || dims[0] != N || dims[1] != 32) {
        fprintf(stderr, "[test_hip_tex_dit] bad velocity shape (expect [N,32])\n");
        return 4;
    }

    fprintf(stderr, "[test_hip_tex_dit] N=%d n_cond=%d t=%.6f\n", N, n_cond, t_val);

    float *out = (float *)malloc((size_t)N * 32 * sizeof(float));
    /* PyT dump bypasses sampler and passes raw t=0.5 to the model directly.
     * Our runner ×1000 internally (matches SS DiT and PyT sampler path),
     * so divide by 1000 here to reproduce the dump's effective t. */
    if (hip_trellis2_tex_dit_step(r, x_t, coords, N, t_val / 1000.0f,
                                   cond, n_cond, out) != 0) {
        fprintf(stderr, "[test_hip_tex_dit] step failed\n");
        return 5;
    }

    /* Cosine + max abs */
    int total = N * 32;
    double dot = 0.0, na = 0.0, nb = 0.0;
    float max_abs = 0.0f;
    for (int i = 0; i < total; i++) {
        double a = out[i], b = ref[i];
        dot += a * b;
        na  += a * a;
        nb  += b * b;
        float d = fabsf((float)(a - b));
        if (d > max_abs) max_abs = d;
    }
    double cosine = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    int pass = (cosine >= 0.999);
    printf("[test_hip_tex_dit] N=%d t=%.6f cosine=%.6f max_abs=%.6g %s\n",
           N, t_val, cosine, (double)max_abs, pass ? "PASS" : "FAIL");

    free(out); free(x_t); free(coords); free(cond_raw); free(ref);
    hip_trellis2_free(r);
    return pass ? 0 : 1;
}
