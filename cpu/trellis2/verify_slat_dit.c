/*
 * verify_slat_dit — TRELLIS.2 SLAT (Sparse-Latent) DiT CPU verify harness.
 *
 * Loads dumps produced by the ROCm/PyTorch SLAT DiT step and compares the
 * CPU forward against the reference velocity output.
 *
 * Args:
 *   verify_slat_dit <safetensors> <dump_dir>
 *
 * Reads from <dump_dir>:
 *   06b_slat_dit_step_x_t.npy       [N, 32]    f32
 *   06b_slat_dit_step_coords.npy    [N, 4]     i32
 *   06b_slat_dit_step_t.npy         [1]        f32
 *   06b_slat_dit_step_cond.npy      [1, M, C]  f32 (drops the leading batch)
 *   06b_slat_dit_step_velocity.npy  [N, 32]    f32 (reference)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define T2SLATDIT_IMPLEMENTATION
#include "../../common/trellis2_slat_dit.h"

#include "../../common/npy_io.h"

static void *load_npy_or_die(const char *dir, const char *name,
                             int *ndim, int *dims, int *is_f32) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    void *p = npy_load(path, ndim, dims, is_f32);
    if (!p) {
        fprintf(stderr, "[verify_slat_dit] missing/unreadable: %s\n", path);
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
    const char *dir = argv[2];

    int n_threads = 1;
    {
        const char *env = getenv("T2_THREADS");
        if (env) n_threads = atoi(env);
        if (n_threads <= 0) n_threads = 1;
    }

    fprintf(stderr, "[verify_slat_dit] safetensors: %s\n", st_path);
    fprintf(stderr, "[verify_slat_dit] dump_dir   : %s\n", dir);
    fprintf(stderr, "[verify_slat_dit] threads    : %d\n", n_threads);

    t2slatdit_model *m = t2slatdit_load_safetensors(st_path);
    if (!m) {
        fprintf(stderr, "[verify_slat_dit] load failed\n");
        return 3;
    }

    int nd, dims[8], is_f32;

    float *x_t = (float *)load_npy_or_die(dir, "06b_slat_dit_step_x_t.npy",
                                          &nd, dims, &is_f32);
    if (!is_f32 || nd != 2 || dims[1] != m->in_channels) {
        fprintf(stderr, "[verify_slat_dit] bad x_t shape (nd=%d dims=[%d,%d] f32=%d) "
                        "expected [N,%d]\n",
                nd, dims[0], dims[1], is_f32, m->in_channels);
        return 4;
    }
    int N = dims[0];

    int32_t *coords = (int32_t *)load_npy_or_die(dir,
        "06b_slat_dit_step_coords.npy", &nd, dims, &is_f32);
    if (is_f32 || nd != 2 || dims[0] != N || dims[1] != 4) {
        fprintf(stderr, "[verify_slat_dit] bad coords shape (nd=%d dims=[%d,%d] f32=%d) "
                        "expected [%d,4] i32\n", nd, dims[0], dims[1], is_f32, N);
        return 4;
    }

    float *t_arr = (float *)load_npy_or_die(dir, "06b_slat_dit_step_t.npy",
                                            &nd, dims, &is_f32);
    if (!is_f32) { fprintf(stderr, "[verify_slat_dit] bad t dtype\n"); return 4; }
    float t_val = t_arr[0];
    free(t_arr);

    float *cond_raw = (float *)load_npy_or_die(dir,
        "06b_slat_dit_step_cond.npy", &nd, dims, &is_f32);
    int n_cond, cond_C;
    const float *cond;
    if (nd == 3) {
        if (dims[0] != 1) {
            fprintf(stderr, "[verify_slat_dit] cond batch != 1 (got %d)\n", dims[0]);
            return 4;
        }
        n_cond = dims[1];
        cond_C = dims[2];
        cond = cond_raw; /* batch=1 leading is just a stride; data is contiguous. */
    } else if (nd == 2) {
        n_cond = dims[0];
        cond_C = dims[1];
        cond = cond_raw;
    } else {
        fprintf(stderr, "[verify_slat_dit] bad cond ndim=%d\n", nd);
        return 4;
    }
    if (cond_C != m->cond_dim) {
        fprintf(stderr, "[verify_slat_dit] cond_dim mismatch: file=%d model=%d\n",
                cond_C, m->cond_dim);
        return 4;
    }

    float *ref = (float *)load_npy_or_die(dir,
        "06b_slat_dit_step_velocity.npy", &nd, dims, &is_f32);
    if (!is_f32 || nd != 2 || dims[0] != N || dims[1] != m->in_channels) {
        fprintf(stderr, "[verify_slat_dit] bad velocity shape (nd=%d dims=[%d,%d] f32=%d)\n",
                nd, dims[0], dims[1], is_f32);
        return 4;
    }

    fprintf(stderr, "[verify_slat_dit] N=%d n_cond=%d cond_C=%d t=%.6f\n",
            N, n_cond, cond_C, t_val);

    float *out = (float *)malloc((size_t)N * m->in_channels * sizeof(float));
    t2slatdit_forward(out, x_t, coords, N, t_val, cond, n_cond, m, n_threads);

    /* Cosine + max abs */
    int total = N * m->in_channels;
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
    printf("[verify_slat_dit] N=%d t=%.6f cosine=%.6f  max_abs=%.6g  %s\n",
           N, t_val, cosine, (double)max_abs, pass ? "PASS" : "FAIL");

    free(out);
    free(x_t); free(coords); free(cond_raw); free(ref);
    t2slatdit_free(m);
    return pass ? 0 : 1;
}
