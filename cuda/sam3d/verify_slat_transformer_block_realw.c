/*
 * verify_slat_transformer_block_realw - Phase 5b.12..5b.15 real-weight verifier.
 *
 * Runs checkpoint SLAT transformer block(s) on CUDA, starting from traced
 * post-APE activations. Default verifies one block; --stack verifies blocks
 * 0..--block and compares against the traced CPU output for that block.
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

#include "../../common/sam3d_slat_dit.h"
#include "cuda_sam3d_slat_dit_gpu.h"
#include "cuda_sam3d_slat_dit_forward.h"
#include "../../common/npy_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static float max_abs_f32(const float *a, const float *b, size_t n, double *mean_out)
{
    double sum = 0.0;
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n ? n : 1);
    return mx;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--safetensors-dir DIR] [--refdir DIR] [--block I] "
            "[--stack] [--threshold F] [--repeat N] [-v]\n", argv0);
}

int main(int argc, char **argv)
{
    const char *safetensors_dir = "/mnt/disk01/models/sam3d/safetensors";
    const char *refdir = "/tmp/sam3d_ref";
    int block_idx = 0;
    int stack = 0;
    int repeat = 3;
    float threshold = 2e-3f;
    int threshold_set = 0;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i + 1 < argc) safetensors_dir = argv[++i];
        else if (!strcmp(a, "--refdir")          && i + 1 < argc) refdir = argv[++i];
        else if (!strcmp(a, "--block")           && i + 1 < argc) block_idx = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")          && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold")       && i + 1 < argc) { threshold = strtof(argv[++i], NULL); threshold_set = 1; }
        else if (!strcmp(a, "--stack")) stack = 1;
        else if (!strcmp(a, "-v")) verbose = 1;
        else { usage(argv[0]); return 2; }
    }
    if (stack && !threshold_set) threshold = 1e-2f;

    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/sam3d_slat_dit.safetensors", safetensors_dir);
    sam3d_slat_dit_model *m = sam3d_slat_dit_load_safetensors(model_path);
    if (!m) return 3;
    if (block_idx < 0 || block_idx >= m->n_blocks) {
        fprintf(stderr, "bad --block %d (n_blocks=%d)\n", block_idx, m->n_blocks);
        sam3d_slat_dit_free(m);
        return 2;
    }

    int dim = m->dim;
    char path[512];
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    snprintf(path, sizeof(path), "%s/c_h_after_ape.npy", refdir);
    float *x = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!x || !is_f32 || nd != 2 || dims[1] != dim) { fprintf(stderr, "bad %s\n", path); return 4; }
    int N = dims[0];

    snprintf(path, sizeof(path), "%s/c_h_after_block_%d.npy", refdir, block_idx);
    float *ref = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!ref || !is_f32 || nd != 2 || dims[0] != N || dims[1] != dim) { fprintf(stderr, "bad %s\n", path); return 4; }

    snprintf(path, sizeof(path), "%s/slat_dit_cond.npy", refdir);
    float *cond = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!cond || !is_f32 || !((nd == 2 && dims[1] == dim) || (nd == 3 && dims[0] == 1 && dims[2] == dim))) {
        fprintf(stderr, "bad %s\n", path); return 4;
    }
    int Nc = (nd == 3) ? dims[1] : dims[0];

    snprintf(path, sizeof(path), "%s/c_t_emb.npy", refdir);
    float *t_emb = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!t_emb || !is_f32 || nd != 1 || dims[0] != dim) { fprintf(stderr, "bad %s\n", path); return 4; }

    float *dst = (float *)malloc((size_t)N * dim * sizeof(float));
    if (!dst) return 5;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 6;
    if (cuInit(0) != CUDA_SUCCESS) return 6;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 6;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 6;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "verify_slat_transformer_block_realw") < 0) return 7;
    cs3d_slatdit_fns fns;
    if (cs3d_slatdit_fns_lookup(&fns, mod) != 0) return 7;

    cs3d_slatdit_gpu g;
    if (cs3d_slatdit_gpu_load_transformer(&g, m, verbose) != 0) return 5;
    cs3d_slatdit_block_ws ws;
    if (cs3d_slatdit_block_ws_alloc(&ws, N, Nc, g.dim, g.mlp_hidden) != 0) return 5;

    size_t XD = (size_t)N * dim;
    size_t CD = (size_t)Nc * dim;
    CUdeviceptr d_x0 = cu_upload_raw(x, XD * sizeof(float));
    CUdeviceptr d_x = cu_upload_raw(x, XD * sizeof(float));
    CUdeviceptr d_cond = cu_upload_raw(cond, CD * sizeof(float));
    CUdeviceptr d_t_emb = cu_upload_raw(t_emb, (size_t)dim * sizeof(float));
    if (!d_x0 || !d_x || !d_cond || !d_t_emb) return 5;

    int first_block = stack ? 0 : block_idx;
    int n_run_blocks = stack ? (block_idx + 1) : 1;
#define RUN_SEQUENCE() do { \
    if (cuMemcpyDtoD(d_x, d_x0, XD * sizeof(float)) != CUDA_SUCCESS) return 5; \
    if (cs3d_slatdit_stack_forward(&fns, &ws, &g, first_block, n_run_blocks, d_t_emb, d_x, N, d_cond, Nc) != 0) return 8; \
} while (0)

    RUN_SEQUENCE();
    cuCtxSynchronize();
    cuMemcpyDtoH(dst, d_x, XD * sizeof(float));

    double mean = 0.0;
    float mx = max_abs_f32(dst, ref, XD, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_SEQUENCE();
        cuCtxSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_SEQUENCE

    fprintf(stderr,
            "[verify_slat_transformer_block_realw] mode=%s block=%d n_run=%d N=%d Nc=%d dim=%d H=%d D_h=%d hidden=%d max_abs=%.6e mean_abs=%.6e avg=%.4f ms x%d %s (threshold %.1g)\n",
            stack ? "stack" : "single", block_idx, n_run_blocks, N, Nc, g.dim, g.n_heads, g.head_dim, g.mlp_hidden,
            (double)mx, mean, avg_ms, repeat, ok ? "OK" : "FAIL", (double)threshold);

    cuMemFree(d_x0); cuMemFree(d_x); cuMemFree(d_cond); cuMemFree(d_t_emb);
    cs3d_slatdit_block_ws_free(&ws);
    cs3d_slatdit_gpu_free(&g);
    free(x); free(ref); free(cond); free(t_emb); free(dst);
    sam3d_slat_dit_free(m);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
