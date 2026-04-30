/*
 * verify_ssdit_block_realw — Phase 2c.11 real-weights single-block diff.
 *
 * Loads the SS Flow DiT checkpoint, uploads weights via cs3d_ssdit_gpu_load,
 * runs ONE MOT block (block 0) end-to-end on the GPU using the real
 * weights, and diffs against the static `ssdit_block_forward` CPU
 * reference (from common/sam3d_ss_flow_dit.h) on the same random
 * inputs. This pins the GPU loader + block-forward driver to the host
 * port at production geometry (D=1024, N_s=4096, N_p=4, N_c=2740).
 *
 * The mod6 modulation vector is generated randomly here (rather than
 * recomputed from t_emb) — the silu+adaln_gemm path is independently
 * verified by Phase 2c.2 and is not exercised in this microbench.
 *
 * Usage:
 *   ./verify_ssdit_block_realw --safetensors-dir <DIR>
 *                              [--ns 4096] [--np 4] [--nc 2740]
 *                              [--block 0] [--threshold 1e-3] [-v]
 */

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "hip_sam3d_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"
#define SAM3D_SS_FLOW_DIT_IMPLEMENTATION
#include "../../common/sam3d_ss_flow_dit.h"

#define HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION
#include "hip_sam3d_ssdit_gpu.h"
#undef  HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION
#define HIP_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#include "hip_sam3d_ssdit_forward.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) / (float)(1u << 24);
}
static float max_abs(const float *a, const float *b, size_t n, double *mean_out) {
    double sum = 0.0; float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL;
    int N_s = 4096, N_p = 4, N_c = 2740, blk = 0;
    float threshold = 1e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(a, "--ns")              && i+1 < argc) N_s = atoi(argv[++i]);
        else if (!strcmp(a, "--np")              && i+1 < argc) N_p = atoi(argv[++i]);
        else if (!strcmp(a, "--nc")              && i+1 < argc) N_c = atoi(argv[++i]);
        else if (!strcmp(a, "--block")           && i+1 < argc) blk = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                              verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (!sft_dir) {
        fprintf(stderr, "usage: %s --safetensors-dir <DIR> [...]\n", argv[0]);
        return 2;
    }

    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_ss_dit.safetensors", sft_dir);
    sam3d_ss_flow_dit_model *m = sam3d_ss_flow_dit_load_safetensors(path);
    if (!m) return 3;
    int dim = m->dim, H = m->n_heads, D_h = m->head_dim, mlp_h = m->mlp_hidden;
    float eps = m->ln_eps;
    if (blk < 0 || blk >= m->n_blocks) {
        fprintf(stderr, "block %d out of range [0, %d)\n", blk, m->n_blocks); return 4;
    }

    /* Random inputs. */
    uint32_t rng = 0xC0FFEEu + (uint32_t)blk;
    size_t cs = (size_t)N_s * dim, cp = (size_t)N_p * dim, cc = (size_t)N_c * dim;
    float *x_s   = (float *)malloc(cs * sizeof(float));
    float *x_p   = (float *)malloc(cp * sizeof(float));
    float *cond  = (float *)malloc(cc * sizeof(float));
    float *mod6  = (float *)malloc((size_t)6 * dim * sizeof(float));
    for (size_t i = 0; i < cs; i++) x_s[i]  = urand(&rng) * 2.f - 1.f;
    for (size_t i = 0; i < cp; i++) x_p[i]  = urand(&rng) * 2.f - 1.f;
    for (size_t i = 0; i < cc; i++) cond[i] = urand(&rng) * 2.f - 1.f;
    for (int i = 0; i < dim; i++) {
        mod6[0*dim + i] = (urand(&rng) * 2.f - 1.f) * 0.05f;  /* shift_msa */
        mod6[1*dim + i] = (urand(&rng) * 2.f - 1.f) * 0.10f;  /* scale_msa */
        mod6[2*dim + i] = (urand(&rng) * 2.f - 1.f) * 0.50f;  /* gate_msa */
        mod6[3*dim + i] = (urand(&rng) * 2.f - 1.f) * 0.05f;  /* shift_mlp */
        mod6[4*dim + i] = (urand(&rng) * 2.f - 1.f) * 0.10f;  /* scale_mlp */
        mod6[5*dim + i] = (urand(&rng) * 2.f - 1.f) * 0.50f;  /* gate_mlp */
    }

    /* Host ref: deep-copy x_s/x_p, then call the static
     * ssdit_block_forward (in-place updater). */
    float *xs_ref = (float *)malloc(cs * sizeof(float));
    float *xp_ref = (float *)malloc(cp * sizeof(float));
    memcpy(xs_ref, x_s, cs * sizeof(float));
    memcpy(xp_ref, x_p, cp * sizeof(float));
    /* Pose stream is N_p tokens; the loaded model has N_p_total = sum of
     * 4 token_lens. Here we just exercise N_p arbitrary tokens, which is
     * fine since norm/attn/ffn are token-permutation-equivariant. */
    ssdit_block_forward(m, &m->blocks[blk], mod6,
                        xs_ref, N_s, xp_ref, N_p,
                        cond, N_c, /*nthr*/ 1);

    /* Device. */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 5;
    if (hipInit(0) != hipSuccess) return 5;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 5;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 5;
    hipModule_t cmod = 0;
    if (hip_compile_kernels(&cmod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "verify_ssdit_block_realw") < 0) return 6;
    cs3d_ssdit_fns fns;
    if (cs3d_ssdit_fns_lookup(&fns, cmod) < 0) return 6;

    cs3d_ssdit_gpu g;
    if (cs3d_ssdit_gpu_load(&g, m, verbose) < 0) return 7;
    cs3d_ssdit_block_ws ws;
    if (cs3d_ssdit_block_ws_alloc(&ws, N_s, N_p, N_c, dim, mlp_h) < 0) return 7;

    hipDeviceptr_t d_xs   = hip_upload_raw(x_s,  cs * sizeof(float));
    hipDeviceptr_t d_xp   = hip_upload_raw(x_p,  cp * sizeof(float));
    hipDeviceptr_t d_cond = hip_upload_raw(cond, cc * sizeof(float));
    hipDeviceptr_t d_mod6 = hip_upload_raw(mod6, (size_t)6 * dim * sizeof(float));

    if (cs3d_ssdit_block_forward(&fns, &ws, &g.blocks[blk],
                                 d_mod6, d_xs, N_s, d_xp, N_p, d_cond, N_c,
                                 dim, H, D_h, mlp_h, eps) < 0) {
        fprintf(stderr, "block_forward launch failed\n"); return 8;
    }
    hipDeviceSynchronize();

    float *xs_dst = (float *)malloc(cs * sizeof(float));
    float *xp_dst = (float *)malloc(cp * sizeof(float));
    hipMemcpyDtoH(xs_dst, d_xs, cs * sizeof(float));
    hipMemcpyDtoH(xp_dst, d_xp, cp * sizeof(float));

    double mean_s = 0.0, mean_p = 0.0;
    float mx_s = max_abs(xs_dst, xs_ref, cs, &mean_s);
    float mx_p = max_abs(xp_dst, xp_ref, cp, &mean_p);
    int ok = (mx_s <= threshold) && (mx_p <= threshold);

    fprintf(stderr,
        "[verify_ssdit_block_realw] block=%d N_s=%d N_p=%d N_c=%d D=%d  "
        "shape max_abs=%.4g (mean %.4g)  pose max_abs=%.4g (mean %.4g)  %s (threshold %.1g)\n",
        blk, N_s, N_p, N_c, dim,
        (double)mx_s, mean_s, (double)mx_p, mean_p,
        ok ? "OK" : "FAIL", (double)threshold);

    free(x_s); free(x_p); free(cond); free(mod6);
    free(xs_ref); free(xp_ref); free(xs_dst); free(xp_dst);
    hipFree(d_xs); hipFree(d_xp); hipFree(d_cond); hipFree(d_mod6);
    cs3d_ssdit_block_ws_free(&ws);
    cs3d_ssdit_gpu_free(&g);
    hipModuleUnload(cmod);
    hipCtxDestroy(cu_ctx);
    sam3d_ss_flow_dit_free(m);
    return ok ? 0 : 9;
}
