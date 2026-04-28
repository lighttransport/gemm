/*
 * verify_ssdit_full_realw — Phase 2c.12 real-weights full-stack diff.
 *
 * Loads the SS Flow DiT checkpoint, runs the full GPU outer driver
 * (`cs3d_ssdit_outer_forward`) and diffs against the host CPU
 * `sam3d_ss_flow_dit_forward` on the same inputs. This pins the
 * t/d embedder + 24-block loop + per-modality input/output projection
 * end-to-end at production geometry (D=1024, N_s=4096, N_p=4, N_c=2740).
 *
 * Usage:
 *   ./verify_ssdit_full_realw --safetensors-dir <DIR>
 *                             [--nc 2740] [--threshold 5e-3] [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"
#define SAM3D_SS_FLOW_DIT_IMPLEMENTATION
#include "../../common/sam3d_ss_flow_dit.h"

#define CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION
#include "cuda_sam3d_ssdit_gpu.h"
#undef  CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION
#define CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#include "cuda_sam3d_ssdit_forward.h"
#undef  CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#define CUDA_SAM3D_SSDIT_OUTER_IMPLEMENTATION
#include "cuda_sam3d_ssdit_outer.h"

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
        float dif = fabsf(a[i] - b[i]);
        if (dif > mx) mx = dif;
        sum += dif;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL;
    int N_c = 2740;
    float threshold = 5e-3f;
    int verbose = 0;
    float t_val = 0.5f, d_val = 0.0f;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(a, "--nc")              && i+1 < argc) N_c = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--t")               && i+1 < argc) t_val = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--d")               && i+1 < argc) d_val = strtof(argv[++i], NULL);
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
    int dim = m->dim;
    int n_lat = SAM3D_SS_DIT_N_LATENTS;

    /* Random latents per modality + cond. Use the same buffers for both
     * the CPU host and the GPU paths. */
    uint32_t rng = 0xDA7Au;
    float *lin[5]      = {0};
    float *lout_cpu[5] = {0};
    float *lout_gpu[5] = {0};
    for (int i = 0; i < n_lat; i++) {
        const sam3d_ss_latent_map *L = &m->latent[i];
        size_t nin  = (size_t)L->token_len * L->in_channels;
        size_t nout = (size_t)L->token_len * L->in_channels;
        lin[i]      = (float *)malloc(nin  * sizeof(float));
        lout_cpu[i] = (float *)malloc(nout * sizeof(float));
        lout_gpu[i] = (float *)malloc(nout * sizeof(float));
        for (size_t k = 0; k < nin; k++) lin[i][k] = urand(&rng) * 2.f - 1.f;
    }
    size_t cond_n = (size_t)N_c * dim;
    float *cond = (float *)malloc(cond_n * sizeof(float));
    for (size_t k = 0; k < cond_n; k++) cond[k] = urand(&rng) * 2.f - 1.f;

    /* === CPU host reference === */
    fprintf(stderr, "[verify_ssdit_full_realw] running CPU reference (24 blocks)...\n");
    if (sam3d_ss_flow_dit_forward(m,
                                  (const float *const *)lin,
                                  (float *const *)lout_cpu,
                                  cond, N_c, t_val, d_val, /*nthr*/ 32) < 0) {
        fprintf(stderr, "CPU forward failed\n"); return 4;
    }

    /* === GPU outer driver === */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 5;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 5;
    CUmodule cmod = 0;
    if (cu_compile_kernels(&cmod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "verify_ssdit_full_realw") < 0) return 6;
    cs3d_ssdit_outer_fns fns;
    if (cs3d_ssdit_outer_fns_lookup(&fns, cmod) < 0) return 6;

    cs3d_ssdit_gpu g;
    if (cs3d_ssdit_gpu_load(&g, m, verbose) < 0) return 7;
    cs3d_ssdit_block_ws ws_block;
    int N_s = g.latent[SAM3D_SS_LAT_SHAPE].token_len;
    int N_p = 0;
    for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++)
        N_p += g.latent[i].token_len;
    if (cs3d_ssdit_block_ws_alloc(&ws_block, N_s, N_p, N_c, dim, g.mlp_hidden) < 0) return 7;
    cs3d_ssdit_outer_ws ws_outer;
    if (cs3d_ssdit_outer_ws_alloc(&ws_outer, &g, N_c) < 0) return 7;

    fprintf(stderr, "[verify_ssdit_full_realw] running GPU outer driver...\n");
    if (cs3d_ssdit_outer_forward(&g, &fns, &ws_block, &ws_outer,
                                 (const float *const *)lin,
                                 (float *const *)lout_gpu,
                                 cond, N_c, t_val, d_val) < 0) {
        fprintf(stderr, "GPU outer forward failed\n"); return 8;
    }
    cuCtxSynchronize();

    /* Diff per modality. */
    int ok = 1;
    static const char *names[5] = {
        "shape", "6drot", "translation", "scale", "translation_scale"
    };
    for (int i = 0; i < n_lat; i++) {
        const sam3d_ss_latent_map *L = &m->latent[i];
        size_t n = (size_t)L->token_len * L->in_channels;
        double mean = 0.0;
        float mx = max_abs(lout_gpu[i], lout_cpu[i], n, &mean);
        int pass = (mx <= threshold);
        if (!pass) ok = 0;
        fprintf(stderr,
            "  [%-18s] T=%d Cin=%d  max_abs=%.4g (mean %.4g)  %s\n",
            names[i], L->token_len, L->in_channels,
            (double)mx, mean, pass ? "OK" : "FAIL");
    }
    fprintf(stderr,
        "[verify_ssdit_full_realw] D=%d N_s=%d N_p=%d N_c=%d  %s (threshold %.1g)\n",
        dim, N_s, N_p, N_c, ok ? "OK" : "FAIL", (double)threshold);

    for (int i = 0; i < n_lat; i++) {
        free(lin[i]); free(lout_cpu[i]); free(lout_gpu[i]);
    }
    free(cond);
    cs3d_ssdit_outer_ws_free(&ws_outer);
    cs3d_ssdit_block_ws_free(&ws_block);
    cs3d_ssdit_gpu_free(&g);
    cuModuleUnload(cmod);
    cuCtxDestroy(cu_ctx);
    sam3d_ss_flow_dit_free(m);
    return ok ? 0 : 9;
}
