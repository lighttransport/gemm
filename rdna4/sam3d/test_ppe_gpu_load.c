/*
 * test_ppe_gpu_load — Phase 2b.8c standalone verification.
 *
 * Loads the real PPE safetensors checkpoint, uploads weights to the
 * device via cs3d_ppe_gpu_load, runs cs3d_ppe_forward on a fixed-seed
 * synthetic pointmap, and diffs against sam3d_ppe_encode (CPU).
 *
 * Threshold accounts for the fp32-vs-double drift compounded across
 * the LN/gemm/softmax chain (PPE has a single ViT block, so much
 * smaller than DINOv2-L's 24-block stack).
 *
 * Usage:
 *   ./test_ppe_gpu_load --safetensors-dir <DIR> [--threshold 5e-4] [-v]
 */

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"


#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SAM3D_COND_FUSER_IMPLEMENTATION
#include "../../common/sam3d_cond_fuser.h"

#include "hip_sam3d_kernels.h"
#include "hip_sam3d_ppe_forward.h"
#define HIP_SAM3D_PPE_GPU_IMPLEMENTATION
#include "hip_sam3d_ppe_gpu.h"

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
    float threshold = 5e-4f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                              verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (!sft_dir) {
        fprintf(stderr, "usage: %s --safetensors-dir <DIR> [--threshold 5e-4] [-v]\n", argv[0]);
        return 2;
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_point_patch_embed.safetensors", sft_dir);
    sam3d_ppe_model *m = sam3d_ppe_load_safetensors(path);
    if (!m) { fprintf(stderr, "load %s failed\n", path); return 3; }
    int Np = m->num_patches, P = m->patch_size, D = m->embed_dim;
    int H = m->n_heads, S = m->input_size, Nwin = Np * Np;
    fprintf(stderr, "ppe: Np=%d P=%d D=%d H=%d S=%d ffn=%d\n",
            Np, P, D, H, S, m->ffn_hidden);

    /* Synthetic pointmap [S, S, 3], some NaN sprinkled. */
    float *h_pmap = (float *)malloc((size_t)S * S * 3 * sizeof(float));
    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < (size_t)S * S * 3; i++) {
        h_pmap[i] = urand(&rng) * 2.0f - 1.0f;
        if (urand(&rng) < 0.03f) h_pmap[i] = NAN;
    }

    /* CPU ref. */
    float *h_ref = sam3d_ppe_encode(m, h_pmap, S, S, NULL, 1);
    if (!h_ref) { fprintf(stderr, "sam3d_ppe_encode failed\n"); return 4; }

    /* GPU path. */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 5;
    if (hipInit(0) != hipSuccess) return 5;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 5;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 5;

    hipModule_t mod = 0;
    int sm = hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_ppe_gpu_load");
    if (sm < 0) return 6;
    cs3d_ppe_fns fns;
    if (cs3d_ppe_fns_lookup(&fns, mod) < 0) return 6;

    cs3d_ppe_gpu g;
    if (cs3d_ppe_gpu_load(&g, m, verbose) < 0) return 7;

    cs3d_ppe_ws ws = {0};
    if (cs3d_ppe_ws_alloc(&ws, g.Np, g.P, g.D, g.ffn) < 0) return 7;

    hipDeviceptr_t d_pmap = hip_upload_raw(h_pmap, (size_t)S * S * 3 * sizeof(float));
    hipDeviceptr_t d_o = 0;
    if (hipMalloc(&d_o, (size_t)Nwin * D * sizeof(float)) != hipSuccess) return 7;

    if (cs3d_ppe_forward(&fns, &ws, &g.w, d_pmap, d_o, g.n_heads, g.ln_eps) < 0) {
        fprintf(stderr, "ppe forward launch failed\n"); return 8;
    }
    hipDeviceSynchronize();

    float *h_dst = (float *)malloc((size_t)Nwin * D * sizeof(float));
    hipMemcpyDtoH(h_dst, d_o, (size_t)Nwin * D * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, (size_t)Nwin * D, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_ppe_gpu_load] Nwin=%d D=%d  max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        Nwin, D, (double)mx, mean, ok ? "OK" : "FAIL", (double)threshold);

    free(h_pmap); free(h_ref); free(h_dst);
    hipFree(d_pmap); hipFree(d_o);
    cs3d_ppe_ws_free(&ws);
    cs3d_ppe_gpu_free(&g);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    sam3d_ppe_free(m);
    return ok ? 0 : 9;
}
