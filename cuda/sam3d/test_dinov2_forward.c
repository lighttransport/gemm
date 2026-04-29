/*
 * test_dinov2_forward — Phase 1b.7b end-to-end DINOv2 numerics gate.
 *
 * Loads sam3d_dinov2.safetensors twice — once via the CPU `dinov2.h`
 * loader (the existing reference path used by verify_dinov2 / test_sam3d)
 * and once via `cs3d_dinov2_gpu_load` (Phase 1b.0 device buffers) — runs
 * a random ImageNet-normalized [3, 518, 518] image through both, and
 * diffs the final tokens.
 *
 * Validates the full Phase 1b.7b composition: patch_embed → prepend +
 * pos_embed → 24 ViT blocks (LN→QKV→split→SDPA→out_proj→ls1+resid →
 * LN→fc1→gelu→fc2→ls2+resid) → final LN.
 *
 * Usage:
 *   ./test_dinov2_forward --safetensors-dir DIR [--threshold 5e-2] [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"

#define DINOV2_IMPLEMENTATION
#include "../../common/dinov2.h"

#define CUDA_SAM3D_DINOV2_GPU_IMPLEMENTATION
#include "cuda_sam3d_dinov2_gpu.h"

#include "cuda_sam3d_kernels.h"
#include "cuda_sam3d_dinov2_forward.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static float urand(uint32_t *state) {
    *state = (*state) * 1664525u + 1013904223u;
    return (float)((*state) >> 8) / (float)(1u << 24);
}

static float max_abs(const float *a, const float *b, size_t n,
                     double *mean_out, size_t *argmax_out) {
    double sum = 0.0; float mx = 0.0f; size_t am = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; am = i; }
        sum += d;
    }
    if (mean_out)   *mean_out   = sum / (n > 0 ? n : 1);
    if (argmax_out) *argmax_out = am;
    return mx;
}

int main(int argc, char **argv)
{
    const char *sft_dir   = NULL;
    float       threshold = 5e-2f;
    int         verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir   = argv[++i];
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                              verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (!sft_dir) {
        fprintf(stderr, "usage: %s --safetensors-dir DIR [--threshold F] [-v]\n", argv[0]);
        return 2;
    }
    char st_path[1200];
    snprintf(st_path, sizeof(st_path), "%s/sam3d_dinov2.safetensors", sft_dir);

    /* ---- CUDA init. ---- */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    int sm = cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_dinov2_forward");
    if (sm < 0) return 4;
    cs3d_dinov2_fns fns;
    if (cs3d_dinov2_fns_lookup(&fns, mod) < 0) return 4;

    /* ---- Load weights both sides. ---- */
    double t0 = now_ms();
    dinov2_model *cpu_m = dinov2_load_safetensors(st_path);
    if (!cpu_m) { fprintf(stderr, "dinov2_load_safetensors failed: %s\n", st_path); return 5; }
    fprintf(stderr, "[load] CPU dinov2 model in %.1f ms\n", now_ms() - t0);

    cs3d_dinov2_gpu gpu = {0};
    t0 = now_ms();
    if (cs3d_dinov2_gpu_load(&gpu, st_path, verbose) < 0) {
        fprintf(stderr, "cs3d_dinov2_gpu_load failed\n"); return 5;
    }
    fprintf(stderr, "[load] GPU dinov2 weights in %.1f ms (%.1f MiB)\n",
            now_ms() - t0, (double)gpu.total_bytes / (1024.0 * 1024.0));

    /* Sanity: the two paths should agree on geometry. */
    if (gpu.dim != cpu_m->dim || gpu.n_blocks != cpu_m->n_blocks ||
        gpu.n_tokens != cpu_m->n_tokens) {
        fprintf(stderr, "geometry mismatch: gpu(dim=%d nb=%d nt=%d) cpu(dim=%d nb=%d nt=%d)\n",
                gpu.dim, gpu.n_blocks, gpu.n_tokens,
                cpu_m->dim, cpu_m->n_blocks, cpu_m->n_tokens);
        return 5;
    }

    /* ---- Random ImageNet-normalized [3, H, W] f32 image. The two paths
     * see the same bytes (host buffer + a device upload of the same). ---- */
    int H = gpu.image_size, W = gpu.image_size;
    size_t n_img = (size_t)3 * H * W;
    float *h_img = (float *)malloc(n_img * sizeof(float));
    if (!h_img) { fprintf(stderr, "img alloc failed\n"); return 5; }
    uint32_t rng = 0xC0FFEEu;
    /* ~ImageNet-normalized range: zero mean, ~unit std. Generate uniform
     * [-2, 2] which is wide enough to exercise the network. */
    for (size_t i = 0; i < n_img; i++) h_img[i] = (urand(&rng) * 4.0f - 2.0f);

    /* ---- CPU forward. ---- */
    t0 = now_ms();
    int n_threads = 1;
    {
        const char *e = getenv("OMP_NUM_THREADS");
        if (e) { int v = atoi(e); if (v > 0) n_threads = v; }
    }
    dinov2_result cpu_r = dinov2_encode_f32(cpu_m, h_img, H, W, n_threads);
    if (!cpu_r.features) { fprintf(stderr, "CPU dinov2_encode_f32 failed\n"); return 6; }
    fprintf(stderr, "[forward] CPU full forward in %.1f ms (n_threads=%d)\n",
            now_ms() - t0, n_threads);

    /* ---- GPU forward. ---- */
    CUdeviceptr d_img = cu_upload_raw(h_img, n_img * sizeof(float));
    CUdeviceptr d_out = 0;
    size_t n_out = (size_t)gpu.n_tokens * gpu.dim;
    if (cuMemAlloc(&d_out, n_out * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc d_out failed\n"); return 5;
    }
    /* Zero-init so any leftover bytes don't hide a kernel that fails to
     * write to a region (e.g. last patch row). */
    {
        float *zeros = (float *)calloc(n_out, sizeof(float));
        cuMemcpyHtoD(d_out, zeros, n_out * sizeof(float));
        free(zeros);
    }

    cs3d_dinov2_block_ws ws = {0};
    if (cs3d_dinov2_block_ws_alloc(&ws, gpu.n_tokens, gpu.dim, gpu.ffn_hidden) < 0) {
        fprintf(stderr, "ws alloc failed\n"); return 5;
    }

    t0 = now_ms();
    if (cs3d_dinov2_gpu_forward(&fns, &ws, &gpu, d_img, d_out) < 0) {
        fprintf(stderr, "cs3d_dinov2_gpu_forward failed\n"); return 6;
    }
    fprintf(stderr, "[forward] GPU full forward in %.1f ms\n", now_ms() - t0);

    /* ---- Diff. ---- */
    float *h_out = (float *)malloc(n_out * sizeof(float));
    cuMemcpyDtoH(h_out, d_out, n_out * sizeof(float));

    double mean = 0.0; size_t am = 0;
    float mx = max_abs(h_out, cpu_r.features, n_out, &mean, &am);
    int ok = (mx <= threshold);
    int tok = (int)(am / gpu.dim);
    int chn = (int)(am % gpu.dim);
    fprintf(stderr,
        "[test_dinov2_forward] n_tok=%d dim=%d  n_out=%zu  "
        "max_abs=%.4g (tok=%d ch=%d) mean_abs=%.4g  %s (threshold %.1g)\n",
        gpu.n_tokens, gpu.dim, n_out, (double)mx, tok, chn, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    cs3d_dinov2_block_ws_free(&ws);
    cuMemFree(d_img); cuMemFree(d_out);
    free(h_out); free(h_img);
    cs3d_dinov2_gpu_free(&gpu);
    dinov2_result_free(&cpu_r);
    dinov2_free(cpu_m);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
