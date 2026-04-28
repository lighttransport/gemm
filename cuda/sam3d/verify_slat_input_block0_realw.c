/* verify_slat_input_block0_realw - real-weight SLAT input_blocks[0] CUDA gate. */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"
#include "sam3d_cpu.h"
#include "../../common/sam3d_slat_dit.h"
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

static float max_abs_f32(const float *a, const float *b, size_t n, double *mean_out,
                         int *bad_out)
{
    double sum = 0.0;
    float mx = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (!isfinite(d)) {
            bad++;
            continue;
        }
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n ? n : 1);
    if (bad_out) *bad_out = bad;
    return mx;
}

static int upload_qtensor(const qtensor *t, const char *name, CUdeviceptr *out)
{
    if (!t || !t->data || t->n_rows <= 0 || t->n_cols <= 0) {
        fprintf(stderr, "dequant failed: %s\n", name);
        return -1;
    }
    int n = t->n_rows * t->n_cols;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return -1;
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        int bs = 0, ts = 0;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32;  ts = 34;  break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default:
                free(buf);
                fprintf(stderr, "unsupported tensor type for %s\n", name);
                return -1;
        }
        size_t row_bytes = (size_t)((t->n_cols + bs - 1) / bs) * ts;
        for (int r = 0; r < t->n_rows; r++) {
            const void *row = (const uint8_t *)t->data + (size_t)r * row_bytes;
            dequant_row(t->type, row, buf + (size_t)r * t->n_cols, t->n_cols);
        }
    }
    *out = cu_upload_raw(buf, (size_t)n * sizeof(float));
    free(buf);
    if (!*out) {
        fprintf(stderr, "upload failed: %s\n", name);
        return -1;
    }
    return 0;
}

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s [--safetensors-dir DIR] [--refdir DIR] [--repeat N] [--threshold F] [-v]\n", argv0);
}

int main(int argc, char **argv)
{
    const char *safetensors_dir = "/mnt/disk01/models/sam3d/safetensors";
    const char *refdir = "/tmp/sam3d_ref";
    int repeat = 20;
    float threshold = 5e-4f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i + 1 < argc) safetensors_dir = argv[++i];
        else if (!strcmp(a, "--refdir")          && i + 1 < argc) refdir = argv[++i];
        else if (!strcmp(a, "--repeat")          && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold")       && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { usage(argv[0]); return 2; }
    }

    sam3d_cpu_slat_dit *cw = sam3d_cpu_slat_dit_load(safetensors_dir);
    if (!cw) return 3;
    sam3d_slat_dit_model *m = (sam3d_slat_dit_model *)sam3d_cpu_slat_dit_model(cw);
    if (!m || m->n_io_res_blocks < 1) return 3;
    const sam3d_slat_io_block *bk = &m->input_blocks[0];
    int C = bk->C_in;
    int C_out = bk->C_out;
    int dim = m->dim;
    if (C != C_out || C <= 0) {
        fprintf(stderr, "expected identity-width input block 0, got C_in=%d C_out=%d\n", C, C_out);
        return 3;
    }

    char path[512];
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    snprintf(path, sizeof(path), "%s/c_h_after_input_layer.npy", refdir);
    float *x = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!x || !is_f32 || nd != 2 || dims[1] != C) { fprintf(stderr, "bad %s\n", path); return 4; }
    int N = dims[0];

    snprintf(path, sizeof(path), "%s/c_coords_after_input_block_0.npy", refdir);
    float *coords_f = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!coords_f || !is_f32 || nd != 2 || dims[0] != N || dims[1] != 4) { fprintf(stderr, "bad %s\n", path); return 4; }
    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    if (!coords) return 5;
    for (int i = 0; i < N * 4; i++) coords[i] = (int32_t)lrintf(coords_f[i]);

    snprintf(path, sizeof(path), "%s/c_t_emb.npy", refdir);
    float *t_emb = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!t_emb || !is_f32 || nd != 1 || dims[0] != dim) { fprintf(stderr, "bad %s\n", path); return 4; }

    snprintf(path, sizeof(path), "%s/c_h_after_input_block_0.npy", refdir);
    float *ref = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!ref || !is_f32 || nd != 2 || dims[0] != N || dims[1] != C) { fprintf(stderr, "bad %s\n", path); return 4; }
    float *dst = (float *)malloc((size_t)N * C * sizeof(float));
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
                           "verify_slat_input_block0_realw") < 0) return 7;

    CUfunction fn_gemm = NULL, fn_idx = NULL, fn_ln = NULL, fn_silu = NULL;
    CUfunction fn_conv = NULL, fn_modln = NULL, fn_resadd = NULL;
    if (cuModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_idx, mod, "slat_build_coord_index64_i32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_ln, mod, "layernorm_token_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_silu, mod, "silu_inplace_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_conv, mod, "slat_submconv3x3_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_modln, mod, "modulated_ln_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_resadd, mod, "residual_add_f32") != CUDA_SUCCESS)
        return 7;

    size_t XC = (size_t)N * C;
    CUdeviceptr d_coords = cu_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    CUdeviceptr d_x = cu_upload_raw(x, XC * sizeof(float));
    CUdeviceptr d_t = cu_upload_raw(t_emb, (size_t)dim * sizeof(float));
    CUdeviceptr d_nw = 0, d_nb = 0, d_w1 = 0, d_b1 = 0, d_w2 = 0, d_b2 = 0, d_ew = 0, d_eb = 0;
    if (!d_coords || !d_x || !d_t ||
        upload_qtensor(&bk->norm1_w, "norm1_w", &d_nw) != 0 ||
        upload_qtensor(&bk->norm1_b, "norm1_b", &d_nb) != 0 ||
        upload_qtensor(&bk->conv1_w, "conv1_w", &d_w1) != 0 ||
        upload_qtensor(&bk->conv1_b, "conv1_b", &d_b1) != 0 ||
        upload_qtensor(&bk->conv2_w, "conv2_w", &d_w2) != 0 ||
        upload_qtensor(&bk->conv2_b, "conv2_b", &d_b2) != 0 ||
        upload_qtensor(&bk->emb_w, "emb_w", &d_ew) != 0 ||
        upload_qtensor(&bk->emb_b, "emb_b", &d_eb) != 0)
        return 5;

    CUdeviceptr d_index = 0, d_t_silu = 0, d_emb = 0, d_h1 = 0, d_h2 = 0, d_h3 = 0, d_h4 = 0;
    if (cuMemAlloc(&d_index, (size_t)64 * 64 * 64 * sizeof(int)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_t_silu, (size_t)dim * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_emb, (size_t)2 * C * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_h1, XC * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_h2, XC * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_h3, XC * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_h4, XC * sizeof(float)) != CUDA_SUCCESS)
        return 5;

    float eps = m->ln_eps;
    int affine = 1;
    int n_elem = (int)XC;
    int twoC = 2 * C;
    int one = 1;
    int total_conv = N * C;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    CUdeviceptr d_scale = d_emb;
    CUdeviceptr d_shift = d_emb + (size_t)C * sizeof(float);

#define RUN_BLOCK0() do { \
    if (cuMemsetD8(d_index, 0xff, (size_t)64 * 64 * 64 * sizeof(int)) != CUDA_SUCCESS) return 5; \
    if (cuMemcpyDtoD(d_t_silu, d_t, (size_t)dim * sizeof(float)) != CUDA_SUCCESS) return 5; \
    void *a_idx[] = { &d_coords, &N, &d_index }; \
    if (cuLaunchKernel(fn_idx, (N + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_idx, NULL) != CUDA_SUCCESS) return 8; \
    void *a_s0[] = { &d_t_silu, &dim }; \
    if (cuLaunchKernel(fn_silu, (dim + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s0, NULL) != CUDA_SUCCESS) return 8; \
    { unsigned gx = 1, gy = (twoC + 15) / 16; void *a[] = { &d_emb, &d_t_silu, &d_ew, &d_eb, &one, &dim, &twoC }; \
      if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != CUDA_SUCCESS) return 8; } \
    void *a_ln[] = { &d_h1, &d_x, &d_nw, &d_nb, &N, &C, &eps, &affine }; \
    if (cuLaunchKernel(fn_ln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ln, NULL) != CUDA_SUCCESS) return 8; \
    void *a_s1[] = { &d_h1, &n_elem }; \
    if (cuLaunchKernel(fn_silu, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s1, NULL) != CUDA_SUCCESS) return 8; \
    void *a_c1[] = { &d_coords, &d_h1, &d_index, &d_w1, &d_b1, &N, &C, &C, &d_h2 }; \
    if (cuLaunchKernel(fn_conv, (total_conv + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_c1, NULL) != CUDA_SUCCESS) return 8; \
    void *a_ml[] = { &d_h3, &d_h2, &d_shift, &d_scale, &N, &C, &eps }; \
    if (cuLaunchKernel(fn_modln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ml, NULL) != CUDA_SUCCESS) return 8; \
    void *a_s2[] = { &d_h3, &n_elem }; \
    if (cuLaunchKernel(fn_silu, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s2, NULL) != CUDA_SUCCESS) return 8; \
    void *a_c2[] = { &d_coords, &d_h3, &d_index, &d_w2, &d_b2, &N, &C, &C, &d_h4 }; \
    if (cuLaunchKernel(fn_conv, (total_conv + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_c2, NULL) != CUDA_SUCCESS) return 8; \
    void *a_ra[] = { &d_h4, &d_x, &n_elem }; \
    if (cuLaunchKernel(fn_resadd, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_ra, NULL) != CUDA_SUCCESS) return 8; \
} while (0)

    RUN_BLOCK0();
    cuCtxSynchronize();
    cuMemcpyDtoH(dst, d_h4, XC * sizeof(float));
    double mean = 0.0;
    int bad = 0;
    float mx = max_abs_f32(dst, ref, XC, &mean, &bad);
    int ok = (bad == 0 && mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_BLOCK0();
        cuCtxSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_BLOCK0

    fprintf(stderr,
            "[verify_slat_input_block0_realw] N=%d C=%d dim=%d max_abs=%.6e mean_abs=%.6e avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, C, dim, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);
    if (bad) fprintf(stderr, "[verify_slat_input_block0_realw] nonfinite_diffs=%d\n", bad);

    cuMemFree(d_coords); cuMemFree(d_x); cuMemFree(d_t); cuMemFree(d_nw); cuMemFree(d_nb);
    cuMemFree(d_w1); cuMemFree(d_b1); cuMemFree(d_w2); cuMemFree(d_b2); cuMemFree(d_ew); cuMemFree(d_eb);
    cuMemFree(d_index); cuMemFree(d_t_silu); cuMemFree(d_emb); cuMemFree(d_h1); cuMemFree(d_h2); cuMemFree(d_h3); cuMemFree(d_h4);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    free(x); free(coords_f); free(coords); free(t_emb); free(ref); free(dst);
    sam3d_cpu_slat_dit_free(cw);
    return ok ? 0 : 9;
}
