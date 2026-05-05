/*
 * test_paint_unet.c - Native CUDA SD-2.1 paint UNet (Phase 3 skeleton).
 *
 * Loads stock paint UNet weights (paint_unet_stock.safetensors produced by
 * ref/hy3d/export_paint_unet_safetensors.py), runs forward pieces, and
 * diffs them against the diffusers reference dump from
 * ref/hy3d/dump_paint_unet.py.
 *
 * Phase 3 incremental: at each iteration we add another stage and validate
 * one intermediate. Current stages live behind --stage <name>:
 *   time_emb : timestep_embedding + time MLP -> [B, 1280]
 *   conv_in  : conv_in 12->320 -> [B, 320, 64, 64]
 *
 * Usage:
 *   ./test_paint_unet --stage conv_in \\
 *       /mnt/disk01/.../unet/paint_unet_stock.safetensors \\
 *       /tmp/hy3d_paint_unet_ref/
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_unet_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ===== .npy I/O (float32 + int64) ========================================= */

static void *read_npy(const char *path, int *out_ndim, uint64_t *out_shape,
                       size_t *out_n, char *out_dtype) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return NULL; }
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) {
        fprintf(stderr, "ERROR: not a .npy file: %s\n", path); fclose(f); return NULL;
    }
    uint8_t ver[2]; if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t hlen; if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return NULL; }
    char hdr[1024];
    if (hlen >= sizeof(hdr)) { fclose(f); return NULL; }
    if (fread(hdr, 1, hlen, f) != hlen) { fclose(f); return NULL; }
    hdr[hlen] = 0;
    /* dtype */
    const char *dt;
    int elt;
    if ((dt = strstr(hdr, "'descr': '<f4'"))) { strcpy(out_dtype, "f4"); elt = 4; }
    else if ((dt = strstr(hdr, "'descr': '<i8'"))) { strcpy(out_dtype, "i8"); elt = 8; }
    else { fprintf(stderr, "ERROR: unsupported dtype in %s\n", path); fclose(f); return NULL; }
    /* shape */
    const char *p = strstr(hdr, "'shape': (");
    if (!p) { fclose(f); return NULL; }
    p += strlen("'shape': (");
    int nd = 0; uint64_t shape[8]; size_t total = 1;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        char *end;
        uint64_t v = strtoull(p, &end, 10);
        shape[nd++] = v; total *= v;
        p = end;
    }
    *out_ndim = nd;
    for (int i = 0; i < nd; i++) out_shape[i] = shape[i];
    *out_n = total;
    void *buf = malloc(total * (size_t)elt);
    if (fread(buf, (size_t)elt, total, f) != total) {
        fprintf(stderr, "ERROR: short read on %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

/* ===== Weight upload ====================================================== */

static CUdeviceptr upload_st(const st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "ERROR: tensor not found: %s\n", name);
        return 0;
    }
    if (strcmp(safetensors_dtype(st, idx), "F32")) {
        fprintf(stderr, "ERROR: %s dtype %s, expected F32\n", name,
                safetensors_dtype(st, idx));
        return 0;
    }
    size_t bytes = safetensors_nbytes(st, idx);
    CUdeviceptr d;
    cuMemAlloc(&d, bytes);
    cuMemcpyHtoD(d, safetensors_data(st, idx), bytes);
    return d;
}

/* ===== Diff helper ======================================================== */

static int diff_against(const float *cu, const char *ref_path, size_t expect_n,
                          float warn_mae) {
    int nd; uint64_t shape[8]; size_t n; char dt[8];
    float *ref = (float *)read_npy(ref_path, &nd, shape, &n, dt);
    if (!ref) return -1;
    if (n != expect_n) {
        fprintf(stderr, "ERROR: ref %s has %zu elements, expected %zu\n",
                ref_path, n, expect_n);
        free(ref); return -1;
    }
    double sae = 0, smax = 0, sum_r = 0, sum_c = 0, sum_rc = 0, sum_rr = 0, sum_cc = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)cu[i] - (double)ref[i];
        if (d < 0) d = -d;
        sae += d; if (d > smax) smax = d;
        double r = ref[i], c = cu[i];
        sum_r += r; sum_c += c; sum_rc += r*c; sum_rr += r*r; sum_cc += c*c;
    }
    double mae = sae / n;
    double mr = sum_r/n, mc = sum_c/n;
    double cov = sum_rc/n - mr*mc;
    double vr  = sum_rr/n - mr*mr;
    double vc  = sum_cc/n - mc*mc;
    double corr = cov / sqrt(vr * vc + 1e-30);
    int ok = mae <= warn_mae;
    fprintf(stderr, "  vs %s : mae=%.4e max=%.4e corr=%.6f  %s\n",
            ref_path, mae, smax, corr, ok ? "OK" : "WARN");
    free(ref);
    return ok ? 0 : 1;
}

/* ===== Kernels ============================================================ */

typedef struct {
    CUmodule mod;
    CUfunction f_tse;     /* unet_timestep_embed_f32 */
    CUfunction f_lin;     /* unet_linear_f32 */
    CUfunction f_silu;    /* unet_silu_f32 */
    CUfunction f_conv;    /* unet_conv2d_f32 */
} pu_kernels;

static void k_timestep_embed(const pu_kernels *kk, CUdeviceptr out,
                              CUdeviceptr ts, int B, int dim) {
    void *args[] = { &out, &ts, &B, &dim };
    int tx = 64;
    cuLaunchKernel(kk->f_tse, (unsigned)B, (unsigned)((dim + tx - 1) / tx), 1,
                    tx, 1, 1, 0, 0, args, NULL);
}

static void k_linear(const pu_kernels *kk, CUdeviceptr y, CUdeviceptr x,
                      CUdeviceptr W, CUdeviceptr b, int M, int K, int N) {
    void *args[] = { &y, &x, &W, &b, &M, &K, &N };
    unsigned gx = (unsigned)((N + 15) / 16), gy = (unsigned)((M + 15) / 16);
    cuLaunchKernel(kk->f_lin, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL);
}

static void k_silu(const pu_kernels *kk, CUdeviceptr x, int n) {
    void *args[] = { &x, &n };
    cuLaunchKernel(kk->f_silu, (unsigned)((n + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_conv(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                    CUdeviceptr W, CUdeviceptr b,
                    int ci, int h, int w, int co, int kh, int kw, int pad) {
    void *args[] = { &out, &in, &W, &b, &ci, &h, &w, &co, &kh, &kw, &pad };
    int total = co * h * w;
    cuLaunchKernel(kk->f_conv, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

/* ===== main =============================================================== */

int main(int argc, char **argv) {
    const char *stage = "conv_in";
    int argi = 1;
    if (argi < argc && !strcmp(argv[argi], "--stage")) {
        stage = argv[argi+1]; argi += 2;
    }
    if (argc - argi < 2) {
        fprintf(stderr,
            "Usage: %s [--stage time_emb|conv_in] <unet.safetensors> <ref_dir>\n",
            argv[0]);
        return 1;
    }
    const char *st_path = argv[argi];
    const char *ref_dir = argv[argi+1];

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    pu_kernels kk = {0};
    if (cu_compile_kernels(&kk.mod, dev, cuda_paint_unet_kernels_src,
                             "hy3d_paint_unet", 1, "HY3D-PAINT-UNET") < 0)
        return 1;
    cuModuleGetFunction(&kk.f_tse,  kk.mod, "unet_timestep_embed_f32");
    cuModuleGetFunction(&kk.f_lin,  kk.mod, "unet_linear_f32");
    cuModuleGetFunction(&kk.f_silu, kk.mod, "unet_silu_f32");
    cuModuleGetFunction(&kk.f_conv, kk.mod, "unet_conv2d_f32");

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "ERROR: cannot open %s\n", st_path); return 1; }
    fprintf(stderr, "loaded safetensors %s\n", st_path);

    /* Load reference inputs from the dump dir */
    char path[512];
    int nd; uint64_t shape[8]; size_t n; char dt[8];
    snprintf(path, sizeof(path), "%s/ref_timestep.npy", ref_dir);
    int64_t *ts = (int64_t *)read_npy(path, &nd, shape, &n, dt);
    if (!ts) return 1;
    int B = (int)shape[0];
    fprintf(stderr, "B=%d, timestep[0]=%lld\n", B, (long long)ts[0]);

    if (!strcmp(stage, "time_emb")) {
        /* timestep -> sinusoidal[320] -> linear(320,1280) silu linear(1280,1280)
         * Output [B,1280] vs ref_time_emb.npy */
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_emb;  cuMemAlloc(&d_emb, B * 320 * sizeof(float));
        CUdeviceptr d_h1;   cuMemAlloc(&d_h1,  B * 1280 * sizeof(float));
        CUdeviceptr d_h2;   cuMemAlloc(&d_h2,  B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");

        k_timestep_embed(&kk, d_emb, d_ts, B, 320);
        k_linear(&kk, d_h1, d_emb, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_h1, B * 1280);
        k_linear(&kk, d_h2, d_h1, l2_w, l2_b, B, 1280, 1280);
        cuCtxSynchronize();

        float *cu = (float *)malloc(B * 1280 * sizeof(float));
        cuMemcpyDtoH(cu, d_h2, B * 1280 * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_time_emb.npy", ref_dir);
        diff_against(cu, path, (size_t)B * 1280, 1e-3f);
        free(cu);
    } else if (!strcmp(stage, "conv_in")) {
        /* Read sample, run conv_in 12->320, compare to ref_conv_in.npy */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) {
            fprintf(stderr, "ERROR: expected sample channels=12, got %d\n", IC);
            return 1;
        }
        fprintf(stderr, "sample [%d, %d, %d, %d]\n", B, IC, H, W);
        size_t in_n  = (size_t)B * IC * H * W;
        size_t out_n = (size_t)B * 320 * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n  * sizeof(float));
        CUdeviceptr d_out; cuMemAlloc(&d_out, out_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        /* Batch loop: kernel handles one sample (CHW) at a time. */
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b  = d_in  + (CUdeviceptr)b * IC  * H * W * sizeof(float);
            CUdeviceptr out_b = d_out + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_conv(&kk, out_b, in_b, cw, cb, IC, H, W, 320, 3, 3, 1);
        }
        cuCtxSynchronize();

        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, d_out, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_conv_in.npy", ref_dir);
        diff_against(cu, path, out_n, 1e-3f);
        free(cu); free(sample);
    } else {
        fprintf(stderr, "unknown stage: %s\n", stage); return 1;
    }

    free(ts);
    safetensors_close(st);
    cuModuleUnload(kk.mod);
    cuCtxDestroy(ctx);
    return 0;
}
