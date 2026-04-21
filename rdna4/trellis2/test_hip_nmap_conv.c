/*
 * test_hip_nmap_conv.c - validate sparse_conv3d_nmap_tiled_f32 against ref.
 *
 * Uses pre-dumped feats, neighbor_map, weights from gen_stage2_ref.py.
 *   ./test_hip_nmap_conv <feats_in.npy> <nmap.npy> <weight.npy> <bias.npy> <feats_out_ref.npy>
 * Weight layout in .npy: [out_C, 3, 3, 3, in_C] (PyTorch conv weight permuted).
 * Kernel expects [out_C, 27, in_C] — same memory, reinterpreted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "hip_tex_dec_kernels.h"

#define SYNC() hipDeviceSynchronize()

static void *read_npy(const char *p, int *nd, int *dd, size_t elt) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); return NULL; }
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    void *d = malloc(n * elt); fread(d, elt, n, f);
    fclose(f); free(h); return d;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s feats_in.npy nmap.npy weight.npy bias.npy feats_out_ref.npy\n", argv[0]);
        return 1;
    }
    int fd[8], nd, md[8], mnd, wd[8], wnd, bd[8], bnd, rd[8], rnd;
    float *feats = read_npy(argv[1], &nd, fd, sizeof(float));
    uint32_t *nmap = read_npy(argv[2], &mnd, md, sizeof(uint32_t));
    float *weight  = read_npy(argv[3], &wnd, wd, sizeof(float));
    float *bias    = read_npy(argv[4], &bnd, bd, sizeof(float));
    float *ref     = read_npy(argv[5], &rnd, rd, sizeof(float));
    if (!feats || !nmap || !weight || !bias || !ref) return 1;

    int N = fd[0], in_C = fd[1];
    int out_C = wd[0];
    fprintf(stderr, "feats (%d, %d), nmap (%d, %d), weight dims=%d shape[0..%d]=",
            fd[0], fd[1], md[0], md[1], wnd, wnd-1);
    for (int i = 0; i < wnd; i++) fprintf(stderr, "%d ", wd[i]);
    fprintf(stderr, "\nbias %d, ref (%d, %d)\n", bd[0], rd[0], rd[1]);
    if (rd[0] != N || rd[1] != out_C) {
        fprintf(stderr, "shape mismatch: expected ref (%d, %d)\n", N, out_C);
        return 1;
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) return 1;
    hipSetDevice(0);
    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, hip_tex_dec_kernels_src, "nmap_conv", 1, "HIP") <= 0) return 1;
    hipFunction_t k_tiled, k_scalar;
    hipModuleGetFunction(&k_tiled, mod, "sparse_conv3d_nmap_tiled_f32");
    hipModuleGetFunction(&k_scalar, mod, "sparse_conv3d_nmap_f32");

    void *d_feats = hip_upload_raw(feats, (size_t)N*in_C*sizeof(float));
    void *d_nmap  = hip_upload_raw(nmap,  (size_t)N*27*sizeof(uint32_t));
    void *d_w     = hip_upload_raw(weight,(size_t)out_C*27*in_C*sizeof(float));
    void *d_b     = hip_upload_raw(bias,  (size_t)out_C*sizeof(float));
    void *d_out=NULL; hipMalloc(&d_out, (size_t)N*out_C*sizeof(float));

    int use_tiled = (out_C % 64 == 0) && (in_C % 32 == 0);
    void *a[] = {&d_out, &d_feats, &d_nmap, &d_w, &d_b, &in_C, &out_C};
    if (use_tiled) {
        hipModuleLaunchKernel(k_tiled, N, out_C/64, 1, 64, 1, 1, 0, 0, a, NULL);
    } else {
        hipModuleLaunchKernel(k_scalar, N, 1, 1, 256, 1, 1, 0, 0, a, NULL);
    }
    SYNC();

    float *got = malloc((size_t)N*out_C*sizeof(float));
    hipMemcpy(got, d_out, (size_t)N*out_C*sizeof(float), hipMemcpyDeviceToHost);

    double sse=0, sref=0; float mx=0;
    for (size_t i = 0; i < (size_t)N*out_C; i++) {
        double dv = (double)got[i] - ref[i];
        sse += dv*dv; sref += (double)ref[i]*ref[i];
        float a2 = (float)fabs(dv); if (a2 > mx) mx = a2;
    }
    double rel = sqrt(sse / (sref + 1e-30));
    fprintf(stderr, "kernel=%s rel=%.3e max_abs=%.3f  got[0..4]=", use_tiled?"tiled":"scalar", rel, mx);
    for (int i = 0; i < 4; i++) fprintf(stderr, "%+.4f ", got[i]);
    fprintf(stderr, "\n                               ref[0..4]=");
    for (int i = 0; i < 4; i++) fprintf(stderr, "%+.4f ", ref[i]);
    fprintf(stderr, "\n");

    free(got); free(feats); free(nmap); free(weight); free(bias); free(ref);
    return rel < 1e-3 ? 0 : 2;
}
