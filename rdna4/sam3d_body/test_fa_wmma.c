/* Standalone numeric test: flash_attn_tiled_wmma_f32 (BF16-WMMA, hd=64) vs the
 * scalar flash_attn_tiled_f32 ground truth, on synthetic Q/K/V. Isolates the
 * new attention kernel from the full pipeline for fast correctness iteration.
 *
 * Build:  see Makefile target test_fa_wmma
 * Run:    ./test_fa_wmma            (default dinov3 config + a partial-tile case)
 */
#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "../hip_kernels_common.h"
#include "hip_sam3d_body_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef HIP_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)(uintptr_t)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void *)(uintptr_t)0x02)
#define HIP_LAUNCH_PARAM_END            ((void *)(uintptr_t)0x03)
#endif

static int launch(hipFunction_t fn, unsigned gx, unsigned gy, unsigned gz,
                  unsigned bx, unsigned by, unsigned bz, unsigned shmem,
                  void *p, size_t pb) {
    void *cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, p,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &pb, HIP_LAUNCH_PARAM_END };
    hipError_t e = hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, 0, NULL, cfg);
    if (e != hipSuccess) { fprintf(stderr, "launch err=%d\n", (int)e); return -1; }
    return hipDeviceSynchronize() == hipSuccess ? 0 : -1;
}

static int run_case(hipModule_t mod, int n_tok, int n_heads) {
    const int hd = 64;
    const int dim = n_heads * hd;
    const float scale = 1.0f / sqrtf((float)hd);

    hipFunction_t fn_kv, fn_sc, fn_wm;
    if (hipModuleGetFunction(&fn_kv, mod, "kv_transpose") != hipSuccess ||
        hipModuleGetFunction(&fn_sc, mod, "flash_attn_tiled_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_wm, mod, "flash_attn_tiled_wmma_f32") != hipSuccess) {
        fprintf(stderr, "kernel lookup failed (wmma present? gfx12?)\n");
        return -1;
    }

    size_t nq3 = (size_t)n_tok * 3 * dim;
    size_t nqd = (size_t)n_tok * dim;
    size_t nkt = (size_t)n_heads * n_tok * hd;
    float *h_qkv = (float *)malloc(nq3 * sizeof(float));
    float *h_ref = (float *)malloc(nqd * sizeof(float));
    float *h_wm  = (float *)malloc(nqd * sizeof(float));
    unsigned seed = 12345u;
    for (size_t i = 0; i < nq3; i++) {
        seed = seed * 1103515245u + 12345u;
        h_qkv[i] = ((float)((seed >> 16) & 0x7fff) / 16384.0f - 1.0f); /* [-1,1) */
    }

    void *d_qkv, *d_kt, *d_vt, *d_ref, *d_wm;
    hipMalloc(&d_qkv, nq3 * sizeof(float));
    hipMalloc(&d_kt,  nkt * sizeof(float));
    hipMalloc(&d_vt,  nkt * sizeof(float));
    hipMalloc(&d_ref, nqd * sizeof(float));
    hipMalloc(&d_wm,  nqd * sizeof(float));
    hipMemcpy(d_qkv, h_qkv, nq3 * sizeof(float), hipMemcpyHostToDevice);

    /* kv_transpose: K_t,V_t <- qkv. params {K,V,qkv,n_tok,dim,heads,hd}. */
    struct __attribute__((packed)) { void *K,*V; const void *qkv; int n_tok,dim,heads,hd; }
        pkv = { d_kt, d_vt, d_qkv, n_tok, dim, n_heads, hd };
    unsigned total = (unsigned)(n_tok * dim);
    if (launch(fn_kv, (total + 255)/256, 1, 1, 256, 1, 1, 0, &pkv, sizeof(pkv)) < 0) return -1;

    /* scalar FA (ground truth): block 64, grid (heads, ceil(N/64)), shmem 2*16*64*4. */
    struct __attribute__((packed)) { void *out; const void *qkv,*K,*V; int n_tok,dim,heads,hd; float sc; }
        psc = { d_ref, d_qkv, d_kt, d_vt, n_tok, dim, n_heads, hd, scale };
    unsigned gy = (unsigned)((n_tok + 63)/64);
    if (launch(fn_sc, n_heads, gy, 1, 64, 1, 1, 2u*16u*64u*sizeof(float), &psc, sizeof(psc)) < 0) return -1;

    /* WMMA FA: block 128, grid (heads, ceil(N/64)), shmem 0 (static LDS). */
    struct __attribute__((packed)) { void *out; const void *qkv,*K,*V; int n_tok,dim,heads,hd; float sc; }
        pwm = { d_wm, d_qkv, d_kt, d_vt, n_tok, dim, n_heads, hd, scale };
    if (launch(fn_wm, n_heads, gy, 1, 128, 1, 1, 0, &pwm, sizeof(pwm)) < 0) return -1;

    hipMemcpy(h_ref, d_ref, nqd * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_wm,  d_wm,  nqd * sizeof(float), hipMemcpyDeviceToHost);

    double dot=0, na=0, nb=0, maxd=0;
    for (size_t i = 0; i < nqd; i++) {
        double a = h_ref[i], b = h_wm[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > maxd) maxd = d;
    }
    double cos = dot / (sqrt(na)*sqrt(nb) + 1e-30);
    printf("  n_tok=%-5d heads=%-3d hd=64 dim=%-5d : cosine=%.8f  max_abs=%.4e  %s\n",
           n_tok, n_heads, dim, cos, maxd, (cos > 0.999) ? "PASS" : "FAIL");

    free(h_qkv); free(h_ref); free(h_wm);
    hipFree(d_qkv); hipFree(d_kt); hipFree(d_vt); hipFree(d_ref); hipFree(d_wm);
    return (cos > 0.999) ? 0 : -1;
}

int main(void) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "rocew init failed\n"); return 2;
    }
    size_t la = strlen(hip_kernels_common_src), lb = strlen(hip_sam3d_body_kernels_src);
    char *src = (char *)malloc(la + lb + 1);
    memcpy(src, hip_kernels_common_src, la);
    memcpy(src + la, hip_sam3d_body_kernels_src, lb + 1);
    hipModule_t mod;
    int rc = hip_compile_kernels(&mod, 0, src, "fa_wmma_test", 1, "fa_test");
    free(src);
    if (rc < 0) { fprintf(stderr, "HIPRTC compile failed\n"); return 3; }

    printf("flash_attn_tiled_wmma_f32 vs scalar flash_attn_tiled_f32:\n");
    int bad = 0;
    bad |= run_case(mod, 1029, 20);  /* real dinov3 config */
    bad |= run_case(mod,  768, 16);  /* multiple-of-... partial last q-tile */
    bad |= run_case(mod,  130,  2);  /* tiny, partial q + kv tiles */
    bad |= run_case(mod,   64,  1);  /* single full tile */
    printf("%s\n", bad ? "OVERALL: FAIL" : "OVERALL: PASS");
    return bad ? 1 : 0;
}
