/*
 * test_hip_nmap.c - validate HIP neighbor_map builder against flex_gemm dump.
 *
 * Usage: test_hip_nmap <coords.npy> <ref_nmap.npy>
 * Builds submanifold conv 3x3x3 neighbor_map on HIP from coords, compares
 * against the reference dump from gen_stage2_ref.py --skip-dit --dump-stages
 * (stage0_b0_nbr_cache_neighbor_map.npy). Expects bit-for-bit equality.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
    if (argc < 3) { fprintf(stderr, "usage: %s coords.npy ref_nmap.npy\n", argv[0]); return 1; }

    int cnd, cd[8];
    int32_t *coords = read_npy(argv[1], &cnd, cd, sizeof(int32_t));
    if (!coords) return 1;
    int N = cd[0];
    fprintf(stderr, "coords: N=%d cols=%d\n", N, cd[1]);

    int rnd, rd[8];
    uint32_t *ref = read_npy(argv[2], &rnd, rd, sizeof(uint32_t));
    if (!ref) return 1;
    fprintf(stderr, "ref nmap: shape (%d, %d)\n", rd[0], rd[1]);
    if (rd[0] != N || rd[1] != 27) {
        fprintf(stderr, "shape mismatch\n"); return 1;
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) return 1;
    hipSetDevice(0);
    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, hip_tex_dec_kernels_src, "nmap", 1, "HIP") <= 0) return 1;
    hipFunction_t ins, bldnm;
    hipModuleGetFunction(&ins, mod, "hash_insert_kernel");
    hipModuleGetFunction(&bldnm, mod, "t2_build_nmap_f32");

    void *d_coords = hip_upload_raw(coords, (size_t)N*4*sizeof(int32_t));
    int cap = 1; while (cap < N*2) cap <<= 1; int cap_mask = cap - 1;
    void *d_keys=NULL, *d_vals=NULL;
    hipMalloc(&d_keys, (size_t)cap*sizeof(uint64_t));
    hipMalloc(&d_vals, (size_t)cap*sizeof(int32_t));
    hipMemset(d_keys, 0, (size_t)cap*sizeof(uint64_t));
    hipMemset(d_vals, 0xff, (size_t)cap*sizeof(int32_t));
    {
        void *a[] = {&d_keys, &d_vals, &cap_mask, &d_coords, &N};
        hipModuleLaunchKernel(ins, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
    }

    void *d_nmap=NULL;
    hipMalloc(&d_nmap, (size_t)N*27*sizeof(uint32_t));
    {
        void *a[] = {&d_nmap, &d_coords, &d_keys, &d_vals, &cap_mask, &N};
        int gx = (N + 7) / 8;
        hipModuleLaunchKernel(bldnm, gx, 1, 1, 27, 8, 1, 0, 0, a, NULL); SYNC();
    }

    uint32_t *got = malloc((size_t)N*27*sizeof(uint32_t));
    hipMemcpy(got, d_nmap, (size_t)N*27*sizeof(uint32_t), hipMemcpyDeviceToHost);

    size_t total = (size_t)N * 27;
    size_t mismatches = 0, both_valid_diff = 0, hip_miss = 0, ref_miss = 0;
    for (size_t i = 0; i < total; i++) {
        if (got[i] != ref[i]) {
            mismatches++;
            if (got[i] == 0xFFFFFFFFu) hip_miss++;
            else if (ref[i] == 0xFFFFFFFFu) ref_miss++;
            else both_valid_diff++;
            if (mismatches < 10) {
                int v = (int)(i % 27), idx = (int)(i / 27);
                fprintf(stderr, "  diff [%d, %d] hip=%u ref=%u\n",
                        idx, v, got[i], ref[i]);
            }
        }
    }
    fprintf(stderr, "nmap compare: total=%zu mismatches=%zu (hip_miss=%zu ref_miss=%zu both_diff=%zu)\n",
            total, mismatches, hip_miss, ref_miss, both_valid_diff);

    free(got); free(coords); free(ref);
    hipFree(d_coords); hipFree(d_keys); hipFree(d_vals); hipFree(d_nmap);
    return mismatches == 0 ? 0 : 2;
}
