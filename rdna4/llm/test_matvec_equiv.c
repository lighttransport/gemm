/*
 * test_matvec_equiv.c - bitwise A/B of the native decode matvec kernels.
 *
 * Compiles two versions of qwen35_matvec_iq.hip and qwen35_matvec_q2k.hip
 * (e.g. `git show HEAD:...` vs the working tree), runs every decode kernel on
 * identical random weights/activations at the Qwen3.8-27B decode shapes and
 * launch geometries, and requires bit-identical outputs.  Also reports the
 * time and effective weight bandwidth of both versions.
 *
 * Usage: test_matvec_equiv old_iq.hip new_iq.hip old_q2k.hip new_q2k.hip [iters]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

/* Read a HIP source, expanding local `#include "..."` lines recursively. */
static char *read_expand(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = malloc((size_t)n + 1);
    if (fread(src, 1, (size_t)n, f) != (size_t)n) { fclose(f); exit(2); }
    src[n] = 0;
    fclose(f);
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) slash[1] = 0; else dir[0] = 0;
    size_t cap = (size_t)n * 4 + 1024, len = 0;
    char *out = malloc(cap);
    for (char *line = src; *line;) {
        char *eol = strchr(line, '\n');
        size_t ll = eol ? (size_t)(eol - line + 1) : strlen(line);
        char inc[512];
        if (sscanf(line, "#include \"%511[^\"]\"", inc) == 1) {
            char ipath[1600];
            snprintf(ipath, sizeof(ipath), "%s%s", dir, inc);
            char *sub = read_expand(ipath);
            size_t sl = strlen(sub);
            while (len + sl + 2 > cap) { cap *= 2; out = realloc(out, cap); }
            memcpy(out + len, sub, sl); len += sl;
            out[len++] = '\n';
            free(sub);
        } else {
            while (len + ll + 1 > cap) { cap *= 2; out = realloc(out, cap); }
            memcpy(out + len, line, ll); len += ll;
        }
        line += ll;
    }
    out[len] = 0;
    free(src);
    return out;
}

typedef struct {
    const char *kernel;
    int q2k;          /* module: 0 = iq, 1 = q2k, 2 = iq1 (main-module IQ1, fast math) */
    int block_bytes;  /* bytes per 256 weights */
    int d_off;        /* f16 scale offset(s) in the block (-1: none, -2: IQ1_M packed) */
    int d_off2;
    int threads;      /* 0: one 256-thread block per row */
    int rows, cols;
    int sig;          /* 0: (y,w,q,s,rows,cols); 1: IQ1 (y,w,q,sd,ss,nr,nc,M=1);
                       * 2: eight-row verifier (y,w,q[8],s[8],rows,cols,count=8) */
} kcase;

/* Launch geometries mirror hip_llm_runner.c (launch_matvec_* native paths). */
static const kcase cases[] = {
    { "qwen35_matvec_iq2xxs", 0, 66, 0, -1, 256, 17408, 5120 },
    { "qwen35_matvec_iq2xxs", 0, 66, 0, -1, 256, 5120, 17408 },
    { "qwen35_matvec_iq2xxs", 0, 66, 0, -1, 256, 5120, 6144 },
    { "qwen35_matvec_iq2xxs", 0, 66, 0, -1, 256, 10240, 5120 },
    { "qwen35_matvec_iq2xs",  0, 74, 0, -1, 512, 17408, 5120 },
    { "qwen35_matvec_iq2xs",  0, 74, 0, -1, 512, 5120, 17408 },
    { "qwen35_matvec_iq2xs",  0, 74, 0, -1, 512, 10240, 5120 },
    { "qwen35_matvec_iq2s",   0, 82, 0, -1, 512, 17408, 5120 },
    { "qwen35_matvec_iq2s",   0, 82, 0, -1, 256, 5120, 17408 },
    { "qwen35_matvec_iq2s",   0, 82, 0, -1, 256, 6144, 5120 },
    { "qwen35_matvec_iq3xxs", 0, 98, 0, -1, 256, 5120, 17408 },
    { "qwen35_matvec_iq3xxs", 0, 98, 0, -1, 256, 17408, 5120 },
    { "qwen35_matvec_iq3s",   0, 110, 0, -1, 128, 5120, 17408 },
    { "qwen35_matvec_iq3s",   0, 110, 0, -1, 128, 17408, 5120 },
    { "qwen35_matvec_iq3s",   0, 110, 0, -1, 256, 6144, 5120 },
    { "qwen35_matvec_iq4xs",  0, 136, 0, -1, 256, 5120, 17408 },
    { "qwen35_matvec_iq4xs",  0, 136, 0, -1, 256, 248320, 5120 },
    { "qwen35_matvec_q2k_rows", 1, 84, 80, 82, 128, 17408, 5120 },
    { "qwen35_matvec_q2k_rows", 1, 84, 80, 82, 256, 12288, 5120 },
    { "qwen35_matvec_q2k",    1, 84, 80, 82, 0, 5120, 17408 },
    { "qwen35_matvec_q2k",    1, 84, 80, 82, 0, 5120, 6144 },
    /* IQ1 Q8_1 kernels (dst,w,q,sd,ss,nr,nc,M=1); geometry from launch_iq1_q81_scalar */
    { "matvec_iq1_s_q81_batch",  2, 50, 0, -1, 256, 17408, 5120, 1 },
    { "matvec_iq1_s_q81_batch",  2, 50, 0, -1, 64, 5120, 17408, 1 },
    { "matvec_iq1_s_mmq_scales", 2, 50, 0, -1, 256, 17408, 5120, 1 },
    { "matvec_iq1_s_mmq_scales", 2, 50, 0, -1, 64, 5120, 17408, 1 },
    { "matvec_iq1_s_mmq_scales", 2, 50, 0, -1, 256, 10240, 5120, 1 },
    { "matvec_iq1_s_mmq_scales", 2, 50, 0, -1, 256, 6144, 5120, 1 },
    { "matvec_iq1_s_mmq_scales", 2, 50, 0, -1, 64, 5120, 6144, 1 },
    { "matvec_iq1_m_q81_batch",  2, 56, -2, -1, 64, 17408, 5120, 1 },
    { "matvec_iq1_m_q81_batch",  2, 56, -2, -1, 512, 5120, 17408, 1 },
    { "matvec_iq1_m_q81_batch",  2, 56, -2, -1, 64, 10240, 5120, 1 },
    { "matvec_iq1_m_q81_batch",  2, 56, -2, -1, 64, 6144, 5120, 1 },
    { "matvec_iq1_m_q81_batch",  2, 56, -2, -1, 512, 5120, 6144, 1 },
    /* DFlash2 K=7 eight-row verifier kernels */
    { "qwen35_matvec_iq2xxs_fixed8", 0, 66, 0, -1, 256, 17408, 5120, 2 },
    { "qwen35_matvec_iq2xxs_fixed8", 0, 66, 0, -1, 256, 5120, 17408, 2 },
    { "qwen35_matvec_iq2xxs_fixed8", 0, 66, 0, -1, 256, 10240, 5120, 2 },
    { "qwen35_matvec_iq2xs_fixed8",  0, 74, 0, -1, 256, 17408, 5120, 2 },
    { "qwen35_matvec_iq2xs_fixed8",  0, 74, 0, -1, 256, 5120, 17408, 2 },
    { "qwen35_matvec_iq2s_fixed8",   0, 82, 0, -1, 256, 17408, 5120, 2 },
    { "qwen35_matvec_iq2s_fixed8",   0, 82, 0, -1, 256, 5120, 6144, 2 },
    { "qwen35_matvec_iq3s_fixed8",   0, 110, 0, -1, 256, 5120, 17408, 2 },
    { "qwen35_matvec_iq3s_fixed8",   0, 110, 0, -1, 256, 17408, 5120, 2 },
    { "qwen35_matvec_iq3xxs_fixed8", 0, 98, 0, -1, 256, 5120, 17408, 2 },
    { "qwen35_matvec_q2k_fixed8",    1, 84, 80, 82, 256, 17408, 5120, 2 },
    { "qwen35_matvec_q2k_fixed8",    1, 84, 80, 82, 256, 6144, 5120, 2 },
    { "qwen35_matvec_q2k_fixed8",    1, 84, 80, 82, 256, 5120, 6144, 2 },
    { "qwen35_matvec_iq4xs_5120_multi8", 0, 136, 0, -1, 256, 248320, 5120, 2 },
};
#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

static uint64_t rs = 0x243f6a8885a308d3ULL;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 32);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s old_iq.hip new_iq.hip old_q2k.hip new_q2k.hip [iters [old_iq1.hip new_iq1.hip]]\n", argv[0]);
        return 2;
    }
    int iters = argc > 5 ? atoi(argv[5]) : 200;
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 1;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(0));
    hipModule_t mod[2][3];
    int nmod = argc > 7 ? 3 : 2;
    for (int v = 0; v < 2; v++)
        for (int m = 0; m < nmod; m++) {
            char *src = read_expand(m < 2 ? argv[1 + m * 2 + v] : argv[6 + v]);
            char name[32];
            snprintf(name, sizeof(name), "mv_equiv_%d%d", v, m);
            /* iq/q2k modules are precise (no fast-math) like the runner's;
             * the IQ1 kernels live in the fast-math main module. */
            if (hip_compile_kernels_ex(&mod[v][m], 0, src, name, 0, name, m < 2 ? 1 : 0) <= 0) return 1;
            free(src);
        }
    hipStream_t st; hipEvent_t e0, e1;
    HIP_CHECK(hipStreamCreate(&st));
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    int fails = 0;
    double tot[2] = {0, 0};
    for (int ci = 0; ci < NCASES; ci++) {
        const kcase *c = &cases[ci];
        const char *only = getenv("MV_ONLY");
        if (only && !strstr(c->kernel, only)) continue;
        hipFunction_t fn[2];
        int ok = c->q2k < nmod;
        for (int v = 0; v < 2 && ok; v++)
            if (hipModuleGetFunction(&fn[v], mod[v][c->q2k], c->kernel) != hipSuccess) ok = 0;
        if (!ok) { printf("skip %s (missing)\n", c->kernel); continue; }
        size_t nb = (size_t)c->rows * (c->cols / 256);
        size_t wbytes = nb * c->block_bytes;
        unsigned char *hw = malloc(wbytes);
        for (size_t i = 0; i < wbytes; i++) hw[i] = (unsigned char)rnd();
        for (size_t b = 0; b < nb; b++) {
            unsigned char *blk = hw + b * c->block_bytes;
            uint16_t h = (uint16_t)(0x1c00 + (rnd() & 0x3ff));   /* ~0.004..0.008 */
            if (c->d_off >= 0) memcpy(blk + c->d_off, &h, 2);
            if (c->d_off == -2) {   /* IQ1_M: scale nibbles in sc[0..3] >> 12 */
                uint16_t sc[4];
                memcpy(sc, blk + 48, 8);
                for (int k = 0; k < 4; k++)
                    sc[k] = (uint16_t)((sc[k] & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12));
                memcpy(blk + 48, sc, 8);
            }
            h = (uint16_t)(0x1800 + (rnd() & 0x3ff));
            if (c->d_off2 >= 0) memcpy(blk + c->d_off2, &h, 2);
        }
        int nact = c->sig == 2 ? 8 : 1;
        signed char *hq = malloc((size_t)c->cols * nact);
        float *hs = malloc(sizeof(float) * (c->cols / 32) * nact);
        for (int i = 0; i < c->cols * nact; i++) hq[i] = (signed char)(rnd() % 255 - 127);
        for (int i = 0; i < c->cols / 32 * nact; i++) hs[i] = (float)(rnd() % 1000 + 1) * 1e-4f;
        /* Rotate through >= 384 MB of weight copies so timing streams from
         * DRAM like real decode instead of hitting the 64 MB Infinity Cache. */
        int ncopy = (int)((384ull << 20) / wbytes) + 1;
        if (ncopy > 64) ncopy = 64;
        void *dwc[64];
        void *dw, *dq, *ds, *dy[2];
        HIP_CHECK(hipMalloc(&dw, wbytes * ncopy));
        for (int i = 0; i < ncopy; i++) {
            dwc[i] = (char *)dw + (size_t)i * wbytes;
            HIP_CHECK(hipMemcpy(dwc[i], hw, wbytes, hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipMalloc(&dq, (size_t)c->cols * nact));
        HIP_CHECK(hipMalloc(&ds, sizeof(float) * (c->cols / 32) * nact));
        void *ds2;
        HIP_CHECK(hipMalloc(&ds2, sizeof(float) * (c->cols / 32)));
        for (int i = 0; i < c->cols / 32; i++) hs[i] = (float)(rnd() % 2000) * 1e-3f - 1.0f;
        HIP_CHECK(hipMemcpy(ds2, hs, sizeof(float) * (c->cols / 32), hipMemcpyHostToDevice));
        for (int i = 0; i < c->cols / 32; i++) hs[i] = (float)(rnd() % 1000 + 1) * 1e-4f;
        size_t ny = (size_t)c->rows * nact;
        HIP_CHECK(hipMalloc(&dy[0], sizeof(float) * ny));
        HIP_CHECK(hipMalloc(&dy[1], sizeof(float) * ny));
        HIP_CHECK(hipMemcpy(dq, hq, (size_t)c->cols * nact, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(ds, hs, sizeof(float) * (c->cols / 32) * nact, hipMemcpyHostToDevice));
        unsigned grid = c->threads ? (unsigned)((c->rows + c->threads / 32 - 1) / (c->threads / 32))
                                   : (unsigned)c->rows;
        unsigned threads = c->threads ? (unsigned)c->threads : 256u;
        /* MV_GRID_CAP=n: launch at most n blocks of the NEW version (kernels
         * that grid-stride over rows). */
        unsigned grid_new = grid, threads_new = threads;
        const char *rpw = getenv("MV_ROWS_PER_WARP");
        if (rpw && atoi(rpw) > 1 && c->threads) {
            int rw = atoi(rpw), wpb = (int)threads / 32;
            grid_new = (unsigned)((c->rows + rw * wpb - 1) / (rw * wpb));
        }
        const char *tn = getenv("MV_THREADS_NEW");
        if (tn && atoi(tn) > 0 && c->threads) {
            threads_new = (unsigned)atoi(tn);
            grid_new = (unsigned)((c->rows + threads_new / 32 - 1) / (threads_new / 32));
        }
        const char *cap = getenv("MV_GRID_CAP");
        if (cap && atoi(cap) > 0 && strstr(c->kernel, getenv("MV_GRID_KERNEL") ? getenv("MV_GRID_KERNEL") : "") &&
            grid_new > (unsigned)atoi(cap)) grid_new = (unsigned)atoi(cap);
        double us[2];
        for (int v = 0; v < 2; v++) {
            void *wcur = dwc[0];
            int one = 1;
            void *a6[] = { &dy[v], &wcur, &dq, &ds, (void *)&c->rows, (void *)&c->cols };
            void *a8[] = { &dy[v], &wcur, &dq, &ds, &ds2, (void *)&c->rows, (void *)&c->cols, &one };
            int eight = 8;
            void *a7[] = { &dy[v], &wcur, &dq, &ds, (void *)&c->rows, (void *)&c->cols, &eight };
            void **a = c->sig == 1 ? a8 : c->sig == 2 ? a7 : a6;
            HIP_CHECK(hipMemset(dy[v], 0xff, sizeof(float) * ny));
            for (int i = 0; i < 20; i++) {
                wcur = dwc[i % ncopy];
                hipModuleLaunchKernel(fn[v], v ? grid_new : grid, 1, 1, v ? threads_new : threads, 1, 1, 0, st, a, NULL);
            }
            hipStreamSynchronize(st);
            float best = 1e30f;
            for (int rep = 0; rep < 3; rep++) {
                hipEventRecord(e0, st);
                for (int i = 0; i < iters; i++) {
                    wcur = dwc[i % ncopy];
                    hipModuleLaunchKernel(fn[v], v ? grid_new : grid, 1, 1, v ? threads_new : threads, 1, 1, 0, st, a, NULL);
                }
                hipEventRecord(e1, st);
                hipEventSynchronize(e1);
                float ms;
                hipEventElapsedTime(&ms, e0, e1);
                if (ms < best) best = ms;
            }
            us[v] = best * 1000.0 / iters;
            tot[v] += us[v];
        }
        float *y0 = malloc(sizeof(float) * ny), *y1 = malloc(sizeof(float) * ny);
        HIP_CHECK(hipMemcpy(y0, dy[0], sizeof(float) * ny, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(y1, dy[1], sizeof(float) * ny, hipMemcpyDeviceToHost));
        int same = memcmp(y0, y1, sizeof(float) * ny) == 0;
        if (!same) {
            fails++;
            double md = 0; int nd = 0;
            for (size_t i = 0; i < ny; i++)
                if (memcmp(&y0[i], &y1[i], 4)) {
                    nd++;
                    double d = fabs((double)y0[i] - y1[i]) / (fabs(y0[i]) + 1e-30);
                    if (d > md) md = d;
                }
            printf("  %d/%d rows differ, max rel %.3g\n", nd, c->rows, md);
        }
        printf("%-24s %6dx%-6d thr=%3d old=%7.2fus (%5.0f GB/s) new=%7.2fus (%5.0f GB/s) %s\n",
               c->kernel, c->rows, c->cols, threads, us[0], wbytes / us[0] * 1e-3,
               us[1], wbytes / us[1] * 1e-3, same ? "SAME" : "DIFF");
        hipFree(dw); hipFree(dq); hipFree(ds); hipFree(ds2); hipFree(dy[0]); hipFree(dy[1]);
        free(hw); free(hq); free(hs); free(y0); free(y1);
    }
    printf("# total old=%.1fus new=%.1fus  %s\n", tot[0], tot[1], fails ? "FAIL" : "ALL SAME");
    return fails ? 1 : 0;
}
