/* R16 repacked Q4_K/Q5_K decode rows: correctness against the scalar model,
 * an independent float dequantization and the production iqf_rows kernel,
 * then a 47-thread expert-shaped timing (gate/up 512x4096 Q4_K + down
 * 4096x256 Q5_K per part) against iqf_rows on the same weights.
 * run: OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores ./test_glm53f_iq_r16 [parts=5] [pool=96] */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.c"
#include "glm53f_iq_r16.h"
#include <omp.h>

static unsigned rng = 12345;
static unsigned next(void) { return rng = rng * 1664525u + 1013904223u; }
static uint8_t *matrix(int type, int rows, int columns) {
    size_t rb = glm53f_iq_row_size(type, columns), bytes = rb * rows;
    uint8_t *w = malloc(bytes);
    if (!w) return NULL;
    for (size_t i = 0; i < bytes; ++i) w[i] = next() >> 24;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < columns / 256; ++b) {
            uint8_t *p = w + r * rb + b * rb / (columns / 256);
            const float d = 0.0001f * (1 + (next() >> 28)), dm = 0.00005f * (1 + (next() >> 28));
            if (type == GLM53F_GGML_Q4_K) { ((block_q4_K *)p)->d = ggml_fp32_to_fp16(d); ((block_q4_K *)p)->dmin = ggml_fp32_to_fp16(dm); }
            else { ((block_q5_K *)p)->d = ggml_fp32_to_fp16(d); ((block_q5_K *)p)->dmin = ggml_fp32_to_fp16(dm); }
        }
    return w;
}
static void act_from(glm53f_r16_act *a, const iqf_src_block *q, int blocks) {
    for (int b = 0; b < blocks; ++b) {
        memcpy(a[b].q, q[b].q, 256);
        for (int g = 0; g < 8; ++g) { int32_t s = 0; for (int i = 0; i < 32; ++i) s += q[b].q[g * 32 + i]; a[b].s[g] = s; }
        a[b].d = q[b].d;
    }
}
static double rel(const float *a, const float *b, int n) {
    double e = 0, r = 0;
    for (int i = 0; i < n; ++i) { double d = (double)a[i] - b[i]; e += d * d; r += (double)b[i] * b[i]; }
    return sqrt(e / (r + 1e-300));
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int check(void) {
    enum { MAXR = 512 };
    float x[4096];
    for (int i = 0; i < 4096; ++i) x[i] = ((int)(next() >> 16) - 32768) / 32768.f;
    iqf_src_block q[16]; iqf_act ia[16]; glm53f_r16_act ra[16];
    glm5_iq_quant_q8((glm5_iq_q8_block *)q, x, 4096);
    iqf_init(); iqf_prepare(ia, q, 16); act_from(ra, q, 16);
    int fail = 0;
    for (int type = GLM53F_GGML_Q4_K; type <= GLM53F_GGML_Q5_K; ++type)
        for (int blocks = 1; blocks <= 16; blocks *= 4)
            for (int rows = 16; rows <= MAXR; rows *= 8) {
                const int q5 = type == GLM53F_GGML_Q5_K;
                uint8_t *w = matrix(type, rows, blocks * 256);
                const size_t rb = glm53f_iq_row_size(type, blocks * 256);
                uint8_t *r16 = aligned_alloc(256, (glm53f_r16_bytes(rows, blocks, q5) + 255) & ~(size_t)255);
                glm53f_r16_repack(r16, w, rb, rows, blocks, q5);
                static float sve[MAXR], ref[MAXR], prod[MAXR], deq[MAXR];
                glm53f_r16_tiles(sve, r16, 0, rows / 16, blocks, q5, ra);
                for (int t = 0; t < rows / 16; ++t) glm53f_r16_tile_ref(ref + t * 16, r16 + (size_t)t * blocks * glm53f_r16_block_bytes(q5), blocks, q5, ra);
                iqf_rows(prod, w, rb, rows, ia, blocks, q5);
                for (int r = 0; r < rows; ++r) {   /* independent float model from the GGUF bytes */
                    double s = 0;
                    for (int b = 0; b < blocks; ++b) {
                        uint8_t v[256], sc[8], mn[8]; float d, dm;
                        glm53f_r16_decode(w + (size_t)r * rb + (size_t)b * (q5 ? 176 : 144), q5, v, sc, mn, &d, &dm);
                        for (int i = 0; i < 256; ++i)
                            s += ((double)d * sc[i / 32] * v[i] - (double)dm * mn[i / 32]) * q[b].q[i] * q[b].d;
                    }
                    deq[r] = (float)s;
                }
                const double e_ref = rel(sve, ref, rows), e_deq = rel(sve, deq, rows), e_prod = rel(sve, prod, rows);
                const int ok = e_ref < 1e-6 && e_deq < 2e-6 && e_prod < 2e-6;
                printf("R16_CHECK type=%s blocks=%d rows=%d rel_model=%.2e rel_float=%.2e rel_iqf=%.2e %s\n",
                       q5 ? "Q5_K" : "Q4_K", blocks, rows, e_ref, e_deq, e_prod, ok ? "PASS" : "FAIL");
                fail |= !ok;
                free(w); free(r16);
            }
    return fail;
}

int main(int argc, char **argv) {
    const int parts = argc > 1 ? atoi(argv[1]) : 5, pool = argc > 2 ? atoi(argv[2]) : 96, iters = 300;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (check()) { printf("R16_CHECK FAIL\n"); return 1; }
    printf("R16_CHECK ALL PASS\n");
    enum { H = 4096, INTER = 256, GU = 2 * INTER };
    const size_t grb = glm53f_iq_row_size(GLM53F_GGML_Q4_K, H), drb = glm53f_iq_row_size(GLM53F_GGML_Q5_K, INTER);
    uint8_t **gu = malloc(pool * sizeof(*gu)), **dn = malloc(pool * sizeof(*dn)), **rgu = malloc(pool * sizeof(*rgu)), **rdn = malloc(pool * sizeof(*rdn));
    for (int p = 0; p < pool; ++p) {
        gu[p] = matrix(GLM53F_GGML_Q4_K, GU, H); dn[p] = matrix(GLM53F_GGML_Q5_K, H, INTER);
        rgu[p] = aligned_alloc(256, glm53f_r16_bytes(GU, 16, 0)); rdn[p] = aligned_alloc(256, glm53f_r16_bytes(H, 1, 1));
        glm53f_r16_repack(rgu[p], gu[p], grb, GU, 16, 0); glm53f_r16_repack(rdn[p], dn[p], drb, H, 1, 1);
    }
    float x[H], ygu[16][GU], ydn[16][H];
    for (int i = 0; i < H; ++i) x[i] = ((int)(next() >> 16) - 32768) / 32768.f;
    iqf_src_block q[16]; iqf_act ia[16]; glm53f_r16_act ra[16], rd[16];
    glm5_iq_quant_q8((glm5_iq_q8_block *)q, x, H);
    iqf_prepare(ia, q, 16); act_from(ra, q, 16);
    iqf_src_block qd[1]; iqf_act iad[1];
    glm5_iq_quant_q8((glm5_iq_q8_block *)qd, x, INTER);
    iqf_prepare(iad, qd, 1); act_from(rd, qd, 1);
    for (int mode = 0; mode < 2; ++mode) {
        double best = 1e30, sum = 0;
        for (int it = -20; it < iters; ++it) {
            const int span = pool - parts + 1 > 0 ? pool - parts + 1 : 1, base = ((it + 20) * 7 + 13) % span;
            const double t0 = now();
#pragma omp parallel
            {
                const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
                if (mode == 0) {   /* production row kernel, same static split as iq_fast_worker */
                    const int total = parts * GU, q0 = (int)((long long)total * tid / nt), q1 = (int)((long long)total * (tid + 1) / nt);
                    for (int qq = q0; qq < q1;) {
                        const int k = qq / GU, r = qq % GU, n = ((k + 1) * GU < q1 ? (k + 1) * GU : q1) - qq;
                        iqf_rows(ygu[k] + r, gu[base + k] + (size_t)r * grb, grb, n, ia, 16, 0);
                        qq += n;
                    }
#pragma omp barrier
                    const int r0 = (int)((long long)H * tid / nt), r1 = (int)((long long)H * (tid + 1) / nt);
                    for (int k = 0; k < parts; ++k) iqf_rows(ydn[k] + r0, dn[base + k] + (size_t)r0 * drb, drb, r1 - r0, iad, 1, 1);
                } else {           /* R16 tiles, static split of 16-row tiles */
                    const int total = parts * (GU / 16), q0 = (int)((long long)total * tid / nt), q1 = (int)((long long)total * (tid + 1) / nt);
                    for (int qq = q0; qq < q1; ++qq)
                        glm53f_r16_tiles(ygu[qq / (GU / 16)], rgu[base + qq / (GU / 16)], qq % (GU / 16), qq % (GU / 16) + 1, 16, 0, ra);
#pragma omp barrier
                    const int t0 = (int)((long long)(H / 16) * tid / nt), t1 = (int)((long long)(H / 16) * (tid + 1) / nt);
                    for (int k = 0; k < parts; ++k) glm53f_r16_tiles(ydn[k], rdn[base + k], t0, t1, 1, 1, rd);
                }
            }
            const double t = now() - t0;
            if (it >= 0) { sum += t; if (t < best) best = t; }
        }
        const double mb = parts * (glm53f_r16_bytes(GU, 16, 0) + glm53f_r16_bytes(H, 1, 1)) / 1e6;
        printf("R16_TIME mode=%s parts=%d threads=%d mean_us=%.1f best_us=%.1f weights_MB=%.2f GBs=%.0f chk=%g\n",
               mode ? "r16" : "iqf", parts, omp_get_max_threads(), sum / iters * 1e6, best * 1e6, mb, mb / (sum / iters) / 1e3,
               (double)ygu[0][7] + ydn[0][11]);
    }
    return 0;
}
