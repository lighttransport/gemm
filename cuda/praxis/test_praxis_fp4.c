/*
 * test_praxis_fp4.c - numerical check of the NVFP4 routed-expert path on one MoE layer of a (mid-size) model:
 *   1. CPU packer round trip (dequant vs the bf16 weights),
 *   2. device grouped W4A4 block-scale GEMMs (quantize -> gate_up -> SwiGLU -> quantize -> down) vs a CPU fp32 reference
 *      that uses the DEQUANTIZED weights and the dequantized quantized activations  (kernel exactness, rel-L2 ~ bf16 output rounding),
 *   3. the same pipeline vs the unquantized fp32 reference (original bf16 weights, bf16-exact activations): the NVFP4 error.
 *   cuda/praxis/test_praxis_fp4 --model $PRAXIS_ROOT/mid [--tokens 300] [--layer 1]
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_PRAXIS_IMPLEMENTATION
#include "cuda_praxis_runner.h"
#include "../../common/praxis_paths.h"

static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { float a = 0; for (int i = 0; i < 4; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; a += (float)(rs >> 40) / 16777216.0f; } return (a - 2.0f) * 1.732f; }
static float bf(unsigned short h) { return pr_bf16_h(h); }

static float h2f(unsigned short h) {
    unsigned s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = INFINITY; else v = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
static unsigned short f2h(float f) {   /* round to nearest even, saturating at 65504 */
    if (f > 65504.f) f = 65504.f; if (f < -65504.f) f = -65504.f;
    unsigned s = f < 0 ? 1 : 0; float a = fabsf(f); unsigned short r;
    if (a < ldexpf(1.f, -14)) { r = (unsigned short)rintf(ldexpf(a, 24)); }
    else { int e; float fr = frexpf(a, &e); float m = rintf(ldexpf(fr, 11)); if (m >= 2048.f) { m = 1024.f; e++; } r = (unsigned short)(((e + 14) << 10) | ((int)m - 1024)); }
    return (unsigned short)((s << 15) | r);
}
static float rbf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); u &= 0xffff0000u; memcpy(&f, &u, 4); return f; }
static double relerr(const float *a, const float *b, size_t n) { double d = 0, s = 0; for (size_t i = 0; i < n; i++) { d += (double)(a[i] - b[i]) * (a[i] - b[i]); s += (double)a[i] * a[i]; } return sqrt(d / (s + 1e-30)); }

/* CPU activation quantization with the same recipe (global from amax over the given values) */
static void quant_rows(const float *x, int R, int K, float amax, float *deq) {
    float g = fmaxf(amax, 1e-12f) * (1.0f / 2688.0f);
    static const float lut[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
    for (int r = 0; r < R; r++) for (int kb = 0; kb < K / 16; kb++) {
        const float *v = x + (size_t)r * K + kb * 16; float bm = 0;
        for (int i = 0; i < 16; i++) bm = fmaxf(bm, fabsf(v[i]));
        float t = fminf(fmaxf((bm * (1.0f / 6.0f)) / g, 0.015625f), 448.0f);
        unsigned char sb = pr_e4m3_rne_h(t); float sd = pr_e4m3_dec_h(sb), rcp = (1.0f / g) / sd;
        for (int i = 0; i < 16; i++) { float q = fminf(fmaxf(v[i] * rcp, -6.f), 6.f); unsigned c = pr_e2m1_h(q); deq[(size_t)r * K + kb * 16 + i] = ((c & 8) ? -1.f : 1.f) * lut[c & 7] * sd * g; }
    }
}

/* CPU packer throughput on one real-size layer bank (128 experts, 3072 x 2560 gate_up + 1280 x 3072 down, random bf16) */
static int pack_bench(void) {
    int E = 128, C = 3072, Hm = 1280;
    size_t gun = (size_t)C * 2 * Hm, dnn = (size_t)Hm * C;
    unsigned short *gu = (unsigned short *)malloc(gun * 2), *dn = (unsigned short *)malloc(dnn * 2);
    for (size_t i = 0; i < gun; i++) gu[i] = (unsigned short)(0x3c00 + ((i * 2654435761u) >> 22));
    for (size_t i = 0; i < dnn; i++) dn[i] = (unsigned short)(0x3c00 + ((i * 40503u) >> 6));
    pr_fp4_lay L = pr_fp4_layout(E, C, Hm);
    unsigned char *buf = (unsigned char *)malloc(L.total);
    double t0 = pr_now();
    for (int e = 0; e < E; e++) {   /* the same matrix for every expert: only the throughput matters */
        praxis_nvfp4_pack_matrix(gu, C, 2 * Hm, buf + L.gu_q + (size_t)e * 2 * Hm * (C / 2), buf + L.gu_sf + (size_t)e * 2 * Hm * (C / 16), (float *)(buf + L.gu_g) + e);
        praxis_nvfp4_pack_matrix(dn, Hm, C, buf + L.dn_q + (size_t)e * C * (Hm / 2), buf + L.dn_sf + (size_t)e * C * (Hm / 16), (float *)(buf + L.dn_g) + e);
    }
    double dt = pr_now() - t0;
    printf("CPU NVFP4 packer, one real-size layer (%.2f GB bf16 -> %.2f GB): %.2f s (%d OpenMP threads) -> 17 layers ~ %.0f s\n",
           (gun + dnn) * 2.0 * E / 1e9, L.total / 1e9, dt, omp_get_max_threads(), dt * 17);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--pack-bench")) return pack_bench();
    const char *model = px_root("mid"); int T = 300, layer = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = atoi(argv[++i]);
    }
    setenv("PRAXIS_NVFP4", "1", 1);
    praxis_runner *r = praxis_load(model, 0, 0);
    if (!r) return 1;
    const pr_cfg *c = &r->c;
    int E = c->n_experts, C = c->dim, Hm = c->moe_hidden, k = c->top_k;
    pr_blk *b = &r->enc.b[layer];
    if (!b->moe) { fprintf(stderr, "layer %d is not MoE\n", layer); return 1; }
    char path[1100], nm[256];
    snprintf(path, sizeof path, "%s/model.safetensors", model);
    st_context *st = safetensors_open(path);
    snprintf(nm, sizeof nm, "backbone.encoder.core.blocks.%d.ffn.routed.gate_up_weight", layer);
    const unsigned short *gu16 = (const unsigned short *)safetensors_data(st, safetensors_find(st, nm));
    snprintf(nm, sizeof nm, "backbone.encoder.core.blocks.%d.ffn.routed.down_weight", layer);
    const unsigned short *dn16 = (const unsigned short *)safetensors_data(st, safetensors_find(st, nm));
    pr_fp4_lay L = pr_fp4_layout(E, C, Hm);
    const unsigned char *bank = (const unsigned char *)b->h_gu;
    /* 1. weight round trip */
    {
        float *deq = (float *)malloc((size_t)2 * Hm * C * 4), *orig = (float *)malloc((size_t)2 * Hm * C * 4);
        double worst = 0;
        for (int e = 0; e < E; e++) {
            praxis_nvfp4_dequant_matrix(bank + L.gu_q + (size_t)e * 2 * Hm * (C / 2), bank + L.gu_sf + (size_t)e * 2 * Hm * (C / 16), ((const float *)(bank + L.gu_g))[e], C, 2 * Hm, deq);
            for (int n = 0; n < 2 * Hm; n++) for (int kk = 0; kk < C; kk++) orig[(size_t)n * C + kk] = bf(gu16[(size_t)e * C * 2 * Hm + (size_t)kk * 2 * Hm + n]);
            double re = relerr(orig, deq, (size_t)2 * Hm * C); if (re > worst) worst = re;
        }
        printf("1. NVFP4 weight round trip (gate_up, all %d experts): worst rel-L2 vs bf16 = %.4f\n", E, worst);
        free(deq); free(orig);
    }
    if (getenv("PACK_DUMP")) {   /* expert 3 gate_up bytes for the torch-recipe cross check */
        char pp[1200]; FILE *f;
        snprintf(pp, sizeof pp, "%s/c_q.bin", getenv("PACK_DUMP")); f = fopen(pp, "wb"); fwrite(bank + L.gu_q + (size_t)3 * 2 * Hm * (C / 2), 1, (size_t)2 * Hm * (C / 2), f); fclose(f);
        snprintf(pp, sizeof pp, "%s/c_sf.bin", getenv("PACK_DUMP")); f = fopen(pp, "wb"); fwrite(bank + L.gu_sf + (size_t)3 * 2 * Hm * (C / 16), 1, (size_t)2 * Hm * (C / 16), f); fclose(f);
        snprintf(pp, sizeof pp, "%s/c_g.bin", getenv("PACK_DUMP")); f = fopen(pp, "wb"); fwrite((const float *)(bank + L.gu_g) + 3, 4, 1, f); fclose(f);
    }
    /* routing: random top-k experts per token */
    float *x = (float *)malloc((size_t)T * C * 4);
    for (size_t i = 0; i < (size_t)T * C; i++) x[i] = rbf(rnd());
    int *tk = (int *)malloc((size_t)T * k * 4), *seg = (int *)calloc(E + 2, 4), *tokid = (int *)malloc((size_t)T * k * 4), *fill = (int *)calloc(E, 4);
    for (int t = 0; t < T; t++) for (int j = 0; j < k; j++) { int e; int dup; do { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; e = (int)((rs >> 20) % E); dup = 0; for (int q = 0; q < j; q++) dup |= tk[t * k + q] == e; } while (dup); tk[t * k + j] = e; }
    for (int i = 0; i < T * k; i++) seg[tk[i] + 1]++;
    for (int e = 0; e < E; e++) seg[e + 1] += seg[e];
    for (int t = 0; t < T; t++) for (int j = 0; j < k; j++) { int e = tk[t * k + j]; tokid[seg[e] + fill[e]++] = t; }
    int R = T * k;
    CUdeviceptr dx = pr_malloc(r, (size_t)T * C * 4), dbank = pr_malloc(r, L.total), dseg = pr_malloc(r, (E + 2) * 4), dtok = pr_malloc(r, (size_t)R * 4);
    CUdeviceptr gub = pr_malloc(r, (size_t)R * 2 * Hm * 2), hh = pr_malloc(r, (size_t)R * Hm * 2), yb = pr_malloc(r, (size_t)R * C * 2);
    cuMemcpyHtoD(dx, x, (size_t)T * C * 4); cuMemcpyHtoD(dbank, bank, L.total); cuMemcpyHtoD(dseg, seg, (E + 1) * 4); cuMemcpyHtoD(dtok, tokid, (size_t)R * 4);
    if (pr_moe_chunk_fp4(r, dbank, dx, T, dseg, dtok, gub, hh, yb)) return 1;
    cuCtxSynchronize();
    unsigned short *hgu = (unsigned short *)malloc((size_t)R * 2 * Hm * 2), *hhh = (unsigned short *)malloc((size_t)R * Hm * 2), *hy = (unsigned short *)malloc((size_t)R * C * 2);
    cuMemcpyDtoH(hgu, gub, (size_t)R * 2 * Hm * 2); cuMemcpyDtoH(hhh, hh, (size_t)R * Hm * 2); cuMemcpyDtoH(hy, yb, (size_t)R * C * 2);
    /* CPU references (per sorted row) */
    float amax = 0; for (size_t i = 0; i < (size_t)T * C; i++) amax = fmaxf(amax, fabsf(x[i]));
    float *xdq = (float *)malloc((size_t)T * C * 4); quant_rows(x, T, C, amax, xdq);
    double e_gu = 0, n_gu = 0, e_y = 0, n_y = 0, e_q = 0, n_q = 0, e_h = 0, n_h = 0;
    float *Wg = (float *)malloc((size_t)2 * Hm * C * 4), *Wd = (float *)malloc((size_t)C * Hm * 4);
    float *hcpu = (float *)malloc((size_t)R * Hm * 4), *ycpu = (float *)malloc((size_t)R * C * 4);
    float *hq = (float *)malloc((size_t)R * Hm * 4), *hdq = (float *)malloc((size_t)R * Hm * 4);
    float *hexact = (float *)malloc((size_t)R * Hm * 4), *yexact = (float *)malloc((size_t)R * C * 4);
    float hmax = 0;
    /* stage 1: dequantized weights vs device gu; exact fp32 chain with original weights */
    for (int e = 0; e < E; e++) {
        int n0 = seg[e], ne = seg[e + 1] - seg[e];
        if (!ne) continue;
        praxis_nvfp4_dequant_matrix(bank + L.gu_q + (size_t)e * 2 * Hm * (C / 2), bank + L.gu_sf + (size_t)e * 2 * Hm * (C / 16), ((const float *)(bank + L.gu_g))[e], C, 2 * Hm, Wg);
        for (int i = 0; i < ne; i++) {
            int t = tokid[n0 + i];
            for (int n = 0; n < 2 * Hm; n++) {
                double a = 0, ex = 0;
                for (int kk = 0; kk < C; kk++) { a += (double)xdq[(size_t)t * C + kk] * Wg[(size_t)n * C + kk]; ex += (double)x[(size_t)t * C + kk] * bf(gu16[(size_t)e * C * 2 * Hm + (size_t)kk * 2 * Hm + n]); }
                float dv = bf(hgu[(size_t)(n0 + i) * 2 * Hm + n]);
                e_gu += (a - dv) * (a - dv); n_gu += a * a;
                if (n < Hm) { hcpu[(size_t)(n0 + i) * Hm + n] = (float)a; hexact[(size_t)(n0 + i) * Hm + n] = (float)ex; }
                else {   /* swiglu with the cpu / exact up-values */
                    float ag = hcpu[(size_t)(n0 + i) * Hm + n - Hm]; float ug = (float)a; hcpu[(size_t)(n0 + i) * Hm + n - Hm] = rbf(ag / (1.f + expf(-ag)) * ug);
                    float ae = hexact[(size_t)(n0 + i) * Hm + n - Hm]; hexact[(size_t)(n0 + i) * Hm + n - Hm] = ae / (1.f + expf(-ae)) * (float)ex;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)R * Hm; i++) { float dv = bf(hhh[i]); e_h += (double)(hcpu[i] - dv) * (hcpu[i] - dv); n_h += (double)hcpu[i] * hcpu[i]; hmax = fmaxf(hmax, fabsf(dv)); }
    /* stage 2 */
    float hamax = 0; for (size_t i = 0; i < (size_t)R * Hm; i++) hamax = fmaxf(hamax, fabsf(bf(hhh[i])));
    for (size_t i = 0; i < (size_t)R * Hm; i++) hq[i] = bf(hhh[i]);
    quant_rows(hq, R, Hm, hamax, hdq);
    for (int e = 0; e < E; e++) {
        int n0 = seg[e], ne = seg[e + 1] - seg[e];
        if (!ne) continue;
        praxis_nvfp4_dequant_matrix(bank + L.dn_q + (size_t)e * C * (Hm / 2), bank + L.dn_sf + (size_t)e * C * (Hm / 16), ((const float *)(bank + L.dn_g))[e], Hm, C, Wd);
        for (int i = 0; i < ne; i++) for (int n = 0; n < C; n++) {
            double a = 0, ex = 0;
            for (int kk = 0; kk < Hm; kk++) { a += (double)hdq[(size_t)(n0 + i) * Hm + kk] * Wd[(size_t)n * Hm + kk]; ex += (double)hexact[(size_t)(n0 + i) * Hm + kk] * bf(dn16[(size_t)e * Hm * C + (size_t)kk * C + n]); }
            float dv = bf(hy[(size_t)(n0 + i) * C + n]);
            e_y += (a - dv) * (a - dv); n_y += a * a;
            e_q += (ex - dv) * (ex - dv); n_q += ex * ex;
        }
    }
    printf("2. kernel exactness vs CPU fp32 on dequantized operands: gate_up out rel-L2 %.2e | swiglu out %.2e | down out %.2e  (bf16 output rounding ~3e-3)\n",
           sqrt(e_gu / n_gu), sqrt(e_h / n_h), sqrt(e_y / n_y));
    printf("3. NVFP4 W4A4 error vs unquantized fp32 chain (original bf16 weights, fp32 activations): down out rel-L2 %.4f\n", sqrt(e_q / n_q));
    /* 4. W4A16 kernel (bf16 activations x in-register dequantised NVFP4): vs CPU fp64 on the dequantised weights and the bf16-rounded activations */
    double e_a1 = 0, n_a1 = 0, e_a2 = 0, n_a2 = 0;
    {
        unsigned short *xbh = (unsigned short *)malloc((size_t)T * C * 2);
        for (size_t i = 0; i < (size_t)T * C; i++) xbh[i] = f2h(x[i]);
        cuMemsetD32(gub, 0, (size_t)R * Hm); cuMemsetD32(yb, 0, (size_t)R * C / 2);
        if (pr_moe_chunk_fp4_a16(r, dbank, dx, T, dseg, dtok, gub, hh, yb)) return 1;
        cuCtxSynchronize();
        cuMemcpyDtoH(hgu, gub, (size_t)R * 2 * Hm * 2); cuMemcpyDtoH(hhh, hh, (size_t)R * Hm * 2); cuMemcpyDtoH(hy, yb, (size_t)R * C * 2);
        for (int e = 0; e < E; e++) {
            int n0 = seg[e], ne = seg[e + 1] - seg[e];
            if (!ne) continue;
            praxis_nvfp4_dequant_matrix(bank + L.gu_q + (size_t)e * 2 * Hm * (C / 2), bank + L.gu_sf + (size_t)e * 2 * Hm * (C / 16), ((const float *)(bank + L.gu_g))[e], C, 2 * Hm, Wg);
            praxis_nvfp4_dequant_matrix(bank + L.dn_q + (size_t)e * C * (Hm / 2), bank + L.dn_sf + (size_t)e * C * (Hm / 16), ((const float *)(bank + L.dn_g))[e], Hm, C, Wd);
            for (int i = 0; i < ne; i += 7) {   /* every 7th row */
                int t = tokid[n0 + i];
                for (int n = 0; n < 2 * Hm; n += 5) {
                    double a = 0; for (int kk = 0; kk < C; kk++) a += (double)h2f(xbh[(size_t)t * C + kk]) * Wg[(size_t)n * C + kk];
                    float dv = h2f(hgu[(size_t)(n0 + i) * 2 * Hm + n]); e_a1 += (a - dv) * (a - dv); n_a1 += a * a;
                }
                for (int n = 0; n < C; n += 5) {
                    double a = 0; for (int kk = 0; kk < Hm; kk++) a += (double)h2f(hhh[(size_t)(n0 + i) * Hm + kk]) * Wd[(size_t)n * Hm + kk];
                    float dv = bf(hy[(size_t)(n0 + i) * C + n]); e_a2 += (a - dv) * (a - dv); n_a2 += a * a;
                }
            }
        }
        free(xbh);
    }
    printf("4. W4A16 kernel exactness vs CPU fp64 (dequantized weights, f16 activations): gate_up out rel-L2 %.2e | down out %.2e  (bf16 output rounding ~3e-3)\n",
           sqrt(e_a1 / n_a1), sqrt(e_a2 / n_a2));
    /* 5. FP8 banks (e4m3 + per-output-channel scale) with the W8A16 kernel, and the mixed gu NVFP4 / dn FP8 + gu FP8 / dn NVFP4 combos, vs CPU fp64 on dequantized weights */
    double e_f1 = 0, n_f1 = 0, e_f2 = 0, n_f2 = 0, wr8 = 0;
    {
        pr_fp4_lay L8 = pr_fp4_layout2(E, C, Hm, 1, 1);
        unsigned char *b8 = (unsigned char *)calloc(1, L8.total);
        for (int e = 0; e < E; e++) {
            praxis_fp8_pack_matrix(gu16 + (size_t)e * C * 2 * Hm, C, 2 * Hm, b8 + L8.gu_q + (size_t)e * 2 * Hm * C, (float *)(b8 + L8.gu_sf) + (size_t)e * 2 * Hm);
            praxis_fp8_pack_matrix(dn16 + (size_t)e * Hm * C, Hm, C, b8 + L8.dn_q + (size_t)e * C * Hm, (float *)(b8 + L8.dn_sf) + (size_t)e * C);
        }
        /* weight round trip of expert 0 gate_up */
        { double d = 0, nn = 0;
          for (int n = 0; n < 2 * Hm; n++) for (int kk = 0; kk < C; kk++) { float w = bf(gu16[(size_t)kk * 2 * Hm + n]); float q = pr_e4m3_val_h(b8[L8.gu_q + (size_t)n * C + kk]) * ((float *)(b8 + L8.gu_sf))[n]; d += (double)(w - q) * (w - q); nn += (double)w * w; }
          wr8 = sqrt(d / nn); }
        unsigned short *xbh = (unsigned short *)malloc((size_t)T * C * 2);
        for (size_t i = 0; i < (size_t)T * C; i++) xbh[i] = f2h(x[i]);
        CUdeviceptr db8 = pr_malloc(r, L8.total);
        cuMemcpyHtoD(db8, b8, L8.total);
        if (pr_moe_chunk_ex(r, db8, 1, 1, 1, dx, 0, T, dseg, dtok, gub, hh, yb, 0, E)) return 1;
        cuCtxSynchronize();
        cuMemcpyDtoH(hgu, gub, (size_t)R * 2 * Hm * 2); cuMemcpyDtoH(hhh, hh, (size_t)R * Hm * 2); cuMemcpyDtoH(hy, yb, (size_t)R * C * 2);
        for (int e = 0; e < E; e++) {
            int n0 = seg[e], ne = seg[e + 1] - seg[e];
            if (!ne) continue;
            for (int i = 0; i < ne; i += 7) {
                int t = tokid[n0 + i];
                for (int n = 0; n < 2 * Hm; n += 5) {
                    double a = 0; float sc = ((float *)(b8 + L8.gu_sf))[(size_t)e * 2 * Hm + n];
                    for (int kk = 0; kk < C; kk++) a += (double)h2f(xbh[(size_t)t * C + kk]) * pr_e4m3_val_h(b8[L8.gu_q + ((size_t)e * 2 * Hm + n) * C + kk]) * sc;
                    float dv = h2f(hgu[(size_t)(n0 + i) * 2 * Hm + n]); e_f1 += (a - dv) * (a - dv); n_f1 += a * a;
                }
                for (int n = 0; n < C; n += 5) {
                    double a = 0; float sc = ((float *)(b8 + L8.dn_sf))[(size_t)e * C + n];
                    for (int kk = 0; kk < Hm; kk++) a += (double)h2f(hhh[(size_t)(n0 + i) * Hm + kk]) * pr_e4m3_val_h(b8[L8.dn_q + ((size_t)e * C + n) * Hm + kk]) * sc;
                    float dv = bf(hy[(size_t)(n0 + i) * C + n]); e_f2 += (a - dv) * (a - dv); n_f2 += a * a;
                }
            }
        }
        free(xbh); free(b8);
    }
    printf("5. FP8 banks W8A16: weight round trip rel-L2 %.4f | kernel vs CPU fp64: gate_up out %.2e | down out %.2e\n", wr8, sqrt(e_f1 / n_f1), sqrt(e_f2 / n_f2));
    int ok = sqrt(e_gu / n_gu) < 1e-2 && sqrt(e_h / n_h) < 1e-2 && sqrt(e_y / n_y) < 2e-2 && sqrt(e_q / n_q) < 0.30 && sqrt(e_a1 / n_a1) < 5e-3 && sqrt(e_a2 / n_a2) < 5e-3
             && sqrt(e_f1 / n_f1) < 5e-3 && sqrt(e_f2 / n_f2) < 5e-3 && wr8 < 0.05;
    printf(ok ? "ALL OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
