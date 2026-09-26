/* q38p: compute-bound batched prefill for the q38d engine (included into
 * q38d_engine.c, shares its engine state E).
 *
 * Arithmetic: int16 x int16 SDOT (.d), exact int64 accumulation over
 * 512-column blocks, fp64 across blocks.
 *  - Weights: every NVFP4 row of this model spans <= 6 binades of per-16
 *    scales, so a row is exactly int16 (|w| <= 12*15*2^6) times 2^(b_row-21).
 *    32-row x 512-column tiles are expanded from the decode F4 groups on the
 *    fly into an L1 buffer (no second weight copy). The few Q4_K/Q8_K
 *    matrices (ssm alpha/beta, attention k/v) are converted once to int16
 *    tiles with a per-row scale (rounded, <= 2^-16 of the row maximum).
 *  - Activations: int16 per (token, 512-column block).
 * Microkernel q38p_mk_r4t5 (gen_pf.py): 32 rows x 5 tokens, weight tile
 * resident in L1, token groups streamed.
 *
 * Prompt chunks of PF_N tokens run layer by layer over all 48 workers: GEMMs
 * by dynamic 32-row tiles, SSM heads sequentially per token (decode's
 * ssm_head_io, one head per worker), attention per (KV head, token). The KV
 * cache, SSM states and conv histories end in decode's layout; decode then
 * continues from the last prompt token. */

#define PF_R 4          /* weight vectors (x8 rows) per tile */
#define PF_T 5          /* tokens per group */
#define PF_ROWS (PF_R * 8)
#define PF_KB 512       /* columns per K block */
#define PF_STEPS (PF_KB / 4)
void q38p_mk_r4t5(const void *W, const void *A, const double *S, double *O, long ngroups, long steps);

static int pf_bench_depth, pf_bench_synthetic;
static unsigned pf_bench_seed = 1;
static int pf_on, pf_chunk = 240, pf_test, pp_on;
static int pf_attn_cmg = 1;  /* Q38P_ATTN_CMG=0: global query work queue */
static int pf_attn_cache = 1;
static int pf_attn_prof;
static int pf_attn_gemm = 1; /* sequential-FMA QK */
static int pf_attn_pv_gemm, pf_attn_score_pad, pf_attn_pv_int16;
static int pf_attn_pv_residual, pf_attn_pv_p_residual;
#include "q38p_pv_int16.h"
#include "q38p_gemm_attention.h"
static uint64_t pf_attn_times[NT][8];
static int pf_attn_ktile = 32;
static int pf_attn_qtile = 8; /* activated at context >=4096; 1 disables query tiling */
static q38d_line pf_attn_work[NCMG];
static int PF_N, PF_NG;   /* tokens per chunk (multiple of PF_T), groups */
#ifdef Q38P_MPI
#include "q38p_partition.h"
static double pp_attn_cost = 326.6;
static uint64_t pf_t_unit[2 * NLAYER];
static int pp_n = 1, pp_r, pp_prompts = 1, pp_u0, pp_u1, pp_bad, pp_decode;
/* A unit is either the mixer or the FFN of a layer. Cuts may fall between
 * them, which keeps 6-, 8-, 12-, and larger allocations balanced. */
static void pf_stage_bounds(int n, int rank, int *lo, int *hi) {
    double weights[2 * NLAYER];
    for (int u = 0; u < 2 * NLAYER; u++)
        weights[u] = (u & 1) ? 566.1 : (E.L[u / 2].ssm ? 294.8 : pp_attn_cost);
    int cut[Q38P_MAX_STAGES + 1];
    if (!q38p_partition(weights, 2 * NLAYER, n, cut)) state_fail("invalid pipeline partition");
    *lo = cut[rank]; *hi = cut[rank + 1];
}
#endif

/* per matrix prefill descriptor */
typedef struct {
    const q38d_mat *m;
    int rows, cols, ntile;            /* ntile: 32-row tiles over the CMG parts */
    int *tile_c, *tile_r0;            /* CMG part and first local row of each tile */
    float *rscale;                    /* per global row */
    int8_t *rbase;                    /* F4: per global row base exponent */
    int16_t *pre;                     /* non-F4: pre-expanded tiles [tile][kb][PF_STEPS][PF_R][32] */
    const uint8_t **sub[NCMG];        /* F4: (group, K chunk) -> sub-group [g * kch + k] */
} q38p_mat;

typedef struct {
    q38p_mat qkv, z, alpha, beta, out, q, k, v, o, gate, up, down;
} q38p_layer;
static q38p_layer *PL;

/* chunk buffers */
static struct {
    float *x;                 /* residual [N][EMBD] */
    int16_t *aq;              /* activation panel [kb][g][step][T][4], K up to NFF */
    double *as;               /* [kb][g][T] */
    float *y0, *y1, *y2, *y3; /* GEMM outputs (token-major) */
    float *o;                 /* core outputs [N][max(DINNER, NHEAD*HD)] */
    float *qh;                /* attention: normed/roped queries [N][NHEAD][HD] */
    int16_t *wt[NT];          /* per-worker expanded tile */
    double *acc[NT];          /* per-worker fp64 accumulators [g][R][T][8] */
    float *gk[NATTN][NKV];
    int16_t *iv[NATTN][NKV];
    float *ivscale[NATTN][NKV];
    int16_t *ivr[NATTN][NKV];
    float *ivrscale[NATTN][NKV];
    int16_t *ia[NT];
    int16_t *ia_res[NT];
    double *iacc[NT];
    float *gq[NT], *gp[NT], *gb[NT], *gc[NT];
    float *sc[NT];            /* per-worker attention scores */
    int ntok, pos0, ng;       /* current chunk; ng = occupied token groups */
} PB;
static void pf_cache_keys(int tid, int ai, int begin, int end) {
    int c = tid / PER, lane = tid % PER;
    float *dst = PB.gk[ai][c];
    if (!dst) return;
    int b0 = begin / 32, b1 = (end + 31) / 32;
    for (int b = b0 + lane; b < b1; b += PER) {
        int n = E.max_seq - b * 32 < 32 ? E.max_seq - b * 32 : 32;
        q38p_pack_keys(E.kcp[ai][c][0], b * 32, n, dst + (size_t)b * (HD + 1) * 32);
    }
}
static void pf_cache_values(int tid, int ai, int begin, int end) {
    int c = tid / PER, lane = tid % PER;
    int16_t *dst = PB.iv[ai][c];
    if (!dst) return;
    int b0 = begin / Q38I_BK, b1 = (end + Q38I_BK - 1) / Q38I_BK;
    for (int task = lane; task < (b1 - b0) * Q38I_DT; task += PER) {
        int b = b0 + task / Q38I_DT, d = task % Q38I_DT;
        int valid = end - b * Q38I_BK;
        if (valid > Q38I_BK) valid = Q38I_BK;
        q38p_pack_v16(E.vcp[ai][c][0], b, valid, d,
                       dst + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                       PB.ivscale[ai][c] + (size_t)b * Q38I_DT + d);
        if (pf_attn_pv_residual)
            q38p_pack_v16_residual(E.vcp[ai][c][0], b, valid, d,
                                    PB.ivscale[ai][c][(size_t)b * Q38I_DT + d],
                                    PB.ivr[ai][c] + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                                    PB.ivrscale[ai][c] + (size_t)b * Q38I_DT + d);
    }
}
static uint64_t pf_last_hash(void) {
    const uint8_t *p = (const uint8_t *)(PB.x + (size_t)(PB.ntok - 1) * EMBD);
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < EMBD * sizeof(float); i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static _Atomic int pf_ctr[4];
static int pf_calls[NT][16];   /* per-worker GEMM call count (identical on every worker) */

/* ---------------------------------------------------------------- */
/* weight tiles                                                       */

/* (group g, K chunk k) sub-group of a K-chunked matrix: items (g, k) of
 * worker l's range are stored chunk-major (see kchunk_copy_or_write). */
static const uint8_t *pf_sub(const q38d_mat *m, int c, int g, int k) {
    if (m->kch <= 1) return m->part[c] + (size_t)g * q38d_group_bytes(m->fmt, m->cols);
    int it = g * m->kch + k, l, i0 = 0, i1 = 0;
    for (l = 0; l < PER; l++) { lane_items(m, c, l, &i0, &i1); if (it >= i0 && it < i1) break; }
    size_t sb = q38d_group_bytes(m->fmt, m->cols / m->kch);
    int pos = 0;
    for (int kk = 0; kk < m->kch; kk++)
        for (int gg = i0 / m->kch; gg <= (i1 - 1) / m->kch; gg++) {
            int j = gg * m->kch + kk;
            if (j < i0 || j >= i1) continue;
            if (kk == k && gg == g) return m->part[c] + (size_t)(i0 + pos) * sb;
            pos++;
        }
    fprintf(stderr, "q38p: pf_sub miss\n"); exit(1);
}
static inline const uint8_t *pf_subp(const q38p_mat *P, int c, int g, int k) {
    return P->sub[c][g * (P->m->kch > 1 ? P->m->kch : 1) + k];
}
/* scale code and nibble code of (group g of part c, row r, column k) */
static inline void pf_f4_codes(const q38p_mat *P, int c, int g, int r, int k, unsigned *code, unsigned *sc) {
    const q38d_mat *m = P->m;
    int kc = m->kch > 1 ? m->cols / m->kch : m->cols, kk = k / kc;
    const uint8_t *G = pf_subp(P, c, g, m->kch > 1 ? kk : 0);
    int np = kc / 32, p = (k % kc) / 32, nib;
    size_t i = q38d_code_index(r, k % 32, &nib);
    *code = (G[(size_t)p * 128 + i] >> (nib * 4)) & 15;
    *sc = G[(size_t)np * 128 + (size_t)p * 16 + 2 * r + (k % 32) / 16];
}

static void pf_mat_init(q38p_mat *P, const q38d_mat *m) {
    memset(P, 0, sizeof *P);
    P->m = m; P->rows = m->rows; P->cols = m->cols;
    int nt = 0;
    for (int c = 0; c < NCMG; c++) nt += (m->first[c + 1] - m->first[c] + PF_ROWS - 1) / PF_ROWS;
    P->ntile = nt;
    P->tile_c = malloc(nt * sizeof(int)); P->tile_r0 = malloc(nt * sizeof(int));
    nt = 0;
    for (int c = 0; c < NCMG; c++)
        for (int r0 = 0; r0 < m->first[c + 1] - m->first[c]; r0 += PF_ROWS) { P->tile_c[nt] = c; P->tile_r0[nt] = r0; nt++; }
    P->rscale = malloc((size_t)m->rows * sizeof(float));
    if (m->fmt == Q38D_F4) {
        int kch = m->kch > 1 ? m->kch : 1, kc = m->cols / kch, npc = kc / 32;
        for (int c = 0; c < NCMG; c++) {
            int G8 = (m->first[c + 1] - m->first[c] + 7) / 8;
            P->sub[c] = malloc((size_t)G8 * kch * sizeof(uint8_t *));
            for (int g = 0; g < G8; g++) for (int k = 0; k < kch; k++) P->sub[c][g * kch + k] = pf_sub(m, c, g, k);
        }
        P->rbase = malloc(m->rows);
        for (int c = 0; c < NCMG; c++)
            for (int lr = 0; lr < m->first[c + 1] - m->first[c]; lr++) {
                int g = lr / 8, r = lr % 8, mn = 1000, mx = -1000;
                for (int k = 0; k < kch; k++) {
                    const uint8_t *scb = P->sub[c][g * kch + k] + (size_t)npc * 128;
                    for (int pu = 0; pu < 2 * npc; pu++) {
                        unsigned sc = scb[(pu / 2) * 16 + 2 * r + pu % 2];
                        if (sc) { int e = sc >> 3; if (e < mn) mn = e; if (e > mx) mx = e; }
                    }
                }
                if (mn == 1000) mn = mx = 0;
                if (mx - mn > 6) { fprintf(stderr, "q38p: row scale span %d > 6 (int16 not exact)\n", mx - mn); exit(1); }
                int row = m->first[c] + lr;
                P->rbase[row] = (int8_t)mn;
                P->rscale[row] = ldexpf(1.0f, mn - 21);
            }
        return;
    }
    /* other formats: exact fp32 row -> int16 with a per-row scale */
    int nkb = m->cols / PF_KB;
    P->pre = aligned_alloc(256, (size_t)P->ntile * nkb * PF_STEPS * PF_R * 32 * 2);
    float *row = malloc((size_t)m->cols * 4);
    for (int t = 0; t < P->ntile; t++) {
        int c = P->tile_c[t], nrows = m->first[c + 1] - m->first[c];
        for (int q = 0; q < PF_R; q++)
            for (int r = 0; r < 8; r++) {
                int lr = P->tile_r0[t] + q * 8 + r;
                float mx = 0;
                if (lr < nrows) {
                    const uint8_t *G = m->part[c] + (size_t)(lr / 8) * q38d_group_bytes(m->fmt, m->cols);
                    for (int k = 0; k < m->cols; k++) { row[k] = q38d_weight(G, m->fmt, m->cols, lr % 8, k); if (fabsf(row[k]) > mx) mx = fabsf(row[k]); }
                    P->rscale[m->first[c] + lr] = mx / 32767.0f;
                } else memset(row, 0, (size_t)m->cols * 4);
                float inv = mx > 0 ? 32767.0f / mx : 0.f;
                for (int k = 0; k < m->cols; k++) {
                    int kb = k / PF_KB, st = (k % PF_KB) / 4;
                    int16_t v = (int16_t)lrintf(row[k] * inv);
                    P->pre[((((size_t)t * nkb + kb) * PF_STEPS + st) * PF_R + q) * 32 + r * 4 + k % 4] = v;
                }
            }
    }
    free(row);
}

/* Expand tile t, K block kb of an F4 matrix into Wt[PF_STEPS][PF_R][32]
 * (vector q = rows 8q..8q+7, lane r = 4 consecutive columns). */
static void pf_expand_ref(const q38p_mat *P, int t, int kb, int16_t *Wt) {
    const q38d_mat *m = P->m;
    int c = P->tile_c[t], nrows = m->first[c + 1] - m->first[c];
    for (int q = 0; q < PF_R; q++)
        for (int r = 0; r < 8; r++) {
            int lr = P->tile_r0[t] + q * 8 + r;
            for (int kk = 0; kk < PF_KB; kk++) {
                int k = kb * PF_KB + kk, v = 0;
                if (lr < nrows) {
                    unsigned cd, sc;
                    pf_f4_codes(P, c, lr / 8, lr % 8, k, &cd, &sc);
                    if (sc) v = q38d_lut_f4[cd] * (int)((8 + (sc & 7)) << ((int)(sc >> 3) - P->rbase[m->first[c] + lr]));
                }
                Wt[((size_t)(kk / 4) * PF_R + q) * 32 + r * 4 + kk % 4] = (int16_t)v;
            }
        }
}

/* SVE expansion. Per 32-column pair of an 8-row group the F4 codes are two
 * 64-byte halves h, byte 8r + 4u + b holding column 16u + 8*half + 4h + b in
 * nibble `half`; step (4 columns) 4u + 2*half + h of the pair is therefore
 * bytes {8r + 4u + b} of half h, one nibble. */
static uint8_t pf_sel[2][64] __attribute__((aligned(64)));   /* u -> byte index 8r+4u+b at lane 4r+b */
static uint8_t pf_fsel[2][64] __attribute__((aligned(64)));  /* u -> scale byte 2r+u at lane 4r+b */
static int8_t pf_lut8[64] __attribute__((aligned(64)));
static void pf_tables(void) {
    for (int u = 0; u < 2; u++)
        for (int i = 0; i < 64; i++) {
            int r = i / 4, b = i % 4;
            pf_sel[u][i] = i < 32 ? (uint8_t)(8 * r + 4 * u + b) : 255;
            pf_fsel[u][i] = i < 32 ? (uint8_t)(2 * r + u) : 255;
        }
    for (int i = 0; i < 64; i++) pf_lut8[i] = i < 16 ? q38d_lut_f4[i] : 0;
}
static void pf_expand(const q38p_mat *P, int t, int kb, int16_t *Wt) {
    const q38d_mat *m = P->m;
    int c = P->tile_c[t], nrows = m->first[c + 1] - m->first[c];
    const svbool_t pb = svptrue_b8(), ph = svptrue_b16(), p32 = svwhilelt_b8(0, 32);
    const svint8_t lut = svld1_s8(pb, pf_lut8);
    const svuint8_t sel0 = svld1_u8(pb, pf_sel[0]), sel1 = svld1_u8(pb, pf_sel[1]);
    const svuint8_t fs0 = svld1_u8(pb, pf_fsel[0]), fs1 = svld1_u8(pb, pf_fsel[1]);
    int kc = m->kch > 1 ? m->cols / m->kch : m->cols, npc = kc / 32;
    for (int q = 0; q < PF_R; q++) {
        int lr0 = P->tile_r0[t] + q * 8;
        if (lr0 >= nrows) {
            for (int st = 0; st < PF_STEPS; st++) svst1_s16(ph, Wt + ((size_t)st * PF_R + q) * 32, svdup_n_s16(0));
            continue;
        }
        int g = lr0 / 8;
        /* per-row base exponent at lanes 4r+b (int16) */
        int16_t bb[32];
        for (int i = 0; i < 32; i++) bb[i] = P->rbase[m->first[c] + lr0 + i / 4];
        const svint16_t base = svld1_s16(ph, bb);
        for (int pp = 0; pp < PF_KB / 32; pp++) {
            int k = kb * PF_KB + pp * 32, kk = k / kc, p = (k % kc) / 32;
            const uint8_t *G = pf_subp(P, c, g, m->kch > 1 ? kk : 0);
            const uint8_t *cod = G + (size_t)p * 128, *scb = G + (size_t)npc * 128 + (size_t)p * 16;
            svuint8_t sraw = svld1_u8(svwhilelt_b8(0, 16), scb);
            for (int u = 0; u < 2; u++) {
                /* factor (8 + mant) << (e - base), 0 for a zero scale code, at lanes 4r+b */
                svint16_t s16 = svreinterpret_s16_u16(svunpklo_u16(svtbl_u8(sraw, u ? fs1 : fs0)));
                svint16_t e = svsub_s16_x(ph, svasr_n_s16_x(ph, s16, 3), base);
                svint16_t mant = svadd_n_s16_x(ph, svand_n_s16_x(ph, s16, 7), 8);
                svint16_t f = svlsl_s16_x(ph, mant, svreinterpret_u16_s16(e));
                f = svsel_s16(svcmpeq_n_s16(ph, s16, 0), svdup_n_s16(0), f);
                for (int h = 0; h < 2; h++) {
                    svuint8_t raw = svld1_u8(pb, cod + 64 * h);
                    svuint8_t picked = svtbl_u8(raw, u ? sel1 : sel0);
                    for (int half = 0; half < 2; half++) {
                        svuint8_t nib = half ? svlsr_n_u8_x(pb, picked, 4) : svand_n_u8_x(pb, picked, 15);
                        svint8_t v8 = svtbl_s8(lut, nib);
                        svint16_t w = svmul_s16_x(ph, svunpklo_s16(v8), f);
                        int st = pp * 8 + 4 * u + 2 * half + h;
                        svst1_s16(ph, Wt + ((size_t)st * PF_R + q) * 32, w);
                    }
                }
            }
            (void)p32;
        }
    }
}

/* ---------------------------------------------------------------- */
/* activations                                                        */

/* Quantize token t's row x[K] into the panel (int16 per 512-column block). */
static void pf_quant_row(const float *x, int K, int t) {
    const svbool_t pf = svptrue_b32();
    int g = t / PF_T, tt = t % PF_T, nkb = K / PF_KB;
    for (int kb = 0; kb < nkb; kb++) {
        const float *xb = x + kb * PF_KB;
        svfloat32_t mv = svdup_n_f32(0);
        for (int i = 0; i < PF_KB; i += 16) mv = svmax_f32_x(pf, mv, svabs_f32_x(pf, svld1_f32(pf, xb + i)));
        float mx = svmaxv_f32(pf, mv);
        float inv = mx > 0 ? 32767.0f / mx : 0.f;
        PB.as[((size_t)kb * PF_NG + g) * PF_T + tt] = mx > 0 ? (double)mx / 32767.0 : 0.0;
        int16_t *dst = PB.aq + (((size_t)kb * PF_NG + g) * PF_STEPS * PF_T + tt) * 4;
        int16_t tmp[PF_KB] __attribute__((aligned(64)));
        for (int i = 0; i < PF_KB; i += 16) {
            svint32_t qv = svcvt_s32_f32_x(pf, svrintn_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, xb + i), inv)));
            svst1h_s32(pf, tmp + i, qv);
        }
        /* Keep the int16 panel's effective type: uint64_t pointer casts let
         * -O3 discard these stores under strict aliasing. */
        for (int st = 0; st < PF_STEPS; st++)
            memcpy(dst + (size_t)st * PF_T * 4, tmp + 4 * st, 4 * sizeof(int16_t));
    }
}
/* padded tokens of the last chunk: zero rows */
static void pf_quant_zero(int K, int t) {
    int g = t / PF_T, tt = t % PF_T, nkb = K / PF_KB;
    for (int kb = 0; kb < nkb; kb++) {
        PB.as[((size_t)kb * PF_NG + g) * PF_T + tt] = 0.0;
        int16_t *dst = PB.aq + (((size_t)kb * PF_NG + g) * PF_STEPS * PF_T + tt) * 4;
        for (int s = 0; s < PF_STEPS; s++) memset(dst + (size_t)s * PF_T * 4, 0, 8);
    }
}

/* ---------------------------------------------------------------- */
/* GEMM: Y[t][row] = sum_k W[row][k] x_t[k] for all matrices in the list  */

typedef struct { const q38p_mat *P; float *y; int ystride; } pf_job;
static double pf_t_expand[NT], pf_t_mk[NT], pf_t_zero[NT], pf_t_epilogue[NT], pf_t_dispatch[NT], pf_t_barrier[NT];
static int pf_tail_split = 1;
/* Work units: whole tiles in full rounds of NT, then the remaining tiles
 * split into token-group parts so that every worker gets a similar share. */
static void pf_gemm(int tid, const pf_job *jobs, int nj, int K, int *gs) {
    int total = 0;
    for (int j = 0; j < nj; j++) total += jobs[j].P->ntile;
    const int ng = PB.ng;
    int full = pf_tail_split ? total / NT * NT : total, rem = total - full;
    int S = rem ? (NT + rem - 1) / rem : 1;
    if (S > ng) S = ng;
    int units = full + rem * S;
    int ci = pf_calls[tid][0]++;
    _Atomic int *ctr = &pf_ctr[ci & 3];
    if (tid == 0) atomic_store_explicit(&pf_ctr[(ci + 2) & 3], 0, memory_order_relaxed);   /* last used two calls ago */
    int nkb = K / PF_KB;
    double *O = PB.acc[tid];
    int16_t *Wt = PB.wt[tid];
    uint64_t dispatch_start = ticks();
    for (;;) {
        int u = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
        if (u >= units) break;
        int it, g0 = 0, g1 = ng;
        if (u < full) it = u;
        else { int v = u - full, part = v % S; it = full + v / S; g0 = ng * part / S; g1 = ng * (part + 1) / S; }
        if (g1 <= g0) continue;
        int j = 0;
        while (it >= jobs[j].P->ntile) { it -= jobs[j].P->ntile; j++; }
        const q38p_mat *P = jobs[j].P;
        const int gn = g1 - g0;
        uint64_t zero_start = ticks();
        pf_t_dispatch[tid] += zero_start - dispatch_start;
        memset(O, 0, (size_t)gn * PF_R * PF_T * 8 * sizeof(double));
        pf_t_zero[tid] += ticks() - zero_start;
        for (int kb = 0; kb < nkb; kb++) {
            uint64_t a = ticks();
            const int16_t *W;
            if (P->pre) W = P->pre + ((size_t)it * nkb + kb) * PF_STEPS * PF_R * 32;
            else { pf_expand(P, it, kb, Wt); W = Wt; }
            uint64_t b = ticks();
            q38p_mk_r4t5(W, PB.aq + ((size_t)kb * PF_NG + g0) * PF_STEPS * PF_T * 4, PB.as + ((size_t)kb * PF_NG + g0) * PF_T, O, gn, PF_STEPS);
            pf_t_expand[tid] += b - a; pf_t_mk[tid] += ticks() - b;
        }
        /* epilogue: O[g][q][tt][r] * rscale -> Y[g*T+tt][row] */
        uint64_t epilogue_start = ticks();
        const q38d_mat *m = P->m;
        int c = P->tile_c[it], nrows = m->first[c + 1] - m->first[c];
        const svbool_t pd = svptrue_b64();
        int row[PF_R], nr[PF_R], nq = 0;
        for (int q = 0; q < PF_R; q++) {
            int lr = P->tile_r0[it] + q * 8;
            if (lr >= nrows) break;
            row[q] = m->first[c] + lr;
            nr[q] = nrows - lr < 8 ? nrows - lr : 8;
            nq++;
        }
        /* Y is token-major: finish adjacent row vectors before moving to
         * the next token, so their partial cache lines stay resident. */
        for (int g = 0; g < gn; g++)
            for (int tt = 0; tt < PF_T; tt++) {
                int tok = (g0 + g) * PF_T + tt;
                if (tok >= PB.ntok) break;
                float *y = jobs[j].y + (size_t)tok * jobs[j].ystride;
                for (int q = 0; q < nq; q++) {
                    svbool_t pr = svwhilelt_b64(0, nr[q]);
                    svfloat64_t rs = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pr, (const uint32_t *)(P->rscale + row[q]))));
                    svfloat64_t v = svmul_f64_x(pd, svld1_f64(pd, O + (((size_t)g * PF_R + q) * PF_T + tt) * 8), rs);
                    svst1w_u64(pr, (uint32_t *)(y + row[q]), svreinterpret_u64_f32(svcvt_f32_f64_x(pd, v)));
                }
            }
        dispatch_start = ticks();
        pf_t_epilogue[tid] += dispatch_start - epilogue_start;
    }
    uint64_t barrier_start = ticks();
    pf_t_dispatch[tid] += barrier_start - dispatch_start;
    gbarrier(tid, gs);
    pf_t_barrier[tid] += ticks() - barrier_start;
}

/* ---------------------------------------------------------------- */
/* init                                                               */

static float *pf_tmp[NT];     /* per-worker row scratch (NFF floats) */
static double pf_t[16];       /* thread-0 phase times */
enum { PT_EMBED, PT_QUANT, PT_GEMM, PT_SSM, PT_ATTPREP, PT_ATT, PT_ELT, PT_HEAD, PT_N };
static const char *pf_tname[PT_N] = {"embed", "norm+quant", "gemm", "ssm_core", "attn_prep", "attn", "eltwise", "head"};

static void pf_init(int tid, int *gs) {
    if (tid == 0) {
        pf_tables();
        PF_N = (pf_chunk + PF_T - 1) / PF_T * PF_T;
        PF_NG = PF_N / PF_T;
        fprintf(stderr, "q38p_attention: gemm=%d key_cache=%d pv_gemm=%d pv_int16=%d pv_residual=%d p_residual=%d score_pad=%d qtile=%d kv_tile=%d chunk=%d\n",
                pf_attn_gemm, pf_attn_gemm && pf_attn_cache && pp_on,
                pf_attn_pv_gemm, pf_attn_pv_int16, pf_attn_pv_residual,
                pf_attn_pv_p_residual, pf_attn_score_pad, pf_attn_qtile, pf_attn_ktile, PF_N);
        PL = calloc(NLAYER, sizeof *PL);
        PB.x = aligned_alloc(256, (size_t)PF_N * EMBD * 4);
        PB.aq = aligned_alloc(256, (size_t)PF_N * NFF * 2 + 4096);
        PB.as = aligned_alloc(256, (size_t)(NFF / PF_KB) * PF_NG * PF_T * 8 + 256);
        size_t ymax = (size_t)PF_N * NFF;
        PB.y0 = aligned_alloc(256, ymax * 4);
        PB.y1 = aligned_alloc(256, ymax * 4);
        PB.y2 = aligned_alloc(256, (size_t)PF_N * EMBD * 4);
        PB.y3 = aligned_alloc(256, (size_t)PF_N * NVH * 4 + 256);
        PB.o = aligned_alloc(256, (size_t)PF_N * DINNER * 4);
        PB.qh = aligned_alloc(256, (size_t)PF_N * NHEAD * HD * 4);
    }
    PB.wt[tid] = aligned_alloc(256, (size_t)PF_STEPS * PF_R * 32 * 2);
    gbarrier(tid, gs);
    PB.acc[tid] = aligned_alloc(256, (size_t)PF_NG * PF_R * PF_T * 8 * sizeof(double));
    PB.sc[tid] = aligned_alloc(256, ((size_t)pf_attn_qtile * 6 * (E.max_seq + 16 + pf_attn_score_pad) * 4 + 255) & ~(size_t)255);
    pf_tmp[tid] = aligned_alloc(256, (size_t)(NFF + pf_attn_qtile * 6 * HD) * 4);
    if (pf_attn_gemm || pf_attn_pv_gemm) {
        size_t pairs = (pf_attn_qtile + 1) / 2;
        if (pf_attn_gemm) PB.gq[tid] = malloc(pairs * (HD + 1) * 12 * sizeof(float));
        if (pf_attn_pv_gemm) PB.gp[tid] = malloc(pairs * (pf_attn_ktile + 1) * 12 * sizeof(float));
        size_t kn = pf_attn_ktile > HD ? pf_attn_ktile : HD;
        PB.gb[tid] = malloc((kn + 1) * 32 * sizeof(float));
        if (pf_attn_gemm) PB.gc[tid] = malloc(12 * 32 * sizeof(float));
        if ((pf_attn_gemm && (!PB.gq[tid] || !PB.gc[tid])) ||
            (pf_attn_pv_gemm && !PB.gp[tid]) || !PB.gb[tid]) state_fail("packed attention allocation");
    }
    if (pf_attn_pv_int16 && pp_on) {
        PB.ia[tid] = aligned_alloc(256, (size_t)Q38I_AS * sizeof(int16_t));
        if (pf_attn_pv_p_residual)
            PB.ia_res[tid] = aligned_alloc(256, (size_t)Q38I_AS * sizeof(int16_t));
        PB.iacc[tid] = aligned_alloc(256, (size_t)Q38I_DT * Q38I_OS * sizeof(double));
        if (!PB.ia[tid] || (pf_attn_pv_p_residual && !PB.ia_res[tid]) || !PB.iacc[tid])
            state_fail("int16 PV scratch allocation");
    }
#ifdef Q38P_MPI
    if (pf_attn_gemm && pf_attn_cache && pp_on && tid % PER == 0)
        for (int u = pp_u0; u < pp_u1; u++) {
            if ((u & 1) || E.L[u / 2].ssm) continue;
            int ai = E.L[u / 2].ai, c = tid / PER;
            PB.gk[ai][c] = cmg_alloc((size_t)((E.max_seq + 31) / 32) * (HD + 1) * 32 * sizeof(float), c);
        }
    if (pf_attn_pv_int16 && pp_on && tid % PER == 0)
        for (int u = pp_u0; u < pp_u1; u++) {
            if ((u & 1) || E.L[u / 2].ssm) continue;
            int ai = E.L[u / 2].ai, c = tid / PER;
            size_t blocks = ((size_t)E.max_seq + Q38I_BK - 1) / Q38I_BK;
            PB.iv[ai][c] = cmg_alloc(blocks * Q38I_DT * Q38I_WS * sizeof(int16_t), c);
            PB.ivscale[ai][c] = cmg_alloc(blocks * Q38I_DT * sizeof(float), c);
            if (pf_attn_pv_residual) {
                PB.ivr[ai][c] = cmg_alloc(blocks * Q38I_DT * Q38I_WS * sizeof(int16_t), c);
                PB.ivrscale[ai][c] = cmg_alloc(blocks * Q38I_DT * sizeof(float), c);
            }
        }
#endif
    /* matrices, round-robin over the workers */
    int item = 0;
    for (int l = 0; l < NLAYER; l++) {
        const q38d_layer *L = &E.L[l];
        q38p_layer *P = &PL[l];
        struct { q38p_mat *p; const q38d_mat *m; } it[9];
        int n = 0;
        if (L->ssm) { it[n++] = (typeof(it[0])){&P->qkv, &L->qkv}; it[n++] = (typeof(it[0])){&P->z, &L->z};
                      it[n++] = (typeof(it[0])){&P->alpha, &L->alpha}; it[n++] = (typeof(it[0])){&P->beta, &L->beta};
                      it[n++] = (typeof(it[0])){&P->out, &L->out}; }
        else { it[n++] = (typeof(it[0])){&P->q, &L->q}; it[n++] = (typeof(it[0])){&P->k, &L->k};
               it[n++] = (typeof(it[0])){&P->v, &L->v}; it[n++] = (typeof(it[0])){&P->o, &L->o}; }
        it[n++] = (typeof(it[0])){&P->gate, &L->gate}; it[n++] = (typeof(it[0])){&P->up, &L->up};
        it[n++] = (typeof(it[0])){&P->down, &L->down};
        for (int i = 0; i < n; i++, item++) {
#ifdef Q38P_MPI
            if (pp_on) {
                int unit = 2 * l + (i >= n - 3);
                if (unit < pp_u0 || unit >= pp_u1) continue;
            }
#endif
            if (item % NT == tid) pf_mat_init(it[i].p, it[i].m);
        }
    }
    gbarrier(tid, gs);
}

/* exactness self-test of the tile expansion (Q38P_TEST=1) */
static void pf_selftest(void) {
    const q38p_mat *T[3] = {&PL[0].qkv, &PL[3].o, &PL[NLAYER - 1].down};
    const char *nm[3] = {"L0.qkv", "L3.o", "L63.down"};
    int16_t *a = aligned_alloc(256, (size_t)PF_STEPS * PF_R * 64), *b = aligned_alloc(256, (size_t)PF_STEPS * PF_R * 64);
    for (int i = 0; i < 3; i++) {
        const q38p_mat *P = T[i];
        long bad = 0, badw = 0, n = 0;
        int nkb = P->cols / PF_KB;
        for (int t = 0; t < P->ntile; t += P->ntile / 7 + 1)
            for (int kb = 0; kb < nkb; kb += nkb / 5 + 1) {
                pf_expand(P, t, kb, a); pf_expand_ref(P, t, kb, b);
                for (int e = 0; e < PF_STEPS * PF_R * 32; e++) if (a[e] != b[e]) bad++;
                /* reference against the decode's exact weight */
                const q38d_mat *m = P->m;
                int c = P->tile_c[t], kch = m->kch > 1 ? m->kch : 1, kc = m->cols / kch;
                for (int q = 0; q < PF_R; q++)
                    for (int r = 0; r < 8; r++) {
                        int lr = P->tile_r0[t] + q * 8 + r;
                        if (lr >= m->first[c + 1] - m->first[c]) continue;
                        for (int kk = 0; kk < PF_KB; kk += 7) {
                            int k = kb * PF_KB + kk;
                            const uint8_t *G = pf_subp(P, c, lr / 8, k / kc);
                            float w = q38d_weight(G, m->fmt, kc, lr % 8, k % kc);
                            float v = (float)b[((size_t)(kk / 4) * PF_R + q) * 32 + r * 4 + kk % 4] * P->rscale[m->first[c] + lr];
                            if (w != v) badw++;
                            n++;
                        }
                    }
            }
        fprintf(stderr, "q38p selftest %s (kch %d): SVE vs ref mismatches %ld, ref vs decode weight mismatches %ld of %ld\n",
                nm[i], P->m->kch, bad, badw, n);
    }
    free(a); free(b);
}

/* ---------------------------------------------------------------- */
/* chunk pipeline                                                     */

static void pf_norm_quant(int tid, const float *w, int *gs) {
    uint64_t a = ticks();
    float *tmp = pf_tmp[tid];
    const svbool_t pf = svptrue_b32();
    for (int t = tid; t < PB.ng * PF_T; t += NT) {
        if (t >= PB.ntok) { pf_quant_zero(EMBD, t); continue; }
        const float *x = PB.x + (size_t)t * EMBD;
        float inv = 1.0f / sqrtf(sumsq(x, EMBD) / EMBD + E.eps);
        for (int i = 0; i < EMBD; i += 16)
            svst1_f32(pf, tmp + i, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, x + i), inv), svld1_f32(pf, w + i)));
        pf_quant_row(tmp, EMBD, t);
    }
    gbarrier(tid, gs);
    if (!tid) pf_t[PT_QUANT] += ticks() - a;
}
static void pf_quant_rows(int tid, const float *y, int K, int stride, int *gs) {
    uint64_t a = ticks();
    for (int t = tid; t < PB.ng * PF_T; t += NT) {
        if (t >= PB.ntok) pf_quant_zero(K, t);
        else pf_quant_row(y + (size_t)t * stride, K, t);
    }
    gbarrier(tid, gs);
    if (!tid) pf_t[PT_QUANT] += ticks() - a;
}
static void pf_gemm_t(int tid, const pf_job *j, int nj, int K, int *gs) {
    uint64_t a = ticks();
    pf_gemm(tid, j, nj, K, gs);
    if (!tid) pf_t[PT_GEMM] += ticks() - a;
}
static void pf_residual(int tid, const float *y, int *gs) {
    uint64_t a = ticks();
    const svbool_t pf = svptrue_b32();
    for (int t = tid; t < PB.ntok; t += NT) {
        float *x = PB.x + (size_t)t * EMBD;
        const float *yr = y + (size_t)t * EMBD;
        for (int i = 0; i < EMBD; i += 16) svst1_f32(pf, x + i, svadd_f32_x(pf, svld1_f32(pf, x + i), svld1_f32(pf, yr + i)));
    }
    gbarrier(tid, gs);
    if (!tid) pf_t[PT_ELT] += ticks() - a;
}

/* One deep PP query per stage: split the packed-KV scan over all 12 workers
 * of each KV-head CMG, then merge their stable softmax partials. */
static void pf_i6_attention_single(int tid, int ai, int pos, int *gs) {
    int c = tid / PER, lane = tid % PER, extent = pos + 1;
    int t0 = (int)((int64_t)extent * lane / PER);
    int t1 = (int)((int64_t)extent * (lane + 1) / PER);
    int nt = t1 - t0;
    float *part = E.apart[c][0] + (size_t)lane * 6 * (2 + HD);
    float *sc = PB.sc[tid];
    const float *qh = PB.qh + (size_t)6 * c * HD;
    const svbool_t pf = svptrue_b32();
    if (nt) {
        attention_scores_i6(qh, (const uint8_t *)KI.k[ai][c][0], KI.ks[ai][c][0],
                            t0, t1, sc, nt, 1.0f / 16.0f);
        for (int h = 0; h < 6; h++) {
            float *s_h = sc + (size_t)h * nt;
            float m = q38d_score_max(s_h, nt, score_max_vector);
            svfloat32_t lsv = svdup_n_f32(0);
            for (int i = 0; i < nt; i += 16) {
                svbool_t pg = svwhilelt_b32(i, nt);
                svfloat32_t e = q38d_exp_sve(pg, svsub_n_f32_x(pg, svld1_f32(pg, s_h + i), m));
                svst1_f32(pg, s_h + i, e);
                lsv = svadd_f32_m(pg, lsv, e);
            }
            float *p = part + (size_t)h * (2 + HD);
            p[0] = m;
            p[1] = svaddv_f32(pf, lsv);
        }
        attention_pv_i6((const uint8_t *)KI.v[ai][c][0], KI.vs[ai][c][0],
                        t0, t1, sc, nt, part + 2, 2 + HD);
    } else for (int h = 0; h < 6; h++) {
        float *p = part + (size_t)h * (2 + HD);
        p[0] = -INFINITY; p[1] = 0; memset(p + 2, 0, HD * sizeof(float));
    }
    gbarrier(tid, gs);
    if (tid == 0) {
        for (int head = 0; head < NKV; head++)
            for (int h = 0; h < 6; h++) {
                float m = -INFINITY, denom = 0.0f, acc[HD] = {0};
                for (int l = 0; l < PER; l++) {
                    float *p = E.apart[head][0] + ((size_t)l * 6 + h) * (2 + HD);
                    if (p[0] > m) m = p[0];
                }
                for (int l = 0; l < PER; l++) {
                    float *p = E.apart[head][0] + ((size_t)l * 6 + h) * (2 + HD);
                    float weight = p[1] > 0 ? expf(p[0] - m) : 0.0f;
                    denom += weight * p[1];
                    for (int d = 0; d < HD; d++) acc[d] = fmaf(weight, p[d + 2], acc[d]);
                }
                int hq = head * 6 + h;
                const float *gate = PB.y0 + (size_t)hq * 2 * HD + HD;
                float *dst = PB.o + (size_t)hq * HD;
                float inv = 1.0f / denom;
                for (int d = 0; d < HD; d += 16)
                    svst1_f32(pf, dst + d,
                              svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, acc + d), inv),
                                          q38d_sigmoid_sve(pf, svld1_f32(pf, gate + d))));
            }
    }
    gbarrier(tid, gs);
}

/* Reuse each packed row across adjacent causal queries. The per-query score
 * and PV accumulation order remains increasing in key position. */
static void pf_i6_scores_tile(int ai, int c, int tb, int te, int pos0,
                              float *sc, int stride) {
    int extent = pos0 + te;
    const uint8_t *k = (const uint8_t *)KI.k[ai][c][0];
    const float *ks = KI.ks[ai][c][0];
    size_t row_bytes = kv_row_bytes();
    for (int key = 0; key < extent; key++) {
        float unpacked[HD];
        q38d_kv_i6_unpack_f32_256(k + (size_t)key * row_bytes, unpacked);
        int first = key > pos0 + tb ? key - pos0 : tb;
        for (int t = first; t < te; t++) {
            const float *q = PB.qh + ((size_t)t * NHEAD + 6 * c) * HD;
            float *s = sc + (size_t)(t - tb) * 6 * stride + key;
            for (int h = 0; h < 6; h++)
                s[(size_t)h * stride] = dot(q + (size_t)h * HD, unpacked, HD) * (ks[key] * (1.0f / 16.0f));
        }
    }
}
static void pf_i6_pv_tile(int ai, int c, int tb, int te, int pos0,
                          const float *sc, int stride, float *out) {
    int extent = pos0 + te;
    const uint8_t *v = (const uint8_t *)KI.v[ai][c][0];
    const float *vs = KI.vs[ai][c][0];
    size_t row_bytes = kv_row_bytes();
    const svbool_t pf = svptrue_b32();
    memset(out, 0, (size_t)(te - tb) * 6 * HD * sizeof(float));
    for (int key = 0; key < extent; key++) {
        float unpacked[HD];
        q38d_kv_i6_unpack_f32_256(v + (size_t)key * row_bytes, unpacked);
        int first = key > pos0 + tb ? key - pos0 : tb;
        for (int t = first; t < te; t++) {
            const float *s = sc + (size_t)(t - tb) * 6 * stride + key;
            float *dst = out + (size_t)(t - tb) * 6 * HD;
            for (int h = 0; h < 6; h++) {
                float w = s[(size_t)h * stride] * vs[key];
                for (int d = 0; d < HD; d += 16)
                    svst1_f32(pf, dst + (size_t)h * HD + d,
                              svmla_n_f32_x(pf, svld1_f32(pf, dst + (size_t)h * HD + d),
                                              svld1_f32(pf, unpacked + d), w));
            }
        }
    }
}

static void pf_layer_run(int tid, int layer, int do_mixer, int do_ffn, int *gs) {
    const q38d_layer *L = &E.L[layer];
    q38p_layer *P = &PL[layer];
    const int N = PB.ntok, pos0 = PB.pos0;
#ifdef Q38P_MPI
    uint64_t unit_start = pp_on && pf_attn_prof && tid == 0 ? ticks() : 0;
#endif
    if (do_mixer) {
        pf_norm_quant(tid, L->attn_norm, gs);
        if (L->ssm) {
            pf_job j[4] = {{&P->qkv, PB.y0, QKVD}, {&P->z, PB.y1, DINNER}, {&P->alpha, PB.y2, NVH}, {&P->beta, PB.y3, NVH}};
            pf_gemm_t(tid, j, 4, EMBD, gs);
            uint64_t a = ticks();
            int hh = ssm_head_of(tid);
            if (hh >= 0)
                for (int t = 0; t < N; t++) {
                    q38d_io io = {PB.y0 + (size_t)t * QKVD, PB.y1 + (size_t)t * DINNER, PB.y2 + (size_t)t * NVH, PB.y3 + (size_t)t * NVH,
                                  NULL, NULL, NULL, PB.o + (size_t)t * DINNER, NULL, -1};
                    ssm_head_io(layer, hh, pos0 + t, &io);
                }
            gbarrier(tid, gs);
            if (!tid) pf_t[PT_SSM] += ticks() - a;
            pf_quant_rows(tid, PB.o, DINNER, DINNER, gs);
            pf_job jo = {&P->out, PB.y2, EMBD};
            pf_gemm_t(tid, &jo, 1, DINNER, gs);
            pf_residual(tid, PB.y2, gs);
        } else {
            pf_job j[3] = {{&P->q, PB.y0, NHEAD * 2 * HD}, {&P->k, PB.y1, NKV * HD}, {&P->v, PB.y2, NKV * HD}};
            pf_gemm_t(tid, j, 3, EMBD, gs);
            uint64_t a = ticks();
            int ai = L->ai;
            /* q/k norm + RoPE, K/V into the cache (cpk = 1: kcp[..][0] is position-major) */
            for (int t = tid; t < N; t += NT) {
                int pos = pos0 + t;
                for (int hq = 0; hq < NHEAD; hq++) {
                    float *qd = PB.qh + ((size_t)t * NHEAD + hq) * HD;
                    memcpy(qd, PB.y0 + (size_t)t * NHEAD * 2 * HD + (size_t)hq * 2 * HD, HD * 4);
                    head_rmsnorm_rope(qd, L->q_norm, pos);
                }
                for (int c = 0; c < NKV; c++) {
                    float kv[HD];
                    memcpy(kv, PB.y1 + (size_t)t * NKV * HD + c * HD, sizeof kv);
                    head_rmsnorm_rope(kv, L->k_norm, pos);
                    if (pf_kv_i6_mode) {
                        if (q38d_kv_i6_pack_row(kv,
                                                 (uint8_t *)KI.k[ai][c][0] + (size_t)pos * kv_row_bytes(),
                                                 HD, KI.ks[ai][c][0] + pos) ||
                            q38d_kv_i6_pack_row(PB.y2 + (size_t)t * NKV * HD + c * HD,
                                                 (uint8_t *)KI.v[ai][c][0] + (size_t)pos * kv_row_bytes(),
                                                 HD, KI.vs[ai][c][0] + pos))
                            state_fail("nonfinite PP INT6 KV row");
                    } else {
                        memcpy(E.kcp[ai][c][0] + (size_t)pos * HD, kv, sizeof kv);
                        memcpy(E.vcp[ai][c][0] + (size_t)pos * HD,
                               PB.y2 + (size_t)t * NKV * HD + c * HD, HD * 4);
                    }
                }
            }
            if (pf_attn_cmg && tid % PER == 0)
                atomic_store_explicit(&pf_attn_work[tid / PER].v, 0, memory_order_relaxed);
            gbarrier(tid, gs);
            if (pf_attn_gemm && PB.gk[ai][tid / PER]) {
                pf_cache_keys(tid, ai, pos0, pos0 + N);
                gbarrier(tid, gs);
            }
            if (pf_attn_pv_int16 && PB.iv[ai][tid / PER]) {
                pf_cache_values(tid, ai, pos0, pos0 + N);
                gbarrier(tid, gs);
            }
            uint64_t b = ticks();
            if (!tid) pf_t[PT_ATTPREP] += b - a;
            if (pf_kv_i6_mode && pf_i6_parallel && N == 1 && pos0 >= 4096) {
                pf_i6_attention_single(tid, ai, pos0, gs);
            } else {
                /* causal attention: tasks (KV head, token), largest contexts first */
                int ci = pf_calls[tid][0]++;
                _Atomic int *ctr = &pf_ctr[ci & 3];
                if (tid == 0) atomic_store_explicit(&pf_ctr[(ci + 2) & 3], 0, memory_order_relaxed);
                float *sc = PB.sc[tid], *out = pf_tmp[tid];
                const svbool_t pf = svptrue_b32();
                const int qt = pos0 + N < 4096 ? 1 : pf_attn_qtile;
                const int groups = (N + qt - 1) / qt;
                for (;;) {
                    int task = atomic_fetch_add_explicit(pf_attn_cmg ? &pf_attn_work[tid / PER].v : ctr,
                                                         1, memory_order_relaxed);
                    if (task >= (pf_attn_cmg ? groups : NKV * groups)) break;
                    int te = N - (pf_attn_cmg ? task : task / NKV) * qt;
                    int tb = te > qt ? te - qt : 0;
                    int c = pf_attn_cmg ? tid / PER : task % NKV, extent = pos0 + te;
                    int stride = extent + pf_attn_score_pad;
                    const float *K = E.kcp[ai][c][0], *V = E.vcp[ai][c][0];
                    /* Share the same KV cache tile across adjacent queries. The
                     * optional GEMM uses sequential QK FMAs; PV order is unchanged. */
                    uint64_t profile_start = pf_attn_prof ? ticks() : 0;
                    int block = qt == 1 ? extent : pf_attn_ktile;
                    if (pf_kv_i6_mode) {
                        pf_i6_scores_tile(ai, c, tb, te, pos0, sc, stride);
                    } else if (pf_attn_gemm) {
                        q38p_pack_queries(PB.qh + ((size_t)tb * NHEAD + 6 * c) * HD, te - tb, NHEAD * HD, PB.gq[tid]);
                        q38p_scores_gemm(K, 0, extent, PB.gq[tid], te - tb, sc, stride, PB.gb[tid], PB.gc[tid], PB.gk[ai][c]);
                    } else for (int k0 = 0; k0 < extent; k0 += block)
                        for (int t = tb; t < te; t++) {
                            int n = pos0 + t + 1, k1 = k0 + block < n ? k0 + block : n;
                            if (k0 < k1)
                                attn_scores_prefill(PB.qh + ((size_t)t * NHEAD + 6 * c) * HD,
                                            K, k0, k1, sc + (size_t)(t - tb) * 6 * stride + k0,
                                            stride, 1.0f / 16.0f);
                        }
                    if (pf_attn_prof) { uint64_t now = ticks(); pf_attn_times[tid][0] += now - profile_start; profile_start = now; }
                    float sums[16][6];
                    for (int t = tb; t < te; t++)
                        for (int hh = 0; hh < 6; hh++) {
                            int n = pos0 + t + 1;
                            float *s_h = sc + ((size_t)(t - tb) * 6 + hh) * stride;
                            float m = q38d_score_max(s_h, n, score_max_vector);
                            svfloat32_t lsv = svdup_n_f32(0);
                            for (int i = 0; i < n; i += 16) {
                                svbool_t pg = svwhilelt_b32(i, n);
                                svfloat32_t e = q38d_exp_sve(pg, svsub_n_f32_x(pg, svld1_f32(pg, s_h + i), m));
                                svst1_f32(pg, s_h + i, e);
                                lsv = svadd_f32_m(pg, lsv, e);
                            }
                            sums[t - tb][hh] = svaddv_f32(pf, lsv);
                        }
                    if (pf_attn_prof) { uint64_t now = ticks(); pf_attn_times[tid][1] += now - profile_start; profile_start = now; }
                    if (pf_kv_i6_mode) {
                        pf_i6_pv_tile(ai, c, tb, te, pos0, sc, stride, out);
                    } else if (pf_attn_pv_int16) {
                        q38p_pv_int16_eval(PB.iv[ai][c], PB.ivscale[ai][c],
                                           PB.ivr[ai][c], PB.ivrscale[ai][c], sc, stride,
                                           te - tb, pos0 + tb, out, PB.ia[tid],
                                           PB.ia_res[tid], PB.iacc[tid]);
                    } else {
                    block = pf_attn_pv_gemm ? pf_attn_ktile : qt == 1 ? extent : pf_attn_ktile;
                    for (int k0 = 0; k0 < extent; k0 += block)
                        if (pf_attn_pv_gemm) {
                            int k1 = k0 + block < extent ? k0 + block : extent;
                            q38p_pv_gemm(V, k0, k1, sc, stride, te - tb,
                                          pos0 + tb, out, PB.gp[tid], PB.gb[tid]);
                        } else for (int t = tb; t < te; t++) {
                            int n = pos0 + t + 1, k1 = k0 + block < n ? k0 + block : n;
                            if (k0 < k1)
                                q38d_attn_pv6_range(V, k0, k1,
                                    sc + (size_t)(t - tb) * 6 * stride + k0, stride,
                                    out + (size_t)(t - tb) * 6 * HD, HD, att_pv_block, k0 == 0);
                        }
                    }
                    if (pf_attn_prof) pf_attn_times[tid][2] += ticks() - profile_start;
                    for (int t = tb; t < te; t++)
                        for (int hh = 0; hh < 6; hh++) {
                            int hq = 6 * c + hh;
                            const float *gate = PB.y0 + (size_t)t * NHEAD * 2 * HD + (size_t)hq * 2 * HD + HD;
                            float *dst = PB.o + (size_t)t * NHEAD * HD + (size_t)hq * HD;
                            const float *src = out + ((size_t)(t - tb) * 6 + hh) * HD;
                            float rl = 1.0f / sums[t - tb][hh];
                            for (int i = 0; i < HD; i += 16)
                                svst1_f32(pf, dst + i, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, src + i), rl),
                                                                 q38d_sigmoid_sve(pf, svld1_f32(pf, gate + i))));
                        }
                }
                gbarrier(tid, gs);
            }
            if (!tid) pf_t[PT_ATT] += ticks() - b;
            pf_quant_rows(tid, PB.o, NHEAD * HD, NHEAD * HD, gs);
            pf_job jo = {&P->o, PB.y2, EMBD};
            pf_gemm_t(tid, &jo, 1, NHEAD * HD, gs);
            pf_residual(tid, PB.y2, gs);
        }
    }
#ifdef Q38P_MPI
    if (unit_start && do_mixer) { pf_t_unit[2 * layer] += ticks() - unit_start; unit_start = ticks(); }
#endif
    /* FFN */
    if (do_ffn) {
        pf_norm_quant(tid, L->post_norm, gs);
        pf_job jf[2] = {{&P->gate, PB.y0, NFF}, {&P->up, PB.y1, NFF}};
        pf_gemm_t(tid, jf, 2, EMBD, gs);
        {
            uint64_t a = ticks();
            const svbool_t pf = svptrue_b32();
            for (int t = tid; t < PB.ng * PF_T; t += NT) {
                if (t >= N) { pf_quant_zero(NFF, t); continue; }
                float *g = PB.y0 + (size_t)t * NFF;
                const float *u = PB.y1 + (size_t)t * NFF;
                for (int i = 0; i < NFF; i += 16) {
                    svfloat32_t gv = svld1_f32(pf, g + i);
                    svst1_f32(pf, g + i, svmul_f32_x(pf, svmul_f32_x(pf, gv, q38d_sigmoid_sve(pf, gv)), svld1_f32(pf, u + i)));
                }
                pf_quant_row(g, NFF, t);
            }
            gbarrier(tid, gs);
            if (!tid) pf_t[PT_QUANT] += ticks() - a;
        }
        pf_job jd = {&P->down, PB.y2, EMBD};
        pf_gemm_t(tid, &jd, 1, NFF, gs);
        pf_residual(tid, PB.y2, gs);
    }
#ifdef Q38P_MPI
    if (unit_start && do_ffn) pf_t_unit[2 * layer + 1] += ticks() - unit_start;
#endif
}

#ifdef Q38P_MPI
static void pp_transfer_state(float *p, int n, int owner) {
    if (pp_r == owner) MPI_Send(p, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    else if (pp_r == 0) MPI_Recv(p, n, MPI_FLOAT, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
static uint32_t pf_random(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b);
    return x ^ (x >> 16);
}
static size_t pf_proc_kib(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    size_t value = 0;
    while (fgets(line, sizeof(line), f))
        if (!strncmp(line, key, strlen(key)) && sscanf(line + strlen(key), "%zu", &value) == 1) break;
    fclose(f);
    return value;
}
/* Pipeline benchmark and first-token correctness path. Every rank retains
 * the decode image for now; only the assigned prefill descriptors are built.
 * MPI calls stay on worker 0, which is the process's main thread. */
static void q38p_pipeline(int tid, const int *tok, int pn, int *gs) {
    pf_init(tid, gs);
    if (tid == 0) { pf_active = 1; MPI_Barrier(MPI_COMM_WORLD); }
    gbarrier(tid, gs);
    float *prefix_ssm[NLAYER] = {0}, *prefix_conv[NLAYER] = {0};
    const int measured = pn - pf_bench_depth;
    double setup_start = tid == 0 ? MPI_Wtime() : 0;
    if (pf_bench_depth && pf_bench_synthetic) {
        for (int u = pp_u0; u < pp_u1; u += 1) {
            if ((u & 1) || E.L[u / 2].ssm) continue;
            int ai = E.L[u / 2].ai, c = tid / PER;
            if (pf_kv_i6_mode) {
                size_t row_bytes = kv_row_bytes();
                for (size_t i = (size_t)(tid % PER); i < (size_t)pf_bench_depth; i += PER) {
                    memset((uint8_t *)KI.k[ai][c][0] + i * row_bytes, 0, row_bytes);
                    memset((uint8_t *)KI.v[ai][c][0] + i * row_bytes, 0, row_bytes);
                    KI.ks[ai][c][0][i] = 1.0f;
                    KI.vs[ai][c][0][i] = 1.0f;
                }
            } else {
                size_t count = (size_t)pf_bench_depth * HD;
                for (size_t i = (size_t)(tid % PER) * 16; i < count; i += PER * 16)
                    for (size_t j = i; j < i + 16 && j < count; j++) {
                        uint32_t r = pf_random((uint32_t)j ^ pf_bench_seed ^ (u * 1237u + c * 7919u));
                        E.kcp[ai][c][0][j] = ((int)(r & 65535) - 32768) * (1.0f / 32768);
                        E.vcp[ai][c][0][j] = ((int)(r >> 16) - 32768) * (1.0f / 32768);
                    }
            }
        }
    }
    gbarrier(tid, gs);
    if (tid == 0 && pf_bench_depth && pf_bench_synthetic) {
        size_t rss = pf_proc_kib("/proc/self/status", "VmRSS:");
        size_t available = pf_proc_kib("/proc/meminfo", "MemAvailable:");
        fprintf(stderr, "q38p_capacity: rank=%d depth=%d stage_units=[%d,%d) rss=%.3f GiB MemAvailable=%.3f GiB\n",
                pp_r, pf_bench_depth, pp_u0, pp_u1,
                (double)rss / (double)(1ull << 20), (double)available / (double)(1ull << 20));
        if (available < (size_t)2 * 1024 * 1024) state_fail("PP synthetic depth crossed 2 GiB MemAvailable guard");
    }
    if (pf_bench_depth && pf_bench_synthetic && (pf_attn_gemm || pf_attn_pv_int16)) {
        for (int u = pp_u0; u < pp_u1; u++)
            if (!(u & 1) && !E.L[u / 2].ssm) {
                if (pf_attn_gemm) pf_cache_keys(tid, E.L[u / 2].ai, 0, pf_bench_depth);
                if (pf_attn_pv_int16) pf_cache_values(tid, E.L[u / 2].ai, 0, pf_bench_depth);
            }
    }
    gbarrier(tid, gs);
    if (tid == 0) MPI_Barrier(MPI_COMM_WORLD);
    gbarrier(tid, gs);
    double t0 = tid == 0 ? MPI_Wtime() : 0, first_done = 0, last_done = 0;
    double recv_time = 0, send_time = 0, compute_time = 0, steady_rate = 0;
    int first_token = -1;
    uint64_t first_hash = 0;
    for (int prompt = pf_bench_depth && !pf_bench_synthetic ? -1 : 0; prompt < pp_prompts; prompt++) {
        /* Stage-local state can be reused as soon as this stage completes its
         * previous prompt; other stages may still be working on that prompt. */
        for (int u = pp_u0; u < pp_u1; u++) {
            if (u & 1) continue;
            int layer = u / 2;
            if (!E.L[layer].ssm) continue;
            int h = ssm_head_of(tid);
            if (h >= 0) {
                if (prompt >= 0 && prefix_ssm[layer]) {
                    memcpy(ssm_st(layer, h, 0), prefix_ssm[layer], (DS * DS + 2 * DS) * sizeof(float));
                    memcpy(E.conv_hist[layer][h], prefix_conv[layer], 8 * 384 * sizeof(float));
                } else {
                    memset(ssm_st(layer, h, 0), 0, (DS * DS + 2 * DS) * sizeof(float));
                    memset(E.conv_hist[layer][h], 0, 8 * 384 * sizeof(float));
                }
            }
            if (tid == 0) memset(E.conv_state[layer], 0, (size_t)4 * QKVD * sizeof(float));
        }
        if (tid == 0) pf_active = 1;
        gbarrier(tid, gs);
        int begin = prompt < 0 ? 0 : pf_bench_depth;
        int end = prompt < 0 ? pf_bench_depth : pn;
        if (prompt == 0 && pf_bench_depth && tid == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            memset(pf_t, 0, sizeof pf_t);
            memset(pf_attn_times, 0, sizeof pf_attn_times);
            memset(pf_t_unit, 0, sizeof pf_t_unit);
            recv_time = send_time = compute_time = 0;
            fprintf(stderr, "q38p_depth: rank=%d depth=%d suffix=%d fill=%s seed=%u setup=%.6f s\n",
                    pp_r, pf_bench_depth, measured, pf_bench_synthetic ? "synthetic-kv" : "evaluated-random-tokens",
                    pf_bench_seed, t0 - setup_start);
        }
        gbarrier(tid, gs);
        for (int pos0 = begin; pos0 < end; pos0 += PF_N) {
            int ntok = end - pos0 < PF_N ? end - pos0 : PF_N;
            if (tid == 0) {
                PB.pos0 = pos0; PB.ntok = ntok; PB.ng = (ntok + PF_T - 1) / PF_T;
                if (pp_r > 0) {
                    double a = MPI_Wtime();
                    MPI_Recv(PB.x, ntok * EMBD, MPI_FLOAT, pp_r - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    recv_time += MPI_Wtime() - a;
                }
            }
            gbarrier(tid, gs);
            if (pp_r == 0)
                for (int t = tid; t < ntok; t += NT)
                    dequant_row(GGML_TYPE_Q6_K, E.embed_q6k + (size_t)tok[pos0 + t] * E.embed_row_bytes,
                                PB.x + (size_t)t * EMBD, EMBD);
            gbarrier(tid, gs);
            double work_start = tid == 0 ? MPI_Wtime() : 0;
            for (int u = pp_u0; u < pp_u1;) {
                int mixer = (u % 2 == 0), ffn = mixer && u + 1 < pp_u1;
                pf_layer_run(tid, u / 2, mixer, ffn || !mixer, gs);
                u += mixer && ffn ? 2 : 1;
            }
            gbarrier(tid, gs);
            if (tid == 0) {
                compute_time += MPI_Wtime() - work_start;
                if (pp_r + 1 < pp_n) {
                    double a = MPI_Wtime();
                    MPI_Send(PB.x, ntok * EMBD, MPI_FLOAT, pp_r + 1, 0, MPI_COMM_WORLD);
                    send_time += MPI_Wtime() - a;
                }
            }
            gbarrier(tid, gs);
        }
        if (prompt < 0) {
            int h = ssm_head_of(tid);
            if (h >= 0)
                for (int u = pp_u0; u < pp_u1; u++) {
                    if ((u & 1) || !E.L[u / 2].ssm) continue;
                    int l = u / 2;
                    prefix_ssm[l] = malloc((DS * DS + 2 * DS) * sizeof(float));
                    prefix_conv[l] = malloc(8 * 384 * sizeof(float));
                    if (!prefix_ssm[l] || !prefix_conv[l]) state_fail("benchmark prefix allocation");
                    memcpy(prefix_ssm[l], ssm_st(l, h, 0), (DS * DS + 2 * DS) * sizeof(float));
                    memcpy(prefix_conv[l], E.conv_hist[l][h], 8 * 384 * sizeof(float));
                }
            gbarrier(tid, gs);
            continue;
        }
        if (pp_r == pp_n - 1) {
            if (tid == 0) {
                memcpy(E.x, PB.x + (size_t)(PB.ntok - 1) * EMBD, EMBD * 4);
                pf_active = 0; ssq_valid = 0; tp_ssq_ok = 0; tp_prod_ok = 0;
            }
            gbarrier(tid, gs);
            uint64_t tt = ticks();
            head_argmax(tid, E.out_norm, gs, &tt);
            if (tid == 0) {
                uint64_t h = pf_last_hash();
                if (prompt == 0) { first_done = MPI_Wtime(); first_token = E.next_token; first_hash = h; }
                if (h != first_hash || E.next_token != first_token) {
                    pp_bad = 1;
                    fprintf(stderr, "q38p_pp: FATAL prompt=%d residual/token mismatch hash=%016llx token=%d\n",
                            prompt, (unsigned long long)h, E.next_token);
                }
                last_done = MPI_Wtime();
            }
        }
        gbarrier(tid, gs);
    }
    if (tid == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        double wall = MPI_Wtime() - t0;
        double metrics[3] = {first_done - t0, last_done - first_done, (double)first_token};
        MPI_Bcast(metrics, 3, MPI_DOUBLE, pp_n - 1, MPI_COMM_WORLD);
        MPI_Bcast(&first_hash, 1, MPI_UINT64_T, pp_n - 1, MPI_COMM_WORLD);
        MPI_Bcast(&pp_bad, 1, MPI_INT, pp_n - 1, MPI_COMM_WORLD);
        first_token = (int)metrics[2];
        steady_rate = pp_prompts > 1 ? (pp_prompts - 1) * (double)measured / metrics[1] : 0;
        fprintf(stderr, "q38p_pp: rank=%d/%d units=[%d,%d) prompts=%d tokens=%d wall=%.3f first_token=%d last_residual_hash=%016llx\n",
                pp_r, pp_n, pp_u0, pp_u1, pp_prompts, pn, wall, (int)metrics[2], (unsigned long long)first_hash);
        fprintf(stderr, "q38p_pp: rank=%d compute=%.3f recv_wait=%.3f send_wait=%.3f gemm=%.3f norm_quant=%.3f ssm=%.3f attn=%.3f\n",
                pp_r, compute_time, recv_time, send_time, pf_t[PT_GEMM] / tick_hz(), pf_t[PT_QUANT] / tick_hz(),
                pf_t[PT_SSM] / tick_hz(), pf_t[PT_ATT] / tick_hz());
        if (pf_attn_prof) {
            double sum[3] = {0};
            for (int w = 0; w < NT; w++) for (int k = 0; k < 3; k++) sum[k] += pf_attn_times[w][k] / tick_hz() / NT;
            fprintf(stderr, "q38p_attn_profile: rank=%d worker_mean_qk=%.6f softmax=%.6f pv=%.6f s\n", pp_r, sum[0], sum[1], sum[2]);
            double gemm_parts[6] = {0};
            for (int w = 0; w < NT; w++) {
                gemm_parts[0] += pf_t_expand[w] / tick_hz() / NT;
                gemm_parts[1] += pf_t_mk[w] / tick_hz() / NT;
                gemm_parts[2] += pf_t_zero[w] / tick_hz() / NT;
                gemm_parts[3] += pf_t_epilogue[w] / tick_hz() / NT;
                gemm_parts[4] += pf_t_dispatch[w] / tick_hz() / NT;
                gemm_parts[5] += pf_t_barrier[w] / tick_hz() / NT;
            }
            fprintf(stderr, "q38p_gemm_profile: rank=%d worker_mean_expand=%.6f kernel=%.6f zero=%.6f epilogue=%.6f dispatch=%.6f barrier=%.6f s\n",
                    pp_r, gemm_parts[0], gemm_parts[1], gemm_parts[2],
                    gemm_parts[3], gemm_parts[4], gemm_parts[5]);
            fprintf(stderr, "q38p_unit_profile: rank=%d", pp_r);
            for (int u = pp_u0; u < pp_u1; u++)
                fprintf(stderr, " %d:%.6f", u, pf_t_unit[u] / tick_hz());
            fputc('\n', stderr);
        }
        if (pp_r == 0)
            fprintf(stderr, "q38p_pp: nodes=%d prompts=%d prompt_tokens=%d end_to_end=%.3f tok/s single_prompt=%.3f s steady=%.3f tok/s\n",
                    pp_n, pp_prompts, pn, pp_prompts * (double)measured / wall, metrics[0],
                    pp_prompts > 1 ? (pp_prompts - 1) * (double)measured / metrics[1] : 0.0);
        if (pp_r == 0)
            fprintf(stderr, "q38p_prefill: rate_kind=end_to_end nodes=%d tokens=%d seconds=%.6f tok_s=%.3f tok_s_node=%.3f target_tok_s_node=150 target_met=%d\n",
                    pp_n, pp_prompts * measured, wall, pp_prompts * (double)measured / wall,
                    pp_prompts * (double)measured / wall / pp_n, pp_prompts * (double)measured / wall / pp_n >= 150);
    }
    if (tid == 0 && pp_r == 0 && pf_bench_depth)
        fprintf(stderr, "q38p_depth_result: depth=%d suffix=%d repeats=%d fill=%s qtile=%d attn_cost=%.3f kv_tile=%d qk6=%d chunk=%d gemm=%d key_cache=%d pv_gemm=%d pv_int16=%d pv_residual=%d p_residual=%d score_pad=%d steady_tok_s=%.3f steady_tok_s_node=%.3f target_tok_s_node=130 target_met=%d\n",
                pf_bench_depth, measured, pp_prompts, pf_bench_synthetic ? "synthetic-kv" : "evaluated-random-tokens", pf_attn_qtile, pp_attn_cost, pf_attn_ktile, att_qk6, PF_N, pf_attn_gemm, pf_attn_cache, pf_attn_pv_gemm, pf_attn_pv_int16, pf_attn_pv_residual, pf_attn_pv_p_residual, pf_attn_score_pad, steady_rate, steady_rate / pp_n, steady_rate / pp_n >= 130);
    gbarrier(tid, gs);
    for (int l = 0; l < NLAYER; l++) { free(prefix_ssm[l]); free(prefix_conv[l]); }
    if (state_out) {
        if (tid == 0) {
            double a = MPI_Wtime();
            for (int u = pp_u0; u < pp_u1; u++)
                if (!(u & 1)) state_export_layer(u / 2);
            MPI_Barrier(MPI_COMM_WORLD);
            if (pp_r == pp_n - 1) state_export_meta();
            MPI_Barrier(MPI_COMM_WORLD);
            state_write_sec = MPI_Wtime() - a;
            if (pp_r == 0) fprintf(stderr, "q38d_state: producer_nodes=%d export=%.6f s path=%s\n",
                                    pp_n, state_write_sec, state_out);
        }
        gbarrier(tid, gs);
    }
    if (pp_decode) {
        double a = tid == 0 ? MPI_Wtime() : 0;
        if (tid == 0) {
            for (int r = 1; r < pp_n; r++) {
                int u0, u1;
                pf_stage_bounds(pp_n, r, &u0, &u1);
                for (int u = u0; u < u1; u++) {
                    if (u & 1) continue;
                    int layer = u / 2;
                    if (E.L[layer].ssm) {
                        for (int h = 0; h < NVH; h++) {
                            pp_transfer_state(ssm_st(layer, h, 0), DS * DS + 2 * DS, r);
                            pp_transfer_state(E.conv_hist[layer][h], 8 * 384, r);
                        }
                    } else {
                        int ai = E.L[layer].ai;
                        for (int h = 0; h < NKV; h++) {
                            pp_transfer_state(E.kcp[ai][h][0], pn * HD, r);
                            pp_transfer_state(E.vcp[ai][h][0], pn * HD, r);
                        }
                    }
                }
            }
            if (pp_r == pp_n - 1) MPI_Send(E.x, EMBD, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
            if (pp_r == 0) MPI_Recv(E.x, EMBD, MPI_FLOAT, pp_n - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pf_active = 0; ssq_valid = 0; tp_ssq_ok = 0; tp_prod_ok = 0;
        }
        gbarrier(tid, gs);
        if (pp_r == 0) {
            uint64_t tt = ticks();
            head_argmax(tid, E.out_norm, gs, &tt);
            if (tid == 0) {
                if (E.next_token != first_token) pp_bad = 1;
                fprintf(stderr, "q38p_pp: decode handoff %.3f s first_token=%d match=%d\n",
                        MPI_Wtime() - a, E.next_token, E.next_token == first_token);
            }
        }
        gbarrier(tid, gs);
    }
}
#endif

/* Prefill the whole prompt; afterwards E.next_token holds the first
 * generated token and decode continues at position pn. */
static void q38p_prefill(int tid, const int *tok, int pn, int *gs) {
    static int inited;
    if (!inited) {
        pf_init(tid, gs);
        if (tid == 0 && pf_test) pf_selftest();
        gbarrier(tid, gs);
        if (tid == 0) inited = 1;
        gbarrier(tid, gs);
    }
    if (tid == 0) {
        pf_active = 1;
        memset(pf_t, 0, sizeof pf_t);
        memset(pf_t_expand, 0, sizeof pf_t_expand);
        memset(pf_t_mk, 0, sizeof pf_t_mk);
        memset(pf_t_zero, 0, sizeof pf_t_zero);
        memset(pf_t_epilogue, 0, sizeof pf_t_epilogue);
        memset(pf_t_dispatch, 0, sizeof pf_t_dispatch);
        memset(pf_t_barrier, 0, sizeof pf_t_barrier);
    }
    for (int pos0 = 0; pos0 < pn; pos0 += PF_N) {
        if (tid == 0) { PB.pos0 = pos0; PB.ntok = pn - pos0 < PF_N ? pn - pos0 : PF_N; PB.ng = (PB.ntok + PF_T - 1) / PF_T; }
        gbarrier(tid, gs);
        uint64_t a = ticks();
        for (int t = tid; t < PB.ntok; t += NT)
            dequant_row(GGML_TYPE_Q6_K, E.embed_q6k + (size_t)tok[pos0 + t] * E.embed_row_bytes, PB.x + (size_t)t * EMBD, EMBD);
        gbarrier(tid, gs);
        if (!tid) pf_t[PT_EMBED] += ticks() - a;
        for (int layer = 0; layer < NLAYER; layer++) pf_layer_run(tid, layer, 1, 1, gs);
    }
    /* head for the last prompt token through decode's head */
    uint64_t a = ticks();
    if (tid == 0) { memcpy(E.x, PB.x + (size_t)(PB.ntok - 1) * EMBD, EMBD * 4); pf_active = 0; }
    ssq_valid = 0; tp_ssq_ok = 0; tp_prod_ok = 0;
    gbarrier(tid, gs);
    uint64_t tt = ticks();
    head_argmax(tid, E.out_norm, gs, &tt);
    if (tid == 0)
        fprintf(stderr, "q38p: last_residual_hash=%016llx\n", (unsigned long long)pf_last_hash());
    if (!tid) pf_t[PT_HEAD] += ticks() - a;
}
static void q38p_report(int pn, double sec) {
    double hz = tick_hz(), tot = 0;
    for (int i = 0; i < PT_N; i++) tot += pf_t[i];
    fprintf(stderr, "q38p: prefill %d tok in %.3f s = %.1f tok/s (chunk %d); phases s:", pn, sec, pn / sec, PF_N);
    for (int i = 0; i < PT_N; i++) fprintf(stderr, " %s=%.3f", pf_tname[i], pf_t[i] / hz);
    double e = 0, k = 0, z = 0, ep = 0, d = 0, b = 0;
    for (int i = 0; i < NT; i++) {
        e += pf_t_expand[i]; k += pf_t_mk[i]; z += pf_t_zero[i];
        ep += pf_t_epilogue[i]; d += pf_t_dispatch[i]; b += pf_t_barrier[i];
    }
    fprintf(stderr, "\nq38p: gemm worker-mean s: expand=%.3f kernel=%.3f zero=%.3f epilogue=%.3f dispatch=%.3f barrier=%.3f; kernel MAC rate %.1f%% of int16 peak\n",
            e / NT / hz, k / NT / hz, z / NT / hz, ep / NT / hz, d / NT / hz, b / NT / hz,
            100.0 * (24.350556e9 * ((pn + PF_T - 1) / PF_T * PF_T)) / ((k / NT / hz) * 48 * 2e9 * 64));
}
