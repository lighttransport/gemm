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

static int pf_on, pf_chunk = 240, pf_test, pp_on;
static int PF_N, PF_NG;   /* tokens per chunk (multiple of PF_T), groups */
#ifdef Q38P_MPI
#define Q38P_MAX_STAGES (2 * NLAYER)
static int pp_n = 1, pp_r, pp_prompts = 1, pp_u0, pp_u1, pp_bad, pp_decode;
/* A unit is either the mixer or the FFN of a layer. Cuts may fall between
 * them, which keeps 6-, 8-, 12-, and larger allocations balanced. */
static void pf_stage_bounds(int n, int rank, int *lo, int *hi) {
    double prefix[2 * NLAYER + 1] = {0};
    for (int u = 0; u < 2 * NLAYER; u++)
        prefix[u + 1] = prefix[u] + ((u & 1) ? 267.4 : (E.L[u / 2].ssm ? 115.8 : 104.9));
    /* One rank per mixer/FFN unit is the architectural limit. */
    int cut[Q38P_MAX_STAGES + 1] = {0};
    for (int s = 1; s < n; s++) {
        double target = prefix[2 * NLAYER] * s / n;
        int best = cut[s - 1] + 1;
        for (int u = best + 1; u <= Q38P_MAX_STAGES - (n - s); u++)
            if (fabs(prefix[u] - target) < fabs(prefix[best] - target)) best = u;
        cut[s] = best;
    }
    cut[n] = Q38P_MAX_STAGES;
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
    float *sc[NT];            /* per-worker attention scores */
    int ntok, pos0, ng;       /* current chunk; ng = occupied token groups */
} PB;
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
    PB.sc[tid] = aligned_alloc(256, (size_t)6 * (E.max_seq + 16) * 4);
    pf_tmp[tid] = aligned_alloc(256, (size_t)(NFF + 6 * HD) * 4);
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

static void pf_layer_run(int tid, int layer, int do_mixer, int do_ffn, int *gs) {
    const q38d_layer *L = &E.L[layer];
    q38p_layer *P = &PL[layer];
    const int N = PB.ntok, pos0 = PB.pos0;
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
                    memcpy(E.kcp[ai][c][0] + (size_t)pos * HD, kv, sizeof kv);
                    memcpy(E.vcp[ai][c][0] + (size_t)pos * HD, PB.y2 + (size_t)t * NKV * HD + c * HD, HD * 4);
                }
            }
            gbarrier(tid, gs);
            uint64_t b = ticks();
            if (!tid) pf_t[PT_ATTPREP] += b - a;
            /* causal attention: tasks (KV head, token), largest contexts first */
            int ci = pf_calls[tid][0]++;
            _Atomic int *ctr = &pf_ctr[ci & 3];
            if (tid == 0) atomic_store_explicit(&pf_ctr[(ci + 2) & 3], 0, memory_order_relaxed);
            float *sc = PB.sc[tid], *out = pf_tmp[tid];
            const svbool_t pf = svptrue_b32();
            for (;;) {
                int task = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
                if (task >= NKV * N) break;
                int t = N - 1 - task / NKV, c = task % NKV, n = pos0 + t + 1;
                const float *K = E.kcp[ai][c][0], *V = E.vcp[ai][c][0];
                attn_scores(PB.qh + ((size_t)t * NHEAD + 6 * c) * HD, K, 0, n, sc, n, 1.0f / 16.0f);
                float l[6];
                for (int hh = 0; hh < 6; hh++) {
                    float *s_h = sc + (size_t)hh * n, m = -INFINITY;
                    for (int i = 0; i < n; i++) if (s_h[i] > m) m = s_h[i];
                    svfloat32_t lsv = svdup_n_f32(0);
                    for (int i = 0; i < n; i += 16) {
                        svbool_t pg = svwhilelt_b32(i, n);
                        svfloat32_t e = q38d_exp_sve(pg, svsub_n_f32_x(pg, svld1_f32(pg, s_h + i), m));
                        svst1_f32(pg, s_h + i, e);
                        lsv = svadd_f32_m(pg, lsv, e);
                    }
                    l[hh] = svaddv_f32(pf, lsv);
                }
                attn_pv6(V, 0, n, sc, n, out, HD);
                for (int hh = 0; hh < 6; hh++) {
                    int hq = 6 * c + hh;
                    const float *gate = PB.y0 + (size_t)t * NHEAD * 2 * HD + (size_t)hq * 2 * HD + HD;
                    float *dst = PB.o + (size_t)t * NHEAD * HD + (size_t)hq * HD, rl = 1.0f / l[hh];
                    for (int i = 0; i < HD; i += 16)
                        svst1_f32(pf, dst + i, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, out + hh * HD + i), rl),
                                                         q38d_sigmoid_sve(pf, svld1_f32(pf, gate + i))));
                }
            }
            gbarrier(tid, gs);
            if (!tid) pf_t[PT_ATT] += ticks() - b;
            pf_quant_rows(tid, PB.o, NHEAD * HD, NHEAD * HD, gs);
            pf_job jo = {&P->o, PB.y2, EMBD};
            pf_gemm_t(tid, &jo, 1, NHEAD * HD, gs);
            pf_residual(tid, PB.y2, gs);
        }
    }
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
}

#ifdef Q38P_MPI
static void pp_transfer_state(float *p, int n, int owner) {
    if (pp_r == owner) MPI_Send(p, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    else if (pp_r == 0) MPI_Recv(p, n, MPI_FLOAT, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
/* Pipeline benchmark and first-token correctness path. Every rank retains
 * the decode image for now; only the assigned prefill descriptors are built.
 * MPI calls stay on worker 0, which is the process's main thread. */
static void q38p_pipeline(int tid, const int *tok, int pn, int *gs) {
    pf_init(tid, gs);
    if (tid == 0) { pf_active = 1; MPI_Barrier(MPI_COMM_WORLD); }
    gbarrier(tid, gs);
    double t0 = tid == 0 ? MPI_Wtime() : 0, first_done = 0, last_done = 0;
    double recv_time = 0, send_time = 0, compute_time = 0;
    int first_token = -1;
    uint64_t first_hash = 0;
    for (int prompt = 0; prompt < pp_prompts; prompt++) {
        /* Stage-local state can be reused as soon as this stage completes its
         * previous prompt; other stages may still be working on that prompt. */
        for (int u = pp_u0; u < pp_u1; u++) {
            if (u & 1) continue;
            int layer = u / 2;
            if (!E.L[layer].ssm) continue;
            int h = ssm_head_of(tid);
            if (h >= 0) {
                memset(ssm_st(layer, h, 0), 0, (DS * DS + 2 * DS) * sizeof(float));
                memset(E.conv_hist[layer][h], 0, 8 * 384 * sizeof(float));
            }
            if (tid == 0) memset(E.conv_state[layer], 0, (size_t)4 * QKVD * sizeof(float));
        }
        if (tid == 0) pf_active = 1;
        gbarrier(tid, gs);
        for (int pos0 = 0; pos0 < pn; pos0 += PF_N) {
            int ntok = pn - pos0 < PF_N ? pn - pos0 : PF_N;
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
        fprintf(stderr, "q38p_pp: rank=%d/%d units=[%d,%d) prompts=%d tokens=%d wall=%.3f first_token=%d last_residual_hash=%016llx\n",
                pp_r, pp_n, pp_u0, pp_u1, pp_prompts, pn, wall, (int)metrics[2], (unsigned long long)first_hash);
        fprintf(stderr, "q38p_pp: rank=%d compute=%.3f recv_wait=%.3f send_wait=%.3f gemm=%.3f norm_quant=%.3f ssm=%.3f attn=%.3f\n",
                pp_r, compute_time, recv_time, send_time, pf_t[PT_GEMM] / tick_hz(), pf_t[PT_QUANT] / tick_hz(),
                pf_t[PT_SSM] / tick_hz(), pf_t[PT_ATT] / tick_hz());
        if (pp_r == 0)
            fprintf(stderr, "q38p_pp: nodes=%d prompts=%d prompt_tokens=%d end_to_end=%.3f tok/s single_prompt=%.3f s steady=%.3f tok/s\n",
                    pp_n, pp_prompts, pn, pp_prompts * (double)pn / wall, metrics[0],
                    pp_prompts > 1 ? (pp_prompts - 1) * (double)pn / metrics[1] : 0.0);
    }
    gbarrier(tid, gs);
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
