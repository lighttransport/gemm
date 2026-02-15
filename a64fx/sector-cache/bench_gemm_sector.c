/*
 * GEMM Sector Cache Benchmark (single-thread + multi-core)
 *
 * Build (single-threaded):
 *   fcc -Nnoclang -O2 -Kocl,hpctag -o bench_gemm_sector \
 *       bench_gemm_sector.c ../fused-gemm/micro_kernel_8x3.S -lm
 *
 * Build (multi-core, 12 threads on one CMG):
 *   fcc -Nnoclang -O2 -Kocl,hpctag,openmp -o bench_gemm_sector \
 *       bench_gemm_sector.c ../fused-gemm/micro_kernel_8x3.S -lm
 *
 * Run:
 *   FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE FLIB_L2_SCCR_CNTL_EX=TRUE \
 *   OMP_NUM_THREADS=12 numactl --cpunodebind=0 --membind=0 ./bench_gemm_sector
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MR  8
#define NR  48
#define KU  4

extern void micro_kernel_fp32_8x3_unroll4(
    const float *A_packed, const float *B_packed,
    float *C, int64_t K, int64_t unused, int64_t ldc_bytes);

static inline uint64_t rdtsc(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}
static inline void barrier(void) {
    __asm__ volatile("dsb ish" ::: "memory");
}

static int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

static volatile float g_sink;

/* Packing */
static void pack_A(int M, int K, const float *A, int lda, float *Ap) {
    int Kr = round_up(K, KU);
    for (int mb = 0; mb < M; mb += MR) {
        for (int k = 0; k < Kr; k++) {
            for (int m = 0; m < MR; m++) {
                int row = mb + m;
                *Ap++ = (row < M && k < K) ? A[row * lda + k] : 0.0f;
            }
        }
    }
}

static void pack_B(int K, int N, const float *B, int ldb, float *Bp) {
    int Kr = round_up(K, KU);
    for (int nb = 0; nb < N; nb += NR) {
        for (int k = 0; k < Kr; k++) {
            for (int n = 0; n < NR; n++) {
                int col = nb + n;
                *Bp++ = (col < N && k < K) ? B[k * ldb + col] : 0.0f;
            }
        }
    }
}

/* ──────────────────────────────────────────────────────────────────
 * GEMM: no sector cache (baseline)
 * ────────────────────────────────────────────────────────────────── */
void gemm_nohint(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc)
{
    int nb, mb;
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = Bp + (long)nb * Kr * NR;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = Ap + (long)mb * Kr * MR;
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

/* ──────────────────────────────────────────────────────────────────
 * GEMM: with sector cache pragma (B → sector 1)
 * ────────────────────────────────────────────────────────────────── */
void gemm_sector(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bp
    int nb, mb;
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = Bp + (long)nb * Kr * NR;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = Ap + (long)mb * Kr * MR;
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

/* ──────────────────────────────────────────────────────────────────
 * GEMM: manual SCCR override
 * ────────────────────────────────────────────────────────────────── */
void gemm_manual(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc,
    unsigned l1s0, unsigned l1s1, unsigned l2s0, unsigned l2s1)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bp
    /* Override SCCR with specified values */
    uint64_t l1v = (l1s1 & 0x7) << 4 | (l1s0 & 0x7);
    uint64_t l2v = ((uint64_t)(l2s1 & 0x1F) << 8) | (l2s0 & 0x1F);
    __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1v));
    __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2v));
    __asm__ volatile("isb" ::: "memory");

    int nb, mb;
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = Bp + (long)nb * Kr * NR;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = Ap + (long)mb * Kr * MR;
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

/* ──────────────────────────────────────────────────────────────────
 * GEMM: address-tagged pointers (like Fujitsu reference DGEMM)
 *
 * Tag bits in address [59:56]:
 *   bit 59: SCE (Sector Cache Enable)
 *   bit 58: PFE (Prefetch Enable)
 *   bit 57-56: sector_id
 *
 * A → tag 0x4 (sector 0, prefetch enable)  — keep in cache
 * B → tag 0xA (SCE + sector 2)             — stream, separate sector
 * C → no tag (sector 0)                    — reused, keep with A
 * ────────────────────────────────────────────────────────────────── */
#define APPLY_TAG(ptr, tag) \
    ((__typeof__(ptr))((uintptr_t)(ptr) | ((uint64_t)(tag) << 56)))

void gemm_tagged(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc,
    unsigned a_tag, unsigned b_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bp
    int nb, mb;
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = APPLY_TAG(Bp + (long)nb * Kr * NR, b_tag);
        __asm__ volatile("" : "+r"(B_tile));
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = APPLY_TAG(Ap + (long)mb * Kr * MR, a_tag);
            __asm__ volatile("" : "+r"(A_tile));
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

/* ──────────────────────────────────────────────────────────────────
 * OpenMP GEMM: parallelized over N-tiles (12 cores share L2)
 *
 * A_packed is shared read-only — should stay in L2 (sector 0)
 * B_packed streams per-thread — cycles through L2 (sector 1)
 * ────────────────────────────────────────────────────────────────── */
#ifdef _OPENMP

void gemm_nohint_omp(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc)
{
    int nb;
#pragma omp parallel for schedule(static)
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = Bp + (long)nb * Kr * NR;
        int mb;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = Ap + (long)mb * Kr * MR;
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

void gemm_sector_omp(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bp
    int nb;
#pragma omp parallel for schedule(static)
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = Bp + (long)nb * Kr * NR;
        int mb;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = Ap + (long)mb * Kr * MR;
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

void gemm_tagged_omp(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc,
    unsigned a_tag, unsigned b_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bp
    int nb;
#pragma omp parallel for schedule(static)
    for (nb = 0; nb < N_tiles; nb++) {
        const float *B_tile = APPLY_TAG(Bp + (long)nb * Kr * NR, b_tag);
        __asm__ volatile("" : "+r"(B_tile));
        int mb;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = APPLY_TAG(Ap + (long)mb * Kr * MR, a_tag);
            __asm__ volatile("" : "+r"(A_tile));
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

void gemm_manual_omp(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    const float *Ap, const float *Bp, float *C, int ldc,
    unsigned l1s0, unsigned l1s1, unsigned l2s0, unsigned l2s1,
    unsigned a_tag, unsigned b_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bp
    uint64_t l1v = (l1s1 & 0x7) << 4 | (l1s0 & 0x7);
    uint64_t l2v = ((uint64_t)(l2s1 & 0x1F) << 8) | (l2s0 & 0x1F);
    int nb;
#pragma omp parallel for schedule(static)
    for (nb = 0; nb < N_tiles; nb++) {
        /* Each thread writes its own core's SCCR */
        __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1v));
        __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2v));
        __asm__ volatile("isb" ::: "memory");
        const float *B_tile = APPLY_TAG(Bp + (long)nb * Kr * NR, b_tag);
        __asm__ volatile("" : "+r"(B_tile));
        int mb;
        for (mb = 0; mb < M_tiles; mb++) {
            const float *A_tile = APPLY_TAG(Ap + (long)mb * Kr * MR, a_tag);
            __asm__ volatile("" : "+r"(A_tile));
            float *C_tile = C + mb * MR * ldc + nb * NR;
            micro_kernel_fp32_8x3_unroll4(
                A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
        }
    }
}

/*
 * Independent GEMM per thread: each thread has its OWN A, B, C
 * (simulates batched GEMM — no shared-A cache benefit)
 * 12 separate A buffers compete for L2 → much more cache pressure
 */
void gemm_indep_nohint(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    float **Aps, float **Bps, float **Cs, int ldc, int nthreads)
{
    int tid;
#pragma omp parallel for schedule(static, 1)
    for (tid = 0; tid < nthreads; tid++) {
        const float *Ap = Aps[tid];
        const float *Bp = Bps[tid];
        float *C = Cs[tid];
        int nb, mb;
        for (nb = 0; nb < N_tiles; nb++) {
            const float *B_tile = Bp + (long)nb * Kr * NR;
            for (mb = 0; mb < M_tiles; mb++) {
                const float *A_tile = Ap + (long)mb * Kr * MR;
                float *C_tile = C + mb * MR * ldc + nb * NR;
                micro_kernel_fp32_8x3_unroll4(
                    A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
            }
        }
    }
}

void gemm_indep_tagged(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    float **Aps, float **Bps, float **Cs, int ldc, int nthreads,
    unsigned a_tag, unsigned b_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bps
    int tid;
#pragma omp parallel for schedule(static, 1)
    for (tid = 0; tid < nthreads; tid++) {
        const float *Ap = Aps[tid];
        const float *Bp = Bps[tid];
        float *C = Cs[tid];
        int nb, mb;
        for (nb = 0; nb < N_tiles; nb++) {
            const float *B_tile = APPLY_TAG(Bp + (long)nb * Kr * NR, b_tag);
            __asm__ volatile("" : "+r"(B_tile));
            for (mb = 0; mb < M_tiles; mb++) {
                const float *A_tile = APPLY_TAG(Ap + (long)mb * Kr * MR, a_tag);
                __asm__ volatile("" : "+r"(A_tile));
                float *C_tile = C + mb * MR * ldc + nb * NR;
                micro_kernel_fp32_8x3_unroll4(
                    A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
            }
        }
    }
}

void gemm_indep_manual(
    int M, int K, int N, int Kr, int M_tiles, int N_tiles,
    float **Aps, float **Bps, float **Cs, int ldc, int nthreads,
    unsigned l1s0, unsigned l1s1, unsigned l2s0, unsigned l2s1,
    unsigned a_tag, unsigned b_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign Bps
    uint64_t l1v = (l1s1 & 0x7) << 4 | (l1s0 & 0x7);
    uint64_t l2v = ((uint64_t)(l2s1 & 0x1F) << 8) | (l2s0 & 0x1F);
    int tid;
#pragma omp parallel for schedule(static, 1)
    for (tid = 0; tid < nthreads; tid++) {
        __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1v));
        __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2v));
        __asm__ volatile("isb" ::: "memory");
        const float *Ap = Aps[tid];
        const float *Bp = Bps[tid];
        float *C = Cs[tid];
        int nb, mb;
        for (nb = 0; nb < N_tiles; nb++) {
            const float *B_tile = APPLY_TAG(Bp + (long)nb * Kr * NR, b_tag);
            __asm__ volatile("" : "+r"(B_tile));
            for (mb = 0; mb < M_tiles; mb++) {
                const float *A_tile = APPLY_TAG(Ap + (long)mb * Kr * MR, a_tag);
                __asm__ volatile("" : "+r"(A_tile));
                float *C_tile = C + mb * MR * ldc + nb * NR;
                micro_kernel_fp32_8x3_unroll4(
                    A_tile, B_tile, C_tile, Kr, 0, (int64_t)ldc * 4);
            }
        }
    }
}

#endif /* _OPENMP */

/* checksum to prevent dead-code elimination */
static float checksum(const float *C, int M, int ldc) {
    float s = 0;
    int i;
    for (i = 0; i < M; i += 13)
        s += C[i * ldc];
    return s;
}

static void run_test(int M, int K, int N, const char *desc) {
    uint64_t freq = rdfreq();
    double peak = 128.0;

    int Kr = round_up(K, KU);
    int M_tiles = (M + MR - 1) / MR;
    int N_tiles = (N + NR - 1) / NR;
    int Nr = N_tiles * NR;

    long Ap_bytes = (long)M_tiles * Kr * MR * 4;
    long Bp_bytes = (long)N_tiles * Kr * NR * 4;
    long C_bytes  = (long)M * Nr * 4;
    long A_tile_bytes = (long)Kr * MR * 4;
    long B_tile_bytes = (long)Kr * NR * 4;

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("M=%d, K=%d, N=%d  [%s]\n", M, K, N, desc);
    printf("  A_packed: %ld KB total (%ld KB/tile × %d tiles)\n",
           Ap_bytes/1024, A_tile_bytes/1024, M_tiles);
    printf("  B_packed: %ld KB total (%ld KB/tile × %d tiles)\n",
           Bp_bytes/1024, B_tile_bytes/1024, N_tiles);
    printf("  C output: %ld KB\n\n", C_bytes/1024);

    printf("  Cache pressure:\n");
    printf("    L1 (64KB 4-way): A_tile=%ldKB (%.1f ways) + B_tile=%ldKB (%.1f ways)",
           A_tile_bytes/1024, (double)A_tile_bytes/(16*1024),
           B_tile_bytes/1024, (double)B_tile_bytes/(16*1024));
    if (A_tile_bytes + B_tile_bytes > 64*1024)
        printf(" → OVERFLOW\n");
    else
        printf(" → fits\n");
    printf("    L2 (8MB 14way):  A_all=%ldKB (%.1f ways) + B_all=%ldKB (%.1f ways)",
           Ap_bytes/1024, (double)Ap_bytes/(512*1024),
           Bp_bytes/1024, (double)Bp_bytes/(512*1024));
    if (Ap_bytes + Bp_bytes > 7L*1024*1024)
        printf(" → OVERFLOW\n");
    else
        printf(" → fits\n");
    printf("\n");

    /* Allocate */
    float *A  = (float*)aligned_alloc(256, (long)M * K * 4);
    float *B  = (float*)aligned_alloc(256, (long)K * N * 4);
    float *Ap = (float*)aligned_alloc(256, Ap_bytes);
    float *Bp = (float*)aligned_alloc(256, Bp_bytes);
    float *C  = (float*)aligned_alloc(256, C_bytes);
    if (!A || !B || !Ap || !Bp || !C) {
        fprintf(stderr, "alloc failed\n");
        return;
    }

    srand(42);
    { int i; for (i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f; }
    { int i; for (i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f; }
    pack_A(M, K, A, K, Ap);
    pack_B(K, N, B, N, Bp);

    /* Determine iteration count (~1s total) */
    double flops_per = 2.0 * M * K * N;
    int iters = (int)(104.0e9 / flops_per);  /* assume ~104 GFLOPS */
    if (iters < 3) iters = 3;
    if (iters > 200) iters = 200;

    double gf, gf_base, sec;
    uint64_t t0, t1, ticks;
    int i;

    /* ── Baseline (no sector cache) ── */
    memset(C, 0, C_bytes);
    gemm_nohint(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
    barrier();

    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_nohint(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
        __asm__ volatile("" ::: "memory");
    }
    barrier();
    t1 = rdtsc();
    g_sink += checksum(C, M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    gf_base = gf;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks\n",
           "nohint (baseline)", gf, gf/peak*100.0, ticks);

    /* ── Pragma sector cache: L2=5, L1=2 ── */
    memset(C, 0, C_bytes);
    gemm_sector(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
    barrier();

    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_sector(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
        __asm__ volatile("" ::: "memory");
    }
    barrier();
    t1 = rdtsc();
    g_sink += checksum(C, M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
           "sector L2=5 L1=2 (pragma)", gf, gf/peak*100.0, ticks, gf/gf_base);

    /* ── Manual SCCR config (max-keep) ── */
    memset(C, 0, C_bytes);
    gemm_manual(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr, 3, 1, 13, 1);
    barrier();

    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_manual(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr, 3, 1, 13, 1);
        __asm__ volatile("" ::: "memory");
    }
    barrier();
    t1 = rdtsc();
    g_sink += checksum(C, M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
           "manual L1(3,1) L2(13,1)", gf, gf/peak*100.0, ticks, gf/gf_base);

    /* ── Address-tagged pointer configs ── */
    printf("  --- address-tagged pointers (tag applied to asm kernel args) ---\n");
    {
        /* Tag format: [59:56] = {SCE, PFE, sec_id[1:0]}
         * 0x0 = no tag
         * 0x2 = sector 2 (no SCE)
         * 0x4 = prefetch enable (sector 0)
         * 0x8 = SCE (sector 0)
         * 0xA = SCE + sector 2
         * 0xB = SCE + sector 3 (L1 bypass)
         * 0xC = SCE + PFE (sector 0)
         */
        unsigned t_atag[] = { 0x0, 0x4, 0xC, 0x0, 0x4 };
        unsigned t_btag[] = { 0xA, 0xA, 0xA, 0xB, 0xB };
        const char *t_label[] = {
            "A=0x0 B=0xA (B:SCE+sec2)",
            "A=0x4 B=0xA (A:PF, B:SCE+sec2)",
            "A=0xC B=0xA (A:SCE+PF, B:SCE+sec2)",
            "A=0x0 B=0xB (B:SCE+sec3=bypass)",
            "A=0x4 B=0xB (A:PF, B:bypass)"
        };
        int nt = 5;
        int t;

        for (t = 0; t < nt; t++) {
            memset(C, 0, C_bytes);
            gemm_tagged(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                        t_atag[t], t_btag[t]);
            barrier();

            barrier();
            t0 = rdtsc();
            for (i = 0; i < iters; i++) {
                gemm_tagged(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                            t_atag[t], t_btag[t]);
                __asm__ volatile("" ::: "memory");
            }
            barrier();
            t1 = rdtsc();
            g_sink += checksum(C, M, Nr);
            ticks = t1 - t0;
            sec = (double)ticks / freq;
            gf = flops_per * iters / sec / 1e9;
            printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
                   t_label[t], gf, gf/peak*100.0, ticks, gf/gf_base);
        }
    }

    printf("\n");
    free(A); free(B); free(Ap); free(Bp); free(C);
}

/* ══════════════════════════════════════════════════════════════════
 * Multi-core test (OpenMP): 12 cores sharing one CMG's 8MB L2
 * ══════════════════════════════════════════════════════════════════ */
#ifdef _OPENMP

static void run_test_omp(int M, int K, int N, const char *desc) {
    uint64_t freq = rdfreq();
    int nthreads = omp_get_max_threads();
    double peak = 128.0 * nthreads;

    int Kr = round_up(K, KU);
    int M_tiles = (M + MR - 1) / MR;
    int N_tiles = (N + NR - 1) / NR;
    int Nr = N_tiles * NR;

    long Ap_bytes = (long)M_tiles * Kr * MR * 4;
    long Bp_bytes = (long)N_tiles * Kr * NR * 4;
    long C_bytes  = (long)M * Nr * 4;
    long A_tile_bytes = (long)Kr * MR * 4;
    long B_tile_bytes = (long)Kr * NR * 4;

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("M=%d, K=%d, N=%d  [%s] (%d threads)\n", M, K, N, desc, nthreads);
    printf("  A_packed: %ld KB total (shared, %ld KB/tile x %d tiles)\n",
           Ap_bytes/1024, A_tile_bytes/1024, M_tiles);
    printf("  B_packed: %ld KB total (streamed, %ld KB/tile x %d tiles)\n",
           Bp_bytes/1024, B_tile_bytes/1024, N_tiles);
    printf("  C output: %ld KB,  tiles/thread: %d\n\n",
           C_bytes/1024, N_tiles / nthreads);

    printf("  Shared L2 pressure (%d cores on 8MB L2, 14 usable ways):\n", nthreads);
    printf("    A_packed=%ld KB (%.1f ways) -- shared read-only (should KEEP)\n",
           Ap_bytes/1024, (double)Ap_bytes/(512*1024));
    printf("    B per-thread: %d tiles x %ldKB = %ldKB streaming\n",
           N_tiles/nthreads, B_tile_bytes/1024,
           (long)(N_tiles/nthreads) * B_tile_bytes/1024);
    printf("    %d threads x B_stream → %ldKB total B through L2\n",
           nthreads, Bp_bytes/1024);
    double bw_needed = (double)(Bp_bytes + Ap_bytes) * 3.0 / 1e9; /* rough, 3 iters */
    printf("    HBM2 BW/CMG ~256 GB/s, peak=%d GFLOPS → %.0f GFLOPS BW-limited\n",
           (int)peak, 256.0 * 3.4);
    printf("\n");

    /* Allocate (NUMA-local via first-touch in OMP) */
    float *A  = (float*)aligned_alloc(256, (long)M * K * 4);
    float *B  = (float*)aligned_alloc(256, (long)K * N * 4);
    float *Ap = (float*)aligned_alloc(256, Ap_bytes);
    float *Bp = (float*)aligned_alloc(256, Bp_bytes);
    float *C  = (float*)aligned_alloc(256, C_bytes);
    if (!A || !B || !Ap || !Bp || !C) {
        fprintf(stderr, "alloc failed\n");
        return;
    }

    srand(42);
    { int i; for (i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f; }
    { int i; for (i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f; }
    pack_A(M, K, A, K, Ap);
    pack_B(K, N, B, N, Bp);

    /* Iteration count: target ~1s per config */
    double flops_per = 2.0 * M * K * N;
    int iters = (int)(peak * 1e9 / flops_per);
    if (iters < 2) iters = 2;
    if (iters > 100) iters = 100;

    double gf, gf_base, sec;
    uint64_t t0, t1, ticks;
    int i;

    /* ── Baseline (no sector cache, OMP) ── */
    memset(C, 0, C_bytes);
    gemm_nohint_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
    barrier();

    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_nohint_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
        barrier();
    }
    barrier();
    t1 = rdtsc();
    g_sink += checksum(C, M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    gf_base = gf;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks\n",
           "nohint (baseline)", gf, gf/peak*100.0, ticks);

    /* ── Pragma sector cache: L2=5, L1=2 ── */
    memset(C, 0, C_bytes);
    gemm_sector_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
    barrier();

    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_sector_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr);
        barrier();
    }
    barrier();
    t1 = rdtsc();
    g_sink += checksum(C, M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
           "sector L2=5 L1=2 (pragma)", gf, gf/peak*100.0, ticks, gf/gf_base);

    /* ── Address-tagged pointers ── */
    printf("  --- address-tagged pointers ---\n");
    {
        unsigned t_atag[] = { 0x0, 0x4, 0xC, 0x4 };
        unsigned t_btag[] = { 0xA, 0xA, 0xA, 0xB };
        const char *t_label[] = {
            "A=0x0 B=0xA (B:SCE+sec2)",
            "A=0x4 B=0xA (A:PF, B:SCE+sec2)",
            "A=0xC B=0xA (A:SCE+PF, B:SCE+sec2)",
            "A=0x4 B=0xB (A:PF, B:bypass)"
        };
        int nt = 4;
        int t;

        for (t = 0; t < nt; t++) {
            memset(C, 0, C_bytes);
            gemm_tagged_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                            t_atag[t], t_btag[t]);
            barrier();

            barrier();
            t0 = rdtsc();
            for (i = 0; i < iters; i++) {
                gemm_tagged_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                                t_atag[t], t_btag[t]);
                barrier();
            }
            barrier();
            t1 = rdtsc();
            g_sink += checksum(C, M, Nr);
            ticks = t1 - t0;
            sec = (double)ticks / freq;
            gf = flops_per * iters / sec / 1e9;
            printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
                   t_label[t], gf, gf/peak*100.0, ticks, gf/gf_base);
        }
    }

    /* ── Manual SCCR + tagged pointers (best configs) ── */
    printf("  --- manual SCCR + tagged pointers ---\n");
    {
        /* L1(3,1) L2(13,1): max A-keep, 1 way for B stream */
        memset(C, 0, C_bytes);
        gemm_manual_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                        3, 1, 13, 1, 0x4, 0xA);
        barrier();

        barrier();
        t0 = rdtsc();
        for (i = 0; i < iters; i++) {
            gemm_manual_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                            3, 1, 13, 1, 0x4, 0xA);
            barrier();
        }
        barrier();
        t1 = rdtsc();
        g_sink += checksum(C, M, Nr);
        ticks = t1 - t0;
        sec = (double)ticks / freq;
        gf = flops_per * iters / sec / 1e9;
        printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
               "L1(3,1)L2(13,1) A=PF B=SCE+s2", gf, gf/peak*100.0, ticks, gf/gf_base);

        /* L1(3,1) L2(12,2): slightly more B room */
        memset(C, 0, C_bytes);
        gemm_manual_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                        3, 1, 12, 2, 0x4, 0xA);
        barrier();

        barrier();
        t0 = rdtsc();
        for (i = 0; i < iters; i++) {
            gemm_manual_omp(M, K, N, Kr, M_tiles, N_tiles, Ap, Bp, C, Nr,
                            3, 1, 12, 2, 0x4, 0xA);
            barrier();
        }
        barrier();
        t1 = rdtsc();
        g_sink += checksum(C, M, Nr);
        ticks = t1 - t0;
        sec = (double)ticks / freq;
        gf = flops_per * iters / sec / 1e9;
        printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
               "L1(3,1)L2(12,2) A=PF B=SCE+s2", gf, gf/peak*100.0, ticks, gf/gf_base);
    }

    printf("\n");
    free(A); free(B); free(Ap); free(Bp); free(C);
}

/* ══════════════════════════════════════════════════════════════════
 * Independent GEMM per thread: each has its OWN A, B, C
 * No shared-A cache benefit → 12 separate A buffers compete for L2
 * ══════════════════════════════════════════════════════════════════ */
static void run_test_indep(int M, int K, int N, const char *desc) {
    uint64_t freq = rdfreq();
    int nthreads = omp_get_max_threads();
    double peak = 128.0 * nthreads;

    int Kr = round_up(K, KU);
    int M_tiles = (M + MR - 1) / MR;
    int N_tiles = (N + NR - 1) / NR;
    int Nr = N_tiles * NR;

    long Ap_bytes = (long)M_tiles * Kr * MR * 4;
    long Bp_bytes = (long)N_tiles * Kr * NR * 4;
    long C_bytes  = (long)M * Nr * 4;

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("INDEPENDENT: M=%d K=%d N=%d  [%s] (%d threads, each own A/B/C)\n",
           M, K, N, desc, nthreads);
    printf("  Per-thread: A=%ldKB  B=%ldKB  C=%ldKB\n",
           Ap_bytes/1024, Bp_bytes/1024, C_bytes/1024);
    printf("  Total: A=%ldKB x %d = %ldKB (%.1f L2 ways)\n",
           Ap_bytes/1024, nthreads, Ap_bytes*nthreads/1024,
           (double)Ap_bytes*nthreads/(512*1024));
    printf("  Total: B=%ldKB x %d = %ldKB\n\n",
           Bp_bytes/1024, nthreads, Bp_bytes*nthreads/1024);

    /* Allocate per-thread arrays */
    float *Aps[12], *Bps[12], *Cs[12];
    int tid;
    for (tid = 0; tid < nthreads; tid++) {
        Aps[tid] = (float*)aligned_alloc(256, Ap_bytes);
        Bps[tid] = (float*)aligned_alloc(256, Bp_bytes);
        Cs[tid]  = (float*)aligned_alloc(256, C_bytes);
        if (!Aps[tid] || !Bps[tid] || !Cs[tid]) {
            fprintf(stderr, "alloc failed thread %d\n", tid);
            return;
        }
    }

    /* Init data per-thread (different data → different cache lines) */
    srand(42);
    for (tid = 0; tid < nthreads; tid++) {
        float *A = (float*)aligned_alloc(256, (long)M * K * 4);
        float *B = (float*)aligned_alloc(256, (long)K * N * 4);
        int j;
        for (j = 0; j < M * K; j++) A[j] = (float)((rand() + tid) % 100) / 100.0f;
        for (j = 0; j < K * N; j++) B[j] = (float)((rand() + tid) % 100) / 100.0f;
        pack_A(M, K, A, K, Aps[tid]);
        pack_B(K, N, B, N, Bps[tid]);
        memset(Cs[tid], 0, C_bytes);
        free(A);
        free(B);
    }

    double flops_per = 2.0 * M * K * N * nthreads;
    int iters = (int)(peak * 1e9 / (2.0 * M * K * N));
    if (iters < 2) iters = 2;
    if (iters > 50) iters = 50;

    double gf, gf_base, sec;
    uint64_t t0, t1, ticks;
    int i;

    /* ── Baseline ── */
    gemm_indep_nohint(M, K, N, Kr, M_tiles, N_tiles, Aps, Bps, Cs, Nr, nthreads);
    barrier();

    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_indep_nohint(M, K, N, Kr, M_tiles, N_tiles, Aps, Bps, Cs, Nr, nthreads);
        barrier();
    }
    barrier();
    t1 = rdtsc();
    for (tid = 0; tid < nthreads; tid++) g_sink += checksum(Cs[tid], M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    gf_base = gf;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks\n",
           "nohint (baseline)", gf, gf/peak*100.0, ticks);

    /* ── Tagged: A=0x4 B=0xA ── */
    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_indep_tagged(M, K, N, Kr, M_tiles, N_tiles, Aps, Bps, Cs, Nr, nthreads,
                          0x4, 0xA);
        barrier();
    }
    barrier();
    t1 = rdtsc();
    for (tid = 0; tid < nthreads; tid++) g_sink += checksum(Cs[tid], M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
           "A=0x4 B=0xA (A:PF, B:SCE+sec2)", gf, gf/peak*100.0, ticks, gf/gf_base);

    /* ── Tagged: A=0xC B=0xA (both SCE) ── */
    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_indep_tagged(M, K, N, Kr, M_tiles, N_tiles, Aps, Bps, Cs, Nr, nthreads,
                          0xC, 0xA);
        barrier();
    }
    barrier();
    t1 = rdtsc();
    for (tid = 0; tid < nthreads; tid++) g_sink += checksum(Cs[tid], M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
           "A=0xC B=0xA (A:SCE+PF, B:SCE+sec2)", gf, gf/peak*100.0, ticks, gf/gf_base);

    /* ── Manual SCCR + tagged ── */
    barrier();
    t0 = rdtsc();
    for (i = 0; i < iters; i++) {
        gemm_indep_manual(M, K, N, Kr, M_tiles, N_tiles, Aps, Bps, Cs, Nr, nthreads,
                          3, 1, 13, 1, 0x4, 0xA);
        barrier();
    }
    barrier();
    t1 = rdtsc();
    for (tid = 0; tid < nthreads; tid++) g_sink += checksum(Cs[tid], M, Nr);
    ticks = t1 - t0;
    sec = (double)ticks / freq;
    gf = flops_per * iters / sec / 1e9;
    printf("  %-40s %8.2f GFLOPS (%5.1f%%) %10lu ticks  %.3fx\n",
           "L1(3,1)L2(13,1) A=PF B=SCE+s2", gf, gf/peak*100.0, ticks, gf/gf_base);

    printf("\n");
    for (tid = 0; tid < nthreads; tid++) {
        free(Aps[tid]); free(Bps[tid]); free(Cs[tid]);
    }
}

#endif /* _OPENMP */

int main(int argc, char **argv) {
    uint64_t freq = rdfreq();
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    double peak_per_core = 128.0;
    double peak = peak_per_core * nthreads;

    printf("=== GEMM Sector Cache Benchmark (FP32 8x3 SVE Kernel) ===\n\n");
    printf("Micro-tile : MR=%d x NR=%d (8 rows x 48 cols = 3 SVE vectors)\n", MR, NR);
    printf("Threads    : %d\n", nthreads);
    printf("Peak FP32  : %.0f GFLOPS (%d cores x 128 GFLOPS/core)\n", peak, nthreads);
    printf("Timer freq : %lu MHz\n\n", freq / 1000000);

    const char *e1 = getenv("FLIB_SCCR_CNTL");
    const char *e2 = getenv("FLIB_L1_SCCR_CNTL");
    const char *e3 = getenv("FLIB_L2_SCCR_CNTL_EX");
    printf("FLIB_SCCR_CNTL       = %s\n", e1 ? e1 : "(not set)");
    printf("FLIB_L1_SCCR_CNTL    = %s\n", e2 ? e2 : "(not set)");
    printf("FLIB_L2_SCCR_CNTL_EX = %s\n\n", e3 ? e3 : "(not set)");

#ifdef _OPENMP
    if (nthreads > 1) {
        /*
         * Multi-core: 12 cores share 8MB L2, HBM2 ~256 GB/s per CMG
         * Kernel becomes memory-bound: 12x128=1536 GFLOPS vs 256*3.4=870 GFLOPS
         *
         * A_packed = shared read-only → should stay in L2 sector 0
         * B_packed = streamed per-thread → cycles through L2 sector 1
         */
        /* Shared-A tests (cooperative GEMM, N-tile parallel) */
        run_test_omp(1024,  512, 4096, "shared-A: A=2MB B=8MB");
        run_test_omp(1024, 1024, 4096, "shared-A: A=4MB B=8MB K=1024");

        /* Independent-A tests (batched GEMM, each thread own data) */
        run_test_indep( 256,  512,  768, "indep: per-T A=512KB B=1.5MB");
        run_test_indep( 512,  512,  768, "indep: per-T A=1MB B=1.5MB");
        run_test_indep( 512, 1024,  768, "indep: per-T A=2MB B=3MB");
    } else
#endif
    {
        /* Single-threaded tests */
        run_test( 256, 512,  768, "1T: L1 stress (B_tile=96KB)");
        run_test(1024, 512, 4096, "1T: A=2MB+B=8MB overflows L2");
    }

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    return 0;
}
