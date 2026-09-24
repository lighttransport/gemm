/* Isolated packed NVFP4 N=3 projection for qlair A64FX cycle/bandwidth study.
 * Synthetic weights have the real Qwen3.8 matrix dimensions and packed bytes. */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

extern int q38_nvfp4_packed_n3_mt(float *, const void *, const float *,
                                   int, int, int);
extern int q38_nvfp4_packed_n1_rows(float *, const void *, const float *,
                                    int, int);
typedef struct { int8_t lo[64], hi[64]; } packed_a8_act;
extern void q38_nvfp4_packed_a8_prepare(packed_a8_act *, const int8_t *, int);
extern int q38_nvfp4_packed_a8_rows(float *, const void *,
                                    const packed_a8_act *, float, int, int);
extern int q38_super_bench_quantize(int8_t *, int8_t *, float[3],
                                    const float *, int);
extern void q38_super_bench_rows(float *, const int8_t *, const int32_t *,
                                  const size_t *, const int8_t *, const int8_t *,
                                  const float[3], const float *,
                                  int, int, int, int);
extern void q38_super_bench_rows_a8(float *, const int8_t *, const int8_t *,
                                     const float[3], int, int, int, int);
extern void q38_super_bench_rows_a8_n1(float *, const int8_t *, const int8_t *,
                                        float, int, int, int);
__attribute__((noinline)) void __qlair_sim_start(unsigned long id) {
    (void)id; __asm__ volatile("" ::: "memory");
}
__attribute__((noinline)) void __qlair_sim_end(unsigned long id) {
    (void)id; __asm__ volatile("" ::: "memory");
}
static uint64_t ticks(void) {
    uint64_t v;
    __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v));
    return v;
}
typedef struct { float d[8]; uint8_t qs[64]; } packed_subblock;
typedef struct { packed_subblock s[4]; } packed_block;
_Static_assert(sizeof(packed_block) == 384, "packed tile width");

static void init_weights(packed_block *w, size_t blocks) {
    for (size_t b = 0; b < blocks; b++)
        for (int s = 0; s < 4; s++) {
            packed_subblock *p = &w[b].s[s];
            for (int r = 0; r < 8; r++) {
                static const float scales[] = {
                    0x1p-10f, 0x1p-9f, 0x1.8p-9f, 0x1p-8f};
                p->d[r] = scales[(b + s + r) & 3];
                for (int j = 0; j < 8; j++)
                    p->qs[r * 8 + j] = (uint8_t)(((b + r + j) & 15) |
                        (((b * 3 + r * 5 + j) & 15) << 4));
            }
        }
}

static uint8_t stream_weights(const uint8_t *p, size_t bytes) {
    const svbool_t pg = svptrue_b8();
    svuint8_t a = svdup_u8(0);
    for (size_t i = 0; i < bytes; i += 64)
        a = sveor_u8_x(pg, a, svld1_u8(pg, p + i));
    return svorv_u8(pg, a);
}

static float *activation;
static int8_t *digits;
static packed_a8_act *packed_act;
static float digit_scale[3];
static int bench_cols, bench_sdot, bench_n1, bench_pack_a8;

/* Validate one row per worker outside the marked region. This catches
 * incomplete simulator execution as well as nibble/order regressions. */
static int check_first_row(const uint8_t *weights, const float *y,
                           int local_rows) {
    static const float code[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    for (int c = 0; c < (bench_n1 ? 1 : 3); c++) {
        float expected = 0.0f;
        if (bench_sdot) {
            const int8_t *w = (const int8_t *)weights;
            int64_t sum = 0;
            for (int k = 0; k < bench_cols; k++)
                sum += (int)w[(size_t)(k & ~3) * 64 + k % 4] *
                       (int)digits[(size_t)c * bench_cols + k];
            expected = (float)sum * digit_scale[c];
        } else if (bench_pack_a8) {
            const packed_block *w = (const packed_block *)weights;
            for (int ib = 0; ib < bench_cols / 64; ib++)
                for (int s = 0; s < 4; s++) {
                    const packed_subblock *p = &w[ib].s[s];
                    int sum = 0;
                    for (int j = 0; j < 8; j++) {
                        uint8_t z = p->qs[j];
                        int k = ib * 64 + s * 16 + j;
                        sum += (int)code[z & 15] * digits[k];
                        sum += (int)code[z >> 4] * digits[k + 8];
                    }
                    expected += (float)sum * p->d[0] * digit_scale[0];
                }
        } else {
            const packed_block *w = (const packed_block *)weights;
            for (int ib = 0; ib < bench_cols / 64; ib++)
                for (int s = 0; s < 4; s++) {
                    const packed_subblock *p = &w[ib].s[s];
                    for (int j = 0; j < 8; j++) {
                        uint8_t z = p->qs[j];
                        int k = ib * 64 + s * 16 + j;
                        expected = fmaf(code[z & 15] * p->d[0],
                            activation[(size_t)c * bench_cols + k], expected);
                    }
                    for (int j = 0; j < 8; j++) {
                        uint8_t z = p->qs[j];
                        int k = ib * 64 + s * 16 + j;
                        expected = fmaf(code[z >> 4] * p->d[0],
                            activation[(size_t)c * bench_cols + k + 8], expected);
                    }
                }
        }
        float actual = y[(size_t)c * local_rows];
        if (!isfinite(actual) || fabsf(actual - expected) >
                0.0005f + 0.0005f * fabsf(expected)) return 0;
    }
    return 1;
}

static pthread_barrier_t start_barrier, end_barrier;
static uint8_t *segments[4];
static int bench_rows, bench_cores, bench_passes, bench_warmup;
static int bench_compute, bench_a8;
typedef struct {
    int tid, error;
    uint64_t begin, end;
    float checksum;
} bench_worker;
static bench_worker workers[48];

static void *run_worker(void *arg) {
    bench_worker *worker = arg;
    int tid = worker->tid;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(12 + tid, &mask);
    worker->error = sched_setaffinity(0, sizeof(mask), &mask) != 0;
    int group_rows = bench_sdot ? 64 : 8;
    int groups = bench_rows / group_rows, nb = bench_cols / 64;
    int cmgs = (bench_cores + 11) / 12, cmg = tid / 12;
    int first = groups * tid / bench_cores;
    int last = groups * (tid + 1) / bench_cores;
    int cmg_first = groups * cmg / cmgs;
    int local_rows = (last - first) * group_rows;
    size_t group_bytes = bench_sdot ? (size_t)bench_cols * 64 :
                                      (size_t)nb * sizeof(packed_block);
    uint8_t *weights = segments[cmg] + (size_t)(first - cmg_first) * group_bytes;
    size_t bytes = (size_t)(last - first) * group_bytes;
    float *y = NULL;
    int32_t *weight_sum = NULL;
    size_t *rare_offsets = NULL;
    if (posix_memalign((void **)&y, 256,
                       (size_t)3 * local_rows * sizeof(float)))
        worker->error = 1;
    if (bench_sdot && !bench_a8 &&
        (posix_memalign((void **)&weight_sum, 256,
                        (size_t)local_rows * sizeof(int32_t)) ||
         posix_memalign((void **)&rare_offsets, 256,
                        ((size_t)local_rows + 1) * sizeof(size_t))))
        worker->error = 1;
    if (bench_sdot && !bench_a8 && !worker->error) {
        memset(weight_sum, 0, (size_t)local_rows * sizeof(int32_t));
        memset(rare_offsets, 0, ((size_t)local_rows + 1) * sizeof(size_t));
    }
    volatile uint8_t stream_sum = 0;
    for (int rep = 0; rep < bench_warmup && !worker->error; rep++) {
        if (bench_compute) {
            if (bench_pack_a8) {
                if (!q38_nvfp4_packed_a8_rows(y, weights, packed_act,
                        digit_scale[0], local_rows, bench_cols)) worker->error = 1;
            } else if (bench_a8 && bench_n1)
                q38_super_bench_rows_a8_n1(y, (const int8_t *)weights,
                    digits, digit_scale[0], bench_cols, 0, local_rows / 64);
            else if (bench_a8)
                q38_super_bench_rows_a8(y, (const int8_t *)weights, digits,
                    digit_scale, local_rows, bench_cols, 0, local_rows / 64);
            else if (bench_sdot)
                q38_super_bench_rows(y, (const int8_t *)weights, weight_sum,
                    rare_offsets, digits, digits + (size_t)3 * bench_cols,
                    digit_scale, activation, local_rows, bench_cols,
                    0, local_rows / 64);
            else if (bench_n1 ?
                     !q38_nvfp4_packed_n1_rows(y, weights, activation,
                                               local_rows, bench_cols) :
                     !q38_nvfp4_packed_n3_mt(y, weights, activation,
                                              local_rows, bench_cols, 1))
                    worker->error = 1;
        } else stream_sum ^= stream_weights((const uint8_t *)weights, bytes);
    }
    pthread_barrier_wait(&start_barrier);
    __qlair_sim_start(0);
    worker->begin = ticks();
    for (int rep = 0; rep < bench_passes && !worker->error; rep++) {
        if (bench_compute) {
            if (bench_pack_a8) {
                if (!q38_nvfp4_packed_a8_rows(y, weights, packed_act,
                        digit_scale[0], local_rows, bench_cols)) worker->error = 1;
            } else if (bench_a8 && bench_n1)
                q38_super_bench_rows_a8_n1(y, (const int8_t *)weights,
                    digits, digit_scale[0], bench_cols, 0, local_rows / 64);
            else if (bench_a8)
                q38_super_bench_rows_a8(y, (const int8_t *)weights, digits,
                    digit_scale, local_rows, bench_cols, 0, local_rows / 64);
            else if (bench_sdot)
                q38_super_bench_rows(y, (const int8_t *)weights, weight_sum,
                    rare_offsets, digits, digits + (size_t)3 * bench_cols,
                    digit_scale, activation, local_rows, bench_cols,
                    0, local_rows / 64);
            else if (bench_n1 ?
                     !q38_nvfp4_packed_n1_rows(y, weights, activation,
                                               local_rows, bench_cols) :
                     !q38_nvfp4_packed_n3_mt(y, weights, activation,
                                              local_rows, bench_cols, 1))
                    worker->error = 1;
        } else stream_sum ^= stream_weights((const uint8_t *)weights, bytes);
    }
    worker->end = ticks();
    __qlair_sim_end(0);
    if (bench_compute && !worker->error && (bench_a8 || !bench_sdot) &&
        !check_first_row(weights, y, local_rows)) worker->error = 1;
    if (bench_compute && !worker->error && !isfinite(y[0])) worker->error = 1;
    worker->checksum = bench_compute && !worker->error ? y[0] : (float)stream_sum;
    pthread_barrier_wait(&end_barrier);
    free(y); free(weight_sum); free(rare_offsets);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 7 || (strcmp(argv[1], "compute") &&
                      strcmp(argv[1], "compute1") &&
                      strcmp(argv[1], "packed8_1") &&
                      strcmp(argv[1], "stream") &&
                      strcmp(argv[1], "stream1") &&
                      strcmp(argv[1], "sdot") &&
                      strcmp(argv[1], "sdot_stream") &&
                      strcmp(argv[1], "sdot8") &&
                      strcmp(argv[1], "sdot8_1") &&
                      strcmp(argv[1], "sdot8_stream"))) {
        fprintf(stderr, "usage: %s compute|compute1|packed8_1|stream|stream1|sdot|sdot_stream|sdot8|sdot8_1|sdot8_stream ROWS COLS CORES PASSES WARMUP\n", argv[0]);
        return 2;
    }
    bench_compute = !strcmp(argv[1], "compute") || !strcmp(argv[1], "compute1") || !strcmp(argv[1], "packed8_1") || !strcmp(argv[1], "sdot") ||
                    !strcmp(argv[1], "sdot8") || !strcmp(argv[1], "sdot8_1");
    bench_pack_a8 = !strcmp(argv[1], "packed8_1");
    bench_n1 = !strcmp(argv[1], "compute1") || !strcmp(argv[1], "stream1") ||
               !strcmp(argv[1], "sdot8_1") || bench_pack_a8;
    bench_a8 = !strcmp(argv[1], "sdot8") || !strcmp(argv[1], "sdot8_1") ||
               !strcmp(argv[1], "sdot8_stream");
    bench_sdot = bench_a8 || !strcmp(argv[1], "sdot") ||
                 !strcmp(argv[1], "sdot_stream");
    bench_rows = atoi(argv[2]); bench_cols = atoi(argv[3]);
    bench_cores = atoi(argv[4]); bench_passes = atoi(argv[5]);
    bench_warmup = atoi(argv[6]);
    int group_rows = bench_sdot ? 64 : 8;
    if (bench_rows < group_rows || bench_rows % group_rows || bench_cols < 64 ||
        bench_cols % 64 || (bench_cores != 1 && bench_cores != 12 &&
                              bench_cores != 48) ||
        bench_rows / group_rows < bench_cores || bench_passes < 1 ||
        bench_passes > 1000 || bench_warmup < 0 || bench_warmup > 10 ||
        svcntb() != 64) return 2;
    int groups = bench_rows / group_rows, nb = bench_cols / 64;
    int cmgs = (bench_cores + 11) / 12;
    size_t group_bytes = bench_sdot ? (size_t)bench_cols * 64 :
                                      (size_t)nb * sizeof(packed_block);
    size_t total_bytes = (size_t)groups * group_bytes;
    for (int cmg = 0; cmg < cmgs; cmg++) {
        size_t cmg_groups = (size_t)(groups * (cmg + 1) / cmgs -
                                     groups * cmg / cmgs);
        size_t bytes = cmg_groups * group_bytes;
        size_t alloc = (bytes + 2097151) & ~(size_t)2097151;
        if (posix_memalign((void **)&segments[cmg], 2097152, alloc)) return 1;
        if (!getenv("Q38_QLAIR_NO_MBIND")) {
            unsigned long nodes = 1UL << (4 + cmg);
            if (syscall(SYS_mbind, segments[cmg], alloc, 2, &nodes, 64UL, 0UL)) {
                perror("mbind");
                return 1;
            }
        }
        if (bench_sdot) {
            for (size_t i = 0; i < bytes; i++)
                segments[cmg][i] = (uint8_t)((int)((i * 37 + cmg * 11) % 241) - 120);
        } else init_weights((packed_block *)segments[cmg], cmg_groups * nb);
    }
    if (posix_memalign((void **)&activation, 256,
                       (size_t)3 * bench_cols * sizeof(float))) return 1;
    for (int i = 0; i < 3 * bench_cols; i++)
        activation[i] = (float)((i * 17) % 101 - 50) * 0.015625f;
    if (bench_sdot || bench_pack_a8) {
        if (posix_memalign((void **)&digits, 256,
                           (size_t)6 * bench_cols)) return 1;
        if (bench_a8 || bench_pack_a8) {
            for (int c = 0; c < 3; c++) {
                float maxabs = 0.0f;
                for (int k = 0; k < bench_cols; k++) {
                    float a = fabsf(activation[(size_t)c * bench_cols + k]);
                    if (a > maxabs) maxabs = a;
                }
                digit_scale[c] = maxabs / (127.0f * (bench_pack_a8 ? 1.0f : 1024.0f));
                float inv = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
                for (int k = 0; k < bench_cols; k++) {
                    int q = (int)lrintf(activation[(size_t)c * bench_cols + k] * inv);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    digits[(size_t)c * bench_cols + k] = (int8_t)q;
                }
            }
        } else if (q38_super_bench_quantize(digits,
                   digits + (size_t)3 * bench_cols, digit_scale,
                   activation, bench_cols)) return 1;
    }
    if (bench_pack_a8) {
        if (posix_memalign((void **)&packed_act, 256,
                           (size_t)(bench_cols / 16) * sizeof(*packed_act))) return 1;
        q38_nvfp4_packed_a8_prepare(packed_act, digits, bench_cols);
    }
    pthread_barrier_init(&start_barrier, NULL, (unsigned)bench_cores);
    pthread_barrier_init(&end_barrier, NULL, (unsigned)bench_cores);
    pthread_t tids[48];
    for (int tid = 0; tid < bench_cores; tid++) workers[tid].tid = tid;
    for (int tid = 1; tid < bench_cores; tid++)
        if (pthread_create(&tids[tid], NULL, run_worker, &workers[tid])) return 1;
    run_worker(&workers[0]);
    for (int tid = 1; tid < bench_cores; tid++) pthread_join(tids[tid], NULL);
    uint64_t slow = 0, first = UINT64_MAX, last = 0;
    int errors = 0;
    for (int tid = 0; tid < bench_cores; tid++) {
        uint64_t duration = workers[tid].end - workers[tid].begin;
        if (duration > slow) slow = duration;
        if (workers[tid].begin < first) first = workers[tid].begin;
        if (workers[tid].end > last) last = workers[tid].end;
        errors += workers[tid].error;
    }
    printf("##FP4 mode=%s rows=%d cols=%d cores=%d passes=%d bytes=%zu "
           "worker_ticks=%llu makespan_ticks=%llu correct=%d\n",
           argv[1], bench_rows, bench_cols, bench_cores, bench_passes,
           total_bytes * (size_t)bench_passes, (unsigned long long)slow,
           (unsigned long long)(last - first), errors == 0);
    for (int cmg = 0; cmg < cmgs; cmg++) free(segments[cmg]);
    free(activation); free(digits); free(packed_act);
    pthread_barrier_destroy(&start_barrier);
    pthread_barrier_destroy(&end_barrier);
    return errors ? 1 : 0;
}
