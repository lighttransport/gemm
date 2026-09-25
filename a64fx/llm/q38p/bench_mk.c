/* q38p microkernel: exactness vs scalar reference, single-core and all-core rate */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
void q38p_mk_r4t5(const void *, const void *, const double *, double *, long, long);
void q38p_mk_r5t4(const void *, const void *, const double *, double *, long, long);
void q38p_mk_r3t7(const void *, const void *, const double *, double *, long, long);
void q38p_mk_r4t4(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r4t5_p0(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r4t5_p1024(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r4t5_p2048(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r4t5_p4096(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r3t7_p0(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r3t7_p1024(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r3t7_p2048(const void *, const void *, const double *, double *, long, long);
void q38p_mkp_r3t7_p4096(const void *, const void *, const double *, double *, long, long);
typedef void (*mkf)(const void *, const void *, const double *, double *, long, long);
static struct { const char *n; mkf f; int R, T; } K[] = {{"r4t5", q38p_mk_r4t5, 4, 5}, {"r5t4", q38p_mk_r5t4, 5, 4}, {"r3t7", q38p_mk_r3t7, 3, 7}, {"r4t4", q38p_mk_r4t4, 4, 4}, {"r4t5_p0", q38p_mkp_r4t5_p0, 4, 5},{"r4t5_p1024", q38p_mkp_r4t5_p1024, 4, 5},{"r4t5_p2048", q38p_mkp_r4t5_p2048, 4, 5},{"r4t5_p4096", q38p_mkp_r4t5_p4096, 4, 5},{"r3t7_p0", q38p_mkp_r3t7_p0, 3, 7},{"r3t7_p1024", q38p_mkp_r3t7_p1024, 3, 7},{"r3t7_p2048", q38p_mkp_r3t7_p2048, 3, 7},{"r3t7_p4096", q38p_mkp_r3t7_p4096, 3, 7}};
static inline uint64_t cyc(void) { uint64_t v; __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline uint64_t hz(void) { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
static uint64_t rs = 88172645463325252ull;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static long STEPS = 128, NG = 48, REPS = 200;
static int KI;
typedef struct { int cpu; double sdot_per_cycle; } targ;
static void *run(void *p) {
    targ *ta = p;
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(ta->cpu, &m); sched_setaffinity(0, sizeof m, &m);
    int R = K[KI].R, T = K[KI].T;
    int16_t *W = aligned_alloc(256, STEPS * R * 64 + 256);
    int16_t *A = aligned_alloc(256, NG * STEPS * T * 8 + 256);
    double *S = aligned_alloc(256, NG * T * 8 + 256), *O = aligned_alloc(256, NG * R * T * 64 + 256);
    for (long i = 0; i < STEPS * R * 32; i++) W[i] = (int16_t)((int)(rnd() % 23041) - 11520);
    for (long i = 0; i < NG * STEPS * T * 4; i++) A[i] = (int16_t)((int)(rnd() % 65535) - 32767);
    for (long i = 0; i < NG * T; i++) S[i] = ldexp(1.0 + (rnd() % 1000) / 1000.0, -20);
    memset(O, 0, NG * R * T * 64);
    K[KI].f(W, A, S, O, NG, STEPS);
    if (ta->cpu == 12) {   /* exactness check on one thread */
        long bad = 0;
        for (long g = 0; g < NG; g++) for (int r = 0; r < R; r++) for (int t = 0; t < T; t++) for (int l = 0; l < 8; l++) {
            int64_t acc = 0;
            for (long s = 0; s < STEPS; s++) for (int k = 0; k < 4; k++)
                acc += (int64_t)W[(s * R + r) * 32 + l * 4 + k] * A[((g * STEPS + s) * T + t) * 4 + k];
            double ref = fma((double)acc, S[g * T + t], 0.0);
            if (O[((g * R + r) * T + t) * 8 + l] != ref) bad++;
        }
        printf("%s exact check: %ld mismatches\n", K[KI].n, bad);
    }
    uint64_t a = cyc();
    for (long rep = 0; rep < REPS; rep++) K[KI].f(W, A, S, O, NG, STEPS);
    uint64_t b = cyc();
    double cycles = (double)(b - a) * 2.0e9 / hz();
    ta->sdot_per_cycle = (double)REPS * NG * STEPS * R * T / cycles;
    return NULL;
}
int main(int argc, char **argv) {
    int nth = argc > 1 ? atoi(argv[1]) : 1;
    if (argc > 2) STEPS = atol(argv[2]);
    if (argc > 3) NG = atol(argv[3]);
    for (KI = 0; KI < (int)(sizeof K / sizeof K[0]); KI++) {
        pthread_t th[48]; targ ta[48];
        for (int i = 0; i < nth; i++) { ta[i].cpu = 12 + i; pthread_create(&th[i], NULL, run, &ta[i]); }
        double mn = 1e9, sum = 0;
        for (int i = 0; i < nth; i++) { pthread_join(th[i], NULL); sum += ta[i].sdot_per_cycle; if (ta[i].sdot_per_cycle < mn) mn = ta[i].sdot_per_cycle; }
        printf("%s threads=%d steps=%ld groups=%ld: sdot/cycle mean %.3f (%.1f%%) min %.3f (%.1f%%)\n", K[KI].n, nth, STEPS, NG,
               sum / nth, 50 * sum / nth, mn, 50 * mn);
    }
    return 0;
}
