// bench_attention_debug.c
// Debug version testing only L=512

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fused_attention.h"

static inline uint64_t rdtimer(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdfreq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

// Initialize with random values
static void init_matrix(int8_t* M, int L, int d) {
    for (int i = 0; i < L * d; i++) {
        M[i] = (int8_t)((rand() % 7) - 3);
    }
}

// Initialize V with positive bias
static void init_matrix_positive(int8_t* M, int L, int d) {
    for (int i = 0; i < L * d; i++) {
        M[i] = (int8_t)(rand() % 5);
    }
}

// Compare with tolerance
static int compare_outputs(const int32_t* O_fused, const int32_t* O_ref,
                           int L, int d, float* max_rel_err) {
    int errors = 0;
    *max_rel_err = 0.0f;

    for (int i = 0; i < L * d; i++) {
        float v_fused = (float)O_fused[i];
        float v_ref = (float)O_ref[i];
        float diff = fabsf(v_fused - v_ref);
        float ref_abs = fabsf(v_ref);
        float rel_err = (ref_abs > 1.0f) ? (diff / ref_abs) : diff;

        if (rel_err > *max_rel_err) *max_rel_err = rel_err;

        if (rel_err > 0.5f && diff > 5.0f) {
            if (errors < 10) {
                printf("Mismatch at [%d,%d]: fused=%.1f, ref=%.1f, rel_err=%.2f\n",
                       i / d, i % d, v_fused, v_ref, rel_err);
            }
            errors++;
        }
    }
    return errors;
}

int main(int argc, char** argv) {
    printf("=== Debug Attention Benchmark (L=512 only) ===\n\n");

    uint64_t freq = rdfreq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    if (freq == 0) {
        printf("ERROR: Invalid timer frequency\n");
        return 1;
    }

    int L = 512;
    int d = 256;
    float scale = 1.0f / sqrtf((float)d);
    printf("L=%d, d=%d, scale=%.6f\n\n", L, d, scale);

    // Allocate matrices
    printf("Allocating matrices...\n");
    int8_t* Q = malloc((size_t)L * d);
    int8_t* K = malloc((size_t)L * d);
    int8_t* V = malloc((size_t)L * d);
    int32_t* O_fused = calloc((size_t)L * d, sizeof(int32_t));
    int32_t* O_ref = calloc((size_t)L * d, sizeof(int32_t));

    if (!Q || !K || !V || !O_fused || !O_ref) {
        printf("Memory allocation failed\n");
        return 1;
    }
    printf("Allocated %zu bytes\n", 3 * (size_t)L * d + 2 * (size_t)L * d * sizeof(int32_t));

    // Initialize
    printf("Initializing matrices...\n");
    srand(42);
    init_matrix(Q, L, d);
    init_matrix(K, L, d);
    init_matrix_positive(V, L, d);

    // Pack matrices
    printf("Packing matrices...\n");
    fflush(stdout);

    uint64_t t0 = rdtimer();
    fused_matrix_t* Qpack = pack_A_fused(Q, L, d);
    uint64_t t1 = rdtimer();
    printf("  Qpack: %p, size=%zu, time=%.3f ms\n",
           (void*)Qpack, Qpack ? Qpack->size : 0,
           (double)(t1 - t0) / (double)freq * 1000.0);
    fflush(stdout);

    t0 = rdtimer();
    fused_matrix_t* Kpack = pack_B_fused(K, L, d);
    t1 = rdtimer();
    printf("  Kpack: %p, size=%zu, time=%.3f ms\n",
           (void*)Kpack, Kpack ? Kpack->size : 0,
           (double)(t1 - t0) / (double)freq * 1000.0);
    fflush(stdout);

    t0 = rdtimer();
    fused_matrix_t* Vpack = pack_C_fused(V, L, d);
    t1 = rdtimer();
    printf("  Vpack: %p, size=%zu, time=%.3f ms\n",
           (void*)Vpack, Vpack ? Vpack->size : 0,
           (double)(t1 - t0) / (double)freq * 1000.0);
    fflush(stdout);

    if (!Qpack || !Kpack || !Vpack) {
        printf("Packing failed\n");
        return 1;
    }

    // Compute reference
    printf("Computing reference...\n");
    fflush(stdout);
    t0 = rdtimer();
    ref_attention(Q, K, V, O_ref, L, d, scale);
    t1 = rdtimer();
    printf("Reference time: %.3f ms\n", (double)(t1 - t0) / (double)freq * 1000.0);
    fflush(stdout);

    // Print some reference output values
    printf("Reference O[0,0..7]: ");
    for (int j = 0; j < 8; j++) {
        printf("%d ", O_ref[j]);
    }
    printf("\n");

    // Test online softmax
    printf("\n--- Online Softmax ---\n");
    fflush(stdout);

    memset(O_fused, 0, (size_t)L * d * sizeof(int32_t));
    t0 = rdtimer();
    fused_attention_online(Qpack, Kpack, V, O_fused, d, scale);
    t1 = rdtimer();
    printf("Online softmax time: %.3f ms\n", (double)(t1 - t0) / (double)freq * 1000.0);
    fflush(stdout);

    // Print some fused output values
    printf("Fused O[0,0..7]: ");
    for (int j = 0; j < 8; j++) {
        printf("%d ", O_fused[j]);
    }
    printf("\n");

    // Compare
    float max_rel_err;
    int errors = compare_outputs(O_fused, O_ref, L, d, &max_rel_err);
    printf("Errors: %d, max_rel_err: %.2f\n", errors, max_rel_err);

    // Cleanup
    free_fused_matrix(Qpack);
    free_fused_matrix(Kpack);
    free_fused_matrix(Vpack);
    free(Q);
    free(K);
    free(V);
    free(O_fused);
    free(O_ref);

    printf("\n=== Done ===\n");
    return 0;
}
