/*
 * Fused Online Softmax Kernel for Flash Attention
 *
 * Traditional approach:
 *   S[M,N] = Q[M,D] @ K[N,D]^T     // Full Phase 1
 *   P[M,N] = softmax(S)             // Full softmax
 *   O[M,D] = P[M,N] @ V[N,D]        // Full Phase 2
 *
 * Online (chunked) approach:
 *   for each N_chunk:
 *     S_chunk = Q @ K_chunk^T       // Partial Phase 1
 *     Update running max/sum        // Online softmax
 *     Scale previous O              // Correction
 *     O += softmax(S_chunk) @ V_chunk  // Partial Phase 2
 *
 * Benefits:
 *   - S_chunk fits in registers (no memory store)
 *   - Eliminates D-tile loop overhead
 *   - Better cache utilization (Q reused, K/V streamed)
 *
 * Parameters:
 *   M = 4 rows (MR=4 for register fit)
 *   D = 256 (head dimension)
 *   N = 64 (sequence length)
 *   N_chunk = 16 (chunk size)
 */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static inline uint64_t rdcycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// ============================================================
// Reference: Separate Phase 1 + Softmax + Phase 2
// ============================================================
void attention_reference(
    const int8_t* Q,    // [4, 256] row-major
    const int8_t* K,    // [64, 256] row-major
    const int8_t* V,    // [64, 256] row-major
    float* O,           // [4, 256] output
    int M, int N, int D
) {
    // Phase 1: Q @ K^T -> S[M, N]
    float S[4 * 64];
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int d = 0; d < D; d++) {
                sum += (int32_t)Q[m * D + d] * (int32_t)K[n * D + d];
            }
            S[m * N + n] = (float)sum * 0.01f;  // Scale for softmax stability
        }
    }

    // Softmax per row
    float P[4 * 64];
    for (int m = 0; m < M; m++) {
        float max_val = -FLT_MAX;
        for (int n = 0; n < N; n++) {
            if (S[m * N + n] > max_val) max_val = S[m * N + n];
        }
        float sum = 0;
        for (int n = 0; n < N; n++) {
            P[m * N + n] = expf(S[m * N + n] - max_val);
            sum += P[m * N + n];
        }
        for (int n = 0; n < N; n++) {
            P[m * N + n] /= sum;
        }
    }

    // Phase 2: P @ V -> O[M, D]
    for (int m = 0; m < M; m++) {
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int n = 0; n < N; n++) {
                sum += P[m * N + n] * (float)V[n * D + d];
            }
            O[m * D + d] = sum;
        }
    }
}

// ============================================================
// Online Softmax: Process N in chunks
// ============================================================
void attention_online(
    const int8_t* Q,    // [4, 256]
    const int8_t* K,    // [64, 256]
    const int8_t* V,    // [64, 256]
    float* O,           // [4, 256]
    int M, int N, int D, int N_chunk
) {
    // Running statistics for online softmax
    float running_max[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float running_sum[4] = {0, 0, 0, 0};

    // Initialize output to zero
    memset(O, 0, M * D * sizeof(float));

    for (int n_start = 0; n_start < N; n_start += N_chunk) {
        int chunk_size = (n_start + N_chunk <= N) ? N_chunk : (N - n_start);

        // Phase 1 chunk: Q @ K_chunk^T -> S_chunk[M, chunk_size]
        float S_chunk[4 * 16];  // Max chunk size 16
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < chunk_size; n++) {
                int32_t sum = 0;
                for (int d = 0; d < D; d++) {
                    sum += (int32_t)Q[m * D + d] * (int32_t)K[(n_start + n) * D + d];
                }
                S_chunk[m * N_chunk + n] = (float)sum * 0.01f;
            }
        }

        // Online softmax update
        for (int m = 0; m < M; m++) {
            // Find chunk max
            float chunk_max = -FLT_MAX;
            for (int n = 0; n < chunk_size; n++) {
                if (S_chunk[m * N_chunk + n] > chunk_max) {
                    chunk_max = S_chunk[m * N_chunk + n];
                }
            }

            // Update global max
            float new_max = (running_max[m] > chunk_max) ? running_max[m] : chunk_max;

            // Correction factor for previous values
            float correction = expf(running_max[m] - new_max);

            // Scale previous O and sum
            for (int d = 0; d < D; d++) {
                O[m * D + d] *= correction;
            }
            running_sum[m] *= correction;

            // Compute exp and accumulate
            float exp_sum = 0;
            float P_chunk[16];
            for (int n = 0; n < chunk_size; n++) {
                P_chunk[n] = expf(S_chunk[m * N_chunk + n] - new_max);
                exp_sum += P_chunk[n];
            }
            running_sum[m] += exp_sum;
            running_max[m] = new_max;

            // Phase 2 chunk: P_chunk @ V_chunk -> accumulate into O
            for (int d = 0; d < D; d++) {
                float sum = 0;
                for (int n = 0; n < chunk_size; n++) {
                    sum += P_chunk[n] * (float)V[(n_start + n) * D + d];
                }
                O[m * D + d] += sum;
            }
        }
    }

    // Final normalization
    for (int m = 0; m < M; m++) {
        for (int d = 0; d < D; d++) {
            O[m * D + d] /= running_sum[m];
        }
    }
}

// ============================================================
// Optimized Fused GEMM-only (no softmax, just memory pattern)
// ============================================================
// This tests the fused GEMM structure without softmax overhead
__attribute__((noinline))
void fused_gemm_chunked(
    const int8_t* Q,    // [4, 256] - packed for SDOT
    const int8_t* K,    // [64, 64, 4] - interleaved
    const int8_t* V,    // [64, 256] - will be accessed in chunks
    int32_t* O,         // [4, 256]
    int32_t* S_buf      // [4, 64] scratch for S values
) {
    // Process N in chunks of 16
    // For each chunk: compute S_chunk, then immediately use for P@V

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Initialize O to zero
        "mov x0, %[O]\n\t"
        "eor z0.d, z0.d, z0.d\n\t"
        "mov x1, #16\n\t"  // 4 rows × 4 D-tiles = 16 stores
        "1:\n\t"
        "st1w {z0.s}, p1, [x0]\n\t"
        "add x0, x0, #64\n\t"
        "subs x1, x1, #1\n\t"
        "b.ne 1b\n\t"

        // Process 4 N-chunks of 16 each
        "mov x14, #4\n\t"           // N-chunk counter
        "mov x2, %[K]\n\t"          // K pointer
        "mov x3, %[V]\n\t"          // V pointer

        "10:\n\t"  // N-chunk loop

        // ===== Phase 1: Q[4,256] @ K_chunk[256,16]^T -> S[4,16] =====
        // S will be stored in z8-z11 (4 rows × 1 vector of 16 int32)
        "eor z8.d, z8.d, z8.d\n\t"   // S row 0
        "eor z9.d, z9.d, z9.d\n\t"   // S row 1
        "eor z10.d, z10.d, z10.d\n\t" // S row 2
        "eor z11.d, z11.d, z11.d\n\t" // S row 3

        "mov x4, %[Q]\n\t"          // Q pointer row 0
        "add x17, x4, #256\n\t"     // Q pointer row 1
        "add x18, x4, #512\n\t"     // Q pointer row 2
        "add x19, x4, #768\n\t"     // Q pointer row 3
        "mov x5, x2\n\t"            // K_chunk pointer
        "mov x6, #64\n\t"           // 256/4 = 64 K-groups

        "20:\n\t"  // K-group loop for Phase 1
        // Load K_chunk: 16 N values × 4 bytes = 64 bytes = 1 vector
        // K is [N, D] but we need K^T, so K[n, k_group] for all n
        // With interleaved layout: K[k_group, n, 4] at x5
        "ld1b {z0.b}, p0/z, [x5]\n\t"  // K[k_group, 0:16, 4]

        // Load Q: 4 rows × 4 bytes = 16 bytes broadcast
        "ld1rw {z4.s}, p1/z, [x4]\n\t"         // Q[0, k_group*4:(k_group+1)*4]
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x17]\n\t"        // Q[1, ...]
        "sdot z9.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x18]\n\t"        // Q[2, ...]
        "sdot z10.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x19]\n\t"        // Q[3, ...]
        "sdot z11.s, z0.b, z4.b\n\t"

        "add x4, x4, #4\n\t"        // Next K-group in Q row 0
        "add x17, x17, #4\n\t"      // Next K-group in Q row 1
        "add x18, x18, #4\n\t"      // Next K-group in Q row 2
        "add x19, x19, #4\n\t"      // Next K-group in Q row 3
        "add x5, x5, #64\n\t"       // Next K-group in K (16 N × 4 bytes)
        "subs x6, x6, #1\n\t"
        "b.ne 20b\n\t"

        // S[4,16] is now in z8-z11
        // Store to scratch buffer for Phase 2 (or keep in registers if possible)
        "mov x7, %[S_buf]\n\t"
        "st1w {z8.s}, p1, [x7]\n\t"
        "st1w {z9.s}, p1, [x7, #1, mul vl]\n\t"
        "st1w {z10.s}, p1, [x7, #2, mul vl]\n\t"
        "st1w {z11.s}, p1, [x7, #3, mul vl]\n\t"

        // ===== Phase 2: S[4,16] @ V_chunk[16,256] -> O[4,256] =====
        // Process D in 4 tiles of 64 each
        "mov x8, #4\n\t"            // D-tile counter
        "mov x9, %[O]\n\t"          // O pointer
        "mov x10, x3\n\t"           // V_chunk pointer

        "30:\n\t"  // D-tile loop
        // Zero accumulators for this D-tile (4 rows × 4 K-tiles = 16 regs)
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"
        "eor z20.d, z20.d, z20.d\n\t" "eor z21.d, z21.d, z21.d\n\t"
        "eor z22.d, z22.d, z22.d\n\t" "eor z23.d, z23.d, z23.d\n\t"
        "eor z24.d, z24.d, z24.d\n\t" "eor z25.d, z25.d, z25.d\n\t"
        "eor z26.d, z26.d, z26.d\n\t" "eor z27.d, z27.d, z27.d\n\t"

        // Load S values (will broadcast each)
        "mov x7, %[S_buf]\n\t"
        "mov x11, x10\n\t"          // V pointer for this D-tile
        "mov x12, #16\n\t"          // N counter (16 values in chunk)

        "40:\n\t"  // N loop within chunk
        // Load V: 4 K-tiles × 64 bytes = 256 bytes
        "ld1b {z0.b}, p0/z, [x11]\n\t"
        "ld1b {z1.b}, p0/z, [x11, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x11, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x11, #3, mul vl]\n\t"

        // Broadcast S[row, n] and accumulate
        // S is stored as [4, 16] int32, so S[m, n] at offset m*64 + n*4
        "ld1rw {z4.s}, p1/z, [x7]\n\t"  // S[0, n]
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #64]\n\t"  // S[1, n]
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #128]\n\t"  // S[2, n]
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #192]\n\t"  // S[3, n]
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"

        "add x7, x7, #4\n\t"        // Next N in S
        "add x11, x11, #256\n\t"    // Next N in V (D stride)
        "subs x12, x12, #1\n\t"
        "b.ne 40b\n\t"

        // Reduce K-tiles: 4→2→1
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"

        // Accumulate into O (load, add, store)
        "ld1w {z0.s}, p1/z, [x9]\n\t"
        "add z12.s, z12.s, z0.s\n\t"
        "st1w {z12.s}, p1, [x9]\n\t"
        "add x9, x9, #1024\n\t"  // Next row in O

        "ld1w {z0.s}, p1/z, [x9]\n\t"
        "add z16.s, z16.s, z0.s\n\t"
        "st1w {z16.s}, p1, [x9]\n\t"
        "add x9, x9, #1024\n\t"

        "ld1w {z0.s}, p1/z, [x9]\n\t"
        "add z20.s, z20.s, z0.s\n\t"
        "st1w {z20.s}, p1, [x9]\n\t"
        "add x9, x9, #1024\n\t"

        "ld1w {z0.s}, p1/z, [x9]\n\t"
        "add z24.s, z24.s, z0.s\n\t"
        "st1w {z24.s}, p1, [x9]\n\t"

        // Move to next D-tile
        "sub x9, x9, #2048\n\t"
        "sub x9, x9, #1024\n\t"
        "add x9, x9, #64\n\t"
        "add x10, x10, #64\n\t"     // Next D-tile in V
        "subs x8, x8, #1\n\t"
        "b.ne 30b\n\t"

        // Move to next N-chunk
        "add x2, x2, #4096\n\t"     // K: 16 N × 64 K-groups × 4 = 4096
        "add x3, x3, #4096\n\t"     // V: 16 N × 256 D = 4096
        "subs x14, x14, #1\n\t"
        "b.ne 10b\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [V] "r"(V), [O] "r"(O), [S_buf] "r"(S_buf)
        : "memory",
          "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
          "x10", "x11", "x12", "x14", "x17", "x18", "x19",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "p0", "p1"
    );
}

// ============================================================
// Baseline separate Phase 1 + Phase 2 (for comparison)
// ============================================================
__attribute__((noinline))
void phase1_only(const int8_t* Q, const int8_t* K, int32_t* S) {
    // Q[4, 256] @ K[64, 256]^T -> S[4, 64]
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Process 16 N at a time (matches Phase 2's 16 N inner loop)
        "mov x15, #4\n\t"           // 4 N-chunks of 16
        "mov x2, %[K]\n\t"
        "mov x3, %[S]\n\t"

        "2:\n\t"
        "eor z8.d, z8.d, z8.d\n\t"
        "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t"
        "eor z11.d, z11.d, z11.d\n\t"

        "mov x4, %[Q]\n\t"
        "add x17, x4, #256\n\t"     // Q row 1
        "add x18, x4, #512\n\t"     // Q row 2
        "add x19, x4, #768\n\t"     // Q row 3
        "mov x5, x2\n\t"
        "mov x6, #64\n\t"           // 256/4 K-groups

        "3:\n\t"
        "ld1b {z0.b}, p0/z, [x5]\n\t"
        "ld1rw {z4.s}, p1/z, [x4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x17]\n\t"
        "sdot z9.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x18]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x19]\n\t"
        "sdot z11.s, z0.b, z4.b\n\t"
        "add x4, x4, #4\n\t"
        "add x17, x17, #4\n\t"
        "add x18, x18, #4\n\t"
        "add x19, x19, #4\n\t"
        "add x5, x5, #64\n\t"
        "subs x6, x6, #1\n\t"
        "b.ne 3b\n\t"

        "st1w {z8.s}, p1, [x3]\n\t"
        "st1w {z9.s}, p1, [x3, #1, mul vl]\n\t"
        "st1w {z10.s}, p1, [x3, #2, mul vl]\n\t"
        "st1w {z11.s}, p1, [x3, #3, mul vl]\n\t"

        "add x2, x2, #4096\n\t"
        "add x3, x3, #64\n\t"
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x2", "x3", "x4", "x5", "x6", "x15", "x17", "x18", "x19",
          "z0", "z4", "z8", "z9", "z10", "z11", "p0", "p1"
    );
}

__attribute__((noinline))
void phase2_only(const int32_t* S, const int8_t* V, int32_t* O) {
    // S[4, 64] @ V[64, 256] -> O[4, 256]
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        "mov x15, #4\n\t"           // 4 D-tiles
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"

        "2:\n\t"
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"
        "eor z20.d, z20.d, z20.d\n\t" "eor z21.d, z21.d, z21.d\n\t"
        "eor z22.d, z22.d, z22.d\n\t" "eor z23.d, z23.d, z23.d\n\t"

        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "mov x4, x16\n\t"
        "mov x10, #16\n\t"          // 64 N / 4 = 16 iterations

        "3:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"

        "add x4, x4, #256\n\t"      // V stride = D = 256
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"

        "st1w {z8.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z20.s}, p1, [x17]\n\t"
        "sub x17, x17, #2048\n\t"
        "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Fused Online Softmax Kernel Benchmark\n");
    printf("==============================================\n\n");

    int M = 4;   // Rows (MR)
    int N = 64;  // Sequence length
    int D = 256; // Head dimension

    // Allocate buffers
    int8_t* Q = aligned_alloc(256, M * D);
    int8_t* K = aligned_alloc(256, N * D);
    int8_t* V = aligned_alloc(256, N * D);
    int32_t* S = aligned_alloc(256, M * N * 4);
    int32_t* O1 = aligned_alloc(256, M * D * 4);
    int32_t* O2 = aligned_alloc(256, M * D * 4);
    int32_t* S_buf = aligned_alloc(256, M * 16 * 4);  // Scratch for fused

    // Initialize with test data
    for (int i = 0; i < M * D; i++) Q[i] = (i % 7) - 3;
    for (int i = 0; i < N * D; i++) K[i] = (i % 5) - 2;
    for (int i = 0; i < N * D; i++) V[i] = (i % 9) - 4;

    int iters = 10000;
    double peak_sdot_per_tick = 40.0;

    // SDOT counts for M=4:
    // Phase 1: M × N × D / 64 = 4 × 64 × 256 / 64 = 1024 SDOT
    // Phase 2: M × D × N / 64 = 4 × 256 × 64 / 64 = 1024 SDOT
    // Total: 2048 SDOT
    int total_sdot = 2048;

    printf("Configuration: M=%d, N=%d, D=%d\n", M, N, D);
    printf("SDOT operations: Phase1=%d, Phase2=%d, Total=%d\n\n",
           total_sdot/2, total_sdot/2, total_sdot);

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase1_only(Q, K, S);
        phase2_only(S, V, O1);
        fused_gemm_chunked(Q, K, V, O2, S_buf);
    }

    // Benchmark separate Phase1 + Phase2
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase1_only(Q, K, S);
        phase2_only(S, V, O1);
    }
    uint64_t end = rdcycle();
    double ticks_sep = (double)(end - start) / iters;
    double eff_sep = (total_sdot / ticks_sep) / peak_sdot_per_tick * 100;

    // Benchmark fused
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_gemm_chunked(Q, K, V, O2, S_buf);
    }
    end = rdcycle();
    double ticks_fused = (double)(end - start) / iters;
    double eff_fused = (total_sdot / ticks_fused) / peak_sdot_per_tick * 100;

    printf("%-30s %10s %10s %12s\n", "Approach", "Ticks", "Efficiency", "SDOT/tick");
    printf("------------------------------------------------------------------\n");
    printf("%-30s %10.1f %9.1f%% %12.2f\n", "Separate (Phase1 + Phase2)",
           ticks_sep, eff_sep, total_sdot/ticks_sep);
    printf("%-30s %10.1f %9.1f%% %12.2f\n", "Fused (N-chunked)",
           ticks_fused, eff_fused, total_sdot/ticks_fused);

    printf("\n");
    double speedup = (ticks_sep - ticks_fused) / ticks_sep * 100;
    if (speedup > 0) {
        printf("Fused kernel is %.1f%% faster!\n", speedup);
    } else {
        printf("Separate kernels are %.1f%% faster.\n", -speedup);
    }

    printf("\n");
    printf("Note: Fused kernel eliminates:\n");
    printf("  - Full S matrix store/load between phases\n");
    printf("  - D-tile loop overhead in Phase 2\n");
    printf("  - But adds N-chunk loop overhead\n");

    free(Q); free(K); free(V); free(S); free(O1); free(O2); free(S_buf);
    return 0;
}
