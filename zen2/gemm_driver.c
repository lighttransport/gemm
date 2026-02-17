#include "gemm.h"
#include "gemm_pack.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>   /* madvise */

/*
 * Scalar microkernel for edge tiles where mr < MR or nr < NR.
 * C[mr×nr] = A_pack[K×MR] × B_pack[K×NR] (overwrite mode).
 */
static void gemm_kernel_edge(const float *A_pack, const float *B_pack,
                             float *C, int ldc,
                             int mr, int nr, int K)
{
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            C[i * ldc + j] = 0.0f;
        }
    }

    for (int k = 0; k < K; k++) {
        for (int i = 0; i < mr; i++) {
            float a_val = A_pack[k * MR + i];
            for (int j = 0; j < nr; j++) {
                C[i * ldc + j] += a_val * B_pack[k * NR + j];
            }
        }
    }
}

void gemm_fp32(const float *A, int lda,
               const float *B, int ldb,
               float *C, int ldc,
               int M, int N, int K)
{
    /* Adaptive MC: use ~75% of L2 (512 KB) for A panel.
     * A panel = Mc * K * 4 bytes.  Mc = 384*1024/(K*4), rounded to MR. */
    int Mc;
    {
        int mc_max = (384 * 1024) / (K * (int)sizeof(float));
        mc_max = (mc_max / MR) * MR;
        if (mc_max < MR) mc_max = MR;
        Mc = mc_max;
    }
    int Nc = NC_DEFAULT;

    if (Mc > M) Mc = ((M + MR - 1) / MR) * MR;
    if (Nc > N) Nc = ((N + NR - 1) / NR) * NR;

    float *A_pack = NULL;
    float *B_pack = NULL;
    size_t a_pack_sz = (size_t)Mc * K * sizeof(float);
    size_t b_pack_sz = (size_t)Nc * K * sizeof(float);
    posix_memalign((void **)&A_pack, 64, a_pack_sz);
    posix_memalign((void **)&B_pack, 64, b_pack_sz);

    /* Request transparent huge pages for packed buffers */
    madvise(A_pack, a_pack_sz, MADV_HUGEPAGE);
    madvise(B_pack, b_pack_sz, MADV_HUGEPAGE);

    int64_t ldc_bytes = (int64_t)ldc * sizeof(float);

    for (int jc = 0; jc < N; jc += Nc) {
        int nc = (jc + Nc <= N) ? Nc : (N - jc);

        /* Pack B panel */
        for (int j = 0; j < nc; j += NR) {
            int nr = (j + NR <= nc) ? NR : (nc - j);
            pack_b(B + (jc + j) * ldb, ldb,
                   B_pack + (size_t)j * K, nr, K);
        }

        for (int ic = 0; ic < M; ic += Mc) {
            int mc = (ic + Mc <= M) ? Mc : (M - ic);

            /* Pack A panel */
            for (int i = 0; i < mc; i += MR) {
                int mr = (i + MR <= mc) ? MR : (mc - i);
                pack_a(A + (ic + i) * lda, lda,
                       A_pack + (size_t)i * K, mr, K);
            }

            /* Macro-kernel */
            for (int jr = 0; jr < nc; jr += NR) {
                int nr = (jr + NR <= nc) ? NR : (nc - jr);
                const float *B_tile = B_pack + (size_t)jr * K;

                for (int ir = 0; ir < mc; ir += MR) {
                    int mr = (ir + MR <= mc) ? MR : (mc - ir);

                    float *C_tile = C + (ic + ir) * ldc + (jc + jr);
                    const float *A_tile = A_pack + (size_t)ir * K;

                    if (mr == MR && nr == NR) {
                        gemm_kernel_6x16(A_tile, B_tile,
                                         C_tile, (int64_t)K, ldc_bytes);
                    } else {
                        gemm_kernel_edge(A_tile, B_tile,
                                         C_tile, ldc, mr, nr, K);
                    }
                }
            }
        }
    }

    free(A_pack);
    free(B_pack);
}
