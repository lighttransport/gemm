/* Compact A15 NVFP4 arithmetic and row-lane oracle, runnable under SVE QEMU. */
#include <omp.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
#if !defined(__ARM_FEATURE_SVE)
    return 77;
#else
    if (svcntb() != 64) return 77;
    tf_nvfp4_tiled_block w;
    memset(&w, 0, sizeof(w));
    const uint8_t scales[8] = {0, 1, 2, 3, 10, 11, 15, 32};
    for (int s = 0; s < 4; s++) {
        tf_nvfp4_tiled_subblock *p = &w.s[s];
        for (int row = 0; row < 8; row++) {
            p->d[row] = scales[row];
            for (int j = 0; j < 8; j++) {
                int lo = (row * 3 + j + s) & 15;
                int hi = (row + j * 5 + s) & 15;
                p->qs[row * 8 + j] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
    qtensor mat = {0};
    mat.type = GGML_TYPE_NVFP4;
    mat.n_rows = 8;
    mat.n_cols = 64;
    mat.nvfp4_tiled = 1;
    mat.data = (uint8_t *)&w;
    float x[64], exact[8], approximate[8];
    int checked = 0;
    for (int trial = 0; trial < 68; trial++) {
        for (int k = 0; k < 64; k++) {
            x[k] = trial < 64 ? (k == trial ? (k & 1 ? -3.5f : 3.5f) : 0.0f) :
                trial == 64 ? 0.0f :
                trial == 65 ? (float)((k * 17) % 43 - 21) * 0.0625f :
                trial == 66 ? (float)((k & 1) ? -1 : 1) * 10000.0f :
                              (float)((k * 7) % 13 - 6) * 0.0001f;
        }
        tf_nvfp4_tiled_matvec_rows(exact, &mat, x, 0, 8);
        tf_nvfp4_a15_matvec_rows(approximate, &mat, x, 0, 8);
        for (int row = 0; row < 8; row++) {
            double abs_sum = 0.0;
            for (int s = 0; s < 4; s++) {
                const tf_nvfp4_tiled_subblock *p = &w.s[s];
                float scale = tf_nvfp4_scale_fast(p->d[row]);
                for (int j = 0; j < 16; j++) {
                    int packed = p->qs[row * 8 + (j & 7)];
                    int code = j < 8 ? packed & 15 : packed >> 4;
                    abs_sum += fabs((double)ds4f_kvalues_mxfp4_f32[code] *
                                    scale * x[s * 16 + j]);
                }
            }
            double error = fabs((double)approximate[row] - exact[row]);
            double limit = 0.001 * abs_sum + 1e-5;
            if (!isfinite(approximate[row]) || error > limit) {
                fprintf(stderr, "FAIL trial=%d row=%d exact=%g got=%g error=%g limit=%g\n",
                        trial, row, exact[row], approximate[row], error, limit);
                return 1;
            }
            checked++;
        }
    }
    tf_q6_exact_block head[16] = {{0}};
    for (int row = 0; row < 16; row++) {
        for (int k = 0; k < 256; k++) head[row].q[k] = 1;
        for (int s = 0; s < 16; s++) head[row].scale[s] = 1.0f;
    }
    qtensor output = {0};
    output.type = GGML_TYPE_Q6_K;
    output.n_rows = 16;
    output.n_cols = 256;
    output.q6_decoded = 1;
    output.data = (uint8_t *)head;
    float hx[256], logits[16], scratch[256];
    float *thread_tmp[1] = {scratch};
    transformer_model model = {0};
    model.n_threads = 1;
    model.thread_tmp = thread_tmp;
    for (int k = 0; k < 256; k++) hx[k] = 1.0f;
    for (int row = 0; row < 16; row++) logits[row] = -123.0f;
    tf_qmatvec_pool(&model, logits, &output, hx, 8);
    for (int row = 0; row < 16; row++)
        if (logits[row] != (row < 8 ? 256.0f : -123.0f)) {
            fprintf(stderr, "FAIL partial Q6 head row=%d got=%g\n", row, logits[row]);
            return 1;
        }
    printf("PASS compact_a15 outputs=%d sve_bytes=%zu\n", checked,
           (size_t)svcntb());
    return 0;
#endif
}
