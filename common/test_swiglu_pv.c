/* Exercise the actual PV producer and activation with deliberately skewed
 * scheduling: one worker finishes both before the next starts. This catches
 * cross-worker reads which can be hidden by synchronous benchmark execution.
 * fcc -Nclang -O1 -march=armv8.2-a+sve -D_GNU_SOURCE -DTF_LINK_PODD \
 *   -ffunction-sections -fdata-sections common/test_swiglu_pv.c \
 *   a64fx/gemma4-kernels/sgemm_bf16_2x12.S -Wl,--gc-sections \
 *   -lm -lpthread -o /local/u14346/test_swiglu_pv
 */
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

int main(void) {
#if defined(__ARM_FEATURE_SVE)
    const int sizes[] = {8, 88, 256, 4352};
    const int teams[] = {1, 3, 12, 47, 48};
    const int k = 32;
    int cases = 0;
    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        uint16_t *w = malloc((size_t)n * k * sizeof(*w));
        float *gate = malloc((size_t)n * sizeof(*gate));
        float *up = malloc((size_t)n * sizeof(*up));
        float *out = malloc((size_t)n * sizeof(*out));
        float x[32];
        if (!w || !gate || !up || !out) return 2;
        /* All weights 1 and activations 1/32: exact dot product is 1. */
        for (int i = 0; i < n * k; i++) w[i] = 0x3f80;
        for (int i = 0; i < k; i++) x[i] = 1.0f / 32;
        qtensor mat = {.data=w, .type=GGML_TYPE_BF16, .n_rows=n,
                       .n_cols=k, .bf16_pv=1};
        for (unsigned t = 0; t < sizeof(teams) / sizeof(teams[0]); t++) {
            for (int i = 0; i < n; i++) gate[i] = up[i] = out[i] = -12345.0f;
            for (int tid = teams[t] - 1; tid >= 0; tid--) {
                tf_thread_matvec(gate, &mat, x, n, tid, teams[t]);
                tf_thread_matvec(up, &mat, x, n, tid, teams[t]);
                tf_swiglu_pv_worker(out, gate, up, n, tid, teams[t]);
            }
            float ref = 1.0f / (1.0f + expf(-1.0f));
            for (int i = 0; i < n; i++) {
                if (fabsf(out[i] - ref) > 1e-6f) {
                    fprintf(stderr, "FAIL rows=%d threads=%d row=%d got=%g ref=%g\n",
                            n, teams[t], i, out[i], ref);
                    return 1;
                }
            }
            cases++;
        }
        free(w); free(gate); free(up); free(out);
    }
    printf("PASS: %d PV producer/activation scheduling cases\n", cases);
    return 0;
#else
    fprintf(stderr, "SKIP: requires SVE\n");
    return 77;
#endif
}
