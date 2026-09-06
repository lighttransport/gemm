/* A64FX bitwise-exact dimension-split PV accumulation.
 * fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -D_GNU_SOURCE \
 *   -DTF_LINK_PODD -ffunction-sections -fdata-sections common/test_attn_pv_slice.c \
 *   a64fx/gemma4-kernels/sgemm_bf16_2x12.S -Wl,--gc-sections -lm -lpthread \
 *   -o /local/u14346/test_attn_pv_slice
 */
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#if defined(__ARM_FEATURE_SVE)
/* Original sequence-major SVE loop from tf_attn_worker. */
static void reference(float *out, const float *a, const float *v,
                        int seq, int stride, int hd) {
    svbool_t pg = svptrue_b32();
    int vl = (int)svcntw();
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int p = 0; p < seq; p++) {
        const float *vp = v + (size_t)p * stride;
        if (p + 2 < seq) __builtin_prefetch(v + (size_t)(p + 2) * stride, 0, 1);
        svfloat32_t va = svdup_f32(a[p]);
        for (int d = 0; d < hd; d += vl)
            svst1(pg, out + d, svmla_x(pg, svld1(pg, out + d), va, svld1(pg, vp + d)));
    }
}
#endif

int main(void) {
#if defined(__ARM_FEATURE_SVE)
    const int seqs[] = {1, 127, 512, 2048, 8192};
    int cases = 0;
    for (int hd = 128; hd <= 256; hd *= 2) {
        for (int multiple = 1; multiple <= 2; multiple++) {
            int stride = multiple * hd;
            for (unsigned s = 0; s < sizeof(seqs) / sizeof(seqs[0]); s++) {
                int seq = seqs[s];
                float *v = malloc((size_t)seq * stride * sizeof(float));
                float *a = malloc((size_t)seq * sizeof(float));
                float out[256], ref[256];
                if (!v || !a) return 2;
                for (int p = 0; p < seq; p++) {
                    a[p] = (float)((p * 13 % 31) + 1) / (16.0f * seq);
                    for (int d = 0; d < stride; d++)
                        v[(size_t)p * stride + d] = (float)((p * 7 + d * 11) % 127 - 63) / 32;
                }
                reference(ref, a, v, seq, stride, hd);
                /* Uneven partitions and reversed worker order exercise tails
                 * and prove that no slice needs another slice's output. */
                for (int lanes = 1; lanes <= 17; lanes++) {
                    for (int d = 0; d < hd; d++) out[d] = -12345.0f;
                    for (int lane = lanes - 1; lane >= 0; lane--)
                        tf_attn_pv_slice_sve(out, a, v, seq, stride,
                            hd * lane / lanes, hd * (lane + 1) / lanes);
                    if (memcmp(out, ref, (size_t)hd * sizeof(float))) {
                        fprintf(stderr, "FAIL slices seq=%d hd=%d stride=%d lanes=%d\n",
                                seq, hd, stride, lanes);
                        return 1;
                    }
                    cases++;
                }
                free(a); free(v);
            }
        }
    }
    printf("PASS: %d bitwise-exact PV cases\n", cases);
    return 0;
#else
    return 77;
#endif
}
