#include <stdlib.h>
#include "../../common/ggml_dequant.h"
#include "kv_f16_sve.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    int exact = 0, nan_count = 0;
    for (unsigned bits = 0; bits < 65536; bits++) {
        float got = tf_kv_f16_to_f32_sve((uint16_t)bits);
        float ref = ggml_fp16_to_fp32((uint16_t)bits);
        if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x3ffu)) {
            if (!isnan(got)) return 1;
            nan_count++;
        } else {
            if (memcmp(&got, &ref, sizeof(got))) {
                fprintf(stderr, "FP16 widening mismatch bits=%04x got=%a ref=%a\n",
                        bits, got, ref);
                return 1;
            }
            exact++;
        }
    }
    printf("KV FP16 widening PASS: %d finite/infinite exact, %d NaNs\n",
           exact, nan_count);
    const int widths[] = {1, 15, 16, 17, 128, 256, 257};
    uint32_t rng = 923;
    for (size_t shape = 0; shape < sizeof(widths) / sizeof(widths[0]); shape++) {
        int n = widths[shape];
        float got[257], ref[257];
        uint16_t values[257];
        for (int d = 0; d < n; d++) got[d] = ref[d] = (float)(d - 123) / 97.0f;
        for (int pos = 0; pos < 129; pos++) {
            float scale = (float)(pos % 17 - 8) / 19.0f;
            for (int d = 0; d < n; d++) {
                rng = rng * 1664525u + 1013904223u;
                values[d] = (uint16_t)((rng % 0x7c00u) | ((rng >> 16) & 0x8000u));
                ref[d] = fmaf(scale, ggml_fp16_to_fp32(values[d]), ref[d]);
            }
            tf_kv_f16_axpy_sve(got, values, scale, n);
            if (memcmp(got, ref, (size_t)n * sizeof(float))) {
                fprintf(stderr, "KV PV mismatch width=%d position=%d\n", n, pos);
                return 1;
            }
        }
    }
    puts("KV FP16 PV PASS: exact sequence-order FMA, full vectors and tails");
    for (size_t shape = 0; shape < sizeof(widths) / sizeof(widths[0]); shape++) {
        int n = widths[shape], stride = n + 17;
        uint16_t keys[4 * (257 + 17)];
        float query[257], got[4], ref[4] = {0};
        for (int d = 0; d < n; d++) {
            rng = rng * 1664525u + 1013904223u;
            query[d] = (float)((int)(rng >> 24) - 128) / 97.0f;
            for (int p = 0; p < 4; p++) {
                rng = rng * 1664525u + 1013904223u;
                keys[p * stride + d] =
                    (uint16_t)((rng % 0x7c00u) | ((rng >> 16) & 0x8000u));
                ref[p] = fmaf(query[d], ggml_fp16_to_fp32(keys[p * stride + d]), ref[p]);
            }
        }
        tf_kv_f16_dot4_sve(got, query, keys, stride, n);
        if (memcmp(got, ref, sizeof(got))) {
            fprintf(stderr, "KV QK mismatch width=%d\n", n);
            return 1;
        }
    }
    puts("KV FP16 QK PASS: four scores, exact ascending-dimension FMA");
    return 0;
}
