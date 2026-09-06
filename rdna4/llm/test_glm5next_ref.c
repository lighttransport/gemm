#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/glm5next_ref.h"

static int close_vec(const float *a, const float *b, int n, float tol) {
    for (int i = 0; i < n; ++i)
        if (!isfinite(a[i]) || !isfinite(b[i]) || fabsf(a[i] - b[i]) > tol)
            return 0;
    return 1;
}

int main(void) {
    glm5next_config c = {0};
    glm5next_decode_state state;
    const int kd = 8, vd = 6;
    float a[kd * vd], b[kd * vd], q[kd], k[kd], v[vd], decay[kd];
    float out_a[vd], out_b[vd], work[vd];
    float combine[16] = { 0.2f, -0.4f, 0.1f, 0.8f,
                          -0.1f, 0.3f, 0.7f, -0.2f,
                          0.5f, 0.1f, -0.6f, 0.4f,
                          0.0f, -0.3f, 0.2f, 0.9f };
    float row_sum, col_sum;
    int i;

    c.n_layers = 45; c.hidden_size = 4096; c.kv_lora_rank = 512;
    c.linear_head_dim = 128; c.short_conv_kernel = 4;
    c.indexer_key_length = 128; c.hc_count = 4;
    if (glm5next_decode_state_alloc(&c, 16, &state) != 0) return 1;
    if (glm5next_decode_state_bytes(&c, 16) == 0) return 1;
    memset(a, 0, sizeof(a));
    for (i = 0; i < kd; ++i) {
        q[i] = 0.05f * (float)(i + 1);
        k[i] = -0.03f * (float)(i + 2);
        decay[i] = -0.01f * (float)(i + 1);
    }
    for (i = 0; i < vd; ++i) v[i] = 0.1f * (float)(i - 2);
    memcpy(b, a, sizeof(a));
    glm53f_kda_step_vec(a, q, k, v, decay, 0.7f, kd, vd, out_a);
    glm5next_kda_step(b, q, k, v, decay, 0.7f, kd, vd, out_b, work);
    if (!close_vec(out_a, out_b, vd, 2e-6f) || !close_vec(a, b, kd * vd, 2e-6f)) return 1;
    glm5next_mhc_sinkhorn(combine, 4, 20, 1e-6f);
    for (i = 0; i < 4; ++i) {
        row_sum = col_sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            row_sum += combine[i * 4 + j];
            col_sum += combine[j * 4 + i];
        }
        if (!isfinite(row_sum) || !isfinite(col_sum) || row_sum < 0.9f || col_sum < 0.9f) return 1;
    }
    glm5next_decode_state_reset(&c, &state);
    if (state.selected_count[0] != 0 || state.kda_recurrent[0] != 0.0f) return 1;
    glm5next_decode_state_free(&state);
    puts("GLM5NEXT_REF PASS");
    return 0;
}
