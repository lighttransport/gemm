#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../common/glm53f_ref.h"

int main(void) {
    float s[6] = {0}, q[2] = {3, 4}, k[2] = {0, 1}, v[3] = {2, 4, 6}, out[3];
    float s2[6] = {0}, out2[3], work[3];
    float logits[6] = {-3, 2, 0, 4, -1, 1}, bias[6] = {0};
    float w[2], comb[4] = {1, 2, 3, 4};
    int ids[2];
    glm53f_l2norm(q, 2, 1e-6f); glm53f_l2norm(k, 2, 1e-6f);
    glm53f_kda_step(s, q, k, v, 0.0f, 1.0f, 2, 3, out);
    glm53f_kda_step_streamed(s2, q, k, v, 0.0f, 1.0f, 2, 3, out2, work);
    if (fabsf(out[0] - 1.1313708f) > 2e-5f || fabsf(out[2] - 3.3941125f) > 2e-5f) return 1;
    if (fabsf(out2[0] - out[0]) > 2e-6f || fabsf(out2[2] - out[2]) > 2e-6f) return 5;
    glm53f_router_topk(logits, bias, 6, 2, 2.5f, ids, w);
    if (ids[0] != 3 || ids[1] != 1 || fabsf(w[0] + w[1] - 2.5f) > 1e-6f) return 2;
    glm53f_mhc_sinkhorn(comb, 2, 20, 1e-6f);
    if (fabsf(comb[0] + comb[1] - 1.0f) > 1e-4f || fabsf(comb[0] + comb[2] - 1.0f) > 1e-4f) return 3;
    if (glm53f_cp_owner(25, 12) != 1 || glm53f_cp_slot(25, 12) != 2 || glm53f_cp_slots(26, 12) != 3) return 4;
    printf("GLM53F_REF kda=ok router=ok mhc=ok cp=ok\n");
    return 0;
}
