#include <stdio.h>
#include "../../common/glm53f_arch.h"

int main(void) {
    glm53f_arch a = glm53f_arch_default();
    int sparse = 0;
    for (size_t layer = 0; layer < (size_t)a.n_layers; ++layer)
        sparse += glm53f_layer_type(layer) == GLM53F_SPARSE_ATTENTION;
    if (a.n_layers != 45 || a.hidden_size != 4096 || a.n_routed_experts != 288 ||
        sparse != 11 || !glm53f_is_moe(44) || !glm53f_is_moe(45) ||
        !glm53f_is_mtp(45) || glm53f_is_mtp(44))
        return 1;
    for (int e = 0; e < 288; ++e) {
        int seen[12] = {0};
        for (int p = 0; p < 4; ++p) {
            int r = glm53f_expert_part_owner(e, p, 4, 12), b, n;
            if (r < 0 || seen[r]++) return 2;
            glm53f_balanced_slice(2048, p, 4, &b, &n);
            if (b != p * 512 || n != 512) return 3;
        }
    }
    for (int e = 0; e < 288; ++e) {
        int seen[12] = {0}, total = 0, wide = 0;
        for (int p = 0; p < 12; ++p) {
            int r = glm53f_expert_part_owner(e, p, 12, 12), b, n;
            if (r < 0 || seen[r]++ ||
                glm53f_block_aligned_slice(2048, 128, p, 12, &b, &n) ||
                b % 128 || (n != 128 && n != 256)) return 4;
            total += n;
            wide += n == 256;
        }
        if (total != 2048 || wide != 4) return 5;
    }
    printf("layers=%d hidden=%d sparse=%d moe=%d mtp=%d\n",
           a.n_layers, a.hidden_size, sparse, glm53f_is_moe(45), glm53f_is_mtp(45));
    return 0;
}
