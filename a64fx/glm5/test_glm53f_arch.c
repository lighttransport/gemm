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
    printf("layers=%d hidden=%d sparse=%d moe=%d mtp=%d\n",
           a.n_layers, a.hidden_size, sparse, glm53f_is_moe(45), glm53f_is_mtp(45));
    return 0;
}
