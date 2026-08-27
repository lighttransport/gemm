#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"

#include <stdio.h>

int main(int argc, char **argv) {
    glm53f_st_context *ctx;
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    ctx = glm53f_st_open(argv[1]);
    if (!ctx || glm53f_st_validate_contract(ctx, 1) != 0) {
        glm53f_st_close(ctx);
        return 1;
    }
    glm53f_st_close(ctx);
    return 0;
}
