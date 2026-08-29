#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    {
        const char *name = "model.language_model.layers.44.self_attn.f_b_proj.weight";
        const size_t rows = 8192, columns = 128, begin = 17, count = 32;
        uint16_t *full = malloc(rows * columns * sizeof(*full));
        uint16_t *packed = malloc(rows * count * sizeof(*packed));
        int ok = full && packed &&
            !glm53f_st_read(ctx, name, 0, full, rows * columns * sizeof(*full)) &&
            !glm53f_st_read_columns(ctx, name, columns * sizeof(*full),
                                    begin * sizeof(*full), count * sizeof(*full), packed);
        for (size_t r = 0; ok && r < rows; ++r)
            ok = !memcmp(full + r * columns + begin, packed + r * count,
                         count * sizeof(*full));
        printf("GLM53F_ST_COLUMNS rows=%zu begin=%zu count=%zu %s\n",
               rows, begin, count, ok ? "PASS" : "FAIL");
        free(packed);
        free(full);
        glm53f_st_close(ctx);
        return ok ? 0 : 1;
    }
}
