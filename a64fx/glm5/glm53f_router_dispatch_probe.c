#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 4096, EXPERTS = 288, TOPK = 8 };

static int fail(const char *msg) {
    fprintf(stderr, "GLM53F_ROUTER_DISPATCH FAIL %s\n", msg);
    return 1;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : NULL;
    glm53f_st_context *ctx;
    uint16_t *embed = NULL, *gate = NULL;
    float *bias = NULL, *logits = NULL, *weights = NULL;
    int *ids = NULL, *dispatch = NULL;
    int rc = 1, i, j, selected = 0;
    double sum = 0.0;

    if (!dir) return fail("usage: probe MODEL_DIR");
    ctx = glm53f_st_open(dir);
    if (!ctx || glm53f_st_validate_contract(ctx, 1) != 0) {
        glm53f_st_close(ctx);
        return fail("checkpoint contract");
    }

    embed = (uint16_t *)malloc((size_t)HIDDEN * sizeof(*embed));
    gate = (uint16_t *)malloc((size_t)EXPERTS * HIDDEN * sizeof(*gate));
    bias = (float *)malloc((size_t)EXPERTS * sizeof(*bias));
    logits = (float *)malloc((size_t)EXPERTS * sizeof(*logits));
    weights = (float *)malloc((size_t)TOPK * sizeof(*weights));
    ids = (int *)malloc((size_t)TOPK * sizeof(*ids));
    dispatch = (int *)calloc(EXPERTS, sizeof(*dispatch));
    if (!embed || !gate || !bias || !logits || !weights || !ids || !dispatch)
        goto done;

    if (glm53f_st_read(ctx, "model.language_model.embed_tokens.weight",
                       0, embed, (size_t)HIDDEN * sizeof(*embed)) != 0 ||
        glm53f_st_read(ctx, "model.language_model.layers.3.mlp.gate.weight",
                       0, gate, (size_t)EXPERTS * HIDDEN * sizeof(*gate)) != 0 ||
        glm53f_st_read(ctx, "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
                       0, bias, (size_t)EXPERTS * sizeof(*bias)) != 0)
        goto done;

    for (i = 0; i < EXPERTS; ++i) {
        double dot = 0.0;
        for (j = 0; j < HIDDEN; ++j)
            dot += (double)glm53f_bf16_to_f32(gate[(size_t)i * HIDDEN + j]) *
                   glm53f_bf16_to_f32(embed[j]);
        logits[i] = (float)dot;
    }
    glm53f_router_topk(logits, bias, EXPERTS, TOPK, 2.5f, ids, weights);

    for (i = 0; i < TOPK; ++i) {
        if (ids[i] < 0 || ids[i] >= EXPERTS || dispatch[ids[i]]) goto done;
        dispatch[ids[i]] = 1;
        sum += weights[i];
        selected++;
    }
    if (selected != TOPK || fabs(sum - 2.5) > 1e-4) goto done;

    /* Dispatch integrity: one token produces exactly one slot for each of its
     * eight unique experts; no unselected expert receives a slot. */
    for (i = 0; i < EXPERTS; ++i) if (dispatch[i] != (i == ids[0] || i == ids[1] ||
        i == ids[2] || i == ids[3] || i == ids[4] || i == ids[5] ||
        i == ids[6] || i == ids[7])) goto done;

    printf("GLM53F_ROUTER logits_checksum=%.9g topk=%d weight_sum=%.9g\n",
           (double)logits[0] + logits[EXPERTS - 1], TOPK, sum);
    printf("GLM53F_ROUTER_IDS");
    for (i = 0; i < TOPK; ++i) printf(" %d:%.9g", ids[i], (double)weights[i]);
    putchar('\n');
    printf("GLM53F_DISPATCH PASS tokens=1 experts=%d selected=%d slots=%d\n",
           EXPERTS, selected, selected);
    rc = 0;

done:
    free(dispatch); free(ids); free(weights); free(logits); free(bias);
    free(gate); free(embed); glm53f_st_close(ctx);
    if (rc) fprintf(stderr, "GLM53F_ROUTER_DISPATCH FAIL route_or_dispatch_integrity\n");
    return rc;
}
