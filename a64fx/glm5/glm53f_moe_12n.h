#ifndef GLM53F_MOE_12N_H
#define GLM53F_MOE_12N_H

#include <mpi.h>
#include <stddef.h>
#include <string.h>
#include "glm53f_expert_kern.h"
#include "glm53f_iq_bridge.h"

#ifndef GLM53F_MOE_FUSED_WEIGHTED
#define GLM53F_MOE_FUSED_WEIGHTED 1
#endif

enum {
    GLM53F_MOE_HIDDEN = 4096,
    GLM53F_MOE_MAX_PARTS = 9,
    GLM53F_MOE_GATE_UP_STRIDE = 1024,
    GLM53F_MOE_INTER_STRIDE = 512
};

typedef struct {
    float up[GLM53F_MOE_MAX_PARTS * GLM53F_MOE_GATE_UP_STRIDE];
    float activation[GLM53F_MOE_MAX_PARTS * GLM53F_MOE_INTER_STRIDE];
    float part_output[GLM53F_MOE_MAX_PARTS * GLM53F_MOE_HIDDEN];
    float local_output[GLM53F_MOE_HIDDEN];
} glm53f_moe_scratch_12n;

typedef struct {
    const glm53f_expert_part *part;
    const float *part_weight;
    int count;
    glm53f_moe_scratch_12n *scratch;
    MPI_Comm comm;
} glm53f_moe_callback_12n;

/* `part_weight` is indexed exactly like `part`, not like the global top-k.
 * This matters for expert_parts < 12, where a rank skips unowned routes. */
static inline void glm53f_moe_local_12n(
        float *output, const glm53f_expert_part *part,
        const float *part_weight, int count, const float *x,
        glm53f_moe_scratch_12n *scratch) {
    if (count <= 0) {
        memset(output, 0, GLM53F_MOE_HIDDEN * sizeof(*output));
        return;
    }
#if GLM53F_MOE_FUSED_WEIGHTED
    int niq = 0, nfp = 0;
    glm53f_iq_part iq[GLM53F_MOE_MAX_PARTS];
    glm53f_expert_part fp[GLM53F_MOE_MAX_PARTS];
    float iq_weight[GLM53F_MOE_MAX_PARTS], fp_weight[GLM53F_MOE_MAX_PARTS];
    for (int k = 0; k < count; ++k) {
        if (part[k].gate_type || part[k].down_type) {
            iq[niq] = (glm53f_iq_part){ part[k].gate_up, part[k].down,
                part[k].gate_type, part[k].down_type, part[k].inter };
            iq_weight[niq++] = part_weight[k];
        } else {
            fp[nfp] = part[k]; fp_weight[nfp++] = part_weight[k];
        }
    }
    if (!niq) {
        glm53f_expert_batch_weighted_bits(fp, fp_weight, nfp, x,
                                           scratch->up, scratch->activation, output);
    } else {
        float *iq_output = scratch->local_output;
        float *fp_output = scratch->part_output;
        if (glm53f_iq_expert_weighted(iq_output, iq, iq_weight, niq, x,
                                      scratch->up, scratch->activation)) {
            memset(iq_output, 0, GLM53F_MOE_HIDDEN * sizeof(*iq_output));
        }
        if (nfp)
            glm53f_expert_batch_weighted_bits(fp, fp_weight, nfp, x,
                scratch->up, scratch->activation, fp_output);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < GLM53F_MOE_HIDDEN; ++i)
            output[i] = iq_output[i] + (nfp ? fp_output[i] : 0.0f);
    }
#else
    glm53f_expert_batch_bits(part, count, x, scratch->up,
                             scratch->activation, scratch->part_output);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < GLM53F_MOE_HIDDEN; ++i) {
        float value = 0.0f;
        for (int k = 0; k < count; ++k)
            value += part_weight[k] *
                     scratch->part_output[(size_t)k * GLM53F_MOE_HIDDEN + i];
        output[i] = value;
    }
#endif
}

static inline int glm53f_moe_forward_12n(
        float *output, const glm53f_expert_part *part,
        const float *part_weight, int count, const float *x,
        glm53f_moe_scratch_12n *scratch, MPI_Comm comm) {
    if (count < 0 || count > GLM53F_MOE_MAX_PARTS) return -1;
    glm53f_moe_local_12n(scratch->local_output, part, part_weight,
                         count, x, scratch);
    return MPI_Allreduce(scratch->local_output, output, GLM53F_MOE_HIDDEN,
                         MPI_FLOAT, MPI_SUM, comm) == MPI_SUCCESS ? 0 : -1;
}

static inline int glm53f_moe_sublayer_12n(
        void *context, float *output, const float *normalized_input) {
    glm53f_moe_callback_12n *moe = (glm53f_moe_callback_12n *)context;
    return glm53f_moe_forward_12n(output, moe->part, moe->part_weight,
                                  moe->count, normalized_input, moe->scratch,
                                  moe->comm);
}

#endif
