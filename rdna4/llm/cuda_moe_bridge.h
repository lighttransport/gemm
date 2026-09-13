#ifndef GLM5NEXT_CUDA_MOE_BRIDGE_H
#define GLM5NEXT_CUDA_MOE_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct glm5next_cuda_moe_bridge glm5next_cuda_moe_bridge;

/* CUDA-side IQ1_S routed + shared expert FFN.  Each pointer names one raw
 * [rows, cols] IQ1_S matrix in GGUF row-major layout. */
glm5next_cuda_moe_bridge *glm5next_cuda_moe_init(int device, int verbose);
void glm5next_cuda_moe_free(glm5next_cuda_moe_bridge *bridge);
int glm5next_cuda_moe_compute(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, const void *shared_gate, const void *shared_up,
        const void *shared_down, int shared_ff, float *out);

/* Routed-only path for the GLM5 IQ1_S model variant used here: IQ2_XXS
 * gate/up and IQ3_XXS down.  The caller adds the shared expert on its native
 * backend. */
int glm5next_cuda_moe_compute_iq2_iq3(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out);

int glm5next_cuda_moe_compute_iq2xs_iq3(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out);

/* Routed-only IQ1_S path.  Matrices are retained in the CUDA LRU cache. */
int glm5next_cuda_moe_compute_iq1(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float clamp, float *out);
int glm5next_cuda_moe_compute_iq1_begin(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float clamp);
int glm5next_cuda_moe_compute_iq1_finish(glm5next_cuda_moe_bridge *bridge,
        int hidden_size, float *out);

int glm5next_cuda_moe_compute_iq2_iq4nl(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out);

int glm5next_cuda_moe_compute_iq2_iq4xs(glm5next_cuda_moe_bridge *bridge,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out);

int glm5next_cuda_moe_compute_shared_q5q6(glm5next_cuda_moe_bridge *bridge,
        const void *gate, const void *up, const void *down,
        int expert_ff, int hidden_size, const float *hidden, float *out);

#ifdef __cplusplus
}
#endif

#endif
