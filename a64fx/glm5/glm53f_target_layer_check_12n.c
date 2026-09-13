/* One complete real-weight GLM-5.3F target layer: mHC + KDA + MoE. */
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/glm53f_safetensors.h"
#include "glm53f_kda_12n.h"
#include "glm53f_moe_stage_12n.h"
#include "glm53f_sparse_12n.h"
#include "glm53f_target_layer_12n.h"

enum { HIDDEN = 4096, STREAMS = 4, FLAT = HIDDEN * STREAMS, MIX = 24 };

static void *aligned_alloc_256(size_t bytes) {
    void *p = NULL;
    return posix_memalign(&p, 256, bytes) ? NULL : p;
}

static void *load_exact(glm53f_st_context *st, const char *name, size_t bytes) {
    const st_tensor_info *tensor = glm53f_st_find(st, name, NULL);
    void *p = aligned_alloc_256(bytes);
    if (!tensor || tensor->nbytes != bytes || !p ||
        glm53f_st_read(st, name, 0, p, bytes)) {
        fprintf(stderr, "target layer load failed: %s\n", name);
        free(p);
        return NULL;
    }
    return p;
}

static int load_layer_weights(glm53f_st_context *st, int layer,
                              glm53f_target_layer_weights_12n *w) {
    char name[256];
#define LOAD_SITE(FIELD, WHICH) do { \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_fn",layer); \
    w->FIELD.fn=load_exact(st,name,(size_t)MIX*FLAT*sizeof(uint16_t)); \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_base",layer); \
    w->FIELD.base=load_exact(st,name,MIX*sizeof(float)); \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_scale",layer); \
    w->FIELD.scale=load_exact(st,name,3*sizeof(float)); \
} while(0)
    LOAD_SITE(attention_mhc, "attn");
    LOAD_SITE(ffn_mhc, "ffn");
#undef LOAD_SITE
    snprintf(name,sizeof(name),"model.language_model.layers.%d.input_layernorm.weight",layer);
    w->input_norm=load_exact(st,name,HIDDEN*sizeof(uint16_t));
    snprintf(name,sizeof(name),"model.language_model.layers.%d.post_attention_layernorm.weight",layer);
    w->post_attention_norm=load_exact(st,name,HIDDEN*sizeof(uint16_t));
    return w->attention_mhc.fn && w->attention_mhc.base && w->attention_mhc.scale &&
           w->ffn_mhc.fn && w->ffn_mhc.base && w->ffn_mhc.scale &&
           w->input_norm && w->post_attention_norm ? 0 : -1;
}

int main(int argc, char **argv) {
    int rank, size, layer = argc > 4 ? atoi(argv[4]) : 44, local_ok, ok;
    glm53f_st_context *st;
    glm53f_target_layer_weights_12n weights;
    glm53f_target_layer_scratch_12n *scratch;
    glm53f_kda_context_12n *kda;
    glm53f_sparse_context_12n *sparse;
    glm53f_moe_stage_context_12n *moe;
    float *initial, *streams[2];
    double elapsed[2], max_elapsed;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    memset(&weights, 0, sizeof(weights));
    if (argc < 4 || size != 12 || layer < 3 || layer >= 45)
        MPI_Abort(MPI_COMM_WORLD, 2);
    st = glm53f_st_open(argv[1]);
    if (!st || load_layer_weights(st, layer, &weights)) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_st_close(st);
    kda = NULL;
    sparse = NULL;
    if (layer % 4 == 3) sparse = glm53f_sparse_create_12n(argv[1], layer, 8);
    else kda = glm53f_kda_create_12n(argv[1], layer);
    moe = glm53f_moe_stage_create_12n(argv[2], argv[3], argv[1], 3, 42);
    if ((!kda && !sparse) || !moe) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_moe_stage_set_layer_12n(moe, layer);
    scratch = aligned_alloc_256(sizeof(*scratch));
    initial = aligned_alloc_256((size_t)FLAT * sizeof(float));
    streams[0] = aligned_alloc_256((size_t)FLAT * sizeof(float));
    streams[1] = aligned_alloc_256((size_t)FLAT * sizeof(float));
    if (!scratch || !initial || !streams[0] || !streams[1]) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int i = 0; i < FLAT; ++i)
        initial[i] = (float)(((i * 17 + 3) % 251) - 125) / 125.0f;
    for (int pass = 0; pass < 2; ++pass) {
        memcpy(streams[pass], initial, (size_t)FLAT * sizeof(float));
        if (kda) glm53f_kda_reset_12n(kda);
        else glm53f_sparse_reset_12n(sparse);
        MPI_Barrier(MPI_COMM_WORLD);
        double begin = MPI_Wtime();
        local_ok = !glm53f_target_layer_forward_12n(
            streams[pass], &weights,
            kda ? glm53f_kda_sublayer_12n : glm53f_sparse_sublayer_12n,
            kda ? (void *)kda : (void *)sparse,
            glm53f_moe_stage_sublayer_12n, moe, scratch);
        elapsed[pass] = MPI_Wtime() - begin;
        if (!local_ok) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    local_ok = !memcmp(streams[0], streams[1], (size_t)FLAT * sizeof(float));
    double checksum = 0.0;
    for (int i = 0; i < FLAT; ++i) {
        local_ok &= isfinite(streams[0][i]);
        checksum += streams[0][i];
    }
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed[1], &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
        printf("GLM53F_TARGET_LAYER_12N layer=%d attention=%s real_router_moe=1 "
               "max_ms=%.3f repeat=%s checksum=%.9g %s\n", layer,
               kda ? "KDA" : "SPARSE", max_elapsed * 1e3,
               ok ? "BIT_EXACT" : "FAIL", checksum,
               ok ? "PASS" : "FAIL");
    glm53f_moe_stage_free_12n(moe);
    glm53f_sparse_free_12n(sparse);
    glm53f_kda_free_12n(kda);
    MPI_Finalize();
    return ok ? 0 : 1;
}
