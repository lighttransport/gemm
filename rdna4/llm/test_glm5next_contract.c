/* Validate and print the shared GLM5Next GGUF contract without requiring HIP. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/glm5next.h"

int main(int argc, char **argv) {
    gguf_shards *model;
    glm5next_config config;
    glm5next_state_layout layout;
    char error[160];
    int max_seq = 4096;
    int kda = 0, dsa = 0, i;
    uint64_t tensor_count = 0;
    glm5next_tensor_view view;

    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [max_seq_len]\n", argv[0]);
        return 2;
    }
    if (argc > 2) max_seq = atoi(argv[2]);
    model = gguf_open_shards(argv[1], 2);
    if (!model) {
        fprintf(stderr, "failed to open GGUF shards: %s\n", argv[1]);
        return 1;
    }
    memset(&config, 0, sizeof(config));
    if (glm5next_config_load(model->metadata, &config, error, sizeof(error)) != 0) {
        fprintf(stderr, "GLM5Next contract: FAIL: %s\n", error);
        gguf_close_shards(model);
        return 1;
    }
    if (glm5next_validate_tensors(model, &config, error, sizeof(error)) != 0) {
        fprintf(stderr, "GLM5Next tensors: FAIL: %s\n", error);
        glm5next_config_free(&config);
        gguf_close_shards(model);
        return 1;
    }
    if (glm5next_tensor_view_get(model, "blk.0.attn_q.weight", 1, &view) != 0 ||
        view.n_dims != 2 || view.dims[0] != 4096 || view.dims[1] != 8192 ||
        view.data == NULL) {
        fprintf(stderr, "GLM5Next tensor view: FAIL for blk.0.attn_q.weight\n");
        glm5next_config_free(&config);
        gguf_close_shards(model);
        return 1;
    }
    if (glm5next_tensor_view_get(model, "blk.3.attn_k_b.weight", 1, &view) != 0 ||
        view.n_dims != 3 || view.dims[0] != 256 || view.dims[1] != 512 ||
        view.dims[2] != 64 || view.data == NULL) {
        fprintf(stderr, "GLM5Next tensor view: FAIL for blk.3.attn_k_b.weight\n");
        glm5next_config_free(&config);
        gguf_close_shards(model);
        return 1;
    }
    for (i = 0; i < model->n_shards; ++i)
        tensor_count += model->shards[i]->n_tensors;
    if (glm5next_state_layout_compute(&config, max_seq, &layout) != 0) {
        fprintf(stderr, "GLM5Next state layout: FAIL\n");
        glm5next_config_free(&config);
        gguf_close_shards(model);
        return 1;
    }
    for (i = 0; i < config.n_layers; ++i) {
        if (glm5next_layer_type(&config, i) == GLM5NEXT_LAYER_KDA) ++kda;
        else ++dsa;
    }
    printf("GLM5NEXT_CONTRACT PASS\n");
    printf("  shards=%u tensors=%llu layers=%d nextn=%d hidden=%d vocab=%d context=%d\n",
           model->n_shards, (unsigned long long)tensor_count,
           config.n_layers, config.n_nextn_layers, config.hidden_size,
           config.vocab_size, config.context_length);
    printf("  schedule kda=%d dsa=%d heads=%d q_lora=%d kv_lora=%d nope=%d value=%d\n",
           kda, dsa, config.attention_heads, config.q_lora_rank,
           config.kv_lora_rank, config.qk_nope_head_dim, config.value_head_dim);
    printf("  moe experts=%d used=%d expert_ff=%d shared=%d/%d\n",
           config.expert_count, config.expert_used_count, config.expert_ff_length,
           config.shared_expert_count, config.shared_expert_ff_length);
    printf("  index heads=%d key=%d top_k=%d kpool=%d mhc=%d sinkhorn=%d\n",
           config.indexer_heads, config.indexer_key_length, config.indexer_top_k,
           config.indexer_kpool, config.hc_count, config.hc_sinkhorn_iterations);
    printf("  state conv=%.2f MiB recurrent=%.2f MiB latent_kv=%.2f MiB indexer=%.2f MiB\n",
           (double)layout.conv_bytes / (1024.0 * 1024.0),
           (double)layout.recurrent_bytes / (1024.0 * 1024.0),
           (double)layout.latent_kv_bytes / (1024.0 * 1024.0),
           (double)layout.indexer_bytes / (1024.0 * 1024.0));
    glm5next_config_free(&config);
    gguf_close_shards(model);
    return 0;
}
