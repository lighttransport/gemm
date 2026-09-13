/* Single-node Qwen3.8 Flash bring-up runner.
 *
 * Current mode is a safe checkpoint/partial-layer validator. It binds the
 * selected backbone layers and the PLE metadata, but does not execute the
 * numerical Qwen4-Exp graph yet.
 */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"
#define Q38FN_RUNTIME_IMPLEMENTATION
#include "../common/q38fn_runtime.h"
#define GLM5_BPE_IMPLEMENTATION
#include "../common/glm5_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int dump_vector(const char *name, const float *values, size_t count)
{
    const char *dir = getenv("Q38FN_DUMP_DIR");
    char path[4096];
    FILE *file;
    if (!dir || !*dir) return 0;
    if (snprintf(path, sizeof(path), "%s/serial-%s.f32", dir, name) >= (int)sizeof(path) ||
        !(file = fopen(path, "wb"))) return -1;
    int rc = fwrite(values, sizeof(*values), count, file) == count && !fclose(file) ? 0 : -1;
    return rc;
}

static int layer_of(const char *name) {
    static const char prefix[] = "model.language_model.layers.";
    const char *p;
    char *e; long n;
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) return -1;
    p = name + sizeof(prefix) - 1;
    n = strtol(p, &e, 10);
    return e == p || *e != '.' || n < 0 || n >= Q38FN_LAYERS ? -1 : (int)n;
}

static char *read_text_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    long size;
    char *text;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 ||
        size > 1024 * 1024 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    text = (char *)malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
        free(text);
        fclose(f);
        return NULL;
    }
    fclose(f);
    text[size] = '\0';
    return text;
}

int main(int argc, char **argv) {
    const char *dir = NULL, *prompt = NULL, *target = NULL, *target_file = NULL;
    char *target_owned = NULL;
    int layers = 3;
    int probe_token = -1;
    int probe_hc = 0;
    int probe_delta = 0;
    int probe_layer = 0;
    int probe_ple = 0;
    int probe_attention = 0;
    glm53f_st_context *ctx;
    int tensor_count = 0, ple_parts = 0;
    size_t bytes = 0;
    const char *ple_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) target = argv[++i];
        else if (!strcmp(argv[i], "--target-file") && i + 1 < argc) target_file = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-token") && i + 1 < argc) probe_token = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-hc")) probe_hc = 1;
        else if (!strcmp(argv[i], "--probe-delta")) probe_delta = probe_hc = 1;
        else if (!strcmp(argv[i], "--probe-layer")) probe_layer = probe_delta = probe_hc = 1;
        else if (!strcmp(argv[i], "--probe-ple")) probe_ple = probe_layer = probe_delta = probe_hc = 1;
        else if (!strcmp(argv[i], "--probe-attention")) probe_attention = 1;
        else if (argv[i][0] != '-' && !dir) dir = argv[i];
        else {
            fprintf(stderr, "usage: %s MODEL_DIR [--layers 1..3] [--probe-token ID [--probe-hc|--probe-delta]] [--prompt TEXT] [--target TEXT|--target-file PATH]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!dir || layers < 1 || layers > 3 || probe_token >= Q38FN_VOCAB ||
        (probe_hc && probe_token < 0) ||
        (target && target_file) ||
        ((target || target_file) && !prompt)) {
        fprintf(stderr, "usage: %s MODEL_DIR [--layers 1..3] [--probe-token ID [--probe-hc|--probe-delta]] [--prompt TEXT] [--target TEXT|--target-file PATH]\n", argv[0]);
        return 2;
    }
    if (target_file) {
        target_owned = read_text_file(target_file);
        if (!target_owned) {
            fprintf(stderr, "q38fn_runner: cannot read target file: %s\n", target_file);
            return 1;
        }
        target = target_owned;
    }
    ctx = glm53f_st_open(dir);
    if (!ctx || !glm53f_st_find(ctx, "model.language_model.embed_tokens.weight", NULL)) {
        fprintf(stderr, "q38fn_runner: checkpoint open failed\n");
        free(target_owned); glm53f_st_close(ctx); return 1;
    }
    for (int i = 0; i < ctx->n_entries; ++i) {
        int layer = layer_of(ctx->entries[i].name);
        const st_context *owner = ctx->shards[ctx->entries[i].shard].st;
        const st_tensor_info *t = &owner->tensors[ctx->entries[i].tensor];
        if (layer >= 0 && layer < layers) { tensor_count++; bytes += t->nbytes; }
        if (!strncmp(ctx->entries[i].name, ple_prefix, strlen(ple_prefix)))
            ple_parts++;
    }
    printf("Q38FN_RUNNER_VALIDATE layers=%d tensors=%d logical_bytes=%zu "
           "ple_parts=%d ple_rows=%d ple_row_bytes=%d full_attention_layers=%d\n",
           layers, tensor_count, bytes, ple_parts, Q38FN_NGRAM_ROWS,
           Q38FN_NGRAM_ROW_BYTES, (Q38FN_LAYERS - 1) / 4);
    if (probe_token >= 0) {
        float embedding[Q38FN_HIDDEN];
        double sum = 0.0, sumsq = 0.0;
        if (q38fn_read_bf16_row(ctx, "model.language_model.embed_tokens.weight",
                                (size_t)probe_token, Q38FN_HIDDEN, embedding) != 0) {
            fprintf(stderr, "q38fn_runner: embedding read failed for token %d\n", probe_token);
            free(target_owned); glm53f_st_close(ctx); return 1;
        }
        for (int i = 0; i < Q38FN_HIDDEN; ++i) {
            sum += embedding[i];
            sumsq += (double)embedding[i] * embedding[i];
        }
        printf("Q38FN_RUNNER_EMBED token=%d width=%d sum=%.9g l2=%.9g checksum=%016llx\n",
               probe_token, Q38FN_HIDDEN, sum, sumsq,
               (unsigned long long)q38fn_f32_checksum(embedding, Q38FN_HIDDEN));
        if (probe_attention) {
            float *hyper = (float *)malloc(Q38FN_HC_COUNT * Q38FN_HIDDEN * sizeof(*hyper));
            float mixed[Q38FN_HIDDEN], injection[Q38FN_HC_COUNT], attention[Q38FN_HIDDEN];
            q38fn_attention_state attention_state = {0};
            if (!hyper) {
                free(target_owned); glm53f_st_close(ctx); return 1;
            }
            for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream)
                memcpy(hyper + stream * Q38FN_HIDDEN, embedding, sizeof(embedding));
            if (q38fn_attention_state_init(&attention_state, 16) != 0 ||
                q38fn_gated_residual_mix(ctx, 3, "attn", hyper, mixed, injection) != 0 ||
                q38fn_attention_step(ctx, 3, &attention_state, mixed, attention) != 0) {
                fprintf(stderr, "q38fn_runner: layer-3 attention failed\n");
                q38fn_attention_state_destroy(&attention_state); free(hyper);
                free(target_owned); glm53f_st_close(ctx); return 1;
            }
            printf("Q38FN_RUNNER_ATTN layer=3 position=0 output_checksum=%016llx injection=",
                   (unsigned long long)q38fn_f32_checksum(attention, Q38FN_HIDDEN));
            for (int i = 0; i < Q38FN_HC_COUNT; ++i) printf("%s%.9g", i ? "," : "", injection[i]);
            putchar('\n');
            q38fn_attention_state_destroy(&attention_state); free(hyper);
        }
        if (probe_hc) {
            float *hyper = (float *)malloc(Q38FN_HC_COUNT * Q38FN_HIDDEN * sizeof(*hyper));
            float mixed[Q38FN_HIDDEN], injection[Q38FN_HC_COUNT];
            if (!hyper) {
                fprintf(stderr, "q38fn_runner: hyperconnection allocation failed\n");
                free(target_owned); glm53f_st_close(ctx); return 1;
            }
            for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream)
                memcpy(hyper + stream * Q38FN_HIDDEN, embedding, sizeof(embedding));
            if (q38fn_gated_residual_mix(ctx, 0, "attn", hyper, mixed, injection) != 0) {
                fprintf(stderr, "q38fn_runner: layer-0 attention hyperconnection failed\n");
                free(hyper); free(target_owned); glm53f_st_close(ctx); return 1;
            }
            printf("Q38FN_RUNNER_HC layer=0 block=attn mixed_checksum=%016llx injection=",
                   (unsigned long long)q38fn_f32_checksum(mixed, Q38FN_HIDDEN));
            for (int i = 0; i < Q38FN_HC_COUNT; ++i)
                printf("%s%.9g", i ? "," : "", injection[i]);
            putchar('\n');
            if (dump_vector("hc-attn-x", mixed, Q38FN_HIDDEN) ||
                dump_vector("hc-attn-inj", injection, Q38FN_HC_COUNT)) return 1;
            if (probe_delta) {
                q38fn_delta_state delta_state = {0};
                float attention[Q38FN_HIDDEN];
                if (q38fn_delta_state_init(&delta_state) != 0 ||
                    q38fn_gated_delta_step(ctx, 0, &delta_state, mixed, attention) != 0) {
                    fprintf(stderr, "q38fn_runner: layer-0 Gated DeltaNet failed\n");
                    q38fn_delta_state_destroy(&delta_state);
                    free(hyper); free(target_owned); glm53f_st_close(ctx); return 1;
                }
                for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream)
                    for (int i = 0; i < Q38FN_HIDDEN; ++i)
                        hyper[stream * Q38FN_HIDDEN + i] += injection[stream] * attention[i];
                printf("Q38FN_RUNNER_DELTA layer=0 output_checksum=%016llx hyper_checksum=%016llx\n",
                       (unsigned long long)q38fn_f32_checksum(attention, Q38FN_HIDDEN),
                       (unsigned long long)q38fn_f32_checksum(
                           hyper, Q38FN_HC_COUNT * Q38FN_HIDDEN));
                if (dump_vector("delta-y", attention, Q38FN_HIDDEN) ||
                    dump_vector("after-attn-residual", hyper,
                                Q38FN_HC_COUNT * Q38FN_HIDDEN)) return 1;
                if (probe_layer) {
                    float mlp_input[Q38FN_HIDDEN], mlp_injection[Q38FN_HC_COUNT];
                    float moe_output[Q38FN_HIDDEN];
                    int experts[Q38FN_ACTIVE_EXPERTS];
                    if (q38fn_gated_residual_mix(ctx, 0, "mlp", hyper, mlp_input,
                                                 mlp_injection) != 0 ||
                        q38fn_moe_step(ctx, 0, mlp_input, moe_output, experts) != 0) {
                        fprintf(stderr, "q38fn_runner: layer-0 MoE failed\n");
                        q38fn_delta_state_destroy(&delta_state);
                        free(hyper); free(target_owned); glm53f_st_close(ctx); return 1;
                    }
                    for (int stream = 0; stream < Q38FN_HC_COUNT; ++stream)
                        for (int i = 0; i < Q38FN_HIDDEN; ++i)
                            hyper[stream * Q38FN_HIDDEN + i] += mlp_injection[stream] * moe_output[i];
                    printf("Q38FN_RUNNER_LAYER layer=0 moe_checksum=%016llx hyper_checksum=%016llx experts=",
                           (unsigned long long)q38fn_f32_checksum(moe_output, Q38FN_HIDDEN),
                           (unsigned long long)q38fn_f32_checksum(
                               hyper, Q38FN_HC_COUNT * Q38FN_HIDDEN));
                    for (int i = 0; i < Q38FN_ACTIVE_EXPERTS; ++i)
                        printf("%s%d", i ? "," : "", experts[i]);
                    putchar('\n');
                    if (dump_vector("hc-mlp-x", mlp_input, Q38FN_HIDDEN) ||
                        dump_vector("hc-mlp-inj", mlp_injection, Q38FN_HC_COUNT) ||
                        dump_vector("moe-y", moe_output, Q38FN_HIDDEN) ||
                        dump_vector("layer-0", hyper,
                                    Q38FN_HC_COUNT * Q38FN_HIDDEN)) return 1;
                    if (probe_ple) {
                        q38fn_ple_state ple_state = {0};
                        if (q38fn_ple_state_init(&ple_state) != 0 ||
                            q38fn_ple_step(ctx, Q38FN_PLE_LAYER, &ple_state,
                                            (uint64_t)probe_token, hyper) != 0) {
                            fprintf(stderr, "q38fn_runner: layer-1 PLE failed\n");
                            q38fn_ple_state_destroy(&ple_state);
                            q38fn_delta_state_destroy(&delta_state);
                            free(hyper); free(target_owned); glm53f_st_close(ctx); return 1;
                        }
                        printf("Q38FN_RUNNER_PLE layer=1 token=%d hyper_checksum=%016llx\n",
                               probe_token, (unsigned long long)q38fn_f32_checksum(
                                   hyper, Q38FN_HC_COUNT * Q38FN_HIDDEN));
                        q38fn_ple_state_destroy(&ple_state);
                    }
                }
                q38fn_delta_state_destroy(&delta_state);
            }
            free(hyper);
        }
    }
    if (prompt) {
        char tokenizer[4096];
        glm5_bpe bpe = {0};
        char *rendered = NULL;
        int32_t *tokens = NULL;
        int ntokens;
        int n = snprintf(tokenizer, sizeof(tokenizer), "%s/tokenizer.json", dir);
        if (n < 0 || (size_t)n >= sizeof(tokenizer) ||
            glm5_bpe_load(tokenizer, &bpe) != 0) {
            fprintf(stderr, "q38fn_runner: tokenizer load failed: %s\n", tokenizer);
            glm5_bpe_free(&bpe);
            free(target_owned);
            glm53f_st_close(ctx);
            return 1;
        }
        /* Q38FN keeps chat control tokens in added_tokens rather than the
         * model vocabulary object. These IDs are read from the checkpoint's
         * tokenizer metadata and are fixed by the shipped vocabulary. */
        bpe.im_start = 248045;
        bpe.im_end = 248046;
        bpe.think = 248068;
        bpe.end_think = 248069;
        bpe.endoftext = 248044;
        size_t cap = strlen(prompt) + 128;
        rendered = (char *)malloc(cap);
        tokens = (int32_t *)malloc((cap + 1) * sizeof(*tokens));
        if (!rendered || !tokens ||
            snprintf(rendered, cap, "<|im_start|>user\n%s<|im_end|>\n"
                     "<|im_start|>assistant\n<think>\n", prompt) < 0) {
            fprintf(stderr, "q38fn_runner: prompt allocation failed\n");
            free(rendered); free(tokens); glm5_bpe_free(&bpe); free(target_owned); glm53f_st_close(ctx);
            return 1;
        }
        ntokens = glm5_bpe_encode(&bpe, rendered, (int *)tokens, (int)(cap + 1));
        if (ntokens < 0) {
            fprintf(stderr, "q38fn_runner: prompt tokenization failed\n");
            free(rendered); free(tokens); glm5_bpe_free(&bpe);
            free(target_owned); glm53f_st_close(ctx);
            return 1;
        }
        printf("Q38FN_RUNNER_PROMPT text=%s tokens=%d ids=", prompt, ntokens);
        for (int i = 0; i < ntokens; ++i) printf("%s%d", i ? "," : "", tokens[i]);
        putchar('\n');
        if (target) {
            size_t target_cap = strlen(target) + 128;
            char *target_text = (char *)malloc(target_cap);
            int32_t *target_tokens = (int32_t *)malloc((target_cap + 1) * sizeof(*target_tokens));
            int target_count = -1;
            if (target_text && target_tokens &&
                snprintf(target_text, target_cap, "\n</think>\n\n%s<|im_end|>\n", target) >= 0)
                target_count = glm5_bpe_encode(&bpe, target_text,
                                               (int *)target_tokens,
                                               (int)(target_cap + 1));
            if (target_count < 0) {
                fprintf(stderr, "q38fn_runner: target tokenization failed\n");
                free(target_text); free(target_tokens); free(rendered); free(tokens);
                glm5_bpe_free(&bpe); free(target_owned); glm53f_st_close(ctx); return 1;
            }
            printf("Q38FN_RUNNER_TARGET text=%s tokens=%d ids=", target, target_count);
            for (int i = 0; i < target_count; ++i)
                printf("%s%d", i ? "," : "", target_tokens[i]);
            putchar('\n');
            free(target_text); free(target_tokens);
        }
        free(rendered); free(tokens); glm5_bpe_free(&bpe);
    }
    glm53f_st_close(ctx);
    free(target_owned);
    return ple_parts == Q38FN_NGRAM_SHARDS ? 0 : 1;
}
