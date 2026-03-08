/*
 * test_transformer_parallel.c - Unified MPI parallel transformer inference test
 *
 * Supports three orthogonal parallelism modes:
 *   --tp N : Tensor Parallel (split heads/FFN across N ranks)
 *   --pp N : Pipeline Parallel (split layers across N ranks)
 *   --dp N : Data Parallel (N independent replicas)
 *
 * Total MPI ranks must equal tp * pp * dp.
 *
 * Build:
 *   mpicc -O2 -march=native -o test_transformer_parallel test_transformer_parallel.c -lm -lpthread
 *
 * Run:
 *   mpirun -np 2 ./test_transformer_parallel <model.gguf> --pp 2 "Hello" 16
 *   mpirun -np 4 ./test_transformer_parallel <model.gguf> --tp 2 --pp 2 "Hello" 16
 *   mpirun -np 4 ./test_transformer_parallel <model.gguf> --dp 4 "Hello" 16
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#define PARALLEL_IMPLEMENTATION
#include "parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void usage(const char *prog) {
    fprintf(stderr, "Usage: mpirun -np N %s <model.gguf> [options] [prompt] [max_gen] [n_threads]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --tp N    Tensor parallel size (default: 1)\n");
    fprintf(stderr, "  --pp N    Pipeline parallel size (default: 1)\n");
    fprintf(stderr, "  --dp N    Data parallel size (default: 1)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Total MPI ranks must equal tp * pp * dp.\n");
}

/* Parse command line: extract --tp/--pp/--dp, return remaining positional args */
static void parse_args(int argc, char **argv,
                       int *tp, int *pp, int *dp,
                       const char **model_path, const char **prompt,
                       int *max_gen, int *n_threads) {
    *tp = 1; *pp = 1; *dp = 1;
    *model_path = NULL;
    *prompt = "Hello";
    *max_gen = 32;
    *n_threads = 1;

    int pos_idx = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tp") == 0 && i + 1 < argc) {
            *tp = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pp") == 0 && i + 1 < argc) {
            *pp = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dp") == 0 && i + 1 < argc) {
            *dp = atoi(argv[++i]);
        } else {
            switch (pos_idx) {
                case 0: *model_path = argv[i]; break;
                case 1: *prompt = argv[i]; break;
                case 2: *max_gen = atoi(argv[i]); break;
                case 3: *n_threads = atoi(argv[i]); break;
            }
            pos_idx++;
        }
    }
}

/* ---- Pipeline-parallel inference (PP only, no TP) ---- */
static void run_pipeline_parallel(parallel_config *cfg, transformer_model *model,
                                   bpe_vocab *vocab, const int32_t *tokens, int n_tokens,
                                   int max_gen) {
    int n_embd = model->n_embd;
    int pp_rank = cfg->pp_rank;
    int pp_size = cfg->pp_size;

    /* Prefill */
    if (pp_rank == 0)
        fprintf(stderr, "\n=== PP Prefill (%d tokens, %d stages) ===\n", n_tokens, pp_size);

    double t0 = MPI_Wtime();
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (pp_rank == 0)
            transformer_embed_token(model, tokens[i]);

        if (pp_rank > 0)
            parallel_pp_recv_hidden(cfg, transformer_get_hidden(model), n_embd, pos);

        transformer_forward_partial(model, pos, cfg->pp_layer_start, cfg->pp_layer_end);

        if (pp_rank < pp_size - 1)
            parallel_pp_send_hidden(cfg, transformer_get_hidden(model), n_embd, pos);

        if (i == n_tokens - 1 && pp_rank == pp_size - 1)
            transformer_compute_logits(model);

        pos++;
    }
    double t1 = MPI_Wtime();

    if (pp_rank == 0) {
        fprintf(stderr, "Prefill: %.3f s (%.1f ms/tok)\n", t1 - t0, (t1 - t0) * 1000.0 / n_tokens);
        printf("%s", ""); /* placeholder for prompt echo */
        fflush(stdout);
    }

    /* Generation */
    if (pp_rank == 0)
        fprintf(stderr, "\n=== PP Generation (max %d tokens) ===\n", max_gen);

    double t_gen_start = MPI_Wtime();
    int gen_count = 0;
    int32_t next_token = -1;

    for (int g = 0; g < max_gen; g++) {
        float *logits = NULL;

        if (g == 0) {
            if (pp_rank == pp_size - 1)
                logits = model->logits;
        } else {
            if (pp_rank == 0)
                transformer_embed_token(model, next_token);

            if (pp_rank > 0)
                parallel_pp_recv_hidden(cfg, transformer_get_hidden(model), n_embd, pos);

            transformer_forward_partial(model, pos, cfg->pp_layer_start, cfg->pp_layer_end);

            if (pp_rank < pp_size - 1)
                parallel_pp_send_hidden(cfg, transformer_get_hidden(model), n_embd, pos);

            if (pp_rank == pp_size - 1)
                logits = transformer_compute_logits(model);

            pos++;
        }

        /* Greedy sample on last PP rank */
        if (pp_rank == pp_size - 1 && logits) {
            next_token = 0;
            for (int j = 1; j < model->n_vocab; j++)
                if (logits[j] > logits[next_token]) next_token = j;
        }

        /* Broadcast next_token to all PP ranks */
        parallel_pp_bcast_from_last(cfg, &next_token);

        /* Check EOS on rank 0, broadcast stop */
        int stop = 0;
        if (pp_rank == 0 && vocab) {
            if (next_token == vocab->eos_id || next_token == vocab->eot_id) {
                stop = 1;
            } else {
                const char *tok_str = bpe_token_to_str(vocab, next_token);
                if (tok_str) {
                    int dec_len;
                    char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                    fwrite(decoded, 1, dec_len, stdout);
                    fflush(stdout);
                    free(decoded);
                }
            }
        }
        parallel_pp_bcast_from_first(cfg, &stop);
        if (stop) { gen_count = g + 1; break; }

        if (g == 0) pos++;
        gen_count = g + 1;
    }

    double t_gen_end = MPI_Wtime();
    if (pp_rank == 0) {
        printf("\n");
        fprintf(stderr, "\nPP Generation: %.3f s (%d tokens, %.1f tok/s)\n",
                t_gen_end - t_gen_start, gen_count,
                gen_count > 0 ? gen_count / (t_gen_end - t_gen_start) : 0.0);
    }
}

/* ---- Data-parallel inference (each DP rank runs independently) ---- */
static void run_data_parallel(parallel_config *cfg, transformer_model *model,
                               bpe_vocab *vocab, const int32_t *tokens, int n_tokens,
                               int max_gen) {
    /* Each DP rank processes the same prompt independently.
     * In a real scenario, each would process different inputs. */
    fprintf(stderr, "[DP rank %d] Running independent inference\n", cfg->dp_rank);

    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (i == n_tokens - 1)
            transformer_forward_logits(model, tokens[i], pos);
        else
            transformer_forward(model, tokens[i], pos);
        pos++;
    }

    /* Generation */
    int32_t next_token = -1;
    int gen_count = 0;

    for (int g = 0; g < max_gen; g++) {
        float *logits;
        if (g == 0) {
            logits = model->logits;
        } else {
            logits = transformer_forward_logits(model, next_token, pos);
            pos++;
        }

        if (!logits) break;

        next_token = 0;
        for (int j = 1; j < model->n_vocab; j++)
            if (logits[j] > logits[next_token]) next_token = j;

        if (next_token == vocab->eos_id || next_token == vocab->eot_id)
            break;

        /* Only DP rank 0 prints output */
        if (cfg->dp_rank == 0) {
            const char *tok_str = bpe_token_to_str(vocab, next_token);
            if (tok_str) {
                int dec_len;
                char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                fwrite(decoded, 1, dec_len, stdout);
                fflush(stdout);
                free(decoded);
            }
        }

        if (g == 0) pos++;
        gen_count = g + 1;
    }

    if (cfg->dp_rank == 0)
        printf("\n");

    /* Verify all DP replicas got the same output (sanity check) */
    int32_t all_last_token = next_token;
    MPI_Bcast(&all_last_token, 1, MPI_INT32_T, 0, cfg->dp_comm);
    if (all_last_token != next_token) {
        fprintf(stderr, "[DP rank %d] WARNING: last token %d != rank 0's %d (divergence!)\n",
                cfg->dp_rank, next_token, all_last_token);
    } else {
        fprintf(stderr, "[DP rank %d] Verified: output matches rank 0 (%d tokens)\n",
                cfg->dp_rank, gen_count);
    }
}

/* ---- Tensor-parallel inference (TP, foundation/verification) ---- */
/* This is a minimal TP implementation that verifies the sliced matvec primitives.
 * It runs column-parallel QKV and FFN gate/up, then row-parallel output proj and FFN down.
 * Currently works for the standard (non-hybrid) attention path. */
static void run_tensor_parallel(parallel_config *cfg, transformer_model *model,
                                 bpe_vocab *vocab, const int32_t *tokens, int n_tokens,
                                 int max_gen) {
    int tp_rank = cfg->tp_rank;
    int tp_size = cfg->tp_size;
    int n_embd = model->n_embd;
    int n_heads = model->n_heads;
    int n_kv_heads = model->n_kv_heads;
    int head_dim = model->head_dim;
    int n_ff = model->n_ff;

    /* Verify divisibility */
    if (n_heads % tp_size != 0 || n_kv_heads % tp_size != 0 || n_ff % tp_size != 0) {
        if (tp_rank == 0)
            fprintf(stderr, "TP error: n_heads=%d, n_kv_heads=%d, n_ff=%d not divisible by tp_size=%d\n",
                    n_heads, n_kv_heads, n_ff, tp_size);
        return;
    }

    /* Configure the model for TP */
    transformer_set_tp(model, tp_rank, tp_size,
                        parallel_tp_allreduce_cb, cfg);

    /* TP dimensions */
    int tp_nh = n_heads / tp_size;       /* heads per TP rank */
    int tp_nkh = n_kv_heads / tp_size;   /* KV heads per TP rank */
    int tp_qd = tp_nh * head_dim;        /* Q dim per TP rank */
    int tp_kvd = tp_nkh * head_dim;      /* KV dim per TP rank */
    int tp_ff = n_ff / tp_size;          /* FFN neurons per TP rank */

    int q_row_start = tp_rank * tp_qd;
    int kv_row_start = tp_rank * tp_kvd;
    int ff_row_start = tp_rank * tp_ff;

    if (tp_rank == 0)
        fprintf(stderr, "\n=== TP Inference (tp_size=%d, tp_nh=%d, tp_nkh=%d, tp_ff=%d) ===\n",
                tp_size, tp_nh, tp_nkh, tp_ff);

    /* Allocate local TP buffers */
    float *tp_q   = (float *)calloc(tp_qd, sizeof(float));
    float *tp_k   = (float *)calloc(tp_kvd, sizeof(float));
    float *tp_v   = (float *)calloc(tp_kvd, sizeof(float));
    float *tp_att = (float *)calloc(tp_nh * model->max_seq_len, sizeof(float));
    float *tp_xb2 = (float *)calloc(tp_qd, sizeof(float));  /* attention output */
    float *tp_ff1 = (float *)calloc(tp_ff, sizeof(float));   /* gate */
    float *tp_ff2 = (float *)calloc(tp_ff, sizeof(float));   /* up */
    float *tp_ff3 = (float *)calloc(tp_ff, sizeof(float));   /* silu(gate)*up */

    /* Local KV cache: [n_layers][max_seq_len * tp_kvd] */
    float **tp_key_cache = (float **)calloc(model->n_layers, sizeof(float *));
    float **tp_val_cache = (float **)calloc(model->n_layers, sizeof(float *));
    for (int l = 0; l < model->n_layers; l++) {
        tp_key_cache[l] = (float *)calloc(model->max_seq_len * tp_kvd, sizeof(float));
        tp_val_cache[l] = (float *)calloc(model->max_seq_len * tp_kvd, sizeof(float));
    }

    double t0 = MPI_Wtime();
    int pos = 0;

    for (int i = 0; i < n_tokens + max_gen; i++) {
        int32_t token_id;
        int is_prefill = (i < n_tokens);

        if (is_prefill) {
            token_id = tokens[i];
        } else if (i == n_tokens) {
            /* First gen step: logits already computed below */
            token_id = -1;
        } else {
            token_id = -1; /* set below from broadcast */
        }

        /* Embed token (all TP ranks need the full embedding) */
        if (i < n_tokens) {
            transformer_embed_token(model, token_id);
        }

        float *x = model->x;  /* current hidden state [n_embd] */

        /* Process each layer */
        for (int l = 0; l < model->n_layers; l++) {
            transformer_layer *layer = &model->layers[l];

            /* Skip SSM layers for TP foundation (run full on each rank) */
            if (model->is_hybrid && layer->is_ssm) {
                /* RMSNorm */
                tf_rmsnorm(model->xb, x, &layer->attn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);
                /* Run SSM fully on each rank (no TP for SSM) */
                tf_ssm_deltanet_forward(model, l);
                tf_vadd(x, model->xb, n_embd);
                /* FFN for SSM layers is also not split */
                tf_rmsnorm(model->xb, x, &layer->ffn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);
                if (layer->ffn_gate.data) {
                    tf_qmatvec_pool(model, model->ffn_buf1, &layer->ffn_gate, model->xb, n_ff);
                    tf_qmatvec_pool(model, model->ffn_buf2, &layer->ffn_up, model->xb, n_ff);
                    tf_silu_mul_avx2(model->ffn_buf3, model->ffn_buf1, model->ffn_buf2, n_ff);
                    tf_qmatvec_pool(model, model->xb, &layer->ffn_down, model->ffn_buf3, n_embd);
                    tf_vadd(x, model->xb, n_embd);
                }
                continue;
            }

            /* --- Attention (TP) --- */
            /* RMSNorm (all ranks compute the same result) */
            tf_rmsnorm(model->xb, x, &layer->attn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);

            /* Column-parallel QKV: each rank computes its head slice */
            if (model->is_hybrid && !layer->is_ssm) {
                /* Gated attention: Q has 2*q_dim interleaved rows */
                int q2_dim = 2 * n_heads * head_dim;
                int tp_q2_start = tp_rank * (2 * tp_nh * head_dim);
                int tp_q2_count = 2 * tp_nh * head_dim;
                float *tp_q2 = (float *)alloca(tp_q2_count * sizeof(float));
                tf_qmatvec_row_slice(model, tp_q2, &layer->attn_q, model->xb,
                                      tp_q2_start, tp_q2_start + tp_q2_count);
                /* De-interleave into tp_q and gate */
                float *tp_gate = (float *)alloca(tp_qd * sizeof(float));
                for (int h = 0; h < tp_nh; h++) {
                    memcpy(tp_q + h * head_dim, tp_q2 + h * 2 * head_dim, head_dim * sizeof(float));
                    memcpy(tp_gate + h * head_dim, tp_q2 + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
                }
                tf_qmatvec_row_slice(model, tp_k, &layer->attn_k, model->xb,
                                      kv_row_start, kv_row_start + tp_kvd);
                tf_qmatvec_row_slice(model, tp_v, &layer->attn_v, model->xb,
                                      kv_row_start, kv_row_start + tp_kvd);

                /* QK-Norm on local heads */
                if (layer->attn_q_norm.data)
                    tf_qk_norm(tp_q, tp_nh, head_dim, &layer->attn_q_norm, model->rms_norm_eps, model->matvec_tmp);
                if (layer->attn_k_norm.data)
                    tf_qk_norm(tp_k, tp_nkh, head_dim, &layer->attn_k_norm, model->rms_norm_eps, model->matvec_tmp);

                /* RoPE on local heads */
                tf_apply_rope(model, tp_q, tp_k, tp_nh, tp_nkh, head_dim, pos, pos, pos);

                /* Store to local KV cache */
                float *kc = tp_key_cache[l] + pos * tp_kvd;
                float *vc = tp_val_cache[l] + pos * tp_kvd;
                memcpy(kc, tp_k, tp_kvd * sizeof(float));
                memcpy(vc, tp_v, tp_kvd * sizeof(float));

                /* Local attention on this rank's heads */
                int seq_len = pos + 1;
                float scale = 1.0f / sqrtf((float)head_dim);
                int tp_gqa = tp_nh / tp_nkh;
                memset(tp_xb2, 0, tp_qd * sizeof(float));

                for (int h = 0; h < tp_nh; h++) {
                    float *q_h = tp_q + h * head_dim;
                    int kv_h = h / tp_gqa;
                    float *att_h = tp_att + h * model->max_seq_len;

                    /* QK dot products */
                    for (int t = 0; t < seq_len; t++) {
                        float *kt = tp_key_cache[l] + t * tp_kvd + kv_h * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) dot += q_h[d] * kt[d];
                        att_h[t] = dot * scale;
                    }

                    /* Softmax */
                    float max_val = att_h[0];
                    for (int t = 1; t < seq_len; t++)
                        if (att_h[t] > max_val) max_val = att_h[t];
                    float sum = 0.0f;
                    for (int t = 0; t < seq_len; t++) {
                        att_h[t] = expf(att_h[t] - max_val);
                        sum += att_h[t];
                    }
                    for (int t = 0; t < seq_len; t++) att_h[t] /= sum;

                    /* Weighted sum of values */
                    float *out_h = tp_xb2 + h * head_dim;
                    for (int t = 0; t < seq_len; t++) {
                        float *vt = tp_val_cache[l] + t * tp_kvd + kv_h * head_dim;
                        float w = att_h[t];
                        for (int d = 0; d < head_dim; d++) out_h[d] += w * vt[d];
                    }
                }

                /* Apply sigmoid gate */
                for (int j = 0; j < tp_qd; j++)
                    tp_xb2[j] *= 1.0f / (1.0f + expf(-tp_gate[j]));

                /* Row-parallel output projection: each rank has tp_qd columns */
                tf_qmatvec_col_slice_pool(model, model->xb, &layer->attn_output,
                                           tp_xb2, n_embd, q_row_start, q_row_start + tp_qd);
            } else {
                /* Standard attention */
                tf_qmatvec_row_slice(model, tp_q, &layer->attn_q, model->xb,
                                      q_row_start, q_row_start + tp_qd);
                tf_qmatvec_row_slice(model, tp_k, &layer->attn_k, model->xb,
                                      kv_row_start, kv_row_start + tp_kvd);
                tf_qmatvec_row_slice(model, tp_v, &layer->attn_v, model->xb,
                                      kv_row_start, kv_row_start + tp_kvd);

                if (layer->attn_q_norm.data)
                    tf_qk_norm(tp_q, tp_nh, head_dim, &layer->attn_q_norm, model->rms_norm_eps, model->matvec_tmp);
                if (layer->attn_k_norm.data)
                    tf_qk_norm(tp_k, tp_nkh, head_dim, &layer->attn_k_norm, model->rms_norm_eps, model->matvec_tmp);

                tf_apply_rope(model, tp_q, tp_k, tp_nh, tp_nkh, head_dim, pos, pos, pos);

                float *kc = tp_key_cache[l] + pos * tp_kvd;
                float *vc = tp_val_cache[l] + pos * tp_kvd;
                memcpy(kc, tp_k, tp_kvd * sizeof(float));
                memcpy(vc, tp_v, tp_kvd * sizeof(float));

                int seq_len = pos + 1;
                float scale = 1.0f / sqrtf((float)head_dim);
                int tp_gqa = tp_nh / tp_nkh;
                memset(tp_xb2, 0, tp_qd * sizeof(float));

                for (int h = 0; h < tp_nh; h++) {
                    float *q_h = tp_q + h * head_dim;
                    int kv_h = h / tp_gqa;
                    float *att_h = tp_att + h * model->max_seq_len;

                    for (int t = 0; t < seq_len; t++) {
                        float *kt = tp_key_cache[l] + t * tp_kvd + kv_h * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) dot += q_h[d] * kt[d];
                        att_h[t] = dot * scale;
                    }

                    float max_val = att_h[0];
                    for (int t = 1; t < seq_len; t++)
                        if (att_h[t] > max_val) max_val = att_h[t];
                    float sum = 0.0f;
                    for (int t = 0; t < seq_len; t++) {
                        att_h[t] = expf(att_h[t] - max_val);
                        sum += att_h[t];
                    }
                    for (int t = 0; t < seq_len; t++) att_h[t] /= sum;

                    float *out_h = tp_xb2 + h * head_dim;
                    for (int t = 0; t < seq_len; t++) {
                        float *vt = tp_val_cache[l] + t * tp_kvd + kv_h * head_dim;
                        float w = att_h[t];
                        for (int d = 0; d < head_dim; d++) out_h[d] += w * vt[d];
                    }
                }

                /* Row-parallel output projection */
                tf_qmatvec_col_slice_pool(model, model->xb, &layer->attn_output,
                                           tp_xb2, n_embd, q_row_start, q_row_start + tp_qd);
            }

            /* Residual (all ranks have the same allreduced result) */
            tf_vadd(x, model->xb, n_embd);

            /* --- FFN (TP) --- */
            tf_rmsnorm(model->xb, x, &layer->ffn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);

            if (model->use_moe && layer->ffn_gate_inp.data) {
                /* MoE: no TP for foundation, run fully on each rank */
                tf_qmatvec_pool(model, model->ffn_buf1, &layer->ffn_gate_inp, model->xb, model->n_expert);
                tf_softmax(model->ffn_buf1, model->n_expert);
                /* Select top experts */
                int n_top = model->n_expert_used;
                int n_ff_exp = model->n_ff_expert;
                int *top_idx = (int *)alloca(n_top * sizeof(int));
                float *top_w = (float *)alloca(n_top * sizeof(float));
                int k = 0;
                for (int e = 0; e < model->n_expert; e++) {
                    float w = model->ffn_buf1[e];
                    if (k < n_top) {
                        top_idx[k] = e; top_w[k] = w; k++;
                    } else if (w > top_w[0]) {
                        /* Find min and replace (simple for small n_top) */
                        int mi = 0;
                        for (int j = 1; j < n_top; j++) if (top_w[j] < top_w[mi]) mi = j;
                        top_w[mi] = w; top_idx[mi] = e;
                    }
                }
                float wsum = 0.0f;
                for (int j = 0; j < n_top; j++) wsum += top_w[j];
                if (wsum > 0.0f) for (int j = 0; j < n_top; j++) top_w[j] /= wsum;

                memset(model->xb2, 0, n_embd * sizeof(float));
                for (int ei = 0; ei < n_top; ei++) {
                    tf_qmatvec_expert_pool(model, model->ffn_buf2, &layer->ffn_up_exps, top_idx[ei], model->xb, n_ff_exp);
                    tf_qmatvec_expert_pool(model, model->ffn_buf3, &layer->ffn_gate_exps, top_idx[ei], model->xb, n_ff_exp);
                    tf_silu_mul_avx2(model->ffn_buf3, model->ffn_buf3, model->ffn_buf2, n_ff_exp);
                    tf_qmatvec_expert_pool(model, model->q, &layer->ffn_down_exps, top_idx[ei], model->ffn_buf3, n_embd);
                    for (int j = 0; j < n_embd; j++) model->xb2[j] += top_w[ei] * model->q[j];
                }
                tf_vadd(x, model->xb2, n_embd);
            } else if (layer->ffn_gate.data) {
                /* Dense FFN with TP: column-parallel gate/up, row-parallel down */
                tf_qmatvec_row_slice(model, tp_ff1, &layer->ffn_gate, model->xb,
                                      ff_row_start, ff_row_start + tp_ff);
                tf_qmatvec_row_slice(model, tp_ff2, &layer->ffn_up, model->xb,
                                      ff_row_start, ff_row_start + tp_ff);
                tf_silu_mul_avx2(tp_ff3, tp_ff1, tp_ff2, tp_ff);

                /* Row-parallel down projection */
                tf_qmatvec_col_slice_pool(model, model->xb, &layer->ffn_down,
                                           tp_ff3, n_embd, ff_row_start, ff_row_start + tp_ff);
                tf_vadd(x, model->xb, n_embd);
            }
        }

        /* Final norm (all ranks compute same thing) */
        tf_rmsnorm(x, x, &model->output_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);

        /* Compute logits (all ranks compute full logits for simplicity) */
        float *logits = NULL;
        if (model->has_lm_head) {
            tf_qmatvec_pool(model, model->logits, &model->output, x, model->n_vocab);
            logits = model->logits;
        }

        /* Greedy sample */
        int32_t next = 0;
        if (logits) {
            for (int j = 1; j < model->n_vocab; j++)
                if (logits[j] > logits[next]) next = j;
        }

        /* Print output (only on TP rank 0) */
        if (tp_rank == 0) {
            if (i >= n_tokens) {
                if (next == vocab->eos_id || next == vocab->eot_id) {
                    fprintf(stderr, "  [EOS %d]\n", next);
                    break;
                }
                const char *tok_str = bpe_token_to_str(vocab, next);
                if (tok_str) {
                    int dec_len;
                    char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                    fwrite(decoded, 1, dec_len, stdout);
                    fflush(stdout);
                    free(decoded);
                }
            }
        }

        /* All TP ranks need the same next token for embedding */
        MPI_Bcast(&next, 1, MPI_INT32_T, 0, cfg->tp_comm);

        /* For next iteration, embed this token */
        if (i >= n_tokens - 1) {
            pos++;
            /* Set next token for embedding in next iteration */
            if (i + 1 < n_tokens + max_gen) {
                transformer_embed_token(model, next);
            }
        } else {
            pos++;
        }
    }

    double t1 = MPI_Wtime();
    if (tp_rank == 0) {
        printf("\n");
        fprintf(stderr, "TP Total time: %.3f s\n", t1 - t0);
    }

    /* Cleanup */
    free(tp_q); free(tp_k); free(tp_v);
    free(tp_att); free(tp_xb2);
    free(tp_ff1); free(tp_ff2); free(tp_ff3);
    for (int l = 0; l < model->n_layers; l++) {
        free(tp_key_cache[l]); free(tp_val_cache[l]);
    }
    free(tp_key_cache); free(tp_val_cache);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Parse args */
    int tp_size, pp_size, dp_size;
    const char *model_path, *prompt;
    int max_gen, n_threads;
    parse_args(argc, argv, &tp_size, &pp_size, &dp_size,
               &model_path, &prompt, &max_gen, &n_threads);

    if (!model_path) {
        if (world_rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* Initialize parallel config */
    parallel_config cfg;
    if (parallel_init(&cfg, tp_size, pp_size, dp_size) != 0) {
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0)
        fprintf(stderr, "Parallel config: tp=%d pp=%d dp=%d (total=%d ranks)\n",
                tp_size, pp_size, dp_size, world_size);

    /* Load model (all ranks load the full model) */
    gguf_context *gguf = gguf_open(model_path, (world_rank == 0) ? 1 : 0);
    if (!gguf) {
        fprintf(stderr, "[rank %d] Failed to open GGUF\n", world_rank);
        MPI_Finalize();
        return 1;
    }

    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "[rank %d] Failed to load vocab\n", world_rank);
        gguf_close(gguf);
        MPI_Finalize();
        return 1;
    }

    int max_seq_len = 256;
    transformer_model *model = transformer_load(gguf, max_seq_len);
    if (!model) {
        fprintf(stderr, "[rank %d] Failed to load model\n", world_rank);
        MPI_Finalize();
        return 1;
    }

    if (n_threads > 1) transformer_set_threads(model, n_threads);

    /* Compute PP layer partition */
    parallel_compute_pp_layers(&cfg, model->n_layers);

    fprintf(stderr, "[rank %d] tp=%d/%d pp=%d/%d dp=%d/%d layers=[%d,%d) n_threads=%d\n",
            world_rank, cfg.tp_rank, tp_size, cfg.pp_rank, pp_size, cfg.dp_rank, dp_size,
            cfg.pp_layer_start, cfg.pp_layer_end, n_threads);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Tokenize (rank 0 broadcasts) */
    int32_t tokens[256];
    int n_tokens = 0;
    if (world_rank == 0) {
        n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 256);
        fprintf(stderr, "Prompt: \"%s\" -> %d tokens\n", prompt, n_tokens);
    }
    MPI_Bcast(&n_tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tokens, n_tokens, MPI_INT32_T, 0, MPI_COMM_WORLD);

    if (!model->has_lm_head) {
        if (world_rank == 0) fprintf(stderr, "Model has no LM head\n");
        goto cleanup;
    }

    /* Print prompt (rank 0 only) */
    if (world_rank == 0) {
        printf("%s", prompt);
        fflush(stdout);
    }

    /* Dispatch to the appropriate parallelism mode */
    if (tp_size > 1 && pp_size == 1 && dp_size == 1) {
        /* Pure TP */
        run_tensor_parallel(&cfg, model, vocab, tokens, n_tokens, max_gen);
    } else if (pp_size > 1 && tp_size == 1 && dp_size == 1) {
        /* Pure PP */
        run_pipeline_parallel(&cfg, model, vocab, tokens, n_tokens, max_gen);
    } else if (dp_size > 1 && tp_size == 1 && pp_size == 1) {
        /* Pure DP */
        run_data_parallel(&cfg, model, vocab, tokens, n_tokens, max_gen);
    } else if (tp_size == 1 && pp_size == 1 && dp_size == 1) {
        /* Single rank: normal inference */
        if (world_rank == 0) {
            int pos = 0;
            for (int i = 0; i < n_tokens; i++) {
                if (i == n_tokens - 1)
                    transformer_forward_logits(model, tokens[i], pos);
                else
                    transformer_forward(model, tokens[i], pos);
                pos++;
            }

            for (int g = 0; g < max_gen; g++) {
                float *logits = (g == 0) ? model->logits
                    : transformer_forward_logits(model, tokens[0], pos);
                if (g > 0) pos++;
                if (!logits) break;

                int32_t next = 0;
                for (int j = 1; j < model->n_vocab; j++)
                    if (logits[j] > logits[next]) next = j;

                if (next == vocab->eos_id || next == vocab->eot_id) break;

                const char *tok_str = bpe_token_to_str(vocab, next);
                if (tok_str) {
                    int dec_len;
                    char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                    fwrite(decoded, 1, dec_len, stdout);
                    fflush(stdout);
                    free(decoded);
                }

                tokens[0] = next;
                if (g == 0) pos++;
            }
            printf("\n");
        }
    } else {
        if (world_rank == 0)
            fprintf(stderr, "Combined TP+PP or TP+DP modes not yet implemented in this test.\n"
                            "Use pure --tp, --pp, or --dp for now.\n");
    }

cleanup:
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);
    parallel_finalize(&cfg);
    MPI_Finalize();
    return 0;
}
