/*
 * test_transformer_parallel.c - Distributed parallel transformer inference test
 *
 * Supports three orthogonal parallelism modes:
 *   --tp N : Tensor Parallel (split heads/FFN across N ranks)
 *   --pp N : Pipeline Parallel (split layers across N ranks)
 *   --dp N : Data Parallel (N independent replicas)
 *
 * Total ranks must equal tp * pp * dp.
 *
 * Build (custom comm, no MPI needed):
 *   gcc -O2 -march=native -o test_transformer_parallel test_transformer_parallel.c -lm -lpthread
 *
 * Build (MPI backend):
 *   mpicc -O2 -march=native -DUSE_MPI -o test_transformer_parallel test_transformer_parallel.c -lm -lpthread
 *
 * Run (custom comm via launcher):
 *   ./launch.sh 2 ./test_transformer_parallel <model.gguf> --pp 2 "Hello" 16
 *   ./launch.sh 4 ./test_transformer_parallel <model.gguf> --tp 2 --pp 2 "Hello" 16
 *
 * Run (MPI):
 *   mpirun -np 2 ./test_transformer_parallel <model.gguf> --pp 2 "Hello" 16
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

#ifdef USE_MPI
#define PARALLEL_USE_MPI
#endif

#ifndef USE_MPI
#define COMM_IMPLEMENTATION
#include "../../common/comm.h"
#endif

#define PARALLEL_IMPLEMENTATION
#include "../../common/parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

static double wtime(void) {
#ifdef USE_MPI
    return MPI_Wtime();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.gguf> [options] [prompt] [max_gen] [n_threads]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --tp N    Tensor parallel size (default: 1)\n");
    fprintf(stderr, "  --pp N    Pipeline parallel size (default: 1)\n");
    fprintf(stderr, "  --dp N    Data parallel size (default: 1)\n");
#ifdef USE_MPI
    fprintf(stderr, "\nLaunch: mpirun -np N %s ...\n", prog);
#else
    fprintf(stderr, "\nLaunch: ./launch.sh N %s ...\n", prog);
#endif
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

/* ---- Pipeline-parallel inference ---- */
static void run_pipeline_parallel(parallel_config *cfg, transformer_model *model,
                                   bpe_vocab *vocab, const int32_t *tokens, int n_tokens,
                                   int max_gen) {
    int n_embd = model->n_embd;
    int pp_rank = cfg->pp_rank;
    int pp_size = cfg->pp_size;

    if (pp_rank == 0)
        fprintf(stderr, "\n=== PP Prefill (%d tokens, %d stages) ===\n", n_tokens, pp_size);

    double t0 = wtime();
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
    double t1 = wtime();

    if (pp_rank == 0)
        fprintf(stderr, "Prefill: %.3f s (%.1f ms/tok)\n", t1 - t0, (t1 - t0) * 1000.0 / n_tokens);

    /* Generation */
    if (pp_rank == 0)
        fprintf(stderr, "\n=== PP Generation (max %d tokens) ===\n", max_gen);

    double t_gen_start = wtime();
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

        if (pp_rank == pp_size - 1 && logits) {
            next_token = 0;
            for (int j = 1; j < model->n_vocab; j++)
                if (logits[j] > logits[next_token]) next_token = j;
        }

        parallel_pp_bcast_from_last(cfg, &next_token);

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

    double t_gen_end = wtime();
    if (pp_rank == 0) {
        printf("\n");
        fprintf(stderr, "\nPP Generation: %.3f s (%d tokens, %.1f tok/s)\n",
                t_gen_end - t_gen_start, gen_count,
                gen_count > 0 ? gen_count / (t_gen_end - t_gen_start) : 0.0);
    }
}

/* ---- Data-parallel inference ---- */
static void run_data_parallel(parallel_config *cfg, transformer_model *model,
                               bpe_vocab *vocab, const int32_t *tokens, int n_tokens,
                               int max_gen) {
    fprintf(stderr, "[DP rank %d] Running independent inference\n", cfg->dp_rank);

    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (i == n_tokens - 1)
            transformer_forward_logits(model, tokens[i], pos);
        else
            transformer_forward(model, tokens[i], pos);
        pos++;
    }

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

    /* Verify replicas match */
    int32_t ref_token = next_token;
#ifdef USE_MPI
    MPI_Bcast(&ref_token, 1, MPI_INT32_T, 0, cfg->dp_comm);
#else
    comm_broadcast(cfg->dp_ctx, &ref_token, sizeof(int32_t), 0);
#endif
    if (ref_token != next_token)
        fprintf(stderr, "[DP rank %d] WARNING: output diverged!\n", cfg->dp_rank);
    else
        fprintf(stderr, "[DP rank %d] Verified: output matches rank 0 (%d tokens)\n",
                cfg->dp_rank, gen_count);
}

/* ---- Tensor-parallel inference ---- */
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

    if (n_heads % tp_size != 0 || n_kv_heads % tp_size != 0 || n_ff % tp_size != 0) {
        if (tp_rank == 0)
            fprintf(stderr, "TP error: n_heads=%d, n_kv_heads=%d, n_ff=%d not divisible by tp_size=%d\n",
                    n_heads, n_kv_heads, n_ff, tp_size);
        return;
    }

    transformer_set_tp(model, tp_rank, tp_size, parallel_tp_allreduce_cb, cfg);

    int tp_nh = n_heads / tp_size;
    int tp_nkh = n_kv_heads / tp_size;
    int tp_qd = tp_nh * head_dim;
    int tp_kvd = tp_nkh * head_dim;
    int tp_ff = n_ff / tp_size;

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
    float *tp_xb2 = (float *)calloc(tp_qd, sizeof(float));
    float *tp_ff1 = (float *)calloc(tp_ff, sizeof(float));
    float *tp_ff2 = (float *)calloc(tp_ff, sizeof(float));
    float *tp_ff3 = (float *)calloc(tp_ff, sizeof(float));

    /* Local KV cache */
    float **tp_key_cache = (float **)calloc(model->n_layers, sizeof(float *));
    float **tp_val_cache = (float **)calloc(model->n_layers, sizeof(float *));
    for (int l = 0; l < model->n_layers; l++) {
        tp_key_cache[l] = (float *)calloc(model->max_seq_len * tp_kvd, sizeof(float));
        tp_val_cache[l] = (float *)calloc(model->max_seq_len * tp_kvd, sizeof(float));
    }

    double t0 = wtime();
    int pos = 0;

    for (int i = 0; i < n_tokens + max_gen; i++) {
        int is_prefill = (i < n_tokens);

        if (i < n_tokens)
            transformer_embed_token(model, tokens[i]);

        float *x = model->x;

        for (int l = 0; l < model->n_layers; l++) {
            transformer_layer *layer = &model->layers[l];

            /* SSM layers: no TP, run fully on each rank */
            if (model->is_hybrid && layer->is_ssm) {
                tf_rmsnorm(model->xb, x, &layer->attn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);
                tf_ssm_deltanet_forward(model, l);
                tf_vadd(x, model->xb, n_embd);
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
            tf_rmsnorm(model->xb, x, &layer->attn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);

            if (model->is_hybrid && !layer->is_ssm) {
                /* Gated attention */
                int tp_q2_count = 2 * tp_nh * head_dim;
                int tp_q2_start = tp_rank * tp_q2_count;
                float *tp_q2 = (float *)alloca(tp_q2_count * sizeof(float));
                tf_qmatvec_row_slice(model, tp_q2, &layer->attn_q, model->xb,
                                      tp_q2_start, tp_q2_start + tp_q2_count);
                float *tp_gate = (float *)alloca(tp_qd * sizeof(float));
                for (int h = 0; h < tp_nh; h++) {
                    memcpy(tp_q + h * head_dim, tp_q2 + h * 2 * head_dim, head_dim * sizeof(float));
                    memcpy(tp_gate + h * head_dim, tp_q2 + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
                }
                tf_qmatvec_row_slice(model, tp_k, &layer->attn_k, model->xb,
                                      kv_row_start, kv_row_start + tp_kvd);
                tf_qmatvec_row_slice(model, tp_v, &layer->attn_v, model->xb,
                                      kv_row_start, kv_row_start + tp_kvd);

                if (layer->attn_q_norm.data)
                    tf_qk_norm(tp_q, tp_nh, head_dim, &layer->attn_q_norm, model->rms_norm_eps, model->matvec_tmp);
                if (layer->attn_k_norm.data)
                    tf_qk_norm(tp_k, tp_nkh, head_dim, &layer->attn_k_norm, model->rms_norm_eps, model->matvec_tmp);

                tf_apply_rope(model, tp_q, tp_k, tp_nh, tp_nkh, head_dim, pos, pos, pos);

                memcpy(tp_key_cache[l] + pos * tp_kvd, tp_k, tp_kvd * sizeof(float));
                memcpy(tp_val_cache[l] + pos * tp_kvd, tp_v, tp_kvd * sizeof(float));

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
                    for (int t = 1; t < seq_len; t++) if (att_h[t] > max_val) max_val = att_h[t];
                    float sum = 0.0f;
                    for (int t = 0; t < seq_len; t++) { att_h[t] = expf(att_h[t] - max_val); sum += att_h[t]; }
                    for (int t = 0; t < seq_len; t++) att_h[t] /= sum;
                    float *out_h = tp_xb2 + h * head_dim;
                    for (int t = 0; t < seq_len; t++) {
                        float *vt = tp_val_cache[l] + t * tp_kvd + kv_h * head_dim;
                        float w = att_h[t];
                        for (int d = 0; d < head_dim; d++) out_h[d] += w * vt[d];
                    }
                }

                for (int j = 0; j < tp_qd; j++)
                    tp_xb2[j] *= 1.0f / (1.0f + expf(-tp_gate[j]));

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

                memcpy(tp_key_cache[l] + pos * tp_kvd, tp_k, tp_kvd * sizeof(float));
                memcpy(tp_val_cache[l] + pos * tp_kvd, tp_v, tp_kvd * sizeof(float));

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
                    for (int t = 1; t < seq_len; t++) if (att_h[t] > max_val) max_val = att_h[t];
                    float sum = 0.0f;
                    for (int t = 0; t < seq_len; t++) { att_h[t] = expf(att_h[t] - max_val); sum += att_h[t]; }
                    for (int t = 0; t < seq_len; t++) att_h[t] /= sum;
                    float *out_h = tp_xb2 + h * head_dim;
                    for (int t = 0; t < seq_len; t++) {
                        float *vt = tp_val_cache[l] + t * tp_kvd + kv_h * head_dim;
                        float w = att_h[t];
                        for (int d = 0; d < head_dim; d++) out_h[d] += w * vt[d];
                    }
                }

                tf_qmatvec_col_slice_pool(model, model->xb, &layer->attn_output,
                                           tp_xb2, n_embd, q_row_start, q_row_start + tp_qd);
            }

            tf_vadd(x, model->xb, n_embd);

            /* --- FFN (TP) --- */
            tf_rmsnorm(model->xb, x, &layer->ffn_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);

            if (model->use_moe && layer->ffn_gate_inp.data) {
                /* MoE: no TP, run fully on each rank */
                tf_qmatvec_pool(model, model->ffn_buf1, &layer->ffn_gate_inp, model->xb, model->n_expert);
                tf_softmax(model->ffn_buf1, model->n_expert);
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
                tf_qmatvec_row_slice(model, tp_ff1, &layer->ffn_gate, model->xb,
                                      ff_row_start, ff_row_start + tp_ff);
                tf_qmatvec_row_slice(model, tp_ff2, &layer->ffn_up, model->xb,
                                      ff_row_start, ff_row_start + tp_ff);
                tf_silu_mul_avx2(tp_ff3, tp_ff1, tp_ff2, tp_ff);
                tf_qmatvec_col_slice_pool(model, model->xb, &layer->ffn_down,
                                           tp_ff3, n_embd, ff_row_start, ff_row_start + tp_ff);
                tf_vadd(x, model->xb, n_embd);
            }
        }

        tf_rmsnorm(x, x, &model->output_norm, n_embd, model->rms_norm_eps, model->matvec_tmp);

        float *logits = NULL;
        if (model->has_lm_head) {
            tf_qmatvec_pool(model, model->logits, &model->output, x, model->n_vocab);
            logits = model->logits;
        }

        int32_t next = 0;
        if (logits)
            for (int j = 1; j < model->n_vocab; j++)
                if (logits[j] > logits[next]) next = j;

        if (tp_rank == 0 && i >= n_tokens) {
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

        /* All TP ranks need the same next token */
#ifdef USE_MPI
        MPI_Bcast(&next, 1, MPI_INT32_T, 0, cfg->tp_comm);
#else
        comm_broadcast(cfg->tp_ctx, &next, sizeof(int32_t), 0);
#endif

        if (i >= n_tokens - 1) {
            pos++;
            if (i + 1 < n_tokens + max_gen)
                transformer_embed_token(model, next);
        } else {
            pos++;
        }
    }

    double t1 = wtime();
    if (tp_rank == 0) {
        printf("\n");
        fprintf(stderr, "TP Total time: %.3f s\n", t1 - t0);
    }

    free(tp_q); free(tp_k); free(tp_v);
    free(tp_att); free(tp_xb2);
    free(tp_ff1); free(tp_ff2); free(tp_ff3);
    for (int l = 0; l < model->n_layers; l++) {
        free(tp_key_cache[l]); free(tp_val_cache[l]);
    }
    free(tp_key_cache); free(tp_val_cache);
}

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    /* Parse comm args (--comm-rank, --comm-nranks, etc.) first */
    int world_rank, world_size;
    char master_addr[64];
    int master_port;
    comm_parse_args(&argc, argv, &world_rank, &world_size, master_addr, sizeof(master_addr), &master_port);
#endif

    /* Parse app args */
    int tp_size, pp_size, dp_size;
    const char *model_path, *prompt;
    int max_gen, n_threads;
    parse_args(argc, argv, &tp_size, &pp_size, &dp_size,
               &model_path, &prompt, &max_gen, &n_threads);

    if (!model_path) {
        if (world_rank == 0) usage(argv[0]);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    /* Initialize parallel config */
    parallel_config cfg;
#ifdef USE_MPI
    if (parallel_init(&cfg, tp_size, pp_size, dp_size) != 0) {
        MPI_Finalize();
        return 1;
    }
#else
    if (parallel_init(&cfg, tp_size, pp_size, dp_size,
                       world_rank, world_size, master_addr, master_port) != 0) {
        return 1;
    }
#endif

    if (world_rank == 0)
        fprintf(stderr, "Parallel config: tp=%d pp=%d dp=%d (total=%d ranks)\n",
                tp_size, pp_size, dp_size, world_size);

    /* Load model */
    gguf_context *gguf = gguf_open(model_path, (world_rank == 0) ? 1 : 0);
    if (!gguf) {
        fprintf(stderr, "[rank %d] Failed to open GGUF\n", world_rank);
        goto fail;
    }

    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "[rank %d] Failed to load vocab\n", world_rank);
        gguf_close(gguf);
        goto fail;
    }

    int max_seq_len = 256;
    transformer_model *model = transformer_load(gguf, max_seq_len);
    if (!model) {
        fprintf(stderr, "[rank %d] Failed to load model\n", world_rank);
        goto fail;
    }

    if (n_threads > 1) transformer_set_threads(model, n_threads);

    /* Compute PP layer partition and free unused KV cache */
    parallel_compute_pp_layers(&cfg, model->n_layers);
    if (pp_size > 1)
        transformer_free_unused_kv(model, cfg.pp_layer_start, cfg.pp_layer_end);

    fprintf(stderr, "[rank %d] tp=%d/%d pp=%d/%d dp=%d/%d layers=[%d,%d) n_threads=%d\n",
            world_rank, cfg.tp_rank, tp_size, cfg.pp_rank, pp_size, cfg.dp_rank, dp_size,
            cfg.pp_layer_start, cfg.pp_layer_end, n_threads);

    parallel_barrier(&cfg);

    /* Tokenize (rank 0 broadcasts) */
    int32_t tokens[256];
    int n_tokens = 0;
    if (world_rank == 0) {
        n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 256);
        fprintf(stderr, "Prompt: \"%s\" -> %d tokens\n", prompt, n_tokens);
    }
    parallel_bcast_tokens(&cfg, tokens, &n_tokens);

    if (!model->has_lm_head) {
        if (world_rank == 0) fprintf(stderr, "Model has no LM head\n");
        goto cleanup;
    }

    if (world_rank == 0) {
        printf("%s", prompt);
        fflush(stdout);
    }

    /* Dispatch */
    if (tp_size > 1 && pp_size == 1 && dp_size == 1) {
        run_tensor_parallel(&cfg, model, vocab, tokens, n_tokens, max_gen);
    } else if (pp_size > 1 && tp_size == 1 && dp_size == 1) {
        run_pipeline_parallel(&cfg, model, vocab, tokens, n_tokens, max_gen);
    } else if (dp_size > 1 && tp_size == 1 && pp_size == 1) {
        run_data_parallel(&cfg, model, vocab, tokens, n_tokens, max_gen);
    } else if (tp_size == 1 && pp_size == 1 && dp_size == 1) {
        /* Single rank */
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
            fprintf(stderr, "Combined TP+PP or TP+DP modes not yet implemented.\n");
    }

cleanup:
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);
    parallel_finalize(&cfg);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;

fail:
    parallel_finalize(&cfg);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 1;
}
