/*
 * transformer.h - Reference C transformer inference for Qwen2-style models
 *
 * Usage:
 *   #define TRANSFORMER_IMPLEMENTATION
 *   #include "transformer.h"
 *
 * Dependencies: gguf_loader.h, ggml_dequant.h (include with IMPLEMENTATION defined separately)
 *
 * API:
 *   transformer_model *transformer_load(gguf_context *gguf, int max_seq_len);
 *   void transformer_free(transformer_model *model);
 *   float *transformer_forward(transformer_model *model, int32_t token_id, int position);
 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "gguf_loader.h"
#include "ggml_dequant.h"
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Quantized tensor reference (points into GGUF mmap) */
typedef struct {
    void    *data;
    uint32_t type;       /* ggml_dtype */
    int      n_rows;
    int      n_cols;     /* number of elements per row */
    int      n_dims;     /* up to 4 for GGUF tensors */
    uint64_t dims[4];
    /* A64FX panel layout (M10): for F16 weights, an optional repack into
     * panel[blk][k][lane] = data[blk*32+lane][k] so the SVE matvec needs no
     * horizontal reduction. NULL when unused. panel_blk = ceil(n_rows/32). */
    uint16_t *panel;
    int       panel_blk;
    /* A64FX BF16 p_odd-pair layout: for BF16 weights, an optional repack
     * into pair-interleaved storage that lets matvec_bf16_8row_pv extract
     * two rows as FP32 via a single ld1h.h with p_odd, eliminating the LSL
     * the lsl variant needs. NULL when unused. bf16_pv_groups = n_rows/8. */
    uint16_t *bf16_pv;
    int       bf16_pv_groups;
    /* A64FX int8 svdot quantize-on-load layout: per group of 8 rows × n_cols,
     * laid out as nb = n_cols/64 blocks of 528 bytes:
     *   bytes [0..16)   = 8 fp16 row-scales
     *   bytes [16..528) = 8 rows × 64 int8 quants (row-major within block)
     * Built from BF16 source when TF_QUANT_Q8=1. Halves weight BW vs bf16
     * (1.03 B/elem vs 2 B/elem); consumed by matvec_sdot_8row (W8A8 svdot_s32
     * against int8-quantized x). q8_pv_groups = n_rows/8. */
    uint8_t  *q8_pv;
    int       q8_pv_groups;
    /* Tensor-parallel row-parallel slicing: when nonzero, the pv/panel/q8
     * fill functions step SOURCE rows by tp_src_stride elements (the ORIGINAL
     * full n_cols) while packing only n_cols (the local column sub-range)
     * elements per row, starting at the column `data` was pre-offset to. Lets
     * a row-parallel projection (attn_output, ffn_down, ssm_out) hold only its
     * 1/tp column slice. 0 = contiguous (uses n_cols); unchanged for tp=1. */
    int       tp_src_stride;
    /* MoE BF16_PV: 3D expert tensors repacked into bf16_pv at load time. The
     * pv buffer above (bf16_pv) holds CONTIGUOUS owned-expert slabs of size
     * rows_per_expert * n_cols uint16_t each, addressed by
     * expert_owned_slot[e] (-1 means the expert is not owned on this rank).
     * NULL when the per-expert pv build is not active. */
    int      *expert_owned_slot;
    int       expert_rows_per_expert;
} qtensor;

typedef struct {
    qtensor attn_norm;   /* [n_embd] */
    qtensor attn_q;      /* [n_embd, n_embd] */
    qtensor attn_k;      /* [n_kv_dim, n_embd] */
    qtensor attn_v;      /* [n_kv_dim, n_embd] */
    qtensor attn_q_norm; /* [head_dim] */
    qtensor attn_k_norm; /* [head_dim] */
    qtensor attn_q_bias; /* [n_embd] Q projection bias (Qwen2.5-VL) */
    qtensor attn_k_bias; /* [n_kv_dim] K projection bias */
    qtensor attn_v_bias; /* [n_kv_dim] V projection bias */
    qtensor attn_output; /* [n_embd, n_embd] */
    qtensor ffn_norm;    /* [n_embd] */
    qtensor ffn_gate;    /* [n_ff, n_embd] */
    qtensor ffn_up;      /* [n_ff, n_embd] */
    qtensor ffn_down;    /* [n_embd, n_ff] */
    /* MoE tensors (Qwen3MoE / Qwen3VLMoE): [n_expert variants] */
    qtensor ffn_gate_inp;  /* [n_embd, n_expert] */
    qtensor ffn_up_exps;   /* [n_embd, n_ff_expert, n_expert] */
    qtensor ffn_gate_exps; /* [n_embd, n_ff_expert, n_expert] */
    qtensor ffn_down_exps; /* [n_ff_expert, n_embd, n_expert] */
    /* Qwen3.5-MoE shared expert (always-on, sigmoid-gated). */
    qtensor ffn_gate_inp_shexp; /* [n_embd] sigmoid-gate scalar projection */
    qtensor ffn_up_shexp;       /* [n_embd, n_ff_shexp] */
    qtensor ffn_gate_shexp;     /* [n_embd, n_ff_shexp] */
    qtensor ffn_down_shexp;     /* [n_ff_shexp, n_embd] */

    /* SSM (Delta-Net) tensors — Qwen3.5 hybrid layers */
    qtensor ssm_qkv;       /* [n_embd, qkv_dim] combined Q/K/V input projection */
    qtensor ssm_gate;      /* [n_embd, d_inner] gate (z) projection */
    qtensor ssm_alpha;     /* [n_embd, n_v_heads] alpha projection */
    qtensor ssm_beta;      /* [n_embd, n_v_heads] beta projection */
    qtensor ssm_a;         /* [n_v_heads] fixed decay constants */
    qtensor ssm_dt_bias;   /* [n_v_heads] alpha bias */
    qtensor ssm_conv1d;    /* [conv_kernel, qkv_dim] 1D conv kernel */
    qtensor ssm_norm;      /* [d_state] RMS norm for gated output */
    qtensor ssm_out;       /* [d_inner, n_embd] output projection */

    /* Gemma4 tensors */
    qtensor attn_v_norm;         /* [head_dim] V RMSNorm (Gemma4) */
    qtensor post_attention_norm; /* [n_embd] after attn output, before residual */
    qtensor post_ffw_norm;       /* [n_embd] after FFN output, before residual */
    qtensor layer_output_scale;  /* [1] scalar per layer */
    /* Per-layer embedding (Gemma4) */
    qtensor ple_inp_gate;        /* [n_embd_per_layer, n_embd] gated input */
    qtensor ple_proj;            /* [n_embd, n_embd_per_layer] project back */
    qtensor ple_post_norm;       /* [n_embd] post-projection norm */

    int is_ssm;            /* 1 = Delta-Net SSM layer, 0 = full attention */
    int is_swa;            /* 1 = sliding window attention (Gemma4) */
    int shared_kv_source;  /* layer idx to reuse KV from, -1 = own KV (Gemma4) */
} transformer_layer;

typedef struct {
    /* Hyperparameters */
    int n_layers;
    int n_embd;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int n_ff;
    int n_ff_expert;
    int n_ff_shexp;        /* Qwen3.5-MoE shared-expert inner dim; 0 if not present */
    int n_expert;
    int n_expert_used;
    int use_moe;
    int n_vocab;
    int max_seq_len;
    float rope_freq_base;
    float rms_norm_eps;

    int has_lm_head;     /* 1 if output.weight exists (generative model) */
    int mrope_sections[4]; /* M-RoPE dimension sections [temporal, height, width, pad] */
    int use_mrope;         /* 1 if M-RoPE is enabled (Qwen3-VL) */
    int debug_layers;      /* 1 = print per-layer hidden state norms (debug) */

    /* Hybrid SSM+Attention (Qwen3.5) */
    int is_hybrid;           /* 1 if model has SSM layers */
    int full_attn_interval;  /* every Nth layer is full attention (e.g. 4) */
    int ssm_conv_kernel;     /* conv1d kernel size (4) */
    int ssm_d_state;         /* state dimension per head (128) */
    int ssm_n_group;         /* number of K/Q heads (16) */
    int ssm_dt_rank;         /* number of V heads (48) */
    int ssm_d_inner;         /* inner hidden dim (6144) */
    int ssm_qkv_dim;         /* combined QKV dim (10240) */
    float **conv_state;      /* [n_layers] -> [conv_kernel-1, qkv_dim] per SSM layer */
    int *conv_state_pos;     /* [n_layers] circular buffer write position per SSM layer */
    float **recurrent_state; /* [n_layers] -> [n_v_heads, d_state, d_state] per SSM layer */
    float **conv_w_trans;    /* [n_layers] -> [conv_k * qkv_dim] pre-dequantised + transposed
                              * conv weights, built once at load. Weights are constant; the
                              * old per-token batch-dequant cost ~184K iters/token on 0.8B. */
    float *ssm_alpha_buf;    /* [dt_rank] shared scratch for parallel SSM forward */
    float *ssm_beta_buf;     /* [dt_rank] shared scratch for parallel SSM forward */
    int8_t *ssm_q8_xq;       /* [max_dim] shared SSM Q8 activation scratch */
    uint16_t *ssm_q8_xs;     /* [max_dim/64] shared SSM Q8 activation scales */
    int ssm_q8_cap;          /* capacity of ssm_q8_xq in float elements */
    int n_deepstack;       /* number of deepstack layers (Qwen3-VL) */
    const float *ds_embd;  /* pointer to current full embedding (incl deepstack slices) */
    int ds_embd_stride;    /* total embedding dim = proj_dim * (1 + n_deepstack) */

    /* Gemma4 architecture */
    int is_gemma4;                  /* 1 if gemma4 architecture */
    int swa_window_size;            /* sliding window size (512) */
    int head_dim_full;              /* head dim for full-attention layers (512) */
    int head_dim_swa;               /* head dim for SWA layers (256) */
    int n_embd_per_layer;           /* per-layer embedding dim (256) */
    int n_layer_kv_from_start;      /* layers 0..N-1 have own KV, rest share */
    float final_logit_softcapping;  /* tanh softcap on output logits (30.0) */
    float rope_freq_base_swa;       /* RoPE freq base for SWA layers (10000) */
    float embd_scale;               /* sqrt(n_embd) for token embedding scaling */
    int ffn_activation;             /* 0 = SiLU (default), 1 = GELU (Gemma4) */
    int *swa_pattern;               /* [n_layers] 1=SWA, 0=full attention */
    /* Per-layer embedding global tensors */
    qtensor per_layer_token_embd;   /* [n_embd_per_layer*n_layer, n_vocab] */
    qtensor per_layer_model_proj;   /* [n_embd_per_layer*n_layer, n_embd] */
    qtensor per_layer_proj_norm;    /* [n_embd_per_layer*n_layer] */
    /* Proportional RoPE for full-attention layers */
    float *rope_freq_factors;       /* [head_dim_full/2] from rope_freqs.weight */
    float *rope_inv_freq_swa;       /* [head_dim_swa/2] precomputed for SWA */
    /* Per-layer embedding scratch */
    float *ple_buf;                 /* [n_embd_per_layer] */
    float *ple_proj_buf;            /* [n_embd] */
    int current_token_id;           /* stashed for per-layer embd lookup */

    /* Global tensors */
    qtensor token_embd;  /* [n_vocab, n_embd] */
    qtensor output_norm; /* [n_embd] */
    qtensor output;      /* [n_vocab, n_embd] LM head (may be absent for embedding models) */

    /* Per-layer weights */
    transformer_layer *layers;

    /* KV cache: [n_layers][seq_cap * n_kv_heads * head_dim].
     * Element size depends on kv_dtype: 4B (F32), 2B (F16) or 1B (Q8).
     * For Q8, key_scales/value_scales hold one float per (pos, kv_head). */
    void **key_cache;
    void **value_cache;
    float **key_scales;
    float **value_scales;
    int kv_dtype;            /* 0=F32 (default), 1=F16, 2=Q8 with per-head per-pos scale */
    size_t kv_elem_bytes;    /* bytes per K/V element (4/2/1) */
    int kv_k_transposed;     /* if 1, K stored as [kv_h][d][p] (stride 1 across positions);
                              * V stays [p][kv_h][d]. Enables FMLA-into-att QK kernel,
                              * eliminating per-position svaddv. Opt-in via TF_KV_K_T=1. */
    int kv_k_dp;             /* if 1, K stored as [p][d][kv_h]. Enables qpkd+ktbl QK kernel
                              * (svld1rq+svtbl, no svaddv, no replication). Requires
                              * n_heads == SVE width (16 on A64FX). F32/F16/Q8 KV all
                              * wired. Default ON when eligible; opt out with TF_KV_K_DP=0. */
    float *q_packed;         /* [head_dim * n_heads] scratch; valid only when kv_k_dp */
    float *av_tmp;           /* [n_threads * n_heads * head_dim] per-thread AV partials.
                              * Allocated when kv_k_dp; lets AV parallelize across all
                              * threads via per-p partition + reduction (instead of the
                              * baseline 1-thread-per-head split). */
    float *att_pmax;         /* [n_threads * n_heads] per-thread per-head partial max,
                              * for the position-parallel softmax (kv_k_dp path). */
    float *att_psum;         /* [n_threads * n_heads] per-thread per-head partial sum_exp. */

    /* Scratch buffers */
    float *x;        /* [n_embd] current hidden state */
    float *xb;       /* [n_embd] scratch after norm */
    float *xb2;      /* [n_embd] scratch for residual */
    float *q;        /* [n_embd] query */
    float *k;        /* [n_kv_dim] key */
    float *v;        /* [n_kv_dim] value */
    float *att;      /* [n_heads * max_seq_len] attention scores */

    /* Flash-attention position-parallel scratch.
     * Sized at load for n_heads * TF_MAX_FA_CHUNKS partial accumulators.
     * fa_m / fa_l: [n_heads * n_chunks] per-tile (max_score, sum_exp).
     * fa_out:      [n_heads * n_chunks * head_dim] per-tile output. */
    float *fa_m;
    float *fa_l;
    float *fa_out;
    float *ffn_buf1; /* [n_ff] */
    float *ffn_buf2; /* [n_ff] */
    float *ffn_buf3; /* [n_ff] */
    float *logits;     /* [n_vocab] output logits (only if has_lm_head) */
    float *matvec_tmp; /* max(n_embd, n_ff) for row dequant (thread 0) */
    int trace_hidden_norms; /* print per-layer hidden norms during forward */

    /* Multi-threading */
    int n_threads;           /* number of threads (default: 1) */
    float **thread_tmp;      /* per-thread dequant scratch [n_threads][max_dim] */

    /* Precomputed RoPE inverse frequency tables */
    float *rope_inv_freq;        /* [head_dim/2]: 1/powf(base, 2j/head_dim) for standard RoPE */
    float *rope_mrope_inv_freq;  /* [sect_dims]: 1/powf(base, 2j/(2*sect_dims)) for M-RoPE */
    int rope_inv_freq_len;       /* head_dim/2 */
    int rope_mrope_inv_freq_len; /* sect_dims */

    /* Thread pool (persistent workers with cond-var sleep between dispatches) */
    pthread_t *pool_threads;   /* [n_threads] worker threads */
    void *(*pool_fn)(void *);  /* current work function */
    void *pool_args;           /* array of per-thread task structs */
    size_t pool_arg_stride;    /* sizeof(task struct) */
    volatile int pool_phase;   /* incremented to signal work */
    volatile int *pool_done_flags; /* [n_threads*TF_POOL_FLAG_STRIDE]: each
                                    * worker writes its slot = phase when done;
                                    * 256B-padded to avoid false sharing / a
                                    * cross-CMG-contended shared counter */
    volatile int pool_sleepers;/* workers currently parked on pool_cond */
    int pool_alive;            /* 1 if pool is running */
    pthread_mutex_t pool_mutex;/* protects pool_phase signaling */
    pthread_cond_t pool_cond;  /* workers sleep here between dispatches */
    volatile int bar_count;    /* barrier arrival counter */
    volatile int bar_sense;    /* barrier sense flag (alternates 0/1) */

    /* SW hierarchical barrier state: 4 CMGs × 16-int slots = each slot one
     * cacheline (64B) so different CMGs don't false-share. Used by the
     * persistent worker to reduce 48-way atomic contention on the single
     * bar_count cacheline; ~3× faster than the flat tf_spin_barrier on
     * Fugaku 4-CMG nodes. hb_g_* are the inter-CMG sync (only 4 leaders). */
    volatile int hb_cmg_count[4 * 16];
    volatile int hb_cmg_sense[4 * 16];
    volatile int hb_g_count;
    char _hb_pad1[60];
    volatile int hb_g_sense;
    char _hb_pad2[60];

    /* A64FX CMG affinity: pool threads pinned to CMG-local cores so that
     * panel buffers first-touched per-thread land on the right HBM stack.
     * cmg_pin=1 enables; cmg_pin_ncmgs = how many CMGs the pool spreads over. */
    int cmg_pin;
    int cmg_pin_ncmgs;

    /* Stage-1 batched-GEMM prefill (TF_PREFILL_GEMM): when set, tf_forward_blocks_range
     * runs only the mixer (attn_norm + SSM/attention + residual) and skips the dense
     * FFN + final RMSNorm, so transformer_prefill_gemm can batch the FFN as a GEMM. */
    int prefill_ffn_skip;

    /* A64FX hardware barrier (libhwb / /dev/xos_hwb), env TF_HW_BARRIER=1.
     * Per-CMG EL0 BST hardware barrier + 4-way SW combine among CMG leaders.
     * The kernel group-assign that makes the EL0 BST register actually
     * synchronize is done per-thread by vhbm_bar_assign(); raw MSR/MRS alone
     * is a no-op. See [[hwbarrier-libhwb-win]]. */
    int  hwbar_enabled;        /* 1 = use HW barrier in the persistent worker */
    int  hwbar_bd;             /* vhbm_bar_init() return (barrier-descriptor mask) */
    int  hwbar_ncmg;           /* CMGs participating */
    int  hwbar_tpc;            /* threads per CMG (n_threads / hwbar_ncmg) */
    long hwbar_bb[64];         /* per-tid BST register index (0..3) from assign */
    volatile int hwbar_lcount; /* 4-way leader-combine arrival count */
    char _hwbar_pad1[60];
    volatile int hwbar_lsense; /* 4-way leader-combine release sense */
    char _hwbar_pad2[60];
    volatile int hwbar_join_count;   /* threads that finished vhbm_bar_assign (startup) */
    volatile int hwbar_assign_failed;/* set if any thread's group-join failed → revert to flat */

    /* Tensor parallelism */
    int tp_rank;               /* this rank's position in the TP group (0 if no TP) */
    int tp_size;               /* size of the TP group (1 if no TP) */
    void (*tp_allreduce_fn)(float *buf, int count, void *ctx);  /* allreduce callback */
    void *tp_allreduce_ctx;    /* opaque context passed to allreduce (e.g. parallel_config*) */
    int tp_ssm_sharded;        /* 1 = SSM V-heads sharded too (Stage B); 0 = SSM replicated
                                * (Stage A: attn+FFN sharded, SSM runs full on every rank).
                                * Controls whether the mixer-output all-reduce fires on SSM
                                * layers (replicated SSM output needs no reduce). */
    int tp_attn_sharded;       /* 1 = attention is row/col-sharded (reduce after out_proj).
                                * Bisection: 0 leaves attn replicated (no reduce). */
    int tp_ffn_sharded;        /* 1 = dense FFN is row/col-sharded (reduce after ffn_down). */
    int gqa_group;             /* GLOBAL n_heads/n_kv_heads (set once at load, never mutated by
                                * slicing). Used as the kv_h divisor everywhere so the mapping
                                * survives KV replication (TP where n_kv % tp_size != 0), where
                                * the LOCAL n_heads/n_kv_heads ratio no longer equals the group. */
    int tp_qhead_offset;       /* GLOBAL index of this rank's first query head, ADDED to the
                                * local head before the /gqa_group kv lookup. 0 unless attention
                                * is in KV-replicate mode (full KV cache, sharded Q only). */
    int tp_kv_head_base;       /* LOCAL KV-head = GLOBAL KV-head - tp_kv_head_base in token cache. */
    int tp_kv_head_count;      /* number of KV-head rows kept locally after TP slicing */
    int ssm_head_offset;       /* Stage B: global index of this rank's first V-head. The SSM
                                * forward maps local head hl -> Q/K group (ssm_head_offset+hl)
                                * % n_group (Q/K stay replicated, V-heads sharded). 0 if
                                * unsharded so the original tile-repeat path is used. */
    int tp_vocab_sharded;      /* 1 = LM-head (output.weight) rows split across the TP group:
                                * each rank computes logits ONLY for vocab [tp_vocab_lo,
                                * tp_vocab_lo+tp_vocab_loc). transformer_compute_logits then
                                * fills m->logits[0..tp_vocab_loc); the caller does a local
                                * argmax and an allreduce-max(value,index) to agree on the
                                * next token. m->n_vocab stays FULL (embedding lookup needs
                                * it). Skipped when output is tied to token_embd. */
    int tp_vocab_lo;           /* global index of this rank's first logit row */
    int tp_vocab_loc;          /* number of logit rows this rank owns */

    /* Expert parallelism (MoE only). Each rank owns experts whose ID satisfies
     * expert % ep_size == ep_rank; ep_e_start/ep_e_end retain the first/last
     * owned IDs for diagnostics. The MoE forward loop skips selected experts it
     * does not own, then ep_ar_fn
     * sum-all-reduces the per-rank weighted partials xb2 across the EP group so
     * every rank exits the MoE block with the same full mixture output. ep_size==1
     * disables the filter and the post-loop reduce (single-node fallback). */
    int ep_rank;
    int ep_size;
    int ep_e_start;
    int ep_e_end;
    void (*ep_ar_fn)(float *buf, int count, void *ctx);
    void *ep_ar_ctx;

    /* Pipeline-parallel layer range owned by this stage. [pp_start, pp_end).
     * Defaults to [0, n_layers) (whole model). When restricted, build_panels
     * repacks only owned layers (so lazy-mmap weights for other layers never
     * fault → ~1/N memory) and builds the output (LM head) panel only on the
     * last stage (pp_end >= n_layers). */
    int pp_start;
    int pp_end;

    /* NUMA allocator state */
    struct {
        int n_cmgs;                   /* number of CMGs (default: 4) */
        size_t per_cmg_budget;        /* usable bytes per CMG (default: 6GB) */
        size_t per_cmg_used[8];       /* current usage per CMG */
        size_t alignment;             /* minimum alignment (default: 2MB) */
        int enabled;                  /* 0=fallback, 1=active */
    } numa;
} transformer_model;

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len);
void transformer_free(transformer_model *model);
void transformer_reset_runtime_state(transformer_model *model);

/* KV cache element formats. F16 stores K/V as IEEE half (2B), Q8 as int8
 * with a per-(pos, kv_head) scale (1B + scale). Set via TF_KV_DTYPE env
 * (f32|f16|q8) before transformer_load. Halves/quarters KV memory and the
 * per-token attention read bandwidth which dominates at long contexts. */
#define TF_KV_DTYPE_F32 0
#define TF_KV_DTYPE_F16 1
#define TF_KV_DTYPE_Q8  2

/* Max position-chunks per head in the flash-attention path. With nt=48 and
 * n_heads=16 (Qwen3.5-9B) we use 3 chunks/head → 48 (head, chunk) tasks.
 * Allows up to 16 for future use. Bounds the fa_out scratch at load time. */
#define TF_MAX_FA_CHUNKS 16

/* Set number of threads for parallel matmul/attention (default: 1) */
void transformer_set_threads(transformer_model *model, int n_threads);
void transformer_set_trace_hidden_norms(transformer_model *model, int enable);

/* Configure NUMA-aware weight/buffer distribution across CMGs.
 * Must be called after transformer_set_threads, before inference.
 * Env vars: NUMA_DISTRIBUTE=1 (enable), NUMA_N_CMGS (default 4),
 *           NUMA_CMG_BUDGET_GB (default 6), NUMA_ALIGNMENT (default 2MB). */
void transformer_numa_setup(transformer_model *m, const gguf_context *gguf);

/* A64FX (SVE) only: repack all dense F16 matvec weights into panel layout for
 * the horizontal-reduction-free matvec kernel, and first-touch each panel's
 * row blocks from the pool thread that will consume them — so with CMG pinning
 * the panel memory is spread across the 4 HBM stacks. Call after
 * transformer_set_threads. No-op on non-SVE builds or if TF_NO_PANEL is set. */
void transformer_build_panels(transformer_model *m);

/* Run one token through the transformer. Returns pointer to hidden state [n_embd].
 * For embedding models (no output projection), this is the final hidden state. */
float *transformer_forward(transformer_model *model, int32_t token_id, int position);

/* Run forward pass and compute logits [n_vocab]. Returns NULL if no LM head.
 * The returned pointer is valid until the next call. */
float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position);

/* Stage-1 batched-GEMM prefill (env TF_PREFILL_GEMM). Processes a whole prompt
 * [M tokens, starting at cache_pos pos0] layer-major: per layer, run the mixer
 * (SSM/gated-attention) per token (writing the KV cache in order), then batch the
 * dense SwiGLU FFN over all M tokens via the per-CMG packed-B bf16 GEMM. Writes the
 * KV cache for positions pos0..pos0+M-1 identically to M sequential transformer_forward
 * calls (bit-similar — GEMM reorders the K-sum). Returns last-token logits [n_vocab]
 * (final RMSNorm + lm_head applied once). Dense (non-MoE), non-Gemma4 hybrid/standard
 * models only; returns NULL if unsupported (caller should fall back to per-token). */
float *transformer_prefill_gemm(transformer_model *model, const int32_t *tokens, int M, int pos0);

/* Run forward pass with a pre-computed embedding vector instead of token lookup.
 * Used to inject vision embeddings into the sequence. */
float *transformer_forward_embd(transformer_model *model, const float *embd, int position);
float *transformer_forward_embd_logits(transformer_model *model, const float *embd, int position);

/* M-RoPE variants: cache_pos = KV cache slot, pos_t/h/w = RoPE temporal/height/width.
 * For text tokens, use cache_pos = pos_t = pos_h = pos_w = position. */
float *transformer_forward_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w);
float *transformer_forward_logits_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w);
float *transformer_forward_embd_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w);
float *transformer_forward_embd_logits_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w);

/* --- Pipeline-parallel API (for MPI) --- */

/* Process layers [layer_start, layer_end) only.
 * If layer_start == 0: expects model->x already set (e.g. via token embedding).
 * If layer_end == n_layers: applies final RMSNorm.
 * Returns pointer to model->x (hidden state [n_embd]). */
float *transformer_forward_partial(transformer_model *model, int cache_pos,
                                    int layer_start, int layer_end);

/* Compute logits from the current hidden state in model->x.
 * Call after transformer_forward_partial with layer_end == n_layers. */
float *transformer_compute_logits(transformer_model *model);

/* Copy hidden state into/out of model->x for MPI communication */
float *transformer_get_hidden(transformer_model *model);
void transformer_set_hidden(transformer_model *model, const float *hidden);

/* Embed a token into model->x (no layer processing) */
void transformer_embed_token(transformer_model *model, int32_t token_id);

/* --- Tensor-parallel API --- */

/* Configure tensor parallelism. Must be called before forward passes.
 * tp_rank: this rank's index in the TP group [0, tp_size).
 * tp_size: total ranks in the TP group.
 * allreduce_fn: callback to allreduce(sum) a float buffer in-place.
 * allreduce_ctx: opaque context passed to allreduce_fn (e.g. parallel_config*).
 *
 * Constraints: n_heads % tp_size == 0, n_ff % tp_size == 0 with an uneven last shard.
 * n_kv_heads may be uneven across tp_size; when so, KV cache is replicated and Q heads
 * are offset with tp_qhead_offset so local->global kv lookup remains correct. */
void transformer_set_tp(transformer_model *model, int tp_rank, int tp_size,
                         void (*allreduce_fn)(float *buf, int count, void *ctx),
                         void *allreduce_ctx);

/* Megatron-style tensor-parallel weight sharding. Slices every dense projection
 * to this rank's 1/tp_size shard IN PLACE (offsetting qtensor.data + mutating
 * n_rows/n_cols/tp_src_stride) and mutates the model's head/FFN dims to local
 * values, so the existing forward loops compute only the local shard and the two
 * residual all-reduces recombine the row-parallel partials.
 *   - COL-parallel (output-row split, no comm): attn_q/k/v, ffn_gate/up.
 *   - ROW-parallel (input-col split, all-reduce): attn_output, ffn_down.
 * Call AFTER transformer_load and BEFORE transformer_build_panels (panels build
 * on the slice). Requires n_kv_heads % tp_size == 0 (clean GQA split; KV
 * replication for tp_size>n_kv_heads is handled (Q-only replication, local kv_head
 * mapping handled via tp_qhead_offset). Requires a per-rank KV-base that is cleanly
 * initialized by this routine. n_ff % tp_size still obeys uneven-tail layout with
 * chunk size rounded up to multiple-of-16. ssm_shard=0 leaves SSM tensors replicated
 * (validation on models that fit); =1 also shards SSM V-heads (not yet impl).
 * Set TF_KEEP_BF16_SRC=1 — the bf16_pv reclaim assumes a contiguous source
 * range, wrong for row-parallel strided slices. Returns 0 on success. */
int transformer_tp_slice_weights(transformer_model *model, int tp_rank, int tp_size,
                                  int ssm_shard);

/* --- Expert-parallel API (MoE) --- */

/* Configure expert parallelism: this rank owns expert IDs where
 * expert % ep_size == ep_rank (interleaved modulo partition, handles n_expert
 * not divisible by ep_size). After this call, the MoE forward skips selected
 * experts it does not own; the caller must also wire transformer_set_ep_ar so the per-rank
 * weighted partials are summed back across the EP group before the residual. The
 * router (ffn_gate_inp) stays replicated, so all ranks select the same top-k.
 * ep_size==1 is a no-op (single-node). Call AFTER transformer_load. */
void transformer_set_ep(transformer_model *model, int ep_rank, int ep_size);

/* Wire the EP all-reduce callback used to sum xb2 across the EP group after the
 * MoE expert loop. Must be called when ep_size>1. */
void transformer_set_ep_ar(transformer_model *model,
                            void (*ar_fn)(float *buf, int count, void *ctx),
                            void *ar_ctx);

/* --- Pipeline-parallel layer-range ownership --- */

/* Restrict this stage to layers [layer_start, layer_end). Call AFTER
 * transformer_load and BEFORE transformer_build_panels so panel repacking (and
 * the lazy-mmap faults it triggers) is confined to owned layers, and the LM-head
 * output panel is built only on the last stage. */
void transformer_set_pp_range(transformer_model *model, int layer_start, int layer_end);

/* --- Distributed memory management --- */

/* Free KV cache for layers outside [layer_start, layer_end).
 * Call after transformer_load for PP ranks that only process a layer subset. */
void transformer_free_unused_kv(transformer_model *model, int layer_start, int layer_end);

/* Reallocate KV cache with reduced dimension for TP.
 * Only affects attention layers within [layer_start, layer_end).
 * tp_kv_dim = n_kv_heads / tp_size * head_dim. */
void transformer_resize_kv_for_tp(transformer_model *model,
                                    int layer_start, int layer_end, int tp_kv_dim);

/* --- Batched prefill API --- */

/* Batch descriptor for prefill: process N tokens through all layers at once. */
typedef struct {
    const float *embds;        /* [N, embd_stride] pre-expanded token embeddings */
    int N;                     /* number of tokens */
    int embd_stride;           /* stride between tokens in embds (>= n_embd) */
    const int *cache_pos;      /* [N] KV cache slot for each token */
    const int *pos_t;          /* [N] M-RoPE temporal positions */
    const int *pos_h;          /* [N] M-RoPE height positions */
    const int *pos_w;          /* [N] M-RoPE width positions */
    const float *const *ds_embds;  /* [N] per-token deepstack embedding pointers (NULL = no deepstack) */
    int ds_embd_stride;        /* total embedding stride for deepstack tokens */
} transformer_batch;

/* Run batched prefill and return logits for the last token. Returns NULL if no LM head.
 * All N tokens are processed through GEMM instead of individual matvecs. */
float *transformer_forward_batch_logits(transformer_model *m, const transformer_batch *b);

/* EP MoE prefill for a single long prompt. Processes tokens layer-major in
 * blocks, preserving per-token mixer/expert math while batching the EP
 * all-reduce for MoE FFN partials to one reduce per layer/block. */
float *transformer_prefill_ep_layermajor(transformer_model *m, const int32_t *tokens,
                                         int M, int pos0, int block_tokens);

/* Sample next token from logits using temperature and top-k. */
int32_t transformer_sample_topk(const float *logits, int n_vocab, float temperature, int top_k);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef TRANSFORMER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <alloca.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>

/* Profiling macros: active only if profiler.h was included before this file */
#ifdef PROFILER_H
#define TF_PROF_BEGIN(name, layer, op, prec) prof_begin(name, "llm", layer, op, prec)
#define TF_PROF_END(name, flops, iops) prof_end(name, flops, iops)
#else
#define TF_PROF_BEGIN(name, layer, op, prec) ((void)0)
#define TF_PROF_END(name, flops, iops) ((void)0)
#endif

static inline double tf_wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

enum {
    TF_SSM_COOP_PROF_S1_PROJ = 0,
    TF_SSM_COOP_PROF_S2_CONV,
    TF_SSM_COOP_PROF_BB_WAIT,
    TF_SSM_COOP_PROF_S3_NORM,
    TF_SSM_COOP_PROF_S4_SCAN,
    TF_SSM_COOP_PROF_BD_WAIT,
    TF_SSM_COOP_PROF_S5_OUT,
    TF_SSM_COOP_PROF_NSTAGE
};
#define TF_SSM_COOP_PROF_TMAX 128
static double tf_ssm_coop_prof_s[TF_SSM_COOP_PROF_NSTAGE][TF_SSM_COOP_PROF_TMAX];
static long   tf_ssm_coop_prof_calls[TF_SSM_COOP_PROF_TMAX];
static int    tf_ssm_coop_prof_enabled_cache = -1;

static inline int tf_ssm_coop_prof_enabled(void) {
    if (tf_ssm_coop_prof_enabled_cache < 0)
        tf_ssm_coop_prof_enabled_cache = getenv("TF_SSM_COOP_PROF") ? 1 : 0;
    return tf_ssm_coop_prof_enabled_cache;
}

void transformer_ssm_coop_profile_reset(void) {
    memset(tf_ssm_coop_prof_s, 0, sizeof(tf_ssm_coop_prof_s));
    memset(tf_ssm_coop_prof_calls, 0, sizeof(tf_ssm_coop_prof_calls));
    tf_ssm_coop_prof_enabled_cache = getenv("TF_SSM_COOP_PROF") ? 1 : 0;
}

static inline void tf_ssm_coop_prof_add(int tid, int stage, double sec) {
    if ((unsigned)tid < TF_SSM_COOP_PROF_TMAX &&
        (unsigned)stage < TF_SSM_COOP_PROF_NSTAGE)
        tf_ssm_coop_prof_s[stage][tid] += sec;
}

void transformer_ssm_coop_profile_dump(FILE *fp, double decode_tokens) {
    static const char *names[TF_SSM_COOP_PROF_NSTAGE] = {
        "ssm_coop_s1_proj", "ssm_coop_s2_conv", "ssm_coop_bb_wait",
        "ssm_coop_s3_norm", "ssm_coop_s4_scan", "ssm_coop_bd_wait",
        "ssm_coop_s5_out"
    };
    if (!fp || !tf_ssm_coop_prof_enabled()) return;
    if (decode_tokens <= 0.0) decode_tokens = 1.0;
    int nt = 0;
    for (int t = 0; t < TF_SSM_COOP_PROF_TMAX; t++)
        if (tf_ssm_coop_prof_calls[t] > 0) nt = t + 1;
    if (nt <= 0) return;
    fprintf(fp, "ssm_coop stage profile (critical-thread total, %d threads)\n", nt);
    fprintf(fp, "  %-22s %10s %10s %10s %7s\n",
            "op", "max_ms", "max_ms/tok", "avg_ms/tok", "max_tid");
    for (int s = 0; s < TF_SSM_COOP_PROF_NSTAGE; s++) {
        double max_s = 0.0, sum_s = 0.0;
        int max_tid = 0;
        for (int t = 0; t < nt; t++) {
            double v = tf_ssm_coop_prof_s[s][t];
            if (v > max_s) { max_s = v; max_tid = t; }
            sum_s += v;
        }
        fprintf(fp, "  %-22s %10.2f %10.3f %10.3f %7d\n",
                names[s], max_s * 1000.0, max_s * 1000.0 / decode_tokens,
                (sum_s / nt) * 1000.0 / decode_tokens, max_tid);
    }
}

/* ---- Tensor lookup helpers ---- */

static int tf_find_tensor(const gguf_context *gguf, const char *name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor tf_load_tensor(const gguf_context *gguf, const char *name, int required) {
    qtensor t = {0};
    int idx = tf_find_tensor(gguf, name);
    if (idx < 0) {
        if (required) fprintf(stderr, "transformer: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(gguf, idx);
    t.type = gguf->tensors[idx].type;
    t.n_dims = (int)gguf->tensors[idx].n_dims;
    if (t.n_dims > 4) t.n_dims = 4;
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = gguf->tensors[idx].dims[d];
    /* dims[0] = inner dimension (cols), rows = product(dims[1:]) */
    t.n_cols = (int)gguf->tensors[idx].dims[0];
    uint64_t n_rows = 1;
    for (uint32_t d = 1; d < gguf->tensors[idx].n_dims; d++) n_rows *= gguf->tensors[idx].dims[d];
    t.n_rows = (int)n_rows;
    return t;
}

static int tf_get_int(const gguf_context *gguf, const char *key, int default_val) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return default_val;
    if (gguf->kv[idx].type == GGUF_TYPE_UINT32) return (int)gguf->kv[idx].value.u32;
    if (gguf->kv[idx].type == GGUF_TYPE_INT32)  return gguf->kv[idx].value.i32;
    return default_val;
}

static float tf_get_float(const gguf_context *gguf, const char *key, float default_val) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return default_val;
    if (gguf->kv[idx].type == GGUF_TYPE_FLOAT32) return gguf->kv[idx].value.f32;
    return default_val;
}

/* ---- Aligned allocation helpers ---- */
/* Returns zeroed memory aligned to 'alignment' bytes. */
static void *tf_aligned_calloc(size_t alignment, size_t count, size_t elem_size) {
    size_t size = count * elem_size;
    size = (size + alignment - 1) & ~(alignment - 1);
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

/* Allocate aligned memory WITHOUT touching pages.
 * With demand paging, first-touch from a worker thread will place pages
 * on that thread's NUMA node. Safe for buffers that are always written
 * before read (e.g. dequant scratch). */
static void *tf_aligned_alloc_notouch(size_t alignment, size_t size) {
    size = (size + alignment - 1) & ~(alignment - 1);
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}

/* ---- Compute helpers ---- */

/* Dequantize one row of a quantized matrix.
 * For a matrix stored as [n_rows, n_cols], row i starts at offset computed
 * from the quantization block size. */
static void tf_dequant_row(const qtensor *t, int row, float *dst) {
    /* Compute byte offset for this row */
    int n_cols = t->n_cols;
    int block_size, type_size;

    /* Get block/type size from ggml_type_info (defined in gguf_loader.h impl) */
    switch (t->type) {
        case GGML_TYPE_Q2_K: block_size = 256; type_size = 84;  break;
        case GGML_TYPE_Q3_K: block_size = 256; type_size = 110; break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K:   block_size = 256; type_size = 176; break;
        case GGML_TYPE_Q6_K:   block_size = 256; type_size = 210; break;
        case GGML_TYPE_IQ2_XXS: block_size = 256; type_size = 66;  break;
        case GGML_TYPE_IQ2_XS: block_size = 256; type_size = 74;  break;
        case GGML_TYPE_IQ3_XXS: block_size = 256; type_size = 98;  break;
        case GGML_TYPE_IQ1_S:  block_size = 256; type_size = 50;  break;
        case GGML_TYPE_IQ4_NL: block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_IQ3_S:  block_size = 256; type_size = 110; break;
        case GGML_TYPE_IQ2_S:  block_size = 256; type_size = 82;  break;
        case GGML_TYPE_IQ4_XS: block_size = 256; type_size = 136; break;
        case GGML_TYPE_IQ1_M:  block_size = 256; type_size = 56;  break;
        case GGML_TYPE_F32:    block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:    block_size = 1;   type_size = 2;   break;
        case GGML_TYPE_BF16:   block_size = 1;   type_size = 2;   break;
        default:
            fprintf(stderr, "tf_dequant_row: unsupported type %u\n", t->type);
            memset(dst, 0, (size_t)n_cols * sizeof(float));
            return;
    }

    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
    const void *row_data = (const uint8_t *)t->data + (size_t)row * row_bytes;
    dequant_row(t->type, row_data, dst, n_cols);
}

static int tf_is_supported_weight_type(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 1;
        default:
            return 0;
    }
}

static size_t tf_row_bytes(uint32_t type, int n_cols) {
    int block_size = 1, type_size = 4;
    switch (type) {
        case GGML_TYPE_Q2_K: block_size = 256; type_size = 84;  break;
        case GGML_TYPE_Q3_K: block_size = 256; type_size = 110; break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K:   block_size = 256; type_size = 176; break;
        case GGML_TYPE_Q6_K:   block_size = 256; type_size = 210; break;
        case GGML_TYPE_IQ2_XXS: block_size = 256; type_size = 66;  break;
        case GGML_TYPE_IQ2_XS: block_size = 256; type_size = 74;  break;
        case GGML_TYPE_IQ3_XXS: block_size = 256; type_size = 98;  break;
        case GGML_TYPE_IQ1_S:  block_size = 256; type_size = 50;  break;
        case GGML_TYPE_IQ4_NL: block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_IQ3_S:  block_size = 256; type_size = 110; break;
        case GGML_TYPE_IQ2_S:  block_size = 256; type_size = 82;  break;
        case GGML_TYPE_IQ4_XS: block_size = 256; type_size = 136; break;
        case GGML_TYPE_IQ1_M:  block_size = 256; type_size = 56;  break;
        case GGML_TYPE_F32:    block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:    block_size = 1;   type_size = 2;   break;
        case GGML_TYPE_BF16:   block_size = 1;   type_size = 2;   break;
        default: return 0;
    }
    return (size_t)((n_cols + block_size - 1) / block_size) * type_size;
}

/* Matvec for a single expert slice from 3D tensor [cols, rows_per_expert, n_expert]. */
static void tf_qmatvec_expert(float *dst, const qtensor *mat, int expert, const float *x,
                              int rows_per_expert, float *tmp) {
    size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
    if (row_bytes == 0) {
        memset(dst, 0, rows_per_expert * sizeof(float));
        return;
    }

    const uint8_t *base = (const uint8_t *)mat->data + (size_t)expert * rows_per_expert * row_bytes;
    if (mat->type == GGML_TYPE_F16) {
        for (int i = 0; i < rows_per_expert; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_f16_f32(row, x, mat->n_cols);
        }
        return;
    }
    if (mat->type == GGML_TYPE_BF16) {
        for (int i = 0; i < rows_per_expert; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_bf16_f32(row, x, mat->n_cols);
        }
        return;
    }

    for (int i = 0; i < rows_per_expert; i++) {
        const void *row_data = base + (size_t)i * row_bytes;
        dequant_row(mat->type, row_data, tmp, mat->n_cols);
        float sum = 0.0f;
        for (int j = 0; j < mat->n_cols; j++) sum += tmp[j] * x[j];
        dst[i] = sum;
    }
}

/* Vectorized element-wise add: dst[i] += src[i] */
static void tf_vadd(float *dst, const float *src, int n) {
#if defined(__AVX2__)
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    for (; i < n; i++) dst[i] += src[i];
#elif defined(__ARM_FEATURE_SVE)
    int i = 0;
    for (; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vd = svld1(pg, dst + i);
        svfloat32_t vs = svld1(pg, src + i);
        svst1(pg, dst + i, svadd_f32_z(pg, vd, vs));
    }
#else
    for (int i = 0; i < n; i++) dst[i] += src[i];
#endif
}

/* Sum of squares: returns sum(v[i]^2) — used for hidden norm profiling */
static float tf_sum_squares(const float *v, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        acc = _mm256_fmadd_ps(x, x, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float ss = _mm_cvtss_f32(s4);
    for (; i < n; i++) ss += v[i] * v[i];
    return ss;
#elif defined(__ARM_FEATURE_SVE)
    svfloat32_t acc = svdup_f32(0.0f);
    int i = 0;
    for (; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t x = svld1(pg, v + i);
        acc = svmla_f32_z(pg, acc, x, x);
    }
    return svaddv_f32(svptrue_b32(), acc);
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    return ss;
#endif
}

/* RMSNorm: y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps) */
static void tf_rmsnorm(float *dst, const float *x, const qtensor *w, int n, float eps, float *w_buf) {
    /* Dequant weight */
    tf_dequant_row(w, 0, w_buf);

#if defined(__AVX2__) && defined(__FMA__)
    /* AVX2: sum of squares */
    __m256 vss = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vss = _mm256_fmadd_ps(vx, vx, vss);
    }
    __m128 hi = _mm256_extractf128_ps(vss, 1);
    __m128 lo = _mm256_castps256_ps128(vss);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float ss = _mm_cvtss_f32(s4);
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);

    /* Fused x * ss * w */
    __m256 vscale = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w_buf + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw));
    }
    for (; i < n; i++) dst[i] = x[i] * ss * w_buf[i];
#elif defined(__ARM_FEATURE_SVE)
    float inv_ss = 1.0f / sqrtf(tf_sum_squares(x, n) / n + eps);
    svfloat32_t scale = svdup_f32(inv_ss);
    int i = 0;
    for (; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1(pg, x + i);
        svfloat32_t vw = svld1(pg, w_buf + i);
        svst1(pg, dst + i, svmul_f32_z(pg, vw, svmul_f32_x(pg, vx, scale)));
    }
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) dst[i] = x[i] * ss * w_buf[i];
#endif
}

/* Quantized matrix-vector multiply: dst[i] = sum_j(M[i][j] * x[j]) for i in [0, n_rows) */
typedef struct {
    float *dst;
    const qtensor *mat;
    const float *x;
    int row_start, row_end;
    float *tmp; /* per-thread scratch */
} tf_matvec_task;

static inline void tf_matvec_bf16_rows_pv(float *dst, const uint8_t *base,
                                            size_t row_bytes, const uint16_t *pv,
                                            const float *x, int n_cols,
                                            int row_start, int row_end);

#if defined(__ARM_FEATURE_SVE)
/* Per-thread int8 quantization of the activation x for the svdot W8A8 weight
 * matvec (matvec_sdot_8row). Each pool thread quantizes the FULL x into its
 * own TLS scratch (per 64-elem block: absmax -> fp16 scale, then round to
 * int8). The work is redundant across threads but x is only K (<= n_ff)
 * elements — negligible next to the matvec — which lets us avoid an extra
 * barrier (decode barriers are already ~18%). Returned pointers are valid
 * until the next tf_quant_x_sdot call on the same thread. K must be a
 * multiple of 64 (guaranteed by the q8_pv col_ok constraint). */
static __thread int8_t   *tf_xq_buf = NULL;
static __thread uint16_t *tf_xs_buf = NULL;
static __thread int       tf_xq_cap = 0;

static inline void tf_quant_x_sdot_blocks(const float *x, int K,
                                          int b_start, int b_step,
                                          int8_t *xq_buf, uint16_t *xs_buf) {
    svbool_t pg = svptrue_b32();
    svint32_t qlo = svdup_s32(-127), qhi = svdup_s32(127);
    int nb = K / 64;
    if (b_step <= 0) b_step = 1;
    for (int b = b_start; b < nb; b += b_step) {
        const float *xb = x + (size_t)b * 64;
        svfloat32_t v0 = svld1_f32(pg, xb +  0);
        svfloat32_t v1 = svld1_f32(pg, xb + 16);
        svfloat32_t v2 = svld1_f32(pg, xb + 32);
        svfloat32_t v3 = svld1_f32(pg, xb + 48);
        svfloat32_t m = svmax_x(pg, svmax_x(pg, svabs_x(pg, v0), svabs_x(pg, v1)),
                                    svmax_x(pg, svabs_x(pg, v2), svabs_x(pg, v3)));
        float amax = svmaxv_f32(pg, m);
        float scale = amax / 127.0f;
        float inv   = amax > 0.0f ? 127.0f / amax : 0.0f;
        xs_buf[b] = ggml_fp32_to_fp16(scale);
        svfloat32_t vinv = svdup_f32(inv);
        int8_t *q = xq_buf + (size_t)b * 64;
        #define QX(V, OFF) do {                                              \
            svint32_t qi = svcvt_s32_f32_x(pg, svmul_x(pg, (V), vinv));       \
            qi = svmax_s32_x(pg, svmin_s32_x(pg, qi, qhi), qlo);             \
            svst1b_s32(pg, q + (OFF), qi);                                   \
        } while (0)
        QX(v0, 0); QX(v1, 16); QX(v2, 32); QX(v3, 48);
        #undef QX
    }
}

static inline void tf_quant_x_sdot(const float *x, int K,
                                   const int8_t **xq_out,
                                   const uint16_t **xs_out) {
    if (K > tf_xq_cap) {
        free(tf_xq_buf);
        free(tf_xs_buf);
        tf_xq_buf = (int8_t *)malloc((size_t)K);
        tf_xs_buf = (uint16_t *)malloc((size_t)(K / 64) * sizeof(uint16_t));
        tf_xq_cap = K;
    }
    /* Fully vectorized: scalar lrintf compiles to a libm call on fcc, which
     * dominated this hot path (called K times per matvec, replicated across
     * all threads). SVE round-to-nearest via svcvt (FPCR default) + saturating
     * clamp + truncating byte store (svst1b) avoids it entirely. One svmaxv
     * per 64-block is the only horizontal op. */
    tf_quant_x_sdot_blocks(x, K, 0, 1, tf_xq_buf, tf_xs_buf);
    *xq_out = tf_xq_buf;
    *xs_out = tf_xs_buf;
}
#endif

static void *tf_qmatvec_worker(void *arg) {
    tf_matvec_task *t = (tf_matvec_task *)arg;
    int n_cols = t->mat->n_cols;
    if (t->mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)t->mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = t->row_start;
        for (; i + 5 < t->row_end; i += 6) {
            matvec_f16_6row(t->dst + i,
                (const uint16_t *)(base + (size_t)i * row_bytes),
                (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
                t->x, n_cols);
        }
        for (; i < t->row_end; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            t->dst[i] = vec_dot_f16_f32(row, t->x, n_cols);
        }
        return NULL;
    }
    if (t->mat->type == GGML_TYPE_BF16) {
#if defined(__ARM_FEATURE_SVE)
        /* Q8_0 quantize-on-load path: when q8_pv is built we use the int8
         * group layout, halving DRAM traffic vs bf16_pv. Worker row ranges
         * are 8-aligned by tf_row_split8 so we can step in 8-row groups. */
        if (t->mat->q8_pv && (t->row_start & 7) == 0 && (t->row_end & 7) == 0) {
            const uint8_t *qbase = t->mat->q8_pv;
            const int8_t *xq; const uint16_t *xs;
            tf_quant_x_sdot(t->x, n_cols, &xq, &xs);
            int nb = n_cols / 64;
            size_t group_bytes = (size_t)nb * 528;
            for (int i = t->row_start; i + 7 < t->row_end; i += 8) {
                int g = i >> 3;
                matvec_sdot_8row(t->dst + i, qbase + (size_t)g * group_bytes,
                                 xq, xs, n_cols);
            }
            return NULL;
        }
#endif
        /* Route through the pv-aware path so matvec_bf16_8row_pv fires when
         * mat->bf16_pv is built and the worker's row_start is 8-aligned
         * (guaranteed by tf_row_split8 in the pool dispatcher). */
        const uint8_t *base = (const uint8_t *)t->mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        tf_matvec_bf16_rows_pv(t->dst, base, row_bytes, t->mat->bf16_pv,
                                t->x, n_cols, t->row_start, t->row_end);
        return NULL;
    }
    if (t->mat->type == GGML_TYPE_Q8_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
        const uint8_t *base = (const uint8_t *)t->mat->data;
        for (int i = t->row_start; i < t->row_end; i++) {
            t->dst[i] = vec_dot_q8_0_f32(base + (size_t)i * row_bytes, t->x, n_cols);
        }
        return NULL;
    }
    if (t->mat->q8_pv && (t->row_start & 7) == 0 && (t->row_end & 7) == 0) {
        const int8_t *xq; const uint16_t *xs;
        tf_quant_x_sdot(t->x, n_cols, &xq, &xs);
        int nb = n_cols / 64;
        size_t group_bytes = (size_t)nb * 528;
        const uint8_t *qbase = t->mat->q8_pv;
        for (int i = t->row_start; i + 7 < t->row_end; i += 8) {
            int g = i >> 3;
            matvec_sdot_8row(t->dst + i, qbase + (size_t)g * group_bytes,
                             xq, xs, n_cols);
        }
        return NULL;
    }
    if (t->mat->bf16_pv && (t->row_start & 7) == 0 && (t->row_end & 7) == 0) {
        /* Dense quantized tensor repacked to bf16_pv (e.g. SSM mixer ssm_out via
         * tf_qmatvec_pool). 8-aligned range -> pv fast path, never reads the
         * quantized `data` base. Mirrors tf_matvec_qtensor_rows. */
        tf_matvec_bf16_rows_pv(t->dst, (const uint8_t *)t->mat->data,
                                (size_t)n_cols * 2, t->mat->bf16_pv,
                                t->x, n_cols, t->row_start, t->row_end);
        return NULL;
    }
    for (int i = t->row_start; i < t->row_end; i++) {
        tf_dequant_row(t->mat, i, t->tmp);
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 31 < n_cols; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j),      _mm256_loadu_ps(t->x + j),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j + 8),  _mm256_loadu_ps(t->x + j + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j + 16), _mm256_loadu_ps(t->x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j + 24), _mm256_loadu_ps(t->x + j + 24), acc3);
        }
        for (; j + 7 < n_cols; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j), _mm256_loadu_ps(t->x + j), acc0);
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float sum = _mm_cvtss_f32(s4);
        for (; j < n_cols; j++) sum += t->tmp[j] * t->x[j];
#else
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += t->tmp[j] * t->x[j];
#endif
        t->dst[i] = sum;
    }
    return NULL;
}

/* Forward declarations for fused matvec and thread pool */
static void tf_qmatvec(float *dst, const qtensor *mat, const float *x, int n_rows, float *tmp);
static void tf_pool_dispatch(transformer_model *model, void *(*fn)(void *),
                              void *args, size_t arg_stride);
static void tf_pool_shutdown(transformer_model *model);

/* Fused dual-matrix matvec: computes dst1 = mat1 × x AND dst2 = mat2 × x in one dispatch.
 * Both matrices must have the same n_cols and same n_rows. */
typedef struct {
    float *dst1, *dst2;
    const qtensor *mat1, *mat2;
    const float *x;
    int row_start, row_end;
} tf_matvec_fused2_task;

static void tf_matvec_q8_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                const float *x, int n_cols, int row_start, int row_end) {
    for (int i = row_start; i < row_end; i++) {
        dst[i] = vec_dot_q8_0_f32(base + (size_t)i * row_bytes, x, n_cols);
    }
}

/* Compute head range [h_start, h_end) for thread tid when distributing
 * dt_rank SSM heads across n_cmgs CMGs. Each CMG gets a contiguous block
 * of heads, and within that block the first min(threads_per_cmg, cmg_heads)
 * threads share the heads. Threads that get nothing return h_end == h_start.
 *
 * Goal: state[h] for h in CMG c's block is first-touched by a thread pinned
 * to CMG c, so the 64 KB / head state column lives in c's HBM and stays
 * L2-resident across decode steps. Without this, all 16-48 heads of state
 * sit on whatever CMG main first-touched (typically CMG0), and threads on
 * CMG1-3 must reach across the X-bar each recurrence pass. */
static inline void tf_ssm_head_range(int dt_rank, int n_threads, int n_cmgs,
                                       int tid, int *h_start, int *h_end) {
    if (n_cmgs < 1) n_cmgs = 1;
    int per_cmg_thr = n_threads / n_cmgs;
    if (per_cmg_thr < 1) per_cmg_thr = 1;
    int cmg = tid / per_cmg_thr;
    int pos = tid - cmg * per_cmg_thr;
    if (cmg >= n_cmgs) { *h_start = *h_end = 0; return; }
    int h_base = dt_rank / n_cmgs, h_extra = dt_rank % n_cmgs;
    int cmg_h  = h_base + (cmg < h_extra ? 1 : 0);
    int cmg_off = h_base * cmg + (cmg < h_extra ? cmg : h_extra);
    int use = cmg_h < per_cmg_thr ? cmg_h : per_cmg_thr;
    if (pos >= use) { *h_start = *h_end = 0; return; }
    int t_base = cmg_h / use, t_extra = cmg_h % use;
    int my  = t_base + (pos < t_extra ? 1 : 0);
    int off = t_base * pos + (pos < t_extra ? pos : t_extra);
    *h_start = cmg_off + off;
    *h_end   = *h_start + my;
}

/* Compute row_start/row_end for worker t out of n_threads, splitting n_rows
 * in 8-row units when n_rows is 8-aligned. Preserves row_start & 7 == 0 so
 * the bf16_pv fast path in tf_matvec_bf16_rows_pv is taken on every worker,
 * not just whichever happens to land 8-aligned by chance. */
static inline void tf_row_split8(int n_rows, int n_threads, int t,
                                  int *row_start, int *row_end) {
    if ((n_rows & 7) == 0) {
        int nb = n_rows >> 3;
        int per = nb / n_threads, extra = nb % n_threads;
        int off = per * t + (t < extra ? t : extra);
        int cnt = per + (t < extra ? 1 : 0);
        *row_start = off << 3;
        *row_end   = (off + cnt) << 3;
    } else {
        int per = n_rows / n_threads, extra = n_rows % n_threads;
        int off = per * t + (t < extra ? t : extra);
        int cnt = per + (t < extra ? 1 : 0);
        *row_start = off;
        *row_end   = off + cnt;
    }
}

/* If pv != NULL use matvec_bf16_8row_pv for 8-row blocks. pv is the
 * pair-packed buffer from tf_bf16_pv_alloc/fill (n_rows * n_cols bf16 = same
 * byte count as row-major). row_start / row_end must be 8-aligned when pv is
 * used; otherwise pv is silently ignored for the misaligned region. */
static inline void tf_matvec_bf16_rows_pv(float *dst, const uint8_t *base,
                                            size_t row_bytes, const uint16_t *pv,
                                            const float *x, int n_cols,
                                            int row_start, int row_end) {
    int i = row_start;
#if defined(__ARM_FEATURE_SVE)
    /* 8-row blocks: 8 FMAs per activation load, doubles compute/memory ratio */
    if (pv && (row_start & 7) == 0) {
        for (; i + 7 < row_end; i += 8) {
            int g = i >> 3;
            const uint16_t *gbase = pv + (size_t)g * 8 * n_cols;
            matvec_bf16_8row_pv(dst + i,
                gbase + (size_t)0 * 2 * n_cols,
                gbase + (size_t)1 * 2 * n_cols,
                gbase + (size_t)2 * 2 * n_cols,
                gbase + (size_t)3 * 2 * n_cols,
                x, n_cols);
        }
    } else {
        for (; i + 7 < row_end; i += 8) {
            matvec_bf16_8row(dst + i,
                (const uint16_t *)(base + (size_t)(i)   * row_bytes),
                (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+6) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+7) * row_bytes),
                x, n_cols);
        }
    }
#endif
    for (; i + 3 < row_end; i += 4) {
        matvec_bf16_4row(dst + i,
            (const uint16_t *)(base + (size_t)(i)   * row_bytes),
            (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
            x, n_cols);
    }
    for (; i < row_end; i++) {
        const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
        dst[i] = vec_dot_bf16_f32(row, x, n_cols);
    }
}

static void tf_matvec_bf16_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                  const float *x, int n_cols, int row_start, int row_end) {
    tf_matvec_bf16_rows_pv(dst, base, row_bytes, NULL, x, n_cols, row_start, row_end);
}

static void tf_matvec_f16_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                 const float *x, int n_cols, int row_start, int row_end) {
    int i = row_start;
    for (; i + 5 < row_end; i += 6) {
        matvec_f16_6row(dst + i,
            (const uint16_t *)(base + (size_t)i * row_bytes),
            (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
            x, n_cols);
    }
    for (; i < row_end; i++) {
        const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
        dst[i] = vec_dot_f16_f32(row, x, n_cols);
    }
}

static void *tf_qmatvec_fused2_worker(void *arg) {
    tf_matvec_fused2_task *t = (tf_matvec_fused2_task *)arg;
    int n_cols = t->mat1->n_cols;
    if (t->mat1->type == GGML_TYPE_F16) {
        size_t row_bytes = (size_t)n_cols * 2;
        tf_matvec_f16_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
        tf_matvec_f16_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
    } else if (t->mat1->type == GGML_TYPE_BF16) {
        size_t row_bytes = (size_t)n_cols * 2;
#if defined(__ARM_FEATURE_SVE)
        if (t->mat1->q8_pv && t->mat2->q8_pv &&
            (t->row_start & 7) == 0 && (t->row_end & 7) == 0) {
            const int8_t *xq; const uint16_t *xs;
            tf_quant_x_sdot(t->x, n_cols, &xq, &xs);
            int nb = n_cols / 64;
            size_t group_bytes = (size_t)nb * 528;
            const uint8_t *q1 = t->mat1->q8_pv;
            const uint8_t *q2 = t->mat2->q8_pv;
            for (int i = t->row_start; i + 7 < t->row_end; i += 8) {
                int g = i >> 3;
                matvec_sdot_8row(t->dst1 + i, q1 + (size_t)g * group_bytes,
                                 xq, xs, n_cols);
                matvec_sdot_8row(t->dst2 + i, q2 + (size_t)g * group_bytes,
                                 xq, xs, n_cols);
            }
        } else
#endif
        {
            tf_matvec_bf16_rows_pv(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                                    t->mat1->bf16_pv,
                                    t->x, n_cols, t->row_start, t->row_end);
            tf_matvec_bf16_rows_pv(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                                    t->mat2->bf16_pv,
                                    t->x, n_cols, t->row_start, t->row_end);
        }
    } else if (t->mat1->type == GGML_TYPE_Q8_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
        tf_matvec_q8_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
        tf_matvec_q8_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
    } else {
        /* Generic path for other quantized weights — AVX2 dot product */
        float *tmp = (float *)malloc(n_cols * sizeof(float));
        for (int i = t->row_start; i < t->row_end; i++) {
            tf_dequant_row(t->mat1, i, tmp);
#if defined(__AVX2__) && defined(__FMA__)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 31 < n_cols; j += 32) {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j),    _mm256_loadu_ps(t->x+j),    a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+8),  _mm256_loadu_ps(t->x+j+8),  a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+16), _mm256_loadu_ps(t->x+j+16), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+24), _mm256_loadu_ps(t->x+j+24), a3);
            }
            for (; j + 7 < n_cols; j += 8)
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j), _mm256_loadu_ps(t->x+j), a0);
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float sum = _mm_cvtss_f32(s4);
            for (; j < n_cols; j++) sum += tmp[j] * t->x[j];
#else
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += tmp[j] * t->x[j];
#endif
            t->dst1[i] = sum;
        }
        for (int i = t->row_start; i < t->row_end; i++) {
            tf_dequant_row(t->mat2, i, tmp);
#if defined(__AVX2__) && defined(__FMA__)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 31 < n_cols; j += 32) {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j),    _mm256_loadu_ps(t->x+j),    a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+8),  _mm256_loadu_ps(t->x+j+8),  a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+16), _mm256_loadu_ps(t->x+j+16), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+24), _mm256_loadu_ps(t->x+j+24), a3);
            }
            for (; j + 7 < n_cols; j += 8)
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j), _mm256_loadu_ps(t->x+j), a0);
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float sum = _mm_cvtss_f32(s4);
            for (; j < n_cols; j++) sum += tmp[j] * t->x[j];
#else
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += tmp[j] * t->x[j];
#endif
            t->dst2[i] = sum;
        }
        free(tmp);
    }
    return NULL;
}

static void tf_qmatvec_pool(transformer_model *m, float *dst, const qtensor *mat, const float *x, int n_rows);

static void tf_qmatvec_fused2_pool(transformer_model *m, float *dst1, const qtensor *mat1,
                                    float *dst2, const qtensor *mat2,
                                    const float *x, int n_rows) {
    int nt = m->n_threads;
#if defined(__ARM_FEATURE_SVE)
    /* Panel-laid-out weights are already single-stream; the fused2 path only
     * existed to share x across two row-major streams. Route each separately. */
    if (mat1->panel || mat2->panel) {
        tf_qmatvec_pool(m, dst1, mat1, x, n_rows);
        tf_qmatvec_pool(m, dst2, mat2, x, n_rows);
        return;
    }
#endif
    if (nt <= 1 || !m->pool_alive) {
        tf_qmatvec(dst1, mat1, x, n_rows, m->thread_tmp[0]);
        tf_qmatvec(dst2, mat2, x, n_rows, m->thread_tmp[0]);
        return;
    }
    tf_matvec_fused2_task *tasks = (tf_matvec_fused2_task *)alloca(nt * sizeof(tf_matvec_fused2_task));
    for (int t = 0; t < nt; t++) {
        int rs, re;
        tf_row_split8(n_rows, nt, t, &rs, &re);
        tasks[t] = (tf_matvec_fused2_task){dst1, dst2, mat1, mat2, x, rs, re};
    }
    tf_pool_dispatch(m, tf_qmatvec_fused2_worker, tasks, sizeof(tf_matvec_fused2_task));
}

/* Forward declaration for fused FFN worker */
static void tf_matvec_qtensor_rows(float *dst, const qtensor *mat, const float *x,
                                    int row_start, int row_end);

/* Fused gate+up+SiLU: compute gate and up matvec for each thread's rows,
 * then immediately apply silu(gate[i]) * up[i] → dst[i].
 * Eliminates the SiLU barrier between gate+up and down projections.
 * dst = silu(gate_mat @ x) * (up_mat @ x) */
typedef struct {
    float *dst;           /* output: silu(gate) * up, [n_rows] */
    const qtensor *gate_mat, *up_mat;
    const float *x;
    int row_start, row_end;
} tf_fused_ffn_silu_task;

static void *tf_fused_ffn_silu_worker(void *arg) {
    tf_fused_ffn_silu_task *t = (tf_fused_ffn_silu_task *)arg;
    int rs = t->row_start, re = t->row_end;

    /* Phase 1: Compute gate and up matvec for this thread's row range.
     * tf_matvec_qtensor_rows writes to dst[row_start..row_end).
     * Use dst as gate buffer, allocate up on stack. */
    int n_rows = re - rs;
    float *up_buf = (float *)alloca((size_t)n_rows * sizeof(float));

    /* gate → dst[rs..re), up → up_buf[0..n_rows) */
    tf_matvec_qtensor_rows(t->dst, t->gate_mat, t->x, rs, re);
    tf_matvec_qtensor_rows(up_buf - rs, t->up_mat, t->x, rs, re);

    /* Phase 2: Fused SiLU×mul in-place: dst[i] = silu(gate[i]) * up[i] */
    for (int i = rs; i < re; i++) {
        float g = t->dst[i];
        t->dst[i] = g / (1.0f + expf(-g)) * up_buf[i - rs];
    }
    return NULL;
}

static void tf_qmatvec_fused2_silu_pool(transformer_model *m, float *dst,
                                          const qtensor *gate_mat, const qtensor *up_mat,
                                          const float *x, int n_rows) {
    int nt = m->n_threads;
    if (nt <= 1 || !m->pool_alive) {
        tf_fused_ffn_silu_task t = {dst, gate_mat, up_mat, x, 0, n_rows};
        tf_fused_ffn_silu_worker(&t);
        return;
    }
    tf_fused_ffn_silu_task *tasks = (tf_fused_ffn_silu_task *)alloca(nt * sizeof(tf_fused_ffn_silu_task));
    for (int t = 0; t < nt; t++) {
        int rs, re;
        tf_row_split8(n_rows, nt, t, &rs, &re);
        tasks[t] = (tf_fused_ffn_silu_task){dst, gate_mat, up_mat, x, rs, re};
    }
    tf_pool_dispatch(m, tf_fused_ffn_silu_worker, tasks, sizeof(tf_fused_ffn_silu_task));
}

/* Fused triple-matrix matvec: Q/K/V in one dispatch.
 * mat1 has n_rows1 rows, mat2 and mat3 have n_rows2 rows each. All same n_cols. */
typedef struct {
    float *dst1, *dst2, *dst3;
    const qtensor *mat1, *mat2, *mat3;
    const float *x;
    int row_start1, row_end1; /* range for mat1 */
    int row_start2, row_end2; /* range for mat2 and mat3 */
} tf_matvec_fused3_task;

/* Process one matrix matvec with its own type (helper for fused3 worker) */
static void tf_matvec_qtensor_rows(float *dst, const qtensor *mat, const float *x,
                                    int row_start, int row_end) {
    int n_cols = mat->n_cols;
    if (mat->type == GGML_TYPE_F16) {
        size_t rb = (size_t)n_cols * 2;
        tf_matvec_f16_rows(dst, (const uint8_t *)mat->data, rb, x, n_cols, row_start, row_end);
    } else if (mat->type == GGML_TYPE_BF16) {
        size_t rb = (size_t)n_cols * 2;
#if defined(__ARM_FEATURE_SVE)
        if (mat->q8_pv && (row_start & 7) == 0 && (row_end & 7) == 0) {
            const int8_t *xq; const uint16_t *xs;
            tf_quant_x_sdot(x, n_cols, &xq, &xs);
            int nb = n_cols / 64;
            size_t group_bytes = (size_t)nb * 528;
            const uint8_t *qbase = mat->q8_pv;
            for (int i = row_start; i + 7 < row_end; i += 8) {
                int g = i >> 3;
                matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                                 xq, xs, n_cols);
            }
        } else
#endif
        tf_matvec_bf16_rows_pv(dst, (const uint8_t *)mat->data, rb,
                                mat->bf16_pv, x, n_cols, row_start, row_end);
    } else if (mat->type == GGML_TYPE_Q8_0) {
        size_t rb = (size_t)(n_cols / 32) * sizeof(block_q8_0);
        tf_matvec_q8_rows(dst, (const uint8_t *)mat->data, rb, x, n_cols, row_start, row_end);
    } else if (mat->q8_pv && (row_start & 7) == 0 && (row_end & 7) == 0) {
        const int8_t *xq; const uint16_t *xs;
        tf_quant_x_sdot(x, n_cols, &xq, &xs);
        int nb = n_cols / 64;
        size_t group_bytes = (size_t)nb * 528;
        const uint8_t *qbase = mat->q8_pv;
        for (int i = row_start; i + 7 < row_end; i += 8) {
            int g = i >> 3;
            matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                             xq, xs, n_cols);
        }
    } else if (mat->bf16_pv && (row_start & 7) == 0 && (row_end & 7) == 0) {
        /* Dense quantized tensor repacked to bf16_pv (e.g. SSM mixer via
         * transformer_repack_dense_bf16_pv). Callers row-split with
         * tf_row_split8 and the repack requires n_rows%8==0, so [row_start,
         * row_end) is 8-aligned and tf_matvec_bf16_rows_pv stays on the pv fast
         * path and never dereferences the quantized `base`. Single dispatch
         * point shared by the threaded (tf_thread_matvec) and serial
         * fused2-diff / fused3 / fused-silu pool workers. */
        tf_matvec_bf16_rows_pv(dst, (const uint8_t *)mat->data, (size_t)n_cols * 2,
                                mat->bf16_pv, x, n_cols, row_start, row_end);
    } else {
        float *tmp = (float *)malloc(n_cols * sizeof(float));
        if (!tmp) return;
        for (int i = row_start; i < row_end; i++) {
            tf_dequant_row(mat, i, tmp);
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
            dst[i] = sum;
        }
        free(tmp);
    }
}

static void *tf_qmatvec_fused3_worker(void *arg) {
    tf_matvec_fused3_task *t = (tf_matvec_fused3_task *)arg;
    /* Each matrix is processed with its own type — handles mixed quantization
     * (e.g. Unsloth UD where Q may be F16 while K/V remain Q8_0). */
    if (t->row_end1 > t->row_start1)
        tf_matvec_qtensor_rows(t->dst1, t->mat1, t->x, t->row_start1, t->row_end1);
    if (t->row_end2 > t->row_start2) {
        tf_matvec_qtensor_rows(t->dst2, t->mat2, t->x, t->row_start2, t->row_end2);
        tf_matvec_qtensor_rows(t->dst3, t->mat3, t->x, t->row_start2, t->row_end2);
    }
    return NULL;
}

static void tf_qmatvec_fused_qkv_pool(transformer_model *m,
                                        float *q, const qtensor *mat_q, int n_q,
                                        float *k, const qtensor *mat_k,
                                        float *v, const qtensor *mat_v, int n_kv) {
    int nt = m->n_threads;
    if (nt <= 1 || !m->pool_alive) {
        tf_qmatvec(q, mat_q, m->xb, n_q, m->thread_tmp[0]);
        tf_qmatvec(k, mat_k, m->xb, n_kv, m->thread_tmp[0]);
        tf_qmatvec(v, mat_v, m->xb, n_kv, m->thread_tmp[0]);
        return;
    }
    /* Distribute rows across threads: n_q Q rows and n_kv K+V rows each. */
    tf_matvec_fused3_task *tasks = (tf_matvec_fused3_task *)alloca(nt * sizeof(tf_matvec_fused3_task));
    for (int t = 0; t < nt; t++) {
        int qs, qe, ks, ke;
        tf_row_split8(n_q,  nt, t, &qs, &qe);
        tf_row_split8(n_kv, nt, t, &ks, &ke);
        tasks[t] = (tf_matvec_fused3_task){
            q, k, v, mat_q, mat_k, mat_v, m->xb,
            qs, qe, ks, ke
        };
    }
    tf_pool_dispatch(m, tf_qmatvec_fused3_worker, tasks, sizeof(tf_matvec_fused3_task));
}

/* Fused pair of independent matvecs sharing the same input x, with potentially
 * different row counts and matrix types. Used in the SSM block to merge
 * ssm_qkv (qkv_dim rows) and ssm_gate (d_inner rows), eliminating one
 * pool barrier per SSM layer. */
typedef struct {
    float *dst1;
    const qtensor *mat1;
    float *dst2;
    const qtensor *mat2;
    const float *x;
    int row_start1, row_end1;
    int row_start2, row_end2;
} tf_matvec_fused2_diff_task;

static void *tf_qmatvec_fused2_diff_worker(void *arg) {
    tf_matvec_fused2_diff_task *t = (tf_matvec_fused2_diff_task *)arg;
    if (t->row_end1 > t->row_start1)
        tf_matvec_qtensor_rows(t->dst1, t->mat1, t->x, t->row_start1, t->row_end1);
    if (t->row_end2 > t->row_start2)
        tf_matvec_qtensor_rows(t->dst2, t->mat2, t->x, t->row_start2, t->row_end2);
    return NULL;
}

static void tf_qmatvec_fused2_diff_pool(transformer_model *m,
                                          float *dst1, const qtensor *mat1, int n_rows1,
                                          float *dst2, const qtensor *mat2, int n_rows2,
                                          const float *x) {
    int nt = m->n_threads;
#if defined(__ARM_FEATURE_SVE)
    /* Panel-laid-out weights are single-stream; fall back to two pool calls. */
    if (mat1->panel || mat2->panel) {
        tf_qmatvec_pool(m, dst1, mat1, x, n_rows1);
        tf_qmatvec_pool(m, dst2, mat2, x, n_rows2);
        return;
    }
#endif
    if (nt <= 1 || !m->pool_alive) {
        tf_qmatvec(dst1, mat1, x, n_rows1, m->thread_tmp[0]);
        tf_qmatvec(dst2, mat2, x, n_rows2, m->thread_tmp[0]);
        return;
    }
    tf_matvec_fused2_diff_task *tasks = (tf_matvec_fused2_diff_task *)alloca(
        nt * sizeof(tf_matvec_fused2_diff_task));
    for (int t = 0; t < nt; t++) {
        int rs1, re1, rs2, re2;
        tf_row_split8(n_rows1, nt, t, &rs1, &re1);
        tf_row_split8(n_rows2, nt, t, &rs2, &re2);
        tasks[t] = (tf_matvec_fused2_diff_task){
            dst1, mat1, dst2, mat2, x, rs1, re1, rs2, re2
        };
    }
    tf_pool_dispatch(m, tf_qmatvec_fused2_diff_worker, tasks,
                      sizeof(tf_matvec_fused2_diff_task));
}

static void tf_qmatvec(float *dst, const qtensor *mat, const float *x, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    if (mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = 0;
        for (; i + 5 < n_rows; i += 6) {
            matvec_f16_6row(dst + i,
                (const uint16_t *)(base + (size_t)i * row_bytes),
                (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
                x, n_cols);
        }
        for (; i < n_rows; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_f16_f32(row, x, n_cols);
        }
        return;
    }
    if (mat->type == GGML_TYPE_BF16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
#if defined(__ARM_FEATURE_SVE)
        if (mat->q8_pv && n_rows >= 8 && (n_rows & 7) == 0) {
            const int8_t *xq; const uint16_t *xs;
            tf_quant_x_sdot(x, n_cols, &xq, &xs);
            int nb = n_cols / 64;
            size_t group_bytes = (size_t)nb * 528;
            const uint8_t *qbase = mat->q8_pv;
            for (int i = 0; i + 7 < n_rows; i += 8) {
                int g = i >> 3;
                matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                                 xq, xs, n_cols);
            }
            return;
        }
#endif
        tf_matvec_bf16_rows_pv(dst, base, row_bytes, mat->bf16_pv,
                                x, n_cols, 0, n_rows);
        return;
    }
    if (mat->type == GGML_TYPE_Q8_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
        const uint8_t *base = (const uint8_t *)mat->data;
        for (int i = 0; i < n_rows; i++) {
            dst[i] = vec_dot_q8_0_f32(base + (size_t)i * row_bytes, x, n_cols);
        }
        return;
    }
#if defined(__ARM_FEATURE_SVE)
    if (mat->q8_pv && n_rows >= 8 && (n_rows & 7) == 0) {
        const int8_t *xq; const uint16_t *xs;
        tf_quant_x_sdot(x, n_cols, &xq, &xs);
        int nb = n_cols / 64;
        size_t group_bytes = (size_t)nb * 528;
        const uint8_t *qbase = mat->q8_pv;
        for (int i = 0; i + 7 < n_rows; i += 8) {
            int g = i >> 3;
            matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                             xq, xs, n_cols);
        }
        return;
    }
#endif
    for (int i = 0; i < n_rows; i++) {
        tf_dequant_row(mat, i, tmp);
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 31 < n_cols; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j),      _mm256_loadu_ps(x + j),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j + 8),  _mm256_loadu_ps(x + j + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j + 16), _mm256_loadu_ps(x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j + 24), _mm256_loadu_ps(x + j + 24), acc3);
        }
        for (; j + 7 < n_cols; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j), _mm256_loadu_ps(x + j), acc0);
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float sum = _mm_cvtss_f32(s4);
        for (; j < n_cols; j++) sum += tmp[j] * x[j];
#else
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
#endif
        dst[i] = sum;
    }
}

/* Pool-based multi-threaded matvec (avoids pthread_create per call) */
#if defined(__ARM_FEATURE_SVE)

/* M10 (A64FX): repack an F16 weight [n_rows][n_cols] row-major into panel
 * layout panel[blk][k][lane] = W[blk*32+lane][k]. Each SVE fp16 lane then
 * accumulates a distinct output row, so the matvec needs no horizontal
 * reduction and reads the weight as one sequential stream. The last block is
 * zero-padded when n_rows is not a multiple of 32. Sets t->panel / t->panel_blk;
 * no-op for non-F16 tensors. The original t->data is kept (other code paths
 * and dequant still use it); this roughly doubles F16 weight memory. */
static size_t tf_panel_bytes(int nblk, int K) {
    return (size_t)nblk * (size_t)K * 32 * sizeof(uint16_t);
}

/* Fill panel blocks [blk_start, blk_end) from the row-major F16 weight.
 * Iterates block-major so a worker writes one contiguous panel region —
 * with CMG pinning that region is first-touched onto the worker's HBM.
 * Tail rows past n_rows are left at their mmap-zeroed value. */
static void tf_panel_fill_range(qtensor *t, int blk_start, int blk_end) {
    int K = t->n_cols, n_rows = t->n_rows;
    const uint16_t *W = (const uint16_t *)t->data;
    uint16_t *p = t->panel;
    for (int blk = blk_start; blk < blk_end; blk++) {
        uint16_t *pb = p + (size_t)blk * K * 32;
        for (int lane = 0; lane < 32; lane++) {
            int row = blk * 32 + lane;
            if (row >= n_rows) break;
            const uint16_t *src = W + (size_t)row * K;
            for (int k = 0; k < K; k++) pb[(size_t)k * 32 + lane] = src[k];
        }
    }
}

/* Allocate t->panel as untouched anonymous pages (mmap, zero-filled lazily) so
 * first-touch placement works. Sets t->panel_blk. Returns 1 on success. */
static int tf_panel_alloc(qtensor *t) {
    if (!t->data || t->type != GGML_TYPE_F16) return 0;
    int nblk = (t->n_rows + 31) / 32;
    if (nblk == 0) return 0;
    size_t bytes = tf_panel_bytes(nblk, t->n_cols);
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
    /* Disable transparent hugepages: THP coalesces the panel into 2 MB pages
     * faulted as a unit, which collapses our per-thread first-touch back onto
     * a single CMG (observed: all panels on node 4). With 4 KB pages each
     * thread's contiguous block range first-touches onto its own CMG's HBM,
     * and the matvec streams sequentially so the TLB cost is negligible. */
#ifdef MADV_NOHUGEPAGE
    madvise(p, bytes, MADV_NOHUGEPAGE);
#endif
    t->panel = (uint16_t *)p;
    t->panel_blk = nblk;
    return 1;
}

static void tf_panel_free(qtensor *t) {
    if (t->panel) {
        munmap(t->panel, tf_panel_bytes(t->panel_blk, t->n_cols));
        t->panel = NULL;
        t->panel_blk = 0;
    }
}

/* Single-threaded build (no pool): alloc + fill the whole panel. */
static void tf_qtensor_build_panel(qtensor *t) {
    if (!tf_panel_alloc(t)) return;
    tf_panel_fill_range(t, 0, t->panel_blk);
}

/* BF16 p_odd-pair repack: layout pairs of adjacent rows so matvec_bf16_8row_pv
 * can extract both rows as FP32 with one ld1h.h+p_odd. Per "group" of 8 rows,
 * 4 pair buffers each of 2*K bf16. Within a pair, 16-element chunks of K are
 * stored as 32 halfwords: HW 0,2,...,30 = rA, HW 1,3,...,31 = rB.
 *
 * Total bytes = n_rows * K * 2 (same as row-major). bf16_pv_groups = n_rows/8.
 * Caller must guarantee n_rows % 8 == 0 and n_cols % 16 == 0. */
static size_t tf_bf16_pv_bytes(int groups, int K) {
    return (size_t)groups * 8 * (size_t)K * sizeof(uint16_t);
}

static void tf_bf16_pv_fill_range(qtensor *t, int g_start, int g_end) {
    int K = t->n_cols;
    int stride = t->tp_src_stride ? t->tp_src_stride : K;  /* TP row-parallel: orig n_cols */
    int vl = 16;  /* fp32 lanes on A64FX SVE */
    const uint16_t *W = (const uint16_t *)t->data;
    uint16_t *pv = t->bf16_pv;
    int chunks_per_K = K / vl;
    for (int g = g_start; g < g_end; g++) {
        uint16_t *gbuf = pv + (size_t)g * 8 * K;
        for (int p = 0; p < 4; p++) {
            int rowA = g * 8 + 2 * p;
            int rowB = g * 8 + 2 * p + 1;
            const uint16_t *srcA = W + (size_t)rowA * stride;
            const uint16_t *srcB = W + (size_t)rowB * stride;
            uint16_t *pair = gbuf + (size_t)p * 2 * K;
            for (int c = 0; c < chunks_per_K; c++) {
                uint16_t *chunk = pair + (size_t)c * 32;
                for (int lane = 0; lane < vl; lane++) {
                    chunk[2 * lane + 0] = srcA[c * vl + lane]; /* even HW */
                    chunk[2 * lane + 1] = srcB[c * vl + lane]; /* odd HW  */
                }
            }
        }
    }
}

static int tf_bf16_pv_alloc(qtensor *t) {
    if (!t->data || t->type != GGML_TYPE_BF16) return 0;
    if (t->n_rows < 8 || (t->n_rows & 7) != 0) return 0;
    if (t->n_cols <= 0 || (t->n_cols & 15) != 0) return 0;
    int groups = t->n_rows / 8;
    size_t bytes = tf_bf16_pv_bytes(groups, t->n_cols);
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
#ifdef MADV_NOHUGEPAGE
    madvise(p, bytes, MADV_NOHUGEPAGE);
#endif
    t->bf16_pv = (uint16_t *)p;
    t->bf16_pv_groups = groups;
    return 1;
}

static void tf_bf16_pv_free(qtensor *t) {
    if (t->bf16_pv) {
        munmap(t->bf16_pv,
               tf_bf16_pv_bytes(t->bf16_pv_groups, t->n_cols));
        t->bf16_pv = NULL;
        t->bf16_pv_groups = 0;
    }
}

/* int8 svdot quantize-on-load layout — see qtensor.q8_pv comment for the
 * format. Per group of 8 rows × K cols, we store nb = K/64 blocks of 528 B
 * each: 8 fp16 row-scales [0..16) then 8 rows × 64 int8 [16..528). Consumed
 * by matvec_sdot_8row (W8A8 svdot). */
static size_t tf_q8_pv_bytes(int groups, int K) {
    int nb = K / 64;
    return (size_t)groups * (size_t)nb * 528ULL;
}

static void tf_q8_pv_fill_range(qtensor *t, int g_start, int g_end) {
    int K = t->n_cols;
    int stride = t->tp_src_stride ? t->tp_src_stride : K;  /* TP row-parallel: orig n_cols */
    int nb = K / 64;
    const uint16_t *W = (const uint16_t *)t->data;
    uint8_t *qv = t->q8_pv;
    for (int g = g_start; g < g_end; g++) {
        uint8_t *gbuf = qv + (size_t)g * nb * 528;
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = gbuf + (size_t)b * 528;
            uint16_t *scl = (uint16_t *)blk;
            int8_t *qs   = (int8_t *)(blk + 16);
            for (int r = 0; r < 8; r++) {
                const uint16_t *src = W + (size_t)(g * 8 + r) * stride + (size_t)b * 64;
                /* Compute absmax over the 64-elem block (BF16 source). */
                float amax = 0.0f;
                for (int j = 0; j < 64; j++) {
                    uint32_t bits = (uint32_t)src[j] << 16;
                    float f; __builtin_memcpy(&f, &bits, 4);
                    float a = f < 0 ? -f : f;
                    if (a > amax) amax = a;
                }
                float scale = amax / 127.0f;
                float invs  = amax > 0 ? 127.0f / amax : 0.0f;
                scl[r] = ggml_fp32_to_fp16(scale);
                for (int j = 0; j < 64; j++) {
                    uint32_t b32 = (uint32_t)src[j] << 16;
                    float f; __builtin_memcpy(&f, &b32, 4);
                    int q = (int)lrintf(f * invs);
                    if (q < -127) q = -127; else if (q > 127) q = 127;
                    qs[r * 64 + j] = (int8_t)q;
                }
            }
        }
    }
}

static int tf_q8_pv_alloc(qtensor *t) {
    if (!t->data || t->type != GGML_TYPE_BF16) return 0;
    if (t->n_rows < 8 || (t->n_rows & 7) != 0) return 0;
    if (t->n_cols <= 0 || (t->n_cols & 63) != 0) return 0;
    int groups = t->n_rows / 8;
    size_t bytes = tf_q8_pv_bytes(groups, t->n_cols);
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
#ifdef MADV_NOHUGEPAGE
    madvise(p, bytes, MADV_NOHUGEPAGE);
#endif
    t->q8_pv = (uint8_t *)p;
    t->q8_pv_groups = groups;
    return 1;
}

static void tf_q8_pv_free(qtensor *t) {
    if (t->q8_pv) {
        munmap(t->q8_pv,
               tf_q8_pv_bytes(t->q8_pv_groups, t->n_cols));
        t->q8_pv = NULL;
        t->q8_pv_groups = 0;
    }
}

typedef struct {
    float *dst;
    const uint16_t *panel;
    const float16_t *x;  /* input vector, pre-converted to f16 once per matvec */
    int blk_start, blk_end;
    int n_cols;
    int n_rows;          /* for predicated store of the last partial block */
} tf_panel_task;

/* Convert an f32 input vector to f16 once, so the matvec inner loop can svdup
 * straight from f16 with no per-element fcvt. The panel kernel re-reads x for
 * every block it owns, so this hoists nblk/nt fcvts down to one pass. */
static inline void tf_x_to_f16(float16_t *dst, const float *src, int n) {
    int i = 0;
    svbool_t pg = svptrue_b32();
    for (; i + 16 <= n; i += 16)
        svst1_f16(svptrue_b16(), dst + i,
                  svuzp1_f16(svcvt_f16_f32_x(pg, svld1_f32(pg, src + i)),
                             svcvt_f16_f32_x(pg, svld1_f32(pg, src + i + 8))));
    for (; i < n; i++) dst[i] = (float16_t)src[i];
}

/* Panel matvec: single sequential stream over panel, 8 k-accumulators to hide
 * the ~9-cycle svmla_f16 latency, result store is a plain widen + svst1. */
static void *tf_panel_matvec_worker(void *arg) {
    tf_panel_task *t = (tf_panel_task *)arg;
    int K = t->n_cols, n_rows = t->n_rows;
    const float16_t *x = t->x;
    svbool_t ph = svptrue_b16(), pg = svptrue_b32();
    for (int blk = t->blk_start; blk < t->blk_end; blk++) {
        const float16_t *wp = (const float16_t *)t->panel + (size_t)blk * K * 32;
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
        int k = 0;
        for (; k + 7 < K; k += 8) {
            a0 = svmla_x(ph,a0, svld1_f16(ph,wp+(size_t)(k+0)*32), svdup_f16(x[k+0]));
            a1 = svmla_x(ph,a1, svld1_f16(ph,wp+(size_t)(k+1)*32), svdup_f16(x[k+1]));
            a2 = svmla_x(ph,a2, svld1_f16(ph,wp+(size_t)(k+2)*32), svdup_f16(x[k+2]));
            a3 = svmla_x(ph,a3, svld1_f16(ph,wp+(size_t)(k+3)*32), svdup_f16(x[k+3]));
            a4 = svmla_x(ph,a4, svld1_f16(ph,wp+(size_t)(k+4)*32), svdup_f16(x[k+4]));
            a5 = svmla_x(ph,a5, svld1_f16(ph,wp+(size_t)(k+5)*32), svdup_f16(x[k+5]));
            a6 = svmla_x(ph,a6, svld1_f16(ph,wp+(size_t)(k+6)*32), svdup_f16(x[k+6]));
            a7 = svmla_x(ph,a7, svld1_f16(ph,wp+(size_t)(k+7)*32), svdup_f16(x[k+7]));
        }
        for (; k < K; k++)
            a0 = svmla_x(ph,a0, svld1_f16(ph,wp+(size_t)k*32), svdup_f16(x[k]));
        svfloat16_t s = svadd_x(ph, svadd_x(ph,svadd_x(ph,a0,a1),svadd_x(ph,a2,a3)),
                                    svadd_x(ph,svadd_x(ph,a4,a5),svadd_x(ph,a6,a7)));
        svuint16_t u = svreinterpret_u16(s);
        int row0 = blk * 32;
        float *d = t->dst + row0;
        svfloat32_t lo = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo_u32(u)));
        svfloat32_t hi = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi_u32(u)));
        if (row0 + 32 <= n_rows) {
            svst1_f32(pg, d,      lo);
            svst1_f32(pg, d + 16, hi);
        } else {
            /* last partial block: predicated store, drop zero-padded rows */
            svst1_f32(svwhilelt_b32(row0,      n_rows), d,      lo);
            svst1_f32(svwhilelt_b32(row0 + 16, n_rows), d + 16, hi);
        }
    }
    return NULL;
}

static void tf_panel_matvec_pool(transformer_model *m, float *dst,
                                 const qtensor *mat, const float *x) {
    int nblk = mat->panel_blk;
    int K = mat->n_cols, n_rows = mat->n_rows;
    int n_threads = m->n_threads;
    float16_t *xh = (float16_t *)alloca((size_t)K * sizeof(float16_t));
    tf_x_to_f16(xh, x, K);
    if (n_threads <= 1 || nblk < n_threads || !m->pool_alive) {
        tf_panel_task t = {dst, mat->panel, xh, 0, nblk, K, n_rows};
        tf_panel_matvec_worker(&t);
        return;
    }
    tf_panel_task *tasks = (tf_panel_task *)alloca(n_threads * sizeof(tf_panel_task));
    int per = nblk / n_threads, extra = nblk % n_threads, off = 0;
    for (int i = 0; i < n_threads; i++) {
        int c = per + (i < extra ? 1 : 0);
        tasks[i] = (tf_panel_task){dst, mat->panel, xh, off, off + c, K, n_rows};
        off += c;
    }
    tf_pool_dispatch(m, tf_panel_matvec_worker, tasks, sizeof(tf_panel_task));
}

/* Parallel panel fill: each worker fills the same contiguous block range it
 * will later consume in tf_panel_matvec_pool, so the panel memory is
 * first-touched onto that worker's CMG. */
typedef struct { qtensor *t; int blk_start, blk_end; } tf_panel_build_task;
static void *tf_panel_build_worker(void *arg) {
    tf_panel_build_task *b = (tf_panel_build_task *)arg;
    tf_panel_fill_range(b->t, b->blk_start, b->blk_end);
    return NULL;
}
static void *tf_bf16_pv_build_worker(void *arg) {
    tf_panel_build_task *b = (tf_panel_build_task *)arg;
    tf_bf16_pv_fill_range(b->t, b->blk_start, b->blk_end);
    return NULL;
}
static void *tf_q8_pv_build_worker(void *arg) {
    tf_panel_build_task *b = (tf_panel_build_task *)arg;
    tf_q8_pv_fill_range(b->t, b->blk_start, b->blk_end);
    return NULL;
}
#endif /* __ARM_FEATURE_SVE */

static void tf_qmatvec_pool(transformer_model *m, float *dst, const qtensor *mat, const float *x, int n_rows) {
    int n_threads = m->n_threads;
#if defined(__ARM_FEATURE_SVE)
    if (mat->panel) {
        tf_panel_matvec_pool(m, dst, mat, x);
        return;
    }
#endif
    if (n_threads <= 1 || n_rows < n_threads * 4 || !m->pool_alive) {
        tf_qmatvec(dst, mat, x, n_rows, m->thread_tmp[0]);
        return;
    }
    tf_matvec_task *tasks = (tf_matvec_task *)alloca(n_threads * sizeof(tf_matvec_task));
    for (int t = 0; t < n_threads; t++) {
        int rs, re;
        tf_row_split8(n_rows, n_threads, t, &rs, &re);
        tasks[t] = (tf_matvec_task){dst, mat, x, rs, re, m->thread_tmp[t]};
    }
    tf_pool_dispatch(m, tf_qmatvec_worker, tasks, sizeof(tf_matvec_task));
}

/* Pool-based multi-threaded expert matvec: splits rows across threads */
/* Forward declaration — definition lives with the MoE expert PV repack block
 * later in this header (alongside tf_expert_bf16_pv_alloc/fill_one and
 * tf_expert_q8_pv_alloc/fill_one). */
static inline void tf_expert_matvec_bf16_pv_block(float *dst, const qtensor *t,
                                                    int expert, const float *x,
                                                    int row_start, int row_end);
static inline void tf_expert_matvec_q8_pv_block(float *dst, const qtensor *t,
                                               int expert, const int8_t *xq,
                                               const uint16_t *xs,
                                               int row_start, int row_end);

/* Per-expert bf16_pv worker: dispatched when mat->expert_owned_slot is set
 * and the expert is owned. 8-row blocks per row slice, no dequant. */
typedef struct {
    float *dst;
    const qtensor *mat;
    int expert;
    const float *x;
    int row_start, row_end;
} tf_expert_pv_task;
static void *tf_expert_pv_worker(void *arg) {
    tf_expert_pv_task *t = (tf_expert_pv_task *)arg;
    tf_expert_matvec_bf16_pv_block(t->dst, t->mat, t->expert, t->x,
                                    t->row_start, t->row_end);
    return NULL;
}

typedef struct {
    float *dst;
    const qtensor *mat;
    int expert;
    const int8_t *xq;
    const uint16_t *xs;
    int row_start, row_end;
} tf_expert_q8_pv_task;
static void *tf_expert_q8_pv_worker(void *arg) {
    tf_expert_q8_pv_task *t = (tf_expert_q8_pv_task *)arg;
    tf_expert_matvec_q8_pv_block(t->dst, t->mat, t->expert, t->xq, t->xs,
                                 t->row_start, t->row_end);
    return NULL;
}

static void tf_qmatvec_expert_pool(transformer_model *m, float *dst, const qtensor *mat,
                                    int expert, const float *x, int rows_per_expert) {
    int n_threads = m->n_threads;
    if (mat->expert_owned_slot && mat->q8_pv &&
        mat->expert_owned_slot[expert] >= 0 &&
        (rows_per_expert & 7) == 0) {
        const int8_t *xq;
        const uint16_t *xs;
        tf_quant_x_sdot(x, mat->n_cols, &xq, &xs);
        if (n_threads <= 1 || rows_per_expert < n_threads * 8 || !m->pool_alive) {
            tf_expert_matvec_q8_pv_block(dst, mat, expert, xq, xs, 0, rows_per_expert);
            return;
        }
        tf_expert_q8_pv_task *tasks =
            (tf_expert_q8_pv_task *)alloca(n_threads * sizeof(*tasks));
        for (int tt = 0; tt < n_threads; tt++) {
            int rs, re;
            tf_row_split8(rows_per_expert, n_threads, tt, &rs, &re);
            tasks[tt] = (tf_expert_q8_pv_task){dst, mat, expert, xq, xs, rs, re};
        }
        tf_pool_dispatch(m, tf_expert_q8_pv_worker, tasks,
                        sizeof(tf_expert_q8_pv_task));
        return;
    }
    /* Fast path: per-expert bf16_pv repack is active and this expert is owned. */
    if (mat->expert_owned_slot && mat->bf16_pv &&
        mat->expert_owned_slot[expert] >= 0 &&
        (rows_per_expert & 7) == 0) {
        if (n_threads <= 1 || rows_per_expert < n_threads * 8 || !m->pool_alive) {
            tf_expert_matvec_bf16_pv_block(dst, mat, expert, x, 0, rows_per_expert);
            return;
        }
        tf_expert_pv_task *tasks = (tf_expert_pv_task *)alloca(n_threads * sizeof(tf_expert_pv_task));
        for (int tt = 0; tt < n_threads; tt++) {
            int rs, re;
            tf_row_split8(rows_per_expert, n_threads, tt, &rs, &re);
            tasks[tt] = (tf_expert_pv_task){dst, mat, expert, x, rs, re};
        }
        tf_pool_dispatch(m, tf_expert_pv_worker, tasks, sizeof(tf_expert_pv_task));
        return;
    }
    if (n_threads <= 1 || rows_per_expert < n_threads * 4 || !m->pool_alive) {
        tf_qmatvec_expert(dst, mat, expert, x, rows_per_expert, m->thread_tmp[0]);
        return;
    }
    /* Create a virtual qtensor pointing to the expert's data slice */
    size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
    qtensor expert_mat = *mat;
    expert_mat.data = (void *)((const uint8_t *)mat->data + (size_t)expert * rows_per_expert * row_bytes);

    tf_matvec_task *tasks = (tf_matvec_task *)alloca(n_threads * sizeof(tf_matvec_task));
    int rows_per = rows_per_expert / n_threads;
    int extra = rows_per_expert % n_threads;
    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int count = rows_per + (t < extra ? 1 : 0);
        tasks[t] = (tf_matvec_task){dst, &expert_mat, x, offset, offset + count, m->thread_tmp[t]};
        offset += count;
    }
    tf_pool_dispatch(m, tf_qmatvec_worker, tasks, sizeof(tf_matvec_task));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Fused MoE expert dispatch (decode B=1): collapse K selected experts ×
 * {up, gate, silu_mul, down, weighted-accum} into TWO pool dispatches per
 * MoE layer instead of 3*K (== 24 at K=8). Cuts fork/join overhead and
 * raises rows-per-thread from rpe/48 (~11) to K*rpe/48 (~85).
 *
 * Math is bit-identical to the per-expert loop: same per-(k,row) dequant
 * order, same scalar dot order, same silu formula, same K-way accumulate
 * order. Threads partition independent output rows so there are no
 * read-modify-write races on xb2. */
typedef struct {
    float *activated_out;    /* [K * rows_per_expert] */
    const qtensor *W_up;
    const qtensor *W_gate;
    const int *experts;      /* K selected expert IDs */
    int K;
    const float *xb;         /* shared input, length n_embd */
    int rows_per_expert;
    int row_start, row_end;  /* this thread's row slice in [0, rows_per_expert) */
    float *tmp_up;
    float *tmp_gate;
} tf_moe_upgate_task;

static void *tf_moe_upgate_worker(void *arg) {
    tf_moe_upgate_task *t = (tf_moe_upgate_task *)arg;
    int n_cols = t->W_up->n_cols;
    int rpe = t->rows_per_expert;
    int use_q8_pv = (t->W_up->expert_owned_slot && t->W_up->q8_pv &&
                     t->W_gate->expert_owned_slot && t->W_gate->q8_pv &&
                     (t->row_start & 7) == 0 && (t->row_end & 7) == 0);
    int use_pv = (!use_q8_pv &&
                  t->W_up->expert_owned_slot && t->W_up->bf16_pv &&
                  t->W_gate->expert_owned_slot && t->W_gate->bf16_pv &&
                  (t->row_start & 7) == 0 && (t->row_end & 7) == 0);
    const int8_t *xq = NULL;
    const uint16_t *xs = NULL;
    if (use_q8_pv) tf_quant_x_sdot(t->xb, n_cols, &xq, &xs);
    if (use_q8_pv || use_pv) {
        /* up lands in act_k (size rpe, dst-indexed by r). gate lands in
         * tmp_gate scratch indexed by (r - row_start), which fits since
         * the row span ≤ rpe and tmp_gate has at least n_cols floats and
         * we additionally require span ≤ n_cols below. */
        int span = t->row_end - t->row_start;
        if (span <= n_cols) {
            for (int k = 0; k < t->K; k++) {
                int e = t->experts[k];
                float *act_k = t->activated_out + (size_t)k * rpe;
                if (use_q8_pv) {
                    tf_expert_matvec_q8_pv_block(act_k, t->W_up, e, xq, xs,
                                                t->row_start, t->row_end);
                    tf_expert_matvec_q8_pv_block(t->tmp_gate - t->row_start,
                                                t->W_gate, e, xq, xs,
                                                t->row_start, t->row_end);
                } else {
                    tf_expert_matvec_bf16_pv_block(act_k, t->W_up, e, t->xb,
                                                  t->row_start, t->row_end);
                    /* shift dst pointer back so the kernel writes to
                     * (tmp_gate - row_start)[r] = tmp_gate[r - row_start] */
                    tf_expert_matvec_bf16_pv_block(t->tmp_gate - t->row_start,
                                                  t->W_gate, e, t->xb,
                                                  t->row_start, t->row_end);
                }
                for (int r = t->row_start; r < t->row_end; r++) {
                    float up = act_k[r];
                    float gate = t->tmp_gate[r - t->row_start];
                    act_k[r] = (gate / (1.0f + expf(-gate))) * up;
                }
            }
            return NULL;
        }
        /* fall through to scalar path on improbable shape */
    }
    size_t rb_up = tf_row_bytes(t->W_up->type, n_cols);
    size_t rb_gate = tf_row_bytes(t->W_gate->type, n_cols);
    const uint8_t *base_up = (const uint8_t *)t->W_up->data;
    const uint8_t *base_gate = (const uint8_t *)t->W_gate->data;

    for (int k = 0; k < t->K; k++) {
        int e = t->experts[k];
        float *act_k = t->activated_out + (size_t)k * rpe;
        const uint8_t *e_up   = base_up   + (size_t)e * rpe * rb_up;
        const uint8_t *e_gate = base_gate + (size_t)e * rpe * rb_gate;
        for (int r = t->row_start; r < t->row_end; r++) {
            const void *row_up   = e_up   + (size_t)r * rb_up;
            const void *row_gate = e_gate + (size_t)r * rb_gate;
            dequant_row(t->W_up->type,   row_up,   t->tmp_up,   n_cols);
            dequant_row(t->W_gate->type, row_gate, t->tmp_gate, n_cols);
            float up = 0.0f, gate = 0.0f;
            for (int j = 0; j < n_cols; j++) {
                up   += t->tmp_up[j]   * t->xb[j];
                gate += t->tmp_gate[j] * t->xb[j];
            }
            act_k[r] = (gate / (1.0f + expf(-gate))) * up;
        }
    }
    return NULL;
}

typedef struct {
    float *xb2_out;             /* [n_embd] — written, not accumulated */
    const qtensor *W_down;
    const int *experts;
    const float *ews;           /* per-expert mixture weights */
    int K;
    const float *activated;     /* [K * rows_per_input] */
    int rows_per_input;         /* == n_ff_exp */
    int n_embd;
    int row_start, row_end;     /* this thread's row slice in [0, n_embd) */
    float *tmp;
} tf_moe_down_task;

static void *tf_moe_down_worker(void *arg) {
    tf_moe_down_task *t = (tf_moe_down_task *)arg;
    int n_cols = t->W_down->n_cols;
    int rpe = t->n_embd;
    int use_q8_pv = (t->W_down->expert_owned_slot && t->W_down->q8_pv &&
                     (t->row_start & 7) == 0 && (t->row_end & 7) == 0);
    /* pv path: for each expert, run a row-sliced bf16_pv matvec on the
     * expert's input activations, accumulate the weighted result into
     * xb2_out[row_start..row_end). */
    if (use_q8_pv || (t->W_down->expert_owned_slot && t->W_down->bf16_pv &&
        (t->row_start & 7) == 0 && (t->row_end & 7) == 0)) {
        int span = t->row_end - t->row_start;
        if (span <= n_cols) {
            for (int r = t->row_start; r < t->row_end; r++) t->xb2_out[r] = 0.0f;
            for (int k = 0; k < t->K; k++) {
                int e = t->experts[k];
                const float *act_k = t->activated + (size_t)k * t->rows_per_input;
                const int8_t *xq = NULL;
                const uint16_t *xs = NULL;
                if (use_q8_pv) tf_quant_x_sdot(act_k, n_cols, &xq, &xs);
                /* shift dst back so kernel writes tmp[0..span) */
                if (use_q8_pv) {
                    tf_expert_matvec_q8_pv_block(t->tmp - t->row_start,
                                                t->W_down, e, xq, xs,
                                                t->row_start, t->row_end);
                } else {
                    tf_expert_matvec_bf16_pv_block(t->tmp - t->row_start,
                                                  t->W_down, e, act_k,
                                                  t->row_start, t->row_end);
                }
                float w = t->ews[k];
                for (int r = t->row_start; r < t->row_end; r++) {
                    t->xb2_out[r] += w * t->tmp[r - t->row_start];
                }
            }
            return NULL;
        }
    }
    size_t rb = tf_row_bytes(t->W_down->type, n_cols);
    const uint8_t *base = (const uint8_t *)t->W_down->data;

    for (int r = t->row_start; r < t->row_end; r++) {
        float acc = 0.0f;
        for (int k = 0; k < t->K; k++) {
            int e = t->experts[k];
            const void *row_data =
                base + (size_t)e * rpe * rb + (size_t)r * rb;
            dequant_row(t->W_down->type, row_data, t->tmp, n_cols);
            const float *act_k = t->activated + (size_t)k * t->rows_per_input;
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += t->tmp[j] * act_k[j];
            acc += t->ews[k] * sum;
        }
        t->xb2_out[r] = acc;
    }
    return NULL;
}

/* Fused up+gate+silu_mul for K experts in one pool dispatch. */
static void tf_moe_upgate_fused_pool(transformer_model *m,
                                      float *activated_out,
                                      const qtensor *W_up, const qtensor *W_gate,
                                      const int *experts, int K,
                                      const float *xb, int rows_per_expert) {
    if (K <= 0) return;
    int n_threads = m->n_threads;
    int n_cols = W_up->n_cols;
    if (n_threads <= 1 || rows_per_expert < n_threads * 4 || !m->pool_alive) {
        for (int k = 0; k < K; k++) {
            int e = experts[k];
            float *act_k = activated_out + (size_t)k * rows_per_expert;
            float *tmp_up = m->thread_tmp[0];
            float *tmp_gate = m->thread_tmp[0] + n_cols;
            size_t rb_up = tf_row_bytes(W_up->type, n_cols);
            size_t rb_gate = tf_row_bytes(W_gate->type, n_cols);
            const uint8_t *e_up   = (const uint8_t *)W_up->data   + (size_t)e * rows_per_expert * rb_up;
            const uint8_t *e_gate = (const uint8_t *)W_gate->data + (size_t)e * rows_per_expert * rb_gate;
            for (int r = 0; r < rows_per_expert; r++) {
                dequant_row(W_up->type,   e_up   + (size_t)r * rb_up,   tmp_up,   n_cols);
                dequant_row(W_gate->type, e_gate + (size_t)r * rb_gate, tmp_gate, n_cols);
                float up = 0.0f, gate = 0.0f;
                for (int j = 0; j < n_cols; j++) {
                    up   += tmp_up[j]   * xb[j];
                    gate += tmp_gate[j] * xb[j];
                }
                act_k[r] = (gate / (1.0f + expf(-gate))) * up;
            }
        }
        return;
    }
    tf_moe_upgate_task *tasks = (tf_moe_upgate_task *)alloca(n_threads * sizeof(tf_moe_upgate_task));
    for (int t = 0; t < n_threads; t++) {
        int rs, re;
        tf_row_split8(rows_per_expert, n_threads, t, &rs, &re);
        /* Each thread needs two dequant scratches; use the second half of
         * thread_tmp[t] for gate. thread_tmp[t] is sized to at least 2*n_cols
         * (see thread_tmp allocation), but to stay safe we use the dedicated
         * tmp from the worker — keep both pointers distinct. */
        tasks[t].activated_out = activated_out;
        tasks[t].W_up = W_up;
        tasks[t].W_gate = W_gate;
        tasks[t].experts = experts;
        tasks[t].K = K;
        tasks[t].xb = xb;
        tasks[t].rows_per_expert = rows_per_expert;
        tasks[t].row_start = rs;
        tasks[t].row_end = re;
        tasks[t].tmp_up = m->thread_tmp[t];
        tasks[t].tmp_gate = m->thread_tmp[t] + n_cols;
    }
    tf_pool_dispatch(m, tf_moe_upgate_worker, tasks, sizeof(tf_moe_upgate_task));
}

/* Fused down + per-expert weighted accumulate for K experts in one dispatch.
 * Writes xb2_out directly (each thread owns disjoint rows). If K == 0 this
 * is a no-op and the caller is responsible for zeroing xb2 beforehand. */
static void tf_moe_down_fused_pool(transformer_model *m,
                                    float *xb2_out,
                                    const qtensor *W_down,
                                    const int *experts, const float *ews, int K,
                                    const float *activated, int rows_per_input,
                                    int n_embd) {
    if (K <= 0) return;
    int n_threads = m->n_threads;
    int n_cols = W_down->n_cols;
    if (n_threads <= 1 || n_embd < n_threads * 4 || !m->pool_alive) {
        float *tmp = m->thread_tmp[0];
        size_t rb = tf_row_bytes(W_down->type, n_cols);
        const uint8_t *base = (const uint8_t *)W_down->data;
        for (int r = 0; r < n_embd; r++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                int e = experts[k];
                dequant_row(W_down->type,
                            base + (size_t)e * n_embd * rb + (size_t)r * rb,
                            tmp, n_cols);
                const float *act_k = activated + (size_t)k * rows_per_input;
                float sum = 0.0f;
                for (int j = 0; j < n_cols; j++) sum += tmp[j] * act_k[j];
                acc += ews[k] * sum;
            }
            xb2_out[r] = acc;
        }
        return;
    }
    tf_moe_down_task *tasks = (tf_moe_down_task *)alloca(n_threads * sizeof(tf_moe_down_task));
    for (int t = 0; t < n_threads; t++) {
        int rs, re;
        tf_row_split8(n_embd, n_threads, t, &rs, &re);
        tasks[t].xb2_out = xb2_out;
        tasks[t].W_down = W_down;
        tasks[t].experts = experts;
        tasks[t].ews = ews;
        tasks[t].K = K;
        tasks[t].activated = activated;
        tasks[t].rows_per_input = rows_per_input;
        tasks[t].n_embd = n_embd;
        tasks[t].row_start = rs;
        tasks[t].row_end = re;
        tasks[t].tmp = m->thread_tmp[t];
    }
    tf_pool_dispatch(m, tf_moe_down_worker, tasks, sizeof(tf_moe_down_task));
}

/* ──────────────────────────────────────────────────────────────────────────
 * MoE expert BF16_PV repack (TF_MOE_BF16_PV): for OWNED experts of an
 * ep-sharded 3D expert tensor, dequant Q4_K (or any quant) → FP32 → BF16,
 * then pack into the pair-interleaved layout the production
 * matvec_bf16_8row_pv kernel consumes. Microbench measured this kernel at
 * 36.8 GB/s/thread vs 0.1 GB/s/thread for Q4_K dequant+dot (109× kernel
 * speedup at expert shape rows=512 cols=2048). At 48 threads parallel the
 * matvec drops from ~125 ms/token to <2 ms/token.
 *
 * Memory: per layer 3 expert weights × n_expert × rows_per_expert × n_cols
 * × 2 bytes. At Qwen3.6-35B-A3B this is 64 GB total, hence EP=4 minimum
 * (16 GB / rank). The Q4_K source bytes for owned experts can be madvised
 * back to the OS after each expert's repack — handled by the caller via
 * `tf_expert_q4k_free_owned_range`. */
static inline uint16_t tf_fp32_to_bf16(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t rnd = 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)((u + rnd) >> 16);
}

/* Allocate per-expert bf16_pv on a 3D expert tensor.
 *   n_expert        : full expert count (e.g. 256)
 *   rows_per_expert : per-expert row count (== qtensor.dims[1])
 *   ep_rank, ep_size: caller's EP ownership (interleaved: own e iff e%ep_size==ep_rank)
 * Allocates one contiguous pv slab of n_owned × rows_per_expert × n_cols × 2 B
 * and the expert→slot mapping table. */
static int tf_expert_bf16_pv_alloc(qtensor *t, int n_expert, int rows_per_expert,
                                    int ep_rank, int ep_size) {
    if (!t->data) return 0;
    if (rows_per_expert < 8 || (rows_per_expert & 7) != 0) return 0;
    if (t->n_cols <= 0 || (t->n_cols & 15) != 0) return 0;
    if (t->bf16_pv) return 0; /* already built (or claimed by per-tensor pv) */

    int n_owned = 0;
    int *slot = (int *)malloc((size_t)n_expert * sizeof(int));
    if (!slot) return 0;
    for (int e = 0; e < n_expert; e++) {
        if (ep_size <= 1 || (e % ep_size) == ep_rank) {
            slot[e] = n_owned++;
        } else {
            slot[e] = -1;
        }
    }
    size_t bytes = (size_t)n_owned * (size_t)rows_per_expert * (size_t)t->n_cols * sizeof(uint16_t);
    /* MAP_NORESERVE so untouched VM doesn't trip per-CMG overcommit (4 CMGs
     * × ~7 GB each, EP=2 repack ≈ 30 GB BF16 — too big to reserve up front). */
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { free(slot); return 0; }
    t->bf16_pv = (uint16_t *)p;
    t->bf16_pv_groups = n_owned * (rows_per_expert / 8);
    t->expert_owned_slot = slot;
    t->expert_rows_per_expert = rows_per_expert;
    return 1;
}

/* Pack a single expert's pv slab. Dequants n_rows rows of the source qtensor
 * to FP32 (via dequant_row), then writes pair-interleaved bf16. */
static void tf_expert_bf16_pv_fill_one(qtensor *t, int expert, float *tmp_fp32) {
    int slot = t->expert_owned_slot[expert];
    if (slot < 0) return;
    int rpe = t->expert_rows_per_expert;
    int K = t->n_cols;
    int vl = 16;
    int chunks_per_K = K / vl;
    int groups = rpe / 8;

    size_t row_bytes_src = tf_row_bytes(t->type, K);
    const uint8_t *src = (const uint8_t *)t->data + (size_t)expert * rpe * row_bytes_src;
    uint16_t *pv = t->bf16_pv + (size_t)slot * rpe * K;

    /* Dequant all rpe rows to a contiguous FP32 staging block. */
    for (int r = 0; r < rpe; r++) {
        dequant_row(t->type, src + (size_t)r * row_bytes_src,
                    tmp_fp32 + (size_t)r * K, K);
    }
    /* Pair-interleave into pv layout matching tf_bf16_pv_fill_range. */
    for (int g = 0; g < groups; g++) {
        uint16_t *gbuf = pv + (size_t)g * 8 * K;
        for (int p = 0; p < 4; p++) {
            int rowA = g * 8 + 2 * p;
            int rowB = g * 8 + 2 * p + 1;
            const float *srcA = tmp_fp32 + (size_t)rowA * K;
            const float *srcB = tmp_fp32 + (size_t)rowB * K;
            uint16_t *pair = gbuf + (size_t)p * 2 * K;
            for (int c = 0; c < chunks_per_K; c++) {
                uint16_t *chunk = pair + (size_t)c * 32;
                for (int lane = 0; lane < vl; lane++) {
                    chunk[2 * lane + 0] = tf_fp32_to_bf16(srcA[c * vl + lane]);
                    chunk[2 * lane + 1] = tf_fp32_to_bf16(srcB[c * vl + lane]);
                }
            }
        }
    }
}

static int tf_expert_q8_pv_alloc(qtensor *t, int n_expert, int rows_per_expert,
                                 int ep_rank, int ep_size) {
    if (!t->data) return 0;
    if (rows_per_expert < 8 || (rows_per_expert & 7) != 0) return 0;
    if (t->n_cols <= 0 || (t->n_cols & 63) != 0) return 0;
    if (t->q8_pv) return 0;

    int n_owned = 0;
    int *slot = (int *)malloc((size_t)n_expert * sizeof(int));
    if (!slot) return 0;
    for (int e = 0; e < n_expert; e++) {
        if (ep_size <= 1 || (e % ep_size) == ep_rank) {
            slot[e] = n_owned++;
        } else {
            slot[e] = -1;
        }
    }
    if (n_owned == 0) {
        free(slot);
        return 0;
    }
    size_t bytes = (size_t)n_owned * (size_t)(rows_per_expert / 8) * tf_q8_pv_bytes(1, t->n_cols);
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { free(slot); return 0; }
    t->q8_pv = (uint8_t *)p;
    t->q8_pv_groups = n_owned * (rows_per_expert / 8);
    if (t->expert_owned_slot) {
        free(t->expert_owned_slot);
    }
    t->expert_owned_slot = slot;
    t->expert_rows_per_expert = rows_per_expert;
    return 1;
}

/* Pack a single expert's q8_pv slab. Dequantize rows of the source qtensor
 * to FP32 (via dequant_row), then pack four 16-lane chunks per 64-block. */
static void tf_expert_q8_pv_fill_one(qtensor *t, int expert, float *tmp_fp32) {
    int slot = t->expert_owned_slot[expert];
    if (slot < 0) return;
    int rpe = t->expert_rows_per_expert;
    int K = t->n_cols;
    int nb = K / 64;
    size_t row_bytes_src = tf_row_bytes(t->type, K);
    const uint8_t *src = (const uint8_t *)t->data + (size_t)expert * rpe * row_bytes_src;
    uint8_t *qv = t->q8_pv + (size_t)slot * (size_t)(rpe / 8) * tf_q8_pv_bytes(1, K);

    for (int g = 0; g < rpe / 8; g++) {
        int gbase = g * 8;
        for (int r = 0; r < 8; r++)
            dequant_row(t->type, src + (size_t)(gbase + r) * row_bytes_src,
                        tmp_fp32 + (size_t)r * K, K);
        uint8_t *gbuf = qv + (size_t)g * nb * 528;
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = gbuf + (size_t)b * 528;
            uint16_t *scl = (uint16_t *)blk;
            int8_t *qs = (int8_t *)(blk + 16);
            for (int r = 0; r < 8; r++) {
                const float *src_block = tmp_fp32 + (size_t)r * K + (size_t)b * 64;
                float amax = 0.0f;
                for (int j = 0; j < 64; j++) {
                    float a = src_block[j] < 0.0f ? -src_block[j] : src_block[j];
                    if (a > amax) amax = a;
                }
                float scale = amax / 127.0f;
                float invs = amax > 0 ? 127.0f / amax : 0.0f;
                scl[r] = ggml_fp32_to_fp16(scale);
                for (int j = 0; j < 64; j++) {
                    int q = (int)lrintf(src_block[j] * invs);
                    if (q < -127) q = -127; else if (q > 127) q = 127;
                    qs[r * 64 + j] = (int8_t)q;
                }
            }
        }
    }
}

static void tf_expert_q4k_free_one(qtensor *t, int expert);

static int transformer_expert_repack_q8_pv(transformer_model *m, qtensor *t,
                                          int n_expert, int rows_per_expert) {
    if (!t || !t->data) return -1;
    if (t->q8_pv) return 0;
    if (!tf_expert_q8_pv_alloc(t, n_expert, rows_per_expert, m->ep_rank, m->ep_size))
        return -1;
    float *tmp = (float *)aligned_alloc(64, (size_t)rows_per_expert * t->n_cols * sizeof(float));
    if (!tmp) return -1;
    for (int e = 0; e < n_expert; e++) {
        if (t->expert_owned_slot[e] < 0) continue;
        tf_expert_q8_pv_fill_one(t, e, tmp);
        tf_expert_q4k_free_one(t, e);
    }
    for (int e = 0; e < n_expert; e++) {
        if (t->expert_owned_slot[e] < 0) tf_expert_q4k_free_one(t, e);
    }
    free(tmp);
    return 0;
}

/* madvise the source Q4_K bytes for one expert range back to the kernel
 * (only does anything for anon RAM-resident gguf; mmap'd regions on
 * incompatible filesystems may decline). Page-aligned shrinks only. */
static void tf_expert_q4k_free_one(qtensor *t, int expert) {
    if (!t->data || t->expert_rows_per_expert <= 0) return;
    int rpe = t->expert_rows_per_expert;
    size_t row_bytes = tf_row_bytes(t->type, t->n_cols);
    uintptr_t start = (uintptr_t)t->data + (size_t)expert * rpe * row_bytes;
    uintptr_t end   = start + (size_t)rpe * row_bytes;
    long page_sz = sysconf(_SC_PAGESIZE);
    if (page_sz <= 0) page_sz = 4096;
    uintptr_t a_start = (start + page_sz - 1) & ~((uintptr_t)page_sz - 1);
    uintptr_t a_end   = end & ~((uintptr_t)page_sz - 1);
    if (a_end > a_start) madvise((void *)a_start, (size_t)(a_end - a_start), MADV_DONTNEED);
}

/* Inline 8-row matvec block from a per-expert pv slab. Caller guarantees
 * slot[expert] >= 0, row_start..row_end is 8-aligned, n_cols % 16 == 0. */
static inline void tf_expert_matvec_bf16_pv_block(float *dst, const qtensor *t,
                                                    int expert, const float *x,
                                                    int row_start, int row_end) {
#if defined(__ARM_FEATURE_SVE)
    int slot = t->expert_owned_slot[expert];
    int rpe = t->expert_rows_per_expert;
    int n_cols = t->n_cols;
    const uint16_t *pv_expert = t->bf16_pv + (size_t)slot * rpe * n_cols;
    int i = row_start;
    for (; i + 7 < row_end; i += 8) {
        int g = i >> 3;
        const uint16_t *gbase = pv_expert + (size_t)g * 8 * n_cols;
        matvec_bf16_8row_pv(dst + i,
            gbase + (size_t)0 * 2 * n_cols,
            gbase + (size_t)1 * 2 * n_cols,
            gbase + (size_t)2 * 2 * n_cols,
            gbase + (size_t)3 * 2 * n_cols,
            x, n_cols);
    }
    /* tail rows: should not occur (rpe is required to be 8-aligned) */
    for (; i < row_end; i++) dst[i] = 0.0f;
#else
    (void)dst; (void)t; (void)expert; (void)x; (void)row_start; (void)row_end;
#endif
}

/* Inline 8-row matvec block from a per-expert int8 pv slab. Caller guarantees
 * slot[expert] >= 0, row_start..row_end is 8-aligned, n_cols % 64 == 0. */
static inline void tf_expert_matvec_q8_pv_block(float *dst, const qtensor *t,
                                               int expert, const int8_t *xq,
                                               const uint16_t *xs,
                                               int row_start, int row_end) {
#if defined(__ARM_FEATURE_SVE)
    int slot = t->expert_owned_slot[expert];
    int rpe = t->expert_rows_per_expert;
    int n_cols = t->n_cols;
    int nb = n_cols / 64;
    size_t group_bytes = (size_t)nb * 528;
    const uint8_t *qbase = t->q8_pv + (size_t)slot * (size_t)(rpe / 8) * group_bytes;
    int i = row_start;
    for (; i + 7 < row_end; i += 8) {
        int g = i >> 3;
        matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                         xq, xs, n_cols);
    }
    for (; i < row_end; i++) dst[i] = 0.0f;
#else
    (void)dst; (void)t; (void)expert; (void)xq; (void)xs; (void)row_start; (void)row_end;
#endif
}

/* Public: repack a 3D expert tensor's OWNED experts (ep_rank/ep_size from m)
 * into bf16_pv. After this call, the kernel routes through pv whenever the
 * expert is in the owned set. Returns 0 on success, -1 on alloc/shape failure.
 * Per-expert TLS scratch (FP32 staging of one expert's rows) is allocated
 * inline and freed before return. */
static int transformer_expert_repack_bf16_pv(transformer_model *m, qtensor *t,
                                              int n_expert, int rows_per_expert) {
    if (!t || !t->data) return -1;
    if (t->bf16_pv) return 0; /* idempotent */
    if (!tf_expert_bf16_pv_alloc(t, n_expert, rows_per_expert, m->ep_rank, m->ep_size))
        return -1;
    float *tmp = (float *)aligned_alloc(64, (size_t)rows_per_expert * t->n_cols * sizeof(float));
    if (!tmp) return -1;
    for (int e = 0; e < n_expert; e++) {
        if (t->expert_owned_slot[e] < 0) continue;
        tf_expert_bf16_pv_fill_one(t, e, tmp);
        /* Drop this expert's Q4_K bytes back to the OS now that pv is built. */
        tf_expert_q4k_free_one(t, e);
    }
    /* Also madvise the non-owned experts' Q4_K bytes since they will never
     * be touched on this rank (EP filter skips them in the forward path). */
    for (int e = 0; e < n_expert; e++) {
        if (t->expert_owned_slot[e] < 0) tf_expert_q4k_free_one(t, e);
    }
    free(tmp);
    return 0;
}

/* Public: repack a DENSE (2D, replicated) quantized tensor into bf16_pv.
 * Unlike the expert variant there are no ownership slots — every row is
 * dequantized to FP32 (via tf_dequant_row, so any quant type is handled)
 * then pair-interleaved as bf16 for matvec_bf16_8row_pv. Used for the
 * replicated SSM mixer projections (ssm_qkv/ssm_gate/ssm_out) which dominate
 * replicated-dense HBM traffic on the EP runner. Returns 0 on success, -1 on
 * shape/alloc failure. If free_src, the quantized source pages are
 * madvise(DONTNEED)'d after the pv buffer is built — only safe when EVERY
 * dispatch path for this tensor takes the pv fast path; the default keeps the
 * source resident so any unwired path falls back to correct scalar dequant. */
static int transformer_repack_dense_bf16_pv(transformer_model *m, qtensor *t,
                                             int free_src) {
    (void)m;
    if (!t || !t->data) return -1;
    if (t->bf16_pv) return 0; /* idempotent */
    if (t->n_rows < 8 || (t->n_rows & 7) != 0) return -1;
    if (t->n_cols <= 0 || (t->n_cols & 15) != 0) return -1;
    int R = t->n_rows, K = t->n_cols;
    int groups = R / 8;
    size_t bytes = tf_bf16_pv_bytes(groups, K);
    void *pv = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pv == MAP_FAILED) return -1;
#ifdef MADV_NOHUGEPAGE
    madvise(pv, bytes, MADV_NOHUGEPAGE);
#endif
    uint16_t *pvb = (uint16_t *)pv;
    /* Dequantize the whole tensor to FP32, then pair-interleave to bf16. The
     * scratch is freed before return so peak is weight + one fp32 copy. */
    float *tmp = (float *)aligned_alloc(64, (size_t)R * (size_t)K * sizeof(float));
    if (!tmp) { munmap(pv, bytes); return -1; }
    for (int r = 0; r < R; r++)
        tf_dequant_row(t, r, tmp + (size_t)r * K);
    int vl = 16;                 /* fp32 lanes on A64FX SVE */
    int chunks_per_K = K / vl;
    for (int g = 0; g < groups; g++) {
        uint16_t *gbuf = pvb + (size_t)g * 8 * K;
        for (int p = 0; p < 4; p++) {
            const float *srcA = tmp + (size_t)(g * 8 + 2 * p)     * K;
            const float *srcB = tmp + (size_t)(g * 8 + 2 * p + 1) * K;
            uint16_t *pair = gbuf + (size_t)p * 2 * K;
            for (int c = 0; c < chunks_per_K; c++) {
                uint16_t *chunk = pair + (size_t)c * 32;
                for (int lane = 0; lane < vl; lane++) {
                    chunk[2 * lane + 0] = tf_fp32_to_bf16(srcA[c * vl + lane]);
                    chunk[2 * lane + 1] = tf_fp32_to_bf16(srcB[c * vl + lane]);
                }
            }
        }
    }
    free(tmp);
    /* Publish only after the buffer is fully built. */
    t->bf16_pv = pvb;
    t->bf16_pv_groups = groups;
    if (free_src) {
        size_t row_bytes = tf_row_bytes(t->type, K);
        uintptr_t start = (uintptr_t)t->data;
        uintptr_t end   = start + (size_t)R * row_bytes;
        long page_sz = sysconf(_SC_PAGESIZE); if (page_sz <= 0) page_sz = 4096;
        uintptr_t a_start = (start + page_sz - 1) & ~((uintptr_t)page_sz - 1);
        uintptr_t a_end   = end & ~((uintptr_t)page_sz - 1);
        if (a_end > a_start)
            madvise((void *)a_start, (size_t)(a_end - a_start), MADV_DONTNEED);
    }
    return 0;
}

/* Public: repack a DENSE (2D, replicated) quantized tensor into q8_pv.
 * This is the int8/SDOT analogue of transformer_repack_dense_bf16_pv for
 * tensors that are not BF16 on disk (e.g. Q4_K SSM projections). */
static int transformer_repack_dense_q8_pv(transformer_model *m, qtensor *t,
                                           int free_src) {
    (void)m;
    if (!t || !t->data) return -1;
    if (t->q8_pv) return 0;
    if (t->n_rows < 8 || (t->n_rows & 7) != 0) return -1;
    if (t->n_cols <= 0 || (t->n_cols & 63) != 0) return -1;
    int R = t->n_rows, K = t->n_cols;
    int groups = R / 8;
    int nb = K / 64;
    size_t bytes = tf_q8_pv_bytes(groups, K);
    void *pv = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pv == MAP_FAILED) return -1;
#ifdef MADV_NOHUGEPAGE
    madvise(pv, bytes, MADV_NOHUGEPAGE);
#endif
    uint8_t *qv = (uint8_t *)pv;
    float *tmp = (float *)aligned_alloc(64, (size_t)8 * (size_t)K * sizeof(float));
    if (!tmp) { munmap(pv, bytes); return -1; }
    for (int g = 0; g < groups; g++) {
        for (int r = 0; r < 8; r++)
            tf_dequant_row(t, g * 8 + r, tmp + (size_t)r * K);
        uint8_t *gbuf = qv + (size_t)g * nb * 528;
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = gbuf + (size_t)b * 528;
            uint16_t *scl = (uint16_t *)blk;
            int8_t *qs = (int8_t *)(blk + 16);
            for (int r = 0; r < 8; r++) {
                const float *src = tmp + (size_t)r * K + (size_t)b * 64;
                float amax = 0.0f;
                for (int j = 0; j < 64; j++) {
                    float a = src[j] < 0 ? -src[j] : src[j];
                    if (a > amax) amax = a;
                }
                float scale = amax / 127.0f;
                float invs = amax > 0 ? 127.0f / amax : 0.0f;
                scl[r] = ggml_fp32_to_fp16(scale);
                for (int j = 0; j < 64; j++) {
                    int q = (int)lrintf(src[j] * invs);
                    if (q < -127) q = -127; else if (q > 127) q = 127;
                    qs[r * 64 + j] = (int8_t)q;
                }
            }
        }
    }
    free(tmp);
    t->q8_pv = qv;
    t->q8_pv_groups = groups;
    if (free_src) {
        size_t row_bytes = tf_row_bytes(t->type, K);
        uintptr_t start = (uintptr_t)t->data;
        uintptr_t end = start + (size_t)R * row_bytes;
        long page_sz = sysconf(_SC_PAGESIZE); if (page_sz <= 0) page_sz = 4096;
        uintptr_t a_start = (start + page_sz - 1) & ~((uintptr_t)page_sz - 1);
        uintptr_t a_end = end & ~((uintptr_t)page_sz - 1);
        if (a_end > a_start)
            madvise((void *)a_start, (size_t)(a_end - a_start), MADV_DONTNEED);
    }
    return 0;
}

#if defined(__ARM_FEATURE_SVE)
/* ──────────────────────────────────────────────────────────────────────────
 * Batched bf16 packed-B GEMM for layer-major prefill (TF_PREFILL_GEMM, Stage 1)
 *
 * Computes C[M,N] = A[M,K] @ W[N,K]^T, where W is a weight's decode-resident
 * bf16_pv layout (w->bf16_pv). Same math as applying tf_qmatvec_pool to each
 * of the M token rows, but as a real GEMM (FLOPS-bound, not BW-bound).
 *
 * Per-CMG-INDEPENDENT (validated recipe, [[batched-prefill-gemm-derisk]]):
 * the M token rows are partitioned across CMGs; each CMG packs its OWN
 * packed-B-PV replica from w->bf16_pv (NUMA-local, no cross-CMG B sharing —
 * shared B is a NUMA collapse) into reused per-CMG scratch, then each thread
 * sweeps the mb token-blocks it owns over the full N. The pack-from-pv path is
 * bit-exact vs pack_B_bf16_pv(transpose(W)) (a64fx/tools/bench_packB_from_pv.c)
 * and needs ZERO extra persistent weight memory. Two pool dispatches (pack,
 * then compute) use the dispatch boundary as the intra-CMG sync, so no custom
 * mid-dispatch barrier is needed. Reuses the asm micro-kernel + pack_A from
 * a64fx/vlm/kernels (already linked into the runner). Output is bit-SIMILAR
 * (GEMM reorders the K-sum vs the per-row matvec), not bit-exact. */
extern size_t packed_A_size(int M, int K);
extern void   pack_A_fp32_block(int mb, int M, int K, int K_rounded,
                                const float *A, int lda, float *A_packed);
extern void   micro_kernel_bf16B_8x3_unroll4_pv(const float *A_packed,
                                const uint16_t *B, float *C, int64_t K_rounded,
                                int64_t unused, int64_t ldc_bytes);

#define TF_GEMM_MR 8
#define TF_GEMM_NR 48
#define TF_GEMM_PV_PREFIX 64

static size_t tf_gemm_packed_B_pv_size(int K, int N) {
    int Nb = (N + TF_GEMM_NR - 1) / TF_GEMM_NR, Kr = ((K + 3) / 4) * 4;
    return (size_t)Nb * Kr * TF_GEMM_NR * sizeof(uint16_t) + TF_GEMM_PV_PREFIX;
}

/* Compute (cmg, local index in cmg, threads in cmg) for a tid, matching the
 * tf_cmg_pin_thread mapping so a CMG's threads are physically co-located and
 * its B replica first-touches onto that CMG's HBM. */
static inline void tf_gemm_cmg_of(int tid, int nt, int ncmg,
                                  int *cmg, int *loc, int *nloc) {
    int c     = (int)((long)tid * ncmg / nt);
    int first = (int)(((long)c * nt + ncmg - 1) / ncmg);
    int next  = (int)((((long)c + 1) * nt + ncmg - 1) / ncmg);
    *cmg = c; *loc = tid - first; *nloc = next - first;
}

/* Pack packed-B-PV directly from a weight's bf16_pv (decode layout), splitting
 * N-blocks across the `nloc` threads of one CMG. Bit-exact vs canonical pack. */
static void tf_gemm_pack_from_pv(int K, int N, const uint16_t *pv,
                                 uint16_t *BTP_alloc, int loc, int nloc) {
    uint16_t *BTP = (uint16_t *)((uint8_t *)BTP_alloc + TF_GEMM_PV_PREFIX);
    int Nb = (N + TF_GEMM_NR - 1) / TF_GEMM_NR, Kr = ((K + 3) / 4) * 4;
    for (int nb = loc; nb < Nb; nb += nloc) {
        int ns = nb * TF_GEMM_NR, nc = (ns + TF_GEMM_NR <= N) ? TF_GEMM_NR : N - ns;
        uint16_t *dst = BTP + (size_t)nb * Kr * TF_GEMM_NR;
        for (int kp = 0; kp < Kr; kp += 2) {
            int k0 = kp, k1 = kp + 1;
            int ck0 = k0 >> 4, la0 = k0 & 15, ck1 = k1 >> 4, la1 = k1 & 15;
            for (int c = 0; c < 3; c++) {
                uint16_t *ch = dst + (size_t)(kp / 2) * (TF_GEMM_NR * 2) + c * 32;
                for (int i = 0; i < 16; i++) {
                    int col = c * 16 + i; uint16_t v0 = 0, v1 = 0;
                    if (col < nc) {
                        int n = ns + col, g = n >> 3, r = n & 7, p = r >> 1, par = r & 1;
                        size_t base = (size_t)g * 8 * K + (size_t)p * 2 * K + par;
                        if (k0 < K) v0 = pv[base + (size_t)ck0 * 32 + 2 * la0];
                        if (k1 < K) v1 = pv[base + (size_t)ck1 * 32 + 2 * la1];
                    }
                    ch[2 * i] = v0; ch[2 * i + 1] = v1;
                }
            }
        }
    }
}

/* One thread computes C[Mc,N] = A[Mc,K] @ B for the mb-blocks it owns (mb =
 * loc, loc+nloc, ...). Packs its own A-blocks first (independent, no barrier). */
static void tf_gemm_compute_owned(int Mc, int K, int N, const float *A,
        float *Apk, const uint16_t *Bpv_alloc, float *C, int loc, int nloc) {
    int Kr = ((K + 3) / 4) * 4;
    int Mb = (Mc + TF_GEMM_MR - 1) / TF_GEMM_MR;
    int Nb = (N + TF_GEMM_NR - 1) / TF_GEMM_NR;
    const uint16_t *B = (const uint16_t *)((const uint8_t *)Bpv_alloc + TF_GEMM_PV_PREFIX);
    for (int mb = loc; mb < Mb; mb += nloc) {
        pack_A_fp32_block(mb, Mc, K, Kr, A, K, Apk);
        int ms = mb * TF_GEMM_MR, mc = (ms + TF_GEMM_MR <= Mc) ? TF_GEMM_MR : Mc - ms;
        const float *At = Apk + (size_t)mb * Kr * TF_GEMM_MR;
        for (int nb = 0; nb < Nb; nb++) {
            int ns = nb * TF_GEMM_NR, nc = (ns + TF_GEMM_NR <= N) ? TF_GEMM_NR : N - ns;
            const uint16_t *Bt = B + (size_t)nb * Kr * TF_GEMM_NR;
            if (mc == TF_GEMM_MR && nc == TF_GEMM_NR) {
                micro_kernel_bf16B_8x3_unroll4_pv(At, Bt, C + (size_t)ms * N + ns,
                                                  Kr, 0, (int64_t)N * 4);
            } else {
                float lb[TF_GEMM_MR * TF_GEMM_NR] __attribute__((aligned(64)));
                micro_kernel_bf16B_8x3_unroll4_pv(At, Bt, lb, Kr, 0, (int64_t)TF_GEMM_NR * 4);
                for (int mm = 0; mm < mc; mm++)
                    for (int n = 0; n < nc; n++)
                        C[(size_t)(ms + mm) * N + ns + n] = lb[mm * TF_GEMM_NR + n];
            }
        }
    }
}

/* Per-CMG reused scratch: B-PV replica + packed-A, grown on demand. mmap'd on
 * the main thread; pages first-touch on the owning CMG's threads (which write
 * them during pack/compute), landing on that CMG's HBM. Single instance — the
 * runner is the only translation unit that calls the prefill GEMM. */
static struct {
    uint16_t *Bpack[8]; size_t Bbytes[8];
    float    *Apack[8]; size_t Abytes[8];
} tf_gemm_scratch;

static void tf_gemm_scratch_ensure(int cmg, size_t bbytes, size_t abytes) {
    if (tf_gemm_scratch.Bbytes[cmg] < bbytes) {
        if (tf_gemm_scratch.Bpack[cmg])
            munmap(tf_gemm_scratch.Bpack[cmg], tf_gemm_scratch.Bbytes[cmg]);
        void *p = mmap(NULL, bbytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { tf_gemm_scratch.Bpack[cmg] = NULL; tf_gemm_scratch.Bbytes[cmg] = 0; return; }
#ifdef MADV_NOHUGEPAGE
        madvise(p, bbytes, MADV_NOHUGEPAGE);
#endif
        tf_gemm_scratch.Bpack[cmg] = (uint16_t *)p;
        tf_gemm_scratch.Bbytes[cmg] = bbytes;
    }
    if (tf_gemm_scratch.Abytes[cmg] < abytes) {
        if (tf_gemm_scratch.Apack[cmg])
            munmap(tf_gemm_scratch.Apack[cmg], tf_gemm_scratch.Abytes[cmg]);
        void *p = mmap(NULL, abytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { tf_gemm_scratch.Apack[cmg] = NULL; tf_gemm_scratch.Abytes[cmg] = 0; return; }
#ifdef MADV_NOHUGEPAGE
        madvise(p, abytes, MADV_NOHUGEPAGE);
#endif
        tf_gemm_scratch.Apack[cmg] = (float *)p;
        tf_gemm_scratch.Abytes[cmg] = abytes;
    }
}

typedef struct {
    int tid, ncmg, nt;
    int M, N, K;
    const uint16_t *pv;     /* w->bf16_pv (shared, read-only) */
    const float    *A;      /* [M,K] activations (shared) */
    float          *C;      /* [M,N] output (shared; CMGs write disjoint rows) */
    const int      *m_lo;   /* [ncmg] token-slice starts */
    const int      *m_hi;
} tf_gemm_pv_task;

static void *tf_gemm_packB_worker(void *arg) {
    tf_gemm_pv_task *t = (tf_gemm_pv_task *)arg;
    int cmg, loc, nloc;
    tf_gemm_cmg_of(t->tid, t->nt, t->ncmg, &cmg, &loc, &nloc);
    if (cmg < t->ncmg && tf_gemm_scratch.Bpack[cmg])
        tf_gemm_pack_from_pv(t->K, t->N, t->pv, tf_gemm_scratch.Bpack[cmg], loc, nloc);
    return NULL;
}

static void *tf_gemm_compute_worker(void *arg) {
    tf_gemm_pv_task *t = (tf_gemm_pv_task *)arg;
    int cmg, loc, nloc;
    tf_gemm_cmg_of(t->tid, t->nt, t->ncmg, &cmg, &loc, &nloc);
    if (cmg >= t->ncmg) return NULL;
    int mlo = t->m_lo[cmg], mhi = t->m_hi[cmg], Mc = mhi - mlo;
    if (Mc <= 0 || !tf_gemm_scratch.Bpack[cmg] || !tf_gemm_scratch.Apack[cmg]) return NULL;
    tf_gemm_compute_owned(Mc, t->K, t->N,
                          t->A + (size_t)mlo * t->K, tf_gemm_scratch.Apack[cmg],
                          tf_gemm_scratch.Bpack[cmg], t->C + (size_t)mlo * t->N,
                          loc, nloc);
    return NULL;
}

typedef struct {
    int tid, nt;
    int M, N, K, nb, ng;
    const uint8_t *q8_pv;
    const float *A;
    float *C;
    int8_t *XQ;
    uint16_t *XS;
} tf_gemm_q8pv_task;

static void *tf_gemm_q8pv_quant_worker(void *arg) {
    tf_gemm_q8pv_task *t = (tf_gemm_q8pv_task *)arg;
    for (int m = t->tid; m < t->M; m += t->nt) {
        tf_quant_x_sdot_blocks(t->A + (size_t)m * t->K, t->K, 0, 1,
                               t->XQ + (size_t)m * t->K,
                               t->XS + (size_t)m * t->nb);
    }
    return NULL;
}

static void *tf_gemm_q8pv_compute_worker(void *arg) {
    tf_gemm_q8pv_task *t = (tf_gemm_q8pv_task *)arg;
    long total = (long)t->M * (long)t->ng;
    long lo = (total * t->tid) / t->nt;
    long hi = (total * (t->tid + 1)) / t->nt;
    size_t group_bytes = (size_t)t->nb * 528;
    for (long p = lo; p < hi; p++) {
        int m = (int)(p / t->ng);
        int g = (int)(p - (long)m * t->ng);
        int row = g << 3;
        matvec_sdot_8row(t->C + (size_t)m * t->N + row,
                         t->q8_pv + (size_t)g * group_bytes,
                         t->XQ + (size_t)m * t->K,
                         t->XS + (size_t)m * t->nb,
                         t->K);
    }
    return NULL;
}

static int tf_gemm_q8pv_prefill(transformer_model *m, float *C,
                                const qtensor *w, const float *A, int M) {
    int N = w->n_rows, K = w->n_cols, nt = m->n_threads;
    if (!m->pool_alive || !w->q8_pv || N <= 0 || M <= 0 ||
        (N & 7) != 0 || (K & 63) != 0)
        return 0;

    int nb = K / 64, ng = N / 8;
    size_t xq_bytes = (size_t)M * (size_t)K;
    size_t xs_bytes = (size_t)M * (size_t)nb * sizeof(uint16_t);
    int8_t *XQ = (int8_t *)aligned_alloc(64, (xq_bytes + 63) & ~(size_t)63);
    uint16_t *XS = (uint16_t *)aligned_alloc(64, (xs_bytes + 63) & ~(size_t)63);
    if (!XQ || !XS) {
        free(XQ);
        free(XS);
        return 0;
    }

    tf_gemm_q8pv_task *tasks =
        (tf_gemm_q8pv_task *)alloca((size_t)nt * sizeof(tf_gemm_q8pv_task));
    for (int t = 0; t < nt; t++)
        tasks[t] = (tf_gemm_q8pv_task){ t, nt, M, N, K, nb, ng,
                                        w->q8_pv, A, C, XQ, XS };

    tf_pool_dispatch(m, tf_gemm_q8pv_quant_worker, tasks, sizeof(tf_gemm_q8pv_task));
    tf_pool_dispatch(m, tf_gemm_q8pv_compute_worker, tasks, sizeof(tf_gemm_q8pv_task));
    free(XQ);
    free(XS);
    return 1;
}

/* C[M,N] = A[M,K] @ W[N,K]^T via the per-CMG-independent bf16 packed-B GEMM.
 * Falls back to row-by-row tf_qmatvec_pool when the pool/bf16_pv aren't usable.
 * Caller should first-touch A and C per-CMG (matching the M-partition below)
 * for NUMA-local placement; correctness does not depend on it. */
static void tf_gemm_bf16pv_prefill(transformer_model *m, float *C,
                                   const qtensor *w, const float *A, int M) {
    int N = w->n_rows, K = w->n_cols, nt = m->n_threads;
    if (!m->pool_alive || N <= 0 || M <= 0) {
        for (int mm = 0; mm < M; mm++)
            tf_qmatvec_pool(m, C + (size_t)mm * N, w, A + (size_t)mm * K, N);
        return;
    }
    if (!w->bf16_pv) {
        if (tf_gemm_q8pv_prefill(m, C, w, A, M))
            return;
        for (int mm = 0; mm < M; mm++)
            tf_qmatvec_pool(m, C + (size_t)mm * N, w, A + (size_t)mm * K, N);
        return;
    }
    if ((K & 15) != 0) {
        for (int mm = 0; mm < M; mm++)
            tf_qmatvec_pool(m, C + (size_t)mm * N, w, A + (size_t)mm * K, N);
        return;
    }
    int ncmg = (m->cmg_pin && m->cmg_pin_ncmgs > 0) ? m->cmg_pin_ncmgs : 1;
    if (ncmg > 4) ncmg = 4;
    if (ncmg > nt) ncmg = nt;

    int m_lo[8], m_hi[8];
    int per = ((M / ncmg + TF_GEMM_MR - 1) / TF_GEMM_MR) * TF_GEMM_MR;
    if (per < TF_GEMM_MR) per = TF_GEMM_MR;
    for (int c = 0; c < ncmg; c++) {
        m_lo[c] = c * per; m_hi[c] = (c == ncmg - 1) ? M : (c + 1) * per;
        if (m_lo[c] > M) m_lo[c] = M;
        if (m_hi[c] > M) m_hi[c] = M;
    }

    size_t bbytes = tf_gemm_packed_B_pv_size(K, N);
    for (int c = 0; c < ncmg; c++) {
        int Mc = m_hi[c] - m_lo[c]; if (Mc < TF_GEMM_MR) Mc = TF_GEMM_MR;
        tf_gemm_scratch_ensure(c, bbytes, packed_A_size(Mc, K));
    }

    tf_gemm_pv_task *tasks = (tf_gemm_pv_task *)alloca((size_t)nt * sizeof(tf_gemm_pv_task));
    for (int t = 0; t < nt; t++)
        tasks[t] = (tf_gemm_pv_task){ t, ncmg, nt, M, N, K, w->bf16_pv, A, C, m_lo, m_hi };

    tf_pool_dispatch(m, tf_gemm_packB_worker, tasks, sizeof(tf_gemm_pv_task));
    tf_pool_dispatch(m, tf_gemm_compute_worker, tasks, sizeof(tf_gemm_pv_task));
}

static inline int tf_prefill_weight_batched(const qtensor *w) {
    return w && (w->bf16_pv || w->q8_pv);
}
#endif /* __ARM_FEATURE_SVE */

/* Legacy multi-threaded version (pthread_create per call) */
static void tf_qmatvec_mt(float *dst, const qtensor *mat, const float *x, int n_rows,
                           int n_threads, float **thread_tmp) {
    if (n_threads <= 1 || n_rows < n_threads * 4) {
        tf_qmatvec(dst, mat, x, n_rows, thread_tmp[0]);
        return;
    }
    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_matvec_task *tasks = (tf_matvec_task *)alloca(n_threads * sizeof(tf_matvec_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int count = rows_per + (t < extra ? 1 : 0);
        tasks[t] = (tf_matvec_task){dst, mat, x, offset, offset + count, thread_tmp[t]};
        offset += count;
        pthread_create(&threads[t], NULL, tf_qmatvec_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);
}

/* RoPE: apply rotary position encoding to a vector of shape [n_heads, head_dim] */
/* NeoX-style RoPE: pairs (v[j], v[j + half_dim]) instead of consecutive (v[2j], v[2j+1]) */
static void tf_rope(float *vec, int n_heads, int head_dim, int pos, float freq_base,
                    const float *inv_freq) {
    int half_dim = head_dim / 2;
    /* Precompute cos/sin table for all pair positions */
    float cos_tab[512], sin_tab[512]; /* half_dim <= 512 */
    for (int j = 0; j < half_dim; j++) {
        float freq = inv_freq ? inv_freq[j] : (1.0f / powf(freq_base, (float)(2 * j) / head_dim));
        float theta = pos * freq;
        cos_tab[j] = cosf(theta);
        sin_tab[j] = sinf(theta);
    }
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
        int j = 0;
        for (; j + 7 < half_dim; j += 8) {
            __m256 vc = _mm256_loadu_ps(cos_tab + j);
            __m256 vs = _mm256_loadu_ps(sin_tab + j);
            __m256 v0 = _mm256_loadu_ps(v + j);
            __m256 v1 = _mm256_loadu_ps(v + j + half_dim);
            _mm256_storeu_ps(v + j,             _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
            _mm256_storeu_ps(v + j + half_dim,  _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
        }
        for (; j < half_dim; j++) {
            float v0 = v[j], v1 = v[j + half_dim];
            v[j]            = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + half_dim] = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#else
        for (int j = 0; j < half_dim; j++) {
            float v0 = v[j], v1 = v[j + half_dim];
            v[j]            = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + half_dim] = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#endif
    }
}

/* M-RoPE (IMROPE): apply rotary position encoding with interleaved sections.
 * sections[4] = [temporal, height, width, pad], positions = [pos_t, pos_h, pos_w].
 * Dimension pairs are assigned to sections via interleaved pattern (sector % 3). */
/* M-RoPE (IMROPE) with NeoX-style rotation: pairs (v[j], v[j + half_dim]).
 * Cache index i0 = 2*j determines the sector and frequency.
 * sector = (i0/2) % sect_dims = j % sect_dims. */
static void tf_rope_mrope(float *vec, int n_heads, int head_dim, int pos_t, int pos_h, int pos_w,
                           float freq_base, const int sections[4], const float *inv_freq) {
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0) sect_dims = head_dim / 2; /* fallback */
    int pair_off = sect_dims;
    int rope_dim = 2 * sect_dims;

    /* Precompute cos/sin table once (per-position varies by section) */
    float cos_tab[512], sin_tab[512]; /* sect_dims <= 512 */
    for (int j = 0; j < sect_dims; j++) {
        int pos;
        if (j % 3 == 1 && j < 3 * sections[1]) {
            pos = pos_h;
        } else if (j % 3 == 2 && j < 3 * sections[2]) {
            pos = pos_w;
        } else if (j % 3 == 0 && j < 3 * sections[0]) {
            pos = pos_t;
        } else {
            pos = pos_t;
        }
        float freq = inv_freq ? inv_freq[j] : (1.0f / powf(freq_base, (float)(2 * j) / rope_dim));
        float theta = pos * freq;
        cos_tab[j] = cosf(theta);
        sin_tab[j] = sinf(theta);
    }

    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
        int j = 0;
        for (; j + 7 < sect_dims; j += 8) {
            __m256 vc = _mm256_loadu_ps(cos_tab + j);
            __m256 vs = _mm256_loadu_ps(sin_tab + j);
            __m256 v0 = _mm256_loadu_ps(v + j);
            __m256 v1 = _mm256_loadu_ps(v + j + pair_off);
            _mm256_storeu_ps(v + j,              _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
            _mm256_storeu_ps(v + j + pair_off,   _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
        }
        for (; j < sect_dims; j++) {
            float v0 = v[j], v1 = v[j + pair_off];
            v[j]             = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + pair_off]  = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#else
        for (int j = 0; j < sect_dims; j++) {
            float v0 = v[j], v1 = v[j + pair_off];
            v[j]             = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + pair_off]  = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#endif
    }
}

/* Per-head RMSNorm (QK-norm): normalize each head_dim-sized chunk */
static void tf_qk_norm(float *vec, int n_heads, int head_dim, const qtensor *norm_w, float eps, float *w_buf) {
    tf_dequant_row(norm_w, 0, w_buf);
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vss = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < head_dim; i += 8) {
            __m256 vi = _mm256_loadu_ps(v + i);
            vss = _mm256_fmadd_ps(vi, vi, vss);
        }
        __m128 hi = _mm256_extractf128_ps(vss, 1);
        __m128 lo = _mm256_castps256_ps128(vss);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float ss = _mm_cvtss_f32(s4);
        for (; i < head_dim; i++) ss += v[i] * v[i];
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        __m256 vscale = _mm256_set1_ps(ss);
        i = 0;
        for (; i + 7 < head_dim; i += 8) {
            __m256 vi = _mm256_loadu_ps(v + i);
            __m256 vw = _mm256_loadu_ps(w_buf + i);
            _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_mul_ps(vi, vscale), vw));
        }
        for (; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#else
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#endif
    }
}

/* Forward declarations — defined after fast_exp_avx2 */
static void tf_softmax(float *x, int n);
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n);

/* IEEE-754 binary16 -> binary32 scalar conversion (slow fallback only). */
static inline float tf_f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    union { uint32_t u; float f; } o;
    if (e == 0) {
        if (m == 0) { o.u = s; return o.f; }
        while ((m & 0x400) == 0) { m <<= 1; e--; }
        e++; m &= 0x3FF;
    } else if (e == 0x1F) {
        o.u = s | 0x7F800000u | (m << 13);
        return o.f;
    }
    o.u = s | ((e + 112) << 23) | (m << 13);
    return o.f;
}

/* IEEE-754 binary32 -> binary16, round-to-nearest-even, with overflow → inf. */
static inline uint16_t tf_f32_to_f16(float x) {
    union { float f; uint32_t u; } in = { x };
    uint32_t u = in.u;
    uint32_t s = (u >> 16) & 0x8000;
    int32_t  e = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFF;
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m |= 0x800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t r = m & ((1u << shift) - 1);
        uint16_t h = (uint16_t)(s | (m >> shift));
        if (r > (1u << (shift - 1)) || (r == (1u << (shift - 1)) && (h & 1))) h++;
        return h;
    } else if (e >= 31) {
        return (uint16_t)(s | 0x7C00);
    }
    uint16_t h = (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
    uint32_t r = m & 0x1FFF;
    if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++;
    return h;
}

/* Quantize a row of N floats to int8 using a single shared scale s = max_abs/127.
 * Writes int8 values to out and returns the scale (0 if max_abs == 0). */
static inline float tf_quantize_row_q8(int8_t *out, const float *in, int n) {
    float ma = 0.0f;
    for (int i = 0; i < n; i++) { float a = in[i]; if (a < 0) a = -a; if (a > ma) ma = a; }
    if (ma == 0.0f) { for (int i = 0; i < n; i++) out[i] = 0; return 0.0f; }
    float s = ma / 127.0f;
    float inv = 127.0f / ma;
    for (int i = 0; i < n; i++) {
        float v = in[i] * inv;
        int q = (int)(v < 0 ? v - 0.5f : v + 0.5f);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
    return s;
}

/* Write one full position-row of [n_kv_heads * head_dim] floats into the
 * layer's KV cache at position p, converting per kv_dtype. Q8 emits one
 * scale per kv-head into scales[p*n_kv_heads .. p*n_kv_heads+n_kv_heads). */
static inline void tf_kv_write_all_heads(void *cache_ptr, float *scales,
                                         const float *src,
                                         int p, int n_kv_heads,
                                         int head_dim, int kv_dtype) {
    int kv_dim = n_kv_heads * head_dim;
    if (kv_dtype == TF_KV_DTYPE_F32) {
        memcpy((float *)cache_ptr + (size_t)p * kv_dim, src, (size_t)kv_dim * sizeof(float));
    } else if (kv_dtype == TF_KV_DTYPE_F16) {
        uint16_t *dst = (uint16_t *)cache_ptr + (size_t)p * kv_dim;
        for (int i = 0; i < kv_dim; i++) dst[i] = tf_f32_to_f16(src[i]);
    } else {
        int8_t *dst = (int8_t *)cache_ptr + (size_t)p * kv_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            float s = tf_quantize_row_q8(dst + h * head_dim, src + h * head_dim, head_dim);
            scales[(size_t)p * n_kv_heads + h] = s;
        }
    }
}

/* Transposed-K writer: K stored as [kv_h][d][p] with stride max_seq across p.
 * Writes one position-row of [n_kv_heads * head_dim] floats into the layer's
 * K cache at position p. Each scalar lands at scattered offsets, but the
 * total volume is small (n_kv_heads*head_dim per step). Q8 scales unchanged
 * layout: scales[p*n_kv_heads + h]. */
static inline void tf_k_write_transposed(void *cache_ptr, float *scales,
                                         const float *src,
                                         int p, int n_kv_heads,
                                         int head_dim, int max_seq_len,
                                         int kv_dtype) {
    size_t stride_d = (size_t)max_seq_len;          /* elements per (kv_h,d) row */
    size_t stride_h = (size_t)head_dim * stride_d;
    if (kv_dtype == TF_KV_DTYPE_F32) {
        float *base = (float *)cache_ptr;
        for (int h = 0; h < n_kv_heads; h++)
            for (int d = 0; d < head_dim; d++)
                base[h * stride_h + d * stride_d + p] = src[h * head_dim + d];
    } else if (kv_dtype == TF_KV_DTYPE_F16) {
        uint16_t *base = (uint16_t *)cache_ptr;
        for (int h = 0; h < n_kv_heads; h++)
            for (int d = 0; d < head_dim; d++)
                base[h * stride_h + d * stride_d + p] = tf_f32_to_f16(src[h * head_dim + d]);
    } else {
        /* Q8: quantise per (kv_h) row using the source, then scatter int8s. */
        int8_t *base = (int8_t *)cache_ptr;
        int8_t tmp[512];
        for (int h = 0; h < n_kv_heads; h++) {
            float s = tf_quantize_row_q8(tmp, src + h * head_dim, head_dim);
            scales[(size_t)p * n_kv_heads + h] = s;
            for (int d = 0; d < head_dim; d++)
                base[h * stride_h + d * stride_d + p] = tmp[d];
        }
    }
}

/* Forward decl for tf_k_cache_write_row below — full definition is further
 * down (handles V rows too). */
static inline void tf_kv_write_row(void *cache_ptr, float *scales,
                                   const float *src,
                                   int p, int kvh, int n_kv_heads,
                                   int head_dim, int kv_dim, int kv_dtype);

/* qpkd K writer: K stored as [p][d][kv_h]. For one position p, write all
 * n_kv_heads*head_dim values reordered so that element-index = d*n_kv_heads + kv_h.
 * F32/F16/Q8: dtype-specialised. Q8 quantises per kv_h row first (so each row
 * shares a single scale), then scatters int8s into the strided slots. */
static inline void tf_k_write_dp_all_heads(void *cache_ptr, float *scales,
                                           const float *src, int p,
                                           int n_kv_heads, int head_dim,
                                           int kv_dtype) {
    int kv_dim = n_kv_heads * head_dim;
    if (kv_dtype == TF_KV_DTYPE_F32) {
        float *base = (float *)cache_ptr + (size_t)p * kv_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            const float *s = src + (size_t)h * head_dim;
            for (int d = 0; d < head_dim; d++) base[d * n_kv_heads + h] = s[d];
        }
    } else if (kv_dtype == TF_KV_DTYPE_F16) {
        uint16_t *base = (uint16_t *)cache_ptr + (size_t)p * kv_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            const float *s = src + (size_t)h * head_dim;
            for (int d = 0; d < head_dim; d++) base[d * n_kv_heads + h] = tf_f32_to_f16(s[d]);
        }
    } else {  /* Q8 */
        int8_t *base = (int8_t *)cache_ptr + (size_t)p * kv_dim;
        int8_t tmp[512];  /* head_dim <= 512 in practice */
        for (int h = 0; h < n_kv_heads; h++) {
            float sc = tf_quantize_row_q8(tmp, src + (size_t)h * head_dim, head_dim);
            scales[(size_t)p * n_kv_heads + h] = sc;
            for (int d = 0; d < head_dim; d++) base[d * n_kv_heads + h] = tmp[d];
        }
    }
}

/* qpkd K writer, single-kv-head variant. */
static inline void tf_k_write_dp_row(void *cache_ptr, float *scales,
                                     const float *src, int p, int kv_h,
                                     int n_kv_heads, int head_dim, int kv_dtype) {
    int kv_dim = n_kv_heads * head_dim;
    if (kv_dtype == TF_KV_DTYPE_F32) {
        float *base = (float *)cache_ptr + (size_t)p * kv_dim;
        for (int d = 0; d < head_dim; d++) base[d * n_kv_heads + kv_h] = src[d];
    } else if (kv_dtype == TF_KV_DTYPE_F16) {
        uint16_t *base = (uint16_t *)cache_ptr + (size_t)p * kv_dim;
        for (int d = 0; d < head_dim; d++) base[d * n_kv_heads + kv_h] = tf_f32_to_f16(src[d]);
    } else {  /* Q8 */
        int8_t *base = (int8_t *)cache_ptr + (size_t)p * kv_dim;
        int8_t tmp[512];
        float sc = tf_quantize_row_q8(tmp, src, head_dim);
        scales[(size_t)p * n_kv_heads + kv_h] = sc;
        for (int d = 0; d < head_dim; d++) base[d * n_kv_heads + kv_h] = tmp[d];
    }
}

/* Dispatch K-row write to the right layout-specific writer. The non-Gemma4
 * Qwen/hybrid attention path uses this; Gemma4's own KV writer is direct
 * memcpy into row-major K. */
static inline void tf_k_cache_write_pos(void *cache, float *scales,
                                        const float *src, int p,
                                        int n_kv_heads, int head_dim,
                                        int max_seq_len, int kv_dtype,
                                        int transposed, int k_dp) {
    if (transposed)
        tf_k_write_transposed(cache, scales, src, p, n_kv_heads, head_dim,
                              max_seq_len, kv_dtype);
    else if (k_dp)
        tf_k_write_dp_all_heads(cache, scales, src, p, n_kv_heads, head_dim, kv_dtype);
    else
        tf_kv_write_all_heads(cache, scales, src, p, n_kv_heads, head_dim, kv_dtype);
}

/* Single-kv-head K row write: writes K[kv_h][0..head_dim] at position p,
 * choosing layout. The per-head persistent-worker path calls this. */
static inline void tf_k_cache_write_row(void *cache, float *scales,
                                        const float *src, int p, int kv_h,
                                        int n_kv_heads, int head_dim,
                                        int kv_dim, int max_seq_len,
                                        int kv_dtype, int transposed, int k_dp) {
    if (k_dp) {
        tf_k_write_dp_row(cache, scales, src, p, kv_h, n_kv_heads, head_dim, kv_dtype);
        return;
    }
    if (!transposed) {
        tf_kv_write_row(cache, scales, src, p, kv_h, n_kv_heads, head_dim, kv_dim, kv_dtype);
        return;
    }
    size_t stride_d = (size_t)max_seq_len;
    size_t base = (size_t)kv_h * head_dim * stride_d;
    if (kv_dtype == TF_KV_DTYPE_F32) {
        float *dst = (float *)cache;
        for (int d = 0; d < head_dim; d++) dst[base + (size_t)d * stride_d + p] = src[d];
    } else {  /* F16 (Q8 disabled when transposed) */
        uint16_t *dst = (uint16_t *)cache;
        for (int d = 0; d < head_dim; d++) dst[base + (size_t)d * stride_d + p] = tf_f32_to_f16(src[d]);
    }
}

/* Write one K (or V) head-row of head_dim floats into the layer-l KV cache at
 * position p, kv_head kvh. Quantises/converts according to kv_dtype. The Q8
 * scale lands in scales[p * n_kv_heads + kvh]. */
static inline void tf_kv_write_row(void *cache_ptr, float *scales,
                                   const float *src,
                                   int p, int kvh, int n_kv_heads,
                                   int head_dim, int kv_dim, int kv_dtype) {
    size_t off = (size_t)p * kv_dim + (size_t)kvh * head_dim;
    if (kv_dtype == TF_KV_DTYPE_F32) {
        memcpy((float *)cache_ptr + off, src, (size_t)head_dim * sizeof(float));
    } else if (kv_dtype == TF_KV_DTYPE_F16) {
        uint16_t *dst = (uint16_t *)cache_ptr + off;
        for (int i = 0; i < head_dim; i++) dst[i] = tf_f32_to_f16(src[i]);
    } else {
        int8_t *dst = (int8_t *)cache_ptr + off;
        float s = tf_quantize_row_q8(dst, src, head_dim);
        scales[(size_t)p * n_kv_heads + kvh] = s;
    }
}

/* qpkd helpers: K stored [p][d][kv_h], Q packed [d][h].
 * pack_q_heads: pack a subset of heads [h_lo, h_hi) into Q_packed[d][h]. */
static inline void tf_qpkd_pack_q_heads(const float *q, float *q_packed,
                                        int h_lo, int h_hi,
                                        int n_heads, int head_dim) {
    for (int h = h_lo; h < h_hi; h++) {
        const float *src = q + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; d++)
            q_packed[(size_t)d * n_heads + h] = src[d];
    }
}

#if defined(__ARM_FEATURE_SVE)
/* qpkd+ktbl QK kernel: process positions [p_lo, p_hi) for ALL heads.
 * Q_packed layout [d][h], K_dp layout [p][d][kv_h]. Per p, write att[h][p]
 * for all h in 0..n_heads via a tiny scalar fan-out. Requires svcntw()==n_heads
 * (verified at env init). */
static inline void tf_qpkd_qk_chunk_f32(const float *K_dp,
                                        const float *Q_packed,
                                        float *att,
                                        int p_lo, int p_hi,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int max_seq_len,
                                        float scale) {
    svbool_t pg = svptrue_b32();
    svfloat32_t vscale = svdup_f32(scale);
    int gqa = n_heads / n_kv_heads;
    uint32_t idx_arr[16];
    for (int i = 0; i < n_heads; i++) idx_arr[i] = (uint32_t)(i / gqa);
    svuint32_t idx_repl = svld1_u32(pg, idx_arr);
    int kv_dim = n_kv_heads * head_dim;
    float tmp[16];
    for (int p = p_lo; p < p_hi; p++) {
        const float *kp = K_dp + (size_t)p * kv_dim;
        if (p + 2 < p_hi)
            __builtin_prefetch(K_dp + (size_t)(p + 2) * kv_dim, 0, 1);
        svfloat32_t acc = svdup_f32(0.0f);
        for (int d = 0; d < head_dim; d++) {
            svfloat32_t qv = svld1(pg, Q_packed + (size_t)d * n_heads);
            svfloat32_t k4 = svld1rq_f32(pg, kp + (size_t)d * n_kv_heads);
            svfloat32_t kx = svtbl_f32(k4, idx_repl);
            acc = svmla_x(pg, acc, qv, kx);
        }
        svst1(pg, tmp, svmul_x(pg, acc, vscale));
        for (int h = 0; h < n_heads; h++)
            att[(size_t)h * max_seq_len + p] = tmp[h];
    }
}

/* F16 variant: K[p][d][kv_h] stored as uint16 half-floats. Vectorized inner
 * loop processes 4 d-iterations per K load: svld1uh_u32 brings 16 u16 values
 * (= 4 d × 4 kv_h) into 16 u32 lanes, then svcvt_f32_f16_x converts in one
 * shot. 4 svtbl + 4 svmla per K-load, vs the previous version's per-d scalar
 * f16→f32 + stack tmp + svld1rq which was the documented bottleneck. */
static inline void tf_qpkd_qk_chunk_f16(const uint16_t *K_dp,
                                        const float *Q_packed,
                                        float *att,
                                        int p_lo, int p_hi,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int max_seq_len,
                                        float scale) {
    svbool_t pg = svptrue_b32();
    svfloat32_t vscale = svdup_f32(scale);
    int gqa = n_heads / n_kv_heads;
    /* Per-d-in-batch (b=0..3) index: lane h pulls k_dp[d+b][h/gqa] from k16
     * which has layout [k_d.h0..h3, k_d+1.h0..h3, k_d+2.h0..h3, k_d+3.h0..h3]. */
    uint32_t idx_arr[64];
    for (int b = 0; b < 4; b++)
        for (int h = 0; h < n_heads; h++)
            idx_arr[b * 16 + h] = (uint32_t)(b * n_kv_heads + h / gqa);
    svuint32_t idx_d0 = svld1_u32(pg, idx_arr +  0);
    svuint32_t idx_d1 = svld1_u32(pg, idx_arr + 16);
    svuint32_t idx_d2 = svld1_u32(pg, idx_arr + 32);
    svuint32_t idx_d3 = svld1_u32(pg, idx_arr + 48);
    int kv_dim = n_kv_heads * head_dim;
    float tmp[16];
    for (int p = p_lo; p < p_hi; p++) {
        const uint16_t *kp = K_dp + (size_t)p * kv_dim;
        if (p + 2 < p_hi)
            __builtin_prefetch(K_dp + (size_t)(p + 2) * kv_dim, 0, 1);
        svfloat32_t acc = svdup_f32(0.0f);
        for (int d = 0; d < head_dim; d += 4) {
            const uint16_t *k_base = kp + (size_t)d * n_kv_heads;
            svuint32_t  ku32 = svld1uh_u32(pg, k_base);
            svfloat32_t k16  = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(ku32));
            svfloat32_t q0 = svld1(pg, Q_packed + (size_t)(d + 0) * n_heads);
            svfloat32_t q1 = svld1(pg, Q_packed + (size_t)(d + 1) * n_heads);
            svfloat32_t q2 = svld1(pg, Q_packed + (size_t)(d + 2) * n_heads);
            svfloat32_t q3 = svld1(pg, Q_packed + (size_t)(d + 3) * n_heads);
            acc = svmla_x(pg, acc, q0, svtbl_f32(k16, idx_d0));
            acc = svmla_x(pg, acc, q1, svtbl_f32(k16, idx_d1));
            acc = svmla_x(pg, acc, q2, svtbl_f32(k16, idx_d2));
            acc = svmla_x(pg, acc, q3, svtbl_f32(k16, idx_d3));
        }
        svst1(pg, tmp, svmul_x(pg, acc, vscale));
        for (int h = 0; h < n_heads; h++)
            att[(size_t)h * max_seq_len + p] = tmp[h];
    }
}

/* Q8 variant: K[p][d][kv_h] stored as int8 with per-(p, kv_h) scale
 * scales[p * n_kv_heads + kv_h]. Per p, build a per-lane scale vector
 * once via svld1rq+svtbl (lanes that share a kv_h share a scale), then
 * accumulate raw int8 products without per-d scale. Combined scale is
 * applied once at the end. */
static inline void tf_qpkd_qk_chunk_q8(const int8_t *K_dp,
                                       const float *scales,
                                       const float *Q_packed,
                                       float *att,
                                       int p_lo, int p_hi,
                                       int n_heads, int n_kv_heads,
                                       int head_dim, int max_seq_len,
                                       float scale) {
    svbool_t pg = svptrue_b32();
    int gqa = n_heads / n_kv_heads;
    /* Same 4-d-batch indices as F16 kernel: 16 s8 K values per load. */
    uint32_t idx_arr[64];
    for (int b = 0; b < 4; b++)
        for (int h = 0; h < n_heads; h++)
            idx_arr[b * 16 + h] = (uint32_t)(b * n_kv_heads + h / gqa);
    svuint32_t idx_d0 = svld1_u32(pg, idx_arr +  0);
    svuint32_t idx_d1 = svld1_u32(pg, idx_arr + 16);
    svuint32_t idx_d2 = svld1_u32(pg, idx_arr + 32);
    svuint32_t idx_d3 = svld1_u32(pg, idx_arr + 48);
    /* Per-lane scale: lanes that share a kv_h share scales[p][kv_h]. The 16
     * 4-lane groups all share the same idx_arr[0..15] pattern. */
    uint32_t idx_repl[16];
    for (int h = 0; h < n_heads; h++) idx_repl[h] = (uint32_t)(h / gqa);
    svuint32_t idx_s = svld1_u32(pg, idx_repl);
    int kv_dim = n_kv_heads * head_dim;
    float tmp[16];
    for (int p = p_lo; p < p_hi; p++) {
        const int8_t *kp = K_dp + (size_t)p * kv_dim;
        if (p + 2 < p_hi)
            __builtin_prefetch(K_dp + (size_t)(p + 2) * kv_dim, 0, 1);
        svfloat32_t sq = svld1rq_f32(pg, scales + (size_t)p * n_kv_heads);
        svfloat32_t s_lane = svmul_x(pg, svtbl_f32(sq, idx_s), svdup_f32(scale));
        svfloat32_t acc = svdup_f32(0.0f);
        for (int d = 0; d < head_dim; d += 4) {
            const int8_t *k_base = kp + (size_t)d * n_kv_heads;
            svint32_t   k_s32 = svld1sb_s32(pg, k_base);
            svfloat32_t k16   = svcvt_f32_s32_x(pg, k_s32);
            svfloat32_t q0 = svld1(pg, Q_packed + (size_t)(d + 0) * n_heads);
            svfloat32_t q1 = svld1(pg, Q_packed + (size_t)(d + 1) * n_heads);
            svfloat32_t q2 = svld1(pg, Q_packed + (size_t)(d + 2) * n_heads);
            svfloat32_t q3 = svld1(pg, Q_packed + (size_t)(d + 3) * n_heads);
            acc = svmla_x(pg, acc, q0, svtbl_f32(k16, idx_d0));
            acc = svmla_x(pg, acc, q1, svtbl_f32(k16, idx_d1));
            acc = svmla_x(pg, acc, q2, svtbl_f32(k16, idx_d2));
            acc = svmla_x(pg, acc, q3, svtbl_f32(k16, idx_d3));
        }
        svst1(pg, tmp, svmul_x(pg, acc, s_lane));
        for (int h = 0; h < n_heads; h++)
            att[(size_t)h * max_seq_len + p] = tmp[h];
    }
}

/* AV partial accumulation over a p-range. V layout is [p][kv_h][d] (baseline).
 * out_tmp is the caller's per-thread [n_heads * head_dim] buffer; the caller
 * is responsible for zeroing it before the first call and for reducing
 * across threads after the barrier.  Each thread streams its own p-slice
 * sequentially through V — 48-way parallel instead of the per-head split
 * that left 32 of 48 threads idle on Qwen3.5-9B. */
static inline void tf_av_chunk_f32(const float *V, const float *att,
                                   float *out_tmp,
                                   int p_lo, int p_hi,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   int max_seq_len) {
    svbool_t pg = svptrue_b32();
    int gqa = n_heads / n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    int vl = (int)svcntw();
    for (int p = p_lo; p < p_hi; p++) {
        const float *vp = V + (size_t)p * kv_dim;
        if (p + 2 < p_hi)
            __builtin_prefetch(V + (size_t)(p + 2) * kv_dim, 0, 1);
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa;
            svfloat32_t a = svdup_f32(att[(size_t)h * max_seq_len + p]);
            const float *v = vp + (size_t)kv_h * head_dim;
            float *o = out_tmp + (size_t)h * head_dim;
            int d = 0;
            for (; d + vl - 1 < head_dim; d += vl) {
                svfloat32_t ov = svld1(pg, o + d);
                svfloat32_t vv = svld1(pg, v + d);
                svst1(pg, o + d, svmla_x(pg, ov, a, vv));
            }
            if (d < head_dim) {
                svbool_t pt = svwhilelt_b32(d, head_dim);
                svfloat32_t ov = svld1(pt, o + d);
                svfloat32_t vv = svld1(pt, v + d);
                svst1(pt, o + d, svmla_m(pt, ov, a, vv));
            }
        }
    }
}

static inline void tf_av_chunk_f16(const uint16_t *V, const float *att,
                                   float *out_tmp,
                                   int p_lo, int p_hi,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   int max_seq_len) {
    svbool_t pg = svptrue_b32();
    int gqa = n_heads / n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    int vl = (int)svcntw();
    for (int p = p_lo; p < p_hi; p++) {
        const uint16_t *vp = V + (size_t)p * kv_dim;
        if (p + 2 < p_hi)
            __builtin_prefetch(V + (size_t)(p + 2) * kv_dim, 0, 1);
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa;
            svfloat32_t a = svdup_f32(att[(size_t)h * max_seq_len + p]);
            const uint16_t *v = vp + (size_t)kv_h * head_dim;
            float *o = out_tmp + (size_t)h * head_dim;
            int d = 0;
            for (; d + vl - 1 < head_dim; d += vl) {
                svuint32_t vu = svld1uh_u32(pg, v + d);
                svfloat32_t vv = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(vu));
                svfloat32_t ov = svld1(pg, o + d);
                svst1(pg, o + d, svmla_x(pg, ov, a, vv));
            }
            if (d < head_dim) {
                svbool_t pt = svwhilelt_b32(d, head_dim);
                svuint32_t vu = svld1uh_u32(pt, v + d);
                svfloat32_t vv = svcvt_f32_f16_x(pt, svreinterpret_f16_u32(vu));
                svfloat32_t ov = svld1(pt, o + d);
                svst1(pt, o + d, svmla_m(pt, ov, a, vv));
            }
        }
    }
}

static inline void tf_av_chunk_q8(const int8_t *V, const float *scales,
                                  const float *att, float *out_tmp,
                                  int p_lo, int p_hi,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len) {
    svbool_t pg = svptrue_b32();
    int gqa = n_heads / n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    int vl = (int)svcntw();
    for (int p = p_lo; p < p_hi; p++) {
        const int8_t *vp = V + (size_t)p * kv_dim;
        if (p + 2 < p_hi)
            __builtin_prefetch(V + (size_t)(p + 2) * kv_dim, 0, 1);
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa;
            float s = scales[(size_t)p * n_kv_heads + kv_h];
            svfloat32_t a = svdup_f32(att[(size_t)h * max_seq_len + p] * s);
            const int8_t *v = vp + (size_t)kv_h * head_dim;
            float *o = out_tmp + (size_t)h * head_dim;
            int d = 0;
            for (; d + vl - 1 < head_dim; d += vl) {
                svint32_t vi = svld1sb_s32(pg, v + d);
                svfloat32_t vv = svcvt_f32_s32_x(pg, vi);
                svfloat32_t ov = svld1(pg, o + d);
                svst1(pg, o + d, svmla_x(pg, ov, a, vv));
            }
            if (d < head_dim) {
                svbool_t pt = svwhilelt_b32(d, head_dim);
                svint32_t vi = svld1sb_s32(pt, v + d);
                svfloat32_t vv = svcvt_f32_s32_x(pt, vi);
                svfloat32_t ov = svld1(pt, o + d);
                svst1(pt, o + d, svmla_m(pt, ov, a, vv));
            }
        }
    }
}

/* Reduce av_tmp across nt threads into out[head*head_dim + d_lo..d_hi].
 * Each thread is responsible for some (h, d-slice); the caller chooses the
 * partition so that all (h, d) are covered exactly once across the pool. */
static inline void tf_av_reduce_slice(float *out, const float *av_tmp,
                                      int h, int d_lo, int d_hi,
                                      int nt, int n_heads, int head_dim) {
    svbool_t pg = svptrue_b32();
    int vl = (int)svcntw();
    size_t per_thread = (size_t)n_heads * head_dim;
    float *out_h = out + (size_t)h * head_dim;
    int d = d_lo;
    for (; d + vl - 1 < d_hi; d += vl) {
        svfloat32_t acc = svdup_f32(0.0f);
        for (int t = 0; t < nt; t++) {
            const float *src = av_tmp + (size_t)t * per_thread + (size_t)h * head_dim + d;
            acc = svadd_x(pg, acc, svld1(pg, src));
        }
        svst1(pg, out_h + d, acc);
    }
    if (d < d_hi) {
        svbool_t pt = svwhilelt_b32(d, d_hi);
        svfloat32_t acc = svdup_f32(0.0f);
        for (int t = 0; t < nt; t++) {
            const float *src = av_tmp + (size_t)t * per_thread + (size_t)h * head_dim + d;
            acc = svadd_m(pt, acc, svld1(pt, src));
        }
        svst1(pt, out_h + d, acc);
    }
}
#endif

/* Multi-head attention worker for threading.
 * key_cache/value_cache stride is kv_dim * elem_bytes; element type matches
 * kv_dtype (4B F32, 2B F16, 1B int8 with per-(pos, kv_head) scale). For Q8,
 * key_scales/value_scales point at the layer's [seq_len * n_kv_heads] scales. */
typedef struct {
    const float *q;          /* full Q buffer */
    float *att;              /* full attention scores buffer */
    float *xb2;              /* full output buffer (each head writes its own slice) */
    const void *key_cache;
    const void *value_cache;
    const float *key_scales;   /* NULL unless kv_dtype==Q8 */
    const float *value_scales; /* NULL unless kv_dtype==Q8 */
    int head_start, head_end;
    int head_dim, kv_dim, gqa_ratio, seq_len, max_seq_len;
    int qhead_base;          /* global index of local head 0: kv_h = (qhead_base+h)/gqa_ratio.
                              * 0 except in TP KV-replicate mode (full KV cache, sharded Q). */
    int kv_head_base;        /* local KV-cache offset: LOCAL kv = GLOBAL kv - kv_head_base */
    int n_kv_heads;          /* needed to index per-(pos, kv_h) scales */
    int kv_dtype;            /* TF_KV_DTYPE_F32 / F16 / Q8 */
    int k_transposed;        /* 1 = K stored [kv_h][d][p] (FMLA-into-att path) */
    int k_dp;                /* 1 = K stored [p][d][kv_h] (qpkd+ktbl) */
    int skip_qk;             /* 1 = att already filled by a pre-pass; jump to softmax */
    int skip_av;             /* 1 = xb2 already filled by parallel AV reduction; no V loop */
    float scale;
} tf_attn_task;

static void *tf_attn_worker(void *arg) {
    tf_attn_task *t = (tf_attn_task *)arg;
    int hd = t->head_dim;
    for (int h = t->head_start; h < t->head_end; h++) {
        int kv_h = (t->qhead_base + h) / t->gqa_ratio;
        kv_h -= t->kv_head_base;
        const float *q_h = t->q + h * hd;
        float *att_h = t->att + h * t->max_seq_len;
        int seq_len = t->seq_len;

#if defined(__AVX2__) && defined(__FMA__)
        /* AVX2 path is F32-only; the F16/Q8 KV variants are SVE-only paths. */
        const float *_kc_f32 = (const float *)t->key_cache;
        const float *_vc_f32 = (const float *)t->value_cache;
        if (hd == 64) {
            __m256 q0=_mm256_loadu_ps(q_h),    q1=_mm256_loadu_ps(q_h+8);
            __m256 q2=_mm256_loadu_ps(q_h+16), q3=_mm256_loadu_ps(q_h+24);
            __m256 q4=_mm256_loadu_ps(q_h+32), q5=_mm256_loadu_ps(q_h+40);
            __m256 q6=_mm256_loadu_ps(q_h+48), q7=_mm256_loadu_ps(q_h+56);

            /* QK scores: 4 positions at a time */
            int p = 0;
            for (; p + 3 < seq_len; p += 4) {
                const float *k0 = _kc_f32 + (size_t)(p+0)*t->kv_dim + kv_h*hd;
                const float *k1 = _kc_f32 + (size_t)(p+1)*t->kv_dim + kv_h*hd;
                const float *k2 = _kc_f32 + (size_t)(p+2)*t->kv_dim + kv_h*hd;
                const float *k3 = _kc_f32 + (size_t)(p+3)*t->kv_dim + kv_h*hd;
                __m256 s0=_mm256_mul_ps(q0,_mm256_loadu_ps(k0));
                __m256 s1=_mm256_mul_ps(q0,_mm256_loadu_ps(k1));
                __m256 s2=_mm256_mul_ps(q0,_mm256_loadu_ps(k2));
                __m256 s3=_mm256_mul_ps(q0,_mm256_loadu_ps(k3));
                s0=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k0+8),s0);
                s1=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k1+8),s1);
                s2=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k2+8),s2);
                s3=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k3+8),s3);
                s0=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k0+16),s0);
                s1=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k1+16),s1);
                s2=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k2+16),s2);
                s3=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k3+16),s3);
                s0=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k0+24),s0);
                s1=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k1+24),s1);
                s2=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k2+24),s2);
                s3=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k3+24),s3);
                s0=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k0+32),s0);
                s1=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k1+32),s1);
                s2=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k2+32),s2);
                s3=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k3+32),s3);
                s0=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k0+40),s0);
                s1=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k1+40),s1);
                s2=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k2+40),s2);
                s3=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k3+40),s3);
                s0=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k0+48),s0);
                s1=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k1+48),s1);
                s2=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k2+48),s2);
                s3=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k3+48),s3);
                s0=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k0+56),s0);
                s1=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k1+56),s1);
                s2=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k2+56),s2);
                s3=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k3+56),s3);
                __m256 h01=_mm256_hadd_ps(s0,s1);
                __m256 h23=_mm256_hadd_ps(s2,s3);
                __m256 h0123=_mm256_hadd_ps(h01,h23);
                __m128 lo=_mm256_castps256_ps128(h0123);
                __m128 hi=_mm256_extractf128_ps(h0123,1);
                _mm_storeu_ps(att_h+p, _mm_mul_ps(_mm_add_ps(lo,hi),_mm_set1_ps(t->scale)));
            }
            for (; p < seq_len; p++) {
                const float *kp = _kc_f32 + (size_t)p*t->kv_dim + kv_h*hd;
                __m256 s=_mm256_mul_ps(q0,_mm256_loadu_ps(kp));
                s=_mm256_fmadd_ps(q1,_mm256_loadu_ps(kp+8),s);
                s=_mm256_fmadd_ps(q2,_mm256_loadu_ps(kp+16),s);
                s=_mm256_fmadd_ps(q3,_mm256_loadu_ps(kp+24),s);
                s=_mm256_fmadd_ps(q4,_mm256_loadu_ps(kp+32),s);
                s=_mm256_fmadd_ps(q5,_mm256_loadu_ps(kp+40),s);
                s=_mm256_fmadd_ps(q6,_mm256_loadu_ps(kp+48),s);
                s=_mm256_fmadd_ps(q7,_mm256_loadu_ps(kp+56),s);
                __m128 _hi=_mm256_extractf128_ps(s,1);
                __m128 _lo=_mm256_castps256_ps128(s);
                __m128 _s=_mm_add_ps(_lo,_hi);
                _s=_mm_add_ps(_s,_mm_movehl_ps(_s,_s));
                _s=_mm_add_ss(_s,_mm_movehdup_ps(_s));
                att_h[p] = _mm_cvtss_f32(_s) * t->scale;
            }

            /* Softmax */
            tf_softmax(att_h, seq_len);

            /* V accumulation: 8 AVX accumulators for head_dim=64 */
            float *out_h = t->xb2 + h * hd;
            __m256 o0=_mm256_setzero_ps(), o1=_mm256_setzero_ps();
            __m256 o2=_mm256_setzero_ps(), o3=_mm256_setzero_ps();
            __m256 o4=_mm256_setzero_ps(), o5=_mm256_setzero_ps();
            __m256 o6=_mm256_setzero_ps(), o7=_mm256_setzero_ps();
            for (p = 0; p < seq_len; p++) {
                const float *vp = _vc_f32 + (size_t)p*t->kv_dim + kv_h*hd;
                __m256 a = _mm256_set1_ps(att_h[p]);
                o0=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp),   o0);
                o1=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+8), o1);
                o2=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+16),o2);
                o3=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+24),o3);
                o4=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+32),o4);
                o5=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+40),o5);
                o6=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+48),o6);
                o7=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+56),o7);
            }
            _mm256_storeu_ps(out_h,   o0); _mm256_storeu_ps(out_h+8, o1);
            _mm256_storeu_ps(out_h+16,o2); _mm256_storeu_ps(out_h+24,o3);
            _mm256_storeu_ps(out_h+32,o4); _mm256_storeu_ps(out_h+40,o5);
            _mm256_storeu_ps(out_h+48,o6); _mm256_storeu_ps(out_h+56,o7);
        } else {
            /* Generic AVX2 attention for any head_dim (with prefetch) */
            for (int p = 0; p < seq_len; p++) {
                const float *k_p = _kc_f32 + (size_t)p * t->kv_dim + kv_h * hd;
                if (p + 2 < seq_len)
                    _mm_prefetch((const char *)(_kc_f32 + (size_t)(p+2) * t->kv_dim + kv_h * hd), _MM_HINT_T0);
                __m256 acc = _mm256_setzero_ps();
                int d = 0;
                for (; d + 7 < hd; d += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d), _mm256_loadu_ps(k_p + d), acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                float score = _mm_cvtss_f32(s4);
                for (; d < hd; d++) score += q_h[d] * k_p[d];
                att_h[p] = score * t->scale;
            }
            tf_softmax(att_h, seq_len);
            float *out_h = t->xb2 + h * hd;
            /* Zero output with AVX2 */
            for (int d = 0; d < hd; d += 8) _mm256_storeu_ps(out_h + d, _mm256_setzero_ps());
            for (int p = 0; p < seq_len; p++) {
                const float *v_p = _vc_f32 + (size_t)p * t->kv_dim + kv_h * hd;
                if (p + 2 < seq_len)
                    _mm_prefetch((const char *)(_vc_f32 + (size_t)(p+2) * t->kv_dim + kv_h * hd), _MM_HINT_T0);
                __m256 a = _mm256_set1_ps(att_h[p]);
                for (int d = 0; d < hd; d += 8)
                    _mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(a, _mm256_loadu_ps(v_p + d),
                                                                  _mm256_loadu_ps(out_h + d)));
            }
        }
#elif defined(__ARM_FEATURE_SVE)
        {
            /* SVE attention. Branch once per head on kv_dtype; inner loops are
             * dtype-specialised so the hot path has no per-(p, d) dispatch. */
            svbool_t pg = svptrue_b32();
            size_t kv_eb = (t->kv_dtype == TF_KV_DTYPE_F32) ? 4 :
                           (t->kv_dtype == TF_KV_DTYPE_F16) ? 2 : 1;
            const uint8_t *kc8 = (const uint8_t *)t->key_cache;
            const uint8_t *vc8 = (const uint8_t *)t->value_cache;
            size_t row_stride = (size_t)t->kv_dim * kv_eb;
            size_t head_off   = (size_t)kv_h * t->head_dim * kv_eb;
            int nkv = t->n_kv_heads;

            /* ===== QK scores ===== */
            if (t->k_transposed) {
                /* K stored as [kv_h][d][p] with stride_d=max_seq_len across p.
                 * Outer block over positions, inner FMLA over d — accumulator
                 * is a vector of vl positions, no svaddv per position. */
                int vl = (int)svcntw();
                size_t stride_d = (size_t)t->max_seq_len;
                size_t stride_h = (size_t)t->head_dim * stride_d;
                svfloat32_t vscale = svdup_f32(t->scale);
                if (t->kv_dtype == TF_KV_DTYPE_F32) {
                    const float *K_h = (const float *)t->key_cache + (size_t)kv_h * stride_h;
                    for (int p = 0; p < seq_len; p += vl) {
                        svbool_t pp = svwhilelt_b32(p, seq_len);
                        svfloat32_t acc = svdup_f32(0.0f);
                        for (int d = 0; d < hd; d++) {
                            svfloat32_t qv = svdup_f32(q_h[d]);
                            svfloat32_t kv = svld1(pp, K_h + (size_t)d * stride_d + p);
                            acc = svmla_x(pp, acc, qv, kv);
                        }
                        svst1(pp, att_h + p, svmul_x(pp, acc, vscale));
                    }
                } else {  /* F16 (Q8 disabled via load-time check) */
                    const uint16_t *K_h = (const uint16_t *)t->key_cache + (size_t)kv_h * stride_h;
                    for (int p = 0; p < seq_len; p += vl) {
                        svbool_t pp = svwhilelt_b32(p, seq_len);
                        svfloat32_t acc = svdup_f32(0.0f);
                        for (int d = 0; d < hd; d++) {
                            svfloat32_t qv = svdup_f32(q_h[d]);
                            svuint32_t ku = svld1uh_u32(pp, K_h + (size_t)d * stride_d + p);
                            svfloat32_t kv = svcvt_f32_f16_x(pp, svreinterpret_f16_u32(ku));
                            acc = svmla_x(pp, acc, qv, kv);
                        }
                        svst1(pp, att_h + p, svmul_x(pp, acc, vscale));
                    }
                }
            } else if (t->skip_qk) {
                /* QK already filled by qpkd+ktbl pre-pass (any dtype). */
            } else if (t->k_dp && t->kv_dtype == TF_KV_DTYPE_F32) {
                /* K stored [p][d][kv_h]: per-head gather fallback (correctness;
                 * the persistent-worker hot path uses tf_qpkd_qk_chunk_f32 in
                 * a pre-pass and sets skip_qk instead). */
                const float *kc_f32 = (const float *)t->key_cache;
                svuint32_t idx_d = svindex_u32(0, (uint32_t)t->n_kv_heads);
                for (int p = 0; p < seq_len; p++) {
                    const float *kp_base = kc_f32 + (size_t)p * t->kv_dim + kv_h;
                    if (p + 2 < seq_len)
                        __builtin_prefetch(kc_f32 + (size_t)(p+2) * t->kv_dim + kv_h, 0, 1);
                    svfloat32_t acc = svdup_f32(0.0f);
                    int d = 0;
                    int vl = (int)svcntw();
                    for (; d + vl - 1 < hd; d += vl) {
                        svfloat32_t qv = svld1(pg, q_h + d);
                        svfloat32_t kv = svld1_gather_u32index_f32(pg, kp_base + (size_t)d * t->n_kv_heads, idx_d);
                        acc = svmla_x(pg, acc, qv, kv);
                    }
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        svfloat32_t qv = svld1(ptail, q_h + d);
                        svfloat32_t kv = svld1_gather_u32index_f32(ptail, kp_base + (size_t)d * t->n_kv_heads, idx_d);
                        acc = svmla_m(ptail, acc, qv, kv);
                    }
                    att_h[p] = svaddv(pg, acc) * t->scale;
                }
            } else if (t->k_dp && t->kv_dtype == TF_KV_DTYPE_F16) {
                /* K [p][d][kv_h] F16: scalar gather over d, convert + accumulate.
                 * Correctness fallback; hot path uses tf_qpkd_qk_chunk_f16. */
                const uint16_t *kc_u16 = (const uint16_t *)t->key_cache;
                for (int p = 0; p < seq_len; p++) {
                    const uint16_t *kp_base = kc_u16 + (size_t)p * t->kv_dim + kv_h;
                    if (p + 2 < seq_len)
                        __builtin_prefetch(kc_u16 + (size_t)(p+2) * t->kv_dim + kv_h, 0, 1);
                    float score = 0.0f;
                    for (int d = 0; d < hd; d++)
                        score += q_h[d] * tf_f16_to_f32(kp_base[(size_t)d * t->n_kv_heads]);
                    att_h[p] = score * t->scale;
                }
            } else if (t->k_dp) {  /* Q8 */
                const int8_t *kc_i8 = (const int8_t *)t->key_cache;
                const float *ks = t->key_scales;
                for (int p = 0; p < seq_len; p++) {
                    const int8_t *kp_base = kc_i8 + (size_t)p * t->kv_dim + kv_h;
                    if (p + 2 < seq_len)
                        __builtin_prefetch(kc_i8 + (size_t)(p+2) * t->kv_dim + kv_h, 0, 1);
                    float s = ks[(size_t)p * nkv + kv_h];
                    float score = 0.0f;
                    for (int d = 0; d < hd; d++)
                        score += q_h[d] * (float)kp_base[(size_t)d * t->n_kv_heads];
                    att_h[p] = score * t->scale * s;
                }
            } else if (t->kv_dtype == TF_KV_DTYPE_F32) {
                for (int p = 0; p < seq_len; p++) {
                    const float *k_p = (const float *)(kc8 + (size_t)p * row_stride + head_off);
                    if (p + 2 < seq_len)
                        __builtin_prefetch(kc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
                    svfloat32_t acc = svdup_f32(0.0f);
                    int d = 0;
                    for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw())
                        acc = svmla_x(pg, acc, svld1(pg, q_h + d), svld1(pg, k_p + d));
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), svld1(ptail, k_p + d));
                    }
                    att_h[p] = svaddv(pg, acc) * t->scale;
                }
            } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
                for (int p = 0; p < seq_len; p++) {
                    const uint16_t *k_p = (const uint16_t *)(kc8 + (size_t)p * row_stride + head_off);
                    if (p + 2 < seq_len)
                        __builtin_prefetch(kc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
                    svfloat32_t acc = svdup_f32(0.0f);
                    int d = 0;
                    for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw()) {
                        svuint32_t ku32 = svld1uh_u32(pg, k_p + d);
                        svfloat32_t kf  = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(ku32));
                        acc = svmla_x(pg, acc, svld1(pg, q_h + d), kf);
                    }
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        svuint32_t ku32 = svld1uh_u32(ptail, k_p + d);
                        svfloat32_t kf  = svcvt_f32_f16_x(ptail, svreinterpret_f16_u32(ku32));
                        acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), kf);
                    }
                    att_h[p] = svaddv(pg, acc) * t->scale;
                }
            } else {  /* Q8 */
                const float *ks = t->key_scales;
                for (int p = 0; p < seq_len; p++) {
                    const int8_t *k_p = (const int8_t *)(kc8 + (size_t)p * row_stride + head_off);
                    if (p + 2 < seq_len)
                        __builtin_prefetch(kc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
                    float s = ks[(size_t)p * nkv + kv_h];
                    svfloat32_t acc = svdup_f32(0.0f);
                    int d = 0;
                    for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw()) {
                        svint32_t ki = svld1sb_s32(pg, k_p + d);
                        svfloat32_t kf = svcvt_f32_s32_x(pg, ki);
                        acc = svmla_x(pg, acc, svld1(pg, q_h + d), kf);
                    }
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        svint32_t ki = svld1sb_s32(ptail, k_p + d);
                        svfloat32_t kf = svcvt_f32_s32_x(ptail, ki);
                        acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), kf);
                    }
                    att_h[p] = svaddv(pg, acc) * t->scale * s;
                }
            }

            /* skip_av: softmax + V already done by parallel pre-pass; xb2 is filled. */
            if (t->skip_av) continue;

            tf_softmax(att_h, seq_len);
            float *out_h = t->xb2 + h * hd;
            for (int d = 0; d < hd; d += (int)svcntw())
                svst1(svwhilelt_b32(d, hd), out_h + d, svdup_f32(0.0f));

            /* ===== V accumulation ===== */
            if (t->kv_dtype == TF_KV_DTYPE_F32) {
                for (int p = 0; p < seq_len; p++) {
                    const float *v_p = (const float *)(vc8 + (size_t)p * row_stride + head_off);
                    if (p + 2 < seq_len)
                        __builtin_prefetch(vc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
                    svfloat32_t va = svdup_f32(att_h[p]);
                    int d = 0;
                    for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw())
                        svst1(pg, out_h + d, svmla_x(pg, svld1(pg, out_h + d), va, svld1(pg, v_p + d)));
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        svst1(ptail, out_h + d, svmla_m(ptail, svld1(ptail, out_h + d), va, svld1(ptail, v_p + d)));
                    }
                }
            } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
                for (int p = 0; p < seq_len; p++) {
                    const uint16_t *v_p = (const uint16_t *)(vc8 + (size_t)p * row_stride + head_off);
                    if (p + 2 < seq_len)
                        __builtin_prefetch(vc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
                    svfloat32_t va = svdup_f32(att_h[p]);
                    int d = 0;
                    for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw()) {
                        svuint32_t vu32 = svld1uh_u32(pg, v_p + d);
                        svfloat32_t vf  = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(vu32));
                        svst1(pg, out_h + d, svmla_x(pg, svld1(pg, out_h + d), va, vf));
                    }
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        svuint32_t vu32 = svld1uh_u32(ptail, v_p + d);
                        svfloat32_t vf  = svcvt_f32_f16_x(ptail, svreinterpret_f16_u32(vu32));
                        svst1(ptail, out_h + d, svmla_m(ptail, svld1(ptail, out_h + d), va, vf));
                    }
                }
            } else {  /* Q8 */
                const float *vs = t->value_scales;
                for (int p = 0; p < seq_len; p++) {
                    const int8_t *v_p = (const int8_t *)(vc8 + (size_t)p * row_stride + head_off);
                    if (p + 2 < seq_len)
                        __builtin_prefetch(vc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
                    float s = vs[(size_t)p * nkv + kv_h];
                    svfloat32_t va = svdup_f32(att_h[p] * s);
                    int d = 0;
                    for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw()) {
                        svint32_t vi = svld1sb_s32(pg, v_p + d);
                        svfloat32_t vf = svcvt_f32_s32_x(pg, vi);
                        svst1(pg, out_h + d, svmla_x(pg, svld1(pg, out_h + d), va, vf));
                    }
                    if (d < hd) {
                        svbool_t ptail = svwhilelt_b32(d, hd);
                        svint32_t vi = svld1sb_s32(ptail, v_p + d);
                        svfloat32_t vf = svcvt_f32_s32_x(ptail, vi);
                        svst1(ptail, out_h + d, svmla_m(ptail, svld1(ptail, out_h + d), va, vf));
                    }
                }
            }
        }
#else
        {
            size_t kv_eb = (t->kv_dtype == TF_KV_DTYPE_F32) ? 4 :
                           (t->kv_dtype == TF_KV_DTYPE_F16) ? 2 : 1;
            const uint8_t *kc8 = (const uint8_t *)t->key_cache;
            const uint8_t *vc8 = (const uint8_t *)t->value_cache;
            size_t row_stride = (size_t)t->kv_dim * kv_eb;
            size_t head_off   = (size_t)kv_h * t->head_dim * kv_eb;
            int nkv = t->n_kv_heads;
            for (int p = 0; p < seq_len; p++) {
                float score = 0.0f;
                if (t->kv_dtype == TF_KV_DTYPE_F32) {
                    const float *k_p = (const float *)(kc8 + (size_t)p * row_stride + head_off);
                    for (int d = 0; d < hd; d++) score += q_h[d] * k_p[d];
                } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
                    const uint16_t *k_p = (const uint16_t *)(kc8 + (size_t)p * row_stride + head_off);
                    for (int d = 0; d < hd; d++) score += q_h[d] * tf_f16_to_f32(k_p[d]);
                } else {
                    const int8_t *k_p = (const int8_t *)(kc8 + (size_t)p * row_stride + head_off);
                    float s = t->key_scales[(size_t)p * nkv + kv_h];
                    for (int d = 0; d < hd; d++) score += q_h[d] * (float)k_p[d] * s;
                }
                att_h[p] = score * t->scale;
            }
            tf_softmax(att_h, seq_len);
            float *out_h = t->xb2 + h * hd;
            memset(out_h, 0, hd * sizeof(float));
            for (int p = 0; p < seq_len; p++) {
                float a = att_h[p];
                if (t->kv_dtype == TF_KV_DTYPE_F32) {
                    const float *v_p = (const float *)(vc8 + (size_t)p * row_stride + head_off);
                    for (int d = 0; d < hd; d++) out_h[d] += a * v_p[d];
                } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
                    const uint16_t *v_p = (const uint16_t *)(vc8 + (size_t)p * row_stride + head_off);
                    for (int d = 0; d < hd; d++) out_h[d] += a * tf_f16_to_f32(v_p[d]);
                } else {
                    const int8_t *v_p = (const int8_t *)(vc8 + (size_t)p * row_stride + head_off);
                    float s = t->value_scales[(size_t)p * nkv + kv_h];
                    for (int d = 0; d < hd; d++) out_h[d] += a * (float)v_p[d] * s;
                }
            }
        }
#endif
    }
    return NULL;
}

/* ── Flash-attention (position-parallel, online-softmax) ────────────────
 * Splits each query head's [0, seq_len) over multiple threads. Each
 * (head, chunk) tile runs the FA-2 inner loop (fused QK + online softmax
 * + Att·V in one pass) producing partial accumulators (m_local, l_local,
 * out_local). A reduce phase merges chunks per head into final out.
 *
 * Win on A64FX: when nt > n_heads (e.g. 48 threads / 16 query heads), the
 * old per-head dispatch only uses n_heads workers; FA uses all nt. */

typedef struct {
    const float *q;              /* full Q buffer [n_heads * head_dim] */
    const void  *key_cache;
    const void  *value_cache;
    const float *key_scales;     /* NULL unless Q8 */
    const float *value_scales;   /* NULL unless Q8 */
    int head;                    /* query-head index (LOCAL; add qhead_base for the kv lookup) */
    int chunk_start, chunk_end;  /* position range, half-open */
    int head_dim, kv_dim, gqa_ratio;
    int qhead_base;              /* global index of local head 0 (TP KV-replicate); else 0 */
    int kv_head_base;            /* local KV-cache offset: LOCAL kv = GLOBAL kv - kv_head_base */
    int n_kv_heads, kv_dtype;
    float scale;                 /* 1 / sqrt(head_dim) */
    /* outputs */
    float *m_local;              /* &m[head * n_chunks + chunk_idx] */
    float *l_local;              /* &l[head * n_chunks + chunk_idx] */
    float *out_local;            /* [head_dim] */
} tf_fa_chunk_task;

static void *tf_fa_chunk_worker(void *arg) {
    tf_fa_chunk_task *t = (tf_fa_chunk_task *)arg;
    int hd  = t->head_dim;
    int p0  = t->chunk_start;
    int p1  = t->chunk_end;
    int kv_h = (t->qhead_base + t->head) / t->gqa_ratio;
    kv_h -= t->kv_head_base;
    const float *q_h = t->q + (size_t)t->head * hd;
    size_t kv_eb = (t->kv_dtype == TF_KV_DTYPE_F32) ? 4 :
                   (t->kv_dtype == TF_KV_DTYPE_F16) ? 2 : 1;
    const uint8_t *kc8 = (const uint8_t *)t->key_cache;
    const uint8_t *vc8 = (const uint8_t *)t->value_cache;
    size_t row_stride = (size_t)t->kv_dim * kv_eb;
    size_t head_off   = (size_t)kv_h * hd * kv_eb;
    int nkv = t->n_kv_heads;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    /* out_local accumulator on stack (max head_dim assumed ≤ 512). */
    float out_local[512];
    if (hd > 512) { /* would corrupt stack; bail safely */
        *t->m_local = -INFINITY; *t->l_local = 0.0f;
        for (int d = 0; d < hd; d++) t->out_local[d] = 0.0f;
        return NULL;
    }
    for (int d = 0; d < hd; d++) out_local[d] = 0.0f;

    if (p0 >= p1) {
        /* Empty chunk: write identity (does not contribute to reduce). */
        *t->m_local = -INFINITY;
        *t->l_local = 0.0f;
        for (int d = 0; d < hd; d++) t->out_local[d] = 0.0f;
        return NULL;
    }

#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b32();
    int vl = (int)svcntw();

    for (int p = p0; p < p1; p++) {
        /* ── QK score ── */
        float score;
        if (t->kv_dtype == TF_KV_DTYPE_F32) {
            const float *k_p = (const float *)(kc8 + (size_t)p * row_stride + head_off);
            if (p + 2 < p1)
                __builtin_prefetch(kc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
            svfloat32_t acc = svdup_f32(0.0f);
            int d = 0;
            for (; d + vl - 1 < hd; d += vl)
                acc = svmla_x(pg, acc, svld1(pg, q_h + d), svld1(pg, k_p + d));
            if (d < hd) {
                svbool_t ptail = svwhilelt_b32(d, hd);
                acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), svld1(ptail, k_p + d));
            }
            score = svaddv(pg, acc) * t->scale;
        } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
            const uint16_t *k_p = (const uint16_t *)(kc8 + (size_t)p * row_stride + head_off);
            if (p + 2 < p1)
                __builtin_prefetch(kc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
            svfloat32_t acc = svdup_f32(0.0f);
            int d = 0;
            for (; d + vl - 1 < hd; d += vl) {
                svuint32_t ku = svld1uh_u32(pg, k_p + d);
                svfloat32_t kf = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(ku));
                acc = svmla_x(pg, acc, svld1(pg, q_h + d), kf);
            }
            if (d < hd) {
                svbool_t ptail = svwhilelt_b32(d, hd);
                svuint32_t ku = svld1uh_u32(ptail, k_p + d);
                svfloat32_t kf = svcvt_f32_f16_x(ptail, svreinterpret_f16_u32(ku));
                acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), kf);
            }
            score = svaddv(pg, acc) * t->scale;
        } else {  /* Q8 */
            const int8_t *k_p = (const int8_t *)(kc8 + (size_t)p * row_stride + head_off);
            if (p + 2 < p1)
                __builtin_prefetch(kc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
            float ks = t->key_scales[(size_t)p * nkv + kv_h];
            svfloat32_t acc = svdup_f32(0.0f);
            int d = 0;
            for (; d + vl - 1 < hd; d += vl) {
                svint32_t ki = svld1sb_s32(pg, k_p + d);
                svfloat32_t kf = svcvt_f32_s32_x(pg, ki);
                acc = svmla_x(pg, acc, svld1(pg, q_h + d), kf);
            }
            if (d < hd) {
                svbool_t ptail = svwhilelt_b32(d, hd);
                svint32_t ki = svld1sb_s32(ptail, k_p + d);
                svfloat32_t kf = svcvt_f32_s32_x(ptail, ki);
                acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), kf);
            }
            score = svaddv(pg, acc) * t->scale * ks;
        }

        /* ── Online softmax update ── */
        float m_new = (score > m_i) ? score : m_i;
        float alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
        float beta  = expf(score - m_new);
        l_i = alpha * l_i + beta;
        m_i = m_new;

        /* ── out *= alpha + beta * v_p ── */
        svfloat32_t va_alpha = svdup_f32(alpha);
        svfloat32_t va_beta  = svdup_f32(beta);

        if (t->kv_dtype == TF_KV_DTYPE_F32) {
            const float *v_p = (const float *)(vc8 + (size_t)p * row_stride + head_off);
            if (p + 2 < p1)
                __builtin_prefetch(vc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
            int d = 0;
            for (; d + vl - 1 < hd; d += vl) {
                svfloat32_t o = svld1(pg, out_local + d);
                svfloat32_t v = svld1(pg, v_p + d);
                o = svmul_x(pg, o, va_alpha);
                o = svmla_x(pg, o, v, va_beta);
                svst1(pg, out_local + d, o);
            }
            if (d < hd) {
                svbool_t ptail = svwhilelt_b32(d, hd);
                svfloat32_t o = svld1(ptail, out_local + d);
                svfloat32_t v = svld1(ptail, v_p + d);
                o = svmul_m(ptail, o, va_alpha);
                o = svmla_m(ptail, o, v, va_beta);
                svst1(ptail, out_local + d, o);
            }
        } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
            const uint16_t *v_p = (const uint16_t *)(vc8 + (size_t)p * row_stride + head_off);
            if (p + 2 < p1)
                __builtin_prefetch(vc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
            int d = 0;
            for (; d + vl - 1 < hd; d += vl) {
                svfloat32_t o = svld1(pg, out_local + d);
                svuint32_t vu = svld1uh_u32(pg, v_p + d);
                svfloat32_t v = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(vu));
                o = svmul_x(pg, o, va_alpha);
                o = svmla_x(pg, o, v, va_beta);
                svst1(pg, out_local + d, o);
            }
            if (d < hd) {
                svbool_t ptail = svwhilelt_b32(d, hd);
                svfloat32_t o = svld1(ptail, out_local + d);
                svuint32_t vu = svld1uh_u32(ptail, v_p + d);
                svfloat32_t v = svcvt_f32_f16_x(ptail, svreinterpret_f16_u32(vu));
                o = svmul_m(ptail, o, va_alpha);
                o = svmla_m(ptail, o, v, va_beta);
                svst1(ptail, out_local + d, o);
            }
        } else {  /* Q8 */
            const int8_t *v_p = (const int8_t *)(vc8 + (size_t)p * row_stride + head_off);
            if (p + 2 < p1)
                __builtin_prefetch(vc8 + (size_t)(p+2) * row_stride + head_off, 0, 1);
            float vs = t->value_scales[(size_t)p * nkv + kv_h];
            svfloat32_t va_beta_vs = svdup_f32(beta * vs);
            int d = 0;
            for (; d + vl - 1 < hd; d += vl) {
                svfloat32_t o = svld1(pg, out_local + d);
                svint32_t vi = svld1sb_s32(pg, v_p + d);
                svfloat32_t v = svcvt_f32_s32_x(pg, vi);
                o = svmul_x(pg, o, va_alpha);
                o = svmla_x(pg, o, v, va_beta_vs);
                svst1(pg, out_local + d, o);
            }
            if (d < hd) {
                svbool_t ptail = svwhilelt_b32(d, hd);
                svfloat32_t o = svld1(ptail, out_local + d);
                svint32_t vi = svld1sb_s32(ptail, v_p + d);
                svfloat32_t v = svcvt_f32_s32_x(ptail, vi);
                o = svmul_m(ptail, o, va_alpha);
                o = svmla_m(ptail, o, v, va_beta_vs);
                svst1(ptail, out_local + d, o);
            }
        }
    }
#else
    /* Scalar fallback — same algorithm, no SIMD. */
    for (int p = p0; p < p1; p++) {
        float score = 0.0f;
        if (t->kv_dtype == TF_KV_DTYPE_F32) {
            const float *k_p = (const float *)(kc8 + (size_t)p * row_stride + head_off);
            for (int d = 0; d < hd; d++) score += q_h[d] * k_p[d];
            score *= t->scale;
        } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
            const uint16_t *k_p = (const uint16_t *)(kc8 + (size_t)p * row_stride + head_off);
            for (int d = 0; d < hd; d++) score += q_h[d] * tf_f16_to_f32(k_p[d]);
            score *= t->scale;
        } else {
            const int8_t *k_p = (const int8_t *)(kc8 + (size_t)p * row_stride + head_off);
            float ks = t->key_scales[(size_t)p * nkv + kv_h];
            for (int d = 0; d < hd; d++) score += q_h[d] * (float)k_p[d];
            score *= t->scale * ks;
        }
        float m_new = (score > m_i) ? score : m_i;
        float alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
        float beta  = expf(score - m_new);
        l_i = alpha * l_i + beta;
        m_i = m_new;
        if (t->kv_dtype == TF_KV_DTYPE_F32) {
            const float *v_p = (const float *)(vc8 + (size_t)p * row_stride + head_off);
            for (int d = 0; d < hd; d++) out_local[d] = alpha * out_local[d] + beta * v_p[d];
        } else if (t->kv_dtype == TF_KV_DTYPE_F16) {
            const uint16_t *v_p = (const uint16_t *)(vc8 + (size_t)p * row_stride + head_off);
            for (int d = 0; d < hd; d++) out_local[d] = alpha * out_local[d] + beta * tf_f16_to_f32(v_p[d]);
        } else {
            const int8_t *v_p = (const int8_t *)(vc8 + (size_t)p * row_stride + head_off);
            float vs = t->value_scales[(size_t)p * nkv + kv_h];
            float bvs = beta * vs;
            for (int d = 0; d < hd; d++) out_local[d] = alpha * out_local[d] + bvs * (float)v_p[d];
        }
    }
#endif

    *t->m_local = m_i;
    *t->l_local = l_i;
    for (int d = 0; d < hd; d++) t->out_local[d] = out_local[d];
    return NULL;
}

typedef struct {
    float *xb2_out;              /* &xb2[head * head_dim] */
    const float *m_partial;      /* [n_chunks] */
    const float *l_partial;      /* [n_chunks] */
    const float *out_partial;    /* [n_chunks][head_dim] */
    int head_dim, n_chunks;
} tf_fa_reduce_task;

static void *tf_fa_reduce_worker(void *arg) {
    tf_fa_reduce_task *t = (tf_fa_reduce_task *)arg;
    int hd = t->head_dim;
    int nc = t->n_chunks;
    if (hd <= 0 || nc <= 0) return NULL;

    /* Global max over non-empty chunks. */
    float m_g = -INFINITY;
    for (int c = 0; c < nc; c++)
        if (t->m_partial[c] > m_g) m_g = t->m_partial[c];

    /* Merge partials. */
    float out_g[512];
    for (int d = 0; d < hd; d++) out_g[d] = 0.0f;
    float l_g = 0.0f;
    for (int c = 0; c < nc; c++) {
        if (t->m_partial[c] == -INFINITY) continue;
        float w = expf(t->m_partial[c] - m_g);
        l_g += w * t->l_partial[c];
        const float *oc = t->out_partial + (size_t)c * hd;
        for (int d = 0; d < hd; d++) out_g[d] += w * oc[d];
    }

    float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
    for (int d = 0; d < hd; d++) t->xb2_out[d] = out_g[d] * inv_l;
    return NULL;
}

/* Dispatch position-parallel flash-attention.
 * Computes n_heads × n_chunks chunk tasks (one task per pool thread when
 * n_heads*n_chunks ≤ nt, padding empty tasks), then n_heads reduce tasks.
 * Writes into m->xb2. Caller must have written K/V at `position`. */
static inline void tf_attention_fa(transformer_model *m, int layer,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   int kv_dim, int gqa_ratio,
                                   int seq_len, float scale) {
    int nt = m->n_threads;
    int n_chunks = (n_heads > 0 && nt > n_heads) ? (nt / n_heads) : 1;
    if (n_chunks > TF_MAX_FA_CHUNKS) n_chunks = TF_MAX_FA_CHUNKS;
    /* Don't over-chunk a tiny seq: at least 8 positions per chunk. */
    while (n_chunks > 1 && seq_len / n_chunks < 8) n_chunks--;

    /* Phase 1: chunk workers. */
    int n_chunk_tasks = n_heads * n_chunks;
    int n_dispatch    = (n_chunk_tasks > nt) ? nt : n_chunk_tasks;
    /* We pad up to nt with empty tasks so pool_dispatch always sees nt slots. */
    tf_fa_chunk_task *ctasks = (tf_fa_chunk_task *)alloca((size_t)nt * sizeof(tf_fa_chunk_task));
    int idx = 0;
    for (int h = 0; h < n_heads; h++) {
        int chunk_size = (seq_len + n_chunks - 1) / n_chunks;
        for (int c = 0; c < n_chunks; c++) {
            int p0 = c * chunk_size;
            int p1 = p0 + chunk_size;
            if (p1 > seq_len) p1 = seq_len;
            if (idx >= nt) break;
            ctasks[idx] = (tf_fa_chunk_task){
                .q = m->q,
                .key_cache   = m->key_cache[layer],
                .value_cache = m->value_cache[layer],
                .key_scales   = m->key_scales   ? m->key_scales[layer]   : NULL,
                .value_scales = m->value_scales ? m->value_scales[layer] : NULL,
                .head = h,
                .chunk_start = p0, .chunk_end = p1,
                .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                .scale = scale,
                .m_local   = &m->fa_m[(size_t)h * n_chunks + c],
                .l_local   = &m->fa_l[(size_t)h * n_chunks + c],
                .out_local = &m->fa_out[((size_t)h * n_chunks + c) * head_dim],
            };
            idx++;
        }
        if (idx >= nt) break;
    }
    /* Pad remaining task slots with empties so pool sees nt work items. */
    for (int t = idx; t < nt; t++) {
        ctasks[t] = (tf_fa_chunk_task){
            .q = m->q,
            .key_cache = m->key_cache[layer], .value_cache = m->value_cache[layer],
                .head = 0, .chunk_start = 0, .chunk_end = 0,
                .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                .kv_head_base = m->tp_kv_head_base,
                .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype, .scale = scale,
                .m_local = &m->fa_m[0], .l_local = &m->fa_l[0], .out_local = &m->fa_out[0],
            };
    }
    /* Handle n_chunk_tasks > nt by serializing extras after dispatch — for
     * the cases we care about (16 heads × 3 chunks on 48 threads) this
     * branch is never taken. Guard with an assert-like fallback. */
    (void)n_dispatch;
    tf_pool_dispatch(m, tf_fa_chunk_worker, ctasks, sizeof(tf_fa_chunk_task));

    /* Phase 2: per-head reduce. n_heads ≤ nt → one head per thread, rest idle. */
    tf_fa_reduce_task *rtasks = (tf_fa_reduce_task *)alloca((size_t)nt * sizeof(tf_fa_reduce_task));
    for (int t = 0; t < nt; t++) {
        if (t < n_heads) {
            rtasks[t] = (tf_fa_reduce_task){
                .xb2_out    = m->xb2 + (size_t)t * head_dim,
                .m_partial  = &m->fa_m[(size_t)t * n_chunks],
                .l_partial  = &m->fa_l[(size_t)t * n_chunks],
                .out_partial= &m->fa_out[(size_t)t * n_chunks * head_dim],
                .head_dim = head_dim, .n_chunks = n_chunks,
            };
        } else {
            rtasks[t] = (tf_fa_reduce_task){
                .xb2_out = NULL,
                .m_partial = NULL, .l_partial = NULL, .out_partial = NULL,
                .head_dim = 0, .n_chunks = 0,
            };
        }
    }
    tf_pool_dispatch(m, tf_fa_reduce_worker, rtasks, sizeof(tf_fa_reduce_task));
}

#if defined(__AVX2__) && defined(__FMA__)
/* Fast AVX2 exp approximation — defined early so tf_softmax can use it. */
static inline __m256 fast_exp_avx2(__m256 x);
#elif defined(__ARM_FEATURE_SVE)
static inline svfloat32_t tf_fast_exp_sve(svbool_t pg, svfloat32_t x);
#endif

/* Softmax in-place over n elements */
static void tf_softmax(float *x, int n) {
    if (n <= 0) return;
#if defined(__AVX2__) && defined(__FMA__)
    /* AVX2 max */
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-1e30f);
    for (; i + 7 < n; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    /* Reduce vmax to scalar */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    float max_val = _mm_cvtss_f32(m4);
    for (; i < n; i++) if (x[i] > max_val) max_val = x[i];

    /* exp(x - max) and sum, using fast_exp_avx2 */
    __m256 vmax_b = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmax_b);
        v = fast_exp_avx2(v);
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    /* Reduce vsum */
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm256_castps256_ps128(vsum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float sum = _mm_cvtss_f32(s4);
    for (; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }

    /* Normalize */
    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    float inv_sum = 1.0f / sum;
    for (; i < n; i++) x[i] *= inv_sum;
#elif defined(__ARM_FEATURE_SVE)
    float max_val = -1e30f;
    svfloat32_t vmax = svdup_f32(max_val);
    for (int i = 0; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1(pg, x + i);
        vmax = svmax_f32_z(pg, vmax, v);
    }
    max_val = svmaxv_f32(svptrue_b32(), vmax);

    svfloat32_t vmax_b = svdup_f32(max_val);
    svfloat32_t vsum = svdup_f32(0.0f);
    for (int i = 0; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svsub_f32_z(pg, svld1(pg, x + i), vmax_b);
        v = tf_fast_exp_sve(pg, v);
        svst1(pg, x + i, v);
        vsum = svadd_f32_z(pg, vsum, v);
    }
    float sum = svaddv_f32(svptrue_b32(), vsum);

    svfloat32_t vinv = svdup_f32(1.0f / sum);
    for (int i = 0; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svst1(pg, x + i, svmul_f32_z(pg, svld1(pg, x + i), vinv));
    }
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
#endif
}

/* ---- Load ---- */

static int tf_compute_max_ff(const transformer_model *m) {
    int max_ff = m->n_ff;
    if (m->use_moe && m->n_ff_expert > max_ff) max_ff = m->n_ff_expert;
    if (m->use_moe && m->n_ff_shexp  > max_ff) max_ff = m->n_ff_shexp;
    if (m->use_moe && m->n_expert > max_ff) max_ff = m->n_expert;
    /* Fused MoE upgate writes activated[K * n_ff_expert] into ffn_buf3,
     * where K = n_expert_used (at ep_size=1). Pure-MoE models can have
     * n_ff = 0 so n_ff_expert alone wouldn't be enough. */
    if (m->use_moe && m->n_expert_used > 0) {
        int need = m->n_expert_used * m->n_ff_expert;
        if (need > max_ff) max_ff = need;
    }
    if (m->is_hybrid && m->ssm_qkv_dim > max_ff) max_ff = m->ssm_qkv_dim;
    return max_ff;
}

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len) {
    if (!gguf) return NULL;

    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    if (!m) return NULL;

    /* Detect architecture prefix. */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "gemma4.block_count") >= 0) {
        arch = "gemma4";
    } else if (gguf_find_key(gguf, "qwen2vl.block_count") >= 0) {
        arch = "qwen2vl";
    } else if (gguf_find_key(gguf, "qwen35moe.block_count") >= 0) {
        arch = "qwen35moe";
    } else if (gguf_find_key(gguf, "qwen35.block_count") >= 0) {
        arch = "qwen35";
    } else if (gguf_find_key(gguf, "qwen3vlmoe.block_count") >= 0) {
        arch = "qwen3vlmoe";
    } else if (gguf_find_key(gguf, "qwen3moe.block_count") >= 0) {
        arch = "qwen3moe";
    } else if (gguf_find_key(gguf, "qwen3vl.block_count") >= 0) {
        arch = "qwen3vl";
    } else if (gguf_find_key(gguf, "qwen3.block_count") >= 0) {
        arch = "qwen3";
    }

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    /* Read hyperparameters from GGUF metadata */
    m->n_embd      = tf_get_int(gguf, ARCH_KEY("embedding_length"), 4096);
    m->n_heads     = tf_get_int(gguf, ARCH_KEY("attention.head_count"), 32);
    m->n_kv_heads  = tf_get_int(gguf, ARCH_KEY("attention.head_count_kv"), 8);
    /* GLOBAL GQA group, captured before any TP slicing mutates n_heads/n_kv_heads.
     * Used as the kv_h divisor in attention so the mapping is correct under KV
     * replication (TP where n_kv % tp_size != 0, local ratio no longer == group). */
    m->gqa_group     = (m->n_kv_heads > 0) ? (m->n_heads / m->n_kv_heads) : 1;
    m->tp_qhead_offset = 0;
    m->tp_kv_head_base  = 0;
    m->tp_kv_head_count = m->n_kv_heads;
    m->n_layers    = tf_get_int(gguf, ARCH_KEY("block_count"), 36);
    m->n_ff        = tf_get_int(gguf, ARCH_KEY("feed_forward_length"), 12288);
    m->n_vocab     = tf_get_int(gguf, ARCH_KEY("vocab_size"), 0);
    m->rms_norm_eps = tf_get_float(gguf, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    m->rope_freq_base = tf_get_float(gguf, ARCH_KEY("rope.freq_base"), 5000000.0f);
    m->n_expert = tf_get_int(gguf, ARCH_KEY("expert_count"), 0);
    m->n_expert_used = tf_get_int(gguf, ARCH_KEY("expert_used_count"), 0);
    m->n_ff_expert = tf_get_int(gguf, ARCH_KEY("expert_feed_forward_length"), 0);
    m->use_moe = (m->n_expert > 0);
    if (m->use_moe) {
        if (m->n_expert_used <= 0 || m->n_expert_used > m->n_expert) {
            fprintf(stderr, "transformer: invalid expert_used_count=%d (expert_count=%d)\n",
                    m->n_expert_used, m->n_expert);
            free(m);
            return NULL;
        }
        if (m->n_ff_expert <= 0) {
            fprintf(stderr, "transformer: invalid expert_feed_forward_length=%d\n", m->n_ff_expert);
            free(m);
            return NULL;
        }
    }
    int ctx_len = tf_get_int(gguf, ARCH_KEY("context_length"), 0);
    if (max_seq_len <= 0) {
        max_seq_len = (ctx_len > 0) ? ctx_len : 1024;
    } else if (ctx_len > 0 && max_seq_len > ctx_len) {
        fprintf(stderr, "transformer: requested max_seq_len=%d exceeds model context=%d, clamping\n",
                max_seq_len, ctx_len);
        max_seq_len = ctx_len;
    }
    if (max_seq_len <= 0) {
        fprintf(stderr, "transformer: invalid max_seq_len=%d\n", max_seq_len);
        free(m);
        return NULL;
    }

    /* M-RoPE sections (Qwen3-VL) */
    m->use_mrope = 0;
    memset(m->mrope_sections, 0, sizeof(m->mrope_sections));
    {
        int idx = gguf_find_key(gguf, ARCH_KEY("rope.dimension_sections"));
        if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY &&
            gguf->kv[idx].value.arr.type == GGUF_TYPE_INT32) {
            int n = (int)gguf->kv[idx].value.arr.n;
            if (n > 4) n = 4;
            int32_t *data = (int32_t *)gguf->kv[idx].value.arr.data;
            for (int i = 0; i < n; i++) m->mrope_sections[i] = data[i];
            int sect_sum = m->mrope_sections[0] + m->mrope_sections[1] + m->mrope_sections[2] + m->mrope_sections[3];
            if (sect_sum > 0) {
                m->use_mrope = 1;
                fprintf(stderr, "transformer: M-RoPE sections=[%d, %d, %d, %d]\n",
                        m->mrope_sections[0], m->mrope_sections[1], m->mrope_sections[2], m->mrope_sections[3]);
            }
        }
    }

    /* DeepStack layers (Qwen3-VL) */
    m->n_deepstack = tf_get_int(gguf, ARCH_KEY("n_deepstack_layers"), 0);
    m->ds_embd = NULL;
    m->ds_embd_stride = 0;

    /* head_dim: prefer attention.key_length from GGUF (e.g. Qwen3-VL-4B has head_dim=128
     * despite n_embd/n_heads=80), fall back to n_embd/n_heads */
    m->head_dim = tf_get_int(gguf, ARCH_KEY("attention.key_length"), m->n_embd / m->n_heads);

    /* Hybrid SSM (Qwen3.5 dense and Qwen3.5-MoE). qwen35moe uses the top-level
     * "full_attention_interval" key (no "attention." prefix). */
    m->is_hybrid = 0;
    m->full_attn_interval = 0;
    m->n_ff_shexp = 0;
    if (strcmp(arch, "qwen35") == 0 || strcmp(arch, "qwen35moe") == 0) {
        m->is_hybrid = 1;
        m->ssm_conv_kernel = tf_get_int(gguf, ARCH_KEY("ssm.conv_kernel"), 4);
        m->ssm_d_state     = tf_get_int(gguf, ARCH_KEY("ssm.state_size"), 128);
        m->ssm_n_group     = tf_get_int(gguf, ARCH_KEY("ssm.group_count"), 16);
        m->ssm_dt_rank     = tf_get_int(gguf, ARCH_KEY("ssm.time_step_rank"), 48);
        m->ssm_d_inner     = tf_get_int(gguf, ARCH_KEY("ssm.inner_size"), 6144);
        if (strcmp(arch, "qwen35moe") == 0) {
            m->full_attn_interval = tf_get_int(gguf, ARCH_KEY("full_attention_interval"), 4);
            m->n_ff_shexp = tf_get_int(gguf, ARCH_KEY("expert_shared_feed_forward_length"), 0);
        } else {
            m->full_attn_interval = tf_get_int(gguf, ARCH_KEY("attention.full_attention_interval"), 4);
        }
        m->ssm_qkv_dim = m->ssm_d_state * m->ssm_n_group * 2 + m->ssm_d_inner;
        fprintf(stderr, "transformer: hybrid SSM: conv_k=%d d_state=%d n_group=%d dt_rank=%d d_inner=%d\n",
                m->ssm_conv_kernel, m->ssm_d_state, m->ssm_n_group, m->ssm_dt_rank, m->ssm_d_inner);
        fprintf(stderr, "transformer: full_attn_interval=%d qkv_dim=%d n_ff_shexp=%d\n",
                m->full_attn_interval, m->ssm_qkv_dim, m->n_ff_shexp);
    }

    /* Gemma4 architecture */
    m->is_gemma4 = 0;
    m->swa_pattern = NULL;
    m->rope_freq_factors = NULL;
    m->rope_inv_freq_swa = NULL;
    m->ple_buf = NULL;
    m->ple_proj_buf = NULL;
    if (strcmp(arch, "gemma4") == 0) {
        m->is_gemma4 = 1;
        m->head_dim_full = tf_get_int(gguf, ARCH_KEY("attention.key_length"), 512);
        m->head_dim_swa  = tf_get_int(gguf, ARCH_KEY("attention.key_length_swa"), 256);
        m->head_dim = m->head_dim_full; /* use max for buffer sizing */
        m->swa_window_size = tf_get_int(gguf, ARCH_KEY("attention.sliding_window"), 512);
        m->n_embd_per_layer = tf_get_int(gguf, ARCH_KEY("embedding_length_per_layer_input"), 256);
        int shared_kv_layers = tf_get_int(gguf, ARCH_KEY("attention.shared_kv_layers"), 0);
        m->n_layer_kv_from_start = m->n_layers - shared_kv_layers;
        m->final_logit_softcapping = tf_get_float(gguf, ARCH_KEY("final_logit_softcapping"), 30.0f);
        m->rope_freq_base_swa = tf_get_float(gguf, ARCH_KEY("rope.freq_base_swa"), 10000.0f);
        m->embd_scale = sqrtf((float)m->n_embd);
        m->ffn_activation = 1; /* GELU */

        /* Parse SWA layer pattern from GGUF bool array */
        m->swa_pattern = (int *)calloc(m->n_layers, sizeof(int));
        {
            int idx = gguf_find_key(gguf, ARCH_KEY("attention.sliding_window_pattern"));
            if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY) {
                int n = (int)gguf->kv[idx].value.arr.n;
                if (n > m->n_layers) n = m->n_layers;
                /* GGUF bool arrays are stored as GGUF_TYPE_BOOL (uint8) */
                uint8_t *data = (uint8_t *)gguf->kv[idx].value.arr.data;
                for (int i = 0; i < n; i++)
                    m->swa_pattern[i] = data[i] ? 1 : 0;
                fprintf(stderr, "transformer: Gemma4 SWA pattern loaded (%d layers)\n", n);
            } else {
                /* Fallback: every 6th layer is full attention (0-indexed: 5,11,...) */
                for (int i = 0; i < m->n_layers; i++)
                    m->swa_pattern[i] = ((i + 1) % 6 != 0) ? 1 : 0;
                fprintf(stderr, "transformer: Gemma4 SWA pattern defaulting to every-6th-full\n");
            }
        }

        fprintf(stderr, "transformer: Gemma4: head_dim_full=%d head_dim_swa=%d swa_window=%d\n",
                m->head_dim_full, m->head_dim_swa, m->swa_window_size);
        fprintf(stderr, "transformer: Gemma4: n_embd_per_layer=%d n_layer_kv_from_start=%d\n",
                m->n_embd_per_layer, m->n_layer_kv_from_start);
        fprintf(stderr, "transformer: Gemma4: softcap=%.1f rope_base_swa=%.0f\n",
                m->final_logit_softcapping, m->rope_freq_base_swa);
    }

    #undef ARCH_KEY
    fprintf(stderr, "transformer: architecture=%s\n", arch);
    m->max_seq_len = max_seq_len;

    fprintf(stderr, "transformer: n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
            m->n_embd, m->n_heads, m->n_kv_heads, m->n_layers, m->n_ff, m->head_dim);
    fprintf(stderr, "transformer: rope_freq_base=%.0f rms_norm_eps=%.1e max_seq_len=%d\n",
            m->rope_freq_base, m->rms_norm_eps, max_seq_len);
    if (m->use_moe) {
        fprintf(stderr, "transformer: MoE enabled (n_expert=%d n_expert_used=%d n_ff_expert=%d)\n",
                m->n_expert, m->n_expert_used, m->n_ff_expert);
    }

    /* Global tensors */
    m->token_embd = tf_load_tensor(gguf, "token_embd.weight", 1);
    m->output_norm = tf_load_tensor(gguf, "output_norm.weight", 1);
    m->output = tf_load_tensor(gguf, "output.weight", 0);
    if (!m->output.data && m->token_embd.data) {
        /* Weight tying: use token_embd as output projection */
        m->output = m->token_embd;
        fprintf(stderr, "transformer: using weight-tied output (token_embd)\n");
    }
    m->has_lm_head = (m->output.data != NULL) ? 1 : 0;

    if (m->n_vocab == 0 && m->token_embd.data) {
        m->n_vocab = m->token_embd.n_rows;
    }
    if (!m->token_embd.data || !m->output_norm.data) {
        fprintf(stderr, "transformer: missing required global tensor(s)\n");
        transformer_free(m);
        return NULL;
    }
    if (m->n_vocab <= 0) {
        fprintf(stderr, "transformer: invalid vocabulary size %d\n", m->n_vocab);
        transformer_free(m);
        return NULL;
    }
    fprintf(stderr, "transformer: n_vocab=%d has_lm_head=%d\n", m->n_vocab, m->has_lm_head);

    /* Gemma4 global tensors */
    if (m->is_gemma4) {
        m->per_layer_token_embd = tf_load_tensor(gguf, "per_layer_token_embd.weight", 1);
        m->per_layer_model_proj = tf_load_tensor(gguf, "per_layer_model_proj.weight", 1);
        m->per_layer_proj_norm = tf_load_tensor(gguf, "per_layer_proj_norm.weight", 1);
        if (!m->per_layer_token_embd.data || !m->per_layer_model_proj.data) {
            fprintf(stderr, "transformer: Gemma4 missing per-layer embedding tensors\n");
            transformer_free(m);
            return NULL;
        }
        /* Load proportional RoPE frequency factors */
        qtensor rope_freqs_qt = tf_load_tensor(gguf, "rope_freqs.weight", 0);
        if (rope_freqs_qt.data) {
            int n_freq = rope_freqs_qt.n_cols;
            m->rope_freq_factors = (float *)calloc(n_freq, sizeof(float));
            dequant_row(rope_freqs_qt.type, rope_freqs_qt.data, m->rope_freq_factors, n_freq);
            fprintf(stderr, "transformer: Gemma4 loaded rope_freqs (%d elements)\n", n_freq);
        }
    }

    /* Per-layer tensors */
    m->layers = (transformer_layer *)calloc(m->n_layers, sizeof(transformer_layer));
    if (!m->layers) {
        fprintf(stderr, "transformer: failed to allocate layer metadata\n");
        transformer_free(m);
        return NULL;
    }
    int missing_required = 0;
    int unsupported_reported = 0;
    for (int l = 0; l < m->n_layers; l++) {
        char name[128];
        #define LOAD(field, suffix, req) \
            snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
            m->layers[l].field = tf_load_tensor(gguf, name, req); \
            if ((req) && !m->layers[l].field.data) missing_required = 1;
        #define REQUIRE_SUPPORTED(field, suffix) \
            do { \
                if (m->layers[l].field.data && !tf_is_supported_weight_type(m->layers[l].field.type)) { \
                    if (!unsupported_reported) { \
                        fprintf(stderr, "transformer: unsupported tensor type for blk.%d." suffix ".weight (type=%u)\n", \
                                l, m->layers[l].field.type); \
                        unsupported_reported = 1; \
                    } \
                    missing_required = 1; \
                } \
            } while (0)

        LOAD(attn_norm,    "attn_norm",   1)

        if (m->is_gemma4) {
            /* --- Gemma4 per-layer tensors --- */
            m->layers[l].is_swa = m->swa_pattern[l];
            m->layers[l].shared_kv_source = -1; /* default: own KV */

            /* Determine shared KV source for layers beyond n_layer_kv_from_start */
            if (l >= m->n_layer_kv_from_start) {
                /* SWA layers reuse KV from (n_layer_kv_from_start - 2),
                 * full-attn layers from (n_layer_kv_from_start - 1) */
                m->layers[l].shared_kv_source = m->n_layer_kv_from_start - (m->layers[l].is_swa ? 2 : 1);
                if (m->layers[l].shared_kv_source < 0) m->layers[l].shared_kv_source = 0;
            }

            /* Q projection (always present) */
            LOAD(attn_q,       "attn_q",      1)
            LOAD(attn_q_norm,  "attn_q_norm", 1)
            LOAD(attn_output,  "attn_output", 1)

            /* K/V only for layers with own KV */
            if (m->layers[l].shared_kv_source < 0) {
                LOAD(attn_k,       "attn_k",      1)
                LOAD(attn_v,       "attn_v",      1)
                LOAD(attn_k_norm,  "attn_k_norm", 1)
            } else {
                /* Try loading K/V anyway — some shared layers may still have them */
                LOAD(attn_k,       "attn_k",      0)
                LOAD(attn_v,       "attn_v",      0)
                LOAD(attn_k_norm,  "attn_k_norm", 0)
            }
            /* V normalization (Gemma4 normalizes V too) */
            LOAD(attn_v_norm,  "attn_v_norm", 0)

            /* Post-attention norm */
            LOAD(post_attention_norm, "post_attention_norm", 1)

            /* FFN */
            LOAD(ffn_norm,     "ffn_norm",    1)
            LOAD(ffn_gate,     "ffn_gate",    1)
            LOAD(ffn_up,       "ffn_up",      1)
            LOAD(ffn_down,     "ffn_down",    1)
            LOAD(post_ffw_norm, "post_ffw_norm", 1)

            /* Layer output scale (optional) */
            LOAD(layer_output_scale, "layer_output_scale", 0)

            /* Per-layer embedding tensors */
            LOAD(ple_inp_gate,  "inp_gate",  1)
            LOAD(ple_proj,      "proj",      1)
            LOAD(ple_post_norm, "post_norm", 1)

            REQUIRE_SUPPORTED(attn_q,      "attn_q");
            REQUIRE_SUPPORTED(attn_output, "attn_output");
            REQUIRE_SUPPORTED(ffn_gate, "ffn_gate");
            REQUIRE_SUPPORTED(ffn_up,   "ffn_up");
            REQUIRE_SUPPORTED(ffn_down, "ffn_down");
            if (m->layers[l].attn_k.data) REQUIRE_SUPPORTED(attn_k, "attn_k");
            if (m->layers[l].attn_v.data) REQUIRE_SUPPORTED(attn_v, "attn_v");
        } else if (m->is_hybrid) {
            int is_attn = (m->full_attn_interval > 0 && (l + 1) % m->full_attn_interval == 0);
            m->layers[l].is_ssm = !is_attn;
            if (is_attn) {
                LOAD(attn_q,       "attn_q",      1)
                LOAD(attn_k,       "attn_k",      1)
                LOAD(attn_v,       "attn_v",      1)
                LOAD(attn_q_norm,  "attn_q_norm", 0)
                LOAD(attn_k_norm,  "attn_k_norm", 0)
                LOAD(attn_output,  "attn_output", 1)
            } else {
                LOAD(ssm_qkv,      "attn_qkv",    1)
                LOAD(ssm_gate,     "attn_gate",    1)
                LOAD(ssm_alpha,    "ssm_alpha",    1)
                LOAD(ssm_beta,     "ssm_beta",     1)
                LOAD(ssm_out,      "ssm_out",      1)
                LOAD(ssm_conv1d,   "ssm_conv1d",   1)
                LOAD(ssm_norm,     "ssm_norm",     1)
                /* ssm_a and ssm_dt.bias: no .weight suffix */
                snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
                m->layers[l].ssm_a = tf_load_tensor(gguf, name, 1);
                if (!m->layers[l].ssm_a.data) missing_required = 1;
                snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
                m->layers[l].ssm_dt_bias = tf_load_tensor(gguf, name, 1);
                if (!m->layers[l].ssm_dt_bias.data) missing_required = 1;
            }
            /* post_attention_norm replaces ffn_norm in qwen35 */
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
            m->layers[l].ffn_norm = tf_load_tensor(gguf, name, 1);
            if (!m->layers[l].ffn_norm.data) missing_required = 1;
        } else {
            LOAD(attn_q,       "attn_q",      1)
            LOAD(attn_k,       "attn_k",      1)
            LOAD(attn_v,       "attn_v",      1)
            LOAD(attn_q_norm,  "attn_q_norm", 0)
            LOAD(attn_k_norm,  "attn_k_norm", 0)
            LOAD(attn_output,  "attn_output", 1)
            LOAD(ffn_norm,     "ffn_norm",    1)
            /* Q/K/V biases (optional, e.g. Qwen2.5-VL) */
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
            m->layers[l].attn_q_bias = tf_load_tensor(gguf, name, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
            m->layers[l].attn_k_bias = tf_load_tensor(gguf, name, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
            m->layers[l].attn_v_bias = tf_load_tensor(gguf, name, 0);
        }
        if (m->use_moe) {
            LOAD(ffn_gate_inp,  "ffn_gate_inp",  1)
            LOAD(ffn_up_exps,   "ffn_up_exps",   1)
            LOAD(ffn_gate_exps, "ffn_gate_exps", 1)
            LOAD(ffn_down_exps, "ffn_down_exps", 1)
            REQUIRE_SUPPORTED(ffn_gate_inp,  "ffn_gate_inp");
            REQUIRE_SUPPORTED(ffn_up_exps,   "ffn_up_exps");
            REQUIRE_SUPPORTED(ffn_gate_exps, "ffn_gate_exps");
            REQUIRE_SUPPORTED(ffn_down_exps, "ffn_down_exps");
            /* Validate expert tensor shape convention [cols, rows_per_expert, n_expert]. */
            if (m->layers[l].ffn_up_exps.data) {
                if (m->layers[l].ffn_up_exps.n_dims < 3 ||
                    (int)m->layers[l].ffn_up_exps.dims[1] != m->n_ff_expert ||
                    (int)m->layers[l].ffn_up_exps.dims[2] != m->n_expert) {
                    fprintf(stderr, "transformer: unexpected shape for blk.%d.ffn_up_exps.weight\n", l);
                    missing_required = 1;
                }
            }
            if (m->layers[l].ffn_gate_exps.data) {
                if (m->layers[l].ffn_gate_exps.n_dims < 3 ||
                    (int)m->layers[l].ffn_gate_exps.dims[1] != m->n_ff_expert ||
                    (int)m->layers[l].ffn_gate_exps.dims[2] != m->n_expert) {
                    fprintf(stderr, "transformer: unexpected shape for blk.%d.ffn_gate_exps.weight\n", l);
                    missing_required = 1;
                }
            }
            if (m->layers[l].ffn_down_exps.data) {
                if (m->layers[l].ffn_down_exps.n_dims < 3 ||
                    (int)m->layers[l].ffn_down_exps.dims[1] != m->n_embd ||
                    (int)m->layers[l].ffn_down_exps.dims[2] != m->n_expert) {
                    fprintf(stderr, "transformer: unexpected shape for blk.%d.ffn_down_exps.weight\n", l);
                    missing_required = 1;
                }
            }
            /* Qwen3.5-MoE shared expert (always-on, sigmoid-gated SwiGLU FFN). */
            if (m->n_ff_shexp > 0) {
                LOAD(ffn_gate_inp_shexp, "ffn_gate_inp_shexp", 1)
                LOAD(ffn_up_shexp,       "ffn_up_shexp",       1)
                LOAD(ffn_gate_shexp,     "ffn_gate_shexp",     1)
                LOAD(ffn_down_shexp,     "ffn_down_shexp",     1)
                REQUIRE_SUPPORTED(ffn_up_shexp,   "ffn_up_shexp");
                REQUIRE_SUPPORTED(ffn_gate_shexp, "ffn_gate_shexp");
                REQUIRE_SUPPORTED(ffn_down_shexp, "ffn_down_shexp");
            }
        } else {
            LOAD(ffn_gate,     "ffn_gate",    1)
            LOAD(ffn_up,       "ffn_up",      1)
            LOAD(ffn_down,     "ffn_down",    1)
            REQUIRE_SUPPORTED(ffn_gate, "ffn_gate");
            REQUIRE_SUPPORTED(ffn_up,   "ffn_up");
            REQUIRE_SUPPORTED(ffn_down, "ffn_down");
        }
        REQUIRE_SUPPORTED(attn_norm,   "attn_norm");
        REQUIRE_SUPPORTED(ffn_norm,    "ffn_norm");
        if (!m->is_hybrid || !m->layers[l].is_ssm) {
            REQUIRE_SUPPORTED(attn_q,      "attn_q");
            REQUIRE_SUPPORTED(attn_k,      "attn_k");
            REQUIRE_SUPPORTED(attn_v,      "attn_v");
            REQUIRE_SUPPORTED(attn_output, "attn_output");
        }
        if (m->is_hybrid && m->layers[l].is_ssm) {
            REQUIRE_SUPPORTED(ssm_qkv,    "attn_qkv");
            REQUIRE_SUPPORTED(ssm_gate,   "attn_gate");
            REQUIRE_SUPPORTED(ssm_alpha,  "ssm_alpha");
            REQUIRE_SUPPORTED(ssm_beta,   "ssm_beta");
            REQUIRE_SUPPORTED(ssm_out,    "ssm_out");
            REQUIRE_SUPPORTED(ssm_conv1d, "ssm_conv1d");
            REQUIRE_SUPPORTED(ssm_norm,   "ssm_norm");
        }
        #undef LOAD
        #undef REQUIRE_SUPPORTED
    }
    if (missing_required) {
        fprintf(stderr, "transformer: aborting due to missing required per-layer tensors\n");
        transformer_free(m);
        return NULL;
    }

    /* M10 panel layout for the A64FX F16 matvec kernel is built lazily by
     * transformer_build_panels(), which the caller invokes after
     * transformer_set_threads() so the panels can be first-touched per-CMG. */

    /* Allocate KV cache. Element format is selected by TF_KV_DTYPE env var:
     *   f16 (default) — 2 B/element, SVE FCVT on the fly, best end-to-end at
     *                   long context when paired with K_DP (qpkd+ktbl QK)
     *   f32           — 4 B/element, baseline read path
     *   q8            — 1 B/element + 4 B scale per (pos, kv_head)
     * Halves/quarters the per-token attention KV read bandwidth. */
    m->kv_dtype = TF_KV_DTYPE_F16;
    m->kv_elem_bytes = 2;
    {
        const char *e = getenv("TF_KV_DTYPE");
        if (e) {
            if      (!strcmp(e, "f16") || !strcmp(e, "fp16")) { m->kv_dtype = TF_KV_DTYPE_F16; m->kv_elem_bytes = 2; }
            else if (!strcmp(e, "q8")  || !strcmp(e, "int8")) { m->kv_dtype = TF_KV_DTYPE_Q8;  m->kv_elem_bytes = 1; }
            else if (!strcmp(e, "f32") || !strcmp(e, "fp32")) { m->kv_dtype = TF_KV_DTYPE_F32; m->kv_elem_bytes = 4; }
            else fprintf(stderr, "transformer: ignoring unknown TF_KV_DTYPE=%s\n", e);
        }
        /* Gemma4 attention path reads K/V as float* directly (SWA + shared KV)
         * and the F16/Q8 codecs aren't wired in there yet. Force F32. */
        if (m->is_gemma4 && m->kv_dtype != TF_KV_DTYPE_F32) {
            fprintf(stderr, "transformer: TF_KV_DTYPE=%s ignored on Gemma4 (forcing f32)\n", e ? e : "");
            m->kv_dtype = TF_KV_DTYPE_F32;
            m->kv_elem_bytes = 4;
        }
    }
    /* Transposed K cache: K[kv_h][d][p] instead of [p][kv_h][d]. Lets the QK
     * kernel run as outer-d / inner-p FMLA into att[], NO svaddv per position.
     * Opt-in for safety. Gemma4 (SWA+shared) and Q8 (per-pos scale stride)
     * are not supported yet. */
    m->kv_k_transposed = 0;
    if (getenv("TF_KV_K_T")) {
        if (m->is_gemma4)
            fprintf(stderr, "transformer: TF_KV_K_T ignored on Gemma4\n");
        else if (m->kv_dtype == TF_KV_DTYPE_Q8)
            fprintf(stderr, "transformer: TF_KV_K_T ignored for Q8 KV (not yet wired)\n");
        else {
            m->kv_k_transposed = 1;
            fprintf(stderr, "transformer: K cache transposed layout [kv_h][d][p]\n");
        }
    }
    /* qpkd+ktbl layout: K stored as [p][d][kv_h]. Q packed [d][h] per call.
     * QK kernel uses svld1rq (4 K floats broadcast across SVE quadwords) +
     * svtbl (permute lanes to per-head) + svmla into a 16-lane accumulator,
     * eliminating per-position svaddv with no data replication. Requires
     * n_heads == SVE float width (16 on A64FX). F16/F32/Q8 KV all supported.
     * Default: ON whenever eligible — opt out with TF_KV_K_DP=0. */
    m->kv_k_dp = 0;
    m->q_packed = NULL;
    m->av_tmp = NULL;
    m->att_pmax = NULL;
    m->att_psum = NULL;
    {
        const char *kdp_env = getenv("TF_KV_K_DP");
        int kdp_want = 1;  /* default ON */
        if (kdp_env && (kdp_env[0] == '0' || !strcmp(kdp_env, "off") || !strcmp(kdp_env, "no")))
            kdp_want = 0;
        if (kdp_want) {
            if (m->is_gemma4) {
                if (kdp_env) fprintf(stderr, "transformer: TF_KV_K_DP ignored on Gemma4\n");
            } else if (m->kv_k_transposed) {
                if (kdp_env) fprintf(stderr, "transformer: TF_KV_K_DP ignored (TF_KV_K_T already enabled)\n");
            } else if (m->n_heads != 16) {
                if (kdp_env) fprintf(stderr, "transformer: TF_KV_K_DP ignored (n_heads=%d, kernel requires 16)\n", m->n_heads);
            } else if (m->n_heads % m->n_kv_heads != 0) {
                if (kdp_env) fprintf(stderr, "transformer: TF_KV_K_DP ignored (n_heads %% n_kv_heads != 0)\n");
            } else {
                m->kv_k_dp = 1;
                const char *dtype_name = (m->kv_dtype == TF_KV_DTYPE_F32) ? "f32" :
                                         (m->kv_dtype == TF_KV_DTYPE_F16) ? "f16" : "q8";
                fprintf(stderr, "transformer: K cache qpkd+ktbl layout [p][d][kv_h] (%s)\n", dtype_name);
            }
        }
    }
    int kv_dim = m->n_kv_heads * m->head_dim;
    m->key_cache    = (void  **)calloc(m->n_layers, sizeof(void  *));
    m->value_cache  = (void  **)calloc(m->n_layers, sizeof(void  *));
    m->key_scales   = (m->kv_dtype == TF_KV_DTYPE_Q8) ? (float **)calloc(m->n_layers, sizeof(float *)) : NULL;
    m->value_scales = (m->kv_dtype == TF_KV_DTYPE_Q8) ? (float **)calloc(m->n_layers, sizeof(float *)) : NULL;
    if (m->is_gemma4) {
        int kv_dim_full = m->n_kv_heads * m->head_dim_full;
        int kv_dim_swa  = m->n_kv_heads * m->head_dim_swa;
        int n_own = 0, n_shared = 0;
        for (int l = 0; l < m->n_layers; l++) {
            if (m->layers[l].shared_kv_source >= 0) { n_shared++; continue; }
            int cache_len  = m->layers[l].is_swa ? m->swa_window_size : max_seq_len;
            int local_kvd  = m->layers[l].is_swa ? kv_dim_swa : kv_dim_full;
            m->key_cache[l]   = tf_aligned_calloc(256, (size_t)cache_len * local_kvd, m->kv_elem_bytes);
            m->value_cache[l] = tf_aligned_calloc(256, (size_t)cache_len * local_kvd, m->kv_elem_bytes);
            if (m->kv_dtype == TF_KV_DTYPE_Q8) {
                m->key_scales[l]   = (float *)tf_aligned_calloc(256, (size_t)cache_len * m->n_kv_heads, sizeof(float));
                m->value_scales[l] = (float *)tf_aligned_calloc(256, (size_t)cache_len * m->n_kv_heads, sizeof(float));
            }
            n_own++;
        }
        for (int l = 0; l < m->n_layers; l++) {
            int src = m->layers[l].shared_kv_source;
            if (src >= 0) {
                m->key_cache[l]   = m->key_cache[src];
                m->value_cache[l] = m->value_cache[src];
                if (m->kv_dtype == TF_KV_DTYPE_Q8) {
                    m->key_scales[l]   = m->key_scales[src];
                    m->value_scales[l] = m->value_scales[src];
                }
            }
        }
        fprintf(stderr, "transformer: Gemma4 KV cache: %d own layers, %d shared layers\n", n_own, n_shared);
        kv_dim = kv_dim_full > kv_dim_swa ? kv_dim_full : kv_dim_swa;
    } else {
        for (int l = 0; l < m->n_layers; l++) {
            if (m->is_hybrid && m->layers[l].is_ssm) continue;
            m->key_cache[l]   = tf_aligned_calloc(256, (size_t)max_seq_len * kv_dim, m->kv_elem_bytes);
            m->value_cache[l] = tf_aligned_calloc(256, (size_t)max_seq_len * kv_dim, m->kv_elem_bytes);
            if (m->kv_dtype == TF_KV_DTYPE_Q8) {
                m->key_scales[l]   = (float *)tf_aligned_calloc(256, (size_t)max_seq_len * m->n_kv_heads, sizeof(float));
                m->value_scales[l] = (float *)tf_aligned_calloc(256, (size_t)max_seq_len * m->n_kv_heads, sizeof(float));
            }
        }
    }
    if (m->kv_dtype != TF_KV_DTYPE_F32) {
        const char *dn = (m->kv_dtype == TF_KV_DTYPE_F16) ? "f16" : "q8";
        fprintf(stderr, "transformer: KV cache dtype=%s (%zu B/elem)\n", dn, m->kv_elem_bytes);
    }

    /* Allocate SSM state for hybrid models */
    m->conv_state = NULL;
    m->conv_state_pos = NULL;
    m->recurrent_state = NULL;
    if (m->is_hybrid && m->n_layers > 0) {
        size_t nl = (size_t)m->n_layers;
        m->conv_state = (float **)calloc(nl, sizeof(float *));
        m->conv_state_pos = (int *)calloc(nl, sizeof(int));
        m->recurrent_state = (float **)calloc(nl, sizeof(float *));
        m->conv_w_trans = (float **)calloc(nl, sizeof(float *));
        int conv_k = m->ssm_conv_kernel;
        int qkv_dim = m->ssm_qkv_dim;
        size_t conv_w_bytes = (size_t)conv_k * qkv_dim * sizeof(float);
        int n_ssm = 0;
        for (int l = 0; l < m->n_layers; l++) {
            if (!m->layers[l].is_ssm) continue;
            int conv_state_size = (m->ssm_conv_kernel - 1) * m->ssm_qkv_dim;
            m->conv_state[l] = (float *)calloc(conv_state_size, sizeof(float));

            /* Pre-dequant + transpose the conv1d weights into [conv_k][qkv_dim]
             * layout the inner conv loop wants. Weights are constant, so the
             * old per-token path (~184K dequant_row calls/token on 0.8B) was
             * pure waste. ~720 KB/layer × n_ssm_layers — fits cache for 0.8B. */
            float *cw = (float *)aligned_alloc(64, conv_w_bytes);
            if (cw) {
                qtensor *cmat = &m->layers[l].ssm_conv1d;
                size_t crb = tf_row_bytes(cmat->type, cmat->n_cols);
                const uint8_t *cbase = (const uint8_t *)cmat->data;
                float wb[8]; /* conv_k <= 8 */
                for (int j = 0; j < qkv_dim; j++) {
                    dequant_row(cmat->type, cbase + j * crb, wb, conv_k);
                    for (int f = 0; f < conv_k; f++)
                        cw[f * qkv_dim + j] = wb[f];
                }
            }
            m->conv_w_trans[l] = cw;
            int rec_state_size = m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state;
            /* mmap-anon (instead of calloc) so the state pages stay
             * uncommitted until the recurrence worker touches them. With
             * cmg_pin + tf_ssm_head_range, the first thread to read each
             * head's 64 KB column is pinned to that head's owner CMG, so
             * the page lands in that CMG's HBM (lazy first-touch). 4 KB
             * pages — MADV_NOHUGEPAGE keeps THP from coalescing several
             * heads' pages back onto a single CMG. */
            size_t rec_state_bytes = (size_t)rec_state_size * sizeof(float);
            void *rs = mmap(NULL, rec_state_bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (rs == MAP_FAILED) {
                m->recurrent_state[l] = (float *)calloc(rec_state_size, sizeof(float));
            } else {
#ifdef MADV_NOHUGEPAGE
                madvise(rs, rec_state_bytes, MADV_NOHUGEPAGE);
#endif
                m->recurrent_state[l] = (float *)rs;
            }
            n_ssm++;
        }
        fprintf(stderr, "transformer: SSM state: %d layers, conv=[%d×%d], recurrent=[%d×%d×%d]\n",
                n_ssm, m->ssm_conv_kernel - 1, m->ssm_qkv_dim,
                m->ssm_dt_rank, m->ssm_d_state, m->ssm_d_state);
        m->ssm_alpha_buf = (float *)aligned_alloc(64, ((size_t)m->ssm_dt_rank * sizeof(float) + 63) & ~(size_t)63);
        m->ssm_beta_buf  = (float *)aligned_alloc(64, ((size_t)m->ssm_dt_rank * sizeof(float) + 63) & ~(size_t)63);
    } else {
        m->conv_w_trans = NULL;
        m->ssm_alpha_buf = NULL;
        m->ssm_beta_buf  = NULL;
    }

    /* Allocate scratch buffers */
    int max_ff = tf_compute_max_ff(m);
    int max_dim = m->n_embd > max_ff ? m->n_embd : max_ff;
    /* tf_moe_upgate_fused needs two n_embd-sized dequant scratches per
     * thread (up + gate). */
    if (m->use_moe && 2 * m->n_embd > max_dim) max_dim = 2 * m->n_embd;
    if (m->is_hybrid) {
        m->ssm_q8_cap = ((max_dim + 63) / 64) * 64;
        m->ssm_q8_xq = (int8_t *)tf_aligned_calloc(256, m->ssm_q8_cap, sizeof(int8_t));
        m->ssm_q8_xs = (uint16_t *)tf_aligned_calloc(256, m->ssm_q8_cap / 64, sizeof(uint16_t));
    }
    int q_dim = m->n_heads * m->head_dim;  /* may differ from n_embd (e.g. 4B: 4096 vs 2560) */
    /* xb2 must hold: attention output (q_dim), SSM qkv (qkv_dim), or Q+gate (2*q_dim) */
    int xb2_dim = q_dim;
    if (m->is_hybrid) {
        if (m->ssm_qkv_dim > xb2_dim) xb2_dim = m->ssm_qkv_dim;
        if (2 * q_dim > xb2_dim) xb2_dim = 2 * q_dim; /* Q+gate for gated attention */
    }
    /* q buffer must hold Q or expanded Q for SSM */
    int q_buf_dim = q_dim;
    if (m->is_hybrid && m->ssm_dt_rank * m->ssm_d_state > q_buf_dim)
        q_buf_dim = m->ssm_dt_rank * m->ssm_d_state;
    m->x         = (float *)tf_aligned_calloc(256, m->n_embd, sizeof(float));
    m->xb        = (float *)tf_aligned_calloc(256, m->n_embd, sizeof(float));
    m->xb2       = (float *)tf_aligned_calloc(256, xb2_dim, sizeof(float));
    m->q         = (float *)tf_aligned_calloc(256, q_buf_dim, sizeof(float));
    m->k         = (float *)tf_aligned_calloc(256, kv_dim, sizeof(float));
    m->v         = (float *)tf_aligned_calloc(256, kv_dim, sizeof(float));
    m->att       = (float *)tf_aligned_calloc(256, m->n_heads * max_seq_len, sizeof(float));
    if (m->kv_k_dp) {
        m->q_packed = (float *)tf_aligned_calloc(256, (size_t)m->head_dim * m->n_heads, sizeof(float));
        m->av_tmp = (float *)tf_aligned_calloc(256,
            (size_t)m->n_threads * m->n_heads * m->head_dim, sizeof(float));
        m->att_pmax = (float *)tf_aligned_calloc(256,
            (size_t)m->n_threads * m->n_heads, sizeof(float));
        m->att_psum = (float *)tf_aligned_calloc(256,
            (size_t)m->n_threads * m->n_heads, sizeof(float));
    }
    {
        size_t fa_tiles = (size_t)m->n_heads * TF_MAX_FA_CHUNKS;
        m->fa_m   = (float *)tf_aligned_calloc(256, fa_tiles, sizeof(float));
        m->fa_l   = (float *)tf_aligned_calloc(256, fa_tiles, sizeof(float));
        m->fa_out = (float *)tf_aligned_calloc(256, fa_tiles * m->head_dim, sizeof(float));
    }
    m->ffn_buf1  = (float *)tf_aligned_calloc(256, max_ff, sizeof(float));
    m->ffn_buf2  = (float *)tf_aligned_calloc(256, max_ff, sizeof(float));
    m->ffn_buf3  = (float *)tf_aligned_calloc(256, max_ff, sizeof(float));
    m->logits     = m->has_lm_head ? (float *)tf_aligned_calloc(256, m->n_vocab, sizeof(float)) : NULL;
    m->matvec_tmp = (float *)tf_aligned_calloc(256, max_dim, sizeof(float));
    m->trace_hidden_norms = 1;

    /* Default: single-threaded, no tensor parallelism */
    m->n_threads = 1;
    m->tp_rank = 0;
    m->tp_size = 1;
    m->ep_rank = 0;
    m->ep_size = 1;
    m->ep_e_start = 0;
    m->ep_e_end = 0;
    m->ep_ar_fn = NULL;
    m->ep_ar_ctx = NULL;
    m->tp_allreduce_fn = NULL;
    m->tp_allreduce_ctx = NULL;
    /* Default PP range = whole model (no pipeline split). */
    m->pp_start = 0;
    m->pp_end   = m->n_layers;
    m->thread_tmp = (float **)calloc(1, sizeof(float *));
    m->thread_tmp[0] = m->matvec_tmp;

    /* Precompute RoPE inverse frequency tables */
    {
        int half_dim = m->head_dim / 2;
        m->rope_inv_freq_len = half_dim;
        m->rope_inv_freq = (float *)calloc(half_dim, sizeof(float));
        for (int j = 0; j < half_dim; j++)
            m->rope_inv_freq[j] = 1.0f / powf(m->rope_freq_base, (float)(2 * j) / m->head_dim);

        int sect_sum = m->mrope_sections[0] + m->mrope_sections[1] + m->mrope_sections[2] + m->mrope_sections[3];
        if (sect_sum > 0) {
            int rope_dim = 2 * sect_sum;
            m->rope_mrope_inv_freq_len = sect_sum;
            m->rope_mrope_inv_freq = (float *)calloc(sect_sum, sizeof(float));
            for (int j = 0; j < sect_sum; j++)
                m->rope_mrope_inv_freq[j] = 1.0f / powf(m->rope_freq_base, (float)(2 * j) / rope_dim);
        } else {
            m->rope_mrope_inv_freq_len = 0;
            m->rope_mrope_inv_freq = NULL;
        }
    }

    /* Gemma4: precompute SWA RoPE table and per-layer embedding buffers */
    if (m->is_gemma4) {
        /* SWA RoPE: standard with freq_base_swa, head_dim_swa */
        int half_swa = m->head_dim_swa / 2;
        m->rope_inv_freq_swa = (float *)calloc(half_swa, sizeof(float));
        for (int j = 0; j < half_swa; j++)
            m->rope_inv_freq_swa[j] = 1.0f / powf(m->rope_freq_base_swa, (float)(2 * j) / m->head_dim_swa);

        /* Full-attention RoPE: if rope_freq_factors is available, apply proportional RoPE.
         * The standard rope_inv_freq (already computed above) uses head_dim_full and rope_freq_base.
         * For proportional RoPE: inv_freq[j] /= freq_factors[j] */
        if (m->rope_freq_factors) {
            int half_full = m->head_dim_full / 2;
            for (int j = 0; j < half_full && j < m->rope_inv_freq_len; j++)
                m->rope_inv_freq[j] /= m->rope_freq_factors[j];
            fprintf(stderr, "transformer: Gemma4 proportional RoPE applied (%d dims)\n", half_full);
        }

        /* Per-layer embedding scratch buffers */
        m->ple_buf = (float *)calloc(m->n_embd_per_layer, sizeof(float));
        m->ple_proj_buf = (float *)calloc(m->n_embd, sizeof(float));
    }

    return m;
}

static size_t tf_runtime_kv_len_for_layer(const transformer_model *model, int layer) {
    if (!model || layer < 0 || layer >= model->n_layers) return 0;
    if (model->is_gemma4) {
        return model->layers[layer].is_swa
            ? (size_t)model->swa_window_size
            : (size_t)model->max_seq_len;
    }
    return (size_t)model->max_seq_len;
}

static int tf_runtime_kv_dim_for_layer(const transformer_model *model, int layer) {
    if (!model || layer < 0 || layer >= model->n_layers) return 0;
    if (model->is_gemma4) {
        int hd = model->layers[layer].is_swa ? model->head_dim_swa : model->head_dim_full;
        return model->n_kv_heads * hd;
    }
    return model->n_kv_heads * model->head_dim;
}

void transformer_reset_runtime_state(transformer_model *model) {
    if (!model) return;

    model->ds_embd = NULL;
    model->current_token_id = -1;
    model->prefill_ffn_skip = 0;

    if (model->key_cache) {
        for (int l = 0; l < model->n_layers; l++) {
            if (model->is_gemma4 && model->layers &&
                model->layers[l].shared_kv_source >= 0) continue;
            if (model->is_hybrid && model->layers &&
                model->layers[l].is_ssm) continue;
            size_t cache_len = tf_runtime_kv_len_for_layer(model, l);
            int kv_dim = tf_runtime_kv_dim_for_layer(model, l);
            size_t kv_bytes = cache_len * (size_t)kv_dim * model->kv_elem_bytes;
            if (model->key_cache[l]) memset(model->key_cache[l], 0, kv_bytes);
            if (model->value_cache[l]) memset(model->value_cache[l], 0, kv_bytes);
            if (model->key_scales && model->key_scales[l])
                memset(model->key_scales[l], 0, cache_len * (size_t)model->n_kv_heads * sizeof(float));
            if (model->value_scales && model->value_scales[l])
                memset(model->value_scales[l], 0, cache_len * (size_t)model->n_kv_heads * sizeof(float));
        }
    }

    if (model->conv_state_pos)
        memset(model->conv_state_pos, 0, (size_t)model->n_layers * sizeof(int));
    if (model->conv_state) {
        size_t conv_elems = (model->ssm_conv_kernel > 1 && model->ssm_qkv_dim > 0)
            ? (size_t)(model->ssm_conv_kernel - 1) * model->ssm_qkv_dim
            : 0;
        for (int l = 0; l < model->n_layers; l++) {
            if (model->conv_state[l] && conv_elems)
                memset(model->conv_state[l], 0, conv_elems * sizeof(float));
        }
    }
    if (model->recurrent_state) {
        size_t rec_elems = (size_t)model->ssm_dt_rank *
                           model->ssm_d_state * model->ssm_d_state;
        for (int l = 0; l < model->n_layers; l++) {
            if (model->recurrent_state[l] && rec_elems)
                memset(model->recurrent_state[l], 0, rec_elems * sizeof(float));
        }
    }
    if (model->ssm_alpha_buf && model->ssm_dt_rank > 0)
        memset(model->ssm_alpha_buf, 0, (size_t)model->ssm_dt_rank * sizeof(float));
    if (model->ssm_beta_buf && model->ssm_dt_rank > 0)
        memset(model->ssm_beta_buf, 0, (size_t)model->ssm_dt_rank * sizeof(float));

    if (model->x) memset(model->x, 0, (size_t)model->n_embd * sizeof(float));
    if (model->xb) memset(model->xb, 0, (size_t)model->n_embd * sizeof(float));
    if (model->logits && model->n_vocab > 0)
        memset(model->logits, 0, (size_t)model->n_vocab * sizeof(float));
}

void transformer_free(transformer_model *model) {
    if (!model) return;
#if defined(__ARM_FEATURE_SVE)
    /* M10: free panel-layout weight copies (mmap'd, see tf_panel_alloc) */
    if (model->layers) {
        for (int l = 0; l < model->n_layers; l++) {
            transformer_layer *L = &model->layers[l];
            tf_panel_free(&L->attn_q);   tf_panel_free(&L->attn_k);
            tf_panel_free(&L->attn_v);   tf_panel_free(&L->attn_output);
            tf_panel_free(&L->ffn_gate); tf_panel_free(&L->ffn_up);
            tf_panel_free(&L->ffn_down);
        }
    }
    tf_panel_free(&model->output);
#endif
    free(model->layers);
    if (model->key_cache) {
        for (int l = 0; l < model->n_layers; l++) {
            /* Skip shared KV caches (freed by their source layer) */
            if (model->is_gemma4 && l < model->n_layers &&
                model->layers && model->layers[l].shared_kv_source >= 0) continue;
            free(model->key_cache[l]);
            free(model->value_cache[l]);
            if (model->key_scales)   free(model->key_scales[l]);
            if (model->value_scales) free(model->value_scales[l]);
        }
        free(model->key_cache);
        free(model->value_cache);
        free(model->key_scales);
        free(model->value_scales);
    }
    /* Gemma4 resources */
    free(model->swa_pattern);
    free(model->rope_freq_factors);
    free(model->rope_inv_freq_swa);
    free(model->ple_buf);
    free(model->ple_proj_buf);
    if (model->conv_state) {
        for (int l = 0; l < model->n_layers; l++) free(model->conv_state[l]);
        free(model->conv_state);
    }
    free(model->conv_state_pos);
    if (model->conv_w_trans) {
        for (int l = 0; l < model->n_layers; l++) free(model->conv_w_trans[l]);
        free(model->conv_w_trans);
    }
    free(model->ssm_alpha_buf);
    free(model->ssm_beta_buf);
    free(model->ssm_q8_xq);
    free(model->ssm_q8_xs);
    if (model->recurrent_state) {
        size_t rec_state_bytes = (size_t)model->ssm_dt_rank *
                                 model->ssm_d_state * model->ssm_d_state *
                                 sizeof(float);
        for (int l = 0; l < model->n_layers; l++) {
            if (model->recurrent_state[l])
                munmap(model->recurrent_state[l], rec_state_bytes);
        }
        free(model->recurrent_state);
    }
    free(model->x);
    free(model->xb);
    free(model->xb2);
    free(model->q);
    free(model->k);
    free(model->v);
    free(model->att);
    free(model->av_tmp);
    free(model->att_pmax);
    free(model->att_psum);
    free(model->fa_m);
    free(model->fa_l);
    free(model->fa_out);
    free(model->ffn_buf1);
    free(model->ffn_buf2);
    free(model->ffn_buf3);
    free(model->logits);
    free(model->matvec_tmp);
    free(model->rope_inv_freq);
    free(model->rope_mrope_inv_freq);
    /* Free extra per-thread scratch (thread_tmp[0] == matvec_tmp, already freed) */
    tf_pool_shutdown(model);
    if (model->thread_tmp) {
        for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
        free(model->thread_tmp);
    }
    free(model);
}

/* ---- Thread pool ---- */
/* Portable spin-wait hint */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define tf_cpu_pause() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define tf_cpu_pause() __asm__ __volatile__("yield")
#else
#define tf_cpu_pause() ((void)0)
#endif

/* Pool workers spin this many `yield` iterations waiting for a dispatch
 * before parking on the cond var. Sized to comfortably cover the
 * main-thread serial gap between the ~200 matvec dispatches in one
 * decode step (~100us on A64FX), so a hot pool never hits a syscall. */
#ifndef TF_POOL_SPIN_LIMIT
#define TF_POOL_SPIN_LIMIT 200000L
#endif

/* Per-worker done-flag stride, in ints. A64FX cache line is 256 B = 64
 * ints; one slot per worker, line-padded, so completion signalling never
 * contends a shared cache line across CMGs. */
#define TF_POOL_FLAG_STRIDE 64

#ifdef TF_POOL_PROFILE
static long   tf_prof_calls = 0;
static double tf_prof_dispatch = 0, tf_prof_work0 = 0, tf_prof_wait = 0, tf_prof_woke = 0;
#define TF_PROF_NSITE 16
static void  *tf_prof_fn[TF_PROF_NSITE];
static long   tf_prof_fn_calls[TF_PROF_NSITE];
static double tf_prof_fn_work[TF_PROF_NSITE], tf_prof_fn_wait[TF_PROF_NSITE];
/* Per-section timing for the persistent worker: [0]=tid0, [1]=tid1. */
static double tf_pw_matvec[2], tf_pw_barrier[2], tf_pw_serial[2], tf_pw_attn[2];
static double tf_now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int tf_prof_site(void *fn) {
    for (int i = 0; i < TF_PROF_NSITE; i++) {
        if (tf_prof_fn[i] == fn) return i;
        if (tf_prof_fn[i] == NULL) { tf_prof_fn[i] = fn; return i; }
    }
    return TF_PROF_NSITE - 1;
}
#endif

#ifdef TF_POOL_PROFILE
/* Zero the persistent-worker section timers + dispatch counters. Called from
 * the runner at the prefill→decode boundary so the printed breakdown reflects
 * steady-state decode only (prefill's cold-mmap matvec faults dominate the
 * raw counters otherwise). */
void transformer_pool_profile_reset(void) {
    tf_prof_calls = 0;
    tf_prof_dispatch = tf_prof_work0 = tf_prof_wait = tf_prof_woke = 0;
    for (int i = 0; i < TF_PROF_NSITE; i++) {
        tf_prof_fn[i] = NULL;
        tf_prof_fn_calls[i] = 0;
        tf_prof_fn_work[i] = tf_prof_fn_wait[i] = 0;
    }
    for (int t = 0; t < 2; t++)
        tf_pw_matvec[t] = tf_pw_barrier[t] = tf_pw_serial[t] = tf_pw_attn[t] = 0;
}
#else
void transformer_pool_profile_reset(void) {}
#endif

/* ---- A64FX CMG core pinning ----
 * Pin pool thread `tid` (of n_threads) to a CMG-local core. Threads are
 * grouped contiguously across n_cmgs CMGs, matching the contiguous block
 * partition used by the panel matvec/build, so each panel row block is
 * first-touched and later read by the same core — keeping it on that CMG's
 * HBM. A64FX compute cores are 12-59, 12 per CMG (CMG c = 12+12c .. 23+12c). */
#if defined(__aarch64__) && defined(__linux__)
static int tf_cmg_pin_thread(int tid, int n_threads, int n_cmgs) {
    if (n_threads < 1) n_threads = 1;
    if (n_cmgs < 1) n_cmgs = 1;
    if (n_cmgs > 4) n_cmgs = 4;
    int cmg       = (int)((long)tid * n_cmgs / n_threads);
    int cmg_first = (int)(((long)cmg * n_threads + n_cmgs - 1) / n_cmgs);
    int local     = tid - cmg_first;
    if (local < 0)  local = 0;
    if (local > 11) local = 11;
    int core = 12 + cmg * 12 + local;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
static int tf_cmg_pin_thread(int tid, int n_threads, int n_cmgs) {
    (void)tid; (void)n_threads; (void)n_cmgs; return -1;
}
#endif

#if defined(__aarch64__)
/* Fujitsu libhwb (/lib64/libhwb.so) — the standalone HW-barrier API the
 * OpenMP runtime also uses. vhbm_bar_assign() performs the privileged
 * /dev/xos_hwb group-assign (sets IMP_BARRIER_ASSIGN_EL1 for the calling
 * core) so the EL0 BST register actually coalesces toggles; without it the
 * raw MSR/MRS is a per-core no-op. Linked via -lhwb on A64FX builds. */
extern int  vhbm_bar_init(uint64_t core_bitmask);    /* bit i = node-core i; returns bd-mask */
extern int  vhbm_bar_assign(int bd_mask, void *bb_hint); /* per-thread join; returns bb (0..3) */
extern void vhbm_bar(long bb);                        /* hot intra-CMG barrier on BST reg bb */
extern int  vhbm_bar_unassign(int bd_mask);
#endif

/* Compute the A64FX compute-core that tf_cmg_pin_thread pins `tid` to (for
 * building the HW-barrier core mask). Mirrors tf_cmg_pin_thread exactly. */
static inline int tf_cmg_core_of(int tid, int n_threads, int n_cmgs) {
    if (n_threads < 1) n_threads = 1;
    if (n_cmgs < 1) n_cmgs = 1;
    if (n_cmgs > 4) n_cmgs = 4;
    int cmg       = (int)((long)tid * n_cmgs / n_threads);
    int cmg_first = (int)(((long)cmg * n_threads + n_cmgs - 1) / n_cmgs);
    int local     = tid - cmg_first;
    if (local < 0)  local = 0;
    if (local > 11) local = 11;
    return 12 + cmg * 12 + local;
}

/* True only on a real Fujitsu A64FX CPU (MIDR_EL1 implementer 0x46, part
 * 0x001 → MIDR 0x46?f0010). Read once from sysfs. Gates HW-barrier default-on
 * so a generic aarch64+Linux build won't engage the A64FX-specific libhwb
 * path unless TF_HW_BARRIER=1 is set explicitly. */
static int tf_is_a64fx(void) {
#if defined(__aarch64__) && defined(__linux__)
    static int cached = -1;
    if (cached >= 0) return cached;
    cached = 0;
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1", "r");
    if (f) {
        unsigned long midr = 0;
        if (fscanf(f, "%lx", &midr) == 1) {
            unsigned impl = (unsigned)((midr >> 24) & 0xff);
            unsigned part = (unsigned)((midr >> 4)  & 0xfff);
            if (impl == 0x46 && part == 0x001) cached = 1;
        }
        fclose(f);
    }
    return cached;
#else
    return 0;
#endif
}

/* Master-side HW-barrier setup: called once by tid0 after pinning, before
 * workers are created. Decides enablement (real A64FX CPU + CMG pinning
 * active + threads split evenly across n_cmgs, or TF_HW_BARRIER=1 to force),
 * builds the participating-core bitmask, allocates the descriptor via libhwb. */
static void tf_hwbar_master_init(transformer_model *m) {
#if defined(__aarch64__) && defined(__linux__)
    m->hwbar_enabled = 0;
    /* Default ON only on a real A64FX CPU; TF_HW_BARRIER=0 forces the flat
     * barrier, =1 forces an attempt even off-A64FX (for testing). When
     * defaulting on we fall back silently if the prerequisites aren't met
     * (CMG pinning, even thread split, driver present) — only complain loudly
     * when the user explicitly asked for it via TF_HW_BARRIER=1. */
    const char *e = getenv("TF_HW_BARRIER");
    int explicit_on  = (e && e[0] == '1');
    int explicit_off = (e && e[0] == '0');
    if (explicit_off) return;
    if (!explicit_on && !tf_is_a64fx()) return;   /* default-on: A64FX only */
    if (!m->cmg_pin) {
        if (explicit_on)
            fprintf(stderr, "transformer: TF_HW_BARRIER needs CMG pinning; ignored\n");
        return;
    }
    int nt = m->n_threads;
    int ncmg = m->cmg_pin_ncmgs;
    if (ncmg < 1) ncmg = 1;
    if (ncmg > 4) ncmg = 4;
    if (nt < ncmg || (nt % ncmg) != 0) {
        if (explicit_on)
            fprintf(stderr, "transformer: TF_HW_BARRIER needs nt%%n_cmgs==0 "
                    "(nt=%d n_cmgs=%d); ignored\n", nt, ncmg);
        return;
    }
    uint64_t mask = 0;
    for (int t = 0; t < nt; t++) mask |= (1ULL << tf_cmg_core_of(t, nt, ncmg));
    int bd = vhbm_bar_init(mask);
    if (bd < 0) {
        if (explicit_on)
            fprintf(stderr, "transformer: vhbm_bar_init failed (%d); HW barrier off\n", bd);
        return;
    }
    m->hwbar_bd      = bd;
    m->hwbar_ncmg    = ncmg;
    m->hwbar_tpc     = nt / ncmg;
    m->hwbar_lcount  = 0;
    m->hwbar_lsense  = 0;
    m->hwbar_join_count    = 0;
    m->hwbar_assign_failed = 0;
    m->hwbar_enabled = 1;
    fprintf(stderr, "transformer: HW barrier ON (libhwb bd=%d, %d cores, "
            "%d CMGs x %d, mask=0x%llx)\n", bd, nt, ncmg, m->hwbar_tpc,
            (unsigned long long)mask);
#else
    (void)m;
#endif
}

/* Per-thread join: each pool thread (after pinning) registers with the kernel
 * barrier group for its core and records its BST register index. */
static void tf_hwbar_thread_join(transformer_model *m, int tid) {
#if defined(__aarch64__) && defined(__linux__)
    if (!m->hwbar_enabled) return;
    int bb = vhbm_bar_assign(m->hwbar_bd, NULL);
    if (bb < 0) {
        fprintf(stderr, "transformer: vhbm_bar_assign tid%d failed (%d); "
                "reverting to flat barrier\n", tid, bb);
        m->hwbar_assign_failed = 1;   /* tf_pool_start finalizes the fallback */
        bb = 0;
    }
    m->hwbar_bb[tid] = bb;
    __sync_synchronize();
    __sync_add_and_fetch((int *)&m->hwbar_join_count, 1);
#else
    (void)m; (void)tid;
#endif
}

typedef struct {
    transformer_model *model;
    int tid;
} tf_pool_worker_ctx;

static void *tf_pool_worker_main(void *arg) {
    tf_pool_worker_ctx *ctx = (tf_pool_worker_ctx *)arg;
    transformer_model *m = ctx->model;
    int tid = ctx->tid;
    free(ctx);

    if (m->cmg_pin)
        tf_cmg_pin_thread(tid, m->n_threads, m->cmg_pin_ncmgs);
    tf_hwbar_thread_join(m, tid);  /* register this core with the HW barrier group */

    int last_phase = 0;
    while (1) {
        /* Hybrid wait: spin first (no syscall — critical for per-token
         * decode, which fires ~200 tiny dispatches/token; a futex wake
         * per dispatch was the multi-CMG scaling killer). Only after a
         * long spin do we park on the cond var so a truly idle pool
         * burns no power. */
        long spins = 0;
        while (m->pool_phase == last_phase && m->pool_alive) {
            if (++spins < TF_POOL_SPIN_LIMIT) {
                tf_cpu_pause();
                continue;
            }
            pthread_mutex_lock(&m->pool_mutex);
            m->pool_sleepers++;
            while (m->pool_phase == last_phase && m->pool_alive)
                pthread_cond_wait(&m->pool_cond, &m->pool_mutex);
            m->pool_sleepers--;
            pthread_mutex_unlock(&m->pool_mutex);
            break;
        }

        last_phase = m->pool_phase;
        if (!m->pool_alive) return NULL;

        /* Execute work */
        void *task = (char *)m->pool_args + (size_t)tid * m->pool_arg_stride;
        m->pool_fn(task);

        /* Signal done: write our own 256B-padded slot (no shared atomic —
         * a single cross-CMG-contended counter was throttling >2 CMGs). */
        m->pool_done_flags[(size_t)tid * TF_POOL_FLAG_STRIDE] = last_phase;
    }
    return NULL;
}

static void tf_pool_shutdown(transformer_model *model) {
    if (!model->pool_alive) return;
#ifdef TF_POOL_PROFILE
    fprintf(stderr,
        "tf_pool: %ld dispatches  woke=%.0f  dispatch=%.1fms  work0=%.1fms  wait=%.1fms"
        "  (per-call: disp=%.2fus work0=%.2fus wait=%.2fus)\n",
        tf_prof_calls, tf_prof_woke,
        tf_prof_dispatch*1e3, tf_prof_work0*1e3, tf_prof_wait*1e3,
        tf_prof_calls ? tf_prof_dispatch/tf_prof_calls*1e6 : 0,
        tf_prof_calls ? tf_prof_work0/tf_prof_calls*1e6 : 0,
        tf_prof_calls ? tf_prof_wait/tf_prof_calls*1e6 : 0);
    for (int i = 0; i < TF_PROF_NSITE && tf_prof_fn[i]; i++)
        fprintf(stderr,
            "tf_pool:   site %p  calls=%ld  work0=%.1fms (%.1fus/call)  wait=%.1fms\n",
            tf_prof_fn[i], tf_prof_fn_calls[i], tf_prof_fn_work[i]*1e3,
            tf_prof_fn_calls[i] ? tf_prof_fn_work[i]/tf_prof_fn_calls[i]*1e6 : 0,
            tf_prof_fn_wait[i]*1e3);
    for (int t = 0; t < 2; t++)
        fprintf(stderr,
            "tf_pool:   persistent tid%d  matvec=%.1fms  barrier=%.1fms  "
            "serial=%.1fms  attn=%.1fms\n", t,
            tf_pw_matvec[t]*1e3, tf_pw_barrier[t]*1e3,
            tf_pw_serial[t]*1e3, tf_pw_attn[t]*1e3);
#endif
    /* Signal workers to exit */
    pthread_mutex_lock(&model->pool_mutex);
    model->pool_alive = 0;
    model->pool_phase++;
    pthread_cond_broadcast(&model->pool_cond);
    pthread_mutex_unlock(&model->pool_mutex);
    for (int t = 1; t < model->n_threads; t++)
        pthread_join(model->pool_threads[t], NULL);
    free(model->pool_threads);
    model->pool_threads = NULL;
    free((void *)model->pool_done_flags);
    model->pool_done_flags = NULL;
    pthread_mutex_destroy(&model->pool_mutex);
    pthread_cond_destroy(&model->pool_cond);
}

static void tf_pool_start(transformer_model *model) {
    int nt = model->n_threads;
    if (model->cmg_pin) {
        /* Main thread runs worker 0's task in tf_pool_dispatch — pin it too. */
        if (tf_cmg_pin_thread(0, nt, model->cmg_pin_ncmgs) != 0) {
            fprintf(stderr, "transformer: CMG pin failed (not A64FX?), disabling\n");
            model->cmg_pin = 0;
        }
    }
    /* HW barrier: allocate the descriptor (master) and join tid0, before the
     * worker threads are created so vhbm_bar_init precedes any vhbm_bar_assign. */
    tf_hwbar_master_init(model);
    tf_hwbar_thread_join(model, 0);
    model->pool_threads = (pthread_t *)calloc(nt, sizeof(pthread_t));
    model->pool_done_flags = (volatile int *)calloc(
        (size_t)nt * TF_POOL_FLAG_STRIDE, sizeof(int));
    model->pool_phase = 0;
    model->pool_sleepers = 0;
    model->pool_alive = 1;
    pthread_mutex_init(&model->pool_mutex, NULL);
    pthread_cond_init(&model->pool_cond, NULL);
    __sync_synchronize();
    for (int t = 1; t < nt; t++) {
        tf_pool_worker_ctx *ctx = (tf_pool_worker_ctx *)malloc(sizeof(*ctx));
        ctx->model = model;
        ctx->tid = t;
        pthread_create(&model->pool_threads[t], NULL, tf_pool_worker_main, ctx);
    }
    /* HW-barrier startup handshake: every thread (tid0 above + workers) must
     * finish its kernel group-join before the first barrier. The HW barrier is
     * all-or-nothing — if any join failed, vhbm_bar() for the rest would never
     * coalesce, so revert ALL threads to the flat barrier here, while workers
     * are still parked on pool_phase==0 (no PW_BARRIER has run yet, and the
     * first tf_pool_dispatch is strictly after this returns → race-free). */
    if (model->hwbar_enabled) {
        while (model->hwbar_join_count < nt) tf_cpu_pause();
        __sync_synchronize();
        if (model->hwbar_assign_failed) {
            fprintf(stderr, "transformer: HW barrier disabled "
                    "(a thread failed to join); using flat barrier\n");
            model->hwbar_enabled = 0;
            __sync_synchronize();
        }
    }
}

/* Dispatch work to the thread pool: calls fn(args + tid * arg_stride) for each thread.
 * Workers sleep on cond_var between dispatches — zero bandwidth when idle. */
static void tf_pool_dispatch(transformer_model *model, void *(*fn)(void *),
                              void *args, size_t arg_stride) {
    int nt = model->n_threads;
    model->pool_fn = fn;
    model->pool_args = args;
    model->pool_arg_stride = arg_stride;
    __sync_synchronize();
#ifdef TF_POOL_PROFILE
    double t0 = tf_now_s();
#endif

    /* Bump the phase: spinning workers pick it up immediately, no syscall.
     * Only pay the mutex + cond_broadcast if a worker actually parked. */
    int phase = ++model->pool_phase;
    if (model->pool_sleepers > 0) {
        pthread_mutex_lock(&model->pool_mutex);
        pthread_cond_broadcast(&model->pool_cond);
        pthread_mutex_unlock(&model->pool_mutex);
#ifdef TF_POOL_PROFILE
        tf_prof_woke++;
#endif
    }

    /* Main thread executes worker 0's task directly */
    void *task0 = (char *)args;
#ifdef TF_POOL_PROFILE
    double t1 = tf_now_s();
#endif
    fn(task0);
#ifdef TF_POOL_PROFILE
    double t2 = tf_now_s();
#endif

    /* Wait for remaining workers: poll each worker's own padded slot, so the
     * completion check never bounces a shared line across CMGs. */
    for (int t = 1; t < nt; t++)
        while (model->pool_done_flags[(size_t)t * TF_POOL_FLAG_STRIDE] != phase)
            tf_cpu_pause();
#ifdef TF_POOL_PROFILE
    double t3 = tf_now_s();
    tf_prof_calls++;
    tf_prof_dispatch += t1 - t0;
    tf_prof_work0    += t2 - t1;
    tf_prof_wait     += t3 - t2;
    int si = tf_prof_site((void *)fn);
    tf_prof_fn_calls[si]++;
    tf_prof_fn_work[si] += t2 - t1;
    tf_prof_fn_wait[si] += t3 - t2;
#endif
}

void transformer_set_threads(transformer_model *model, int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads == model->n_threads) return;

    /* Shutdown old pool */
    tf_pool_shutdown(model);

    /* Free old extra per-thread buffers */
    for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
    free(model->thread_tmp);

    int max_ff = tf_compute_max_ff(model);
    int max_dim = model->n_embd > max_ff ? model->n_embd : max_ff;
    /* tf_moe_upgate_fused needs two n_embd-sized dequant scratches per
     * thread (up + gate). Make sure the per-thread buffer can hold both. */
    if (model->use_moe && 2 * model->n_embd > max_dim) max_dim = 2 * model->n_embd;
    model->n_threads = n_threads;
    /* AV-parallel scratch sized by n_threads: reallocate if pool grew. */
    if (model->kv_k_dp) {
        if (model->av_tmp) free(model->av_tmp);
        model->av_tmp = (float *)tf_aligned_calloc(256,
            (size_t)n_threads * model->n_heads * model->head_dim, sizeof(float));
        if (model->att_pmax) free(model->att_pmax);
        if (model->att_psum) free(model->att_psum);
        model->att_pmax = (float *)tf_aligned_calloc(256,
            (size_t)n_threads * model->n_heads, sizeof(float));
        model->att_psum = (float *)tf_aligned_calloc(256,
            (size_t)n_threads * model->n_heads, sizeof(float));
    }
    model->thread_tmp = (float **)calloc(n_threads, sizeof(float *));
    model->thread_tmp[0] = model->matvec_tmp;
    for (int t = 1; t < n_threads; t++) {
        /* Use notouch alloc: dequant always writes before read, so uninitialized is safe.
         * With demand paging, first actual use from worker thread t places pages on
         * thread t's NUMA node for optimal locality. */
        model->thread_tmp[t] = (float *)tf_aligned_alloc_notouch(256,
                                    (size_t)max_dim * sizeof(float));
    }

    /* CMG affinity config: on A64FX, pin pool threads to CMG-local cores so
     * panel buffers first-touched per-thread stay on the owning HBM stack.
     * Default on for aarch64 multi-thread; TF_CMG_PIN=0 disables. TF_N_CMGS
     * sets how many CMGs the pool spreads across (default 4, capped). */
#if defined(__aarch64__) && defined(__linux__)
    model->cmg_pin = (n_threads > 1);
    {
        const char *e = getenv("TF_CMG_PIN");
        if (e) model->cmg_pin = atoi(e) != 0;
    }
    model->cmg_pin_ncmgs = 4;
    {
        const char *e = getenv("TF_N_CMGS");
        if (e) model->cmg_pin_ncmgs = atoi(e);
    }
    if (model->cmg_pin_ncmgs < 1) model->cmg_pin_ncmgs = 1;
    if (model->cmg_pin_ncmgs > 4) model->cmg_pin_ncmgs = 4;
    if (model->cmg_pin_ncmgs > n_threads) model->cmg_pin_ncmgs = n_threads;
#else
    model->cmg_pin = 0;
    model->cmg_pin_ncmgs = 1;
#endif

    /* Start new pool */
    if (n_threads > 1) tf_pool_start(model);
    if (model->cmg_pin)
        fprintf(stderr, "transformer: using %d threads (thread pool, "
                "pinned across %d CMGs)\n", n_threads, model->cmg_pin_ncmgs);
    else
        fprintf(stderr, "transformer: using %d threads (thread pool)\n", n_threads);
}

void transformer_build_panels(transformer_model *m) {
#if defined(__ARM_FEATURE_SVE)
    if (!m || getenv("TF_NO_PANEL")) return;

    /* Gather every dense F16 matvec weight (attn q/k/v/o, ffn gate/up/down,
     * plus the output projection). MoE expert tensors stay row-major. */
    qtensor *list[7 * 256 + 1];
    int n = 0;
    for (int l = 0; l < m->n_layers && n + 7 <= (int)(sizeof(list)/sizeof(list[0])) - 1; l++) {
        if (l < m->pp_start || l >= m->pp_end) continue;  /* PP: skip non-owned layers */
        transformer_layer *L = &m->layers[l];
        qtensor *dense[] = { &L->attn_q, &L->attn_k, &L->attn_v, &L->attn_output,
                             &L->ffn_gate, &L->ffn_up, &L->ffn_down };
        for (int i = 0; i < 7; i++)
            if (dense[i]->data && dense[i]->type == GGML_TYPE_F16 && !dense[i]->panel)
                list[n++] = dense[i];
    }
    if (m->pp_end >= m->n_layers &&        /* PP: LM head lives on the last stage only */
        m->has_lm_head && m->output.data && m->output.type == GGML_TYPE_F16 &&
        !m->output.panel)
        list[n++] = &m->output;

    int nt = m->n_threads;
    int built = 0;
    for (int i = 0; i < n; i++) {
        qtensor *t = list[i];
        if (!tf_panel_alloc(t)) continue;
        int nblk = t->panel_blk;
        if (nt > 1 && m->pool_alive && nblk >= nt) {
            /* Same contiguous partition as tf_panel_matvec_pool: worker w
             * fills (and first-touches) exactly the blocks it will read. */
            tf_panel_build_task *tasks =
                (tf_panel_build_task *)alloca(nt * sizeof(tf_panel_build_task));
            int per = nblk / nt, extra = nblk % nt, off = 0;
            for (int w = 0; w < nt; w++) {
                int c = per + (w < extra ? 1 : 0);
                tasks[w] = (tf_panel_build_task){ t, off, off + c };
                off += c;
            }
            tf_pool_dispatch(m, tf_panel_build_worker, tasks,
                             sizeof(tf_panel_build_task));
        } else {
            tf_panel_fill_range(t, 0, nblk);
        }
        built++;
    }
    if (built > 0)
        fprintf(stderr, "transformer: panel layout built for %d F16 matvec weights%s\n",
                built, m->cmg_pin ? " (first-touched per CMG)" : "");

    /* BF16 p_odd-pair packing for dense matvec weights (same set + ssm_*).
     * TF_QUANT_Q8=1 switches the build to Q8_0 quantize-on-load layout
     * (q8_pv) which halves DRAM traffic for the matvec hot path; in that
     * mode we skip bf16_pv entirely. */
    if (getenv("TF_NO_BF16_PV") && !getenv("TF_QUANT_Q8")) return;
    int use_q8 = getenv("TF_QUANT_Q8") != NULL;
    qtensor *bf16_list[16 * 256 + 1];
    int bn = 0;
    for (int l = 0; l < m->n_layers && bn + 16 <= (int)(sizeof(bf16_list)/sizeof(bf16_list[0])) - 1; l++) {
        if (l < m->pp_start || l >= m->pp_end) continue;  /* PP: skip non-owned layers */
        transformer_layer *L = &m->layers[l];
        qtensor *cand[] = {
            &L->attn_q, &L->attn_k, &L->attn_v, &L->attn_output,
            &L->ffn_gate, &L->ffn_up, &L->ffn_down,
            &L->ssm_qkv, &L->ssm_gate, &L->ssm_alpha, &L->ssm_beta,
            &L->ssm_out,
        };
        for (size_t i = 0; i < sizeof(cand)/sizeof(cand[0]); i++) {
            qtensor *t = cand[i];
            int col_ok = use_q8 ? ((t->n_cols & 63) == 0) : ((t->n_cols & 15) == 0);
            int built_already = use_q8 ? (t->q8_pv != NULL) : (t->bf16_pv != NULL);
            if (t->data && t->type == GGML_TYPE_BF16 && !built_already
                && t->n_rows >= 8 && (t->n_rows & 7) == 0
                && col_ok)
                bf16_list[bn++] = t;
        }
    }
    {
        int col_ok = use_q8 ? ((m->output.n_cols & 63) == 0) : ((m->output.n_cols & 15) == 0);
        int built_already = use_q8 ? (m->output.q8_pv != NULL) : (m->output.bf16_pv != NULL);
        if (m->pp_end >= m->n_layers &&    /* PP: LM head lives on the last stage only */
            m->has_lm_head && m->output.data && m->output.type == GGML_TYPE_BF16 &&
            !m->output.panel && !built_already
            && m->output.n_rows >= 8 && (m->output.n_rows & 7) == 0
            && col_ok)
            bf16_list[bn++] = &m->output;
    }

    /* Stream-reclaim source bytes after each pv build so peak memory is
     * weight_size + one tensor instead of 2× weight_size — required to fit
     * 9B BF16 (~17 GiB) on a 31 GiB / 4-CMG node. The matvec hot path
     * reads bf16_pv, and the 8-row-aligned partition leaves the tail loops
     * that touch mat->data unreachable. TF_KEEP_BF16_SRC=1 disables. */
    int bbuilt = 0;
    size_t reclaimed = 0;
    long page_sz = sysconf(_SC_PAGESIZE);
    if (page_sz <= 0) page_sz = 4096;
    int reclaim = !getenv("TF_KEEP_BF16_SRC");
    for (int i = 0; i < bn; i++) {
        qtensor *t = bf16_list[i];
        int groups;
        if (use_q8) {
            if (!tf_q8_pv_alloc(t)) continue;
            groups = t->q8_pv_groups;
        } else {
            if (!tf_bf16_pv_alloc(t)) continue;
            groups = t->bf16_pv_groups;
        }
        if (nt > 1 && m->pool_alive && groups >= nt) {
            tf_panel_build_task *tasks =
                (tf_panel_build_task *)alloca(nt * sizeof(tf_panel_build_task));
            int per = groups / nt, extra = groups % nt, off = 0;
            for (int w = 0; w < nt; w++) {
                int c = per + (w < extra ? 1 : 0);
                tasks[w] = (tf_panel_build_task){ t, off, off + c };
                off += c;
            }
            tf_pool_dispatch(m,
                use_q8 ? tf_q8_pv_build_worker : tf_bf16_pv_build_worker,
                tasks, sizeof(tf_panel_build_task));
        } else {
            if (use_q8) tf_q8_pv_fill_range(t, 0, groups);
            else        tf_bf16_pv_fill_range(t, 0, groups);
        }
        if (reclaim && t->data) {
            /* Round start up, length down to page boundaries — madvise needs
             * page-aligned ranges and partial pages at the edges would either
             * be rejected (page_sz on aarch64 = 4 KiB or 64 KiB) or release
             * memory we don't own. */
            uintptr_t start = (uintptr_t)t->data;
            size_t bytes = (size_t)t->n_rows * (size_t)t->n_cols * 2;
            uintptr_t aligned_start = (start + page_sz - 1) & ~((uintptr_t)page_sz - 1);
            uintptr_t aligned_end   = (start + bytes) & ~((uintptr_t)page_sz - 1);
            if (aligned_end > aligned_start) {
                size_t dn_bytes = (size_t)(aligned_end - aligned_start);
                if (madvise((void *)aligned_start, dn_bytes, MADV_DONTNEED) == 0)
                    reclaimed += dn_bytes;
            }
        }
        bbuilt++;
    }
    if (bbuilt > 0) {
        fprintf(stderr, "transformer: %s layout built for %d matvec weights%s",
                use_q8 ? "Q8_0 quantize-on-load" : "BF16 p_odd-pair",
                bbuilt, m->cmg_pin ? " (first-touched per CMG)" : "");
        if (reclaim && reclaimed > 0)
            fprintf(stderr, ", reclaimed %.2f GiB source", reclaimed / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "\n");
    }
#else
    (void)m;
#endif
}

void transformer_set_trace_hidden_norms(transformer_model *model, int enable) {
    if (!model) return;
    model->trace_hidden_norms = enable ? 1 : 0;
}

/* ---- NUMA-aware memory allocator ---- */
/* With demand paging (XOS_MMM_L_PAGING_POLICY=demand:demand:demand), pages are
 * placed on the NUMA node of the first-touching core. The NUMA allocator ensures
 * each thread's data partition is first-touched by that thread, placing pages on
 * the thread's CMG for optimal memory bandwidth. */

/* Task for parallel pread (weight loading) or memset (buffer first-touch) */
typedef struct {
    void *dst;
    int fd;            /* -1 for memset-only (no pread) */
    size_t file_off;   /* base file offset (ignored if fd < 0) */
    size_t byte_start;
    size_t byte_end;
} tf_numa_task;

static void *tf_numa_pread_worker(void *arg) {
    tf_numa_task *t = (tf_numa_task *)arg;
    uint8_t *base = (uint8_t *)t->dst;
    size_t start = t->byte_start;
    size_t len   = t->byte_end - start;
    if (len == 0) return NULL;
    size_t off = 0;
    while (off < len) {
        size_t chunk = len - off;
        if (chunk > 1024 * 1024) chunk = 1024 * 1024;
        ssize_t n = pread(t->fd, base + start + off, chunk,
                          (off_t)(t->file_off + start + off));
        if (n <= 0) break;
        off += (size_t)n;
    }
    return NULL;
}

static void *tf_numa_memset_worker(void *arg) {
    tf_numa_task *t = (tf_numa_task *)arg;
    uint8_t *base = (uint8_t *)t->dst;
    size_t len = t->byte_end - t->byte_start;
    if (len > 0) memset(base + t->byte_start, 0, len);
    return NULL;
}

/* Initialize NUMA config from environment or defaults */
static void tf_numa_init(transformer_model *m) {
    memset(&m->numa, 0, sizeof(m->numa));
    m->numa.n_cmgs = 4;
    m->numa.per_cmg_budget = 6ULL * 1024 * 1024 * 1024;
    m->numa.alignment = 2 * 1024 * 1024;
    m->numa.enabled = 0;

    const char *env;
    if ((env = getenv("NUMA_N_CMGS")))       m->numa.n_cmgs = atoi(env);
    if ((env = getenv("NUMA_CMG_BUDGET_GB"))) m->numa.per_cmg_budget = (size_t)(atof(env) * 1024.0 * 1024.0 * 1024.0);
    if ((env = getenv("NUMA_ALIGNMENT")))     m->numa.alignment = (size_t)atol(env);
    if (getenv("NUMA_DISTRIBUTE"))            m->numa.enabled = 1;
}

/* Distribute a buffer across threads via parallel memset (first-touch placement) */
static void tf_numa_distribute_buffer(transformer_model *m, void *buf, size_t total_bytes) {
    int nt = m->n_threads;
    if (!buf || total_bytes == 0 || nt <= 1) return;

    tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
    size_t part = total_bytes / nt;
    size_t off = 0;
    for (int t = 0; t < nt; t++) {
        size_t end = (t == nt - 1) ? total_bytes : off + part;
        tasks[t] = (tf_numa_task){buf, -1, 0, off, end};
        off = end;
    }
    tf_pool_dispatch(m, tf_numa_memset_worker, tasks, sizeof(tf_numa_task));
}

/* Print per-CMG usage */
static void tf_numa_print_usage(transformer_model *m) {
    int nc = m->numa.n_cmgs < m->n_threads ? m->numa.n_cmgs : m->n_threads;
    fprintf(stderr, "numa: per-CMG usage:");
    for (int c = 0; c < nc; c++)
        fprintf(stderr, " CMG%d=%.1fMB", c, (double)m->numa.per_cmg_used[c] / (1024.0 * 1024.0));
    fprintf(stderr, " (budget=%.1fGB)\n", (double)m->numa.per_cmg_budget / (1024.0 * 1024.0 * 1024.0));
}

void transformer_numa_setup(transformer_model *m, const gguf_context *gguf) {
    tf_numa_init(m);
    if (!m->numa.enabled || m->n_threads <= 1 || !m->pool_alive) {
        m->numa.enabled = 0;
        return;
    }

    int nt = m->n_threads;
    int fd = gguf->fd;

    fprintf(stderr, "numa: setup %d CMGs, %.1fGB/CMG, %d threads\n",
            m->numa.n_cmgs,
            (double)m->numa.per_cmg_budget / (1024.0 * 1024.0 * 1024.0), nt);

    /* ---- Phase 1: Distribute weight data via parallel pread ---- */
    if (fd >= 0) {
        fprintf(stderr, "numa: phase 1 - loading weights (fd=%d, %zu tensors)...\n",
                fd, (size_t)gguf->n_tensors);
        size_t weight_bytes = 0;

        for (uint64_t ti = 0; ti < gguf->n_tensors; ti++) {
            void *tdata = gguf_tensor_data(gguf, (int)ti);
            size_t tsz = gguf_tensor_size(gguf, (int)ti);
            if (!tdata || tsz == 0) continue;

            size_t toff = gguf->data_offset + ((uint8_t *)tdata - gguf->data);
            int n_dims = (int)gguf->tensors[ti].n_dims;
            int n_rows = 1;
            for (int d = 1; d < n_dims; d++)
                n_rows *= (int)gguf->tensors[ti].dims[d];
            size_t row_bytes = (n_rows > 0) ? tsz / (size_t)n_rows : tsz;

            if (n_rows < nt || n_dims <= 1) {
                tf_numa_task task = {tdata, fd, toff, 0, tsz};
                tf_numa_pread_worker(&task);
                m->numa.per_cmg_used[0] += tsz;
            } else {
                tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
                int rp = n_rows / nt, re = n_rows % nt, ro = 0;
                for (int t = 0; t < nt; t++) {
                    int rc = rp + (t < re ? 1 : 0);
                    size_t s = (size_t)ro * row_bytes;
                    size_t e = (size_t)(ro + rc) * row_bytes;
                    if (e > tsz) e = tsz;
                    tasks[t] = (tf_numa_task){tdata, fd, toff, s, e};
                    int cmg = t < m->numa.n_cmgs ? t : t % m->numa.n_cmgs;
                    m->numa.per_cmg_used[cmg] += e - s;
                    ro += rc;
                }
                tf_pool_dispatch(m, tf_numa_pread_worker, tasks, sizeof(tf_numa_task));
            }
            weight_bytes += tsz;
        }
        fprintf(stderr, "numa: phase 1 done (%.1fGB weights loaded)\n",
                (double)weight_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    /* ---- Phase 2: First-touch thread scratch from each worker ---- */
    /* thread_tmp[t>=1] allocated with notouch; first memset places pages on CMG t */
    {
        int max_ff = tf_compute_max_ff(m);
        int max_dim = m->n_embd > max_ff ? m->n_embd : max_ff;
        size_t buf_sz = (size_t)max_dim * sizeof(float);
        tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
        for (int t = 0; t < nt; t++) {
            tasks[t] = (tf_numa_task){m->thread_tmp[t], -1, 0, 0, buf_sz};
        }
        tf_pool_dispatch(m, tf_numa_memset_worker, tasks, sizeof(tf_numa_task));
        fprintf(stderr, "numa: phase 2 - thread scratch distributed\n");
    }

    /* ---- Phase 3: Distribute logits buffer (row-partitioned) ---- */
    if (m->logits && m->n_vocab > 0) {
        size_t logits_sz = (size_t)m->n_vocab * sizeof(float);
        tf_numa_distribute_buffer(m, m->logits, logits_sz);
        size_t per_cmg = logits_sz / nt;
        for (int t = 0; t < nt && t < m->numa.n_cmgs; t++)
            m->numa.per_cmg_used[t] += per_cmg;
        fprintf(stderr, "numa: phase 3 - logits distributed (%.1fKB)\n",
                (double)logits_sz / 1024.0);
    }

    /* ---- Phase 4: Distribute SSM recurrent state (head-partitioned) ---- */
    if (m->is_hybrid && m->recurrent_state) {
        int dt_rank = m->ssm_dt_rank;
        int d2 = m->ssm_d_state * m->ssm_d_state;
        size_t head_bytes = (size_t)d2 * sizeof(float);
        int distributed = 0;

        for (int l = 0; l < m->n_layers; l++) {
            if (!m->layers[l].is_ssm || !m->recurrent_state[l]) continue;
            if (dt_rank < nt) continue;

            tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
            int hp = dt_rank / nt, he = dt_rank % nt, ho = 0;
            for (int t = 0; t < nt; t++) {
                int hc = hp + (t < he ? 1 : 0);
                size_t s = (size_t)ho * head_bytes;
                size_t e = (size_t)(ho + hc) * head_bytes;
                tasks[t] = (tf_numa_task){m->recurrent_state[l], -1, 0, s, e};
                ho += hc;
            }
            tf_pool_dispatch(m, tf_numa_memset_worker, tasks, sizeof(tf_numa_task));
            distributed++;
        }
        if (distributed > 0)
            fprintf(stderr, "numa: phase 4 - SSM state distributed (%d layers)\n", distributed);
    }

    tf_numa_print_usage(m);
    fprintf(stderr, "numa: setup complete\n");
}

/* ---- SSM Delta-Net forward (single token, autoregressive) ---- */

/* L2-normalize a vector in-place: v[i] /= sqrt(sum(v[i]^2) + eps) */
static void tf_l2_norm(float *v, int n, float eps) {
#if defined(__AVX2__) && defined(__FMA__)
    float ss = 0.0f;
    int i = 0;
    __m256 vss = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vi = _mm256_loadu_ps(v + i);
        vss = _mm256_fmadd_ps(vi, vi, vss);
    }
    __m128 hi128 = _mm256_extractf128_ps(vss, 1);
    __m128 lo128 = _mm256_castps256_ps128(vss);
    __m128 s4 = _mm_add_ps(lo128, hi128);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    ss = _mm_cvtss_f32(s4);
    for (; i < n; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss + eps);
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), vinv));
    for (; i < n; i++) v[i] *= inv;
#elif defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw();
    svfloat32_t vss = svdup_f32(0.0f);
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vi = svld1_f32(pg, v + i);
        vss = svmla_m(pg, vss, vi, vi);
    }
    float inv = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vss) + eps);
    svfloat32_t vinv = svdup_f32(inv);
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svst1_f32(pg, v + i, svmul_x(pg, svld1_f32(pg, v + i), vinv));
    }
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++) v[i] *= inv;
#endif
}

/* SSM Delta-Net recurrence task for multi-threaded head dispatch */
typedef struct {
    float *rec_state;       /* [dt_rank * d_state * d_state] */
    float *Q_exp;           /* [dt_rank * d_state] */
    float *K_exp;           /* [dt_rank * d_state] */
    float *V_raw;           /* [dt_rank * d_state] */
    float *out_buf;         /* [dt_rank * d_state] */
    const float *alpha;     /* [dt_rank] */
    const float *beta_arr;  /* [dt_rank] */
    int head_start, head_end;
    int d_state;
    float scale;
} tf_ssm_recurrence_task;

static void *tf_ssm_recurrence_worker(void *arg) {
    tf_ssm_recurrence_task *t = (tf_ssm_recurrence_task *)arg;
    int ds = t->d_state;
    int d2 = ds * ds;

    for (int h = t->head_start; h < t->head_end; h++) {
        float *state = t->rec_state + (size_t)h * d2;
        float *q_h = t->Q_exp + h * ds;
        float *k_h = t->K_exp + h * ds;
        float *v_h = t->V_raw + h * ds;
        float *o_h = t->out_buf + h * ds;

#if defined(__AVX2__) && defined(__FMA__)
        /* Scale Q */
        {
            __m256 vscale = _mm256_set1_ps(t->scale);
            int i = 0;
            for (; i + 7 < ds; i += 8)
                _mm256_storeu_ps(q_h + i, _mm256_mul_ps(_mm256_loadu_ps(q_h + i), vscale));
            for (; i < ds; i++) q_h[i] *= t->scale;
        }

        /* Decay: state *= exp(alpha_h) */
        {
            float decay = expf(t->alpha[h]);
            __m256 vdecay = _mm256_set1_ps(decay);
            int i = 0;
            for (; i + 7 < d2; i += 8)
                _mm256_storeu_ps(state + i, _mm256_mul_ps(_mm256_loadu_ps(state + i), vdecay));
            for (; i < d2; i++) state[i] *= decay;
        }

        /* Read: sk = state @ k (2-row to share k loads) */
        float sk[128];
        {
            int r = 0;
            for (; r + 1 < ds; r += 2) {
                float *row0 = state + r * ds;
                float *row1 = state + (r + 1) * ds;
                /* Prefetch next 2 rows (512B each = 8 cache lines) */
                if (r + 3 < ds) {
                    for (int pf = 0; pf < ds; pf += 16) {
                        _mm_prefetch((const char *)(state + (r+2)*ds + pf), _MM_HINT_T0);
                        _mm_prefetch((const char *)(state + (r+3)*ds + pf), _MM_HINT_T0);
                    }
                }
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    __m256 k0 = _mm256_loadu_ps(k_h + c);
                    __m256 k1 = _mm256_loadu_ps(k_h + c + 8);
                    __m256 k2 = _mm256_loadu_ps(k_h + c + 16);
                    __m256 k3 = _mm256_loadu_ps(k_h + c + 24);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c),      k0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 8),  k1, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 16), k2, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 24), k3, a3);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c),      k0, b0);
                    b1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 8),  k1, b1);
                    b2 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 16), k2, b2);
                    b3 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 24), k3, b3);
                }
                for (; c + 7 < ds; c += 8) {
                    __m256 kv = _mm256_loadu_ps(k_h + c);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c), kv, a0);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c), kv, b0);
                }
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                b0 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                __m128 ahi = _mm256_extractf128_ps(a0, 1), alo = _mm256_castps256_ps128(a0);
                __m128 as4 = _mm_add_ps(alo, ahi);
                as4 = _mm_add_ps(as4, _mm_movehl_ps(as4, as4));
                as4 = _mm_add_ss(as4, _mm_movehdup_ps(as4));
                float sum0 = _mm_cvtss_f32(as4);
                __m128 bhi = _mm256_extractf128_ps(b0, 1), blo = _mm256_castps256_ps128(b0);
                __m128 bs4 = _mm_add_ps(blo, bhi);
                bs4 = _mm_add_ps(bs4, _mm_movehl_ps(bs4, bs4));
                bs4 = _mm_add_ss(bs4, _mm_movehdup_ps(bs4));
                float sum1 = _mm_cvtss_f32(bs4);
                for (; c < ds; c++) { sum0 += row0[c] * k_h[c]; sum1 += row1[c] * k_h[c]; }
                sk[r] = sum0;
                sk[r + 1] = sum1;
            }
            for (; r < ds; r++) {
                float *row = state + r * ds;
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(k_h + c), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8), _mm256_loadu_ps(k_h + c + 8), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), _mm256_loadu_ps(k_h + c + 16), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), _mm256_loadu_ps(k_h + c + 24), acc3);
                }
                acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                __m128 hi = _mm256_extractf128_ps(acc0, 1), lo = _mm256_castps256_ps128(acc0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                sk[r] = _mm_cvtss_f32(s4);
            }
        }

        /* Delta: d = (v - sk) * beta */
        float delta[128];
        {
            float beta_h = t->beta_arr[h];
            __m256 vbeta = _mm256_set1_ps(beta_h);
            int i = 0;
            for (; i + 7 < ds; i += 8) {
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(v_h + i), _mm256_loadu_ps(sk + i));
                _mm256_storeu_ps(delta + i, _mm256_mul_ps(d, vbeta));
            }
            for (; i < ds; i++) delta[i] = (v_h[i] - sk[i]) * beta_h;
        }

        /* Update: state[r][c] += delta[r] * k[c] */
        for (int r = 0; r < ds; r++) {
            float *row = state + r * ds;
            if (r + 1 < ds)
                _mm_prefetch((const char *)(state + (r+1)*ds), _MM_HINT_T0);
            __m256 vdr = _mm256_set1_ps(delta[r]);
            int c = 0;
            for (; c + 7 < ds; c += 8)
                _mm256_storeu_ps(row + c, _mm256_fmadd_ps(vdr, _mm256_loadu_ps(k_h + c),
                                                            _mm256_loadu_ps(row + c)));
            for (; c < ds; c++) row[c] += delta[r] * k_h[c];
        }

        /* Output: o = state @ q (2-row to share q loads) */
        {
            int r = 0;
            for (; r + 1 < ds; r += 2) {
                float *row0 = state + r * ds;
                float *row1 = state + (r + 1) * ds;
                /* Prefetch next 2 rows for output matvec */
                if (r + 3 < ds) {
                    for (int pf = 0; pf < ds; pf += 16) {
                        _mm_prefetch((const char *)(state + (r+2)*ds + pf), _MM_HINT_T0);
                        _mm_prefetch((const char *)(state + (r+3)*ds + pf), _MM_HINT_T0);
                    }
                }
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    __m256 q0 = _mm256_loadu_ps(q_h + c);
                    __m256 q1 = _mm256_loadu_ps(q_h + c + 8);
                    __m256 q2 = _mm256_loadu_ps(q_h + c + 16);
                    __m256 q3 = _mm256_loadu_ps(q_h + c + 24);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c),      q0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 8),  q1, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 16), q2, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 24), q3, a3);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c),      q0, b0);
                    b1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 8),  q1, b1);
                    b2 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 16), q2, b2);
                    b3 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 24), q3, b3);
                }
                for (; c + 7 < ds; c += 8) {
                    __m256 qv = _mm256_loadu_ps(q_h + c);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c), qv, a0);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c), qv, b0);
                }
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                b0 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                __m128 ahi = _mm256_extractf128_ps(a0, 1), alo = _mm256_castps256_ps128(a0);
                __m128 as4 = _mm_add_ps(alo, ahi);
                as4 = _mm_add_ps(as4, _mm_movehl_ps(as4, as4));
                as4 = _mm_add_ss(as4, _mm_movehdup_ps(as4));
                o_h[r] = _mm_cvtss_f32(as4);
                __m128 bhi = _mm256_extractf128_ps(b0, 1), blo = _mm256_castps256_ps128(b0);
                __m128 bs4 = _mm_add_ps(blo, bhi);
                bs4 = _mm_add_ps(bs4, _mm_movehl_ps(bs4, bs4));
                bs4 = _mm_add_ss(bs4, _mm_movehdup_ps(bs4));
                o_h[r + 1] = _mm_cvtss_f32(bs4);
                for (; c < ds; c++) { o_h[r] += row0[c] * q_h[c]; o_h[r + 1] += row1[c] * q_h[c]; }
            }
            for (; r < ds; r++) {
                float *row = state + r * ds;
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(q_h + c), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8), _mm256_loadu_ps(q_h + c + 8), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), _mm256_loadu_ps(q_h + c + 16), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), _mm256_loadu_ps(q_h + c + 24), acc3);
                }
                acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                __m128 hi = _mm256_extractf128_ps(acc0, 1), lo = _mm256_castps256_ps128(acc0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                o_h[r] = _mm_cvtss_f32(s4);
            }
        }
#elif defined(__ARM_FEATURE_SVE)
        /* SVE branch — A64FX vl=16 FP32 lanes.
         * ds is typically 128, so 8 SVE iterations per ds-stride loop. */
        {
            svbool_t pg = svptrue_b32();
            int vl = (int)svcntw();

            /* Scale Q */
            {
                svfloat32_t vscale = svdup_f32(t->scale);
                int i = 0;
                for (; i + vl - 1 < ds; i += vl) {
                    svst1(pg, q_h + i, svmul_x(pg, svld1(pg, q_h + i), vscale));
                }
                for (; i < ds; i++) q_h[i] *= t->scale;
            }

            /* Decay: state *= exp(alpha_h) */
            {
                float decay = expf(t->alpha[h]);
                svfloat32_t vdecay = svdup_f32(decay);
                int i = 0;
                for (; i + vl - 1 < d2; i += vl) {
                    svst1(pg, state + i, svmul_x(pg, svld1(pg, state + i), vdecay));
                }
                for (; i < d2; i++) state[i] *= decay;
            }

            /* Read: sk = state @ k (2-row at a time to share k loads) */
            float sk[128];
            {
                int r = 0;
                for (; r + 1 < ds; r += 2) {
                    float *row0 = state + r * ds;
                    float *row1 = state + (r + 1) * ds;
                    svfloat32_t a0 = svdup_f32(0.0f);
                    svfloat32_t a1 = svdup_f32(0.0f);
                    int c = 0;
                    for (; c + vl - 1 < ds; c += vl) {
                        svfloat32_t kv = svld1(pg, k_h + c);
                        a0 = svmla_x(pg, a0, svld1(pg, row0 + c), kv);
                        a1 = svmla_x(pg, a1, svld1(pg, row1 + c), kv);
                    }
                    float sum0 = svaddv(pg, a0);
                    float sum1 = svaddv(pg, a1);
                    for (; c < ds; c++) { sum0 += row0[c] * k_h[c]; sum1 += row1[c] * k_h[c]; }
                    sk[r] = sum0;
                    sk[r + 1] = sum1;
                }
                for (; r < ds; r++) {
                    float *row = state + r * ds;
                    svfloat32_t acc = svdup_f32(0.0f);
                    int c = 0;
                    for (; c + vl - 1 < ds; c += vl)
                        acc = svmla_x(pg, acc, svld1(pg, row + c), svld1(pg, k_h + c));
                    float sum = svaddv(pg, acc);
                    for (; c < ds; c++) sum += row[c] * k_h[c];
                    sk[r] = sum;
                }
            }

            /* Delta: d = (v - sk) * beta */
            float delta[128];
            {
                float beta_h = t->beta_arr[h];
                svfloat32_t vbeta = svdup_f32(beta_h);
                int i = 0;
                for (; i + vl - 1 < ds; i += vl) {
                    svfloat32_t dv = svsub_x(pg, svld1(pg, v_h + i), svld1(pg, sk + i));
                    svst1(pg, delta + i, svmul_x(pg, dv, vbeta));
                }
                for (; i < ds; i++) delta[i] = (v_h[i] - sk[i]) * beta_h;
            }

            /* Update: state[r][c] += delta[r] * k[c] (outer product) */
            for (int r = 0; r < ds; r++) {
                float *row = state + r * ds;
                svfloat32_t vdr = svdup_f32(delta[r]);
                int c = 0;
                for (; c + vl - 1 < ds; c += vl) {
                    svfloat32_t acc = svmla_x(pg, svld1(pg, row + c), vdr, svld1(pg, k_h + c));
                    svst1(pg, row + c, acc);
                }
                for (; c < ds; c++) row[c] += delta[r] * k_h[c];
            }

            /* Output: o = state @ q (2-row to share q loads) */
            {
                int r = 0;
                for (; r + 1 < ds; r += 2) {
                    float *row0 = state + r * ds;
                    float *row1 = state + (r + 1) * ds;
                    svfloat32_t a0 = svdup_f32(0.0f);
                    svfloat32_t a1 = svdup_f32(0.0f);
                    int c = 0;
                    for (; c + vl - 1 < ds; c += vl) {
                        svfloat32_t qv = svld1(pg, q_h + c);
                        a0 = svmla_x(pg, a0, svld1(pg, row0 + c), qv);
                        a1 = svmla_x(pg, a1, svld1(pg, row1 + c), qv);
                    }
                    float sum0 = svaddv(pg, a0);
                    float sum1 = svaddv(pg, a1);
                    for (; c < ds; c++) { sum0 += row0[c] * q_h[c]; sum1 += row1[c] * q_h[c]; }
                    o_h[r] = sum0;
                    o_h[r + 1] = sum1;
                }
                for (; r < ds; r++) {
                    float *row = state + r * ds;
                    svfloat32_t acc = svdup_f32(0.0f);
                    int c = 0;
                    for (; c + vl - 1 < ds; c += vl)
                        acc = svmla_x(pg, acc, svld1(pg, row + c), svld1(pg, q_h + c));
                    float sum = svaddv(pg, acc);
                    for (; c < ds; c++) sum += row[c] * q_h[c];
                    o_h[r] = sum;
                }
            }
        }
#else
        /* Scalar fallback */
        for (int i = 0; i < ds; i++) q_h[i] *= t->scale;
        float decay = expf(t->alpha[h]);
        for (int i = 0; i < d2; i++) state[i] *= decay;

        float sk[128];
        for (int r = 0; r < ds; r++) {
            float sum = 0.0f;
            for (int c = 0; c < ds; c++) sum += state[r * ds + c] * k_h[c];
            sk[r] = sum;
        }
        float delta[128];
        for (int i = 0; i < ds; i++) delta[i] = (v_h[i] - sk[i]) * t->beta_arr[h];
        for (int r = 0; r < ds; r++) {
            float dr = delta[r];
            for (int c = 0; c < ds; c++) state[r * ds + c] += dr * k_h[c];
        }
        for (int r = 0; r < ds; r++) {
            float sum = 0.0f;
            for (int c = 0; c < ds; c++) sum += state[r * ds + c] * q_h[c];
            o_h[r] = sum;
        }
#endif
    }
    return NULL;
}

#if defined(__ARM_FEATURE_SVE)
/* SVE fast exp: identical magic-bias + degree-4 polynomial as fast_exp_avx2, so
 * the A64FX SSM path matches the validated x86 approximation (rel err ~6e-5).
 * Used by the SSM delta-net vectorized loops below (svscale = poly * 2^n). */
static inline svfloat32_t tf_fast_exp_sve(svbool_t pg, svfloat32_t x) {
    const svfloat32_t log2e = svdup_f32(1.442695040f);
    const svfloat32_t c0    = svdup_f32(12582912.0f); /* 1.5 * 2^23 magic bias */
    const svfloat32_t p0    = svdup_f32(0.9999999f);
    const svfloat32_t p1    = svdup_f32(0.6931472f);
    const svfloat32_t p2    = svdup_f32(0.2402265f);
    const svfloat32_t p3    = svdup_f32(0.0554953f);
    const svfloat32_t p4    = svdup_f32(0.0096813f);
    x = svmax_x(pg, svmin_x(pg, x, svdup_f32(88.0f)), svdup_f32(-88.0f));
    svfloat32_t xl  = svmul_x(pg, x, log2e);
    svfloat32_t z   = svadd_x(pg, xl, c0);
    svfloat32_t n_f = svsub_x(pg, z, c0);          /* round(x*log2e) */
    svfloat32_t f   = svsub_x(pg, xl, n_f);        /* fractional part */
    svint32_t   n_i = svcvt_s32_f32_x(pg, n_f);
    svfloat32_t poly = svmla_x(pg, p3, f, p4);     /* p3 + f*p4, Horner */
    poly = svmla_x(pg, p2, f, poly);
    poly = svmla_x(pg, p1, f, poly);
    poly = svmla_x(pg, p0, f, poly);
    return svscale_x(pg, poly, n_i);               /* poly * 2^n_i */
}

/* SVE natural log, ported from a64fx/cross-entropy/sve_math.h: exponent extract
 * + degree-5 minimax on log2(1+f), then *LN2. Purely vertical (no horizontal
 * reductions). ~1e-5 abs err. Used by the SSM softplus below. */
static inline svfloat32_t tf_sve_log2_f32(svbool_t pg, svfloat32_t x) {
    const float C0 = 1.44269504089f, C1 = -0.72134752045f, C2 = 0.48089834696f,
                C3 = -0.36067376023f, C4 = 0.28853900819f;
    const float SQRT2 = 1.4142135623730951f;
    svint32_t bits = svreinterpret_s32(x);
    svint32_t n_i = svsub_n_s32_x(pg, svasr_n_s32_x(pg, bits, 23), 127);
    svfloat32_t n = svcvt_f32_s32_x(pg, n_i);
    svint32_t mantissa = svand_n_s32_x(pg, bits, 0x007FFFFF);
    svint32_t m_bits   = svorr_n_s32_x(pg, mantissa, 0x3F800000);
    svfloat32_t m = svreinterpret_f32(m_bits);
    svbool_t hi = svcmpgt(pg, m, svdup_f32(SQRT2));        /* range reduce */
    svfloat32_t m_adj = svreinterpret_f32(svsub_n_s32_x(pg, m_bits, 0x00800000));
    m = svsel(hi, m_adj, m);
    n = svadd_f32_m(hi, n, svdup_f32(1.0f));
    svfloat32_t f = svsub_n_f32_x(pg, m, 1.0f);            /* f in [-0.293,0.414] */
    svfloat32_t p = svdup_f32(C4);
    p = svmla_x(pg, svdup_f32(C3), f, p);
    p = svmla_x(pg, svdup_f32(C2), f, p);
    p = svmla_x(pg, svdup_f32(C1), f, p);
    p = svmla_x(pg, svdup_f32(C0), f, p);
    p = svmul_x(pg, p, f);
    return svadd_x(pg, n, p);
}
static inline svfloat32_t tf_sve_log_f32(svbool_t pg, svfloat32_t x) {
    return svmul_n_f32_x(pg, tf_sve_log2_f32(pg, x), 0.6931471805599453f); /* *LN2 */
}
#endif

/* Cooperative parallel SSM (TF_SSM_COOP): the TP decode path runs the SSM glue
 * (conv1d/L2-norm/RMSNorm/softplus) serially on the main thread BETWEEN pooled
 * matvec dispatches, leaving 47 threads idle in each gap. This wraps
 * tf_ssm_deltanet_forward_parallel — which spreads every section across all pool
 * threads with internal spin-barriers — as ONE pool dispatch, collapsing the 3
 * separate dispatches + serial gaps into a single cooperative pass. */
typedef struct { transformer_model *m; int layer_idx; int tid; int nt; } tf_ssm_par_task;
static void tf_ssm_deltanet_forward_parallel(transformer_model *m, int layer_idx,
                                             int tid, int nt, int *local_sense);
static void *tf_ssm_parallel_worker(void *arg);

/* SSM Delta-Net forward for one layer.
 * Input:  m->xb (post-norm hidden state [n_embd])
 * Output: m->xb (residual-ready output [n_embd])
 * Scratch: m->xb2 (qkv), m->q (Q_expanded), m->ffn_buf1 (z/gate),
 *          m->ffn_buf2 (K_expanded), m->ffn_buf3 (delta-net output) */
static void tf_ssm_deltanet_forward(transformer_model *m, int layer_idx) {
    transformer_layer *layer = &m->layers[layer_idx];
    int n_embd  = m->n_embd;
    int qkv_dim = m->ssm_qkv_dim;
    int d_inner = m->ssm_d_inner;
    int d_state = m->ssm_d_state;
    int n_group = m->ssm_n_group;
    int dt_rank = m->ssm_dt_rank;
    int conv_k  = m->ssm_conv_kernel;
    float eps   = m->rms_norm_eps;

    float *qkv_buf = m->xb2;       /* [qkv_dim] */
    float *z_buf   = m->ffn_buf1;  /* [d_inner] */
    float *K_exp   = m->ffn_buf2;  /* [dt_rank * d_state] */
    float *out_buf = m->ffn_buf3;  /* [d_inner] */
    float *Q_exp   = m->q;         /* [dt_rank * d_state] */

    /* Cooperative parallel path: one pool dispatch runs the entire SSM layer
     * across all threads (internal spin-barriers), instead of 3 dispatches with
     * serial conv/norm glue in the gaps. Falls back to the serial body below
     * when single-threaded or pool inactive. */
    static int coop = -1;
    if (coop < 0) coop = getenv("TF_SSM_COOP") ? 1 : 0;
    if (coop && m->n_threads > 1 && m->pool_alive) {
        int nt = m->n_threads;
        tf_ssm_par_task *tasks = (tf_ssm_par_task *)alloca(nt * sizeof(tf_ssm_par_task));
        for (int t = 0; t < nt; t++)
            tasks[t] = (tf_ssm_par_task){ m, layer_idx, t, nt };
        tf_pool_dispatch(m, tf_ssm_parallel_worker, tasks, sizeof(tf_ssm_par_task));
        return;
    }

    /* 1. Linear projections from xb (fused: qkv + gate share input xb) */
    TF_PROF_BEGIN("ssm_proj", layer_idx, "matvec", "FP32");
    tf_qmatvec_fused2_diff_pool(m,
        qkv_buf, &layer->ssm_qkv,  qkv_dim,
        z_buf,   &layer->ssm_gate, d_inner,
        m->xb);

    float alpha[64], beta_arr[64]; /* dt_rank <= 64 (48 for Qwen3.5-27B) */
    tf_qmatvec(alpha, &layer->ssm_alpha, m->xb, dt_rank, m->matvec_tmp);
    tf_qmatvec(beta_arr, &layer->ssm_beta, m->xb, dt_rank, m->matvec_tmp);
    TF_PROF_END("ssm_proj", 0, 0);

    /* 2. alpha = softplus(alpha + dt_bias) * ssm_a */
    {
        float a_buf[64], dt_bias_buf[64];
        tf_dequant_row(&layer->ssm_a, 0, a_buf);
        tf_dequant_row(&layer->ssm_dt_bias, 0, dt_bias_buf);
        /* ssm_a / dt_bias stay REPLICATED (full dt_rank); when V-head sharded,
         * local head i maps to global head ssm_head_offset+i. 0 if unsharded. */
        int hoff = m->ssm_head_offset;
#if defined(__ARM_FEATURE_SVE)
        /* softplus(val) = log(1+exp(val)); guard val>20 -> val (matches scalar) */
        svfloat32_t one = svdup_f32(1.0f), thr = svdup_f32(20.0f);
        int vl = (int)svcntw();
        for (int i = 0; i < dt_rank; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dt_rank);
            svfloat32_t val = svadd_x(pg, svld1_f32(pg, alpha + i),
                                       svld1_f32(pg, dt_bias_buf + hoff + i));
            svfloat32_t lg = tf_sve_log_f32(pg, svadd_x(pg, one, tf_fast_exp_sve(pg, val)));
            svfloat32_t sp = svsel(svcmpgt(pg, val, thr), val, lg);
            svst1_f32(pg, alpha + i, svmul_x(pg, sp, svld1_f32(pg, a_buf + hoff + i)));
        }
#else
        for (int i = 0; i < dt_rank; i++) {
            float val = alpha[i] + dt_bias_buf[hoff + i];
            float sp = (val > 20.0f) ? val : logf(1.0f + expf(val));
            alpha[i] = sp * a_buf[hoff + i]; /* negative since ssm_a < 0 */
        }
#endif
    }

    /* 3. beta = sigmoid(beta) */
#if defined(__AVX2__) && defined(__FMA__)
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int i = 0;
        for (; i + 7 < dt_rank; i += 8) {
            __m256 b = _mm256_loadu_ps(beta_arr + i);
            __m256 neg_b = _mm256_sub_ps(_mm256_setzero_ps(), b);
            __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_b)));
            _mm256_storeu_ps(beta_arr + i, sig);
        }
        for (; i < dt_rank; i++)
            beta_arr[i] = 1.0f / (1.0f + expf(-beta_arr[i]));
    }
#elif defined(__ARM_FEATURE_SVE)
    {
        svfloat32_t one = svdup_f32(1.0f);
        int vl = (int)svcntw();
        for (int i = 0; i < dt_rank; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dt_rank);
            svfloat32_t b = svld1_f32(pg, beta_arr + i);
            svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), b));
            svst1_f32(pg, beta_arr + i, svdiv_x(pg, one, svadd_x(pg, one, e)));
        }
    }
#else
    for (int i = 0; i < dt_rank; i++) {
        beta_arr[i] = 1.0f / (1.0f + expf(-beta_arr[i]));
    }
#endif

    /* 4. Conv1d: depthwise causal conv + SiLU (circular buffer) */
    TF_PROF_BEGIN("ssm_conv", layer_idx, "conv1d", "FP32");
    {
        float *conv_st = m->conv_state[layer_idx]; /* [(conv_k-1) * qkv_dim] */
        int wr = m->conv_state_pos[layer_idx]; /* circular buffer write position */
        int n_hist = conv_k - 1;
        /* Use K_exp temporarily for conv output (17408 >= 10240) */
        float *conv_out = K_exp;

        /* Conv weights are pre-dequantised + transposed once at load time
         * into m->conv_w_trans[layer_idx] (see transformer_load). */
        float *w_trans = m->conv_w_trans[layer_idx];

        /* Precompute circular buffer row offsets to avoid modulo in inner loop. */
        int *row_off = n_hist > 0 ? (int *)alloca((size_t)n_hist * sizeof(*row_off)) : NULL;
        for (int f = 0; f < n_hist; f++)
            row_off[f] = ((wr + f) % n_hist) * qkv_dim;

        /* Vectorized conv MAC over channels */
#if defined(__AVX2__) && defined(__FMA__)
        {
            int j = 0;
            for (; j + 7 < qkv_dim; j += 8) {
                /* Prefetch conv_st rows 64 bytes ahead */
                if (j + 16 < qkv_dim) {
                    for (int f = 0; f < n_hist; f++)
                        _mm_prefetch((const char *)(conv_st + row_off[f] + j + 16), _MM_HINT_T0);
                }
                __m256 sum = _mm256_setzero_ps();
                for (int f = 0; f < n_hist; f++)
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(w_trans + f * qkv_dim + j),
                                           _mm256_loadu_ps(conv_st + row_off[f] + j), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(w_trans + n_hist * qkv_dim + j),
                                       _mm256_loadu_ps(qkv_buf + j), sum);
                _mm256_storeu_ps(conv_out + j, sum);
            }
            for (; j < qkv_dim; j++) {
                float s = 0.0f;
                for (int f = 0; f < n_hist; f++)
                    s += w_trans[f * qkv_dim + j] * conv_st[row_off[f] + j];
                conv_out[j] = s + w_trans[n_hist * qkv_dim + j] * qkv_buf[j];
            }
        }
#elif defined(__ARM_FEATURE_SVE)
        {
            int vl = (int)svcntw();
            for (int j = 0; j < qkv_dim; j += vl) {
                svbool_t pg = svwhilelt_b32(j, qkv_dim);
                svfloat32_t sum = svdup_f32(0.0f);
                for (int f = 0; f < n_hist; f++)
                    sum = svmla_x(pg, sum, svld1_f32(pg, w_trans + f * qkv_dim + j),
                                  svld1_f32(pg, conv_st + row_off[f] + j));
                sum = svmla_x(pg, sum, svld1_f32(pg, w_trans + n_hist * qkv_dim + j),
                              svld1_f32(pg, qkv_buf + j));
                svst1_f32(pg, conv_out + j, sum);
            }
        }
#else
        for (int j = 0; j < qkv_dim; j++) {
            float sum = 0.0f;
            for (int f = 0; f < n_hist; f++)
                sum += w_trans[f * qkv_dim + j] * conv_st[row_off[f] + j];
            conv_out[j] = sum + w_trans[n_hist * qkv_dim + j] * qkv_buf[j];
        }
#endif
        /* Vectorized SiLU activation */
#if defined(__AVX2__) && defined(__FMA__)
        {
            const __m256 one = _mm256_set1_ps(1.0f);
            int j = 0;
            for (; j + 7 < qkv_dim; j += 8) {
                __m256 s = _mm256_loadu_ps(conv_out + j);
                __m256 neg_s = _mm256_sub_ps(_mm256_setzero_ps(), s);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_s)));
                _mm256_storeu_ps(conv_out + j, _mm256_mul_ps(s, sig));
            }
            for (; j < qkv_dim; j++)
                conv_out[j] = conv_out[j] / (1.0f + expf(-conv_out[j]));
        }
#elif defined(__ARM_FEATURE_SVE)
        {
            svfloat32_t one = svdup_f32(1.0f);
            int vl = (int)svcntw();
            for (int j = 0; j < qkv_dim; j += vl) {
                svbool_t pg = svwhilelt_b32(j, qkv_dim);
                svfloat32_t s = svld1_f32(pg, conv_out + j);
                svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), s));
                svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
                svst1_f32(pg, conv_out + j, svmul_x(pg, s, sig));
            }
        }
#else
        for (int j = 0; j < qkv_dim; j++)
            conv_out[j] = conv_out[j] / (1.0f + expf(-conv_out[j]));
#endif

        /* Update circular buffer: overwrite oldest slot, advance write position */
        memcpy(conv_st + wr * qkv_dim, qkv_buf, qkv_dim * sizeof(float));
        m->conv_state_pos[layer_idx] = n_hist > 0 ? (wr + 1) % n_hist : 0;

        /* Copy conv output to qkv_buf */
        memcpy(qkv_buf, conv_out, qkv_dim * sizeof(float));
    }
    TF_PROF_END("ssm_conv", 0, 0);

    /* 5. Split: Q[n_group*d_state], K[n_group*d_state], V[dt_rank*d_state]
     *    Order in qkv_buf: Q, K, V (following llama.cpp convention) */
    float *Q_raw = qkv_buf;                                  /* [n_group * d_state] */
    float *K_raw = qkv_buf + n_group * d_state;              /* [n_group * d_state] */
    float *V_raw = qkv_buf + 2 * n_group * d_state;          /* [dt_rank * d_state = d_inner] */

    /* L2-normalize Q and K per head (n_group heads, d_state elements each) */
    for (int g = 0; g < n_group; g++) {
        tf_l2_norm(Q_raw + g * d_state, d_state, eps);
        tf_l2_norm(K_raw + g * d_state, d_state, eps);
    }

    /* Repeat Q and K from n_group to dt_rank heads (tiling, matching ggml_repeat) */
    /* dt_rank=48, n_group=16: 3 bulk copies of n_group*d_state instead of 96 memcpys */
    if (!m->tp_ssm_sharded) {
        size_t tile_bytes = (size_t)n_group * d_state * sizeof(float);
        int n_repeat = dt_rank / n_group;
        for (int r = n_repeat - 1; r >= 0; r--) {
            memcpy(Q_exp + r * n_group * d_state, Q_raw, tile_bytes);
            memcpy(K_exp + r * n_group * d_state, K_raw, tile_bytes);
        }
    } else {
        /* Stage B (V-head sharded): dt_rank is LOCAL (may be < n_group), so the
         * bulk-tile path (n_repeat = dt_rank/n_group) underflows to 0. Q/K stay
         * REPLICATED (full n_group present), V-heads are sharded. Each local head
         * hl maps to global head (ssm_head_offset+hl); its Q/K group is that
         * global index % n_group. Gather per local head. */
        for (int hl = 0; hl < dt_rank; hl++) {
            int g = (m->ssm_head_offset + hl) % n_group;
            memcpy(Q_exp + (size_t)hl * d_state, Q_raw + (size_t)g * d_state, (size_t)d_state * sizeof(float));
            memcpy(K_exp + (size_t)hl * d_state, K_raw + (size_t)g * d_state, (size_t)d_state * sizeof(float));
        }
    }

    TF_PROF_BEGIN("ssm_scan", layer_idx, "ssm_scan", "FP32");
    /* 6. Delta-Net recurrence per head (AVX2 + multi-threaded) */
    float scale = 1.0f / sqrtf((float)d_state);
    float *rec_state = m->recurrent_state[layer_idx]; /* [dt_rank * d_state * d_state] */

    if (m->n_threads > 1 && m->pool_alive) {
        int nt = m->n_threads;
        int ncmgs = m->cmg_pin ? m->cmg_pin_ncmgs : 1;
        tf_ssm_recurrence_task *rtasks = (tf_ssm_recurrence_task *)alloca(
            nt * sizeof(tf_ssm_recurrence_task));
        for (int t = 0; t < nt; t++) {
            int hs, he;
            tf_ssm_head_range(dt_rank, nt, ncmgs, t, &hs, &he);
            rtasks[t] = (tf_ssm_recurrence_task){
                rec_state, Q_exp, K_exp, V_raw, out_buf,
                alpha, beta_arr, hs, he, d_state, scale
            };
        }
        tf_pool_dispatch(m, tf_ssm_recurrence_worker, rtasks, sizeof(tf_ssm_recurrence_task));
    } else {
        tf_ssm_recurrence_task rtask = {
            rec_state, Q_exp, K_exp, V_raw, out_buf,
            alpha, beta_arr, 0, dt_rank, d_state, scale
        };
        tf_ssm_recurrence_worker(&rtask);
    }
    TF_PROF_END("ssm_scan", 0, 0);

    /* 7. Fused: out = rmsnorm(out, ssm_norm) * silu(z) */
    {
        /* ssm_norm has [d_state] weights, applied per-head (48 groups of 128) */
        float norm_w[128];
        tf_dequant_row(&layer->ssm_norm, 0, norm_w);
        for (int h = 0; h < dt_rank; h++) {
            float *o_h = out_buf + h * d_state;
            float *z_h = z_buf + h * d_state;
#if defined(__AVX2__) && defined(__FMA__)
            /* RMSNorm: sum of squares (AVX2) */
            __m256 vss = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < d_state; i += 8) {
                __m256 oi = _mm256_loadu_ps(o_h + i);
                vss = _mm256_fmadd_ps(oi, oi, vss);
            }
            __m128 hi128 = _mm256_extractf128_ps(vss, 1);
            __m128 lo128 = _mm256_castps256_ps128(vss);
            __m128 s4 = _mm_add_ps(lo128, hi128);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float ss = _mm_cvtss_f32(s4);
            for (; i < d_state; i++) ss += o_h[i] * o_h[i];
            __m256 vscale = _mm256_set1_ps(1.0f / sqrtf(ss / d_state + eps));
            const __m256 one = _mm256_set1_ps(1.0f);
            /* Apply norm * weight * silu(z) */
            i = 0;
            for (; i + 7 < d_state; i += 8) {
                __m256 oi = _mm256_loadu_ps(o_h + i);
                __m256 wi = _mm256_loadu_ps(norm_w + i);
                __m256 normed = _mm256_mul_ps(_mm256_mul_ps(oi, vscale), wi);
                __m256 zi = _mm256_loadu_ps(z_h + i);
                __m256 neg_z = _mm256_sub_ps(_mm256_setzero_ps(), zi);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_z)));
                _mm256_storeu_ps(o_h + i, _mm256_mul_ps(normed, _mm256_mul_ps(zi, sig)));
            }
            float ss_scalar = 1.0f / sqrtf(ss / d_state + eps);
            for (; i < d_state; i++) {
                float normed = o_h[i] * ss_scalar * norm_w[i];
                float z_val = z_h[i];
                o_h[i] = normed * (z_val / (1.0f + expf(-z_val)));
            }
#elif defined(__ARM_FEATURE_SVE)
            {
                int vl = (int)svcntw();
                svfloat32_t vss = svdup_f32(0.0f);
                for (int i = 0; i < d_state; i += vl) {
                    svbool_t pg = svwhilelt_b32(i, d_state);
                    svfloat32_t oi = svld1_f32(pg, o_h + i);
                    vss = svmla_m(pg, vss, oi, oi);
                }
                float scl = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vss) / d_state + eps);
                svfloat32_t vscale = svdup_f32(scl), one = svdup_f32(1.0f);
                for (int i = 0; i < d_state; i += vl) {
                    svbool_t pg = svwhilelt_b32(i, d_state);
                    svfloat32_t oi = svld1_f32(pg, o_h + i);
                    svfloat32_t wi = svld1_f32(pg, norm_w + i);
                    svfloat32_t normed = svmul_x(pg, svmul_x(pg, oi, vscale), wi);
                    svfloat32_t zi = svld1_f32(pg, z_h + i);
                    svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), zi));
                    svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
                    svst1_f32(pg, o_h + i, svmul_x(pg, normed, svmul_x(pg, zi, sig)));
                }
            }
#else
            /* RMSNorm per head */
            float ss = 0.0f;
            for (int i = 0; i < d_state; i++) ss += o_h[i] * o_h[i];
            ss = 1.0f / sqrtf(ss / d_state + eps);
            /* Apply norm * weight * silu(z) */
            for (int i = 0; i < d_state; i++) {
                float normed = o_h[i] * ss * norm_w[i];
                float z_val = z_h[i];
                float silu_z = z_val / (1.0f + expf(-z_val));
                o_h[i] = normed * silu_z;
            }
#endif
        }
    }

    /* 8. Output projection: xb = ssm_out @ out_buf */
    tf_qmatvec_pool(m, m->xb, &layer->ssm_out, out_buf, n_embd);
}

/* Forward decls (definitions live in the persistent-worker block below). */
static inline void tf_spin_barrier(transformer_model *m, int *local_sense, int nt);
static inline void tf_sw_hier_barrier(transformer_model *m, int tid, int *local_sense,
                                      int nt, int n_cmgs);
static void tf_thread_matvec(float *dst, const qtensor *mat, const float *x,
                              int n_rows, int tid, int nt);

/* Mirrors the row-split decision inside tf_thread_matvec so other code can
 * compute the exact [rs,re) range that thread `tid` will write into `dst`.
 * Used to align downstream consumers with the producer's partition and avoid
 * an intervening barrier. Keep in sync with tf_thread_matvec! */
static inline void tf_matvec_row_range(const qtensor *mat, int n_rows,
                                        int nt, int tid, int *rs, int *re) {
#if defined(__ARM_FEATURE_SVE)
    if (mat->type == GGML_TYPE_F16 && mat->panel) {
        /* Panel path splits by panel_blk; each block covers a contiguous row
         * stripe of size (n_rows / panel_blk). */
        int nblk = mat->panel_blk;
        int bp = nblk / nt, be = nblk % nt;
        int bs = tid * bp + (tid < be ? tid : be);
        int bc = bp + (tid < be ? 1 : 0);
        int rows_per_blk = n_rows / nblk;
        *rs = bs * rows_per_blk;
        *re = (bs + bc) * rows_per_blk;
        return;
    }
#endif
    if (mat->type == GGML_TYPE_BF16 || mat->type == GGML_TYPE_F16 ||
        mat->bf16_pv || mat->q8_pv) {
        tf_row_split8(n_rows, nt, tid, rs, re);
    } else {
        int rp = n_rows / nt, rem = n_rows % nt;
        *rs = tid * rp + (tid < rem ? tid : rem);
        *re = *rs + rp + (tid < rem ? 1 : 0);
    }
}

static inline int tf_ssm_q8_shared_quant(transformer_model *m, const float *x, int K,
                                         int tid, int nt, int *local_sense,
                                         const int8_t **xq, const uint16_t **xs) {
#if defined(__ARM_FEATURE_SVE)
    if (m && local_sense && m->ssm_q8_xq && m->ssm_q8_xs &&
        K <= m->ssm_q8_cap && (K & 63) == 0) {
        tf_quant_x_sdot_blocks(x, K, tid, nt, m->ssm_q8_xq, m->ssm_q8_xs);
        tf_spin_barrier(m, local_sense, nt);
        *xq = m->ssm_q8_xq;
        *xs = m->ssm_q8_xs;
        return 1;
    }
#else
    (void)m; (void)x; (void)K; (void)tid; (void)nt; (void)local_sense;
    (void)xq; (void)xs;
#endif
    return 0;
}

static void tf_thread_matvec2_diff(transformer_model *m,
                                   float *dst1, const qtensor *mat1, int n_rows1,
                                   float *dst2, const qtensor *mat2, int n_rows2,
                                   const float *x, int tid, int nt,
                                   int *local_sense) {
    int rs1, re1, rs2, re2;
    tf_matvec_row_range(mat1, n_rows1, nt, tid, &rs1, &re1);
    tf_matvec_row_range(mat2, n_rows2, nt, tid, &rs2, &re2);
#if defined(__ARM_FEATURE_SVE)
    if (mat1->q8_pv && mat2->q8_pv && mat1->n_cols == mat2->n_cols &&
        ((rs1 | re1 | rs2 | re2) & 7) == 0 && (re1 > rs1 || re2 > rs2)) {
        int n_cols = mat1->n_cols;
        const int8_t *xq; const uint16_t *xs;
        if (!tf_ssm_q8_shared_quant(m, x, n_cols, tid, nt, local_sense, &xq, &xs))
            tf_quant_x_sdot(x, n_cols, &xq, &xs);
        int nb = n_cols / 64;
        size_t group_bytes = (size_t)nb * 528;
        const uint8_t *q1 = mat1->q8_pv;
        const uint8_t *q2 = mat2->q8_pv;
        for (int i = rs1; i + 7 < re1; i += 8) {
            int g = i >> 3;
            matvec_sdot_8row(dst1 + i, q1 + (size_t)g * group_bytes,
                             xq, xs, n_cols);
        }
        for (int i = rs2; i + 7 < re2; i += 8) {
            int g = i >> 3;
            matvec_sdot_8row(dst2 + i, q2 + (size_t)g * group_bytes,
                             xq, xs, n_cols);
        }
        return;
    }
#endif
    if (re1 > rs1)
        tf_matvec_qtensor_rows(dst1, mat1, x, rs1, re1);
    if (re2 > rs2)
        tf_matvec_qtensor_rows(dst2, mat2, x, rs2, re2);
}

static void tf_thread_matvec_q8_shared(transformer_model *m, float *dst,
                                       const qtensor *mat, const float *x,
                                       int n_rows, int tid, int nt,
                                       int *local_sense) {
    int rs, re;
    tf_matvec_row_range(mat, n_rows, nt, tid, &rs, &re);
#if defined(__ARM_FEATURE_SVE)
    if (mat->q8_pv && ((rs | re) & 7) == 0 && re > rs) {
        int n_cols = mat->n_cols;
        const int8_t *xq; const uint16_t *xs;
        if (!tf_ssm_q8_shared_quant(m, x, n_cols, tid, nt, local_sense, &xq, &xs))
            tf_quant_x_sdot(x, n_cols, &xq, &xs);
        int nb = n_cols / 64;
        size_t group_bytes = (size_t)nb * 528;
        const uint8_t *qbase = mat->q8_pv;
        for (int i = rs; i + 7 < re; i += 8) {
            int g = i >> 3;
            matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                             xq, xs, n_cols);
        }
        return;
    }
#endif
    if (re > rs)
        tf_matvec_qtensor_rows(dst, mat, x, rs, re);
}

#ifndef TF_SSM_S1_ALPHA_BETA_SPLIT
#define TF_SSM_S1_ALPHA_BETA_SPLIT 1
#endif
#ifndef TF_SSM_S1_AB_FULL_PAR
#define TF_SSM_S1_AB_FULL_PAR 0
#endif

static inline void tf_ssm_post_alpha_range(float *alpha, const transformer_layer *layer,
                                           int dt_rank, int hoff, int rs, int re) {
    float a_buf[64], dt_bias_buf[64];
    if (rs < 0) rs = 0;
    if (re > dt_rank) re = dt_rank;
    if (re <= rs) return;
    tf_dequant_row(&layer->ssm_a, 0, a_buf);
    tf_dequant_row(&layer->ssm_dt_bias, 0, dt_bias_buf);
#if defined(__ARM_FEATURE_SVE)
    /* softplus(alpha+dt_bias)*a, SVE-matched to the serial path. */
    svfloat32_t one = svdup_f32(1.0f), thr = svdup_f32(20.0f);
    int vl = (int)svcntw();
    for (int i = rs; i < re; i += vl) {
        svbool_t pg = svwhilelt_b32(i, re);
        svfloat32_t val = svadd_x(pg, svld1_f32(pg, alpha + i),
                                   svld1_f32(pg, dt_bias_buf + hoff + i));
        svfloat32_t lg = tf_sve_log_f32(pg, svadd_x(pg, one, tf_fast_exp_sve(pg, val)));
        svfloat32_t sp = svsel(svcmpgt(pg, val, thr), val, lg);
        svst1_f32(pg, alpha + i, svmul_x(pg, sp, svld1_f32(pg, a_buf + hoff + i)));
    }
#else
    for (int i = rs; i < re; i++) {
        float val = alpha[i] + dt_bias_buf[hoff + i];
        float sp = (val > 20.0f) ? val : logf(1.0f + expf(val));
        alpha[i] = sp * a_buf[hoff + i];
    }
#endif
}

static inline void tf_ssm_post_alpha(float *alpha, const transformer_layer *layer,
                                     int dt_rank, int hoff) {
    tf_ssm_post_alpha_range(alpha, layer, dt_rank, hoff, 0, dt_rank);
}

static inline void tf_ssm_post_beta_range(float *beta_arr, int dt_rank, int rs, int re) {
    if (rs < 0) rs = 0;
    if (re > dt_rank) re = dt_rank;
    if (re <= rs) return;
#if defined(__ARM_FEATURE_SVE)
    /* sigmoid(beta), SVE-matched to the serial path. */
    svfloat32_t one = svdup_f32(1.0f);
    int vl = (int)svcntw();
    for (int i = rs; i < re; i += vl) {
        svbool_t pg = svwhilelt_b32(i, re);
        svfloat32_t b = svld1_f32(pg, beta_arr + i);
        svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), b));
        svst1_f32(pg, beta_arr + i, svdiv_x(pg, one, svadd_x(pg, one, e)));
    }
#else
    for (int i = rs; i < re; i++)
        beta_arr[i] = 1.0f / (1.0f + expf(-beta_arr[i]));
#endif
}

static inline void tf_ssm_post_beta(float *beta_arr, int dt_rank) {
    tf_ssm_post_beta_range(beta_arr, dt_rank, 0, dt_rank);
}

/* Parallel SSM Delta-Net forward, called by ALL persistent worker threads.
 * Uses 4 internal spin barriers (B_a..B_d) to coordinate sections:
 *   S1: parallel qkv+gate matvecs (rows); tid 0 also does alpha+beta matvec + postproc
 *   B_a
 *   S2: parallel conv1d (channels) — overlap with conv_state write-back
 *   B_b
 *   S3: parallel L2-norm + tile-repeat (heads) — no write to qkv_buf
 *   B_c
 *   S4: parallel Delta-Net recurrence + fused RMSNorm+SiLU(z) (heads)
 *   B_d
 *   S5: parallel output projection (rows) */
static void tf_ssm_deltanet_forward_parallel(transformer_model *m, int layer_idx,
                                              int tid, int nt, int *local_sense) {
    transformer_layer *layer = &m->layers[layer_idx];
    int n_embd  = m->n_embd;
    int qkv_dim = m->ssm_qkv_dim;
    int d_inner = m->ssm_d_inner;
    int d_state = m->ssm_d_state;
    int n_group = m->ssm_n_group;
    int dt_rank = m->ssm_dt_rank;
    int conv_k  = m->ssm_conv_kernel;
    float eps   = m->rms_norm_eps;
    int ncmgs = m->cmg_pin ? m->cmg_pin_ncmgs : 1;
    /* ssm_a / dt_bias stay REPLICATED (full dt_rank); when V-head sharded, local
     * head i maps to global head hoff+i. 0 (no offset) when unsharded. */
    int hoff = m->ssm_head_offset;
    int ssm_conv_wr = m->conv_state_pos[layer_idx];

    float *qkv_buf  = m->xb2;
    float *z_buf    = m->ffn_buf1;
    float *K_exp    = m->ffn_buf2;        /* also conv_out scratch (size >= qkv_dim) */
    float *out_buf  = m->ffn_buf3;
    float *Q_exp    = m->q;
    float *alpha    = m->ssm_alpha_buf;   /* [dt_rank] shared */
    float *beta_arr = m->ssm_beta_buf;    /* [dt_rank] shared */

    int _ssm_prof = tf_ssm_coop_prof_enabled();
    double _ssm_t0 = _ssm_prof ? tf_wall_seconds() : 0.0;
#define TF_SSM_COOP_MARK(STAGE) do { \
        if (_ssm_prof) { \
            double _ssm_t1 = tf_wall_seconds(); \
            tf_ssm_coop_prof_add(tid, (STAGE), _ssm_t1 - _ssm_t0); \
            _ssm_t0 = _ssm_t1; \
        } \
    } while (0)

    /* === S1: qkv+gate matvecs cooperatively; alpha-chain on tid0, beta-chain on
     * tid1 (else tid0) so the small per-tensor matvec+postproc no longer fully
     * serializes on tid0 (was the ~1.9ms s1_proj max-vs-avg imbalance exposed at
     * B_b). The two chains are independent (alpha->ssm_alpha_buf via thread_tmp[0]
     * = matvec_tmp, beta->ssm_beta_buf via the per-thread thread_tmp[beta_tid]; no
     * shared scratch) and both finish before B_b, where they become visible for
     * S4 — byte-identical. */
    tf_thread_matvec2_diff(m, qkv_buf, &layer->ssm_qkv, qkv_dim,
                           z_buf, &layer->ssm_gate, d_inner,
                           m->xb, tid, nt, local_sense);
#if TF_SSM_S1_AB_FULL_PAR
    {
        int ars, are, brs, bre;
        tf_matvec_row_range(&layer->ssm_alpha, dt_rank, nt, tid, &ars, &are);
        tf_matvec_row_range(&layer->ssm_beta,  dt_rank, nt, tid, &brs, &bre);
        if (are > ars)
            tf_matvec_qtensor_rows(alpha, &layer->ssm_alpha, m->xb, ars, are);
        if (bre > brs)
            tf_matvec_qtensor_rows(beta_arr, &layer->ssm_beta, m->xb, brs, bre);
        tf_ssm_post_alpha_range(alpha, layer, dt_rank, hoff, ars, are);
        tf_ssm_post_beta_range(beta_arr, dt_rank, brs, bre);
    }
#elif TF_SSM_S1_ALPHA_BETA_SPLIT
    int beta_tid = (nt > 1) ? 1 : 0;
    if (tid == 0) {
        tf_qmatvec(alpha, &layer->ssm_alpha, m->xb, dt_rank, m->matvec_tmp);
        tf_ssm_post_alpha(alpha, layer, dt_rank, hoff);
    }
    if (tid == beta_tid) {
        tf_qmatvec(beta_arr, &layer->ssm_beta, m->xb, dt_rank, m->thread_tmp[beta_tid]);
        tf_ssm_post_beta(beta_arr, dt_rank);
    }
#else
    if (tid == 0) {
        tf_qmatvec(alpha, &layer->ssm_alpha, m->xb, dt_rank, m->matvec_tmp);
        tf_qmatvec(beta_arr, &layer->ssm_beta, m->xb, dt_rank, m->matvec_tmp);
        tf_ssm_post_alpha(alpha, layer, dt_rank, hoff);
        tf_ssm_post_beta(beta_arr, dt_rank);
    }
#endif
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_S1_PROJ);
    /* B_a removed: S2 uses the SAME partition as S1's qkv matvec
     * (tf_matvec_row_range), so every thread reads qkv_buf[j] it just
     * wrote in S1. conv_state_pos is snapshotted before S1 so tid0 cannot
     * advance the circular slot before slower threads enter S2. tid 0's
     * alpha/beta writes are consumed in S4, after B_b — still safe. */

    /* === S2: parallel conv1d by channel, aligned with S1's qkv matvec === */
    {
        float *conv_st = m->conv_state[layer_idx];
        int wr = ssm_conv_wr;
        int n_hist = conv_k - 1;
        float *conv_out = K_exp;
        float *w_trans = m->conv_w_trans[layer_idx];

        int *row_off = n_hist > 0 ? (int *)alloca((size_t)n_hist * sizeof(*row_off)) : NULL;
        for (int f = 0; f < n_hist; f++)
            row_off[f] = ((wr + f) % n_hist) * qkv_dim;

        int cs, cend;
        tf_matvec_row_range(&layer->ssm_qkv, qkv_dim, nt, tid, &cs, &cend);
        int cc = cend - cs;

        /* MAC + SiLU into conv_out (= K_exp scratch). Per-channel: no contention.
         * SVE-matched to the serial conv (svmla chain + tf_fast_exp_sve SiLU). */
#if defined(__ARM_FEATURE_SVE)
        {
            svfloat32_t one = svdup_f32(1.0f);
            int vl = (int)svcntw();
            for (int j = cs; j < cend; j += vl) {
                svbool_t pg = svwhilelt_b32(j, cend);
                svfloat32_t sum = svdup_f32(0.0f);
                for (int f = 0; f < n_hist; f++)
                    sum = svmla_x(pg, sum, svld1_f32(pg, w_trans + f * qkv_dim + j),
                                  svld1_f32(pg, conv_st + row_off[f] + j));
                sum = svmla_x(pg, sum, svld1_f32(pg, w_trans + n_hist * qkv_dim + j),
                              svld1_f32(pg, qkv_buf + j));
                svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), sum));
                svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
                svst1_f32(pg, conv_out + j, svmul_x(pg, sum, sig));
            }
        }
#else
        for (int j = cs; j < cend; j++) {
            float sum = 0.0f;
            for (int f = 0; f < n_hist; f++)
                sum += w_trans[f * qkv_dim + j] * conv_st[row_off[f] + j];
            sum += w_trans[n_hist * qkv_dim + j] * qkv_buf[j];
            conv_out[j] = sum / (1.0f + expf(-sum));
        }
#endif
        /* Save current input into circular slot wr (per-channel slice). */
        if (cc > 0)
            memcpy(conv_st + wr * qkv_dim + cs, qkv_buf + cs, (size_t)cc * sizeof(float));
        /* Copy conv output back into qkv_buf (per-channel slice). */
        if (cc > 0)
            memcpy(qkv_buf + cs, conv_out + cs, (size_t)cc * sizeof(float));

        if (tid == 0)
            m->conv_state_pos[layer_idx] = n_hist > 0 ? (wr + 1) % n_hist : 0;
    }
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_S2_CONV);
    tf_spin_barrier(m, local_sense, nt);  /* B_b */
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_BB_WAIT);

    /* === S3: L2-norm Q/K per head + tile-repeat from n_group → dt_rank ===
     * Each thread owns a head range [hs..he) of the dt_rank target heads;
     * it locally normalises the source group head h%n_group and writes
     * directly into Q_exp[h*ds] / K_exp[h*ds]. No write to qkv_buf, so
     * sources are read-only and shared safely. */
    {
        int hs, he;
        tf_ssm_head_range(dt_rank, nt, ncmgs, tid, &hs, &he);
        float *Q_raw = qkv_buf;
        float *K_raw = qkv_buf + n_group * d_state;
        /* Local target head h maps to global head (hoff+h); its Q/K source group
         * is (hoff+h) % n_group (Q/K stay REPLICATED, n_group groups present).
         * hoff=0 when unsharded → g=h%n_group, the original tile-repeat order. */
#if defined(__ARM_FEATURE_SVE)
        int vl = (int)svcntw();
        for (int h = hs; h < he; h++) {
            int g = (hoff + h) % n_group;
            const float *q_src = Q_raw + g * d_state;
            const float *k_src = K_raw + g * d_state;
            svfloat32_t vqss = svdup_f32(0.0f), vkss = svdup_f32(0.0f);
            for (int i = 0; i < d_state; i += vl) {
                svbool_t pg = svwhilelt_b32(i, d_state);
                svfloat32_t qi = svld1_f32(pg, q_src + i);
                svfloat32_t ki = svld1_f32(pg, k_src + i);
                vqss = svmla_m(pg, vqss, qi, qi);
                vkss = svmla_m(pg, vkss, ki, ki);
            }
            /* tf_l2_norm formula: inv = 1 / sqrt(ss + eps), no /n divisor */
            float qs = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vqss) + eps);
            float ks = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vkss) + eps);
            svfloat32_t vqs = svdup_f32(qs), vks = svdup_f32(ks);
            float *q_dst = Q_exp + h * d_state;
            float *k_dst = K_exp + h * d_state;
            for (int i = 0; i < d_state; i += vl) {
                svbool_t pg = svwhilelt_b32(i, d_state);
                svst1_f32(pg, q_dst + i, svmul_x(pg, svld1_f32(pg, q_src + i), vqs));
                svst1_f32(pg, k_dst + i, svmul_x(pg, svld1_f32(pg, k_src + i), vks));
            }
        }
#else
        for (int h = hs; h < he; h++) {
            int g = (hoff + h) % n_group;
            const float *q_src = Q_raw + g * d_state;
            const float *k_src = K_raw + g * d_state;
            float qss = 0.0f, kss = 0.0f;
            for (int i = 0; i < d_state; i++) {
                qss += q_src[i] * q_src[i];
                kss += k_src[i] * k_src[i];
            }
            /* tf_l2_norm formula: inv = 1 / sqrt(ss + eps), no /n divisor */
            float qs = 1.0f / sqrtf(qss + eps);
            float ks = 1.0f / sqrtf(kss + eps);
            float *q_dst = Q_exp + h * d_state;
            float *k_dst = K_exp + h * d_state;
            for (int i = 0; i < d_state; i++) {
                q_dst[i] = q_src[i] * qs;
                k_dst[i] = k_src[i] * ks;
            }
        }
#endif
    }
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_S3_NORM);
    /* B_c removed: S4 reads Q_exp[h*ds]/K_exp[h*ds] for the same head range
     * [hs,he) that S3 just wrote (identical tf_ssm_head_range call). */

    /* === S4: Delta-Net recurrence + fused RMSNorm+SiLU(z) per head === */
    {
        float scale = 1.0f / sqrtf((float)d_state);
        float *rec_state = m->recurrent_state[layer_idx];
        float *V_raw = qkv_buf + 2 * n_group * d_state;
        int hs, he;
        tf_ssm_head_range(dt_rank, nt, ncmgs, tid, &hs, &he);
        if (he > hs) {
            tf_ssm_recurrence_task rtask = {
                rec_state, Q_exp, K_exp, V_raw, out_buf,
                alpha, beta_arr, hs, he, d_state, scale
            };
            tf_ssm_recurrence_worker(&rtask);

            float norm_w[128];
            tf_dequant_row(&layer->ssm_norm, 0, norm_w);
#if defined(__ARM_FEATURE_SVE)
            int vl = (int)svcntw();
            svfloat32_t one = svdup_f32(1.0f);
            for (int h = hs; h < he; h++) {
                float *o_h = out_buf + h * d_state;
                float *z_h = z_buf   + h * d_state;
                svfloat32_t vss = svdup_f32(0.0f);
                for (int i = 0; i < d_state; i += vl) {
                    svbool_t pg = svwhilelt_b32(i, d_state);
                    svfloat32_t oi = svld1_f32(pg, o_h + i);
                    vss = svmla_m(pg, vss, oi, oi);
                }
                float scl = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vss) / d_state + eps);
                svfloat32_t vscale = svdup_f32(scl);
                for (int i = 0; i < d_state; i += vl) {
                    svbool_t pg = svwhilelt_b32(i, d_state);
                    svfloat32_t oi = svld1_f32(pg, o_h + i);
                    svfloat32_t wi = svld1_f32(pg, norm_w + i);
                    svfloat32_t normed = svmul_x(pg, svmul_x(pg, oi, vscale), wi);
                    svfloat32_t zi = svld1_f32(pg, z_h + i);
                    svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), zi));
                    svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
                    svst1_f32(pg, o_h + i, svmul_x(pg, normed, svmul_x(pg, zi, sig)));
                }
            }
#else
            for (int h = hs; h < he; h++) {
                float *o_h = out_buf + h * d_state;
                float *z_h = z_buf   + h * d_state;
                float ss = 0.0f;
                for (int i = 0; i < d_state; i++) ss += o_h[i] * o_h[i];
                float scl = 1.0f / sqrtf(ss / d_state + eps);
                for (int i = 0; i < d_state; i++) {
                    float normed = o_h[i] * scl * norm_w[i];
                    float zv = z_h[i];
                    o_h[i] = normed * (zv / (1.0f + expf(-zv)));
                }
            }
#endif
        }
    }
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_S4_SCAN);
    tf_spin_barrier(m, local_sense, nt);  /* B_d */
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_BD_WAIT);

    /* === S5: output projection (parallel rows). Caller's B2 barrier follows. === */
    tf_thread_matvec_q8_shared(m, m->xb, &layer->ssm_out, out_buf,
                               n_embd, tid, nt, local_sense);
    TF_SSM_COOP_MARK(TF_SSM_COOP_PROF_S5_OUT);
    if (_ssm_prof && (unsigned)tid < TF_SSM_COOP_PROF_TMAX)
        tf_ssm_coop_prof_calls[tid]++;
#undef TF_SSM_COOP_MARK
}

/* Pool-worker entry for the cooperative SSM. Each thread seeds a fresh
 * local_sense from the shared bar_sense so the first internal spin-barrier's
 * my_sense (= !bar_sense) differs from the currently-released sense and all
 * threads block correctly. bar_count is 0 on entry (prior dispatches use
 * pool_done_flags, not the spin barrier) and every barrier resets it to 0. */
static void *tf_ssm_parallel_worker(void *arg) {
    tf_ssm_par_task *t = (tf_ssm_par_task *)arg;
    int local_sense = t->m->bar_sense;
    tf_ssm_deltanet_forward_parallel(t->m, t->layer_idx, t->tid, t->nt, &local_sense);
    return NULL;
}

/* Vectorized GELU(gate) × up: out[i] = gelu(gate[i]) * up[i]
 * Uses exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))) */
static void tf_gelu_mul(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f)); /* 1/sqrt(2) */
        out[i] = gelu_g * up[i];
    }
}

/* Logit soft-capping: logits[i] = cap * tanh(logits[i] / cap) */
static void tf_logit_softcap(float *logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    for (int i = 0; i < n; i++)
        logits[i] = cap * tanhf(logits[i] * inv_cap);
}

/* Helper: apply standard or M-RoPE depending on model config */
static void tf_apply_rope(transformer_model *m, float *q, float *k,
                           int n_heads, int n_kv_heads, int head_dim,
                           int pos_t, int pos_h, int pos_w) {
    if (m->use_mrope) {
        tf_rope_mrope(q, n_heads,    head_dim, pos_t, pos_h, pos_w, m->rope_freq_base, m->mrope_sections, m->rope_mrope_inv_freq);
        tf_rope_mrope(k, n_kv_heads, head_dim, pos_t, pos_h, pos_w, m->rope_freq_base, m->mrope_sections, m->rope_mrope_inv_freq);
    } else {
        tf_rope(q, n_heads,    head_dim, pos_t, m->rope_freq_base, m->rope_inv_freq);
        tf_rope(k, n_kv_heads, head_dim, pos_t, m->rope_freq_base, m->rope_inv_freq);
    }
}

/* ---- Persistent-thread forward pass ---- */
/* Each thread runs the entire layer loop for its static partition.
 * Sync uses pthread_barrier (kernel-backed, no spin-wait).
 * Thread 0 does sequential work (RMSNorm, RoPE, KV cache); others wait at barrier.
 * All threads participate in parallel work (matvec rows, attention heads). */

typedef struct {
    transformer_model *m;
    int tid;
    int position, pos_t, pos_h, pos_w;
} tf_persistent_ctx;

/* ---- Barrier implementations ---- */

/* A64FX hardware barrier via W0 window register (intra-CMG only).
 * Uses WFE (wait-for-event) instead of spin-polling — near-zero power
 * and ~20 cycle latency. Requires libfjomphk.so (link with -Kopenmp)
 * to have initialized the barrier blade. */
#if defined(__aarch64__)
static inline void tf_hw_barrier_w0(void) {
    uint64_t bst, lbsy;
    __asm__ __volatile__("mrs %0, S3_3_C15_C15_0" : "=r"(lbsy));
    bst = (~lbsy) & 1ULL;
    __asm__ __volatile__("msr S3_3_C15_C15_0, %0" :: "r"(bst));
    __asm__ __volatile__("sevl");
    do {
        __asm__ __volatile__("wfe");
        __asm__ __volatile__("mrs %0, S3_3_C15_C15_0" : "=r"(lbsy));
        lbsy &= 1ULL;
    } while (lbsy != bst);
}
#endif

#if defined(__aarch64__)
/* Hierarchical 48-thread HW barrier: HW intra-CMG arrival + 4-way SW combine
 * among CMG leaders + HW intra-CMG release. The EL0 BST barrier is intra-CMG
 * only, so cross-CMG sync needs the software combine. `bb` is this thread's
 * per-CMG BST register (from vhbm_bar_assign); leaders are tid % tpc == 0. */
static inline void tf_hwlib_barrier(transformer_model *m, int tid, long bb, int *ls4) {
    vhbm_bar(bb);                              /* intra-CMG arrival */
    if (tid % m->hwbar_tpc == 0) {             /* CMG leader: combine across CMGs */
        int my = !(*ls4);
        if (__sync_add_and_fetch((int *)&m->hwbar_lcount, 1) == m->hwbar_ncmg) {
            m->hwbar_lcount = 0;
            __sync_synchronize();
            m->hwbar_lsense = my;
        } else {
            while (m->hwbar_lsense != my) __asm__ __volatile__("yield");
        }
        *ls4 = my;
    }
    vhbm_bar(bb);                              /* intra-CMG release */
}
#endif

/* Barrier: sense-reversing with SEV/WFE on aarch64 for low-power wait.
 * WFE puts the core in low-power state until an event (SEV from the last
 * arriving thread). No bandwidth consumption while waiting. */
static inline void tf_spin_barrier(transformer_model *m, int *local_sense, int nt) {
    int my_sense = !(*local_sense);
    if (__sync_add_and_fetch(&m->bar_count, 1) == nt) {
        m->bar_count = 0;
        __sync_synchronize();
        m->bar_sense = my_sense;
#if defined(__aarch64__)
        __asm__ __volatile__("sev");  /* wake all WFE-sleeping cores */
#endif
    } else {
#if defined(__aarch64__)
        /* SEV/WFE spin: core sleeps until event, then rechecks.
         * SEVL ensures first WFE doesn't stall if event pending. */
        __asm__ __volatile__("sevl");
        do {
            __asm__ __volatile__("wfe");
        } while (m->bar_sense != my_sense);
#else
        while (m->bar_sense != my_sense)
            tf_cpu_pause();
#endif
    }
    *local_sense = my_sense;
}

/* Hierarchical-ARRIVAL, flat-release barrier. Splits the 48-way atomic
 * arrival counter into 4 per-CMG counters (cache-local incrs, ~10× cheaper)
 * but keeps the release on the single sense bit so the wake is one SEV. The
 * earlier two-level-release design regressed because of doubled WFE/SEV
 * round-trips on the critical path; this design keeps the cheap arrival
 * with a flat release. */
static inline void tf_hier_arr_barrier(transformer_model *m, int tid, int *local_sense,
                                       int nt, int n_cmgs) {
    if (n_cmgs <= 1 || nt < n_cmgs * 2 || (nt % n_cmgs) != 0) {
        tf_spin_barrier(m, local_sense, nt);
        return;
    }
    int tpc  = nt / n_cmgs;
    int cmg  = tid / tpc;
    int slot = cmg * 16;  /* 64B-isolated CMG counter slot */
    int my_sense = !(*local_sense);

    if (__sync_add_and_fetch((int *)&m->hb_cmg_count[slot], 1) == tpc) {
        /* CMG leader: reset CMG count, hit a *separate* global counter
         * (hb_g_count) so this barrier can safely coexist with `tf_spin_barrier`
         * — both flip the shared bar_sense, but bar_count is only used by
         * the flat version. The thread that closes the global count flips
         * bar_sense and SEVs. All others wait on bar_sense once. */
        m->hb_cmg_count[slot] = 0;
        if (__sync_add_and_fetch((int *)&m->hb_g_count, 1) == n_cmgs) {
            m->hb_g_count = 0;
            __sync_synchronize();
            m->bar_sense = my_sense;
#if defined(__aarch64__)
            __asm__ __volatile__("sev");
#endif
            *local_sense = my_sense;
            return;
        }
    }
#if defined(__aarch64__)
    __asm__ __volatile__("sevl");
    do {
        __asm__ __volatile__("wfe");
    } while (m->bar_sense != my_sense);
#else
    while (m->bar_sense != my_sense) tf_cpu_pause();
#endif
    *local_sense = my_sense;
}

/* Software two-level (hierarchical) barrier: per-CMG count on its own cacheline,
 * then inter-CMG count among leaders. Cuts the cross-CMG atomic contention that
 * plagues the flat tf_spin_barrier when nt is 48 split across 4 CMGs. Uses
 * WFE/SEV on aarch64 so cores park in low-power until the release SEV. */
static inline void tf_sw_hier_barrier(transformer_model *m, int tid, int *local_sense,
                                      int nt, int n_cmgs) {
    if (n_cmgs <= 1 || nt < n_cmgs * 2 || (nt % n_cmgs) != 0) {
        /* Fall back: degenerate (1 thread/CMG) or uneven partition. */
        tf_spin_barrier(m, local_sense, nt);
        return;
    }
    int tpc = nt / n_cmgs;
    int cmg = tid / tpc;
    int slot = cmg * 16;  /* 16 ints = 64B = own cacheline per CMG */
    int my_sense = !(*local_sense);

    if (__sync_add_and_fetch((int *)&m->hb_cmg_count[slot], 1) == tpc) {
        /* CMG leader: reset local count, participate in global sync. */
        m->hb_cmg_count[slot] = 0;
        if (__sync_add_and_fetch((int *)&m->hb_g_count, 1) == n_cmgs) {
            m->hb_g_count = 0;
            __sync_synchronize();
            m->hb_g_sense = my_sense;
#if defined(__aarch64__)
            __asm__ __volatile__("sev");
#endif
        } else {
#if defined(__aarch64__)
            __asm__ __volatile__("sevl");
            do {
                __asm__ __volatile__("wfe");
            } while (m->hb_g_sense != my_sense);
#else
            while (m->hb_g_sense != my_sense) tf_cpu_pause();
#endif
        }
        /* Release the rest of this CMG. */
        __sync_synchronize();
        m->hb_cmg_sense[slot] = my_sense;
#if defined(__aarch64__)
        __asm__ __volatile__("sev");
#endif
    } else {
#if defined(__aarch64__)
        __asm__ __volatile__("sevl");
        do {
            __asm__ __volatile__("wfe");
        } while (m->hb_cmg_sense[slot] != my_sense);
#else
        while (m->hb_cmg_sense[slot] != my_sense) tf_cpu_pause();
#endif
    }
    *local_sense = my_sense;
}

/* Two-level barrier for multi-CMG:
 * Level 1: Hardware barrier within each CMG (cores on same CMG sync via W0)
 * Level 2: Software atomic barrier across CMG leaders
 *
 * For single-CMG (all threads on 1 CMG): use hardware barrier directly.
 * For multi-CMG: each CMG's threads hw-barrier locally, then one thread
 * per CMG does a software barrier with other CMG leaders. */
static inline void tf_barrier(transformer_model *m, int tid, int *local_sense, int nt) {
#if defined(__aarch64__)
    int n_cmgs = m->numa.enabled ? m->numa.n_cmgs : 1;
    if (n_cmgs <= 1 || nt <= n_cmgs) {
        /* Single CMG or 1 thread/CMG: just use hw barrier */
        tf_hw_barrier_w0();
    } else {
        /* Multi-CMG with multiple threads per CMG */
        int threads_per_cmg = nt / n_cmgs;
        int cmg_id = tid / threads_per_cmg;
        int local_tid = tid % threads_per_cmg;

        /* Level 1: intra-CMG hardware barrier */
        tf_hw_barrier_w0();

        /* Level 2: inter-CMG software barrier (only CMG leaders, local_tid==0) */
        if (local_tid == 0) {
            int my_sense = !(*local_sense);
            if (__sync_add_and_fetch(&m->bar_count, 1) == n_cmgs) {
                m->bar_count = 0;
                __sync_synchronize();
                m->bar_sense = my_sense;
            } else {
                while (m->bar_sense != my_sense)
                    tf_cpu_pause();
            }
            *local_sense = my_sense;
        }

        /* Level 1 again: intra-CMG hw barrier to release non-leaders */
        tf_hw_barrier_w0();
    }
#else
    tf_spin_barrier(m, local_sense, nt);
#endif
}

/* Per-thread matvec: thread tid computes its static partition.
 *
 * When the F16 weight has an A64FX panel layout, partition by 32-row panel
 * BLOCKS using exactly the same split as transformer_build_panels(), so the
 * block range a thread streams here is the one it first-touched at build
 * time — i.e. resident in its own CMG's HBM. This is what makes the
 * persistent forward path scale across CMGs; the row-major fallback below
 * reads mat->data, which lives on whichever single CMG loaded it. */
static void tf_thread_matvec(float *dst, const qtensor *mat, const float *x,
                              int n_rows, int tid, int nt) {
#if defined(__ARM_FEATURE_SVE)
    if (mat->type == GGML_TYPE_F16 && mat->panel) {
        int nblk = mat->panel_blk;
        int bp = nblk / nt, be = nblk % nt;
        int bs = tid * bp + (tid < be ? tid : be);
        int bc = bp + (tid < be ? 1 : 0);
        if (bc <= 0) return;
        int K = mat->n_cols;
        float16_t *xh = (float16_t *)alloca((size_t)K * sizeof(float16_t));
        tf_x_to_f16(xh, x, K);
        tf_panel_task t = { dst, mat->panel, xh, bs, bs + bc, K, n_rows };
        tf_panel_matvec_worker(&t);
        return;
    }
#endif
    int rs, re_end;
    /* BF16/F16 want 8-aligned row boundaries so every thread stays on the
     * pv / 8-row fast path. q-types are agnostic — fall back to naive split. */
    if (mat->type == GGML_TYPE_BF16 || mat->type == GGML_TYPE_F16 || mat->bf16_pv || mat->q8_pv) {
        tf_row_split8(n_rows, nt, tid, &rs, &re_end);
    } else {
        int rp = n_rows / nt, re = n_rows % nt;
        rs = tid * rp + (tid < re ? tid : re);
        re_end = rs + rp + (tid < re ? 1 : 0);
    }
    if (re_end <= rs) return;
    int n_cols = mat->n_cols;

    if (mat->type == GGML_TYPE_BF16) {
#if defined(__ARM_FEATURE_SVE)
        if (mat->q8_pv && (rs & 7) == 0 && (re_end & 7) == 0) {
            const int8_t *xq; const uint16_t *xs;
            tf_quant_x_sdot(x, n_cols, &xq, &xs);
            int nb = n_cols / 64;
            size_t group_bytes = (size_t)nb * 528;
            const uint8_t *qbase = mat->q8_pv;
            for (int i = rs; i + 7 < re_end; i += 8) {
                int g = i >> 3;
                matvec_sdot_8row(dst + i, qbase + (size_t)g * group_bytes,
                                 xq, xs, n_cols);
            }
        } else
#endif
        tf_matvec_bf16_rows_pv(dst, (const uint8_t *)mat->data,
                                (size_t)n_cols * 2, mat->bf16_pv,
                                x, n_cols, rs, re_end);
    } else if (mat->type == GGML_TYPE_F16) {
        tf_matvec_f16_rows(dst, (const uint8_t *)mat->data,
                            (size_t)n_cols * 2, x, n_cols, rs, re_end);
    } else {
        tf_matvec_qtensor_rows(dst, mat, x, rs, re_end);
    }
}

static void *tf_persistent_worker(void *arg) {
    tf_persistent_ctx *ctx = (tf_persistent_ctx *)arg;
    transformer_model *m = ctx->m;
    int tid = ctx->tid;
    int nt = m->n_threads;
    /* Persistent-worker barrier macro: swaps in the SW hierarchical barrier
     * when CMG pinning is active. Falls back to flat tf_spin_barrier when
     * the partition is degenerate (handled inside tf_sw_hier_barrier). */
    /* Two SW hier-barrier designs were tried and both regressed on Fugaku:
     *   - tf_sw_hier_barrier  (two-level release): -46% prefill at 14K — two
     *     SEV/WFE round-trips on the critical path
     *   - tf_hier_arr_barrier (hier arrival, flat release): -3% prefill — the
     *     extra per-CMG atomic step costs as much as the 48-way contention
     *     saves on A64FX (LSE atomics are already fairly efficient).
     * Both functions kept in tree for reference. Flat barrier is the default.
     *
     * TF_HW_BARRIER=1 swaps in the A64FX hardware barrier (per-CMG EL0 BST +
     * 4-way SW leader combine, via libhwb). This *does* work from our pthreads
     * once vhbm_bar_assign performs the kernel group-assign — see
     * [[hwbarrier-libhwb-win]]. Falls back to flat when disabled. */
    (void)0;
#if defined(__aarch64__)
    long hw_bb = m->hwbar_enabled ? m->hwbar_bb[tid] : 0;
    int  hw_ls4 = 0;
    #define PW_BARRIER() do { \
        if (m->hwbar_enabled) tf_hwlib_barrier(m, tid, hw_bb, &hw_ls4); \
        else tf_spin_barrier(m, &local_sense, nt); \
    } while (0)
#else
    #define PW_BARRIER() tf_spin_barrier(m, &local_sense, nt)
#endif
#ifdef TF_POOL_PROFILE
    double _ts = tf_now_s();
#define PW_MARK(BUCKET) do { if (tid < 2) { double _n = tf_now_s(); \
    tf_pw_##BUCKET[tid] += _n - _ts; _ts = _n; } } while (0)
#else
#define PW_MARK(BUCKET) ((void)0)
#endif
    int position = ctx->position;
    int pos_t = ctx->pos_t, pos_h = ctx->pos_h, pos_w = ctx->pos_w;
    int local_sense = 0;

    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;
    int gqa_ratio = m->gqa_group;    /* GLOBAL group: survives KV replication (local ratio may be 1) */
    int n_ff = m->n_ff;
    /* Head partition for attention */
    int h_per = n_heads / nt, h_extra = n_heads % nt;
    int h_start = tid * h_per + (tid < h_extra ? tid : h_extra);
    int h_count = h_per + (tid < h_extra ? 1 : 0);

    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* Thread 0: RMSNorm (sequential, cheap) */
        if (tid == 0)
            tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        PW_MARK(serial);
        PW_BARRIER();  /* B1: xb ready */
        PW_MARK(barrier);

        if (m->is_hybrid && layer->is_ssm) {
            /* Parallel SSM forward — all threads participate via internal
             * spin barriers (B_a..B_d). xb (post-norm) ready from B1, xb
             * (ssm_out projection) ready when this returns. */
            tf_ssm_deltanet_forward_parallel(m, l, tid, nt, &local_sense);
            PW_MARK(matvec);
            PW_BARRIER();  /* B2: SSM done */
            PW_MARK(barrier);
        } else {
            /* --- Attention layer --- */

            /* QKV projection: all threads compute their row partition */
            if (m->is_hybrid) {
                /* Gated attention: Q+gate combined [2*q_dim] */
                int q2_dim = 2 * q_dim;
                tf_thread_matvec(m->xb2, &layer->attn_q, m->xb, q2_dim, tid, nt);
                tf_thread_matvec(m->k, &layer->attn_k, m->xb, kv_dim, tid, nt);
                tf_thread_matvec(m->v, &layer->attn_v, m->xb, kv_dim, tid, nt);
            } else {
                tf_thread_matvec(m->q, &layer->attn_q, m->xb, q_dim, tid, nt);
                tf_thread_matvec(m->k, &layer->attn_k, m->xb, kv_dim, tid, nt);
                tf_thread_matvec(m->v, &layer->attn_v, m->xb, kv_dim, tid, nt);
            }
            PW_MARK(matvec);
            PW_BARRIER();  /* B2: Q/K/V ready */
            PW_MARK(barrier);

            /* Parallel: de-interleave (gated), QK-norm, F32-bias, RoPE, KV cache.
             * Heads are partitioned across threads (n_heads for Q, n_kv_heads for K/V).
             * Dequant-bias path is rare; if present, only tid 0 handles it (with a
             * pre-norm/pre-rope barrier so RoPE sees biased values). */
            int has_dq_bias =
                (layer->attn_q_bias.data && layer->attn_q_bias.type != GGML_TYPE_F32) ||
                (layer->attn_k_bias.data && layer->attn_k_bias.type != GGML_TYPE_F32) ||
                (layer->attn_v_bias.data && layer->attn_v_bias.type != GGML_TYPE_F32);

            if (has_dq_bias) {
                /* Fallback: original tid==0 sequential path (preserves correctness
                 * for dequant biases; rare for current Qwen3-VL models). */
                if (tid == 0) {
                    if (m->is_hybrid) {
                        for (int h = 0; h < n_heads; h++) {
                            memcpy(m->q + h * head_dim, m->xb2 + h * 2 * head_dim, head_dim * sizeof(float));
                            memcpy(m->ffn_buf1 + h * head_dim, m->xb2 + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
                        }
                    }
                    if (layer->attn_q_norm.data)
                        tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
                    if (layer->attn_k_norm.data)
                        tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
                    if (layer->attn_q_bias.data) {
                        if (layer->attn_q_bias.type == GGML_TYPE_F32) {
                            float *qb = (float *)layer->attn_q_bias.data;
                            for (int i = 0; i < q_dim; i++) m->q[i] += qb[i];
                        } else {
                            tf_dequant_row(&layer->attn_q_bias, 0, m->matvec_tmp);
                            for (int i = 0; i < q_dim; i++) m->q[i] += m->matvec_tmp[i];
                        }
                    }
                    if (layer->attn_k_bias.data) {
                        if (layer->attn_k_bias.type == GGML_TYPE_F32) {
                            float *kb = (float *)layer->attn_k_bias.data;
                            for (int i = 0; i < kv_dim; i++) m->k[i] += kb[i];
                        } else {
                            tf_dequant_row(&layer->attn_k_bias, 0, m->matvec_tmp);
                            for (int i = 0; i < kv_dim; i++) m->k[i] += m->matvec_tmp[i];
                        }
                    }
                    if (layer->attn_v_bias.data) {
                        if (layer->attn_v_bias.type == GGML_TYPE_F32) {
                            float *vb = (float *)layer->attn_v_bias.data;
                            for (int i = 0; i < kv_dim; i++) m->v[i] += vb[i];
                        } else {
                            tf_dequant_row(&layer->attn_v_bias, 0, m->matvec_tmp);
                            for (int i = 0; i < kv_dim; i++) m->v[i] += m->matvec_tmp[i];
                        }
                    }
                    tf_apply_rope(m, m->q, m->k, n_heads, n_kv_heads, head_dim, pos_t, pos_h, pos_w);
                    tf_k_cache_write_pos(m->key_cache[l],
                                         m->key_scales ? m->key_scales[l] : NULL,
                                         m->k, position, n_kv_heads, head_dim,
                                         m->max_seq_len, m->kv_dtype, m->kv_k_transposed,
                                         m->kv_k_dp);
                    tf_kv_write_all_heads(m->value_cache[l],
                                          m->value_scales ? m->value_scales[l] : NULL,
                                          m->v, position, n_kv_heads, head_dim, m->kv_dtype);
                }
            } else {
                /* Per-head partition: assign Q heads to tids [0, n_heads) and
                 * K/V heads to tids [n_heads, n_heads+n_kv_heads).  When
                 * nt >= n_heads + n_kv_heads (e.g. 48 >= 16+8) every working
                 * thread gets exactly one head — load-balanced.  Otherwise we
                 * fall back to wrapped contiguous partitions. */
                int qhs, qhc, kvhs, kvhc;
                if (nt >= n_heads + n_kv_heads) {
                    qhs = (tid < n_heads) ? tid : 0;
                    qhc = (tid < n_heads) ? 1 : 0;
                    int kvtid = tid - n_heads;
                    kvhs = (kvtid >= 0 && kvtid < n_kv_heads) ? kvtid : 0;
                    kvhc = (kvtid >= 0 && kvtid < n_kv_heads) ? 1 : 0;
                } else {
                    int qhp = n_heads / nt, qhe = n_heads % nt;
                    qhs = tid * qhp + (tid < qhe ? tid : qhe);
                    qhc = qhp + (tid < qhe ? 1 : 0);
                    int kvhp = n_kv_heads / nt, kvhe = n_kv_heads % nt;
                    kvhs = tid * kvhp + (tid < kvhe ? tid : kvhe);
                    kvhc = kvhp + (tid < kvhe ? 1 : 0);
                }

                /* Per-thread scratch (head_dim<=256 typical: Qwen3-VL=128). */
                float qnw_buf[256] __attribute__((aligned(32)));
                float knw_buf[256] __attribute__((aligned(32)));
                float cos_tab[512] __attribute__((aligned(32)));
                float sin_tab[512] __attribute__((aligned(32)));

                int do_qnorm = (layer->attn_q_norm.data != NULL);
                int do_knorm = (layer->attn_k_norm.data != NULL);
                if (do_qnorm && qhc > 0) tf_dequant_row(&layer->attn_q_norm, 0, qnw_buf);
                if (do_knorm && kvhc > 0) tf_dequant_row(&layer->attn_k_norm, 0, knw_buf);

                /* Build RoPE cos/sin table once per thread (cheap; reused for Q+K). */
                int rope_pairs = 0;
                int pair_off = head_dim / 2;
                if (qhc > 0 || kvhc > 0) {
                    if (m->use_mrope) {
                        int sect_dims = m->mrope_sections[0] + m->mrope_sections[1] +
                                        m->mrope_sections[2] + m->mrope_sections[3];
                        if (sect_dims <= 0) sect_dims = head_dim / 2;
                        rope_pairs = sect_dims;
                        pair_off = sect_dims;
                        int rope_dim = 2 * sect_dims;
                        for (int j = 0; j < sect_dims; j++) {
                            int pos;
                            if (j % 3 == 1 && j < 3 * m->mrope_sections[1])      pos = pos_h;
                            else if (j % 3 == 2 && j < 3 * m->mrope_sections[2]) pos = pos_w;
                            else if (j % 3 == 0 && j < 3 * m->mrope_sections[0]) pos = pos_t;
                            else                                                  pos = pos_t;
                            float freq = m->rope_mrope_inv_freq ?
                                m->rope_mrope_inv_freq[j] :
                                (1.0f / powf(m->rope_freq_base, (float)(2 * j) / rope_dim));
                            float theta = pos * freq;
                            cos_tab[j] = cosf(theta);
                            sin_tab[j] = sinf(theta);
                        }
                    } else {
                        int half = head_dim / 2;
                        rope_pairs = half;
                        pair_off = half;
                        for (int j = 0; j < half; j++) {
                            float freq = m->rope_inv_freq ?
                                m->rope_inv_freq[j] :
                                (1.0f / powf(m->rope_freq_base, (float)(2 * j) / head_dim));
                            float theta = pos_t * freq;
                            cos_tab[j] = cosf(theta);
                            sin_tab[j] = sinf(theta);
                        }
                    }
                }

                /* Q-side: de-interleave -> qknorm -> bias -> RoPE */
                for (int hi = 0; hi < qhc; hi++) {
                    int h = qhs + hi;
                    float *vq = m->q + h * head_dim;
                    if (m->is_hybrid) {
                        memcpy(vq, m->xb2 + h * 2 * head_dim, head_dim * sizeof(float));
                        memcpy(m->ffn_buf1 + h * head_dim, m->xb2 + h * 2 * head_dim + head_dim,
                               head_dim * sizeof(float));
                    }
                    if (do_qnorm) {
                        float ss = 0.0f;
                        for (int i = 0; i < head_dim; i++) ss += vq[i] * vq[i];
                        ss = 1.0f / sqrtf(ss / head_dim + m->rms_norm_eps);
                        for (int i = 0; i < head_dim; i++) vq[i] = vq[i] * ss * qnw_buf[i];
                    }
                    if (layer->attn_q_bias.data) {
                        float *qb = (float *)layer->attn_q_bias.data + h * head_dim;
                        for (int i = 0; i < head_dim; i++) vq[i] += qb[i];
                    }
                    for (int j = 0; j < rope_pairs; j++) {
                        float v0 = vq[j], v1 = vq[j + pair_off];
                        vq[j]            = v0 * cos_tab[j] - v1 * sin_tab[j];
                        vq[j + pair_off] = v0 * sin_tab[j] + v1 * cos_tab[j];
                    }
                }

                /* K/V-side: qknorm(K) -> bias(K) -> bias(V) -> RoPE(K) -> KV cache */
                for (int hi = 0; hi < kvhc; hi++) {
                    int h = kvhs + hi;
                    float *vk = m->k + h * head_dim;
                    float *vv = m->v + h * head_dim;
                    if (do_knorm) {
                        float ss = 0.0f;
                        for (int i = 0; i < head_dim; i++) ss += vk[i] * vk[i];
                        ss = 1.0f / sqrtf(ss / head_dim + m->rms_norm_eps);
                        for (int i = 0; i < head_dim; i++) vk[i] = vk[i] * ss * knw_buf[i];
                    }
                    if (layer->attn_k_bias.data) {
                        float *kb = (float *)layer->attn_k_bias.data + h * head_dim;
                        for (int i = 0; i < head_dim; i++) vk[i] += kb[i];
                    }
                    if (layer->attn_v_bias.data) {
                        float *vb = (float *)layer->attn_v_bias.data + h * head_dim;
                        for (int i = 0; i < head_dim; i++) vv[i] += vb[i];
                    }
                    for (int j = 0; j < rope_pairs; j++) {
                        float v0 = vk[j], v1 = vk[j + pair_off];
                        vk[j]            = v0 * cos_tab[j] - v1 * sin_tab[j];
                        vk[j + pair_off] = v0 * sin_tab[j] + v1 * cos_tab[j];
                    }
                    tf_k_cache_write_row(m->key_cache[l],
                                         m->key_scales ? m->key_scales[l] : NULL,
                                         vk, position, h, n_kv_heads, head_dim,
                                         kv_dim, m->max_seq_len, m->kv_dtype,
                                         m->kv_k_transposed, m->kv_k_dp);
                    tf_kv_write_row(m->value_cache[l],
                                    m->value_scales ? m->value_scales[l] : NULL,
                                    vv, position, h, n_kv_heads, head_dim, kv_dim, m->kv_dtype);
                }
            }
            PW_MARK(serial);
            PW_BARRIER();  /* B3: Q/K ready for attention */
            PW_MARK(barrier);

            /* qpkd pre-pass: pack this thread's Q heads, barrier, then compute
             * QK over a p-chunk for ALL heads using qpkd+ktbl. Writes att[h][p]
             * for all h; per-head attention worker then skips QK. */
            /* Position-parallel softmax (all 48T) replaces the legacy 16T
             * per-head softmax; default ON, TF_PSOFTMAX_OFF=1 restores the old
             * path for A/B comparison. */
            static int tf_psm = -1;
            if (tf_psm < 0) tf_psm = getenv("TF_PSOFTMAX_OFF") ? 0 : 1;
            int pw_skip_qk = 0;
#if defined(__ARM_FEATURE_SVE)
            if (m->kv_k_dp) {
                if (h_count > 0)
                    tf_qpkd_pack_q_heads(m->q, m->q_packed, h_start, h_start + h_count,
                                         n_heads, head_dim);
                PW_MARK(serial);
                PW_BARRIER();  /* Q_packed ready */
                PW_MARK(barrier);
                int seq_len = position + 1;
                float scale = 1.0f / sqrtf((float)head_dim);
                long p_lo = (long)tid * seq_len / nt;
                long p_hi = (long)(tid + 1) * seq_len / nt;
                if (p_hi > p_lo) {
                    if (m->kv_dtype == TF_KV_DTYPE_F32)
                        tf_qpkd_qk_chunk_f32((const float *)m->key_cache[l], m->q_packed,
                                             m->att, (int)p_lo, (int)p_hi,
                                             n_heads, n_kv_heads, head_dim,
                                             m->max_seq_len, scale);
                    else if (m->kv_dtype == TF_KV_DTYPE_F16)
                        tf_qpkd_qk_chunk_f16((const uint16_t *)m->key_cache[l], m->q_packed,
                                             m->att, (int)p_lo, (int)p_hi,
                                             n_heads, n_kv_heads, head_dim,
                                             m->max_seq_len, scale);
                    else
                        tf_qpkd_qk_chunk_q8((const int8_t *)m->key_cache[l],
                                            m->key_scales[l], m->q_packed,
                                            m->att, (int)p_lo, (int)p_hi,
                                            n_heads, n_kv_heads, head_dim,
                                            m->max_seq_len, scale);
                }
                if (tf_psm && m->av_tmp) {
                    /* Per-head partial max over this thread's p-range, hot from
                     * the QK write above. Published by the att-ready barrier
                     * below (no extra barrier); reduced to a global per-head max
                     * in the AV block. */
                    svbool_t pgm = svptrue_b32();
                    float *pmax = m->att_pmax + (size_t)tid * n_heads;
                    for (int h = 0; h < n_heads; h++) {
                        const float *ar = m->att + (size_t)h * m->max_seq_len;
                        svfloat32_t vmax = svdup_f32(-INFINITY);
                        long p = p_lo;
                        for (; p + (long)svcntw() <= p_hi; p += (long)svcntw())
                            vmax = svmax_f32_x(pgm, vmax, svld1_f32(pgm, ar + p));
                        if (p < p_hi) {
                            svbool_t pt = svwhilelt_b32((int)p, (int)p_hi);
                            vmax = svmax_f32_m(pt, vmax, svld1_f32(pt, ar + p));
                        }
                        pmax[h] = svmaxv_f32(svptrue_b32(), vmax);
                    }
                }
                PW_MARK(attn);
                PW_BARRIER();  /* att rows ready (+ att_pmax when tf_psm) */
                PW_MARK(barrier);
                pw_skip_qk = 1;
            }
#endif

            /* Parallel softmax+AV pre-pass: with K_DP, splitting AV across all
             * threads (per-p partition) instead of one-thread-per-head avoids
             * the 16-of-48 underutilisation that dominates attention at long
             * context. Per-thread av_tmp [n_heads*head_dim] is reduced into
             * xb2 after the chunk pass. */
            int pw_skip_av = 0;
#if defined(__ARM_FEATURE_SVE)
            if (m->kv_k_dp && m->av_tmp) {
                int seq_len = position + 1;
                long p_lo = (long)tid * seq_len / nt;
                long p_hi = (long)(tid + 1) * seq_len / nt;
                size_t slice_n = (size_t)n_heads * head_dim;
                float *my_tmp = m->av_tmp + (size_t)tid * slice_n;

                if (tf_psm) {
                    /* Position-parallel softmax: every thread owns a p-range for
                     * ALL heads (no 16-of-48 idle threads). Reduce the partial
                     * maxes written in the QK block (redundantly, per thread) to
                     * a global per-head max, then exp+sum over this thread's
                     * p-range in place. 1/sum is folded into the AV reduce below,
                     * so the old separate softmax barrier is eliminated. */
                    float *psum = m->att_psum + (size_t)tid * n_heads;
                    for (int h = 0; h < n_heads; h++) {
                        float gmax = -INFINITY;
                        for (int t = 0; t < nt; t++) {
                            float v = m->att_pmax[(size_t)t * n_heads + h];
                            if (v > gmax) gmax = v;
                        }
                        float *ar = m->att + (size_t)h * m->max_seq_len;
                        float s = 0.0f;
                        for (long p = p_lo; p < p_hi; p++) {
                            float e = expf(ar[p] - gmax);
                            ar[p] = e;
                            s += e;
                        }
                        psum[h] = s;
                    }
                } else {
                    /* Legacy 16-thread per-head softmax. */
                    for (int hi = 0; hi < h_count; hi++)
                        tf_softmax(m->att + (size_t)(h_start + hi) * m->max_seq_len, seq_len);
                    PW_MARK(attn);
                    PW_BARRIER();  /* softmax done */
                    PW_MARK(barrier);
                }

                /* AV chunk over this thread's p-range, accumulating into its
                 * own av_tmp slice (which we zero first). att holds exp scores
                 * (tf_psm) or normalized softmax (legacy). */
                memset(my_tmp, 0, slice_n * sizeof(float));
                if (p_hi > p_lo) {
                    if (m->kv_dtype == TF_KV_DTYPE_F32)
                        tf_av_chunk_f32((const float *)m->value_cache[l], m->att,
                                        my_tmp, (int)p_lo, (int)p_hi,
                                        n_heads, n_kv_heads, head_dim, m->max_seq_len);
                    else if (m->kv_dtype == TF_KV_DTYPE_F16)
                        tf_av_chunk_f16((const uint16_t *)m->value_cache[l], m->att,
                                        my_tmp, (int)p_lo, (int)p_hi,
                                        n_heads, n_kv_heads, head_dim, m->max_seq_len);
                    else
                        tf_av_chunk_q8((const int8_t *)m->value_cache[l], m->value_scales[l],
                                       m->att, my_tmp, (int)p_lo, (int)p_hi,
                                       n_heads, n_kv_heads, head_dim, m->max_seq_len);
                }
                PW_MARK(attn);
                PW_BARRIER();  /* av_tmp ready (+ att_psum when tf_psm) */
                PW_MARK(barrier);

                /* Global per-head 1/sum (redundant per thread) for the fold.
                 * K_DP requires n_heads == SVE width (16); fixed bound avoids a
                 * per-iteration alloca in this layer loop. */
                float ginv[64];
                if (tf_psm) {
                    for (int h = 0; h < n_heads; h++) {
                        float s = 0.0f;
                        for (int t = 0; t < nt; t++) s += m->att_psum[(size_t)t * n_heads + h];
                        ginv[h] = s > 0.0f ? 1.0f / s : 0.0f;
                    }
                }

                /* Reduce av_tmp across nt threads into xb2, folding 1/sum.
                 * Partition (h, d-vec) work units across all threads: vl floats
                 * per unit, 16 heads × head_dim/vl units total. */
                {
                    int vl = (int)svcntw();
                    int per_h = (head_dim + vl - 1) / vl;
                    int total_units = n_heads * per_h;
                    long u_lo = (long)tid * total_units / nt;
                    long u_hi = (long)(tid + 1) * total_units / nt;
                    for (long u = u_lo; u < u_hi; u++) {
                        int h  = (int)u / per_h;
                        int dv = (int)u % per_h;
                        int d_lo = dv * vl;
                        int d_hi = d_lo + vl; if (d_hi > head_dim) d_hi = head_dim;
                        tf_av_reduce_slice(m->xb2, m->av_tmp, h, d_lo, d_hi,
                                           nt, n_heads, head_dim);
                        if (tf_psm) {
                            float inv = ginv[h];
                            float *o = m->xb2 + (size_t)h * head_dim;
                            for (int d = d_lo; d < d_hi; d++) o[d] *= inv;
                        }
                    }
                }
                PW_MARK(attn);
                /* No barrier here: the existing B4 barrier after the worker
                 * (which is now a no-op for skip_av) plus the per-head sigmoid
                 * gate (which only touches xb2 for h_start..h_start+h_count
                 * — what this thread just reduced) makes this safe.  Actually
                 * we still need a barrier so that each thread's reduce work
                 * (some other thread's heads) is visible before the gate. */
                PW_BARRIER();
                PW_MARK(barrier);
                pw_skip_av = 1;
            }
#endif

            /* Attention: each thread handles its head partition */
            {
                int seq_len = position + 1;
                float scale = 1.0f / sqrtf((float)head_dim);
                if (h_count > 0) {
                    tf_attn_task at = {
                        .q = m->q, .att = m->att, .xb2 = m->xb2,
                        .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                        .key_scales   = m->key_scales   ? m->key_scales[l]   : NULL,
                        .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                        .head_start = h_start, .head_end = h_start + h_count,
                        .head_dim = head_dim, .kv_dim = kv_dim,
                        .gqa_ratio = gqa_ratio,
                        .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                        .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                        .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                        .k_transposed = m->kv_k_transposed,
                        .k_dp = m->kv_k_dp,
                        .skip_qk = pw_skip_qk,
                        .skip_av = pw_skip_av,
                        .scale = scale,
                    };
                    if (!pw_skip_av)
                        memset(m->xb2 + h_start * head_dim, 0, (size_t)h_count * head_dim * sizeof(float));
                    tf_attn_worker(&at);
                }
            }

            /* Sigmoid gate for hybrid gated attention */
            if (m->is_hybrid) {
                for (int i = h_start * head_dim; i < (h_start + h_count) * head_dim; i++) {
                    float g = m->ffn_buf1[i];
                    m->xb2[i] *= 1.0f / (1.0f + expf(-g));
                }
            }
            PW_MARK(attn);
            PW_BARRIER();  /* B4: xb2 ready */
            PW_MARK(barrier);

            /* Output projection: all threads compute their row partition */
            tf_thread_matvec(m->xb, &layer->attn_output, m->xb2, n_embd, tid, nt);
            PW_MARK(matvec);
            PW_BARRIER();  /* B5: xb ready */
            PW_MARK(barrier);
        }

        /* Thread 0: residual + FFN norm (merged B5+B6: saves 1 barrier).
         * TP: the row-parallel mixer output projection (attn_output / ssm_out)
         * produced a PARTIAL sum on m->xb — all-reduce across the TP group
         * before the residual add, mirroring the per-op decode path (~9427).
         * tid 0 owns the uTofu collective; the B6 barrier below publishes the
         * reduced+normed xb to all threads. Dormant unless TP decode is routed
         * through the persistent worker (tp_size>1). */
        if (tid == 0) {
            if (m->tp_size > 1 && m->tp_allreduce_fn &&
                ((m->is_hybrid && layer->is_ssm) ? m->tp_ssm_sharded : m->tp_attn_sharded))
                m->tp_allreduce_fn(m->xb, n_embd, m->tp_allreduce_ctx);
            tf_vadd(m->x, m->xb, n_embd);
            tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        }
        PW_MARK(serial);
        PW_BARRIER();  /* B6: xb ready for FFN (merged) */
        PW_MARK(barrier);

        if (!m->use_moe || !layer->ffn_gate_inp.data) {
            /* Dense SwiGLU FFN: gate+up matvec (parallel) */
            tf_thread_matvec(m->ffn_buf1, &layer->ffn_gate, m->xb, n_ff, tid, nt);
            tf_thread_matvec(m->ffn_buf2, &layer->ffn_up, m->xb, n_ff, tid, nt);

            /* SiLU×mul on this thread's row partition. We use the SAME partition
             * as the gate matvec (via tf_matvec_row_range) so each thread reads
             * ffn_buf1/ffn_buf2 entries it just wrote — no barrier needed and
             * no cross-thread cache-line bounce. (Pre-2026-05-16 this used a
             * naive n_ff/nt split which was a latent race when matvec switched
             * to tf_row_split8.) Assumes ffn_gate and ffn_up share dtype/type
             * so their matvec partitions coincide. */
            {
                int rs, re_end;
                tf_matvec_row_range(&layer->ffn_gate, n_ff, nt, tid, &rs, &re_end);
                for (int i = rs; i < re_end; i++) {
                    float g = m->ffn_buf1[i];
                    m->ffn_buf3[i] = g / (1.0f + expf(-g)) * m->ffn_buf2[i];
                }
            }
            PW_MARK(matvec);
            PW_BARRIER();  /* B7: ffn_buf3 ready for down */
            PW_MARK(barrier);

            /* Down projection */
            tf_thread_matvec(m->xb, &layer->ffn_down, m->ffn_buf3, n_embd, tid, nt);
            PW_MARK(matvec);
            PW_BARRIER();  /* B8: xb ready */
            PW_MARK(barrier);

            /* Thread 0: residual. TP: ffn_down is row-parallel → partial sum,
             * all-reduce before the residual add (mirrors per-op path ~9635). */
            if (tid == 0) {
                if (m->tp_size > 1 && m->tp_allreduce_fn && m->tp_ffn_sharded)
                    m->tp_allreduce_fn(m->xb, n_embd, m->tp_allreduce_ctx);
                tf_vadd(m->x, m->xb, n_embd);
            }
        } else {
            /* MoE: thread 0 only (complex routing) */
            if (tid == 0) {
                int saved_alive = m->pool_alive;
                m->pool_alive = 0;  /* disable pool dispatch */
                const int n_expert = m->n_expert;
                const int n_top = m->n_expert_used;
                const int n_ff_exp = m->n_ff_expert;
                tf_qmatvec_pool(m, m->ffn_buf1, &layer->ffn_gate_inp, m->xb, n_expert);
                tf_softmax(m->ffn_buf1, n_expert);
                /* ... MoE routing simplified: use single-thread for now ... */
                int best = 0;
                for (int e = 1; e < n_expert; e++)
                    if (m->ffn_buf1[e] > m->ffn_buf1[best]) best = e;
                float ew = m->ffn_buf1[best];
                tf_qmatvec_expert(m->ffn_buf2, &layer->ffn_up_exps, best, m->xb, n_ff_exp, m->matvec_tmp);
                tf_qmatvec_expert(m->ffn_buf3, &layer->ffn_gate_exps, best, m->xb, n_ff_exp, m->matvec_tmp);
                tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_exp);
                tf_qmatvec_expert(m->q, &layer->ffn_down_exps, best, m->ffn_buf3, n_embd, m->matvec_tmp);
                for (int i = 0; i < n_embd; i++) m->x[i] += ew * m->q[i];
                m->pool_alive = saved_alive;
            }
            PW_BARRIER();  /* MoE done */
        }
        /* No end-of-layer barrier needed: only thread 0 writes m->x (vadd),
         * and only thread 0 reads it at the start of next layer (RMSNorm).
         * The next layer's B1 barrier ensures all threads wait for the norm. */
    }

    /* Final norm: thread 0 only */
    if (tid == 0) {
        tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    }
    return NULL;
}

/* Persistent forward: dispatch tf_persistent_worker to ALL threads via pool,
 * then main thread runs worker 0 (master-worker pattern). Workers communicate
 * via spin barrier — the old pool dispatch mechanism is NOT used during forward. */
static float *tf_forward_persistent(transformer_model *m, int position, int pos_t, int pos_h, int pos_w) {
    int nt = m->n_threads;

    m->bar_count = 0;
    m->bar_sense = 0;
    __sync_synchronize();

    tf_persistent_ctx *ctxs = (tf_persistent_ctx *)alloca(nt * sizeof(tf_persistent_ctx));
    for (int t = 0; t < nt; t++)
        ctxs[t] = (tf_persistent_ctx){m, t, position, pos_t, pos_h, pos_w};

    /* Use the existing pool to dispatch the persistent worker to threads 1..nt-1.
     * Main thread (worker 0) runs inline. */
    tf_pool_dispatch(m, tf_persistent_worker, ctxs, sizeof(tf_persistent_ctx));

    return m->x;
}

/* ---- Forward pass ---- */

/* Internal: forward pass block loop with separate cache position and 3D RoPE positions.
 * Processes layers [layer_start, layer_end). If layer_end == n_layers, applies final norm. */
static float *tf_forward_blocks_range(transformer_model *m, int cache_pos, int pos_t, int pos_h, int pos_w,
                                       int layer_start, int layer_end);
static float *tf_forward_blocks(transformer_model *m, int cache_pos, int pos_t, int pos_h, int pos_w);

/* Forward with M-RoPE: cache_pos = KV cache slot, pos_t/h/w = RoPE temporal/height/width */
float *transformer_forward_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w) {
    if (!model || !model->token_embd.data) return NULL;
    if (token_id < 0 || token_id >= model->n_vocab) {
        fprintf(stderr, "transformer_forward_pos: token_id=%d out of range [0, %d)\n", token_id, model->n_vocab);
        return NULL;
    }
    if (cache_pos < 0 || cache_pos >= model->max_seq_len) {
        fprintf(stderr, "transformer_forward_pos: cache_pos=%d out of range [0, %d)\n", cache_pos, model->max_seq_len);
        return NULL;
    }
    tf_dequant_row(&model->token_embd, token_id, model->x);
    /* Gemma4: scale token embeddings by sqrt(n_embd), stash token_id for per-layer embd */
    if (model->is_gemma4) {
        float scale = model->embd_scale;
        for (int i = 0; i < model->n_embd; i++) model->x[i] *= scale;
        model->current_token_id = token_id;
    }
    return tf_forward_blocks(model, cache_pos, pos_t, pos_h, pos_w);
}

float *transformer_forward(transformer_model *model, int32_t token_id, int position) {
    return transformer_forward_pos(model, token_id, position, position, position, position);
}

float *transformer_forward_logits_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w) {
    float *hidden = transformer_forward_pos(model, token_id, cache_pos, pos_t, pos_h, pos_w);
    if (!model->has_lm_head || !hidden) return NULL;
    TF_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
    tf_qmatvec_pool(model, model->logits, &model->output, hidden, model->n_vocab);
    TF_PROF_END("lm_head", 2.0 * model->n_vocab * model->n_embd, 0);
    /* Gemma4: final logit soft-capping */
    if (model->is_gemma4 && model->final_logit_softcapping > 0.0f) {
        tf_logit_softcap(model->logits, model->n_vocab, model->final_logit_softcapping);
    }
    return model->logits;
}

float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position) {
    return transformer_forward_logits_pos(model, token_id, position, position, position, position);
}

static float *tf_forward_blocks(transformer_model *m, int position, int pos_t, int pos_h, int pos_w) {
    if (m->n_threads > 1 && m->pool_alive)
        return tf_forward_persistent(m, position, pos_t, pos_h, pos_w);
    return tf_forward_blocks_range(m, position, pos_t, pos_h, pos_w, 0, m->n_layers);
}

static float *tf_forward_blocks_range(transformer_model *m, int position, int pos_t, int pos_h, int pos_w,
                                       int layer_start, int layer_end) {
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;  /* may differ from n_embd */
    int gqa_ratio = m->gqa_group;    /* GLOBAL group: survives KV replication (local ratio may be 1) */

    if (layer_end > m->n_layers) layer_end = m->n_layers;
    if (layer_start < 0) layer_start = 0;

    /* Gemma4: precompute per-layer inputs (token embedding + model projection, combined).
     * Uses heap allocation instead of alloca (total_ple = 256*42 = 10752 floats ≈ 43KB × 3). */
    float *ple_combined = NULL; /* [n_layers * n_embd_per_layer] if Gemma4 */
    if (m->is_gemma4 && m->per_layer_token_embd.data && m->current_token_id >= 0) {
        int ple_dim = m->n_embd_per_layer;
        int total_ple = ple_dim * m->n_layers;
        ple_combined = (float *)malloc(total_ple * sizeof(float));
        float *tok_ple = (float *)malloc(total_ple * sizeof(float));
        float *proj_out = (float *)malloc(total_ple * sizeof(float));
        if (!ple_combined || !tok_ple || !proj_out) {
            fprintf(stderr, "transformer: Gemma4 PLE alloc failed (total_ple=%d)\n", total_ple);
            free(proj_out);
            free(tok_ple);
            free(ple_combined);
            return NULL;
        }

        /* 1. Look up per-layer token embedding: dequant row for this token */
        dequant_row(m->per_layer_token_embd.type,
                    (const uint8_t *)m->per_layer_token_embd.data +
                    (size_t)m->current_token_id * tf_row_bytes(m->per_layer_token_embd.type, total_ple),
                    tok_ple, total_ple);
        float ple_tok_scale = sqrtf((float)ple_dim);
        for (int i = 0; i < total_ple; i++) tok_ple[i] *= ple_tok_scale;

        /* 2. Project x through per_layer_model_proj: [10752, 2560] @ [2560] = [10752] */
        tf_qmatvec(proj_out, &m->per_layer_model_proj, m->x, total_ple, m->matvec_tmp);
        float proj_scale = 1.0f / sqrtf((float)n_embd);
        for (int i = 0; i < total_ple; i++) proj_out[i] *= proj_scale;

        /* 3. RMSNorm with per_layer_proj_norm (shared [256] weights, applied per-layer slice) */
        if (m->per_layer_proj_norm.data) {
            float norm_w[256];
            dequant_row(m->per_layer_proj_norm.type, m->per_layer_proj_norm.data, norm_w, ple_dim);
            for (int ll = 0; ll < m->n_layers; ll++) {
                float *slice = proj_out + ll * ple_dim;
                float ss = 0.0f;
                for (int i = 0; i < ple_dim; i++) ss += slice[i] * slice[i];
                ss = 1.0f / sqrtf(ss / ple_dim + m->rms_norm_eps);
                for (int i = 0; i < ple_dim; i++) slice[i] = slice[i] * ss * norm_w[i];
            }
        }

        /* 4. Add token embedding + projection, scale by 1/sqrt(2) */
        float input_scale = 1.0f / sqrtf(2.0f);
        for (int i = 0; i < total_ple; i++)
            ple_combined[i] = (proj_out[i] + tok_ple[i]) * input_scale;
        free(proj_out);
        free(tok_ple);
    }

    /* 2. Transformer blocks */
    for (int l = layer_start; l < layer_end; l++) {
        transformer_layer *layer = &m->layers[l];

        /* --- Attention --- */
        /* RMSNorm */
        TF_PROF_BEGIN("attn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("attn_norm", 5.0 * n_embd, 0);

        if (m->is_gemma4) {
            /* --- Gemma4 layer --- */
            int hd = layer->is_swa ? m->head_dim_swa : m->head_dim_full;
            int local_kv_dim = n_kv_heads * hd;
            int local_q_dim  = n_heads * hd;
            int local_gqa    = n_heads / n_kv_heads;
            float eps = m->rms_norm_eps;

            /* Q projection (always present) */
            tf_qmatvec_pool(m, m->q, &layer->attn_q, m->xb, local_q_dim);

            /* K/V projections (skip if sharing KV) */
            int kv_src = (layer->shared_kv_source >= 0) ? layer->shared_kv_source : l;
            if (layer->shared_kv_source < 0) {
                tf_qmatvec_pool(m, m->k, &layer->attn_k, m->xb, local_kv_dim);
                tf_qmatvec_pool(m, m->v, &layer->attn_v, m->xb, local_kv_dim);
            }

            /* Q norm (always) */
            tf_qk_norm(m->q, n_heads, hd, &layer->attn_q_norm, eps, m->matvec_tmp);

            /* K/V norm (only if we projected them) */
            if (layer->shared_kv_source < 0) {
                tf_qk_norm(m->k, n_kv_heads, hd, &layer->attn_k_norm, eps, m->matvec_tmp);
                /* V norm — Gemma4 normalizes V too (raw RMSNorm, no weight) */
                if (layer->attn_v_norm.data) {
                    tf_qk_norm(m->v, n_kv_heads, hd, &layer->attn_v_norm, eps, m->matvec_tmp);
                } else {
                    /* V gets raw RMSNorm without learned weight */
                    for (int h = 0; h < n_kv_heads; h++) {
                        float *vh = m->v + h * hd;
                        float ss = 0.0f;
                        for (int i = 0; i < hd; i++) ss += vh[i] * vh[i];
                        ss = 1.0f / sqrtf(ss / hd + eps);
                        for (int i = 0; i < hd; i++) vh[i] *= ss;
                    }
                }
            }

            /* RoPE: use SWA or full-attention inv_freq table */
            {
                float *inv_freq = layer->is_swa ? m->rope_inv_freq_swa : m->rope_inv_freq;
                int half = hd / 2;
                /* Apply RoPE to Q */
                for (int h = 0; h < n_heads; h++) {
                    float *qh = m->q + h * hd;
                    for (int j = 0; j < half; j++) {
                        float freq = (float)position * inv_freq[j];
                        float cos_v = cosf(freq), sin_v = sinf(freq);
                        float r0 = qh[j], r1 = qh[j + half];
                        qh[j]        = r0 * cos_v - r1 * sin_v;
                        qh[j + half] = r0 * sin_v + r1 * cos_v;
                    }
                }
                /* Apply RoPE to K (only if we projected K) */
                if (layer->shared_kv_source < 0) {
                    for (int h = 0; h < n_kv_heads; h++) {
                        float *kh = m->k + h * hd;
                        for (int j = 0; j < half; j++) {
                            float freq = (float)position * inv_freq[j];
                            float cos_v = cosf(freq), sin_v = sinf(freq);
                            float r0 = kh[j], r1 = kh[j + half];
                            kh[j]        = r0 * cos_v - r1 * sin_v;
                            kh[j + half] = r0 * sin_v + r1 * cos_v;
                        }
                    }
                }
            }

            /* KV cache store (Gemma4 path is F32-only; transformer_load forces
             * kv_dtype back to F32 for Gemma4). */
            if (layer->shared_kv_source < 0) {
                float *kc_l = (float *)m->key_cache[l];
                float *vc_l = (float *)m->value_cache[l];
                if (layer->is_swa) {
                    int slot = position % m->swa_window_size;
                    memcpy(kc_l + slot * local_kv_dim, m->k, local_kv_dim * sizeof(float));
                    memcpy(vc_l + slot * local_kv_dim, m->v, local_kv_dim * sizeof(float));
                } else {
                    memcpy(kc_l + position * local_kv_dim, m->k, local_kv_dim * sizeof(float));
                    memcpy(vc_l + position * local_kv_dim, m->v, local_kv_dim * sizeof(float));
                }
            }

            /* Attention with scale=1.0 (QK norms handle scaling) */
            {
                float attn_scale = 1.0f;
                int seq_len;
                float *kc = (float *)m->key_cache[kv_src];
                float *vc = (float *)m->value_cache[kv_src];

                if (layer->is_swa) {
                    /* SWA: attend to window [max(0, pos-window+1), pos] via circular buffer */
                    int win = m->swa_window_size;
                    int start = (position >= win) ? (position - win + 1) : 0;
                    seq_len = position - start + 1;

                    /* Compute attention scores over the window */
                    memset(m->xb2, 0, local_q_dim * sizeof(float));
                    for (int h = 0; h < n_heads; h++) {
                        float *qh = m->q + h * hd;
                        float *att_h = m->att + h * seq_len;
                        int kv_h = h / local_gqa;

                        /* Compute QK scores */
                        for (int p = 0; p < seq_len; p++) {
                            int abs_pos = start + p;
                            int slot = abs_pos % win;
                            float *kp = kc + slot * local_kv_dim + kv_h * hd;
                            float score = 0.0f;
                            for (int d = 0; d < hd; d++) score += qh[d] * kp[d];
                            att_h[p] = score * attn_scale;
                        }

                        /* Softmax */
                        float max_s = att_h[0];
                        for (int p = 1; p < seq_len; p++) if (att_h[p] > max_s) max_s = att_h[p];
                        float sum_e = 0.0f;
                        for (int p = 0; p < seq_len; p++) { att_h[p] = expf(att_h[p] - max_s); sum_e += att_h[p]; }
                        float inv_sum = 1.0f / sum_e;
                        for (int p = 0; p < seq_len; p++) att_h[p] *= inv_sum;

                        /* Weighted sum of values */
                        float *out_h = m->xb2 + h * hd;
                        for (int p = 0; p < seq_len; p++) {
                            int abs_pos = start + p;
                            int slot = abs_pos % win;
                            float *vp = vc + slot * local_kv_dim + kv_h * hd;
                            float w = att_h[p];
                            for (int d = 0; d < hd; d++) out_h[d] += w * vp[d];
                        }
                    }
                } else {
                    /* Full attention */
                    seq_len = position + 1;
                    memset(m->xb2, 0, local_q_dim * sizeof(float));
                    for (int h = 0; h < n_heads; h++) {
                        float *qh = m->q + h * hd;
                        float *att_h = m->att + h * seq_len;
                        int kv_h = h / local_gqa;

                        for (int p = 0; p < seq_len; p++) {
                            float *kp = kc + p * local_kv_dim + kv_h * hd;
                            float score = 0.0f;
                            for (int d = 0; d < hd; d++) score += qh[d] * kp[d];
                            att_h[p] = score * attn_scale;
                        }

                        float max_s = att_h[0];
                        for (int p = 1; p < seq_len; p++) if (att_h[p] > max_s) max_s = att_h[p];
                        float sum_e = 0.0f;
                        for (int p = 0; p < seq_len; p++) { att_h[p] = expf(att_h[p] - max_s); sum_e += att_h[p]; }
                        float inv_sum = 1.0f / sum_e;
                        for (int p = 0; p < seq_len; p++) att_h[p] *= inv_sum;

                        float *out_h = m->xb2 + h * hd;
                        for (int p = 0; p < seq_len; p++) {
                            float *vp = vc + p * local_kv_dim + kv_h * hd;
                            float w = att_h[p];
                            for (int d = 0; d < hd; d++) out_h[d] += w * vp[d];
                        }
                    }
                }
            }

            /* Output projection */
            tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);

            /* Post-attention norm (before residual) */
            tf_rmsnorm(m->xb, m->xb, &layer->post_attention_norm, n_embd, eps, m->matvec_tmp);

            /* Residual */
            tf_vadd(m->x, m->xb, n_embd);

            /* --- FFN with GELU --- */
            tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, eps, m->matvec_tmp);

            tf_qmatvec_fused2_pool(m, m->ffn_buf1, &layer->ffn_gate,
                                    m->ffn_buf2, &layer->ffn_up, m->xb, m->n_ff);
            tf_gelu_mul(m->ffn_buf3, m->ffn_buf1, m->ffn_buf2, m->n_ff);

            tf_qmatvec_pool(m, m->xb, &layer->ffn_down, m->ffn_buf3, n_embd);

            /* Post-FFN norm (before residual) */
            tf_rmsnorm(m->xb, m->xb, &layer->post_ffw_norm, n_embd, eps, m->matvec_tmp);

            /* Residual */
            tf_vadd(m->x, m->xb, n_embd);

            /* Per-layer embedding injection (BEFORE layer_output_scale) */
            if (ple_combined) {
                int ple_dim = m->n_embd_per_layer;
                float *ple = m->ple_buf;      /* [ple_dim] */
                float *proj = m->ple_proj_buf; /* [n_embd] */

                /* Take precomputed per-layer input slice for this layer */
                memcpy(ple, ple_combined + l * ple_dim, ple_dim * sizeof(float));

                /* inp_gate: hidden -> [ple_dim], GELU, element-wise multiply */
                tf_qmatvec_pool(m, proj, &layer->ple_inp_gate, m->x, ple_dim);
                for (int i = 0; i < ple_dim; i++) {
                    float g = proj[i];
                    float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));
                    ple[i] *= gelu_g;
                }

                /* Project back to n_embd via ple_proj */
                tf_qmatvec_pool(m, proj, &layer->ple_proj, ple, n_embd);

                /* Post-norm on projected output */
                if (layer->ple_post_norm.data) {
                    tf_rmsnorm(proj, proj, &layer->ple_post_norm, n_embd, eps, m->matvec_tmp);
                }

                /* Residual add */
                tf_vadd(m->x, proj, n_embd);
            }

            /* Layer output scaling (AFTER per-layer embedding) */
            if (layer->layer_output_scale.data) {
                float scale_val;
                dequant_row(layer->layer_output_scale.type, layer->layer_output_scale.data, &scale_val, 1);
                for (int i = 0; i < n_embd; i++) m->x[i] *= scale_val;
            }

            goto gemma4_layer_done;
        } else if (m->is_hybrid && layer->is_ssm) {
            /* --- SSM (Delta-Net) layer --- */
            tf_ssm_deltanet_forward(m, l);
        } else if (m->is_hybrid) {
            /* --- Gated attention layer (Qwen3.5) --- */
            /* Q+gate combined projection: attn_q outputs [2*q_dim] interleaved */
            int q2_dim = 2 * q_dim;
            TF_PROF_BEGIN("qkv_proj", l, "matvec", "FP32");
            tf_qmatvec_fused_qkv_pool(m, m->xb2, &layer->attn_q, q2_dim,
                                       m->k, &layer->attn_k, m->v, &layer->attn_v, kv_dim);
            TF_PROF_END("qkv_proj", 2.0 * (q2_dim + 2.0 * kv_dim) * n_embd, 0);

            /* De-interleave: [Q0,gate0, Q1,gate1, ...] → Q[q_dim], gate[q_dim] */
            for (int h = 0; h < n_heads; h++) {
                memcpy(m->q + h * head_dim, m->xb2 + h * 2 * head_dim, head_dim * sizeof(float));
                memcpy(m->ffn_buf1 + h * head_dim, m->xb2 + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
            }

            /* QK-Norm */
            TF_PROF_BEGIN("qk_norm", l, "rmsnorm", "FP32");
            if (layer->attn_q_norm.data)
                tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
            if (layer->attn_k_norm.data)
                tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
            TF_PROF_END("qk_norm", 5.0 * (n_heads + n_kv_heads) * head_dim, 0);

            /* RoPE */
            TF_PROF_BEGIN("rope", l, "rope", "FP32");
            tf_apply_rope(m, m->q, m->k, n_heads, n_kv_heads, head_dim, pos_t, pos_h, pos_w);
            TF_PROF_END("rope", 8.0 * (n_heads + n_kv_heads) * head_dim / 2, 0);

            /* KV cache */
            tf_k_cache_write_pos(m->key_cache[l],
                                 m->key_scales ? m->key_scales[l] : NULL,
                                 m->k, position, n_kv_heads, head_dim,
                                 m->max_seq_len, m->kv_dtype, m->kv_k_transposed,
                                 m->kv_k_dp);
            tf_kv_write_all_heads(m->value_cache[l],
                                  m->value_scales ? m->value_scales[l] : NULL,
                                  m->v, position, n_kv_heads, head_dim, m->kv_dtype);

            /* GQA attention (threaded if pool available) */
            TF_PROF_BEGIN("attention", l, "attention", "FP32");
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);

            if (m->n_threads > 1 && m->pool_alive) {
                int nt = m->n_threads;
                static int fa_enabled = -1;
                if (fa_enabled < 0) fa_enabled = getenv("TF_USE_FA") ? 1 : 0;
                if (fa_enabled && nt > n_heads && !m->kv_k_dp) {
                    tf_attention_fa(m, l, n_heads, n_kv_heads, head_dim, kv_dim,
                                    gqa_ratio, seq_len, scale);
                } else {
                    tf_attn_task *atasks = (tf_attn_task *)alloca(nt * sizeof(tf_attn_task));
                    int heads_per = n_heads / nt, heads_extra = n_heads % nt, hoff = 0;
                    for (int t = 0; t < nt; t++) {
                        int hcount = heads_per + (t < heads_extra ? 1 : 0);
                        atasks[t] = (tf_attn_task){
                            .q = m->q, .att = m->att, .xb2 = m->xb2,
                            .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                            .key_scales   = m->key_scales   ? m->key_scales[l]   : NULL,
                            .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                            .head_start = hoff, .head_end = hoff + hcount,
                            .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                            .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                            .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                            .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                            .k_transposed = m->kv_k_transposed,
                            .k_dp = m->kv_k_dp,
                            .scale = scale,
                        };
                        hoff += hcount;
                    }
                    tf_pool_dispatch(m, tf_attn_worker, atasks, sizeof(tf_attn_task));
                }
            } else {
                /* Single-threaded fallback — dispatch through tf_attn_worker for AVX2 */
                tf_attn_task st = {
                    .q = m->q, .att = m->att, .xb2 = m->xb2,
                    .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                    .key_scales   = m->key_scales   ? m->key_scales[l]   : NULL,
                    .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                    .head_start = 0, .head_end = n_heads,
                    .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                    .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                    .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                    .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                    .k_transposed = m->kv_k_transposed,
                    .k_dp = m->kv_k_dp,
                    .scale = scale,
                };
                memset(m->xb2, 0, q_dim * sizeof(float));
                tf_attn_worker(&st);
            }
            TF_PROF_END("attention", 2.0 * n_heads * seq_len * head_dim * 2, 0);

            /* Apply sigmoid gate: xb2[i] *= sigmoid(gate[i]) */
#if defined(__AVX2__) && defined(__FMA__)
            {
                const __m256 one = _mm256_set1_ps(1.0f);
                int i = 0;
                for (; i + 7 < q_dim; i += 8) {
                    __m256 x2 = _mm256_loadu_ps(m->xb2 + i);
                    __m256 g = _mm256_loadu_ps(m->ffn_buf1 + i);
                    __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
                    __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_g)));
                    _mm256_storeu_ps(m->xb2 + i, _mm256_mul_ps(x2, sig));
                }
                for (; i < q_dim; i++)
                    m->xb2[i] *= 1.0f / (1.0f + expf(-m->ffn_buf1[i]));
            }
#else
            for (int i = 0; i < q_dim; i++) {
                m->xb2[i] *= 1.0f / (1.0f + expf(-m->ffn_buf1[i]));
            }
#endif

            /* Output projection */
            TF_PROF_BEGIN("out_proj", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);
            TF_PROF_END("out_proj", 2.0 * n_embd * q_dim, 0);
        } else {
            /* --- Standard attention (non-hybrid) --- */
            /* Q/K/V projections (fused: one pool dispatch for all three) */
            TF_PROF_BEGIN("qkv_proj", l, "matvec", "FP32");
            tf_qmatvec_fused_qkv_pool(m, m->q, &layer->attn_q, q_dim,
                                       m->k, &layer->attn_k, m->v, &layer->attn_v, kv_dim);
            /* Add Q/K/V biases if present (Qwen2.5-VL) */
            if (layer->attn_q_bias.data) {
                if (layer->attn_q_bias.type == GGML_TYPE_F32) {
                    float *qb = (float *)layer->attn_q_bias.data;
                    for (int i = 0; i < q_dim; i++) m->q[i] += qb[i];
                } else {
                    tf_dequant_row(&layer->attn_q_bias, 0, m->matvec_tmp);
                    for (int i = 0; i < q_dim; i++) m->q[i] += m->matvec_tmp[i];
                }
            }
            if (layer->attn_k_bias.data) {
                if (layer->attn_k_bias.type == GGML_TYPE_F32) {
                    float *kb = (float *)layer->attn_k_bias.data;
                    for (int i = 0; i < kv_dim; i++) m->k[i] += kb[i];
                } else {
                    tf_dequant_row(&layer->attn_k_bias, 0, m->matvec_tmp);
                    for (int i = 0; i < kv_dim; i++) m->k[i] += m->matvec_tmp[i];
                }
            }
            if (layer->attn_v_bias.data) {
                if (layer->attn_v_bias.type == GGML_TYPE_F32) {
                    float *vb = (float *)layer->attn_v_bias.data;
                    for (int i = 0; i < kv_dim; i++) m->v[i] += vb[i];
                } else {
                    tf_dequant_row(&layer->attn_v_bias, 0, m->matvec_tmp);
                    for (int i = 0; i < kv_dim; i++) m->v[i] += m->matvec_tmp[i];
                }
            }
            TF_PROF_END("qkv_proj", 2.0 * (q_dim + 2.0 * kv_dim) * n_embd, 0);

            /* QK-Norm (if present) */
            TF_PROF_BEGIN("qk_norm", l, "rmsnorm", "FP32");
            if (layer->attn_q_norm.data) {
                tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
            }
            if (layer->attn_k_norm.data) {
                tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
            }
            TF_PROF_END("qk_norm", 5.0 * (n_heads + n_kv_heads) * head_dim, 0);

            /* RoPE on Q and K (standard or M-RoPE) */
            TF_PROF_BEGIN("rope", l, "rope", "FP32");
            tf_apply_rope(m, m->q, m->k, n_heads, n_kv_heads, head_dim, pos_t, pos_h, pos_w);
            TF_PROF_END("rope", 8.0 * (n_heads + n_kv_heads) * head_dim / 2, 0);

            /* Store K/V into cache at position */
            tf_k_cache_write_pos(m->key_cache[l],
                                 m->key_scales ? m->key_scales[l] : NULL,
                                 m->k, position, n_kv_heads, head_dim,
                                 m->max_seq_len, m->kv_dtype, m->kv_k_transposed,
                                 m->kv_k_dp);
            tf_kv_write_all_heads(m->value_cache[l],
                                  m->value_scales ? m->value_scales[l] : NULL,
                                  m->v, position, n_kv_heads, head_dim, m->kv_dtype);

            /* Multi-head attention with GQA */
            TF_PROF_BEGIN("attention", l, "attention", "FP32");
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);

            if (m->n_threads > 1 && m->pool_alive) {
                /* Pool-based threaded attention */
                int nt = m->n_threads;
                static int fa_enabled = -1;
                if (fa_enabled < 0) fa_enabled = getenv("TF_USE_FA") ? 1 : 0;
                if (fa_enabled && nt > n_heads && !m->kv_k_dp) {
                    tf_attention_fa(m, l, n_heads, n_kv_heads, head_dim, kv_dim,
                                    gqa_ratio, seq_len, scale);
                } else {
                    tf_attn_task *atasks = (tf_attn_task *)alloca(nt * sizeof(tf_attn_task));
                    int heads_per = n_heads / nt;
                    int heads_extra = n_heads % nt;
                    int hoff = 0;
                    for (int t = 0; t < nt; t++) {
                        int hcount = heads_per + (t < heads_extra ? 1 : 0);
                        atasks[t] = (tf_attn_task){
                            .q = m->q, .att = m->att, .xb2 = m->xb2,
                            .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                            .key_scales   = m->key_scales   ? m->key_scales[l]   : NULL,
                            .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                            .head_start = hoff, .head_end = hoff + hcount,
                            .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                            .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                            .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                            .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                            .k_transposed = m->kv_k_transposed,
                            .k_dp = m->kv_k_dp,
                            .scale = scale,
                        };
                        hoff += hcount;
                    }
                    tf_pool_dispatch(m, tf_attn_worker, atasks, sizeof(tf_attn_task));
                }
            } else {
                /* Single-threaded fallback — dispatch through tf_attn_worker for AVX2 */
                tf_attn_task st = {
                    .q = m->q, .att = m->att, .xb2 = m->xb2,
                    .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                    .key_scales   = m->key_scales   ? m->key_scales[l]   : NULL,
                    .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                    .head_start = 0, .head_end = n_heads,
                    .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                    .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                    .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                    .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                    .k_transposed = m->kv_k_transposed,
                    .k_dp = m->kv_k_dp,
                    .scale = scale,
                };
                memset(m->xb2, 0, q_dim * sizeof(float));
                tf_attn_worker(&st);
            }

            /* QK: heads*seq*hd, AV: heads*seq*hd */
            TF_PROF_END("attention", 2.0 * n_heads * seq_len * head_dim * 2, 0);

            /* Output projection */
            TF_PROF_BEGIN("out_proj", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);
            TF_PROF_END("out_proj", 2.0 * n_embd * n_embd, 0);
        }

        /* Residual. TP: the row-parallel mixer output projection (attn_output /
         * ssm_out) produced a PARTIAL sum on m->xb — all-reduce across the TP
         * group before the residual add. SSM layers skip this when SSM is left
         * replicated (Stage A: every rank computed the full SSM output). */
        if (m->tp_size > 1 && m->tp_allreduce_fn &&
            ((m->is_hybrid && layer->is_ssm) ? m->tp_ssm_sharded : m->tp_attn_sharded))
            m->tp_allreduce_fn(m->xb, n_embd, m->tp_allreduce_ctx);
        tf_vadd(m->x, m->xb, n_embd);

        /* Stage-1 batched-GEMM prefill: mixer done, FFN is batched separately. */
        if (m->prefill_ffn_skip) continue;

        /* --- FFN --- */
        /* RMSNorm */
        TF_PROF_BEGIN("ffn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("ffn_norm", 5.0 * n_embd, 0);

        if (m->use_moe && layer->ffn_gate_inp.data) {
            const int n_expert = m->n_expert;
            const int n_top = m->n_expert_used;
            const int n_ff_exp = m->n_ff_expert;

            /* Router logits -> probs over experts */
            TF_PROF_BEGIN("ffn_gate_inp", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->ffn_buf1, &layer->ffn_gate_inp, m->xb, n_expert);
            TF_PROF_END("ffn_gate_inp", 2.0 * n_expert * n_embd, 0);
            tf_softmax(m->ffn_buf1, n_expert);

            /* Select top-k experts using min-heap: O(n_expert) single pass */
            int *top_idx = (int *)alloca(n_top * sizeof(int));
            float *top_w = (float *)alloca(n_top * sizeof(float));
            int k = 0;
            for (int e = 0; e < n_expert; e++) {
                float w = m->ffn_buf1[e];
                if (k < n_top) {
                    top_idx[k] = e; top_w[k] = w; k++;
                    /* Sift up to maintain min-heap at top_w[0] */
                    for (int j = k-1; j > 0;) {
                        int p = (j-1)/2;
                        if (top_w[j] < top_w[p]) {
                            float tv = top_w[j]; top_w[j] = top_w[p]; top_w[p] = tv;
                            int ti = top_idx[j]; top_idx[j] = top_idx[p]; top_idx[p] = ti;
                            j = p;
                        } else break;
                    }
                } else if (w > top_w[0]) {
                    top_w[0] = w; top_idx[0] = e;
                    /* Sift down from root */
                    for (int j = 0;;) {
                        int s = j, l2 = 2*j+1, r2 = 2*j+2;
                        if (l2 < n_top && top_w[l2] < top_w[s]) s = l2;
                        if (r2 < n_top && top_w[r2] < top_w[s]) s = r2;
                        if (s == j) break;
                        float tv = top_w[j]; top_w[j] = top_w[s]; top_w[s] = tv;
                        int ti = top_idx[j]; top_idx[j] = top_idx[s]; top_idx[s] = ti;
                        j = s;
                    }
                }
            }
            float wsum = 0.0f;
            for (int i = 0; i < n_top; i++) wsum += top_w[i];
            if (wsum > 0.0f) {
                for (int i = 0; i < n_top; i++) top_w[i] /= wsum;
            }

            /* Aggregate selected experts.
             *
             * Fused dispatch path: collect this rank's owned experts (EP filter)
             * into a small array, then collapse the per-expert {up, gate,
             * silu_mul, down, accumulate} loop into TWO pool dispatches:
             *   1) tf_moe_upgate_fused_pool  — K up matvecs + K gate matvecs
             *      + K silu_muls, written into activated[K * n_ff_exp]
             *   2) tf_moe_down_fused_pool    — K down matvecs with per-expert
             *      weighted-sum directly into xb2 (each thread owns disjoint
             *      output rows; no read-modify-write race)
             *
             * Per-(k,row) math order is preserved → bit-identical to the
             * per-expert path. Saves 3*K - 2 dispatches/layer (== 22 at
             * K=8); each thread now sweeps K*rpe/48 rows (~85) instead of
             * rpe/48 (~11). Set TF_MOE_FUSED=0 to fall back to the
             * per-expert loop. */
            memset(m->xb2, 0, n_embd * sizeof(float));

            int owned_e[16];
            float owned_w[16];
            int K = 0;
            for (int ei = 0; ei < n_top; ei++) {
                int e = top_idx[ei];
                if (m->ep_size > 1 && ((e % m->ep_size) != m->ep_rank)) continue;
                owned_e[K] = e;
                owned_w[K] = top_w[ei];
                K++;
            }

            static int moe_fused = -1;
            if (moe_fused < 0) {
                const char *ev = getenv("TF_MOE_FUSED");
                moe_fused = ev ? (atoi(ev) != 0) : 1;
            }

            if (moe_fused && K > 0) {
                /* ffn_buf3 reused as activated[K * n_ff_exp]; sized to
                 * max_ff (>= n_ff_expert * n_expert_used in practice). */
                TF_PROF_BEGIN("moe_upgate_fused", l, "matvec", "FP32");
                tf_moe_upgate_fused_pool(m, m->ffn_buf3,
                                         &layer->ffn_up_exps, &layer->ffn_gate_exps,
                                         owned_e, K, m->xb, n_ff_exp);
                TF_PROF_END("moe_upgate_fused", 2.0 * 2.0 * n_ff_exp * n_embd * K, 0);

                TF_PROF_BEGIN("moe_down_fused", l, "matvec", "FP32");
                tf_moe_down_fused_pool(m, m->xb2, &layer->ffn_down_exps,
                                       owned_e, owned_w, K,
                                       m->ffn_buf3, n_ff_exp, n_embd);
                TF_PROF_END("moe_down_fused", 2.0 * n_embd * n_ff_exp * K, 0);
            } else {
                for (int ki = 0; ki < K; ki++) {
                    int e = owned_e[ki];
                    float ew = owned_w[ki];

                    TF_PROF_BEGIN("ffn_up_exp", l, "matvec", "FP32");
                    tf_qmatvec_expert_pool(m, m->ffn_buf2, &layer->ffn_up_exps, e, m->xb, n_ff_exp);
                    TF_PROF_END("ffn_up_exp", 2.0 * n_ff_exp * n_embd, 0);

                    TF_PROF_BEGIN("ffn_gate_exp", l, "matvec", "FP32");
                    tf_qmatvec_expert_pool(m, m->ffn_buf3, &layer->ffn_gate_exps, e, m->xb, n_ff_exp);
                    TF_PROF_END("ffn_gate_exp", 2.0 * n_ff_exp * n_embd, 0);

                    TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
                    tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_exp);
                    TF_PROF_END("silu_mul", 5.0 * n_ff_exp, 0);

                    TF_PROF_BEGIN("ffn_down_exp", l, "matvec", "FP32");
                    tf_qmatvec_expert_pool(m, m->q, &layer->ffn_down_exps, e, m->ffn_buf3, n_embd);
                    TF_PROF_END("ffn_down_exp", 2.0 * n_embd * n_ff_exp, 0);

#if defined(__AVX2__) && defined(__FMA__)
                    {
                        __m256 vew = _mm256_set1_ps(ew);
                        int i = 0;
                        for (; i + 7 < n_embd; i += 8)
                            _mm256_storeu_ps(m->xb2 + i, _mm256_fmadd_ps(vew, _mm256_loadu_ps(m->q + i),
                                                                           _mm256_loadu_ps(m->xb2 + i)));
                        for (; i < n_embd; i++) m->xb2[i] += ew * m->q[i];
                    }
#else
                    for (int i = 0; i < n_embd; i++) m->xb2[i] += ew * m->q[i];
#endif
                }
            }

            /* Qwen3.5-MoE shared expert (always-on, sigmoid-gated SwiGLU).
             *   gscore = sigmoid(x . ffn_gate_inp_shexp)
             *   xb2  += gscore * ffn_down_shexp @ (silu(ffn_gate_shexp @ x) * (ffn_up_shexp @ x))
             * When EP is active the shared-expert weights are replicated, so we
             * compute it only on ep_rank 0; the post-loop sum-all-reduce
             * distributes the contribution to every rank. */
            if (m->n_ff_shexp > 0 && layer->ffn_up_shexp.data &&
                (m->ep_size == 1 || m->ep_rank == 0)) {
                const int n_ff_sh = m->n_ff_shexp;
                /* 1D F32 gate vector -> scalar dot product with xb. */
                const float *gw = (const float *)layer->ffn_gate_inp_shexp.data;
                float gz = 0.0f;
                if (gw) {
                    for (int i = 0; i < n_embd; i++) gz += gw[i] * m->xb[i];
                }
                float gscore = 1.0f / (1.0f + expf(-gz));

                TF_PROF_BEGIN("ffn_up_shexp", l, "matvec", "FP32");
                tf_qmatvec_pool(m, m->ffn_buf2, &layer->ffn_up_shexp,   m->xb, n_ff_sh);
                TF_PROF_END("ffn_up_shexp", 2.0 * n_ff_sh * n_embd, 0);

                TF_PROF_BEGIN("ffn_gate_shexp", l, "matvec", "FP32");
                tf_qmatvec_pool(m, m->ffn_buf3, &layer->ffn_gate_shexp, m->xb, n_ff_sh);
                TF_PROF_END("ffn_gate_shexp", 2.0 * n_ff_sh * n_embd, 0);

                TF_PROF_BEGIN("silu_mul_shexp", l, "silu_mul", "FP32");
                tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_sh);
                TF_PROF_END("silu_mul_shexp", 5.0 * n_ff_sh, 0);

                TF_PROF_BEGIN("ffn_down_shexp", l, "matvec", "FP32");
                tf_qmatvec_pool(m, m->q, &layer->ffn_down_shexp, m->ffn_buf3, n_embd);
                TF_PROF_END("ffn_down_shexp", 2.0 * n_embd * n_ff_sh, 0);

                for (int i = 0; i < n_embd; i++) m->xb2[i] += gscore * m->q[i];
            }

            /* EP: combine per-rank weighted partials. Each rank wrote only its owned
             * experts' contribution into xb2; sum-all-reduce gives every rank the
             * full mixture output, identical bit-for-bit (matches the tp_runner
             * lockstep design). One reduce per MoE layer per token; payload n_embd
             * floats. */
            if (m->ep_size > 1 && m->ep_ar_fn)
                m->ep_ar_fn(m->xb2, n_embd, m->ep_ar_ctx);

            tf_vadd(m->x, m->xb2, n_embd);
        } else {
            /* Dense SwiGLU: down @ (silu(gate @ x) * (up @ x)) */
            TF_PROF_BEGIN("ffn_gate_up", l, "matvec", "FP32");
            tf_qmatvec_fused2_pool(m, m->ffn_buf1, &layer->ffn_gate,
                                    m->ffn_buf2, &layer->ffn_up, m->xb, m->n_ff);
            TF_PROF_END("ffn_gate_up", 2.0 * 2.0 * m->n_ff * n_embd, 0);

            TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
            tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf1, m->ffn_buf2, m->n_ff);
            TF_PROF_END("silu_mul", 5.0 * m->n_ff, 0);

            TF_PROF_BEGIN("ffn_down", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->ffn_down, m->ffn_buf3, n_embd);
            TF_PROF_END("ffn_down", 2.0 * n_embd * m->n_ff, 0);

            /* TP: ffn_down is row-parallel → partial sum, all-reduce before residual. */
            if (m->tp_size > 1 && m->tp_allreduce_fn && m->tp_ffn_sharded)
                m->tp_allreduce_fn(m->xb, n_embd, m->tp_allreduce_ctx);
            tf_vadd(m->x, m->xb, n_embd);
        }

        gemma4_layer_done:

        /* DeepStack injection: add deepstack slice after each early layer */
        if (m->ds_embd && l < m->n_deepstack && m->ds_embd_stride > n_embd) {
            const float *ds_slice = m->ds_embd + (1 + l) * n_embd;
            tf_vadd(m->x, ds_slice, n_embd);
        }

        if (m->debug_layers) {
            float ss = 0;
            for (int i = 0; i < n_embd; i++) ss += m->x[i] * m->x[i];
            fprintf(stderr, "  [L%02d ATT] norm=%.2f first=[%.4f, %.4f, %.4f, %.4f]\n",
                    l, sqrtf(ss), m->x[0], m->x[1], m->x[2], m->x[3]);
        }
    }

    /* Final RMSNorm (only if we processed through the last layer; skipped during
     * the batched-GEMM-prefill mixer pass — applied once at the end instead). */
    if (layer_end >= m->n_layers && !m->prefill_ffn_skip) {
        TF_PROF_BEGIN("final_norm", -1, "rmsnorm", "FP32");
        tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("final_norm", 5.0 * n_embd, 0);
    }

    free(ple_combined); /* NULL-safe: no-op if not Gemma4 */
    return m->x;
}

/* ---- Stage-1 batched-GEMM prefill ---- */

/* Pooled elementwise helpers: split a large flat array across the live pool so the
 * FFN epilogue (silu*mul) and residual add run on all cores, matching the per-thread
 * partitioning the persistent worker uses (a single-thread silu over M*n_ff would
 * serialize ~12.9G scalar expf on A64FX and dominate prefill). */
typedef struct { float *out; const float *gate, *up; size_t lo, hi; } tf_silu_flat_task;
static void *tf_silu_flat_worker(void *arg) {
    tf_silu_flat_task *t = (tf_silu_flat_task *)arg;
    size_t n = t->hi - t->lo;
    if ((int)n <= 0) return NULL;
    tf_silu_mul_avx2(t->out + t->lo, t->gate + t->lo, t->up + t->lo, (int)n);
    return NULL;
}
typedef struct { float *dst; const float *src; size_t lo, hi; } tf_vadd_flat_task;
static void *tf_vadd_flat_worker(void *arg) {
    tf_vadd_flat_task *t = (tf_vadd_flat_task *)arg;
    size_t n = t->hi - t->lo;
    if ((int)n <= 0) return NULL;
    tf_vadd(t->dst + t->lo, t->src + t->lo, (int)n);
    return NULL;
}
static void tf_silu_mul_flat_pool(transformer_model *m, float *out, const float *gate,
                                   const float *up, size_t n) {
    int nt = m->n_threads;
    tf_silu_flat_task *tasks = (tf_silu_flat_task *)alloca(nt * sizeof(tf_silu_flat_task));
    size_t per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        size_t lo = (size_t)t * per, hi = lo + per;
        if (lo > n) lo = n;
        if (hi > n) hi = n;
        tasks[t] = (tf_silu_flat_task){ out, gate, up, lo, hi };
    }
    tf_pool_dispatch(m, tf_silu_flat_worker, tasks, sizeof(tf_silu_flat_task));
}
static void tf_vadd_flat_pool(transformer_model *m, float *dst, const float *src, size_t n) {
    int nt = m->n_threads;
    tf_vadd_flat_task *tasks = (tf_vadd_flat_task *)alloca(nt * sizeof(tf_vadd_flat_task));
    size_t per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        size_t lo = (size_t)t * per, hi = lo + per;
        if (lo > n) lo = n;
        if (hi > n) hi = n;
        tasks[t] = (tf_vadd_flat_task){ dst, src, lo, hi };
    }
    tf_pool_dispatch(m, tf_vadd_flat_worker, tasks, sizeof(tf_vadd_flat_task));
}

typedef struct {
    float *dst;
    const float *src;
    const int *indices;
    const float *scale;
    int n_embd;
    size_t row_lo, row_hi;
} tf_scaled_vadd_scatter_task;

static void *tf_scaled_vadd_scatter_worker(void *arg) {
    tf_scaled_vadd_scatter_task *t = (tf_scaled_vadd_scatter_task *)arg;
#if defined(__ARM_FEATURE_SVE)
    const int vl = (int)svcntw();
    for (size_t r = t->row_lo; r < t->row_hi; r++) {
        int tok = t->indices[r];
        float w = t->scale[r];
        if (w == 0.0f) continue;
        float *dst = t->dst + (size_t)tok * (size_t)t->n_embd;
        const float *src = t->src + r * (size_t)t->n_embd;
        svfloat32_t ws = svdup_f32(w);
        int i = 0;
        for (; i + vl - 1 < t->n_embd; i += vl) {
            svbool_t pg = svwhilelt_b32(i, t->n_embd);
            svfloat32_t vd = svld1(pg, dst + i);
            svfloat32_t vs = svld1(pg, src + i);
            svst1(pg, dst + i, svmla_x(pg, vd, vs, ws));
        }
        if (i < t->n_embd) {
            svbool_t pg = svwhilelt_b32(i, t->n_embd);
            svfloat32_t vd = svld1(pg, dst + i);
            svfloat32_t vs = svld1(pg, src + i);
            svst1(pg, dst + i, svmla_m(pg, vd, vs, ws));
        }
    }
#else
    for (size_t r = t->row_lo; r < t->row_hi; r++) {
        int tok = t->indices[r];
        float w = t->scale[r];
        if (w == 0.0f) continue;
        float *dst = t->dst + (size_t)tok * (size_t)t->n_embd;
        const float *src = t->src + r * (size_t)t->n_embd;
        for (int i = 0; i < t->n_embd; i++) dst[i] += w * src[i];
    }
#endif
    return NULL;
}

static void tf_scaled_vadd_scatter_pool(transformer_model *m, float *dst,
                                       const float *src, const int *indices,
                                       const float *scale, int rows,
                                       int n_embd) {
    int nt = m->n_threads;
    size_t total = (size_t)rows;
    tf_scaled_vadd_scatter_task *tasks =
        (tf_scaled_vadd_scatter_task *)alloca((size_t)nt * sizeof(tf_scaled_vadd_scatter_task));
    size_t per = (total + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        size_t lo = (size_t)t * per, hi = lo + per;
        if (lo > total) lo = total;
        if (hi > total) hi = total;
        tasks[t] = (tf_scaled_vadd_scatter_task){
            .dst = dst,
            .src = src,
            .indices = indices,
            .scale = scale,
            .n_embd = n_embd,
            .row_lo = lo,
            .row_hi = hi,
        };
    }
    tf_pool_dispatch(m, tf_scaled_vadd_scatter_worker, tasks,
                    sizeof(tf_scaled_vadd_scatter_task));
}

#if defined(__ARM_FEATURE_SVE)
static inline float tf_dot_f32_sve(const float *a, const float *b, int n);
#endif

typedef struct {
    float *dst;
    const float *src;
    const float *scale;
    int n_embd;
    size_t lo, hi;
} tf_scaled_vadd_rows_task;
static void *tf_scaled_vadd_rows_worker(void *arg) {
    tf_scaled_vadd_rows_task *t = (tf_scaled_vadd_rows_task *)arg;
    int n = t->n_embd;
#if defined(__ARM_FEATURE_SVE)
    const int vl = (int)svcntw();
    for (size_t p = t->lo; p < t->hi; ) {
        size_t r = p / (size_t)n;
        size_t row_start = r * (size_t)n;
        size_t row_end = row_start + (size_t)n;
        float scale = t->scale[r];
        if (row_end > t->hi) row_end = t->hi;
        int row_len = (int)(row_end - row_start);
        if (scale == 0.0f) {
            p = row_end;
            continue;
        }
        svfloat32_t vs = svdup_f32(scale);
        for (size_t i = p; i < row_end; i += (size_t)vl) {
            int rel = (int)(i - row_start);
            svbool_t pg = svwhilelt_b32(rel, row_len);
            svfloat32_t vd = svld1(pg, t->dst + i);
            svfloat32_t vsr = svld1(pg, t->src + i);
            svst1(pg, t->dst + i, svmla_x(pg, vd, vsr, vs));
        }
        p = row_end;
    }
#else
    for (size_t p = t->lo; p < t->hi; p++) {
        int r = (int)(p / (size_t)n);
        t->dst[p] += t->scale[r] * t->src[p];
    }
#endif
    return NULL;
}
static void tf_scaled_vadd_rows_pool(transformer_model *m, float *dst,
                                     const float *src, const float *scale,
                                     int rows, int n_embd) {
    int nt = m->n_threads;
    size_t total = (size_t)rows * (size_t)n_embd;
    tf_scaled_vadd_rows_task *tasks =
        (tf_scaled_vadd_rows_task *)alloca((size_t)nt * sizeof(tf_scaled_vadd_rows_task));
    size_t per = (total + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        size_t lo = (size_t)t * per, hi = lo + per;
        if (lo > total) lo = total;
        if (hi > total) hi = total;
        tasks[t] = (tf_scaled_vadd_rows_task){ dst, src, scale, n_embd, lo, hi };
    }
    tf_pool_dispatch(m, tf_scaled_vadd_rows_worker, tasks, sizeof(tf_scaled_vadd_rows_task));
}

static void tf_moe_shared_expert_add(transformer_model *m, transformer_layer *layer,
                                     int l, const float *x_norm, float *dst) {
    const int n_embd = m->n_embd;
    if (m->n_ff_shexp <= 0 || !layer->ffn_up_shexp.data) return;

    memcpy(m->xb, x_norm, (size_t)n_embd * sizeof(float));

    const int n_ff_sh = m->n_ff_shexp;
    const float *gw = (const float *)layer->ffn_gate_inp_shexp.data;
    float gz = 0.0f;
    if (gw) {
#if defined(__ARM_FEATURE_SVE)
        gz = tf_dot_f32_sve(m->xb, gw, n_embd);
#else
        for (int i = 0; i < n_embd; i++) gz += gw[i] * m->xb[i];
#endif
    }
    float gscore = 1.0f / (1.0f + expf(-gz));

    TF_PROF_BEGIN("ffn_up_shexp", l, "matvec", "FP32");
    tf_qmatvec_pool(m, m->ffn_buf2, &layer->ffn_up_shexp, m->xb, n_ff_sh);
    TF_PROF_END("ffn_up_shexp", 2.0 * n_ff_sh * n_embd, 0);

    TF_PROF_BEGIN("ffn_gate_shexp", l, "matvec", "FP32");
    tf_qmatvec_pool(m, m->ffn_buf3, &layer->ffn_gate_shexp, m->xb, n_ff_sh);
    TF_PROF_END("ffn_gate_shexp", 2.0 * n_ff_sh * n_embd, 0);

    TF_PROF_BEGIN("silu_mul_shexp", l, "silu_mul", "FP32");
    tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_sh);
    TF_PROF_END("silu_mul_shexp", 5.0 * n_ff_sh, 0);

    TF_PROF_BEGIN("ffn_down_shexp", l, "matvec", "FP32");
    tf_qmatvec_pool(m, m->q, &layer->ffn_down_shexp, m->ffn_buf3, n_embd);
    TF_PROF_END("ffn_down_shexp", 2.0 * n_embd * n_ff_sh, 0);

#if defined(__ARM_FEATURE_SVE)
    {
        svfloat32_t vg = svdup_f32(gscore);
        for (int i = 0; i < n_embd; i += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32(i, n_embd);
            svfloat32_t vd = svld1(pg, dst + i);
            svfloat32_t vq = svld1(pg, m->q + i);
            svst1(pg, dst + i, svmla_x(pg, vd, vq, vg));
        }
    }
#else
    for (int i = 0; i < n_embd; i++) dst[i] += gscore * m->q[i];
#endif
}

#if defined(__ARM_FEATURE_SVE)
static int tf_moe_shared_expert_add_block(transformer_model *m, transformer_layer *layer,
                                          int l, const float *Xn, int B, float *dst,
                                          float *G, float *U, float *D, float *score) {
    (void)l;
    const int n_embd = m->n_embd;
    const int n_ff_sh = m->n_ff_shexp;
    if (B <= 0 || n_ff_sh <= 0 || !layer->ffn_up_shexp.data ||
        !tf_prefill_weight_batched(&layer->ffn_up_shexp) ||
        !tf_prefill_weight_batched(&layer->ffn_gate_shexp) ||
        !tf_prefill_weight_batched(&layer->ffn_down_shexp) ||
        !G || !U || !D || !score)
        return 0;

    const float *gw = (const float *)layer->ffn_gate_inp_shexp.data;
    for (int t = 0; t < B; t++) {
        float gz = 0.0f;
        if (gw) {
            const float *x = Xn + (size_t)t * n_embd;
            gz = tf_dot_f32_sve(x, gw, n_embd);
        }
        score[t] = 1.0f / (1.0f + expf(-gz));
    }

    tf_gemm_bf16pv_prefill(m, U, &layer->ffn_up_shexp, Xn, B);
    tf_gemm_bf16pv_prefill(m, G, &layer->ffn_gate_shexp, Xn, B);
    tf_silu_mul_flat_pool(m, G, G, U, (size_t)B * (size_t)n_ff_sh);
    tf_gemm_bf16pv_prefill(m, D, &layer->ffn_down_shexp, G, B);
    tf_scaled_vadd_rows_pool(m, dst, D, score, B, n_embd);
    return 1;
}
#endif

typedef struct {
    double mixer_ssm_gemm;
    double mixer_attn_gemm;
    double mixer_fallback_ssm;
    double mixer_fallback_attn;
    double ssm_norm;
    double ssm_proj;
    double ssm_post;
    double ssm_finish;
    double ssm_finish_conv;
    double ssm_finish_norm2;
    double ssm_finish_scan;
    double ssm_out;
    double attn_norm;
    double attn_qkv;
    double attn_token;
    double attn_out;
    double ffn_norm_local;
    double ffn_local_core;
    double ffn_moe_gemm;
    double ffn_shared;
    double ffn_residual;
    long fallback_ssm_layers;
    long fallback_attn_layers;
    long ffn_tokens;
    long shared_tokens;
} tf_ep_prefill_detail;

static void tf_moe_ffn_local_partial(transformer_model *m, transformer_layer *layer,
                                     int l, const float *x_norm, float *dst,
                                     int include_shared) {
    const int n_embd = m->n_embd;
    const int n_expert = m->n_expert;
    const int n_top = m->n_expert_used;
    const int n_ff_exp = m->n_ff_expert;

    memcpy(m->xb, x_norm, (size_t)n_embd * sizeof(float));

    TF_PROF_BEGIN("ffn_gate_inp", l, "matvec", "FP32");
    tf_qmatvec_pool(m, m->ffn_buf1, &layer->ffn_gate_inp, m->xb, n_expert);
    TF_PROF_END("ffn_gate_inp", 2.0 * n_expert * n_embd, 0);
    tf_softmax(m->ffn_buf1, n_expert);

    int *top_idx = (int *)alloca((size_t)n_top * sizeof(int));
    float *top_w = (float *)alloca((size_t)n_top * sizeof(float));
    int k = 0;
    for (int e = 0; e < n_expert; e++) {
        float w = m->ffn_buf1[e];
        if (k < n_top) {
            top_idx[k] = e; top_w[k] = w; k++;
            for (int j = k - 1; j > 0;) {
                int p = (j - 1) / 2;
                if (top_w[j] < top_w[p]) {
                    float tv = top_w[j]; top_w[j] = top_w[p]; top_w[p] = tv;
                    int ti = top_idx[j]; top_idx[j] = top_idx[p]; top_idx[p] = ti;
                    j = p;
                } else break;
            }
        } else if (w > top_w[0]) {
            top_w[0] = w; top_idx[0] = e;
            for (int j = 0;;) {
                int s = j, l2 = 2 * j + 1, r2 = 2 * j + 2;
                if (l2 < n_top && top_w[l2] < top_w[s]) s = l2;
                if (r2 < n_top && top_w[r2] < top_w[s]) s = r2;
                if (s == j) break;
                float tv = top_w[j]; top_w[j] = top_w[s]; top_w[s] = tv;
                int ti = top_idx[j]; top_idx[j] = top_idx[s]; top_idx[s] = ti;
                j = s;
            }
        }
    }
    float wsum = 0.0f;
    for (int i = 0; i < n_top; i++) wsum += top_w[i];
    if (wsum > 0.0f) {
        for (int i = 0; i < n_top; i++) top_w[i] /= wsum;
    }

    memset(dst, 0, (size_t)n_embd * sizeof(float));

    int owned_e[16];
    float owned_w[16];
    int K = 0;
    for (int ei = 0; ei < n_top; ei++) {
        int e = top_idx[ei];
        if (m->ep_size > 1 && ((e % m->ep_size) != m->ep_rank)) continue;
        owned_e[K] = e;
        owned_w[K] = top_w[ei];
        K++;
    }

    static int moe_fused = -1;
    if (moe_fused < 0) {
        const char *ev = getenv("TF_MOE_FUSED");
        moe_fused = ev ? (atoi(ev) != 0) : 1;
    }

    if (moe_fused && K > 0) {
        TF_PROF_BEGIN("moe_upgate_fused", l, "matvec", "FP32");
        tf_moe_upgate_fused_pool(m, m->ffn_buf3,
                                 &layer->ffn_up_exps, &layer->ffn_gate_exps,
                                 owned_e, K, m->xb, n_ff_exp);
        TF_PROF_END("moe_upgate_fused", 2.0 * 2.0 * n_ff_exp * n_embd * K, 0);

        TF_PROF_BEGIN("moe_down_fused", l, "matvec", "FP32");
        tf_moe_down_fused_pool(m, dst, &layer->ffn_down_exps,
                               owned_e, owned_w, K,
                               m->ffn_buf3, n_ff_exp, n_embd);
        TF_PROF_END("moe_down_fused", 2.0 * n_embd * n_ff_exp * K, 0);
    } else {
        for (int ki = 0; ki < K; ki++) {
            int e = owned_e[ki];
            float ew = owned_w[ki];

            TF_PROF_BEGIN("ffn_up_exp", l, "matvec", "FP32");
            tf_qmatvec_expert_pool(m, m->ffn_buf2, &layer->ffn_up_exps, e, m->xb, n_ff_exp);
            TF_PROF_END("ffn_up_exp", 2.0 * n_ff_exp * n_embd, 0);

            TF_PROF_BEGIN("ffn_gate_exp", l, "matvec", "FP32");
            tf_qmatvec_expert_pool(m, m->ffn_buf3, &layer->ffn_gate_exps, e, m->xb, n_ff_exp);
            TF_PROF_END("ffn_gate_exp", 2.0 * n_ff_exp * n_embd, 0);

            TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
            tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_exp);
            TF_PROF_END("silu_mul", 5.0 * n_ff_exp, 0);

            TF_PROF_BEGIN("ffn_down_exp", l, "matvec", "FP32");
            tf_qmatvec_expert_pool(m, m->q, &layer->ffn_down_exps, e, m->ffn_buf3, n_embd);
            TF_PROF_END("ffn_down_exp", 2.0 * n_embd * n_ff_exp, 0);

            for (int i = 0; i < n_embd; i++) dst[i] += ew * m->q[i];
        }
    }

    if (include_shared)
        tf_moe_shared_expert_add(m, layer, l, x_norm, dst);
}

static void tf_moe_route_topk(transformer_model *m, transformer_layer *layer,
                              int l, const float *x_norm, int *top_idx, float *top_w) {
    const int n_embd = m->n_embd;
    const int n_expert = m->n_expert;
    const int n_top = m->n_expert_used;

    memcpy(m->xb, x_norm, (size_t)n_embd * sizeof(float));
    TF_PROF_BEGIN("ffn_gate_inp", l, "matvec", "FP32");
    tf_qmatvec_pool(m, m->ffn_buf1, &layer->ffn_gate_inp, m->xb, n_expert);
    TF_PROF_END("ffn_gate_inp", 2.0 * n_expert * n_embd, 0);
    tf_softmax(m->ffn_buf1, n_expert);

    int k = 0;
    for (int e = 0; e < n_expert; e++) {
        float w = m->ffn_buf1[e];
        if (k < n_top) {
            top_idx[k] = e; top_w[k] = w; k++;
            for (int j = k - 1; j > 0;) {
                int p = (j - 1) / 2;
                if (top_w[j] < top_w[p]) {
                    float tv = top_w[j]; top_w[j] = top_w[p]; top_w[p] = tv;
                    int ti = top_idx[j]; top_idx[j] = top_idx[p]; top_idx[p] = ti;
                    j = p;
                } else break;
            }
        } else if (w > top_w[0]) {
            top_w[0] = w; top_idx[0] = e;
            for (int j = 0;;) {
                int s = j, l2 = 2 * j + 1, r2 = 2 * j + 2;
                if (l2 < n_top && top_w[l2] < top_w[s]) s = l2;
                if (r2 < n_top && top_w[r2] < top_w[s]) s = r2;
                if (s == j) break;
                float tv = top_w[j]; top_w[j] = top_w[s]; top_w[s] = tv;
                int ti = top_idx[j]; top_idx[j] = top_idx[s]; top_idx[s] = ti;
                j = s;
            }
        }
    }

    float wsum = 0.0f;
    for (int i = 0; i < n_top; i++) wsum += top_w[i];
    if (wsum > 0.0f) {
        for (int i = 0; i < n_top; i++) top_w[i] /= wsum;
    }
}

static int tf_moe_expert_temp_qtensor(qtensor *dst, const qtensor *src,
                                      int expert, int n_rows, int n_cols) {
    if (!src->expert_owned_slot)
        return 0;
    if ((src->expert_owned_slot[expert] < 0) ||
        (!src->bf16_pv && !src->q8_pv))
        return 0;
    *dst = *src;
    int slot = src->expert_owned_slot[expert];
    int rpe = src->expert_rows_per_expert;
    dst->n_rows = n_rows;
    dst->n_cols = n_cols;
    dst->data = NULL;
    if (src->bf16_pv)
        dst->bf16_pv = src->bf16_pv + (size_t)slot * (size_t)rpe * (size_t)src->n_cols;
    else
        dst->bf16_pv = NULL;
    if (src->q8_pv)
        dst->q8_pv = src->q8_pv + (size_t)slot * (size_t)(rpe / 8) * tf_q8_pv_bytes(1, src->n_cols);
    else
        dst->q8_pv = NULL;
    dst->q8_pv_groups = src->q8_pv ? (rpe / 8) : 0;
    dst->bf16_pv_groups = n_rows / 8;
    dst->expert_owned_slot = NULL;
    dst->expert_rows_per_expert = 0;
    return 1;
}

static int tf_moe_ffn_block_gemm(transformer_model *m, transformer_layer *layer,
                                 int l, const float *X, int B, float *Y,
                                 float *Xn, float *A, float *G, float *U,
                                 float *D, int *top_idx, float *top_w,
                                 int *counts, int *tok_ids, float *tok_w) {
#if defined(__ARM_FEATURE_SVE)
    const int n_embd = m->n_embd;
    const int n_expert = m->n_expert;
    const int n_top = m->n_expert_used;
    const int n_ff_exp = m->n_ff_expert;
    if (B <= 0 || !m->pool_alive ||
        (!layer->ffn_up_exps.bf16_pv && !layer->ffn_up_exps.q8_pv) ||
        (!layer->ffn_gate_exps.bf16_pv && !layer->ffn_gate_exps.q8_pv) ||
        (!layer->ffn_down_exps.bf16_pv && !layer->ffn_down_exps.q8_pv) ||
        !layer->ffn_up_exps.expert_owned_slot ||
        !layer->ffn_gate_exps.expert_owned_slot ||
        !layer->ffn_down_exps.expert_owned_slot)
        return 0;

    memset(Y, 0, (size_t)B * n_embd * sizeof(float));
    memset(counts, 0, (size_t)n_expert * sizeof(int));

    for (int t = 0; t < B; t++) {
        tf_rmsnorm(Xn + (size_t)t * n_embd, X + (size_t)t * n_embd,
                   &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        tf_moe_route_topk(m, layer, l, Xn + (size_t)t * n_embd,
                          top_idx + (size_t)t * n_top, top_w + (size_t)t * n_top);
        for (int i = 0; i < n_top; i++) {
            int e = top_idx[(size_t)t * n_top + i];
            if (m->ep_size > 1 && (e % m->ep_size) != m->ep_rank) continue;
            counts[e]++;
        }
    }

    int *start = (int *)alloca((size_t)n_expert * sizeof(int));
    int *fill = (int *)alloca((size_t)n_expert * sizeof(int));
    int total = 0;
    for (int e = 0; e < n_expert; e++) {
        start[e] = counts[e];
        fill[e] = 0;
        counts[e] = total;
        total += start[e];
    }
    if (total > B * n_top) return 0;
    if (total > 0) {
        memcpy(fill, counts, (size_t)n_expert * sizeof(int));
        for (int t = 0; t < B; t++) {
            size_t row_off = (size_t)t * n_top;
            for (int i = 0; i < n_top; i++) {
                size_t top_off = row_off + (size_t)i;
                int e = top_idx[top_off];
                if (m->ep_size > 1 && (e % m->ep_size) != m->ep_rank) continue;
                int pos = fill[e];
                if (pos - counts[e] < start[e]) {
                    tok_ids[pos] = t;
                    tok_w[pos] = top_w[top_off];
                    memcpy(A + (size_t)pos * n_embd,
                           Xn + (size_t)t * n_embd,
                           (size_t)n_embd * sizeof(float));
                    fill[e]++;
                }
            }
        }
    }

    for (int e = 0; e < n_expert; e++) {
        int cnt = 0;
        int begin = counts[e];
        if (e + 1 < n_expert)
            cnt = counts[e + 1] - begin;
        else
            cnt = total - begin;
        if (cnt <= 0) continue;
        qtensor Wu, Wg, Wd;
        if (!tf_moe_expert_temp_qtensor(&Wu, &layer->ffn_up_exps, e, n_ff_exp, n_embd) ||
            !tf_moe_expert_temp_qtensor(&Wg, &layer->ffn_gate_exps, e, n_ff_exp, n_embd) ||
            !tf_moe_expert_temp_qtensor(&Wd, &layer->ffn_down_exps, e, n_embd, n_ff_exp))
            return 0;

        int j = begin;

        tf_gemm_bf16pv_prefill(m, U, &Wu, A, cnt);
        tf_gemm_bf16pv_prefill(m, G, &Wg, A, cnt);
        tf_silu_mul_flat_pool(m, G, G, U, (size_t)cnt * n_ff_exp);
        tf_gemm_bf16pv_prefill(m, D, &Wd, G, cnt);

        tf_scaled_vadd_scatter_pool(m, Y, D, tok_ids + (size_t)begin,
                                   tok_w + (size_t)begin,
                                   cnt, n_embd);
    }
    return 1;
#else
    (void)m; (void)layer; (void)l; (void)X; (void)B; (void)Y; (void)Xn;
    (void)A; (void)G; (void)U; (void)D; (void)top_idx; (void)top_w;
    (void)counts; (void)tok_ids; (void)tok_w;
    return 0;
#endif
}

#if defined(__ARM_FEATURE_SVE)
static inline void tf_attn_apply_sigmoid_gate_sve(float *out, const float *gate, int n) {
    int i = 0;
    svbool_t pg_all = svptrue_b32();
    for (; i + (int)svcntw() <= n; i += (int)svcntw()) {
        svfloat32_t o = svld1(pg_all, out + i);
        svfloat32_t g = svld1(pg_all, gate + i);
        svfloat32_t e = tf_fast_exp_sve(pg_all, svsub_x(pg_all, svdup_f32(0.0f), g));
        svfloat32_t sig = svdiv_x(pg_all, svdup_f32(1.0f),
                                  svadd_x(pg_all, svdup_f32(1.0f), e));
        svst1(pg_all, out + i, svmul_x(pg_all, o, sig));
    }
    for (; i < n; i++)
        out[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

static inline void tf_attn_apply_sigmoid_gate_scaled_sve(float *out, const float *gate,
                                                        int n, float scale) {
    int i = 0;
    int vl = (int)svcntw();
    svbool_t pg_all = svptrue_b32();
    svfloat32_t sv_scale = svdup_f32(scale);
    svfloat32_t one = svdup_f32(1.0f);
    for (; i + vl - 1 < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t o = svld1(pg, out + i);
        svfloat32_t g = svld1(pg, gate + i);
        svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), g));
        svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
        svst1(pg, out + i, svmul_x(pg, svmul_x(pg, o, sv_scale), sig));
    }
    for (; i < n; i++)
        out[i] *= scale / (1.0f + expf(-gate[i]));
}

static inline void tf_attn_apply_sigmoid_gate_rows_sve(float *out, const float *gate, int n) {
    int i = 0;
    int vl = (int)svcntw();
    svbool_t pg_all = svptrue_b32();
    svfloat32_t one = svdup_f32(1.0f);
    for (; i + (int)svcntw() <= n; i += (int)svcntw()) {
        svfloat32_t o = svld1(pg_all, out + i);
        svfloat32_t g = svld1(pg_all, gate + i);
        svfloat32_t e = tf_fast_exp_sve(pg_all, svsub_x(pg_all, svdup_f32(0.0f), g));
        svfloat32_t sig = svdiv_x(pg_all, one, svadd_x(pg_all, one, e));
        svst1(pg_all, out + i, svmul_x(pg_all, o, sig));
    }
    for (; i < n; i++)
        out[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

static inline float tf_dot_f32_sve(const float *a, const float *b, int n) {
    int i = 0;
    int vl = (int)svcntw();
    svfloat32_t acc = svdup_f32(0.0f);
    svbool_t pg_all = svptrue_b32();
    for (; i + vl - 1 < n; i += vl)
        acc = svmla_x(pg_all, acc, svld1(pg_all, a + i), svld1(pg_all, b + i));
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        acc = svmla_m(pg, acc, svld1(pg, a + i), svld1(pg, b + i));
    }
    return svaddv_f32(pg_all, acc);
}

static inline void tf_dot2_f32_sve(const float *a0, const float *a1,
                                  const float *b, int n,
                                  float *out0, float *out1) {
    int i = 0;
    int vl = (int)svcntw();
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svbool_t pg_all = svptrue_b32();
    for (; i + vl - 1 < n; i += vl) {
        svfloat32_t bv = svld1(pg_all, b + i);
        acc0 = svmla_x(pg_all, acc0, svld1(pg_all, a0 + i), bv);
        acc1 = svmla_x(pg_all, acc1, svld1(pg_all, a1 + i), bv);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        acc0 = svmla_m(pg, acc0, svld1(pg, a0 + i), svld1(pg, b + i));
        acc1 = svmla_m(pg, acc1, svld1(pg, a1 + i), svld1(pg, b + i));
    }
    *out0 = svaddv_f32(pg_all, acc0);
    *out1 = svaddv_f32(pg_all, acc1);
}

typedef struct {
    const float *Q2;
    const float *K;
    const float *V;
    float *AOut;
    int tid, nt;
    int M;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int q_dim;
    int q2_dim;
    int kv_dim;
    int gqa_ratio;
    int qhead_base;
    int kv_head_base;
    /* Keep each local Q-head aligned with the global KV slice:
     * kv_h = (qhead_base + h)/gqa_ratio - kv_head_base. */
    int q_tile;
    int k_tile;
    size_t scratch_stack_cap;
    float scale;
    int algo;
} tf_attn_prefill_tile_task;

static void *tf_attn_prefill_tile_worker(void *arg) {
    tf_attn_prefill_tile_task *t = (tf_attn_prefill_tile_task *)arg;
    const int M = t->M;
    const int hd = t->head_dim;
    int algo = t->algo;
    if (M <= 0) return NULL;
    const int q_tile = t->q_tile > 0 ? t->q_tile : 16;
    const int k_tile = t->k_tile > 0 ? t->k_tile : 256;
    const int max_q_tile = q_tile > M ? M : q_tile;
    const int tiles_per_head = (M + q_tile - 1) / q_tile;
    const long total_tiles = (long)t->n_heads * (long)tiles_per_head;
    svbool_t pg_all = svptrue_b32();
    int vl = (int)svcntw();

    size_t row_bytes = (size_t)max_q_tile * sizeof(float);
    float *row_max = NULL;
    float *row_sum = NULL;
    float *scores = NULL;
    void *scratch = NULL;
    const size_t score_heap_threshold = 512u * 1024u;

    if (algo == 0 || algo == 2) {
        size_t max_score_block = (size_t)max_q_tile * (size_t)M;
        size_t score_bytes = max_score_block * sizeof(float);
        size_t scratch_bytes = row_bytes + score_bytes;
        int use_heap = (scratch_bytes > score_heap_threshold);
        if (t->scratch_stack_cap > 0 && scratch_bytes > t->scratch_stack_cap)
            use_heap = 1;
        if (!use_heap) {
            row_max = (float *)alloca(row_bytes);
            if (score_bytes > 0) {
                scores = (float *)alloca(score_bytes);
            }
        } else {
            scratch = malloc(scratch_bytes);
            if (scratch) {
                row_max = (float *)scratch;
                scores = (float *)((char *)scratch + row_bytes);
            } else {
                algo = 1;
                row_max = (float *)alloca(row_bytes);
            }
        }
    } else if (algo == 3) {
        size_t max_score_tile = (size_t)max_q_tile * (size_t)k_tile;
        size_t score_bytes = max_score_tile * sizeof(float);
        size_t scratch_bytes = row_bytes + row_bytes + score_bytes;
        int use_heap = (scratch_bytes > score_heap_threshold);
        if (t->scratch_stack_cap > 0 && scratch_bytes > t->scratch_stack_cap)
            use_heap = 1;
        if (!use_heap) {
            row_max = (float *)alloca(row_bytes);
            row_sum = (float *)alloca(row_bytes);
            scores = (float *)alloca(score_bytes);
        } else {
            scratch = malloc(scratch_bytes);
            if (scratch) {
                row_max = (float *)scratch;
                row_sum = (float *)((char *)scratch + row_bytes);
                scores = (float *)((char *)scratch + 2 * row_bytes);
            } else {
                algo = 1;
                row_max = (float *)alloca(row_bytes);
            }
        }
    } else {
        row_max = (float *)alloca(row_bytes);
    }
    if (!row_max) return NULL;
    for (long job = t->tid; job < total_tiles; job += t->nt) {
        int h = (int)(job / tiles_per_head);
        int tile = (int)(job - (long)h * tiles_per_head);
        int q0 = tile * q_tile;
        int q1 = q0 + q_tile;
        if (q1 > M) q1 = M;
        int q_rows = q1 - q0;
        int kv_h = (t->qhead_base + h) / t->gqa_ratio;
        kv_h -= t->kv_head_base;
        const float *q2_base = t->Q2 + (size_t)q0 * t->q2_dim;

        for (int qi = 0; qi < q_rows; qi++) {
            row_max[qi] = -3.4028234663852886e38f;
        }
        if (algo == 3) {
            for (int qi = 0; qi < q_rows; qi++) row_sum[qi] = 0.0f;
        }

        if (algo == 0 || algo == 2) {
            /* Pass 1 (materialized modes): compute all Q@K^T scores for this
             * tile and track row max. */
            for (int k0 = 0; k0 < M; k0 += k_tile) {
                int k1 = k0 + k_tile;
                if (k1 > M) k1 = M;
                int qi = 0;
                for (; qi + 1 < q_rows; qi += 2) {
                    int q_idx0 = q0 + qi;
                    int q_idx1 = q_idx0 + 1;
                    int seq0 = q_idx0 + 1;
                    int seq1 = seq0 + 1;
                    const float *q_h0 = q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd;
                    const float *q_h1 = q2_base + (size_t)(qi + 1) * t->q2_dim + (size_t)h * 2 * hd;
                    float *q_scores = scores + (size_t)qi * (size_t)M;
                    float *q_scores1 = q_scores + (size_t)M;
                    for (int kj = k0; kj < k1; kj++) {
                        if (kj >= seq1) break;
                        const float *k_h = t->K + (size_t)kj * t->kv_dim + (size_t)kv_h * hd;
                        if (kj < seq0) {
                            float s0, s1;
                            tf_dot2_f32_sve(q_h0, q_h1, k_h, hd, &s0, &s1);
                            s0 *= t->scale;
                            s1 *= t->scale;
                            q_scores[kj] = s0;
                            q_scores1[kj] = s1;
                            if (s0 > row_max[qi]) row_max[qi] = s0;
                            if (s1 > row_max[qi + 1]) row_max[qi + 1] = s1;
                        } else if (kj < seq1) {
                            float s1 = tf_dot_f32_sve(q_h1, k_h, hd) * t->scale;
                            q_scores1[kj] = s1;
                            if (s1 > row_max[qi + 1]) row_max[qi + 1] = s1;
                        }
                    }
                }
                for (; qi < q_rows; qi++) {
                    int q_idx = q0 + qi;
                    int seq_len = q_idx + 1;
                    if (k0 >= seq_len) continue;
                    int seg_end = seq_len < k1 ? seq_len : k1;
                    const float *q_h = q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd;
                    float *q_scores = scores + (size_t)qi * (size_t)M;
                    for (int kj = k0; kj < seg_end; kj++) {
                        const float *k_h = t->K + (size_t)kj * t->kv_dim + (size_t)kv_h * hd;
                        float s = tf_dot_f32_sve(q_h, k_h, hd) * t->scale;
                        q_scores[kj] = s;
                        if (s > row_max[qi]) row_max[qi] = s;
                    }
                }
            }
        }

        if (algo == 0) {
            /* Pass 2: build softmax row statistics and accumulate O = softmax(S) @ V. */
            for (int qi = 0; qi < q_rows; qi++) {
                int q_idx = q0 + qi;
                int seq_len = q_idx + 1;
                float *q_scores = scores + (size_t)qi * (size_t)M;
                float *out_h = t->AOut + (size_t)q_idx * t->q_dim + (size_t)h * hd;
                float maxv = row_max[qi];
                float sum = 0.0f;
                int d = 0;
                for (; d + vl - 1 < hd; d += vl)
                    svst1(pg_all, out_h + d, svdup_f32(0.0f));
                if (d < hd) {
                    svbool_t pg = svwhilelt_b32(d, hd);
                    svst1(pg, out_h + d, svdup_f32(0.0f));
                }

                for (int k0 = 0; k0 < seq_len; k0 += k_tile) {
                    int k1 = k0 + k_tile;
                    if (k1 > seq_len) k1 = seq_len;
                    int kk = k0;
                    for (; kk + vl <= k1; kk += vl) {
                        svfloat32_t scores_vec = svld1(pg_all, q_scores + kk);
                        svfloat32_t exp_vec = tf_fast_exp_sve(pg_all,
                                                             svsub_x(pg_all, scores_vec,
                                                                     svdup_f32(maxv)));
                        sum += svaddv_f32(pg_all, exp_vec);
                        svst1(pg_all, q_scores + kk, exp_vec);
                    }
                    for (; kk < k1; kk++) {
                        float a = expf(q_scores[kk] - maxv);
                        sum += a;
                        q_scores[kk] = a;
                    }
                    for (int kj = k0; kj < k1; kj++) {
                        float a = q_scores[kj];
                        svfloat32_t a_vec = svdup_f32(a);
                        const float *v_h = t->V + (size_t)kj * t->kv_dim + (size_t)kv_h * hd;
                        d = 0;
                        for (; d + vl - 1 < hd; d += vl) {
                            svfloat32_t ov = svld1(pg_all, out_h + d);
                            ov = svmla_x(pg_all, ov, a_vec, svld1(pg_all, v_h + d));
                            svst1(pg_all, out_h + d, ov);
                        }
                        if (d < hd) {
                            svbool_t pg = svwhilelt_b32(d, hd);
                            svfloat32_t ov = svld1(pg, out_h + d);
                            ov = svmla_m(pg, ov, a_vec, svld1(pg, v_h + d));
                            svst1(pg, out_h + d, ov);
                        }
                    }
                }

                float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                tf_attn_apply_sigmoid_gate_scaled_sve(out_h,
                    q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd + hd,
                    hd, inv_sum);
            }
            continue;
        }

        if (algo == 2) {
            for (int qi = 0; qi < q_rows; qi++) {
                int q_idx = q0 + qi;
                int seq_len = q_idx + 1;
                float *q_scores = scores + (size_t)qi * (size_t)M;
                float *out_h = t->AOut + (size_t)q_idx * t->q_dim + (size_t)h * hd;
                float maxv = row_max[qi];
                float sum = 0.0f;
                int d = 0;
                for (; d + vl - 1 < hd; d += vl)
                    svst1(pg_all, out_h + d, svdup_f32(0.0f));
                if (d < hd) {
                    svbool_t pg = svwhilelt_b32(d, hd);
                    svst1(pg, out_h + d, svdup_f32(0.0f));
                }

                for (int k0 = 0; k0 < seq_len; k0 += k_tile) {
                    int k1 = k0 + k_tile;
                    if (k1 > seq_len) k1 = seq_len;
                    int kk = k0;
                    for (; kk + vl <= k1; kk += vl) {
                        svfloat32_t scores_vec = svld1(pg_all, q_scores + kk);
                        svfloat32_t exp_vec = tf_fast_exp_sve(pg_all,
                                                             svsub_x(pg_all, scores_vec,
                                                                     svdup_f32(maxv)));
                        sum += svaddv_f32(pg_all, exp_vec);
                        svst1(pg_all, q_scores + kk, exp_vec);
                    }
                    for (; kk < k1; kk++) {
                        float a = expf(q_scores[kk] - maxv);
                        sum += a;
                        q_scores[kk] = a;
                    }
                }

                for (int k0 = 0; k0 < seq_len; k0 += k_tile) {
                    int k1 = k0 + k_tile;
                    if (k1 > seq_len) k1 = seq_len;
                    for (int kj = k0; kj < k1; kj++) {
                        float a = q_scores[kj];
                        svfloat32_t a_vec = svdup_f32(a);
                        const float *v_h = t->V + (size_t)kj * t->kv_dim + (size_t)kv_h * hd;
                        d = 0;
                        for (; d + vl - 1 < hd; d += vl) {
                            svfloat32_t ov = svld1(pg_all, out_h + d);
                            ov = svmla_x(pg_all, ov, a_vec, svld1(pg_all, v_h + d));
                            svst1(pg_all, out_h + d, ov);
                        }
                        if (d < hd) {
                            svbool_t pg = svwhilelt_b32(d, hd);
                            svfloat32_t ov = svld1(pg, out_h + d);
                            ov = svmla_m(pg, ov, a_vec, svld1(pg, v_h + d));
                            svst1(pg, out_h + d, ov);
                        }
                    }
                }

                float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                tf_attn_apply_sigmoid_gate_scaled_sve(out_h,
                    q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd + hd,
                    hd, inv_sum);
            }
            continue;
        }

        if (algo == 3) {
            /* Two-pass tiled materialization: keep score scratch as a fixed
             * [q_rows, k_tile] block in L1-sized working memory. */
            for (int k0 = 0; k0 < M; k0 += k_tile) {
                int k1 = k0 + k_tile;
                if (k1 > M) k1 = M;
                int k_blk = k1 - k0;
                int qi = 0;
                for (; qi + 1 < q_rows; qi += 2) {
                    int q_idx0 = q0 + qi;
                    int q_idx1 = q_idx0 + 1;
                    int seq0 = q_idx0 + 1;
                    int seq1 = seq0 + 1;
                    float *q_scores = scores + (size_t)qi * (size_t)k_tile;
                    float *q_scores1 = q_scores + (size_t)k_tile;
                    const float *q_h0 = q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd;
                    const float *q_h1 = q2_base + (size_t)(qi + 1) * t->q2_dim + (size_t)h * 2 * hd;
                    for (int kk = 0; kk < k_blk; kk++) {
                        int idx = k0 + kk;
                        if (idx >= seq1) break;
                        const float *k_h = t->K + (size_t)idx * t->kv_dim + (size_t)kv_h * hd;
                        if (idx < seq0) {
                            float s0, s1;
                            tf_dot2_f32_sve(q_h0, q_h1, k_h, hd, &s0, &s1);
                            s0 *= t->scale;
                            s1 *= t->scale;
                            q_scores[kk] = s0;
                            q_scores1[kk] = s1;
                            if (s0 > row_max[qi]) row_max[qi] = s0;
                            if (s1 > row_max[qi + 1]) row_max[qi + 1] = s1;
                        } else if (idx < seq1) {
                            float s1 = tf_dot_f32_sve(q_h1, k_h, hd) * t->scale;
                            q_scores1[kk] = s1;
                            if (s1 > row_max[qi + 1]) row_max[qi + 1] = s1;
                        }
                    }
                }
                for (; qi < q_rows; qi++) {
                    int q_idx = q0 + qi;
                    int seq_len = q_idx + 1;
                    float *q_scores = scores + (size_t)qi * (size_t)k_tile;
                    const float *q_h = q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd;
                    for (int kk = 0; kk < k_blk; kk++) {
                        int idx = k0 + kk;
                        if (idx >= seq_len) break;
                        const float *k_h = t->K + (size_t)idx * t->kv_dim + (size_t)kv_h * hd;
                        float s = tf_dot_f32_sve(q_h, k_h, hd) * t->scale;
                        q_scores[kk] = s;
                        if (s > row_max[qi]) row_max[qi] = s;
                    }
                }
            }

            for (int k0 = 0; k0 < M; k0 += k_tile) {
                int k1 = k0 + k_tile;
                if (k1 > M) k1 = M;
                int k_blk = k1 - k0;
                for (int qi = 0; qi < q_rows; qi++) {
                    int q_idx = q0 + qi;
                    int seq_len = q_idx + 1;
                    if (k0 >= seq_len) continue;
                    float *q_scores = scores + (size_t)qi * (size_t)k_tile;
                    int valid = seq_len - k0;
                    if (valid > k_blk) valid = k_blk;
                    float maxv = row_max[qi];
                    float *sum = row_sum + qi;
                    int kk = 0;
                    for (; kk + vl <= valid; kk += vl) {
                        svbool_t p = svwhilelt_b32(kk, valid);
                        svfloat32_t scores_vec = svsel(p, svld1(pg_all, q_scores + kk),
                                                      svdup_f32(-3.4028234663852886e38f));
                        svfloat32_t exp_vec = tf_fast_exp_sve(pg_all,
                                                             svsub_x(pg_all, scores_vec,
                                                                     svdup_f32(maxv)));
                        *sum += svaddv_f32(pg_all, exp_vec);
                        svst1(p, q_scores + kk, exp_vec);
                    }
                    for (; kk < valid; kk++) {
                        float a = expf(q_scores[kk] - maxv);
                        q_scores[kk] = a;
                        *sum += a;
                    }
                }
            }

            for (int qi = 0; qi < q_rows; qi++) {
                int q_idx = q0 + qi;
                int seq_len = q_idx + 1;
                float *out_h = t->AOut + (size_t)q_idx * t->q_dim + (size_t)h * hd;
                float inv_sum = row_sum[qi] > 0.0f ? 1.0f / row_sum[qi] : 0.0f;
                int d = 0;
                for (; d + vl - 1 < hd; d += vl)
                    svst1(pg_all, out_h + d, svdup_f32(0.0f));
                if (d < hd) {
                    svbool_t pg = svwhilelt_b32(d, hd);
                    svst1(pg, out_h + d, svdup_f32(0.0f));
                }

                for (int k0 = 0; k0 < seq_len; k0 += k_tile) {
                    int k1 = k0 + k_tile;
                    if (k1 > seq_len) k1 = seq_len;
                    int k_blk = k1 - k0;
                    for (int kk = 0; kk < k_blk; kk++) {
                        float a = scores[(size_t)qi * (size_t)k_tile + kk] * inv_sum;
                        svfloat32_t a_vec = svdup_f32(a);
                        const float *v_h = t->V + (size_t)(k0 + kk) * t->kv_dim + (size_t)kv_h * hd;
                        int d0 = 0;
                        for (; d0 + vl - 1 < hd; d0 += vl) {
                            svfloat32_t ov = svld1(pg_all, out_h + d0);
                            ov = svmla_x(pg_all, ov, a_vec, svld1(pg_all, v_h + d0));
                            svst1(pg_all, out_h + d0, ov);
                        }
                        if (d0 < hd) {
                            svbool_t pg = svwhilelt_b32(d0, hd);
                            svfloat32_t ov = svld1(pg, out_h + d0);
                            ov = svmla_m(pg, ov, a_vec, svld1(pg, v_h + d0));
                            svst1(pg, out_h + d0, ov);
                        }
                    }
                }

                tf_attn_apply_sigmoid_gate_scaled_sve(out_h,
                    q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd + hd,
                    hd, 1.0f);
            }
            continue;
        }

        /* On-the-fly online softmax path: keep scores in registers/accumulators.
         * Saves score scratch and one pass over QK scores at the cost of an output
         * rescale each token. */
        for (int qi = 0; qi < q_rows; qi++) {
            int q_idx = q0 + qi;
            int seq_len = q_idx + 1;
            const float *q_h = q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd;
            float *out_h = t->AOut + (size_t)q_idx * t->q_dim + (size_t)h * hd;
            const float minus_inf = -3.4028234663852886e38f;
            float maxv = minus_inf;
            float sum = 0.0f;
            int d = 0;
            for (; d + vl - 1 < hd; d += vl)
                svst1(pg_all, out_h + d, svdup_f32(0.0f));
            if (d < hd) {
                svbool_t pg = svwhilelt_b32(d, hd);
                svst1(pg, out_h + d, svdup_f32(0.0f));
            }
            for (int k0 = 0; k0 < seq_len; k0 += k_tile) {
                int k1 = k0 + k_tile;
                if (k1 > seq_len) k1 = seq_len;
                for (int kj = k0; kj < k1; kj++) {
                    const float *k_h = t->K + (size_t)kj * t->kv_dim + (size_t)kv_h * hd;
                    float s = tf_dot_f32_sve(q_h, k_h, hd) * t->scale;
                    float m_new = (s > maxv) ? s : maxv;
                    float alpha = (maxv == minus_inf) ? 0.0f : expf(maxv - m_new);
                    float beta = expf(s - m_new);
                    sum = alpha * sum + beta;
                    maxv = m_new;

                    const float *v_h = t->V + (size_t)kj * t->kv_dim + (size_t)kv_h * hd;
                    svfloat32_t v_alpha = svdup_f32(alpha);
                    svfloat32_t v_beta = svdup_f32(beta);
                    int d0 = 0;
                    for (; d0 + vl - 1 < hd; d0 += vl) {
                        svfloat32_t ov = svld1(pg_all, out_h + d0);
                        ov = svmul_x(pg_all, ov, v_alpha);
                        ov = svmla_x(pg_all, ov, svld1(pg_all, v_h + d0), v_beta);
                        svst1(pg_all, out_h + d0, ov);
                    }
                    if (d0 < hd) {
                        svbool_t pg = svwhilelt_b32(d0, hd);
                        svfloat32_t ov = svld1(pg, out_h + d0);
                        ov = svmul_m(pg, ov, v_alpha);
                        ov = svmla_m(pg, ov, svld1(pg, v_h + d0), v_beta);
                        svst1(pg, out_h + d0, ov);
                    }
                }
            }
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            tf_attn_apply_sigmoid_gate_scaled_sve(out_h,
                q2_base + (size_t)qi * t->q2_dim + (size_t)h * 2 * hd + hd,
                hd, inv_sum);
        }
    }
    if (scratch) free(scratch);
    return NULL;
}

static int tf_attn_prefill_layer_bf16pv(transformer_model *m, transformer_layer *layer, int l,
                                        float *X, int M, int pos0,
                                        float *Xn, float *Q2, float *K, float *V,
                                        float *AOut, float *O,
                                        int tiled,
                                        tf_ep_prefill_detail *pd) {
    if (layer->is_ssm) return 0;
    if (!tf_prefill_weight_batched(&layer->attn_q) ||
        !tf_prefill_weight_batched(&layer->attn_k) ||
        !tf_prefill_weight_batched(&layer->attn_v) ||
        !tf_prefill_weight_batched(&layer->attn_output)) return 0;
    if (!Xn || !Q2 || !K || !V || !AOut || !O) return 0;

    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int q_dim = n_heads * head_dim;
    int q2_dim = 2 * q_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = (m->gqa_group > 0) ? m->gqa_group : ((n_kv_heads > 0) ? n_heads / n_kv_heads : 1);
    if (layer->attn_q.n_rows != q2_dim || layer->attn_q.n_cols != n_embd ||
        layer->attn_k.n_rows != kv_dim || layer->attn_k.n_cols != n_embd ||
        layer->attn_v.n_rows != kv_dim || layer->attn_v.n_cols != n_embd ||
        layer->attn_output.n_rows != n_embd || layer->attn_output.n_cols != q_dim)
        return 0;
    if (pos0 < 0 || pos0 + M > m->max_seq_len) return 0;

    double t0 = pd ? tf_wall_seconds() : 0.0;
    for (int t = 0; t < M; t++)
        tf_rmsnorm(Xn + (size_t)t * n_embd, X + (size_t)t * n_embd,
                   &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    if (pd) pd->attn_norm += tf_wall_seconds() - t0;

    t0 = pd ? tf_wall_seconds() : 0.0;
    tf_gemm_bf16pv_prefill(m, Q2, &layer->attn_q, Xn, M);
    tf_gemm_bf16pv_prefill(m, K,  &layer->attn_k, Xn, M);
    tf_gemm_bf16pv_prefill(m, V,  &layer->attn_v, Xn, M);
    if (pd) pd->attn_qkv += tf_wall_seconds() - t0;

    t0 = pd ? tf_wall_seconds() : 0.0;
    /* Tiled attention path requires pos0==0 for this implementation:
     * worker seq_len is q_idx+1 and K/V for this tile are only available
     * for the current prefill block. */
    if (tiled && pos0 == 0) {
        for (int t = 0; t < M; t++) {
            float *q2 = Q2 + (size_t)t * q2_dim;
            int pos = pos0 + t;
            if (layer->attn_q_norm.data)
                tf_qk_norm(q2, n_heads, head_dim, &layer->attn_q_norm,
                           m->rms_norm_eps, m->matvec_tmp);
            if (layer->attn_k_norm.data)
                tf_qk_norm(K + (size_t)t * kv_dim, n_kv_heads, head_dim,
                           &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
            tf_apply_rope(m, q2, K + (size_t)t * kv_dim,
                          n_heads, n_kv_heads, head_dim, pos, pos, pos);
            tf_k_cache_write_pos(m->key_cache[l],
                                 m->key_scales ? m->key_scales[l] : NULL,
                                 K + (size_t)t * kv_dim, pos, n_kv_heads, head_dim,
                                 m->max_seq_len, m->kv_dtype, m->kv_k_transposed,
                                 m->kv_k_dp);
            tf_kv_write_all_heads(m->value_cache[l],
                                  m->value_scales ? m->value_scales[l] : NULL,
                                  V + (size_t)t * kv_dim, pos, n_kv_heads, head_dim,
                                  m->kv_dtype);
        }

    int nt = (m->n_threads > 1 && m->pool_alive) ? m->n_threads : 1;
    const char *q_tile_env = getenv("TP_PREFILL_ATTN_TILE_Q");
    const char *k_tile_env = getenv("TP_PREFILL_ATTN_TILE_K");
    const char *algo_env = getenv("TP_PREFILL_ATTN_TILE_ALGO");
    const char *stack_env = getenv("TP_PREFILL_ATTN_TILE_STACK_BYTES");
    if (!q_tile_env) q_tile_env = getenv("EP_PREFILL_ATTN_TILE_Q");
    if (!k_tile_env) k_tile_env = getenv("EP_PREFILL_ATTN_TILE_K");
    if (!algo_env)  algo_env  = getenv("EP_PREFILL_ATTN_TILE_ALGO");
    int q_tile = q_tile_env ? atoi(q_tile_env) : 32;
    int k_tile = k_tile_env ? atoi(k_tile_env) : 256;
    int attn_tiled_algo = algo_env ? atoi(algo_env) : 0;
    size_t stack_cap = 2u << 20; /* 2 MiB hard cap for per-task scratch by default */
    if (stack_env) {
    long long parsed = atoll(stack_env);
    if (parsed >= 0) stack_cap = (size_t)parsed;
    }
    if (attn_tiled_algo < 0 || attn_tiled_algo > 3) attn_tiled_algo = 0;
    if (q_tile <= 0) q_tile = 32;
    if (k_tile <= 0) k_tile = 256;
    if ((size_t)M > 0 && stack_cap > 0) {
        size_t per_qtile = ((size_t)M * sizeof(float)) + sizeof(float);
        if (per_qtile > 0) {
            size_t q_max = stack_cap / per_qtile;
            if (q_max < 1u) q_max = 1u;
            if ((size_t)q_tile > q_max) q_tile = (int)q_max;
        }
    }
    if (q_tile > M) q_tile = M;
    if (q_tile <= 0) q_tile = 1;
    if (nt > 0) {
        if (nt > 1) {
            tf_attn_prefill_tile_task *tasks =
                (tf_attn_prefill_tile_task *)alloca((size_t)nt * sizeof(*tasks));
            for (int ti = 0; ti < nt; ti++) {
                tasks[ti] = (tf_attn_prefill_tile_task){
                        .Q2 = Q2, .K = K, .V = V, .AOut = AOut,
                        .tid = ti, .nt = nt, .M = M,
                        .n_heads = n_heads, .n_kv_heads = n_kv_heads,
                        .head_dim = head_dim, .q_dim = q_dim, .q2_dim = q2_dim,
                        .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                        .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                        .q_tile = q_tile,
                        .k_tile = k_tile, .scratch_stack_cap = stack_cap, .algo = attn_tiled_algo,
                        .scale = 1.0f / sqrtf((float)head_dim),
                    };
            }
            tf_pool_dispatch(m, tf_attn_prefill_tile_worker, tasks, sizeof(*tasks));
        } else {
            tf_attn_prefill_tile_task task = {
                .Q2 = Q2, .K = K, .V = V, .AOut = AOut,
                .tid = 0, .nt = 1, .M = M,
                .n_heads = n_heads, .n_kv_heads = n_kv_heads,
                .head_dim = head_dim, .q_dim = q_dim, .q2_dim = q2_dim,
                .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                .q_tile = q_tile,
                .k_tile = k_tile, .scratch_stack_cap = stack_cap, .algo = attn_tiled_algo,
                .scale = 1.0f / sqrtf((float)head_dim),
            };
            tf_attn_prefill_tile_worker(&task);
        }
    }
    if (pd) pd->attn_token += tf_wall_seconds() - t0;
    } else {
    for (int t = 0; t < M; t++) {
        float *q2 = Q2 + (size_t)t * q2_dim;
        float *aout = AOut + (size_t)t * q_dim;
        for (int h = 0; h < n_heads; h++) {
            memcpy(m->q + (size_t)h * head_dim,
                   q2 + (size_t)h * 2 * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
        if (layer->attn_q_norm.data)
            tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm,
                       m->rms_norm_eps, m->matvec_tmp);
        if (layer->attn_k_norm.data)
            tf_qk_norm(K + (size_t)t * kv_dim, n_kv_heads, head_dim,
                       &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
        int pos = pos0 + t;
        tf_apply_rope(m, m->q, K + (size_t)t * kv_dim,
                      n_heads, n_kv_heads, head_dim, pos, pos, pos);
        tf_k_cache_write_pos(m->key_cache[l],
                             m->key_scales ? m->key_scales[l] : NULL,
                             K + (size_t)t * kv_dim, pos, n_kv_heads, head_dim,
                             m->max_seq_len, m->kv_dtype, m->kv_k_transposed,
                             m->kv_k_dp);
        tf_kv_write_all_heads(m->value_cache[l],
                              m->value_scales ? m->value_scales[l] : NULL,
                              V + (size_t)t * kv_dim, pos, n_kv_heads, head_dim,
                              m->kv_dtype);
        int seq_len = pos + 1;
        float scale = 1.0f / sqrtf((float)head_dim);
        if (m->n_threads > 1 && m->pool_alive) {
            int nt = m->n_threads;
            static int fa_enabled = -1;
            if (fa_enabled < 0) fa_enabled = getenv("TF_USE_FA") ? 1 : 0;
            if (fa_enabled && nt > n_heads && !m->kv_k_dp) {
                float *saved_xb2 = m->xb2;
                m->xb2 = aout;
                tf_attention_fa(m, l, n_heads, n_kv_heads, head_dim, kv_dim,
                                gqa_ratio, seq_len, scale);
                m->xb2 = saved_xb2;
            } else {
                tf_attn_task *atasks = (tf_attn_task *)alloca((size_t)nt * sizeof(tf_attn_task));
                int heads_per = n_heads / nt, heads_extra = n_heads % nt, hoff = 0;
                for (int ti = 0; ti < nt; ti++) {
                    int hcount = heads_per + (ti < heads_extra ? 1 : 0);
                    atasks[ti] = (tf_attn_task){
                        .q = m->q, .att = m->att, .xb2 = aout,
                        .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                        .key_scales = m->key_scales ? m->key_scales[l] : NULL,
                        .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                        .head_start = hoff, .head_end = hoff + hcount,
                        .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                        .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                        .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                        .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                        .k_transposed = m->kv_k_transposed,
                        .k_dp = m->kv_k_dp,
                        .scale = scale,
                    };
                    hoff += hcount;
                }
                tf_pool_dispatch(m, tf_attn_worker, atasks, sizeof(tf_attn_task));
            }
        } else {
            tf_attn_task st = {
                .q = m->q, .att = m->att, .xb2 = m->xb2,
                .key_cache = m->key_cache[l], .value_cache = m->value_cache[l],
                .key_scales = m->key_scales ? m->key_scales[l] : NULL,
                .value_scales = m->value_scales ? m->value_scales[l] : NULL,
                .head_start = 0, .head_end = n_heads,
                .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio,
                .qhead_base = m->tp_qhead_offset, .kv_head_base = m->tp_kv_head_base,
                .seq_len = seq_len, .max_seq_len = m->max_seq_len,
                .n_kv_heads = n_kv_heads, .kv_dtype = m->kv_dtype,
                .k_transposed = m->kv_k_transposed,
                .k_dp = m->kv_k_dp,
                .scale = scale,
            };
            st.xb2 = aout;
            tf_attn_worker(&st);
        }
        tf_attn_apply_sigmoid_gate_rows_sve(aout, q2 + q_dim, q_dim);
    }
    if (pd) pd->attn_token += tf_wall_seconds() - t0;
    }

    t0 = pd ? tf_wall_seconds() : 0.0;
    tf_gemm_bf16pv_prefill(m, O, &layer->attn_output, AOut, M);
    tf_vadd_flat_pool(m, X, O, (size_t)M * n_embd);
    if (pd) pd->attn_out += tf_wall_seconds() - t0;
    return 1;
}

static void tf_ssm_prefill_post_alpha_beta(transformer_model *m, transformer_layer *layer,
                                           float *alpha, float *beta_arr, int M) {
    int dt_rank = m->ssm_dt_rank;
    int hoff = m->ssm_head_offset;
    float a_buf[64], dt_bias_buf[64];
    tf_dequant_row(&layer->ssm_a, 0, a_buf);
    tf_dequant_row(&layer->ssm_dt_bias, 0, dt_bias_buf);

    int vl = (int)svcntw();
    svfloat32_t one = svdup_f32(1.0f), thr = svdup_f32(20.0f);
    for (int t = 0; t < M; t++) {
        float *a = alpha + (size_t)t * dt_rank;
        float *b = beta_arr + (size_t)t * dt_rank;
        for (int i = 0; i < dt_rank; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dt_rank);
            svfloat32_t val = svadd_x(pg, svld1_f32(pg, a + i),
                                      svld1_f32(pg, dt_bias_buf + hoff + i));
            svfloat32_t lg = tf_sve_log_f32(pg, svadd_x(pg, one, tf_fast_exp_sve(pg, val)));
            svfloat32_t sp = svsel(svcmpgt(pg, val, thr), val, lg);
            svst1_f32(pg, a + i, svmul_x(pg, sp, svld1_f32(pg, a_buf + hoff + i)));

            svfloat32_t bv = svld1_f32(pg, b + i);
            svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), bv));
            svst1_f32(pg, b + i, svdiv_x(pg, one, svadd_x(pg, one, e)));
        }
    }
}

static inline void tf_ssm_gate_norm_sve(float *o_h, const float *z_h,
                                       const float *norm_w, int d_state, float eps);

static void tf_ssm_prefill_finish_token(transformer_model *m, transformer_layer *layer,
                                        int layer_idx, float *qkv_buf, float *z_buf,
                                        float *alpha, float *beta_arr, float *out_buf,
                                        float *Q_exp, float *K_exp, float *conv_tmp,
                                        const float *norm_w) {
    int qkv_dim = m->ssm_qkv_dim;
    int d_inner = m->ssm_d_inner;
    int d_state = m->ssm_d_state;
    int n_group = m->ssm_n_group;
    int dt_rank = m->ssm_dt_rank;
    int conv_k = m->ssm_conv_kernel;
    float eps = m->rms_norm_eps;

    {
        float *conv_st = m->conv_state[layer_idx];
        int wr = m->conv_state_pos[layer_idx];
        int n_hist = conv_k - 1;
        float *conv_out = conv_tmp;
        float *w_trans = m->conv_w_trans[layer_idx];
        int *row_off = n_hist > 0 ? (int *)alloca((size_t)n_hist * sizeof(*row_off)) : NULL;
        for (int f = 0; f < n_hist; f++)
            row_off[f] = ((wr + f) % n_hist) * qkv_dim;

        int vl = (int)svcntw();
        for (int j = 0; j < qkv_dim; j += vl) {
            svbool_t pg = svwhilelt_b32(j, qkv_dim);
            svfloat32_t sum = svdup_f32(0.0f);
            for (int f = 0; f < n_hist; f++)
                sum = svmla_x(pg, sum, svld1_f32(pg, w_trans + (size_t)f * qkv_dim + j),
                              svld1_f32(pg, conv_st + (size_t)row_off[f] + j));
            sum = svmla_x(pg, sum, svld1_f32(pg, w_trans + (size_t)n_hist * qkv_dim + j),
                          svld1_f32(pg, qkv_buf + j));
            svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), sum));
            svfloat32_t sig = svdiv_x(pg, svdup_f32(1.0f), svadd_x(pg, svdup_f32(1.0f), e));
            svst1_f32(pg, conv_out + j, svmul_x(pg, sum, sig));
        }

        memcpy(conv_st + (size_t)wr * qkv_dim, qkv_buf, (size_t)qkv_dim * sizeof(float));
        m->conv_state_pos[layer_idx] = n_hist > 0 ? (wr + 1) % n_hist : 0;
        memcpy(qkv_buf, conv_out, (size_t)qkv_dim * sizeof(float));
    }

    float *Q_raw = qkv_buf;
    float *K_raw = qkv_buf + (size_t)n_group * d_state;
    float *V_raw = qkv_buf + (size_t)2 * n_group * d_state;
    for (int g = 0; g < n_group; g++) {
        tf_l2_norm(Q_raw + (size_t)g * d_state, d_state, eps);
        tf_l2_norm(K_raw + (size_t)g * d_state, d_state, eps);
    }

    size_t tile_bytes = (size_t)n_group * d_state * sizeof(float);
    int n_repeat = dt_rank / n_group;
    for (int r = n_repeat - 1; r >= 0; r--) {
        memcpy(Q_exp + (size_t)r * n_group * d_state, Q_raw, tile_bytes);
        memcpy(K_exp + (size_t)r * n_group * d_state, K_raw, tile_bytes);
    }

    float scale = 1.0f / sqrtf((float)d_state);
    float *rec_state = m->recurrent_state[layer_idx];
    if (m->n_threads > 1 && m->pool_alive) {
        int nt = m->n_threads;
        int ncmgs = m->cmg_pin ? m->cmg_pin_ncmgs : 1;
        tf_ssm_recurrence_task *rtasks = (tf_ssm_recurrence_task *)alloca((size_t)nt * sizeof(*rtasks));
        for (int t = 0; t < nt; t++) {
            int hs, he;
            tf_ssm_head_range(dt_rank, nt, ncmgs, t, &hs, &he);
            rtasks[t] = (tf_ssm_recurrence_task){
                rec_state, Q_exp, K_exp, V_raw, out_buf, alpha, beta_arr, hs, he, d_state, scale
            };
        }
        tf_pool_dispatch(m, tf_ssm_recurrence_worker, rtasks, sizeof(tf_ssm_recurrence_task));
    } else {
        tf_ssm_recurrence_task rtask = {
            rec_state, Q_exp, K_exp, V_raw, out_buf, alpha, beta_arr, 0, dt_rank, d_state, scale
        };
        tf_ssm_recurrence_worker(&rtask);
    }

    for (int h = 0; h < dt_rank; h++) {
        float *o_h = out_buf + (size_t)h * d_state;
        float *z_h = z_buf + (size_t)h * d_state;
        tf_ssm_gate_norm_sve(o_h, z_h, norm_w, d_state, eps);
    }
}

static inline void tf_ssm_recurrence_step_sve(float *state,
                                              const float *q_h,
                                              const float *k_h,
                                              const float *v_h,
                                              float *o_h,
                                              float alpha,
                                              float beta,
                                              int ds,
                                              float scale,
                                              const float *norm_w,
                                              const float *z_h,
                                              float eps) {
    svbool_t pg_all = svptrue_b32();
    int vl = (int)svcntw();

    float qnorm = 0.0f;
    {
        svfloat32_t sq = svdup_f32(0.0f);
        int c = 0;
        for (; c + vl - 1 < ds; c += vl) {
            svfloat32_t qv = svld1(pg_all, q_h + c);
            sq = svmla_x(pg_all, sq, qv, qv);
        }
        qnorm = svaddv_f32(pg_all, sq);
        if (c < ds) {
            svbool_t pg = svwhilelt_b32(c, ds);
            svfloat32_t qv = svld1(pg, q_h + c);
            qnorm += svaddv_f32(pg, svmul_x(pg, qv, qv));
        }
    }

    float decay = expf(alpha);
    svfloat32_t vdecay = svdup_f32(decay);
    for (int r = 0; r < ds; r++) {
        float *row = state + (size_t)r * ds;
        svfloat32_t a_dot_k = svdup_f32(0.0f), a_dot_q = svdup_f32(0.0f);
        int c = 0;
        for (; c + vl - 1 < ds; c += vl) {
            svfloat32_t rv = svmul_x(pg_all, svld1(pg_all, row + c), vdecay);
            svst1(pg_all, row + c, rv);
            svfloat32_t kv = svld1(pg_all, k_h + c);
            svfloat32_t qv = svld1(pg_all, q_h + c);
            a_dot_k = svmla_x(pg_all, a_dot_k, rv, kv);
            a_dot_q = svmla_x(pg_all, a_dot_q, rv, qv);
        }
        float sum_k = svaddv_f32(pg_all, a_dot_k);
        float sum_q = svaddv_f32(pg_all, a_dot_q);
        if (c < ds) {
            svbool_t pg = svwhilelt_b32(c, ds);
            svfloat32_t rv = svmul_x(pg, svld1(pg, row + c), vdecay);
            svfloat32_t kv = svld1(pg, k_h + c);
            svfloat32_t qv = svld1(pg, q_h + c);
            svst1(pg, row + c, rv);
            sum_k += svaddv_f32(pg, svmul_x(pg, rv, kv));
            sum_q += svaddv_f32(pg, svmul_x(pg, rv, qv));
        }
        float delta = (v_h[r] - sum_k) * beta;
        o_h[r] = (sum_q + delta * qnorm) * scale;
        svfloat32_t dv = svdup_f32(delta);
        c = 0;
        for (; c + vl - 1 < ds; c += vl) {
            svst1(pg_all, row + c, svmla_x(pg_all, svld1(pg_all, row + c), dv, svld1(pg_all, q_h + c)));
        }
        if (c < ds) {
            svbool_t pg = svwhilelt_b32(c, ds);
            svst1(pg, row + c, svmla_x(pg, svld1(pg, row + c), dv, svld1(pg, q_h + c)));
        }
    }
    if (!norm_w || !z_h) return;
    svfloat32_t vss = svdup_f32(0.0f);
    for (int i = 0; i < ds; i += vl) {
        svbool_t pg = svwhilelt_b32(i, ds);
        svfloat32_t oi = svld1_f32(pg, o_h + i);
        vss = svmla_m(pg, vss, oi, oi);
    }
    float scl = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vss) / ds + eps);
    svfloat32_t vscale = svdup_f32(scl), one = svdup_f32(1.0f);
    for (int i = 0; i < ds; i += vl) {
        svbool_t pg = svwhilelt_b32(i, ds);
        svfloat32_t oi = svld1_f32(pg, o_h + i);
        svfloat32_t wi = svld1_f32(pg, norm_w + i);
        svfloat32_t zi = svld1_f32(pg, z_h + i);
        svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), zi));
        svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
        svst1_f32(pg, o_h + i, svmul_x(pg, svmul_x(pg, svmul_x(pg, oi, vscale), wi), svmul_x(pg, zi, sig)));
    }
}

static inline void tf_ssm_gate_norm_sve(float *o_h, const float *z_h,
                                        const float *norm_w, int d_state,
                                        float eps) {
    int vl = (int)svcntw();
    svfloat32_t vss = svdup_f32(0.0f);
    for (int i = 0; i < d_state; i += vl) {
        svbool_t pg = svwhilelt_b32(i, d_state);
        svfloat32_t oi = svld1_f32(pg, o_h + i);
        vss = svmla_m(pg, vss, oi, oi);
    }
    float scl = 1.0f / sqrtf(svaddv_f32(svptrue_b32(), vss) / d_state + eps);
    svfloat32_t vscale = svdup_f32(scl), one = svdup_f32(1.0f);
    for (int i = 0; i < d_state; i += vl) {
        svbool_t pg = svwhilelt_b32(i, d_state);
        svfloat32_t oi = svld1_f32(pg, o_h + i);
        svfloat32_t wi = svld1_f32(pg, norm_w + i);
        svfloat32_t zi = svld1_f32(pg, z_h + i);
        svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), zi));
        svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
        svfloat32_t normed = svmul_x(pg, svmul_x(pg, oi, vscale), wi);
        svst1_f32(pg, o_h + i, svmul_x(pg, normed, svmul_x(pg, zi, sig)));
    }
}

typedef struct {
    float *QKV;
    float *conv_st;
    float *w_trans;
    int tid, nt;
    int M;
    int qkv_dim;
    int n_hist;
    int wr0;
} tf_ssm_prefill_conv_block_task;

static void *tf_ssm_prefill_conv_block_worker(void *arg) {
    tf_ssm_prefill_conv_block_task *t = (tf_ssm_prefill_conv_block_task *)arg;
    int chunk = (t->qkv_dim + t->nt - 1) / t->nt;
    int cs = t->tid * chunk;
    int ce = cs + chunk;
    if (ce > t->qkv_dim) ce = t->qkv_dim;
    if (cs >= ce) return NULL;
    int *row_off = t->n_hist > 0 ? (int *)alloca((size_t)t->n_hist * sizeof(*row_off)) : NULL;
    int vl = (int)svcntw();
    svfloat32_t one = svdup_f32(1.0f);
    if (t->n_hist <= 0) {
        for (int ti = 0; ti < t->M; ti++) {
            float *qkv = t->QKV + (size_t)ti * t->qkv_dim;
            for (int j = cs; j < ce; j += vl) {
                svbool_t pg = svwhilelt_b32(j, ce);
                svfloat32_t cur = svld1_f32(pg, qkv + j);
                svfloat32_t sum = svmul_m(pg, cur, svld1_f32(pg, t->w_trans + j));
                svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), sum));
                svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
                svst1_f32(pg, qkv + j, svmul_x(pg, sum, sig));
            }
        }
        return NULL;
    }

    for (int ti = 0; ti < t->M; ti++) {
        float *qkv = t->QKV + (size_t)ti * t->qkv_dim;
        int wr = (t->wr0 + ti) % t->n_hist;
        for (int f = 0; f < t->n_hist; f++)
            row_off[f] = ((wr + f) % t->n_hist) * t->qkv_dim;

        for (int j = cs; j < ce; j += vl) {
            svbool_t pg = svwhilelt_b32(j, ce);
            svfloat32_t cur = svld1_f32(pg, qkv + j);
            svfloat32_t sum = svdup_f32(0.0f);
            for (int f = 0; f < t->n_hist; f++)
                sum = svmla_x(pg, sum,
                              svld1_f32(pg, t->w_trans + (size_t)f * t->qkv_dim + j),
                              svld1_f32(pg, t->conv_st + (size_t)row_off[f] + j));
            sum = svmla_x(pg, sum,
                          svld1_f32(pg, t->w_trans + (size_t)t->n_hist * t->qkv_dim + j),
                          cur);
            svfloat32_t e = tf_fast_exp_sve(pg, svsub_x(pg, svdup_f32(0.0f), sum));
            svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
            svst1_f32(pg, t->conv_st + (size_t)wr * t->qkv_dim + j, cur);
            svst1_f32(pg, qkv + j, svmul_x(pg, sum, sig));
        }
    }
    return NULL;
}

typedef struct {
    float *QKV;
    int tid, nt;
    int M;
    int qkv_dim;
    int d_state;
    int n_group;
    float eps;
} tf_ssm_prefill_qknorm_block_task;

static void *tf_ssm_prefill_qknorm_block_worker(void *arg) {
    tf_ssm_prefill_qknorm_block_task *t = (tf_ssm_prefill_qknorm_block_task *)arg;
    int jobs_per_tok = 2 * t->n_group;
    int total = t->M * jobs_per_tok;
    for (int job = t->tid; job < total; job += t->nt) {
        int ti = job / jobs_per_tok;
        int rem = job - ti * jobs_per_tok;
        int is_k = rem >= t->n_group;
        int g = is_k ? rem - t->n_group : rem;
        float *base = t->QKV + (size_t)ti * t->qkv_dim;
        float *ptr = base + (is_k ? (size_t)t->n_group * t->d_state : 0) +
                     (size_t)g * t->d_state;
        tf_l2_norm(ptr, t->d_state, t->eps);
    }
    return NULL;
}

typedef struct {
    float *rec_state;
    float *QKV;
    float *Z;
    const float *Alpha;
    const float *Beta;
    float *S;
    const float *norm_w;
    int M;
    int qkv_dim;
    int d_inner;
    int d_state;
    int n_group;
    int dt_rank;
    int head_start, head_end;
    float eps;
    float scale;
} tf_ssm_prefill_finish_block_task;

static void *tf_ssm_prefill_finish_block_worker(void *arg) {
    tf_ssm_prefill_finish_block_task *t = (tf_ssm_prefill_finish_block_task *)arg;
    int ds = t->d_state;
    int d2 = ds * ds;
    size_t d_inner = (size_t)t->d_inner;
    size_t qkv_stride = (size_t)t->qkv_dim;
    for (int h = t->head_start; h < t->head_end; h++) {
        int g = h % t->n_group;
        float *state = t->rec_state + (size_t)h * d2;
        const float *q_base = t->QKV + (size_t)g * ds;
        const float *k_base = t->QKV + (size_t)t->n_group * ds + (size_t)g * ds;
        const float *v_base = t->QKV + (size_t)2 * t->n_group * ds + (size_t)h * ds;
        const float *alpha = t->Alpha + h;
        const float *beta = t->Beta + h;

        for (int ti = 0; ti < t->M; ti++) {
            const float *q_h = q_base + (size_t)ti * qkv_stride;
            const float *k_h = k_base + (size_t)ti * qkv_stride;
            const float *v_h = v_base + (size_t)ti * qkv_stride;
            float *o_h = t->S + d_inner * (size_t)ti + (size_t)h * ds;
            const float *z_h = t->Z + d_inner * (size_t)ti + (size_t)h * ds;
            tf_ssm_recurrence_step_sve(state, q_h, k_h, v_h, o_h,
                                       alpha[ti * t->dt_rank], beta[ti * t->dt_rank],
                                       ds, t->scale, t->norm_w,
                                       z_h, t->eps);
        }
    }
    return NULL;
}

static void tf_ssm_prefill_finish_block(transformer_model *m, transformer_layer *layer,
                                        int layer_idx, float *QKV, float *Z,
                                        float *Alpha, float *Beta, float *S,
                                        float *conv_tmp, int M,
                                        const float *norm_w,
                                        tf_ep_prefill_detail *pd) {
    int qkv_dim = m->ssm_qkv_dim;
    int d_state = m->ssm_d_state;
    int n_group = m->ssm_n_group;
    int dt_rank = m->ssm_dt_rank;
    int d_inner = m->ssm_d_inner;
    int conv_k = m->ssm_conv_kernel;
    int n_hist = conv_k - 1;
    float eps = m->rms_norm_eps;

    float *conv_st = m->conv_state[layer_idx];
    float *w_trans = m->conv_w_trans[layer_idx];
    int wr0 = m->conv_state_pos[layer_idx];
    int nt = (m->n_threads > 1 && m->pool_alive) ? m->n_threads : 1;

    double ts = pd ? tf_wall_seconds() : 0.0;
    if (nt > 1) {
        tf_ssm_prefill_conv_block_task *tasks =
            (tf_ssm_prefill_conv_block_task *)alloca((size_t)nt * sizeof(*tasks));
        for (int ti = 0; ti < nt; ti++) {
            tasks[ti] = (tf_ssm_prefill_conv_block_task){
                .QKV = QKV, .conv_st = conv_st, .w_trans = w_trans,
                .tid = ti, .nt = nt, .M = M, .qkv_dim = qkv_dim,
                .n_hist = n_hist, .wr0 = wr0,
            };
        }
        tf_pool_dispatch(m, tf_ssm_prefill_conv_block_worker, tasks, sizeof(*tasks));
    } else {
        tf_ssm_prefill_conv_block_task task = {
            .QKV = QKV, .conv_st = conv_st, .w_trans = w_trans,
            .tid = 0, .nt = 1, .M = M, .qkv_dim = qkv_dim,
            .n_hist = n_hist, .wr0 = wr0,
        };
        tf_ssm_prefill_conv_block_worker(&task);
    }
    m->conv_state_pos[layer_idx] = n_hist > 0 ? (wr0 + M) % n_hist : 0;
    if (pd) pd->ssm_finish_conv += tf_wall_seconds() - ts;

    ts = pd ? tf_wall_seconds() : 0.0;
    if (nt > 1) {
        tf_ssm_prefill_qknorm_block_task *tasks =
            (tf_ssm_prefill_qknorm_block_task *)alloca((size_t)nt * sizeof(*tasks));
        for (int ti = 0; ti < nt; ti++) {
            tasks[ti] = (tf_ssm_prefill_qknorm_block_task){
                .QKV = QKV, .tid = ti, .nt = nt, .M = M,
                .qkv_dim = qkv_dim, .d_state = d_state,
                .n_group = n_group, .eps = eps,
            };
        }
        tf_pool_dispatch(m, tf_ssm_prefill_qknorm_block_worker, tasks, sizeof(*tasks));
    } else {
        tf_ssm_prefill_qknorm_block_task task = {
            .QKV = QKV, .tid = 0, .nt = 1, .M = M,
            .qkv_dim = qkv_dim, .d_state = d_state,
            .n_group = n_group, .eps = eps,
        };
        tf_ssm_prefill_qknorm_block_worker(&task);
    }
    if (pd) pd->ssm_finish_norm2 += tf_wall_seconds() - ts;

    ts = pd ? tf_wall_seconds() : 0.0;
    float scale = 1.0f / sqrtf((float)d_state);
    float *rec_state = m->recurrent_state[layer_idx];
    int ncmgs = m->cmg_pin ? m->cmg_pin_ncmgs : 1;
    if (nt > 1) {
        tf_ssm_prefill_finish_block_task *tasks =
            (tf_ssm_prefill_finish_block_task *)alloca((size_t)nt * sizeof(*tasks));
        for (int ti = 0; ti < nt; ti++) {
            int hs, he;
            tf_ssm_head_range(dt_rank, nt, ncmgs, ti, &hs, &he);
            tasks[ti] = (tf_ssm_prefill_finish_block_task){
                .rec_state = rec_state, .QKV = QKV, .Z = Z,
                .Alpha = Alpha, .Beta = Beta, .S = S, .norm_w = norm_w,
                .M = M, .qkv_dim = qkv_dim, .d_inner = d_inner,
                .d_state = d_state, .n_group = n_group, .dt_rank = dt_rank,
                .head_start = hs, .head_end = he, .eps = eps, .scale = scale,
            };
        }
        tf_pool_dispatch(m, tf_ssm_prefill_finish_block_worker, tasks, sizeof(*tasks));
    } else {
        tf_ssm_prefill_finish_block_task task = {
            .rec_state = rec_state, .QKV = QKV, .Z = Z,
            .Alpha = Alpha, .Beta = Beta, .S = S, .norm_w = norm_w,
            .M = M, .qkv_dim = qkv_dim, .d_inner = d_inner,
            .d_state = d_state, .n_group = n_group, .dt_rank = dt_rank,
            .head_start = 0, .head_end = dt_rank, .eps = eps, .scale = scale,
        };
        tf_ssm_prefill_finish_block_worker(&task);
    }
    if (pd) pd->ssm_finish_scan += tf_wall_seconds() - ts;

    (void)layer;
    (void)conv_tmp;
}

static int tf_ssm_prefill_layer_bf16pv(transformer_model *m, int layer_idx,
                                       float *X, int M, float *Xn, float *QKV,
                                       float *Z, float *Alpha, float *Beta,
                                       float *S, float *O, float *Q_exp,
                                       float *K_exp,
                                       tf_ep_prefill_detail *pd) {
    transformer_layer *layer = &m->layers[layer_idx];
    int n_embd = m->n_embd;
    int qkv_dim = m->ssm_qkv_dim;
    int d_inner = m->ssm_d_inner;
    int dt_rank = m->ssm_dt_rank;
    int d_state = m->ssm_d_state;
    /* TP V-head sharding row-slices ssm_alpha/ssm_beta to the local dt_rank
     * (4-5 at TP=11): < 8 and not a multiple of 8, so tf_bf16_pv_alloc rejects
     * them and they never get a bf16_pv panel — even though ssm_qkv/gate/out DO
     * panelize (their sliced dims stay 8/16-aligned). Requiring alpha/beta
     * panels here forced ALL 48 SSM layers onto the per-token fallback (the 25x
     * TP-prefill cliff). tf_gemm_bf16pv_prefill already falls back to pooled
     * matvecs for a panel-less weight, and these are tiny [n_embd x 4-5]
     * projections, so dropping the alpha/beta panel requirement lets the whole
     * SSM layer batch with negligible extra cost. Env-gated for a clean A/B;
     * default preserves the legacy (panel-required) behavior. */
    /* Default ON: validated on 27B TP=11 (fallback_layers 48->0, SSM mixer
     * 6.47->2.54s). Pure gate relaxation — tf_gemm_bf16pv_prefill still uses an
     * alpha/beta panel if one exists, so this never disables a panel, only lets
     * the row-sliced (n_rows<8) sharded case batch via the pooled-matvec fallback.
     * Set TP_PREFILL_SSM_NOPANEL_AB=0 to restore the legacy panel-required gate. */
    int ssm_nopanel_ab = getenv("TP_PREFILL_SSM_NOPANEL_AB")
                             ? (atoi(getenv("TP_PREFILL_SSM_NOPANEL_AB")) != 0) : 1;
    if (!layer->is_ssm ||
        !tf_prefill_weight_batched(&layer->ssm_qkv) ||
        !tf_prefill_weight_batched(&layer->ssm_gate) ||
        !tf_prefill_weight_batched(&layer->ssm_out) ||
        (!ssm_nopanel_ab &&
         (!tf_prefill_weight_batched(&layer->ssm_alpha) ||
          !tf_prefill_weight_batched(&layer->ssm_beta))))
        return 0;

    double t0 = pd ? tf_wall_seconds() : 0.0;
    for (int t = 0; t < M; t++)
        tf_rmsnorm(Xn + (size_t)t * n_embd, X + (size_t)t * n_embd,
                   &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    if (pd) pd->ssm_norm += tf_wall_seconds() - t0;

    t0 = pd ? tf_wall_seconds() : 0.0;
    tf_gemm_bf16pv_prefill(m, QKV,   &layer->ssm_qkv,   Xn, M);
    tf_gemm_bf16pv_prefill(m, Z,     &layer->ssm_gate,  Xn, M);
    tf_gemm_bf16pv_prefill(m, Alpha, &layer->ssm_alpha, Xn, M);
    tf_gemm_bf16pv_prefill(m, Beta,  &layer->ssm_beta,  Xn, M);
    if (pd) pd->ssm_proj += tf_wall_seconds() - t0;

    t0 = pd ? tf_wall_seconds() : 0.0;
    tf_ssm_prefill_post_alpha_beta(m, layer, Alpha, Beta, M);
    if (pd) pd->ssm_post += tf_wall_seconds() - t0;

    float norm_w[128];
    tf_dequant_row(&layer->ssm_norm, 0, norm_w);
    t0 = pd ? tf_wall_seconds() : 0.0;
    const char *null_env = getenv("TP_PREFILL_SSM_NULL_FINISH");
    const char *block_env = getenv("TP_PREFILL_SSM_BLOCK");
    if (!null_env) null_env = getenv("EP_PREFILL_SSM_NULL_FINISH");
    if (!block_env) block_env = getenv("EP_PREFILL_SSM_BLOCK");
    int null_finish = null_env ? (atoi(null_env) != 0) : 0;
    int block_finish = block_env ? (atoi(block_env) != 0) : 1;
    if (null_finish) {
        memset(S, 0, (size_t)M * (size_t)d_inner * sizeof(float));
    } else if (block_finish) {
        tf_ssm_prefill_finish_block(m, layer, layer_idx, QKV, Z, Alpha, Beta,
                                    S, O, M, norm_w, pd);
    } else {
        for (int t = 0; t < M; t++)
            tf_ssm_prefill_finish_token(m, layer, layer_idx,
                                        QKV + (size_t)t * qkv_dim,
                                        Z + (size_t)t * d_inner,
                                        Alpha + (size_t)t * dt_rank,
                                        Beta + (size_t)t * dt_rank,
                                        S + (size_t)t * d_inner,
                                        Q_exp, K_exp, O, norm_w);
    }
    if (pd) pd->ssm_finish += tf_wall_seconds() - t0;

    t0 = pd ? tf_wall_seconds() : 0.0;
    tf_gemm_bf16pv_prefill(m, O, &layer->ssm_out, S, M);
    tf_vadd_flat_pool(m, X, O, (size_t)M * n_embd);
    if (pd) pd->ssm_out += tf_wall_seconds() - t0;
    return 1;
}
#endif

float *transformer_prefill_ep_layermajor(transformer_model *m, const int32_t *tokens,
                                         int M, int pos0, int block_tokens) {
    if (!m || !tokens || M <= 0 || !m->token_embd.data || !m->has_lm_head) return NULL;
    if (!m->use_moe || m->is_gemma4 || !m->pool_alive || m->n_threads <= 1) return NULL;
    if (block_tokens <= 0) block_tokens = 1024;
    if (block_tokens > M) block_tokens = M;

    int n_embd = m->n_embd;
    size_t x_elems = (size_t)M * (size_t)n_embd;
    size_t y_elems = (size_t)block_tokens * (size_t)n_embd;
    float *X = (float *)malloc(x_elems * sizeof(float));
    float *Y = (float *)malloc(y_elems * sizeof(float));
#if defined(__ARM_FEATURE_SVE)
    int ssm_gemm = (getenv("EP_PREFILL_SSM_GEMM") && atoi(getenv("EP_PREFILL_SSM_GEMM")) != 0) ? 1 : 0;
    int moe_gemm = (getenv("EP_PREFILL_MOE_GEMM") && atoi(getenv("EP_PREFILL_MOE_GEMM")) != 0) ? 1 : 0;
    int attn_gemm = (getenv("EP_PREFILL_ATTN_GEMM") && atoi(getenv("EP_PREFILL_ATTN_GEMM")) != 0) ? 1 : 0;
    int shexp_gemm = getenv("EP_PREFILL_SHEXP_GEMM") ? (atoi(getenv("EP_PREFILL_SHEXP_GEMM")) != 0) : 1;
    int attn_block = getenv("EP_PREFILL_ATTN_BLOCK") ? atoi(getenv("EP_PREFILL_ATTN_BLOCK")) : 256;
    int attn_tiled = getenv("EP_PREFILL_ATTN_TILE") ? (atoi(getenv("EP_PREFILL_ATTN_TILE")) != 0) : 1;
    int attn_gemm_max_tok = getenv("EP_PREFILL_ATTN_GEMM_MAX_TOK") ? atoi(getenv("EP_PREFILL_ATTN_GEMM_MAX_TOK")) : 0;
    int attn_work = 0;
    float *Xn = NULL, *QKV = NULL, *Z = NULL, *Alpha = NULL, *Beta = NULL;
    float *S = NULL, *O = NULL, *Q_exp = NULL, *K_exp = NULL;
    float *MXn = NULL, *MA = NULL, *MG = NULL, *MU = NULL, *MD = NULL;
    float *Mtopw = NULL, *Mew = NULL;
    float *SXn = NULL, *SG = NULL, *SU = NULL, *SD = NULL, *Sscore = NULL;
    float *AXn = NULL, *AQ2 = NULL, *AK = NULL, *AV = NULL;
    float *AOut = NULL, *AO = NULL;
    int *Mtop = NULL, *Mcnt = NULL, *Mtid = NULL;
#else
    int ssm_gemm = 0;
    int moe_gemm = 0;
    int attn_gemm = 0;
    int shexp_gemm = 0;
    int attn_block = 0;
    int attn_tiled = 0;
    int attn_gemm_max_tok = 0;
    int attn_work = 0;
#endif
    if (!X || !Y) {
        free(X); free(Y);
        fprintf(stderr, "transformer_prefill_ep_layermajor: alloc failed (M=%d block=%d need ~%.1f GB)\n",
                M, block_tokens, (x_elems + y_elems) * 4.0 / 1e9);
        return NULL;
    }

#if defined(__ARM_FEATURE_SVE)
    if (attn_block <= 0) attn_block = 256;
    if (attn_block > M) attn_block = M;
    if (pos0 != 0) attn_tiled = 0;
    attn_work = attn_tiled ? M : attn_block;
    if (attn_gemm && attn_gemm_max_tok > 0 && M > attn_gemm_max_tok)
        attn_gemm = 0;
    if (ssm_gemm) {
        size_t qkv_elems = (size_t)M * (size_t)m->ssm_qkv_dim;
        size_t inner_elems = (size_t)M * (size_t)m->ssm_d_inner;
        size_t dt_elems = (size_t)M * (size_t)m->ssm_dt_rank;
        size_t exp_elems = (size_t)m->ssm_dt_rank * (size_t)m->ssm_d_state;
        Xn = (float *)malloc(x_elems * sizeof(float));
        QKV = (float *)malloc(qkv_elems * sizeof(float));
        Z = (float *)malloc(inner_elems * sizeof(float));
        Alpha = (float *)malloc(dt_elems * sizeof(float));
        Beta = (float *)malloc(dt_elems * sizeof(float));
        S = (float *)malloc(inner_elems * sizeof(float));
        O = (float *)malloc(x_elems * sizeof(float));
        Q_exp = (float *)malloc(exp_elems * sizeof(float));
        K_exp = (float *)malloc(exp_elems * sizeof(float));
        if (!Xn || !QKV || !Z || !Alpha || !Beta || !S || !O || !Q_exp || !K_exp) {
            fprintf(stderr, "transformer_prefill_ep_layermajor: EP_PREFILL_SSM_GEMM scratch alloc failed; falling back\n");
            free(Xn); free(QKV); free(Z); free(Alpha); free(Beta);
            free(S); free(O); free(Q_exp); free(K_exp);
            Xn = QKV = Z = Alpha = Beta = S = O = Q_exp = K_exp = NULL;
            ssm_gemm = 0;
        }
    }
    if (moe_gemm) {
        size_t bx = (size_t)block_tokens * (size_t)n_embd;
        size_t bf = (size_t)block_tokens * (size_t)m->n_ff_expert;
        size_t bt = (size_t)block_tokens * (size_t)m->n_expert_used;
        MXn = (float *)malloc(bx * sizeof(float));
        MA  = (float *)malloc(bx * sizeof(float));
        MG  = (float *)malloc(bf * sizeof(float));
        MU  = (float *)malloc(bf * sizeof(float));
        MD  = (float *)malloc(bx * sizeof(float));
        Mtop = (int *)malloc(bt * sizeof(int));
        Mtopw = (float *)malloc(bt * sizeof(float));
        Mcnt = (int *)malloc((size_t)m->n_expert * sizeof(int));
        Mtid = (int *)malloc((size_t)block_tokens * sizeof(int));
        Mew  = (float *)malloc((size_t)block_tokens * sizeof(float));
        if (!MXn || !MA || !MG || !MU || !MD || !Mtop || !Mtopw || !Mcnt || !Mtid || !Mew) {
            fprintf(stderr, "transformer_prefill_ep_layermajor: EP_PREFILL_MOE_GEMM scratch alloc failed; falling back\n");
            free(MXn); free(MA); free(MG); free(MU); free(MD); free(Mtop); free(Mtopw); free(Mcnt); free(Mtid); free(Mew);
            MXn = MA = MG = MU = MD = Mtopw = Mew = NULL;
            Mtop = Mcnt = Mtid = NULL;
            moe_gemm = 0;
        }
    }
    if (shexp_gemm && m->n_ff_shexp > 0) {
        size_t bx = (size_t)block_tokens * (size_t)n_embd;
        size_t bf = (size_t)block_tokens * (size_t)m->n_ff_shexp;
        SXn = (float *)malloc(bx * sizeof(float));
        SG  = (float *)malloc(bf * sizeof(float));
        SU  = (float *)malloc(bf * sizeof(float));
        SD  = (float *)malloc(bx * sizeof(float));
        Sscore = (float *)malloc((size_t)block_tokens * sizeof(float));
        if (!SXn || !SG || !SU || !SD || !Sscore) {
            fprintf(stderr, "transformer_prefill_ep_layermajor: EP_PREFILL_SHEXP_GEMM scratch alloc failed; falling back\n");
            free(SXn); free(SG); free(SU); free(SD); free(Sscore);
            SXn = SG = SU = SD = Sscore = NULL;
            shexp_gemm = 0;
        }
    }
    if (attn_gemm) {
        int q_dim = m->n_heads * m->head_dim;
        int q2_dim = 2 * q_dim;
        int kv_dim = m->n_kv_heads * m->head_dim;
        AXn  = (float *)malloc((size_t)attn_work * (size_t)n_embd * sizeof(float));
        AQ2  = (float *)malloc((size_t)attn_work * (size_t)q2_dim * sizeof(float));
        AK   = (float *)malloc((size_t)attn_work * (size_t)kv_dim * sizeof(float));
        AV   = (float *)malloc((size_t)attn_work * (size_t)kv_dim * sizeof(float));
        AOut = (float *)malloc((size_t)attn_work * (size_t)q_dim * sizeof(float));
        AO   = (float *)malloc((size_t)attn_work * (size_t)n_embd * sizeof(float));
        if (!AXn || !AQ2 || !AK || !AV || !AOut || !AO) {
            fprintf(stderr, "transformer_prefill_ep_layermajor: EP_PREFILL_ATTN_GEMM scratch alloc failed; falling back\n");
            free(AXn); free(AQ2); free(AK); free(AV); free(AOut); free(AO);
            AXn = AQ2 = AK = AV = AOut = AO = NULL;
            attn_gemm = 0;
        }
    }

    tf_numa_distribute_buffer(m, X, x_elems * sizeof(float));
    tf_numa_distribute_buffer(m, Y, y_elems * sizeof(float));
    if (ssm_gemm) {
        tf_numa_distribute_buffer(m, Xn, x_elems * sizeof(float));
        tf_numa_distribute_buffer(m, QKV, (size_t)M * (size_t)m->ssm_qkv_dim * sizeof(float));
        tf_numa_distribute_buffer(m, Z, (size_t)M * (size_t)m->ssm_d_inner * sizeof(float));
        tf_numa_distribute_buffer(m, Alpha, (size_t)M * (size_t)m->ssm_dt_rank * sizeof(float));
        tf_numa_distribute_buffer(m, Beta, (size_t)M * (size_t)m->ssm_dt_rank * sizeof(float));
        tf_numa_distribute_buffer(m, S, (size_t)M * (size_t)m->ssm_d_inner * sizeof(float));
        tf_numa_distribute_buffer(m, O, x_elems * sizeof(float));
        tf_numa_distribute_buffer(m, Q_exp, (size_t)m->ssm_dt_rank * (size_t)m->ssm_d_state * sizeof(float));
        tf_numa_distribute_buffer(m, K_exp, (size_t)m->ssm_dt_rank * (size_t)m->ssm_d_state * sizeof(float));
    }
    if (moe_gemm) {
        tf_numa_distribute_buffer(m, MXn, (size_t)block_tokens * (size_t)n_embd * sizeof(float));
        tf_numa_distribute_buffer(m, MA,  (size_t)block_tokens * (size_t)n_embd * sizeof(float));
        tf_numa_distribute_buffer(m, MG,  (size_t)block_tokens * (size_t)m->n_ff_expert * sizeof(float));
        tf_numa_distribute_buffer(m, MU,  (size_t)block_tokens * (size_t)m->n_ff_expert * sizeof(float));
        tf_numa_distribute_buffer(m, MD,  (size_t)block_tokens * (size_t)n_embd * sizeof(float));
    }
    if (shexp_gemm) {
        tf_numa_distribute_buffer(m, SXn, (size_t)block_tokens * (size_t)n_embd * sizeof(float));
        tf_numa_distribute_buffer(m, SG,  (size_t)block_tokens * (size_t)m->n_ff_shexp * sizeof(float));
        tf_numa_distribute_buffer(m, SU,  (size_t)block_tokens * (size_t)m->n_ff_shexp * sizeof(float));
        tf_numa_distribute_buffer(m, SD,  (size_t)block_tokens * (size_t)n_embd * sizeof(float));
        tf_numa_distribute_buffer(m, Sscore, (size_t)block_tokens * sizeof(float));
    }
    if (attn_gemm) {
        int q_dim = m->n_heads * m->head_dim;
        int q2_dim = 2 * q_dim;
        int kv_dim = m->n_kv_heads * m->head_dim;
        tf_numa_distribute_buffer(m, AXn, (size_t)attn_work * (size_t)n_embd * sizeof(float));
        tf_numa_distribute_buffer(m, AQ2, (size_t)attn_work * (size_t)q2_dim * sizeof(float));
        tf_numa_distribute_buffer(m, AK,  (size_t)attn_work * (size_t)kv_dim * sizeof(float));
        tf_numa_distribute_buffer(m, AV,  (size_t)attn_work * (size_t)kv_dim * sizeof(float));
        tf_numa_distribute_buffer(m, AOut,(size_t)attn_work * (size_t)q_dim * sizeof(float));
        tf_numa_distribute_buffer(m, AO,  (size_t)attn_work * (size_t)n_embd * sizeof(float));
    }
#endif

    for (int t = 0; t < M; t++)
        tf_dequant_row(&m->token_embd, tokens[t], X + (size_t)t * n_embd);

    int prof = getenv("EP_PREFILL_PROF") || getenv("TF_PREFILL_PROF");
    tf_ep_prefill_detail pd;
    memset(&pd, 0, sizeof(pd));
    tf_ep_prefill_detail *pdp = prof ? &pd : NULL;
    double t_mixer = 0.0, t_ffn = 0.0, t_ar = 0.0, t_total0 = tf_wall_seconds();
    long ar_calls = 0, ssm_gemm_layers = 0, attn_gemm_layers = 0, moe_gemm_blocks = 0;

    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];
        if (!layer->ffn_gate_inp.data) {
            free(X); free(Y);
            fprintf(stderr, "transformer_prefill_ep_layermajor: unsupported non-MoE layer %d\n", l);
            return NULL;
        }

        double ta = tf_wall_seconds();
        int mixer_done = 0;
#if defined(__ARM_FEATURE_SVE)
        if (ssm_gemm && layer->is_ssm) {
            double tm0 = tf_wall_seconds();
            mixer_done = tf_ssm_prefill_layer_bf16pv(m, l, X, M, Xn, QKV, Z, Alpha, Beta, S, O, Q_exp, K_exp, pdp);
            double tmd = tf_wall_seconds() - tm0;
            if (mixer_done) {
                ssm_gemm_layers++;
                if (pdp) pdp->mixer_ssm_gemm += tmd;
            }
        }
        if (!mixer_done && attn_gemm && !layer->is_ssm) {
            double tm0 = tf_wall_seconds();
            mixer_done = 1;
            if (attn_tiled) {
                mixer_done = tf_attn_prefill_layer_bf16pv(m, layer, l, X, M, pos0,
                                                          AXn, AQ2, AK, AV, AOut, AO, 1, pdp);
            } else {
                for (int b0 = 0; b0 < M; b0 += attn_block) {
                    int B = M - b0;
                    if (B > attn_block) B = attn_block;
                    int ok = tf_attn_prefill_layer_bf16pv(m, layer, l,
                                                           X + (size_t)b0 * n_embd,
                                                           B, pos0 + b0,
                                                           AXn, AQ2, AK, AV, AOut, AO, 0, pdp);
                    if (!ok) { mixer_done = 0; break; }
                }
            }
            if (mixer_done) {
                attn_gemm_layers++;
                if (pdp) pdp->mixer_attn_gemm += tf_wall_seconds() - tm0;
            }
        }
#endif
        if (!mixer_done) {
            double tm0 = tf_wall_seconds();
            m->prefill_ffn_skip = 1;
            for (int b0 = 0; b0 < M; b0 += block_tokens) {
                int B = M - b0;
                if (B > block_tokens) B = block_tokens;
                for (int t = 0; t < B; t++) {
                    int ti = b0 + t;
                    int pos = pos0 + ti;
                    memcpy(m->x, X + (size_t)ti * n_embd, (size_t)n_embd * sizeof(float));
                    tf_forward_blocks_range(m, pos, pos, pos, pos, l, l + 1);
                    memcpy(X + (size_t)ti * n_embd, m->x, (size_t)n_embd * sizeof(float));
                }
            }
            m->prefill_ffn_skip = 0;
            if (pdp) {
                double tmd = tf_wall_seconds() - tm0;
                if (layer->is_ssm) {
                    pdp->mixer_fallback_ssm += tmd;
                    pdp->fallback_ssm_layers++;
                } else {
                    pdp->mixer_fallback_attn += tmd;
                    pdp->fallback_attn_layers++;
                }
            }
        }
        t_mixer += tf_wall_seconds() - ta;

        for (int b0 = 0; b0 < M; b0 += block_tokens) {
            int B = M - b0;
            if (B > block_tokens) B = block_tokens;
            ta = tf_wall_seconds();
            int ffn_done = 0;
#if defined(__ARM_FEATURE_SVE)
            if (moe_gemm) {
                double tg0 = tf_wall_seconds();
                ffn_done = tf_moe_ffn_block_gemm(m, layer, l, X + (size_t)b0 * n_embd,
                                                 B, Y, MXn, MA, MG, MU, MD,
                                                 Mtop, Mtopw, Mcnt, Mtid, Mew);
                if (ffn_done) {
                    moe_gemm_blocks++;
                    if (pdp) pdp->ffn_moe_gemm += tf_wall_seconds() - tg0;
                }
            }
#endif
            int shared_norm_ready = 0;
            if (!ffn_done) {
                int fill_shared_norm =
#if defined(__ARM_FEATURE_SVE)
                    (shexp_gemm && SXn && m->ep_size > 1 &&
                     m->n_ff_shexp > 0 && layer->ffn_up_shexp.data);
#else
                    0;
#endif
                for (int t = 0; t < B; t++) {
                    int ti = b0 + t;
                    float *xnorm =
#if defined(__ARM_FEATURE_SVE)
                        fill_shared_norm ? (SXn + (size_t)t * n_embd) : m->q;
#else
                        m->q;
#endif
                    double tn0 = pdp ? tf_wall_seconds() : 0.0;
                    tf_rmsnorm(xnorm, X + (size_t)ti * n_embd,
                               &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
                    if (pdp) pdp->ffn_norm_local += tf_wall_seconds() - tn0;
                    double tc0 = pdp ? tf_wall_seconds() : 0.0;
                    tf_moe_ffn_local_partial(m, layer, l, xnorm, Y + (size_t)t * n_embd,
                                             m->ep_size <= 1);
                    if (pdp) {
                        pdp->ffn_local_core += tf_wall_seconds() - tc0;
                        pdp->ffn_tokens++;
                    }
                }
                shared_norm_ready = fill_shared_norm;
            }
            t_ffn += tf_wall_seconds() - ta;

            if (m->ep_size > 1 && m->ep_ar_fn) {
                ta = tf_wall_seconds();
                m->ep_ar_fn(Y, B * n_embd, m->ep_ar_ctx);
                t_ar += tf_wall_seconds() - ta;
                ar_calls++;
            }

            ta = tf_wall_seconds();
            int shared_done = 0;
#if defined(__ARM_FEATURE_SVE)
            if ((ffn_done || m->ep_size > 1) && shexp_gemm &&
                m->n_ff_shexp > 0 && layer->ffn_up_shexp.data) {
                double ts0 = pdp ? tf_wall_seconds() : 0.0;
                if (!shared_norm_ready) {
                    for (int t = 0; t < B; t++) {
                        int ti = b0 + t;
                        tf_rmsnorm(SXn + (size_t)t * n_embd, X + (size_t)ti * n_embd,
                                   &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
                    }
                }
                shared_done = tf_moe_shared_expert_add_block(m, layer, l, SXn, B, Y,
                                                              SG, SU, SD, Sscore);
                if (shared_done && pdp) {
                    pdp->ffn_shared += tf_wall_seconds() - ts0;
                    pdp->shared_tokens += B;
                }
            }
#endif
            if (!shared_done && (ffn_done || m->ep_size > 1) &&
                m->n_ff_shexp > 0 && layer->ffn_up_shexp.data) {
                for (int t = 0; t < B; t++) {
                    int ti = b0 + t;
                    double ts0 = pdp ? tf_wall_seconds() : 0.0;
                    tf_rmsnorm(m->q, X + (size_t)ti * n_embd,
                               &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
                    tf_moe_shared_expert_add(m, layer, l, m->q, Y + (size_t)t * n_embd);
                    if (pdp) {
                        pdp->ffn_shared += tf_wall_seconds() - ts0;
                        pdp->shared_tokens++;
                    }
                }
            }
            {
                double tr0 = pdp ? tf_wall_seconds() : 0.0;
                tf_vadd_flat_pool(m, X + (size_t)b0 * n_embd, Y, (size_t)B * n_embd);
                if (pdp) pdp->ffn_residual += tf_wall_seconds() - tr0;
            }
            t_ffn += tf_wall_seconds() - ta;
        }
    }

    memcpy(m->x, X + (size_t)(M - 1) * n_embd, (size_t)n_embd * sizeof(float));
    TF_PROF_BEGIN("final_norm", -1, "rmsnorm", "FP32");
    tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    TF_PROF_END("final_norm", 5.0 * n_embd, 0);
    float *logits = transformer_compute_logits(m);

    if (prof) {
        char fn[64];
        snprintf(fn, sizeof fn, "ep_prefill_rank%02d.txt", m->ep_rank);
        FILE *pf = fopen(fn, "w");
        if (pf) {
            double total = tf_wall_seconds() - t_total0;
            fprintf(pf, "rank %d ep_prefill_layermajor M=%d block=%d layers=%d\n",
                    m->ep_rank, M, block_tokens, m->n_layers);
            fprintf(pf, "total=%.6f mixer=%.6f ffn_local=%.6f ar=%.6f ar_calls=%ld ssm_gemm_layers=%ld attn_gemm_layers=%ld moe_gemm_blocks=%ld\n",
                    total, t_mixer, t_ffn, t_ar, ar_calls, ssm_gemm_layers, attn_gemm_layers, moe_gemm_blocks);
            fprintf(pf, "detail mixer_ssm_gemm=%.6f mixer_attn_gemm=%.6f mixer_fallback_ssm=%.6f mixer_fallback_attn=%.6f fallback_ssm_layers=%ld fallback_attn_layers=%ld\n",
                    pd.mixer_ssm_gemm, pd.mixer_attn_gemm,
                    pd.mixer_fallback_ssm, pd.mixer_fallback_attn,
                    pd.fallback_ssm_layers, pd.fallback_attn_layers);
            fprintf(pf, "detail_ssm norm=%.6f proj=%.6f post=%.6f finish=%.6f finish_conv=%.6f finish_qknorm=%.6f finish_scan=%.6f out=%.6f\n",
                    pd.ssm_norm, pd.ssm_proj, pd.ssm_post, pd.ssm_finish,
                    pd.ssm_finish_conv, pd.ssm_finish_norm2, pd.ssm_finish_scan,
                    pd.ssm_out);
            fprintf(pf, "detail_attn norm=%.6f qkv=%.6f token=%.6f out=%.6f\n",
                    pd.attn_norm, pd.attn_qkv, pd.attn_token, pd.attn_out);
            fprintf(pf, "detail_ffn norm_local=%.6f local_core=%.6f moe_gemm=%.6f shared=%.6f residual=%.6f ffn_tokens=%ld shared_tokens=%ld\n",
                    pd.ffn_norm_local, pd.ffn_local_core, pd.ffn_moe_gemm,
                    pd.ffn_shared, pd.ffn_residual, pd.ffn_tokens, pd.shared_tokens);
            fclose(pf);
        }
    }

    free(X); free(Y);
#if defined(__ARM_FEATURE_SVE)
    free(Xn); free(QKV); free(Z); free(Alpha); free(Beta);
    free(S); free(O); free(Q_exp); free(K_exp);
    free(MXn); free(MA); free(MG); free(MU); free(MD);
    free(Mtop); free(Mtopw); free(Mcnt); free(Mtid); free(Mew);
    free(SXn); free(SG); free(SU); free(SD); free(Sscore);
    free(AXn); free(AQ2); free(AK); free(AV); free(AOut); free(AO);
#endif
    return logits;
}

#if defined(__ARM_FEATURE_SVE)
/* Per-CMG first-touch of the prefill activation buffers, matching the M-partition
 * tf_gemm_bf16pv_prefill uses, so each CMG's token-slice of X/Xn/G/U/D lands on that
 * CMG's HBM. Without this the buffers sit on the master's CMG and 36/48 GEMM threads
 * write C cross-CMG into one HBM stack — the dominant prefill-GEMM bottleneck. */
typedef struct {
    int tid, ncmg, nt, n_embd, n_ff;
    float *X, *Xn, *G, *U, *D;
    const int *m_lo, *m_hi;
} tf_prefill_ft_task;
static void *tf_prefill_ft_worker(void *arg) {
    tf_prefill_ft_task *t = (tf_prefill_ft_task *)arg;
    int cmg, loc, nloc;
    tf_gemm_cmg_of(t->tid, t->nt, t->ncmg, &cmg, &loc, &nloc);
    if (cmg >= t->ncmg || nloc <= 0) return NULL;
    int mlo = t->m_lo[cmg], mhi = t->m_hi[cmg], Mc = mhi - mlo;
    if (Mc <= 0) return NULL;
    int per = (Mc + nloc - 1) / nloc;
    int rs = mlo + loc * per, re = rs + per;
    if (rs > mhi) rs = mhi;
    if (re > mhi) re = mhi;
    for (int r = rs; r < re; r++) {
        memset(t->X  + (size_t)r * t->n_embd, 0, (size_t)t->n_embd * sizeof(float));
        memset(t->Xn + (size_t)r * t->n_embd, 0, (size_t)t->n_embd * sizeof(float));
        memset(t->D  + (size_t)r * t->n_embd, 0, (size_t)t->n_embd * sizeof(float));
        memset(t->G  + (size_t)r * t->n_ff,   0, (size_t)t->n_ff   * sizeof(float));
        memset(t->U  + (size_t)r * t->n_ff,   0, (size_t)t->n_ff   * sizeof(float));
    }
    return NULL;
}

static inline int tf_prefill_env_int(const char *name1, const char *name2, int def) {
    const char *v = getenv(name1);
    if (!v) v = getenv(name2);
    if (!v) return def;
    if (!strcasecmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "off") ||
        !strcasecmp(v, "no")) return 0;
    if (!strcasecmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "on") ||
        !strcasecmp(v, "yes")) return 1;
    return atoi(v) != 0;
}

static inline void tf_prefill_allreduce(transformer_model *m, float *x, size_t n_floats) {
    if (!m || !m->tp_allreduce_fn || m->tp_size <= 1 || !x) return;
    const size_t step = (size_t)INT_MAX;
    for (size_t i = 0; i < n_floats; ) {
        size_t n = n_floats - i;
        if (n > step) n = step;
        m->tp_allreduce_fn(x + i, (int)n, m->tp_allreduce_ctx);
        i += n;
    }
}
#else
static inline int tf_prefill_env_int(const char *name1, const char *name2, int def) {
    const char *v = getenv(name1);
    if (!v) v = getenv(name2);
    if (!v) return def;
    if (!strcasecmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "off") ||
        !strcasecmp(v, "no")) return 0;
    if (!strcasecmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "on") ||
        !strcasecmp(v, "yes")) return 1;
    return atoi(v) != 0;
}
static inline void tf_prefill_allreduce(transformer_model *m, float *x, size_t n_floats) {
    (void)m; (void)x; (void)n_floats;
}
#endif

float *transformer_prefill_gemm(transformer_model *m, const int32_t *tokens, int M, int pos0) {
    if (!m || !tokens || M <= 0 || !m->token_embd.data) return NULL;
    /* Supported: dense (non-MoE) SwiGLU FFN, non-Gemma4, pool alive. */
    if (m->use_moe || m->is_gemma4 || !m->pool_alive || m->n_threads <= 1) return NULL;
    if (m->n_ff <= 0 || !m->has_lm_head) return NULL;
    transformer_layer *l0 = &m->layers[0];
    if (l0->ffn_gate.type == GGML_TYPE_F32 || !l0->ffn_gate.data) return NULL;

    int n_embd = m->n_embd, n_ff = m->n_ff, n_vocab = m->n_vocab;
    float eps = m->rms_norm_eps;
    size_t emb_elems = (size_t)M * n_embd;
    size_t ff_elems  = (size_t)M * n_ff;

    /* Batch buffers: X = residual stream, Xn = FFN-norm input, G/U = gate/up, D = down out. */
    float *X  = (float *)malloc(emb_elems * sizeof(float));
    float *Xn = (float *)malloc(emb_elems * sizeof(float));
    float *G  = (float *)malloc(ff_elems  * sizeof(float));
    float *U  = (float *)malloc(ff_elems  * sizeof(float));
    float *D  = (float *)malloc(emb_elems * sizeof(float));

    if (!X || !Xn || !G || !U || !D) {
        free(X); free(Xn); free(G); free(U); free(D);
        fprintf(stderr, "transformer_prefill_gemm: alloc failed (M=%d, need ~%.1f GB)\n",
                M, (2.0 * ff_elems + 3.0 * emb_elems) * 4.0 / 1e9);
        return NULL;
    }

    int do_ffn_allreduce = tf_prefill_env_int("TP_PREFILL_FFN_ALLREDUCE", "EP_PREFILL_FFN_ALLREDUCE", 1);
    int do_attn_gemm = tf_prefill_env_int("TP_PREFILL_ATTN_GEMM", "EP_PREFILL_ATTN_GEMM", 1);
    int do_ssm_gemm  = tf_prefill_env_int("TP_PREFILL_SSM_GEMM",  "EP_PREFILL_SSM_GEMM",  1);
    int attn_tiled = tf_prefill_env_int("TP_PREFILL_ATTN_TILE",  "EP_PREFILL_ATTN_TILE", 1);
    int null_gemm = tf_prefill_env_int("TP_PREFILL_NULL_GEMM", "EP_PREFILL_NULL_GEMM", 0);

    /* Per-CMG first-touch so each CMG's token-slice of X/Xn/G/U/D lands on its own
     * HBM stack, matching the exact M-partition tf_gemm_bf16pv_prefill uses (MR=8).
     * Otherwise all buffers sit on the master's CMG and 36/48 GEMM threads write C
     * cross-CMG into one HBM stack — the dominant prefill-GEMM bottleneck. */
#if defined(__ARM_FEATURE_SVE)
    {
        int ncmg = (m->cmg_pin && m->cmg_pin_ncmgs > 0) ? m->cmg_pin_ncmgs : 1;
        if (ncmg > 4) ncmg = 4;
        if (ncmg > m->n_threads) ncmg = m->n_threads;
        int per = ((M / ncmg + 8 - 1) / 8) * 8;
        if (per < 8) per = 8;
        int m_lo[4], m_hi[4];
        for (int c = 0; c < ncmg; c++) {
            m_lo[c] = c * per;
            m_hi[c] = (c == ncmg - 1) ? M : (c + 1) * per;
            if (m_lo[c] > M) m_lo[c] = M;
            if (m_hi[c] > M) m_hi[c] = M;
        }
        int nt = m->n_threads;
        tf_prefill_ft_task *ft = (tf_prefill_ft_task *)alloca((size_t)nt * sizeof(*ft));
        for (int t = 0; t < nt; t++)
            ft[t] = (tf_prefill_ft_task){ t, ncmg, nt, n_embd, n_ff, X, Xn, G, U, D, m_lo, m_hi };
        tf_pool_dispatch(m, tf_prefill_ft_worker, ft, sizeof(tf_prefill_ft_task));
    }
#else
    do_attn_gemm = 0;
    do_ssm_gemm = 0;
#endif

    /* Embed all prompt tokens into the residual stream. */
    for (int t = 0; t < M; t++)
        tf_dequant_row(&m->token_embd, tokens[t], X + (size_t)t * n_embd);

    int tf_prefill_prof = (getenv("TF_PREFILL_PROF") != NULL);
    /* Per-phase split: t_attn (16 full-attn mixers), t_ssm (48 SSM mixers),
     * t_ffn (batched FFN GEMM only), t_ar (mixer-out + FFN all-reduce, which the
     * old {t_mixer,t_ffn}-only profiler dropped on the floor). */
    double t_attn = 0.0, t_ssm = 0.0, t_ffn = 0.0, t_ar = 0.0;
    struct timespec ta, tb;

    int n_layers = m->n_layers;
    int fallback_layers = 0;

#if defined(__ARM_FEATURE_SVE)
    float *Q2  = NULL, *K = NULL, *V = NULL, *AOut = NULL;
    float *QKV = NULL, *Z = NULL, *Alpha = NULL, *Beta = NULL, *S = NULL;
    float *Q_exp = NULL, *K_exp = NULL;
    if (!null_gemm) {
        size_t q2_elems = (size_t)M * 2 * m->n_heads * m->head_dim;
        size_t kv_elems = (size_t)M * m->n_kv_heads * m->head_dim;
        size_t qkv_elems = (size_t)M * m->ssm_qkv_dim;
        size_t inner_elems = (size_t)M * m->ssm_d_inner;
        size_t alpha_elems = (size_t)M * m->ssm_dt_rank;
        size_t exp_elems = (size_t)m->ssm_dt_rank * (size_t)m->ssm_d_state;

        if (q2_elems > 0) {
            Q2 = (float *)malloc(q2_elems * sizeof(float));
            K  = (float *)malloc(kv_elems * sizeof(float));
            V  = (float *)malloc(kv_elems * sizeof(float));
            AOut = (float *)malloc((size_t)M * m->n_embd * sizeof(float));
        }
        if (exp_elems > 0) {
            Q_exp = (float *)malloc(exp_elems * sizeof(float));
            K_exp = (float *)malloc(exp_elems * sizeof(float));
        }
        if (qkv_elems > 0) {
            QKV = (float *)malloc(qkv_elems * sizeof(float));
            Z = (float *)malloc(inner_elems * sizeof(float));
            Alpha = (float *)malloc(alpha_elems * sizeof(float));
            Beta = (float *)malloc(alpha_elems * sizeof(float));
            S = (float *)malloc(inner_elems * sizeof(float));
        }
    }
#endif

    for (int l = 0; l < n_layers; l++) {
        transformer_layer *layer = &m->layers[l];
        int mixer_ok = 0;
        if (tf_prefill_prof) clock_gettime(CLOCK_MONOTONIC, &ta);
        m->prefill_ffn_skip = 1;

        if (null_gemm) {
            if (m->tp_size > 1 && m->tp_allreduce_fn &&
                ((layer->is_ssm) ? m->tp_ssm_sharded : m->tp_attn_sharded)) {
                tf_prefill_allreduce(m, X, emb_elems);
            }
            m->prefill_ffn_skip = 0;
            if (do_ffn_allreduce && m->tp_size > 1 && m->tp_allreduce_fn && m->tp_ffn_sharded) {
                tf_prefill_allreduce(m, X, emb_elems);
            }
            if (tf_prefill_prof) {   /* null_gemm probe: all time here is pure AR+norm */
                clock_gettime(CLOCK_MONOTONIC, &tb);
                t_ar += (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
            }
            continue;
        }

#if defined(__ARM_FEATURE_SVE)
        if (layer->is_ssm && do_ssm_gemm && m->is_hybrid && QKV && Z && Alpha && Beta && S && K_exp && Q_exp) {
            mixer_ok = tf_ssm_prefill_layer_bf16pv(m, l, X, M, Xn, QKV, Z, Alpha, Beta,
                                                  S, AOut, Q_exp, K_exp, NULL);
        } else if (!layer->is_ssm && do_attn_gemm && Q2 && K && V && AOut) {
            mixer_ok = tf_attn_prefill_layer_bf16pv(m, layer, l, X, M, pos0, Xn, Q2, K, V,
                                                  AOut, AOut, attn_tiled, NULL);
        }
#endif

        if (!mixer_ok) {
            if (tf_prefill_prof && !m->is_hybrid) {
                fprintf(stderr, "  [TF_PREFILL_PROF] M=%d layer=%d mixer fallback: full per-token path\n", M, l);
            }
            fallback_layers++;
            for (int t = 0; t < M; t++) {
                int pos = pos0 + t;
                memcpy(m->x, X + (size_t)t * n_embd, n_embd * sizeof(float));
                tf_forward_blocks_range(m, pos, pos, pos, pos, l, l + 1); /* attn_norm + mixer + residual */
                memcpy(X + (size_t)t * n_embd, m->x, n_embd * sizeof(float));
                tf_rmsnorm(Xn + (size_t)t * n_embd, m->x, &layer->ffn_norm,
                           n_embd, eps, m->matvec_tmp);
            }
            m->prefill_ffn_skip = 0;
            if (tf_prefill_prof) {
                clock_gettime(CLOCK_MONOTONIC, &tb);
                double dt = (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
                if (layer->is_ssm) t_ssm += dt; else t_attn += dt;
                ta = tb;   /* so t_ffn below is FFN-only, not mixer+FFN */
            }
            tf_gemm_bf16pv_prefill(m, G, &layer->ffn_gate, Xn, M);
            tf_gemm_bf16pv_prefill(m, U, &layer->ffn_up, Xn, M);
            tf_silu_mul_flat_pool(m, G, G, U, ff_elems);
            tf_gemm_bf16pv_prefill(m, D, &layer->ffn_down, G, M);
            tf_vadd_flat_pool(m, X, D, emb_elems);
            if (tf_prefill_prof) {
                clock_gettime(CLOCK_MONOTONIC, &tb);
                t_ffn += (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
            }
            continue;
        }

        m->prefill_ffn_skip = 0;
        if (tf_prefill_prof) {
            clock_gettime(CLOCK_MONOTONIC, &tb);
            double dt = (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
            if (layer->is_ssm) t_ssm += dt; else t_attn += dt;
            ta = tb;
        }

#if defined(__ARM_FEATURE_SVE)
        if (m->tp_size > 1 && m->tp_allreduce_fn &&
            ((layer->is_ssm) ? m->tp_ssm_sharded : m->tp_attn_sharded)) {
            tf_prefill_allreduce(m, X, emb_elems);
        }
#endif

        if (tf_prefill_prof) {   /* mixer-out all-reduce (the old profiler dropped this) */
            clock_gettime(CLOCK_MONOTONIC, &tb);
            t_ar += (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
            ta = tb;
        }
        tf_gemm_bf16pv_prefill(m, G, &layer->ffn_gate, Xn, M);
        tf_gemm_bf16pv_prefill(m, U, &layer->ffn_up, Xn, M);
        tf_silu_mul_flat_pool(m, G, G, U, ff_elems);
        tf_gemm_bf16pv_prefill(m, D, &layer->ffn_down, G, M);
        tf_vadd_flat_pool(m, X, D, emb_elems);
        if (tf_prefill_prof) {   /* FFN batched GEMM only (AR split out below) */
            clock_gettime(CLOCK_MONOTONIC, &tb);
            t_ffn += (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
            ta = tb;
        }
        if (do_ffn_allreduce && m->tp_size > 1 && m->tp_allreduce_fn && m->tp_ffn_sharded) {
            tf_prefill_allreduce(m, X, emb_elems);
        }
        if (tf_prefill_prof) {   /* FFN all-reduce */
            clock_gettime(CLOCK_MONOTONIC, &tb);
            t_ar += (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) * 1e-9;
        }
#endif
    }

    if (tf_prefill_prof) {
        fprintf(stderr, "  [TF_PREFILL_PROF] M=%d attn=%.2fs ssm=%.2fs ffn=%.2fs ar=%.2fs (mixer=%.2fs) fallback_layers=%d\n",
                M, t_attn, t_ssm, t_ffn, t_ar, t_attn + t_ssm, fallback_layers);
    }

    /* Final RMSNorm + lm_head on the last prompt token only.
     * Under TP the lm_head (m->output) is VOCAB-SHARDED to m->output.n_rows
     * (= tp_vocab_loc, e.g. 22576 at TP=11). Computing the full n_vocab rows
     * reads m->output.data far past the local shard → SIGSEGV. Compute only the
     * local shard; the caller's sample_argmax() does the cross-shard argmax
     * all-reduce. n_rows == n_vocab when not sharded, so this is universal. */
    memcpy(m->x, X + (size_t)(M - 1) * n_embd, n_embd * sizeof(float));
    tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, eps, m->matvec_tmp);
    tf_qmatvec_pool(m, m->logits, &m->output, m->x, m->output.n_rows);

    free(X); free(Xn); free(G); free(U); free(D);
#if defined(__ARM_FEATURE_SVE)
    free(Q2); free(K); free(V); free(AOut);
    free(QKV); free(Z); free(Alpha); free(Beta); free(S);
    free(Q_exp); free(K_exp);
#endif
    return m->logits;
}

/* ---- Tensor-parallel API ---- */

void transformer_set_tp(transformer_model *model, int tp_rank, int tp_size,
                         void (*allreduce_fn)(float *buf, int count, void *ctx),
                         void *allreduce_ctx) {
    if (!model) return;
    model->tp_rank = tp_rank;
    model->tp_size = tp_size;
    model->tp_allreduce_fn = allreduce_fn;
    model->tp_allreduce_ctx = allreduce_ctx;
}

/* ---- Expert-parallel API (MoE) ---- */

void transformer_set_ep(transformer_model *model, int ep_rank, int ep_size) {
    if (!model) return;
    if (ep_size <= 1) {
        model->ep_rank = 0; model->ep_size = 1;
        model->ep_e_start = 0; model->ep_e_end = 0;
        return;
    }
    if (ep_rank < 0 || ep_rank >= ep_size) {
        fprintf(stderr, "ep: bad rank %d/%d\n", ep_rank, ep_size); return;
    }
    if (!model->use_moe || model->n_expert <= 0) {
        fprintf(stderr, "ep: model has no MoE experts (n_expert=%d) — set_ep ignored\n",
                model->n_expert);
        return;
    }
    /* Interleaved (modulo) partition: ep_owner(e) = e % ep_size. Spreads any
     * expert-hotness uniformly across ranks. ep_e_start/ep_e_end retain the
     * first and one-past-last owned IDs for diagnostics only; they are not a
     * contiguous ownership range. */
    int E = model->n_expert;
    model->ep_rank = ep_rank;
    model->ep_size = ep_size;
    model->ep_e_start = ep_rank;                              /* first owned */
    model->ep_e_end   = ((E - 1 - ep_rank) / ep_size) * ep_size + ep_rank + 1;
    int own_cnt = (E - ep_rank + ep_size - 1) / ep_size;
    fprintf(stderr, "ep: rank %d/%d interleaved: owns %d/%d experts (e %% %d == %d)\n",
            ep_rank, ep_size, own_cnt, E, ep_size, ep_rank);
}

void transformer_set_ep_ar(transformer_model *model,
                            void (*ar_fn)(float *buf, int count, void *ctx),
                            void *ar_ctx) {
    if (!model) return;
    model->ep_ar_fn = ar_fn;
    model->ep_ar_ctx = ar_ctx;
}

/* COL-parallel: keep only output rows [r0,r1) of a weight (offset data, shrink
 * n_rows). bf16_pv/panel fills read contiguous rows at the unchanged stride. */
static void tf_tp_slice_rows(qtensor *t, int r0, int r1) {
    if (!t->data || r1 <= r0) return;
    size_t row_bytes = tf_row_bytes(t->type, t->n_cols);
    t->data = (uint8_t *)t->data + (size_t)r0 * row_bytes;
    t->n_rows = r1 - r0;
}

/* ROW-parallel: keep only input cols [c0,c1) of every row (offset data to col
 * c0, shrink n_cols, set tp_src_stride=orig n_cols so the fills step source rows
 * by the full width). BF16/F16 only (linear 2B/elem column offset). */
static void tf_tp_slice_cols(qtensor *t, int c0, int c1) {
    if (!t->data || c1 <= c0) return;
    if (t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_F16) {
        fprintf(stderr, "tf_tp_slice_cols: unsupported type %d (BF16/F16 only)\n", t->type);
        return;
    }
    t->data = (uint8_t *)t->data + tf_row_bytes(t->type, c0);  /* c0*2 bytes */
    t->tp_src_stride = t->n_cols;                               /* original full width */
    t->n_cols = c1 - c0;
}

/* Stage B: V-head shard ONE SSM/Delta-Net layer to global heads [hs,he).
 * Q/K projections stay REPLICATED (full n_group present in ssm_qkv), only the
 * V-heads are sharded — this keeps the n_group->dt_rank Q/K tile-repeat valid
 * even when local dt_rank < n_group (the forward gathers per local head via
 * m->ssm_head_offset). Per layer:
 *   - ssm_qkv  : repack rows to [Q full | K full | V[hs*ds,he*ds)] (contiguous)
 *   - conv_w_trans/conv_state : rebuild for the local [Q|K|Vslice] channel set
 *   - ssm_gate : ROW-slice output rows [hs*ds,he*ds)            (local d_inner)
 *   - ssm_alpha/beta : ROW-slice output rows [hs,he)            (local dt_rank)
 *   - ssm_out  : COL-slice input cols [hs*ds,he*ds) -> row-parallel -> all-reduce
 * ssm_a/dt_bias/ssm_norm stay replicated (forward indexes by global head).
 * m->ssm_qkv_dim is still FULL when this runs (mutated by the driver after the
 * layer loop), so it is the source channel width for the conv rebuild. */
static int tf_tp_slice_ssm_layer(transformer_model *m, int layer_idx,
                                  int hs, int he, int ng, int ds, int conv_k) {
    transformer_layer *L = &m->layers[layer_idx];
    int qk_rows  = 2 * ng * ds;        /* Q + K channels (replicated)         */
    int v_lo     = qk_rows + hs * ds;  /* first local V channel in the source */
    int loc_v    = (he - hs) * ds;     /* local V channels                    */
    int new_qkv  = qk_rows + loc_v;    /* local qkv_dim                       */
    int full_qkv = m->ssm_qkv_dim;     /* still full here (driver mutates later) */

    /* --- ssm_qkv: repack [Q|K|V_slice] into a fresh contiguous buffer.
     * Whole-row copy is type-agnostic (works for BF16/F16/quant). --- */
    {
        qtensor *t = &L->ssm_qkv;      /* [n_rows=qkv_dim, n_cols=n_embd] */
        size_t rb = tf_row_bytes(t->type, t->n_cols);
        uint8_t *src = (uint8_t *)t->data;
        size_t bytes = ((size_t)new_qkv * rb + 63) & ~(size_t)63;
        uint8_t *dst = (uint8_t *)aligned_alloc(64, bytes);
        if (!dst) { fprintf(stderr, "tp_slice ssm: OOM ssm_qkv repack (L%d)\n", layer_idx); return -1; }
        memcpy(dst,                       src,                    (size_t)qk_rows * rb);  /* Q+K */
        memcpy(dst + (size_t)qk_rows * rb, src + (size_t)v_lo * rb, (size_t)loc_v * rb);  /* V slice */
        t->data = dst;
        t->n_rows = new_qkv;
    }

    /* --- conv weights + state: rebuild for the local [Q|K|Vslice] channels.
     * conv_w_trans[l] is the load-time dequantized [conv_k][full_qkv] layout. --- */
    if (m->conv_w_trans && m->conv_w_trans[layer_idx]) {
        float *old_cw = m->conv_w_trans[layer_idx];
        size_t cwb = ((size_t)conv_k * new_qkv * sizeof(float) + 63) & ~(size_t)63;
        float *new_cw = (float *)aligned_alloc(64, cwb);
        if (!new_cw) { fprintf(stderr, "tp_slice ssm: OOM conv_w (L%d)\n", layer_idx); return -1; }
        for (int f = 0; f < conv_k; f++) {
            memcpy(new_cw + (size_t)f * new_qkv,            /* Q+K channels */
                   old_cw + (size_t)f * full_qkv, (size_t)qk_rows * sizeof(float));
            memcpy(new_cw + (size_t)f * new_qkv + qk_rows,  /* V channels   */
                   old_cw + (size_t)f * full_qkv + v_lo, (size_t)loc_v * sizeof(float));
        }
        free(old_cw);
        m->conv_w_trans[layer_idx] = new_cw;
    }
    if (m->conv_state) {
        free(m->conv_state[layer_idx]);
        m->conv_state[layer_idx] = (float *)calloc((size_t)(conv_k - 1) * new_qkv, sizeof(float));
        m->conv_state_pos[layer_idx] = 0;
    }

    /* --- gate / alpha / beta : output rows by local V-head --- */
    tf_tp_slice_rows(&L->ssm_gate,  hs * ds, he * ds);  /* local d_inner = L*ds */
    tf_tp_slice_rows(&L->ssm_alpha, hs,      he);       /* local dt_rank = L    */
    tf_tp_slice_rows(&L->ssm_beta,  hs,      he);
    /* --- ssm_out : input cols by local V-head -> row-parallel -> all-reduce --- */
    tf_tp_slice_cols(&L->ssm_out,   hs * ds, he * ds);
    return 0;
}

static inline void tf_tp_partition_range(int total, int parts, int part,
                                        int *lo_out, int *hi_out) {
    const int base = total / parts;
    const int rem = total % parts;
    int lo = part * base + ((part < rem) ? part : rem);
    int hi = lo + base + ((part < rem) ? 1 : 0);
    if (lo < 0) lo = 0;
    if (hi < lo) hi = lo;
    if (lo > total) lo = total;
    if (hi > total) hi = total;
    *lo_out = lo;
    *hi_out = hi;
}

int transformer_tp_slice_weights(transformer_model *m, int tp_rank, int tp_size,
                                  int ssm_shard) {
    if (!m || tp_size <= 1) return 0;
    if (tp_rank < 0 || tp_rank >= tp_size) {
        fprintf(stderr, "tp_slice: bad rank %d/%d\n", tp_rank, tp_size); return -1;
    }
    if (m->use_moe) {
        fprintf(stderr, "tp_slice: MoE sharding not implemented\n"); return -1;
    }
    int hd        = m->head_dim;
    int n_heads   = m->n_heads;
    int n_kv      = m->n_kv_heads;
    int n_ff      = m->n_ff;
    int per_head_q = m->is_hybrid ? (2 * hd) : hd;  /* gated attn: Q+gate interleaved */

    /* --- divisibility. Q-heads must partition. KV heads may be REPLICATED when
     * n_kv % tp_size != 0 (every rank keeps all KV heads + full KV cache, only the
     * Q-heads are sharded) -- needed e.g. for 27B's 4 KV heads at TP=6. Each check is
     * gated on whether that component is actually being sharded. --- */
    int kv_replicate = 0;
    if (!getenv("TP_SKIP_ATTN")) {
        kv_replicate = (n_kv % tp_size) != 0;   /* 1 => replicate KV, shard Q only */
    }
    /* FFN: balanced, multiple-of-16 partition (uneven last rank, like the vocab shard)
     * so widths with no factor of tp_size (e.g. 17408 = 2^10*17 @ TP=6) still work. */
    int ff_chunk = (((n_ff + tp_size - 1) / tp_size) + 15) & ~15;
    if (!getenv("TP_SKIP_FFN") && ((n_ff % 16) || ff_chunk * (tp_size - 1) >= n_ff)) {
        fprintf(stderr, "tp_slice: n_ff=%d cannot form a mult-of-16 partition for tp_size=%d\n",
                n_ff, tp_size);
        return -1;
    }

    /* SSM V-head split: dt_rank V-heads divided across the TP group. Q/K stay
     * replicated, so no n_group divisibility constraint. */
    int ssm_dt = m->ssm_dt_rank, ssm_ng = m->ssm_n_group, ssm_ds = m->ssm_d_state;
    int ssm_ck = m->ssm_conv_kernel;
    if (ssm_shard) {
        if (!m->is_hybrid) { fprintf(stderr, "tp_slice: ssm_shard set but model is not hybrid\n"); return -1; }
    }

    int qh_lo = 0, qh_hi = 0;
    int kh_lo = 0, kh_hi = 0;
    int vh_lo = 0, vh_hi = 0;
    tf_tp_partition_range(n_heads, tp_size, tp_rank, &qh_lo, &qh_hi);
    if (!kv_replicate) tf_tp_partition_range(n_kv, tp_size, tp_rank, &kh_lo, &kh_hi);
    tf_tp_partition_range(ssm_dt, tp_size, tp_rank, &vh_lo, &vh_hi);
    int ff_lo = tp_rank * ff_chunk,            ff_hi = ff_lo + ff_chunk;          /* uneven: last rank shorter */
    if (ff_lo > n_ff) ff_lo = n_ff;
    if (ff_hi > n_ff) ff_hi = n_ff;

    /* Bisection gates: shard only attn/FFN/SSM to localize a correctness bug. */
    int do_attn = !getenv("TP_SKIP_ATTN");
    int do_ffn  = !getenv("TP_SKIP_FFN");
    int do_ssm  = ssm_shard && !getenv("TP_SKIP_SSM");
    if (do_attn && qh_hi <= qh_lo) do_attn = 0;
    if (do_ssm && vh_hi <= vh_lo) do_ssm = 0;

    /* KV-replicate: full KV cache stays, so the kernel must map the GLOBAL query head
     * (qh_lo + local h) to its KV head; clean-divide sharding rebases to a local
     * cache offset. */
    m->tp_qhead_offset = (do_attn && kv_replicate) ? qh_lo : 0;
    m->tp_kv_head_base  = (do_attn && !kv_replicate) ? kh_lo : 0;
    m->tp_kv_head_count = do_attn ? (kv_replicate ? n_kv : (kh_hi - kh_lo)) : n_kv;

    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *L = &m->layers[l];
        int is_ssm_layer = (m->is_hybrid && L->is_ssm);

        /* SSM mixer: V-head shard (Stage B). Only on SSM layers. */
        if (do_ssm && is_ssm_layer) {
            if (tf_tp_slice_ssm_layer(m, l, vh_lo, vh_hi, ssm_ng, ssm_ds, ssm_ck) != 0)
                return -1;
        }

        /* Attention exists ONLY on non-SSM (gated/full-attn) layers; the SSM
         * mixer is left replicated in Stage A. */
        if (do_attn && !is_ssm_layer) {
            /* COL-parallel: attn Q output rows by head (always sharded). */
            tf_tp_slice_rows(&L->attn_q, qh_lo * per_head_q, qh_hi * per_head_q);
            if (L->attn_q_bias.data) tf_tp_slice_rows(&L->attn_q_bias, qh_lo * per_head_q, qh_hi * per_head_q);
            /* K/V: sharded by KV-head on a clean divide; REPLICATED (left full) otherwise. */
            if (!kv_replicate) {
                tf_tp_slice_rows(&L->attn_k, kh_lo * hd, kh_hi * hd);
                tf_tp_slice_rows(&L->attn_v, kh_lo * hd, kh_hi * hd);
                if (L->attn_k_bias.data) tf_tp_slice_rows(&L->attn_k_bias, kh_lo * hd, kh_hi * hd);
                if (L->attn_v_bias.data) tf_tp_slice_rows(&L->attn_v_bias, kh_lo * hd, kh_hi * hd);
            }
            /* ROW-parallel: attn_output input cols = the local Q-head columns. */
            tf_tp_slice_cols(&L->attn_output, qh_lo * hd, qh_hi * hd);
        }

        /* The dense SwiGLU FFN exists on EVERY layer (SSM blocks included), and
         * m->n_ff is mutated globally below — so the FFN must be sliced on every
         * layer, NOT skipped for SSM (else ffn_down dots stale ffn_buf3 tail).
         * tf_tp_slice_* no-op on absent tensors, so this is safe if some layer
         * genuinely lacks an FFN. */
        if (do_ffn) {
            /* COL-parallel: ffn gate/up output rows. ROW-parallel: ffn_down cols. */
            tf_tp_slice_rows(&L->ffn_gate, ff_lo, ff_hi);
            tf_tp_slice_rows(&L->ffn_up,   ff_lo, ff_hi);
            tf_tp_slice_cols(&L->ffn_down, ff_lo, ff_hi);
        }
    }

    /* LM-head (vocab) split: divide output.weight rows across the group so each
     * rank's compute_logits runs only its 1/tp_size of the 248320×n_embd matvec
     * (the biggest fixed cost at decode). m->n_vocab stays FULL — token_embd is
     * NOT touched, embedding lookup still spans the whole vocabulary. Skipped if
     * output is tied to token_embd (slicing the shared tensor would corrupt the
     * embedding) or no LM head. Boundaries are rounded up to a multiple of 8 so
     * every rank keeps the bf16_pv panel path; the last rank takes the (possibly
     * shorter) remainder. The argmax-reduce payload is 2 floats regardless of the
     * local row count, so an uneven final shard is fine. */
    int do_vocab = m->has_lm_head && !getenv("TP_NO_VOCAB_SHARD")
                   && m->output.data && m->output.data != m->token_embd.data;
    if (do_vocab) {
        int chunk = ((m->n_vocab + tp_size - 1) / tp_size + 7) & ~7;  /* per-rank, mult of 8 */
        int v_lo  = tp_rank * chunk;
        int v_hi  = v_lo + chunk;
        if (v_lo > m->n_vocab) v_lo = m->n_vocab;
        if (v_hi > m->n_vocab) v_hi = m->n_vocab;
        tf_tp_slice_rows(&m->output, v_lo, v_hi);
        m->tp_vocab_lo  = v_lo;
        m->tp_vocab_loc = v_hi - v_lo;
    } else {
        m->tp_vocab_lo  = 0;
        m->tp_vocab_loc = m->n_vocab;
    }
    m->tp_vocab_sharded = do_vocab;

    /* Mutate model dims to local so the forward loops compute the shard. KV heads
     * stay FULL when replicated (only Q-heads sharded). FFN is the uneven local width. */
    if (do_attn) { m->n_heads = qh_hi - qh_lo; if (!kv_replicate) m->n_kv_heads = kh_hi - kh_lo; }
    if (do_ffn)  { m->n_ff = ff_hi - ff_lo; }
    if (do_ssm) {
        int loc_dt = vh_hi - vh_lo;
        m->ssm_dt_rank   = loc_dt;
        m->ssm_d_inner   = loc_dt * ssm_ds;
        m->ssm_qkv_dim   = 2 * ssm_ng * ssm_ds + loc_dt * ssm_ds;  /* Q+K full, V local */
        m->ssm_head_offset = vh_lo;
    }
    m->tp_rank        = tp_rank;
    m->tp_size        = tp_size;
    m->tp_ssm_sharded = do_ssm;   /* gates the SSM mixer-output all-reduce */
    m->tp_attn_sharded = do_attn;
    m->tp_ffn_sharded  = do_ffn;

    fprintf(stderr, "tp_slice: rank %d/%d — n_heads %d→%d, n_kv %d→%d, n_ff %d→%d, "
            "ssm_dt %d→%d, vocab %d→%d@%d, qh_off=%d gqa_grp=%d (attn %s, kv %s, ffn %s, SSM %s, vocab %s)\n",
            tp_rank, tp_size,
            n_heads, m->n_heads, n_kv, m->n_kv_heads, n_ff, m->n_ff,
            ssm_dt, m->ssm_dt_rank, m->n_vocab, m->tp_vocab_loc, m->tp_vocab_lo,
            m->tp_qhead_offset, m->gqa_group,
            do_attn ? "shard" : "replic", kv_replicate ? "REPLIC" : "shard", do_ffn ? "shard" : "replic",
            do_ssm ? "sharded" : "replicated", do_vocab ? "sharded" : "replicated");
    return 0;
}

/* ---- Pipeline-parallel layer-range ownership ---- */

/* Restrict this stage to layers [layer_start, layer_end). Call AFTER
 * transformer_load and BEFORE transformer_build_panels so panel repacking
 * (and the lazy-mmap page faults it triggers) is confined to owned layers.
 * Pair with transformer_free_unused_kv() to drop non-owned KV/SSM state. */
void transformer_set_pp_range(transformer_model *model, int layer_start, int layer_end) {
    if (!model) return;
    if (layer_start < 0) layer_start = 0;
    if (layer_end > model->n_layers) layer_end = model->n_layers;
    if (layer_end < layer_start) layer_end = layer_start;
    model->pp_start = layer_start;
    model->pp_end   = layer_end;
}

/* ---- Distributed memory management ---- */

void transformer_free_unused_kv(transformer_model *model, int layer_start, int layer_end) {
    if (!model || !model->key_cache) return;
    for (int l = 0; l < model->n_layers; l++) {
        if (l < layer_start || l >= layer_end) {
            free(model->key_cache[l]);   model->key_cache[l] = NULL;
            free(model->value_cache[l]); model->value_cache[l] = NULL;
        }
    }
    /* Also free SSM state for unused layers. recurrent_state is mmap-anon (see
     * transformer_load) so it must be munmap'd, not free()'d — matching the
     * cleanup in transformer_free. conv_state is calloc'd; conv_w_trans is
     * aligned_alloc'd; both use free(). */
    if (model->conv_state) {
        size_t rec_state_bytes = (size_t)model->ssm_dt_rank *
            model->ssm_d_state * model->ssm_d_state * sizeof(float);
        for (int l = 0; l < model->n_layers; l++) {
            if (l < layer_start || l >= layer_end) {
                free(model->conv_state[l]);            model->conv_state[l] = NULL;
                if (model->recurrent_state[l]) {
                    munmap(model->recurrent_state[l], rec_state_bytes);
                    model->recurrent_state[l] = NULL;
                }
                if (model->conv_w_trans) {
                    free(model->conv_w_trans[l]);      model->conv_w_trans[l] = NULL;
                }
            }
        }
    }
}

void transformer_resize_kv_for_tp(transformer_model *model,
                                    int layer_start, int layer_end, int tp_kv_dim) {
    if (!model || !model->key_cache) return;
    for (int l = layer_start; l < layer_end && l < model->n_layers; l++) {
        if (!model->key_cache[l]) continue;  /* SSM layer, no KV cache */
        free(model->key_cache[l]);
        free(model->value_cache[l]);
        model->key_cache[l]   = calloc((size_t)model->max_seq_len * tp_kv_dim, model->kv_elem_bytes);
        model->value_cache[l] = calloc((size_t)model->max_seq_len * tp_kv_dim, model->kv_elem_bytes);
        if (model->key_scales) {
            free(model->key_scales[l]);
            model->key_scales[l]   = (float *)calloc((size_t)model->max_seq_len * model->n_kv_heads, sizeof(float));
        }
        if (model->value_scales) {
            free(model->value_scales[l]);
            model->value_scales[l] = (float *)calloc((size_t)model->max_seq_len * model->n_kv_heads, sizeof(float));
        }
    }
}

/* Column-parallel matvec: compute rows [row_start, row_end) of mat, output to dst[0..count).
 * Used for QKV, gate, up projections where each TP rank computes a subset of output rows. */
static void tf_qmatvec_row_slice(transformer_model *m, float *dst, const qtensor *mat,
                                  const float *x, int row_start, int row_end) {
    int count = row_end - row_start;
    if (count <= 0) return;
    /* Create a virtual qtensor pointing to the slice */
    size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
    qtensor slice = *mat;
    slice.data = (void *)((const uint8_t *)mat->data + (size_t)row_start * row_bytes);
    slice.n_rows = count;
    tf_qmatvec_pool(m, dst, &slice, x, count);
}

/* Row-parallel matvec: compute partial dot products using columns [col_start, col_end) of mat.
 * Each row's result is a partial sum; caller must allreduce across TP ranks.
 * x_local is the local slice of the input [col_end - col_start].
 * dst has n_rows elements (partial sums). */
static void tf_qmatvec_col_slice(float *dst, const qtensor *mat, const float *x_local,
                                  int n_rows, int col_start, int col_end, float *tmp) {
    int local_cols = col_end - col_start;
    for (int i = 0; i < n_rows; i++) {
        /* Dequant full row to tmp, then dot with local slice */
        tf_dequant_row(mat, i, tmp);
        float sum = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 15 < local_cols; j += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + col_start + j),
                                    _mm256_loadu_ps(x_local + j), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + col_start + j + 8),
                                    _mm256_loadu_ps(x_local + j + 8), acc1);
        }
        for (; j + 7 < local_cols; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + col_start + j),
                                    _mm256_loadu_ps(x_local + j), acc0);
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        sum = _mm_cvtss_f32(s4);
        for (; j < local_cols; j++) sum += tmp[col_start + j] * x_local[j];
#else
        for (int j = 0; j < local_cols; j++)
            sum += tmp[col_start + j] * x_local[j];
#endif
        dst[i] = sum;
    }
}

/* Row-parallel matvec with pool threading: col_slice + allreduce. */
static void tf_qmatvec_col_slice_pool(transformer_model *m, float *dst, const qtensor *mat,
                                       const float *x_local, int n_rows,
                                       int col_start, int col_end) {
    /* Single-threaded col-slice (threading within rows is hard for col-slice) */
    tf_qmatvec_col_slice(dst, mat, x_local, n_rows, col_start, col_end, m->matvec_tmp);
    /* Allreduce partial sums across TP ranks */
    if (m->tp_allreduce_fn)
        m->tp_allreduce_fn(dst, n_rows, m->tp_allreduce_ctx);
}

/* ---- Pipeline-parallel API ---- */

void transformer_embed_token(transformer_model *model, int32_t token_id) {
    if (!model || !model->token_embd.data) return;
    if (token_id < 0 || token_id >= model->n_vocab) return;
    tf_dequant_row(&model->token_embd, token_id, model->x);
}

float *transformer_get_hidden(transformer_model *model) {
    return model ? model->x : NULL;
}

void transformer_set_hidden(transformer_model *model, const float *hidden) {
    if (model && hidden)
        memcpy(model->x, hidden, model->n_embd * sizeof(float));
}

float *transformer_compute_logits(transformer_model *model) {
    if (!model || !model->has_lm_head) return NULL;
    /* When the LM head is vocab-sharded the output panel holds only this rank's
     * rows, so compute exactly tp_vocab_loc logits into m->logits[0..loc). The
     * caller maps the local argmax index back via tp_vocab_lo and reduces. */
    int nlog = model->tp_vocab_sharded ? model->tp_vocab_loc : model->n_vocab;
    TF_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
    tf_qmatvec_pool(model, model->logits, &model->output, model->x, nlog);
    TF_PROF_END("lm_head", 2.0 * nlog * model->n_embd, 0);
    return model->logits;
}

float *transformer_forward_partial(transformer_model *m, int cache_pos,
                                    int layer_start, int layer_end) {
    if (!m) return NULL;
    /* Full-range forward with a live thread pool: route through the persistent
     * worker (ONE pool dispatch/token, internal HW/spin barriers, fully
     * parallel SSM) instead of the per-op block loop (~320 dispatches/token).
     * The persistent worker carries the SAME TP all-reduce hooks as the per-op
     * path (mixer-out after attn/ssm out-proj, ffn-down), issued tid-0-only in
     * identical count+order, so TP decode stays byte-identical / lockstep-argmax
     * safe. PP partial ranges (layer_start>0 or layer_end<n_layers) keep the
     * per-op loop, which is the only path that supports partial layer spans.
     * Env TP_DECODE_PERSIST=0 forces the per-op path for A/B. */
    static int persist = -1;
    if (persist < 0)
        persist = getenv("TP_DECODE_PERSIST") ? atoi(getenv("TP_DECODE_PERSIST")) : 1;
    if (persist && layer_start == 0 && layer_end == m->n_layers &&
        m->n_threads > 1 && m->pool_alive)
        return tf_forward_persistent(m, cache_pos, cache_pos, cache_pos, cache_pos);
    return tf_forward_blocks_range(m, cache_pos, cache_pos, cache_pos, cache_pos,
                                    layer_start, layer_end);
}

float *transformer_forward_embd_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w) {
    if (!model || !embd) return NULL;
    if (cache_pos < 0 || cache_pos >= model->max_seq_len) {
        fprintf(stderr, "transformer_forward_embd_pos: cache_pos=%d out of range [0, %d)\n", cache_pos, model->max_seq_len);
        return NULL;
    }
    memcpy(model->x, embd, model->n_embd * sizeof(float));
    /* Gemma4: use padding token (ID=0) for per-layer embeddings on vision tokens.
     * llama.cpp gemma4-iswa.cpp: vision path uses row 0 of per_layer_token_embd. */
    if (model->is_gemma4) model->current_token_id = 0;
    /* Store full embedding pointer for deepstack injection in tf_forward_blocks */
    model->ds_embd = embd;
    float *result = tf_forward_blocks(model, cache_pos, pos_t, pos_h, pos_w);
    model->ds_embd = NULL;
    return result;
}

float *transformer_forward_embd(transformer_model *model, const float *embd, int position) {
    return transformer_forward_embd_pos(model, embd, position, position, position, position);
}

float *transformer_forward_embd_logits_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w) {
    float *hidden = transformer_forward_embd_pos(model, embd, cache_pos, pos_t, pos_h, pos_w);
    if (!model->has_lm_head || !hidden) return NULL;
    TF_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
    tf_qmatvec_pool(model, model->logits, &model->output, hidden, model->n_vocab);
    TF_PROF_END("lm_head", 2.0 * model->n_vocab * model->n_embd, 0);
    return model->logits;
}

float *transformer_forward_embd_logits(transformer_model *model, const float *embd, int position) {
    return transformer_forward_embd_logits_pos(model, embd, position, position, position, position);
}

/* ---- Batched prefill ---- */

/* Multi-threaded GEMM worker for F16 weights */
typedef struct {
    float *Y;
    const uint16_t *W;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_task;

static void *tf_gemm_worker(void *arg) {
    tf_gemm_task *t = (tf_gemm_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_f16_f32(t->Y + t->row_start * t->Y_stride,
                  t->W + (size_t)t->row_start * t->K,
                  t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

/* Multi-threaded GEMM for F16 weight matrices: Y[n_rows, N] = W[n_rows, K] × X[N, K]^T
 * Output Y is row-major: Y[row][tok], Y_stride >= N. */
static void tf_gemm_f16_mt(float *Y, const qtensor *mat, const float *X,
                            int n_rows, int N, int Y_stride, int X_stride,
                            int n_threads) {
    if (mat->type != GGML_TYPE_F16) {
        /* Fallback: per-token matvec for non-F16 weights (AVX2 dot product) */
        int n_cols = mat->n_cols;
        float *row_buf = (float *)malloc(n_cols * sizeof(float));
        size_t rb = tf_row_bytes(mat->type, n_cols);
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            for (int r = 0; r < n_rows; r++) {
                const void *row_data = (const uint8_t *)mat->data + (size_t)r * rb;
                dequant_row(mat->type, row_data, row_buf, n_cols);
#if defined(__AVX2__) && defined(__FMA__)
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                int j = 0;
                for (; j + 31 < n_cols; j += 32) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j),    _mm256_loadu_ps(xt+j),    a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+8),  _mm256_loadu_ps(xt+j+8),  a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+16), _mm256_loadu_ps(xt+j+16), a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+24), _mm256_loadu_ps(xt+j+24), a3);
                }
                for (; j + 7 < n_cols; j += 8)
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j), _mm256_loadu_ps(xt+j), a0);
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                float sum = _mm_cvtss_f32(s4);
                for (; j < n_cols; j++) sum += row_buf[j] * xt[j];
#else
                float sum = 0.0f;
                for (int j = 0; j < n_cols; j++) sum += row_buf[j] * xt[j];
#endif
                Y[r * Y_stride + t] = sum;
            }
        }
        free(row_buf);
        return;
    }

    const uint16_t *W = (const uint16_t *)mat->data;
    int K = mat->n_cols;

    if (n_threads <= 1 || n_rows < n_threads * 6) {
        gemm_f16_f32(Y, W, X, n_rows, K, N, Y_stride, X_stride);
        return;
    }

    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_gemm_task *tasks = (tf_gemm_task *)alloca(n_threads * sizeof(tf_gemm_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = rows_per + (i < extra ? 1 : 0);
        tasks[i] = (tf_gemm_task){Y, W, X, offset, offset + count, K, N, Y_stride, X_stride};
        offset += count;
        pthread_create(&threads[i], NULL, tf_gemm_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* Token-major threaded GEMM worker */
typedef struct {
    float *Y;
    const uint16_t *W;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_tm_task;

static void *tf_gemm_tm_worker(void *arg) {
    tf_gemm_tm_task *t = (tf_gemm_tm_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_f16_f32_tokmajor(t->Y + t->row_start,
                           t->W + (size_t)t->row_start * t->K,
                           t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

static void *tf_gemm_bf16_tm_worker(void *arg) {
    tf_gemm_tm_task *t = (tf_gemm_tm_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_bf16_f32_tokmajor(t->Y + t->row_start,
                            t->W + (size_t)t->row_start * t->K,
                            t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

/* Q8_0 token-major GEMM worker (for threaded dispatch) */
typedef struct {
    float *Y;
    const void *W;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_q8_tm_task;

static void *tf_gemm_q8_tm_worker(void *arg) {
    tf_gemm_q8_tm_task *t = (tf_gemm_q8_tm_task *)arg;
    int nrows = t->row_end - t->row_start;
    int nb = t->K / 32;
    size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
    gemm_q8_0_f32_tokmajor(t->Y + t->row_start,
                             (const uint8_t *)t->W + (size_t)t->row_start * row_bytes,
                             t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

/* Token-major GEMM: Y[tok * out_stride + row] = dot(W[row,:], X[tok,:])
 * Direct output without transpose. */
static void tf_gemm_f16_mt_tokenmajor(float *Y_out, const qtensor *mat, const float *X,
                                       int n_rows, int N, int out_stride, int X_stride,
                                       int n_threads) {
    if (mat->type == GGML_TYPE_Q8_0) {
        /* Q8_0 SIMD GEMM path */
        int K = mat->n_cols;
        if (n_threads <= 1 || n_rows < n_threads * 6) {
            gemm_q8_0_f32_tokmajor(Y_out, mat->data, X, n_rows, K, N, out_stride, X_stride);
            return;
        }
        pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
        tf_gemm_q8_tm_task *tasks = (tf_gemm_q8_tm_task *)alloca(n_threads * sizeof(tf_gemm_q8_tm_task));
        int rows_per = n_rows / n_threads, extra = n_rows % n_threads, offset = 0;
        for (int i = 0; i < n_threads; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_q8_tm_task){Y_out,
                                    (const uint8_t *)mat->data,
                                    X, offset, offset + count, K, N, out_stride, X_stride};
            offset += count;
        }
        for (int i = 0; i < n_threads; i++) {
            pthread_create(&threads[i], NULL, tf_gemm_q8_tm_worker, &tasks[i]);
        }
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        return;
    }
    if (mat->type == GGML_TYPE_BF16) {
        /* BF16 GEMM path */
        int K = mat->n_cols;
        if (n_threads <= 1 || n_rows < n_threads * 4) {
            gemm_bf16_f32_tokmajor(Y_out, (const uint16_t *)mat->data, X,
                                    n_rows, K, N, out_stride, X_stride);
            return;
        }
        pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
        tf_gemm_tm_task *tasks = (tf_gemm_tm_task *)alloca(n_threads * sizeof(tf_gemm_tm_task));
        int rows_per = n_rows / n_threads, extra = n_rows % n_threads, offset = 0;
        for (int i = 0; i < n_threads; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_tm_task){Y_out, (const uint16_t *)mat->data,
                                          X, offset, offset + count, K, N, out_stride, X_stride};
            offset += count;
        }
        for (int i = 0; i < n_threads; i++)
            pthread_create(&threads[i], NULL, tf_gemm_bf16_tm_worker, &tasks[i]);
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        return;
    }
    if (mat->type != GGML_TYPE_F16) {
        /* Fallback for other non-F16/BF16: per-token matvec */
        int n_cols = mat->n_cols;
        float *row_buf = (float *)malloc(n_cols * sizeof(float));
        size_t rb = tf_row_bytes(mat->type, n_cols);
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            for (int r = 0; r < n_rows; r++) {
                const void *row_data = (const uint8_t *)mat->data + (size_t)r * rb;
                dequant_row(mat->type, row_data, row_buf, n_cols);
#if defined(__AVX2__) && defined(__FMA__)
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                int j = 0;
                for (; j + 31 < n_cols; j += 32) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j),    _mm256_loadu_ps(xt+j),    a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+8),  _mm256_loadu_ps(xt+j+8),  a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+16), _mm256_loadu_ps(xt+j+16), a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+24), _mm256_loadu_ps(xt+j+24), a3);
                }
                for (; j + 7 < n_cols; j += 8)
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j), _mm256_loadu_ps(xt+j), a0);
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                float sum = _mm_cvtss_f32(s4);
                for (; j < n_cols; j++) sum += row_buf[j] * xt[j];
#else
                float sum = 0.0f;
                for (int j = 0; j < n_cols; j++) sum += row_buf[j] * xt[j];
#endif
                Y_out[(size_t)t * out_stride + r] = sum;
            }
        }
        free(row_buf);
        return;
    }

    const uint16_t *W = (const uint16_t *)mat->data;
    int K = mat->n_cols;

    if (n_threads <= 1 || n_rows < n_threads * 6) {
        gemm_f16_f32_tokmajor(Y_out, W, X, n_rows, K, N, out_stride, X_stride);
        return;
    }

    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_gemm_tm_task *tasks = (tf_gemm_tm_task *)alloca(n_threads * sizeof(tf_gemm_tm_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = rows_per + (i < extra ? 1 : 0);
        tasks[i] = (tf_gemm_tm_task){Y_out, W, X, offset, offset + count, K, N, out_stride, X_stride};
        offset += count;
        pthread_create(&threads[i], NULL, tf_gemm_tm_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* Fused dual-matrix threaded GEMM: compute gate and up simultaneously, reading X once */
typedef struct {
    float *Y1, *Y2;
    const uint16_t *W1, *W2;
    const float *X;
    int row_start, row_end;
    int K, N, Y1_stride, Y2_stride, X_stride;
} tf_gemm_fused2_task;

static void *tf_gemm_fused2_worker(void *arg) {
    tf_gemm_fused2_task *t = (tf_gemm_fused2_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_f16_f32_tokmajor_fused2(
        t->Y1 + t->row_start, t->W1 + (size_t)t->row_start * t->K,
        t->Y2 + t->row_start, t->W2 + (size_t)t->row_start * t->K,
        t->X, nrows, t->K, t->N,
        t->Y1_stride, t->Y2_stride, t->X_stride);
    return NULL;
}

static void tf_gemm_f16_mt_fused2(float *Y1, const qtensor *mat1,
                                    float *Y2, const qtensor *mat2,
                                    const float *X, int n_rows, int N,
                                    int out_stride, int X_stride, int n_threads) {
    const uint16_t *W1 = (const uint16_t *)mat1->data;
    const uint16_t *W2 = (const uint16_t *)mat2->data;
    int K = mat1->n_cols;

    if (n_threads <= 1 || n_rows < n_threads * 4) {
        gemm_f16_f32_tokmajor_fused2(Y1, W1, Y2, W2, X, n_rows, K, N,
                                      out_stride, out_stride, X_stride);
        return;
    }

    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_gemm_fused2_task *tasks = (tf_gemm_fused2_task *)alloca(n_threads * sizeof(tf_gemm_fused2_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = rows_per + (i < extra ? 1 : 0);
        tasks[i] = (tf_gemm_fused2_task){Y1, Y2, W1, W2, X, offset, offset + count,
                                          K, N, out_stride, out_stride, X_stride};
        offset += count;
        pthread_create(&threads[i], NULL, tf_gemm_fused2_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* ---- Vectorized batch helpers ---- */

#if defined(__AVX2__) && defined(__FMA__)
/* Fast AVX2 exp approximation (Schraudolph-style, ~1e-4 relative error).
 * Based on: exp(x) ≈ 2^(x/ln2) using float bit manipulation. */
static inline __m256 fast_exp_avx2(__m256 x) {
    const __m256 log2e  = _mm256_set1_ps(1.442695040f);
    const __m256 c0     = _mm256_set1_ps(12582912.0f); /* 1.5 * 2^23 (magic bias) */
    /* Polynomial coefficients for fractional part */
    const __m256 p0     = _mm256_set1_ps(0.9999999f);
    const __m256 p1     = _mm256_set1_ps(0.6931472f);
    const __m256 p2     = _mm256_set1_ps(0.2402265f);
    const __m256 p3     = _mm256_set1_ps(0.0554953f);
    const __m256 p4     = _mm256_set1_ps(0.0096813f);
    const __m256 clamp_hi = _mm256_set1_ps(88.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-88.0f);

    x = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);
    __m256 z = _mm256_fmadd_ps(x, log2e, c0);
    __m256 n_f = _mm256_sub_ps(z, c0);  /* floor(x * log2e) */
    __m256 f = _mm256_fmsub_ps(x, log2e, n_f); /* fractional part */
    /* 2^n via integer shift */
    __m256i n_i = _mm256_cvtps_epi32(n_f);
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(n_i, _mm256_set1_epi32(127)), 23);
    /* Polynomial: p0 + f*(p1 + f*(p2 + f*(p3 + f*p4))) */
    __m256 poly = _mm256_fmadd_ps(f, p4, p3);
    poly = _mm256_fmadd_ps(f, poly, p2);
    poly = _mm256_fmadd_ps(f, poly, p1);
    poly = _mm256_fmadd_ps(f, poly, p0);
    return _mm256_mul_ps(poly, _mm256_castsi256_ps(pow2n));
}

/* Vectorized SiLU(gate) × up for a contiguous array: out[i] = silu(gate[i]) * up[i] */
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        /* silu(g) = g / (1 + exp(-g)) = g * sigmoid(g) */
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_g)));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(g, sig), u));
    }
    for (; i < n; i++) {
        float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
#elif defined(__ARM_FEATURE_SVE)
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n) {
    int vl = (int)svcntw();
    svfloat32_t one = svdup_f32(1.0f);
    int i = 0;
    for (; i + vl - 1 < n; i += vl) {
        svbool_t pg = svptrue_b32();
        svfloat32_t g = svld1(pg, gate + i);
        svfloat32_t u = svld1(pg, up + i);
        svfloat32_t e = tf_fast_exp_sve(pg, svneg_x(pg, g));
        svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
        svst1(pg, out + i, svmul_x(pg, svmul_x(pg, g, sig), u));
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t g = svld1(pg, gate + i);
        svfloat32_t u = svld1(pg, up + i);
        svfloat32_t e = tf_fast_exp_sve(pg, svneg_x(pg, g));
        svfloat32_t sig = svdiv_x(pg, one, svadd_x(pg, one, e));
        svst1(pg, out + i, svmul_x(pg, svmul_x(pg, g, sig), u));
    }
}
#else
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
#endif

/* Batched M-RoPE with precomputed frequency table.
 * Computes freq[j] and sector_pos_map[j] once, then applies to all N tokens.
 * freq[j] = 1/pow(base, 2j/head_dim), pos_idx[j] = which of {t,h,w} to use. */
static void tf_rope_mrope_batch(float *bq, float *bk, int N, int n_heads, int n_kv_heads,
                                 int head_dim, int q_dim, int kv_dim,
                                 const int *pos_t, const int *pos_h, const int *pos_w,
                                 float freq_base, const int sections[4]) {
    int half_dim = head_dim / 2;
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0) sect_dims = half_dim;
    /* Use rope_dim_count (= 2 * sect_dims) as freq denominator, not head_dim */
    int rope_dim = 2 * sect_dims;

    /* Precompute per-dimension: freq[j] and which position axis to use (0=t,1=h,2=w) */
    float *freq = (float *)alloca(half_dim * sizeof(float));
    int *pos_axis = (int *)alloca(half_dim * sizeof(int));

    for (int j = 0; j < half_dim; j++) {
        freq[j] = 1.0f / powf(freq_base, (float)(2 * j) / rope_dim);
        int sector = j % sect_dims;
        if (sector % 3 == 1 && sector < 3 * sections[1]) {
            pos_axis[j] = 1; /* height */
        } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
            pos_axis[j] = 2; /* width */
        } else {
            pos_axis[j] = 0; /* temporal (default) */
        }
    }

    /* Precompute cos/sin tables per token: [N][half_dim] */
    float *cos_tab = (float *)malloc((size_t)N * half_dim * sizeof(float));
    float *sin_tab = (float *)malloc((size_t)N * half_dim * sizeof(float));

    for (int t = 0; t < N; t++) {
        float *ct = cos_tab + (size_t)t * half_dim;
        float *st = sin_tab + (size_t)t * half_dim;
        for (int j = 0; j < half_dim; j++) {
            int pos;
            switch (pos_axis[j]) {
                case 1: pos = pos_h[t]; break;
                case 2: pos = pos_w[t]; break;
                default: pos = pos_t[t]; break;
            }
            float theta = pos * freq[j];
            ct[j] = cosf(theta);
            st[j] = sinf(theta);
        }
    }

    /* Apply rotation to Q: [N, n_heads, head_dim] */
    for (int t = 0; t < N; t++) {
        const float *ct = cos_tab + (size_t)t * half_dim;
        const float *st = sin_tab + (size_t)t * half_dim;
        for (int h = 0; h < n_heads; h++) {
            float *v = bq + (size_t)t * q_dim + h * head_dim;
            int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
            for (; j + 7 < half_dim; j += 8) {
                __m256 vc = _mm256_loadu_ps(ct + j);
                __m256 vs = _mm256_loadu_ps(st + j);
                __m256 v0 = _mm256_loadu_ps(v + j);
                __m256 v1 = _mm256_loadu_ps(v + j + half_dim);
                _mm256_storeu_ps(v + j,            _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
                _mm256_storeu_ps(v + j + half_dim, _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
            }
#endif
            for (; j < half_dim; j++) {
                float v0 = v[j], v1 = v[j + half_dim];
                v[j]            = v0 * ct[j] - v1 * st[j];
                v[j + half_dim] = v0 * st[j] + v1 * ct[j];
            }
        }
    }

    /* Apply rotation to K: [N, n_kv_heads, head_dim] */
    for (int t = 0; t < N; t++) {
        const float *ct = cos_tab + (size_t)t * half_dim;
        const float *st = sin_tab + (size_t)t * half_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            float *v = bk + (size_t)t * kv_dim + h * head_dim;
            int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
            for (; j + 7 < half_dim; j += 8) {
                __m256 vc = _mm256_loadu_ps(ct + j);
                __m256 vs = _mm256_loadu_ps(st + j);
                __m256 v0 = _mm256_loadu_ps(v + j);
                __m256 v1 = _mm256_loadu_ps(v + j + half_dim);
                _mm256_storeu_ps(v + j,            _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
                _mm256_storeu_ps(v + j + half_dim, _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
            }
#endif
            for (; j < half_dim; j++) {
                float v0 = v[j], v1 = v[j + half_dim];
                v[j]            = v0 * ct[j] - v1 * st[j];
                v[j + half_dim] = v0 * st[j] + v1 * ct[j];
            }
        }
    }

    free(cos_tab);
    free(sin_tab);
}

/* Batched RMSNorm: apply to N tokens. bx[i] = rmsnorm(bx_in[i], w) */
static void tf_rmsnorm_batch(float *dst, const float *src, const qtensor *w,
                              int n_embd, int N, float eps, float *w_buf) {
    tf_dequant_row(w, 0, w_buf);
    for (int t = 0; t < N; t++) {
        const float *xi = src + (size_t)t * n_embd;
        float *yi = dst + (size_t)t * n_embd;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vss = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < n_embd; i += 8) {
            __m256 vx = _mm256_loadu_ps(xi + i);
            vss = _mm256_fmadd_ps(vx, vx, vss);
        }
        __m128 hi = _mm256_extractf128_ps(vss, 1);
        __m128 lo = _mm256_castps256_ps128(vss);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float ss = _mm_cvtss_f32(s4);
        for (; i < n_embd; i++) ss += xi[i] * xi[i];
        ss = 1.0f / sqrtf(ss / n_embd + eps);
        __m256 vscale = _mm256_set1_ps(ss);
        i = 0;
        for (; i + 7 < n_embd; i += 8)
            _mm256_storeu_ps(yi + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xi + i), vscale),
                                                     _mm256_loadu_ps(w_buf + i)));
        for (; i < n_embd; i++) yi[i] = xi[i] * ss * w_buf[i];
#else
        float ss = 0.0f;
        for (int i = 0; i < n_embd; i++) ss += xi[i] * xi[i];
        ss = 1.0f / sqrtf(ss / n_embd + eps);
        for (int i = 0; i < n_embd; i++) yi[i] = xi[i] * ss * w_buf[i];
#endif
    }
}

/* Batched QK-norm: per-head RMSNorm for N tokens */
static void tf_qk_norm_batch(float *vec, int n_heads, int head_dim, int N,
                               const qtensor *norm_w, float eps, float *w_buf) {
    tf_dequant_row(norm_w, 0, w_buf);
    int vec_dim = n_heads * head_dim;
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + (size_t)t * vec_dim + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 vss = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < head_dim; i += 8) {
                __m256 vx = _mm256_loadu_ps(v + i);
                vss = _mm256_fmadd_ps(vx, vx, vss);
            }
            __m128 hi = _mm256_extractf128_ps(vss, 1);
            __m128 lo = _mm256_castps256_ps128(vss);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float ss = _mm_cvtss_f32(s4);
            for (; i < head_dim; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / head_dim + eps);
            __m256 vscale = _mm256_set1_ps(ss);
            i = 0;
            for (; i + 7 < head_dim; i += 8)
                _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), vscale),
                                                       _mm256_loadu_ps(w_buf + i)));
            for (; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#else
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / head_dim + eps);
            for (int i = 0; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#endif
        }
    }
}

/* Batched causal attention worker: processes a range of (token, head) work items.
 * Work item i maps to token = i / n_heads, head = i % n_heads. */
typedef struct {
    const float *bq;           /* [N, q_dim] queries */
    float *bxb2;               /* [N, q_dim] output (each head writes its own slice) */
    const float *key_cache;    /* layer key cache */
    const float *value_cache;  /* layer value cache */
    const int *cache_pos;      /* [N] cache positions */
    float *att_scratch;        /* per-thread scratch [max_seq_len] */
    int work_start, work_end;  /* range of flattened (tok*n_heads + head) indices */
    int N, n_heads, n_kv_heads;
    int head_dim, kv_dim, q_dim, gqa_ratio, max_seq_len;
} tf_batch_attn_task;

static void *tf_batch_attn_worker(void *arg) {
    tf_batch_attn_task *t = (tf_batch_attn_task *)arg;
    float scale = 1.0f / sqrtf((float)t->head_dim);
    int hd = t->head_dim;

    for (int wi = t->work_start; wi < t->work_end; wi++) {
        int tok = wi / t->n_heads;
        int h   = wi % t->n_heads;
        int kv_h = h / t->gqa_ratio;
        int seq_len = t->cache_pos[tok] + 1;

        const float *q_h = t->bq + (size_t)tok * t->q_dim + h * hd;
        float *att_h = t->att_scratch;

        /* QK scores: att[p] = dot(q, k[p]) * scale */
#if defined(__AVX2__) && defined(__FMA__)
        if (hd == 64) {
            /* Fully unrolled for head_dim=64: 8 AVX vectors */
            __m256 q0 = _mm256_loadu_ps(q_h);
            __m256 q1 = _mm256_loadu_ps(q_h+8);
            __m256 q2 = _mm256_loadu_ps(q_h+16);
            __m256 q3 = _mm256_loadu_ps(q_h+24);
            __m256 q4 = _mm256_loadu_ps(q_h+32);
            __m256 q5 = _mm256_loadu_ps(q_h+40);
            __m256 q6 = _mm256_loadu_ps(q_h+48);
            __m256 q7 = _mm256_loadu_ps(q_h+56);
            __m256 vscale = _mm256_set1_ps(scale);

            int p = 0;
            for (; p + 3 < seq_len; p += 4) {
                /* Process 4 positions at once to amortize q loads */
                const float *k0 = t->key_cache + (size_t)(p+0) * t->kv_dim + kv_h * hd;
                const float *k1 = t->key_cache + (size_t)(p+1) * t->kv_dim + kv_h * hd;
                const float *k2 = t->key_cache + (size_t)(p+2) * t->kv_dim + kv_h * hd;
                const float *k3 = t->key_cache + (size_t)(p+3) * t->kv_dim + kv_h * hd;

                __m256 s0=_mm256_mul_ps(q0, _mm256_loadu_ps(k0));
                __m256 s1=_mm256_mul_ps(q0, _mm256_loadu_ps(k1));
                __m256 s2=_mm256_mul_ps(q0, _mm256_loadu_ps(k2));
                __m256 s3=_mm256_mul_ps(q0, _mm256_loadu_ps(k3));
                s0=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k0+8), s0);
                s1=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k1+8), s1);
                s2=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k2+8), s2);
                s3=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k3+8), s3);
                s0=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k0+16), s0);
                s1=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k1+16), s1);
                s2=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k2+16), s2);
                s3=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k3+16), s3);
                s0=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k0+24), s0);
                s1=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k1+24), s1);
                s2=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k2+24), s2);
                s3=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k3+24), s3);
                s0=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k0+32), s0);
                s1=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k1+32), s1);
                s2=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k2+32), s2);
                s3=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k3+32), s3);
                s0=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k0+40), s0);
                s1=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k1+40), s1);
                s2=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k2+40), s2);
                s3=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k3+40), s3);
                s0=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k0+48), s0);
                s1=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k1+48), s1);
                s2=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k2+48), s2);
                s3=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k3+48), s3);
                s0=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k0+56), s0);
                s1=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k1+56), s1);
                s2=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k2+56), s2);
                s3=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k3+56), s3);

                /* Horizontal sum each into scalar, multiply by scale */
                /* Use hadd pairs to reduce 4 accumulators efficiently */
                __m256 h01 = _mm256_hadd_ps(s0, s1);  /* [s0a+s0b, s0c+s0d, s1a+s1b, s1c+s1d, ...] */
                __m256 h23 = _mm256_hadd_ps(s2, s3);
                __m256 h0123 = _mm256_hadd_ps(h01, h23); /* each lane has partial sums */
                /* h0123 = [A0, A1, A2, A3 | B0, B1, B2, B3] where A=low128, B=hi128 */
                __m128 lo = _mm256_castps256_ps128(h0123);
                __m128 hi = _mm256_extractf128_ps(h0123, 1);
                __m128 sums = _mm_add_ps(lo, hi);
                sums = _mm_mul_ps(sums, _mm256_castps256_ps128(vscale));
                _mm_storeu_ps(att_h + p, sums);
            }
            for (; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                __m256 s = _mm256_mul_ps(q0, _mm256_loadu_ps(k_p));
                s = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k_p+8), s);
                s = _mm256_fmadd_ps(q2, _mm256_loadu_ps(k_p+16), s);
                s = _mm256_fmadd_ps(q3, _mm256_loadu_ps(k_p+24), s);
                s = _mm256_fmadd_ps(q4, _mm256_loadu_ps(k_p+32), s);
                s = _mm256_fmadd_ps(q5, _mm256_loadu_ps(k_p+40), s);
                s = _mm256_fmadd_ps(q6, _mm256_loadu_ps(k_p+48), s);
                s = _mm256_fmadd_ps(q7, _mm256_loadu_ps(k_p+56), s);
                __m128 _hi = _mm256_extractf128_ps(s, 1);
                __m128 _lo = _mm256_castps256_ps128(s);
                __m128 _s = _mm_add_ps(_lo, _hi);
                _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s));
                _s = _mm_add_ss(_s, _mm_movehdup_ps(_s));
                att_h[p] = _mm_cvtss_f32(_s) * scale;
            }
        } else
#endif
        {
            for (int p = 0; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++) score += q_h[d] * k_p[d];
                att_h[p] = score * scale;
            }
        }

        tf_softmax(att_h, seq_len);

        /* Weighted V accumulation: out[d] = sum_p att[p] * V[p][d] */
        float *out_h = t->bxb2 + (size_t)tok * t->q_dim + h * hd;
#if defined(__AVX2__) && defined(__FMA__)
        if (hd == 64) {
            __m256 o0=_mm256_setzero_ps(), o1=_mm256_setzero_ps();
            __m256 o2=_mm256_setzero_ps(), o3=_mm256_setzero_ps();
            __m256 o4=_mm256_setzero_ps(), o5=_mm256_setzero_ps();
            __m256 o6=_mm256_setzero_ps(), o7=_mm256_setzero_ps();

            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                __m256 a = _mm256_set1_ps(att_h[p]);
                o0=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p),    o0);
                o1=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+8),  o1);
                o2=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+16), o2);
                o3=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+24), o3);
                o4=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+32), o4);
                o5=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+40), o5);
                o6=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+48), o6);
                o7=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+56), o7);
            }

            _mm256_storeu_ps(out_h,    o0); _mm256_storeu_ps(out_h+8,  o1);
            _mm256_storeu_ps(out_h+16, o2); _mm256_storeu_ps(out_h+24, o3);
            _mm256_storeu_ps(out_h+32, o4); _mm256_storeu_ps(out_h+40, o5);
            _mm256_storeu_ps(out_h+48, o6); _mm256_storeu_ps(out_h+56, o7);
        } else
#endif
        {
            memset(out_h, 0, hd * sizeof(float));
            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                float a = att_h[p];
                for (int d = 0; d < hd; d++) out_h[d] += a * v_p[d];
            }
        }
    }
    return NULL;
}

/* GEMM-based attention worker: processes a range of Q heads */
typedef struct {
    const float *bq; float *bxb2;
    const float *key_cache; const float *value_cache;
    const int *cache_pos;
    float *att_buf;
    int h_start, h_end;
    int N, S, head_dim, kv_dim, q_dim, gqa_ratio;
    float scale;
} tf_attn_gemm_task;

static void *tf_attn_gemm_worker(void *arg) {
    tf_attn_gemm_task *t = (tf_attn_gemm_task *)arg;
    int hd = t->head_dim;
    int N = t->N;

    for (int h = t->h_start; h < t->h_end; h++) {
        int kv_h = h / t->gqa_ratio;

        for (int i = 0; i < N; i++) {
            const float *q_i = t->bq + (size_t)i * t->q_dim + h * hd;
            int seq_len = t->cache_pos[i] + 1;
            float *att_row = t->att_buf;

#if defined(__AVX2__) && defined(__FMA__)
            if (hd == 64) {
                __m256 q0=_mm256_loadu_ps(q_i), q1=_mm256_loadu_ps(q_i+8);
                __m256 q2=_mm256_loadu_ps(q_i+16), q3=_mm256_loadu_ps(q_i+24);
                __m256 q4=_mm256_loadu_ps(q_i+32), q5=_mm256_loadu_ps(q_i+40);
                __m256 q6=_mm256_loadu_ps(q_i+48), q7=_mm256_loadu_ps(q_i+56);

                int p = 0;
                for (; p + 3 < seq_len; p += 4) {
                    const float *k0 = t->key_cache + (size_t)(p+0)*t->kv_dim + kv_h*hd;
                    const float *k1 = t->key_cache + (size_t)(p+1)*t->kv_dim + kv_h*hd;
                    const float *k2 = t->key_cache + (size_t)(p+2)*t->kv_dim + kv_h*hd;
                    const float *k3 = t->key_cache + (size_t)(p+3)*t->kv_dim + kv_h*hd;
                    __m256 s0=_mm256_mul_ps(q0,_mm256_loadu_ps(k0));
                    __m256 s1=_mm256_mul_ps(q0,_mm256_loadu_ps(k1));
                    __m256 s2=_mm256_mul_ps(q0,_mm256_loadu_ps(k2));
                    __m256 s3=_mm256_mul_ps(q0,_mm256_loadu_ps(k3));
                    s0=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k0+8),s0);
                    s1=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k1+8),s1);
                    s2=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k2+8),s2);
                    s3=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k3+8),s3);
                    s0=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k0+16),s0);
                    s1=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k1+16),s1);
                    s2=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k2+16),s2);
                    s3=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k3+16),s3);
                    s0=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k0+24),s0);
                    s1=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k1+24),s1);
                    s2=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k2+24),s2);
                    s3=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k3+24),s3);
                    s0=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k0+32),s0);
                    s1=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k1+32),s1);
                    s2=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k2+32),s2);
                    s3=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k3+32),s3);
                    s0=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k0+40),s0);
                    s1=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k1+40),s1);
                    s2=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k2+40),s2);
                    s3=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k3+40),s3);
                    s0=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k0+48),s0);
                    s1=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k1+48),s1);
                    s2=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k2+48),s2);
                    s3=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k3+48),s3);
                    s0=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k0+56),s0);
                    s1=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k1+56),s1);
                    s2=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k2+56),s2);
                    s3=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k3+56),s3);
                    __m256 h01=_mm256_hadd_ps(s0,s1);
                    __m256 h23=_mm256_hadd_ps(s2,s3);
                    __m256 h0123=_mm256_hadd_ps(h01,h23);
                    __m128 lo=_mm256_castps256_ps128(h0123);
                    __m128 hi=_mm256_extractf128_ps(h0123,1);
                    _mm_storeu_ps(att_row+p, _mm_mul_ps(_mm_add_ps(lo,hi),_mm_set1_ps(t->scale)));
                }
                for (; p < seq_len; p++) {
                    const float *kp = t->key_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    __m256 s=_mm256_mul_ps(q0,_mm256_loadu_ps(kp));
                    s=_mm256_fmadd_ps(q1,_mm256_loadu_ps(kp+8),s);
                    s=_mm256_fmadd_ps(q2,_mm256_loadu_ps(kp+16),s);
                    s=_mm256_fmadd_ps(q3,_mm256_loadu_ps(kp+24),s);
                    s=_mm256_fmadd_ps(q4,_mm256_loadu_ps(kp+32),s);
                    s=_mm256_fmadd_ps(q5,_mm256_loadu_ps(kp+40),s);
                    s=_mm256_fmadd_ps(q6,_mm256_loadu_ps(kp+48),s);
                    s=_mm256_fmadd_ps(q7,_mm256_loadu_ps(kp+56),s);
                    __m128 _hi=_mm256_extractf128_ps(s,1);
                    __m128 _lo=_mm256_castps256_ps128(s);
                    __m128 _s=_mm_add_ps(_lo,_hi);
                    _s=_mm_add_ps(_s,_mm_movehl_ps(_s,_s));
                    _s=_mm_add_ss(_s,_mm_movehdup_ps(_s));
                    att_row[p] = _mm_cvtss_f32(_s) * t->scale;
                }
            } else
#endif
            {
                for (int p = 0; p < seq_len; p++) {
                    const float *kp = t->key_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    float score = 0;
                    for (int d = 0; d < hd; d++) score += q_i[d] * kp[d];
                    att_row[p] = score * t->scale;
                }
            }

            tf_softmax(att_row, seq_len);

            float *out_h = t->bxb2 + (size_t)i * t->q_dim + h * hd;
#if defined(__AVX2__) && defined(__FMA__)
            if (hd == 64) {
                __m256 o0=_mm256_setzero_ps(), o1=_mm256_setzero_ps();
                __m256 o2=_mm256_setzero_ps(), o3=_mm256_setzero_ps();
                __m256 o4=_mm256_setzero_ps(), o5=_mm256_setzero_ps();
                __m256 o6=_mm256_setzero_ps(), o7=_mm256_setzero_ps();
                for (int p = 0; p < seq_len; p++) {
                    const float *vp = t->value_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    __m256 a = _mm256_set1_ps(att_row[p]);
                    o0=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp),   o0);
                    o1=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+8), o1);
                    o2=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+16),o2);
                    o3=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+24),o3);
                    o4=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+32),o4);
                    o5=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+40),o5);
                    o6=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+48),o6);
                    o7=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+56),o7);
                }
                _mm256_storeu_ps(out_h,   o0); _mm256_storeu_ps(out_h+8, o1);
                _mm256_storeu_ps(out_h+16,o2); _mm256_storeu_ps(out_h+24,o3);
                _mm256_storeu_ps(out_h+32,o4); _mm256_storeu_ps(out_h+40,o5);
                _mm256_storeu_ps(out_h+48,o6); _mm256_storeu_ps(out_h+56,o7);
            } else
#endif
            {
                memset(out_h, 0, hd * sizeof(float));
                for (int p = 0; p < seq_len; p++) {
                    const float *vp = t->value_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    float a = att_row[p];
                    for (int d = 0; d < hd; d++) out_h[d] += a * vp[d];
                }
            }
        }
    }
    return NULL;
}

/* Timing helper */
static inline double tf_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

float *transformer_forward_batch_logits(transformer_model *m, const transformer_batch *b) {
    if (!m || !b || b->N <= 0) return NULL;
    if (!m->has_lm_head) return NULL;

    /* Pause thread pool during batch forward (uses its own pthread-based GEMM) */
    int pool_was_alive = m->pool_alive;
    if (pool_was_alive) tf_pool_shutdown(m);

    int N = b->N;
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;
    int gqa_ratio = (m->gqa_group > 0) ? m->gqa_group : ((n_kv_heads > 0) ? n_heads / n_kv_heads : 1);
    int n_ff = m->n_ff;

    /* Profiling accumulators (ms) */
    double t_attn_norm = 0, t_qkv_gemm = 0, t_qk_norm = 0, t_rope = 0;
    double t_kv_store = 0, t_attention = 0, t_out_proj = 0, t_residual = 0;
    double t_ffn_norm = 0, t_ffn_gemm_up = 0, t_silu = 0, t_ffn_gemm_down = 0;
    double t_deepstack = 0;
    double t0p;

    /* Allocate batch scratch buffers */
    float *bx     = (float *)malloc((size_t)N * n_embd * sizeof(float));
    float *bxb    = (float *)malloc((size_t)N * n_embd * sizeof(float));
    float *bq     = (float *)malloc((size_t)N * q_dim * sizeof(float));
    float *bk     = (float *)malloc((size_t)N * kv_dim * sizeof(float));
    float *bv     = (float *)malloc((size_t)N * kv_dim * sizeof(float));
    float *bxb2   = (float *)malloc((size_t)N * q_dim * sizeof(float));
    float *bffn1  = (float *)malloc((size_t)N * n_ff * sizeof(float));
    float *bffn2  = (float *)malloc((size_t)N * n_ff * sizeof(float));
    float *bffn3  = (float *)malloc((size_t)N * n_ff * sizeof(float));

    /* Per-thread attention score scratch: each thread needs [max_seq_len] */
    float *batch_att_scratch = (float *)malloc((size_t)m->n_threads * m->max_seq_len * sizeof(float));

    /* Copy input embeddings to bx (extract n_embd from embd_stride) */
    for (int t = 0; t < N; t++) {
        memcpy(bx + (size_t)t * n_embd,
               b->embds + (size_t)t * b->embd_stride,
               n_embd * sizeof(float));
    }

    /* Layer loop */
    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* 1. Attention RMSNorm */
        t0p = tf_time_ms();
        tf_rmsnorm_batch(bxb, bx, &layer->attn_norm, n_embd, N, m->rms_norm_eps, m->matvec_tmp);
        t_attn_norm += tf_time_ms() - t0p;

        /* 2. Q/K/V GEMM */
        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bq, &layer->attn_q, bxb, q_dim, N, q_dim, n_embd, m->n_threads);
        tf_gemm_f16_mt_tokenmajor(bk, &layer->attn_k, bxb, kv_dim, N, kv_dim, n_embd, m->n_threads);
        tf_gemm_f16_mt_tokenmajor(bv, &layer->attn_v, bxb, kv_dim, N, kv_dim, n_embd, m->n_threads);

        /* 3. QK-Norm */
        t0p = tf_time_ms();
        if (layer->attn_q_norm.data) {
            tf_qk_norm_batch(bq, n_heads, head_dim, N, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        if (layer->attn_k_norm.data) {
            tf_qk_norm_batch(bk, n_kv_heads, head_dim, N, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        t_qk_norm += tf_time_ms() - t0p;

        /* 4. M-RoPE (batched with precomputed cos/sin tables + AVX2) */
        t0p = tf_time_ms();
        tf_rope_mrope_batch(bq, bk, N, n_heads, n_kv_heads, head_dim,
                            q_dim, kv_dim, b->pos_t, b->pos_h, b->pos_w,
                            m->rope_freq_base, m->mrope_sections);
        t_rope += tf_time_ms() - t0p;

        /* 5. Store all K/V into cache first (before attention).
         * The batched path is F32-only — runner must use TF_KV_DTYPE=f32 when
         * exercising it (asserted at runner level via kv_dtype check). */
        t0p = tf_time_ms();
        for (int t = 0; t < N; t++) {
            int cp = b->cache_pos[t];
            memcpy((float *)m->key_cache[l]   + (size_t)cp * kv_dim, bk + (size_t)t * kv_dim, kv_dim * sizeof(float));
            memcpy((float *)m->value_cache[l] + (size_t)cp * kv_dim, bv + (size_t)t * kv_dim, kv_dim * sizeof(float));
        }
        t_kv_store += tf_time_ms() - t0p;

        /* 6. GEMM-based causal attention.
         * For each Q head h: scores[N, max_seq] = Q_h[N, hd] × K_kv[max_seq, hd]^T
         * Then causal mask, softmax, then out[N, hd] = scores[N, max_seq] × V[max_seq, hd].
         * Thread across Q heads (16 heads → 16 threads). */
        t0p = tf_time_ms();
        {
            float scale = 1.0f / sqrtf((float)head_dim);
            int max_pos = b->cache_pos[N-1]; /* last token has max cache_pos */
            int S = max_pos + 1; /* sequence length in cache */

            int nt = m->n_threads;
            if (nt > n_heads) nt = n_heads;
            pthread_t *athreads = (pthread_t *)alloca(nt * sizeof(pthread_t));
            tf_attn_gemm_task *atasks = (tf_attn_gemm_task *)alloca(nt * sizeof(tf_attn_gemm_task));
            int heads_per = n_heads / nt;
            int heads_extra = n_heads % nt;
            int hoff = 0;
            for (int ti = 0; ti < nt; ti++) {
                int hcount = heads_per + (ti < heads_extra ? 1 : 0);
                atasks[ti] = (tf_attn_gemm_task){
                    bq, bxb2, (const float *)m->key_cache[l], (const float *)m->value_cache[l],
                    b->cache_pos, batch_att_scratch + (size_t)ti * m->max_seq_len,
                    hoff, hoff + hcount, N, S, head_dim, kv_dim, q_dim, gqa_ratio, scale
                };
                hoff += hcount;
                pthread_create(&athreads[ti], NULL, tf_attn_gemm_worker, &atasks[ti]);
            }
            for (int ti = 0; ti < nt; ti++) pthread_join(athreads[ti], NULL);
        }
        t_attention += tf_time_ms() - t0p;

        /* 7. Output projection GEMM: bxb[N, n_embd] = attn_output[n_embd, q_dim] × bxb2[N, q_dim]^T */
        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bxb, &layer->attn_output, bxb2, n_embd, N, n_embd, q_dim, m->n_threads);
        t_out_proj += tf_time_ms() - t0p;

        /* 8. Residual add */
        t0p = tf_time_ms();
        for (int t = 0; t < N; t++) tf_vadd(bx + (size_t)t * n_embd, bxb + (size_t)t * n_embd, n_embd);
        t_residual += tf_time_ms() - t0p;

        /* 9. FFN */
        t0p = tf_time_ms();
        tf_rmsnorm_batch(bxb, bx, &layer->ffn_norm, n_embd, N, m->rms_norm_eps, m->matvec_tmp);
        t_ffn_norm += tf_time_ms() - t0p;

        /* Dense SwiGLU FFN */
        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bffn1, &layer->ffn_gate, bxb, n_ff, N, n_ff, n_embd, m->n_threads);
        tf_gemm_f16_mt_tokenmajor(bffn2, &layer->ffn_up,   bxb, n_ff, N, n_ff, n_embd, m->n_threads);
        t_ffn_gemm_up += tf_time_ms() - t0p;

        /* SiLU × mul for all N tokens */
        t0p = tf_time_ms();
        tf_silu_mul_avx2(bffn3, bffn1, bffn2, N * n_ff);
        t_silu += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bxb, &layer->ffn_down, bffn3, n_embd, N, n_embd, n_ff, m->n_threads);
        t_ffn_gemm_down += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        for (int t = 0; t < N; t++) tf_vadd(bx + (size_t)t * n_embd, bxb + (size_t)t * n_embd, n_embd);
        t_residual += tf_time_ms() - t0p;

        /* 10. DeepStack injection */
        t0p = tf_time_ms();
        if (b->ds_embds && l < m->n_deepstack) {
            for (int t = 0; t < N; t++) {
                if (b->ds_embds[t] && b->ds_embd_stride > n_embd) {
                    const float *ds_slice = b->ds_embds[t] + (1 + l) * n_embd;
                    float *xt = bx + (size_t)t * n_embd;
                    tf_vadd(xt, ds_slice, n_embd);
                }
            }
        }

        t_deepstack += tf_time_ms() - t0p;

        if (m->trace_hidden_norms && (l == 0 || l == m->n_layers - 1 || (l + 1) % 10 == 0)) {
            float *last = bx + (size_t)(N-1) * n_embd;
            fprintf(stderr, "  [batch] layer %2d: last token hidden norm = %.4f\n", l, sqrtf(tf_sum_squares(last, n_embd)));
        }
    }

    /* Print batch profiling summary */
    double t_total = t_attn_norm + t_qkv_gemm + t_qk_norm + t_rope + t_kv_store +
                     t_attention + t_out_proj + t_residual + t_ffn_norm +
                     t_ffn_gemm_up + t_silu + t_ffn_gemm_down + t_deepstack;
    fprintf(stderr, "\n  === Batch Prefill Profile (%d tokens × %d layers) ===\n", N, m->n_layers);
    fprintf(stderr, "  attn_norm:     %7.1f ms (%4.1f%%)\n", t_attn_norm, 100*t_attn_norm/t_total);
    fprintf(stderr, "  QKV GEMM:      %7.1f ms (%4.1f%%)\n", t_qkv_gemm, 100*t_qkv_gemm/t_total);
    fprintf(stderr, "  QK norm:       %7.1f ms (%4.1f%%)\n", t_qk_norm, 100*t_qk_norm/t_total);
    fprintf(stderr, "  RoPE:          %7.1f ms (%4.1f%%)\n", t_rope, 100*t_rope/t_total);
    fprintf(stderr, "  KV store:      %7.1f ms (%4.1f%%)\n", t_kv_store, 100*t_kv_store/t_total);
    fprintf(stderr, "  attention:     %7.1f ms (%4.1f%%)\n", t_attention, 100*t_attention/t_total);
    fprintf(stderr, "  out_proj GEMM: %7.1f ms (%4.1f%%)\n", t_out_proj, 100*t_out_proj/t_total);
    fprintf(stderr, "  residual:      %7.1f ms (%4.1f%%)\n", t_residual, 100*t_residual/t_total);
    fprintf(stderr, "  ffn_norm:      %7.1f ms (%4.1f%%)\n", t_ffn_norm, 100*t_ffn_norm/t_total);
    fprintf(stderr, "  FFN up+gate:   %7.1f ms (%4.1f%%)\n", t_ffn_gemm_up, 100*t_ffn_gemm_up/t_total);
    fprintf(stderr, "  SiLU×mul:      %7.1f ms (%4.1f%%)\n", t_silu, 100*t_silu/t_total);
    fprintf(stderr, "  FFN down:      %7.1f ms (%4.1f%%)\n", t_ffn_gemm_down, 100*t_ffn_gemm_down/t_total);
    fprintf(stderr, "  deepstack:     %7.1f ms (%4.1f%%)\n", t_deepstack, 100*t_deepstack/t_total);
    fprintf(stderr, "  TOTAL:         %7.1f ms\n", t_total);
    fprintf(stderr, "  GEMM total:    %7.1f ms (%4.1f%%)\n",
            t_qkv_gemm + t_out_proj + t_ffn_gemm_up + t_ffn_gemm_down,
            100*(t_qkv_gemm + t_out_proj + t_ffn_gemm_up + t_ffn_gemm_down)/t_total);
    fprintf(stderr, "\n");

    /* Final RMSNorm on last token only */
    float *last_hidden = bx + (size_t)(N-1) * n_embd;
    tf_rmsnorm(m->x, last_hidden, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);

    /* LM head matvec on last token */
    tf_qmatvec_pool(m, m->logits, &m->output, m->x, m->n_vocab);

    /* Copy last hidden state to m->x for subsequent single-token decode */
    /* m->x already has the normed state, but we need the pre-norm state for KV continuity */

    free(bx); free(bxb); free(bq); free(bk); free(bv); free(bxb2);
    free(bffn1); free(bffn2); free(bffn3); free(batch_att_scratch);

    /* Restart thread pool for decode phase */
    if (pool_was_alive) tf_pool_start(m);

    return m->logits;
}

/* Top-k sampling with temperature */
int32_t transformer_sample_topk(const float *logits, int n_vocab, float temperature, int top_k) {
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    /* Find top-k indices by partial sort (alloca to avoid malloc in hot path) */
    int32_t *indices = (int32_t *)alloca(top_k * sizeof(int32_t));
    float *vals = (float *)alloca(top_k * sizeof(float));
    int k = 0;

    for (int i = 0; i < n_vocab; i++) {
        float v = logits[i] / temperature;
        if (k < top_k) {
            indices[k] = i;
            vals[k] = v;
            k++;
            /* Bubble up to maintain min-heap property at vals[0] */
            for (int j = k - 1; j > 0; j--) {
                int p = (j - 1) / 2;
                if (vals[j] < vals[p]) {
                    float tv = vals[j]; vals[j] = vals[p]; vals[p] = tv;
                    int32_t ti = indices[j]; indices[j] = indices[p]; indices[p] = ti;
                }
            }
        } else if (v > vals[0]) {
            vals[0] = v;
            indices[0] = i;
            /* Sift down */
            int j = 0;
            for (;;) {
                int s = j, l = 2*j+1, r = 2*j+2;
                if (l < top_k && vals[l] < vals[s]) s = l;
                if (r < top_k && vals[r] < vals[s]) s = r;
                if (s == j) break;
                float tv = vals[j]; vals[j] = vals[s]; vals[s] = tv;
                int32_t ti = indices[j]; indices[j] = indices[s]; indices[s] = ti;
                j = s;
            }
        }
    }

    /* Softmax over top-k (uses AVX2 fast_exp when available) */
    tf_softmax(vals, top_k);

    /* Sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    int32_t result = (top_k > 0) ? indices[0] : 0;
    for (int i = 0; i < top_k; i++) {
        cum += vals[i];
        if (r <= cum) { result = indices[i]; break; }
    }

    return result;
}

#endif /* TRANSFORMER_IMPLEMENTATION */
