/*
 * ds4f.h - DeepSeek-V4-Flash synthetic inference harness for A64FX/Fugaku.
 *
 * Perf/plumbing harness: weights are SYNTHETIC (deterministic junk filled in
 * HBM, no disk load), logits are MEANINGLESS. Validates the parallel forward
 * pass shape/FLOPs/throughput/lockstep/memory-fit, NOT output quality.
 *
 * Weights stay quantized in HBM and are dequantized on demand per token:
 *   - dense (MLA attn + shared expert) = FP8 E4M3, 128x128 block E8M0 scale
 *   - routed experts                   = cfg.expert_qt: split MXFP4 (e2m1), per-32 E8M0
 *                                        scale for Flash/Pro; FP8 E4M3 + 128x128 block
 *                                        E8M0 scale for base (DS4F_MODEL=ds4fbase)
 *   - embed / head / router / norms    = BF16 ;  attn_sink / mHC = F32
 * Kernels (matvec_fp8e4m3_8row / matvec_mxfp4_8row / matvec_bf16_8row) and the
 * e8m0/fp8 helpers live in ggml_dequant.h (validated by ds4f_kernels_bench.c).
 *
 * Parallelism = expert-parallel (EP): experts e with e%ep_size==ep_rank live
 * on this node (~23/node at ep_size=11); dense is replicated and computed
 * redundantly. Single-node Stage 1 uses ep_size=11/ep_rank=0 (no comm) so it
 * fits ~21 GB; the runner adds the uTofu MoE combine for the 11-node stages.
 *
 * Tensor layouts (from ~/models/ds4f safetensors headers, layer 0):
 *   attn.wq_a   F8_E4M3 [1024,4096]   scale [8,32]
 *   attn.q_norm BF16    [1024]
 *   attn.wq_b   F8_E4M3 [32768,1024]  scale [256,8]      (64 heads x 512)
 *   attn.wkv    F8_E4M3 [512,4096]    scale [4,32]       (kv latent, 1 kv head)
 *   attn.kv_norm BF16   [512]
 *   attn.attn_sink F32  [64]
 *   attn.wo_a   F8_E4M3 [8192,4096]   scale [64,32]
 *   attn.wo_b   F8_E4M3 [4096,8192]   scale [32,64]
 *   ffn.gate.weight BF16 [256,4096]   (router)
 *   ffn.shared_experts.w1/w3 F8 [2048,4096] scale[16,32]; w2 F8 [4096,2048] scale[32,16]
 *   ffn.experts.N.w1/w3 I8 [2048,2048] scale[2048,128];  w2 I8 [4096,1024] scale[4096,64]
 *     (ds4fbase instead: w1/w3 F8 [2048,4096] scale[16,32]; w2 F8 [4096,2048] scale[32,16]
 *      -- i.e. the experts take the SAME shape as the shared expert above)
 *   hc_attn_* / hc_ffn_* F32 (small, FLOP stand-in)
 */
#ifndef DS4F_H
#define DS4F_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <alloca.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include "ggml_dequant.h"
#if !defined(__ARM_FEATURE_SVE)
/* Portable stand-ins for the SVE-only small kernels in ds4f_impl.h. */
#include "ds4f_kernels_x86.h"
#endif

/* Weight quant types. Declared up here because ds4f_config carries one (expert_qt);
 * the per-type layout notes live above the ds4f_tensor definition below. */
typedef enum { DS4F_BF16 = 0, DS4F_FP8 = 1, DS4F_MXFP4 = 2, DS4F_F32 = 3, DS4F_BF16_PV = 4, DS4F_Q8_PV = 5 } ds4f_qtype;

/* ===================== config ===================== */
typedef struct {
    int n_layers;       /* 43 */
    int hidden;         /* 4096 */
    int vocab;          /* 129280 */
    /* MLA */
    int n_heads;        /* 64 */
    int q_head_dim;     /* 512  (qk_nope 448 + qk_rope 64) */
    int qk_rope_dim;    /* 64 */
    int q_lora;         /* 1024 */
    int kv_lora;        /* 512  (compressed latent, 1 kv head) */
    int o_inter;        /* 8192 (wo_a out / wo_b in) */
    int o_groups;       /* 8    (32768 -> 4096 reduction before wo_a) */
    /* MoE */
    int n_experts;      /* 256 */
    int n_active;       /* 6 */
    int moe_inter;      /* 2048 */
    int shared_inter;   /* 2048 */
    float routed_scale; /* 1.5 */
    /* mHC (Manifold-Constrained Hyper-Connections): the hidden state is hc_mult
     * residual streams [hc_mult, hidden], not a single vector. Each block collapses
     * 4->1 (hc_pre) before attn/ffn and expands 1->4 (hc_post) after; a final
     * hc_head collapses 4->1 before lm_head. mix_hc = (2+hc_mult)*hc_mult = 24. */
    int hc_mult;        /* 4 */
    int hc_iters;       /* 20 sinkhorn iters */
    float hc_eps;       /* 1e-6 */
    float norm_eps;     /* 1e-6 (RMSNorm + mHC pre-norm) */
    /* sparse lightning indexer (Stage 4): per-layer compress_ratios pick CSA(4)/
     * HCA(128); ratio 0 = dense full attention (layers 0,1,last). The indexer
     * scores nP/ratio compressed blocks over index_head_dim dims, selects the
     * top index_topk original positions, and the expensive weighted-V runs only
     * over them -> O(topk) instead of O(nP) at long context. */
    int index_topk;     /* 512 */
    int index_head_dim; /* 128 */
    int index_n_heads;  /* 64  (Tier-B2 indexer scoring heads) */
    int compress_ratios[64];  /* per-layer; [0..n_layers), 0 = dense */
    /* exact forward math (DS4F_EXACT; the synthetic stand-ins ignore these) */
    int o_lora;             /* 1024 (o_lora_rank; o_inter = o_groups*o_lora) */
    int window_size;        /* 128 sliding-window attention */
    int n_hash_layers;      /* 3 (layers 0..2 route by token id, no bias) */
    float swiglu_limit;     /* 10.0 (clamp up to [-lim,lim], gate to [...,lim]) */
    /* RoPE/YaRN (precompute_freqs_cis): dense layers use rope_theta + no YaRN;
     * sparse layers use compress_rope_theta + YaRN(original_seq_len,factor,beta_*) */
    float rope_theta;            /* 10000 */
    float compress_rope_theta;   /* 160000 */
    int   rope_factor;           /* 16 */
    int   beta_fast, beta_slow;  /* 32, 1 */
    int   original_seq_len;      /* 65536 (YaRN orig ctx; 0 => YaRN off) */
    /* Routed-expert weight format. This is the ONLY thing that differs between the
     * Flash release (~/models/ds4f, expert_dtype=fp4 -> DS4F_MXFP4) and the base
     * release (~/models/ds4fbase, expert_dtype=fp8 -> DS4F_FP8). Everything else --
     * dims, tensor names, tokenizer, dense FP8 -- is identical. */
    ds4f_qtype expert_qt;
    /* runtime */
    int max_pos;        /* KV cache capacity */
} ds4f_config;

static inline ds4f_config ds4f_default_config(void) {
    ds4f_config c = {0};
    c.n_layers = 43; c.hidden = 4096; c.vocab = 129280;
    c.n_heads = 64; c.q_head_dim = 512; c.qk_rope_dim = 64;
    c.q_lora = 1024; c.kv_lora = 512; c.o_inter = 8192; c.o_groups = 8;
    c.n_experts = 256; c.n_active = 6; c.moe_inter = 2048; c.shared_inter = 2048;
    c.routed_scale = 1.5f; c.max_pos = 4096;
    c.expert_qt = DS4F_MXFP4;            /* Flash: expert_dtype=fp4 (see ds4f_base_config) */
    c.hc_mult = 4; c.hc_iters = 20; c.hc_eps = 1e-6f; c.norm_eps = 1e-6f;
    /* lightning indexer (matches DeepSeek-V4-Flash config.json) */
    c.index_topk = 512; c.index_head_dim = 128; c.index_n_heads = 64;
    /* exact-forward hyperparameters (config.json) */
    c.o_lora = c.o_inter / c.o_groups;   /* 1024 */
    c.window_size = 128; c.n_hash_layers = 3; c.swiglu_limit = 10.0f;
    c.rope_theta = 10000.0f; c.compress_rope_theta = 160000.0f;
    c.rope_factor = 16; c.beta_fast = 32; c.beta_slow = 1; c.original_seq_len = 65536;
    /* compress_ratios: EXACT config.json array (layers 0,1 dense; then CSA(4)/HCA(128)
     * alternating even/odd; index 43 dense=MTP). Picks per-layer RoPE config in exact
     * mode (ratio!=0 => YaRN + compress_rope_theta). Only consulted when sparse/exact. */
    static const int RATIOS[44] = {
        0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
        4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0 };
    for (int L = 0; L < 64; L++) c.compress_ratios[L] = (L < 44) ? RATIOS[L] : 0;
    return c;
}

/* DeepSeek-V4-Pro (~/models/ds4p, 805 GB / 64 shards): same deepseek_v4 graph as
 * Flash (identical tensor names, tokenizer, MXFP4 expert packing, kv_lora=512,
 * wo_a in-dim n_heads*q_head_dim/o_groups = 4096 in both) — pure dimension scaling.
 * NOTE: no dense (ratio-0) decode layers — layers 0,1 are HCA(128); index 61 = MTP. */
static inline ds4f_config ds4f_pro_config(void) {
    ds4f_config c = ds4f_default_config();
    c.n_layers = 61; c.hidden = 7168;
    c.n_heads = 128;                       /* q_head_dim/qk_rope unchanged (512/64) */
    c.q_lora = 1536;
    c.o_inter = 16384; c.o_groups = 16;    /* o_lora stays 16384/16 = 1024 */
    c.o_lora = c.o_inter / c.o_groups;
    c.n_experts = 384; c.moe_inter = 3072; c.shared_inter = 3072;
    c.routed_scale = 2.5f;
    c.index_topk = 1024;                   /* index_head_dim/index_n_heads unchanged */
    static const int RATIOS_P[62] = {
        128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0 };
    for (int L = 0; L < 64; L++) c.compress_ratios[L] = (L < 62) ? RATIOS_P[L] : 0;
    return c;
}

/* DeepSeek-V4 BASE (~/models/ds4fbase, 275 GB / 46 shards): the SAME graph and the SAME
 * dimensions as Flash -- config.json differs in exactly one field, expert_dtype fp8 (vs fp4).
 * So the routed experts are FP8-e4m3 with a 128x128 block scale (byte-for-byte the layout the
 * dense tensors already use) instead of MXFP4, and they cost 24 MiB/expert instead of 12.75.
 *
 * The base safetensors also store every *.scale as F32 rather than F8_E8M0. The values are exact
 * powers of two (config.json: scale_fmt="ue8m0"), so ds4f_stage.c folds them losslessly to E8M0
 * bytes at stage time -- by the time the loader sees the blob, base and Flash scales are identical
 * and no kernel here knows the difference.
 *
 * Memory: FP8 experts are 22.17 GiB/node at EP=12 (vs 12.85 for Flash at EP=11), so base needs the
 * dense-TP stack (DS4F_TP_ATTN/OPROJ/WOB/SHARED/HEAD/EMBED) to fit a 32 GB node -- 24.5 GiB with TP
 * vs 30.5 without. See run_ds4fbase_12n.sh. */
static inline ds4f_config ds4f_base_config(void) {
    ds4f_config c = ds4f_default_config();
    c.expert_qt = DS4F_FP8;              /* the only delta vs Flash */
    return c;
}

/* DS4F_MODEL selects the variant: ds4p = Pro, ds4fbase = base, default = Flash.
 *
 * DS4F_EXPERTS=q8pv then overrides the routed-expert rep to the OFFLINE-BAKED int8 W8A8
 * (ds4f_bake.c with DS4F_BAKE_EXPERTS=1). Only meaningful for base: its FP8 experts run the
 * on-demand gather at ~76 GB/s in-model, 5x below the q8-sdot ceiling, and Q8_PV is 1.03 B/elem
 * vs FP8's 1.0 -- essentially free. It also cuts COMM, because the per-layer MoE imbalance
 * (top-6-of-256 landing on <=6 of 12 ranks) IS the straggler skew the all-reduce absorbs.
 * NOT for Flash: MXFP4 -> Q8 would double its experts (11.8 -> 22.9 GiB) and blow the arena. */
static inline ds4f_config ds4f_config_from_env(void) {
    const char *m = getenv("DS4F_MODEL");
    ds4f_config c = (m && strcmp(m, "ds4p") == 0)     ? ds4f_pro_config()
                  : (m && strcmp(m, "ds4fbase") == 0) ? ds4f_base_config()
                                                      : ds4f_default_config();
    {   const char *e = getenv("DS4F_EXPERTS");
        if (e && *e && strcmp(e, "q8pv") == 0) c.expert_qt = DS4F_Q8_PV;
    }
    return c;
}

/* ===================== tensor ===================== */
/* DS4F_BF16_PV: BF16 weights in the pair-interleaved layout consumed by
 * matvec_bf16_8row_pv (p_odd predicated load, +22..28% over plain bf16,
 * BYTE-IDENTICAL result). Layout is tied to the TYPE so fill and read can
 * never disagree — use it ONLY for tensors read via the pv matvec (head,
 * router, predequant dense), never for flat-read norms/embed. Same bytes as
 * DS4F_BF16, no scale. */
/* DS4F_Q8_PV: int8 W8A8 weights in the q8_pv "group" layout (per 8 rows x K
 * cols, K%64==0: nb=K/64 blocks of 528 B = 8 fp16 row-scales + 8 rows x 64 int8).
 * Consumed by matvec_sdot_8row (svdot_s32, 4x int8 MACs/lane). ~1.03 B/elem,
 * and svdot is 4x FMLA throughput -> the only sub-f32 dense lever. Argmax-exact
 * but NOT rel<1e-3 (int8 rounding ~1e-2); used ONLY for the big hidden-layer
 * dense GEMMs (qkv/o_proj/shared), never the argmax-critical router/lm-head. */

typedef struct ds4f_tensor {
    void    *w;       /* weight bytes */
    uint8_t *scale;   /* E8M0 scale bytes (NULL for BF16/F32) */
    ds4f_qtype type;
    int rows, cols;   /* logical [rows, cols] */
    int gpu_id;       /* optional shared dense-device-bank id; -1 means CPU */
} ds4f_tensor;

/* bytes of the weight body for a logical [rows,cols] of the given type */
static inline size_t ds4f_wbytes(ds4f_qtype t, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    switch (t) {
        case DS4F_BF16:    return n * 2;
        case DS4F_BF16_PV: return n * 2;   /* same bytes, pair-interleaved layout */
        case DS4F_FP8:     return n;       /* 1 B/elem */
        case DS4F_MXFP4:   return n / 2;   /* 2 nibbles/byte */
        case DS4F_F32:     return n * 4;
        case DS4F_Q8_PV:   return (size_t)(rows / 8) * (cols / 64) * 528;  /* ~1.03 B/elem */
    }
    return 0;
}
static inline size_t ds4f_sbytes(ds4f_qtype t, int rows, int cols) {
    switch (t) {
        case DS4F_FP8:   return (size_t)((rows + 127) / 128) * ((cols + 127) / 128);
        case DS4F_MXFP4: return (size_t)rows * (cols / 32);
        default:         return 0;
    }
}

/* ===================== layer / model ===================== */
typedef struct ds4f_layer {
    /* norms (BF16) */
    uint16_t *attn_norm, *ffn_norm, *q_norm, *kv_norm;
    /* MLA (FP8) */
    ds4f_tensor wq_a, wq_b, wkv, wo_a, wo_b;
    float *attn_sink;                 /* [n_heads] F32 */
    /* MoE */
    ds4f_tensor gate;                 /* BF16 [n_experts, hidden] router */
    float *gate_bias;                 /* [n_experts] F32 selection bias (exact, layers>=n_hash); NULL=hash/synth */
    ds4f_tensor sh_w1, sh_w2, sh_w3;  /* shared expert (FP8) */
    ds4f_tensor *ex_w1, *ex_w2, *ex_w3; /* owned experts (cfg.expert_qt: MXFP4 | FP8), 0..n_owned-1 */
    int *owned_eid;                   /* global expert id of each owned slot */
    int  n_owned;
    /* mHC (F32): fn = [mix_hc=24, hc_mult*hidden=16384] Linear; base = [24] bias;
     * scale = [3] (pre/post/comb gates). Exact math in ds4f_hc_pre/post. */
    float *hc_attn_fn, *hc_ffn_fn;    /* [24,16384] */
    float *hc_attn_base, *hc_ffn_base;  /* [24] */
    float *hc_attn_scale, *hc_ffn_scale; /* [3] */
    /* per-layer KV cache: [max_pos, kv_lora] BF16 (latent, 1 kv head).
     * 2-byte storage halves the long-context KV footprint (the decode memory
     * dominator) vs f32. BF16 (not FP16): the model is natively bfloat16 and the
     * kv latents carry massive-activation dims (~1e4..1e5) that overflow FP16's
     * 65504 ceiling -> Inf -> NaN; BF16 shares f32's exponent range and the
     * model's 8-bit-mantissa reference precision (faithful, not lossy). Codec:
     * ds4f_bf16f (bf16->f32 lsl) / ds4f_f32bf (f32->bf16 round) in ds4f_impl.h. */
    uint16_t *kv_cache;
    /* kv_cache capacity in positions, and the modulus for ring indexing. Under tierb2,
     * sparse (compress_ratio!=0) layers only ever read the last window_size positions of
     * kv_cache (older history is served by cmp_kv), so they ring-buffer at
     * kv_slots==window_size; everything else (dense layers, non-tierb2 modes, int8_kv)
     * keeps kv_slots==max_pos. All kv_cache indexing is (idx % kv_slots) -- when
     * kv_slots==max_pos the modulus is a no-op (idx<max_pos), so it is bit-exact. Caps the
     * kv_cache ctx-scaling cost to the 2 dense layers (long-ctx memory lever -> ~1M). */
    int kv_slots;
    /* ---- int8 KV cache (DS4F_INT8_KV; S5 static per-channel scale) ----
     * Halves the KV footprint (the 256k-ctx memory dominator). The kv latent's
     * massive-activation channels (~1e3..1e5, positionally CONSISTENT) make naive
     * per-token int8 catastrophic (one sink dim sets the scale -> O(1) dims -> 0).
     * The viable layout is a STATIC per-channel scale calibrated on the first ~CAL
     * positions and applied to all later positions (sinks are positionally stable
     * so early calibration holds). When DS4F_INT8_KV is on, kv_cache is NULL and the
     * store is kv_q (int8); reads dequant via kv_scale[channel]. Streaming: positions
     * [0,CAL) stage to kv_calbuf (bf16) while the per-channel absmax accumulates; at
     * pos>=CAL the scale freezes, calbuf is quantized into kv_q, and the read path
     * switches to int8 (single hoisted branch on kv_frozen). LOSSY (~1% rel) ->
     * argmax NOT bit-exact; coherence is the gate. Only the streaming exact/tierb2
     * decode path is wired (batched prefill is incompatible & unused there). */
    int8_t   *kv_q;        /* [max_pos, kv_lora] int8 store (NULL unless int8_kv) */
    float    *kv_scale;    /* [kv_lora] per-channel dequant scale (absmax/127) */
    float    *kv_iscale;   /* [kv_lora] 1/scale (quantize) */
    float    *kv_absmax;   /* [kv_lora] running absmax during calibration */
    uint16_t *kv_calbuf;   /* [CAL, kv_lora] bf16 staging until freeze */
    int       kv_caln;     /* positions staged in calbuf */
    int       kv_frozen;   /* 1 once scale frozen & calbuf quantized into kv_q */
    /* ---- Tier-B2 (DS4F_TIERB2; off-arena, only on ratio!=0 layers) ----
     * The stateful compressor/indexer long-range compressed-KV term that Tier-B1
     * omits. The matvec weights are stored bf16 (sources are bf16/FP8-e4m3, both
     * lossless into bf16) and widened in-lane by the pooled SVE kernels — half the
     * bytes, bit-identical to the prior f32-widened arena. ape stays f32, norm bf16.
     * All NULL unless m->tierb2 && compress_ratio!=0. */
    uint16_t *cmp_wkv, *cmp_wgate;   /* layer compressor (rotate=0): [coff*kv_lora, hidden] bf16 */
    float    *cmp_ape;               /* [compress_ratio, coff*kv_lora] */
    uint16_t *cmp_norm;              /* [kv_lora] bf16 */
    float    *cmp_kv_state, *cmp_score_state;  /* [coff*ratio, coff*kv_lora] ring state */
    float    *cmp_kv;                /* [max_pos/ratio, kv_lora] compressed latents (NULL if int8_cmp) */
    /* ---- int8 compressed-latent cache (DS4F_INT8_CMP) ----
     * Same S5 static-per-channel scheme as kv_q above, applied to cmp_kv (slot-indexed,
     * slot=pos/ratio). cmp_kv is a kv_lora latent feeding the same attention softmax as
     * the int8 KV window, so per-channel-static is the de-risked layout. When on, cmp_kv
     * is NULL and the store is cmp_q (int8); reads dequant via cmp_scale[channel]. Slots
     * [0,CAL) stage to cmp_calbuf (bf16) accumulating per-channel absmax; at slot>=CAL the
     * scale freezes, calbuf -> cmp_q, reads switch to int8. cmp_kv costs the bulk of the
     * tierb2 physical (THP) footprint at high ctx; int8 reclaims ~3/4. LOSSY -> coherence
     * gate, not bit-exact. Only the streaming exact/tierb2 path is wired. */
    int8_t   *cmp_q;       /* [max_pos/ratio, kv_lora] int8 store (NULL unless int8_cmp && !int4_cmp) */
    uint8_t  *cmp_q4;      /* [max_pos/ratio, kv_lora/2] int4 store, 2 signed-nibbles/byte (DS4F_INT4_CMP).
                            * Halves the dominant ctx-cache (2768->1384 B/pos). Same S5 per-channel
                            * cmp_scale, range +/-7. LOSSY (16 levels) -> coherence gate, not bit-exact. */
    int       cp_on;       /* DS4F_CP: cmp_q4 slot-sharded ([0,CAL) replicated + [cp_t0,cp_t1) tail) */
    int       cp_t0, cp_t1;/* this node's owned compressed-slot range in the sharded tail [CAL, nslot) */
    int       cp_nslot;    /* cmp_q4 slot capacity (DEBUG bounds guards) */
    float    *cmp_scale;   /* [kv_lora] per-channel dequant scale (absmax/127) */
    float    *cmp_iscale;  /* [kv_lora] 1/scale (quantize) */
    float    *cmp_absmax;  /* [kv_lora] running absmax during calibration */
    uint16_t *cmp_calbuf;  /* [CAL, kv_lora] bf16 staging until freeze */
    int       cmp_caln;    /* slots staged in calbuf */
    int       cmp_frozen;  /* 1 once scale frozen & calbuf quantized into cmp_q */
    /* indexer (CSA layers, compress_ratio==4 only) */
    uint16_t *idx_wq_b;              /* [index_n_heads*index_head_dim, q_lora] bf16 (FP8 src, lossless) */
    int8_t   *idx_wq_b_i8;           /* DS4F_IDX_INT8W: int8 W8A8 qproj weight (lazy, lossy, gated) */
    float    *idx_wq_b_sc;           /* per-row absmax scale for idx_wq_b_i8 */
    uint16_t *idx_wproj;             /* [index_n_heads, hidden] bf16 */
    uint16_t *idx_cmp_wkv, *idx_cmp_wgate;  /* indexer compressor (rotate=1): [coff*index_head_dim, hidden] bf16 */
    float    *idx_cmp_ape;           /* [4, coff*index_head_dim] */
    uint16_t *idx_cmp_norm;          /* [index_head_dim] bf16 */
    float    *idx_cmp_kv_state, *idx_cmp_score_state;  /* [coff*4, coff*index_head_dim] */
    float    *idx_kv;                /* [max_pos/ratio, index_head_dim] indexer compressed */
    int8_t   *idx_kv8;               /* [max_pos/ratio, index_head_dim] resident int8 mirror (DS4F_IDX_INT8) */
    uint8_t  *idx_kv8_4;             /* [max_pos/ratio, index_head_dim/2] int4 (2/byte, +/-7), DS4F_IDX_INT4:
                                      * half of idx_kv8 (672->336 B/pos); scan unpacks->int8 temp then svdot */
    float    *idx_pscale;            /* [max_pos/ratio] per-position scale (absmax/127 int8, /7 int4) */
    int       idx_cp_on;             /* DS4F_CP_IDX: idx_kv8_4/idx_pscale slot-sharded [idx_cp_s0,idx_cp_s1) */
    int       idx_cp_s0, idx_cp_s1;  /* this node's owned indexer-slot range (per-slot scale -> no CAL replication) */
    int       idx_cp_nslot;          /* idx_kv8_4 slot capacity (DEBUG bounds guard) */
    /* DS4F_IDX_REUSE: per-layer cached indexer selection (offset-stripped local cmp indices) + the position
     * it was scanned at. Re-scan every N decode steps, reuse in between (the O(T) scan+topk is the only
     * ctx-growing decode term; the selection drifts slowly). Lazily allocated. */
    int      *sel_cache; int sel_cache_n, sel_cache_pos;
} ds4f_layer;

/* Batched concurrent decode (DS4F_DECODE_BATCH): the per-sequence DATA/STATE cache buffers for one
 * (sequence, layer). The batched forward swaps these into ds4f_layer per batch element so the tested
 * per-position attn/tb2/KV-append run unchanged. Weights (cmp_wkv, idx_wq_b, ...) are shared, never
 * swapped. bf16/f32 caches only (int8/int4 modes are a later phase). */
typedef struct {
    uint16_t *kv_cache;
    float    *cmp_kv, *cmp_kv_state, *cmp_score_state;
    float    *idx_kv, *idx_cmp_kv_state, *idx_cmp_score_state;
    /* int8/int4 quantized stores + their per-sequence calibration (DS4F_INT8_KV / INT8_CMP / INT4_CMP /
     * IDX_INT8 / IDX_INT4, needed for batched-decode under those modes incl. CP). NULL when the mode is
     * off. cp_on/cp_t0/cp_t1 stay topology-constant on the base layer (same shard for every sequence). */
    int8_t   *kv_q;    float *kv_scale;  uint16_t *kv_calbuf;  int kv_caln, kv_frozen;    /* window int8 */
    uint8_t  *cmp_q4;  int8_t *cmp_q;    float *cmp_scale, *cmp_iscale, *cmp_absmax;
    uint16_t *cmp_calbuf; int cmp_caln, cmp_frozen;                                       /* compressed int8/int4 */
    uint8_t  *idx_kv8_4; int8_t *idx_kv8; float *idx_pscale;                              /* indexer int8/int4 */
    int      *sel_cache; int sel_cache_n, sel_cache_pos;   /* DS4F_IDX_REUSE: per-sequence cached selection */
} ds4f_lseq;

typedef struct ds4f_pool ds4f_pool;
typedef struct ds4f_mem_pool ds4f_mem_pool;

/* Runtime configuration.  The command-line/JSON path fills this structure;
 * the legacy ds4f_load_real()/ds4f_alloc_synth() wrappers below may still
 * populate it from the environment for old experiments.  Environment values
 * are therefore an explicit compatibility/debug path, not the production
 * configuration interface. */
typedef struct ds4f_runtime_options {
    ds4f_config cfg;
    char stage_dir[1024];
    char tokenizer[1024];
    char tokenizer_py[1024];
    int ep_rank, ep_size;
    int n_threads, n_cmgs;
    int dense_bf16, bf16_pv, dense_mxfp4, q8_dense, mxfp4_w4a8;
    int fp8_magic, mxfp4_gemm_tile;
    int sparse, mhc, exact, tierb2;
    int int8_kv, int8_cmp, int4_cmp, cp;
    int tp_head, tp_shared, tp_shared_full, tp_attn, tp_oproj, tp_embed, tp_wob;
    int mtp, expert_resident;
    int int8kv_cal, int8cmp_cal;
    int zero_copy_experts, load_drop_blob;
    int use_hip, hip_device, hip_async, hip_verbose;
    int hip_shared_bf16, hip_shared_bf16_layers;
    int hip_shared_fp16, hip_shared_fp16_layers;
    int hip_ordered_wkv_layers;          /* CPU-compatible FP8 WKV GEMM for prefill prefix */
    int hip_ordered_fp8_layers;          /* CPU-compatible reduction for all FP8 GEMMs */
    int hip_mxfp4_widen_layers;          /* GPU MXFP4 experts, widened to row-scale FP8 */
    int hip_mxfp4_resident_layers;       /* keep this many widened expert layers on GPU */
    int hip_mxfp4_resident_auto;         /* derive resident prefix from free VRAM */
    int hip_vram_reserve_mb;             /* safety margin for streamed/work buffers */
    int hip_mxfp4_stream_raw;            /* use compact raw MXFP4/LUT for streamed experts */
    int hip_expert_cache_mb;              /* 0=off, -1=auto, >0 raw MXFP4 expert-cache budget */
    int hip_expert_cache_stats;           /* report prompt-hot cache coverage and residency */
    int hip_prefill_attn;                /* experimental opt-in GPU sliding-window attention; exact default is 0 */
    int hip_exact_prefill;              /* keep M>1 prompt GEMMs on CPU reference path */
    int debug_env;
} ds4f_runtime_options;

/* Optional S3 dense-device hook.  The common forward path remains CPU-owned;
 * a backend can attach a persistent FP8 bank and claim only tensors it has
 * explicitly uploaded.  Returning zero means dst was produced; nonzero asks
 * the caller to treat the backend invocation as a hard integration failure. */
typedef int (*ds4f_gpu_dense_matvec_fn)(void *ctx, float *dst,
                                        const ds4f_tensor *t, const float *x);
typedef int (*ds4f_gpu_dense_async_multi_fn)(
    void *ctx, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, int n);
typedef int (*ds4f_gpu_dense_wait_fn)(void *ctx);
/* Fused shared expert: w1/w3 -> SwiGLU -> w2 with both [M, inter]
 * intermediates kept on the device, so only x goes up and only [M, hidden]
 * comes back.  Nonzero means the backend declined and the caller should run
 * the unfused GEMM/SwiGLU/GEMM sequence. */
typedef int (*ds4f_gpu_shared_ffn_fn)(
    void *ctx, float *dst, const ds4f_tensor *w1, const ds4f_tensor *w3,
    const ds4f_tensor *w2, const float *x, int M, int inter, int C, float lim);
typedef int (*ds4f_gpu_dense_blockdiag_fn)(
    void *ctx, float *dst, const ds4f_tensor *t, const float *xbase,
    int gin, int glora, int goff);
/* Batched prefill GEMM: Y[M,Ystride] = W[rows,cols] * X[M,Xstride]^T,
 * with token-major host buffers. The backend may use a device GEMM and must
 * preserve the logical row strides supplied by the caller. */
typedef int (*ds4f_gpu_dense_gemm_fn)(
    void *ctx, float *dst, const ds4f_tensor *t, const float *x,
    int M, int Ystride, int Xstride);
typedef int (*ds4f_gpu_dense_gemm_multi_fn)(
    void *ctx, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ystride,
    const int *Xstride, int n);
typedef int (*ds4f_gpu_dense_layer_fn)(void *ctx, const ds4f_layer *layer);
typedef int (*ds4f_gpu_dense_layer_prefetch_fn)(void *ctx, const ds4f_layer *layer);
typedef int (*ds4f_gpu_prefill_attn_fn)(
    void *ctx, float *dst, const float *q, const uint16_t *kv,
    const float *sink, int M, int pos0, int n_heads, int head_dim,
    int kv_dim, int kv_slots, int window, float scale);

typedef struct {
    ds4f_config cfg;
    int ep_rank, ep_size;
    ds4f_mem_pool *mem;                       /* owns all model-side allocations */
    ds4f_layer *layers;
    /* DS4F_MTP: the multi-token-prediction module (config num_nextn_predict_layers=1, tensors mtp.0.*).
     * A full transformer Block (reuses ds4f_layer: attn + MoE) + the MTP fusion:
     *   x' = e_proj(enorm(embed(next_id))) + h_proj(hnorm(x));  block(x'); head -> logits.
     * Scaffolded (load + forward stub); the draft/verify spec-decode loop is the follow-on. */
    int       has_mtp;                         /* DS4F_MTP loaded */
    /* DS4F_DECODE_BATCH: when dec_batch_seq!=NULL, ds4f_forward_verify decodes dec_nseq INDEPENDENT
     * sequences (batch elem k at position dec_batch_pos[k], reading cache set dec_batch_seq[k*L+layer])
     * instead of K consecutive tokens of one sequence. Set 0 aliases the layers' own live buffers. */
    int        dec_nseq;
    int       *dec_batch_pos;                  /* [dec_nseq] per-sequence positions (NULL = consecutive) */
    ds4f_lseq *dec_batch_seq;                  /* [dec_nseq * n_layers] cache sets (NULL = single-stream) */
    ds4f_layer mtp;                            /* the MTP block (attn + MoE), like a main layer */
    uint16_t *mtp_enorm, *mtp_hnorm, *mtp_norm;/* BF16 [hidden] RMSNorm weights (embed/hidden/final) */
    ds4f_tensor mtp_e_proj, mtp_h_proj;        /* [hidden,hidden] dense fusion projections */
    float    *mtp_hc_fn, *mtp_hc_base, *mtp_hc_scale;  /* MTP head's HC params */
    uint16_t *embed;        /* BF16 [vocab, hidden] (unused in synth decode) */
    ds4f_tensor head;       /* BF16 [vocab, hidden] (TP: only this node's vocab-shard rows) */
    int head_r0;            /* DS4F_TP_HEAD: global vocab offset of this node's head shard
                             * (head.rows = shard row count). 0 + head.rows==vocab => replicated. */
    int sh_r0, sh_rows;     /* DS4F_TP_SHARED: shared_inter shard [sh_r0, sh_r0+sh_rows) for sh_w1/sh_w3. */
    int sh2_r0, sh2_rows;   /* DS4F_TP_SHARED_FULL: hidden-row shard for sh_w2; its partial is folded into EP. */
    int attn_h0, attn_h1;   /* DS4F_TP_ATTN: this node's owned attention heads [attn_h0, attn_h1).
                             * wq_b holds those heads' rows; q-norm/RoPE + the attn worker process only
                             * them; s_o is per-node partial -> reduced. 0/[0,n_heads) => replicated. */
    int oi0, oi_rows;       /* DS4F_TP_OPROJ: wo_a o_inter row-shard [oi0, oi0+oi_rows) (wo_b replicated).
                             * Needs FULL s_attn (reduced first under TP_ATTN). 0/==o_inter => replicated. */
    int emb_r0, emb_rows;   /* DS4F_TP_EMBED: embed vocab-shard [emb_r0, emb_r0+emb_rows). The token's row
                             * lives on one node; embed_lookup zero-fills + ar_cb-SUMs -> full (bit-exact). */
    uint16_t *out_norm;     /* BF16 [hidden] */
    /* global mHC head (collapses the 4 streams 1x before lm_head; NO sinkhorn) */
    float *hc_head_fn;      /* [hc_mult=4, hc_mult*hidden=16384] */
    float *hc_head_base;    /* [hc_mult=4] */
    float *hc_head_scale;   /* [1] */
    uint32_t fp8_lut[256];
    /* dense (MLA + shared-expert) quant type: DS4F_FP8 (on-demand dequant, ~20GB)
     * or DS4F_BF16 (predequant, +5.7GB -> ~26GB, routes the dominant matvecs
     * through the BW-bound bf16 kernel ~400 GB/s instead of the gather-bound
     * fp8 kernel ~70 GB/s). Set via DS4F_FP8_BF16=1. Experts stay MXFP4. */
    ds4f_qtype dense_qt;
    void *gpu_dense_ctx;
    ds4f_gpu_dense_matvec_fn gpu_dense_matvec;
    ds4f_gpu_dense_async_multi_fn gpu_dense_async_multi;
    ds4f_gpu_dense_wait_fn gpu_dense_wait;
    ds4f_gpu_shared_ffn_fn gpu_shared_ffn;
    ds4f_gpu_dense_blockdiag_fn gpu_dense_blockdiag;
    ds4f_gpu_dense_gemm_fn gpu_dense_gemm;
    ds4f_gpu_dense_gemm_multi_fn gpu_dense_gemm_multi;
    ds4f_gpu_dense_layer_prefetch_fn gpu_dense_layer_prefetch;
    ds4f_gpu_dense_layer_fn gpu_dense_layer_begin;
    ds4f_gpu_prefill_attn_fn gpu_prefill_attn;
    int gpu_dense_stream_prefill_only;
    int gpu_dense_mixed;       /* opt-in mixed GPU/CPU independent-GEMM dispatch */
    int gpu_exact_prefill;     /* skip GPU M>1 GEMMs; retain GPU M=1 decode */
    /* FP8 dense decode kernel: 0 = gather (LUT, bit-exact), 1 = magic-multiply
     * (FTZ, ~6 ops/lane, no gather; +2..18% in the HBM-stream decode regime,
     * subnormals flush to 0 -> values ~5e-5 off, fine for the harness). The
     * magic path also enables FTZ on every pool worker. Set via DS4F_FP8_MAGIC=1. */
    int fp8_magic;
    /* x86 MXFP4 expert activation mode.  Allocators initialize this from
     * DS4F_MXFP4_W4A8; keeping it per model lets a correctness harness compare
     * exact-f32 and W4A8 models in one process. */
    int mxfp4_w4a8;
    /* Zero-copy routed experts: the MXFP4 expert tensors point directly at the
     * mmap'd safetensors shards instead of at repacked arena copies, so their
     * ~147 GB stay clean, evictable, file-backed page cache. Requires kernels
     * that consume the ON-DISK nibble order and undo the x0.5 scale in-kernel
     * (see matvec_mxfp4_1row_*_raw). Set by ds4f_load_real from a
     * DS4F_STAGE_NOCOPY manifest; 0 keeps the repack-into-arena behaviour. */
    int mxfp4_raw;
    void  *blob_map;      /* kept mapped for the model's lifetime when mxfp4_raw */
    size_t blob_map_sz;
    /* MXFP4 GEMM (M>1 expert/dense): 0 = svtbl per-token-pair (matvec_mxfp4_8row_2x,
     * default; the M=1 decode + small-M path -- ~84 Gmac/s, dequant re-run per pair);
     * >0 = M threshold above which the GEMM tile-dequants each 8-row group's nibbles
     * ONCE into a bf16 L1 tile (svtbl->bf16, lossless: fp4*pow2 fits bf16) and reuses
     * it across all M tokens via the 8x3 bf16-pv kernel -> compute-bound, scales to
     * ~140 Gmac/s at M>=32 where the flat-84 svtbl re-dequant is the wall. Set via
     * DS4F_MXFP4_GEMM_TILE (e.g. 16). Crossover with svtbl is ~M=8-16. */
    int mxfp4_gemm_tile;
    /* BF16 matvec layout: 0 = row-major (matvec_bf16_8row), 1 = pair-interleaved
     * (matvec_bf16_8row_pv, p_odd predicated-load, +22..28% over plain bf16 and
     * BYTE-IDENTICAL — same column order + 8-row svaddv reduction). Applies to
     * ALL DS4F_BF16 matvecs (predequant dense + router + lm-head). Fill writes
     * the interleaved layout to match. Set via DS4F_BF16_PV=1; pairs naturally
     * with DS4F_FP8_BF16=1 to route the dominant dense matvecs through pv. */
    int bf16_pv;
    ds4f_qtype bf16_mv_qt;  /* DS4F_BF16 or DS4F_BF16_PV; for router gate + lm-head */
    /* int8 W8A8 dense (DS4F_Q8_DENSE): 0 = dense stays bf16-pv (default); 1 = after
     * load/fill the dominant dense weights (wq_a/wq_b/wkv/wo_a/wo_b/shared w1/w2/w3)
     * are repacked bf16-pv -> DS4F_Q8_PV (int8 group layout) so the prefill GEMM
     * runs svdot (4x int8 MACs/FMLA). Router gate + lm-head stay bf16-pv to protect
     * argmax. Requires DS4F_FP8_BF16=1 (the bf16-pv source). Quality is argmax-exact
     * (int8 rel-L2 ~1%), NOT bit-similar -> the rel<1e-3 gate is relaxed for this
     * path. The bf16 source pages are reclaimed (q8 is leaner), so memory shrinks. */
    int q8_dense;
    /* sparse lightning-indexer attention (Stage 4): 0 = dense full attention all
     * layers (default); 1 = on sparse layers (compress_ratios[L]!=0) with nP>topk,
     * cheap compressed index selects topk positions, weighted-V over them only.
     * Synthetic stand-in (logits meaningless); the point is the O(topk) long-ctx
     * perf model. Set via DS4F_SPARSE=1. */
    int sparse;
    /* exact mHC (Manifold-Constrained Hyper-Connections): 0 = plain residual
     * stand-in (default, byte-identical to the Stage 1-4a path); 1 = carry the
     * hidden state as hc_mult=4 streams and run the EXACT hc_pre/hc_post/hc_head
     * + sinkhorn math (so a real-weight loader yields meaningful logits). The
     * mixes Linear (hc_*_fn) is the only added matvec; collapse/expand are cheap
     * scalar folds. Set via DS4F_MHC=1. */
    int mhc;
    /* exact DeepSeek-V4-Flash forward math (DS4F_EXACT): 0 = the synthetic
     * stand-ins (silu o-proj, no RoPE/q-norm, bias-free gate; default, byte-
     * identical to Stage 1-4b); 1 = the real math (per-head q-norm + RoPE/YaRN on
     * q/kv, sliding-window(128)+sink attention, de-rotate + grouped low-rank
     * o-proj, gate bias-select/unbiased-weight, swiglu clamp). With real weights
     * this yields meaningful logits. Tier-B1 (this) is bit-faithful for
     * dense+HCA layers and the sliding-window term of CSA layers. KNOWN
     * omissions, all deferred to Tier-B2 (need extra plumbing, not just math):
     *   1. stateful compressor/indexer long-range compressed-KV term on sparse
     *      layers -> omits CSA compressed tokens at pos>=4 (needs Hadamard +
     *      FP4 act-quant + raw-hidden history);
     *   2. hash-layer routing (layers < n_hash_layers route by tid2eid[token_id])
     *      -> falls back to score-based topk here (the harness threads a hidden
     *      state, not a token id, per position);
     *   3. act_quant FP8-sim of the kv non-rope dims (a QAT precision detail).
     * RoPE/YaRN, per-head q-norm, sliding-window+sink attention, grouped low-rank
     * o-proj, sqrtsoftplus gate (bias-select/unbiased-weight), and swiglu clamp
     * are exact (validated vs pure-Python ref to 5e-8: ds4f_exact_test.c). */
    int exact;
    /* Tier-B2 stateful sparse attention (DS4F_TIERB2): 0 = Tier-B1 (default, the
     * sliding-window-only term; byte-identical). 1 = ALSO run the per-layer
     * compressor/indexer and fold the long-range compressed-KV term into the same
     * softmax (the omission #1 above). Implies exact (reuses q-norm/RoPE/window).
     * Off OR on a dense layer (ratio==0) == exact path, byte-identical. The decode
     * kernels are bit-validated (ds4f_tierb2_test.c); here they are wired stateful
     * token-at-a-time. Set via DS4F_TIERB2=1. */
    int tierb2;
    /* int8 KV cache (DS4F_INT8_KV): store the per-(layer,pos) kv latent as int8 with
     * a static per-channel scale (S5) instead of bf16 -> half the KV footprint at long
     * ctx. See ds4f_layer.kv_q. Requires exact (streaming decode path); incompatible
     * with batched prefill (which is itself unused under tierb2). LOSSY (coherence
     * gate, not bit-exact). Off by default. */
    int int8_kv;
    /* int8 compressed-latent cache (DS4F_INT8_CMP): store the per-(layer,slot) compressed
     * cmp_kv latent as int8 with a static per-channel scale (S5), like int8_kv. cmp_kv is
     * the dominant tierb2 physical (THP) footprint at high ctx; int8 reclaims ~3/4 and
     * lifts the ctx ceiling toward 256k. See ds4f_layer.cmp_q. Requires exact/tierb2
     * streaming. LOSSY (coherence gate). Off by default. */
    int int8_cmp;
    /* DS4F_INT4_CMP: refinement of int8_cmp -- stores cmp_kv as int4 (+/-7, 2/byte) instead of
     * int8, halving the dominant ctx-cache (1384 vs 2768 B/pos) to push ctx past ~2.75M. Implies
     * int8_cmp (reuses its calbuf/scales/exact-path/dispatch); selects cmp_q4 + the i4 codec.
     * 16 levels of an already-compressed latent -> coherence gate (not token-identical). Off by default. */
    int int4_cmp;
    /* RoPE freqs tables (only built when exact): cos/sin[pos*half + k],
     * half = qk_rope_dim/2. Two configs: dense (theta, no YaRN) + comp (YaRN). */
    float *rope_dense_cos, *rope_dense_sin;
    float *rope_comp_cos,  *rope_comp_sin;
    /* arena */
    uint8_t *arena; size_t arena_sz, arena_used;
    /* pool */
    ds4f_pool *pool;
    int n_threads, n_cmgs;
    /* scratch (per-forward, single token) */
    float *s_hn, *s_q, *s_qlat, *s_kvlat, *s_attn, *s_oin, *s_o1, *s_o;
    float *s_h2, *s_router, *s_shg, *s_shu, *s_exg, *s_exu, *s_moe, *s_logits;
    /* S2 routed-expert decode scratch: active experts are evaluated in two
     * dispatches (all gate/up matvecs, then all down matvecs).  The legacy
     * serial path continues to use s_exg/s_exu/s_o. */
    float *s_exb_g, *s_exb_u, *s_exb_o;
    float *s_route;         /* routed-expert partial (owned-only); EP-summed via ar_cb */
    float *s_attn_sc;       /* DS4F_ATTN_GEMM: [n_heads*(window+index_topk)] scores->softmax weights (lazy) */
    float *s_attn_m;             /* DS4F_CP_COMBINE: per-head local max (lazy, [n_heads]) */
    float *s_attn_comb;          /* DS4F_CP_COMBINE: packed [acc: n_heads*q_head_dim | l: n_heads] reduced in
                                  * ONE ar_cb (min collective count on this latency-bound fabric) (lazy) */
    float *p_attn_comb, *p_attn_m;  /* DS4F_CP_COMBINE verify/prefill: the K positions' partials collected so the
                                  * cross-node combine is ONE reduce for the whole chunk (m_tile-sized, lazy) */
    float *s_idx_qpre;      /* batched-prefill: pre-projected indexer q for the current pos (NULL=compute in index_step) */
    float *v_idxq;          /* [m_tile*index_n_heads*index_head_dim] batched qproj output (lazy, verify prefill) */
    /* batched (M>1) prefill scratch (only allocated by ds4f_alloc_prefill_batch;
     * NULL unless DS4F_PREFILL_BATCH is wired). Token-major [m_tile, width].
     * p_x is the carried hidden state for all M tokens. */
    int m_tile;
    float *p_x, *p_hn, *p_qlat, *p_q, *p_kvlat, *p_attn, *p_o1, *p_o;
    float *p_h2, *p_shg, *p_shu, *p_moe, *p_route, *p_router, *p_logits;
    /* [m_tile, vocab] full per-row logits reconstructed for batched SAMPLING under TP_HEAD (zero-fill the
     * owned shard + ar_cb SUM). When the head is replicated it aliases p_logits. Lazily allocated. */
    float *p_logits_full;
    /* expert-grouping prefill scratch: per owned slot a bucket of routed tokens
     * (ex_tok[slot*m_tile+p]=token idx, ex_wt=its routed weight). The p_ex*
     * slabs hold all local assignments contiguously so independent experts can
     * share one gate/up, activation, and down dispatch. */
    int *ex_cnt, *ex_tok; float *ex_wt; float *p_exX, *p_exG, *p_exU, *p_exO; int ex_no;
    /* mHC 4-stream state (only used when m->mhc): x4/resid = [hc_mult*hidden]
     * stream buffers, xc = [hidden] collapsed hc_pre/hc_head output. */
    float *s_x4, *s_resid, *s_xc;
    /* sparse-indexer per-thread scratch: block scores [n_threads * idx_blk_stride]
     * and selected positions [n_threads * index_topk]. Only used when m->sparse. */
    float *s_idx_scores; int *s_idx_sel; int idx_blk_stride;
    /* Tier-B2 per-token scratch (off-arena, only when m->tierb2): compressor output
     * [kv_lora], indexer q [index_n_heads*index_head_dim] + score [max_pos/4],
     * selected LOCAL compressed indices [max_pos/4] + their count for this layer. */
    float *s_cmp_out, *s_idx_q, *s_idx_score; int *s_tb2_sel; int s_tb2_nsel;
    /* EP combine hook: if set, called once per layer on the routed-expert partial
     * [hidden] to sum it across the expert-parallel group (Stage 2 = tp_allreduce_sum).
     * Shared expert stays replicated (added locally). NULL => single-node (all owned). */
    void  (*ar_cb)(float *buf, int count, void *ctx);
    void   *ar_ctx;
    void  (*ar_max_cb)(float *buf, int count, void *ctx);   /* MAX all-reduce (Phase-2 CP combine) */
    void   *ar_max_ctx;
    void  (*ar_argmax_cb)(float *val, int32_t *idx, void *ctx);  /* (val,global-idx) argmax all-reduce
                                                                  * (TP_HEAD batched-prefill head merge) */
    void   *ar_argmax_ctx;
    int     want_full_logits;  /* set by the runner when a caller needs the full-vocab logits vector
                                 * (e.g. temperature/top_p/top_k sampling) rather than just the argmax
                                 * token. Under TP_HEAD, ds4f_forward_token uses this to pick between the
                                 * cheap local-argmax + tiny argmax-reduce path (greedy, want=0) and the
                                 * full zero-fill + [vocab] all-reduce-SUM path (sampling, want=1). */
    int     cp;             /* DS4F_CP: context parallelism — compressed caches sharded by slot,
                             * selected latents gathered (ar_cb-SUM) so attention reads a full set. */
    float  *s_cmp_gather;   /* [index_topk * kv_lora] gathered selected cmp latents (f32) under CP */
    int     cp_gather;      /* set per CSA layer in forward_token: attention reads s_cmp_gather (f32) */
    float  *s_cp_cand_slot; /* [ep_size*index_topk] CP idx-merge: gathered candidate slots (as float) */
    float  *s_cp_cand_score;/* [ep_size*index_topk] CP idx-merge: gathered candidate scores */
    float  *v_x4, *v_resid; /* [K*hc_mult*hidden] M2b batched verify: the K positions' mHC states + residual */
    /* perf accounting (weight HBM bytes touched, reset per token by the runner) */
    size_t bytes_read;
    /* Optional routing telemetry, lazily allocated when DS4F_ROUTE_TELEMETRY=1.
     * route_hits is [n_layers,n_experts]; route_tokens counts routed tokens per
     * layer.  Keeping it model-local makes server and benchmark reports agree. */
    uint64_t *route_hits, *route_tokens;
    int route_telemetry;
    /* per-phase wall-time profiler (seconds, accumulated; printed by runner) */
#define DS4F_NPHASE 24
    double prof[DS4F_NPHASE];
} ds4f_model;

/* phase ids for ds4f_model.prof[]. TB2SCAN..TB2LCMP are SUB-timers of TB2PREP (they
 * partition the index_step/compressor work inside it); they overlap TB2PREP so the
 * percentage column double-counts them -- read their absolute ms, not their %.
 *   tb2scan  = index_score scan (ranks idx_kv) -- the O(ctx/ratio) part
 *   tb2qproj = idx wq_b q-projection [H*hd, qlora]   (O(1)-in-ctx weight matvec, x21 CSA)
 *   tb2rope  = per-head RoPE + rotate + fp4 on q     (O(1))
 *   tb2icmp  = idx compressor (ds4f_compress_step inside index_step)  (O(1))
 *   tb2wproj = weights_proj [H, hidden]              (O(1))
 *   tb2lcmp  = layer compressor (the top-of-tb2_prepare cmp_kv write) (O(1))
 *   tb2topk  = index_topk top-k selection (O(k*T) naive scan!) -- the real ctx-scaling cost
 * tb2prep - (sum of these) = glue. These isolate where the index decode time actually goes. */
enum { DS4F_P_QKV=0, DS4F_P_ATTN=1, DS4F_P_OPROJ=2, DS4F_P_SHARED=3,
       DS4F_P_ROUTER=4, DS4F_P_EXPERTS=5, DS4F_P_HEAD=6, DS4F_P_OTHER=7,
       DS4F_P_TB2PREP=8, DS4F_P_TB2SCAN=9,
       DS4F_P_TB2QPROJ=10, DS4F_P_TB2ROPE=11, DS4F_P_TB2ICMP=12,
       DS4F_P_TB2WPROJ=13, DS4F_P_TB2LCMP=14, DS4F_P_TB2TOPK=15,
       /* QKV_A..QKV_ROPE are SUB-timers of QKV (like TB2SCAN.. are of TB2PREP) */
       DS4F_P_QKV_A=16, DS4F_P_QKV_B=17, DS4F_P_QKV_KV=18, DS4F_P_QKV_ROPE=19,
       /* mHC + comm decode sub-timers (mhc* are inside "other"/oproj/experts; comm = ar_cb) */
       DS4F_P_MHCPRE=20, DS4F_P_MHCPOST=21, DS4F_P_MHCCPY=22, DS4F_P_COMM=23 };
static const char *ds4f_prof_names[24] = {
    "qkv_proj","attn","o_proj","shared","router","experts","head","other","tb2prep","tb2scan",
    "tb2qproj","tb2rope","tb2icmp","tb2wproj","tb2lcmp","tb2topk",
    "qkv_wqa","qkv_wqb","qkv_wkv","qkv_rope",
    "mhc_pre","mhc_post","mhc_cpy","comm" };

/* ===================== thread pool (pinned, spin) ===================== */
typedef void (*ds4f_fn)(void *arg, int tid, int nthr);

/* ---- implementation (relocated to keep this header small) ---- */
#include "ds4f_impl.h"

#endif /* DS4F_H */
