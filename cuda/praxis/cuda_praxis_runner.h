/*
 * cuda_praxis_runner.h - standalone CUDA (NVRTC) runner for the praxis-1 policy:
 * chunk-causal two-tower MoE backbone, action expert and the window_joint Euler
 * sampler. praxis/{backbone,layers,action_expert,model,sampler}.py are the oracle.
 *
 *   #define CUDA_PRAXIS_IMPLEMENTATION   in exactly one .c file
 *   #include "cuda_praxis_runner.h"
 *
 * Config driven: every dim comes from <dir>/config.json (praxis checkpoint
 * format), weights from <dir>/model.safetensors (bf16/f16/f32, kept in their
 * dtype on the device and widened in the GEMM kernel; small vectors -> f32).
 * Activations are fp32. Attention is dense-masked (token-code mask for the
 * teacher-forcing layout, byte masks for the action expert); the structure
 * (codes, per-token rope tables) is what a block-sparse kernel needs later.
 *
 * Routed expert banks live either on the device (resident) or in pinned host
 * memory and are streamed per layer through a ring of R device slots on a copy
 * stream (PRAXIS_STREAM / praxis_set_stream); see praxis_stats().
 */
#ifndef CUDA_PRAXIS_RUNNER_H
#define CUDA_PRAXIS_RUNNER_H
#include <stddef.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct praxis_runner praxis_runner;

typedef struct {
    size_t dev_bytes_now, dev_bytes_peak;      /* all tracked device allocations */
    size_t weights_bytes;                      /* resident non-routed weights */
    size_t routed_dev_bytes;                   /* device bytes holding routed expert banks */
    size_t routed_total_bytes;                 /* routed bytes of the whole model */
    size_t ring_slot_bytes; int ring_slots;    /* 0 when resident */
    size_t h2d_bytes; long n_loads, n_prefetch_hits, n_moe_layers_run;
    size_t ws_bytes;                           /* activation workspace */
    size_t free_min_bytes_used;                /* peak (cuMemGetInfo) device use since load */
    size_t host_rss_peak_load_kb;              /* max VmRSS sampled during NVFP4 conversion */
    int pack_cache_hits, pack_cache_written, n_shards, n_moe_layers_total;
} praxis_stats;

praxis_runner *praxis_load(const char *dir, int device, int verbose);
/* Synthetic model: only <dir>/config.json is read; resident weights are random bf16 device buffers and the routed
 * banks are random pinned-host bf16 banks (nbanks distinct banks, default 4, shared cyclically by the MoE layers;
 * every layer still streams a full-size bank). */
praxis_runner *praxis_load_synthetic(const char *dir, int device, int verbose, int nbanks);
void praxis_set_fast(praxis_runner *r, int fast);
/* Routed expert banks as NVFP4 (E2M1 + e4m3 block scale per 16 + fp32 global per matrix, praxis nvfp4.py recipe):
 * PRAXIS_NVFP4=1 in the environment at load time converts the bf16 banks of the checkpoint (CPU, OpenMP) or, for the
 * synthetic model, generates random packed banks. Requires sm_120a. Call praxis_set_stream afterwards.
 * PRAXIS_EXPERT_FMT (also enables the quantised banks): nvfp4 = W4A4 (same as PRAXIS_NVFP4=1), nvfp4a16 = NVFP4 weights with f16 activations (W4A16),
 * mixed = FP8 down banks + FP8 gate_up of the sensitive layers (encoder blocks 3/10/13), f16 activations, expert-group streaming (best accuracy < 8 GiB).
 * Finer control: PRAXIS_FP8_BANKS (per-matrix FP8, "dn,enc3.gu,-enc13.dn"), PRAXIS_SLOT_MB (ring slot cap -> expert groups), PRAXIS_PACK_SCALE=mse,
 * PRAXIS_PACK_CACHE (non-default plans use <file>.p<plan hash>). See cuda/praxis/README.md. */
int praxis_nvfp4_enabled(const praxis_runner *r);
int praxis_encoder_pair(praxis_runner *r, const float *text, const float *text_u, int S, const float *anchor, int h, int w,
                        const float *noised, const float *clean, const float *t, int K);
/* Encoder context cache across the Euler steps of a sampler run (fast path only; default from PRAXIS_CTX_CACHE, off if unset). */
void praxis_set_ctx_cache(praxis_runner *r, int on);
void praxis_select_state(praxis_runner *r, int i);   /* 0: cond (default), 1: uncond state of the last pair */
void praxis_prof_report(praxis_runner *r, FILE *f);
void praxis_prof_reset(praxis_runner *r);
void praxis_free(praxis_runner *r);
/* ring_slots == 0: routed banks resident on device; >= 1: streamed through that many slots. */
int praxis_set_stream(praxis_runner *r, int ring_slots);
void praxis_set_dump_dir(praxis_runner *r, const char *dir);  /* per-block taps as .npy */
void praxis_set_sample_dump_dir(praxis_runner *r, const char *dir);  /* per-step sampler tensors */
void praxis_stats_get(const praxis_runner *r, praxis_stats *s);

int praxis_n_chunks(const praxis_runner *r);
int praxis_lat_per_chunk(const praxis_runner *r);
int praxis_action_dim(const praxis_runner *r);
int praxis_proprio_dim(const praxis_runner *r);
int praxis_text_dim(const praxis_runner *r);
int praxis_video_c(const praxis_runner *r);

/* Backbone: text [S][text_c]; anchor [C][1][h][w]; noised [C][K*F][h][w];
 * clean [C][(K-1)*F][h][w] (NULL if K == 1); t [K]. Leaves the taps on the device. */
int praxis_encoder(praxis_runner *r, const float *text, int S, const float *anchor, int h, int w,
                   const float *noised, const float *clean, const float *t, int K);
/* After praxis_encoder: velocity [C][K*F][h][w] for the same noised latents. */
int praxis_decoder(praxis_runner *r, const float *noised, const float *t, float *vel_out);
/* After praxis_encoder: action velocity [K*F][action_dim]. proprio [K][proprio_dim]. */
int praxis_expert(praxis_runner *r, const float *noised_a, const float *t, const float *proprio, float *out);

/* window_joint with the sampler shortcuts of praxis: pair_cfg runs the two guidance encoder passes in lockstep (each MoE layer reads
 * its routed experts once; fast path only), skip_last_video makes the last step action-only (no decoder; the unguided pass only if
 * cfg_action != 1). With skip_last_video video_out is zeroed (the actions keep their bits). */
int praxis_sample_window_joint_ex(praxis_runner *r, const float *text, const float *text_u, int S,
                                  const float *anchor, int h, int w, const float *clean,
                                  const float *proprio, const float *noise_v, const float *noise_a,
                                  int K, int num_steps, double t_shift, double cfg_video, double cfg_action,
                                  float *video_out, float *action_out, int pair_cfg, int skip_last_video);

/* Same with its own text length for the unguided (blank caption) branch: praxis encodes the cond and the blank caption separately, so S_u != S
 * (S_u == 1 for the blank caption " ") is the normal case; pair_cfg needs S_u == S and is otherwise ignored. */
int praxis_sample_window_joint_ex2(praxis_runner *r, const float *text, int S, const float *text_u, int S_u,
                                   const float *anchor, int h, int w, const float *clean,
                                   const float *proprio, const float *noise_v, const float *noise_a,
                                   int K, int num_steps, double t_shift, double cfg_video, double cfg_action,
                                   float *video_out, float *action_out, int pair_cfg, int skip_last_video);

/* The window_joint sampler. noise_v [C][K*F][h][w], noise_a [K*F][A] (identical to the
 * oracle's seeded noise), clean as above. Writes the last chunk: video_out
 * [C][F][h][w], action_out [F][A]. */
int praxis_sample_window_joint(praxis_runner *r, const float *text, const float *text_u, int S,
                               const float *anchor, int h, int w, const float *clean,
                               const float *proprio, const float *noise_v, const float *noise_a,
                               int K, int num_steps, double t_shift, double cfg_video, double cfg_action,
                               float *video_out, float *action_out);

#ifdef __cplusplus
}
#endif
#endif /* CUDA_PRAXIS_RUNNER_H */

/* ====================================================================== */
#ifdef CUDA_PRAXIS_IMPLEMENTATION
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <time.h>
#include <omp.h>
#include "../cuew.h"
#include "../cublasew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../../common/safetensors.h"
#include "praxis_kernels.h"
#include "praxis_fp4_kernels.h"

#define PR_MAXD 32
#define PR_MAXTAPS 32
#define PR_MAXSH 64
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/* ---- config ---- */
typedef struct {
    int dim, time_dim, n_reg, dense_hidden, head_dim, n_kv_heads, n_heads;
    int rope_eff; double rope_base; long long rope_tab;
    int n_experts, top_k, moe_hidden; float routed_scale;
    int enc_depth, enc_dense, enc_ntap, enc_taps[PR_MAXTAPS], text_cin, enc_vcin;
    int dec_depth, dec_dense, dec_mod_dim, dec_ph, dec_pw, dec_vcin;
    int row_stride, frame_stride;
    int ae_dim, ae_layers, ae_heads, ae_hd, trunk_dim, act_dim, prop_dim, nfut, ae_max_chunks;
    int tap_norm, clean_only, ae_rope_dim, ae_rope_eff; double ae_rope_base; long long ae_rope_tab;
    int gorder[3];                 /* 0 register, 1 anchor, 2 text */
    int n_chunks, lat;
} pr_cfg;

typedef struct { CUdeviceptr p; int dt; size_t bytes; } pr_mat;   /* matrix in checkpoint dtype */

typedef struct {
    pr_mat q, k, v, gate, out;
    CUdeviceptr out_b, ln_in, ln_out, qn, kn;
    float s;
    int moe;
    pr_mat a, b, down;                      /* dense SwiGLU */
    pr_mat router, sh_gu, sh_dn;            /* MoE */
    CUdeviceptr ebias;
    int moe_id;                             /* global streaming sequence id */
    void *h_gu, *h_dn;                      /* pinned host banks (stream mode) */
    pr_mat gu, dn;                          /* device banks (resident mode) */
    size_t gu_bytes, dn_bytes; int bank_dt;
    int ngrp; size_t grp_bytes;             /* bank_dt 3: expert groups streamed one after the other through a ring slot (1 = whole layer); bytes of one group */
    int fgu, fdn;                           /* bank_dt 3: storage format of the gate_up / down bank (0 NVFP4, 1 FP8 e4m3 + per-row scale) */
    CUdeviceptr mod_w, mod_b;               /* decoder modulator [2C][mod_dim] (dt) */
    pr_mat mod;
} pr_blk;

typedef struct {
    int depth, dense, n_moe;
    pr_blk b[PR_MAXD];
    CUdeviceptr registers, t_freq, t_phase, tp_b;
    pr_mat tp_w;
    pr_mat vin_w; CUdeviceptr vin_b;                 /* video input projection */
    pr_mat tx0_w, tx2_w; CUdeviceptr tx0_b, tx2_b, tx_norm;   /* encoder text MLP */
    pr_mat out_w; CUdeviceptr out_b, out_norm;       /* encoder: out proj; decoder: norm + out proj */
} pr_tower;

typedef struct {
    pr_mat to_q, to_kv, to_out, mlp_in, mlp_out, ada_a, ada_m;
    CUdeviceptr ada_a_b, ada_m_b, mlp_in_b, mlp_out_b;
} pr_ae_blk;

typedef struct {
    pr_mat w1, w2, w3, tau0, tau2, temb0, temb2, embp, tap_proj, text_proj, head;
    CUdeviceptr w1_b, w2_b, w3_b, tau0_b, tau2_b, temb0_b, temb2_b, embp_b, head_b;
    CUdeviceptr future, proprio_tok, chunk_embed, tap_scale, freq256;
    pr_ae_blk b[PR_MAXD];
} pr_ae;

typedef struct { CUdeviceptr p; size_t bytes; long long seq, layer; int grp; CUevent ready, done; int done_valid; } pr_slot;

/* encoder context cache (praxis ctx_cache): keys/values (post q/k-norm + rope, bf16) of the registers/anchor/text and clean tokens of
 * every layer, rows [0,Ng) = globals then [Ng,Ng+Nc) = clean. They are the same in every Euler step of a sampler call. */
typedef struct { int valid, Nctx, Ng, Nc, Nn, K, S; CUdeviceptr k[PR_MAXD], v[PR_MAXD]; size_t cap[PR_MAXD]; } pr_kvc;

typedef struct {   /* state left by praxis_encoder */
    int K, S, h, w, F, tpf, n_reg, n_anc, T, Nn, Nc, Ng;
    long long *pos;
    CUdeviceptr emb, taps[PR_MAXTAPS], cond;
    size_t emb_cap, taps_cap[PR_MAXTAPS], cond_cap;
} pr_enc_state;

enum { WS_X, WS_N1, WS_Q, WS_K, WS_V, WS_G, WS_AO, WS_PROJ, WS_N2, WS_FA, WS_FB, WS_FH, WS_FO, WS_LOG,
       WS_GU, WS_HH, WS_Y, WS_IN, WS_TOK, WS_TMP, WS_MOD, WS_MOD2, WS_COSQ, WS_SINQ, WS_COSK, WS_SINK,
       WS_CODEQ, WS_CODEK, WS_MROW, WS_TV, WS_TOPI, WS_TOPG, WS_TOKID, WS_TOKROW, WS_SEG, WS_MASK,
       WS_AX, WS_ACOND, WS_AKV, WS_AH, WS_AT,
       WS_XB, WS_NB1, WS_NB2, WS_QB, WS_KB, WS_VB, WS_AOB, WS_HB1, WS_HB2, WS_HB3, WS_XG, WS_GUB, WS_HHB, WS_YB,
       WS_KW, WS_VW, WS_KCODE, WS_CNT, WS_SEGD, WS_TILE, WS_FP4A, WS_FP4B, WS_FP4C, WS_FP4D, WS_TPRE, WS_AMAX, WS_N };

struct praxis_runner {
    CUcontext ctx; CUdevice dev; CUmodule mod;
    cublasew_context *cb;
    CUmodule mod4; CUfunction f4_amax_f32, f4_amax_bf16, f4_quant_f32, f4_quant_bf16, f4_tiles, f4_gemm, f4_w16, f8_w16, f4_cvt, f4_swi16, f4_b2h;
    int fp4_a16;                     /* NVFP4 banks with f16 activations (W4A16, PRAXIS_EXPERT_FMT=nvfp4a16 or mixed) */
    int fmt_mixed;                   /* PRAXIS_EXPERT_FMT=mixed: FP8 down banks + FP8 gate_up of the sensitive layers, f16 activations, expert-group streaming */
    int nvfp4;                       /* routed banks stored as NVFP4 (bank_dt == 3) */
    int fast, synth, moe_chunk, prof, synth_banks, skip_h2d, pinned_bf16, mmap_bf16;
    CUfunction f_cvt, f_rmsb, f_qkprep, f_bias, f_swib, f_ctiles, f_flash[4], f_flashiv[4], f_mcount, f_mscan, f_mfill, f_gather, f_combb, f_fillrand;
    double tcat[12]; double tlast;
    int *hseg;
    void *synth_pool[8]; int synth_npool; size_t synth_pool_bytes;
    CUfunction f_gemm, f_rms, f_adaln, f_ln, f_modulate, f_fourier, f_sincos, f_swiglu, f_act, f_add, f_addrow,
        f_future, f_copy, f_chw, f_unpatch, f_patch, f_rope, f_attn, f_topk, f_combine;
    pr_cfg c;
    st_context *st;                  /* shard 0 (kept for compat) */
    st_context *shard[PR_MAXSH]; int n_shards; char *shard_path[PR_MAXSH];
    int pack_cache_hits, pack_cache_written; size_t rss_peak_load;
    pr_tower enc, dec;
    pr_ae ae;
    cu_buf_slot ws[WS_N];
    CUdeviceptr zero_row;                 /* zero vector for decoder register modulation */
    pr_enc_state es_store[2]; pr_enc_state *esp;
    CUdeviceptr *allocs; int n_alloc, cap_alloc;   /* every pr_malloc'd device block, so praxis_free can release them all */
    int ae_flash;                    /* action expert uses the block-sparse flash kernel (fast path; PRAXIS_AE_DENSE=1 disables) */
    int ctx_cache;                   /* sampler uses the encoder context cache (fast path only) */
    pr_kvc kvc[2];                   /* per guidance branch (cond, uncond) */
    int kx_mode, kx_nk; pr_kvc *kx_seq[2]; CUdeviceptr kx_kcode; int kx_Ng, kx_Nn;   /* set around pr_stack: 0 off, 1 store, 2 use */
    double ctx_time_saved;
    /* streaming */
    int ring_n; pr_slot ring[4]; long long seq; CUstream copy_stream;
    int n_moe_total;
    size_t cur_bytes, peak_bytes, w_bytes, routed_dev, routed_total, ring_slot_bytes, h2d_bytes, ws_bytes;
    long n_loads, n_hits, n_moe_run;
    size_t base_free, min_free;
    char dump_dir[1024], smp_dir[1024];
    int verbose;
    float *scratch_h; size_t scratch_cap;
};

#define F4(x) ((size_t)(x) * sizeof(float))
#define PD(x) ((void *)&(CUdeviceptr){(CUdeviceptr)(x)})
#define PI(x) ((void *)&(int){(int)(x)})
#define PL(x) ((void *)&(long long){(long long)(x)})
#define PF(x) ((void *)&(float){(float)(x)})
#define FO(p, n) ((CUdeviceptr)((p) + F4(n)))
#define PRC(call) do { CUresult e_ = (call); if (e_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(e_, &s_); \
    fprintf(stderr, "praxis: CUDA error %s:%d: %s\n", __FILE__, __LINE__, s_); return -1; } } while (0)

/* ---- tracked allocation ---- */
static void pr_track(praxis_runner *r) {
    size_t fr = 0, tot = 0;
    cuMemGetInfo(&fr, &tot);
    if (fr < r->min_free) r->min_free = fr;
}
static CUdeviceptr pr_malloc(praxis_runner *r, size_t bytes) {
    CUdeviceptr p = 0;
    if (cuMemAlloc(&p, bytes ? bytes : 4) != CUDA_SUCCESS) { fprintf(stderr, "praxis: cuMemAlloc(%zu) failed\n", bytes); return 0; }
    r->cur_bytes += bytes;
    if (r->cur_bytes > r->peak_bytes) r->peak_bytes = r->cur_bytes;
    if (r->n_alloc == r->cap_alloc) { r->cap_alloc = r->cap_alloc ? 2 * r->cap_alloc : 1024; r->allocs = (CUdeviceptr *)realloc(r->allocs, (size_t)r->cap_alloc * sizeof(CUdeviceptr)); }
    r->allocs[r->n_alloc++] = p;
    pr_track(r);
    return p;
}
static void pr_mfree(praxis_runner *r, CUdeviceptr p, size_t bytes) {
    if (!p) return;
    cuMemFree(p); r->cur_bytes -= bytes;
    for (int i = r->n_alloc - 1; i >= 0; i--) if (r->allocs[i] == p) { r->allocs[i] = r->allocs[--r->n_alloc]; break; }
}
/* workspace slot of n floats (grows on demand) */
static CUdeviceptr pr_ws(praxis_runner *r, int slot, size_t nfloats) {
    size_t bytes = F4(nfloats) + 64;
    cu_buf_slot *s = &r->ws[slot];
    if (s->ptr && s->cap >= bytes) return s->ptr;
    if (s->ptr) { cuMemFree(s->ptr); r->cur_bytes -= s->cap; r->ws_bytes -= s->cap; s->ptr = 0; }
    if (cuMemAlloc(&s->ptr, bytes) != CUDA_SUCCESS) { fprintf(stderr, "praxis: ws alloc %zu failed\n", bytes); s->ptr = 0; s->cap = 0; return 0; }
    s->cap = bytes; r->cur_bytes += bytes; r->ws_bytes += bytes;
    if (r->cur_bytes > r->peak_bytes) r->peak_bytes = r->cur_bytes;
    pr_track(r);
    return s->ptr;
}
static CUdeviceptr pr_ws_i(praxis_runner *r, int slot, size_t n) { return pr_ws(r, slot, n); }
/* workspace slot of n 16-bit elements */
static CUdeviceptr pr_wsb(praxis_runner *r, int slot, size_t n16) { return pr_ws(r, slot, (n16 + 1) / 2 + 16); }

/* profile categories (PRAXIS_PROFILE=1: sync after each phase and charge the elapsed wall time) */
enum { PT_OTHER, PT_ATTN, PT_PROJ, PT_DENSE, PT_ROUTER, PT_EXPERT, PT_MOEOTH, PT_STALL, PT_FQ, PT_FG1, PT_FSW, PT_FG2, PT_N };
static const char *pr_tnames[PT_N] = { "other", "attn(flash+qk prep)", "attn proj gemms", "dense ffn", "moe router+dispatch",
                                       "moe routed gemms", "moe shared+combine", "wait for expert slot (stall)",
                                       "  nvfp4: act quant (amax+quantize)", "  nvfp4: gate_up gemm", "  nvfp4: swiglu", "  nvfp4: down gemm" };
static double pr_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void pr_tick(praxis_runner *r, int cat) {
    if (!r->prof) return;
    cuStreamSynchronize(0);   /* compute stream only: the copy stream keeps prefetching */
    double n = pr_now();
    r->tcat[cat] += n - r->tlast; r->tlast = n;
}

/* ---- launch ---- */
#define LAUNCH(f, gx, gy, gz, bx, sm, ...) do { void *a_[] = { __VA_ARGS__ }; \
    CUresult e_ = cuLaunchKernel((f), (unsigned)(gx), (unsigned)(gy), (unsigned)(gz), (unsigned)(bx), 1, 1, (unsigned)(sm), 0, a_, NULL); \
    if (e_ != CUDA_SUCCESS) { fprintf(stderr, "praxis: launch failed %s:%d (%d)\n", __FILE__, __LINE__, (int)e_); return -1; } } while (0)
#define NBLK(n) (((long long)(n) + 255) / 256)

static int pr_gemm_naive(praxis_runner *r, CUdeviceptr X, long long ldx, const pr_mat *W, int wl, CUdeviceptr bias,
                         CUdeviceptr Y, long long ldy, int M, int N, int K, int accum) {
    if (M <= 0) return 0;
    LAUNCH(r->f_gemm, (N + 31) / 32, (M + 31) / 32, 1, 256, 0, PD(X), PL(ldx), PD(0), PD(W->p), PI(W->dt), PI(wl), PL(0),
           PD(bias), PD(Y), PL(ldy), PI(M), PI(N), PI(K), PI(accum), PD(0));
    return 0;
}
/* Y[M][N] (fp32 or bf16, ytype 0/1) (+)= X W^T + bias. Fast mode: cuBLAS bf16 x bf16 -> fp32 accumulate when the shapes are
 * tensor-core friendly. Xb (dense bf16 [M][K]) may be given, otherwise Xf (fp32, row stride ldx) is converted on the fly. */
static int pr_gemm2(praxis_runner *r, CUdeviceptr Xf, CUdeviceptr Xb, long long ldx, const pr_mat *W, int wl, CUdeviceptr bias,
                    CUdeviceptr Y, long long ldy, int ytype, int M, int N, int K, int accum) {
    if (M <= 0) return 0;
    int ok = r->fast && W->dt == 1 && (K % 8) == 0 && (N % 8) == 0 && (Xb || Xf) && !(ytype == 1 && bias);
    if (!ok) {
        if (!Xf || ytype) { fprintf(stderr, "praxis: gemm needs the fp32 path (M=%d N=%d K=%d) without fp32 input/output\n", M, N, K); return -1; }
        return pr_gemm_naive(r, Xf, ldx, W, wl, bias, Y, ldy, M, N, K, accum);
    }
    if (!Xb) {
        Xb = pr_wsb(r, WS_XB, (size_t)M * K);
        if (!Xb) return -1;
        LAUNCH(r->f_cvt, NBLK((long long)M * K), 1, 1, 256, 0, PD(Xf), PL(ldx), PD(Xb), PI(K), PL(M));
    }
    if (cublasew_gemm_ex_rm(r->cb, wl, M, N, K, W->p, wl ? N : K, 1, Xb, K, 1, Y, (int)ldy, ytype ? 1 : 0, accum ? 1.0f : 0.0f)) {
        fprintf(stderr, "praxis: cuBLAS gemm failed (M=%d N=%d K=%d wl=%d)\n", M, N, K, wl); return -1; }
    if (bias) LAUNCH(r->f_bias, NBLK((long long)M * N), 1, 1, 256, 0, PD(Y), PL(ldy), PD(bias), PL(M), PI(N));
    return 0;
}
static int pr_gemm(praxis_runner *r, CUdeviceptr X, long long ldx, const pr_mat *W, int wl, CUdeviceptr bias,
                   CUdeviceptr Y, long long ldy, int M, int N, int K, int accum) {
    return pr_gemm2(r, X, 0, ldx, W, wl, bias, Y, ldy, 0, M, N, K, accum);
}
static int pr_rms(praxis_runner *r, CUdeviceptr x, long long ldx, CUdeviceptr y, long long ldy, CUdeviceptr w, int plus1,
                  float eps, float oscale, int C, long long rows, int heads) {
    LAUNCH(r->f_rms, rows * heads, 1, 1, C >= 128 ? 128 : 32, 0, PD(x), PL(ldx), PD(y), PL(ldy), PD(w), PI(plus1), PF(eps), PF(oscale), PI(C), PI(heads));
    return 0;
}
static int pr_addv(praxis_runner *r, CUdeviceptr y, CUdeviceptr x, long long n) {
    LAUNCH(r->f_add, NBLK(n), 1, 1, 256, 0, PD(y), PD(x), PL(n));
    return 0;
}
static int pr_copyrows(praxis_runner *r, CUdeviceptr src, long long lds, CUdeviceptr dst, long long ldd, long long rows, int C) {
    LAUNCH(r->f_copy, NBLK(rows * C), 1, 1, 256, 0, PD(src), PL(lds), PD(dst), PL(ldd), PL(rows), PI(C));
    return 0;
}
static int pr_actk(praxis_runner *r, CUdeviceptr x, CUdeviceptr y, long long n, int mode) {
    LAUNCH(r->f_act, NBLK(n), 1, 1, 256, 0, PD(x), PD(y), PL(n), PI(mode));
    return 0;
}
static int pr_swi(praxis_runner *r, CUdeviceptr a, long long lda, CUdeviceptr b, long long ldb, CUdeviceptr o, long long ldo, int H, long long rows, int mode) {
    LAUNCH(r->f_swiglu, NBLK(rows * H), 1, 1, 256, 0, PD(a), PL(lda), PD(b), PL(ldb), PD(o), PL(ldo), PI(H), PL(rows), PI(mode));
    return 0;
}
static int pr_ropek(praxis_runner *r, CUdeviceptr q, long long ld, int heads, int hd, CUdeviceptr cs, CUdeviceptr sn, long long n) {
    LAUNCH(r->f_rope, NBLK(n * heads * (hd / 2)), 1, 1, 256, 0, PD(q), PL(ld), PI(heads), PI(hd), PD(cs), PD(sn), PL(n));
    return 0;
}
static int pr_attn(praxis_runner *r, CUdeviceptr Q, long long ldq, CUdeviceptr K, long long ldk, CUdeviceptr V, long long ldv,
                   CUdeviceptr O, long long ldo, CUdeviceptr gate, int H, int KVH, int hd, int nq, int nk,
                   int mode, CUdeviceptr mask, CUdeviceptr qcode, CUdeviceptr kcode) {
    size_t sm = F4(hd + nk + 32);
    if (sm > 48 * 1024) {
        if (cuFuncSetAttribute(r->f_attn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)sm) != CUDA_SUCCESS) {
            fprintf(stderr, "praxis: attention needs %zu B smem (kv=%d); block-sparse kernel required\n", sm, nk); return -1; }
    }
    LAUNCH(r->f_attn, nq, H, 1, 128, sm, PD(Q), PL(ldq), PD(K), PL(ldk), PD(V), PL(ldv), PD(O), PL(ldo), PD(gate), PI(H), PI(KVH), PI(hd),
           PI(nq), PI(nk), PF(1.0 / sqrt((double)hd)), PI(mode), PD(mask), PD(qcode), PD(kcode));
    return 0;
}

/* ---- npy output ---- */
static void pr_npy_write(const char *path, const float *d, int nd, const long long *shape) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "praxis: cannot write %s\n", path); return; }
    char hdr[256], sh[200] = "";
    size_t n = 1;
    for (int i = 0; i < nd; i++) { char t[32]; snprintf(t, sizeof t, "%lld, ", shape[i]); strcat(sh, t); n *= shape[i]; }
    int hl = snprintf(hdr, sizeof hdr, "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", sh);
    int pad = (64 - (10 + hl + 1) % 64) % 64;
    unsigned short hlen = (unsigned short)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(d, 4, n, f);
    fclose(f);
}
/* PRAXIS_DUMP_LITE=1: only the routing dumps (router logits, top-k indices) are written, not every block tap (4.7 GB per run) */
static int pr_dump_skip(const char *name) {
    static int lite = -1; if (lite < 0) { const char *e = getenv("PRAXIS_DUMP_LITE"); lite = e && *e && *e != '0'; }
    return lite && !strstr(name, "top_idx") && !strstr(name, "router_logits") && !strstr(name, "moe_x") && !strstr(name, "top_gate");
}
static void pr_dump(praxis_runner *r, const char *name, CUdeviceptr d, long long rows, long long cols) {
    if (!r->dump_dir[0] || pr_dump_skip(name)) return;
    size_t n = (size_t)(rows * cols);
    float *h = (float *)malloc(F4(n));
    cuCtxSynchronize();
    cuMemcpyDtoH(h, d, F4(n));
    char p[1200];
    snprintf(p, sizeof p, "%s/%s.npy", r->dump_dir, name);
    long long sh[2] = { rows, cols };
    pr_npy_write(p, h, cols > 0 ? 2 : 1, sh);
    free(h);
}
static void pr_dump_i(praxis_runner *r, const char *name, CUdeviceptr d, long long rows, long long cols) {
    if (!r->dump_dir[0] || pr_dump_skip(name)) return;
    size_t n = (size_t)(rows * cols);
    int *hi = (int *)malloc(n * 4);
    float *h = (float *)malloc(F4(n));
    cuCtxSynchronize();
    cuMemcpyDtoH(hi, d, n * 4);
    for (size_t i = 0; i < n; i++) h[i] = (float)hi[i];
    char p[1200];
    snprintf(p, sizeof p, "%s/%s.npy", r->dump_dir, name);
    long long sh[2] = { rows, cols };
    pr_npy_write(p, h, 2, sh);
    free(hi); free(h);
}

/* ---- config ---- */
static double pr_jnum(const json_val *o, const char *k, double dflt) {
    json_val *v = json_obj_get(o, k);
    if (!v) return dflt;
    if (v->type == JSON_TRUE) return 1;
    if (v->type == JSON_FALSE || v->type == JSON_NULL) return 0;
    return v->type == JSON_NUMBER ? v->num : dflt;
}
static int pr_parse_config(const char *path, pr_cfg *c) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "praxis: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return -1; }
    fclose(f); buf[n] = 0;
    json_val *root = json_parse(buf, (int)n);
    free(buf);
    json_val *bb = json_obj_get(root, "backbone"), *ex = json_obj_get(root, "action_expert"), *ch = json_obj_get(root, "chunking");
    if (!bb || !ex || !ch) { fprintf(stderr, "praxis: config lacks backbone/action_expert/chunking\n"); json_free(root); return -1; }
    json_val *at = json_obj_get(bb, "attention"), *rp = json_obj_get(bb, "rope"), *mo = json_obj_get(bb, "moe");
    json_val *en = json_obj_get(bb, "encoder"), *de = json_obj_get(bb, "decoder");
    memset(c, 0, sizeof *c);
    c->dim = (int)pr_jnum(bb, "dim", 3072);
    c->time_dim = (int)pr_jnum(bb, "timestep_dim", 128);
    c->n_reg = (int)pr_jnum(bb, "num_registers", 4);
    c->dense_hidden = (int)pr_jnum(bb, "dense_ffn_hidden", 11520);
    c->head_dim = (int)pr_jnum(at, "head_dim", 128);
    c->n_kv_heads = (int)pr_jnum(at, "num_kv_heads", 8);
    c->n_heads = c->dim / c->head_dim;
    c->rope_eff = (int)pr_jnum(rp, "effective_dim", 64);
    c->rope_base = pr_jnum(rp, "base", 26000000);
    c->rope_tab = (long long)pr_jnum(rp, "efficient_cache_max", 1000000);
    c->n_experts = (int)pr_jnum(mo, "num_routed_experts", 128);
    c->top_k = (int)pr_jnum(mo, "top_k_routed", 8);
    c->moe_hidden = (int)pr_jnum(mo, "expert_hidden", 1280);
    c->routed_scale = (float)pr_jnum(mo, "routed_scaling_factor", 2.5);
    c->enc_depth = (int)pr_jnum(en, "depth", 14);
    { json_val *d = json_obj_get(en, "dense_blocks"); c->enc_dense = d ? d->arr.count : 2;
      for (int i = 0; d && i < d->arr.count; i++) if ((int)d->arr.items[i].num != i) { fprintf(stderr, "praxis: dense_blocks must be a prefix\n"); json_free(root); return -1; } }
    { json_val *t = json_obj_get(en, "tap_blocks"); c->enc_ntap = t ? t->arr.count : 0;
      for (int i = 0; t && i < t->arr.count && i < PR_MAXTAPS; i++) c->enc_taps[i] = (int)t->arr.items[i].num; }
    { json_val *ip = json_obj_get(en, "input_projections");
      c->text_cin = (int)pr_jnum(json_obj_get(ip, "TextEncodingTensor"), "c_in", 2048);
      c->enc_vcin = (int)pr_jnum(json_obj_get(ip, "VideoTensor"), "c_in", 192); }
    c->dec_depth = (int)pr_jnum(de, "depth", 6);
    { json_val *d = json_obj_get(de, "dense_blocks"); c->dec_dense = d ? d->arr.count : 1; }
    c->dec_mod_dim = (int)pr_jnum(de, "modulation_dim", c->dim);
    { json_val *u = json_obj_get(de, "unpatch_factor"); c->dec_ph = u ? (int)u->arr.items[1].num : 2; c->dec_pw = u ? (int)u->arr.items[2].num : 2; }
    { json_val *ip = json_obj_get(de, "input_projections"); c->dec_vcin = (int)pr_jnum(json_obj_get(ip, "VideoTensor"), "c_in", 48); }
    c->row_stride = (int)pr_jnum(rp, "max_intra", pr_jnum(bb, "row_stride", 360));
    c->frame_stride = (int)pr_jnum(rp, "max_inter", pr_jnum(bb, "frame_stride", 129600));
    c->ae_dim = (int)pr_jnum(ex, "dim", 1024);
    c->ae_layers = (int)pr_jnum(ex, "num_layers", 16);
    c->ae_heads = (int)pr_jnum(ex, "num_heads", 16);
    c->ae_hd = (int)pr_jnum(ex, "head_dim", 64);
    c->trunk_dim = (int)pr_jnum(ex, "trunk_dim", c->dim);
    c->act_dim = (int)pr_jnum(ex, "action_patch_dim", 40);
    c->prop_dim = (int)pr_jnum(ex, "proprio_patch_dim", 10);
    c->nfut = (int)pr_jnum(ex, "num_future_per_chunk", 8);
    c->ae_max_chunks = (int)pr_jnum(ex, "max_chunks", 16);
    c->tap_norm = (int)pr_jnum(ex, "tap_norm", 1);
    c->clean_only = (int)pr_jnum(ex, "clean_only_conditioning", 0);
    { json_val *r2 = json_obj_get(ex, "rope"); c->ae_rope_dim = (int)pr_jnum(r2, "dim", 64); c->ae_rope_eff = (int)pr_jnum(r2, "effective_dim", 64);
      c->ae_rope_base = pr_jnum(r2, "base", 26000000); c->ae_rope_tab = (long long)pr_jnum(r2, "table_size", 1000000); }
    { json_val *g = json_obj_get(ex, "global_order"); int def[3] = { 0, 1, 2 };
      for (int i = 0; i < 3; i++) c->gorder[i] = def[i];
      for (int i = 0; g && i < g->arr.count && i < 3; i++) {
          const char *s = g->arr.items[i].str.ptr;
          c->gorder[i] = !strncmp(s, "register", 8) ? 0 : !strncmp(s, "anchor", 6) ? 1 : 2; } }
    c->n_chunks = (int)pr_jnum(ch, "n_chunks", 4);
    c->lat = (int)pr_jnum(ch, "latents_per_chunk", 6);
    /* semantics the runner hard-codes (praxis-1-oss defaults): refuse a config that selects another variant instead of silently computing the wrong thing */
    {
        int bad = 0;
        #define PR_REQ_STR(obj, key, want) do { json_val *v_ = json_obj_get(obj, key); if (v_ && v_->type == JSON_STRING && strncmp(v_->str.ptr, want, strlen(want)) != 0) { \
            fprintf(stderr, "praxis: unsupported config %s=\"%.*s\" (runner implements \"%s\")\n", key, (int)v_->str.len, v_->str.ptr, want); bad = 1; } } while (0)
        #define PR_REQ_NUM(obj, key, want) do { json_val *v_ = json_obj_get(obj, key); if (v_ && v_->type == JSON_NUMBER && fabs(v_->num - (want)) > 1e-12) { \
            fprintf(stderr, "praxis: unsupported config %s=%g (runner implements %g)\n", key, v_->num, (double)(want)); bad = 1; } } while (0)
        #define PR_REQ_BOOL(obj, key, want) do { json_val *v_ = json_obj_get(obj, key); if (v_ && (v_->type == JSON_TRUE || v_->type == JSON_FALSE) && ((v_->type == JSON_TRUE) != (want))) { \
            fprintf(stderr, "praxis: unsupported config %s=%s (runner implements %s)\n", key, v_->type == JSON_TRUE ? "true" : "false", (want) ? "true" : "false"); bad = 1; } } while (0)
        PR_REQ_STR(bb, "layout", "chunk_causal"); PR_REQ_STR(bb, "modulation_dtype", "mixed");
        PR_REQ_NUM(bb, "norm_eps", 1e-6); PR_REQ_NUM(at, "qk_norm_eps", 1e-6); PR_REQ_BOOL(at, "gated", 1); PR_REQ_BOOL(at, "rope_after_norm", 0);
        PR_REQ_STR(rp, "ref_placement", "group"); PR_REQ_BOOL(rp, "chunked_video_layout", 1); PR_REQ_NUM(rp, "time_scale", 1.0);
        PR_REQ_STR(mo, "router_activation", "sigmoid"); PR_REQ_STR(mo, "router_dtype", "float32"); PR_REQ_STR(mo, "routed_weight_layout", "expert_first");
        PR_REQ_BOOL(mo, "shared_expert", 1); PR_REQ_BOOL(mo, "no_expert_slots", 0);
        { json_val *tk = json_obj_get(mo, "top_k"); if (tk && tk->type == JSON_NUMBER && (int)tk->num != c->top_k + 1) { fprintf(stderr, "praxis: moe.top_k=%g != top_k_routed+shared=%d\n", tk->num, c->top_k + 1); bad = 1; } }
        PR_REQ_BOOL(ex, "rope_cross", 0); PR_REQ_BOOL(ex, "rope_self", 0); PR_REQ_BOOL(ex, "absolute_positions", 1); PR_REQ_BOOL(ex, "action_pos_table", 0);
        PR_REQ_STR(ex, "future_chunk_offset", "zero");
        { json_val *r2 = json_obj_get(ex, "rope"); if (r2) { PR_REQ_STR(r2, "kind", "trunk"); PR_REQ_STR(r2, "order", "before_norm"); } }
        if (bad) { json_free(root); return -1; }
    }
    json_free(root);
    if (c->enc_depth > PR_MAXD || c->dec_depth > PR_MAXD || c->ae_layers > PR_MAXD || c->top_k > 32 || c->n_experts > 512) {
        fprintf(stderr, "praxis: config exceeds compiled limits\n"); return -1; }
    return 0;
}

/* ---- weight loading ---- */
static unsigned long long pr_name_hash(const char *s) {
    unsigned long long h = 1469598103934665603ULL;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
    return h;
}
/* ---- (sharded) safetensors lookup: model.safetensors.index.json + N shards, or a single model.safetensors ---- */
static int pr_st_find(praxis_runner *r, const char *name, st_context **ctx) {
    { static int lk = -1; if (lk < 0) { const char *e = getenv("PRAXIS_LOG_KEYS"); lk = e && *e && *e != '0'; }
      if (lk) fprintf(stderr, "PRAXIS_KEY %s\n", name); }
    for (int k = 0; k < r->n_shards; k++) {
        int i = safetensors_find(r->shard[k], name);
        if (i >= 0) { *ctx = r->shard[k]; return i; }
    }
    return -1;
}
static int pr_st_open(praxis_runner *r, const char *dir) {
    char path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(path, sizeof path, "%s/model.safetensors", dir);
        r->shard[0] = safetensors_open(path);
        if (!r->shard[0]) return -1;
        r->shard_path[0] = strdup(path); r->n_shards = 1; r->st = r->shard[0];
        return 0;
    }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return -1; }
    fclose(f); buf[n] = 0;
    json_val *root = json_parse(buf, (int)n);
    free(buf);
    json_val *wm = root ? json_obj_get(root, "weight_map") : NULL;
    if (!wm || wm->type != JSON_OBJECT) { fprintf(stderr, "praxis: %s has no weight_map\n", path); json_free(root); return -1; }
    for (int i = 0; i < wm->obj.count; i++) {   /* unique shard file names, in first-seen order */
        json_val *v = &wm->obj.vals[i];
        if (v->type != JSON_STRING) continue;
        char fn[512]; int l = v->str.len < 511 ? v->str.len : 511; memcpy(fn, v->str.ptr, l); fn[l] = 0;
        int seen = 0;
        for (int k = 0; k < r->n_shards; k++) { const char *b = strrchr(r->shard_path[k], '/'); if (!strcmp(b ? b + 1 : r->shard_path[k], fn)) { seen = 1; break; } }
        if (seen) continue;
        if (r->n_shards >= PR_MAXSH) { fprintf(stderr, "praxis: too many shards\n"); json_free(root); return -1; }
        char sp[1800]; snprintf(sp, sizeof sp, "%s/%s", dir, fn);
        st_context *c = safetensors_open(sp);
        if (!c) { fprintf(stderr, "praxis: cannot open shard %s\n", sp); json_free(root); return -1; }
        r->shard[r->n_shards] = c; r->shard_path[r->n_shards] = strdup(sp); r->n_shards++;
    }
    json_free(root);
    if (r->n_shards == 0) return -1;
    r->st = r->shard[0];
    return 0;
}
/* drop the file-backed pages of a tensor range from this process (RSS) once it has been consumed */
static void pr_drop_pages(const void *p, size_t bytes) {
    static int nodrop = -1;
    if (nodrop < 0) { const char *e = getenv("PRAXIS_NO_DROP"); nodrop = e && *e && *e != '0'; }   /* A/B knob for the RSS test */
    if (nodrop) return;
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)p, b = a + bytes;
    a &= ~(uintptr_t)(pg - 1); b = (b + pg - 1) & ~(uintptr_t)(pg - 1);
    madvise((void *)a, b - a, MADV_DONTNEED);
}
static int pr_ld_mat(praxis_runner *r, const char *name, pr_mat *m, size_t expect_elems) {
    if (r->synth) {   /* random bf16 device matrix, no file */
        m->dt = 1; m->bytes = expect_elems * 2;
        m->p = pr_malloc(r, m->bytes);
        if (!m->p) return -1;
        LAUNCH(r->f_fillrand, NBLK((long long)expect_elems), 1, 1, 256, 0, PD(m->p), PL(expect_elems), PL(pr_name_hash(name) & 0xffffffffffffULL), PF(0.03));
        r->w_bytes += m->bytes;
        return 0;
    }
    st_context *st = NULL;
    int i = pr_st_find(r, name, &st);
    if (i < 0) { fprintf(stderr, "praxis: missing tensor %s\n", name); return -1; }
    const char *dt = safetensors_dtype(st, i);
    m->dt = !strcmp(dt, "F32") ? 0 : !strcmp(dt, "BF16") ? 1 : !strcmp(dt, "F16") ? 2 : -1;
    if (m->dt < 0) { fprintf(stderr, "praxis: %s has unsupported dtype %s\n", name, dt); return -1; }
    size_t nb = safetensors_nbytes(st, i);
    if (expect_elems && nb != expect_elems * (m->dt == 0 ? 4 : 2)) { fprintf(stderr, "praxis: %s has %zu bytes, expected %zu elems\n", name, nb, expect_elems); return -1; }
    m->bytes = nb;
    m->p = pr_malloc(r, nb);
    if (!m->p) return -1;
    cuMemcpyHtoD(m->p, safetensors_data(st, i), nb);
    pr_drop_pages(safetensors_data(st, i), nb);
    r->w_bytes += nb;
    return 0;
}
static float pr_bf2f(const void *p, int dt, size_t i) {
    if (dt == 0) return ((const float *)p)[i];
    unsigned short h = ((const unsigned short *)p)[i];
    if (dt == 1) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
    /* f16 */
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float f;
    if (e == 0) f = ldexpf((float)m, -24); else if (e == 31) f = m ? NAN : INFINITY; else f = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -f : f;
}
/* vector / small table -> f32 on device; optionally returns host copy via *host (malloc'd) */
static int pr_ld_vec(praxis_runner *r, const char *name, CUdeviceptr *d, size_t expect, float **host) {
    if (r->synth) {
        size_t n = expect ? expect : 1;
        float *h = (float *)malloc(F4(n));
        unsigned long long st = pr_name_hash(name) | 1;
        size_t ln = strlen(name);
        int is_s = ln >= 2 && !strcmp(name + ln - 2, ".s");
        int is_bias = strstr(name, "expert_bias") != NULL;
        int is_freq = strstr(name, "freqs") != NULL || strstr(name, "phases") != NULL;
        for (size_t k = 0; k < n; k++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            float u = (float)(st >> 40) / 8388608.0f - 1.0f;
            h[k] = is_s ? 1.0f : is_bias ? 0.0f : is_freq ? u : 0.02f * u;
        }
        *d = pr_malloc(r, F4(n));
        if (!*d) { free(h); return -1; }
        cuMemcpyHtoD(*d, h, F4(n));
        r->w_bytes += F4(n);
        if (host) *host = h; else free(h);
        return 0;
    }
    st_context *st = NULL;
    int i = pr_st_find(r, name, &st);
    if (i < 0) { fprintf(stderr, "praxis: missing tensor %s\n", name); return -1; }
    const char *dts = safetensors_dtype(st, i);
    int dt = !strcmp(dts, "F32") ? 0 : !strcmp(dts, "BF16") ? 1 : !strcmp(dts, "F16") ? 2 : -1;
    if (dt < 0) { fprintf(stderr, "praxis: %s dtype %s\n", name, dts); return -1; }
    size_t n = safetensors_nbytes(st, i) / (dt == 0 ? 4 : 2);
    if (expect && n != expect) { fprintf(stderr, "praxis: %s has %zu elems, expected %zu\n", name, n, expect); return -1; }
    float *h = (float *)malloc(F4(n));
    for (size_t k = 0; k < n; k++) h[k] = pr_bf2f(safetensors_data(st, i), dt, k);
    *d = pr_malloc(r, F4(n));
    if (!*d) { free(h); return -1; }
    cuMemcpyHtoD(*d, h, F4(n));
    r->w_bytes += F4(n);
    if (host) *host = h; else free(h);
    return 0;
}

#define LDM(field, fmt, n, ...) do { snprintf(nm, sizeof nm, fmt, __VA_ARGS__); if (pr_ld_mat(r, nm, &(field), (size_t)(n))) return -1; } while (0)
#define LDV(field, fmt, n, ...) do { snprintf(nm, sizeof nm, fmt, __VA_ARGS__); if (pr_ld_vec(r, nm, &(field), (size_t)(n), NULL)) return -1; } while (0)

static int pr_ld_tower(praxis_runner *r, pr_tower *t, const char *pre, int enc, int *moe_counter) {
    const pr_cfg *c = &r->c;
    char nm[512];
    int C = c->dim, hd = c->head_dim, H = c->n_heads, KVH = c->n_kv_heads;
    t->depth = enc ? c->enc_depth : c->dec_depth;
    t->dense = enc ? c->enc_dense : c->dec_dense;
    LDV(t->registers, "%sregisters", (size_t)c->n_reg * C, pre);
    LDV(t->t_freq, "%st_embedding.freqs", c->time_dim, pre);
    LDV(t->t_phase, "%st_embedding.phases", c->time_dim, pre);
    LDM(t->tp_w, "%st_proj.VideoTensor.weight", (size_t)2 * C * c->time_dim, pre);
    LDV(t->tp_b, "%st_proj.VideoTensor.bias", 2 * C, pre);
    LDM(t->vin_w, "%sinput_projections.VideoTensor.proj.weight", (size_t)C * (enc ? c->enc_vcin : c->dec_vcin), pre);
    LDV(t->vin_b, "%sinput_projections.VideoTensor.proj.bias", C, pre);
    if (enc) {
        LDM(t->tx0_w, "%sinput_projections.TextEncodingTensor.proj.0.weight", (size_t)4 * C * c->text_cin, pre);
        LDV(t->tx0_b, "%sinput_projections.TextEncodingTensor.proj.0.bias", 4 * C, pre);
        LDM(t->tx2_w, "%sinput_projections.TextEncodingTensor.proj.2.weight", (size_t)4 * C * C, pre);
        LDV(t->tx2_b, "%sinput_projections.TextEncodingTensor.proj.2.bias", C, pre);
        LDV(t->tx_norm, "%sinput_projections.TextEncodingTensor.proj.3.weight", C, pre);
        LDM(t->out_w, "%soutput_projections.VideoTensor.proj.weight", (size_t)C * C, pre);
        LDV(t->out_b, "%soutput_projections.VideoTensor.proj.bias", C, pre);
    } else {
        LDV(t->out_norm, "%soutput_projections.VideoTensor.proj.0.weight", C, pre);
        LDM(t->out_w, "%soutput_projections.VideoTensor.proj.1.weight", (size_t)c->dec_vcin * C, pre);
        LDV(t->out_b, "%soutput_projections.VideoTensor.proj.1.bias", c->dec_vcin, pre);
    }
    for (int i = 0; i < t->depth; i++) {
        pr_blk *b = &t->b[i];
        float *sh = NULL; CUdeviceptr dummy = 0;
        snprintf(nm, sizeof nm, "%score.blocks.%d.s", pre, i);
        if (pr_ld_vec(r, nm, &dummy, 1, &sh)) return -1;
        b->s = sh[0]; free(sh); pr_mfree(r, dummy, 4); r->w_bytes -= 4;
        LDV(b->ln_in, "%score.blocks.%d.ln_in.weight", C, pre, i);
        LDV(b->ln_out, "%score.blocks.%d.ln_out.weight", C, pre, i);
        LDM(b->q, "%score.blocks.%d.mha.in_proj.q_weight", (size_t)H * hd * C, pre, i);
        LDM(b->k, "%score.blocks.%d.mha.in_proj.k_weight", (size_t)KVH * hd * C, pre, i);
        LDM(b->v, "%score.blocks.%d.mha.in_proj.v_weight", (size_t)KVH * hd * C, pre, i);
        LDM(b->gate, "%score.blocks.%d.mha.gate_proj.weight", (size_t)H * C, pre, i);
        LDV(b->qn, "%score.blocks.%d.mha.q_norm.weight", hd, pre, i);
        LDV(b->kn, "%score.blocks.%d.mha.k_norm.weight", hd, pre, i);
        LDM(b->out, "%score.blocks.%d.mha.out_proj.weight", (size_t)C * H * hd, pre, i);
        LDV(b->out_b, "%score.blocks.%d.mha.out_proj.bias", C, pre, i);
        if (i < t->dense) {
            LDM(b->a, "%score.blocks.%d.ffn.0.linear.a_weight", (size_t)c->dense_hidden * C, pre, i);
            LDM(b->b, "%score.blocks.%d.ffn.0.linear.b_weight", (size_t)c->dense_hidden * C, pre, i);
            LDM(b->down, "%score.blocks.%d.ffn.1.weight", (size_t)C * c->dense_hidden, pre, i);
        } else {
            int E = c->n_experts, Hm = c->moe_hidden;
            b->moe = 1;
            b->moe_id = (*moe_counter)++;
            LDV(b->ebias, "%score.blocks.%d.ffn.expert_bias", E, pre, i);
            LDM(b->router, "%score.blocks.%d.ffn.router.weight", (size_t)E * C, pre, i);
            LDM(b->sh_gu, "%score.blocks.%d.ffn.shared.gate_up", (size_t)C * 2 * Hm, pre, i);
            LDM(b->sh_dn, "%score.blocks.%d.ffn.shared.down", (size_t)Hm * C, pre, i);
            /* routed banks: pinned host copy (stream mode) or device (resident) - decided in praxis_set_stream */
            int ig, id;
            if (r->synth) {
                b->bank_dt = 1;
                b->gu_bytes = (size_t)E * C * 2 * Hm * 2; b->dn_bytes = (size_t)E * Hm * C * 2;
                r->routed_total += b->gu_bytes + b->dn_bytes;
                b->h_gu = b->h_dn = NULL;
                goto banks_done;
            }
            st_context *sg = NULL, *sd = NULL;
            snprintf(nm, sizeof nm, "%score.blocks.%d.ffn.routed.gate_up_weight", pre, i); ig = pr_st_find(r, nm, &sg);
            snprintf(nm, sizeof nm, "%score.blocks.%d.ffn.routed.down_weight", pre, i); id = pr_st_find(r, nm, &sd);
            if (ig < 0 || id < 0) { fprintf(stderr, "praxis: missing routed banks of %s block %d\n", pre, i); return -1; }
            const char *dts = safetensors_dtype(sg, ig);
            b->bank_dt = !strcmp(dts, "F32") ? 0 : !strcmp(dts, "BF16") ? 1 : 2;
            b->gu_bytes = safetensors_nbytes(sg, ig); b->dn_bytes = safetensors_nbytes(sd, id);
            size_t es = b->bank_dt == 0 ? 4 : 2;
            if (b->gu_bytes != (size_t)E * C * 2 * Hm * es || b->dn_bytes != (size_t)E * Hm * C * es) { fprintf(stderr, "praxis: routed bank shape mismatch\n"); return -1; }
            b->h_gu = safetensors_data(sg, ig); b->h_dn = safetensors_data(sd, id);   /* mmap'd; pinned copy made later */
            r->routed_total += b->gu_bytes + b->dn_bytes;
        banks_done: ;
        }
        if (!enc) {
            LDM(b->mod, "%score.modulators.%d.weight", (size_t)2 * C * c->dec_mod_dim, pre, i);
            LDV(b->mod_b, "%score.modulators.%d.bias", 2 * C, pre, i);
        }
    }
    t->n_moe = 0;
    for (int i = 0; i < t->depth; i++) t->n_moe += t->b[i].moe;
    return 0;
}

static int pr_ld_ae(praxis_runner *r) {
    const pr_cfg *c = &r->c;
    pr_ae *a = &r->ae;
    char nm[512];
    int A = c->ae_dim, inner = c->ae_heads * c->ae_hd;
    const char *p = "action_expert.";
    LDM(a->w1, "%saction_encoder.w1.weight", (size_t)A * c->act_dim, p);
    LDV(a->w1_b, "%saction_encoder.w1.bias", A, p);
    LDM(a->w2, "%saction_encoder.w2.weight", (size_t)A * 2 * A, p);
    LDV(a->w2_b, "%saction_encoder.w2.bias", A, p);
    LDM(a->w3, "%saction_encoder.w3.weight", (size_t)A * A, p);
    LDV(a->w3_b, "%saction_encoder.w3.bias", A, p);
    LDM(a->tau0, "%saction_encoder.tau.mlp.0.weight", (size_t)A * 256, p);
    LDV(a->tau0_b, "%saction_encoder.tau.mlp.0.bias", A, p);
    LDM(a->tau2, "%saction_encoder.tau.mlp.2.weight", (size_t)A * A, p);
    LDV(a->tau2_b, "%saction_encoder.tau.mlp.2.bias", A, p);
    LDM(a->temb0, "%stimestep_embedder.mlp.0.weight", (size_t)A * 256, p);
    LDV(a->temb0_b, "%stimestep_embedder.mlp.0.bias", A, p);
    LDM(a->temb2, "%stimestep_embedder.mlp.2.weight", (size_t)A * A, p);
    LDV(a->temb2_b, "%stimestep_embedder.mlp.2.bias", A, p);
    LDV(a->future, "%sfuture_tokens", (size_t)c->nfut * A, p);
    LDV(a->proprio_tok, "%sproprio_token", A, p);
    LDV(a->chunk_embed, "%sfuture_chunk_embed.weight", (size_t)c->ae_max_chunks * A, p);
    LDM(a->embp, "%sembed_proprio.weight", (size_t)A * c->prop_dim, p);
    LDV(a->embp_b, "%sembed_proprio.bias", A, p);
    LDM(a->tap_proj, "%stap_proj.weight", (size_t)A * c->trunk_dim, p);
    LDM(a->text_proj, "%stext_proj.weight", (size_t)A * c->trunk_dim, p);
    if (c->tap_norm) LDV(a->tap_scale, "%stap_norm_scale", c->trunk_dim, p);
    LDM(a->head, "%shead.weight", (size_t)c->act_dim * A, p);
    LDV(a->head_b, "%shead.bias", c->act_dim, p);
    for (int i = 0; i < c->ae_layers; i++) {
        pr_ae_blk *b = &a->b[i];
        int cross = (i % 2 == 0);
        LDM(b->ada_a, "action_expert.blocks.%d.adaln_attn.proj.weight", (size_t)A * A, i);
        LDV(b->ada_a_b, "action_expert.blocks.%d.adaln_attn.proj.bias", A, i);
        LDM(b->to_q, "action_expert.blocks.%d.to_q.weight", (size_t)inner * A, i);
        LDM(b->to_kv, "action_expert.blocks.%d.to_kv.weight", (size_t)2 * inner * A, i);
        (void)cross;
        LDM(b->to_out, "action_expert.blocks.%d.to_out.weight", (size_t)A * inner, i);
        LDM(b->ada_m, "action_expert.blocks.%d.adaln_mlp.proj.weight", (size_t)A * A, i);
        LDV(b->ada_m_b, "action_expert.blocks.%d.adaln_mlp.proj.bias", A, i);
        LDM(b->mlp_in, "action_expert.blocks.%d.mlp_in.weight", (size_t)8 * A * A, i);
        LDV(b->mlp_in_b, "action_expert.blocks.%d.mlp_in.bias", 8 * A, i);
        LDM(b->mlp_out, "action_expert.blocks.%d.mlp_out.weight", (size_t)4 * A * A, i);
        LDV(b->mlp_out_b, "action_expert.blocks.%d.mlp_out.bias", A, i);
    }
    /* expert sinusoid frequency table (256-dim embedder), as torch computes it in fp32 */
    {
        float f[128]; float cm = (float)(-log(10000.0));
        for (int i = 0; i < 128; i++) f[i] = expf((cm * (float)i) / 128.0f);
        a->freq256 = pr_malloc(r, sizeof f);
        cuMemcpyHtoD(a->freq256, f, sizeof f);
        r->w_bytes += sizeof f;
    }
    return 0;
}

/* ---- streaming of routed banks ---- */
static size_t pr_max_bank_bytes(praxis_runner *r) {
    size_t m = 0;
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++)
        if (ts[t]->b[i].moe) { size_t b = ts[t]->b[i].ngrp > 1 ? ts[t]->b[i].grp_bytes : ts[t]->b[i].gu_bytes + ts[t]->b[i].dn_bytes; if (b > m) m = b; }
    return m;
}
static pr_blk *pr_moe_by_id(praxis_runner *r, int id) {
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++)
        if (ts[t]->b[i].moe && ts[t]->b[i].moe_id == id) return &ts[t]->b[i];
    return NULL;
}
static void pr_release_banks(praxis_runner *r) {
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
        pr_blk *b = &ts[t]->b[i];
        if (!b->moe) continue;
        if (b->gu.p) { pr_mfree(r, b->gu.p, b->gu_bytes); b->gu.p = 0; r->routed_dev -= b->gu_bytes; }
        if (b->dn.p) { pr_mfree(r, b->dn.p, b->dn_bytes); b->dn.p = 0; r->routed_dev -= b->dn_bytes; }
    }
    for (int s = 0; s < 4; s++) {
        if (r->ring[s].p) { pr_mfree(r, r->ring[s].p, r->ring[s].bytes); r->routed_dev -= r->ring[s].bytes; r->ring[s].p = 0; }
        if (r->ring[s].ready) { cuEventDestroy(r->ring[s].ready); r->ring[s].ready = 0; }
        if (r->ring[s].done) { cuEventDestroy(r->ring[s].done); r->ring[s].done = 0; }
    }
    r->ring_n = 0; r->ring_slot_bytes = 0;
}

int praxis_set_stream(praxis_runner *r, int ring_slots) {
    cuCtxSetCurrent(r->ctx);
    cuCtxSynchronize();
    pr_release_banks(r);
    r->seq = 0;
    pr_tower *ts[2] = { &r->enc, &r->dec };
    if (ring_slots <= 0) {
        for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
            pr_blk *b = &ts[t]->b[i];
            if (!b->moe) continue;
            b->gu.dt = b->dn.dt = b->bank_dt;
            b->gu.bytes = b->gu_bytes; b->dn.bytes = b->dn_bytes;
            b->gu.p = pr_malloc(r, b->gu_bytes); b->dn.p = pr_malloc(r, b->dn_bytes);
            if (!b->gu.p || !b->dn.p) return -1;
            cuMemcpyHtoD(b->gu.p, b->h_gu, b->gu_bytes);
            cuMemcpyHtoD(b->dn.p, b->h_dn, b->dn_bytes);
            r->routed_dev += b->gu_bytes + b->dn_bytes;
        }
        return 0;
    }
    if (ring_slots > 4) ring_slots = 4;
    r->ring_n = ring_slots;
    r->ring_slot_bytes = pr_max_bank_bytes(r);
    for (int s = 0; s < ring_slots; s++) {
        r->ring[s].p = pr_malloc(r, r->ring_slot_bytes);
        if (!r->ring[s].p) return -1;
        r->ring[s].bytes = r->ring_slot_bytes;
        r->ring[s].seq = -1; r->ring[s].layer = -1; r->ring[s].grp = 0; r->ring[s].done_valid = 0;
        cuEventCreate(&r->ring[s].ready, CU_EVENT_DISABLE_TIMING);
        cuEventCreate(&r->ring[s].done, CU_EVENT_DISABLE_TIMING);
        r->routed_dev += r->ring_slot_bytes;
    }
    return 0;
}

/* pinned staging of the banks (one block per layer, gate_up then down) */

/* ---- NVFP4 / FP8 bank layout / CPU packer ----
 * One bank blob per MoE layer = [gate_up data | gate_up scales | down data | down scales | gate_up globals | down globals].
 * Format 0 (NVFP4): data [E][N][K/2] bytes, scales [E][N][K/16] e4m3, one fp32 global per expert matrix.
 * Format 1 (FP8):   data [E][N][K] e4m3 bytes, scales [E][N] fp32 (per output channel, = amax / 448), globals unused. */
typedef struct { size_t gu_q, gu_sf, dn_q, dn_sf, gu_g, dn_g, total; int fgu, fdn; } pr_fp4_lay;
static size_t pr_al256(size_t x) { return (x + 255) & ~(size_t)255; }
static pr_fp4_lay pr_fp4_layout2(int E, int C, int Hm, int fgu, int fdn) {
    pr_fp4_lay l; size_t o = 0;
    l.fgu = fgu; l.fdn = fdn;
    l.gu_q = o;  o = pr_al256(o + (size_t)E * 2 * Hm * (fgu ? C : C / 2));
    l.gu_sf = o; o = pr_al256(o + (size_t)E * 2 * Hm * (fgu ? 4 : C / 16));
    l.dn_q = o;  o = pr_al256(o + (size_t)E * C * (fdn ? Hm : Hm / 2));
    l.dn_sf = o; o = pr_al256(o + (size_t)E * C * (fdn ? 4 : Hm / 16));
    l.gu_g = o;  o = pr_al256(o + (size_t)E * 4);
    l.dn_g = o;  o = pr_al256(o + (size_t)E * 4);
    l.total = o;
    return l;
}
static pr_fp4_lay pr_fp4_layout(int E, int C, int Hm) { return pr_fp4_layout2(E, C, Hm, 0, 0); }
static float pr_e4m3_dec_h(unsigned char b) {
    int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return ((float)m * 0.125f) * 0.015625f;
    return (1.0f + (float)m * 0.125f) * ldexpf(1.0f, e - 7);
}
/* round-to-nearest-even float -> e4m3 for 2^-6 <= x <= 448 (the clamped block scales) */
static unsigned char pr_e4m3_rne_h(float x) {
    if (x >= 448.0f) return 0x7e;
    int p; float f = frexpf(x, &p);            /* x = f * 2^p, f in [0.5, 1) */
    int e = p - 1;
    float m8 = rintf((2.0f * f - 1.0f) * 8.0f);
    int m = (int)m8;
    if (m == 8) { m = 0; e++; }
    if (e > 8) return 0x7e;
    return (unsigned char)(((e + 7) << 3) | m);
}
static unsigned pr_e2m1_h(float q) {
    unsigned s = q < 0.f ? 8u : 0u;
    float a = fabsf(q);
    unsigned c = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.0f ? 6u : 7u;
    return s | c;
}
static float pr_bf16_h(unsigned short h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
/* block-scale search of the packer: PRAXIS_PACK_SCALE=max6 (praxis recipe: block amax maps to the e2m1 maximum 6, default) or mse (try
 * block amax / {6, 5.5, 5, 4.5, 4, 3.5} as the scale, keep the lowest squared error). Only changes the bytes written, not the kernels. */
static int pr_pack_scale_mode(void) {
    static int m = -1;
    if (m < 0) { const char *e = getenv("PRAXIS_PACK_SCALE"); m = e && !strcmp(e, "mse"); }
    return m;
}
/* quantise one 16-block v with scale divisor dv; returns squared error, writes scale byte and the 16 codes */
static float pr_fp4_block(const float *v, float bm, float g, float dv, unsigned char *sbo, unsigned char *codes) {
    static const float lut[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
    float t = (bm * (1.0f / dv)) / g;
    t = fminf(fmaxf(t, 0.015625f), 448.0f);
    unsigned char sb = pr_e4m3_rne_h(t);
    float sd = pr_e4m3_dec_h(sb);
    float rcp = (1.0f / g) / sd, err = 0.f;
    for (int i = 0; i < 16; i++) {
        float q = fminf(fmaxf(v[i] * rcp, -6.0f), 6.0f);
        unsigned c = pr_e2m1_h(q);
        codes[i] = (unsigned char)c;
        float d = ((c & 8) ? -1.f : 1.f) * lut[c & 7] * sd * g - v[i];
        err += d * d;
    }
    *sbo = sb;
    return err;
}
/* W stored [K][N] bf16 (x @ W layout) -> q [N][K/2] bytes, sf [N][K/16] bytes, *glob. OpenMP over column tiles. */
static void praxis_nvfp4_pack_matrix(const unsigned short *W, int K, int N, unsigned char *q, unsigned char *sf, float *glob) {
    float amax = 0.f;
    #pragma omp parallel for reduction(max:amax) schedule(static)
    for (long long i = 0; i < (long long)K * N; i++) { float v = fabsf(pr_bf16_h(W[i])); if (v > amax) amax = v; }
    float g = fmaxf(amax, 1e-12f) * (1.0f / 2688.0f);
    *glob = g;
    int nkb = K / 16;
    int search = pr_pack_scale_mode();
    static const float divs[6] = { 6.0f, 5.5f, 5.0f, 4.5f, 4.0f, 3.5f };
    #pragma omp parallel for schedule(static)
    for (int n0 = 0; n0 < N; n0 += 32) {
        int nn = N - n0 < 32 ? N - n0 : 32;
        for (int kb = 0; kb < nkb; kb++)
            for (int c = 0; c < nn; c++) {
                int n = n0 + c;
                float v[16], bm = 0.f;
                for (int i = 0; i < 16; i++) { v[i] = pr_bf16_h(W[(size_t)(kb * 16 + i) * N + n]); if (fabsf(v[i]) > bm) bm = fabsf(v[i]); }
                unsigned char sb, codes[16], bc[16], bs = 0;
                float be = pr_fp4_block(v, bm, g, 6.0f, &bs, bc);
                if (search) for (int d = 1; d < 6; d++) {
                    float e = pr_fp4_block(v, bm, g, divs[d], &sb, codes);
                    if (e < be) { be = e; bs = sb; memcpy(bc, codes, 16); }
                }
                sf[(size_t)n * nkb + kb] = bs;
                unsigned char *dst = q + (size_t)n * (K / 2) + (size_t)kb * 8;
                for (int j = 0; j < 8; j++) dst[j] = (unsigned char)(bc[2 * j] | (bc[2 * j + 1] << 4));
            }
    }
}
/* round to nearest even to e4m3 (|x| <= 448, with subnormals), returns the byte incl. sign */
static unsigned char pr_e4m3_full_h(float x) {
    unsigned sgn = x < 0.f ? 0x80u : 0u;
    float a = fabsf(x);
    if (a >= 448.0f) return (unsigned char)(sgn | 0x7e);
    if (a < 0.015625f) { int m = (int)rintf(a * 512.0f); return (unsigned char)(sgn | (unsigned)m); }   /* subnormal: m * 2^-9, m == 8 is the min normal */
    return (unsigned char)(sgn | pr_e4m3_rne_h(a));
}
static float pr_e4m3_val_h(unsigned char b) { float v = pr_e4m3_dec_h(b & 0x7f); return (b & 0x80) ? -v : v; }
/* W stored [K][N] bf16 -> q [N][K] e4m3 bytes, rs [N] fp32 per-output-channel scale (amax over K / 448) */
static void praxis_fp8_pack_matrix(const unsigned short *W, int K, int N, unsigned char *q, float *rs) {
    #pragma omp parallel for schedule(static)
    for (int n0 = 0; n0 < N; n0 += 32) {
        int nn = N - n0 < 32 ? N - n0 : 32;
        float am[32];
        for (int c = 0; c < nn; c++) am[c] = 0.f;
        for (int k = 0; k < K; k++) for (int c = 0; c < nn; c++) { float v = fabsf(pr_bf16_h(W[(size_t)k * N + n0 + c])); if (v > am[c]) am[c] = v; }
        for (int c = 0; c < nn; c++) {
            float sc = fmaxf(am[c], 1e-30f) * (1.0f / 448.0f);
            rs[n0 + c] = sc;
            float inv = 1.0f / sc;
            unsigned char *dst = q + (size_t)(n0 + c) * K;
            for (int k = 0; k < K; k++) dst[k] = pr_e4m3_full_h(pr_bf16_h(W[(size_t)k * N + n0 + c]) * inv);
        }
    }
}
/* dequantize one packed matrix back to fp32 [N][K] (reference / tests) */
static void praxis_nvfp4_dequant_matrix(const unsigned char *q, const unsigned char *sf, float glob, int K, int N, float *out) {
    static const float lut[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            unsigned char byte = q[(size_t)n * (K / 2) + k / 2];
            unsigned nib = (k & 1) ? (byte >> 4) : (byte & 15);
            float v = lut[nib & 7] * ((nib & 8) ? -1.f : 1.f);
            out[(size_t)n * K + k] = v * pr_e4m3_dec_h(sf[(size_t)n * (K / 16) + k / 16]) * glob;
        }
}

/* ---- pack cache: PRAXIS_PACK_CACHE=<file>. One header page + one NVFP4 blob per MoE layer (in moe_id order). A fingerprint of
 * the bank tensors (names, sizes, shard file sizes/mtimes, sampled bytes) invalidates a stale cache. The magic is written last. ---- */
#define PR_PC_MAGIC "PRXFP4C1"
typedef struct { char magic[8]; unsigned long long fp, layer_bytes; int E, C, Hm, n_moe; } pr_pc_hdr;
static unsigned long long pr_bank_fingerprint(praxis_runner *r) {
    unsigned long long h = 1469598103934665603ULL;
    #define MIX(v) do { unsigned long long v_ = (unsigned long long)(v); for (int q_ = 0; q_ < 8; q_++) { h ^= (v_ >> (8 * q_)) & 255; h *= 1099511628211ULL; } } while (0)
    for (int k = 0; k < r->n_shards; k++) {
        struct stat sb; if (stat(r->shard_path[k], &sb) == 0) { MIX(sb.st_size); MIX(sb.st_mtime); }
    }
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
        pr_blk *b = &ts[t]->b[i];
        if (!b->moe) continue;
        MIX(b->gu_bytes); MIX(b->dn_bytes);
        const unsigned char *src[2] = { (const unsigned char *)b->h_gu, (const unsigned char *)b->h_dn };
        size_t nb[2] = { b->gu_bytes, b->dn_bytes };
        for (int m = 0; m < 2; m++) {
            for (int smp = 0; smp < 16; smp++) {
                size_t off = nb[m] > 4096 ? (size_t)((nb[m] - 4096) / 15) * smp : 0;
                size_t len = nb[m] < 4096 ? nb[m] : 4096;
                for (size_t q = 0; q < len; q += 8) { unsigned long long v = 0; memcpy(&v, src[m] + off + q, len - q < 8 ? len - q : 8); MIX(v); }
            }
            pr_drop_pages(src[m], nb[m]);
        }
    }
    #undef MIX
    return h;
}
static size_t pr_rss_kb(const char *key) {   /* VmRSS / VmHWM in kB */
    FILE *f = fopen("/proc/self/status", "r"); if (!f) return 0;
    char line[256]; size_t v = 0; size_t kl = strlen(key);
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, kl)) { v = (size_t)atoll(line + kl + 1); break; }
    fclose(f); return v;
}

/* convert the bf16 banks (file-mapped or pinned) of every MoE layer into pinned NVFP4 banks, layer by layer: only one layer of
 * bf16 pages is touched at a time (dropped from the mapping right after packing), so host residency is the packed set plus one layer. */
/* PRAXIS_SLOT_MB=N (may be fractional): a layer whose bank blob is larger than N MB is streamed as 2/4/8 groups of experts, one group per ring slot (the ring slot
 * is sized for the largest group, so a layer with FP8 banks no longer sets the device peak). Needs the fast path; a layer with several token chunks streams its groups once per chunk. */
static int pr_plan_groups(praxis_runner *r) {
    const char *e = getenv("PRAXIS_SLOT_MB");
    int E = r->c.n_experts, C = r->c.dim, Hm = r->c.moe_hidden;
    size_t cap = 0;
    if (e && *e && atof(e) > 0) cap = (size_t)(atof(e) * 1048576.0);
    else if (r->fmt_mixed) cap = pr_fp4_layout2(E, C, Hm, 0, 1).total;   /* mixed: the slot is one NVFP4-gate_up + FP8-down layer, larger layers stream as expert groups */
    if (!cap) return 0;
    if (!e || !*e) e = "auto";
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
        pr_blk *b = &ts[t]->b[i];
        if (!b->moe) continue;
        size_t tot = pr_fp4_layout2(E, C, Hm, b->fgu, b->fdn).total;
        for (int g = 2; g <= 8 && tot > cap; g *= 2) if (E % g == 0 && pr_fp4_layout2(E / g, C, Hm, b->fgu, b->fdn).total <= cap) { b->ngrp = g; break; }
        if (tot > cap && b->ngrp == 1) { fprintf(stderr, "praxis: layer bank %.0f MB does not fit the slot cap (%s)\n", tot / 1048576.0, e); return -1; }
    }
    return 0;
}

/* per-matrix bank format plan: PRAXIS_FP8_BANKS = comma list of "enc3" (both banks of encoder block 3), "enc3.dn", "dec1.gu", or the wildcards
 * "dn" / "gu" / "all" (every MoE layer); a leading "-" reverts (e.g. "dn,enc13.gu,-enc13.dn"). Matrices not listed stay NVFP4. Also fills the per-layer defaults (0). */
static int pr_plan_parse(praxis_runner *r) {
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) ts[t]->b[i].fgu = ts[t]->b[i].fdn = 0;
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) ts[t]->b[i].ngrp = 1;
    const char *e = getenv("PRAXIS_FP8_BANKS");
    if ((!e || !*e) && r->fmt_mixed) e = "dn,enc3.gu,enc10.gu,enc13.gu";   /* measured on the real checkpoint: the FP4 gate_up banks of these layers dominate the error */
    if (!e || !*e) return pr_plan_groups(r);
    char buf[1024]; snprintf(buf, sizeof buf, "%s", e);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int mgu = 1, mdn = 1, tw = -1, bi = -1, val = 1;
        if (*tok == '-') { val = 0; tok++; }   /* "-enc13.dn": back to NVFP4 */
        char *dot = strchr(tok, '.');
        if (dot) { *dot = 0; if (!strcmp(dot + 1, "gu")) mdn = 0; else if (!strcmp(dot + 1, "dn")) mgu = 0; }
        if (!strcmp(tok, "all")) { }
        else if (!strcmp(tok, "dn")) { mgu = 0; }
        else if (!strcmp(tok, "gu")) { mdn = 0; }
        else if (!strncmp(tok, "enc", 3)) { tw = 0; bi = atoi(tok + 3); }
        else if (!strncmp(tok, "dec", 3)) { tw = 1; bi = atoi(tok + 3); }
        else { fprintf(stderr, "praxis: bad PRAXIS_FP8_BANKS token '%s'\n", tok); return -1; }
        for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
            pr_blk *b = &ts[t]->b[i];
            if (!b->moe) continue;
            if (tw >= 0 && (t != tw || i != bi)) continue;
            if (mgu) b->fgu = val;
            if (mdn) b->fdn = val;
        }
    }
    return pr_plan_groups(r);
}

/* convert the bf16 banks (file-mapped or pinned) of every MoE layer into pinned NVFP4 (and FP8, per plan) banks, layer by layer: only one layer of
 * bf16 pages is touched at a time (dropped from the mapping right after packing), so host residency is the packed set plus one layer. */
static int pr_nvfp4_convert(praxis_runner *r) {
    const pr_cfg *c = &r->c;
    int E = c->n_experts, C = c->dim, Hm = c->moe_hidden;
    if (C % 128 || Hm % 128) { fprintf(stderr, "praxis: NVFP4 needs dim and expert_hidden multiples of 128 (got %d, %d)\n", C, Hm); return -1; }
    if (pr_plan_parse(r)) return -1;
    pr_tower *ts[2] = { &r->enc, &r->dec };
    r->routed_total = 0;
    double t0 = pr_now();
    /* layer blob offsets (moe_id order) and the plan fingerprint (0 = the plain NVFP4 default, the cache format of before the plan existed) */
    int nm = r->n_moe_total;
    size_t *loff = (size_t *)calloc((size_t)nm + 1, sizeof(size_t));
    unsigned long long plan_fp = pr_pack_scale_mode() ? 0x5ca1e0001ULL : 0;
    size_t first_total = 0; int uniform = 1;
    for (int id = 0; id < nm; id++) {
        pr_blk *b = pr_moe_by_id(r, id);
        pr_fp4_lay Li = pr_fp4_layout2(E / b->ngrp, C, Hm, b->fgu, b->fdn);
        size_t lt = Li.total * (size_t)b->ngrp;
        loff[id + 1] = loff[id] + lt;
        if (id == 0) first_total = lt; else if (lt != first_total) uniform = 0;
        if (b->fgu || b->fdn || b->ngrp > 1) plan_fp = plan_fp * 1099511628211ULL + (unsigned long long)(id * 64 + b->ngrp * 4 + b->fgu * 1 + b->fdn * 2 + 1);
    }
    const char *cpath = getenv("PRAXIS_PACK_CACHE");
    char cpath_plan[1200];   /* a non-default bank plan (FP8 banks, expert groups, scale search) gets its own cache file: the default cache is never overwritten */
    if (cpath && *cpath && plan_fp) { snprintf(cpath_plan, sizeof cpath_plan, "%s.p%llx", cpath, plan_fp); cpath = cpath_plan; }
    int cfd = -1, cache_valid = 0, cache_write = 0;
    pr_pc_hdr ch; unsigned long long fp = 0;
    const size_t HDR = 4096, PLAN_OFF = 512;
    if (cpath && *cpath && !r->synth) {
        fp = pr_bank_fingerprint(r);
        cfd = open(cpath, O_RDWR);
        unsigned long long cpf = 0;
        if (cfd >= 0 && pread(cfd, &ch, sizeof ch, 0) == (ssize_t)sizeof ch && pread(cfd, &cpf, 8, PLAN_OFF) == 8 && !memcmp(ch.magic, PR_PC_MAGIC, 8) && ch.fp == fp &&
            ch.layer_bytes == first_total && ch.E == E && ch.C == C && ch.Hm == Hm && ch.n_moe == nm && cpf == plan_fp) cache_valid = 1;
        if (!cache_valid) {
            if (cfd >= 0) close(cfd);
            cfd = open(cpath, O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (cfd < 0) { fprintf(stderr, "praxis: cannot create pack cache %s (continuing without)\n", cpath); }
            else {
                memset(&ch, 0, sizeof ch); memcpy(ch.magic, "PRXFP4C0", 8); ch.fp = fp; ch.layer_bytes = first_total; ch.E = E; ch.C = C; ch.Hm = Hm; ch.n_moe = nm;
                if (pwrite(cfd, &ch, sizeof ch, 0) != (ssize_t)sizeof ch || pwrite(cfd, &plan_fp, 8, PLAN_OFF) != 8) { close(cfd); cfd = -1; } else cache_write = 1;
            }
        }
    }
    (void)uniform;
    size_t max_total = 0;
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
        pr_blk *b = &ts[t]->b[i];
        if (!b->moe) continue;
        int G = b->ngrp, Eg = E / G;
        pr_fp4_lay L = pr_fp4_layout2(Eg, C, Hm, b->fgu, b->fdn);
        size_t LT = L.total * (size_t)G;
        if (G == 1 ? LT > max_total : L.total > max_total) max_total = G == 1 ? LT : L.total;
        unsigned char *buf = NULL;
        if (cuMemHostAlloc((void **)&buf, LT, 0) != CUDA_SUCCESS) { fprintf(stderr, "praxis: pinned NVFP4 bank alloc failed\n"); return -1; }
        off_t boff = (off_t)(HDR + loff[b->moe_id]);
        if (cache_valid) {
            /* 8 parallel preads of the layer blob (one pread is ~0.4 GB/s on this disk, parallel ones saturate it) */
            const int NR = 8; int bad = 0;
            size_t part = ((LT / NR) + 4095) & ~(size_t)4095;
            #pragma omp parallel for num_threads(NR) schedule(static, 1)
            for (int k = 0; k < NR; k++) {
                size_t lo = (size_t)k * part, hi = lo + part > LT ? LT : lo + part, done = lo;
                while (done < hi) {
                    ssize_t n = pread(cfd, buf + done, hi - done, boff + done);
                    if (n <= 0) { bad = 1; break; }
                    done += (size_t)n;
                }
            }
            if (bad) { fprintf(stderr, "praxis: pack cache read failed\n"); return -1; }
            r->pack_cache_hits++;
        } else if (!r->synth) {
            const unsigned short *gu = (const unsigned short *)b->h_gu, *dn = (const unsigned short *)b->h_dn;
            for (int g = 0; g < G; g++) for (int el = 0; el < Eg; el++) {
                int e = g * Eg + el; unsigned char *gb = buf + (size_t)g * L.total;
                if (b->fgu) praxis_fp8_pack_matrix(gu + (size_t)e * C * 2 * Hm, C, 2 * Hm, gb + L.gu_q + (size_t)el * 2 * Hm * C, (float *)(gb + L.gu_sf) + (size_t)el * 2 * Hm);
                else praxis_nvfp4_pack_matrix(gu + (size_t)e * C * 2 * Hm, C, 2 * Hm, gb + L.gu_q + (size_t)el * 2 * Hm * (C / 2),
                                              gb + L.gu_sf + (size_t)el * 2 * Hm * (C / 16), (float *)(gb + L.gu_g) + el);
                if (b->fdn) praxis_fp8_pack_matrix(dn + (size_t)e * Hm * C, Hm, C, gb + L.dn_q + (size_t)el * C * Hm, (float *)(gb + L.dn_sf) + (size_t)el * C);
                else praxis_nvfp4_pack_matrix(dn + (size_t)e * Hm * C, Hm, C, gb + L.dn_q + (size_t)el * C * (Hm / 2),
                                              gb + L.dn_sf + (size_t)el * C * (Hm / 16), (float *)(gb + L.dn_g) + el);
            }
            if (!r->pinned_bf16) { pr_drop_pages(b->h_gu, b->gu_bytes); pr_drop_pages(b->h_dn, b->dn_bytes); }   /* file-backed shard pages */
            if (cache_write) {
                size_t done = 0;
                while (done < LT) {
                    ssize_t n = pwrite(cfd, buf + done, LT - done, boff + done);
                    if (n <= 0) { fprintf(stderr, "praxis: pack cache write failed (disabling)\n"); cache_write = 0; break; }
                    done += (size_t)n;
                }
            }
        } else {   /* random valid packed bytes: any nibble is a valid e2m1 code, scales in e4m3 [0x30, 0x48) ~ [0.5, 2) */
            #pragma omp parallel
            {
                unsigned long long st = 0x9e3779b97f4a7c15ULL * (unsigned long long)(b->moe_id * 131 + 7 + 7919 * omp_get_thread_num()) | 1;
                #pragma omp for schedule(static)
                for (long long k = 0; k < (long long)LT; k++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; buf[k] = (unsigned char)(st >> 32); }
                for (int g = 0; g < G; g++) {
                    unsigned char *gb = buf + (size_t)g * L.total;
                    if (!b->fgu) {
                        #pragma omp for schedule(static)
                        for (long long k = 0; k < (long long)Eg * 2 * Hm * (C / 16); k++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; gb[L.gu_sf + k] = (unsigned char)(0x30 + ((st >> 33) % 24)); }
                    }
                    if (!b->fdn) {
                        #pragma omp for schedule(static)
                        for (long long k = 0; k < (long long)Eg * C * (Hm / 16); k++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; gb[L.dn_sf + k] = (unsigned char)(0x30 + ((st >> 33) % 24)); }
                    }
                }
            }
            for (int g = 0; g < G; g++) {
                unsigned char *gb = buf + (size_t)g * L.total;
                for (int e = 0; e < Eg; e++) { ((float *)(gb + L.gu_g))[e] = 0.01f; ((float *)(gb + L.dn_g))[e] = 0.01f; }
                if (b->fgu) for (long long k = 0; k < (long long)Eg * 2 * Hm; k++) ((float *)(gb + L.gu_sf))[k] = 0.002f;
                if (b->fdn) for (long long k = 0; k < (long long)Eg * C; k++) ((float *)(gb + L.dn_sf))[k] = 0.002f;
            }
        }
        if (r->pinned_bf16) cuMemFreeHost(b->h_gu);
        b->h_gu = buf; b->h_dn = buf + LT;
        b->gu_bytes = LT; b->dn_bytes = 0; b->bank_dt = 3; b->grp_bytes = L.total;
        r->routed_total += LT;
        b->gu.p = 0; b->dn.p = 0;
        size_t hwm = pr_rss_kb("VmRSS:"); if (hwm > r->rss_peak_load) r->rss_peak_load = hwm;
    }
    if (cache_write) {
        memcpy(ch.magic, PR_PC_MAGIC, 8);
        if (pwrite(cfd, &ch, sizeof ch, 0) == (ssize_t)sizeof ch) { r->pack_cache_written = 1; fsync(cfd); }
    }
    if (cfd >= 0) close(cfd);
    free(loff);
    r->synth_npool = 0;
    r->nvfp4 = 1;
    if (r->verbose) fprintf(stderr, "praxis: NVFP4 banks ready (%.1f s, %.2f GB, max %.1f MB/layer%s)\n", pr_now() - t0, r->routed_total / 1073741824.0, max_total / 1048576.0,
                            cache_valid ? ", from pack cache" : cache_write || r->pack_cache_written ? ", pack cache written" : "");
    return 0;
}

static int pr_pin_synth_banks(praxis_runner *r) {
    pr_tower *ts[2] = { &r->enc, &r->dec };
    size_t bytes = pr_max_bank_bytes(r);
    int np = r->synth_banks > 0 ? r->synth_banks : 4;
    if (np > 8) np = 8;
    if (np > r->n_moe_total) np = r->n_moe_total;
    for (int j = 0; j < np; j++) {
        void *p = NULL;
        if (cuMemHostAlloc(&p, bytes, 0) != CUDA_SUCCESS) { fprintf(stderr, "praxis: pinned synthetic bank alloc (%zu B) failed\n", bytes); return -1; }
        size_t n16 = bytes / 2;
        unsigned short *u = (unsigned short *)p;
        #pragma omp parallel
        {
            unsigned long long st = 0x9e3779b97f4a7c15ULL * (unsigned long long)(j * 977 + 13 + 7919 * omp_get_thread_num()) | 1;
            #pragma omp for schedule(static)
            for (long long k = 0; k < (long long)n16; k++) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                float f = 0.03f * ((float)(st >> 40) / 8388608.0f - 1.0f);
                uint32_t bits; memcpy(&bits, &f, 4);
                u[k] = (unsigned short)(bits >> 16);
            }
        }
        r->synth_pool[j] = p;
    }
    r->synth_npool = np; r->synth_pool_bytes = bytes;
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
        pr_blk *b = &ts[t]->b[i];
        if (!b->moe) continue;
        char *p = (char *)r->synth_pool[b->moe_id % np];
        b->h_gu = p; b->h_dn = p + b->gu_bytes;
        b->gu.p = 0; b->dn.p = 0;
    }
    return 0;
}
static int pr_pin_banks(praxis_runner *r) {
    if (r->synth) return pr_pin_synth_banks(r);
    { const char *e = getenv("PRAXIS_BF16_MMAP");   /* parity/debug mode: stream bf16 banks from the mmap'd shards (no 51 GB pinned copy), pages dropped after each copy */
      if (e && *e && *e != '0') { r->mmap_bf16 = 1; return 0; } }
    pr_tower *ts[2] = { &r->enc, &r->dec };
    for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) {
        pr_blk *b = &ts[t]->b[i];
        if (!b->moe) continue;
        void *p = NULL;
        if (cuMemHostAlloc(&p, b->gu_bytes + b->dn_bytes, 0) != CUDA_SUCCESS) { fprintf(stderr, "praxis: pinned alloc failed\n"); return -1; }
        memcpy(p, b->h_gu, b->gu_bytes);
        memcpy((char *)p + b->gu_bytes, b->h_dn, b->dn_bytes);
        pr_drop_pages(b->h_gu, b->gu_bytes); pr_drop_pages(b->h_dn, b->dn_bytes);   /* one layer of file pages resident at a time */
        r->pinned_bf16 = 1;
        b->h_gu = p; b->h_dn = (char *)p + b->gu_bytes;
        b->gu.p = 0; b->dn.p = 0;
    }
    return 0;
}

static int pr_slot_load(praxis_runner *r, pr_slot *s, long long seq, int layer, int grp) {
    pr_blk *b = pr_moe_by_id(r, layer);
    if (s->done_valid) cuStreamWaitEvent(r->copy_stream, s->done, 0);
    size_t off = b->ngrp > 1 ? (size_t)grp * b->grp_bytes : 0, len = b->ngrp > 1 ? b->grp_bytes : b->gu_bytes + b->dn_bytes;
    if (r->mmap_bf16) {
        if (!r->skip_h2d) {
            PRC(cuMemcpyHtoDAsync(s->p, b->h_gu, b->gu_bytes, r->copy_stream));
            PRC(cuMemcpyHtoDAsync(s->p + b->gu_bytes, b->h_dn, b->dn_bytes, r->copy_stream));
            cuStreamSynchronize(r->copy_stream);
            pr_drop_pages(b->h_gu, b->gu_bytes); pr_drop_pages(b->h_dn, b->dn_bytes);
        }
    } else
    if (!r->skip_h2d) PRC(cuMemcpyHtoDAsync(s->p, (char *)b->h_gu + off, len, r->copy_stream));
    cuEventRecord(s->ready, r->copy_stream);
    s->seq = seq; s->layer = layer; s->grp = grp;
    r->h2d_bytes += len;
    r->n_loads++;
    return 0;
}
static int pr_moe_prefetch2(praxis_runner *r, int layer, int grp) {
    if (r->ring_n < 2 || r->n_moe_total <= 0) return 0;   /* prefetch needs a free slot: ring >= 2 */
    long long seq = r->seq + 1;
    pr_slot *s = &r->ring[seq % r->ring_n];
    if (s->seq == seq && s->layer == layer && s->grp == grp) return 0;
    return pr_slot_load(r, s, seq, layer, grp);
}
static int pr_moe_prefetch(praxis_runner *r, int layer) { return pr_moe_prefetch2(r, layer, 0); }
/* device pointers of the banks of a layer (or of expert group grp of it) for this use; stream mode waits on the copy event */
static int pr_moe_acquire2(praxis_runner *r, pr_blk *b, int grp, pr_mat *gu, pr_mat *dn, pr_slot **used) {
    *used = NULL;
    if (r->ring_n <= 0) { *gu = b->gu; *dn = b->dn; if (b->ngrp > 1) gu->p += (CUdeviceptr)((size_t)grp * b->grp_bytes); return 0; }
    pr_slot *s = &r->ring[r->seq % r->ring_n];
    if (s->seq == r->seq && s->layer == b->moe_id && s->grp == grp) r->n_hits++;
    else if (pr_slot_load(r, s, r->seq, b->moe_id, grp)) return -1;
    cuStreamWaitEvent(0, s->ready, 0);
    gu->p = s->p; gu->dt = b->bank_dt; gu->bytes = b->ngrp > 1 ? b->grp_bytes : b->gu_bytes;
    dn->p = s->p + b->gu_bytes; dn->dt = b->bank_dt; dn->bytes = b->dn_bytes;
    *used = s;
    return 0;
}
static int pr_moe_acquire(praxis_runner *r, pr_blk *b, pr_mat *gu, pr_mat *dn, pr_slot **used) { return pr_moe_acquire2(r, b, 0, gu, dn, used); }

/* ---- rope tables (host, float64 like RotaryEmbedding) ---- */
static void pr_rope_table(const long long *pos, int n, int dim, int eff, double base, long long tab, float *cs, float *sn) {
    int half = dim / 2;
    float theta[256] = { 0 };
    for (int j = 0; j < eff / 2 && j < 256; j++) {
        float e = (float)(2 * j) / (float)eff;
        theta[j] = 1.0f / powf((float)base, e);
    }
    for (int i = 0; i < n; i++) {
        long long hi = pos[i] / tab, lo = pos[i] % tab;
        double hp = (double)(float)((float)hi * (float)tab), lp = (double)(float)lo;
        for (int j = 0; j < half; j++) {
            double th = (double)theta[j];
            double ch = (double)(float)cos(hp * th), sh = (double)(float)sin(hp * th);
            double cl = (double)(float)cos(lp * th), sl = (double)(float)sin(lp * th);
            cs[(size_t)i * half + j] = (float)(ch * cl - sh * sl);
            sn[(size_t)i * half + j] = (float)(sh * cl + ch * sl);
        }
    }
}
static int pr_up_rope(praxis_runner *r, const long long *pos, int n, int dim, int eff, double base, long long tab, int slot_c, int slot_s) {
    int half = dim / 2;
    float *c = (float *)malloc(F4((size_t)n * half)), *s = (float *)malloc(F4((size_t)n * half));
    pr_rope_table(pos, n, dim, eff, base, tab, c, s);
    CUdeviceptr dc = pr_ws(r, slot_c, (size_t)n * half), ds = pr_ws(r, slot_s, (size_t)n * half);
    if (!dc || !ds) { free(c); free(s); return -1; }
    cuMemcpyHtoD(dc, c, F4((size_t)n * half)); cuMemcpyHtoD(ds, s, F4((size_t)n * half));
    free(c); free(s);
    return 0;
}

/* ---- lifecycle ---- */
static praxis_runner *pr_load_impl(const char *dir, int device, int verbose, int synth, int nbanks);
praxis_runner *praxis_load(const char *dir, int device, int verbose) { return pr_load_impl(dir, device, verbose, 0, 0); }
praxis_runner *praxis_load_synthetic(const char *dir, int device, int verbose, int nbanks) { return pr_load_impl(dir, device, verbose, 1, nbanks); }
static praxis_runner *pr_load_impl(const char *dir, int device, int verbose, int synth, int nbanks) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) { fprintf(stderr, "praxis: cuewInit failed\n"); return NULL; }
    CU_CHECK_NULL(cuInit(0));
    praxis_runner *r = (praxis_runner *)calloc(1, sizeof *r);
    r->verbose = verbose; r->synth = synth; r->synth_banks = nbanks; r->esp = &r->es_store[0];
    { const char *e = getenv("PRAXIS_EXACT"); r->fast = !(e && *e && *e != '0'); }
    { const char *e = getenv("PRAXIS_MOE_CHUNK"); r->moe_chunk = e ? atoi(e) : 4096; if (r->moe_chunk < 64) r->moe_chunk = 4096; }
    { const char *e = getenv("PRAXIS_PROFILE"); r->prof = e && *e && *e != '0'; }
    { const char *e = getenv("PRAXIS_AE_DENSE"); r->ae_flash = !(e && *e && *e != '0'); }
    { const char *e = getenv("PRAXIS_CTX_CACHE"); r->ctx_cache = e && *e && *e != '0'; }
    { const char *e = getenv("PRAXIS_SKIP_H2D"); r->skip_h2d = e && *e && *e != '0'; }
    if (cublasewInit() != 0 || cublasewCreate(&r->cb, 0) != 0) { fprintf(stderr, "praxis: cuBLAS unavailable, using the exact fp32 path\n"); r->fast = 0; }
    CU_CHECK_NULL(cuDeviceGet(&r->dev, device));
    CU_CHECK_NULL(cuDevicePrimaryCtxRetain(&r->ctx, r->dev));
    CU_CHECK_NULL(cuCtxSetCurrent(r->ctx));
    { size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot); r->base_free = r->min_free = fr; }
    char path[1100];
    snprintf(path, sizeof path, "%s/config.json", dir);
    if (pr_parse_config(path, &r->c)) { free(r); return NULL; }
    if (cu_compile_kernels_ex(&r->mod, r->dev, praxis_kernel_src, "praxis.cu", verbose, "praxis", 0) < 0) { free(r); return NULL; }
    struct { CUfunction *f; const char *n; } fns[] = {
        { &r->f_cvt, "pr_cvt_bf16" }, { &r->f_rmsb, "pr_rmsnorm_bf" }, { &r->f_qkprep, "pr_qkprep" }, { &r->f_bias, "pr_bias" },
        { &r->f_swib, "pr_swiglu_bf" }, { &r->f_ctiles, "pr_code_tiles" }, { &r->f_flash[0], "pr_flash16" }, { &r->f_flash[1], "pr_flash32" },
        { &r->f_flash[2], "pr_flash64" }, { &r->f_flash[3], "pr_flash128" }, { &r->f_mcount, "pr_moe_count" }, { &r->f_mscan, "pr_moe_scan" },
        { &r->f_mfill, "pr_moe_fill" }, { &r->f_gather, "pr_gather_bf" }, { &r->f_combb, "pr_combine_bf" }, { &r->f_fillrand, "pr_fill_rand_bf16" },
        { &r->f_flashiv[0], "pr_flashiv16" }, { &r->f_flashiv[1], "pr_flashiv32" }, { &r->f_flashiv[2], "pr_flashiv64" }, { &r->f_flashiv[3], "pr_flashiv128" },
        { &r->f_gemm, "pr_gemm" }, { &r->f_rms, "pr_rmsnorm" }, { &r->f_adaln, "pr_adaln" }, { &r->f_ln, "pr_layernorm" },
        { &r->f_modulate, "pr_modulate" }, { &r->f_fourier, "pr_fourier" }, { &r->f_sincos, "pr_sincos" },
        { &r->f_swiglu, "pr_swiglu" }, { &r->f_act, "pr_act" }, { &r->f_add, "pr_add" }, { &r->f_addrow, "pr_add_row" },
        { &r->f_future, "pr_future" }, { &r->f_copy, "pr_copy_rows" }, { &r->f_chw, "pr_chw_to_tok" },
        { &r->f_unpatch, "pr_unpatch_tok" }, { &r->f_patch, "pr_patch_out" }, { &r->f_rope, "pr_rope" },
        { &r->f_attn, "pr_attn" }, { &r->f_topk, "pr_topk" }, { &r->f_combine, "pr_combine" } };
    for (size_t i = 0; i < sizeof fns / sizeof fns[0]; i++)
        if (cuModuleGetFunction(fns[i].f, r->mod, fns[i].n) != CUDA_SUCCESS) { fprintf(stderr, "praxis: missing kernel %s\n", fns[i].n); free(r); return NULL; }
    if (!synth) {
        if (pr_st_open(r, dir)) { fprintf(stderr, "praxis: cannot open weights in %s\n", dir); free(r); return NULL; }
    }
    { int hd = r->c.head_dim; if (!(hd == 16 || hd == 32 || hd == 64 || hd == 128)) r->fast = 0; }
    r->hseg = (int *)malloc(4 * (r->c.n_experts + 2));
    int moe_ctr = 0;
    if (pr_ld_tower(r, &r->enc, "backbone.encoder.", 1, &moe_ctr) || pr_ld_tower(r, &r->dec, "backbone.decoder.", 0, &moe_ctr) || pr_ld_ae(r)) {
        fprintf(stderr, "praxis: weight load failed\n"); free(r); return NULL; }
    r->n_moe_total = moe_ctr;
    if (r->enc.b[0].q.dt != 1 || r->dec.b[0].q.dt != 1 || r->enc.b[0].out.dt != 1) r->fast = 0;
    r->zero_row = pr_malloc(r, F4(r->c.dec_mod_dim > r->c.dim ? r->c.dec_mod_dim : r->c.dim));
    cuMemsetD32(r->zero_row, 0, r->c.dec_mod_dim > r->c.dim ? r->c.dec_mod_dim : r->c.dim);
    cuStreamCreate(&r->copy_stream, CU_STREAM_NON_BLOCKING);
    {
        const char *e4 = getenv("PRAXIS_NVFP4"), *efm = getenv("PRAXIS_EXPERT_FMT");
        int fmt_on = efm && (!strcmp(efm, "nvfp4") || !strcmp(efm, "nvfp4a16") || !strcmp(efm, "w4a16") || !strcmp(efm, "mixed"));
        if ((e4 && *e4 && *e4 != '0') || fmt_on) {
            if (cu_compile_kernels_ex(&r->mod4, r->dev, praxis_fp4_kernel_src, "praxis_fp4.cu", verbose, "praxis", CU_COMPILE_ARCH_A) < 0) { fprintf(stderr, "praxis: NVFP4 kernels need sm_120a\n"); free(r); return NULL; }
            struct { CUfunction *f; const char *n; } f4[] = { { &r->f4_amax_f32, "pr4_amax_f32" }, { &r->f4_amax_bf16, "pr4_amax_bf16" },
                { &r->f4_quant_f32, "pr4_quant_f32" }, { &r->f4_quant_bf16, "pr4_quant_bf16" }, { &r->f4_tiles, "pr4_tiles" }, { &r->f4_gemm, "pr4_gemm_grouped" }, { &r->f4_w16, "pr4_gemm_w4a16" }, { &r->f8_w16, "pr8_gemm_w8a16" }, { &r->f4_cvt, "pr4_f32_to_f16" }, { &r->f4_swi16, "pr4_swiglu_f16" }, { &r->f4_b2h, "pr4_bf16_to_f16" } };
            for (size_t i = 0; i < sizeof f4 / sizeof f4[0]; i++)
                if (cuModuleGetFunction(f4[i].f, r->mod4, f4[i].n) != CUDA_SUCCESS) { fprintf(stderr, "praxis: missing kernel %s\n", f4[i].n); free(r); return NULL; }
            { const char *ef = getenv("PRAXIS_EXPERT_FMT"); r->fmt_mixed = ef && !strcmp(ef, "mixed");
              r->fp4_a16 = ef && (!strcmp(ef, "nvfp4a16") || !strcmp(ef, "w4a16") || r->fmt_mixed); }
            if (pr_nvfp4_convert(r)) { free(r); return NULL; }
        } else if (pr_pin_banks(r)) { free(r); return NULL; }
    }
    if (!synth) {
        /* resident banks when they fit comfortably (< 6 GB routed), else a 2-slot streaming ring (PRAXIS_STREAM overrides; 0 = resident) */
        const char *es = getenv("PRAXIS_STREAM");
        int ring = es && *es ? atoi(es) : (r->routed_total > ((size_t)6 << 30) ? 2 : 0);
        if (praxis_set_stream(r, ring)) { free(r); return NULL; }
    }
    if (verbose) fprintf(stderr, "praxis: loaded dim=%d enc %d(+%d dense) dec %d, %d MoE layers, %.2f MB weights, %.2f MB routed\n",
                         r->c.dim, r->c.enc_depth, r->c.enc_dense, r->c.dec_depth, r->n_moe_total, r->w_bytes / 1048576.0, r->routed_total / 1048576.0);
    return r;
}

void praxis_set_dump_dir(praxis_runner *r, const char *dir) {
    if (dir) { snprintf(r->dump_dir, sizeof r->dump_dir, "%s", dir); char b[1100]; snprintf(b, sizeof b, "%s", dir);
        for (char *p = b + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(b, 0755); *p = '/'; } mkdir(b, 0755); }
    else r->dump_dir[0] = 0;
}
void praxis_set_sample_dump_dir(praxis_runner *r, const char *dir) {
    if (dir) { snprintf(r->smp_dir, sizeof r->smp_dir, "%s", dir); char b[1100]; snprintf(b, sizeof b, "%s", dir);
        for (char *p = b + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(b, 0755); *p = '/'; } mkdir(b, 0755); }
    else r->smp_dir[0] = 0;
}
static void pr_smp_dump(praxis_runner *r, const char *name, int n, const float *d, int nd, const long long *sh) {
    if (!r->smp_dir[0]) return;
    char p[1300];
    snprintf(p, sizeof p, "%s/smp_%s.%d.npy", r->smp_dir, name, n);
    pr_npy_write(p, d, nd, sh);
}
void praxis_stats_get(const praxis_runner *r, praxis_stats *s) {
    memset(s, 0, sizeof *s);
    s->dev_bytes_now = r->cur_bytes; s->dev_bytes_peak = r->peak_bytes; s->weights_bytes = r->w_bytes;
    s->routed_dev_bytes = r->routed_dev; s->routed_total_bytes = r->routed_total; s->ring_slot_bytes = r->ring_slot_bytes;
    s->ring_slots = r->ring_n; s->h2d_bytes = r->h2d_bytes; s->n_loads = r->n_loads; s->n_prefetch_hits = r->n_hits;
    s->n_moe_layers_run = r->n_moe_run; s->ws_bytes = r->ws_bytes;
    s->free_min_bytes_used = r->base_free > r->min_free ? r->base_free - r->min_free : 0;
    s->host_rss_peak_load_kb = r->rss_peak_load; s->pack_cache_hits = r->pack_cache_hits; s->pack_cache_written = r->pack_cache_written; s->n_shards = r->n_shards; s->n_moe_layers_total = r->n_moe_total;
}
void praxis_free(praxis_runner *r) {
    if (!r) return;
    cuCtxSetCurrent(r->ctx);
    cuCtxSynchronize();
    cu_buf_slots_free(r->ws, WS_N);
    free(r->es_store[0].pos); free(r->es_store[1].pos);
    for (int s = 0; s < 4; s++) {
        if (r->ring[s].ready) cuEventDestroy(r->ring[s].ready);
        if (r->ring[s].done) cuEventDestroy(r->ring[s].done);
    }
    /* pinned host banks: per-layer buffers (NVFP4 / bf16-pinned) or the synthetic pool */
    if (r->nvfp4 || r->pinned_bf16) {
        pr_tower *ts[2] = { &r->enc, &r->dec };
        for (int t = 0; t < 2; t++) for (int i = 0; i < ts[t]->depth; i++) if (ts[t]->b[i].moe && ts[t]->b[i].h_gu && !r->mmap_bf16) cuMemFreeHost(ts[t]->b[i].h_gu);
    }
    for (int j = 0; j < r->synth_npool; j++) if (r->synth_pool[j]) cuMemFreeHost(r->synth_pool[j]);
    for (int i = 0; i < r->n_alloc; i++) cuMemFree(r->allocs[i]);   /* weights, ring slots, persistent encoder state, context K/V ... */
    free(r->allocs);
    if (r->copy_stream) cuStreamDestroy(r->copy_stream);
    if (r->mod) cuModuleUnload(r->mod);
    if (r->mod4) cuModuleUnload(r->mod4);
    for (int k = 0; k < r->n_shards; k++) { safetensors_close(r->shard[k]); free(r->shard_path[k]); }
    if (r->cb) cublasewDestroy(r->cb);
    free(r->hseg);
    free(r);
}
void praxis_set_fast(praxis_runner *r, int fast) { if (!fast) r->fast = 0; else if (r->cb) { int hd = r->c.head_dim; r->fast = (hd == 16 || hd == 32 || hd == 64 || hd == 128); } }
void praxis_prof_reset(praxis_runner *r) { memset(r->tcat, 0, sizeof r->tcat); r->tlast = pr_now(); }
void praxis_prof_report(praxis_runner *r, FILE *f) {
    double tot = 0;
    for (int i = 0; i < PT_N; i++) tot += r->tcat[i];
    fprintf(f, "  profile (synchronised phases, total %.3f s):\n", tot);
    for (int i = 0; i < PT_N; i++) fprintf(f, "    %-30s %8.3f s  %5.1f%%\n", pr_tnames[i], r->tcat[i], tot > 0 ? 100.0 * r->tcat[i] / tot : 0.0);
}
int praxis_nvfp4_enabled(const praxis_runner *r) { return r->nvfp4; }
int praxis_n_chunks(const praxis_runner *r) { return r->c.n_chunks; }
int praxis_lat_per_chunk(const praxis_runner *r) { return r->c.lat; }
int praxis_action_dim(const praxis_runner *r) { return r->c.act_dim; }
int praxis_proprio_dim(const praxis_runner *r) { return r->c.prop_dim; }
int praxis_text_dim(const praxis_runner *r) { return r->c.text_cin; }
int praxis_video_c(const praxis_runner *r) { return r->c.enc_vcin; }

/* ---- MoE feed-forward ---- */
static int pr_moe_ffn_exact(praxis_runner *r, pr_blk *b, const char *tag, int bi, CUdeviceptr x, int T, CUdeviceptr out) {
    const pr_cfg *c = &r->c;
    int C = c->dim, E = c->n_experts, k = c->top_k, Hm = c->moe_hidden;
    char nm[128];
    CUdeviceptr logits = pr_ws(r, WS_LOG, (size_t)T * E);
    CUdeviceptr topi = pr_ws(r, WS_TOPI, (size_t)T * k), topg = pr_ws(r, WS_TOPG, (size_t)T * k);
    if (!logits || !topi || !topg) return -1;
    if (pr_gemm(r, x, C, &b->router, 0, 0, logits, E, T, E, C, 0)) return -1;
    LAUNCH(r->f_topk, (T + 63) / 64, 1, 1, 64, 0, PD(logits), PD(b->ebias), PD(topi), PD(topg), PI(T), PI(E), PI(k));
    snprintf(nm, sizeof nm, "%s_b%d_router_logits", tag, bi); pr_dump(r, nm, logits, T, E);
    snprintf(nm, sizeof nm, "%s_b%d_top_idx", tag, bi); pr_dump_i(r, nm, topi, T, k);
    /* host dispatch: group token-rows by expert */
    int *hi = (int *)malloc((size_t)T * k * 4);
    PRC(cuMemcpyDtoH(hi, topi, (size_t)T * k * 4));
    int *cnt = (int *)calloc(E + 1, 4), *seg = (int *)calloc(E + 1, 4);
    for (long i = 0; i < (long)T * k; i++) cnt[hi[i]]++;
    int maxc = 0;
    for (int e = 0; e < E; e++) { seg[e + 1] = seg[e] + cnt[e]; if (cnt[e] > maxc) maxc = cnt[e]; }
    int *tokid = (int *)malloc((size_t)T * k * 4), *tokrow = (int *)malloc((size_t)T * k * 4);
    int *fill = (int *)calloc(E, 4);
    for (int t = 0; t < T; t++) for (int j = 0; j < k; j++) {
        int e = hi[t * k + j], row = seg[e] + fill[e]++;
        tokid[row] = t; tokrow[t * k + j] = row;
    }
    CUdeviceptr dseg = pr_ws_i(r, WS_SEG, E + 1), dtok = pr_ws_i(r, WS_TOKID, (size_t)T * k), dtr = pr_ws_i(r, WS_TOKROW, (size_t)T * k);
    cuMemcpyHtoD(dseg, seg, (E + 1) * 4); cuMemcpyHtoD(dtok, tokid, (size_t)T * k * 4); cuMemcpyHtoD(dtr, tokrow, (size_t)T * k * 4);
    free(hi); free(cnt); free(seg); free(tokid); free(tokrow); free(fill);
    /* shared expert */
    CUdeviceptr gus = pr_ws(r, WS_FA, (size_t)T * 2 * Hm), hs = pr_ws(r, WS_FB, (size_t)T * Hm);
    if (!gus || !hs) return -1;
    if (pr_gemm(r, x, C, &b->sh_gu, 1, 0, gus, 2 * Hm, T, 2 * Hm, C, 0)) return -1;
    if (pr_swi(r, gus, 2 * Hm, gus + F4(Hm), 2 * Hm, hs, Hm, Hm, T, 0)) return -1;
    if (pr_gemm(r, hs, Hm, &b->sh_dn, 1, 0, out, C, T, C, Hm, 0)) return -1;
    /* routed experts */
    pr_mat gu, dn; pr_slot *slot = NULL;
    if (pr_moe_acquire(r, b, &gu, &dn, &slot)) return -1;
    pr_moe_prefetch(r, (b->moe_id + 1) % r->n_moe_total);
    size_t R = (size_t)T * k;
    CUdeviceptr gur = pr_ws(r, WS_GU, R * 2 * Hm), hh = pr_ws(r, WS_HH, R * Hm), y = pr_ws(r, WS_Y, R * C);
    if (!gur || !hh || !y) return -1;
    int gy = (maxc + 31) / 32;
    if (gy > 0) {
        LAUNCH(r->f_gemm, (2 * Hm + 31) / 32, gy, E, 256, 0, PD(x), PL(C), PD(dtok), PD(gu.p), PI(gu.dt), PI(1), PL((long long)C * 2 * Hm), PD(0),
               PD(gur), PL(2 * Hm), PI(0), PI(2 * Hm), PI(C), PI(0), PD(dseg));
        if (pr_swi(r, gur, 2 * Hm, gur + F4(Hm), 2 * Hm, hh, Hm, Hm, (long long)R, 0)) return -1;
        LAUNCH(r->f_gemm, (C + 31) / 32, gy, E, 256, 0, PD(hh), PL(Hm), PD(0), PD(dn.p), PI(dn.dt), PI(1), PL((long long)Hm * C), PD(0),
               PD(y), PL(C), PI(0), PI(C), PI(Hm), PI(0), PD(dseg));
        LAUNCH(r->f_combine, NBLK((long long)T * C), 1, 1, 256, 0, PD(out), PD(y), PD(dtr), PD(topg), PF(c->routed_scale), PI(C), PL(T), PI(k));
    }
    if (slot) { cuEventRecord(slot->done, 0); slot->done_valid = 1; }
    r->seq++; r->n_moe_run++;
    return 0;
}

static int pr_dense_ffn_exact(praxis_runner *r, pr_blk *b, CUdeviceptr x, int T, CUdeviceptr out) {
    int C = r->c.dim, Hd = r->c.dense_hidden;
    CUdeviceptr a = pr_ws(r, WS_FA, (size_t)T * Hd), bb = pr_ws(r, WS_FB, (size_t)T * Hd), h = pr_ws(r, WS_FH, (size_t)T * Hd);
    if (!a || !bb || !h) return -1;
    if (pr_gemm(r, x, C, &b->a, 0, 0, a, Hd, T, Hd, C, 0)) return -1;
    if (pr_gemm(r, x, C, &b->b, 0, 0, bb, Hd, T, Hd, C, 0)) return -1;
    if (pr_swi(r, a, Hd, bb, Hd, h, Hd, Hd, T, 0)) return -1;
    return pr_gemm(r, h, Hd, &b->down, 0, 0, out, C, T, C, Hd, 0);
}


/* ---- fast path: bf16 operands, fp32 accumulate, device-side expert dispatch, chunked workspace ---- */
static int pr_flash_ex(praxis_runner *r, CUdeviceptr Qb, long long ldq, CUdeviceptr Kb, long long ldk, CUdeviceptr Vb, long long ldv,
                    CUdeviceptr Ob, long long ldo, CUdeviceptr gate, int H, int KVH, int hd, int nq, int nk,
                    CUdeviceptr qcode, CUdeviceptr kcode, CUdeviceptr ktile, int iv) {
    int fi = hd == 16 ? 0 : hd == 32 ? 1 : hd == 64 ? 2 : 3;
    CUfunction f = iv ? r->f_flashiv[fi] : r->f_flash[fi];
    int ntile = (nk + 31) / 32;
    size_t sm = (size_t)4 * 32 * (hd + 8) * 2 + 4 * (size_t)(ntile + 8);
    if (sm > 48 * 1024 && cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)sm) != CUDA_SUCCESS) {
        fprintf(stderr, "praxis: flash attention smem %zu B too large (nk=%d)\n", sm, nk); return -1; }
    LAUNCH(f, (nq + 63) / 64, H, 1, 128, sm, PD(Qb), PL(ldq), PD(Kb), PL(ldk), PD(Vb), PL(ldv), PD(Ob), PL(ldo), PD(gate),
           PI(H), PI(KVH), PI(nq), PI(nk), PF(1.0 / sqrt((double)hd)), PD(qcode), PD(kcode), PD(ktile));
    return 0;
}
static int pr_flash(praxis_runner *r, CUdeviceptr Qb, long long ldq, CUdeviceptr Kb, long long ldk, CUdeviceptr Vb, long long ldv,
                    CUdeviceptr Ob, long long ldo, CUdeviceptr gate, int H, int KVH, int hd, int nq, int nk,
                    CUdeviceptr qcode, CUdeviceptr kcode, CUdeviceptr ktile) {
    return pr_flash_ex(r, Qb, ldq, Kb, ldk, Vb, ldv, Ob, ldo, gate, H, KVH, hd, nq, nk, qcode, kcode, ktile, 0);
}
static int pr_rms_bf(praxis_runner *r, CUdeviceptr x, long long ldx, CUdeviceptr y, CUdeviceptr yb, CUdeviceptr w, int plus1,
                     float eps, float oscale, int C, long long rows) {
    LAUNCH(r->f_rmsb, rows, 1, 1, C >= 128 ? 128 : 32, 0, PD(x), PL(ldx), PD(y), PD(yb), PD(w), PI(plus1), PF(eps), PF(oscale), PI(C));
    return 0;
}
static int pr_qkprep(praxis_runner *r, CUdeviceptr x, long long ldx, CUdeviceptr yb, long long ldy, CUdeviceptr w, CUdeviceptr cs, CUdeviceptr sn,
                     float eps, int hd, int heads, long long n, long long tseq) {
    int th = hd / 2 >= 32 ? hd / 2 : 32;
    LAUNCH(r->f_qkprep, n * heads, 1, 1, th, (unsigned)(hd * 4), PD(x), PL(ldx), PD(yb), PL(ldy), PD(w), PD(cs), PD(sn), PF(eps), PI(hd), PI(heads), PL(tseq));
    return 0;
}
static int pr_swib(praxis_runner *r, CUdeviceptr a, long long lda, CUdeviceptr b, long long ldb, CUdeviceptr o, long long ldo, int H, long long rows) {
    LAUNCH(r->f_swib, NBLK(rows * H), 1, 1, 256, 0, PD(a), PL(lda), PD(b), PL(ldb), PD(o), PL(ldo), PI(H), PL(rows));
    return 0;
}

static int pr_dense_ffn_fast(praxis_runner *r, pr_blk *b, CUdeviceptr xb, int T, CUdeviceptr out) {
    int C = r->c.dim, Hd = r->c.dense_hidden, TC = r->moe_chunk;
    int Tm = T < TC ? T : TC;
    CUdeviceptr a = pr_wsb(r, WS_HB1, (size_t)Tm * Hd), bb = pr_wsb(r, WS_HB2, (size_t)Tm * Hd), h = pr_wsb(r, WS_HB3, (size_t)Tm * Hd);
    if (!a || !bb || !h) return -1;
    for (int t0 = 0; t0 < T; t0 += TC) {
        int Tc = T - t0 < TC ? T - t0 : TC;
        CUdeviceptr xc = xb + (size_t)t0 * C * 2;
        if (pr_gemm2(r, 0, xc, C, &b->a, 0, 0, a, Hd, 1, Tc, Hd, C, 0)) return -1;
        if (pr_gemm2(r, 0, xc, C, &b->b, 0, 0, bb, Hd, 1, Tc, Hd, C, 0)) return -1;
        if (pr_swib(r, a, Hd, bb, Hd, h, Hd, Hd, Tc)) return -1;
        if (pr_gemm2(r, 0, h, Hd, &b->down, 0, 0, out + F4((size_t)t0 * C), C, 0, Tc, C, Hd, 0)) return -1;
    }
    return 0;
}


/* Routed experts of one token chunk from a quantised bank (per-matrix format fgu / fdn: 0 NVFP4, 1 FP8).
 * NVFP4 matrices with a16 == 0: dynamic W4A4 (activation global amax / 2688), grouped block-scale mma GEMM over the expert-sorted rows (no host
 * readback: the grid is sized by an upper bound and exits on the device tile prefix). FP8 matrices, and NVFP4 ones with a16 == 1, run
 * bf16 activations x in-register dequantised weights (bf16 mma). x: fp32 [Tc][C] chunk rows (A4 only), xb: bf16 copy (A16), gathered through dtok.
 * yb: bf16 [R][C]. */
static int pr_moe_chunk_ex(praxis_runner *r, CUdeviceptr bank, int fgu, int fdn, int a16, CUdeviceptr x, CUdeviceptr xb, int Tc, CUdeviceptr dseg, CUdeviceptr dtok,
                           CUdeviceptr gub, CUdeviceptr hh, CUdeviceptr yb, int e0, int Eg) {
    (void)xb;
    const pr_cfg *c = &r->c;
    int C = c->dim, E = Eg, k = c->top_k, Hm = c->moe_hidden;
    pr_fp4_lay L = pr_fp4_layout2(E, C, Hm, fgu, fdn);
    dseg += (CUdeviceptr)((size_t)e0 * 4);   /* rows of this expert group: seg[] entries e0..e0+Eg */
    long long R = (long long)Tc * k;
    int gu_a4 = !fgu && !a16, dn_a4 = !fdn && !a16;
    CUdeviceptr tpre = pr_ws_i(r, WS_TPRE, E + 4), am = pr_ws_i(r, WS_AMAX, 8);
    CUdeviceptr aq = 0, as = 0, aq2 = 0, as2 = 0, xh = 0;
    if (gu_a4) { aq = pr_ws_i(r, WS_FP4A, (size_t)R * C / 8 + 4); as = pr_ws_i(r, WS_FP4B, (size_t)R * C / 64 + 4); if (!aq || !as) return -1; }
    else { xh = pr_wsb(r, WS_XG, (size_t)Tc * C); if (!xh) return -1; }
    if (dn_a4) { aq2 = pr_ws_i(r, WS_FP4C, (size_t)R * Hm / 8 + 4); as2 = pr_ws_i(r, WS_FP4D, (size_t)R * Hm / 64 + 4); if (!aq2 || !as2) return -1; }
    if (!tpre || !am) return -1;
    pr_tick(r, PT_ROUTER);
    PRC(cuMemsetD32(am, 0, 2));
    LAUNCH(r->f4_tiles, 1, 1, 1, 32, 0, PD(dseg), PI(E), PI(64), PD(tpre));
    int ymax = (int)((R + 63) / 64) + E;
    CUdeviceptr nomap = 0;
    int f16_mid = !dn_a4;   /* hidden activations stay f16 when the down GEMM reads f16; the bf16 path (pr_swib) feeds the A4 quantiser */
    if (gu_a4) {
        LAUNCH(r->f4_amax_f32, 1024, 1, 1, 256, 0, PD(x), PL((long long)Tc * C), PD(am));
        LAUNCH(r->f4_quant_f32, NBLK(R * (C / 16)), 1, 1, 256, 0, PD(x), PL(C), PD(dtok), PD(am), PD(aq), PD(as), PL(R), PI(C));
        pr_tick(r, PT_FQ);
        LAUNCH(r->f4_gemm, (2 * Hm + 127) / 128, ymax, 1, 256, 2 * 16896, PD(aq), PD(as), PD(bank + L.gu_q), PD(bank + L.gu_sf), PD(bank + L.gu_g), PD(am),
               PD(dseg), PD(tpre), PI(E), PD(gub), PL(2 * Hm), PI(2 * Hm), PI(C));
    } else {
        LAUNCH(r->f4_cvt, NBLK((long long)Tc * (C / 2)), 1, 1, 256, 0, PD(x), PL(C), PD(xh), PL(C), PL(Tc), PI(C));
        pr_tick(r, PT_FQ);
        if (fgu) {
            LAUNCH(r->f8_w16, (2 * Hm + 127) / 128, ymax, 1, 256, 2 * 5120 * 4, PD(xh), PL(C), PD(dtok), PD(bank + L.gu_q), PD(bank + L.gu_sf),
                   PD(dseg), PD(tpre), PI(E), PD(gub), PL(2 * Hm), PI(2 * Hm), PI(C), PI(f16_mid));
        } else {
            LAUNCH(r->f4_w16, (2 * Hm + 127) / 128, ymax, 1, 256, 2 * 4224 * 4, PD(xh), PL(C), PD(dtok), PD(bank + L.gu_q), PD(bank + L.gu_sf), PD(bank + L.gu_g),
                   PD(dseg), PD(tpre), PI(E), PD(gub), PL(2 * Hm), PI(2 * Hm), PI(C), PI(f16_mid));
        }
    }
    pr_tick(r, PT_FG1);
    if (!gu_a4 && f16_mid) {
        LAUNCH(r->f4_swi16, NBLK(R * (Hm / 2)), 1, 1, 256, 0, PD(gub), PL(2 * Hm), PD(gub + (size_t)Hm * 2), PL(2 * Hm), PD(hh), PL(Hm), PI(Hm), PL(R));
    } else {
        if (pr_swib(r, gub, 2 * Hm, gub + (size_t)Hm * 2, 2 * Hm, hh, Hm, Hm, R)) return -1;
        if (gu_a4 && !dn_a4) {} /* unreachable combination guard (a4 gu and a4 dn both) */
        if (gu_a4 && f16_mid) LAUNCH(r->f4_b2h, NBLK(R * Hm), 1, 1, 256, 0, PD(hh), PL(R * Hm));
    }
    pr_tick(r, PT_FSW);
    if (dn_a4) {
        LAUNCH(r->f4_amax_bf16, 1024, 1, 1, 256, 0, PD(hh), PL(R * Hm), PD(am + 4));
        LAUNCH(r->f4_quant_bf16, NBLK(R * (Hm / 16)), 1, 1, 256, 0, PD(hh), PL(Hm), PD(am + 4), PD(aq2), PD(as2), PL(R), PI(Hm));
        pr_tick(r, PT_FQ);
        LAUNCH(r->f4_gemm, (C + 127) / 128, ymax, 1, 256, 2 * 16896, PD(aq2), PD(as2), PD(bank + L.dn_q), PD(bank + L.dn_sf), PD(bank + L.dn_g), PD(am + 4),
               PD(dseg), PD(tpre), PI(E), PD(yb), PL(C), PI(C), PI(Hm));
    } else if (fdn) {
        LAUNCH(r->f8_w16, (C + 127) / 128, ymax, 1, 256, 2 * 5120 * 4, PD(hh), PL(Hm), PD(nomap), PD(bank + L.dn_q), PD(bank + L.dn_sf),
               PD(dseg), PD(tpre), PI(E), PD(yb), PL(C), PI(C), PI(Hm), PI(0));
    } else {
        LAUNCH(r->f4_w16, (C + 127) / 128, ymax, 1, 256, 2 * 4224 * 4, PD(hh), PL(Hm), PD(nomap), PD(bank + L.dn_q), PD(bank + L.dn_sf), PD(bank + L.dn_g),
               PD(dseg), PD(tpre), PI(E), PD(yb), PL(C), PI(C), PI(Hm), PI(0));
    }
    pr_tick(r, PT_FG2);
    return 0;
}
static int pr_moe_chunk_fp4(praxis_runner *r, CUdeviceptr bank, CUdeviceptr x, int Tc, CUdeviceptr dseg, CUdeviceptr dtok, CUdeviceptr gub, CUdeviceptr hh, CUdeviceptr yb) {
    return pr_moe_chunk_ex(r, bank, 0, 0, 0, x, 0, Tc, dseg, dtok, gub, hh, yb, 0, r->c.n_experts);
}
static int pr_moe_chunk_fp4_a16(praxis_runner *r, CUdeviceptr bank, CUdeviceptr xb, int Tc, CUdeviceptr dseg, CUdeviceptr dtok, CUdeviceptr gub, CUdeviceptr hh, CUdeviceptr yb) {
    return pr_moe_chunk_ex(r, bank, 0, 0, 1, xb, 0, Tc, dseg, dtok, gub, hh, yb, 0, r->c.n_experts);
}

static int pr_moe_ffn_fast(praxis_runner *r, pr_blk *b, const char *tag, int bi, CUdeviceptr x, CUdeviceptr xb, int T, CUdeviceptr out) {
    const pr_cfg *c = &r->c;
    int C = c->dim, E = c->n_experts, k = c->top_k, Hm = c->moe_hidden, TC = r->moe_chunk;
    char nm[128];
    CUdeviceptr logits = pr_ws(r, WS_LOG, (size_t)T * E);
    CUdeviceptr topi = pr_ws(r, WS_TOPI, (size_t)T * k), topg = pr_ws(r, WS_TOPG, (size_t)T * k);
    if (!logits || !topi || !topg) return -1;
    if (pr_gemm_naive(r, x, C, &b->router, 0, 0, logits, E, T, E, C, 0)) return -1;
    LAUNCH(r->f_topk, (T + 63) / 64, 1, 1, 64, 0, PD(logits), PD(b->ebias), PD(topi), PD(topg), PI(T), PI(E), PI(k));
    snprintf(nm, sizeof nm, "%s_b%d_router_logits", tag, bi); pr_dump(r, nm, logits, T, E);
    snprintf(nm, sizeof nm, "%s_b%d_top_idx", tag, bi); pr_dump_i(r, nm, topi, T, k);
    { static int dm = -1; if (dm < 0) { const char *e = getenv("PRAXIS_DUMP_MOE"); dm = e && *e && *e != '0'; }
      if (dm) {   /* per-layer attribution inputs: the fp32 routed-expert input rows and the combine gates */
          snprintf(nm, sizeof nm, "%s_b%d_moe_x", tag, bi); pr_dump(r, nm, x, T, C);
          snprintf(nm, sizeof nm, "%s_b%d_top_gate", tag, bi); pr_dump(r, nm, topg, T, k);
      } }
    pr_tick(r, PT_ROUTER);
    int Tm = T < TC ? T : TC;
    size_t Rm = (size_t)Tm * k;
    /* shared expert (token chunks) */
    {
        CUdeviceptr gub = pr_wsb(r, WS_GUB, (size_t)Rm * 2 * Hm), hs = pr_wsb(r, WS_HB1, (size_t)Tm * Hm);
        if (!gub || !hs) return -1;
        for (int t0 = 0; t0 < T; t0 += TC) {
            int Tc = T - t0 < TC ? T - t0 : TC;
            if (pr_gemm2(r, 0, xb + (size_t)t0 * C * 2, C, &b->sh_gu, 1, 0, gub, 2 * Hm, 1, Tc, 2 * Hm, C, 0)) return -1;
            if (pr_swib(r, gub, 2 * Hm, gub + (size_t)Hm * 2, 2 * Hm, hs, Hm, Hm, Tc)) return -1;
            if (pr_gemm2(r, 0, hs, Hm, &b->sh_dn, 1, 0, out + F4((size_t)t0 * C), C, 0, Tc, C, Hm, 0)) return -1;
        }
    }
    pr_tick(r, PT_MOEOTH);
    /* routed experts */
    pr_mat gu, dn; pr_slot *slot = NULL;
    if (b->ngrp > 1) {   /* expert-group streaming: the layer is streamed as ngrp contiguous groups of experts, one ring slot each; with several token chunks
                          * the groups are streamed again for every chunk (only a few oversized layers are grouped) */
        if (b->bank_dt != 3) { fprintf(stderr, "praxis: grouped expert banks need the NVFP4/FP8 fast path\n"); return -1; }
        int Eg = E / b->ngrp;
        CUdeviceptr gub = pr_wsb(r, WS_GUB, Rm * 2 * Hm), hh = pr_wsb(r, WS_HHB, Rm * Hm), yb = pr_wsb(r, WS_YB, Rm * C);
        CUdeviceptr dcnt = pr_ws_i(r, WS_CNT, E + 1), dseg = pr_ws_i(r, WS_SEGD, E + 2);
        CUdeviceptr dtok = pr_ws_i(r, WS_TOKID, Rm), dtr = pr_ws_i(r, WS_TOKROW, Rm);
        if (!gub || !hh || !yb || !dcnt || !dseg || !dtok || !dtr) return -1;
        for (int t0 = 0; t0 < T; t0 += TC) {
            int Tc = T - t0 < TC ? T - t0 : TC, n = Tc * k, more = t0 + TC < T;
            CUdeviceptr ti = topi + (size_t)t0 * k * 4, tg = topg + F4((size_t)t0 * k);
            LAUNCH(r->f_mcount, E, 1, 1, 256, 0, PD(ti), PI(n), PD(dcnt));
            LAUNCH(r->f_mscan, 1, 1, 1, 32, 0, PD(dcnt), PI(E), PD(dseg));
            LAUNCH(r->f_mfill, E, 1, 1, 256, 0, PD(ti), PI(n), PI(k), PD(dseg), PD(dtok), PD(dtr));
            for (int g = 0; g < b->ngrp; g++) {
                if (pr_moe_acquire2(r, b, g, &gu, &dn, &slot)) return -1;
                if (g + 1 < b->ngrp) pr_moe_prefetch2(r, b->moe_id, g + 1);
                else if (more) pr_moe_prefetch2(r, b->moe_id, 0);
                else pr_moe_prefetch2(r, (b->moe_id + 1) % r->n_moe_total, 0);
                pr_tick(r, PT_STALL);
                if (pr_moe_chunk_ex(r, gu.p, b->fgu, b->fdn, r->fp4_a16, x + F4((size_t)t0 * C), xb + (size_t)t0 * C * 2, Tc, dseg, dtok, gub, hh, yb, g * Eg, Eg)) return -1;
                pr_tick(r, PT_EXPERT);
                if (slot) { cuEventRecord(slot->done, 0); slot->done_valid = 1; }
                r->seq++;
            }
            LAUNCH(r->f_combb, NBLK((long long)Tc * C), 1, 1, 256, 0, PD(out + F4((size_t)t0 * C)), PD(yb), PD(dtr), PD(tg), PF(c->routed_scale), PI(C), PL(Tc), PI(k));
            pr_tick(r, PT_MOEOTH);
        }
        r->n_moe_run++;
        return 0;
    }
    if (pr_moe_acquire(r, b, &gu, &dn, &slot)) return -1;
    pr_moe_prefetch(r, (b->moe_id + 1) % r->n_moe_total);
    pr_tick(r, PT_STALL);
    CUdeviceptr xg = b->bank_dt == 3 ? 0 : pr_wsb(r, WS_XG, Rm * C), gub = pr_wsb(r, WS_GUB, Rm * 2 * Hm), hh = pr_wsb(r, WS_HHB, Rm * Hm), yb = pr_wsb(r, WS_YB, Rm * C);
    CUdeviceptr dcnt = pr_ws_i(r, WS_CNT, E + 1), dseg = pr_ws_i(r, WS_SEGD, E + 2);
    CUdeviceptr dtok = pr_ws_i(r, WS_TOKID, Rm), dtr = pr_ws_i(r, WS_TOKROW, Rm);
    if ((!xg && b->bank_dt != 3) || !gub || !hh || !yb || !dcnt || !dseg || !dtok || !dtr) return -1;
    size_t gu_e = (size_t)C * 2 * Hm * 2, dn_e = (size_t)Hm * C * 2;
    for (int t0 = 0; t0 < T; t0 += TC) {
        int Tc = T - t0 < TC ? T - t0 : TC;
        int n = Tc * k;
        CUdeviceptr ti = topi + (size_t)t0 * k * 4, tg = topg + F4((size_t)t0 * k);
        LAUNCH(r->f_mcount, E, 1, 1, 256, 0, PD(ti), PI(n), PD(dcnt));
        LAUNCH(r->f_mscan, 1, 1, 1, 32, 0, PD(dcnt), PI(E), PD(dseg));
        LAUNCH(r->f_mfill, E, 1, 1, 256, 0, PD(ti), PI(n), PI(k), PD(dseg), PD(dtok), PD(dtr));
        if (b->bank_dt == 3) {
            if (pr_moe_chunk_ex(r, gu.p, b->fgu, b->fdn, r->fp4_a16, x + F4((size_t)t0 * C), xb + (size_t)t0 * C * 2, Tc, dseg, dtok, gub, hh, yb, 0, E)) return -1;
            pr_tick(r, PT_EXPERT);
            LAUNCH(r->f_combb, NBLK((long long)Tc * C), 1, 1, 256, 0, PD(out + F4((size_t)t0 * C)), PD(yb), PD(dtr), PD(tg), PF(c->routed_scale), PI(C), PL(Tc), PI(k));
            pr_tick(r, PT_MOEOTH);
            continue;
        }
        PRC(cuMemcpyDtoH(r->hseg, dseg, (size_t)(E + 1) * 4));
        pr_tick(r, PT_ROUTER);
        LAUNCH(r->f_gather, NBLK((long long)n * (C / 8)), 1, 1, 256, 0, PD(xb + (size_t)t0 * C * 2), PD(dtok), PD(xg), PI(C), PL(n));
        for (int e = 0; e < E; e++) {
            int ne = r->hseg[e + 1] - r->hseg[e];
            if (ne <= 0) continue;
            pr_mat ge = { gu.p + (CUdeviceptr)(e * gu_e), 1, gu_e };
            if (pr_gemm2(r, 0, xg + (size_t)r->hseg[e] * C * 2, C, &ge, 1, 0, gub + (size_t)r->hseg[e] * 2 * Hm * 2, 2 * Hm, 1, ne, 2 * Hm, C, 0)) return -1;
        }
        if (pr_swib(r, gub, 2 * Hm, gub + (size_t)Hm * 2, 2 * Hm, hh, Hm, Hm, n)) return -1;
        for (int e = 0; e < E; e++) {
            int ne = r->hseg[e + 1] - r->hseg[e];
            if (ne <= 0) continue;
            pr_mat de = { dn.p + (CUdeviceptr)(e * dn_e), 1, dn_e };
            if (pr_gemm2(r, 0, hh + (size_t)r->hseg[e] * Hm * 2, Hm, &de, 1, 0, yb + (size_t)r->hseg[e] * C * 2, C, 1, ne, C, Hm, 0)) return -1;
        }
        pr_tick(r, PT_EXPERT);
        LAUNCH(r->f_combb, NBLK((long long)Tc * C), 1, 1, 256, 0, PD(out + F4((size_t)t0 * C)), PD(yb), PD(dtr), PD(tg), PF(c->routed_scale), PI(C), PL(Tc), PI(k));
        pr_tick(r, PT_MOEOTH);
    }
    if (slot) { cuEventRecord(slot->done, 0); slot->done_valid = 1; }
    r->seq++; r->n_moe_run++;
    return 0;
}

static int pr_moe_ffn(praxis_runner *r, pr_blk *b, const char *tag, int bi, CUdeviceptr x, CUdeviceptr xb, int T, CUdeviceptr out) {
    if (r->fast && (b->bank_dt == 1 || b->bank_dt == 3) && xb) return pr_moe_ffn_fast(r, b, tag, bi, x, xb, T, out);
    return pr_moe_ffn_exact(r, b, tag, bi, x, T, out);
}
static int pr_dense_ffn(praxis_runner *r, pr_blk *b, CUdeviceptr x, CUdeviceptr xb, int T, CUdeviceptr out) {
    if (r->fast && xb && b->a.dt == 1 && b->b.dt == 1 && b->down.dt == 1) return pr_dense_ffn_fast(r, b, xb, T, out);
    return pr_dense_ffn_exact(r, b, x, T, out);
}

/* One transformer stack: x [T][C] in place. mrow/modulation only for the decoder. */
static int pr_stack(praxis_runner *r, pr_tower *tw, const char *tag, CUdeviceptr x, int Tseq, int nb, CUdeviceptr cs, CUdeviceptr sn,
                    CUdeviceptr qcode, CUdeviceptr mod_in, int n_mod_rows, CUdeviceptr mrow,
                    const int *taps, int ntap, CUdeviceptr **tap_out) {
    /* nb sequences of Tseq tokens (identical layout, identical rope/codes) share every row-wise op and every expert read;
     * attention runs per sequence. nb > 1 needs the fast path. */
    int T = Tseq * nb;
    const pr_cfg *c = &r->c;
    int C = c->dim, hd = c->head_dim, H = c->n_heads, KVH = c->n_kv_heads;
    int fast = r->fast;
    CUdeviceptr n1 = pr_ws(r, WS_N1, (size_t)T * C), q = pr_ws(r, WS_Q, (size_t)T * H * hd), k = pr_ws(r, WS_K, (size_t)T * KVH * hd),
                g = pr_ws(r, WS_G, (size_t)T * H), proj = pr_ws(r, WS_PROJ, (size_t)T * C), n2 = pr_ws(r, WS_N2, (size_t)T * C),
                fo = pr_ws(r, WS_FO, (size_t)T * C);
    CUdeviceptr v = 0, ao = 0, n1b = 0, n2b = 0, qb = 0, kb = 0, vb = 0, aob = 0, ktile = 0, kw = 0, vw = 0;
    if (!(n1 && q && k && g && proj && n2 && fo)) return -1;
    if (fast) {
        n1b = pr_wsb(r, WS_NB1, (size_t)T * C); n2b = pr_wsb(r, WS_NB2, (size_t)T * C);
        qb = pr_wsb(r, WS_QB, (size_t)T * H * hd); kb = pr_wsb(r, WS_KB, (size_t)T * KVH * hd); vb = pr_wsb(r, WS_VB, (size_t)T * KVH * hd);
        aob = pr_wsb(r, WS_AOB, (size_t)T * H * hd);
        int nkx = r->kx_mode == 2 ? r->kx_nk : Tseq;
        ktile = pr_ws_i(r, WS_TILE, 3 * (size_t)((nkx + 31) / 32) + 4);
        if (!(n1b && n2b && qb && kb && vb && aob && ktile)) return -1;
        LAUNCH(r->f_ctiles, ((nkx + 31) / 32 + 63) / 64, 1, 1, 64, 0, PD(r->kx_mode == 2 ? r->kx_kcode : qcode), PI(nkx), PI(32), PD(ktile));
        if (r->kx_mode == 2) {
            kw = pr_wsb(r, WS_KW, (size_t)nb * nkx * KVH * hd); vw = pr_wsb(r, WS_VW, (size_t)nb * nkx * KVH * hd);
            if (!kw || !vw) return -1;
        }
    } else {
        v = pr_ws(r, WS_V, (size_t)T * KVH * hd); ao = pr_ws(r, WS_AO, (size_t)T * H * hd);
        if (!v || !ao) return -1;
    }
    char nm[128];
    pr_tick(r, PT_OTHER);
    for (int i = 0; i < tw->depth; i++) {
        pr_blk *b = &tw->b[i];
        if (mod_in) {   /* x = modulate(x, shift, 1 + scale), modulator on the compact conditioning rows */
            CUdeviceptr md = pr_ws(r, WS_MOD2, (size_t)n_mod_rows * 2 * C);
            if (!md) return -1;
            if (pr_gemm(r, mod_in, c->dec_mod_dim, &b->mod, 0, b->mod_b, md, 2 * C, n_mod_rows, 2 * C, c->dec_mod_dim, 0)) return -1;
            LAUNCH(r->f_modulate, NBLK((long long)T * C), 1, 1, 256, 0, PD(x), PD(x), PD(md), PL(2 * C), PD(mrow), PI(C), PL(T));
        }
        pr_tick(r, PT_OTHER);
        if (fast) {
            if (pr_rms_bf(r, x, C, n1, n1b, b->ln_in, 1, 1e-6f, b->s, C, T)) return -1;
            if (pr_gemm2(r, n1, n1b, C, &b->q, 0, 0, q, (long long)H * hd, 0, T, H * hd, C, 0)) return -1;
            if (pr_gemm2(r, n1, n1b, C, &b->k, 0, 0, k, (long long)KVH * hd, 0, T, KVH * hd, C, 0)) return -1;
            if (pr_gemm2(r, n1, n1b, C, &b->v, 0, 0, vb, (long long)KVH * hd, 1, T, KVH * hd, C, 0)) return -1;
            if (pr_gemm2(r, n1, n1b, C, &b->gate, 0, 0, g, H, 0, T, H, C, 0)) return -1;
            pr_tick(r, PT_PROJ);
            if (pr_qkprep(r, q, (long long)H * hd, qb, (long long)H * hd, b->qn, cs, sn, 1e-6f, hd, H, T, Tseq)) return -1;
            if (pr_qkprep(r, k, (long long)KVH * hd, kb, (long long)KVH * hd, b->kn, cs, sn, 1e-6f, hd, KVH, T, Tseq)) return -1;
            if (r->kx_mode == 1) {   /* store the context rows (globals, clean) of this layer's K/V */
                size_t rb = (size_t)KVH * hd * 2;
                for (int sq = 0; sq < nb; sq++) {
                    pr_kvc *kc = r->kx_seq[sq];
                    if (!kc->k[i] || kc->cap[i] < (size_t)kc->Nctx * rb) {
                        if (kc->k[i]) { pr_mfree(r, kc->k[i], kc->cap[i]); pr_mfree(r, kc->v[i], kc->cap[i]); }
                        kc->cap[i] = (size_t)kc->Nctx * rb;
                        kc->k[i] = pr_malloc(r, kc->cap[i]); kc->v[i] = pr_malloc(r, kc->cap[i]);
                        if (!kc->k[i] || !kc->v[i]) return -1;
                    }
                    size_t o = (size_t)sq * Tseq;
                    PRC(cuMemcpyDtoD(kc->k[i], kb + o * rb, (size_t)kc->Ng * rb));
                    PRC(cuMemcpyDtoD(kc->v[i], vb + o * rb, (size_t)kc->Ng * rb));
                    PRC(cuMemcpyDtoD(kc->k[i] + (size_t)kc->Ng * rb, kb + (o + kc->Ng + kc->Nn) * rb, (size_t)kc->Nc * rb));
                    PRC(cuMemcpyDtoD(kc->v[i] + (size_t)kc->Ng * rb, vb + (o + kc->Ng + kc->Nn) * rb, (size_t)kc->Nc * rb));
                }
            }
            if (r->kx_mode == 2) {   /* queries = the noised rows only; keys/values = [cached context | new noised rows] */
                size_t rb = (size_t)KVH * hd * 2;
                for (int sq = 0; sq < nb; sq++) {
                    pr_kvc *kc = r->kx_seq[sq];
                    size_t o = (size_t)sq * Tseq, wo = (size_t)sq * r->kx_nk;
                    PRC(cuMemcpyDtoD(kw + wo * rb, kc->k[i], (size_t)kc->Nctx * rb));
                    PRC(cuMemcpyDtoD(vw + wo * rb, kc->v[i], (size_t)kc->Nctx * rb));
                    PRC(cuMemcpyDtoD(kw + (wo + kc->Nctx) * rb, kb + o * rb, (size_t)Tseq * rb));
                    PRC(cuMemcpyDtoD(vw + (wo + kc->Nctx) * rb, vb + o * rb, (size_t)Tseq * rb));
                    if (pr_flash(r, qb + o * H * hd * 2, (long long)H * hd, kw + wo * rb, (long long)KVH * hd, vw + wo * rb, (long long)KVH * hd,
                                 aob + o * H * hd * 2, (long long)H * hd, g + F4(o * H), H, KVH, hd, Tseq, r->kx_nk, qcode, r->kx_kcode, ktile)) return -1;
                }
            } else
            for (int sq = 0; sq < nb; sq++) {
                size_t o = (size_t)sq * Tseq;
                if (pr_flash(r, qb + o * H * hd * 2, (long long)H * hd, kb + o * KVH * hd * 2, (long long)KVH * hd, vb + o * KVH * hd * 2, (long long)KVH * hd,
                             aob + o * H * hd * 2, (long long)H * hd, g + F4(o * H), H, KVH, hd, Tseq, Tseq, qcode, qcode, ktile)) return -1;
            }
            pr_tick(r, PT_ATTN);
            if (pr_gemm2(r, 0, aob, (long long)H * hd, &b->out, 0, b->out_b, proj, C, 0, T, C, H * hd, 0)) return -1;
            pr_tick(r, PT_PROJ);
        } else {
            if (pr_rms(r, x, C, n1, C, b->ln_in, 1, 1e-6f, b->s, C, T, 1)) return -1;
            if (pr_gemm(r, n1, C, &b->q, 0, 0, q, H * hd, T, H * hd, C, 0)) return -1;
            if (pr_gemm(r, n1, C, &b->k, 0, 0, k, KVH * hd, T, KVH * hd, C, 0)) return -1;
            if (pr_gemm(r, n1, C, &b->v, 0, 0, v, KVH * hd, T, KVH * hd, C, 0)) return -1;
            if (pr_gemm(r, n1, C, &b->gate, 0, 0, g, H, T, H, C, 0)) return -1;
            if (pr_ropek(r, q, (long long)H * hd, H, hd, cs, sn, T)) return -1;
            if (pr_ropek(r, k, (long long)KVH * hd, KVH, hd, cs, sn, T)) return -1;
            if (pr_rms(r, q, (long long)H * hd, q, (long long)H * hd, b->qn, 1, 1e-6f, 1.0f, hd, T, H)) return -1;
            if (pr_rms(r, k, (long long)KVH * hd, k, (long long)KVH * hd, b->kn, 1, 1e-6f, 1.0f, hd, T, KVH)) return -1;
            if (pr_attn(r, q, (long long)H * hd, k, (long long)KVH * hd, v, (long long)KVH * hd, ao, (long long)H * hd, g, H, KVH, hd, T, T, 1, 0, qcode, qcode)) return -1;
            if (pr_gemm(r, ao, (long long)H * hd, &b->out, 0, b->out_b, proj, C, T, C, H * hd, 0)) return -1;
        }
        snprintf(nm, sizeof nm, "%s_b%d_attn", tag, i); pr_dump(r, nm, proj, T, C);
        if (pr_addv(r, x, proj, (long long)T * C)) return -1;
        if (fast) { if (pr_rms_bf(r, x, C, n2, n2b, b->ln_out, 1, 1e-6f, b->s, C, T)) return -1; }
        else if (pr_rms(r, x, C, n2, C, b->ln_out, 1, 1e-6f, b->s, C, T, 1)) return -1;
        pr_tick(r, PT_OTHER);
        if (b->moe) { if (pr_moe_ffn(r, b, tag, i, n2, n2b, T, fo)) return -1; }
        else {
            if (pr_dense_ffn(r, b, n2, n2b, T, fo)) return -1;
            pr_tick(r, PT_DENSE);
        }
        snprintf(nm, sizeof nm, "%s_b%d_ffn", tag, i); pr_dump(r, nm, fo, T, C);
        if (pr_addv(r, x, fo, (long long)T * C)) return -1;
        snprintf(nm, sizeof nm, "%s_b%d_out", tag, i); pr_dump(r, nm, x, T, C);
        for (int j = 0; j < ntap; j++) if (taps && taps[j] == i && tap_out)
            for (int sq = 0; sq < nb; sq++) PRC(cuMemcpyDtoD(tap_out[sq][j] + (r->kx_mode == 2 ? F4((size_t)r->kx_Ng * C) : 0), x + F4((size_t)sq * Tseq * C), F4((size_t)Tseq * C)));
        pr_tick(r, PT_OTHER);
    }
    return 0;
}

/* time embedding rows: out [n][time_dim] = fourier(t) then t_proj -> [n][2C] */
static int pr_tproj(praxis_runner *r, pr_tower *tw, const float *tv, int n, CUdeviceptr out) {
    int D = r->c.time_dim, C = r->c.dim;
    CUdeviceptr dt = pr_ws(r, WS_TV, n), emb = pr_ws(r, WS_TMP, (size_t)n * D);
    if (!dt || !emb) return -1;
    cuMemcpyHtoD(dt, tv, F4(n));
    LAUNCH(r->f_fourier, NBLK((long long)n * D), 1, 1, 256, 0, PD(dt), PI(n), PD(tw->t_freq), PD(tw->t_phase), PI(D), PD(emb));
    return pr_gemm(r, emb, D, &tw->tp_w, 0, tw->tp_b, out, 2 * C, n, 2 * C, D, 0);
}

/* ---- encoder ---- */
static int pr_persist(praxis_runner *r, CUdeviceptr *p, size_t *cap, size_t bytes) {
    if (*p && *cap >= bytes) return 0;
    if (*p) pr_mfree(r, *p, *cap);
    *p = pr_malloc(r, bytes);
    *cap = *p ? bytes : 0;
    return *p ? 0 : -1;
}

/* embeds one sequence into row slot idx of an nb-sequence batch (x = WS_X), fills es (positions, persistent emb/taps/cond buffers)
 * and the shared rope tables / token codes. Returns the sequence base pointer in *xs, the codes in *dcode_out. */
static int pr_enc_prepare(praxis_runner *r, pr_enc_state *es, int nb, int idx, const float *text, int S, const float *anchor, int h, int w,
                          const float *noised, const float *clean, const float *t, int K, CUdeviceptr *xs, CUdeviceptr *dcode_out) {
    const pr_cfg *c = &r->c;
    pr_tower *tw = &r->enc;
    int C = c->dim, F = c->lat, vc = c->enc_vcin, tpf = h * w, hd = c->head_dim;
    int nreg = c->n_reg, Na = tpf;
    int Nn = K * F * tpf, Nc = (K - 1) * F * tpf;
    int T = nreg + Na + S + Nn + Nc, Ng = nreg + Na + S;
    es->K = K; es->S = S; es->h = h; es->w = w; es->F = F; es->tpf = tpf; es->n_reg = nreg; es->n_anc = Na; es->T = T;
    es->Nn = Nn; es->Nc = Nc; es->Ng = Ng;
    CUdeviceptr x = pr_ws(r, WS_X, (size_t)T * C * nb);
    if (x) x += F4((size_t)idx * T * C);
    size_t inmax = (size_t)vc * (Nn > Na ? Nn : Na);
    if (inmax < (size_t)S * c->text_cin) inmax = (size_t)S * c->text_cin;
    CUdeviceptr in = pr_ws(r, WS_IN, inmax), tok = pr_ws(r, WS_TOK, (size_t)vc * (Nn > Na ? Nn : Na));
    if (!x || !in || !tok) return -1;
    PRC(cuMemcpyDtoD(x, tw->registers, F4((size_t)nreg * C)));
    /* anchor tokens */
    PRC(cuMemcpyHtoD(in, anchor, F4((size_t)vc * Na)));
    LAUNCH(r->f_chw, NBLK((long long)Na * vc), 1, 1, 256, 0, PD(in), PD(tok), PL(vc), PI(vc), PL(Na));
    if (pr_gemm(r, tok, vc, &tw->vin_w, 0, tw->vin_b, FO(x, (size_t)nreg * C), C, Na, C, vc, 0)) return -1;
    /* text MLP: Linear -> GELU -> Linear -> zero-centred RMSNorm */
    {
        CUdeviceptr t1 = pr_ws(r, WS_FA, (size_t)S * 4 * C), t2 = pr_ws(r, WS_TMP, (size_t)S * C);
        if (!t1 || !t2) return -1;
        PRC(cuMemcpyHtoD(in, text, F4((size_t)S * c->text_cin)));
        if (pr_gemm(r, in, c->text_cin, &tw->tx0_w, 0, tw->tx0_b, t1, 4 * C, S, 4 * C, c->text_cin, 0)) return -1;
        if (pr_actk(r, t1, t1, (long long)S * 4 * C, 0)) return -1;
        if (pr_gemm(r, t1, 4 * C, &tw->tx2_w, 0, tw->tx2_b, t2, C, S, C, 4 * C, 0)) return -1;
        if (pr_rms(r, t2, C, FO(x, (size_t)(nreg + Na) * C), C, tw->tx_norm, 1, 1e-6f, 1.0f, C, S, 1)) return -1;
    }
    /* noised + clean tokens, then per-chunk timestep modulation (clean rows use t = 0) */
    PRC(cuMemcpyHtoD(in, noised, F4((size_t)vc * Nn)));
    LAUNCH(r->f_chw, NBLK((long long)Nn * vc), 1, 1, 256, 0, PD(in), PD(tok), PL(vc), PI(vc), PL(Nn));
    if (pr_gemm(r, tok, vc, &tw->vin_w, 0, tw->vin_b, FO(x, (size_t)Ng * C), C, Nn, C, vc, 0)) return -1;
    if (Nc > 0) {
        PRC(cuMemcpyHtoD(in, clean, F4((size_t)vc * Nc)));
        LAUNCH(r->f_chw, NBLK((long long)Nc * vc), 1, 1, 256, 0, PD(in), PD(tok), PL(vc), PI(vc), PL(Nc));
        if (pr_gemm(r, tok, vc, &tw->vin_w, 0, tw->vin_b, FO(x, (size_t)(Ng + Nn) * C), C, Nc, C, vc, 0)) return -1;
    }
    {
        float *tv = (float *)calloc(K + 1, 4);
        for (int k = 0; k < K; k++) tv[k] = t[k];
        CUdeviceptr md = pr_ws(r, WS_MOD, (size_t)(K + 1) * 2 * C);
        if (!md || pr_tproj(r, tw, tv, K + 1, md)) { free(tv); return -1; }
        free(tv);
        int *mr = (int *)malloc((size_t)(Nn + Nc) * 4);
        for (int n = 0; n < Nn; n++) mr[n] = n / (F * tpf);
        for (int n = 0; n < Nc; n++) mr[Nn + n] = K;
        CUdeviceptr dm = pr_ws(r, WS_MROW, Nn + Nc);
        cuMemcpyHtoD(dm, mr, (size_t)(Nn + Nc) * 4);
        free(mr);
        LAUNCH(r->f_modulate, NBLK((long long)(Nn + Nc) * C), 1, 1, 256, 0, PD(FO(x, (size_t)Ng * C)), PD(FO(x, (size_t)Ng * C)), PD(md), PL(2 * C), PD(dm), PI(C), PL(Nn + Nc));
    }
    /* positions (chunk_causal_positions) and teacher-forcing codes */
    free(es->pos);
    es->pos = (long long *)malloc((size_t)T * 8);
    int *code = (int *)calloc(T, 4);
    {
        long long FS = c->frame_stride, RS = c->row_stride;
        int o = 0;
        for (int i = 0; i < nreg; i++) es->pos[o++] = 0;
        for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++) es->pos[o++] = 1 + y * RS + xx;
        for (int s = 0; s < S; s++) es->pos[o++] = 1 + s + FS;
        for (int f = 0; f < K * F; f++) for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++) {
            code[o] = 2 * (f / F) + 2;
            es->pos[o++] = 2 * FS + 1 + f * FS + y * RS + xx;
        }
        for (int n = 0; n < Nc; n++) { code[o] = 2 * ((n / tpf) / F) + 1; es->pos[o] = es->pos[Ng + n]; o++; }
    }
    if (pr_up_rope(r, es->pos, T, hd, c->rope_eff, c->rope_base, c->rope_tab, WS_COSQ, WS_SINQ)) return -1;
    CUdeviceptr dcode = pr_ws(r, WS_CODEQ, T);
    cuMemcpyHtoD(dcode, code, (size_t)T * 4);
    free(code);
    /* persistent encoder outputs */
    size_t sz = F4((size_t)T * C);
    if (pr_persist(r, &es->emb, &es->emb_cap, sz)) return -1;
    for (int j = 0; j < c->enc_ntap; j++) if (pr_persist(r, &es->taps[j], &es->taps_cap[j], sz)) return -1;
    if (pr_persist(r, &es->cond, &es->cond_cap, F4((size_t)Nn * C))) return -1;
    PRC(cuMemcpyDtoD(es->emb, x, sz));
    pr_dump(r, "enc_embed", x, T, C);
    *xs = x; *dcode_out = dcode;
    return 0;
}

static int pr_enc_finish(praxis_runner *r, pr_enc_state *es, CUdeviceptr x, int row0) {
    const pr_cfg *c = &r->c;
    int C = c->dim;
    if (pr_gemm(r, FO(x, (size_t)row0 * C), C, &r->enc.out_w, 0, r->enc.out_b, es->cond, C, es->Nn, C, C, 0)) return -1;
    pr_dump(r, "enc_cond", es->cond, es->Nn, C);
    return 0;
}

int praxis_encoder(praxis_runner *r, const float *text, int S, const float *anchor, int h, int w,
                   const float *noised, const float *clean, const float *t, int K) {
    cuCtxSetCurrent(r->ctx);
    pr_enc_state *es = r->esp;
    CUdeviceptr x, dcode;
    if (pr_enc_prepare(r, es, 1, 0, text, S, anchor, h, w, noised, clean, t, K, &x, &dcode)) return -1;
    CUdeviceptr *tp[1] = { es->taps };
    if (pr_stack(r, &r->enc, "enc", x, es->T, 1, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dcode, 0, 0, 0, r->c.enc_taps, r->c.enc_ntap, tp)) return -1;
    return pr_enc_finish(r, es, x, es->Ng);
}

/* Two encoder passes (cond text, uncond text) in lockstep: every layer's routed experts are read once for both.
 * Leaves the cond state in es[0] and the uncond state in es[1]; select with praxis_select_state. */
int praxis_encoder_pair(praxis_runner *r, const float *text, const float *text_u, int S, const float *anchor, int h, int w,
                        const float *noised, const float *clean, const float *t, int K) {
    cuCtxSetCurrent(r->ctx);
    if (!r->fast) { fprintf(stderr, "praxis: paired encoder passes need the fast path\n"); return -1; }
    CUdeviceptr x0, x1, dcode, dcode1;
    if (pr_enc_prepare(r, &r->es_store[0], 2, 0, text, S, anchor, h, w, noised, clean, t, K, &x0, &dcode)) return -1;
    if (pr_enc_prepare(r, &r->es_store[1], 2, 1, text_u, S, anchor, h, w, noised, clean, t, K, &x1, &dcode1)) return -1;
    CUdeviceptr *tp[2] = { r->es_store[0].taps, r->es_store[1].taps };
    if (pr_stack(r, &r->enc, "enc", x0, r->es_store[0].T, 2, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dcode, 0, 0, 0, r->c.enc_taps, r->c.enc_ntap, tp)) return -1;
    if (pr_enc_finish(r, &r->es_store[0], x0, r->es_store[0].Ng) || pr_enc_finish(r, &r->es_store[1], x1, r->es_store[1].Ng)) return -1;
    return 0;
}
void praxis_select_state(praxis_runner *r, int i) { r->esp = &r->es_store[i ? 1 : 0]; }

/* ---- encoder with the context cache (praxis ctx_cache) ----
 * First call of a sampler run (kvc invalid): full pass, the context K/V of every layer are stored. Later calls: only the noised rows are
 * embedded / run through the stack; their keys/values are appended to the cached [globals | clean] ones, attention is rectangular
 * (queries = noised rows). Everything row-wise (norms, MoE routing, FFN) is per token, so the result equals the full pass (up to the
 * different GEMM M shapes: cuBLAS may pick other tile configs). */
static int pr_kvc_matches(const pr_kvc *kc, const pr_enc_state *es, int K, int S) {
    return kc->valid && kc->K == K && kc->S == S && kc->Nn == es->Nn && kc->Ng == es->Ng && kc->Nc == es->Nc && es->T == es->Ng + es->Nn + es->Nc;
}
static void pr_kvc_free(praxis_runner *r, pr_kvc *kc) {
    for (int i = 0; i < PR_MAXD; i++) {
        if (kc->k[i]) { pr_mfree(r, kc->k[i], kc->cap[i]); pr_mfree(r, kc->v[i], kc->cap[i]); kc->k[i] = kc->v[i] = 0; kc->cap[i] = 0; }
    }
    kc->valid = 0;
}
void praxis_set_ctx_cache(praxis_runner *r, int on) {
    cuCtxSetCurrent(r->ctx);
    r->ctx_cache = on && r->fast;
    if (!on) for (int b = 0; b < 2; b++) pr_kvc_free(r, &r->kvc[b]);
}
static void pr_ctx_invalidate(praxis_runner *r) { r->kvc[0].valid = 0; r->kvc[1].valid = 0; }

/* embeds the noised rows of sequence idx of an nb-sequence batch: x = WS_X rows [idx*R, (idx+1)*R) */
static int pr_enc_prepare_nz(praxis_runner *r, pr_enc_state *es, int nb, int idx, const float *noised, const float *t, int K,
                             CUdeviceptr *xs, CUdeviceptr *qcode_out) {
    const pr_cfg *c = &r->c;
    pr_tower *tw = &r->enc;
    int C = c->dim, F = c->lat, vc = c->enc_vcin, tpf = es->tpf, hd = c->head_dim;
    int Nn = es->Nn, Ng = es->Ng, Nc = es->Nc;
    CUdeviceptr x = pr_ws(r, WS_X, (size_t)Nn * C * nb);
    if (x) x += F4((size_t)idx * Nn * C);
    CUdeviceptr in = pr_ws(r, WS_IN, (size_t)vc * Nn), tok = pr_ws(r, WS_TOK, (size_t)vc * Nn);
    if (!x || !in || !tok) return -1;
    PRC(cuMemcpyHtoD(in, noised, F4((size_t)vc * Nn)));
    LAUNCH(r->f_chw, NBLK((long long)Nn * vc), 1, 1, 256, 0, PD(in), PD(tok), PL(vc), PI(vc), PL(Nn));
    if (pr_gemm(r, tok, vc, &tw->vin_w, 0, tw->vin_b, x, C, Nn, C, vc, 0)) return -1;
    {
        float *tv = (float *)calloc(K + 1, 4);
        for (int k = 0; k < K; k++) tv[k] = t[k];
        CUdeviceptr md = pr_ws(r, WS_MOD, (size_t)(K + 1) * 2 * C);
        if (!md || pr_tproj(r, tw, tv, K + 1, md)) { free(tv); return -1; }
        free(tv);
        int *mr = (int *)malloc((size_t)Nn * 4);
        for (int n = 0; n < Nn; n++) mr[n] = n / (F * tpf);
        CUdeviceptr dm = pr_ws(r, WS_MROW, Nn);
        cuMemcpyHtoD(dm, mr, (size_t)Nn * 4);
        free(mr);
        LAUNCH(r->f_modulate, NBLK((long long)Nn * C), 1, 1, 256, 0, PD(x), PD(x), PD(md), PL(2 * C), PD(dm), PI(C), PL(Nn));
    }
    /* rope of the noised rows; q codes (noised); k codes [globals 0 | clean | noised] */
    if (pr_up_rope(r, es->pos + Ng, Nn, hd, c->rope_eff, c->rope_base, c->rope_tab, WS_COSQ, WS_SINQ)) return -1;
    int Nctx = Ng + Nc;
    int *qc = (int *)malloc((size_t)Nn * 4), *kc = (int *)calloc((size_t)Nctx + Nn, 4);
    for (int n = 0; n < Nn; n++) qc[n] = 2 * ((n / tpf) / F) + 2;
    for (int n = 0; n < Nc; n++) kc[Ng + n] = 2 * ((n / tpf) / F) + 1;
    for (int n = 0; n < Nn; n++) kc[Nctx + n] = qc[n];
    CUdeviceptr dq = pr_ws(r, WS_CODEQ, Nn), dk = pr_ws(r, WS_KCODE, (size_t)Nctx + Nn);
    if (!dq || !dk) { free(qc); free(kc); return -1; }
    cuMemcpyHtoD(dq, qc, (size_t)Nn * 4); cuMemcpyHtoD(dk, kc, ((size_t)Nctx + Nn) * 4);
    free(qc); free(kc);
    r->kx_kcode = dk; r->kx_nk = Nctx + Nn; r->kx_Ng = Ng; r->kx_Nn = Nn;
    PRC(cuMemcpyDtoD(es->emb + F4((size_t)Ng * C), x, F4((size_t)Nn * C)));
    *xs = x; *qcode_out = dq;
    return 0;
}
static void pr_kvc_setup(pr_kvc *kc, const pr_enc_state *es, int K, int S) {
    kc->K = K; kc->S = S; kc->Ng = es->Ng; kc->Nn = es->Nn; kc->Nc = es->Nc; kc->Nctx = es->Ng + es->Nc;
}
/* branch = which es_store / kvc the (single) pass uses. use_cache: sampler run with ctx_cache on. */
static int pr_encoder_x(praxis_runner *r, const float *text, int S, const float *anchor, int h, int w, const float *noised,
                        const float *clean, const float *t, int K, int branch) {
    if (!(r->ctx_cache && r->fast)) return praxis_encoder(r, text, S, anchor, h, w, noised, clean, t, K);
    cuCtxSetCurrent(r->ctx);
    pr_enc_state *es = &r->es_store[branch]; pr_kvc *kc = &r->kvc[branch];
    r->esp = es;
    CUdeviceptr x, dcode;
    CUdeviceptr *tp[1] = { es->taps };
    int rc;
    if (!(kc->valid && es->T > 0 && kc->K == K && kc->S == S && kc->Nn == K * r->c.lat * h * w)) {
        if (pr_enc_prepare(r, es, 1, 0, text, S, anchor, h, w, noised, clean, t, K, &x, &dcode)) return -1;
        pr_kvc_setup(kc, es, K, S);
        r->kx_mode = 1; r->kx_seq[0] = kc; r->kx_seq[1] = NULL;
        rc = pr_stack(r, &r->enc, "enc", x, es->T, 1, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dcode, 0, 0, 0, r->c.enc_taps, r->c.enc_ntap, tp);
        r->kx_mode = 0;
        if (rc) return -1;
        kc->valid = 1;
        return pr_enc_finish(r, es, x, es->Ng);
    }
    if (pr_enc_prepare_nz(r, es, 1, 0, noised, t, K, &x, &dcode)) return -1;
    r->kx_mode = 2; r->kx_seq[0] = kc;
    rc = pr_stack(r, &r->enc, "enc", x, es->Nn, 1, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dcode, 0, 0, 0, r->c.enc_taps, r->c.enc_ntap, tp);
    r->kx_mode = 0;
    if (rc) return -1;
    return pr_enc_finish(r, es, x, 0);
}
static int pr_encoder_pair_x(praxis_runner *r, const float *text, const float *text_u, int S, const float *anchor, int h, int w,
                             const float *noised, const float *clean, const float *t, int K) {
    if (!(r->ctx_cache && r->fast)) return praxis_encoder_pair(r, text, text_u, S, anchor, h, w, noised, clean, t, K);
    cuCtxSetCurrent(r->ctx);
    pr_enc_state *e0 = &r->es_store[0], *e1 = &r->es_store[1];
    pr_kvc *k0 = &r->kvc[0], *k1 = &r->kvc[1];
    int Nn = K * r->c.lat * h * w;
    CUdeviceptr x0, x1, dc, dc1;
    CUdeviceptr *tp[2] = { e0->taps, e1->taps };
    int rc;
    if (!(k0->valid && k1->valid && e0->T > 0 && e1->T > 0 && k0->K == K && k0->S == S && k0->Nn == Nn)) {
        if (pr_enc_prepare(r, e0, 2, 0, text, S, anchor, h, w, noised, clean, t, K, &x0, &dc)) return -1;
        if (pr_enc_prepare(r, e1, 2, 1, text_u, S, anchor, h, w, noised, clean, t, K, &x1, &dc1)) return -1;
        pr_kvc_setup(k0, e0, K, S); pr_kvc_setup(k1, e1, K, S);
        r->kx_mode = 1; r->kx_seq[0] = k0; r->kx_seq[1] = k1;
        tp[0] = e0->taps; tp[1] = e1->taps;
        rc = pr_stack(r, &r->enc, "enc", x0, e0->T, 2, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dc, 0, 0, 0, r->c.enc_taps, r->c.enc_ntap, tp);
        r->kx_mode = 0;
        if (rc) return -1;
        k0->valid = k1->valid = 1;
        return pr_enc_finish(r, e0, x0, e0->Ng) || pr_enc_finish(r, e1, x1, e1->Ng) ? -1 : 0;
    }
    if (pr_enc_prepare_nz(r, e0, 2, 0, noised, t, K, &x0, &dc) || pr_enc_prepare_nz(r, e1, 2, 1, noised, t, K, &x1, &dc1)) return -1;
    r->kx_mode = 2; r->kx_seq[0] = k0; r->kx_seq[1] = k1;
    rc = pr_stack(r, &r->enc, "enc", x0, Nn, 2, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dc, 0, 0, 0, r->c.enc_taps, r->c.enc_ntap, tp);
    r->kx_mode = 0;
    if (rc) return -1;
    return pr_enc_finish(r, e0, x0, 0) || pr_enc_finish(r, e1, x1, 0) ? -1 : 0;
}



/* ---- decoder ---- */
int praxis_decoder(praxis_runner *r, const float *noised, const float *t, float *vel_out) {
    cuCtxSetCurrent(r->ctx);
    const pr_cfg *c = &r->c;
    pr_enc_state *es = r->esp;
    pr_tower *tw = &r->dec;
    int C = c->dim, F = c->lat, K = es->K, h = es->h, w = es->w, ph = c->dec_ph, pw = c->dec_pw;
    int Hd = h * ph, Wd = w * pw, tpfd = Hd * Wd, Nd = K * F * tpfd, nreg = c->n_reg, Td = nreg + Nd;
    int vc = c->enc_vcin, vcin = c->dec_vcin, hd = c->head_dim, Nn = es->Nn, tpf = es->tpf;
    if (vcin * ph * pw != vc) { fprintf(stderr, "praxis: decoder patch %d*%d*%d != video channels %d\n", vcin, ph, pw, vc); return -1; }
    if (c->dec_mod_dim != C) { fprintf(stderr, "praxis: decoder modulation_dim != dim not supported\n"); return -1; }
    CUdeviceptr x = pr_ws(r, WS_X, (size_t)Td * C);
    CUdeviceptr in = pr_ws(r, WS_IN, (size_t)vc * K * F * tpf), tok = pr_ws(r, WS_TOK, (size_t)Nd * (vcin > C ? vcin : C));
    if (!x || !in || !tok) return -1;
    PRC(cuMemcpyDtoD(x, tw->registers, F4((size_t)nreg * C)));
    PRC(cuMemcpyHtoD(in, noised, F4((size_t)vc * K * F * tpf)));
    LAUNCH(r->f_unpatch, NBLK((long long)Nd * vcin), 1, 1, 256, 0, PD(in), PD(tok), PI(vcin), PI(ph), PI(pw), PI(K * F), PI(h), PI(w));
    if (pr_gemm(r, tok, vcin, &tw->vin_w, 0, tw->vin_b, FO(x, (size_t)nreg * C), C, Nd, C, vcin, 0)) return -1;
    {
        CUdeviceptr md = pr_ws(r, WS_MOD, (size_t)K * 2 * C);
        if (!md || pr_tproj(r, tw, t, K, md)) return -1;
        int *mr = (int *)malloc((size_t)Nd * 4);
        for (int n = 0; n < Nd; n++) mr[n] = n / (F * tpfd);
        CUdeviceptr dm = pr_ws(r, WS_MROW, Nd);
        cuMemcpyHtoD(dm, mr, (size_t)Nd * 4);
        free(mr);
        LAUNCH(r->f_modulate, NBLK((long long)Nd * C), 1, 1, 256, 0, PD(FO(x, (size_t)nreg * C)), PD(FO(x, (size_t)nreg * C)), PD(md), PL(2 * C), PD(dm), PI(C), PL(Nd));
    }
    /* positions + codes */
    long long *pos = (long long *)malloc((size_t)Td * 8);
    int *code = (int *)calloc(Td, 4);
    {
        long long FS = c->frame_stride, RS = c->row_stride;
        int o = 0;
        for (int i = 0; i < nreg; i++) pos[o++] = 0;
        for (int f = 0; f < K * F; f++) for (int y = 0; y < Hd; y++) for (int xx = 0; xx < Wd; xx++) {
            code[o] = 2 * (f / F) + 2; pos[o++] = 1 + f * FS + y * RS + xx; }
    }
    int rc = pr_up_rope(r, pos, Td, hd, c->rope_eff, c->rope_base, c->rope_tab, WS_COSQ, WS_SINQ);
    free(pos);
    CUdeviceptr dcode = pr_ws(r, WS_CODEQ, Td);
    cuMemcpyHtoD(dcode, code, (size_t)Td * 4);
    free(code);
    if (rc) return -1;
    /* modulation rows: encoder conditioning (nearest-upsampled by index) + one zero row for the registers */
    CUdeviceptr mi = pr_ws(r, WS_MOD, (size_t)(Nn + 1) * C), mrow2 = pr_ws(r, WS_MROW, Td);
    if (!mi || !mrow2) return -1;
    PRC(cuMemcpyDtoD(mi, es->cond, F4((size_t)Nn * C)));
    PRC(cuMemcpyDtoD(FO(mi, (size_t)Nn * C), r->zero_row, F4(C)));
    {
        int *mr = (int *)malloc((size_t)Td * 4);
        int o = 0;
        for (int i = 0; i < nreg; i++) mr[o++] = Nn;
        for (int f = 0; f < K * F; f++) for (int y = 0; y < Hd; y++) for (int xx = 0; xx < Wd; xx++)
            mr[o++] = f * tpf + (y / ph) * w + xx / pw;
        cuMemcpyHtoD(mrow2, mr, (size_t)Td * 4);
        free(mr);
    }
    pr_dump(r, "dec_embed", x, Td, C);
    if (pr_stack(r, tw, "dec", x, Td, 1, pr_ws(r, WS_COSQ, 0), pr_ws(r, WS_SINQ, 0), dcode, mi, Nn + 1, mrow2, NULL, 0, NULL)) return -1;
    CUdeviceptr nrm = pr_ws(r, WS_TMP, (size_t)Nd * C);
    if (!nrm) return -1;
    if (pr_rms(r, FO(x, (size_t)nreg * C), C, nrm, C, tw->out_norm, 1, 1e-6f, 1.0f, C, Nd, 1)) return -1;
    if (pr_gemm(r, nrm, C, &tw->out_w, 0, tw->out_b, tok, vcin, Nd, vcin, C, 0)) return -1;
    LAUNCH(r->f_patch, NBLK((long long)vc * K * F * tpf), 1, 1, 256, 0, PD(tok), PD(in), PI(vcin), PI(ph), PI(pw), PI(K * F), PI(h), PI(w));
    PRC(cuMemcpyDtoH(vel_out, in, F4((size_t)vc * K * F * tpf)));
    return 0;
}

/* ---- action expert ---- */
/* t values -> [n][A] via the sinusoid features and the 2-layer MLP; result into out (row stride ldo) */
static int pr_tembed(praxis_runner *r, const float *tv, int n, const pr_mat *w0, CUdeviceptr b0, const pr_mat *w2, CUdeviceptr b2,
                     CUdeviceptr out, long long ldo) {
    int A = r->c.ae_dim;
    CUdeviceptr dt = pr_ws(r, WS_AT, n), feat = pr_ws(r, WS_TMP, (size_t)n * 256), hh = pr_ws(r, WS_AH, (size_t)n * A);
    if (!dt || !feat || !hh) return -1;
    cuMemcpyHtoD(dt, tv, F4(n));
    LAUNCH(r->f_sincos, NBLK((long long)n * 128), 1, 1, 256, 0, PD(dt), PI(n), PD(r->ae.freq256), PI(128), PD(feat));
    if (pr_gemm(r, feat, 256, w0, 0, b0, hh, A, n, A, 256, 0)) return -1;
    if (pr_actk(r, hh, hh, (long long)n * A, 1)) return -1;
    return pr_gemm(r, hh, A, w2, 0, b2, out, ldo, n, A, A, 0);
}

static int pr_noised_offset(int n_noised, int n_clean, int cs) {
    if (n_clean == 0) return 0;
    if (n_noised == 1) return n_clean;
    if (cs > 1 && n_noised == cs) return n_clean;
    if (n_noised == n_clean + 1) return 0;
    if (cs > 1 && n_noised == n_clean + cs) return 0;
    return -1;
}

int praxis_expert(praxis_runner *r, const float *noised_a, const float *t, const float *proprio, float *out) {
    cuCtxSetCurrent(r->ctx);
    const pr_cfg *c = &r->c;
    pr_enc_state *es = r->esp;
    pr_ae *ae = &r->ae;
    int K = es->K, F = c->lat, cs = F, A = c->ae_dim, T = K * F, nf = c->nfut, C = c->dim;
    int Ct = c->trunk_dim, hd = c->ae_hd, NH = c->ae_heads, inner = NH * hd, tpf = es->tpf;
    int Q = K + K * nf + T;
    int n_c = (K - 1) * F, n_n = K * F;
    int G = es->Ng, Nf = es->Nn + es->Nc, KV = G + Nf;
    if (Ct != C) { fprintf(stderr, "praxis: trunk_dim != dim not supported\n"); return -1; }
    int off = pr_noised_offset(n_n, n_c, cs);
    if (off < 0) { fprintf(stderr, "praxis: unsupported expert window layout\n"); return -1; }
    CUdeviceptr x = pr_ws(r, WS_AX, (size_t)Q * A), cond = pr_ws(r, WS_ACOND, (size_t)Q * A);
    CUdeviceptr hbuf = pr_ws(r, WS_N1, (size_t)Q * A), qb = pr_ws(r, WS_Q, (size_t)Q * inner), kvb = pr_ws(r, WS_K, (size_t)(KV > Q ? KV : Q) * 2 * inner),
        ao = pr_ws(r, WS_AO, (size_t)Q * inner), scl = pr_ws(r, WS_G, (size_t)Q * A), ub = pr_ws(r, WS_FA, (size_t)Q * 8 * A),
        gb = pr_ws(r, WS_FB, (size_t)Q * 4 * A), kvs = pr_ws(r, WS_AKV, (size_t)KV * A), gtok = pr_ws(r, WS_PROJ, (size_t)G * C),
        tkv = pr_ws(r, WS_N2, (size_t)G * A), fraw = pr_ws(r, WS_V, (size_t)Nf * C), fnorm = pr_ws(r, WS_FH, (size_t)Nf * C);
    if (!(x && cond && hbuf && qb && kvb && ao && scl && ub && gb && kvs && gtok && tkv && fraw && fnorm)) return -1;
    /* --- query sequence [proprio | future | actions] --- */
    float *tq = (float *)malloc(F4(Q)), *tact = (float *)malloc(F4(T));
    for (int i = 0; i < T; i++) tact[i] = t[i / cs];
    int *qchunk = (int *)malloc((size_t)Q * 4);
    for (int k = 0; k < K; k++) qchunk[k] = k;
    for (int k = 0; k < K; k++) for (int f = 0; f < nf; f++) qchunk[K + k * nf + f] = k;
    for (int i = 0; i < T; i++) qchunk[K + K * nf + i] = i / cs;
    for (int q = 0; q < Q; q++) { int fr = qchunk[q] * cs; if (fr < 0) fr = 0; if (fr > T - 1) fr = T - 1; tq[q] = tact[fr]; }
    /* actions: ActionTauEncoder */
    {
        CUdeviceptr in = pr_ws(r, WS_IN, (size_t)T * c->act_dim > (size_t)K * c->prop_dim ? (size_t)T * c->act_dim : (size_t)K * c->prop_dim), cat = pr_ws(r, WS_MOD, (size_t)T * 2 * A), mid = pr_ws(r, WS_MOD2, (size_t)T * A);
        if (!in || !cat || !mid) return -1;
        PRC(cuMemcpyHtoD(in, noised_a, F4((size_t)T * c->act_dim)));
        if (pr_gemm(r, in, c->act_dim, &ae->w1, 0, ae->w1_b, cat, 2 * A, T, A, c->act_dim, 0)) return -1;
        if (pr_tembed(r, tact, T, &ae->tau0, ae->tau0_b, &ae->tau2, ae->tau2_b, FO(cat, A), 2 * A)) return -1;
        if (pr_gemm(r, cat, 2 * A, &ae->w2, 0, ae->w2_b, mid, A, T, A, 2 * A, 0)) return -1;
        if (pr_actk(r, mid, mid, (long long)T * A, 1)) return -1;
        if (pr_gemm(r, mid, A, &ae->w3, 0, ae->w3_b, FO(x, (size_t)(K + K * nf) * A), A, T, A, A, 0)) return -1;
        /* proprio tokens: proprio_token + embed_proprio(first latent of each chunk) */
        PRC(cuMemcpyHtoD(in, proprio, F4((size_t)K * c->prop_dim)));
        if (pr_gemm(r, in, c->prop_dim, &ae->embp, 0, ae->embp_b, x, A, K, A, c->prop_dim, 0)) return -1;
        LAUNCH(r->f_addrow, NBLK((long long)K * A), 1, 1, 256, 0, PD(x), PD(ae->proprio_tok), PL(K), PI(A));
        LAUNCH(r->f_future, NBLK((long long)K * nf * A), 1, 1, 256, 0, PD(FO(x, (size_t)K * A)), PD(ae->future), PD(ae->chunk_embed), PI(K), PI(nf), PI(A));
    }
    /* conditioning: timestep embedder of each query token's chunk time */
    if (pr_tembed(r, tq, Q, &ae->temb0, ae->temb0_b, &ae->temb2, ae->temb2_b, cond, A)) return -1;
    if (pr_actk(r, cond, pr_ws(r, WS_AH, (size_t)Q * A), (long long)Q * A, 1)) return -1;   /* silu(cond) -> AH */
    CUdeviceptr silu_c = pr_ws(r, WS_AH, (size_t)Q * A);
    pr_dump(r, "ae_x0", x, Q, A);
    /* --- rope ids, tables, masks --- */
    long long *kid = (long long *)malloc((size_t)KV * 8), *qid = (long long *)malloc((size_t)Q * 8);
    {
        int o = 0;
        int gs[3] = { 0, c->n_reg, c->n_reg + es->n_anc }, gl[3] = { c->n_reg, es->n_anc, es->S };
        for (int g = 0; g < 3; g++) { int kd = c->gorder[g]; for (int i = 0; i < gl[kd]; i++) kid[o++] = es->pos[gs[kd] + i]; }
        for (int i = 0; i < es->Nc; i++) kid[o++] = es->pos[es->Ng + es->Nn + i];
        for (int i = 0; i < es->Nn; i++) kid[o++] = es->pos[es->Ng + i];
        long long *fid = (long long *)malloc((size_t)(n_c + n_n) * 8);
        for (int f = 0; f < n_c + n_n; f++) { long long m = kid[G + (long long)f * tpf]; for (int i = 1; i < tpf; i++) if (kid[G + (long long)f * tpf + i] < m) m = kid[G + (long long)f * tpf + i]; fid[f] = m; }
        for (int q = 0; q < Q; q++) {
            int qf = q < K + K * nf ? qchunk[q] * cs : q - (K + K * nf);
            if (qf < 0) qf = 0;
            qid[q] = fid[n_c + qf];
        }
        free(fid);
    }
    if (pr_up_rope(r, qid, Q, c->ae_rope_dim, c->ae_rope_eff, c->ae_rope_base, c->ae_rope_tab, WS_COSQ, WS_SINQ)) return -1;
    if (pr_up_rope(r, kid, KV, c->ae_rope_dim, c->ae_rope_eff, c->ae_rope_base, c->ae_rope_tab, WS_COSK, WS_SINK)) return -1;
    CUdeviceptr dcq = pr_ws(r, WS_COSQ, 0), dsq = pr_ws(r, WS_SINQ, 0), dck = pr_ws(r, WS_COSK, 0), dsk = pr_ws(r, WS_SINK, 0);
    int use_iv = r->fast && r->ae_flash && (hd == 16 || hd == 32 || hd == 64 || hd == 128) && (inner % 8) == 0 && ae->b[0].to_q.dt == 1 && ae->b[0].to_out.dt == 1;
    unsigned char *mc = NULL, *ms = NULL;
    int *ivc = (int *)malloc((size_t)Q * 6 * 4), *ivs = (int *)malloc((size_t)Q * 6 * 4);
    {
        int window = n_c + n_n, k0 = K;
        if (!use_iv) { mc = (unsigned char *)malloc((size_t)Q * KV); ms = (unsigned char *)malloc((size_t)Q * Q); }
        for (int q = 0; q < Q; q++) {
            int qch = q < k0 ? q : q < k0 * (1 + nf) ? (q - k0) / (nf > 0 ? nf : 1) : (q - k0 * (1 + nf)) / cs;
            int qst = qch * cs + off;
            /* the allowed keys of a query are at most 3 contiguous intervals: cross = [globals | clean frames in window | noised frames of its chunk],
             * self = [its proprio token | its future tokens | its action tokens] */
            {
                int clo = qst - window; if (clo < 0) clo = 0;
                int chi = qst < n_c ? qst : n_c;
                int *v = ivc + (size_t)q * 6;
                v[0] = 0; v[1] = G;
                if (chi > clo) { v[2] = G + clo * tpf; v[3] = G + chi * tpf; } else { v[2] = 0; v[3] = 0; }
                int n0 = qch * cs, n1 = (qch + 1) * cs < n_n ? (qch + 1) * cs : n_n;
                if (!c->clean_only && n1 > n0) { v[4] = G + (n_c + n0) * tpf; v[5] = G + (n_c + n1) * tpf; } else { v[4] = 0; v[5] = 0; }
                int *u = ivs + (size_t)q * 6;
                u[0] = qch; u[1] = qch + 1;
                u[2] = k0 + qch * nf; u[3] = k0 + (qch + 1) * nf;
                u[4] = k0 * (1 + nf) + qch * cs; u[5] = k0 * (1 + nf) + (qch + 1) * cs;
            }
            if (use_iv) continue;
            for (int j = 0; j < KV; j++) {
                int ok;
                if (j < G) ok = 1;
                else {
                    int sp = j - G, fr = sp / tpf;
                    if (fr < n_c) ok = (fr < qst) && (fr >= qst - window);
                    else { int nfr = fr - n_c; ok = !c->clean_only && (nfr / cs == qch); }
                }
                mc[(size_t)q * KV + j] = (unsigned char)ok;
            }
            for (int j = 0; j < Q; j++) {
                int jch = j < k0 ? j : j < k0 * (1 + nf) ? (j - k0) / (nf > 0 ? nf : 1) : (j - k0 * (1 + nf)) / cs;
                ms[(size_t)q * Q + j] = (unsigned char)(jch == qch);
            }
        }
    }
    CUdeviceptr dmc = pr_ws(r, WS_MASK, use_iv ? (size_t)Q * 6 + 1 : ((size_t)Q * KV + 3) / 4 + 1), dms = pr_ws(r, WS_CODEK, use_iv ? (size_t)Q * 6 + 1 : ((size_t)Q * Q + 3) / 4 + 1);
    if (!dmc || !dms) return -1;
    if (use_iv) { cuMemcpyHtoD(dmc, ivc, (size_t)Q * 24); cuMemcpyHtoD(dms, ivs, (size_t)Q * 24); }
    else { cuMemcpyHtoD(dmc, mc, (size_t)Q * KV); cuMemcpyHtoD(dms, ms, (size_t)Q * Q); }
    free(mc); free(ms); free(ivc); free(ivs); free(kid); free(qid); free(tq); free(tact); free(qchunk);
    /* --- global tokens (block-0 input, in global_order) -> text_proj --- */
    {
        int gs[3] = { 0, c->n_reg, c->n_reg + es->n_anc }, gl[3] = { c->n_reg, es->n_anc, es->S };
        int o = 0;
        for (int g = 0; g < 3; g++) { int kd = c->gorder[g]; if (pr_copyrows(r, FO(es->emb, (size_t)gs[kd] * C), C, FO(gtok, (size_t)o * C), C, gl[kd], C)) return -1; o += gl[kd]; }
        if (pr_gemm(r, gtok, C, &ae->text_proj, 0, 0, tkv, A, G, A, C, 0)) return -1;
    }
    int ci = 0;
    char nm[128];
    for (int bi = 0; bi < c->ae_layers; bi++) {
        pr_ae_blk *b = &ae->b[bi];
        int cross = (bi % 2 == 0);
        /* adaln_attn */
        if (pr_gemm(r, silu_c, A, &b->ada_a, 0, b->ada_a_b, scl, A, Q, A, A, 0)) return -1;
        LAUNCH(r->f_adaln, Q, 1, 1, A >= 128 ? 128 : 32, 0, PD(x), PD(scl), PD(hbuf), PF(1e-4), PI(A));
        if (pr_gemm(r, hbuf, A, &b->to_q, 0, 0, qb, inner, Q, inner, A, 0)) return -1;
        int nkv;
        CUdeviceptr ktab_c, ktab_s;
        if (cross) {
            /* kv source = [text_kv | tap_proj(rms_norm(tap) * tap_norm_scale)] (clean frames first, then noised) */
            CUdeviceptr tap = es->taps[ci++];
            if (pr_copyrows(r, FO(tap, (size_t)(es->Ng + es->Nn) * C), C, fraw, C, es->Nc, C)) return -1;
            if (pr_copyrows(r, FO(tap, (size_t)es->Ng * C), C, FO(fraw, (size_t)es->Nc * C), C, es->Nn, C)) return -1;
            if (pr_rms(r, fraw, C, fnorm, C, c->tap_norm ? ae->tap_scale : 0, 0, 1e-4f, 1.0f, C, Nf, 1)) return -1;
            PRC(cuMemcpyDtoD(kvs, tkv, F4((size_t)G * A)));
            if (pr_gemm(r, fnorm, C, &ae->tap_proj, 0, 0, FO(kvs, (size_t)G * A), A, Nf, A, C, 0)) return -1;
            if (pr_gemm(r, kvs, A, &b->to_kv, 0, 0, kvb, 2 * inner, KV, 2 * inner, A, 0)) return -1;
            nkv = KV; ktab_c = dck; ktab_s = dsk;
        } else {
            if (pr_gemm(r, hbuf, A, &b->to_kv, 0, 0, kvb, 2 * inner, Q, 2 * inner, A, 0)) return -1;
            nkv = Q; ktab_c = dcq; ktab_s = dsq;
        }
        if (pr_ropek(r, qb, inner, NH, hd, dcq, dsq, Q)) return -1;
        if (pr_ropek(r, kvb, 2 * inner, NH, hd, ktab_c, ktab_s, nkv)) return -1;
        if (pr_rms(r, qb, inner, qb, inner, 0, 0, 1e-4f, 1.0f, hd, Q, NH)) return -1;
        if (pr_rms(r, kvb, 2 * inner, kvb, 2 * inner, 0, 0, 1e-4f, 1.0f, hd, nkv, NH)) return -1;
        if (use_iv) {   /* block-sparse flash attention over the per-query key intervals (bf16 operands, fp32 accumulate) */
            CUdeviceptr qbb = pr_wsb(r, WS_QB, (size_t)Q * inner), kbb = pr_wsb(r, WS_KB, (size_t)nkv * inner), vbb = pr_wsb(r, WS_VB, (size_t)nkv * inner),
                        aob = pr_wsb(r, WS_AOB, (size_t)Q * inner);
            if (!qbb || !kbb || !vbb || !aob) return -1;
            LAUNCH(r->f_cvt, NBLK((long long)Q * inner), 1, 1, 256, 0, PD(qb), PL(inner), PD(qbb), PI(inner), PL(Q));
            LAUNCH(r->f_cvt, NBLK((long long)nkv * inner), 1, 1, 256, 0, PD(kvb), PL(2 * inner), PD(kbb), PI(inner), PL(nkv));
            LAUNCH(r->f_cvt, NBLK((long long)nkv * inner), 1, 1, 256, 0, PD(FO(kvb, inner)), PL(2 * inner), PD(vbb), PI(inner), PL(nkv));
            if (pr_flash_ex(r, qbb, inner, kbb, inner, vbb, inner, aob, inner, 0, NH, NH, hd, Q, nkv, cross ? dmc : dms, 0, 0, 1)) return -1;
            if (pr_gemm2(r, 0, aob, inner, &b->to_out, 0, 0, x, A, 0, Q, A, inner, 1)) return -1;
        } else {
        if (pr_attn(r, qb, inner, kvb, 2 * inner, FO(kvb, inner), 2 * inner, ao, inner, 0, NH, NH, hd, Q, nkv, 0, cross ? dmc : dms, 0, 0)) return -1;
        if (pr_gemm(r, ao, inner, &b->to_out, 0, 0, x, A, Q, A, inner, 1)) return -1;
        }
        /* adaln_mlp + GEGLU MLP */
        if (pr_gemm(r, silu_c, A, &b->ada_m, 0, b->ada_m_b, scl, A, Q, A, A, 0)) return -1;
        LAUNCH(r->f_adaln, Q, 1, 1, A >= 128 ? 128 : 32, 0, PD(x), PD(scl), PD(hbuf), PF(1e-4), PI(A));
        if (pr_gemm(r, hbuf, A, &b->mlp_in, 0, b->mlp_in_b, ub, 8 * A, Q, 8 * A, A, 0)) return -1;
        if (pr_swi(r, ub, 8 * A, FO(ub, (size_t)4 * A), 8 * A, gb, 4 * A, 4 * A, Q, 1)) return -1;
        if (pr_gemm(r, gb, 4 * A, &b->mlp_out, 0, b->mlp_out_b, x, A, Q, A, 4 * A, 1)) return -1;
        snprintf(nm, sizeof nm, "ae_b%d_out", bi); pr_dump(r, nm, x, Q, A);
    }
    /* head on the action tokens */
    {
        CUdeviceptr ln = pr_ws(r, WS_TMP, (size_t)T * A), o = pr_ws(r, WS_MOD, (size_t)T * c->act_dim);
        if (!ln || !o) return -1;
        LAUNCH(r->f_ln, T, 1, 1, A >= 128 ? 128 : 32, 0, PD(FO(x, (size_t)(K + K * nf) * A)), PL(A), PD(ln), PF(1e-6), PI(A));
        if (pr_gemm(r, ln, A, &ae->head, 0, ae->head_b, o, c->act_dim, T, c->act_dim, A, 0)) return -1;
        PRC(cuMemcpyDtoH(out, o, F4((size_t)T * c->act_dim)));
        pr_dump(r, "ae_out", o, T, c->act_dim);
    }
    return 0;
}

/* ---- window_joint sampler ---- */
int praxis_sample_window_joint_ex(praxis_runner *r, const float *text, const float *text_u, int S,
                                  const float *anchor, int h, int w, const float *clean,
                                  const float *proprio, const float *noise_v, const float *noise_a,
                                  int K, int num_steps, double t_shift, double cfg_video, double cfg_action,
                                  float *video_out, float *action_out, int pair_cfg, int skip_last_video) {
    return praxis_sample_window_joint_ex2(r, text, S, text_u, S, anchor, h, w, clean, proprio, noise_v, noise_a, K, num_steps, t_shift, cfg_video, cfg_action,
                                          video_out, action_out, pair_cfg, skip_last_video);
}
int praxis_sample_window_joint_ex2(praxis_runner *r, const float *text, int S, const float *text_u, int S_u,
                                   const float *anchor, int h, int w, const float *clean,
                                   const float *proprio, const float *noise_v, const float *noise_a,
                                   int K, int num_steps, double t_shift, double cfg_video, double cfg_action,
                                   float *video_out, float *action_out, int pair_cfg, int skip_last_video) {
    const pr_cfg *c = &r->c;
    int F = c->lat, vc = c->enc_vcin, A = c->act_dim;
    size_t nv = (size_t)vc * K * F * h * w, na = (size_t)K * F * A;
    float *xv = (float *)malloc(F4(nv)), *xa = (float *)malloc(F4(na));
    float *vc_ = (float *)malloc(F4(nv)), *vu_ = (float *)malloc(F4(nv)), *ac_ = (float *)malloc(F4(na)), *au_ = (float *)malloc(F4(na));
    memcpy(xv, noise_v, F4(nv)); memcpy(xa, noise_a, F4(na));
    double *ts = (double *)malloc((num_steps + 1) * sizeof(double));
    for (int i = 0; i <= num_steps; i++) {
        double u = 1.0 - (double)i / (double)num_steps;
        ts[i] = t_shift == 1.0 ? u : t_shift * u / (1.0 + (t_shift - 1.0) * u);
    }
    int guided = !(cfg_video == 1.0 && cfg_action == 1.0);
    int rc = 0;
    float *tt = (float *)malloc(F4(K));
    long long shv[5] = { 1, vc, (long long)K * F, h, w }, sha[3] = { 1, (long long)K * F, A };
    int pn = 0;
    pr_ctx_invalidate(r);   /* new window: text/anchor/clean changed, the first step stores the context K/V again */
    for (int step = 0; step < num_steps && !rc; step++) {
        double tc = ts[step], tn = ts[step + 1];
        int skipv = skip_last_video && step == num_steps - 1;
        /* skip_last_video: the last step moves only the actions; the unguided pass is then needed only for action guidance */
        int need_unc = guided && !(skipv && cfg_action == 1.0);
        for (int k = 0; k < K; k++) tt[k] = (float)tc;
        int npass = need_unc ? 2 : 1;
        int paired = need_unc && pair_cfg && r->fast && S_u == S;
        if (paired) rc = pr_encoder_pair_x(r, text, text_u, S, anchor, h, w, xv, clean, tt, K);
        for (int pass = 0; pass < npass && !rc; pass++) {
            float *vo = pass == 0 ? vc_ : vu_, *ao = pass == 0 ? ac_ : au_;
            if (paired) praxis_select_state(r, pass);
            else rc = pr_encoder_x(r, pass == 0 ? text : text_u, pass == 0 ? S : S_u, anchor, h, w, xv, clean, tt, K, pass);
            if (!rc && !skipv) rc = praxis_decoder(r, xv, tt, vo);
            if (!rc) rc = praxis_expert(r, xa, tt, proprio, ao);
            if (!rc) {
                pr_smp_dump(r, "velocity.in.xt_v", pn, xv, 5, shv); pr_smp_dump(r, "velocity.in.xt_a", pn, xa, 3, sha);
                if (!skipv) pr_smp_dump(r, "velocity.out.video", pn, vo, 5, shv);
                pr_smp_dump(r, "velocity.out.action", pn, ao, 3, sha);
                pn++;
            }
        }
        praxis_select_state(r, 0);
        if (rc) break;
        float dt = (float)(tn - tc);
        float *gv = (float *)malloc(F4(nv)), *ga = (float *)malloc(F4(na));
        if (!skipv) for (size_t i = 0; i < nv; i++) {
            float v = vc_[i];
            if (guided && cfg_video != 1.0) v = vu_[i] + (float)cfg_video * (vc_[i] - vu_[i]);
            gv[i] = v;
            xv[i] = xv[i] + dt * v;
        }
        for (size_t i = 0; i < na; i++) {
            float a = ac_[i];
            if (need_unc && cfg_action != 1.0) a = au_[i] + (float)cfg_action * (ac_[i] - au_[i]);
            ga[i] = a;
            xa[i] = xa[i] + dt * a;
        }
        if (!skipv) pr_smp_dump(r, "cfg_combine.video.out", step, gv, 5, shv);
        pr_smp_dump(r, "cfg_combine.action.out", step, ga, 3, sha);
        free(gv); free(ga);
        if (r->verbose) fprintf(stderr, "praxis: euler step %d/%d t %.6f -> %.6f%s%s\n", step + 1, num_steps, tc, tn, paired ? " (paired encoder)" : "", skipv ? " (video skipped)" : "");
    }
    if (!rc) {
        /* last chunk only */
        size_t T = (size_t)F * h * w;
        if (skip_last_video) memset(video_out, 0, F4((size_t)vc * T));
        else for (int ch = 0; ch < vc; ch++)
            memcpy(video_out + (size_t)ch * T, xv + ((size_t)ch * K * F * h * w) + (size_t)(K - 1) * T, F4(T));
        memcpy(action_out, xa + (size_t)(K - 1) * F * A, F4((size_t)F * A));
    }
    free(xv); free(xa); free(vc_); free(vu_); free(ac_); free(au_); free(ts); free(tt);
    return rc;
}
int praxis_sample_window_joint(praxis_runner *r, const float *text, const float *text_u, int S,
                               const float *anchor, int h, int w, const float *clean,
                               const float *proprio, const float *noise_v, const float *noise_a,
                               int K, int num_steps, double t_shift, double cfg_video, double cfg_action,
                               float *video_out, float *action_out) {
    return praxis_sample_window_joint_ex(r, text, text_u, S, anchor, h, w, clean, proprio, noise_v, noise_a, K, num_steps, t_shift,
                                         cfg_video, cfg_action, video_out, action_out, 0, 0);
}
#endif
