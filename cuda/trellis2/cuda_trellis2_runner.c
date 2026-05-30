/*
 * cuda_trellis2_runner.c - CUDA TRELLIS.2 Stage 1 via NVRTC-compiled kernels
 *
 * Pipeline: DINOv3 encoder -> DiT diffusion (flow matching) -> Decoder -> occupancy
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 *
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
/* sparse3d.h provides sp3d_hash_build/free for shape decoder gather map */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"
#include "cuda_trellis2_runner.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

/* Include HY3D kernels for cross_attn, timestep_embed, euler_step, cfg_combine,
 * split_qkv, split_kv, broadcast_add, rms_norm — these are model-specific in HY3D
 * but reused here. We extract only the kernel source string (no ops header). */
#include "../hy3d/cuda_hy3d_kernels.h"

#include "cuda_trellis2_kernels.h"
#include "cuda_trellis2_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ======================================================================== */
/* Model constants                                                          */
/* ======================================================================== */

static int t2_timing_enabled(void) {
    const char *v = getenv("T2_TIMING");
    return (v && v[0] && atoi(v) != 0) ? 1 : 0;
}

static double t2_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void t2_timing_log(const char *label, double t0_ms) {
    if (t2_timing_enabled())
        fprintf(stderr, "T2_TIMING: %s %.3f ms\n", label, t2_now_ms() - t0_ms);
}

/* DINOv3 ViT-L/16 */
#define DINO_HIDDEN     1024
#define DINO_HEADS      16
#define DINO_HEAD_DIM   64
#define DINO_LAYERS     24
#define DINO_FFN        4096
#define DINO_PATCH      16
#define DINO_IMG_SIZE   512
#define DINO_GRID       (DINO_IMG_SIZE / DINO_PATCH)  /* 32 */
#define DINO_N_PATCHES  (DINO_GRID * DINO_GRID)       /* 1024 */
#define DINO_N_STORAGE  4
#define DINO_SEQ_LEN    (1 + DINO_N_STORAGE + DINO_N_PATCHES)  /* 1029 */

/* DiT Stage 1 */
#define DIT_DIM         1536
#define DIT_HEADS       12
#define DIT_HEAD_DIM    128
#define DIT_FFN         8192
#define DIT_DEPTH       30
#define DIT_IN_CH       8
#define DIT_GRID        16
#define DIT_N_TOKENS    (DIT_GRID * DIT_GRID * DIT_GRID)  /* 4096 */
#define DIT_COND_DIM    DINO_HIDDEN                        /* 1024 */

/* ======================================================================== */
/* GPU weight structures                                                    */
/* ======================================================================== */

typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr q_w, q_b, k_w, k_b, v_w, v_b;  /* separate Q/K/V for RoPE */
    CUdeviceptr q_norm_w, k_norm_w;
    CUdeviceptr out_w, out_b;
    CUdeviceptr ls1, ls2;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr fc1_w, fc1_b;
    CUdeviceptr fc2_w, fc2_b;
} dino_layer_gpu;

typedef struct {
    /* Self-attention */
    CUdeviceptr sa_qkv_w, sa_qkv_b;    /* [3*DIT_DIM, DIT_DIM], [3*DIT_DIM] */
    CUdeviceptr sa_q_norm, sa_k_norm;   /* [DIT_HEADS * DIT_HEAD_DIM] */
    CUdeviceptr sa_out_w, sa_out_b;
    /* Cross-attention */
    CUdeviceptr norm2_w, norm2_b;
    CUdeviceptr ca_q_w, ca_q_b;
    CUdeviceptr ca_kv_w, ca_kv_b;
    CUdeviceptr ca_q_norm, ca_k_norm;
    CUdeviceptr ca_out_w, ca_out_b;
    /* MLP */
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;
    /* Per-block modulation bias */
    CUdeviceptr mod_bias;  /* [6*DIT_DIM] */
} dit_block_gpu;

/* CPU-side weight storage for layer streaming (offloading) */
typedef struct {
    void *sa_qkv_w, *sa_qkv_b, *sa_q_norm, *sa_k_norm, *sa_out_w, *sa_out_b;
    void *norm2_w, *norm2_b, *ca_q_w, *ca_q_b, *ca_kv_w, *ca_kv_b;
    void *ca_q_norm, *ca_k_norm, *ca_out_w, *ca_out_b;
    void *mlp_fc1_w, *mlp_fc1_b, *mlp_fc2_w, *mlp_fc2_b;
    void *mod_bias;
    /* Byte sizes for each weight */
    size_t sa_qkv_w_sz, sa_qkv_b_sz, sa_q_norm_sz, sa_k_norm_sz;
    size_t sa_out_w_sz, sa_out_b_sz, norm2_w_sz, norm2_b_sz;
    size_t ca_q_w_sz, ca_q_b_sz, ca_kv_w_sz, ca_kv_b_sz;
    size_t ca_q_norm_sz, ca_k_norm_sz, ca_out_w_sz, ca_out_b_sz;
    size_t mlp_fc1_w_sz, mlp_fc1_b_sz, mlp_fc2_w_sz, mlp_fc2_b_sz;
    size_t mod_bias_sz;
} dit_block_cpu;

/* Generic DiT model (shared by Stage 1 and Stage 2) */
typedef struct {
    int n_blocks;
    int model_channels;   /* D: 1536 */
    int n_heads;          /* 12 */
    int head_dim;         /* 128 */
    int ffn_hidden;       /* 8192 */
    int in_channels;      /* 8 for Stage 1, 32 for Stage 2, 64 for Stage 3 */
    int out_channels;     /* same as in_channels for Stage 1/2, 32 for Stage 3 */
    int cond_dim;         /* 1024 */
    CUdeviceptr t_fc1_w, t_fc1_b, t_fc2_w, t_fc2_b;
    CUdeviceptr mod_w, mod_b;
    CUdeviceptr x_emb_w, x_emb_b;
    CUdeviceptr out_w, out_b;
    dit_block_gpu *blocks;
    CUdeviceptr rope_cos, rope_sin;
    int n_rope_freqs, rope_axis_dim;
    int use_f16;          /* 1 = fp16 MMA path, 0 = F32 GEMM (default, matches bf16) */
} dit_model_gpu;

typedef struct {
    CUdeviceptr conv_w, conv_b;
    CUdeviceptr gn1_w, gn1_b, conv1_w, conv1_b;
    CUdeviceptr gn2_w, gn2_b, conv2_w, conv2_b;
} dec_resblock_gpu;

typedef struct {
    CUdeviceptr from_latent_w, from_latent_b;  /* [1024, 32] */
    CUdeviceptr out_w, out_b;                  /* [out_channels, 64] */
    struct {
        CUdeviceptr conv_w, conv_b;   /* [27, C, C] transposed for gather-GEMM */
        CUdeviceptr norm_w, norm_b;   /* [C] */
        CUdeviceptr mlp0_w, mlp0_b;   /* [4C, C] */
        CUdeviceptr mlp2_w, mlp2_b;   /* [C, 4C] */
    } convnext[4][16];  /* [stage][block] — max 16 blocks per stage */
    struct {
        CUdeviceptr norm1_w, norm1_b;
        CUdeviceptr conv1_w, conv1_b;  /* [27, C_out*8, C_in] transposed */
        CUdeviceptr conv2_w, conv2_b;  /* [27, C_out, C_out] transposed */
        CUdeviceptr subdiv_w, subdiv_b; /* [8, C_in], absent for dense texture decoder */
    } c2s[4];
    int n_convnext[4];  /* {4, 16, 8, 4} */
    int channels[5];    /* {1024, 512, 256, 128, 64} */
    int out_channels;   /* 7 for shape, 6 for texture */
    int use_f16;
    int loaded;
} t2_scvae_dec_gpu;

/* Per-C2S-stage subdivision recorded during the SHAPE decode and replayed by the
 * TEXTURE decode. The texture decoder checkpoint has no `to_subdiv` head, so the
 * PyTorch image-to-3D pipeline feeds it the shape decoder's per-level subdivision
 * (decode_tex_slat(..., guide_subs=subs)). We reproduce that by recording the
 * shape side's (parent idx, child slot, child coords) at each stage and replaying
 * it on the texture side instead of the dense x8 fallback (which explodes to
 * ~17M voxels and OOMs the card). Both decoders are driven from the same res-32
 * sparse coords in the same order, so the recorded indices stay valid. */
typedef struct {
    int      valid;
    int      n_parent;   /* parent voxel count this subdivision was computed from */
    int      n_new;      /* child voxel count kept */
    int32_t *idx;        /* [n_new] parent index into the stage's coord array */
    int32_t *subidx;     /* [n_new] child slot 0..7 */
    int32_t *coords;     /* [n_new*4] child coords (b,z,y,x) */
} t2_subdiv_stage;

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

struct cuda_trellis2_runner {
    CUdevice   device;
    CUcontext  context;
    CUstream   stream;
    CUmodule   module;
    t2_ops     ops;
    int        verbose;

    /* DINOv3 weights */
    CUdeviceptr dino_patch_w, dino_patch_b;
    CUdeviceptr dino_cls_token, dino_storage_tokens;
    CUdeviceptr dino_norm_w, dino_norm_b;
    dino_layer_gpu dino_layers[DINO_LAYERS];
    int dino_has_qk_norm, dino_has_layerscale;
    CUdeviceptr dino_rope_cos, dino_rope_sin;  /* [1024, 64] for 32x32 patches */

    /* Stage 1 DiT */
    CUdeviceptr dit_t_fc1_w, dit_t_fc1_b, dit_t_fc2_w, dit_t_fc2_b;
    CUdeviceptr dit_mod_w, dit_mod_b;
    CUdeviceptr dit_x_emb_w, dit_x_emb_b;
    CUdeviceptr dit_out_w, dit_out_b;
    dit_block_gpu dit_blocks[DIT_DEPTH];
    CUdeviceptr dit_rope_cos, dit_rope_sin;
    int dit_n_freqs, dit_axis_dim;
    int dit_use_f16;  /* 1 if DiT weights stored as F16 */
    dit_block_cpu dit_blocks_cpu[DIT_DEPTH]; /* CPU copies for layer streaming */

    /* Stage 2 shape flow DiT */
    dit_model_gpu stage2;
    dit_block_gpu stage2_blocks[DIT_DEPTH];
    dit_block_cpu stage2_blocks_cpu[DIT_DEPTH];
    int stage2_loaded;

    /* Stage 3 texture flow DiT (same architecture, in_channels=64) */
    dit_model_gpu stage3;
    dit_block_gpu stage3_blocks[DIT_DEPTH];
    dit_block_cpu stage3_blocks_cpu[DIT_DEPTH];
    int stage3_loaded;

    /* Layer streaming config (0 = all on GPU, >0 = max layers on GPU at once) */
    int max_gpu_layers;

    /* Decoder weights */
    CUdeviceptr dec_conv_in_w, dec_conv_in_b;
    dec_resblock_gpu dec_middle[2];
    dec_resblock_gpu dec_res16[2];
    CUdeviceptr dec_up1_w, dec_up1_b;
    dec_resblock_gpu dec_res32[2];
    CUdeviceptr dec_up2_w, dec_up2_b;
    dec_resblock_gpu dec_res64[2];
    CUdeviceptr dec_out_gn_w, dec_out_gn_b;
    CUdeviceptr dec_out_conv_w, dec_out_conv_b;

    /* SC-VAE decoders */
    t2_scvae_dec_gpu shape_dec;
    t2_scvae_dec_gpu tex_dec;

    /* Subdivision plan recorded by the shape decoder, replayed by the texture
     * decoder (which has no to_subdiv head). Indexed by C2S stage (0..3). */
    t2_subdiv_stage subdiv_plan[8];

    /* Cached packed-sparse-conv index set for the current resolution level. The
     * pack (27 src/dst index arrays) is a pure function of (coords, hash) and is
     * IDENTICAL across all ConvNeXt blocks at a level, but t2_sparse_conv_pack_build
     * is host-side (N*27 CPU hash lookups + up to 54 HtoD uploads). Caching it keyed
     * on (coords ptr, N) turns the per-block rebuild into one build per level. */
    t2_sparse_conv_pack pack_cache;
    const int32_t *pack_cache_coords;
    int pack_cache_N;
    int pack_cache_valid;

    /* Cross-attention KV cache (precomputed per-block, reused across timesteps).
     * Slot 0 is the active nonzero condition, slot 1 is the all-zero CFG condition. */
    CUdeviceptr ca_kv_cache_K[2][DIT_DEPTH];
    CUdeviceptr ca_kv_cache_V[2][DIT_DEPTH];
    uint64_t ca_kv_cache_cond_hash[2];
    int ca_kv_cache_n_blocks[2];
    int ca_kv_cache_model_id[2];  /* which model's weights were cached */
    int ca_kv_cache_valid[2];

    /* Sparse Stage-2/3 RoPE tables are a pure function of coords and n_freqs.
     * Cache them across sampler steps to avoid repeated CPU sin/cos + HtoD. */
    CUdeviceptr sparse_rope_cos[4];
    CUdeviceptr sparse_rope_sin[4];
    uint64_t sparse_rope_coord_hash[4];
    int sparse_rope_N[4];
    int sparse_rope_n_freqs[4];
    int sparse_rope_valid[4];

    /* Scratch buffers (expanded to 12 for shape decoder use) */
    CUdeviceptr scratch[12];
    size_t      scratch_size[12];
};

static void ensure_scratch(cuda_trellis2_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] >= bytes) return;
    if (r->scratch[idx]) cuMemFree(r->scratch[idx]);
    CUresult err = cuMemAlloc(&r->scratch[idx], bytes);
    if (err != 0) {
        const char *name = NULL;
        cuGetErrorName(err, &name);
        size_t free_mem = 0, total_mem = 0;
        cuMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "T2: FATAL: cuMemAlloc(%zu bytes = %.1f MB) for scratch[%d] failed: %s "
                "(GPU free=%.0f MB / %.0f MB total)\n",
                bytes, bytes / 1048576.0, idx, name ? name : "?",
                free_mem / 1048576.0, total_mem / 1048576.0);
        r->scratch[idx] = 0;
        r->scratch_size[idx] = 0;
        return;
    }
    r->scratch_size[idx] = bytes;
}

/* Free any recorded shape-decoder subdivision plan (call before a new shape
 * decode and at runner teardown). Safe to call repeatedly. */
static void t2_subdiv_plan_free(cuda_trellis2_runner *r) {
    for (int i = 0; i < 8; i++) {
        free(r->subdiv_plan[i].idx);
        free(r->subdiv_plan[i].subidx);
        free(r->subdiv_plan[i].coords);
        r->subdiv_plan[i].idx = NULL;
        r->subdiv_plan[i].subidx = NULL;
        r->subdiv_plan[i].coords = NULL;
        r->subdiv_plan[i].valid = 0;
        r->subdiv_plan[i].n_parent = 0;
        r->subdiv_plan[i].n_new = 0;
    }
}

/* ======================================================================== */
/* Layer streaming: upload/free individual blocks                           */
/* ======================================================================== */

/* Upload a single block's weights from CPU to GPU */
static void dit_block_upload_to_gpu(dit_block_gpu *gpu, const dit_block_cpu *cpu) {
    #define UP(field) if (cpu->field) gpu->field = cu_upload_raw(cpu->field, cpu->field##_sz); else gpu->field = 0
    UP(sa_qkv_w); UP(sa_qkv_b); UP(sa_q_norm); UP(sa_k_norm);
    UP(sa_out_w); UP(sa_out_b); UP(norm2_w); UP(norm2_b);
    UP(ca_q_w); UP(ca_q_b); UP(ca_kv_w); UP(ca_kv_b);
    UP(ca_q_norm); UP(ca_k_norm); UP(ca_out_w); UP(ca_out_b);
    UP(mlp_fc1_w); UP(mlp_fc1_b); UP(mlp_fc2_w); UP(mlp_fc2_b);
    UP(mod_bias);
    #undef UP
}

/* Free a block's GPU weights */
static void dit_block_free_gpu(dit_block_gpu *gpu) {
    #define FR(field) if (gpu->field) { cuMemFree(gpu->field); gpu->field = 0; }
    FR(sa_qkv_w); FR(sa_qkv_b); FR(sa_q_norm); FR(sa_k_norm);
    FR(sa_out_w); FR(sa_out_b); FR(norm2_w); FR(norm2_b);
    FR(ca_q_w); FR(ca_q_b); FR(ca_kv_w); FR(ca_kv_b);
    FR(ca_q_norm); FR(ca_k_norm); FR(ca_out_w); FR(ca_out_b);
    FR(mlp_fc1_w); FR(mlp_fc1_b); FR(mlp_fc2_w); FR(mlp_fc2_b);
    FR(mod_bias);
    #undef FR
}

/* Save CPU copy of data that was uploaded to GPU.
 * Returns malloc'd CPU copy (caller owns). */
static void *cpu_copy_from_gpu(CUdeviceptr d, size_t bytes) {
    if (!d || !bytes) return NULL;
    void *cpu = malloc(bytes);
    cuMemcpyDtoH(cpu, d, bytes);
    return cpu;
}

/* After loading a block to GPU, save CPU copies for streaming */
static void dit_block_save_cpu_copy(dit_block_cpu *cpu, const dit_block_gpu *gpu,
                                      int use_f16, int D, int H, int HD, int FFN, int COND_DIM) {
    /* Weight sizes depend on F16 vs F32 */
    size_t ws = use_f16 ? sizeof(uint16_t) : sizeof(float); /* weight element size */
    size_t bs = sizeof(float); /* bias always F32 */

    #define SV(field, sz) cpu->field##_sz = (sz); cpu->field = cpu_copy_from_gpu(gpu->field, (sz))
    SV(sa_qkv_w, (size_t)3*D*D*ws);  SV(sa_qkv_b, (size_t)3*D*bs);
    SV(sa_q_norm, (size_t)H*HD*bs);  SV(sa_k_norm, (size_t)H*HD*bs);
    SV(sa_out_w, (size_t)D*D*ws);    SV(sa_out_b, (size_t)D*bs);
    SV(norm2_w, (size_t)D*bs);       SV(norm2_b, (size_t)D*bs);
    SV(ca_q_w, (size_t)D*D*ws);      SV(ca_q_b, (size_t)D*bs);
    SV(ca_kv_w, (size_t)2*D*COND_DIM*ws); SV(ca_kv_b, (size_t)2*D*bs);
    SV(ca_q_norm, (size_t)H*HD*bs);  SV(ca_k_norm, (size_t)H*HD*bs);
    SV(ca_out_w, (size_t)D*D*ws);    SV(ca_out_b, (size_t)D*bs);
    SV(mlp_fc1_w, (size_t)FFN*D*ws); SV(mlp_fc1_b, (size_t)FFN*bs);
    SV(mlp_fc2_w, (size_t)D*FFN*ws); SV(mlp_fc2_b, (size_t)D*bs);
    SV(mod_bias, (size_t)6*D*bs);
    #undef SV
}

static void dit_block_cpu_free(dit_block_cpu *cpu) {
    free(cpu->sa_qkv_w); free(cpu->sa_qkv_b); free(cpu->sa_q_norm); free(cpu->sa_k_norm);
    free(cpu->sa_out_w); free(cpu->sa_out_b); free(cpu->norm2_w); free(cpu->norm2_b);
    free(cpu->ca_q_w); free(cpu->ca_q_b); free(cpu->ca_kv_w); free(cpu->ca_kv_b);
    free(cpu->ca_q_norm); free(cpu->ca_k_norm); free(cpu->ca_out_w); free(cpu->ca_out_b);
    free(cpu->mlp_fc1_w); free(cpu->mlp_fc1_b); free(cpu->mlp_fc2_w); free(cpu->mlp_fc2_b);
    free(cpu->mod_bias);
    memset(cpu, 0, sizeof(*cpu));
}

static void dino_layer_free_gpu(dino_layer_gpu *l) {
    CU_FREE(l->ln1_w); CU_FREE(l->ln1_b);
    CU_FREE(l->q_w); CU_FREE(l->q_b);
    CU_FREE(l->k_w); CU_FREE(l->k_b);
    CU_FREE(l->v_w); CU_FREE(l->v_b);
    CU_FREE(l->q_norm_w); CU_FREE(l->k_norm_w);
    CU_FREE(l->out_w); CU_FREE(l->out_b);
    CU_FREE(l->ls1); CU_FREE(l->ls2);
    CU_FREE(l->ln2_w); CU_FREE(l->ln2_b);
    CU_FREE(l->fc1_w); CU_FREE(l->fc1_b);
    CU_FREE(l->fc2_w); CU_FREE(l->fc2_b);
}

static void dec_resblock_free_gpu(dec_resblock_gpu *rb) {
    CU_FREE(rb->conv_w); CU_FREE(rb->conv_b);
    CU_FREE(rb->gn1_w); CU_FREE(rb->gn1_b);
    CU_FREE(rb->conv1_w); CU_FREE(rb->conv1_b);
    CU_FREE(rb->gn2_w); CU_FREE(rb->gn2_b);
    CU_FREE(rb->conv2_w); CU_FREE(rb->conv2_b);
}

static void dit_model_free_gpu(dit_model_gpu *m) {
    CU_FREE(m->t_fc1_w); CU_FREE(m->t_fc1_b);
    CU_FREE(m->t_fc2_w); CU_FREE(m->t_fc2_b);
    CU_FREE(m->mod_w); CU_FREE(m->mod_b);
    CU_FREE(m->x_emb_w); CU_FREE(m->x_emb_b);
    CU_FREE(m->out_w); CU_FREE(m->out_b);
    if (m->blocks) {
        int n = m->n_blocks > 0 ? m->n_blocks : DIT_DEPTH;
        if (n > DIT_DEPTH) n = DIT_DEPTH;
        for (int i = 0; i < n; i++)
            dit_block_free_gpu(&m->blocks[i]);
    }
    CU_FREE(m->rope_cos);
    CU_FREE(m->rope_sin);
}

static void scvae_decoder_free_gpu(t2_scvae_dec_gpu *d) {
    CU_FREE(d->from_latent_w);
    CU_FREE(d->from_latent_b);
    CU_FREE(d->out_w);
    CU_FREE(d->out_b);
    for (int s = 0; s < 4; s++) {
        for (int b = 0; b < 16; b++) {
            CU_FREE(d->convnext[s][b].conv_w);
            CU_FREE(d->convnext[s][b].conv_b);
            CU_FREE(d->convnext[s][b].norm_w);
            CU_FREE(d->convnext[s][b].norm_b);
            CU_FREE(d->convnext[s][b].mlp0_w);
            CU_FREE(d->convnext[s][b].mlp0_b);
            CU_FREE(d->convnext[s][b].mlp2_w);
            CU_FREE(d->convnext[s][b].mlp2_b);
        }
        CU_FREE(d->c2s[s].norm1_w);
        CU_FREE(d->c2s[s].norm1_b);
        CU_FREE(d->c2s[s].conv1_w);
        CU_FREE(d->c2s[s].conv1_b);
        CU_FREE(d->c2s[s].conv2_w);
        CU_FREE(d->c2s[s].conv2_b);
        CU_FREE(d->c2s[s].subdiv_w);
        CU_FREE(d->c2s[s].subdiv_b);
    }
    d->loaded = 0;
}

static void dbg4(const char *label, CUdeviceptr ptr, CUstream stream) {
    cuStreamSynchronize(stream);
    float d[4];
    cuMemcpyDtoH(d, ptr, 16);
    fprintf(stderr, "  %s: [:4]=%.6f %.6f %.6f %.6f\n", label, d[0], d[1], d[2], d[3]);
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

static float t2_f16_to_f32_value(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    float f;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
            memcpy(&f, &bits, 4);
            return f;
        }
        f = ldexpf((float)mant, -24);
        return sign ? -f : f;
    }
    if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    memcpy(&f, &bits, 4);
    return f;
}

/* Upload safetensors tensor as F32 on GPU (handles F16, BF16, F32 source) */
static CUdeviceptr t2_upload_f32(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [MISSING] %s\n", name);
        return 0;
    }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        return cu_upload_raw(data, nbytes);
    }

    /* Convert to F32 on CPU, then upload */
    size_t n_elem;
    if (strcmp(dtype, "F16") == 0)      n_elem = nbytes / 2;
    else if (strcmp(dtype, "BF16") == 0) n_elem = nbytes / 2;
    else { fprintf(stderr, "  [SKIP] %s: unsupported dtype %s\n", name, dtype); return 0; }

    float *f32 = (float *)malloc(n_elem * sizeof(float));
    const uint16_t *src = (const uint16_t *)data;

    if (strcmp(dtype, "BF16") == 0) {
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else { /* F16 */
        for (size_t i = 0; i < n_elem; i++) {
            f32[i] = t2_f16_to_f32_value(src[i]);
        }
    }

    CUdeviceptr d = cu_upload_raw(f32, n_elem * sizeof(float));
    free(f32);

    if (verbose >= 2) {
        const uint64_t *sh = safetensors_shape(st, idx);
        int nd = safetensors_ndims(st, idx);
        fprintf(stderr, "  %s: %s [", name, dtype);
        for (int d2 = 0; d2 < nd; d2++) fprintf(stderr, "%s%lu", d2?",":"", (unsigned long)sh[d2]);
        fprintf(stderr, "] -> F32 GPU (%.1f MB)\n", (float)(n_elem * 4) / (1024*1024));
    }
    return d;
}

/* Fast path for F16/BF16 checkpoints loaded as F32: upload the raw 16-bit tensor and
 * expand on GPU. This preserves the exact same F32 values as the CPU conversion
 * while halving host-to-device traffic for the large DiT/SC-VAE weights. */
static CUdeviceptr t2_upload_f32_fast(cuda_trellis2_runner *r,
                                      st_context *st,
                                      const char *name,
                                      int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [MISSING] %s\n", name);
        return 0;
    }
    const char *dtype = safetensors_dtype(st, idx);
    int is_bf16 = strcmp(dtype, "BF16") == 0;
    int is_f16 = strcmp(dtype, "F16") == 0;
    int cpu_bf16 = getenv("T2_CPU_BF16_UPLOAD") && atoi(getenv("T2_CPU_BF16_UPLOAD"));
    int cpu_f16 = getenv("T2_CPU_F16_UPLOAD") && atoi(getenv("T2_CPU_F16_UPLOAD"));
    CUfunction cast_fn = r ? (is_bf16 ? r->ops.cast_bf16_to_f32 :
                              is_f16  ? r->ops.cast_f16_to_f32 : NULL) : NULL;
    if (!r || !cast_fn || (!is_bf16 && !is_f16) ||
        (is_bf16 && cpu_bf16) || (is_f16 && cpu_f16)) {
        return t2_upload_f32(st, name, verbose);
    }

    size_t nbytes = safetensors_nbytes(st, idx);
    size_t n_elem = nbytes / 2;
    CUdeviceptr d_bf16 = 0, d_f32 = 0;
    if (cuMemAlloc(&d_bf16, nbytes) != CUDA_SUCCESS)
        return t2_upload_f32(st, name, verbose);
    if (cuMemAlloc(&d_f32, n_elem * sizeof(float)) != CUDA_SUCCESS)
        goto fail;
    if (cuMemcpyHtoDAsync(d_bf16, safetensors_data(st, idx), nbytes,
                          r->stream) != CUDA_SUCCESS)
        goto fail;
    long n = (long)n_elem;
    void *args[] = {&d_bf16, &d_f32, &n};
    if (cuLaunchKernel(cast_fn, (unsigned)((n_elem + 255) / 256),
                       1, 1, 256, 1, 1, 0, r->stream, args, NULL) != CUDA_SUCCESS)
        goto fail;
    if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS)
        goto fail;
    cuMemFree(d_bf16);

    if (verbose >= 2) {
        const uint64_t *sh = safetensors_shape(st, idx);
        int nd = safetensors_ndims(st, idx);
        fprintf(stderr, "  %s: %s [", name, dtype);
        for (int d2 = 0; d2 < nd; d2++) fprintf(stderr, "%s%lu", d2?",":"", (unsigned long)sh[d2]);
        fprintf(stderr, "] -> F32 GPU via 16-bit upload (%.1f MB H2D)\n",
                (float)nbytes / (1024*1024));
    }
    return d_f32;

fail:
    if (d_bf16) cuMemFree(d_bf16);
    if (d_f32) cuMemFree(d_f32);
    return t2_upload_f32(st, name, verbose);
}

/* Upload safetensors tensor as F16 on GPU (handles F16, BF16, F32 source).
 * Halves GPU memory compared to F32, enables MMA tensor core GEMM. */
static CUdeviceptr t2_upload_f16(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [MISSING] %s\n", name);
        return 0;
    }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    if (strcmp(dtype, "F16") == 0) {
        /* Already F16 — upload directly */
        return cu_upload_raw(data, nbytes);
    }

    size_t n_elem;
    if (strcmp(dtype, "BF16") == 0)     n_elem = nbytes / 2;
    else if (strcmp(dtype, "F32") == 0) n_elem = nbytes / 4;
    else { fprintf(stderr, "  [SKIP] %s: unsupported dtype %s for F16\n", name, dtype); return 0; }

    uint16_t *f16 = (uint16_t *)malloc(n_elem * sizeof(uint16_t));
    if (strcmp(dtype, "BF16") == 0) {
        /* BF16 -> F16: convert via F32 intermediate */
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            float f; memcpy(&f, &bits, 4);
            /* F32 -> F16 conversion */
            uint32_t fb; memcpy(&fb, &f, 4);
            uint32_t sign = (fb >> 16) & 0x8000;
            int32_t exp = ((fb >> 23) & 0xff) - 127 + 15;
            uint32_t mant = (fb >> 13) & 0x3ff;
            if (exp <= 0) f16[i] = (uint16_t)sign;           /* underflow to zero */
            else if (exp >= 31) f16[i] = (uint16_t)(sign | 0x7c00); /* overflow to inf */
            else f16[i] = (uint16_t)(sign | (exp << 10) | mant);
        }
    } else { /* F32 -> F16 */
        const float *src = (const float *)data;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t fb; memcpy(&fb, &src[i], 4);
            uint32_t sign = (fb >> 16) & 0x8000;
            int32_t exp = ((fb >> 23) & 0xff) - 127 + 15;
            uint32_t mant = (fb >> 13) & 0x3ff;
            if (exp <= 0) f16[i] = (uint16_t)sign;
            else if (exp >= 31) f16[i] = (uint16_t)(sign | 0x7c00);
            else f16[i] = (uint16_t)(sign | (exp << 10) | mant);
        }
    }

    CUdeviceptr d = cu_upload_raw(f16, n_elem * sizeof(uint16_t));
    free(f16);

    if (verbose >= 2) {
        const uint64_t *sh = safetensors_shape(st, idx);
        int nd = safetensors_ndims(st, idx);
        fprintf(stderr, "  %s: %s [", name, dtype);
        for (int d2 = 0; d2 < nd; d2++) fprintf(stderr, "%s%lu", d2?",":"", (unsigned long)sh[d2]);
        fprintf(stderr, "] -> F16 GPU (%.1f MB)\n", (float)(n_elem * 2) / (1024*1024));
    }
    return d;
}

/* ======================================================================== */
/* Init / Free                                                              */
/* ======================================================================== */

cuda_trellis2_runner *cuda_trellis2_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "T2: failed to load CUDA/NVRTC libraries\n");
        return NULL;
    }
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "T2: cuInit failed (%d)\n", (int)err);
        return NULL;
    }

    cuda_trellis2_runner *r = (cuda_trellis2_runner *)calloc(1, sizeof(*r));
    r->verbose = verbose;

    CU_CHECK_NULL(cuDeviceGet(&r->device, device_id));
    CU_CHECK_NULL(cuCtxCreate(&r->context, 0, r->device));
    CU_CHECK_NULL(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

    /* Query GPU info */
    char name[256];
    cuDeviceGetName(name, sizeof(name), r->device);
    size_t mem;
    cuDeviceTotalMem(&mem, r->device);
    fprintf(stderr, "T2: GPU %d: %s (%.1f GB)\n", device_id, name, (double)mem / (1<<30));

    /* Compile kernels: common + HY3D shared ops + TRELLIS2-specific.
     * HY3D kernels end with "} extern C" so we strip the last line
     * and let trellis2 kernels close the block. */
    size_t hy3d_len = strlen(cuda_hy3d_specific_kernels);
    /* Find and trim the closing "}" from HY3D source */
    const char *hy3d_end = cuda_hy3d_specific_kernels + hy3d_len;
    while (hy3d_end > cuda_hy3d_specific_kernels && hy3d_end[-1] != '}')
        hy3d_end--;
    if (hy3d_end > cuda_hy3d_specific_kernels) hy3d_end--; /* skip the '}' itself */
    size_t hy3d_trimmed = (size_t)(hy3d_end - cuda_hy3d_specific_kernels);

    size_t total_len = strlen(cuda_kernels_common_src) + hy3d_trimmed
                     + strlen(cuda_trellis2_kernel_source) + 16;
    char *full_source = (char *)malloc(total_len);
    sprintf(full_source, "%s\n", cuda_kernels_common_src);
    size_t off = strlen(full_source);
    memcpy(full_source + off, cuda_hy3d_specific_kernels, hy3d_trimmed);
    off += hy3d_trimmed;
    sprintf(full_source + off, "\n%s", cuda_trellis2_kernel_source);

    int sm = cu_compile_kernels(&r->module, r->device, full_source, "trellis2",
                                verbose, "T2");
    free(full_source);
    if (sm < 0) {
        fprintf(stderr, "T2: kernel compilation failed\n");
        cuCtxDestroy(r->context);
        free(r);
        return NULL;
    }
    if (t2_ops_load(&r->ops, r->module, sm) != 0) {
        fprintf(stderr, "T2: failed to load kernel functions\n");
        cuModuleUnload(r->module);
        cuCtxDestroy(r->context);
        free(r);
        return NULL;
    }
    if (cublasewCreate(&r->ops.cublas, r->stream) == 0) {
        fprintf(stderr, "T2: cuBLAS available for opt-in F32 GEMM\n");
    }

    fprintf(stderr, "T2: kernels compiled for sm_%d\n", sm);
    return r;
}

void cuda_trellis2_set_f32_gemm(cuda_trellis2_runner *r, int enable) {
    r->ops.use_f32_gemm = enable;
}

void cuda_trellis2_set_max_gpu_layers(cuda_trellis2_runner *r, int n) {
    r->max_gpu_layers = n;
    if (n > 0)
        fprintf(stderr, "T2: layer streaming enabled: max %d layers on GPU\n", n);
}

void cuda_trellis2_free_buffer(void *p) {
    free(p);
}

static void t2_sparse_conv_pack_free(t2_sparse_conv_pack *pack);  /* defined below */

static void t2_sparse_rope_cache_free(cuda_trellis2_runner *r, int model_id) {
    if (!r || model_id < 0 || model_id >= 4) return;
    CU_FREE(r->sparse_rope_cos[model_id]);
    CU_FREE(r->sparse_rope_sin[model_id]);
    r->sparse_rope_coord_hash[model_id] = 0;
    r->sparse_rope_N[model_id] = 0;
    r->sparse_rope_n_freqs[model_id] = 0;
    r->sparse_rope_valid[model_id] = 0;
}

void cuda_trellis2_free(cuda_trellis2_runner *r) {
    if (!r) return;
    cuCtxSetCurrent(r->context);

    if (r->pack_cache_valid) {
        t2_sparse_conv_pack_free(&r->pack_cache);
        r->pack_cache_valid = 0;
    }

    for (int i = 0; i < 12; i++)
        CU_FREE(r->scratch[i]);
    for (int i = 0; i < 4; i++)
        t2_sparse_rope_cache_free(r, i);

    CU_FREE(r->dit_t_fc1_w); CU_FREE(r->dit_t_fc1_b);
    CU_FREE(r->dit_t_fc2_w); CU_FREE(r->dit_t_fc2_b);
    CU_FREE(r->dit_mod_w); CU_FREE(r->dit_mod_b);
    CU_FREE(r->dit_x_emb_w); CU_FREE(r->dit_x_emb_b);
    CU_FREE(r->dit_out_w); CU_FREE(r->dit_out_b);
    for (int i = 0; i < DIT_DEPTH; i++) {
        dit_block_free_gpu(&r->dit_blocks[i]);
        dit_block_cpu_free(&r->dit_blocks_cpu[i]);
    }
    CU_FREE(r->dit_rope_cos);
    CU_FREE(r->dit_rope_sin);

    dit_model_free_gpu(&r->stage2);
    dit_model_free_gpu(&r->stage3);
    for (int i = 0; i < DIT_DEPTH; i++) {
        dit_block_cpu_free(&r->stage2_blocks_cpu[i]);
        dit_block_cpu_free(&r->stage3_blocks_cpu[i]);
        for (int slot = 0; slot < 2; slot++) {
            CU_FREE(r->ca_kv_cache_K[slot][i]);
            CU_FREE(r->ca_kv_cache_V[slot][i]);
        }
    }

    CU_FREE(r->dino_patch_w); CU_FREE(r->dino_patch_b);
    CU_FREE(r->dino_cls_token); CU_FREE(r->dino_storage_tokens);
    CU_FREE(r->dino_norm_w); CU_FREE(r->dino_norm_b);
    for (int i = 0; i < DINO_LAYERS; i++)
        dino_layer_free_gpu(&r->dino_layers[i]);
    CU_FREE(r->dino_rope_cos);
    CU_FREE(r->dino_rope_sin);

    CU_FREE(r->dec_conv_in_w); CU_FREE(r->dec_conv_in_b);
    for (int i = 0; i < 2; i++) {
        dec_resblock_free_gpu(&r->dec_middle[i]);
        dec_resblock_free_gpu(&r->dec_res16[i]);
        dec_resblock_free_gpu(&r->dec_res32[i]);
        dec_resblock_free_gpu(&r->dec_res64[i]);
    }
    CU_FREE(r->dec_up1_w); CU_FREE(r->dec_up1_b);
    CU_FREE(r->dec_up2_w); CU_FREE(r->dec_up2_b);
    CU_FREE(r->dec_out_gn_w); CU_FREE(r->dec_out_gn_b);
    CU_FREE(r->dec_out_conv_w); CU_FREE(r->dec_out_conv_b);

    scvae_decoder_free_gpu(&r->shape_dec);
    scvae_decoder_free_gpu(&r->tex_dec);
    t2_subdiv_plan_free(r);

    CU_FREE(r->ops.bf16_w);
    CU_FREE(r->ops.bf16_x);
    if (r->ops.cublas) cublasewDestroy(r->ops.cublas);
    cuModuleUnload(r->module);
    cuStreamDestroy(r->stream);
    cuCtxDestroy(r->context);
    free(r);
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

/* DiT precision policy (Stage 1/2/3). PyTorch runs the DiT blocks in bf16,
 * whose exponent range equals f32. Our fp16 MMA path (max ~65504) can clip hot
 * intermediates. Measured single-step vs PyTorch (t=0.5) on RTX 5060 Ti:
 *   Stage 1 (dense 4096 tok): fp16 corr 0.976 / latent cosine 0.727  -> NEEDS F32
 *                             F32  corr 0.9998 / latent cosine 0.989
 *   Stage 2 (sparse 3548 tok): fp16 corr 0.99995 == F32 0.99995      -> fp16 ok
 *   Stage 3 (sparse 3548 tok): fp16 corr 0.99997 == F32 0.99997      -> fp16 ok
 * So Stage 1 defaults to F32; sparse Stage 2/3 stay on fp16 (the slow stages,
 * already bit-faithful). `default_f16` is the per-stage default; env overrides:
 * T2_DIT_F16=1 forces fp16 everywhere, T2_DIT_F32=1 forces F32 everywhere.
 * Pre-Volta (no F16 MMA) always uses F32. */
static int t2_dit_use_f16(cuda_trellis2_runner *r, int default_f16) {
    if (r->ops.sm_version < 70) return 0;
    const char *f16 = getenv("T2_DIT_F16");
    if (f16 && atoi(f16)) return 1;
    const char *f32 = getenv("T2_DIT_F32");
    if (f32 && atoi(f32)) return 0;
    return default_f16;
}

static int t2_dit_keep_cpu_blocks(cuda_trellis2_runner *r) {
    const char *keep = getenv("T2_DIT_KEEP_CPU_BLOCKS");
    return (r->max_gpu_layers > 0 || (keep && atoi(keep))) ? 1 : 0;
}

static int load_dit_weights(cuda_trellis2_runner *r, const char *path) {
    double t_load0 = t2_now_ms();
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading DiT from %s (%d tensors)\n", path, st->n_tensors);

    int v = r->verbose;
    /* Stage 1 defaults to F32 (fp16 clips a hot intermediate: latent cosine
     * 0.727 vs 0.989). T2_DIT_F16=1 opts back into the faster fp16 path. */
    int use_f16 = t2_dit_use_f16(r, 0);

    /* Timestep MLP and modulation: always F32 (n_tok=1, modulation reads directly) */
    r->dit_t_fc1_w = t2_upload_f32_fast(r, st, "t_embedder.mlp.0.weight", v);
    r->dit_t_fc1_b = t2_upload_f32_fast(r, st, "t_embedder.mlp.0.bias", v);
    r->dit_t_fc2_w = t2_upload_f32_fast(r, st, "t_embedder.mlp.2.weight", v);
    r->dit_t_fc2_b = t2_upload_f32_fast(r, st, "t_embedder.mlp.2.bias", v);
    r->dit_mod_w   = t2_upload_f32_fast(r, st, "adaLN_modulation.1.weight", v);
    r->dit_mod_b   = t2_upload_f32_fast(r, st, "adaLN_modulation.1.bias", v);

    /* Input/output embedding and block weights: F16 if supported */
    #define UPW(name) (use_f16 ? t2_upload_f16(st, name, v) : t2_upload_f32_fast(r, st, name, v))
    r->dit_x_emb_w = UPW("input_layer.weight");
    r->dit_x_emb_b = t2_upload_f32_fast(r, st, "input_layer.bias", v);
    r->dit_out_w   = UPW("out_layer.weight");
    r->dit_out_b   = t2_upload_f32_fast(r, st, "out_layer.bias", v);

    if (!r->dit_t_fc1_w) fprintf(stderr, "T2: WARNING: t_embedder missing\n");
    if (!r->dit_mod_w)   fprintf(stderr, "T2: WARNING: adaLN_modulation missing\n");
    if (!r->dit_x_emb_w) fprintf(stderr, "T2: WARNING: input_layer missing\n");

    /* Per-block */
    for (int L = 0; L < DIT_DEPTH; L++) {
        dit_block_gpu *blk = &r->dit_blocks[L];
        char name[256];

        #define BLKW(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                              UPW(name))
        #define BLKB(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                              t2_upload_f32_fast(r, st, name, v >= 2 ? v : 0))

        blk->sa_qkv_w    = BLKW("self_attn.to_qkv.weight");
        blk->sa_qkv_b    = BLKB("self_attn.to_qkv.bias");
        blk->sa_q_norm   = BLKB("self_attn.q_rms_norm.gamma");
        blk->sa_k_norm   = BLKB("self_attn.k_rms_norm.gamma");
        blk->sa_out_w    = BLKW("self_attn.to_out.weight");
        blk->sa_out_b    = BLKB("self_attn.to_out.bias");

        blk->norm2_w     = BLKB("norm2.weight");
        blk->norm2_b     = BLKB("norm2.bias");
        blk->ca_q_w      = BLKW("cross_attn.to_q.weight");
        blk->ca_q_b      = BLKB("cross_attn.to_q.bias");
        blk->ca_kv_w     = BLKW("cross_attn.to_kv.weight");
        blk->ca_kv_b     = BLKB("cross_attn.to_kv.bias");
        blk->ca_q_norm   = BLKB("cross_attn.q_rms_norm.gamma");
        blk->ca_k_norm   = BLKB("cross_attn.k_rms_norm.gamma");
        blk->ca_out_w    = BLKW("cross_attn.to_out.weight");
        blk->ca_out_b    = BLKB("cross_attn.to_out.bias");

        blk->mlp_fc1_w   = BLKW("mlp.mlp.0.weight");
        blk->mlp_fc1_b   = BLKB("mlp.mlp.0.bias");
        blk->mlp_fc2_w   = BLKW("mlp.mlp.2.weight");
        blk->mlp_fc2_b   = BLKB("mlp.mlp.2.bias");

        blk->mod_bias    = BLKB("modulation");

        #undef BLKW
        #undef BLKB

        if (L == 0 && !blk->sa_qkv_w)
            fprintf(stderr, "T2: WARNING: block 0 self_attn missing\n");

        /* Save CPU copies only when layer streaming/debug explicitly needs them.
         * The default full-GPU path used to upload every block then copy it back
         * to CPU even though the copies were never read; skipping that removes a
         * large cold-load cost without changing forward numerics. */
        if (t2_dit_keep_cpu_blocks(r)) {
            cuStreamSynchronize(0);
            dit_block_save_cpu_copy(&r->dit_blocks_cpu[L], blk, use_f16,
                                    DIT_DIM, DIT_HEADS, DIT_HEAD_DIM,
                                    DIT_FFN, DIT_COND_DIM);
        }

        /* If streaming enabled, free GPU weights for blocks beyond limit */
        if (r->max_gpu_layers > 0 && L >= r->max_gpu_layers)
            dit_block_free_gpu(blk);
    }
    #undef UPW

    /* Precompute 3D RoPE tables on CPU, upload to GPU */
    {
        int gs = DIT_GRID, nt = DIT_N_TOKENS;
        int n_freqs = DIT_HEAD_DIM / 6;  /* 21 */
        int axis_dim = 2 * n_freqs;
        r->dit_n_freqs = n_freqs;
        r->dit_axis_dim = axis_dim;

        float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
        for (int j = 0; j < n_freqs; j++)
            freqs[j] = 1.0f / powf(10000.0f, (float)j / (float)n_freqs);

        size_t table_sz = (size_t)nt * 3 * n_freqs * sizeof(float);
        float *cos_tab = (float *)malloc(table_sz);
        float *sin_tab = (float *)malloc(table_sz);

        for (int i = 0; i < nt; i++) {
            int z = i / (gs * gs), y = (i / gs) % gs, x = i % gs;
            float coords[3] = {(float)z, (float)y, (float)x};
            for (int axis = 0; axis < 3; axis++) {
                for (int j = 0; j < n_freqs; j++) {
                    float theta = coords[axis] * freqs[j];
                    cos_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                    sin_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
                }
            }
        }

        r->dit_rope_cos = cu_upload_raw(cos_tab, table_sz);
        r->dit_rope_sin = cu_upload_raw(sin_tab, table_sz);
        free(freqs); free(cos_tab); free(sin_tab);
        fprintf(stderr, "T2: RoPE tables uploaded (%d freqs/axis)\n", n_freqs);
    }

    r->dit_use_f16 = use_f16;
    safetensors_close(st);
    { size_t fr = 0, to = 0; cuMemGetInfo(&fr, &to);
      int on_gpu = r->max_gpu_layers > 0 ? (r->max_gpu_layers < DIT_DEPTH ? r->max_gpu_layers : DIT_DEPTH) : DIT_DEPTH;
      fprintf(stderr, "T2: DiT loaded (%d blocks, %d on GPU, weights=%s, GPU %.0f/%.0f MB free)\n",
              DIT_DEPTH, on_gpu, use_f16 ? "F16" : "F32", fr/1048576.0, to/1048576.0); }
    t2_timing_log("load_stage1_dit", t_load0);
    return 0;
}

static int load_decoder_weights(cuda_trellis2_runner *r, const char *path) {
    double t_load0 = t2_now_ms();
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading decoder from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;

    r->dec_conv_in_w = t2_upload_f32_fast(r, st, "input_layer.weight", v);
    r->dec_conv_in_b = t2_upload_f32_fast(r, st, "input_layer.bias", v);

    /* Load ResBlock weights */
    #define LOAD_RES(rb, prefix) do { \
        char _n[256]; \
        snprintf(_n, sizeof(_n), "%snorm1.weight", prefix); (rb).gn1_w = t2_upload_f32_fast(r, st, _n, v>=2?v:0); \
        snprintf(_n, sizeof(_n), "%snorm1.bias", prefix);   (rb).gn1_b = t2_upload_f32_fast(r, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv1.weight", prefix); (rb).conv1_w = t2_upload_f32_fast(r, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv1.bias", prefix);   (rb).conv1_b = t2_upload_f32_fast(r, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%snorm2.weight", prefix); (rb).gn2_w = t2_upload_f32_fast(r, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%snorm2.bias", prefix);   (rb).gn2_b = t2_upload_f32_fast(r, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv2.weight", prefix); (rb).conv2_w = t2_upload_f32_fast(r, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv2.bias", prefix);   (rb).conv2_b = t2_upload_f32_fast(r, st, _n, 0); \
    } while(0)

    LOAD_RES(r->dec_middle[0], "middle_block.0.");
    LOAD_RES(r->dec_middle[1], "middle_block.1.");
    LOAD_RES(r->dec_res16[0],  "blocks.0.");
    LOAD_RES(r->dec_res16[1],  "blocks.1.");
    r->dec_up1_w = t2_upload_f32_fast(r, st, "blocks.2.conv.weight", v);
    r->dec_up1_b = t2_upload_f32_fast(r, st, "blocks.2.conv.bias", v);
    LOAD_RES(r->dec_res32[0],  "blocks.3.");
    LOAD_RES(r->dec_res32[1],  "blocks.4.");
    r->dec_up2_w = t2_upload_f32_fast(r, st, "blocks.5.conv.weight", v);
    r->dec_up2_b = t2_upload_f32_fast(r, st, "blocks.5.conv.bias", v);
    LOAD_RES(r->dec_res64[0],  "blocks.6.");
    LOAD_RES(r->dec_res64[1],  "blocks.7.");
    r->dec_out_gn_w   = t2_upload_f32_fast(r, st, "out_layer.0.weight", v);
    r->dec_out_gn_b   = t2_upload_f32_fast(r, st, "out_layer.0.bias", v);
    r->dec_out_conv_w  = t2_upload_f32_fast(r, st, "out_layer.2.weight", v);
    r->dec_out_conv_b  = t2_upload_f32_fast(r, st, "out_layer.2.bias", v);

    #undef LOAD_RES

    safetensors_close(st);
    fprintf(stderr, "T2: decoder loaded\n");
    t2_timing_log("load_stage1_decoder", t_load0);
    return 0;
}

static int load_dinov3_weights(cuda_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading DINOv3 from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;

    /* Patch embedding + tokens (timm naming) */
    r->dino_patch_w = t2_upload_f32(st, "patch_embed.proj.weight", v);
    r->dino_patch_b = t2_upload_f32(st, "patch_embed.proj.bias", v);
    r->dino_cls_token = t2_upload_f32(st, "cls_token", v);
    r->dino_storage_tokens = t2_upload_f32(st, "reg_token", v);

    r->dino_has_layerscale = 0;
    r->dino_has_qk_norm = 0;

    /* Per-layer weights (timm naming: fused QKV → split to separate Q/K/V) */
    for (int L = 0; L < DINO_LAYERS; L++) {
        dino_layer_gpu *l = &r->dino_layers[L];
        char name[256];

        #define DINO_LOAD(field, suffix) do { \
            snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix); \
            l->field = t2_upload_f32(st, name, v >= 2 ? v : 0); \
        } while(0)

        DINO_LOAD(ln1_w, "norm1.weight");
        DINO_LOAD(ln1_b, "norm1.bias");

        /* Split fused QKV [3*dim, dim] into separate Q, K, V on CPU */
        snprintf(name, sizeof(name), "blocks.%d.attn.qkv.weight", L);
        int idx = safetensors_find(st, name);
        if (idx >= 0) {
            size_t nbytes = safetensors_nbytes(st, idx);
            const char *dtype = safetensors_dtype(st, idx);
            void *data = safetensors_data(st, idx);
            int n_elem = (int)(nbytes / (strcmp(dtype, "F32") == 0 ? 4 : 2));
            int dim = DINO_HIDDEN;
            float *f32 = (float *)malloc(n_elem * sizeof(float));
            if (strcmp(dtype, "F32") == 0) memcpy(f32, data, n_elem * 4);
            else if (strcmp(dtype, "F16") == 0) {
                const uint16_t *src = (const uint16_t *)data;
                for (int i = 0; i < n_elem; i++) {
                    uint16_t h = src[i]; uint32_t sign = ((uint32_t)h & 0x8000) << 16;
                    uint32_t exp = (h>>10)&0x1f; uint32_t mant = h&0x3ff; uint32_t f;
                    if (exp==0) { if (mant==0) f=sign; else { exp=1; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3ff; f=sign|((exp+127-15)<<23)|(mant<<13); }}
                    else if (exp==31) f=sign|0x7f800000|(mant<<13); else f=sign|((exp+127-15)<<23)|(mant<<13);
                    memcpy(&f32[i], &f, 4);
                }
            }
            /* Split: Q=[dim, dim], K=[dim, dim], V=[dim, dim] */
            l->q_w = cu_upload_raw(f32, (size_t)dim * dim * 4);
            l->k_w = cu_upload_raw(f32 + dim * dim, (size_t)dim * dim * 4);
            l->v_w = cu_upload_raw(f32 + 2 * dim * dim, (size_t)dim * dim * 4);
            free(f32);
            /* timm has no QKV bias — create zeros */
            float *zeros = (float *)calloc(dim, sizeof(float));
            l->q_b = cu_upload_raw(zeros, dim * 4);
            l->k_b = cu_upload_raw(zeros, dim * 4);
            l->v_b = cu_upload_raw(zeros, dim * 4);
            free(zeros);
        }

        DINO_LOAD(out_w, "attn.proj.weight");
        DINO_LOAD(out_b, "attn.proj.bias");
        DINO_LOAD(ln2_w, "norm2.weight");
        DINO_LOAD(ln2_b, "norm2.bias");
        DINO_LOAD(fc1_w, "mlp.fc1.weight");
        DINO_LOAD(fc1_b, "mlp.fc1.bias");
        DINO_LOAD(fc2_w, "mlp.fc2.weight");
        DINO_LOAD(fc2_b, "mlp.fc2.bias");

        /* LayerScale (timm: gamma_1, gamma_2) */
        snprintf(name, sizeof(name), "blocks.%d.gamma_1", L);
        l->ls1 = t2_upload_f32(st, name, 0);
        snprintf(name, sizeof(name), "blocks.%d.gamma_2", L);
        l->ls2 = t2_upload_f32(st, name, 0);
        if (l->ls1) r->dino_has_layerscale = 1;

        /* QK norm (not present in timm ViT-L) */
        l->q_norm_w = 0; l->k_norm_w = 0;

        #undef DINO_LOAD
    }

    /* Precompute 2D RoPE tables for 32×32 grid */
    {
        int gh = DINO_IMG_SIZE / DINO_PATCH;  /* 32 */
        int np = gh * gh;  /* 1024 */
        int hd = DINO_HEAD_DIM;  /* 64 */
        float rope_theta = 100.0f;
        int n_freqs = hd / 4;  /* 16 */

        float *inv_freq = (float *)malloc(n_freqs * sizeof(float));
        for (int j = 0; j < n_freqs; j++)
            inv_freq[j] = 1.0f / powf(rope_theta, (float)(4 * j) / (float)hd);

        float *cos_tab = (float *)malloc((size_t)np * hd * sizeof(float));
        float *sin_tab = (float *)malloc((size_t)np * hd * sizeof(float));

        for (int p = 0; p < np; p++) {
            int py = p / gh, px = p % gh;
            float cy = ((0.5f + py) / gh) * 2.0f - 1.0f;
            float cx = ((0.5f + px) / gh) * 2.0f - 1.0f;

            /* angles: [y_freqs(16), x_freqs(16)] tiled 2× = [y16, x16, y16, x16] = 64 */
            float angles[64];
            for (int j = 0; j < n_freqs; j++) {
                angles[j]              = 2.0f * 3.14159265358979f * cy * inv_freq[j];
                angles[n_freqs + j]    = 2.0f * 3.14159265358979f * cx * inv_freq[j];
                angles[2*n_freqs + j]  = angles[j];       /* tile */
                angles[3*n_freqs + j]  = angles[n_freqs + j];
            }
            for (int d = 0; d < hd; d++) {
                cos_tab[p * hd + d] = cosf(angles[d]);
                sin_tab[p * hd + d] = sinf(angles[d]);
            }
        }
        r->dino_rope_cos = cu_upload_raw(cos_tab, (size_t)np * hd * sizeof(float));
        r->dino_rope_sin = cu_upload_raw(sin_tab, (size_t)np * hd * sizeof(float));
        free(inv_freq); free(cos_tab); free(sin_tab);
        fprintf(stderr, "T2: DINOv3 RoPE tables uploaded (16 freqs, 32x32 grid)\n");
    }

    safetensors_close(st);
    fprintf(stderr, "T2: DINOv3 loaded (%d blocks, layerscale=%d)\n",
            DINO_LAYERS, r->dino_has_layerscale);
    return 0;
}

/* Generic DiT weight loader (used for Stage 2 and Stage 3).
 * Both use identical architecture with different in_channels. */
static int load_dit_model_weights(cuda_trellis2_runner *r, const char *path,
                                    dit_model_gpu *m, dit_block_gpu *blocks_arr,
                                    dit_block_cpu *blocks_cpu_arr,
                                    const char *label) {
    double t_load0 = t2_now_ms();
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading %s from %s (%d tensors)\n", label, path, st->n_tensors);
    int v = r->verbose;
    /* Stage 2/3 (sparse) default to F32 + cuBLAS TF32 (matches Stage 1). Profiling
     * (nsys, 2026-05-30) showed the hand-written F16-MMA gemm_f16_f32 was the #1
     * Stage-2 cost (46.7%) and SLOWER per-token than Stage 1's cuBLAS TF32, despite
     * fewer voxels: full Stage-2 sampler 1924 -> 1417 ms/forward (1.36x) switching
     * F16-MMA -> cuBLAS TF32, with equivalent accuracy (cosine 0.985372 -> 0.985343).
     * VRAM (F32 = 5.3 GB vs F16 2.5 GB) is fine under the default lazy per-stage load
     * (one DiT resident at a time). T2_DIT_F16=1 forces the old fp16 MMA path. */
    int use_f16 = t2_dit_use_f16(r, 0);

    m->n_blocks = DIT_DEPTH;
    m->model_channels = DIT_DIM;
    m->n_heads = DIT_HEADS;
    m->head_dim = DIT_HEAD_DIM;
    m->ffn_hidden = DIT_FFN;
    m->cond_dim = DIT_COND_DIM;
    m->blocks = blocks_arr;
    m->use_f16 = use_f16;

    /* Detect in_channels and out_channels from weight shapes */
    {
        int idx = safetensors_find(st, "input_layer.weight");
        if (idx >= 0) m->in_channels = (int)safetensors_shape(st, idx)[1];
        else m->in_channels = 32;
        idx = safetensors_find(st, "out_layer.weight");
        if (idx >= 0) m->out_channels = (int)safetensors_shape(st, idx)[0];
        else m->out_channels = m->in_channels;
    }

    /* Timestep MLP and modulation: keep F32 (n_tok=1, GEMM perf irrelevant).
     * Modulation kernel reads weights as F32 directly. */
    m->t_fc1_w = t2_upload_f32_fast(r, st, "t_embedder.mlp.0.weight", v);
    m->t_fc1_b = t2_upload_f32_fast(r, st, "t_embedder.mlp.0.bias", v);
    m->t_fc2_w = t2_upload_f32_fast(r, st, "t_embedder.mlp.2.weight", v);
    m->t_fc2_b = t2_upload_f32_fast(r, st, "t_embedder.mlp.2.bias", v);
    m->mod_w   = t2_upload_f32_fast(r, st, "adaLN_modulation.1.weight", v);
    m->mod_b   = t2_upload_f32_fast(r, st, "adaLN_modulation.1.bias", v);

    /* Input/output embedding: F16 weight, F32 bias */
    #define UPW(name) (use_f16 ? t2_upload_f16(st, name, v) : t2_upload_f32_fast(r, st, name, v))
    m->x_emb_w = UPW("input_layer.weight");
    m->x_emb_b = t2_upload_f32_fast(r, st, "input_layer.bias", v);
    m->out_w   = UPW("out_layer.weight");
    m->out_b   = t2_upload_f32_fast(r, st, "out_layer.bias", v);

    for (int L = 0; L < m->n_blocks; L++) {
        dit_block_gpu *blk = &m->blocks[L];
        char name[256];
        #define BLK2W(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                               UPW(name))
        #define BLK2B(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                               t2_upload_f32_fast(r, st, name, v >= 2 ? v : 0))
        blk->sa_qkv_w  = BLK2W("self_attn.to_qkv.weight");
        blk->sa_qkv_b  = BLK2B("self_attn.to_qkv.bias");
        blk->sa_q_norm  = BLK2B("self_attn.q_rms_norm.gamma");
        blk->sa_k_norm  = BLK2B("self_attn.k_rms_norm.gamma");
        blk->sa_out_w   = BLK2W("self_attn.to_out.weight");
        blk->sa_out_b   = BLK2B("self_attn.to_out.bias");
        blk->norm2_w    = BLK2B("norm2.weight");
        blk->norm2_b    = BLK2B("norm2.bias");
        blk->ca_q_w     = BLK2W("cross_attn.to_q.weight");
        blk->ca_q_b     = BLK2B("cross_attn.to_q.bias");
        blk->ca_kv_w    = BLK2W("cross_attn.to_kv.weight");
        blk->ca_kv_b    = BLK2B("cross_attn.to_kv.bias");
        blk->ca_q_norm  = BLK2B("cross_attn.q_rms_norm.gamma");
        blk->ca_k_norm  = BLK2B("cross_attn.k_rms_norm.gamma");
        blk->ca_out_w   = BLK2W("cross_attn.to_out.weight");
        blk->ca_out_b   = BLK2B("cross_attn.to_out.bias");
        blk->mlp_fc1_w  = BLK2W("mlp.mlp.0.weight");
        blk->mlp_fc1_b  = BLK2B("mlp.mlp.0.bias");
        blk->mlp_fc2_w  = BLK2W("mlp.mlp.2.weight");
        blk->mlp_fc2_b  = BLK2B("mlp.mlp.2.bias");
        blk->mod_bias   = BLK2B("modulation");
        #undef BLK2W
        #undef BLK2B

        if (t2_dit_keep_cpu_blocks(r)) {
            cuStreamSynchronize(0);
            dit_block_save_cpu_copy(&blocks_cpu_arr[L], blk, use_f16,
                                    DIT_DIM, DIT_HEADS, DIT_HEAD_DIM,
                                    DIT_FFN, DIT_COND_DIM);
        }

        if (r->max_gpu_layers > 0 && L >= r->max_gpu_layers)
            dit_block_free_gpu(blk);
    }
    #undef UPW

    m->rope_cos = 0; m->rope_sin = 0;
    m->n_rope_freqs = DIT_HEAD_DIM / 6;
    m->rope_axis_dim = 2 * m->n_rope_freqs;

    safetensors_close(st);
    { size_t fr = 0, to = 0; cuMemGetInfo(&fr, &to);
      int on_gpu = r->max_gpu_layers > 0 ? (r->max_gpu_layers < m->n_blocks ? r->max_gpu_layers : m->n_blocks) : m->n_blocks;
      fprintf(stderr, "T2: %s loaded (%d blocks, %d on GPU, in_ch=%d, weights=%s, GPU %.0f/%.0f MB free)\n",
              label, m->n_blocks, on_gpu, m->in_channels, use_f16 ? "F16" : "F32",
              fr/1048576.0, to/1048576.0); }
    {
        char timing_label[128];
        snprintf(timing_label, sizeof(timing_label), "load_%s", label);
        t2_timing_log(timing_label, t_load0);
    }
    return 0;
}

int cuda_trellis2_load_stage2(cuda_trellis2_runner *r, const char *stage2_path) {
    int ret = load_dit_model_weights(r, stage2_path, &r->stage2, r->stage2_blocks,
                                      r->stage2_blocks_cpu, "Stage 2");
    if (ret == 0) {
        r->stage2_loaded = 1;
        cuda_trellis2_invalidate_kv_cache(r);
    }
    return ret;
}

int cuda_trellis2_load_stage3(cuda_trellis2_runner *r, const char *stage3_path) {
    int ret = load_dit_model_weights(r, stage3_path, &r->stage3, r->stage3_blocks,
                                      r->stage3_blocks_cpu, "Stage 3");
    if (ret == 0) {
        r->stage3_loaded = 1;
        cuda_trellis2_invalidate_kv_cache(r);
    }
    return ret;
}

/* ---- Shape decoder weight loading ---- */

/* Upload sparse conv weight transposed: [out_C, 27, in_C] -> [27, out_C, in_C]
 * This layout enables gather-GEMM: for each k in 0..26, use weight[k*out_C*in_C] */
static CUdeviceptr t2_upload_conv_transposed(cuda_trellis2_runner *r,
                                               st_context *st, const char *name,
                                               int out_C, int in_C, int v,
                                               int use_f16) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { if (v) fprintf(stderr, "  [MISSING] %s\n", name); return 0; }
    const char *dtype = safetensors_dtype(st, idx);
    void *data = safetensors_data(st, idx);

    size_t n_elem = (size_t)out_C * 27 * in_C;
    size_t nbytes = safetensors_nbytes(st, idx);

    if (!use_f16) {
        int is_f16 = strcmp(dtype, "F16") == 0;
        int is_bf16 = strcmp(dtype, "BF16") == 0;
        int cpu_conv = getenv("T2_CPU_SCVAE_CONV_UPLOAD") &&
                       atoi(getenv("T2_CPU_SCVAE_CONV_UPLOAD"));
        int cpu_f16 = getenv("T2_CPU_F16_UPLOAD") && atoi(getenv("T2_CPU_F16_UPLOAD"));
        int cpu_bf16 = getenv("T2_CPU_BF16_UPLOAD") && atoi(getenv("T2_CPU_BF16_UPLOAD"));
        CUfunction fn = r ? (is_f16 ? r->ops.conv_transpose_f16_to_f32 :
                             is_bf16 ? r->ops.conv_transpose_bf16_to_f32 : NULL) : NULL;
        if (r && fn && !cpu_conv && !(is_f16 && cpu_f16) && !(is_bf16 && cpu_bf16)) {
            CUdeviceptr d_src = 0, d_dst = 0;
            if (cuMemAlloc(&d_src, nbytes) == CUDA_SUCCESS &&
                cuMemAlloc(&d_dst, n_elem * sizeof(float)) == CUDA_SUCCESS &&
                cuMemcpyHtoDAsync(d_src, data, nbytes, r->stream) == CUDA_SUCCESS) {
                long total = (long)n_elem;
                void *args[] = {&d_src, &d_dst, &out_C, &in_C};
                if (cuLaunchKernel(fn, (unsigned)((total + 255) / 256),
                                   1, 1, 256, 1, 1, 0, r->stream, args, NULL) == CUDA_SUCCESS &&
                    cuStreamSynchronize(r->stream) == CUDA_SUCCESS) {
                    cuMemFree(d_src);
                    if (v >= 2)
                        fprintf(stderr, "  %s: [%d,27,%d] -> [27,%d,%d] F32 GPU via 16-bit upload\n",
                                name, out_C, in_C, out_C, in_C);
                    return d_dst;
                }
            }
            if (d_src) cuMemFree(d_src);
            if (d_dst) cuMemFree(d_dst);
        }
    }

    /* Fallback: convert/transpose on CPU, then upload. */
    float *f32 = (float *)malloc(n_elem * sizeof(float));
    const uint16_t *src16 = (const uint16_t *)data;

    if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, data, n_elem * sizeof(float));
    } else if (strcmp(dtype, "BF16") == 0) {
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src16[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F16") == 0) {
        for (size_t i = 0; i < n_elem; i++) {
            f32[i] = t2_f16_to_f32_value(src16[i]);
        }
    } else {
        free(f32); return 0;
    }

    /* Transpose: [out_C, 27, in_C] -> [27, out_C, in_C] */
    float *trans = (float *)malloc(n_elem * sizeof(float));
    for (int o = 0; o < out_C; o++)
        for (int k = 0; k < 27; k++)
            memcpy(trans + ((size_t)k * out_C + o) * in_C,
                   f32 + ((size_t)o * 27 + k) * in_C,
                   (size_t)in_C * sizeof(float));
    free(f32);

    CUdeviceptr d;
    if (use_f16) {
        /* Convert transposed F32 to F16 */
        uint16_t *f16 = (uint16_t *)malloc(n_elem * sizeof(uint16_t));
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t fb; memcpy(&fb, &trans[i], 4);
            uint32_t s = (fb >> 16) & 0x8000;
            int32_t e = ((fb >> 23) & 0xff) - 127 + 15;
            uint32_t m = (fb >> 13) & 0x3ff;
            if (e <= 0) f16[i] = (uint16_t)s;
            else if (e >= 31) f16[i] = (uint16_t)(s | 0x7c00);
            else f16[i] = (uint16_t)(s | (e << 10) | m);
        }
        d = cu_upload_raw(f16, n_elem * sizeof(uint16_t));
        free(f16);
    } else {
        d = cu_upload_raw(trans, n_elem * sizeof(float));
    }
    free(trans);

    if (v >= 2)
        fprintf(stderr, "  %s: [%d,27,%d] -> [27,%d,%d] %s GPU\n",
                name, out_C, in_C, out_C, in_C, use_f16 ? "F16" : "F32");
    return d;
}

static int cuda_trellis2_load_scvae_decoder(cuda_trellis2_runner *r,
                                             t2_scvae_dec_gpu *dec,
                                             const char *path,
                                             const char *label) {
    double t_load0 = t2_now_ms();
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading %s from %s (%d tensors)\n", label, path, st->n_tensors);
    int v = r->verbose;
    int use_f16 = 0;
    {
        const char *f16 = getenv("T2_SHAPE_DEC_F16");
        if (f16 && atoi(f16) && r->ops.sm_version >= 70) use_f16 = 1;
        const char *e = getenv("T2_SHAPE_DEC_F32");
        if (e && atoi(e)) use_f16 = 0;
    }
    scvae_decoder_free_gpu(dec);
    dec->use_f16 = use_f16;

    /* Weight names match: blocks.{stage}.{block}.{component} */
    #define UPW(n) (use_f16 ? t2_upload_f16(st, n, v) : t2_upload_f32_fast(r, st, n, v))
    #define UPB(n) t2_upload_f32_fast(r, st, n, v)

    dec->from_latent_w = UPW("from_latent.weight");
    dec->from_latent_b = UPB("from_latent.bias");
    dec->out_w = UPW("output_layer.weight");
    dec->out_b = UPB("output_layer.bias");
    {
        int oi = safetensors_find(st, "output_layer.weight");
        if (oi >= 0 && safetensors_ndims(st, oi) >= 2) {
            const uint64_t *sh = safetensors_shape(st, oi);
            dec->out_channels = (int)sh[0];
        } else {
            dec->out_channels = 7;
        }
    }

    /* Channel progression: 1024 -> 512 -> 256 -> 128 -> 64 */
    int ch[] = {1024, 512, 256, 128, 64};
    int nblk[] = {4, 16, 8, 4};
    memcpy(dec->channels, ch, sizeof(ch));
    memcpy(dec->n_convnext, nblk, sizeof(nblk));

    for (int s = 0; s < 4; s++) {
        int C = ch[s];

        /* ConvNeXt blocks */
        for (int b = 0; b < nblk[s]; b++) {
            char name[256];
            #define CN(suf) (snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, b, suf), name)
            dec->convnext[s][b].conv_w = t2_upload_conv_transposed(
                r, st, CN("conv.weight"), C, C, v, use_f16);
            dec->convnext[s][b].conv_b = UPB(CN("conv.bias"));
            dec->convnext[s][b].norm_w = UPB(CN("norm.weight"));
            dec->convnext[s][b].norm_b = UPB(CN("norm.bias"));
            dec->convnext[s][b].mlp0_w = UPW(CN("mlp.0.weight"));
            dec->convnext[s][b].mlp0_b = UPB(CN("mlp.0.bias"));
            dec->convnext[s][b].mlp2_w = UPW(CN("mlp.2.weight"));
            dec->convnext[s][b].mlp2_b = UPB(CN("mlp.2.bias"));
            #undef CN
        }

        /* C2S upsample block */
        {
            char name[256];
            int C_out = ch[s + 1];
            #define C2S(suf) (snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, nblk[s], suf), name)
            dec->c2s[s].norm1_w = UPB(C2S("norm1.weight"));
            dec->c2s[s].norm1_b = UPB(C2S("norm1.bias"));
            dec->c2s[s].conv1_w = t2_upload_conv_transposed(
                r, st, C2S("conv1.weight"), C_out * 8, C, v, use_f16);
            dec->c2s[s].conv1_b = UPB(C2S("conv1.bias"));
            dec->c2s[s].conv2_w = t2_upload_conv_transposed(
                r, st, C2S("conv2.weight"), C_out, C_out, v, use_f16);
            dec->c2s[s].conv2_b = UPB(C2S("conv2.bias"));
            /* to_subdiv may be missing (texture decoder) */
            snprintf(name, sizeof(name), "blocks.%d.%d.to_subdiv.weight", s, nblk[s]);
            int idx = safetensors_find(st, name);
            if (idx >= 0) {
                dec->c2s[s].subdiv_w = UPW(name);
                snprintf(name, sizeof(name), "blocks.%d.%d.to_subdiv.bias", s, nblk[s]);
                dec->c2s[s].subdiv_b = UPB(name);
            } else {
                dec->c2s[s].subdiv_w = 0;
                dec->c2s[s].subdiv_b = 0;
            }
            #undef C2S
        }
    }

    #undef UPW
    #undef UPB

    safetensors_close(st);
    dec->loaded = 1;

    fprintf(stderr, "T2: %s loaded (GPU=%s, out_ch=%d)\n",
            label, use_f16 ? "F16" : "F32", dec->out_channels);
    {
        char timing_label[128];
        snprintf(timing_label, sizeof(timing_label), "load_%s", label);
        t2_timing_log(timing_label, t_load0);
    }
    return 0;
}

int cuda_trellis2_load_shape_decoder(cuda_trellis2_runner *r, const char *path) {
    return cuda_trellis2_load_scvae_decoder(r, &r->shape_dec, path, "shape decoder");
}

int cuda_trellis2_load_texture_decoder(cuda_trellis2_runner *r, const char *path) {
    return cuda_trellis2_load_scvae_decoder(r, &r->tex_dec, path, "texture decoder");
}

void cuda_trellis2_unload_shape_decoder(cuda_trellis2_runner *r) {
    if (r) scvae_decoder_free_gpu(&r->shape_dec);
}

void cuda_trellis2_unload_texture_decoder(cuda_trellis2_runner *r) {
    if (r) scvae_decoder_free_gpu(&r->tex_dec);
}

/* Per-stage DiT unloads. Each is idempotent (CU_FREE / dit_model_free_gpu zero the
 * pointers), so they are safe to call once after a stage finishes AND again via the
 * bulk cuda_trellis2_unload_dit_stages() below. The shared cross-attn KV cache is NOT
 * freed here — it is keyed by (model_id, cond_hash) and recomputed on the next stage's
 * first forward, so it is reused/overwritten rather than leaked between stages. */
void cuda_trellis2_unload_stage1(cuda_trellis2_runner *r) {
    if (!r) return;
    cuCtxSetCurrent(r->context);
    CU_FREE(r->dit_t_fc1_w); CU_FREE(r->dit_t_fc1_b);
    CU_FREE(r->dit_t_fc2_w); CU_FREE(r->dit_t_fc2_b);
    CU_FREE(r->dit_mod_w); CU_FREE(r->dit_mod_b);
    CU_FREE(r->dit_x_emb_w); CU_FREE(r->dit_x_emb_b);
    CU_FREE(r->dit_out_w); CU_FREE(r->dit_out_b);
    for (int i = 0; i < DIT_DEPTH; i++)
        dit_block_free_gpu(&r->dit_blocks[i]);
    CU_FREE(r->dit_rope_cos);
    CU_FREE(r->dit_rope_sin);
}

void cuda_trellis2_unload_stage2(cuda_trellis2_runner *r) {
    if (!r) return;
    cuCtxSetCurrent(r->context);
    dit_model_free_gpu(&r->stage2);
    t2_sparse_rope_cache_free(r, 2);
    r->stage2_loaded = 0;
}

void cuda_trellis2_unload_stage3(cuda_trellis2_runner *r) {
    if (!r) return;
    cuCtxSetCurrent(r->context);
    dit_model_free_gpu(&r->stage3);
    t2_sparse_rope_cache_free(r, 3);
    r->stage3_loaded = 0;
}

/* Free ALL three DiT stages + the shared cross-attn KV cache, once all latents are
 * produced and before the SC-VAE decoders (which need the VRAM: Stage 1 alone is F32
 * ~5.3 GB, and all three resident leave ~0 MB free → the shape decode OOMs at its
 * ~1.47M-voxel finest level). PyTorch frees these via its low_vram CPU offload.
 * Idempotent. For lazy load-run-free pipelining, call the per-stage unloads above. */
void cuda_trellis2_unload_dit_stages(cuda_trellis2_runner *r) {
    if (!r) return;
    cuCtxSetCurrent(r->context);

    cuda_trellis2_unload_stage1(r);
    cuda_trellis2_unload_stage2(r);
    cuda_trellis2_unload_stage3(r);

    /* Cross-attn KV cache (shared across DiT stages; dead once they all are). */
    for (int i = 0; i < DIT_DEPTH; i++) {
        for (int slot = 0; slot < 2; slot++) {
            CU_FREE(r->ca_kv_cache_K[slot][i]);
            CU_FREE(r->ca_kv_cache_V[slot][i]);
        }
    }
    cuda_trellis2_invalidate_kv_cache(r);

    size_t free_mem = 0, total_mem = 0;
    cuMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "T2: unloaded DiT stages 1/2/3 + KV cache (GPU free=%.0f MB / %.0f MB)\n",
            free_mem / 1048576.0, total_mem / 1048576.0);
}

int cuda_trellis2_load_weights(cuda_trellis2_runner *r,
                                const char *dinov3_path,
                                const char *stage1_path,
                                const char *decoder_path) {
    /* GEMM mode is set per-model in each forward wrapper (run_dit_forward,
     * run_stage2_forward). Default to F32 for safety. */
    r->ops.use_f32_gemm = 1;

    if (stage1_path && load_dit_weights(r, stage1_path) != 0) return -1;
    if (decoder_path && load_decoder_weights(r, decoder_path) != 0) return -1;

    if (dinov3_path && load_dinov3_weights(r, dinov3_path) != 0) return -1;
    if (stage1_path) cuda_trellis2_invalidate_kv_cache(r);

    /* Stage 2 shape flow model loads with same code as Stage 1 — just different in_channels */
    r->stage2_loaded = 0;

    return 0;
}

/* ======================================================================== */
/* DINOv3 forward pass                                                      */
/* ======================================================================== */

static void run_dinov3(cuda_trellis2_runner *r, CUdeviceptr d_image, CUdeviceptr d_out) {
    t2_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int dim = DINO_HIDDEN;        /* 1024 */
    const int heads = DINO_HEADS;       /* 16 */
    const int hd = DINO_HEAD_DIM;       /* 64 */
    const int ffn = DINO_FFN;           /* 4096 */
    const int ps = DINO_PATCH;          /* 16 */
    const int gw = DINO_IMG_SIZE / ps;  /* 32 */
    const int np = DINO_N_PATCHES;      /* 1024 */
    const int ns = DINO_N_STORAGE;      /* 4 */
    const int seq = DINO_SEQ_LEN;       /* 1029 */
    const int n_prefix = 1 + ns;        /* 5 */

    /* Scratch layout */
    ensure_scratch(r, 0, (size_t)seq * dim * sizeof(float));       /* hidden */
    ensure_scratch(r, 1, (size_t)3 * seq * dim * sizeof(float));   /* QKV */
    ensure_scratch(r, 2, (size_t)seq * dim * sizeof(float));       /* attn_out */
    ensure_scratch(r, 3, (size_t)seq * ffn * sizeof(float));       /* mlp */
    ensure_scratch(r, 4, (size_t)seq * dim * sizeof(float));       /* normed */

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_mlp    = r->scratch[3];
    CUdeviceptr d_normed = r->scratch[4];

    /* 1. Patch embedding */
    /* Write patches starting at offset n_prefix in hidden buffer */
    CUdeviceptr d_patches = d_hidden + (size_t)n_prefix * dim * sizeof(float);
    {
        int gw2 = gw, dim2 = dim, ps2 = ps, iw = DINO_IMG_SIZE;
        void *args[] = {&d_patches, &d_image, &r->dino_patch_w, &r->dino_patch_b,
                        &gw2, &dim2, &ps2, &iw};
        cuLaunchKernel(ops->dinov3_patch_embed, (unsigned)(gw * gw), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 2. Prepend CLS + register tokens */
    {
        int np2 = np, ns2 = ns, dim2 = dim;
        void *args[] = {&d_hidden, &d_patches, &r->dino_cls_token, &r->dino_storage_tokens,
                        &np2, &ns2, &dim2};
        cuLaunchKernel(ops->dinov3_prepend_tokens,
                       (unsigned)((seq * dim + 255) / 256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 3. Transformer blocks */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        /* LN1 → Q, K, V projections */
        t2_op_layernorm(ops, stream, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        CUdeviceptr d_Q = d_qkv;
        CUdeviceptr d_K = d_qkv + (size_t)seq * dim * sizeof(float);
        CUdeviceptr d_V = d_qkv + (size_t)2 * seq * dim * sizeof(float);
        t2_op_gemm(ops, stream, d_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        t2_op_gemm(ops, stream, d_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        t2_op_gemm(ops, stream, d_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        /* 2D RoPE on Q and K (patch tokens only, skip prefix) */
        {
            int np3 = np, npx = n_prefix, dim3 = dim, nh = heads, hd3 = hd;
            void *args[] = {&d_Q, &d_K, &r->dino_rope_cos, &r->dino_rope_sin,
                            &np3, &npx, &dim3, &nh, &hd3};
            cuLaunchKernel(ops->rope_2d_dinov3, (unsigned)np, 1, 1,
                           256, 1, 1, 0, stream, args, NULL);
        }

        /* Self-attention */
        t2_op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, seq, dim, heads, hd);

        /* Output projection */
        t2_op_gemm(ops, stream, d_normed, l->out_w, d_attn, l->out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (l->ls1) {
            /* layerscale_add: dst[i] += src[i] * scale[i % dim] */
            int n = seq * dim;
            int dim_arg = dim;
            void *ls_args[] = {&d_hidden, &d_normed, &l->ls1, &n, &dim_arg};
            cuLaunchKernel(ops->layerscale_add, (unsigned)((n+255)/256), 1, 1,
                           256, 1, 1, 0, stream, ls_args, NULL);
        } else {
            t2_op_add(ops, stream, d_hidden, d_normed, seq * dim);
        }

        /* LN2 → MLP → LayerScale 2 + residual */
        t2_op_layernorm(ops, stream, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        t2_op_gemm(ops, stream, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        t2_op_gelu(ops, stream, d_mlp, seq * ffn);
        t2_op_gemm(ops, stream, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2) {
            int n = seq * dim;
            int dim_arg = dim;
            void *ls_args[] = {&d_hidden, &d_normed, &l->ls2, &n, &dim_arg};
            cuLaunchKernel(ops->layerscale_add, (unsigned)((n+255)/256), 1, 1,
                           256, 1, 1, 0, stream, ls_args, NULL);
        } else {
            t2_op_add(ops, stream, d_hidden, d_normed, seq * dim);
        }
    }

    /* 4. Final unparameterized LayerNorm (for TRELLIS.2 compatibility) */
    {
        float eps = 1e-6f;
        int dim2 = dim;
        void *args[] = {&d_out, &d_hidden, &dim2, &eps};
        cuLaunchKernel(ops->layernorm_noaffine, (unsigned)seq, 1, 1,
                       256, 1, 1, 512 * sizeof(float), stream, args, NULL);
    }
}

/* ======================================================================== */
/* DiT forward pass (single denoising step)                                 */
/* ======================================================================== */

/* Generic DiT forward: works for both Stage 1 (N=4096, in=8) and Stage 2 (N=variable, in=32) */
static void run_dit_forward_generic(cuda_trellis2_runner *r,
                                      CUdeviceptr d_x, float timestep,
                                      CUdeviceptr d_cond,
                                      CUdeviceptr d_output,
                                      int N,                   /* token count */
                                      int in_ch,               /* input channels */
                                      int out_ch,              /* output channels (may differ from in_ch) */
                                      int n_blocks,
                                      CUdeviceptr t_fc1_w, CUdeviceptr t_fc1_b,
                                      CUdeviceptr t_fc2_w, CUdeviceptr t_fc2_b,
                                      CUdeviceptr mod_w, CUdeviceptr mod_b,
                                      CUdeviceptr x_emb_w, CUdeviceptr x_emb_b,
                                      CUdeviceptr out_w, CUdeviceptr out_b,
                                      dit_block_gpu *blocks,
                                      dit_block_cpu *blocks_cpu, /* CPU copies for streaming (NULL = all on GPU) */
                                      CUdeviceptr rope_cos, CUdeviceptr rope_sin,
                                      int n_freqs, int axis_dim,
                                      int model_id,  /* 1=Stage1, 2=Stage2, 3=Stage3 */
                                      uint64_t cond_hash,
                                      int cache_slot) {
    t2_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int D = DIT_DIM;        /* 1536 */
    const int H = DIT_HEADS;      /* 12 */
    const int HD = DIT_HEAD_DIM;  /* 128 */
    const int FFN = DIT_FFN;      /* 8192 */
    const int ctx_len = DINO_SEQ_LEN;  /* 1029 */

    /* Scratch layout:
     * 0: hidden [N * D]
     * 1: QKV / MLP shared [max(3*N*D, N*FFN)]
     * 2: attn_out + normed [2 * N * D]
     * 3: mod[6*D] + mod_base[6*D] + t_emb[D] + cross_Q[N*D]
     *    + ca_K[ctx*D] + ca_V[ctx*D] */
    size_t qkv_sz = (size_t)3 * N * D * sizeof(float);
    size_t mlp_sz = (size_t)N * FFN * sizeof(float);
    size_t ca_kv_gemm_sz = (size_t)ctx_len * 2 * D * sizeof(float);  /* cross-attn KV GEMM */
    size_t sh1 = qkv_sz > mlp_sz ? qkv_sz : mlp_sz;
    if (ca_kv_gemm_sz > sh1) sh1 = ca_kv_gemm_sz;
    size_t ca_kv_sz = (size_t)ctx_len * D * sizeof(float);
    /* t_emb MLP outputs D floats (not 256), needs D-sized buffer */
    size_t buf3_sz = (size_t)(12*D + D) * sizeof(float)
                   + (size_t)N * D * sizeof(float) + 2 * ca_kv_sz
                   + (size_t)N * D * sizeof(float); /* split V space */

    ensure_scratch(r, 0, (size_t)N * D * sizeof(float));
    ensure_scratch(r, 1, sh1);
    ensure_scratch(r, 2, (size_t)2 * N * D * sizeof(float));
    ensure_scratch(r, 3, buf3_sz);

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_mlp    = r->scratch[1]; /* alias */
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_normed = r->scratch[2] + (size_t)N * D * sizeof(float);

    CUdeviceptr d_mod      = r->scratch[3];
    CUdeviceptr d_mod_base = d_mod + (size_t)6 * D * sizeof(float);
    CUdeviceptr d_temb     = d_mod_base + (size_t)6 * D * sizeof(float);
    CUdeviceptr d_cross_Q = d_temb + (size_t)D * sizeof(float);  /* t_emb is D floats, not 256 */
    CUdeviceptr d_ca_K   = d_cross_Q + (size_t)N * D * sizeof(float);
    CUdeviceptr d_ca_V   = d_ca_K + ca_kv_sz;
    CUdeviceptr d_split_V = d_ca_V + ca_kv_sz;

    /* bf16-block port (ops->bf16_round, set by T2_DIT_BF16): round a buffer to
     * bf16 precision after each block op so the DiT runs PyTorch's bf16 trajectory.
     * No-op when bf16_round==0 (TF32/F16/F32 paths unaffected). */
    #define RB(buf, cnt) do { if (ops->bf16_round) t2_op_round_bf16(ops, stream, (buf), (long)(cnt)); } while (0)

    /* 1. Input embedding: [N, DIT_IN_CH] -> [N, D].
     * PyTorch keeps input_layer in f32, then casts h to bf16 before the blocks.
     * Force f32 here (suppress bf16-GEMM) and round h to bf16 afterwards. */
    {
        int _sb = ops->use_bf16_gemm; ops->use_bf16_gemm = 0;
        t2_op_gemm(ops, stream, d_hidden, x_emb_w, d_x, x_emb_b,
                   D, in_ch, N);
        ops->use_bf16_gemm = _sb;
    }
    RB(d_hidden, (long)N * D);   /* manual_cast(h, bf16) */

    if (r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_hidden, 16); fprintf(stderr, "  input_emb: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

    /* 2. Timestep embedding: sinusoidal(256) -> MLP(256->D->D) */
    /* TRELLIS.2: t*1000, [cos, sin] order (NOT [sin, cos] like HY3D) */
    {
        float t_scaled = timestep * 1000.0f;
        int te_dim = 256;
        void *te_args2[] = {&d_temb, &t_scaled, &te_dim};
        cuLaunchKernel(ops->timestep_embed_cossin, (unsigned)((128+255)/256), 1, 1,
                       256, 1, 1, 0, stream, te_args2, NULL);
    }

    if (r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_temb, 16); fprintf(stderr, "  sinusoidal_emb: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

    /* t_fc1: [D, 256] */
    /* Timestep MLP always uses F32 GEMM (weights are F32, n_tok=1) */
    t2_op_gemm_f32(ops, stream, d_normed, t_fc1_w, d_temb, t_fc1_b,
               D, 256, 1);
    /* SiLU */
    t2_op_silu_inplace(ops, stream, d_normed, D);
    /* t_fc2: [D, D] */
    t2_op_gemm_f32(ops, stream, d_temb, t_fc2_w, d_normed, t_fc2_b,
               D, D, 1);
    /* Now d_temb = [D] timestep embedding */
    RB(d_temb, (long)D);   /* manual_cast(t_emb, bf16) */

    if (r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_temb, 16); fprintf(stderr, "  t_emb_mlp: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

    /* Shared adaLN modulation. The block loop only adds blk->mod_bias. */
    t2_op_modulation(ops, stream, d_mod_base, d_temb,
                     mod_w, mod_b, 0, D, 6 * D);

    /* Precompute cross-attention KV for all blocks if not cached.
     * KV depends on block weights (different per block) but NOT on timestep or x_t.
     * So we compute once and cache for all subsequent calls with same cond. */
    if (cache_slot < 0 || cache_slot > 1) cache_slot = 0;
    int use_kv_cache = (r->ca_kv_cache_valid[cache_slot] &&
                        r->ca_kv_cache_model_id[cache_slot] == model_id &&
                        r->ca_kv_cache_cond_hash[cache_slot] == cond_hash &&
                        r->ca_kv_cache_n_blocks[cache_slot] == n_blocks);
    if (!use_kv_cache && d_cond) {
        /* Allocate/reallocate cache */
        size_t kv_sz = (size_t)ctx_len * D * sizeof(float);
        if (r->ca_kv_cache_n_blocks[cache_slot] != n_blocks) {
            for (int bi = 0; bi < DIT_DEPTH; bi++) {
                if (r->ca_kv_cache_K[cache_slot][bi]) cuMemFree(r->ca_kv_cache_K[cache_slot][bi]);
                if (r->ca_kv_cache_V[cache_slot][bi]) cuMemFree(r->ca_kv_cache_V[cache_slot][bi]);
                r->ca_kv_cache_K[cache_slot][bi] = 0;
                r->ca_kv_cache_V[cache_slot][bi] = 0;
            }
        }
        for (int bi = 0; bi < n_blocks; bi++) {
            dit_block_gpu *blk = &blocks[bi];
            int blk_streamed = 0;
            if (blocks_cpu && !blk->sa_qkv_w) {
                cuStreamSynchronize(stream);
                dit_block_upload_to_gpu(blk, &blocks_cpu[bi]);
                blk_streamed = 1;
            }
            if (!r->ca_kv_cache_K[cache_slot][bi]) cuMemAlloc(&r->ca_kv_cache_K[cache_slot][bi], kv_sz);
            if (!r->ca_kv_cache_V[cache_slot][bi]) cuMemAlloc(&r->ca_kv_cache_V[cache_slot][bi], kv_sz);
            /* KV GEMM: [ctx, cond_dim] -> [ctx, 2*D] */
            t2_op_gemm(ops, stream, d_qkv, blk->ca_kv_w, d_cond, blk->ca_kv_b,
                       2 * D, DIT_COND_DIM, ctx_len);
            t2_op_split_kv_chunk(ops, stream,
                                  r->ca_kv_cache_K[cache_slot][bi],
                                  r->ca_kv_cache_V[cache_slot][bi],
                                  d_qkv, ctx_len, D);
            if (blk->ca_k_norm)
                t2_op_rms_norm_perhead(ops, stream, r->ca_kv_cache_K[cache_slot][bi], blk->ca_k_norm,
                                      ctx_len, H, HD, D);
            RB(r->ca_kv_cache_K[cache_slot][bi], (long)ctx_len * D);
            RB(r->ca_kv_cache_V[cache_slot][bi], (long)ctx_len * D);
            if (blk_streamed) { cuStreamSynchronize(stream); dit_block_free_gpu(blk); }
        }
        r->ca_kv_cache_n_blocks[cache_slot] = n_blocks;
        r->ca_kv_cache_model_id[cache_slot] = model_id;
        r->ca_kv_cache_cond_hash[cache_slot] = cond_hash;
        r->ca_kv_cache_valid[cache_slot] = 1;
        if (r->verbose >= 1)
            fprintf(stderr, "  Cross-attn KV cached for %d blocks (slot=%d)\n",
                    n_blocks, cache_slot);
    }

    /* 3. Transformer blocks */
    for (int bi = 0; bi < n_blocks; bi++) {
        dit_block_gpu *blk = &blocks[bi];
        int streamed = 0;  /* 1 if we uploaded this block from CPU */

        /* Layer streaming: if block not on GPU, upload from CPU */
        if (blocks_cpu && !blk->sa_qkv_w) {
            cuStreamSynchronize(stream);  /* wait for previous work before upload */
            dit_block_upload_to_gpu(blk, &blocks_cpu[bi]);
            streamed = 1;
        }

        /* (Rounding this block's non-matmul weights — norm gammas, biases, mod_bias
         * — to bf16 was tried and is BIT-IDENTICAL: the per-op output rounding (RB)
         * already absorbs sub-bf16 weight differences, so it is omitted. Matmul
         * weights are bf16'd per-call by the bf16-GEMM path.) */

        /* 3a. Modulation: shared adaLN(t_emb) base + block_bias -> [6*D] */
        t2_op_modulation_add_bias(ops, stream, d_mod, d_mod_base,
                                  blk->mod_bias, 6 * D);
        RB(d_mod, (long)6 * D);

        if (bi == 0 && r->verbose >= 2) { cuStreamSynchronize(stream); float _d[8]; cuMemcpyDtoH(_d, d_mod, 32); fprintf(stderr, "  mod_block0: [:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3],_d[4],_d[5],_d[6],_d[7]); }
        if (bi == 0 && r->verbose >= 2) { float _s[4]; cuMemcpyDtoH(_s, d_mod+(size_t)D*sizeof(float), 16); fprintf(stderr, "  scale_sa: [:4]=%.6f %.6f %.6f %.6f\n", _s[0],_s[1],_s[2],_s[3]); }

        /* Split mod into shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp */
        CUdeviceptr d_shift_sa  = d_mod;
        CUdeviceptr d_scale_sa  = d_mod + (size_t)D * sizeof(float);
        CUdeviceptr d_gate_sa   = d_mod + (size_t)2 * D * sizeof(float);
        CUdeviceptr d_shift_mlp = d_mod + (size_t)3 * D * sizeof(float);
        CUdeviceptr d_scale_mlp = d_mod + (size_t)4 * D * sizeof(float);
        CUdeviceptr d_gate_mlp  = d_mod + (size_t)5 * D * sizeof(float);

        /* === Self-attention === */
        /* adaLN: LN(hidden) * (1+scale) + shift */
        t2_op_adaln(ops, stream, d_normed, d_hidden,
                    d_shift_sa, d_scale_sa, N, D);
        RB(d_normed, (long)N * D);

        if (bi == 0 && r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_normed, 16); fprintf(stderr, "  adaln_out: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

        /* QKV projection: [N, D] -> [N, 3*D] */
        t2_op_gemm(ops, stream, d_qkv, blk->sa_qkv_w, d_normed, blk->sa_qkv_b,
                   3 * D, D, N);
        RB(d_qkv, (long)N * 3 * D);

        /* Split QKV -> Q, K, V (standard chunk, NOT interleaved)
         * IMPORTANT: Q must NOT alias d_attn since the attention kernel reads Q
         * and writes to d_attn simultaneously. Use d_cross_Q as temp for Q. */
        CUdeviceptr d_Q = d_cross_Q;  /* use cross_Q buffer as temp for self-attn Q */
        CUdeviceptr d_K = d_normed;
        CUdeviceptr d_V = d_split_V;
        t2_op_split_qkv_chunk(ops, stream, d_Q, d_K, d_V, d_qkv, N, D);

        if (bi==0 && r->verbose>=2) dbg4("Q_raw", d_Q, stream);
        if (bi==0 && r->verbose>=2) dbg4("K_raw", d_K, stream);
        if (bi==0 && r->verbose>=2) dbg4("V_raw", d_V, stream);

        /* QK RMSNorm (per-head, gamma [n_heads * head_dim]) */
        if (blk->sa_q_norm) {
            t2_op_rms_norm_perhead(ops, stream, d_Q, blk->sa_q_norm, N, H, HD, D);
            RB(d_Q, (long)N * D);
        }
        if (blk->sa_k_norm) {
            t2_op_rms_norm_perhead(ops, stream, d_K, blk->sa_k_norm, N, H, HD, D);
            RB(d_K, (long)N * D);
        }

        if (bi==0 && r->verbose>=2) dbg4("Q_rmsnorm", d_Q, stream);
        if (bi==0 && r->verbose>=2) dbg4("K_rmsnorm", d_K, stream);

        /* 3D RoPE on Q and K */
        t2_op_rope_3d(ops, stream, d_Q, rope_cos, rope_sin,
                      N, D, H, HD, n_freqs, axis_dim);
        t2_op_rope_3d(ops, stream, d_K, rope_cos, rope_sin,
                      N, D, H, HD, n_freqs, axis_dim);
        RB(d_Q, (long)N * D);
        RB(d_K, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("Q_rope", d_Q, stream);
        if (bi==0 && r->verbose>=2) dbg4("K_rope", d_K, stream);

        /* Debug: save Q, K, V for token 0 to check attention */
        if (bi == 0 && r->verbose >= 2) {
            cuStreamSynchronize(stream);
            /* Save first head's Q for token 0 (128 floats) */
            float q0[128], k0[128], v0[128];
            cuMemcpyDtoH(q0, d_Q, 128 * sizeof(float));
            cuMemcpyDtoH(k0, d_K, 128 * sizeof(float));
            cuMemcpyDtoH(v0, d_V, 128 * sizeof(float));
            /* Compute Q[0] @ K[0]^T manually for head 0 */
            float dot = 0;
            for (int d = 0; d < 128; d++) dot += q0[d] * k0[d];
            dot /= sqrtf(128.0f);
            fprintf(stderr, "  MANUAL Q[0]@K[0]/sqrt(hd) = %.6f (should produce softmax weight for k=0)\n", dot);
            /* Also check how many Q/K elements are nonzero */
            int nz_q = 0, nz_k = 0;
            for (int d = 0; d < 128; d++) { if (q0[d] != 0.0f) nz_q++; if (k0[d] != 0.0f) nz_k++; }
            fprintf(stderr, "  Q[0] nonzero: %d/128, K[0] nonzero: %d/128\n", nz_q, nz_k);
        }

        /* Self-attention — output goes to d_attn (d_Q is separate buffer) */
        t2_op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, N, D, H, HD);
        RB(d_attn, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("attn_out", d_attn, stream);

        /* Output projection + gated residual */
        t2_op_gemm(ops, stream, d_normed, blk->sa_out_w, d_attn, blk->sa_out_b,
                   D, D, N);
        RB(d_normed, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("sa_proj", d_normed, stream);

        t2_op_gated_add(ops, stream, d_hidden, d_normed, d_gate_sa, N * D, D);
        RB(d_hidden, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("h_after_sa", d_hidden, stream);


        /* === Cross-attention === */
        /* LayerNorm (affine) */
        t2_op_layernorm(ops, stream, d_normed, d_hidden,
                        blk->norm2_w, blk->norm2_b, N, D);
        RB(d_normed, (long)N * D);

        /* Q from tokens */
        t2_op_gemm(ops, stream, d_cross_Q, blk->ca_q_w, d_normed, blk->ca_q_b,
                   D, D, N);
        if (blk->ca_q_norm)
            t2_op_rms_norm_perhead(ops, stream, d_cross_Q, blk->ca_q_norm, N, H, HD, D);
        RB(d_cross_Q, (long)N * D);

        /* KV from conditioning (cached — precomputed above) */
        CUdeviceptr d_ca_K_cached = r->ca_kv_cache_K[cache_slot][bi];
        CUdeviceptr d_ca_V_cached = r->ca_kv_cache_V[cache_slot][bi];

        /* Cross-attention (using cached KV) */
        t2_op_cross_attn(ops, stream, d_attn, d_cross_Q, d_ca_K_cached, d_ca_V_cached,
                         N, ctx_len, D, H, HD);
        RB(d_attn, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("cross_attn_out", d_attn, stream);

        /* Cross-attn output + residual (no gate) */
        t2_op_gemm(ops, stream, d_normed, blk->ca_out_w, d_attn, blk->ca_out_b,
                   D, D, N);
        RB(d_normed, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("ca_proj", d_normed, stream);

        t2_op_add(ops, stream, d_hidden, d_normed, N * D);
        RB(d_hidden, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("h_after_ca", d_hidden, stream);

        /* === MLP === */
        t2_op_adaln(ops, stream, d_normed, d_hidden,
                    d_shift_mlp, d_scale_mlp, N, D);
        RB(d_normed, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("mlp_adaln", d_normed, stream);

        t2_op_gemm(ops, stream, d_mlp, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b,
                   FFN, D, N);
        RB(d_mlp, (long)N * FFN);
        t2_op_gelu(ops, stream, d_mlp, N * FFN);
        RB(d_mlp, (long)N * FFN);
        t2_op_gemm(ops, stream, d_normed, blk->mlp_fc2_w, d_mlp, blk->mlp_fc2_b,
                   D, FFN, N);
        RB(d_normed, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("mlp_out", d_normed, stream);

        t2_op_gated_add(ops, stream, d_hidden, d_normed, d_gate_mlp, N * D, D);
        RB(d_hidden, (long)N * D);

        if (bi==0 && r->verbose>=2) dbg4("h_after_mlp", d_hidden, stream);

        /* Per-block debug */
        if (r->verbose >= 2) {
            cuStreamSynchronize(stream);
            float hs[4];
            cuMemcpyDtoH(hs, d_hidden, 4 * sizeof(float));
            float *hc = (float *)malloc(4096 * sizeof(float));
            cuMemcpyDtoH(hc, d_hidden, 4096 * sizeof(float));
            double sm = 0, sm2 = 0;
            for (int j = 0; j < 4096; j++) { sm += hc[j]; sm2 += (double)hc[j]*hc[j]; }
            fprintf(stderr, "  block %2d: std=%10.4f  h[:4]=%.4f %.4f %.4f %.4f\n",
                    bi, sqrt(sm2/4096 - (sm/4096)*(sm/4096)), hs[0], hs[1], hs[2], hs[3]);
            free(hc);
        }

        /* Layer streaming: free GPU weights after processing this block */
        if (streamed) {
            cuStreamSynchronize(stream);
            dit_block_free_gpu(blk);
        }
    }

    /* 4. Final LayerNorm (no affine) + output projection.
     * PyTorch casts h back to f32 here, then runs out_layer in f32 (the final LN
     * reads the bf16 h but computes/outputs f32; out_layer is f32). So suppress
     * bf16-GEMM for the out projection and do NOT round its output. */
    {
        float eps = 1e-6f;
        int D_arg = D;
        void *ln_args[] = {&d_hidden, &d_hidden, &D_arg, &eps};
        cuLaunchKernel(ops->layernorm_noaffine, (unsigned)N, 1, 1,
                       256, 1, 1, 512 * sizeof(float), stream, ln_args, NULL);
    }
    {
        int _sb = ops->use_bf16_gemm; ops->use_bf16_gemm = 0;
        t2_op_gemm(ops, stream, d_output, out_w, d_hidden, out_b,
                   out_ch, D, N);
        ops->use_bf16_gemm = _sb;
    }
    #undef RB
}

/* Stage 1 wrapper — uses F16 MMA if weights are F16, else F32 */
static void run_dit_forward(cuda_trellis2_runner *r,
                             CUdeviceptr d_x, float timestep,
                             CUdeviceptr d_cond, CUdeviceptr d_output,
                             uint64_t cond_hash, int cache_slot) {
    /* Save GEMM mode and set appropriate mode for Stage 1 */
    int saved_f32 = r->ops.use_f32_gemm;
    int saved_mma = r->ops.use_mma_gemm;
    int saved_tf32 = r->ops.use_tf32_gemm;
    int saved_bf16 = r->ops.use_bf16_gemm;
    int saved_round = r->ops.bf16_round;
    if (r->dit_use_f16) {
        r->ops.use_f32_gemm = 0;
        r->ops.use_mma_gemm = 1;
    } else {
        r->ops.use_f32_gemm = 1;
        r->ops.use_mma_gemm = 0;
        /* T2_DIT_BF16=1 (default OFF): full bf16 DiT-block port for matching the
         * PyTorch reference. PyTorch runs the 30 DiT blocks in bf16 (input_layer/
         * t_embedder/out_layer stay f32; h is cast to bf16 before the blocks and
         * back to f32 after). We replicate that trajectory: bf16_round rounds every
         * block-op OUTPUT to bf16 precision (LN/adaLN/attn/rope/gelu/residuals), and
         * use_bf16_gemm makes every block matmul a true bf16 matmul (W,X->bf16, f32
         * accumulate via CUBLAS_COMPUTE_32F).
         *   RESULT (2026-05-29, 12-step Stage-1 latent cosine vs 03_ss_latent):
         *     TF32 (default)        0.98954
         *     bf16-GEMM only        0.98848  (NEGATIVE: matmul inputs only)
         *     full bf16-block port  0.99739  (this path)  <- 0.990 -> 0.997
         *   The bf16 block is the lever (the gap was the non-matmul ops, not the
         *   matmuls). Negative sub-findings while chasing 0.999:
         *     - rounding non-matmul WEIGHTS (norm gammas/biases/mod_w) to bf16 is
         *       BIT-IDENTICAL — the per-op output rounding already absorbs it.
         *     - rounding attention PROBS to bf16 made it WORSE (0.9974->0.9961):
         *       PyTorch SDPA keeps the attention internals in f32, so our f16-MMA
         *       probs are already the closer match (see attn_mma_hd128_f32).
         *   The residual 0.0026 is irreducible: independent bf16 GEMM libraries
         *   (cuBLAS vs PyTorch/cuDNN) round at different points and accumulate over
         *   the 12 recursive Euler steps (single forward is already cosine 0.9998).
         * Default remains TF32 (fast, f32 range, MORE precise than the bf16 ref —
         * use it for production quality); opt out of TF32 with T2_DIT_NO_TF32=1.
         * NOTE: bf16 mode mutates nothing persistent but re-casts weights per GEMM
         * (a resident bf16 weight cache would speed it up; ~parity with TF32 today). */
        const char *want_bf16 = getenv("T2_DIT_BF16");
        if (want_bf16 && atoi(want_bf16) && r->ops.cublas &&
            r->ops.cast_f32_to_bf16 && r->ops.round_bf16) {
            r->ops.use_bf16_gemm = 1;
            r->ops.bf16_round = 1;
            r->ops.use_tf32_gemm = 0;
        } else {
            const char *no_tf32 = getenv("T2_DIT_NO_TF32");
            r->ops.use_tf32_gemm = (r->ops.cublas && !(no_tf32 && atoi(no_tf32))) ? 1 : 0;
        }
    }

    run_dit_forward_generic(r, d_x, timestep, d_cond, d_output,
        DIT_N_TOKENS, DIT_IN_CH, DIT_IN_CH, DIT_DEPTH,
        r->dit_t_fc1_w, r->dit_t_fc1_b, r->dit_t_fc2_w, r->dit_t_fc2_b,
        r->dit_mod_w, r->dit_mod_b,
        r->dit_x_emb_w, r->dit_x_emb_b,
        r->dit_out_w, r->dit_out_b,
        r->dit_blocks, r->dit_blocks_cpu,
        r->dit_rope_cos, r->dit_rope_sin,
        r->dit_n_freqs, r->dit_axis_dim, 1, cond_hash, cache_slot);

    r->ops.use_f32_gemm = saved_f32;
    r->ops.use_mma_gemm = saved_mma;
    r->ops.use_tf32_gemm = saved_tf32;
    r->ops.use_bf16_gemm = saved_bf16;
    r->ops.bf16_round = saved_round;
}

/* Stage 2 wrapper — uses F16 MMA GEMM if available (weights stored as F16) */
/* Generic sparse DiT forward (used for Stage 2 and Stage 3) */
static void run_sparse_dit_forward(cuda_trellis2_runner *r,
                                     dit_model_gpu *m, dit_block_cpu *blocks_cpu,
                                     CUdeviceptr d_x, float timestep,
                                     CUdeviceptr d_cond, CUdeviceptr d_output,
                                     int N, CUdeviceptr rope_cos, CUdeviceptr rope_sin,
                                     int model_id, uint64_t cond_hash, int cache_slot) {
    /* Save GEMM mode and set precision per the model's loaded weight format.
     * m->use_f16 mirrors the Stage-1 r->dit_use_f16 policy (default F32). */
    int saved_f32 = r->ops.use_f32_gemm;
    int saved_mma = r->ops.use_mma_gemm;
    int saved_tf32 = r->ops.use_tf32_gemm;
    int saved_bf16 = r->ops.use_bf16_gemm;
    int saved_round = r->ops.bf16_round;
    if (m->use_f16) {
        r->ops.use_f32_gemm = 0;
        r->ops.use_mma_gemm = 1;
    } else {
        r->ops.use_f32_gemm = 1;
        r->ops.use_mma_gemm = 0;
        /* T2_SLAT_BF16=1 (default OFF): bf16 DiT-block port for the sparse Stage 2/3
         * flows — identical mechanism to Stage 1's T2_DIT_BF16. PyTorch runs the SLAT
         * flow blocks in bf16, and Stage 2's CFG=7.5 amplifies the per-step f16-vs-bf16
         * difference ~7.5x (hence full-sampler cosine 0.985 vs 07 despite single-step
         * 0.99995). Replicating bf16 should recover it. REQUIRES F32-loaded weights
         * (run with T2_DIT_F32=1): the per-GEMM F32->bf16 cast then recovers the exact
         * original bf16 (bf16->F32 at load is lossless; F16-loaded weights would be
         * doubly lossy and the cast kernel expects F32 input). bf16_round rounds every
         * block-op output; use_bf16_gemm makes block matmuls true bf16; x_emb/out stay
         * f32 (suppressed inside run_dit_forward_generic). Stage 3 has no CFG so it is
         * already 0.99998 and gains little; the lever is Stage 2. */
        const char *want_bf16 = getenv("T2_SLAT_BF16");
        if (want_bf16 && atoi(want_bf16) && r->ops.cublas &&
            r->ops.cast_f32_to_bf16 && r->ops.round_bf16) {
            r->ops.use_bf16_gemm = 1;
            r->ops.bf16_round = 1;
            r->ops.use_tf32_gemm = 0;
        } else {
            /* F32 sparse DiT (only under T2_DIT_F32): TF32 tensor cores, as Stage 1. */
            const char *no_tf32 = getenv("T2_DIT_NO_TF32");
            r->ops.use_tf32_gemm = (r->ops.cublas && !(no_tf32 && atoi(no_tf32))) ? 1 : 0;
        }
    }

    run_dit_forward_generic(r, d_x, timestep, d_cond, d_output,
        N, m->in_channels, m->out_channels, m->n_blocks,
        m->t_fc1_w, m->t_fc1_b, m->t_fc2_w, m->t_fc2_b,
        m->mod_w, m->mod_b,
        m->x_emb_w, m->x_emb_b,
        m->out_w, m->out_b,
        m->blocks, blocks_cpu,
        rope_cos, rope_sin,
        m->n_rope_freqs, m->rope_axis_dim, model_id, cond_hash, cache_slot);

    r->ops.use_f32_gemm = saved_f32;
    r->ops.use_mma_gemm = saved_mma;
    r->ops.use_tf32_gemm = saved_tf32;
    r->ops.use_bf16_gemm = saved_bf16;
    r->ops.bf16_round = saved_round;
}

/* ======================================================================== */
/* Decoder forward                                                          */
/* ======================================================================== */

static void run_resblock(cuda_trellis2_runner *r,
                          CUdeviceptr d_out, CUdeviceptr d_in,
                          const dec_resblock_gpu *rb,
                          int C, int D, int H, int W) {
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;
    int spatial = D * H * W;

    /* Ensure scratch[4,5] for resblock temps */
    ensure_scratch(r, 4, (size_t)C * spatial * sizeof(float));
    ensure_scratch(r, 5, (size_t)C * spatial * sizeof(float));
    CUdeviceptr d_h1 = r->scratch[4];
    CUdeviceptr d_h2 = r->scratch[5];

    /* ChannelLN1 -> SiLU -> Conv1 */
    t2_op_channel_layernorm_3d(ops, s, d_h1, d_in, rb->gn1_w, rb->gn1_b, C, spatial);
    t2_op_silu_inplace(ops, s, d_h1, C * spatial);
    t2_op_conv3d(ops, s, d_h2, d_h1, rb->conv1_w, rb->conv1_b, C, C, D, H, W);

    /* ChannelLN2 -> SiLU -> Conv2 */
    t2_op_channel_layernorm_3d(ops, s, d_h1, d_h2, rb->gn2_w, rb->gn2_b, C, spatial);
    t2_op_silu_inplace(ops, s, d_h1, C * spatial);
    t2_op_conv3d(ops, s, d_out, d_h1, rb->conv2_w, rb->conv2_b, C, C, D, H, W);

    /* Skip (identity) */
    t2_op_add(ops, s, d_out, d_in, C * spatial);
}

static void run_decoder(cuda_trellis2_runner *r,
                         CUdeviceptr d_latent, CUdeviceptr d_output) {
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;
    /* Alloc main decoder buffers */
    ensure_scratch(r, 6, (size_t)1024 * 16 * 16 * 16 * sizeof(float));
    ensure_scratch(r, 7, (size_t)512 * 32 * 32 * 32 * sizeof(float));

    CUdeviceptr d_buf_a = r->scratch[6];
    CUdeviceptr d_buf_b = r->scratch[7];

    /* conv_in: [8, 16^3] -> [512, 16^3] */
    t2_op_conv3d(ops, s, d_buf_a, d_latent, r->dec_conv_in_w, r->dec_conv_in_b,
                 8, 512, 16, 16, 16);

    if (r->verbose >= 2) dbg4("dec_conv_in", d_buf_a, s);

    /* middle + res16 blocks (512 ch, 16^3) */
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_middle[0], 512, 16, 16, 16);
    if (r->verbose >= 2) dbg4("dec_mid0", d_buf_b, s);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_middle[1], 512, 16, 16, 16);
    if (r->verbose >= 2) dbg4("dec_mid1", d_buf_a, s);
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_res16[0], 512, 16, 16, 16);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_res16[1], 512, 16, 16, 16);

    /* Up1: conv 512->1024, pixel_shuffle -> [128, 32^3] */
    t2_op_conv3d(ops, s, d_buf_b, d_buf_a, r->dec_up1_w, r->dec_up1_b,
                 512, 1024, 16, 16, 16);
    t2_op_pixel_shuffle_3d(ops, s, d_buf_a, d_buf_b, 128, 16, 16, 16);

    /* res32 blocks (128 ch, 32^3) */
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_res32[0], 128, 32, 32, 32);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_res32[1], 128, 32, 32, 32);

    /* Up2: conv 128->256, pixel_shuffle -> [32, 64^3] */
    t2_op_conv3d(ops, s, d_buf_b, d_buf_a, r->dec_up2_w, r->dec_up2_b,
                 128, 256, 32, 32, 32);
    /* Need bigger buffer for 64^3 */
    ensure_scratch(r, 6, (size_t)32 * 64 * 64 * 64 * sizeof(float));
    d_buf_a = r->scratch[6];
    t2_op_pixel_shuffle_3d(ops, s, d_buf_a, d_buf_b, 32, 32, 32, 32);

    /* res64 blocks (32 ch, 64^3) */
    ensure_scratch(r, 7, (size_t)32 * 64 * 64 * 64 * sizeof(float));
    d_buf_b = r->scratch[7];
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_res64[0], 32, 64, 64, 64);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_res64[1], 32, 64, 64, 64);

    /* Output: GN -> SiLU -> Conv3d(32->1) */
    /* Output: ChannelLayerNorm -> SiLU -> Conv3d(32->1) */
    t2_op_channel_layernorm_3d(ops, s, d_buf_b, d_buf_a,
                                r->dec_out_gn_w, r->dec_out_gn_b, 32, 64*64*64);
    t2_op_silu_inplace(ops, s, d_buf_b, 32 * 64 * 64 * 64);
    t2_op_conv3d(ops, s, d_output, d_buf_b, r->dec_out_conv_w, r->dec_out_conv_b,
                 32, 1, 64, 64, 64);
}

/* ======================================================================== */
/* Full pipeline                                                            */
/* ======================================================================== */

/* Simple xoshiro256** for public predict() sampling. Matches test harness. */
static uint64_t t2_rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } t2_rng_state;
static uint64_t t2_rng_next(t2_rng_state *r) {
    uint64_t *s = r->s;
    uint64_t result = t2_rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = t2_rotl64(s[3], 45);
    return result;
}
static float t2_rng_randn(t2_rng_state *r) {
    double u1 = ((double)(t2_rng_next(r) >> 11) + 0.5) / (double)(1ULL << 53);
    double u2 = ((double)(t2_rng_next(r) >> 11) + 0.5) / (double)(1ULL << 53);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}
static float t2_rescale_t(float t, float rt) {
    return t * rt / (1.0f + (rt - 1.0f) * t);
}

static float *t2_preprocess_rgb_for_dinov3(const uint8_t *rgb, int w, int h) {
    if (!rgb || w <= 0 || h <= 0) return NULL;
    const int out_w = DINO_IMG_SIZE, out_h = DINO_IMG_SIZE;
    int crop = w < h ? w : h;
    int ox = (w - crop) / 2;
    int oy = (h - crop) / 2;
    float *out = (float *)malloc((size_t)3 * out_w * out_h * sizeof(float));
    if (!out) return NULL;

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int y = 0; y < out_h; y++) {
        float src_y = ((float)y + 0.5f) * (float)crop / (float)out_h - 0.5f;
        int y0 = (int)floorf(src_y);
        float fy = src_y - (float)y0;
        if (y0 < 0) { y0 = 0; fy = 0.0f; }
        if (y0 >= crop - 1) { y0 = crop - 1; fy = 0.0f; }
        int y1 = (y0 + 1 < crop) ? y0 + 1 : y0;
        for (int x = 0; x < out_w; x++) {
            float src_x = ((float)x + 0.5f) * (float)crop / (float)out_w - 0.5f;
            int x0 = (int)floorf(src_x);
            float fx = src_x - (float)x0;
            if (x0 < 0) { x0 = 0; fx = 0.0f; }
            if (x0 >= crop - 1) { x0 = crop - 1; fx = 0.0f; }
            int x1 = (x0 + 1 < crop) ? x0 + 1 : x0;
            const uint8_t *p00 = rgb + ((size_t)(oy + y0) * w + (ox + x0)) * 3;
            const uint8_t *p01 = rgb + ((size_t)(oy + y0) * w + (ox + x1)) * 3;
            const uint8_t *p10 = rgb + ((size_t)(oy + y1) * w + (ox + x0)) * 3;
            const uint8_t *p11 = rgb + ((size_t)(oy + y1) * w + (ox + x1)) * 3;
            for (int ch = 0; ch < 3; ch++) {
                float v0 = (1.0f - fx) * (float)p00[ch] + fx * (float)p01[ch];
                float v1 = (1.0f - fx) * (float)p10[ch] + fx * (float)p11[ch];
                float v = ((1.0f - fy) * v0 + fy * v1) / 255.0f;
                out[(size_t)ch * out_w * out_h + (size_t)y * out_w + x] =
                    (v - mean[ch]) / stdv[ch];
            }
        }
    }
    return out;
}

float *cuda_trellis2_predict(cuda_trellis2_runner *r,
                              const uint8_t *rgb, int w, int h,
                              int n_steps, float cfg_scale,
                              uint32_t seed) {
    if (!r || !rgb || w <= 0 || h <= 0 || n_steps <= 0) return NULL;
    if (!r->dino_patch_w || !r->dit_x_emb_w || !r->dec_conv_in_w) {
        fprintf(stderr, "T2: predict requires DINOv3, Stage 1 DiT, and decoder weights\n");
        return NULL;
    }

    /* Make predict() safe to call repeatedly on a reused runner (e.g. a
     * long-lived server that keeps weights resident). The CUDA context is
     * thread-affine but a threaded server dispatches each request on a
     * different thread, and the cross-attention KV cache still holds the
     * PREVIOUS call's conditioning. Re-bind the context to the calling thread
     * and drop stale KV so the result depends only on (image, seed, steps). */
    cuCtxSetCurrent(r->context);
    cuda_trellis2_invalidate_kv_cache(r);

    const int N = DIT_N_TOKENS;
    const int C = DIT_IN_CH;
    const int cond_tokens = DINO_SEQ_LEN;
    const int cond_dim = DIT_COND_DIM;
    const float rescale = 5.0f;
    const float sigma_min = 1e-5f;
    const float cfg_rescale = 0.7f;

    float *image_f32 = t2_preprocess_rgb_for_dinov3(rgb, w, h);
    float *features = (float *)malloc((size_t)cond_tokens * cond_dim * sizeof(float));
    float *zeros_cond = (float *)calloc((size_t)cond_tokens * cond_dim, sizeof(float));
    float *x = (float *)malloc((size_t)N * C * sizeof(float));
    float *v_cond = (float *)malloc((size_t)N * C * sizeof(float));
    float *v_uncond = (float *)malloc((size_t)N * C * sizeof(float));
    float *occupancy = (float *)malloc((size_t)64 * 64 * 64 * sizeof(float));
    if (!image_f32 || !features || !zeros_cond || !x || !v_cond || !v_uncond || !occupancy) {
        fprintf(stderr, "T2: predict host allocation failed\n");
        free(image_f32); free(features); free(zeros_cond); free(x);
        free(v_cond); free(v_uncond); free(occupancy);
        return NULL;
    }

    if (r->verbose) fprintf(stderr, "T2: predict DINOv3 encode\n");
    if (cuda_trellis2_run_dinov3(r, image_f32, features) != 0) goto fail;
    free(image_f32);
    image_f32 = NULL;

    t2_rng_state rng = {{seed, seed ^ 0x9E3779B97F4A7C15ULL,
                         seed ^ 0x6C62272E07BB0142ULL,
                         seed ^ 0xBF58476D1CE4E5B9ULL}};
    for (int i = 0; i < 8; i++) t2_rng_next(&rng);
    for (int i = 0; i < N * C; i++) x[i] = t2_rng_randn(&rng);

    if (r->verbose)
        fprintf(stderr, "T2: predict Stage 1 flow (%d steps, cfg=%.2f)\n", n_steps, cfg_scale);
    for (int step = 0; step < n_steps; step++) {
        float t_start = 1.0f - (float)step / (float)n_steps;
        float t_end = 1.0f - (float)(step + 1) / (float)n_steps;
        float t_cur = t2_rescale_t(t_start, rescale);
        float t_next = t2_rescale_t(t_end, rescale);
        int apply_cfg = (t_cur >= 0.6f && t_cur <= 1.0f && cfg_scale != 1.0f);

        if (apply_cfg) {
            if (cuda_trellis2_run_dit(r, x, t_cur, features, v_cond) != 0) goto fail;
            if (cuda_trellis2_run_dit(r, x, t_cur, zeros_cond, v_uncond) != 0) goto fail;
            float *pred_v = v_uncond;
            for (int i = 0; i < N * C; i++)
                pred_v[i] = cfg_scale * v_cond[i] + (1.0f - cfg_scale) * v_uncond[i];

            if (cfg_rescale > 0.0f) {
                float tc = sigma_min + (1.0f - sigma_min) * t_cur;
                float one_m_sm = 1.0f - sigma_min;
                double sum_pos = 0.0, sum_cfg = 0.0, sum2_pos = 0.0, sum2_cfg = 0.0;
                for (int i = 0; i < N * C; i++) {
                    float x0p = one_m_sm * x[i] - tc * v_cond[i];
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    sum_pos += x0p; sum2_pos += (double)x0p * x0p;
                    sum_cfg += x0c; sum2_cfg += (double)x0c * x0c;
                }
                double n = (double)(N * C);
                double var_pos = (sum2_pos - sum_pos * sum_pos / n) / (n - 1.0);
                double var_cfg = (sum2_cfg - sum_cfg * sum_cfg / n) / (n - 1.0);
                double std_pos = var_pos > 0.0 ? sqrt(var_pos) : 0.0;
                double std_cfg = var_cfg > 0.0 ? sqrt(var_cfg) : 0.0;
                float ratio = (std_cfg > 1e-8) ? (float)(std_pos / std_cfg) : 1.0f;
                float sc = cfg_rescale * ratio + (1.0f - cfg_rescale);
                for (int i = 0; i < N * C; i++) {
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    pred_v[i] = (one_m_sm * x[i] - sc * x0c) / tc;
                }
            }

            for (int i = 0; i < N * C; i++)
                x[i] -= (t_cur - t_next) * pred_v[i];
        } else {
            if (cuda_trellis2_run_dit(r, x, t_cur, features, v_cond) != 0) goto fail;
            for (int i = 0; i < N * C; i++)
                x[i] -= (t_cur - t_next) * v_cond[i];
        }
        if (r->verbose >= 2)
            fprintf(stderr, "  predict step %d/%d t=%.4f->%.4f %s\n",
                    step + 1, n_steps, t_cur, t_next, apply_cfg ? "CFG" : "noG");
    }

    if (r->verbose) fprintf(stderr, "T2: predict structure decoder\n");
    if (cuda_trellis2_run_decoder(r, x, occupancy) != 0) goto fail;

    free(features); free(zeros_cond); free(x); free(v_cond); free(v_uncond);
    return occupancy;

fail:
    free(image_f32); free(features); free(zeros_cond); free(x);
    free(v_cond); free(v_uncond); free(occupancy);
    return NULL;
}

/* ---- Per-stage APIs ---- */

static uint64_t t2_hash_f32_bytes(const float *data, size_t n) {
    if (!data) return 0;
    const unsigned char *p = (const unsigned char *)data;
    size_t bytes = n * sizeof(float);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t t2_hash_i32_bytes(const int32_t *data, size_t n) {
    if (!data) return 0;
    const unsigned char *p = (const unsigned char *)data;
    size_t bytes = n * sizeof(int32_t);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int t2_all_zero_f32(const float *data, size_t n) {
    if (!data) return 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != 0.0f) return 0;
    }
    return 1;
}

static int t2_cond_cache_slot(const float *cond_features, int ctx_len) {
    if (!cond_features) return 0;
    return t2_all_zero_f32(cond_features, (size_t)ctx_len * DIT_COND_DIM) ? 1 : 0;
}

static int t2_kv_cache_ready(const cuda_trellis2_runner *r, int slot,
                              int model_id, uint64_t cond_hash,
                              int n_blocks) {
    return r && slot >= 0 && slot < 2 &&
           r->ca_kv_cache_valid[slot] &&
           r->ca_kv_cache_model_id[slot] == model_id &&
           r->ca_kv_cache_cond_hash[slot] == cond_hash &&
           r->ca_kv_cache_n_blocks[slot] == n_blocks;
}

static int t2_get_sparse_rope_tables(cuda_trellis2_runner *r, int model_id,
                                      const dit_model_gpu *m,
                                      const int32_t *coords, int N,
                                      CUdeviceptr *out_cos,
                                      CUdeviceptr *out_sin) {
    if (!r || !m || !coords || !out_cos || !out_sin ||
        model_id < 0 || model_id >= 4 || N <= 0) {
        return -1;
    }

    int n_freqs = m->n_rope_freqs;
    uint64_t coord_hash = t2_hash_i32_bytes(coords, (size_t)N * 4);
    if (r->sparse_rope_valid[model_id] &&
        r->sparse_rope_coord_hash[model_id] == coord_hash &&
        r->sparse_rope_N[model_id] == N &&
        r->sparse_rope_n_freqs[model_id] == n_freqs &&
        r->sparse_rope_cos[model_id] &&
        r->sparse_rope_sin[model_id]) {
        *out_cos = r->sparse_rope_cos[model_id];
        *out_sin = r->sparse_rope_sin[model_id];
        return 0;
    }

    t2_sparse_rope_cache_free(r, model_id);

    const float rope_theta = 10000.0f;
    float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
    size_t table_sz = (size_t)N * 3 * n_freqs * sizeof(float);
    float *cos_tab = (float *)malloc(table_sz);
    float *sin_tab = (float *)malloc(table_sz);
    if (!freqs || !cos_tab || !sin_tab) {
        free(freqs); free(cos_tab); free(sin_tab);
        return -1;
    }

    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(rope_theta, (float)j / (float)n_freqs);

    for (int i = 0; i < N; i++) {
        float c[3] = {
            (float)coords[i * 4 + 1],
            (float)coords[i * 4 + 2],
            (float)coords[i * 4 + 3],
        };
        for (int axis = 0; axis < 3; axis++) {
            for (int j = 0; j < n_freqs; j++) {
                float theta = c[axis] * freqs[j];
                size_t idx = (size_t)i * 3 * n_freqs + (size_t)axis * n_freqs + j;
                cos_tab[idx] = cosf(theta);
                sin_tab[idx] = sinf(theta);
            }
        }
    }

    r->sparse_rope_cos[model_id] = cu_upload_raw(cos_tab, table_sz);
    r->sparse_rope_sin[model_id] = cu_upload_raw(sin_tab, table_sz);
    free(freqs); free(cos_tab); free(sin_tab);

    if (!r->sparse_rope_cos[model_id] || !r->sparse_rope_sin[model_id]) {
        t2_sparse_rope_cache_free(r, model_id);
        return -1;
    }

    r->sparse_rope_coord_hash[model_id] = coord_hash;
    r->sparse_rope_N[model_id] = N;
    r->sparse_rope_n_freqs[model_id] = n_freqs;
    r->sparse_rope_valid[model_id] = 1;
    *out_cos = r->sparse_rope_cos[model_id];
    *out_sin = r->sparse_rope_sin[model_id];
    return 0;
}

void cuda_trellis2_invalidate_kv_cache(cuda_trellis2_runner *r) {
    if (!r) return;
    for (int slot = 0; slot < 2; slot++) {
        r->ca_kv_cache_valid[slot] = 0;
        r->ca_kv_cache_model_id[slot] = 0;
        r->ca_kv_cache_cond_hash[slot] = 0;
    }
}

int cuda_trellis2_run_dit(cuda_trellis2_runner *r,
                           const float *x_t, float timestep,
                           const float *cond_features, float *output) {
    CUstream s = r->stream;
    int N = DIT_N_TOKENS, in_ch = DIT_IN_CH;
    int ctx_len = DINO_SEQ_LEN;
    size_t cond_n = (size_t)ctx_len * DIT_COND_DIM;
    uint64_t cond_hash = t2_hash_f32_bytes(cond_features, cond_n);
    int cache_slot = t2_cond_cache_slot(cond_features, ctx_len);

    /* Transpose input from NCDHW [C, spatial] to [spatial, C] on CPU.
     * Official code: x.view(B, C, -1).permute(0, 2, 1) -> [B, N, C] */
    float *x_transposed = (float *)malloc((size_t)N * in_ch * sizeof(float));
    for (int pos = 0; pos < N; pos++)
        for (int ch = 0; ch < in_ch; ch++)
            x_transposed[pos * in_ch + ch] = x_t[ch * N + pos];

    CUdeviceptr d_x = cu_upload_raw(x_transposed, (size_t)N * in_ch * sizeof(float));
    free(x_transposed);
    int have_kv = t2_kv_cache_ready(r, cache_slot, 1, cond_hash, DIT_DEPTH);
    CUdeviceptr d_cond = have_kv ? 0 :
        cu_upload_raw(cond_features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)N * in_ch * sizeof(float));

    run_dit_forward(r, d_x, timestep, d_cond, d_out, cond_hash, cache_slot);

    /* Transpose output from [spatial, C] back to [C, spatial] NCDHW */
    float *out_flat = (float *)malloc((size_t)N * in_ch * sizeof(float));
    cuStreamSynchronize(s);
    cuMemcpyDtoH(out_flat, d_out, (size_t)N * in_ch * sizeof(float));
    for (int pos = 0; pos < N; pos++)
        for (int ch = 0; ch < in_ch; ch++)
            output[ch * N + pos] = out_flat[pos * in_ch + ch];
    free(out_flat);

    cuMemFree(d_x);
    if (d_cond) cuMemFree(d_cond);
    cuMemFree(d_out);
    return 0;
}

int cuda_trellis2_run_decoder(cuda_trellis2_runner *r,
                               const float *latent, float *output) {
    CUstream s = r->stream;
    CUdeviceptr d_lat = cu_upload_raw(latent, (size_t)8 * 16 * 16 * 16 * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)64 * 64 * 64 * sizeof(float));

    run_decoder(r, d_lat, d_out);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)64 * 64 * 64 * sizeof(float));

    cuMemFree(d_lat); cuMemFree(d_out);
    return 0;
}

/* ======================================================================== */
/* Shape decoder forward (SC-VAE, gather-GEMM sparse conv)                  */
/* ======================================================================== */

/* Run one ConvNeXt block on GPU.
 * d_feats: [N, C] in/out features (modified in-place via residual)
 * d_gather_map: [N, 27] precomputed neighbor indices */
static void t2_run_sparse_conv3d(cuda_trellis2_runner *r,
                                  CUdeviceptr dst,
                                  CUdeviceptr feats,
                                  CUdeviceptr weight,
                                  CUdeviceptr bias,
                                  CUdeviceptr gather_map,
                                  CUdeviceptr scratch,
                                  CUdeviceptr scratch2,
                                  const int32_t *coords,
                                  const sp3d_hash *hash,
                                  int N, int in_C, int out_C,
                                  int stage, int block, int op);

static int t2_shape_debug_stop_op(int stage, int block, int op);
static int t2_shape_debug_start_op(int stage, int block);
static int t2_shape_debug_stop_mlp_op(int stage, int block, int op);
static int t2_shape_debug_group_mlp(int stage, int block, int op);
static int t2_shape_debug_cublaslt_mlp(int stage, int block, int op);
static int t2_shape_debug_welford_affine_ln(int stage, int block);
static int t2_shape_debug_welford_affine_c2s(int stage);
static int t2_shape_debug_welford_noaffine_c2s(int stage);
static int t2_shape_debug_sparse_value(const char *name,
                                       int stage, int block, int op,
                                       int *out_value);
static int t2_shape_launch_group_gemm(t2_ops *ops, CUstream s,
                                      CUdeviceptr Y, CUdeviceptr W,
                                      CUdeviceptr X, CUdeviceptr bias,
                                      int n_out, int n_in, int n_tok,
                                      int group);

static int t2_env_flag(const char *name) {
    const char *v = getenv(name);
    return (v && v[0] && atoi(v) != 0) ? 1 : 0;
}

static int run_shape_convnext(cuda_trellis2_runner *r,
                                 CUdeviceptr d_feats, int N, int C,
                                 CUdeviceptr d_gather_map,
                                 const int32_t *coords,
                                 const sp3d_hash *hash,
                                 CUdeviceptr conv_w, CUdeviceptr conv_b,
                                 CUdeviceptr norm_w, CUdeviceptr norm_b,
                                 CUdeviceptr mlp0_w, CUdeviceptr mlp0_b,
                                 CUdeviceptr mlp2_w, CUdeviceptr mlp2_b,
                                 int stage, int block,
                                 CUdeviceptr *debug_feats,
                                 int *debug_C) {
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;

    /* Scratch layout:
     * 8: [N, C] conv output / tmp
     * 9: [N, max(C, 4C)] gathered features / MLP buffer
     * 10: [N, max(C, out_C)] GEMM partial result */
    ensure_scratch(r, 8, (size_t)N * C * sizeof(float));
    ensure_scratch(r, 9, (size_t)N * 4 * C * sizeof(float));
    ensure_scratch(r, 10, (size_t)N * C * sizeof(float));

    CUdeviceptr d_tmp = r->scratch[8];
    CUdeviceptr d_gathered = r->scratch[9];
    CUdeviceptr d_partial = r->scratch[10];
    int start_op = t2_shape_debug_start_op(stage, block);

    /* 1. Sparse conv: feats -> tmp */
    if (start_op <= 0) {
        t2_run_sparse_conv3d(r, d_tmp, d_feats, conv_w, conv_b,
                             d_gather_map, d_gathered, d_partial,
                             coords, hash, N, C, C, stage, block, 0);
        if (t2_shape_debug_stop_op(stage, block, 0)) {
            if (debug_feats) *debug_feats = d_tmp;
            if (debug_C) *debug_C = C;
            return 1;
        }
    } else {
        d_tmp = d_feats;
    }

    /* 2. LayerNorm: tmp -> tmp */
    if (start_op <= 1) {
        if (t2_shape_debug_welford_affine_ln(stage, block)) {
            t2_op_layernorm_welford_eps(ops, s, d_tmp, d_tmp,
                                        norm_w, norm_b, N, C, 1e-6f);
        } else {
            t2_op_layernorm(ops, s, d_tmp, d_tmp, norm_w, norm_b, N, C);
        }
        if (t2_shape_debug_stop_op(stage, block, 1)) {
            if (debug_feats) *debug_feats = d_tmp;
            if (debug_C) *debug_C = C;
            return 1;
        }
    }

    /* 3. MLP: Linear(C->4C) -> SiLU -> Linear(4C->C) */
    CUdeviceptr d_mlp = r->scratch[9];  /* reuse as [N, 4C] */
    int saved_cublas = ops->use_cublas_f32;
    if (t2_env_flag("T2_SCVAE_CUBLAS_MLP") && ops->cublas)
        ops->use_cublas_f32 = 1;
    int lt_mlp0 = t2_shape_debug_cublaslt_mlp(stage, block, 0);
    int group_mlp0 = t2_shape_debug_group_mlp(stage, block, 0);
    if (!(lt_mlp0 &&
          t2_op_gemm_f32_lt_bias(ops, s, d_mlp, mlp0_w, d_tmp, mlp0_b,
                                 4 * C, C, N) == 0) &&
        (!group_mlp0 ||
        t2_shape_launch_group_gemm(ops, s, d_mlp, mlp0_w, d_tmp, mlp0_b,
                                   4 * C, C, N, group_mlp0) != 0)) {
        t2_op_gemm(ops, s, d_mlp, mlp0_w, d_tmp, mlp0_b, 4 * C, C, N);
    }
    if (t2_shape_debug_stop_mlp_op(stage, block, 0)) {
        ops->use_cublas_f32 = saved_cublas;
        if (debug_feats) *debug_feats = d_mlp;
        if (debug_C) *debug_C = 4 * C;
        return 1;
    }
    t2_op_silu_inplace(ops, s, d_mlp, N * 4 * C);
    if (t2_shape_debug_stop_mlp_op(stage, block, 1)) {
        ops->use_cublas_f32 = saved_cublas;
        if (debug_feats) *debug_feats = d_mlp;
        if (debug_C) *debug_C = 4 * C;
        return 1;
    }
    int lt_mlp2 = t2_shape_debug_cublaslt_mlp(stage, block, 2);
    int group_mlp2 = t2_shape_debug_group_mlp(stage, block, 2);
    if (!(lt_mlp2 &&
          t2_op_gemm_f32_lt_bias(ops, s, d_tmp, mlp2_w, d_mlp, mlp2_b,
                                 C, 4 * C, N) == 0) &&
        (!group_mlp2 ||
        t2_shape_launch_group_gemm(ops, s, d_tmp, mlp2_w, d_mlp, mlp2_b,
                                   C, 4 * C, N, group_mlp2) != 0)) {
        t2_op_gemm(ops, s, d_tmp, mlp2_w, d_mlp, mlp2_b, C, 4 * C, N);
    }
    ops->use_cublas_f32 = saved_cublas;
    if (t2_shape_debug_stop_mlp_op(stage, block, 2)) {
        if (debug_feats) *debug_feats = d_tmp;
        if (debug_C) *debug_C = C;
        return 1;
    }
    if (t2_shape_debug_stop_op(stage, block, 2)) {
        if (debug_feats) *debug_feats = d_tmp;
        if (debug_C) *debug_C = C;
        return 1;
    }

    /* 4. Residual: feats += tmp */
    t2_op_residual_add(ops, s, d_feats, d_tmp, N * C);
    return 0;
}

static int t2_make_subdiv_from_logits_host(const float *logits,
                                            const int32_t *coords, int N,
                                            int dense,
                                            int32_t **out_idx,
                                            int32_t **out_subidx,
                                            int32_t **out_coords,
                                            int *out_N) {
    int total = 0;
    if (dense) {
        total = N * 8;
    } else {
        for (int i = 0; i < N * 8; i++)
            if (logits[i] > 0.0f) total++;
    }
    if (total <= 0) {
        *out_idx = NULL; *out_subidx = NULL; *out_coords = NULL; *out_N = 0;
        return -1;
    }
    int32_t *idx = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    int32_t *subidx = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    int32_t *fine_coords = (int32_t *)malloc((size_t)total * 4 * sizeof(int32_t));
    if (!idx || !subidx || !fine_coords) {
        free(idx); free(subidx); free(fine_coords);
        return -1;
    }

    int k = 0;
    for (int i = 0; i < N; i++) {
        int32_t b = coords[i * 4 + 0];
        int32_t z = coords[i * 4 + 1];
        int32_t y = coords[i * 4 + 2];
        int32_t x = coords[i * 4 + 3];
        for (int s = 0; s < 8; s++) {
            if (!dense && logits[i * 8 + s] <= 0.0f) continue;
            /* Match upstream SparseChannel2Spatial: coord dimension i gets
             * bit i of subidx, so coords=(b,z,y,x) maps z=bit0, x=bit2. */
            int dz = s & 1;
            int dy = (s >> 1) & 1;
            int dx = (s >> 2) & 1;
            idx[k] = i;
            subidx[k] = s;
            fine_coords[k * 4 + 0] = b;
            fine_coords[k * 4 + 1] = z * 2 + dz;
            fine_coords[k * 4 + 2] = y * 2 + dy;
            fine_coords[k * 4 + 3] = x * 2 + dx;
            k++;
        }
    }

    *out_idx = idx;
    *out_subidx = subidx;
    *out_coords = fine_coords;
    *out_N = total;
    return 0;
}

static int t2_make_subdiv_from_logits_gpu(cuda_trellis2_runner *r,
                                           CUdeviceptr d_logits,
                                           CUdeviceptr d_coords,
                                           int N,
                                           int dense,
                                           int32_t **out_idx,
                                           int32_t **out_subidx,
                                           int32_t **out_coords,
                                           int *out_N,
                                           CUdeviceptr *out_d_idx,
                                           CUdeviceptr *out_d_subidx,
                                           CUdeviceptr *out_d_coords) {
    if (t2_env_flag("T2_SCVAE_CPU_SUBDIV") ||
        !r->ops.c2s_count_subdiv || !r->ops.c2s_write_subdiv_stable)
        return -1;
    if (!dense && !d_logits) return -1;

    double t0 = t2_now_ms();
    CUdeviceptr d_idx = 0, d_subidx = 0, d_new_coords = 0;
    CUdeviceptr d_counts = 0, d_offsets = 0;
    int *counts = NULL, *offsets = NULL;
    if (cuMemAlloc(&d_counts, (size_t)N * sizeof(int32_t)) != CUDA_SUCCESS) goto fail;
    t2_op_c2s_count_subdiv(&r->ops, r->stream, d_counts, d_logits, N, dense);
    if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS) goto fail;

    counts = (int *)malloc((size_t)N * sizeof(int));
    offsets = (int *)malloc((size_t)N * sizeof(int));
    if (!counts || !offsets) goto fail;
    if (cuMemcpyDtoH(counts, d_counts, (size_t)N * sizeof(int32_t)) != CUDA_SUCCESS)
        goto fail;
    int N_new = 0;
    for (int i = 0; i < N; i++) {
        if (counts[i] < 0 || counts[i] > 8) goto fail;
        offsets[i] = N_new;
        N_new += counts[i];
    }
    if (N_new <= 0 || N_new > N * 8) goto fail;

    if (cuMemAlloc(&d_offsets, (size_t)N * sizeof(int32_t)) != CUDA_SUCCESS) goto fail;
    if (cuMemcpyHtoD(d_offsets, offsets, (size_t)N * sizeof(int32_t)) != CUDA_SUCCESS)
        goto fail;
    if (cuMemAlloc(&d_idx, (size_t)N_new * sizeof(int32_t)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_subidx, (size_t)N_new * sizeof(int32_t)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_new_coords, (size_t)N_new * 4 * sizeof(int32_t)) != CUDA_SUCCESS) goto fail;
    t2_op_c2s_write_subdiv_stable(&r->ops, r->stream, d_idx, d_subidx,
                                  d_new_coords, d_offsets, d_logits, d_coords,
                                  N, dense);
    if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS) goto fail;

    int32_t *idx = (int32_t *)malloc((size_t)N_new * sizeof(int32_t));
    int32_t *subidx = (int32_t *)malloc((size_t)N_new * sizeof(int32_t));
    int32_t *new_coords = (int32_t *)malloc((size_t)N_new * 4 * sizeof(int32_t));
    if (!idx || !subidx || !new_coords) {
        free(idx); free(subidx); free(new_coords);
        goto fail;
    }
    if (cuMemcpyDtoH(idx, d_idx, (size_t)N_new * sizeof(int32_t)) != CUDA_SUCCESS ||
        cuMemcpyDtoH(subidx, d_subidx, (size_t)N_new * sizeof(int32_t)) != CUDA_SUCCESS ||
        cuMemcpyDtoH(new_coords, d_new_coords, (size_t)N_new * 4 * sizeof(int32_t)) != CUDA_SUCCESS) {
        free(idx); free(subidx); free(new_coords);
        goto fail;
    }

    *out_idx = idx;
    *out_subidx = subidx;
    *out_coords = new_coords;
    *out_N = N_new;
    *out_d_idx = d_idx;
    *out_d_subidx = d_subidx;
    *out_d_coords = d_new_coords;
    cuMemFree(d_counts);
    cuMemFree(d_offsets);
    free(counts);
    free(offsets);
    t2_timing_log("c2s_subdiv_gpu", t0);
    return 0;

fail:
    if (d_counts) cuMemFree(d_counts);
    if (d_offsets) cuMemFree(d_offsets);
    if (d_idx) cuMemFree(d_idx);
    if (d_subidx) cuMemFree(d_subidx);
    if (d_new_coords) cuMemFree(d_new_coords);
    free(counts);
    free(offsets);
    return -1;
}

static void t2_free_sparse_gpu_index(sp3d_hash *hash,
                                      CUdeviceptr d_hash_keys,
                                      CUdeviceptr d_hash_vals,
                                      CUdeviceptr d_gather_map) {
    if (d_gather_map) cuMemFree(d_gather_map);
    if (d_hash_keys) cuMemFree(d_hash_keys);
    if (d_hash_vals) cuMemFree(d_hash_vals);
    if (hash) sp3d_hash_free(hash);
}

static void t2_sparse_conv_pack_free(t2_sparse_conv_pack *pack) {
    if (!pack) return;
    if (pack->src_storage) {
        cuMemFree(pack->src_storage);
    } else {
        for (int k = 0; k < 27; k++)
            if (pack->src_idx[k]) cuMemFree(pack->src_idx[k]);
    }
    if (pack->dst_storage) {
        cuMemFree(pack->dst_storage);
    } else {
        for (int k = 0; k < 27; k++)
            if (pack->dst_idx[k]) cuMemFree(pack->dst_idx[k]);
    }
    memset(pack, 0, sizeof(*pack));
}

static int t2_sparse_conv_pack_build(const int32_t *coords, int N,
                                      const sp3d_hash *hash,
                                      t2_sparse_conv_pack *pack) {
    memset(pack, 0, sizeof(*pack));
    int *src = (int *)malloc((size_t)27 * N * sizeof(int));
    int *dst = (int *)malloc((size_t)27 * N * sizeof(int));
    if (!src || !dst) {
        free(src);
        free(dst);
        return -1;
    }

    for (int i = 0; i < N; i++) {
        int32_t b = coords[i * 4 + 0];
        int32_t z = coords[i * 4 + 1];
        int32_t y = coords[i * 4 + 2];
        int32_t x = coords[i * 4 + 3];
        for (int kd = 0; kd < 3; kd++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int k = kd * 9 + kh * 3 + kw;
                    int ni = sp3d_hash_lookup(hash, b, z + kd - 1, y + kh - 1, x + kw - 1);
                    if (ni < 0) continue;
                    int m = pack->M[k]++;
                    src[k * N + m] = ni;
                    dst[k * N + m] = i;
                }
            }
        }
    }

    for (int k = 0; k < 27; k++) {
        int M = pack->M[k];
        if (M <= 0) continue;
        pack->src_idx[k] = cu_upload_raw(src + k * N, (size_t)M * sizeof(int));
        pack->dst_idx[k] = cu_upload_raw(dst + k * N, (size_t)M * sizeof(int));
        if (!pack->src_idx[k] || !pack->dst_idx[k]) {
            free(src);
            free(dst);
            t2_sparse_conv_pack_free(pack);
            return -1;
        }
    }

    free(src);
    free(dst);
    return 0;
}

static int t2_sparse_conv_pack_build_gpu(cuda_trellis2_runner *r,
                                          CUdeviceptr gather_map, int N,
                                          t2_sparse_conv_pack *pack) {
    memset(pack, 0, sizeof(*pack));
    if (!gather_map || !r->ops.sparse_pack_from_gather_map) return -1;

    double t0 = t2_now_ms();
    CUdeviceptr d_src = 0, d_dst = 0, d_counts = 0;
    size_t idx_bytes = (size_t)27 * (size_t)N * sizeof(int);
    if (cuMemAlloc(&d_src, idx_bytes) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_dst, idx_bytes) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_counts, 27 * sizeof(int)) != CUDA_SUCCESS) goto fail;
    if (cuMemsetD8Async(d_counts, 0, 27 * sizeof(int), r->stream) != CUDA_SUCCESS)
        goto fail;

    t2_op_sparse_pack_from_gather_map(&r->ops, r->stream, d_src, d_dst,
                                      d_counts, gather_map, N);
    if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS) goto fail;

    int counts[27];
    if (cuMemcpyDtoH(counts, d_counts, sizeof(counts)) != CUDA_SUCCESS) goto fail;
    cuMemFree(d_counts);
    d_counts = 0;

    pack->src_storage = d_src;
    pack->dst_storage = d_dst;
    for (int k = 0; k < 27; k++) {
        if (counts[k] < 0 || counts[k] > N) goto fail;
        pack->M[k] = counts[k];
        if (counts[k] > 0) {
            pack->src_idx[k] = d_src + (size_t)k * (size_t)N * sizeof(int);
            pack->dst_idx[k] = d_dst + (size_t)k * (size_t)N * sizeof(int);
        }
    }
    t2_timing_log("sparse_pack_gpu", t0);
    return 0;

fail:
    if (d_counts) cuMemFree(d_counts);
    if (d_src) cuMemFree(d_src);
    if (d_dst) cuMemFree(d_dst);
    memset(pack, 0, sizeof(*pack));
    return -1;
}

static void t2_run_sparse_conv3d(cuda_trellis2_runner *r,
                                  CUdeviceptr dst,
                                  CUdeviceptr feats,
                                  CUdeviceptr weight,
                                  CUdeviceptr bias,
                                  CUdeviceptr gather_map,
                                  CUdeviceptr scratch,
                                  CUdeviceptr scratch2,
                                  const int32_t *coords,
                                  const sp3d_hash *hash,
                                  int N, int in_C, int out_C,
                                  int stage, int block, int op) {
    t2_ops *ops = &r->ops;
    int v = 0;
    const char *direct_env = getenv("T2_SCVAE_DIRECT_CONV");
    int direct_mode = (direct_env && direct_env[0]) ? atoi(direct_env) : 0;
    if (t2_shape_debug_sparse_value("T2SD_SPARSE_DIRECT",
                                    stage, block, op, &v)) {
        direct_mode = v;
    }
    if (ops->use_f32_gemm && direct_mode) {
        t2_op_sparse_conv3d_direct(ops, r->stream, dst, feats, weight, bias,
                                   gather_map, N, in_C, out_C, direct_mode);
        return;
    }
    int use_packed = ops->use_packed_sparse_conv;
    if (t2_shape_debug_sparse_value("T2SD_SPARSE_PACKED",
                                    stage, block, op, &v)) {
        use_packed = v ? 1 : 0;
    }
    int use_lt = t2_shape_debug_sparse_value("T2SD_SPARSE_LT",
                                             stage, block, op, NULL);
    if (ops->use_f32_gemm && use_packed && (gather_map || hash) &&
        ops->sparse_pack_rows && ops->scatter_add_rows) {
        /* Reuse the level-cached pack if it matches (coords ptr + N). Every
         * ConvNeXt block at a resolution level — and the c2s conv1 — share the
         * same pack; only a new level (c2s conv2 onward) rebuilds it. This turns
         * the host-side build (N*27 CPU hash lookups + up to 54 HtoD) from
         * once-per-conv into once-per-level. The default builder now derives
         * the same row lists from the already-built GPU gather map, avoiding
         * the remaining per-level CPU hash walk; T2_SCVAE_CPU_PACK_BUILD=1
         * forces the legacy host builder for A/B checks. */
        if (!(r->pack_cache_valid && r->pack_cache_coords == coords &&
              r->pack_cache_N == N)) {
            if (r->pack_cache_valid) {
                t2_sparse_conv_pack_free(&r->pack_cache);
                r->pack_cache_valid = 0;
            }
            int pack_ok = -1;
            if (!t2_env_flag("T2_SCVAE_CPU_PACK_BUILD"))
                pack_ok = t2_sparse_conv_pack_build_gpu(r, gather_map, N,
                                                        &r->pack_cache);
            if (pack_ok != 0 && hash)
                pack_ok = t2_sparse_conv_pack_build(coords, N, hash,
                                                    &r->pack_cache);
            if (pack_ok == 0) {
                r->pack_cache_coords = coords;
                r->pack_cache_N = N;
                r->pack_cache_valid = 1;
            }
        }
        if (r->pack_cache_valid) {
            t2_op_sparse_conv3d_packed(ops, r->stream, dst, feats, weight, bias,
                                       &r->pack_cache, scratch, scratch2,
                                       in_C, out_C, N, use_lt);
            return;
        }
        fprintf(stderr, "T2: packed sparse conv setup failed; using gather-GEMM\n");
    }

    int saved_cublas = ops->use_cublas_f32;
    if (ops->use_f32_gemm &&
        ((t2_env_flag("T2_SCVAE_CUBLAS_SPARSE") && ops->cublas) ||
         t2_shape_debug_sparse_value("T2SD_SPARSE_CUBLAS",
                                     stage, block, op, NULL)))
        ops->use_cublas_f32 = 1;
    t2_op_sparse_conv3d(ops, r->stream, dst, feats, weight, bias,
                        gather_map, scratch, scratch2, N, in_C, out_C, use_lt);
    ops->use_cublas_f32 = saved_cublas;
}

static int t2_build_sparse_gpu_index(cuda_trellis2_runner *r,
                                      const int32_t *coords, int N,
                                      CUdeviceptr d_coords,
                                      sp3d_hash **out_hash,
                                      CUdeviceptr *out_hash_keys,
                                      CUdeviceptr *out_hash_vals,
                                      CUdeviceptr *out_gather_map,
                                      int *out_hash_cap) {
    int hash_cap = 16;
    while (hash_cap < 2 * N) hash_cap *= 2;

    if (!t2_env_flag("T2_SCVAE_CPU_HASH_BUILD") &&
        !t2_env_flag("T2_SCVAE_CPU_GATHER_MAP") &&
        !t2_env_flag("T2_SCVAE_CPU_PACK_BUILD") &&
        r->ops.sparse_hash_insert_coords) {
        double t0 = t2_now_ms();
        CUdeviceptr d_hash_keys = 0, d_hash_vals = 0, d_gather_map = 0;
        if (cuMemAlloc(&d_hash_keys, (size_t)hash_cap * sizeof(uint64_t)) != CUDA_SUCCESS)
            goto gpu_fail;
        if (cuMemAlloc(&d_hash_vals, (size_t)hash_cap * sizeof(int32_t)) != CUDA_SUCCESS)
            goto gpu_fail;
        if (cuMemAlloc(&d_gather_map, (size_t)N * 27 * sizeof(int32_t)) != CUDA_SUCCESS)
            goto gpu_fail;
        cuMemsetD8Async(d_hash_keys, 0xff, (size_t)hash_cap * sizeof(uint64_t), r->stream);
        cuMemsetD8Async(d_hash_vals, 0xff, (size_t)hash_cap * sizeof(int32_t), r->stream);
        t2_op_sparse_hash_insert_coords(&r->ops, r->stream, d_hash_keys,
                                        d_hash_vals, d_coords, N, hash_cap);
        t2_op_sparse_build_gather_map(&r->ops, r->stream, d_gather_map, d_coords, N,
                                      d_hash_keys, d_hash_vals, hash_cap);
        if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS)
            goto gpu_fail;
        *out_hash = NULL;
        *out_hash_keys = d_hash_keys;
        *out_hash_vals = d_hash_vals;
        *out_gather_map = d_gather_map;
        if (out_hash_cap) *out_hash_cap = hash_cap;
        t2_timing_log("sparse_index_gpu", t0);
        return 0;

gpu_fail:
        if (d_gather_map) cuMemFree(d_gather_map);
        if (d_hash_keys) cuMemFree(d_hash_keys);
        if (d_hash_vals) cuMemFree(d_hash_vals);
        fprintf(stderr, "T2: GPU sparse index build failed; falling back to CPU hash build\n");
    }

    double t0 = t2_now_ms();
    sp3d_hash *hash = sp3d_hash_build(coords, N);
    if (!hash) return -1;
    CUdeviceptr d_hash_keys = cu_upload_raw(hash->keys, (size_t)hash->capacity * sizeof(uint64_t));
    CUdeviceptr d_hash_vals = cu_upload_raw(hash->vals, (size_t)hash->capacity * sizeof(int32_t));
    CUdeviceptr d_gather_map = 0;
    if (t2_env_flag("T2_SCVAE_CPU_GATHER_MAP")) {
        int32_t *gm = (int32_t *)malloc((size_t)N * 27 * sizeof(int32_t));
        if (!gm) {
            sp3d_hash_free(hash);
            return -1;
        }
        for (int i = 0; i < N; i++) {
            int32_t b = coords[i * 4 + 0];
            int32_t z = coords[i * 4 + 1];
            int32_t y = coords[i * 4 + 2];
            int32_t x = coords[i * 4 + 3];
            for (int kd = 0; kd < 3; kd++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        gm[i * 27 + kd * 9 + kh * 3 + kw] =
                            sp3d_hash_lookup(hash, b, z + kd - 1, y + kh - 1, x + kw - 1);
        }
        d_gather_map = cu_upload_raw(gm, (size_t)N * 27 * sizeof(int32_t));
        free(gm);
    } else {
        cuMemAlloc(&d_gather_map, (size_t)N * 27 * sizeof(int32_t));
        t2_op_sparse_build_gather_map(&r->ops, r->stream, d_gather_map, d_coords, N,
                                      d_hash_keys, d_hash_vals, hash->capacity);
    }
    *out_hash = hash;
    *out_hash_keys = d_hash_keys;
    *out_hash_vals = d_hash_vals;
    *out_gather_map = d_gather_map;
    if (out_hash_cap) *out_hash_cap = hash->capacity;
    t2_timing_log("sparse_index_cpu", t0);
    return 0;
}

static int t2_shape_debug_return(CUstream s, CUdeviceptr d_feats,
                                  const int32_t *coords, int N, int C,
                                  float **out_feats,
                                  int32_t **out_coords,
                                  int *out_N, int *out_C) {
    float *hf = (float *)malloc((size_t)N * C * sizeof(float));
    int32_t *hc = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    if (!hf || !hc) {
        free(hf); free(hc);
        return -1;
    }
    cuStreamSynchronize(s);
    cuMemcpyDtoH(hf, d_feats, (size_t)N * C * sizeof(float));
    memcpy(hc, coords, (size_t)N * 4 * sizeof(int32_t));
    *out_feats = hf;
    *out_coords = hc;
    if (out_N) *out_N = N;
    if (out_C) *out_C = C;
    return 0;
}

static int t2_shape_debug_stop_stage_block(int stage, int block) {
    const char *sb = getenv("T2SD_STOP_AFTER_STAGE_BLOCK");
    if (!sb || !sb[0]) return 0;
    int ws = -1, wb = -1;
    sscanf(sb, "%d:%d", &ws, &wb);
    return ws == stage && wb == block;
}

static int t2_shape_debug_stop_op(int stage, int block, int op) {
    const char *env = getenv("T2SD_STOP_AFTER_OP");
    if (!env || !env[0]) return 0;
    int ws = -1, wb = -1, wo = -1;
    sscanf(env, "%d:%d:%d", &ws, &wb, &wo);
    return ws == stage && wb == block && wo == op;
}

static int t2_shape_debug_start_op(int stage, int block) {
    const char *env = getenv("T2SD_START_AT_OP");
    if (!env || !env[0]) return 0;
    int ws = -1, wb = -1, wo = 0;
    sscanf(env, "%d:%d:%d", &ws, &wb, &wo);
    return (ws == stage && wb == block) ? wo : 0;
}

static int t2_shape_debug_stop_mlp_op(int stage, int block, int op) {
    const char *env = getenv("T2SD_STOP_AFTER_MLP_OP");
    if (!env || !env[0]) return 0;
    int ws = -1, wb = -1, wo = -1;
    sscanf(env, "%d:%d:%d", &ws, &wb, &wo);
    return ws == stage && wb == block && wo == op;
}

static int t2_shape_debug_group_mlp(int stage, int block, int op) {
    const char *env = getenv("T2SD_GROUP_MLP");
    if (!env || !env[0]) return 0;
    int ws = -1, wb = -1, wo = -1, group = 0;
    sscanf(env, "%d:%d:%d:%d", &ws, &wb, &wo, &group);
    return (ws == stage && (wb == block || wb < 0) &&
            (wo == op || wo < 0) && group > 0)
        ? group : 0;
}

static int t2_shape_debug_cublaslt_mlp(int stage, int block, int op) {
    const char *env = getenv("T2SD_CUBLASLT_MLP");
    if (!env || !env[0]) return 0;
    const char *p = env;
    while (*p) {
        int ws = -1, wb = -1, wo = -1;
        if (sscanf(p, "%d:%d:%d", &ws, &wb, &wo) == 3 &&
            ws == stage && (wb == block || wb < 0) &&
            (wo == op || wo < 0)) {
            return 1;
        }
        p = strpbrk(p, ",;");
        if (!p) break;
        p++;
    }
    return 0;
}

static int t2_shape_debug_welford_affine_ln(int stage, int block) {
    const char *env = getenv("T2SD_WELFORD_AFFINE_LN");
    if (!env || !env[0]) return 0;
    int ws = -1, wb = -1;
    sscanf(env, "%d:%d", &ws, &wb);
    return (ws == stage && (wb == block || wb < 0));
}

static int t2_shape_debug_welford_affine_c2s(int stage) {
    const char *env = getenv("T2SD_WELFORD_AFFINE_C2S");
    if (!env || !env[0]) return 0;
    int ws = -1;
    sscanf(env, "%d", &ws);
    return ws == stage || ws < 0;
}

static int t2_shape_debug_welford_noaffine_c2s(int stage) {
    const char *env = getenv("T2SD_WELFORD_NOAFFINE_C2S");
    if (!env || !env[0]) return 0;
    int ws = -1;
    sscanf(env, "%d", &ws);
    return ws == stage || ws < 0;
}

static int t2_shape_debug_sparse_value(const char *name,
                                       int stage, int block, int op,
                                       int *out_value) {
    const char *env = getenv(name);
    if (!env || !env[0]) return 0;
    int ws = -1, wb = -1, wo = -1, value = 1;
    int n = sscanf(env, "%d:%d:%d:%d", &ws, &wb, &wo, &value);
    if (n < 3) return 0;
    if (ws != stage && ws >= 0) return 0;
    if (wb != block && wb >= 0) return 0;
    if (wo != op && wo >= 0) return 0;
    if (out_value) *out_value = (n >= 4) ? value : 1;
    return 1;
}

static int t2_shape_launch_group_gemm(t2_ops *ops, CUstream s,
                                      CUdeviceptr Y, CUdeviceptr W,
                                      CUdeviceptr X, CUdeviceptr bias,
                                      int n_out, int n_in, int n_tok,
                                      int group) {
    if (!ops->use_f32_gemm || !ops->gemm_f32_group || group <= 0) return -1;
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &group};
    cuLaunchKernel(ops->gemm_f32_group,
                   (unsigned)(((size_t)n_out * n_tok + 255) / 256),
                   1, 1, 256, 1, 1, 0, s, args, NULL);
    return 0;
}

static int t2_shape_debug_stop_c2s_op(int stage, int op) {
    const char *env = getenv("T2SD_STOP_AFTER_C2S_OP");
    if (!env || !env[0]) return 0;
    int ws = -1, wo = -1;
    sscanf(env, "%d:%d", &ws, &wo);
    return ws == stage && wo == op;
}

/* Run shape decoder forward pass on CUDA.
 * slat: [N, 32] structured latent on CPU
 * coords: [N, 4] int32 on CPU
 * out_feats/out_coords are malloc-owned CPU outputs. */
static int cuda_trellis2_run_scvae_decoder_alloc(cuda_trellis2_runner *r,
                                                  t2_scvae_dec_gpu *dec,
                                                  const char *label,
                                                  const float *input_feats,
                                                  int input_C,
                                                  const int32_t *coords, int N,
                                                  int start_stage,
                                                  int start_block,
                                                  float **out_feats,
                                                  int32_t **out_coords,
                                                  int *out_N,
                                                  int *out_C) {
    if (!dec->loaded) {
        fprintf(stderr, "T2: %s not loaded\n", label); return -1;
    }
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;
    *out_feats = NULL;
    *out_coords = NULL;
    if (out_N) *out_N = 0;
    if (out_C) *out_C = 0;

    /* Invalidate any packed-conv pack cached from a prior decode (its coords
     * pointer could alias a new allocation and cause a false hit). */
    if (r->pack_cache_valid) {
        t2_sparse_conv_pack_free(&r->pack_cache);
        r->pack_cache_valid = 0;
    }
    r->pack_cache_coords = NULL;
    r->pack_cache_N = 0;

    /* Match GEMM mode to the loaded shape decoder weight precision. */
    int saved_f32 = ops->use_f32_gemm;
    int saved_mma = ops->use_mma_gemm;
    int saved_cublas = ops->use_cublas_f32;
    int saved_cublas_pedantic = ops->use_cublas_pedantic;
    int saved_packed_conv = ops->use_packed_sparse_conv;
    if (dec->use_f16 && r->ops.sm_version >= 70) {
        ops->use_f32_gemm = 0;
        ops->use_mma_gemm = 1;
        ops->use_cublas_f32 = 0;
        ops->use_packed_sparse_conv = 0;
    } else {
        ops->use_f32_gemm = 1;
        ops->use_mma_gemm = 0;
        const char *use_cb = getenv("T2_SCVAE_CUBLAS");
        ops->use_cublas_f32 = (use_cb && atoi(use_cb) && ops->cublas) ? 1 : 0;
        const char *ped = getenv("T2_SCVAE_CUBLAS_PEDANTIC");
        ops->use_cublas_pedantic = (ped && atoi(ped)) ? 1 : 0;
        const char *packed_conv = getenv("T2_SCVAE_PACKED_CONV");
        const char *no_packed_conv = getenv("T2_SCVAE_NO_PACKED_CONV");
        ops->use_packed_sparse_conv = packed_conv
            ? (atoi(packed_conv) ? 1 : 0)
            : ((no_packed_conv && atoi(no_packed_conv)) ? 0 : 1);
        if (ops->use_cublas_f32) {
            fprintf(stderr, "T2: %s: using cuBLAS F32 GEMM%s\n", label,
                    ops->use_cublas_pedantic ? " (pedantic)" : "");
        }
        if (ops->use_packed_sparse_conv) {
            fprintf(stderr, "T2: %s: using packed-row F32 sparse conv\n", label);
        }
    }
    #define T2_RESTORE_SCVAE_GEMM() do { \
        ops->use_f32_gemm = saved_f32; \
        ops->use_mma_gemm = saved_mma; \
        ops->use_cublas_f32 = saved_cublas; \
        ops->use_cublas_pedantic = saved_cublas_pedantic; \
        ops->use_packed_sparse_conv = saved_packed_conv; \
    } while (0)

    if (start_stage >= 0) {
        if (start_stage > 4) {
            fprintf(stderr, "T2: %s: invalid start_stage=%d\n", label, start_stage);
            T2_RESTORE_SCVAE_GEMM();
            return -1;
        }
        int expect_C = dec->channels[start_stage];
        if (input_C != expect_C) {
            fprintf(stderr, "T2: %s: start_stage=%d expects C=%d, got C=%d\n",
                    label, start_stage, expect_C, input_C);
            T2_RESTORE_SCVAE_GEMM();
            return -1;
        }
        if (start_stage < 4) {
            int nblk = dec->n_convnext[start_stage];
            if (start_block < 0 || start_block > nblk) {
                fprintf(stderr, "T2: %s: invalid start_block=%d for stage %d (nblk=%d)\n",
                        label, start_block, start_stage, nblk);
                T2_RESTORE_SCVAE_GEMM();
                return -1;
            }
        }
    }

    int C = (start_stage >= 0) ? input_C : dec->channels[0];  /* 1024 for full path */

    /* Upload input */
    CUdeviceptr d_coords = cu_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    CUdeviceptr d_feats = 0;
    if (start_stage >= 0) {
        d_feats = cu_upload_raw(input_feats, (size_t)N * C * sizeof(float));
        fprintf(stderr, "T2: %s: start from stage=%d block=%d -> [%d, %d]\n",
                label, start_stage, start_block, N, C);
    } else {
        CUdeviceptr d_slat = cu_upload_raw(input_feats, (size_t)N * 32 * sizeof(float));
        /* from_latent: [N, 32] -> [N, 1024] */
        cuMemAlloc(&d_feats, (size_t)N * C * sizeof(float));
        t2_op_gemm(ops, s, d_feats, dec->from_latent_w, d_slat, dec->from_latent_b,
                   C, 32, N);
        cuMemFree(d_slat);

        fprintf(stderr, "T2: %s: from_latent -> [%d, %d]\n", label, N, C);
    }

    sp3d_hash *hash = NULL;
    CUdeviceptr d_hash_keys = 0, d_hash_vals = 0, d_gather_map = 0;
    int hash_cap = 0;

    if (start_stage < 0) {
        const char *flag = getenv("T2SD_STOP_AFTER_LATENT");
        if (flag && atoi(flag)) {
            int rc = t2_shape_debug_return(s, d_feats, coords, N, C,
                                           out_feats, out_coords, out_N, out_C);
            cuMemFree(d_feats);
            cuMemFree(d_coords);
            T2_RESTORE_SCVAE_GEMM();
            return rc;
        }
    }

    /* Current CPU-side coords for subdivision-list synthesis. */
    int32_t *cur_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    memcpy(cur_coords, coords, (size_t)N * 4 * sizeof(int32_t));

    if (start_stage < 4) {
        if (t2_build_sparse_gpu_index(r, cur_coords, N, d_coords,
                                      &hash, &d_hash_keys, &d_hash_vals,
                                      &d_gather_map, &hash_cap) != 0) {
            cuMemFree(d_feats); cuMemFree(d_coords); free(cur_coords);
            T2_RESTORE_SCVAE_GEMM();
            return -1;
        }

        fprintf(stderr, "T2: %s: gather map built (N=%d, hash_cap=%d)\n",
                label, N, hash_cap);
    }

    {
        const char *flag = getenv("T2SD_STOP_AT_START");
        if (flag && atoi(flag)) {
            int rc = t2_shape_debug_return(s, d_feats, cur_coords, N, C,
                                           out_feats, out_coords, out_N, out_C);
            t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
            cuMemFree(d_feats);
            cuMemFree(d_coords);
            free(cur_coords);
            T2_RESTORE_SCVAE_GEMM();
            return rc;
        }
    }

    /* The shape decoder has per-stage to_subdiv heads (subdiv_w); the texture
     * decoder has none. A "recording" (shape) decode stores its per-stage
     * subdivision; a "replaying" (texture) decode reuses it (PyTorch guide_subs).
     * Reset any stale plan at the start of a full recording decode. */
    int dec_is_recording = 0;
    for (int st = 0; st < 4; st++)
        if (dec->c2s[st].conv1_w && dec->c2s[st].subdiv_w) dec_is_recording = 1;
    if (dec_is_recording && start_stage < 0)
        t2_subdiv_plan_free(r);

    /* Run ConvNeXt blocks + C2S for each stage */
    int first_stage = (start_stage >= 0) ? start_stage : 0;
    for (int stage = first_stage; stage < 4; stage++) {
        int nblk = dec->n_convnext[stage];
        C = dec->channels[stage];
        fprintf(stderr, "T2: %s: stage %d: %d ConvNeXt(%d), N=%d\n",
                label, stage, nblk, C, N);

        struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);

        int first_block = (stage == first_stage && start_stage >= 0) ? start_block : 0;
        for (int b = first_block; b < nblk; b++) {
            CUdeviceptr d_debug_feats = 0;
            int debug_C = C;
            int stop_op = run_shape_convnext(r, d_feats, N, C,
                d_gather_map,
                cur_coords,
                hash,
                dec->convnext[stage][b].conv_w,
                dec->convnext[stage][b].conv_b,
                dec->convnext[stage][b].norm_w,
                dec->convnext[stage][b].norm_b,
                dec->convnext[stage][b].mlp0_w,
                dec->convnext[stage][b].mlp0_b,
                dec->convnext[stage][b].mlp2_w,
                dec->convnext[stage][b].mlp2_b,
                stage, b, &d_debug_feats, &debug_C);
            if (stop_op) {
                int rc = t2_shape_debug_return(s, d_debug_feats, cur_coords, N, debug_C,
                                               out_feats, out_coords, out_N, out_C);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                cuMemFree(d_feats);
                cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
            if (t2_shape_debug_stop_stage_block(stage, b)) {
                int rc = t2_shape_debug_return(s, d_feats, cur_coords, N, C,
                                               out_feats, out_coords, out_N, out_C);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                cuMemFree(d_feats);
                cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
        }

        cuStreamSynchronize(s);
        struct timespec ts1; clock_gettime(CLOCK_MONOTONIC, &ts1);
        double dt_conv = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
        fprintf(stderr, "T2: %s: stage %d convnext %.1f ms\n", label, stage, dt_conv);

        /* C2S upsampling. Default path compacts subdivision decisions on GPU
         * and keeps the new coord table device-resident for the next sparse
         * index build; T2_SCVAE_CPU_SUBDIV=1 restores the legacy host path. */
        if (dec->c2s[stage].conv1_w) {
            int C_in = C;
            int C_out = dec->channels[stage + 1];
            int C_exp = C_out * 8;
            int C_in8 = C_in / 8;

            float *sub_logits = NULL;
            CUdeviceptr d_logits = 0;
            CUdeviceptr d_sub_idx = 0, d_sub_subidx = 0, d_new_coords = 0;
            int dense_subdiv = dec->c2s[stage].subdiv_w ? 0 : 1;
            if (!dense_subdiv) {
                cuMemAlloc(&d_logits, (size_t)N * 8 * sizeof(float));
                t2_op_gemm(ops, s, d_logits,
                           dec->c2s[stage].subdiv_w,
                           d_feats,
                           dec->c2s[stage].subdiv_b,
                           8, C_in, N);
                if (t2_shape_debug_stop_c2s_op(stage, 7)) {
                    int rc = t2_shape_debug_return(s, d_logits, cur_coords, N, 8,
                                                   out_feats, out_coords, out_N, out_C);
                    cuMemFree(d_logits);
                    t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                    cuMemFree(d_feats);
                    cuMemFree(d_coords);
                    free(cur_coords);
                    T2_RESTORE_SCVAE_GEMM();
                    return rc;
                }
            }

            int32_t *sub_idx = NULL, *sub_subidx = NULL, *new_coords = NULL;
            int N_new = 0;

            /* Texture decode (no subdiv head): replay the shape decoder's recorded
             * subdivision for this stage instead of densely subdividing x8 (which
             * explodes to ~17M voxels and OOMs). Both decoders are driven from the
             * same res-32 coords in the same order, so the parent count must match;
             * if it does not (e.g. a standalone texture decode with no prior shape
             * decode), fall through to the dense path. */
            int replayed = 0;
            if (dense_subdiv && stage < 8 && r->subdiv_plan[stage].valid &&
                r->subdiv_plan[stage].n_parent == N) {
                t2_subdiv_stage *p = &r->subdiv_plan[stage];
                N_new = p->n_new;
                sub_idx    = (int32_t *)malloc((size_t)N_new * sizeof(int32_t));
                sub_subidx = (int32_t *)malloc((size_t)N_new * sizeof(int32_t));
                new_coords = (int32_t *)malloc((size_t)N_new * 4 * sizeof(int32_t));
                if (sub_idx && sub_subidx && new_coords) {
                    memcpy(sub_idx,    p->idx,    (size_t)N_new * sizeof(int32_t));
                    memcpy(sub_subidx, p->subidx, (size_t)N_new * sizeof(int32_t));
                    memcpy(new_coords, p->coords, (size_t)N_new * 4 * sizeof(int32_t));
                    replayed = 1;
                    fprintf(stderr, "T2: %s: stage %d replaying shape subdivision "
                            "(%d -> %d voxels)\n", label, stage, N, N_new);
                } else {
                    free(sub_idx); free(sub_subidx); free(new_coords);
                    sub_idx = sub_subidx = new_coords = NULL; N_new = 0;
                }
            } else if (dense_subdiv && stage < 8 && !r->subdiv_plan[stage].valid) {
                fprintf(stderr, "T2: %s: stage %d has no recorded subdivision; "
                        "falling back to dense x8 subdivision\n", label, stage);
            }

            if (!replayed) {
                if (t2_make_subdiv_from_logits_gpu(r, d_logits, d_coords, N, dense_subdiv,
                                                   &sub_idx, &sub_subidx, &new_coords,
                                                   &N_new, &d_sub_idx, &d_sub_subidx,
                                                   &d_new_coords) != 0) {
                    if (!dense_subdiv) {
                        sub_logits = (float *)malloc((size_t)N * 8 * sizeof(float));
                        cuStreamSynchronize(s);
                        cuMemcpyDtoH(sub_logits, d_logits, (size_t)N * 8 * sizeof(float));
                    }
                    if (t2_make_subdiv_from_logits_host(sub_logits, cur_coords, N, dense_subdiv,
                                                        &sub_idx, &sub_subidx, &new_coords,
                                                        &N_new) != 0) {
                        fprintf(stderr, "T2: %s: c2s produced no voxels at stage %d\n", label, stage);
                        free(sub_logits);
                        if (d_logits) cuMemFree(d_logits);
                        t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                        cuMemFree(d_feats); cuMemFree(d_coords); free(cur_coords);
                        T2_RESTORE_SCVAE_GEMM();
                        return -1;
                    }
                }
            }
            free(sub_logits);
            if (d_logits) {
                cuMemFree(d_logits);
                d_logits = 0;
            }

            /* Shape decode: record this stage's subdivision so a subsequent texture
             * decode can replay it (PyTorch decode_tex_slat(..., guide_subs=subs)). */
            if (!dense_subdiv && stage < 8) {
                t2_subdiv_stage *p = &r->subdiv_plan[stage];
                free(p->idx); free(p->subidx); free(p->coords);
                p->idx    = (int32_t *)malloc((size_t)N_new * sizeof(int32_t));
                p->subidx = (int32_t *)malloc((size_t)N_new * sizeof(int32_t));
                p->coords = (int32_t *)malloc((size_t)N_new * 4 * sizeof(int32_t));
                if (p->idx && p->subidx && p->coords) {
                    memcpy(p->idx,    sub_idx,    (size_t)N_new * sizeof(int32_t));
                    memcpy(p->subidx, sub_subidx, (size_t)N_new * sizeof(int32_t));
                    memcpy(p->coords, new_coords, (size_t)N_new * 4 * sizeof(int32_t));
                    p->n_parent = N; p->n_new = N_new; p->valid = 1;
                } else {
                    free(p->idx); free(p->subidx); free(p->coords);
                    p->idx = p->subidx = NULL; p->coords = NULL;
                    p->valid = 0; p->n_parent = 0; p->n_new = 0;
                }
            }

            if (!d_sub_idx)
                d_sub_idx = cu_upload_raw(sub_idx, (size_t)N_new * sizeof(int32_t));
            if (!d_sub_subidx)
                d_sub_subidx = cu_upload_raw(sub_subidx, (size_t)N_new * sizeof(int32_t));
            free(sub_idx); free(sub_subidx);

            ensure_scratch(r, 8, (size_t)N * C_in * sizeof(float));
            ensure_scratch(r, 9, (size_t)N * ((C_in > C_exp) ? C_in : C_exp) * sizeof(float));
            ensure_scratch(r, 10, (size_t)N * C_exp * sizeof(float));
            CUdeviceptr d_norm = r->scratch[8];
            CUdeviceptr d_gather_tmp = r->scratch[9];
            CUdeviceptr d_partial = r->scratch[10];

            if (t2_shape_debug_welford_affine_c2s(stage)) {
                t2_op_layernorm_welford_eps(ops, s, d_norm, d_feats,
                                            dec->c2s[stage].norm1_w,
                                            dec->c2s[stage].norm1_b,
                                            N, C_in, 1e-6f);
            } else {
                t2_op_layernorm(ops, s, d_norm, d_feats,
                                dec->c2s[stage].norm1_w,
                                dec->c2s[stage].norm1_b,
                                N, C_in);
            }
            t2_op_silu_inplace(ops, s, d_norm, N * C_in);
            if (t2_shape_debug_stop_c2s_op(stage, 0)) {
                int rc = t2_shape_debug_return(s, d_norm, cur_coords, N, C_in,
                                               out_feats, out_coords, out_N, out_C);
                cuMemFree(d_sub_idx);
                cuMemFree(d_sub_subidx);
                if (d_new_coords) cuMemFree(d_new_coords);
                free(new_coords);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                cuMemFree(d_feats);
                cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }

            CUdeviceptr d_expanded = 0;
            cuMemAlloc(&d_expanded, (size_t)N * C_exp * sizeof(float));
            t2_run_sparse_conv3d(r, d_expanded, d_norm,
                                  dec->c2s[stage].conv1_w,
                                  dec->c2s[stage].conv1_b,
                                  d_gather_map, d_gather_tmp, d_partial,
                                  cur_coords, hash, N, C_in, C_exp,
                                  stage, -1, 1);
            if (t2_shape_debug_stop_c2s_op(stage, 1)) {
                int rc = t2_shape_debug_return(s, d_expanded, cur_coords, N, C_exp,
                                               out_feats, out_coords, out_N, out_C);
                cuMemFree(d_expanded);
                cuMemFree(d_sub_idx);
                cuMemFree(d_sub_subidx);
                if (d_new_coords) cuMemFree(d_new_coords);
                free(new_coords);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                cuMemFree(d_feats);
                cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }

            CUdeviceptr d_h_fine = 0, d_x_fine = 0;
            cuMemAlloc(&d_h_fine, (size_t)N_new * C_out * sizeof(float));
            cuMemAlloc(&d_x_fine, (size_t)N_new * C_in8 * sizeof(float));
            t2_op_c2s_gather(ops, s, d_h_fine, d_x_fine,
                             d_expanded, d_feats,
                             d_sub_idx, d_sub_subidx,
                             N_new, C_out, C_in8);
            cuStreamSynchronize(s);
            if (t2_shape_debug_stop_c2s_op(stage, 2) ||
                t2_shape_debug_stop_c2s_op(stage, 3)) {
                int want_x = t2_shape_debug_stop_c2s_op(stage, 3);
                int rc = t2_shape_debug_return(s,
                                               want_x ? d_x_fine : d_h_fine,
                                               new_coords, N_new,
                                               want_x ? C_in8 : C_out,
                                               out_feats, out_coords, out_N, out_C);
                cuMemFree(d_expanded);
                cuMemFree(d_sub_idx);
                cuMemFree(d_sub_subidx);
                cuMemFree(d_h_fine);
                cuMemFree(d_x_fine);
                if (d_new_coords) cuMemFree(d_new_coords);
                free(new_coords);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                cuMemFree(d_feats);
                cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
            cuMemFree(d_expanded);
            cuMemFree(d_sub_idx);
            cuMemFree(d_sub_subidx);

            cuMemFree(d_feats);
            d_feats = 0;
            cuMemFree(d_coords);
            d_coords = 0;
            t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
            hash = NULL; d_hash_keys = d_hash_vals = d_gather_map = 0;

            free(cur_coords);
            cur_coords = new_coords;
            N = N_new;
            C = C_out;
            if (d_new_coords) {
                d_coords = d_new_coords;
                d_new_coords = 0;
            } else {
                d_coords = cu_upload_raw(cur_coords, (size_t)N * 4 * sizeof(int32_t));
            }
            if (t2_build_sparse_gpu_index(r, cur_coords, N, d_coords,
                                          &hash, &d_hash_keys, &d_hash_vals,
                                          &d_gather_map, &hash_cap) != 0) {
                cuMemFree(d_h_fine); cuMemFree(d_x_fine); cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return -1;
            }

            ensure_scratch(r, 8, (size_t)N * C * sizeof(float));
            ensure_scratch(r, 9, (size_t)N * C * sizeof(float));
            ensure_scratch(r, 10, (size_t)N * C * sizeof(float));
            CUdeviceptr d_h_norm = r->scratch[8];
            d_gather_tmp = r->scratch[9];
            d_partial = r->scratch[10];

            if (t2_shape_debug_welford_noaffine_c2s(stage)) {
                t2_op_layernorm_noaffine_welford_eps(ops, s, d_h_norm,
                                                     d_h_fine, N, C, 1e-6f);
            } else {
                t2_op_layernorm_noaffine_eps(ops, s, d_h_norm,
                                             d_h_fine, N, C, 1e-6f);
            }
            t2_op_silu_inplace(ops, s, d_h_norm, N * C);
            if (t2_shape_debug_stop_c2s_op(stage, 4)) {
                int rc = t2_shape_debug_return(s, d_h_norm, cur_coords, N, C,
                                               out_feats, out_coords, out_N, out_C);
                cuMemFree(d_h_fine);
                cuMemFree(d_x_fine);
                cuMemFree(d_coords);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
            CUdeviceptr d_c2s_out = 0;
            cuMemAlloc(&d_c2s_out, (size_t)N * C * sizeof(float));
            t2_run_sparse_conv3d(r, d_c2s_out, d_h_norm,
                                  dec->c2s[stage].conv2_w,
                                  dec->c2s[stage].conv2_b,
                                  d_gather_map, d_gather_tmp, d_partial,
                                  cur_coords, hash, N, C, C, stage, -1, 2);
            if (t2_shape_debug_stop_c2s_op(stage, 5)) {
                int rc = t2_shape_debug_return(s, d_c2s_out, cur_coords, N, C,
                                               out_feats, out_coords, out_N, out_C);
                cuMemFree(d_c2s_out);
                cuMemFree(d_h_fine);
                cuMemFree(d_x_fine);
                cuMemFree(d_coords);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
            t2_op_c2s_residual_repeat(ops, s, d_c2s_out, d_x_fine, N, C, C_in8);
            cuStreamSynchronize(s);
            if (t2_shape_debug_stop_c2s_op(stage, 6)) {
                int rc = t2_shape_debug_return(s, d_c2s_out, cur_coords, N, C,
                                               out_feats, out_coords, out_N, out_C);
                cuMemFree(d_c2s_out);
                cuMemFree(d_h_fine);
                cuMemFree(d_x_fine);
                cuMemFree(d_coords);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
            cuMemFree(d_h_fine);
            cuMemFree(d_x_fine);

            d_feats = d_c2s_out;
            struct timespec ts2; clock_gettime(CLOCK_MONOTONIC, &ts2);
            double dt_c2s = (ts2.tv_sec - ts1.tv_sec) * 1000.0 + (ts2.tv_nsec - ts1.tv_nsec) / 1e6;
            fprintf(stderr, "T2: %s: c2s %d->%d %.1f ms, N=%d\n",
                    label, C_in, C_out, dt_c2s, N);
        }

        {
            const char *stop_env = getenv("T2SD_STOP_AFTER_STAGE");
            if (stop_env && atoi(stop_env) == stage) {
                int rc = t2_shape_debug_return(s, d_feats, cur_coords, N, C,
                                               out_feats, out_coords, out_N, out_C);
                t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
                cuMemFree(d_feats);
                cuMemFree(d_coords);
                free(cur_coords);
                T2_RESTORE_SCVAE_GEMM();
                return rc;
            }
        }
    }

    {
        const char *skip_final_ln_env = getenv("T2SD_START_PRE_OUTPUT");
        int skip_final_ln = (start_stage == 4 && skip_final_ln_env &&
                             atoi(skip_final_ln_env) != 0);
        if (!skip_final_ln) {
            int final_ln_mode = 0;
            const char *ln_env = getenv("T2_SCVAE_FINAL_LN_MODE");
            if (ln_env && ln_env[0]) final_ln_mode = atoi(ln_env);
            float final_ln_eps = 1e-5f;
            const char *eps_env = getenv("T2_SCVAE_FINAL_LN_EPS");
            if (eps_env && eps_env[0]) final_ln_eps = (float)atof(eps_env);
            if (final_ln_mode > 0) {
                t2_op_layernorm_noaffine_serial_eps(ops, s, d_feats, d_feats,
                                                    N, C, final_ln_eps, final_ln_mode);
            } else if (t2_env_flag("T2_SCVAE_FINAL_WELFORD_LN")) {
                t2_op_layernorm_noaffine_welford_eps(ops, s, d_feats, d_feats,
                                                     N, C, final_ln_eps);
            } else {
                t2_op_layernorm_noaffine_eps(ops, s, d_feats, d_feats, N, C, final_ln_eps);
            }
        }
    }

    {
        const char *flag = getenv("T2SD_STOP_PRE_OUTPUT");
        if (flag && atoi(flag)) {
            int rc = t2_shape_debug_return(s, d_feats, cur_coords, N, C,
                                           out_feats, out_coords, out_N, out_C);
            free(cur_coords);
            cuMemFree(d_feats);
            cuMemFree(d_coords);
            t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
            T2_RESTORE_SCVAE_GEMM();
            return rc;
        }
    }

    /* output_layer: [N, C=64] -> [N, out_ch] */
    int out_ch = dec->out_channels > 0 ? dec->out_channels : 7;

    cuStreamSynchronize(s);
    CUdeviceptr d_output;
    cuMemAlloc(&d_output, (size_t)N * out_ch * sizeof(float));
    int output_done = 0;
    {
        /* Default to the grouped F32 output kernel (group=25, the resume-validated
         * value). The fall-through cuBLAS/plain path below leaves the small [N,out_ch]
         * output ALL ZERO here (shape/tex decoder), so FDG finds no surface and emits
         * 0 triangles. Callers can still override the value or set =0 to opt out. */
        const char *group_env = getenv("T2_SCVAE_OUTPUT_GROUP");
        int group = group_env && group_env[0] ? atoi(group_env) : 25;
        if (group > 0 &&
            t2_op_gemm_f32_group(ops, s, d_output, dec->out_w, d_feats,
                                 dec->out_b, out_ch, C, N, group) == 0) {
            output_done = 1;
        }
    }
    if (!output_done && getenv("T2_SCVAE_OUTPUT_PAIR32") &&
        t2_op_gemm_f32_pair32(ops, s, d_output, dec->out_w, d_feats,
                              dec->out_b, out_ch, C, N) == 0) {
        output_done = 1;
    }
    if (!output_done && getenv("T2_SCVAE_CUBLASLT_BIAS_GEMM") && dec->out_b &&
        t2_op_gemm_f32_lt_bias(ops, s, d_output, dec->out_w, d_feats,
                               dec->out_b, out_ch, C, N) == 0) {
        output_done = 1;
    }
    if (!output_done && ops->use_cublas_f32 && dec->out_b) {
        t2_op_broadcast_bias(ops, s, d_output, dec->out_b, N, out_ch);
        if (t2_op_gemm_f32_cublas_beta1(ops, s, d_output, dec->out_w,
                                         d_feats, out_ch, C, N) == 0) {
            output_done = 1;
        }
    }
    if (!output_done) {
        t2_op_gemm(ops, s, d_output, dec->out_w, d_feats, dec->out_b,
                   out_ch, C, N);
    }
    cuStreamSynchronize(s);

    float *host_feats = (float *)malloc((size_t)N * out_ch * sizeof(float));
    int32_t *host_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    if (!host_feats || !host_coords) {
        free(host_feats); free(host_coords);
        cuMemFree(d_output);
        free(cur_coords);
        cuMemFree(d_feats);
        cuMemFree(d_coords);
        t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
        T2_RESTORE_SCVAE_GEMM();
        return -1;
    }
    cuMemcpyDtoH(host_feats, d_output, (size_t)N * out_ch * sizeof(float));
    memcpy(host_coords, cur_coords, (size_t)N * 4 * sizeof(int32_t));
    *out_feats = host_feats;
    *out_coords = host_coords;
    if (out_N) *out_N = N;
    if (out_C) *out_C = out_ch;
    cuMemFree(d_output);

    /* Cleanup */
    free(cur_coords);
    cuMemFree(d_feats);
    cuMemFree(d_coords);
    t2_free_sparse_gpu_index(hash, d_hash_keys, d_hash_vals, d_gather_map);
    if (r->pack_cache_valid) {
        t2_sparse_conv_pack_free(&r->pack_cache);
        r->pack_cache_valid = 0;
    }

    T2_RESTORE_SCVAE_GEMM();
    #undef T2_RESTORE_SCVAE_GEMM
    return 0;
}

int cuda_trellis2_run_shape_decoder_alloc(cuda_trellis2_runner *r,
                                           const float *slat,
                                           const int32_t *coords, int N,
                                           float **out_feats,
                                           int32_t **out_coords,
                                           int *out_N,
                                           int *out_C) {
    return cuda_trellis2_run_scvae_decoder_alloc(r, &r->shape_dec, "shape decoder",
                                                  slat, 32, coords, N, -1, 0,
                                                  out_feats, out_coords,
                                                  out_N, out_C);
}

int cuda_trellis2_run_shape_decoder_from_alloc(cuda_trellis2_runner *r,
                                                const float *feats,
                                                int C,
                                                const int32_t *coords, int N,
                                                int start_stage,
                                                int start_block,
                                                float **out_feats,
                                                int32_t **out_coords,
                                                int *out_N,
                                                int *out_C) {
    return cuda_trellis2_run_scvae_decoder_alloc(r, &r->shape_dec, "shape decoder",
                                                  feats, C, coords, N,
                                                  start_stage, start_block,
                                                  out_feats, out_coords,
                                                  out_N, out_C);
}

int cuda_trellis2_run_texture_decoder_alloc(cuda_trellis2_runner *r,
                                             const float *slat,
                                             const int32_t *coords, int N,
                                             float **out_feats,
                                             int32_t **out_coords,
                                             int *out_N,
                                             int *out_C) {
    return cuda_trellis2_run_scvae_decoder_alloc(r, &r->tex_dec, "texture decoder",
                                                  slat, 32, coords, N, -1, 0,
                                                  out_feats, out_coords,
                                                  out_N, out_C);
}

int cuda_trellis2_run_shape_decoder(cuda_trellis2_runner *r,
                                      const float *slat, const int32_t *coords, int N,
                                      float *out_feats, int32_t *out_coords,
                                      int *out_N) {
    float *tmp_feats = NULL;
    int32_t *tmp_coords = NULL;
    int out_count = 0, out_ch = 0;
    int rc = cuda_trellis2_run_shape_decoder_alloc(r, slat, coords, N,
                                                    &tmp_feats, &tmp_coords,
                                                    &out_count, &out_ch);
    if (rc != 0) return rc;
    memcpy(out_feats, tmp_feats, (size_t)out_count * out_ch * sizeof(float));
    memcpy(out_coords, tmp_coords, (size_t)out_count * 4 * sizeof(int32_t));
    if (out_N) *out_N = out_count;
    free(tmp_feats);
    free(tmp_coords);
    return 0;
}

int cuda_trellis2_run_texture_decoder(cuda_trellis2_runner *r,
                                       const float *slat, const int32_t *coords, int N,
                                       float *out_feats, int32_t *out_coords,
                                       int *out_N) {
    float *tmp_feats = NULL;
    int32_t *tmp_coords = NULL;
    int out_count = 0, out_ch = 0;
    int rc = cuda_trellis2_run_texture_decoder_alloc(r, slat, coords, N,
                                                      &tmp_feats, &tmp_coords,
                                                      &out_count, &out_ch);
    if (rc != 0) return rc;
    memcpy(out_feats, tmp_feats, (size_t)out_count * out_ch * sizeof(float));
    memcpy(out_coords, tmp_coords, (size_t)out_count * 4 * sizeof(int32_t));
    if (out_N) *out_N = out_count;
    free(tmp_feats);
    free(tmp_coords);
    return 0;
}

int cuda_trellis2_run_stage2_dit(cuda_trellis2_runner *r,
                                  const float *x_t, float timestep,
                                  const float *cond_features,
                                  const int32_t *coords, int N,
                                  float *output) {
    if (!r->stage2_loaded) {
        fprintf(stderr, "T2: Stage 2 not loaded\n"); return -1;
    }
    CUstream s = r->stream;
    dit_model_gpu *m = &r->stage2;
    int in_ch = m->in_channels;  /* 32 */
    int ctx_len = DINO_SEQ_LEN;
    size_t cond_n = (size_t)ctx_len * DIT_COND_DIM;
    uint64_t cond_hash = t2_hash_f32_bytes(cond_features, cond_n);
    int cache_slot = t2_cond_cache_slot(cond_features, ctx_len);

    CUdeviceptr d_rope_cos = 0, d_rope_sin = 0;
    if (t2_get_sparse_rope_tables(r, 2, m, coords, N,
                                  &d_rope_cos, &d_rope_sin) != 0) {
        fprintf(stderr, "T2: failed to build Stage 2 sparse RoPE tables\n");
        return -1;
    }

    /* Upload input and conditioning. The conditioning tensor is only needed when
     * the per-block cross-attention KV cache is cold for this model/slot. */
    CUdeviceptr d_x = cu_upload_raw(x_t, (size_t)N * in_ch * sizeof(float));
    int have_kv = t2_kv_cache_ready(r, cache_slot, 2, cond_hash, m->n_blocks);
    CUdeviceptr d_cond = have_kv ? 0 :
        cu_upload_raw(cond_features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)N * in_ch * sizeof(float));

    /* Run Stage 2 DiT forward */
    run_sparse_dit_forward(r, m, r->stage2_blocks_cpu,
                            d_x, timestep, d_cond, d_out, N, d_rope_cos, d_rope_sin,
                            2, cond_hash, cache_slot);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)N * in_ch * sizeof(float));

    cuMemFree(d_x);
    if (d_cond) cuMemFree(d_cond);
    cuMemFree(d_out);
    return 0;
}

int cuda_trellis2_run_stage3_dit(cuda_trellis2_runner *r,
                                  const float *x_t, float timestep,
                                  const float *cond_features,
                                  const int32_t *coords, int N,
                                  float *output) {
    if (!r->stage3_loaded) {
        fprintf(stderr, "T2: Stage 3 not loaded\n"); return -1;
    }
    CUstream s = r->stream;
    dit_model_gpu *m = &r->stage3;
    int in_ch = m->in_channels;  /* 64 (noise_32 + shape_slat_32) */
    int ctx_len = DINO_SEQ_LEN;
    size_t cond_n = (size_t)ctx_len * DIT_COND_DIM;
    uint64_t cond_hash = t2_hash_f32_bytes(cond_features, cond_n);
    int cache_slot = t2_cond_cache_slot(cond_features, ctx_len);

    CUdeviceptr d_rope_cos = 0, d_rope_sin = 0;
    if (t2_get_sparse_rope_tables(r, 3, m, coords, N,
                                  &d_rope_cos, &d_rope_sin) != 0) {
        fprintf(stderr, "T2: failed to build Stage 3 sparse RoPE tables\n");
        return -1;
    }

    /* Upload input and conditioning. The conditioning tensor is only needed when
     * the per-block cross-attention KV cache is cold for this model/slot. */
    CUdeviceptr d_x = cu_upload_raw(x_t, (size_t)N * in_ch * sizeof(float));
    int have_kv = t2_kv_cache_ready(r, cache_slot, 3, cond_hash, m->n_blocks);
    CUdeviceptr d_cond = have_kv ? 0 :
        cu_upload_raw(cond_features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    int out_ch = m->out_channels;  /* 32 for Stage 3 */
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)N * out_ch * sizeof(float));

    run_sparse_dit_forward(r, m, r->stage3_blocks_cpu,
                            d_x, timestep, d_cond, d_out, N, d_rope_cos, d_rope_sin,
                            3, cond_hash, cache_slot);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)N * out_ch * sizeof(float));

    cuMemFree(d_x);
    if (d_cond) cuMemFree(d_cond);
    cuMemFree(d_out);
    return 0;
}

int cuda_trellis2_run_dinov3(cuda_trellis2_runner *r,
                              const float *image_f32, float *output) {
    CUstream s = r->stream;
    int seq = DINO_SEQ_LEN;
    int dim = DINO_HIDDEN;

    /* Upload normalized image [3, 512, 512] */
    CUdeviceptr d_image = cu_upload_raw(image_f32,
        (size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)seq * dim * sizeof(float));

    run_dinov3(r, d_image, d_out);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)seq * dim * sizeof(float));

    cuMemFree(d_image); cuMemFree(d_out);
    return 0;
}
