/*
 * paint_stage_unet.c - dual-stream UNet UniPC step driver TU.
 *
 * Lifts the out_loop block from test_paint_unet.c into a reusable opaque API
 * (declared in paint_stages.h). Sole TU including cuda_paint_unet_runner.h so
 * its file-local helpers don't collide with other stage runners.
 *
 * The orchestrator calls:
 *   create -> set_conditioning -> run_dual (once) -> run_step (per UniPC step)
 *
 * No correctness changes vs test_paint_unet --stage out_loop: same kernel
 * sequence, same device-buffer layout, same dual-stream cache semantics.
 * The only differences are (a) conditioning comes from caller-supplied host
 * tensors instead of pyref .npy files, and (b) per-step pyref validation is
 * dropped (the orchestrator drives validation through its own checks).
 */

#include "cuda_paint_unet_runner.h"
#include "paint_stages.h"

#include <stdint.h>

#define MAX_N_BLOCKS 16

struct paint_stage_unet {
    paint_unet_config cfg;

    /* Derived dims */
    int Beff_main;          /* B_outer * N_pbr * N_gen */
    int Beff_dual;          /* B_outer * N_ref */
    size_t per_view;        /* 4 * H0 * W0 */
    size_t per_in_main;     /* IC_main * H0 * W0 = 12 * H0 * W0 */
    size_t txt_per;         /* M_text * cross_dim */
    size_t x_n;             /* Beff_main * per_view (latent element count) */

    /* Kernels + module */
    pu_kernels kk;

    /* Conditioning host scratch (rebuilt by set_conditioning, reused by
     * run_step's input pack). */
    float *packed_main;     /* [Beff_main, 12, H0, W0] - sample slot rewritten per step */
    float *packed_dual;     /* [Beff_dual, 4, H0, W0]  - ref latents */
    float *text_tiled_main; /* [Beff_main, M_text, cross_dim] */

    /* Time-embedding linear weights */
    CUdeviceptr l1_w, l1_b, l2_w, l2_b;          /* main */
    CUdeviceptr l1_wd, l1_bd, l2_wd, l2_bd;      /* dual */

    /* Pre-tiled text inputs */
    CUdeviceptr d_text_dual, d_text_m;

    /* Pre-tiled DINO conditioning (set_conditioning fills d_dino) */
    CUdeviceptr d_dino;
    int M_dino;                                  /* T_dino * EXTRA = T_dino * 4 */

    /* DINO projection weights (resident; reused per set_conditioning) */
    CUdeviceptr dino_pw, dino_pb, dino_png, dino_pnb;

    /* Reference text-clip for dual stream (resident) */
    CUdeviceptr d_text_clip_ref;

    /* Timestep device buffers */
    CUdeviceptr d_ts_main, d_ts_dual;

    /* conv_in / conv_out / norm_out weights */
    CUdeviceptr cw_d, cb_d;                      /* dual conv_in */
    CUdeviceptr cw, cb;                          /* main conv_in */
    CUdeviceptr ng, nb_w, ow, ob_w;              /* main conv_norm_out + conv_out */

    /* Block weights */
    pu_down_block dbd[4];   pu_mid_block midd;   pu_up_block ubd[4]; /* dual */
    pu_down_block db[4];    pu_mid_block mid;    pu_up_block ub[4];  /* main */

    /* Per-step scratch on device */
    CUdeviceptr d_temb_in_d, d_temb_h1_d, d_temb_d;
    CUdeviceptr d_temb_in_m, d_temb_h1_m, d_temb_m;
    CUdeviceptr d_in_raw_d, d_in_raw_m;
    CUdeviceptr d_concat;
    pu_workspace ws;

    /* RA cache backing storage (g_ra_cache slots) */
    pu_ra_slot ra_slots[MAX_N_BLOCKS];

    /* Whether conditioning has been set + dual pass run */
    int cond_set;
    int dual_done;
};

paint_stage_unet *paint_stage_unet_create(CUdevice dev,
                                           const char *unet_safetensors_path,
                                           const paint_unet_config *cfg) {
    (void)dev; /* current ctx is assumed to be active */
    paint_stage_unet *s = (paint_stage_unet *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = *cfg;
    s->Beff_main   = cfg->B_outer * cfg->N_pbr * cfg->N_gen;
    s->Beff_dual   = cfg->B_outer * cfg->N_ref;
    s->per_view    = (size_t)4 * cfg->H0 * cfg->W0;
    s->per_in_main = (size_t)12 * cfg->H0 * cfg->W0;
    s->txt_per     = (size_t)cfg->M_text * cfg->cross_dim;
    s->x_n         = (size_t)s->Beff_main * s->per_view;

    /* Module compile + kernel handles */
    int sm = cu_compile_kernels(&s->kk.mod, dev,
                                 cuda_paint_unet_kernels_src,
                                 "hy3d_paint_unet", 1, "HY3D-PAINT-UNET");
    if (sm < 0) { free(s); return NULL; }
    /* Resolve kernel handles (mirrors test_paint_unet main()). */
    cuModuleGetFunction(&s->kk.f_tse,     s->kk.mod, "unet_timestep_embed_f32");
    cuModuleGetFunction(&s->kk.f_lin,     s->kk.mod, "unet_linear_f32");
    cuModuleGetFunction(&s->kk.f_silu,    s->kk.mod, "unet_silu_f32");
    cuModuleGetFunction(&s->kk.f_conv,    s->kk.mod, "unet_conv2d_f32");
    cuModuleGetFunction(&s->kk.f_gn,      s->kk.mod, "unet_groupnorm_f32");
    cuModuleGetFunction(&s->kk.f_addc,    s->kk.mod, "unet_add_chan_f32");
    cuModuleGetFunction(&s->kk.f_add,     s->kk.mod, "unet_add_f32");
    cuModuleGetFunction(&s->kk.f_ln,      s->kk.mod, "unet_layernorm_f32");
    cuModuleGetFunction(&s->kk.f_chw_nc,  s->kk.mod, "unet_chw_to_nc_f32");
    cuModuleGetFunction(&s->kk.f_nc_chw,  s->kk.mod, "unet_nc_to_chw_f32");
    cuModuleGetFunction(&s->kk.f_mha,     s->kk.mod, "unet_mha_f32");
    cuModuleGetFunction(&s->kk.f_geglu,   s->kk.mod, "unet_geglu_f32");
    cuModuleGetFunction(&s->kk.f_conv_s2, s->kk.mod, "unet_conv2d_stride2_f32");
    cuModuleGetFunction(&s->kk.f_up2x,    s->kk.mod, "unet_upsample2x_f32");
    cuModuleGetFunction(&s->kk.f_concat,  s->kk.mod, "unet_concat_chan_f32");
    cuModuleGetFunction(&s->kk.f_rope,    s->kk.mod, "unet_rope_apply_f32");
    cuModuleGetFunction(&s->kk.f_ra_split_v, s->kk.mod, "unet_ra_split_v_f32");

    /* Open weights */
    st_context *st = safetensors_open(unet_safetensors_path);
    if (!st) {
        fprintf(stderr, "ERROR: cannot open %s\n", unet_safetensors_path);
        cuModuleUnload(s->kk.mod); free(s); return NULL;
    }

    /* Time embedding linears (main + dual) */
    s->l1_w  = upload_st(st, "time_embedding.linear_1.weight");
    s->l1_b  = upload_st(st, "time_embedding.linear_1.bias");
    s->l2_w  = upload_st(st, "time_embedding.linear_2.weight");
    s->l2_b  = upload_st(st, "time_embedding.linear_2.bias");
    s->l1_wd = upload_st(st, "unet_dual.time_embedding.linear_1.weight");
    s->l1_bd = upload_st(st, "unet_dual.time_embedding.linear_1.bias");
    s->l2_wd = upload_st(st, "unet_dual.time_embedding.linear_2.weight");
    s->l2_bd = upload_st(st, "unet_dual.time_embedding.linear_2.bias");

    /* Dual conv_in + blocks */
    g_load_wp = "unet_dual.";
    s->cw_d = upload_st(st, "unet_dual.conv_in.weight");
    s->cb_d = upload_st(st, "unet_dual.conv_in.bias");
    load_down_block(st, &s->dbd[0], 0,  320,  320,  5, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_down_block(st, &s->dbd[1], 1,  320,  640, 10, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_down_block(st, &s->dbd[2], 2,  640, 1280, 20, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_down_block(st, &s->dbd[3], 3, 1280, 1280, 20, 0, 0, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_mid_block(st, &s->midd, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[0], 0, 1280, 1280, 1280,  0, 0, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[1], 1,  640, 1280, 1280, 20, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[2], 2,  320,  640, 1280, 10, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[3], 3,  320,  320,  640,  5, 1, 0, cfg->cross_dim, 0, 0, 0, 0, 0, 0);

    /* Main conv_in + blocks (full 4-attn-path config) */
    g_load_wp = "";
    s->cw = upload_st(st, "conv_in.weight");
    s->cb = upload_st(st, "conv_in.bias");
    load_down_block(st, &s->db[0], 0,  320,  320,  5, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_down_block(st, &s->db[1], 1,  320,  640, 10, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_down_block(st, &s->db[2], 2,  640, 1280, 20, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_down_block(st, &s->db[3], 3, 1280, 1280, 20, 0, 0, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_mid_block(st, &s->mid, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[1], 1,  640, 1280, 1280, 20, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[2], 2,  320,  640, 1280, 10, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[3], 3,  320,  320,  640,  5, 1, 0, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);

    s->ng   = upload_st(st, "conv_norm_out.weight");
    s->nb_w = upload_st(st, "conv_norm_out.bias");
    s->ow   = upload_st(st, "conv_out.weight");
    s->ob_w = upload_st(st, "conv_out.bias");

    /* DINO image proj weights — kept resident; set_conditioning runs the
     * Linear+LayerNorm with the caller-supplied dino_hidden_states. */
    s->M_dino = cfg->T_dino * 4;            /* EXTRA=4 */
    s->dino_pw  = upload_st(st, "image_proj_model_dino.proj.weight");
    s->dino_pb  = upload_st(st, "image_proj_model_dino.proj.bias");
    s->dino_png = upload_st(st, "image_proj_model_dino.norm.weight");
    s->dino_pnb = upload_st(st, "image_proj_model_dino.norm.bias");
    s->d_dino = 0; /* lazy alloc in set_conditioning */

    /* Dual-stream reference text-clip (resident, broadcast at set_conditioning). */
    s->d_text_clip_ref = upload_st(st, "learned_text_clip_ref");
    s->d_text_dual = 0;
    s->d_text_m    = 0;

    safetensors_close(st);

    /* Workspace + per-step buffers */
    const int H0 = cfg->H0, W0 = cfg->W0, M_text = cfg->M_text;
    size_t MAX_ACT  = (size_t)s->Beff_main * 1280 * H0 * W0;
    size_t MAX_CCAT = (size_t)s->Beff_main * 960  * H0 * W0;
    size_t MAX_FF_GH= (size_t)s->Beff_main * 320  * H0 * W0 * 2 * 4;
    size_t MAX_FF_H = (size_t)s->Beff_main * 320  * H0 * W0 * 4;
    size_t MAX_BNC  = (size_t)s->Beff_main * 320  * H0 * W0;
    if ((size_t)s->Beff_main * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)s->Beff_main * 1280 * 16 * 16;
    if ((size_t)s->Beff_main *  640 * 32 * 32 > MAX_BNC) MAX_BNC = (size_t)s->Beff_main *  640 * 32 * 32;
    size_t MAX_BMC = MAX_BNC;
    if ((size_t)s->Beff_main * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)s->Beff_main * 1280 * M_text;
    if ((size_t)s->Beff_main * 1280 * s->M_dino > MAX_BMC) MAX_BMC = (size_t)s->Beff_main * 1280 * s->M_dino;

    cuMemAlloc(&s->ws.d_a, MAX_ACT * sizeof(float));
    cuMemAlloc(&s->ws.d_b, MAX_ACT * sizeof(float));
    cuMemAlloc(&s->ws.d_t1, MAX_CCAT * sizeof(float));
    cuMemAlloc(&s->ws.d_t2, MAX_CCAT * sizeof(float));
    cuMemAlloc(&s->ws.d_temb_act,  s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->ws.d_temb_proj, s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->ws.X.d_resid, MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_nc,    MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_nc_b,  MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_norm,  MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_q,     MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_k,     MAX_BMC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_v,     MAX_BMC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_attn,  MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
    cuMemAlloc(&s->ws.X.d_ff_h,  MAX_FF_H * sizeof(float));
    cuMemAlloc(&s->d_concat,     MAX_CCAT * sizeof(float));

    cuMemAlloc(&s->d_temb_in_d, s->Beff_dual * 320 * sizeof(float));
    cuMemAlloc(&s->d_temb_h1_d, s->Beff_dual * 1280 * sizeof(float));
    cuMemAlloc(&s->d_temb_d,    s->Beff_dual * 1280 * sizeof(float));
    cuMemAlloc(&s->d_temb_in_m, s->Beff_main * 320 * sizeof(float));
    cuMemAlloc(&s->d_temb_h1_m, s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->d_temb_m,    s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->d_in_raw_d,  (size_t)s->Beff_dual * 4 * H0 * W0 * sizeof(float));
    cuMemAlloc(&s->d_in_raw_m,  (size_t)s->Beff_main * 12 * H0 * W0 * sizeof(float));
    cuMemAlloc(&s->d_ts_main,   s->Beff_main * sizeof(int64_t));
    cuMemAlloc(&s->d_ts_dual,   s->Beff_dual * sizeof(int64_t));

    /* RA cache: backed by a fixed 16-slot array; runner header reads from
     * the global g_ra_cache. */
    g_ra_cache.slots = s->ra_slots;
    g_ra_cache.n_slots = MAX_N_BLOCKS;
    g_ra_cache.idx = 0;
    g_ra_n_ref = cfg->N_ref;

    /* Host scratch */
    s->packed_main     = (float *)malloc(s->Beff_main * s->per_in_main * sizeof(float));
    s->packed_dual     = (float *)malloc(s->Beff_dual * s->per_view    * sizeof(float));
    s->text_tiled_main = (float *)malloc(s->Beff_main * s->txt_per     * sizeof(float));

    return s;
}

void paint_stage_unet_set_conditioning(paint_stage_unet *s,
    const float *embeds_normal,
    const float *embeds_position,
    const float *encoder_hidden_states,
    const float *ref_latents,
    const float *dino_hidden_states) {

    const paint_unet_config *cfg = &s->cfg;
    const int N_PBR = cfg->N_pbr, N_GEN = cfg->N_gen;
    const size_t per_view = s->per_view, per_in_main = s->per_in_main;
    const size_t txt_per = s->txt_per;

    /* Pack en/ep into the conditioning slots of packed_main.
     * Layout: per-batch dst[12,H,W] with channel slots [sample | normal | position]. */
    for (int p = 0; p < N_PBR; p++)
        for (int g = 0; g < N_GEN; g++) {
            int b = p * N_GEN + g;
            float *dst = s->packed_main + (size_t)b * per_in_main;
            memcpy(dst + per_view,     embeds_normal   + (size_t)g * per_view, per_view * sizeof(float));
            memcpy(dst + 2 * per_view, embeds_position + (size_t)g * per_view, per_view * sizeof(float));
        }

    /* Dual ref latents */
    memcpy(s->packed_dual, ref_latents, (size_t)s->Beff_dual * per_view * sizeof(float));

    /* Tile text per material */
    for (int p = 0; p < N_PBR; p++)
        for (int g = 0; g < N_GEN; g++) {
            int b = p * N_GEN + g;
            memcpy(s->text_tiled_main + (size_t)b * txt_per,
                   encoder_hidden_states + (size_t)p * txt_per,
                   txt_per * sizeof(float));
        }
    if (!s->d_text_m)
        cuMemAlloc(&s->d_text_m, (size_t)s->Beff_main * txt_per * sizeof(float));
    cuMemcpyHtoD(s->d_text_m, s->text_tiled_main, (size_t)s->Beff_main * txt_per * sizeof(float));

    /* Dual text: broadcast learned_text_clip_ref [1, M_text, cross_dim] to
     * Beff_dual rows. */
    if (!s->d_text_dual)
        cuMemAlloc(&s->d_text_dual, (size_t)s->Beff_dual * txt_per * sizeof(float));
    for (int b = 0; b < s->Beff_dual; b++)
        cuMemcpyDtoD(s->d_text_dual + (CUdeviceptr)b * txt_per * sizeof(float),
                     s->d_text_clip_ref, txt_per * sizeof(float));

    /* DINO projection: Linear(C_in -> EXTRA*CTX) + LayerNorm(CTX), then
     * broadcast to Beff_main rows. */
    {
        const int CTX = 1024, EXTRA = 4;
        const int T = cfg->T_dino, Cin = cfg->C_dino_in;
        const int rows_out = T * EXTRA;
        size_t dino_per = (size_t)rows_out * CTX * sizeof(float);
        CUdeviceptr d_din, d_dlin, d_done;
        cuMemAlloc(&d_din,  (size_t)T * Cin * sizeof(float));
        cuMemAlloc(&d_dlin, (size_t)T * EXTRA * CTX * sizeof(float));
        cuMemAlloc(&d_done, dino_per);
        cuMemcpyHtoD(d_din, dino_hidden_states, (size_t)T * Cin * sizeof(float));
        k_linear(&s->kk, d_dlin, d_din, s->dino_pw, s->dino_pb, T, Cin, EXTRA * CTX);
        k_layernorm(&s->kk, d_done, d_dlin, s->dino_png, s->dino_pnb, rows_out, CTX);
        if (!s->d_dino)
            cuMemAlloc(&s->d_dino, (size_t)s->Beff_main * dino_per);
        for (int b = 0; b < s->Beff_main; b++)
            cuMemcpyDtoD(s->d_dino + (CUdeviceptr)b * dino_per, d_done, dino_per);
        cuMemFree(d_din); cuMemFree(d_dlin); cuMemFree(d_done);
    }

    s->cond_set = 1;
    s->dual_done = 0;
}

void paint_stage_unet_run_dual(paint_stage_unet *s) {
    if (!s->cond_set) {
        fprintf(stderr, "[paint_stage_unet] ERROR: set_conditioning first\n");
        return;
    }
    const paint_unet_config *cfg = &s->cfg;
    const int H0 = cfg->H0, W0 = cfg->W0, M_text = cfg->M_text;
    const int IC_dual = 4;

    g_ra_mode = 1; g_ra_cache.idx = 0;

    int64_t ts_dual_arr[16];
    for (int b = 0; b < s->Beff_dual; b++) ts_dual_arr[b] = 0;
    cuMemcpyHtoD(s->d_ts_dual, ts_dual_arr, s->Beff_dual * sizeof(int64_t));
    k_timestep_embed(&s->kk, s->d_temb_in_d, s->d_ts_dual, s->Beff_dual, 320);
    k_linear(&s->kk, s->d_temb_h1_d, s->d_temb_in_d, s->l1_wd, s->l1_bd, s->Beff_dual, 320, 1280);
    k_silu(&s->kk, s->d_temb_h1_d, s->Beff_dual * 1280);
    k_linear(&s->kk, s->d_temb_d, s->d_temb_h1_d, s->l2_wd, s->l2_bd, s->Beff_dual, 1280, 1280);

    size_t in_n_d = (size_t)s->Beff_dual * IC_dual * H0 * W0;
    cuMemcpyHtoD(s->d_in_raw_d, s->packed_dual, in_n_d * sizeof(float));
    for (int b = 0; b < s->Beff_dual; b++) {
        CUdeviceptr ib = s->d_in_raw_d + (CUdeviceptr)b * IC_dual * H0 * W0 * sizeof(float);
        CUdeviceptr ob = s->ws.d_a     + (CUdeviceptr)b * 320     * H0 * W0 * sizeof(float);
        k_conv(&s->kk, ob, ib, s->cw_d, s->cb_d, IC_dual, H0, W0, 320, 3, 3, 1);
    }
    pu_skip_stack ssd = {.top = 0, .B = s->Beff_dual};
    skip_push_copy(&ssd, s->ws.d_a, 320, H0, W0);
    int H = H0, W = W0;
    run_down_block(&s->kk, &s->dbd[0], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_down_block(&s->kk, &s->dbd[1], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_down_block(&s->kk, &s->dbd[2], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_down_block(&s->kk, &s->dbd[3], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_mid_block(&s->kk, &s->midd, s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, H, W, M_text, &s->ws);
    run_up_block(&s->kk, &s->ubd[0], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_up_block(&s->kk, &s->ubd[1], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_up_block(&s->kk, &s->ubd[2], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_up_block(&s->kk, &s->ubd[3], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    cuCtxSynchronize();
    s->dual_done = 1;
    fprintf(stderr, "[paint_stage_unet] dual-stream RA cache populated (%d slots)\n", g_ra_cache.idx);
}

void paint_stage_unet_run_step(paint_stage_unet *s, long long timestep,
                                const float *x_host, float *noise_pred_host) {
    if (!s->dual_done) {
        fprintf(stderr, "[paint_stage_unet] ERROR: run_dual first\n");
        return;
    }
    const paint_unet_config *cfg = &s->cfg;
    const int H0 = cfg->H0, W0 = cfg->W0, M_text = cfg->M_text;
    const int N_PBR = cfg->N_pbr, N_GEN = cfg->N_gen;
    const int IC_main = 12;
    const size_t per_view = s->per_view, per_in_main = s->per_in_main;
    int H = H0, W = W0;

    /* Pack current x into the sample slot of packed_main */
    for (int p = 0; p < N_PBR; p++)
        for (int g = 0; g < N_GEN; g++) {
            int b = p * N_GEN + g;
            float *dst = s->packed_main + (size_t)b * per_in_main;
            memcpy(dst, x_host + (size_t)b * per_view, per_view * sizeof(float));
        }
    size_t in_n_m = (size_t)s->Beff_main * IC_main * H0 * W0;
    cuMemcpyHtoD(s->d_in_raw_m, s->packed_main, in_n_m * sizeof(float));

    /* Timestep embed for current t */
    int64_t ts_main_arr[16];
    for (int b = 0; b < s->Beff_main; b++) ts_main_arr[b] = (int64_t)timestep;
    cuMemcpyHtoD(s->d_ts_main, ts_main_arr, s->Beff_main * sizeof(int64_t));
    k_timestep_embed(&s->kk, s->d_temb_in_m, s->d_ts_main, s->Beff_main, 320);
    k_linear(&s->kk, s->d_temb_h1_m, s->d_temb_in_m, s->l1_w, s->l1_b, s->Beff_main, 320, 1280);
    k_silu(&s->kk, s->d_temb_h1_m, s->Beff_main * 1280);
    k_linear(&s->kk, s->d_temb_m, s->d_temb_h1_m, s->l2_w, s->l2_b, s->Beff_main, 1280, 1280);

    /* Main forward (RA mode='r', read in-order from cache) */
    g_ra_mode = 2; g_ra_cache.idx = 0;
    for (int b = 0; b < s->Beff_main; b++) {
        CUdeviceptr ib = s->d_in_raw_m + (CUdeviceptr)b * IC_main * H0 * W0 * sizeof(float);
        CUdeviceptr ob = s->ws.d_a     + (CUdeviceptr)b * 320     * H0 * W0 * sizeof(float);
        k_conv(&s->kk, ob, ib, s->cw, s->cb, IC_main, H0, W0, 320, 3, 3, 1);
    }
    pu_skip_stack ss = {.top = 0, .B = s->Beff_main};
    skip_push_copy(&ss, s->ws.d_a, 320, H0, W0);
    run_down_block(&s->kk, &s->db[0], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_down_block(&s->kk, &s->db[1], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_down_block(&s->kk, &s->db[2], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_down_block(&s->kk, &s->db[3], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_mid_block(&s->kk, &s->mid, s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, H, W, M_text, &s->ws);
    run_up_block(&s->kk, &s->ub[0], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_up_block(&s->kk, &s->ub[1], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_up_block(&s->kk, &s->ub[2], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_up_block(&s->kk, &s->ub[3], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);

    for (int b = 0; b < s->Beff_main; b++) {
        CUdeviceptr xb = s->ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
        CUdeviceptr yb = s->ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
        k_groupnorm(&s->kk, yb, xb, s->ng, s->nb_w, 320, H * W, 32, 1);
    }
    for (int b = 0; b < s->Beff_main; b++) {
        CUdeviceptr yb = s->ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
        CUdeviceptr ob = s->ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
        k_conv(&s->kk, ob, yb, s->ow, s->ob_w, 320, H, W, 4, 3, 3, 1);
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(noise_pred_host, s->ws.d_a, s->x_n * sizeof(float));
}

void paint_stage_unet_destroy(paint_stage_unet *s) {
    if (!s) return;
    free(s->packed_main); free(s->packed_dual); free(s->text_tiled_main);
    if (s->kk.mod) cuModuleUnload(s->kk.mod);
    g_ra_mode = 0;
    free(s);
}
