/*
 * test_paint_unet.c - Native CUDA SD-2.1 paint UNet (Phase 3 skeleton).
 *
 * Loads stock paint UNet weights (paint_unet_stock.safetensors produced by
 * ref/hy3d/export_paint_unet_safetensors.py), runs forward pieces, and
 * diffs them against the diffusers reference dump from
 * ref/hy3d/dump_paint_unet.py.
 *
 * Phase 3 incremental: at each iteration we add another stage and validate
 * one intermediate. Current stages live behind --stage <name>:
 *   time_emb : timestep_embedding + time MLP -> [B, 1280]
 *   conv_in  : conv_in 12->320 -> [B, 320, 64, 64]
 *
 * Usage:
 *   ./test_paint_unet --stage conv_in \\
 *       /mnt/disk01/.../unet/paint_unet_stock.safetensors \\
 *       /tmp/hy3d_paint_unet_ref/
 */

#include "cuda_paint_unet_runner.h"
#include "cuda_paint_unipc.h"

int main(int argc, char **argv) {
    const char *stage = "conv_in";
    int argi = 1;
    if (argi < argc && !strcmp(argv[argi], "--stage")) {
        stage = argv[argi+1]; argi += 2;
    }
    if (argc - argi < 2) {
        fprintf(stderr,
            "Usage: %s [--stage time_emb|conv_in|db0_res0|db0_attn0|out] <unet.safetensors> <ref_dir>\n",
            argv[0]);
        return 1;
    }
    const char *st_path = argv[argi];
    const char *ref_dir = argv[argi+1];

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    pu_kernels kk = {0};
    if (cu_compile_kernels(&kk.mod, dev, cuda_paint_unet_kernels_src,
                             "hy3d_paint_unet", 1, "HY3D-PAINT-UNET") < 0)
        return 1;
    cuModuleGetFunction(&kk.f_tse,  kk.mod, "unet_timestep_embed_f32");
    cuModuleGetFunction(&kk.f_lin,  kk.mod, "unet_linear_f32");
    cuModuleGetFunction(&kk.f_silu, kk.mod, "unet_silu_f32");
    cuModuleGetFunction(&kk.f_conv, kk.mod, "unet_conv2d_f32");
    cuModuleGetFunction(&kk.f_gn,   kk.mod, "unet_groupnorm_f32");
    cuModuleGetFunction(&kk.f_addc, kk.mod, "unet_add_chan_f32");
    cuModuleGetFunction(&kk.f_add,  kk.mod, "unet_add_f32");
    cuModuleGetFunction(&kk.f_ln,     kk.mod, "unet_layernorm_f32");
    cuModuleGetFunction(&kk.f_chw_nc, kk.mod, "unet_chw_to_nc_f32");
    cuModuleGetFunction(&kk.f_nc_chw, kk.mod, "unet_nc_to_chw_f32");
    cuModuleGetFunction(&kk.f_mha,    kk.mod, "unet_mha_f32");
    cuModuleGetFunction(&kk.f_geglu,  kk.mod, "unet_geglu_f32");
    cuModuleGetFunction(&kk.f_conv_s2, kk.mod, "unet_conv2d_stride2_f32");
    cuModuleGetFunction(&kk.f_up2x,    kk.mod, "unet_upsample2x_f32");
    cuModuleGetFunction(&kk.f_concat,  kk.mod, "unet_concat_chan_f32");
    cuModuleGetFunction(&kk.f_rope,    kk.mod, "unet_rope_apply_f32");
    cuModuleGetFunction(&kk.f_ra_split_v, kk.mod, "unet_ra_split_v_f32");

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "ERROR: cannot open %s\n", st_path); return 1; }
    fprintf(stderr, "loaded safetensors %s\n", st_path);

    /* Load reference inputs from the dump dir. dino_proj / out_dino stages
     * use the wrapper-style `in_*.npy` prefix and are validated separately
     * below; the original `ref_*.npy` stages still go through this branch. */
    char path[512];
    int nd; uint64_t shape[8]; size_t n; char dt[8];
    int64_t *ts = NULL;
    int B = 0;
    int wrapper_stage = (!strcmp(stage, "dino_proj") || !strcmp(stage, "out_dino")
                          || !strcmp(stage, "out_ma")
                          || !strcmp(stage, "out_ma_rope")
                          || !strcmp(stage, "out_mda")
                          || !strcmp(stage, "out_ra")
                          || !strcmp(stage, "out_all")
                          || !strcmp(stage, "out_all_rope")
                          || !strcmp(stage, "out_loop"));
    if (!wrapper_stage) {
        snprintf(path, sizeof(path), "%s/ref_timestep.npy", ref_dir);
        ts = (int64_t *)read_npy(path, &nd, shape, &n, dt);
        if (!ts) return 1;
        B = (int)shape[0];
        fprintf(stderr, "B=%d, timestep[0]=%lld\n", B, (long long)ts[0]);
    }

    if (!strcmp(stage, "time_emb")) {
        /* timestep -> sinusoidal[320] -> linear(320,1280) silu linear(1280,1280)
         * Output [B,1280] vs ref_time_emb.npy */
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_emb;  cuMemAlloc(&d_emb, B * 320 * sizeof(float));
        CUdeviceptr d_h1;   cuMemAlloc(&d_h1,  B * 1280 * sizeof(float));
        CUdeviceptr d_h2;   cuMemAlloc(&d_h2,  B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");

        k_timestep_embed(&kk, d_emb, d_ts, B, 320);
        k_linear(&kk, d_h1, d_emb, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_h1, B * 1280);
        k_linear(&kk, d_h2, d_h1, l2_w, l2_b, B, 1280, 1280);
        cuCtxSynchronize();

        float *cu = (float *)malloc(B * 1280 * sizeof(float));
        cuMemcpyDtoH(cu, d_h2, B * 1280 * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_time_emb.npy", ref_dir);
        diff_against(cu, path, (size_t)B * 1280, 1e-3f);
        free(cu);
    } else if (!strcmp(stage, "conv_in")) {
        /* Read sample, run conv_in 12->320, compare to ref_conv_in.npy */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) {
            fprintf(stderr, "ERROR: expected sample channels=12, got %d\n", IC);
            return 1;
        }
        fprintf(stderr, "sample [%d, %d, %d, %d]\n", B, IC, H, W);
        size_t in_n  = (size_t)B * IC * H * W;
        size_t out_n = (size_t)B * 320 * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n  * sizeof(float));
        CUdeviceptr d_out; cuMemAlloc(&d_out, out_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        /* Batch loop: kernel handles one sample (CHW) at a time. */
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b  = d_in  + (CUdeviceptr)b * IC  * H * W * sizeof(float);
            CUdeviceptr out_b = d_out + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_conv(&kk, out_b, in_b, cw, cb, IC, H, W, 320, 3, 3, 1);
        }
        cuCtxSynchronize();

        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, d_out, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_conv_in.npy", ref_dir);
        diff_against(cu, path, out_n, 1e-3f);
        free(cu); free(sample);
    } else if (!strcmp(stage, "db0_res0")) {
        /* Pipeline up to and including down_blocks[0].resnets[0]:
         *   time_emb      [B, 1280]
         *   conv_in(x)    [B, 320, 64, 64]
         *   resblock(.,t) [B, 320, 64, 64]   vs ref_db0_res0.npy
         */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) { fprintf(stderr, "ERROR: IC!=12\n"); return 1; }

        /* --- time embedding --- */
        CUdeviceptr d_ts;   cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_temb_in; cuMemAlloc(&d_temb_in, B * 320 * sizeof(float));
        CUdeviceptr d_temb_h1; cuMemAlloc(&d_temb_h1, B * 1280 * sizeof(float));
        CUdeviceptr d_temb;    cuMemAlloc(&d_temb,    B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, B, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_temb_h1, B * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, B, 1280, 1280);

        /* --- conv_in --- */
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        size_t in_n  = (size_t)B * IC * H * W;
        size_t hw_n  = (size_t)B * 320 * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n * sizeof(float));
        CUdeviceptr d_x;   cuMemAlloc(&d_x,   hw_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b = d_in + (CUdeviceptr)b * IC  * H * W * sizeof(float);
            CUdeviceptr x_b  = d_x  + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_conv(&kk, x_b, in_b, cw, cb, IC, H, W, 320, 3, 3, 1);
        }

        /* --- resblock --- */
        pu_resblock r;
        load_resblock(st, &r, "down_blocks.0.resnets.0", 320, 320);
        CUdeviceptr d_out, d_t1, d_t2, d_temb_act, d_temb_proj;
        cuMemAlloc(&d_out, hw_n * sizeof(float));
        cuMemAlloc(&d_t1,  hw_n * sizeof(float));
        cuMemAlloc(&d_t2,  hw_n * sizeof(float));
        cuMemAlloc(&d_temb_act,  B * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_proj, B * 320  * sizeof(float));
        run_resblock(&kk, &r, d_x, d_out, d_t1, d_t2,
                      d_temb, d_temb_act, d_temb_proj, B, H, W, 32);
        cuCtxSynchronize();

        float *cu = (float *)malloc(hw_n * sizeof(float));
        cuMemcpyDtoH(cu, d_out, hw_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_db0_res0.npy", ref_dir);
        diff_against(cu, path, hw_n, 1e-3f);
        free(cu); free(sample);
    } else if (!strcmp(stage, "db0_attn0")) {
        /* Pipeline: time_emb + conv_in + db0.res0 + db0.attn0 (Transformer2D)
         * Validate against ref_db0_attn0.npy [B, 320, 64, 64]. */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) { fprintf(stderr, "ERROR: IC!=12\n"); return 1; }
        snprintf(path, sizeof(path), "%s/ref_encoder_hidden.npy", ref_dir);
        float *text = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!text) return 1;
        int M_text = (int)shape[1], cross_dim = (int)shape[2];
        fprintf(stderr, "text [%d, %d, %d]\n", B, M_text, cross_dim);

        int C = 320, N = H * W;

        /* --- time embedding --- */
        CUdeviceptr d_ts;   cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_temb_in; cuMemAlloc(&d_temb_in, B * 320 * sizeof(float));
        CUdeviceptr d_temb_h1; cuMemAlloc(&d_temb_h1, B * 1280 * sizeof(float));
        CUdeviceptr d_temb;    cuMemAlloc(&d_temb,    B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, B, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_temb_h1, B * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, B, 1280, 1280);

        /* --- conv_in --- */
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        size_t in_n = (size_t)B * IC * H * W;
        size_t hw_n = (size_t)B * C * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n * sizeof(float));
        CUdeviceptr d_x;   cuMemAlloc(&d_x,   hw_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b = d_in + (CUdeviceptr)b * IC * H * W * sizeof(float);
            CUdeviceptr x_b  = d_x  + (CUdeviceptr)b * C  * H * W * sizeof(float);
            k_conv(&kk, x_b, in_b, cw, cb, IC, H, W, C, 3, 3, 1);
        }

        /* --- res0 --- */
        pu_resblock r;
        load_resblock(st, &r, "down_blocks.0.resnets.0", C, C);
        CUdeviceptr d_res, d_t1, d_t2, d_temb_act, d_temb_proj;
        cuMemAlloc(&d_res, hw_n * sizeof(float));
        cuMemAlloc(&d_t1,  hw_n * sizeof(float));
        cuMemAlloc(&d_t2,  hw_n * sizeof(float));
        cuMemAlloc(&d_temb_act,  B * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_proj, B * 1280 * sizeof(float)); /* big enough */
        run_resblock(&kk, &r, d_x, d_res, d_t1, d_t2,
                      d_temb, d_temb_act, d_temb_proj, B, H, W, 32);

        /* --- attn0 (Transformer2DModel, 1 BasicTransformerBlock at this level) --- */
        pu_transformer T;
        load_transformer(st, &T, "down_blocks.0.attentions.0", C,
                          /*num_heads*/ 5, cross_dim, /*num_blocks*/ 1,
                          /*has_dino*/ 0, /*has_ma*/ 0, /*has_mda*/ 0, /*has_ra*/ 0, /*n_pbr*/ 0, /*n_gen*/ 0);
        /* text upload */
        CUdeviceptr d_text; cuMemAlloc(&d_text, (size_t)B * M_text * cross_dim * sizeof(float));
        cuMemcpyHtoD(d_text, text, (size_t)B * M_text * cross_dim * sizeof(float));

        /* Scratch: B*N*C floats per buffer (and the 2x ff_inner one). */
        size_t bnc = (size_t)B * N * C;
        size_t bmc = (size_t)B * (M_text > N ? M_text : N) * C;
        size_t bn2ff = (size_t)B * N * 2 * T.ff_inner;
        size_t bnff  = (size_t)B * N *     T.ff_inner;
        pu_xfm_scratch S;
        cuMemAlloc(&S.d_resid, bnc * sizeof(float));
        cuMemAlloc(&S.d_nc,    bnc * sizeof(float));
        cuMemAlloc(&S.d_nc_b,  bnc * sizeof(float));
        cuMemAlloc(&S.d_norm,  bnc * sizeof(float));
        cuMemAlloc(&S.d_q,     bnc * sizeof(float));
        cuMemAlloc(&S.d_k,     bmc * sizeof(float));
        cuMemAlloc(&S.d_v,     bmc * sizeof(float));
        cuMemAlloc(&S.d_attn,  bnc * sizeof(float));
        cuMemAlloc(&S.d_ff_gh, bn2ff * sizeof(float));
        cuMemAlloc(&S.d_ff_h,  bnff  * sizeof(float));

        run_transformer(&kk, &T, d_res, d_text, /*d_dino*/0, /*M_dino*/0, B, H, W, M_text, &S);
        cuCtxSynchronize();

        float *cu = (float *)malloc(hw_n * sizeof(float));
        cuMemcpyDtoH(cu, d_res, hw_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_db0_attn0.npy", ref_dir);
        diff_against(cu, path, hw_n, 1e-3f);
        free(cu); free(sample); free(text);
    } else if (!strcmp(stage, "out")) {
        /* Full UNet forward -> [B, 4, 64, 64] vs ref_out.npy */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H0 = (int)shape[2], W0 = (int)shape[3];
        if (IC != 12) { fprintf(stderr, "ERROR: IC!=12\n"); return 1; }
        snprintf(path, sizeof(path), "%s/ref_encoder_hidden.npy", ref_dir);
        float *text = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!text) return 1;
        int M_text = (int)shape[1], cross_dim = (int)shape[2];
        fprintf(stderr, "input [%d, %d, %d, %d]  text [%d, %d, %d]\n",
                B, IC, H0, W0, B, M_text, cross_dim);
        if (B != 1) { fprintf(stderr, "ERROR: out stage assumes B=1\n"); return 1; }

        /* time embedding */
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_temb_in, d_temb_h1, d_temb;
        cuMemAlloc(&d_temb_in, B * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1, B * 1280 * sizeof(float));
        cuMemAlloc(&d_temb,    B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, B, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_temb_h1, B * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, B, 1280, 1280);

        /* Activation buffers - pick max sizes for any tensor in the flow.
         * Largest single tensor = 320@64 = 1.31M; concat output max = 640@64
         * = 2.62M; res_in_concat max = 2560@8 = 164K; ff_gh max = 4096*2560
         * = 10.5M. Round generously. */
        size_t MAX_ACT  = (size_t)1280 * 64 * 64;          /* 5.24M */
        /* MAX_CCAT covers worst-case concat C_in (up3 first iter: 960@64 = 3.93M)
         * AND worst-case resnet scratch max(c_in,c_out)*HW (also up3.res0: 960@64). */
        size_t MAX_CCAT = (size_t)960 * 64 * 64;           /* 3.93M */
        size_t MAX_FF_GH = (size_t)320 * 64 * 64 * 2 * 4;  /* 10.5M */
        size_t MAX_FF_H  = (size_t)320 * 64 * 64 * 4;      /* 5.24M */
        size_t MAX_BNC   = (size_t)320 * 64 * 64;          /* 1.31M (token buffers, max channel*N) */
        if ((size_t)1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)1280 * 16 * 16;
        if ((size_t)640 * 32 * 32 > MAX_BNC)  MAX_BNC = (size_t)640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)1280 * M_text > MAX_BMC) MAX_BMC = (size_t)1280 * M_text;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  B * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, B * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));

        /* d_concat is sized for the largest concat (up3 first iter: 640@64). */
        CUdeviceptr d_concat;
        cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        /* conv_in 12->320 into d_a */
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        size_t in_n = (size_t)B * IC * H0 * W0;
        CUdeviceptr d_in_raw; cuMemAlloc(&d_in_raw, in_n * sizeof(float));
        cuMemcpyHtoD(d_in_raw, sample, in_n * sizeof(float));
        k_conv(&kk, ws.d_a, d_in_raw, cw, cb, IC, H0, W0, 320, 3, 3, 1);
        cuMemFree(d_in_raw);

        /* text */
        CUdeviceptr d_text; cuMemAlloc(&d_text, (size_t)B * M_text * cross_dim * sizeof(float));
        cuMemcpyHtoD(d_text, text, (size_t)B * M_text * cross_dim * sizeof(float));

        /* Skip stack: push initial (post conv_in) */
        pu_skip_stack ss = {.top = 0, .B = 1};
        skip_push_copy(&ss, ws.d_a, 320, H0, W0);

        /* Load all blocks */
        pu_down_block db[4];
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 0, 0, 0, 0, 0, 0);
        pu_mid_block mid;
        load_mid_block(st, &mid, cross_dim, 0, 0, 0, 0, 0, 0);
        pu_up_block ub[4];
        /* up_blocks per diffusers: prev_out tracks the previous up's c_out (init=last block_out=1280). */
        load_up_block(st, &ub[0], 0, /*in*/1280, /*out*/1280, /*prev*/1280,  0, 0, 1, cross_dim, 0, 0, 0, 0, 0, 0); /* UpBlock2D */
        load_up_block(st, &ub[1], 1, /*in*/ 640, /*out*/1280, /*prev*/1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ub[2], 2, /*in*/ 320, /*out*/ 640, /*prev*/1280, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ub[3], 3, /*in*/ 320, /*out*/ 320, /*prev*/ 640,  5, 1, 0, cross_dim, 0, 0, 0, 0, 0, 0);

        /* Down path. Current activation lives in ws.d_a. */
        int H = H0, W = W0;
        run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        /* Sanity: H=W=8 here, top of stack = db3.r1[1280@8], 12 entries total */
        if (ss.top != 12) {
            fprintf(stderr, "ERROR: skip stack top=%d, expected 12\n", ss.top);
            return 1;
        }

        /* Mid: in d_a, out also d_a (using d_b as scratch) */
        run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb, d_text, 0, 0, B, H, W, M_text, &ws);

        /* Up path */
        run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, B, &H, &W, M_text, &ws, &ss);

        if (ss.top != 0) {
            fprintf(stderr, "WARN: skip stack not drained, top=%d\n", ss.top);
        }

        /* conv_norm_out (32 grp, +silu) -> d_b ; conv_out 320->4 -> d_a */
        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob = upload_st(st, "conv_out.bias");
        k_groupnorm(&kk, ws.d_b, ws.d_a, ng, nb, 320, H * W, 32, 1);
        k_conv(&kk, ws.d_a, ws.d_b, ow, ob, 320, H, W, 4, 3, 3, 1);
        cuCtxSynchronize();

        size_t out_n = (size_t)B * 4 * H * W;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, ws.d_a, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_out.npy", ref_dir);
        diff_against(cu, path, out_n, 1e-2f);
        free(cu); free(sample); free(text);
    } else if (!strcmp(stage, "out_ma") || !strcmp(stage, "out_ma_rope")
                || !strcmp(stage, "out_mda")) {
        /* Wrapper-style forward with one custom attn path enabled at a time:
         *   out_ma       MA only,  no rope (Phase 4.3a)
         *   out_ma_rope  MA only,  with PoseRoPE (Phase 4.3b)
         *   out_mda      MDA only, per-material self-attn (Phase 4.4) */
        const int with_rope = !strcmp(stage, "out_ma_rope");
        const int with_mda  = !strcmp(stage, "out_mda");
        const int with_ma   = !with_mda;
        const char *out_npy = with_mda ? "out_mda.npy" : "out_ma.npy";
        const int N_PBR = 2, N_GEN = 2;
        const int Beff = N_PBR * N_GEN;
        const int IC = 12;
        const int H0 = 64, W0 = 64;
        const int HEAD_DIM = 64;     /* SD-2.1 head_dim is 64 at every level */

        if (with_rope) {
            /* Per-level voxel res from the dump: [512,256,128,64] for
             * grid [H,H/2,H/4,H/8] with H=64. voxel_indices_<key>.npy
             * key = n_gen * (H*W). */
            int Nps[4]   = { N_GEN * 64*64, N_GEN * 32*32, N_GEN * 16*16, N_GEN * 8*8 };
            int vres[4]  = { 512, 256, 128, 64 };
            for (int i = 0; i < 4; i++) {
                char vp[512]; snprintf(vp, sizeof(vp), "%s/voxel_indices_%d.npy", ref_dir, Nps[i]);
                int nd2; uint64_t sh2[8]; size_t n2; char dt2[8];
                int64_t *vox = (int64_t *)read_npy(vp, &nd2, sh2, &n2, dt2);
                if (!vox) { fprintf(stderr, "ERROR: missing %s\n", vp); return 1; }
                if (build_rope_level_from_voxels(vox, Nps[i], N_PBR, N_GEN,
                                                  HEAD_DIM, vres[i]) < 0) return 1;
                fprintf(stderr, "  rope L Np=%d res=%d N=%d\n",
                         Nps[i], vres[i], Nps[i] / N_GEN);
                free(vox);
            }
        }

        snprintf(path, sizeof(path), "%s/in_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample || n != (size_t)Beff * 4 * H0 * W0) {
            fprintf(stderr, "ERROR: in_sample shape mismatch\n"); return 1;
        }
        snprintf(path, sizeof(path), "%s/in_embeds_normal.npy", ref_dir);
        float *en = (float *)read_npy(path, &nd, shape, &n, dt);  if (!en) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_position.npy", ref_dir);
        float *ep = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ep) return 1;
        snprintf(path, sizeof(path), "%s/in_encoder_hidden_states.npy", ref_dir);
        float *text_in = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!text_in) return 1;
        int M_text = (int)shape[2], cross_dim = (int)shape[3];
        snprintf(path, sizeof(path), "%s/in_timestep.npy", ref_dir);
        int64_t *ts_in = (int64_t *)read_npy(path, &nd, shape, &n, dt);
        if (!ts_in) return 1;
        long long ts_val = ts_in[0];
        fprintf(stderr, "%s: Beff=%d, ts=%lld, M_text=%d, cross=%d\n",
                stage, Beff, ts_val, M_text, cross_dim);

        size_t per_view = (size_t)4 * H0 * W0;
        size_t per_in = (size_t)IC * H0 * W0;
        float *packed = (float *)malloc((size_t)Beff * per_in * sizeof(float));
        for (int p = 0; p < N_PBR; p++) {
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                float *dst = packed + (size_t)b * per_in;
                memcpy(dst,                sample + ((size_t)p * N_GEN + g) * per_view, per_view * sizeof(float));
                memcpy(dst + per_view,     en + (size_t)g * per_view, per_view * sizeof(float));
                memcpy(dst + 2 * per_view, ep + (size_t)g * per_view, per_view * sizeof(float));
            }
        }

        size_t txt_per = (size_t)M_text * cross_dim;
        float *text_tiled = (float *)malloc((size_t)Beff * txt_per * sizeof(float));
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                memcpy(text_tiled + (size_t)b * txt_per,
                       text_in + (size_t)p * txt_per, txt_per * sizeof(float));
            }

        int64_t ts_arr[16];
        for (int b = 0; b < Beff; b++) ts_arr[b] = ts_val;
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, Beff * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts_arr, Beff * sizeof(int64_t));
        CUdeviceptr d_temb_in, d_temb_h1, d_temb;
        cuMemAlloc(&d_temb_in, Beff * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1, Beff * 1280 * sizeof(float));
        cuMemAlloc(&d_temb,    Beff * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, Beff, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, Beff, 320, 1280);
        k_silu(&kk, d_temb_h1, Beff * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, Beff, 1280, 1280);

        size_t in_n  = (size_t)Beff * IC * H0 * W0;
        CUdeviceptr d_in_raw; cuMemAlloc(&d_in_raw, in_n * sizeof(float));
        cuMemcpyHtoD(d_in_raw, packed, in_n * sizeof(float));
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");

        size_t MAX_ACT = (size_t)Beff * 1280 * H0 * W0;
        size_t MAX_CCAT = (size_t)Beff * 960 * H0 * W0;
        size_t MAX_FF_GH = (size_t)Beff * 320 * H0 * W0 * 2 * 4;
        size_t MAX_FF_H  = (size_t)Beff * 320 * H0 * W0 * 4;
        size_t MAX_BNC   = (size_t)Beff * 320 * H0 * W0;
        if ((size_t)Beff * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)Beff * 1280 * 16 * 16;
        if ((size_t)Beff * 640 * 32 * 32  > MAX_BNC) MAX_BNC = (size_t)Beff * 640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)Beff * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)Beff * 1280 * M_text;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  Beff * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, Beff * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));

        CUdeviceptr d_concat; cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        for (int b = 0; b < Beff; b++) {
            CUdeviceptr ib = d_in_raw + (CUdeviceptr)b * IC * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a   + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw, cb, IC, H0, W0, 320, 3, 3, 1);
        }

        CUdeviceptr d_text; cuMemAlloc(&d_text, (size_t)Beff * txt_per * sizeof(float));
        cuMemcpyHtoD(d_text, text_tiled, (size_t)Beff * txt_per * sizeof(float));

        pu_skip_stack ss = {.top = 0, .B = Beff};
        skip_push_copy(&ss, ws.d_a, 320, H0, W0);

        /* Load all blocks with the selected custom attn path, n_pbr=2 n_gen=2,
         * no DINO/RA. has_ma + has_mda are mutually exclusive here. */
        pu_down_block db[4];
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        pu_mid_block mid;
        load_mid_block(st, &mid, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        pu_up_block ub[4];
        load_up_block(st, &ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        load_up_block(st, &ub[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        load_up_block(st, &ub[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);
        load_up_block(st, &ub[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 0, with_ma, with_mda, 0, N_PBR, N_GEN);

        int H = H0, W = W0;
        run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb, d_text, 0, 0, Beff, H, W, M_text, &ws);
        run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb, d_text, 0, 0, Beff, &H, &W, M_text, &ws, &ss);

        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb_w = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob_w = upload_st(st, "conv_out.bias");
        for (int b = 0; b < Beff; b++) {
            CUdeviceptr xb = ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_groupnorm(&kk, yb, xb, ng, nb_w, 320, H * W, 32, 1);
        }
        for (int b = 0; b < Beff; b++) {
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr ob = ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
            k_conv(&kk, ob, yb, ow, ob_w, 320, H, W, 4, 3, 3, 1);
        }
        cuCtxSynchronize();

        size_t out_n = (size_t)Beff * 4 * H * W;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, ws.d_a, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/%s", ref_dir, out_npy);
        diff_against(cu, path, out_n, 5e-3f);
        free(cu); free(packed); free(text_tiled);
        free(sample); free(en); free(ep); free(text_in); free(ts_in);
    } else if (!strcmp(stage, "out_dino")) {
        /* Full UNet forward with DINO cross-attn enabled. Wrapper-style
         * inputs: sample [1, N_pbr=2, N_gen=2, 4, 64, 64] (cat'd with
         * normal+position embed for 12 ch in), text [1, N_pbr, 77, 1024]
         * (tiled per N_gen), DINO [1, 257, 1536] (projected once + tiled).
         * Effective batch B=N_pbr*N_gen=4. Output [4, 4, 64, 64] vs
         * /tmp/.../out_dino.npy. */
        const int N_PBR = 2, N_GEN = 2;
        const int Beff = N_PBR * N_GEN;        /* 4 */
        const int M_DINO = 1028;               /* 257 * 4 */
        const int IC = 12;
        const int H0 = 64, W0 = 64;

        /* --- load wrapper inputs --- */
        snprintf(path, sizeof(path), "%s/in_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample || n != (size_t)Beff * 4 * H0 * W0) {
            fprintf(stderr, "ERROR: in_sample shape mismatch\n"); return 1;
        }
        snprintf(path, sizeof(path), "%s/in_embeds_normal.npy", ref_dir);
        float *en = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!en) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_position.npy", ref_dir);
        float *ep = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!ep) return 1;
        snprintf(path, sizeof(path), "%s/in_encoder_hidden_states.npy", ref_dir);
        float *text_in = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!text_in) return 1;                    /* [1, N_pbr, 77, 1024] */
        int M_text = (int)shape[2], cross_dim = (int)shape[3];
        snprintf(path, sizeof(path), "%s/in_dino_hidden_states.npy", ref_dir);
        float *dino_in = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!dino_in) return 1;                    /* [1, 257, 1536] */
        int T_dino = (int)shape[1], C_dino_in = (int)shape[2];
        snprintf(path, sizeof(path), "%s/in_timestep.npy", ref_dir);
        int64_t *ts_in = (int64_t *)read_npy(path, &nd, shape, &n, dt);
        if (!ts_in) return 1;
        long long ts_val = ts_in[0];
        fprintf(stderr, "out_dino: Beff=%d, ts=%lld, M_text=%d, cross=%d, T_dino=%d, C_dino_in=%d\n",
                Beff, ts_val, M_text, cross_dim, T_dino, C_dino_in);

        /* Pack 12-ch input on host, then upload. sample is [N_pbr, N_gen, 4, 64, 64];
         * en/ep are [N_gen, 4, 64, 64] each (B=1 dropped). Wrapper:
         *   sample.append(embeds_normal.unsqueeze(1).repeat(1, N_pbr, ...))
         *   -> for each (n_pbr, n_gen): cat(sample[n_pbr,n_gen], en[n_gen], ep[n_gen]) on chan
         *   -> rearrange "b n_pbr n c h w -> (b n_pbr n) c h w"
         */
        size_t per_view = (size_t)4 * H0 * W0;
        size_t per_in = (size_t)IC * H0 * W0;
        float *packed = (float *)malloc((size_t)Beff * per_in * sizeof(float));
        for (int p = 0; p < N_PBR; p++) {
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                float *dst = packed + (size_t)b * per_in;
                memcpy(dst,                  sample + ((size_t)p * N_GEN + g) * per_view, per_view * sizeof(float));
                memcpy(dst + per_view,       en + (size_t)g * per_view, per_view * sizeof(float));
                memcpy(dst + 2 * per_view,   ep + (size_t)g * per_view, per_view * sizeof(float));
            }
        }

        /* Tile text [1, N_pbr, 77, 1024] by N_gen -> [Beff, 77, 1024]. */
        size_t txt_per = (size_t)M_text * cross_dim;
        float *text_tiled = (float *)malloc((size_t)Beff * txt_per * sizeof(float));
        for (int p = 0; p < N_PBR; p++) {
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                memcpy(text_tiled + (size_t)b * txt_per,
                       text_in + (size_t)p * txt_per,
                       txt_per * sizeof(float));
            }
        }

        /* --- time embedding for B=Beff (broadcast scalar timestep) --- */
        int64_t ts_arr[16];
        for (int b = 0; b < Beff; b++) ts_arr[b] = ts_val;
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, Beff * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts_arr, Beff * sizeof(int64_t));
        CUdeviceptr d_temb_in, d_temb_h1, d_temb;
        cuMemAlloc(&d_temb_in, Beff * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1, Beff * 1280 * sizeof(float));
        cuMemAlloc(&d_temb,    Beff * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, Beff, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, Beff, 320, 1280);
        k_silu(&kk, d_temb_h1, Beff * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, Beff, 1280, 1280);

        /* --- DINO projection: [1, 257, 1536] -> [1, 1028, 1024] then
         * tile to [Beff, 1028, 1024]. */
        const int CTX = 1024, EXTRA = 4;
        int rows = T_dino;
        int rows_out = rows * EXTRA;
        if (rows_out != M_DINO) {
            fprintf(stderr, "ERROR: DINO rows_out=%d != %d\n", rows_out, M_DINO); return 1;
        }
        CUdeviceptr d_dino_in;  cuMemAlloc(&d_dino_in,  (size_t)rows * C_dino_in * sizeof(float));
        CUdeviceptr d_dino_lin; cuMemAlloc(&d_dino_lin, (size_t)rows * EXTRA * CTX * sizeof(float));
        CUdeviceptr d_dino_one; cuMemAlloc(&d_dino_one, (size_t)rows_out * CTX * sizeof(float));
        cuMemcpyHtoD(d_dino_in, dino_in, (size_t)rows * C_dino_in * sizeof(float));
        CUdeviceptr pw = upload_st(st, "image_proj_model_dino.proj.weight");
        CUdeviceptr pb = upload_st(st, "image_proj_model_dino.proj.bias");
        CUdeviceptr png = upload_st(st, "image_proj_model_dino.norm.weight");
        CUdeviceptr pnb = upload_st(st, "image_proj_model_dino.norm.bias");
        k_linear(&kk, d_dino_lin, d_dino_in, pw, pb, rows, C_dino_in, EXTRA * CTX);
        k_layernorm(&kk, d_dino_one, d_dino_lin, png, pnb, rows_out, CTX);
        /* Tile [1, 1028, 1024] -> [Beff, 1028, 1024] on device. */
        CUdeviceptr d_dino; cuMemAlloc(&d_dino, (size_t)Beff * rows_out * CTX * sizeof(float));
        size_t dino_per = (size_t)rows_out * CTX * sizeof(float);
        for (int b = 0; b < Beff; b++) {
            cuMemcpyDtoD(d_dino + (CUdeviceptr)b * dino_per, d_dino_one, dino_per);
        }

        /* --- conv_in 12 -> 320 (B=Beff, per-sample) --- */
        size_t in_n  = (size_t)Beff * IC * H0 * W0;
        CUdeviceptr d_in_raw; cuMemAlloc(&d_in_raw, in_n * sizeof(float));
        cuMemcpyHtoD(d_in_raw, packed, in_n * sizeof(float));
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");

        /* --- Workspace sizing for B=Beff. Largest tensors:
         *   ACT  : Beff * 1280 * 8 * 8  = 0.33M (mid)
         *          Beff *  320 * 64*64  = 5.24M (top level) -- this is max
         *   CCAT : up3 first iter 960 * 64 * 64 * Beff = 15.7M
         *   FF_GH: Beff * (320*64*64) * 2*1280 = 84M floats? wait recompute:
         *          ff_gh per row = 2*ff_inner; max N=4096; ff_inner=1280;
         *          -> Beff * 4096 * 2*1280 = 168M floats = 672MB. Too large.
         *
         * Worst-case at top level (320ch @ 64x64):
         *   N=4096, C=320, ff_inner=1280
         *   ff_gh = Beff*N*2*ff_inner = 4*4096*2560 = 41.9M floats = 168MB
         *   ff_h  = Beff*N*ff_inner   = 4*4096*1280 = 20.9M floats = 84MB
         * Acceptable. */
        size_t MAX_ACT = (size_t)Beff * 1280 * H0 * W0;       /* 20.9M oversized */
        size_t MAX_CCAT = (size_t)Beff * 960 * H0 * W0;       /* 15.7M */
        size_t MAX_FF_GH = (size_t)Beff * 320 * H0 * W0 * 2 * 4;
        size_t MAX_FF_H  = (size_t)Beff * 320 * H0 * W0 * 4;
        size_t MAX_BNC   = (size_t)Beff * 320 * H0 * W0;
        if ((size_t)Beff * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)Beff * 1280 * 16 * 16;
        if ((size_t)Beff * 640 * 32 * 32  > MAX_BNC) MAX_BNC = (size_t)Beff * 640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)Beff * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)Beff * 1280 * M_text;
        if ((size_t)Beff * 1280 * M_DINO > MAX_BMC) MAX_BMC = (size_t)Beff * 1280 * M_DINO;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  Beff * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, Beff * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));

        CUdeviceptr d_concat; cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        /* conv_in 12->320 per batch */
        for (int b = 0; b < Beff; b++) {
            CUdeviceptr ib = d_in_raw + (CUdeviceptr)b * IC * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a   + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw, cb, IC, H0, W0, 320, 3, 3, 1);
        }

        /* upload tiled text */
        CUdeviceptr d_text; cuMemAlloc(&d_text, (size_t)Beff * txt_per * sizeof(float));
        cuMemcpyHtoD(d_text, text_tiled, (size_t)Beff * txt_per * sizeof(float));

        /* skip stack: push initial conv_in result (Beff samples) */
        pu_skip_stack ss = {.top = 0, .B = Beff};
        skip_push_copy(&ss, ws.d_a, 320, H0, W0);

        /* Load all blocks WITH attn_dino weights */
        pu_down_block db[4];
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim, 1, 0, 0, 0, 0, 0);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim, 1, 0, 0, 0, 0, 0);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim, 1, 0, 0, 0, 0, 0);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 1, 0, 0, 0, 0, 0);
        pu_mid_block mid;
        load_mid_block(st, &mid, cross_dim, 1, 0, 0, 0, 0, 0);
        pu_up_block ub[4];
        load_up_block(st, &ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 1, 0, 0, 0, 0, 0);
        load_up_block(st, &ub[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 1, 0, 0, 0, 0, 0);
        load_up_block(st, &ub[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 1, 0, 0, 0, 0, 0);
        load_up_block(st, &ub[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 1, 0, 0, 0, 0, 0);

        int H = H0, W = W0;
        run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        if (ss.top != 12) {
            fprintf(stderr, "ERROR: skip stack top=%d, expected 12\n", ss.top); return 1;
        }
        run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb, d_text, d_dino, M_DINO, Beff, H, W, M_text, &ws);
        run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb, d_text, d_dino, M_DINO, Beff, &H, &W, M_text, &ws, &ss);

        /* conv_norm_out + conv_out (per batch) */
        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb_w = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob_w = upload_st(st, "conv_out.bias");
        for (int b = 0; b < Beff; b++) {
            CUdeviceptr xb = ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_groupnorm(&kk, yb, xb, ng, nb_w, 320, H * W, 32, 1);
        }
        for (int b = 0; b < Beff; b++) {
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr ob = ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
            k_conv(&kk, ob, yb, ow, ob_w, 320, H, W, 4, 3, 3, 1);
        }
        cuCtxSynchronize();

        size_t out_n = (size_t)Beff * 4 * H * W;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, ws.d_a, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/out_dino.npy", ref_dir);
        diff_against(cu, path, out_n, 5e-3f);
        free(cu); free(packed); free(text_tiled);
        free(sample); free(en); free(ep); free(text_in); free(dino_in); free(ts_in);
    } else if (!strcmp(stage, "out_ra")) {
        /* Phase 4.5: RA + dual-stream.
         *  1. Run unet_dual on ref_latents [B*N_ref, 4, H, W] with ra_mode='w'
         *     to populate g_ra_cache (one slot per transformer block).
         *  2. Run main unet on packed [B*N_pbr*N_gen, 12, H, W] with ra_mode='r';
         *     each block consumes the cached norm_hidden_states for its layer
         *     and applies attn_refview (shared Q/K + per-material V/out).
         * Validate against /tmp/.../out_ra.npy [4, 4, 64, 64]. */
        const int N_PBR = 2, N_GEN = 2, N_REF = 1;
        const int B_outer = 1;                 /* B in the wrapper-input */
        const int Beff_main = B_outer * N_PBR * N_GEN;       /* 4 */
        const int Beff_dual = B_outer * N_REF;               /* 1 */
        const int H0 = 64, W0 = 64;
        const int IC_main = 12, IC_dual = 4;
        const int N_BLOCKS = 16;     /* 6 down + 1 mid + 9 up */

        /* Allocate cache slots; idx walks 0..N_BLOCKS in deterministic order. */
        g_ra_cache.slots = (pu_ra_slot *)calloc(N_BLOCKS, sizeof(pu_ra_slot));
        g_ra_cache.n_slots = N_BLOCKS;
        g_ra_cache.idx = 0;
        g_ra_n_ref = N_REF;

        /* Load shared inputs */
        snprintf(path, sizeof(path), "%s/in_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);  if (!sample) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_normal.npy", ref_dir);
        float *en = (float *)read_npy(path, &nd, shape, &n, dt);  if (!en) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_position.npy", ref_dir);
        float *ep = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ep) return 1;
        snprintf(path, sizeof(path), "%s/in_encoder_hidden_states.npy", ref_dir);
        float *text_in = (float *)read_npy(path, &nd, shape, &n, dt);  if (!text_in) return 1;
        int M_text = (int)shape[2], cross_dim = (int)shape[3];
        snprintf(path, sizeof(path), "%s/in_ref_latents.npy", ref_dir);
        float *ref_latents = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ref_latents) return 1;
        snprintf(path, sizeof(path), "%s/in_timestep.npy", ref_dir);
        int64_t *ts_in = (int64_t *)read_npy(path, &nd, shape, &n, dt);  if (!ts_in) return 1;
        long long ts_val = ts_in[0];
        fprintf(stderr, "out_ra: Beff_main=%d Beff_dual=%d ts=%lld M_text=%d cross=%d\n",
                Beff_main, Beff_dual, ts_val, M_text, cross_dim);

        /* Pack main UNet input [Beff_main, 12, H, W] = sample | normal | position
         * with order (b=0..Beff_main) <- (p in N_PBR) (g in N_GEN). */
        size_t per_view = (size_t)4 * H0 * W0;
        size_t per_in_main = (size_t)IC_main * H0 * W0;
        float *packed_main = (float *)malloc((size_t)Beff_main * per_in_main * sizeof(float));
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                float *dst = packed_main + (size_t)b * per_in_main;
                memcpy(dst,                sample + ((size_t)p * N_GEN + g) * per_view, per_view * sizeof(float));
                memcpy(dst + per_view,     en + (size_t)g * per_view, per_view * sizeof(float));
                memcpy(dst + 2 * per_view, ep + (size_t)g * per_view, per_view * sizeof(float));
            }
        /* Dual UNet input: ref_latents flattened [B*N_ref, 4, H, W]. */
        float *packed_dual = (float *)malloc((size_t)Beff_dual * per_view * sizeof(float));
        memcpy(packed_dual, ref_latents, (size_t)Beff_dual * per_view * sizeof(float));

        /* Tile text per material -> [Beff_main, 77, cross_dim] for main UNet. */
        size_t txt_per = (size_t)M_text * cross_dim;
        float *text_tiled_main = (float *)malloc((size_t)Beff_main * txt_per * sizeof(float));
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                memcpy(text_tiled_main + (size_t)b * txt_per,
                       text_in + (size_t)p * txt_per, txt_per * sizeof(float));
            }
        /* Dual UNet text: learned_text_clip_ref [1, 77, 1024] from main UNet's
         * weights, repeated B*N_ref times. */
        CUdeviceptr d_text_clip_ref = upload_st(st, "learned_text_clip_ref");
        size_t ltcr_n = (size_t)1 * M_text * cross_dim;
        CUdeviceptr d_text_dual; cuMemAlloc(&d_text_dual, (size_t)Beff_dual * txt_per * sizeof(float));
        for (int b = 0; b < Beff_dual; b++)
            cuMemcpyDtoD(d_text_dual + (CUdeviceptr)b * txt_per * sizeof(float),
                          d_text_clip_ref, ltcr_n * sizeof(float));

        /* Time embedding (main and dual share weights but use different ts;
         * dual uses ts=0). */
        int64_t ts_main_arr[16], ts_dual_arr[16];
        for (int b = 0; b < Beff_main; b++) ts_main_arr[b] = ts_val;
        for (int b = 0; b < Beff_dual; b++) ts_dual_arr[b] = 0;
        CUdeviceptr d_ts_main, d_ts_dual;
        cuMemAlloc(&d_ts_main, Beff_main * sizeof(int64_t));
        cuMemAlloc(&d_ts_dual, Beff_dual * sizeof(int64_t));
        cuMemcpyHtoD(d_ts_main, ts_main_arr, Beff_main * sizeof(int64_t));
        cuMemcpyHtoD(d_ts_dual, ts_dual_arr, Beff_dual * sizeof(int64_t));

        /* time_embedding weights (shared across main/dual since unet_dual was
         * deepcopy'd before any modification).
         * Main has bias l1_w/b/l2_w/b under stock prefix. */
        CUdeviceptr l1_w  = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b  = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w  = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b  = upload_st(st, "time_embedding.linear_2.bias");
        CUdeviceptr l1_wd = upload_st(st, "unet_dual.time_embedding.linear_1.weight");
        CUdeviceptr l1_bd = upload_st(st, "unet_dual.time_embedding.linear_1.bias");
        CUdeviceptr l2_wd = upload_st(st, "unet_dual.time_embedding.linear_2.weight");
        CUdeviceptr l2_bd = upload_st(st, "unet_dual.time_embedding.linear_2.bias");

        /* Allocate one shared workspace. Main is the larger pass (Beff=4) so
         * sizes from out_ma branch suffice. */
        size_t MAX_ACT  = (size_t)Beff_main * 1280 * H0 * W0;
        size_t MAX_CCAT = (size_t)Beff_main * 960  * H0 * W0;
        size_t MAX_FF_GH= (size_t)Beff_main * 320  * H0 * W0 * 2 * 4;
        size_t MAX_FF_H = (size_t)Beff_main * 320  * H0 * W0 * 4;
        size_t MAX_BNC  = (size_t)Beff_main * 320  * H0 * W0;
        if ((size_t)Beff_main * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)Beff_main * 1280 * 16 * 16;
        if ((size_t)Beff_main *  640 * 32 * 32 > MAX_BNC) MAX_BNC = (size_t)Beff_main *  640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)Beff_main * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)Beff_main * 1280 * M_text;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));
        CUdeviceptr d_concat; cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        /* ===== Pass 1: dual-stream forward, ra_mode='w' ====================== */
        g_ra_mode = 1; g_ra_cache.idx = 0;

        /* Load dual UNet weights (g_load_wp = "unet_dual."). conv_in is
         * 4-channel; everything else is the stock SD-2.1 layout. has_ra=0
         * for dual blocks (dual just runs vanilla UNet to populate cache). */
        g_load_wp = "unet_dual.";
        CUdeviceptr cw_d = upload_st(st, "unet_dual.conv_in.weight");
        CUdeviceptr cb_d = upload_st(st, "unet_dual.conv_in.bias");
        pu_down_block dbd[4]; pu_mid_block midd; pu_up_block ubd[4];
        load_down_block(st, &dbd[0], 0,  320,  320,  5, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[1], 1,  320,  640, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[2], 2,  640, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 0, 0, 0, 0, 0, 0);
        load_mid_block(st, &midd, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 0, 0, 0, 0, 0, 0);

        /* Time emb for dual */
        CUdeviceptr d_temb_in_d, d_temb_h1_d, d_temb_d;
        cuMemAlloc(&d_temb_in_d, Beff_dual * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1_d, Beff_dual * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_d,    Beff_dual * 1280 * sizeof(float));
        k_timestep_embed(&kk, d_temb_in_d, d_ts_dual, Beff_dual, 320);
        k_linear(&kk, d_temb_h1_d, d_temb_in_d, l1_wd, l1_bd, Beff_dual, 320, 1280);
        k_silu(&kk, d_temb_h1_d, Beff_dual * 1280);
        k_linear(&kk, d_temb_d, d_temb_h1_d, l2_wd, l2_bd, Beff_dual, 1280, 1280);

        /* conv_in: 4 -> 320 */
        size_t in_n_d = (size_t)Beff_dual * IC_dual * H0 * W0;
        CUdeviceptr d_in_raw_d; cuMemAlloc(&d_in_raw_d, in_n_d * sizeof(float));
        cuMemcpyHtoD(d_in_raw_d, packed_dual, in_n_d * sizeof(float));
        for (int b = 0; b < Beff_dual; b++) {
            CUdeviceptr ib = d_in_raw_d + (CUdeviceptr)b * IC_dual * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a     + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw_d, cb_d, IC_dual, H0, W0, 320, 3, 3, 1);
        }
        pu_skip_stack ssd = {.top = 0, .B = Beff_dual};
        skip_push_copy(&ssd, ws.d_a, 320, H0, W0);
        int H = H0, W = W0;
        run_down_block(&kk, &dbd[0], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[1], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[2], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[3], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_mid_block(&kk, &midd, ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, H, W, M_text, &ws);
        run_up_block(&kk, &ubd[0], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[1], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[2], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[3], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        cuCtxSynchronize();
        fprintf(stderr, "  RA write pass: cached %d transformer-block slots\n", g_ra_cache.idx);

        /* ===== Pass 2: main forward, ra_mode='r' =========================== */
        g_ra_mode = 2; g_ra_cache.idx = 0;
        g_load_wp = "";
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        pu_down_block db[4]; pu_mid_block mid; pu_up_block ub[4];
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_mid_block(st, &mid, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 0, 0, 0, 1, N_PBR, N_GEN);

        CUdeviceptr d_temb_in_m, d_temb_h1_m, d_temb_m;
        cuMemAlloc(&d_temb_in_m, Beff_main * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1_m, Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_m,    Beff_main * 1280 * sizeof(float));
        k_timestep_embed(&kk, d_temb_in_m, d_ts_main, Beff_main, 320);
        k_linear(&kk, d_temb_h1_m, d_temb_in_m, l1_w, l1_b, Beff_main, 320, 1280);
        k_silu(&kk, d_temb_h1_m, Beff_main * 1280);
        k_linear(&kk, d_temb_m, d_temb_h1_m, l2_w, l2_b, Beff_main, 1280, 1280);

        size_t in_n_m = (size_t)Beff_main * IC_main * H0 * W0;
        CUdeviceptr d_in_raw_m; cuMemAlloc(&d_in_raw_m, in_n_m * sizeof(float));
        cuMemcpyHtoD(d_in_raw_m, packed_main, in_n_m * sizeof(float));
        CUdeviceptr d_text_m; cuMemAlloc(&d_text_m, (size_t)Beff_main * txt_per * sizeof(float));
        cuMemcpyHtoD(d_text_m, text_tiled_main, (size_t)Beff_main * txt_per * sizeof(float));
        for (int b = 0; b < Beff_main; b++) {
            CUdeviceptr ib = d_in_raw_m + (CUdeviceptr)b * IC_main * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a     + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw, cb, IC_main, H0, W0, 320, 3, 3, 1);
        }
        pu_skip_stack ss = {.top = 0, .B = Beff_main};
        skip_push_copy(&ss, ws.d_a, 320, H0, W0);
        H = H0; W = W0;
        run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb_m, d_text_m, 0, 0, Beff_main, H, W, M_text, &ws);
        run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, 0, 0, Beff_main, &H, &W, M_text, &ws, &ss);

        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb_w = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob_w = upload_st(st, "conv_out.bias");
        for (int b = 0; b < Beff_main; b++) {
            CUdeviceptr xb = ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_groupnorm(&kk, yb, xb, ng, nb_w, 320, H * W, 32, 1);
        }
        for (int b = 0; b < Beff_main; b++) {
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr ob = ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
            k_conv(&kk, ob, yb, ow, ob_w, 320, H, W, 4, 3, 3, 1);
        }
        cuCtxSynchronize();
        size_t out_n = (size_t)Beff_main * 4 * H * W;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, ws.d_a, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/out_ra.npy", ref_dir);
        diff_against(cu, path, out_n, 5e-3f);
        free(cu); free(packed_main); free(packed_dual); free(text_tiled_main);
        free(sample); free(en); free(ep); free(text_in); free(ref_latents); free(ts_in);
        g_ra_mode = 0;
    } else if (!strcmp(stage, "out_all") || !strcmp(stage, "out_all_rope")) {
        /* Phase 4.6/4.7: all 4 attention paths on (DINO + MA + MDA + RA) +
         * dual-stream. PoseRoPE on for out_all_rope. */
        const int with_rope = !strcmp(stage, "out_all_rope");
        const char *out_npy = with_rope ? "out_all_rope.npy" : "out_all.npy";
        const int HEAD_DIM = 64;
        const int N_PBR = 2, N_GEN = 2, N_REF = 1;
        const int B_outer = 1;
        const int Beff_main = B_outer * N_PBR * N_GEN;       /* 4 */
        const int Beff_dual = B_outer * N_REF;               /* 1 */
        const int H0 = 64, W0 = 64;
        const int IC_main = 12, IC_dual = 4;
        const int N_BLOCKS = 16;
        const int M_DINO = 1028;       /* 257 * 4 */

        g_ra_cache.slots = (pu_ra_slot *)calloc(N_BLOCKS, sizeof(pu_ra_slot));
        g_ra_cache.n_slots = N_BLOCKS;
        g_ra_cache.idx = 0;
        g_ra_n_ref = N_REF;

        if (with_rope) {
            /* Per-level voxel res from the dump: [512,256,128,64] for
             * grid [H,H/2,H/4,H/8] with H=64. voxel_indices_<key>.npy
             * key = n_gen * (H*W). */
            int Nps[4]   = { N_GEN * 64*64, N_GEN * 32*32, N_GEN * 16*16, N_GEN * 8*8 };
            int vres[4]  = { 512, 256, 128, 64 };
            for (int i = 0; i < 4; i++) {
                char vp[512]; snprintf(vp, sizeof(vp), "%s/voxel_indices_%d.npy", ref_dir, Nps[i]);
                int nd2; uint64_t sh2[8]; size_t n2; char dt2[8];
                int64_t *vox = (int64_t *)read_npy(vp, &nd2, sh2, &n2, dt2);
                if (!vox) { fprintf(stderr, "ERROR: missing %s\n", vp); return 1; }
                if (build_rope_level_from_voxels(vox, Nps[i], N_PBR, N_GEN,
                                                  HEAD_DIM, vres[i]) < 0) return 1;
                fprintf(stderr, "  rope L Np=%d res=%d N=%d\n",
                         Nps[i], vres[i], Nps[i] / N_GEN);
                free(vox);
            }
        }

        /* --- shared inputs --- */
        snprintf(path, sizeof(path), "%s/in_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);  if (!sample) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_normal.npy", ref_dir);
        float *en = (float *)read_npy(path, &nd, shape, &n, dt);  if (!en) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_position.npy", ref_dir);
        float *ep = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ep) return 1;
        snprintf(path, sizeof(path), "%s/in_encoder_hidden_states.npy", ref_dir);
        float *text_in = (float *)read_npy(path, &nd, shape, &n, dt);  if (!text_in) return 1;
        int M_text = (int)shape[2], cross_dim = (int)shape[3];
        snprintf(path, sizeof(path), "%s/in_ref_latents.npy", ref_dir);
        float *ref_latents = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ref_latents) return 1;
        snprintf(path, sizeof(path), "%s/in_dino_hidden_states.npy", ref_dir);
        float *dino_in = (float *)read_npy(path, &nd, shape, &n, dt);  if (!dino_in) return 1;
        int T_dino = (int)shape[1], C_dino_in = (int)shape[2];
        snprintf(path, sizeof(path), "%s/in_timestep.npy", ref_dir);
        int64_t *ts_in = (int64_t *)read_npy(path, &nd, shape, &n, dt);  if (!ts_in) return 1;
        long long ts_val = ts_in[0];
        fprintf(stderr, "%s: Beff_main=%d Beff_dual=%d ts=%lld M_text=%d cross=%d T_dino=%d rope=%d\n",
                stage, Beff_main, Beff_dual, ts_val, M_text, cross_dim, T_dino, with_rope);

        size_t per_view = (size_t)4 * H0 * W0;
        size_t per_in_main = (size_t)IC_main * H0 * W0;
        float *packed_main = (float *)malloc((size_t)Beff_main * per_in_main * sizeof(float));
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                float *dst = packed_main + (size_t)b * per_in_main;
                memcpy(dst,                sample + ((size_t)p * N_GEN + g) * per_view, per_view * sizeof(float));
                memcpy(dst + per_view,     en + (size_t)g * per_view, per_view * sizeof(float));
                memcpy(dst + 2 * per_view, ep + (size_t)g * per_view, per_view * sizeof(float));
            }
        float *packed_dual = (float *)malloc((size_t)Beff_dual * per_view * sizeof(float));
        memcpy(packed_dual, ref_latents, (size_t)Beff_dual * per_view * sizeof(float));

        size_t txt_per = (size_t)M_text * cross_dim;
        float *text_tiled_main = (float *)malloc((size_t)Beff_main * txt_per * sizeof(float));
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                memcpy(text_tiled_main + (size_t)b * txt_per,
                       text_in + (size_t)p * txt_per, txt_per * sizeof(float));
            }
        CUdeviceptr d_text_clip_ref = upload_st(st, "learned_text_clip_ref");
        size_t ltcr_n = (size_t)1 * M_text * cross_dim;
        CUdeviceptr d_text_dual; cuMemAlloc(&d_text_dual, (size_t)Beff_dual * txt_per * sizeof(float));
        for (int b = 0; b < Beff_dual; b++)
            cuMemcpyDtoD(d_text_dual + (CUdeviceptr)b * txt_per * sizeof(float),
                          d_text_clip_ref, ltcr_n * sizeof(float));

        int64_t ts_main_arr[16], ts_dual_arr[16];
        for (int b = 0; b < Beff_main; b++) ts_main_arr[b] = ts_val;
        for (int b = 0; b < Beff_dual; b++) ts_dual_arr[b] = 0;
        CUdeviceptr d_ts_main, d_ts_dual;
        cuMemAlloc(&d_ts_main, Beff_main * sizeof(int64_t));
        cuMemAlloc(&d_ts_dual, Beff_dual * sizeof(int64_t));
        cuMemcpyHtoD(d_ts_main, ts_main_arr, Beff_main * sizeof(int64_t));
        cuMemcpyHtoD(d_ts_dual, ts_dual_arr, Beff_dual * sizeof(int64_t));

        CUdeviceptr l1_w  = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b  = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w  = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b  = upload_st(st, "time_embedding.linear_2.bias");
        CUdeviceptr l1_wd = upload_st(st, "unet_dual.time_embedding.linear_1.weight");
        CUdeviceptr l1_bd = upload_st(st, "unet_dual.time_embedding.linear_1.bias");
        CUdeviceptr l2_wd = upload_st(st, "unet_dual.time_embedding.linear_2.weight");
        CUdeviceptr l2_bd = upload_st(st, "unet_dual.time_embedding.linear_2.bias");

        /* DINO proj: [1, 257, 1536] -> [1, 1028, 1024] -> tile to Beff_main */
        const int CTX = 1024, EXTRA = 4;
        int rows_out = T_dino * EXTRA;
        if (rows_out != M_DINO) { fprintf(stderr, "ERROR: M_DINO mismatch\n"); return 1; }
        CUdeviceptr d_dino_in;  cuMemAlloc(&d_dino_in,  (size_t)T_dino * C_dino_in * sizeof(float));
        CUdeviceptr d_dino_lin; cuMemAlloc(&d_dino_lin, (size_t)T_dino * EXTRA * CTX * sizeof(float));
        CUdeviceptr d_dino_one; cuMemAlloc(&d_dino_one, (size_t)rows_out * CTX * sizeof(float));
        cuMemcpyHtoD(d_dino_in, dino_in, (size_t)T_dino * C_dino_in * sizeof(float));
        CUdeviceptr pw  = upload_st(st, "image_proj_model_dino.proj.weight");
        CUdeviceptr pb  = upload_st(st, "image_proj_model_dino.proj.bias");
        CUdeviceptr png = upload_st(st, "image_proj_model_dino.norm.weight");
        CUdeviceptr pnb = upload_st(st, "image_proj_model_dino.norm.bias");
        k_linear(&kk, d_dino_lin, d_dino_in, pw, pb, T_dino, C_dino_in, EXTRA * CTX);
        k_layernorm(&kk, d_dino_one, d_dino_lin, png, pnb, rows_out, CTX);
        CUdeviceptr d_dino; cuMemAlloc(&d_dino, (size_t)Beff_main * rows_out * CTX * sizeof(float));
        size_t dino_per = (size_t)rows_out * CTX * sizeof(float);
        for (int b = 0; b < Beff_main; b++)
            cuMemcpyDtoD(d_dino + (CUdeviceptr)b * dino_per, d_dino_one, dino_per);

        /* Workspace sized for the larger main pass; matches out_dino sizing */
        size_t MAX_ACT  = (size_t)Beff_main * 1280 * H0 * W0;
        size_t MAX_CCAT = (size_t)Beff_main * 960  * H0 * W0;
        size_t MAX_FF_GH= (size_t)Beff_main * 320  * H0 * W0 * 2 * 4;
        size_t MAX_FF_H = (size_t)Beff_main * 320  * H0 * W0 * 4;
        size_t MAX_BNC  = (size_t)Beff_main * 320  * H0 * W0;
        if ((size_t)Beff_main * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)Beff_main * 1280 * 16 * 16;
        if ((size_t)Beff_main *  640 * 32 * 32 > MAX_BNC) MAX_BNC = (size_t)Beff_main *  640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)Beff_main * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)Beff_main * 1280 * M_text;
        if ((size_t)Beff_main * 1280 * M_DINO > MAX_BMC) MAX_BMC = (size_t)Beff_main * 1280 * M_DINO;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));
        CUdeviceptr d_concat; cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        /* ===== Pass 1: dual-stream, ra_mode='w' (vanilla, no custom paths) ===== */
        g_ra_mode = 1; g_ra_cache.idx = 0;
        g_load_wp = "unet_dual.";
        CUdeviceptr cw_d = upload_st(st, "unet_dual.conv_in.weight");
        CUdeviceptr cb_d = upload_st(st, "unet_dual.conv_in.bias");
        pu_down_block dbd[4]; pu_mid_block midd; pu_up_block ubd[4];
        load_down_block(st, &dbd[0], 0,  320,  320,  5, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[1], 1,  320,  640, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[2], 2,  640, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 0, 0, 0, 0, 0, 0);
        load_mid_block(st, &midd, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 0, 0, 0, 0, 0, 0);

        CUdeviceptr d_temb_in_d, d_temb_h1_d, d_temb_d;
        cuMemAlloc(&d_temb_in_d, Beff_dual * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1_d, Beff_dual * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_d,    Beff_dual * 1280 * sizeof(float));
        k_timestep_embed(&kk, d_temb_in_d, d_ts_dual, Beff_dual, 320);
        k_linear(&kk, d_temb_h1_d, d_temb_in_d, l1_wd, l1_bd, Beff_dual, 320, 1280);
        k_silu(&kk, d_temb_h1_d, Beff_dual * 1280);
        k_linear(&kk, d_temb_d, d_temb_h1_d, l2_wd, l2_bd, Beff_dual, 1280, 1280);

        size_t in_n_d = (size_t)Beff_dual * IC_dual * H0 * W0;
        CUdeviceptr d_in_raw_d; cuMemAlloc(&d_in_raw_d, in_n_d * sizeof(float));
        cuMemcpyHtoD(d_in_raw_d, packed_dual, in_n_d * sizeof(float));
        for (int b = 0; b < Beff_dual; b++) {
            CUdeviceptr ib = d_in_raw_d + (CUdeviceptr)b * IC_dual * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a     + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw_d, cb_d, IC_dual, H0, W0, 320, 3, 3, 1);
        }
        pu_skip_stack ssd = {.top = 0, .B = Beff_dual};
        skip_push_copy(&ssd, ws.d_a, 320, H0, W0);
        int H = H0, W = W0;
        run_down_block(&kk, &dbd[0], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[1], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[2], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[3], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_mid_block(&kk, &midd, ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, H, W, M_text, &ws);
        run_up_block(&kk, &ubd[0], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[1], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[2], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[3], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        cuCtxSynchronize();
        fprintf(stderr, "  RA write pass: cached %d transformer-block slots\n", g_ra_cache.idx);

        /* ===== Pass 2: main forward, ra_mode='r', all paths on (no rope) ===== */
        g_ra_mode = 2; g_ra_cache.idx = 0;
        g_load_wp = "";
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        pu_down_block db[4]; pu_mid_block mid; pu_up_block ub[4];
        /* has_dino=1, has_ma=1, has_mda=1, has_ra=1 */
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_mid_block(st, &mid, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);

        CUdeviceptr d_temb_in_m, d_temb_h1_m, d_temb_m;
        cuMemAlloc(&d_temb_in_m, Beff_main * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1_m, Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_m,    Beff_main * 1280 * sizeof(float));
        k_timestep_embed(&kk, d_temb_in_m, d_ts_main, Beff_main, 320);
        k_linear(&kk, d_temb_h1_m, d_temb_in_m, l1_w, l1_b, Beff_main, 320, 1280);
        k_silu(&kk, d_temb_h1_m, Beff_main * 1280);
        k_linear(&kk, d_temb_m, d_temb_h1_m, l2_w, l2_b, Beff_main, 1280, 1280);

        size_t in_n_m = (size_t)Beff_main * IC_main * H0 * W0;
        CUdeviceptr d_in_raw_m; cuMemAlloc(&d_in_raw_m, in_n_m * sizeof(float));
        cuMemcpyHtoD(d_in_raw_m, packed_main, in_n_m * sizeof(float));
        CUdeviceptr d_text_m; cuMemAlloc(&d_text_m, (size_t)Beff_main * txt_per * sizeof(float));
        cuMemcpyHtoD(d_text_m, text_tiled_main, (size_t)Beff_main * txt_per * sizeof(float));
        for (int b = 0; b < Beff_main; b++) {
            CUdeviceptr ib = d_in_raw_m + (CUdeviceptr)b * IC_main * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a     + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw, cb, IC_main, H0, W0, 320, 3, 3, 1);
        }
        pu_skip_stack ss = {.top = 0, .B = Beff_main};
        skip_push_copy(&ss, ws.d_a, 320, H0, W0);
        H = H0; W = W0;
        run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, H, W, M_text, &ws);
        run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);

        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb_w = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob_w = upload_st(st, "conv_out.bias");
        for (int b = 0; b < Beff_main; b++) {
            CUdeviceptr xb = ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_groupnorm(&kk, yb, xb, ng, nb_w, 320, H * W, 32, 1);
        }
        for (int b = 0; b < Beff_main; b++) {
            CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            CUdeviceptr ob = ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
            k_conv(&kk, ob, yb, ow, ob_w, 320, H, W, 4, 3, 3, 1);
        }
        cuCtxSynchronize();
        size_t out_n = (size_t)Beff_main * 4 * H * W;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, ws.d_a, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/%s", ref_dir, out_npy);
        diff_against(cu, path, out_n, 5e-3f);
        free(cu); free(packed_main); free(packed_dual); free(text_tiled_main);
        free(sample); free(en); free(ep); free(text_in); free(ref_latents);
        free(dino_in); free(ts_in);
        g_ra_mode = 0;
        g_rope_n_levels = 0;
    } else if (!strcmp(stage, "out_loop")) {
        /* Phase 4.11b: scheduler↔UNet integration loop. Single-batch (no
         * CFG), all 4 attention paths on. Drives N UniPC steps using the
         * same conditioning each step (sample changes, timestep changes).
         * Validates final latent vs loop_x_after_<N-1>.npy. */
        const int HEAD_DIM = 64;
        const int N_PBR = 2, N_GEN = 2, N_REF = 1;
        const int B_outer = 1;
        const int Beff_main = B_outer * N_PBR * N_GEN;
        const int Beff_dual = B_outer * N_REF;
        const int H0 = 64, W0 = 64;
        const int IC_main = 12, IC_dual = 4;
        const int N_BLOCKS = 16;
        const int M_DINO = 1028;
        (void)HEAD_DIM;

        g_ra_cache.slots = (pu_ra_slot *)calloc(N_BLOCKS, sizeof(pu_ra_slot));
        g_ra_cache.n_slots = N_BLOCKS;
        g_ra_cache.idx = 0;
        g_ra_n_ref = N_REF;

        /* shared inputs */
        snprintf(path, sizeof(path), "%s/in_embeds_normal.npy", ref_dir);
        float *en = (float *)read_npy(path, &nd, shape, &n, dt);  if (!en) return 1;
        snprintf(path, sizeof(path), "%s/in_embeds_position.npy", ref_dir);
        float *ep = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ep) return 1;
        snprintf(path, sizeof(path), "%s/in_encoder_hidden_states.npy", ref_dir);
        float *text_in = (float *)read_npy(path, &nd, shape, &n, dt);  if (!text_in) return 1;
        int M_text = (int)shape[2], cross_dim = (int)shape[3];
        snprintf(path, sizeof(path), "%s/in_ref_latents.npy", ref_dir);
        float *ref_latents = (float *)read_npy(path, &nd, shape, &n, dt);  if (!ref_latents) return 1;
        snprintf(path, sizeof(path), "%s/in_dino_hidden_states.npy", ref_dir);
        float *dino_in = (float *)read_npy(path, &nd, shape, &n, dt);  if (!dino_in) return 1;
        int T_dino = (int)shape[1], C_dino_in = (int)shape[2];

        /* loop inputs */
        snprintf(path, sizeof(path), "%s/loop_x0.npy", ref_dir);
        float *x0 = (float *)read_npy(path, &nd, shape, &n, dt);  if (!x0) return 1;
        size_t x_n = (size_t)Beff_main * 4 * H0 * W0;
        if (n != x_n) { fprintf(stderr, "ERROR: loop_x0 size mismatch (%zu vs %zu)\n", n, x_n); return 1; }
        snprintf(path, sizeof(path), "%s/loop_timesteps.npy", ref_dir);
        int64_t *ts_loop = (int64_t *)read_npy(path, &nd, shape, &n, dt);
        if (!ts_loop) return 1;
        int N_steps = (int)n;
        fprintf(stderr, "out_loop: Beff_main=%d N_steps=%d M_text=%d cross=%d T_dino=%d\n",
                Beff_main, N_steps, M_text, cross_dim, T_dino);

        size_t per_view = (size_t)4 * H0 * W0;
        size_t per_in_main = (size_t)IC_main * H0 * W0;
        float *packed_main = (float *)malloc((size_t)Beff_main * per_in_main * sizeof(float));
        /* Fill en/ep portions once; sample portion is rewritten per step.
         * sample input layout in dump is [B, N_pbr, N_gen, 4, H, W];
         * loop_x0 is the Beff-flattened [Beff, 4, H, W] in the same order
         * (b = p*N_GEN + g). */
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                float *dst = packed_main + (size_t)b * per_in_main;
                memcpy(dst + per_view,     en + (size_t)g * per_view, per_view * sizeof(float));
                memcpy(dst + 2 * per_view, ep + (size_t)g * per_view, per_view * sizeof(float));
            }
        float *packed_dual = (float *)malloc((size_t)Beff_dual * per_view * sizeof(float));
        memcpy(packed_dual, ref_latents, (size_t)Beff_dual * per_view * sizeof(float));

        size_t txt_per = (size_t)M_text * cross_dim;
        float *text_tiled_main = (float *)malloc((size_t)Beff_main * txt_per * sizeof(float));
        for (int p = 0; p < N_PBR; p++)
            for (int g = 0; g < N_GEN; g++) {
                int b = p * N_GEN + g;
                memcpy(text_tiled_main + (size_t)b * txt_per,
                       text_in + (size_t)p * txt_per, txt_per * sizeof(float));
            }
        CUdeviceptr d_text_clip_ref = upload_st(st, "learned_text_clip_ref");
        size_t ltcr_n = (size_t)1 * M_text * cross_dim;
        CUdeviceptr d_text_dual; cuMemAlloc(&d_text_dual, (size_t)Beff_dual * txt_per * sizeof(float));
        for (int b = 0; b < Beff_dual; b++)
            cuMemcpyDtoD(d_text_dual + (CUdeviceptr)b * txt_per * sizeof(float),
                          d_text_clip_ref, ltcr_n * sizeof(float));

        int64_t ts_dual_arr[16];
        for (int b = 0; b < Beff_dual; b++) ts_dual_arr[b] = 0;
        CUdeviceptr d_ts_main, d_ts_dual;
        cuMemAlloc(&d_ts_main, Beff_main * sizeof(int64_t));
        cuMemAlloc(&d_ts_dual, Beff_dual * sizeof(int64_t));
        cuMemcpyHtoD(d_ts_dual, ts_dual_arr, Beff_dual * sizeof(int64_t));

        CUdeviceptr l1_w  = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b  = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w  = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b  = upload_st(st, "time_embedding.linear_2.bias");
        CUdeviceptr l1_wd = upload_st(st, "unet_dual.time_embedding.linear_1.weight");
        CUdeviceptr l1_bd = upload_st(st, "unet_dual.time_embedding.linear_1.bias");
        CUdeviceptr l2_wd = upload_st(st, "unet_dual.time_embedding.linear_2.weight");
        CUdeviceptr l2_bd = upload_st(st, "unet_dual.time_embedding.linear_2.bias");

        /* DINO proj once */
        const int CTX = 1024, EXTRA = 4;
        int rows_out = T_dino * EXTRA;
        if (rows_out != M_DINO) { fprintf(stderr, "ERROR: M_DINO mismatch\n"); return 1; }
        CUdeviceptr d_dino_in;  cuMemAlloc(&d_dino_in,  (size_t)T_dino * C_dino_in * sizeof(float));
        CUdeviceptr d_dino_lin; cuMemAlloc(&d_dino_lin, (size_t)T_dino * EXTRA * CTX * sizeof(float));
        CUdeviceptr d_dino_one; cuMemAlloc(&d_dino_one, (size_t)rows_out * CTX * sizeof(float));
        cuMemcpyHtoD(d_dino_in, dino_in, (size_t)T_dino * C_dino_in * sizeof(float));
        CUdeviceptr pw  = upload_st(st, "image_proj_model_dino.proj.weight");
        CUdeviceptr pb  = upload_st(st, "image_proj_model_dino.proj.bias");
        CUdeviceptr png = upload_st(st, "image_proj_model_dino.norm.weight");
        CUdeviceptr pnb = upload_st(st, "image_proj_model_dino.norm.bias");
        k_linear(&kk, d_dino_lin, d_dino_in, pw, pb, T_dino, C_dino_in, EXTRA * CTX);
        k_layernorm(&kk, d_dino_one, d_dino_lin, png, pnb, rows_out, CTX);
        CUdeviceptr d_dino; cuMemAlloc(&d_dino, (size_t)Beff_main * rows_out * CTX * sizeof(float));
        size_t dino_per = (size_t)rows_out * CTX * sizeof(float);
        for (int b = 0; b < Beff_main; b++)
            cuMemcpyDtoD(d_dino + (CUdeviceptr)b * dino_per, d_dino_one, dino_per);

        /* workspace */
        size_t MAX_ACT  = (size_t)Beff_main * 1280 * H0 * W0;
        size_t MAX_CCAT = (size_t)Beff_main * 960  * H0 * W0;
        size_t MAX_FF_GH= (size_t)Beff_main * 320  * H0 * W0 * 2 * 4;
        size_t MAX_FF_H = (size_t)Beff_main * 320  * H0 * W0 * 4;
        size_t MAX_BNC  = (size_t)Beff_main * 320  * H0 * W0;
        if ((size_t)Beff_main * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)Beff_main * 1280 * 16 * 16;
        if ((size_t)Beff_main *  640 * 32 * 32 > MAX_BNC) MAX_BNC = (size_t)Beff_main *  640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)Beff_main * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)Beff_main * 1280 * M_text;
        if ((size_t)Beff_main * 1280 * M_DINO > MAX_BMC) MAX_BMC = (size_t)Beff_main * 1280 * M_DINO;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));
        CUdeviceptr d_concat; cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        /* ===== Pass 1 (once): dual-stream cache populate ===== */
        g_ra_mode = 1; g_ra_cache.idx = 0;
        g_load_wp = "unet_dual.";
        CUdeviceptr cw_d = upload_st(st, "unet_dual.conv_in.weight");
        CUdeviceptr cb_d = upload_st(st, "unet_dual.conv_in.bias");
        pu_down_block dbd[4]; pu_mid_block midd; pu_up_block ubd[4];
        load_down_block(st, &dbd[0], 0,  320,  320,  5, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[1], 1,  320,  640, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[2], 2,  640, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_down_block(st, &dbd[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 0, 0, 0, 0, 0, 0);
        load_mid_block(st, &midd, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 0, 0, 0, 0, 0, 0);
        load_up_block(st, &ubd[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 0, 0, 0, 0, 0, 0);

        CUdeviceptr d_temb_in_d, d_temb_h1_d, d_temb_d;
        cuMemAlloc(&d_temb_in_d, Beff_dual * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1_d, Beff_dual * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_d,    Beff_dual * 1280 * sizeof(float));
        k_timestep_embed(&kk, d_temb_in_d, d_ts_dual, Beff_dual, 320);
        k_linear(&kk, d_temb_h1_d, d_temb_in_d, l1_wd, l1_bd, Beff_dual, 320, 1280);
        k_silu(&kk, d_temb_h1_d, Beff_dual * 1280);
        k_linear(&kk, d_temb_d, d_temb_h1_d, l2_wd, l2_bd, Beff_dual, 1280, 1280);

        size_t in_n_d = (size_t)Beff_dual * IC_dual * H0 * W0;
        CUdeviceptr d_in_raw_d; cuMemAlloc(&d_in_raw_d, in_n_d * sizeof(float));
        cuMemcpyHtoD(d_in_raw_d, packed_dual, in_n_d * sizeof(float));
        for (int b = 0; b < Beff_dual; b++) {
            CUdeviceptr ib = d_in_raw_d + (CUdeviceptr)b * IC_dual * H0 * W0 * sizeof(float);
            CUdeviceptr ob = ws.d_a     + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
            k_conv(&kk, ob, ib, cw_d, cb_d, IC_dual, H0, W0, 320, 3, 3, 1);
        }
        pu_skip_stack ssd = {.top = 0, .B = Beff_dual};
        skip_push_copy(&ssd, ws.d_a, 320, H0, W0);
        int H = H0, W = W0;
        run_down_block(&kk, &dbd[0], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[1], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[2], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_down_block(&kk, &dbd[3], ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_mid_block(&kk, &midd, ws.d_a, ws.d_b, d_temb_d, d_text_dual, 0, 0, Beff_dual, H, W, M_text, &ws);
        run_up_block(&kk, &ubd[0], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[1], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[2], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        run_up_block(&kk, &ubd[3], ws.d_a, ws.d_b, d_concat, d_temb_d, d_text_dual, 0, 0, Beff_dual, &H, &W, M_text, &ws, &ssd);
        cuCtxSynchronize();
        int n_cached_slots = g_ra_cache.idx;
        fprintf(stderr, "  RA write pass: cached %d slots\n", n_cached_slots);

        /* ===== Main pass setup (done once) ===== */
        g_load_wp = "";
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        pu_down_block db[4]; pu_mid_block mid; pu_up_block ub[4];
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_mid_block(st, &mid, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[1], 1,  640, 1280, 1280, 20, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[2], 2,  320,  640, 1280, 10, 1, 1, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);
        load_up_block(st, &ub[3], 3,  320,  320,  640,  5, 1, 0, cross_dim, 1, 1, 1, 1, N_PBR, N_GEN);

        CUdeviceptr d_temb_in_m, d_temb_h1_m, d_temb_m;
        cuMemAlloc(&d_temb_in_m, Beff_main * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1_m, Beff_main * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_m,    Beff_main * 1280 * sizeof(float));

        size_t in_n_m = (size_t)Beff_main * IC_main * H0 * W0;
        CUdeviceptr d_in_raw_m; cuMemAlloc(&d_in_raw_m, in_n_m * sizeof(float));
        CUdeviceptr d_text_m; cuMemAlloc(&d_text_m, (size_t)Beff_main * txt_per * sizeof(float));
        cuMemcpyHtoD(d_text_m, text_tiled_main, (size_t)Beff_main * txt_per * sizeof(float));

        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb_w = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob_w = upload_st(st, "conv_out.bias");

        /* UniPC scheduler */
        pu_unipc sch;
        pu_unipc_init(&sch, N_steps, x_n);
        float *x_host = (float *)malloc(x_n * sizeof(float));
        memcpy(x_host, x0, x_n * sizeof(float));
        float *noise_pred = (float *)malloc(x_n * sizeof(float));

        int all_ok = 1;
        for (int i = 0; i < N_steps; i++) {
            /* Pack current x into the sample slot of packed_main */
            for (int p = 0; p < N_PBR; p++)
                for (int g = 0; g < N_GEN; g++) {
                    int b = p * N_GEN + g;
                    float *dst = packed_main + (size_t)b * per_in_main;
                    memcpy(dst, x_host + (size_t)b * per_view, per_view * sizeof(float));
                }
            cuMemcpyHtoD(d_in_raw_m, packed_main, in_n_m * sizeof(float));

            /* timestep embed for current ts */
            int64_t ts_v = ts_loop[i];
            int64_t ts_main_arr[16];
            for (int b = 0; b < Beff_main; b++) ts_main_arr[b] = ts_v;
            cuMemcpyHtoD(d_ts_main, ts_main_arr, Beff_main * sizeof(int64_t));
            k_timestep_embed(&kk, d_temb_in_m, d_ts_main, Beff_main, 320);
            k_linear(&kk, d_temb_h1_m, d_temb_in_m, l1_w, l1_b, Beff_main, 320, 1280);
            k_silu(&kk, d_temb_h1_m, Beff_main * 1280);
            k_linear(&kk, d_temb_m, d_temb_h1_m, l2_w, l2_b, Beff_main, 1280, 1280);

            /* Main forward (RA mode='r', read in-order from cache) */
            g_ra_mode = 2; g_ra_cache.idx = 0;
            for (int b = 0; b < Beff_main; b++) {
                CUdeviceptr ib = d_in_raw_m + (CUdeviceptr)b * IC_main * H0 * W0 * sizeof(float);
                CUdeviceptr ob = ws.d_a     + (CUdeviceptr)b * 320 * H0 * W0 * sizeof(float);
                k_conv(&kk, ob, ib, cw, cb, IC_main, H0, W0, 320, 3, 3, 1);
            }
            pu_skip_stack ss = {.top = 0, .B = Beff_main};
            skip_push_copy(&ss, ws.d_a, 320, H0, W0);
            H = H0; W = W0;
            run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, H, W, M_text, &ws);
            run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);
            run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb_m, d_text_m, d_dino, M_DINO, Beff_main, &H, &W, M_text, &ws, &ss);

            for (int b = 0; b < Beff_main; b++) {
                CUdeviceptr xb = ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
                CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
                k_groupnorm(&kk, yb, xb, ng, nb_w, 320, H * W, 32, 1);
            }
            for (int b = 0; b < Beff_main; b++) {
                CUdeviceptr yb = ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
                CUdeviceptr ob = ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
                k_conv(&kk, ob, yb, ow, ob_w, 320, H, W, 4, 3, 3, 1);
            }
            cuCtxSynchronize();
            cuMemcpyDtoH(noise_pred, ws.d_a, x_n * sizeof(float));

            /* Optional: validate per-step UNet output against pyref */
            char p2[512];
            snprintf(p2, sizeof(p2), "%s/loop_model_out_%d.npy", ref_dir, i);
            int nd2; uint64_t sh2[8]; size_t n2; char dt2[8];
            float *mo_ref = (float *)read_npy(p2, &nd2, sh2, &n2, dt2);
            double mo_max = 0.0;
            if (mo_ref && n2 == x_n) {
                for (size_t k = 0; k < x_n; k++) {
                    double d = fabs((double)noise_pred[k] - (double)mo_ref[k]);
                    if (d > mo_max) mo_max = d;
                }
            }
            free(mo_ref);

            /* UniPC step on host */
            pu_unipc_step(&sch, noise_pred, x_host);

            /* Validate cumulative latent vs pyref */
            snprintf(p2, sizeof(p2), "%s/loop_x_after_%d.npy", ref_dir, i);
            float *xa_ref = (float *)read_npy(p2, &nd2, sh2, &n2, dt2);
            double x_max = 0.0, x_sum = 0.0;
            if (xa_ref && n2 == x_n) {
                for (size_t k = 0; k < x_n; k++) {
                    double d = fabs((double)x_host[k] - (double)xa_ref[k]);
                    x_sum += d;
                    if (d > x_max) x_max = d;
                }
            }
            double x_mae = x_sum / (double)x_n;
            int ok = (x_max < 5e-2) && (mo_max < 5e-2);
            if (!ok) all_ok = 0;
            fprintf(stderr, "  step %2d  t=%4lld  mo_max=%.3e  x_mae=%.3e  x_max=%.3e  %s\n",
                    i, (long long)ts_v, mo_max, x_mae, x_max, ok ? "OK" : "**MISMATCH**");
            free(xa_ref);
        }

        pu_unipc_free(&sch);
        free(x_host); free(noise_pred);
        free(packed_main); free(packed_dual); free(text_tiled_main);
        free(en); free(ep); free(text_in); free(ref_latents);
        free(dino_in); free(x0); free(ts_loop);
        g_ra_mode = 0;
        fprintf(stderr, "\nout_loop result: %s\n", all_ok ? "PASS" : "FAIL");
        if (!all_ok) return 1;
    } else if (!strcmp(stage, "dino_proj")) {
        /* image_proj_model_dino: Linear(1536 -> 4*1024) + LayerNorm(1024)
         * applied per (token, slot). Input [1, 257, 1536] -> [1, 1028, 1024]
         * vs dino_proj.npy. */
        snprintf(path, sizeof(path), "%s/in_dino_hidden_states.npy", ref_dir);
        float *dino = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!dino) return 1;
        int Bd = (int)shape[0], T_in = (int)shape[1], C_in = (int)shape[2];
        if (C_in != 1536) {
            fprintf(stderr, "ERROR: dino C_in=%d, expected 1536\n", C_in); return 1;
        }
        const int CTX = 1024, EXTRA = 4;
        int rows = Bd * T_in;            /* 257 */
        int rows_out = rows * EXTRA;     /* 1028 */
        fprintf(stderr, "dino [%d, %d, %d] -> [%d, %d, %d]\n",
                Bd, T_in, C_in, Bd, T_in * EXTRA, CTX);

        CUdeviceptr d_in;  cuMemAlloc(&d_in,  (size_t)rows * C_in * sizeof(float));
        CUdeviceptr d_lin; cuMemAlloc(&d_lin, (size_t)rows * EXTRA * CTX * sizeof(float));
        CUdeviceptr d_out; cuMemAlloc(&d_out, (size_t)rows_out * CTX * sizeof(float));
        cuMemcpyHtoD(d_in, dino, (size_t)rows * C_in * sizeof(float));

        CUdeviceptr pw = upload_st(st, "image_proj_model_dino.proj.weight");
        CUdeviceptr pb = upload_st(st, "image_proj_model_dino.proj.bias");
        CUdeviceptr ng = upload_st(st, "image_proj_model_dino.norm.weight");
        CUdeviceptr nb = upload_st(st, "image_proj_model_dino.norm.bias");

        /* proj: [rows, 1536] @ [4096, 1536]^T + [4096] -> [rows, 4096] */
        k_linear(&kk, d_lin, d_in, pw, pb, rows, C_in, EXTRA * CTX);
        /* LN over last dim (1024) on [rows*EXTRA, 1024] reshape */
        k_layernorm(&kk, d_out, d_lin, ng, nb, rows_out, CTX);
        cuCtxSynchronize();

        size_t out_n = (size_t)rows_out * CTX;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, d_out, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/dino_proj.npy", ref_dir);
        diff_against(cu, path, out_n, 1e-4f);
        free(cu); free(dino);
    } else {
        fprintf(stderr, "unknown stage: %s\n", stage); return 1;
    }

    if (ts) free(ts);
    safetensors_close(st);
    cuModuleUnload(kk.mod);
    cuCtxDestroy(ctx);
    return 0;
}
