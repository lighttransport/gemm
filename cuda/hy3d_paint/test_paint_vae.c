/*
 * test_paint_vae.c - Native CUDA SD-2.1 paint VAE encoder + decoder.
 *
 * Phase 2 of the Hunyuan3D-2.1 texgen port. Loads weights from
 * paint_vae.safetensors (run ref/hy3d/export_vae_safetensors.py to convert
 * the upstream .bin), encodes an RGB image .npy → latent .npy or decodes a
 * latent .npy → RGB .npy, for diffing against the diffusers reference
 * produced by ref/hy3d/dump_paint_vae.py.
 *
 * Usage:
 *   ./test_paint_vae decode <vae.safetensors> <latent.npy> <out_recon.npy>
 *   ./test_paint_vae encode <vae.safetensors> <input.npy> <out_latent.npy>
 *
 * Decoder architecture (stock diffusers AutoencoderKL):
 *   post_quant_conv 1x1 (4->4)
 *   conv_in 3x3 (4->512)
 *   mid: ResBlock(512) → Attn(512) → ResBlock(512)
 *   up_blocks[0]: 3 ResBlocks(512->512), upsample(512)
 *   up_blocks[1]: 3 ResBlocks(512->512), upsample(512)
 *   up_blocks[2]: 3 ResBlocks(512->256), upsample(256)
 *   up_blocks[3]: 3 ResBlocks(256->128), no upsample
 *   conv_norm_out 32grp -> SiLU -> conv_out 3x3 (128->3)
 *
 * Encoder architecture (mirror):
 *   conv_in 3x3 (3->128)
 *   down_blocks[0]: 2 ResBlocks(128->128), down(128) (asymmetric pad+stride2)
 *   down_blocks[1]: ResBlock(128->256)+shortcut, ResBlock(256->256), down(256)
 *   down_blocks[2]: ResBlock(256->512)+shortcut, ResBlock(512->512), down(512)
 *   down_blocks[3]: 2 ResBlocks(512->512), no downsample
 *   mid: ResBlock(512) → Attn(512) → ResBlock(512)
 *   conv_norm_out 32grp -> SiLU -> conv_out 3x3 (512->8)
 *   quant_conv 1x1 (8->8); take first 4 channels as mean (deterministic z).
 */

#define CUDA_PAINT_VAE_RUNNER_IMPLEMENTATION
#include "cuda_paint_vae_runner.h"

/* ===== main =============================================================== */

int main(int argc, char **argv) {
    if (argc < 5 ||
        (strcmp(argv[1], "encode") && strcmp(argv[1], "decode"))) {
        fprintf(stderr,
            "Usage: %s decode <vae.safetensors> <latent.npy> <out_recon.npy>\n"
            "       %s encode <vae.safetensors> <input.npy>  <out_latent.npy>\n",
            argv[0], argv[0]);
        return 1;
    }
    int do_encode = !strcmp(argv[1], "encode");
    const char *st_path  = argv[2];
    const char *in_path  = argv[3];
    const char *out_path = argv[4];

    int nd; uint64_t shape[8]; size_t total;
    float *in_buf = read_npy_f32(in_path, &nd, shape, &total);
    if (!in_buf) return 1;

    int IC, IH, IW;     /* input shape (single sample) */
    int OC, OH, OW;     /* output shape (single sample) */
    int B = 1;
    if (do_encode) {
        if (nd == 3 && shape[0] == 3) {
            IC = 3; IH = (int)shape[1]; IW = (int)shape[2];
        } else if (nd == 4 && shape[1] == 3) {
            B = (int)shape[0]; IC = 3; IH = (int)shape[2]; IW = (int)shape[3];
        } else {
            fprintf(stderr, "ERROR: expected input shape [3,H,W] or [B,3,H,W], got nd=%d\n", nd);
            return 1;
        }
        OC = 4; OH = IH / 8; OW = IW / 8;
        fprintf(stderr, "input   [%d, %d, %d, %d]   latent [%d, %d, %d, %d]\n",
                B, IC, IH, IW, B, OC, OH, OW);
    } else {
        if (nd == 3 && shape[0] == 4) {
            IC = 4; IH = (int)shape[1]; IW = (int)shape[2];
        } else if (nd == 4 && shape[1] == 4) {
            B = (int)shape[0]; IC = 4; IH = (int)shape[2]; IW = (int)shape[3];
        } else {
            fprintf(stderr, "ERROR: expected latent shape [4,H,W] or [B,4,H,W], got nd=%d\n", nd);
            return 1;
        }
        OC = 3; OH = IH * 8; OW = IW * 8;
        fprintf(stderr, "latent  [%d, %d, %d, %d]   recon  [%d, %d, %d, %d]\n",
                B, IC, IH, IW, B, OC, OH, OW);
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    pvae_kernels kk = {0};
    int sm = cu_compile_kernels(&kk.mod, dev,
                                cuda_paint_vae_kernels_src,
                                "hy3d_paint_vae", 1, "HY3D-PAINT-VAE");
    if (sm < 0) return 1;
    cuModuleGetFunction(&kk.f_gn,        kk.mod, "vae_groupnorm_f32");
    cuModuleGetFunction(&kk.f_conv,      kk.mod, "vae_conv2d_f32");
    cuModuleGetFunction(&kk.f_conv_down, kk.mod, "vae_conv2d_down_f32");
    cuModuleGetFunction(&kk.f_up2x,      kk.mod, "vae_upsample2x_f32");
    cuModuleGetFunction(&kk.f_add,       kk.mod, "vae_add_f32");
    cuModuleGetFunction(&kk.f_attn,      kk.mod, "vae_attn_f32");
    cuModuleGetFunction(&kk.f_chw_nc,    kk.mod, "vae_chw_to_nc_f32");
    cuModuleGetFunction(&kk.f_nc_chw,    kk.mod, "vae_nc_to_chw_f32");

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "ERROR: cannot open %s\n", st_path); return 1; }
    pvae_decoder D = {0};
    pvae_encoder E = {0};
    if (do_encode) load_encoder(st, &E);
    else           load_decoder(st, &D);
    fprintf(stderr, "loaded %s weights from %s\n",
            do_encode ? "encoder" : "decoder", st_path);

    /* Worst-case workspace per buffer.
     *   encoder stages (image-res H_full): 128@H, 256@H/2, 512@H/4, 512@H/8
     *   decoder stages (image-res H_full = 8*LH):
     *     512@LH, 512@2LH, 512@4LH, 256@4LH, 256@8LH, 128@8LH
     *   So 256 * H_full² is the dominant decoder term. */
    int H_full = do_encode ? IH : OH;
    int W_full = do_encode ? IW : OW;
    size_t max_n = 0;
    if (do_encode) {
        int CH[4] = { 128, 256, 512, 512 };
        for (int k = 0; k < 4; k++) {
            size_t n = (size_t)CH[k] * (H_full >> k) * (W_full >> k);
            if (n > max_n) max_n = n;
        }
    } else {
        /* decoder: enumerate the actual spatial/channel pairs */
        int LH = H_full / 8, LW = W_full / 8;
        size_t cands[] = {
            (size_t)512 * LH * LW,
            (size_t)512 * (LH*2) * (LW*2),
            (size_t)512 * (LH*4) * (LW*4),
            (size_t)256 * (LH*4) * (LW*4),
            (size_t)256 * (LH*8) * (LW*8),
            (size_t)128 * (LH*8) * (LW*8),
        };
        for (size_t i = 0; i < sizeof(cands)/sizeof(cands[0]); i++)
            if (cands[i] > max_n) max_n = cands[i];
    }
    fprintf(stderr, "workspace = %.1f MB / buffer\n",
            max_n * 4 / 1024.0 / 1024.0);

    /* Attention always at lowest resolution: 512 * (H_full/8)^2 floats. */
    size_t attn_n = (size_t)512 * (H_full / 8) * (W_full / 8);

    CUdeviceptr d_in_dev, d_out_dev, d_a, d_b, d_t1, d_t2;
    CUdeviceptr d_qnc, d_knc, d_vnc, d_ync;
    cuMemAlloc(&d_in_dev,  IC * (size_t)IH * IW * sizeof(float));
    cuMemAlloc(&d_out_dev, OC * (size_t)OH * OW * sizeof(float));
    cuMemAlloc(&d_a,   max_n * sizeof(float));
    cuMemAlloc(&d_b,   max_n * sizeof(float));
    cuMemAlloc(&d_t1,  max_n * sizeof(float));
    cuMemAlloc(&d_t2,  max_n * sizeof(float));
    cuMemAlloc(&d_qnc, attn_n * sizeof(float));
    cuMemAlloc(&d_knc, attn_n * sizeof(float));
    cuMemAlloc(&d_vnc, attn_n * sizeof(float));
    cuMemAlloc(&d_ync, attn_n * sizeof(float));
    size_t in_per  = (size_t)IC * IH * IW;
    size_t out_per = (size_t)OC * OH * OW;
    float *out_buf = (float *)malloc((size_t)B * out_per * sizeof(float));
    for (int bi = 0; bi < B; bi++) {
        cuMemcpyHtoD(d_in_dev, in_buf + (size_t)bi * in_per, in_per * sizeof(float));
        if (do_encode) {
            encode(&kk, &E, d_in_dev, IH, IW, d_out_dev,
                    d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
        } else {
            decode(&kk, &D, d_in_dev, IH, IW, d_out_dev,
                    d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(out_buf + (size_t)bi * out_per, d_out_dev, out_per * sizeof(float));
    }

    size_t out_n = (size_t)B * out_per;
    if (B == 1) {
        int sh3[3] = { OC, OH, OW };
        write_npy_f32(out_path, out_buf, sh3, 3);
    } else {
        int sh4[4] = { B, OC, OH, OW };
        write_npy_f32(out_path, out_buf, sh4, 4);
    }
    float mn = out_buf[0], mx = out_buf[0];
    for (size_t i = 1; i < out_n; i++) {
        if (out_buf[i] < mn) mn = out_buf[i];
        if (out_buf[i] > mx) mx = out_buf[i];
    }
    fprintf(stderr, "wrote %s  range=[%.3f, %.3f]\n", out_path, mn, mx);

    free(out_buf); free(in_buf);
    safetensors_close(st);
    cuModuleUnload(kk.mod);
    cuCtxDestroy(ctx);
    return 0;
}
