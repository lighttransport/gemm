/*
 * test_cuda_qimg.c - CUDA Qwen-Image end-to-end test
 *
 * Modes:
 *   --test-init    : Initialize CUDA, compile kernels
 *   --test-load    : Load DiT + VAE weights to GPU
 *   --test-dit     : Run single DiT step on GPU
 *   --generate     : Full text-to-image pipeline
 *
 * Build:
 *   cc -O2 -mavx2 -mfma -I../../common -I.. -o test_cuda_qimg test_cuda_qimg.c ../cuew.c -lm -ldl -lpthread
 */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define QIMG_DIT_IMPLEMENTATION
#define QIMG_VAE_IMPLEMENTATION
#define QIMG_TEXT_ENCODER_IMPLEMENTATION
#define CUDA_QIMG_RUNNER_IMPLEMENTATION

#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/qwen_image_dit.h"
#include "../../common/qwen_image_vae.h"
#include "../../common/qwen_image_text_encoder.h"
#include "cuda_qimg_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Simple PRNG */
static uint64_t rng_state = 42;
static float randn(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[(size_t)c * h * w + y * w + x];
                v = v * 0.5f + 0.5f;
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                px[c] = (uint8_t)(v * 255.0f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

int main(int argc, char **argv) {
    const char *dit_path = "/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors";
    const char *vae_path = "/mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors";
    const char *enc_path = "/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf";
    const char *prompt = "a red apple on a white table";
    const char *mode = NULL;
    int out_h = 256, out_w = 256, n_steps = 20;
    int force_f16 = 0, no_cfg = 0;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test-init") == 0) mode = "init";
        else if (strcmp(argv[i], "--test-load") == 0) mode = "load";
        else if (strcmp(argv[i], "--test-dit") == 0) mode = "dit";
        else if (strcmp(argv[i], "--generate") == 0) mode = "gen";
        else if (strcmp(argv[i], "--no-fp8") == 0) force_f16 = 1;
        else if (strcmp(argv[i], "--no-cfg") == 0) no_cfg = 1;
        else if (strcmp(argv[i], "--dit") == 0 && i+1 < argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0 && i+1 < argc) vae_path = argv[++i];
        else if (strcmp(argv[i], "--enc") == 0 && i+1 < argc) enc_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--height") == 0 && i+1 < argc) out_h = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i+1 < argc) out_w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
    }

    if (!mode) {
        fprintf(stderr, "Usage: %s --test-init|--test-load|--test-dit|--generate\n"
                "  --dit <st>  --vae <st>  --enc <gguf>  --prompt <text>\n"
                "  --height <h>  --width <w>  --steps <n>  --seed <s>  --no-fp8\n", argv[0]);
        return 1;
    }

    int lat_h = out_h / 8, lat_w = out_w / 8;

    fprintf(stderr, "=== CUDA Qwen-Image: %s ===\n", mode);

    /* ---- Init CUDA ---- */
    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    if (r) r->verbose = 2;  /* enable debug dumps */
    if (r && force_f16) { r->use_fp8_gemm = 0; fprintf(stderr, "Forced F16 path\n"); }
    if (!r) { fprintf(stderr, "Init failed\n"); return 1; }

    if (strcmp(mode, "init") == 0) { cuda_qimg_free(r); return 0; }

    /* ---- Load DiT ---- */
    clock_t t0 = clock();
    if (cuda_qimg_load_dit(r, dit_path) != 0) { cuda_qimg_free(r); return 1; }
    fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

    if (strcmp(mode, "load") == 0) {
        cuda_qimg_load_vae(r, vae_path);
        cuda_qimg_free(r); return 0;
    }

    /* ---- test-dit: single step ---- */
    if (strcmp(mode, "dit") == 0) {
        int ps = 2, hp = lat_h/ps, wp = lat_w/ps, n_img = hp*wp, n_txt = 7;
        int in_ch = 64, txt_dim = 3584;
        rng_state = seed;
        float *img = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *txt = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        float *out = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        for (int i = 0; i < n_img*in_ch; i++) img[i] = randn() * 0.1f;
        for (int i = 0; i < n_txt*txt_dim; i++) txt[i] = randn() * 0.1f;
        t0 = clock();
        cuda_qimg_dit_step(r, img, n_img, txt, n_txt, 500.0f, out);
        fprintf(stderr, "DiT step: %.2fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
        { float mn=out[0],mx=out[0],sm=0; int nn=n_img*in_ch, nc=0;
          for(int i=0;i<nn;i++){if(out[i]!=out[i])nc++;else{if(out[i]<mn)mn=out[i];if(out[i]>mx)mx=out[i];sm+=out[i];}}
          fprintf(stderr, "Output: min=%.4f max=%.4f mean=%.4f nan=%d/%d\n",mn,mx,sm/(nn-nc),nc,nn); }
        free(img); free(txt); free(out);
        cuda_qimg_free(r); return 0;
    }

    /* ---- generate: full text-to-image pipeline ---- */
    if (strcmp(mode, "gen") == 0) {
        int ps = 2;
        int hp = lat_h / ps, wp = lat_w / ps;
        int n_img = hp * wp;
        int in_ch = 64, lat_ch = 16;
        int txt_dim = 3584;

        /* 1. Text encoder — try loading ComfyUI .npy first, fall back to GGUF */
        fprintf(stderr, "\n[1/3] Text conditioning...\n");
        int n_txt = 0, n_txt_neg = 0;
        float *txt_hidden = NULL, *txt_neg_hidden = NULL;

        /* Try ComfyUI pre-encoded hidden states */
        {
            /* Try loading from several locations */
            const char *pos_paths[] = {
                "../../ref/qwen_image/output/comfyui_text_hidden.npy",
                "../../../diffusion/ref/qwen_image/output/comfyui_text_hidden.npy",
                "comfyui_text_hidden.npy", NULL};
            const char *pos_path = NULL, *neg_path = NULL;
            for (int pi = 0; pos_paths[pi]; pi++) {
                FILE *tf = fopen(pos_paths[pi], "rb");
                if (tf) { fclose(tf); pos_path = pos_paths[pi]; break; }
            }
            if (!pos_path) pos_path = "comfyui_text_hidden.npy";
            {
                /* Derive neg path from pos path */
                static char neg_buf[512];
                const char *dot = strrchr(pos_path, '.');
                if (dot) {
                    size_t base_len = (size_t)(dot - pos_path);
                    memcpy(neg_buf, pos_path, base_len);
                    snprintf(neg_buf + base_len, sizeof(neg_buf) - base_len, "_neg.npy");
                    neg_path = neg_buf;
                }
            }
            FILE *fp = fopen(pos_path, "rb");
            if (fp) {
                uint8_t hdr[10]; if(fread(hdr,1,10,fp)==10) {
                    int hl = (int)hdr[8] | ((int)hdr[9]<<8);
                    char hbuf[512]; fseek(fp,10,SEEK_SET);
                    if(fread(hbuf,1,(size_t)hl,fp)==(size_t)hl) {
                        hbuf[hl]=0;
                        /* Parse shape (1, N, 3584) */
                        char *sp = strstr(hbuf, "shape");
                        if (sp) { sp = strchr(sp, '(');
                            if (sp) { sp++; /* skip batch dim */
                                while(*sp && *sp!=',') sp++; if(*sp==',') sp++;
                                n_txt = atoi(sp);
                            }
                        }
                    }
                    if (n_txt > 0) {
                        fseek(fp, 10+hl, SEEK_SET);
                        txt_hidden = (float*)malloc((size_t)n_txt*txt_dim*sizeof(float));
                        fread(txt_hidden, sizeof(float), (size_t)n_txt*txt_dim, fp);
                        fprintf(stderr, "  Loaded ComfyUI pos hidden [%d, %d]\n", n_txt, txt_dim);
                    }
                }
                fclose(fp);
                /* Load negative */
                fp = fopen(neg_path, "rb");
                if (fp) {
                    uint8_t hdr2[10]; fread(hdr2,1,10,fp);
                    int hl2 = (int)hdr2[8] | ((int)hdr2[9]<<8);
                    char hb2[512]; fseek(fp,10,SEEK_SET);
                    if(fread(hb2,1,(size_t)hl2,fp)==(size_t)hl2) {
                        hb2[hl2]=0;
                        char *sp2 = strstr(hb2, "shape");
                        if (sp2) { sp2 = strchr(sp2, '(');
                            if (sp2) { sp2++; while(*sp2&&*sp2!=',') sp2++; if(*sp2==',') sp2++;
                                n_txt_neg = atoi(sp2); } }
                    }
                    if (n_txt_neg > 0) {
                        fseek(fp, 10+hl2, SEEK_SET);
                        txt_neg_hidden = (float*)malloc((size_t)n_txt_neg*txt_dim*sizeof(float));
                        fread(txt_neg_hidden, sizeof(float), (size_t)n_txt_neg*txt_dim, fp);
                        fprintf(stderr, "  Loaded ComfyUI neg hidden [%d, %d]\n", n_txt_neg, txt_dim);
                    }
                    fclose(fp);
                }
            }
        }

        /* Try FP8 safetensors text encoder first, then GGUF fallback */
        if (!txt_hidden) {
            const char *st_enc = "/mnt/disk01/models/qwen-image-st/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
            FILE *tf = fopen(st_enc, "rb");
            if (tf) {
                fclose(tf);
                fprintf(stderr, "  Loading FP8 scaled text encoder...\n");
                qimg_text_enc *enc = qimg_text_enc_load_safetensors(st_enc, enc_path);
                if (enc) {
                    txt_hidden = qimg_text_enc_encode(enc, prompt, &n_txt);
                    fprintf(stderr, "  Encoding negative prompt...\n");
                    txt_neg_hidden = qimg_text_enc_encode(enc, " ", &n_txt_neg);
                    qimg_text_enc_free(enc);
                }
            }
        }
        /* GGUF fallback */
        if (!txt_hidden) {
            qimg_text_enc *enc = qimg_text_enc_load(enc_path);
            if (enc) {
                txt_hidden = qimg_text_enc_encode(enc, prompt, &n_txt);
                fprintf(stderr, "  Encoding negative prompt...\n");
                txt_neg_hidden = qimg_text_enc_encode(enc, " ", &n_txt_neg);
                qimg_text_enc_free(enc);
            }
        }
        if (!txt_hidden) {
            fprintf(stderr, "Text encoder failed\n");
            free(txt_neg_hidden);
            cuda_qimg_free(r); return 1;
        }
        /* Pad negative to same length as positive (or vice versa) */
        if (n_txt_neg < n_txt) {
            float *padded = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
            memcpy(padded, txt_neg_hidden, (size_t)n_txt_neg * txt_dim * sizeof(float));
            free(txt_neg_hidden);
            txt_neg_hidden = padded;
            n_txt_neg = n_txt;
        } else if (n_txt < n_txt_neg) {
            float *padded = (float *)calloc((size_t)n_txt_neg * txt_dim, sizeof(float));
            memcpy(padded, txt_hidden, (size_t)n_txt * txt_dim * sizeof(float));
            free(txt_hidden);
            txt_hidden = padded;
            n_txt = n_txt_neg;
        }

        /* 2. DiT denoising loop (CUDA) */
        fprintf(stderr, "\n[2/3] DiT denoising (%d steps, %dx%d, %s)...\n",
                n_steps, out_w, out_h, r->use_fp8_gemm ? "FP8 native" : "F16 fallback");

        /* Initialize noise latent */
        rng_state = seed;
        float *latent = (float *)malloc((size_t)lat_ch * lat_h * lat_w * sizeof(float));
        for (size_t i = 0; i < (size_t)lat_ch * lat_h * lat_w; i++)
            latent[i] = randn();

        /* Scheduler */
        qimg_scheduler sched;
        qimg_sched_init(&sched);
        /* Use ComfyUI-compatible scheduler: shift=3.1, timestep=sigma */
        /* Use original dynamic scheduler (better stability than AuraFlow shift=3.1) */
        qimg_sched_set_timesteps(&sched, n_steps, n_img);

        float cfg_scale = 2.5f;  /* ComfyUI default for Qwen-Image */

        /* Patchify + denoising loop with CFG */
        float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *vel_cond = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *vel_uncond = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

        clock_t denoise_t0 = clock();
        for (int step = 0; step < n_steps; step++) {
            float t_val = sched.timesteps[step];
            clock_t step_t0 = clock();

            /* Patchify latent */
            qimg_dit_patchify(img_tokens, latent, lat_ch, lat_h, lat_w, ps);

            /* Conditional DiT forward (with text) */
            cuda_qimg_dit_step(r, img_tokens, n_img, txt_hidden, n_txt,
                               t_val, vel_cond);

            int lat_n = lat_ch * lat_h * lat_w;
            float *vel_latent = (float *)malloc((size_t)lat_n * sizeof(float));

            if (no_cfg) {
                /* No CFG: just use conditional velocity directly */
                qimg_dit_unpatchify(vel_latent, vel_cond, n_img, lat_ch, lat_h, lat_w, ps);
            } else {
                /* Unconditional DiT forward (negative prompt encoding) */
                cuda_qimg_dit_step(r, img_tokens, n_img, txt_neg_hidden, n_txt,
                                   t_val, vel_uncond);

                /* CFG combine: velocity = uncond + cfg_scale * (cond - uncond) */
                float *vc_lat = (float *)malloc((size_t)lat_n * sizeof(float));
                float *vu_lat = (float *)malloc((size_t)lat_n * sizeof(float));
                qimg_dit_unpatchify(vc_lat, vel_cond, n_img, lat_ch, lat_h, lat_w, ps);
                qimg_dit_unpatchify(vu_lat, vel_uncond, n_img, lat_ch, lat_h, lat_w, ps);
                for (int i = 0; i < lat_n; i++)
                    vel_latent[i] = vu_lat[i] + cfg_scale * (vc_lat[i] - vu_lat[i]);
                free(vc_lat); free(vu_lat);

                /* CFGNorm: normalize combined velocity by its magnitude */
                {
                    float mag = 0;
                    for (int i = 0; i < lat_n; i++) mag += vel_latent[i] * vel_latent[i];
                    mag = sqrtf(mag / (float)lat_n + 1e-8f);
                    float *vu_tmp = (float *)malloc((size_t)lat_n * sizeof(float));
                    qimg_dit_unpatchify(vu_tmp, vel_uncond, n_img, lat_ch, lat_h, lat_w, ps);
                    float umag = 0;
                    for (int i = 0; i < lat_n; i++) umag += vu_tmp[i] * vu_tmp[i];
                    umag = sqrtf(umag / (float)lat_n + 1e-8f);
                    free(vu_tmp);
                    if (mag > 1e-6f) {
                        float scale_norm = umag / mag;
                        for (int i = 0; i < lat_n; i++) vel_latent[i] *= scale_norm;
                    }
                }
            }

            /* Track velocity magnitude */
            { float vmx=0;
              for(int i=0;i<lat_n;i++) if(fabsf(vel_latent[i])>vmx) vmx=fabsf(vel_latent[i]);
              if (r->verbose >= 2)
                fprintf(stderr, "    vel_max=%.2f dt=%.6f step_size=%.2f\n",
                        vmx, sched.dt[step], vmx * fabsf(sched.dt[step])); }

            /* Euler step */
            qimg_sched_step(latent, vel_latent, lat_n, step, &sched);
            free(vel_latent);

            /* Track latent stats */
            { float lmn=latent[0],lmx=latent[0],ls=0,ls2=0;
              int ln=lat_ch*lat_h*lat_w;
              for(int i=0;i<ln;i++){if(latent[i]<lmn)lmn=latent[i];if(latent[i]>lmx)lmx=latent[i];ls+=latent[i];ls2+=latent[i]*latent[i];}
              float mean=ls/ln, std=sqrtf(ls2/ln - mean*mean);
              double step_s = (double)(clock() - step_t0) / CLOCKS_PER_SEC;
              fprintf(stderr, "  step %d/%d  t=%.3f  lat=[%.2f,%.2f] mean=%.3f std=%.3f  %.2fs\n",
                      step+1, n_steps, t_val, lmn, lmx, mean, std, step_s); }
        }
        double denoise_s = (double)(clock() - denoise_t0) / CLOCKS_PER_SEC;
        fprintf(stderr, "Denoising: %.1fs total (%.2fs/step)\n", denoise_s, denoise_s / n_steps);

        free(img_tokens); free(vel_cond); free(vel_uncond);
        free(txt_hidden); free(txt_neg_hidden);

        /* 3. VAE decode (CUDA) */
        fprintf(stderr, "\n[3/3] VAE decode (CUDA)...\n");
        cuda_qimg_load_vae(r, vae_path);
        float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
        t0 = clock();
        cuda_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
        fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
        free(latent);

        /* Save */
        save_ppm("cuda_qimg_output.ppm", rgb, out_h, out_w);
        free(rgb);
    }

    cuda_qimg_free(r);
    return 0;
}
