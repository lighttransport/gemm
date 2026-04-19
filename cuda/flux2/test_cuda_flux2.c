/*
 * test_cuda_flux2.c - CUDA Flux.2 Klein end-to-end test
 *
 * Modes:
 *   --test-init    : Initialize CUDA, compile kernels
 *   --test-load    : Load DiT + VAE weights
 *   --test-dit     : Run single DiT step
 *   --test-vae     : Run VAE decoder
 *   --test-text-enc: Compare CPU and GPU text encoders
 *   --generate     : Full text-to-image pipeline
 *
 * Build:
 *   make test_cuda_flux2
 *   (or: cc -O2 -mavx2 -mfma -I../../common -I.. -o test_cuda_flux2 \
 *        test_cuda_flux2.c ../llm/cuda_llm_runner.c ../cuew.c -lm -ldl -lpthread)
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
#define FLUX2_DIT_IMPLEMENTATION
#define FLUX2_VAE_IMPLEMENTATION
#define FLUX2_TEXT_ENCODER_IMPLEMENTATION
#define CUDA_FLUX2_RUNNER_IMPLEMENTATION

#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/flux2_klein_dit.h"
#include "../../common/flux2_klein_vae.h"
#include "../llm/cuda_llm_runner.h"
#include "../../common/flux2_klein_text_encoder.h"
#include "cuda_flux2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Default weight paths ---- */
static const char *DEFAULT_DIT = "/mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors";
static const char *DEFAULT_VAE = "/mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors";
static const char *DEFAULT_ENC = "/mnt/disk01/models/klein2-4b/text_encoder";
static const char *DEFAULT_TOK = "/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf";

/* ---- PRNG: Box-Muller with pair caching ---- */
static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static float randn(void) {
    if (rng_cached_valid) { rng_cached_valid = 0; return rng_cached; }
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double rv = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(rv * sin(theta));
    rng_cached_valid = 1;
    return (float)(rv * cos(theta));
}

static void save_npy_f32(const char *path, const float *data, int ndims, const int *shape) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    char shape_str[256] = "(";
    for (int d = 0; d < ndims; d++) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%d%s", shape[d], d < ndims-1 ? ", " : "");
        strcat(shape_str, tmp);
    }
    if (ndims == 1) strcat(shape_str, ",");
    strcat(shape_str, ")");
    char dict[512];
    int dlen = snprintf(dict, sizeof(dict),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape_str);
    int total_hdr = 10 + dlen + 1;
    int pad = 64 - (total_hdr % 64);
    if (pad == 64) pad = 0;
    int hdr_data_len = dlen + pad + 1;
    uint8_t magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    magic[8] = (uint8_t)(hdr_data_len & 0xFF);
    magic[9] = (uint8_t)((hdr_data_len >> 8) & 0xFF);
    fwrite(magic, 1, 10, f);
    fwrite(dict, 1, (size_t)dlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= (size_t)shape[d];
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "Saved %s\n", path);
}

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[(size_t)c * h * w + y * w + x] * 0.5f + 0.5f;
                if (v < 0.0f) { v = 0.0f; } if (v > 1.0f) { v = 1.0f; }
                px[c] = (uint8_t)(v * 255.0f + 0.5f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

/* ---- Patchify / unpatchify (same as cpu test) ---- */

static void flux2_patchify(float *out, const float *latent,
                            int lc, int lat_h, int lat_w, int ps) {
    int ph = lat_h / ps, pw = lat_w / ps, pin = lc * ps * ps;
    for (int r = 0; r < ph; r++)
        for (int c = 0; c < pw; c++) {
            float *tok = out + ((size_t)r * pw + c) * pin;
            for (int ch = 0; ch < lc; ch++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++)
                        tok[ch*ps*ps + pr*ps + pc] =
                            latent[(size_t)ch*lat_h*lat_w + (r*ps+pr)*lat_w + (c*ps+pc)];
        }
}

static void flux2_unpatchify(float *latent, const float *tok,
                              int lc, int lat_h, int lat_w, int ps) {
    int ph = lat_h/ps, pw = lat_w/ps, pin = lc*ps*ps;
    for (int r = 0; r < ph; r++)
        for (int c = 0; c < pw; c++) {
            const float *t = tok + ((size_t)r*pw+c)*pin;
            for (int ch = 0; ch < lc; ch++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++)
                        latent[(size_t)ch*lat_h*lat_w+(r*ps+pr)*lat_w+(c*ps+pc)] =
                            t[ch*ps*ps+pr*ps+pc];
        }
}

/* ---- Flux.2 Klein scheduler helpers ---- */

/* Flux.2 Klein uses Flux-style time shift with mu=2.02 (from ComfyUI's
 * ModelSamplingFlux). The Flux shift formula is:
 *   sigma(t) = exp(mu)*t / (1 + (exp(mu)-1)*t)
 * which equals the AuraFlow shift `alpha*t/(1+(alpha-1)*t)` when alpha=exp(mu).
 * So we pass alpha=exp(2.02)≈7.539 to qimg_sched_set_timesteps_comfyui. */
#define FLUX2_KLEIN_SHIFT_MU 2.02f

static void flux2_sched_distilled(qimg_scheduler *s, int n_steps) {
    qimg_sched_init(s);
    qimg_sched_set_timesteps_comfyui(s, n_steps, expf(FLUX2_KLEIN_SHIFT_MU), 1.0f);
}

static void flux2_sched_base(qimg_scheduler *s, int n_steps, int n_img) {
    (void)n_img;
    qimg_sched_init(s);
    qimg_sched_set_timesteps_comfyui(s, n_steps, expf(FLUX2_KLEIN_SHIFT_MU), 1.0f);
}

/* ---- Test modes ---- */

static int test_init(void) {
    fprintf(stderr, "=== CUDA Init Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 2);
    if (!r) return 1;
    fprintf(stderr, "CUDA init OK\n");
    cuda_flux2_free(r);
    return 0;
}

/* ---- Kernel unit tests ----
 * Micro-tests for the new MMA kernels. No model weights required — only the
 * NVRTC module needs to compile cleanly, so this runs quickly. */

/* Test 1: f32_to_bf16_bulk (AVX2 path vs scalar fallback).
 * Pure host-side — checks the AVX2 and scalar branches agree bit-for-bit. */
static int test_kernel_f32_to_bf16_bulk(void) {
    fprintf(stderr, "  [kernel] f32_to_bf16_bulk round-trip... ");
    const int N = 1003;  /* intentionally not a multiple of 8 */
    float *src = (float *)malloc((size_t)N * sizeof(float));
    uint16_t *out = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    uint16_t *ref = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    if (!src || !out || !ref) { free(src); free(out); free(ref); return 1; }
    rng_state = 123456;
    for (int i = 0; i < N; i++) src[i] = randn() * 3.0f;
    f32_to_bf16_bulk(out, src, N);
    /* scalar reference */
    for (int i = 0; i < N; i++) {
        unsigned int b;
        memcpy(&b, &src[i], 4);
        unsigned int r = 0x7FFFu + ((b >> 16) & 1u);
        ref[i] = (uint16_t)((b + r) >> 16);
    }
    int mismatches = 0;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) mismatches++;
    free(src); free(out); free(ref);
    if (mismatches) {
        fprintf(stderr, "FAIL (%d/%d mismatches)\n", mismatches, N);
        return 1;
    }
    fprintf(stderr, "OK (%d elements)\n", N);
    return 0;
}

/* Test 2: gemm_bf16_f32 vs CPU F32 reference.
 * Tolerates BF16 rounding (~1/256 relative). */
static int test_kernel_gemm_bf16(cuda_flux2_runner *r) {
    fprintf(stderr, "  [kernel] gemm_bf16_f32 vs CPU F32 ref... ");
    if (!r->fn_gemm_bf16) {
        fprintf(stderr, "SKIP (kernel not loaded)\n");
        return 0;
    }
    /* Small dims that exercise M/N/K padding without blowing up alloc */
    const int M = 48, K = 80, N = 72;
    size_t n_w = (size_t)N * K, n_x = (size_t)M * K, n_y = (size_t)M * N;

    float *W = (float *)malloc(n_w * sizeof(float));
    float *X = (float *)malloc(n_x * sizeof(float));
    float *Y_gpu = (float *)malloc(n_y * sizeof(float));
    float *Y_cpu = (float *)malloc(n_y * sizeof(float));
    uint16_t *W_bf = (uint16_t *)malloc(n_w * sizeof(uint16_t));
    if (!W || !X || !Y_gpu || !Y_cpu || !W_bf) {
        free(W); free(X); free(Y_gpu); free(Y_cpu); free(W_bf); return 1;
    }
    rng_state = 7654321;
    for (size_t i = 0; i < n_w; i++) W[i] = randn() * 0.1f;
    for (size_t i = 0; i < n_x; i++) X[i] = randn() * 0.3f;
    f32_to_bf16_bulk(W_bf, W, (int)n_w);

    /* CPU F32 ref (using BF16-rounded W to match what the kernel sees) */
    float *W_deq = (float *)malloc(n_w * sizeof(float));
    for (size_t i = 0; i < n_w; i++) {
        unsigned int b = ((unsigned int)W_bf[i]) << 16;
        memcpy(&W_deq[i], &b, 4);
    }
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += X[(size_t)m * K + k] * W_deq[(size_t)n * K + k];
            Y_cpu[(size_t)m * N + n] = s;
        }
    }
    free(W_deq);

    /* Upload, launch, download */
    CUdeviceptr d_W = gpu_upload_bytes(W_bf, n_w * sizeof(uint16_t));
    CUdeviceptr d_X = gpu_upload_f32(X, (int)n_x);
    CUdeviceptr d_Y;
    cuMemAlloc(&d_Y, n_y * sizeof(float));
    op_gemm_bf16(r, d_Y, d_W, d_X, 0, N, K, M);
    cuCtxSynchronize();
    cuMemcpyDtoH(Y_gpu, d_Y, n_y * sizeof(float));
    cuMemFree(d_W); cuMemFree(d_X); cuMemFree(d_Y);

    /* Compare: BF16 has ~1/256 relative precision. For near-zero reference
     * values, a direct abs threshold based on the expected RMS magnitude of
     * the output is more meaningful than relative error. */
    float max_abs = 0.0f;   /* max absolute diff */
    float rms_cpu = 0.0f;   /* RMS of reference values (for scale) */
    for (size_t i = 0; i < n_y; i++) {
        float d = fabsf(Y_gpu[i] - Y_cpu[i]);
        if (d > max_abs) max_abs = d;
        rms_cpu += Y_cpu[i] * Y_cpu[i];
    }
    rms_cpu = sqrtf(rms_cpu / (float)n_y);
    free(W); free(X); free(Y_gpu); free(Y_cpu); free(W_bf);
    /* Allow max_abs up to 3% of the reference RMS (BF16 rounds both operands). */
    float tol = 0.03f * rms_cpu;
    if (max_abs > tol) {
        fprintf(stderr, "FAIL (max_abs=%.6g rms_cpu=%.6g tol=%.6g)\n", max_abs, rms_cpu, tol);
        return 1;
    }
    fprintf(stderr, "OK (max_abs=%.6g rms_cpu=%.6g)\n", max_abs, rms_cpu);
    return 0;
}

/* Test 3: flash_attn_fp8 MMA vs flash_attn_fp8_ref scalar.
 * Both paths share the same per-tensor FP8 quantization, so their outputs
 * should match within a small tolerance (FP8 quant noise is identical; only
 * the numerical order of the mma reductions differs). */
static int test_kernel_fp8_attn(cuda_flux2_runner *r) {
    fprintf(stderr, "  [kernel] flash_attn_fp8 MMA vs scalar ref... ");
    if (!r->fn_flash_attn_fp8 || !r->fn_flash_attn_fp8_ref ||
        !r->fn_quant_fp8 || !r->fn_reduce_max_abs) {
        fprintf(stderr, "SKIP (kernels not loaded)\n");
        return 0;
    }
    const int n_tok = 48;
    const int n_heads = 4;
    const int head_dim = 128;  /* kernel hardcodes this */
    const int dim = n_heads * head_dim;
    size_t n_elems = (size_t)n_tok * dim;

    float *Q = (float *)malloc(n_elems * sizeof(float));
    float *K = (float *)malloc(n_elems * sizeof(float));
    float *V = (float *)malloc(n_elems * sizeof(float));
    float *out_mma = (float *)malloc(n_elems * sizeof(float));
    float *out_ref = (float *)malloc(n_elems * sizeof(float));
    if (!Q || !K || !V || !out_mma || !out_ref) {
        free(Q); free(K); free(V); free(out_mma); free(out_ref); return 1;
    }
    rng_state = 999;
    for (size_t i = 0; i < n_elems; i++) Q[i] = randn() * 0.7f;
    for (size_t i = 0; i < n_elems; i++) K[i] = randn() * 0.7f;
    for (size_t i = 0; i < n_elems; i++) V[i] = randn() * 0.7f;

    CUdeviceptr d_Q = gpu_upload_f32(Q, (int)n_elems);
    CUdeviceptr d_K = gpu_upload_f32(K, (int)n_elems);
    CUdeviceptr d_V = gpu_upload_f32(V, (int)n_elems);
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, n_elems * sizeof(float));

    /* Run MMA path */
    int saved_attn = r->use_fp8_attn;
    r->use_fp8_attn = 1;
    unsetenv("FLUX2_FP8_ATTN_REF");
    op_attn(r, d_out, d_Q, d_K, d_V, n_tok, n_heads, head_dim);
    cuCtxSynchronize();
    cuMemcpyDtoH(out_mma, d_out, n_elems * sizeof(float));

    /* Run scalar ref path */
    setenv("FLUX2_FP8_ATTN_REF", "1", 1);
    op_attn(r, d_out, d_Q, d_K, d_V, n_tok, n_heads, head_dim);
    cuCtxSynchronize();
    cuMemcpyDtoH(out_ref, d_out, n_elems * sizeof(float));
    unsetenv("FLUX2_FP8_ATTN_REF");
    r->use_fp8_attn = saved_attn;

    cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_out);

    /* Compare — both paths quantize to the same FP8, so max diff should be small */
    float max_diff = 0.0f, mean_diff = 0.0f;
    for (size_t i = 0; i < n_elems; i++) {
        float d = fabsf(out_mma[i] - out_ref[i]);
        if (d > max_diff) max_diff = d;
        mean_diff += d;
    }
    mean_diff /= (float)n_elems;
    free(Q); free(K); free(V); free(out_mma); free(out_ref);
    /* Tolerance: online softmax ordering + FP8 P-requant can differ slightly */
    if (max_diff > 0.1f) {
        fprintf(stderr, "FAIL (max_diff=%.6g mean_diff=%.6g)\n", max_diff, mean_diff);
        return 1;
    }
    fprintf(stderr, "OK (max_diff=%.6g mean_diff=%.6g)\n", max_diff, mean_diff);
    return 0;
}

static int test_kernels(void) {
    fprintf(stderr, "=== Kernel Unit Tests ===\n");
    int fail = 0;
    /* Host-side test first — doesn't need the runner */
    fail += test_kernel_f32_to_bf16_bulk();

    cuda_flux2_runner *r = cuda_flux2_init(0, 0);
    if (!r) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    fail += test_kernel_gemm_bf16(r);
    fail += test_kernel_fp8_attn(r);
    cuda_flux2_free(r);

    if (fail) {
        fprintf(stderr, "=== Kernel Tests: %d FAIL(S) ===\n", fail);
        return 1;
    }
    fprintf(stderr, "=== Kernel Tests: all OK ===\n");
    return 0;
}

static int test_load(const char *dit_path, const char *vae_path) {
    fprintf(stderr, "=== CUDA Load Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 1);
    if (!r) return 1;
    if (cuda_flux2_load_dit(r, dit_path) != 0) { cuda_flux2_free(r); return 1; }
    if (cuda_flux2_load_vae(r, vae_path) != 0) { cuda_flux2_free(r); return 1; }
    fprintf(stderr, "Load OK\n");
    cuda_flux2_free(r);
    return 0;
}

static int test_dit(const char *dit_path, const char *enc_path, const char *tok_path,
                    const char *prompt, int lat_h, int lat_w, int n_txt,
                    float img_scale, float txt_scale, float timestep,
                    int use_real_text, int use_real_latent) {
    fprintf(stderr, "=== CUDA DiT Step Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 1);
    if (!r) return 1;
    if (cuda_flux2_load_dit(r, dit_path) != 0) { cuda_flux2_free(r); return 1; }

    int ps = 2;
    int n_img = (lat_h/ps) * (lat_w/ps);
    int pin = r->pin;
    int txt_dim = r->txt_dim;

    float *img_tok = (float *)calloc((size_t)n_img * pin, sizeof(float));
    float *txt_tok = NULL;
    float *vel_out = (float *)malloc((size_t)n_img * pin * sizeof(float));
    if (!img_tok || !vel_out) {
        free(img_tok); free(vel_out); cuda_flux2_free(r); return 1;
    }

    if (use_real_latent) {
        int lc = FLUX2_VAE_LATENT_CHANNELS;
        size_t lat_sz = (size_t)lc * lat_h * lat_w;
        float *latent = (float *)malloc(lat_sz * sizeof(float));
        if (!latent) {
            free(img_tok); free(vel_out); cuda_flux2_free(r); return 1;
        }
        for (size_t i = 0; i < lat_sz; i++) latent[i] = randn() * img_scale;
        flux2_patchify(img_tok, latent, lc, lat_h, lat_w, ps);
        free(latent);
        fprintf(stderr, "Using real latent patchify path: lc=%d lat=%dx%d ps=%d\n",
                lc, lat_w, lat_h, ps);
    } else {
        for (int i = 0; i < n_img*pin; i++) img_tok[i] = randn() * img_scale;
    }
    if (use_real_text) {
        flux2_text_enc *enc = flux2_text_enc_load_safetensors(enc_path, tok_path);
        if (!enc) {
            free(img_tok); free(vel_out); cuda_flux2_free(r); return 1;
        }
        txt_tok = flux2_text_enc_encode(enc, prompt, &n_txt);
        flux2_text_enc_free(enc);
        if (!txt_tok) {
            free(img_tok); free(vel_out); cuda_flux2_free(r); return 1;
        }
        txt_dim = r->txt_dim;
        fprintf(stderr, "Using real text hidden states: n_txt=%d txt_dim=%d\n", n_txt, txt_dim);
    } else {
        txt_tok = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
        if (!txt_tok) {
            free(img_tok); free(vel_out); cuda_flux2_free(r); return 1;
        }
        for (int i = 0; i < n_txt*txt_dim; i++) txt_tok[i] = randn() * txt_scale;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cuda_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, timestep, 0.0f, vel_out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    fprintf(stderr, "DiT step: %.2f s (n_img=%d, n_txt=%d, pin=%d, img_scale=%.3f, txt_scale=%.3f, t=%.3f)\n",
            dt, n_img, n_txt, pin, img_scale, txt_scale, timestep);

    float mn=vel_out[0], mx=vel_out[0];
    for (int i=0;i<n_img*pin;i++) { if(vel_out[i]<mn) mn=vel_out[i]; if(vel_out[i]>mx) mx=vel_out[i]; }
    fprintf(stderr, "GPU velocity: min=%.4f max=%.4f\n", mn, mx);

    float *cpu_out = (float *)malloc((size_t)n_img * pin * sizeof(float));
    flux2_dit_forward(cpu_out, img_tok, n_img, txt_tok, n_txt, timestep, r->dit, 1);
    mn=cpu_out[0]; mx=cpu_out[0];
    for (int i=0;i<n_img*pin;i++) { if(cpu_out[i]<mn) mn=cpu_out[i]; if(cpu_out[i]>mx) mx=cpu_out[i]; }
    fprintf(stderr, "CPU velocity: min=%.4f max=%.4f\n", mn, mx);
    float max_diff = 0, sum_diff = 0;
    for (int i=0;i<n_img*pin;i++) { float d=fabsf(vel_out[i]-cpu_out[i]); if(d>max_diff) max_diff=d; sum_diff+=d; }
    fprintf(stderr, "GPU vs CPU: max_diff=%.6f mean_diff=%.6f\n", max_diff, sum_diff/(n_img*pin));
    for (int i=0;i<5;i++)
        fprintf(stderr, "  [%d] GPU=%.6f CPU=%.6f\n", i, vel_out[i], cpu_out[i]);
    free(cpu_out);

    free(img_tok); free(txt_tok); free(vel_out);
    cuda_flux2_free(r); return 0;
}

static int test_vae(const char *vae_path, int lat_h, int lat_w) {
    fprintf(stderr, "=== CUDA VAE Decode Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 1);
    if (!r) return 1;
    if (cuda_flux2_load_vae(r, vae_path) != 0) { cuda_flux2_free(r); return 1; }

    int lc = FLUX2_VAE_LATENT_CHANNELS;
    size_t lat_sz = (size_t)lc * lat_h * lat_w;
    int out_h = lat_h * 8, out_w = lat_w * 8;
    float *latent = (float *)malloc(lat_sz * sizeof(float));
    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    float *cpu_rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    if (!latent || !rgb || !cpu_rgb) {
        free(latent); free(rgb); free(cpu_rgb); cuda_flux2_free(r); return 1;
    }

    rng_state = 12345;
    for (size_t i = 0; i < lat_sz; i++) latent[i] = randn() * 0.5f;

    /* Save latent for external comparison */
    { int sh[] = {lc, lat_h, lat_w}; save_npy_f32("cuda_flux2_vae_latent.npy", latent, 3, sh); }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cuda_flux2_vae_decode(r, latent, lat_h, lat_w, rgb) != 0) {
        free(latent); free(rgb); cuda_flux2_free(r); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    fprintf(stderr, "VAE decode: %.2f s (%dx%d -> %dx%d)\n", dt, lat_w, lat_h, out_w, out_h);

    float mn = rgb[0], mx = rgb[0];
    for (int i = 0; i < 3 * out_h * out_w; i++) {
        if (rgb[i] < mn) mn = rgb[i];
        if (rgb[i] > mx) mx = rgb[i];
    }
    fprintf(stderr, "GPU rgb: min=%.4f max=%.4f\n", mn, mx);
    { int sh[] = {3, out_h, out_w}; save_npy_f32("cuda_flux2_vae_rgb.npy", rgb, 3, sh); }

    flux2_vae_decode(cpu_rgb, latent, lat_h, lat_w, r->vae);
    float max_diff = 0.0f, sum_diff = 0.0f;
    for (int i = 0; i < 3 * out_h * out_w; i++) {
        float d = fabsf(rgb[i] - cpu_rgb[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    fprintf(stderr, "GPU vs CPU VAE: max_diff=%.6f mean_diff=%.6f\n",
            max_diff, sum_diff / (float)(3 * out_h * out_w));
    save_ppm("cuda_flux2_vae_output.ppm", rgb, out_h, out_w);

    free(latent);
    free(rgb);
    free(cpu_rgb);
    cuda_flux2_free(r);
    return 0;
}

static int test_text_enc(const char *enc_path, const char *tok_path,
                         const char *prompt, int device_id) {
    fprintf(stderr, "=== Flux.2 Klein Text Encoder Test ===\n");
    fprintf(stderr, "Prompt: '%s'\n", prompt);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_text_enc *cpu = flux2_text_enc_load_safetensors(enc_path, tok_path);
    if (!cpu) return 1;
    int n_tok_cpu = 0;
    float *cpu_hidden = flux2_text_enc_encode(cpu, prompt, &n_tok_cpu);
    int n_embd = cpu->n_embd;
    flux2_text_enc_free(cpu);
    if (!cpu_hidden) return 1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "CPU text enc: %.2f s (%d tokens × %d)\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9, n_tok_cpu, n_embd);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    fprintf(stderr, "GPU text encoder weights: %s\n", enc_path);
    flux2_text_enc *gpu = flux2_text_enc_load_gpu(enc_path, tok_path, device_id);
    if (!gpu) { free(cpu_hidden); return 1; }
    int n_tok_gpu = 0;
    float *gpu_hidden = flux2_text_enc_encode(gpu, prompt, &n_tok_gpu);
    if (!gpu_hidden) { free(cpu_hidden); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "GPU text enc: %.2f s (%d tokens × %d)\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9, n_tok_gpu, n_embd);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_text_enc *gpu_cached = flux2_text_enc_load_gpu(enc_path, tok_path, device_id);
    if (!gpu_cached) { free(cpu_hidden); free(gpu_hidden); flux2_text_enc_free(gpu); return 1; }
    int n_tok_gpu_cached = 0;
    float *gpu_hidden_cached = flux2_text_enc_encode(gpu_cached, prompt, &n_tok_gpu_cached);
    if (!gpu_hidden_cached) {
        free(cpu_hidden); free(gpu_hidden); flux2_text_enc_free(gpu_cached); flux2_text_enc_free(gpu); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "GPU text enc (cached): %.2f s (%d tokens × %d)\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9, n_tok_gpu_cached, n_embd);

    if (n_tok_cpu != n_tok_gpu) {
        fprintf(stderr, "Token count mismatch: CPU=%d GPU=%d\n", n_tok_cpu, n_tok_gpu);
        free(cpu_hidden);
        free(gpu_hidden);
        free(gpu_hidden_cached);
        flux2_text_enc_free(gpu_cached);
        flux2_text_enc_free(gpu);
        return 1;
    }
    if (n_tok_cpu != n_tok_gpu_cached) {
        fprintf(stderr, "Token count mismatch: CPU=%d GPU(cached)=%d\n", n_tok_cpu, n_tok_gpu_cached);
        free(cpu_hidden);
        free(gpu_hidden);
        free(gpu_hidden_cached);
        flux2_text_enc_free(gpu_cached);
        flux2_text_enc_free(gpu);
        return 1;
    }

    size_t total = (size_t)n_tok_cpu * n_embd;
    float max_diff = 0.0f, sum_diff = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < total; i++) {
        float d = fabsf(cpu_hidden[i] - gpu_hidden[i]);
        if (d > max_diff) {
            max_diff = d;
            max_idx = i;
        }
        sum_diff += d;
    }
    fprintf(stderr, "GPU vs CPU text enc: max_diff=%.6f mean_diff=%.6f\n",
            max_diff, sum_diff / (float)total);
    fprintf(stderr, "  max_idx=%zu GPU=%.6f CPU=%.6f\n",
            max_idx, gpu_hidden[max_idx], cpu_hidden[max_idx]);

    float max_diff_cached = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float d = fabsf(gpu_hidden[i] - gpu_hidden_cached[i]);
        if (d > max_diff_cached) max_diff_cached = d;
    }
    fprintf(stderr, "GPU cached vs first load: max_diff=%.6f\n", max_diff_cached);

    for (int i = 0; i < 6 && i < n_embd; i++) {
        fprintf(stderr, "  tok0[%d] GPU=%.6f CPU=%.6f\n", i, gpu_hidden[i], cpu_hidden[i]);
    }

    free(cpu_hidden);
    free(gpu_hidden);
    free(gpu_hidden_cached);
    flux2_text_enc_free(gpu_cached);
    flux2_text_enc_free(gpu);
    return 0;
}

static int run_generate_once(cuda_flux2_runner *r,
                         const char *prompt,
                         const float *txt_hidden, int n_txt, int enc_embd,
                         int out_h, int out_w, int n_steps,
                         uint64_t seed, int is_distilled, float cfg_scale,
                         const char *out_path) {
    struct timespec t_start, t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    fprintf(stderr, "Prompt: '%s'\n", prompt);
    fprintf(stderr, "Output: %dx%d, %d steps, seed=%llu, distilled=%d, cfg=%.1f\n",
            out_w, out_h, n_steps, (unsigned long long)seed, is_distilled, cfg_scale);

    fprintf(stderr, "\n[1/3] Text encoding...\n");
    fprintf(stderr, "Text enc: cached (%d tokens × %d)\n", n_txt, enc_embd);

    /* Latent setup */
    int lat_h = out_h / 8, lat_w = out_w / 8;
    int lc = FLUX2_VAE_LATENT_CHANNELS;
    int ps = (r->pin == lc * 4) ? 2 : 1;
    int n_img = (lat_h / ps) * (lat_w / ps);
    int pin = r->pin;

    fprintf(stderr, "Latent: [%d, %d, %d], ps=%d, n_img=%d, pin=%d, n_txt=%d, txt_dim=%d\n",
            lc, lat_h, lat_w, ps, n_img, pin, n_txt, r->txt_dim);

    rng_state = seed;
    size_t lat_sz = (size_t)lc * lat_h * lat_w;
    float *latent = (float *)malloc(lat_sz * sizeof(float));
    for (size_t i = 0; i < lat_sz; i++) latent[i] = randn();

    float *img_tok = (float *)malloc((size_t)n_img * pin * sizeof(float));
    float *vel_out = (float *)malloc((size_t)n_img * pin * sizeof(float));

    qimg_scheduler sched;
    if (is_distilled) flux2_sched_distilled(&sched, n_steps);
    else              flux2_sched_base(&sched, n_steps, n_img);

    /* Denoising loop */
    fprintf(stderr, "\n[2/3] Denoising (%d steps)...\n", n_steps);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < n_steps; step++) {
        float t_sigma = sched.timesteps[step];
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        flux2_patchify(img_tok, latent, lc, lat_h, lat_w, ps);

        if (is_distilled || cfg_scale <= 1.0f) {
            if (cuda_flux2_dit_step(r, img_tok, n_img, txt_hidden, n_txt,
                                    t_sigma, 0.0f, vel_out) != 0) {
                free(img_tok); free(vel_out); free(latent);
                return 1;
            }
            /* Velocity stats */
            float vmn=vel_out[0], vmx=vel_out[0]; double vsum=0;
            for(int i=0;i<n_img*pin;i++){if(vel_out[i]<vmn)vmn=vel_out[i];if(vel_out[i]>vmx)vmx=vel_out[i];vsum+=vel_out[i];}
            fprintf(stderr, "    vel: min=%.4f max=%.4f mean=%.6f\n", vmn, vmx, vsum/(n_img*pin));
        } else {
            float *txt_uncond = (float *)calloc((size_t)n_txt * r->txt_dim, sizeof(float));
            float *vel_uncond = (float *)malloc((size_t)n_img * pin * sizeof(float));
            if (cuda_flux2_dit_step(r, img_tok, n_img, txt_uncond, n_txt,
                                    t_sigma, 0.0f, vel_uncond) != 0 ||
                cuda_flux2_dit_step(r, img_tok, n_img, txt_hidden, n_txt,
                                    t_sigma, 0.0f, vel_out) != 0) {
                free(txt_uncond); free(vel_uncond);
                free(img_tok); free(vel_out); free(latent);
                return 1;
            }
            for (int i = 0; i < n_img * pin; i++)
                vel_out[i] = vel_uncond[i] + cfg_scale * (vel_out[i] - vel_uncond[i]);
            free(txt_uncond); free(vel_uncond);
        }

        {
            CUresult err = cuCtxSynchronize();
            if (err != CUDA_SUCCESS) {
                fprintf(stderr, "generate: cuCtxSynchronize failed after DiT step %d/%d (%d)\n",
                        step + 1, n_steps, (int)err);
                free(img_tok); free(vel_out); free(latent);
                return 1;
            }
        }

        float *vel_lat = (float *)calloc(lat_sz, sizeof(float));
        flux2_unpatchify(vel_lat, vel_out, lc, lat_h, lat_w, ps);

        /* Save raw velocity for comparison */
        {
            char path[64];
            snprintf(path, sizeof(path), "cuda_flux2_vel%d.npy", step);
            int sh3[] = {(int)lc, lat_h, lat_w};
            save_npy_f32(path, vel_lat, 3, sh3);
        }

        qimg_sched_step(latent, vel_lat, (int)lat_sz, step, &sched);
        free(vel_lat);

        /* Save per-step latent for comparison */
        {
            char path[64];
            snprintf(path, sizeof(path), "cuda_flux2_step%d.npy", step);
            int sh3[] = {(int)lc, lat_h, lat_w};
            save_npy_f32(path, latent, 3, sh3);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double sdt = (ts1.tv_sec-ts0.tv_sec)+(ts1.tv_nsec-ts0.tv_nsec)*1e-9;
        fprintf(stderr, "  step %d/%d  sigma=%.4f  %.1f s\n", step+1, n_steps, t_sigma, sdt);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Denoising: %.1f s\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    free(img_tok); free(vel_out);

    /* Save final latent and text hidden states for comparison */
    {
        int sh3[] = {(int)lc, lat_h, lat_w};
        save_npy_f32("cuda_flux2_latent_final.npy", latent, 3, sh3);
    }
    if (txt_hidden) {
        int sh2[] = {n_txt, (int)r->txt_dim};
        save_npy_f32("cuda_flux2_text_hidden.npy", txt_hidden, 2, sh2);
    }

    /* VAE decode */
    fprintf(stderr, "\n[3/3] VAE decoding...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    if (cuda_flux2_vae_decode(r, latent, lat_h, lat_w, rgb) != 0) {
        free(rgb); free(latent);
        return 1;
    }
    {
        CUresult err = cuCtxSynchronize();
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "generate: cuCtxSynchronize failed after VAE decode (%d)\n", (int)err);
            free(rgb); free(latent);
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "VAE: %.1f s\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    save_ppm(out_path, rgb, out_h, out_w);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec-t_start.tv_sec)+(t1.tv_nsec-t_start.tv_nsec)*1e-9;
    fprintf(stderr, "\nTotal: %.1f s\n", total);

    free(rgb); free(latent);
    return 0;
}

static int run_generate(const char *dit_path, const char *vae_path,
                         const char *enc_path, const char *tok_path,
                         const char *prompt,
                         int out_h, int out_w, int n_steps,
                         uint64_t seed, int is_distilled, float cfg_scale,
                         int use_gpu_enc, int device_id, int repeat) {
    fprintf(stderr, "\n=== Flux.2 Klein GPU Pipeline ===\n");
    fprintf(stderr, "Runs: %d\n", repeat);
    if (use_gpu_enc) fprintf(stderr, "GPU text encoder weights: %s\n", enc_path);

    fprintf(stderr, "\n[setup] Text encoder (%s)...\n", use_gpu_enc ? "GPU" : "CPU");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_text_enc *enc = use_gpu_enc
        ? flux2_text_enc_load_gpu(enc_path, tok_path, device_id)
        : flux2_text_enc_load_safetensors(enc_path, tok_path);
    if (!enc) return 1;

    int n_txt = 0;
    int enc_embd = enc->n_embd;
    float *txt_hidden_raw = flux2_text_enc_encode(enc, prompt, &n_txt);
    flux2_text_enc_free(enc);
    if (!txt_hidden_raw) return 1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Shared text enc: %.1f s (%d real tokens × %d)\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9, n_txt, enc_embd);

    /* Front-pad text to 512 tokens with zeros (matches ComfyUI Flux2.extra_conds) */
    const int FLUX2_KLEIN_TXT_LEN = 512;
    float *txt_hidden = (float *)calloc((size_t)FLUX2_KLEIN_TXT_LEN * enc_embd, sizeof(float));
    int pad_front = FLUX2_KLEIN_TXT_LEN - n_txt;
    if (pad_front < 0) pad_front = 0;
    int real_txt = (n_txt < FLUX2_KLEIN_TXT_LEN) ? n_txt : FLUX2_KLEIN_TXT_LEN;
    memcpy(txt_hidden + (size_t)pad_front * enc_embd, txt_hidden_raw,
           (size_t)real_txt * enc_embd * sizeof(float));
    free(txt_hidden_raw);
    n_txt = FLUX2_KLEIN_TXT_LEN;
    fprintf(stderr, "Padded text to %d tokens (front-pad zeros)\n", n_txt);

    fprintf(stderr, "\n[setup] Init CUDA + load DiT + VAE...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cuda_flux2_runner *r = cuda_flux2_init(device_id, 1);
    if (!r) { free(txt_hidden); return 1; }
    if (cuda_flux2_load_dit(r, dit_path) != 0) { free(txt_hidden); cuda_flux2_free(r); return 1; }
    if (cuda_flux2_load_vae(r, vae_path) != 0) { free(txt_hidden); cuda_flux2_free(r); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Shared init+load: %.1f s\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    int rc = 0;
    for (int i = 0; i < repeat; i++) {
        char out_path[256];
        if (repeat == 1) snprintf(out_path, sizeof(out_path), "cuda_flux2_output.ppm");
        else snprintf(out_path, sizeof(out_path), "cuda_flux2_output_%02d.ppm", i);
        fprintf(stderr, "\n--- Run %d/%d ---\n", i + 1, repeat);
        rc = run_generate_once(r, prompt, txt_hidden, n_txt, enc_embd, out_h, out_w, n_steps,
                               seed + (uint64_t)i, is_distilled, cfg_scale, out_path);
        if (rc != 0) break;
    }

    free(txt_hidden);
    cuda_flux2_free(r);
    return rc;
}

int main(int argc, char **argv) {
    const char *dit_path = DEFAULT_DIT;
    const char *vae_path = DEFAULT_VAE;
    const char *enc_path = DEFAULT_ENC;
    const char *tok_path = DEFAULT_TOK;
    const char *prompt = "a red apple on a white table";
    const char *mode = NULL;

    int out_h = 256, out_w = 256, n_steps = 4, repeat = 1, n_txt = 8;
    int is_distilled = 1, use_gpu_enc = 0, device_id = 0;
    float cfg_scale = 1.0f;
    float img_scale = 0.1f, txt_scale = 0.1f;
    float timestep = 750.0f;
    int use_real_text = 0, use_real_latent = 0;
    uint64_t seed = 42;
    int verbose = 1;
    (void)verbose;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--test-init") == 0) mode = "init";
        else if (strcmp(argv[i], "--test-load") == 0) mode = "load";
        else if (strcmp(argv[i], "--test-dit")  == 0) mode = "dit";
        else if (strcmp(argv[i], "--test-vae")  == 0) mode = "vae";
        else if (strcmp(argv[i], "--test-text-enc") == 0) mode = "text";
        else if (strcmp(argv[i], "--test-kernels") == 0) mode = "kernels";
        else if (strcmp(argv[i], "--generate")  == 0) mode = "gen";
        else if (strcmp(argv[i], "--base")       == 0) { is_distilled = 0; n_steps = 20; }
        else if (strcmp(argv[i], "--distilled")  == 0) { is_distilled = 1; n_steps = 4; }
        else if (strcmp(argv[i], "--gpu-enc")    == 0) use_gpu_enc = 1;
        else if (strcmp(argv[i], "--dit")    == 0 && i+1<argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae")    == 0 && i+1<argc) vae_path = argv[++i];
        else if (strcmp(argv[i], "--enc")    == 0 && i+1<argc) enc_path = argv[++i];
        else if (strcmp(argv[i], "--tok")    == 0 && i+1<argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1<argc) prompt   = argv[++i];
        else if (strcmp(argv[i], "--height") == 0 && i+1<argc) out_h    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width")  == 0 && i+1<argc) out_w    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps")  == 0 && i+1<argc) n_steps  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i+1<argc) repeat   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-txt")  == 0 && i+1<argc) n_txt    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--img-scale") == 0 && i+1<argc) img_scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--txt-scale") == 0 && i+1<argc) txt_scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--timestep") == 0 && i+1<argc) timestep = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--real-text") == 0) use_real_text = 1;
        else if (strcmp(argv[i], "--real-latent") == 0) use_real_latent = 1;
        else if (strcmp(argv[i], "--seed")   == 0 && i+1<argc) seed     = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--cfg")    == 0 && i+1<argc) cfg_scale= (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i+1<argc) device_id= atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose")== 0 && i+1<argc) verbose  = atoi(argv[++i]);
    }

    if (!mode) {
        fprintf(stderr,
            "Usage: %s [--test-init|--test-load|--test-dit|--test-vae|--test-text-enc|--test-kernels|--generate]\n"
            "          [--dit PATH] [--vae PATH] [--enc PATH]\n"
            "          [--prompt TEXT] [--height H] [--width W]\n"
            "          [--steps N] [--repeat N] [--n-txt N] [--img-scale S] [--txt-scale S] [--timestep T] [--real-text] [--real-latent] [--seed S] [--cfg SCALE]\n"
            "          [--base|--distilled] [--gpu-enc] [--device N]\n",
            argv[0]);
        return 1;
    }

    if (strcmp(mode, "init") == 0) return test_init();
    if (strcmp(mode, "kernels") == 0) return test_kernels();
    if (strcmp(mode, "load") == 0) return test_load(dit_path, vae_path);
    if (strcmp(mode, "dit")  == 0) return test_dit(dit_path, enc_path, tok_path, prompt,
                                                   out_h/16, out_w/16, n_txt,
                                                   img_scale, txt_scale, timestep,
                                                   use_real_text, use_real_latent);
    if (strcmp(mode, "vae")  == 0) return test_vae(vae_path, out_h/8, out_w/8);
    if (strcmp(mode, "text") == 0) return test_text_enc(enc_path, tok_path, prompt, device_id);
    if (strcmp(mode, "gen")  == 0)
        return run_generate(dit_path, vae_path, enc_path, tok_path, prompt,
                            out_h, out_w, n_steps, seed, is_distilled,
                            cfg_scale, use_gpu_enc, device_id, repeat);

    fprintf(stderr, "Unknown mode: %s\n", mode);
    return 1;
}
