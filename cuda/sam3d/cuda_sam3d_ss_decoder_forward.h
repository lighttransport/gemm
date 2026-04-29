/* Composed GPU forward for SAM3D/TRELLIS.2 SS-VAE decoder. */

#ifndef CUDA_SAM3D_SS_DECODER_FORWARD_H_
#define CUDA_SAM3D_SS_DECODER_FORWARD_H_

#include "../cuew.h"
#include "cuda_sam3d_ss_decoder_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipFunction_t conv3d;
    hipFunction_t cln;
    hipFunction_t silu;
    hipFunction_t pixshuf;
    hipFunction_t resadd;
} cs3d_ssdec_fns;

typedef struct {
    CUdeviceptr a, b, c;
    size_t bytes;
} cs3d_ssdec_ws;

int  cs3d_ssdec_fns_init(cs3d_ssdec_fns *f, hipModule_t mod);
int  cs3d_ssdec_ws_alloc(cs3d_ssdec_ws *ws);
void cs3d_ssdec_ws_free(cs3d_ssdec_ws *ws);
int  cs3d_ssdec_forward(const cs3d_ssdec_gpu *g, const cs3d_ssdec_fns *f,
                        cs3d_ssdec_ws *ws, CUdeviceptr d_latent,
                        CUdeviceptr d_out_logits, int verbose);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_SS_DECODER_FORWARD_H_ */

#ifdef CUDA_SAM3D_SS_DECODER_FORWARD_IMPLEMENTATION

#include <stdio.h>
#include <string.h>
#include <time.h>

static double cs3d_ssdec_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cs3d_ssdec_sync_stage(const char *stage)
{
    CUresult err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        const char *es = "?";
        cuGetErrorString(err, &es);
        fprintf(stderr, "cs3d_ssdec: %s failed: %s (%d)\n", stage, es, (int)err);
        return -1;
    }
    return 0;
}

int cs3d_ssdec_fns_init(cs3d_ssdec_fns *f, hipModule_t mod)
{
    if (!f) return -1;
    memset(f, 0, sizeof(*f));
#define GET_(name, sym) \
    if (cuModuleGetFunction(&f->name, mod, sym) != CUDA_SUCCESS) { \
        fprintf(stderr, "cs3d_ssdec: missing kernel %s\n", sym); return -1; \
    }
    GET_(conv3d,  "conv3d_k3_pad1_f32");
    GET_(cln,     "channel_layernorm_3d_f32");
    GET_(silu,    "silu_inplace_f32");
    GET_(pixshuf, "pixel_shuffle_3d_f32");
    GET_(resadd,  "residual_add_f32");
#undef GET_
    return 0;
}

int cs3d_ssdec_ws_alloc(cs3d_ssdec_ws *ws)
{
    if (!ws) return -1;
    memset(ws, 0, sizeof(*ws));
    ws->bytes = (size_t)32 * 64 * 64 * 64 * sizeof(float);
    if (cuMemAlloc(&ws->a, ws->bytes) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&ws->b, ws->bytes) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&ws->c, ws->bytes) != CUDA_SUCCESS) goto fail;
    return 0;
fail:
    cs3d_ssdec_ws_free(ws);
    return -1;
}

void cs3d_ssdec_ws_free(cs3d_ssdec_ws *ws)
{
    if (!ws) return;
    if (ws->a) cuMemFree(ws->a);
    if (ws->b) cuMemFree(ws->b);
    if (ws->c) cuMemFree(ws->c);
    memset(ws, 0, sizeof(*ws));
}

static int cs3d_launch_conv3d(const cs3d_ssdec_fns *f,
                              CUdeviceptr src, CUdeviceptr dst,
                              CUdeviceptr w, CUdeviceptr b,
                              int Ci, int Co, int D, int H, int W)
{
    int total = Co * D * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    void *args[] = { &src, &dst, &w, &b, &Ci, &Co, &D, &H, &W };
    return cuLaunchKernel(f->conv3d, blocks, 1, 1, threads, 1, 1,
                          0, 0, args, NULL) == CUDA_SUCCESS ? 0 : -1;
}

static int cs3d_launch_cln(const cs3d_ssdec_fns *f,
                           CUdeviceptr src, CUdeviceptr dst,
                           CUdeviceptr gamma, CUdeviceptr beta,
                           int C, int D, int H, int W)
{
    int spatial = D * H * W;
    float eps = 1e-5f;
    int threads = 256;
    unsigned smem = (unsigned)(2 * threads * sizeof(float));
    void *args[] = { &src, &dst, &gamma, &beta, &C, &spatial, &eps };
    return cuLaunchKernel(f->cln, spatial, 1, 1, threads, 1, 1,
                          smem, 0, args, NULL) == CUDA_SUCCESS ? 0 : -1;
}

static int cs3d_launch_silu(const cs3d_ssdec_fns *f, CUdeviceptr x, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *args[] = { &x, &n };
    return cuLaunchKernel(f->silu, blocks, 1, 1, threads, 1, 1,
                          0, 0, args, NULL) == CUDA_SUCCESS ? 0 : -1;
}

static int cs3d_launch_pixshuf(const cs3d_ssdec_fns *f,
                               CUdeviceptr src, CUdeviceptr dst,
                               int C, int D, int H, int W)
{
    int D2 = 2 * D, H2 = 2 * H, W2 = 2 * W;
    int total = C * D2 * H2 * W2;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    void *args[] = { &src, &dst, &C, &D, &H, &W };
    return cuLaunchKernel(f->pixshuf, blocks, 1, 1, threads, 1, 1,
                          0, 0, args, NULL) == CUDA_SUCCESS ? 0 : -1;
}

static int cs3d_launch_resadd(const cs3d_ssdec_fns *f,
                              CUdeviceptr hidden, CUdeviceptr residual, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *args[] = { &hidden, &residual, &n };
    return cuLaunchKernel(f->resadd, blocks, 1, 1, threads, 1, 1,
                          0, 0, args, NULL) == CUDA_SUCCESS ? 0 : -1;
}

static int cs3d_ssdec_resblock(const cs3d_ssdec_resblock_gpu *rb,
                               const cs3d_ssdec_fns *f,
                               CUdeviceptr src, CUdeviceptr dst, CUdeviceptr tmp,
                               int C, int D, int H, int W)
{
    int n = C * D * H * W;
    if (cs3d_launch_cln(f, src, tmp, rb->gn1_w, rb->gn1_b, C, D, H, W) < 0) return -1;
    if (cs3d_launch_silu(f, tmp, n) < 0) return -1;
    if (cs3d_launch_conv3d(f, tmp, dst, rb->conv1_w, rb->conv1_b, C, C, D, H, W) < 0) return -1;
    if (cs3d_launch_cln(f, dst, tmp, rb->gn2_w, rb->gn2_b, C, D, H, W) < 0) return -1;
    if (cs3d_launch_silu(f, tmp, n) < 0) return -1;
    if (cs3d_launch_conv3d(f, tmp, dst, rb->conv2_w, rb->conv2_b, C, C, D, H, W) < 0) return -1;
    if (cs3d_launch_resadd(f, dst, src, n) < 0) return -1;
    return 0;
}

int cs3d_ssdec_forward(const cs3d_ssdec_gpu *g, const cs3d_ssdec_fns *f,
                       cs3d_ssdec_ws *ws, CUdeviceptr d_latent,
                       CUdeviceptr d_out_logits, int verbose)
{
    if (!g || !g->loaded || !f || !ws || !ws->a || !ws->b || !ws->c ||
        !d_latent || !d_out_logits) return -1;
    CUdeviceptr a = ws->a, b = ws->b, c = ws->c;
    double t0 = cs3d_ssdec_time_ms();

    if (cs3d_launch_conv3d(f, d_latent, a, g->conv_in_w, g->conv_in_b, 8, 512, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("conv_in") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: conv_in done\n");

    if (cs3d_ssdec_resblock(&g->middle[0], f, a, b, c, 512, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("middle.0") < 0) return -1;
    if (cs3d_ssdec_resblock(&g->middle[1], f, b, a, c, 512, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("middle.1") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: middle done\n");

    if (cs3d_ssdec_resblock(&g->res_16[0], f, a, b, c, 512, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("res_16.0") < 0) return -1;
    if (cs3d_ssdec_resblock(&g->res_16[1], f, b, a, c, 512, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("res_16.1") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: res_16 done\n");

    if (cs3d_launch_conv3d(f, a, b, g->up1_conv_w, g->up1_conv_b, 512, 1024, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("up1_conv") < 0) return -1;
    if (cs3d_launch_pixshuf(f, b, a, 128, 16, 16, 16) < 0) return -1;
    if (cs3d_ssdec_sync_stage("up1_pixel_shuffle") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: upsample1 done\n");

    if (cs3d_ssdec_resblock(&g->res_32[0], f, a, b, c, 128, 32, 32, 32) < 0) return -1;
    if (cs3d_ssdec_sync_stage("res_32.0") < 0) return -1;
    if (cs3d_ssdec_resblock(&g->res_32[1], f, b, a, c, 128, 32, 32, 32) < 0) return -1;
    if (cs3d_ssdec_sync_stage("res_32.1") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: res_32 done\n");

    if (cs3d_launch_conv3d(f, a, b, g->up2_conv_w, g->up2_conv_b, 128, 256, 32, 32, 32) < 0) return -1;
    if (cs3d_ssdec_sync_stage("up2_conv") < 0) return -1;
    if (cs3d_launch_pixshuf(f, b, a, 32, 32, 32, 32) < 0) return -1;
    if (cs3d_ssdec_sync_stage("up2_pixel_shuffle") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: upsample2 done\n");

    if (cs3d_ssdec_resblock(&g->res_64[0], f, a, b, c, 32, 64, 64, 64) < 0) return -1;
    if (cs3d_ssdec_sync_stage("res_64.0") < 0) return -1;
    if (cs3d_ssdec_resblock(&g->res_64[1], f, b, a, c, 32, 64, 64, 64) < 0) return -1;
    if (cs3d_ssdec_sync_stage("res_64.1") < 0) return -1;
    if (verbose >= 2) fprintf(stderr, "ss_dec_gpu: res_64 done\n");

    if (cs3d_launch_cln(f, a, b, g->out_gn_w, g->out_gn_b, 32, 64, 64, 64) < 0) return -1;
    if (cs3d_launch_silu(f, b, 32 * 64 * 64 * 64) < 0) return -1;
    if (cs3d_launch_conv3d(f, b, d_out_logits, g->out_conv_w, g->out_conv_b, 32, 1, 64, 64, 64) < 0) return -1;
    if (cs3d_ssdec_sync_stage("out_layer") < 0) return -1;
    if (verbose >= 1) {
        fprintf(stderr, "ss_dec_gpu: forward done in %.1f ms, output (1, 64^3)\n",
                cs3d_ssdec_time_ms() - t0);
    }
    return 0;
}

#endif /* CUDA_SAM3D_SS_DECODER_FORWARD_IMPLEMENTATION */
