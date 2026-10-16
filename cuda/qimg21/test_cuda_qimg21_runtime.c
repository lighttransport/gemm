/* Shared-runtime architecture regression: BF16 helpers survive FP8 ISA guards. */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_QIMG_RUNNER_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../qimg/cuda_qimg_runner.h"
#include <dlfcn.h>

typedef int (*conv_fn)(float *, const float *, const float *, const float *,
                      int, int, int, int, int, int, int, int, int, int, CUstream);

static float round_bf16(float x) {
    uint32_t bits = (uint32_t)qimg_f32_to_bf16_bits(x) << 16;
    float value; memcpy(&value, &bits, sizeof(value)); return value;
}

static int check_conv(cuda_qimg_runner *r, conv_fn conv, int ci, int co, int h, int kh) {
    int pad = kh/2, oh = h, ow = h;
    size_t nx = (size_t)ci*h*h, nw = (size_t)co*ci*kh*kh, ny = (size_t)co*h*h;
    float *x = malloc(nx*4), *w = malloc(nw*4), *b = malloc(co*4), *out = malloc(ny*4);
    CUdeviceptr dx = 0, dw = 0, db = 0, dy = 0;
    int ok = x && w && b && out;
    for (size_t i=0;i<nx && ok;i++) x[i] = ((int)(i%17)-8)/16.f;
    for (size_t i=0;i<nw && ok;i++) w[i] = ((int)(i%7)-3)/64.f;
    for (int i=0;i<co && ok;i++) b[i] = (i%5-2)/8.f;
    if (ok) ok = cuMemAlloc(&dx,nx*4)==CUDA_SUCCESS && cuMemAlloc(&dw,nw*4)==CUDA_SUCCESS &&
                 cuMemAlloc(&db,co*4)==CUDA_SUCCESS && cuMemAlloc(&dy,ny*4)==CUDA_SUCCESS;
    if (ok) ok = cuMemcpyHtoD(dx,x,nx*4)==CUDA_SUCCESS && cuMemcpyHtoD(dw,w,nw*4)==CUDA_SUCCESS &&
                 cuMemcpyHtoD(db,b,co*4)==CUDA_SUCCESS;
    if (ok) ok = !conv((float *)(uintptr_t)dy,(const float *)(uintptr_t)dx,
                      (const float *)(uintptr_t)dw,(const float *)(uintptr_t)db,
                      ci,h,h,co,kh,kh,1,pad,oh,ow,r->stream) &&
                 cuStreamSynchronize(r->stream)==CUDA_SUCCESS && cuMemcpyDtoH(out,dy,ny*4)==CUDA_SUCCESS;
    float max_error = 0;
    for (int c=0;c<co && ok;c++) for (int y=0;y<h;y++) for (int col=0;col<h;col++) {
        float sum = 0;
        for (int k=0;k<ci;k++) for (int a=0;a<kh;a++) for (int j=0;j<kh;j++) {
            int yy=y+a-pad,xx=col+j-pad;
            if (yy>=0 && yy<h && xx>=0 && xx<h)
                sum += x[(k*h+yy)*h+xx]*w[((c*ci+k)*kh+a)*kh+j];
        }
        float expected = round_bf16(round_bf16(sum)+b[c]);
        float error = fabsf(out[(c*h+y)*h+col]-expected);
        if (error>max_error) max_error=error;
        if (!isfinite(error) || error>.001f) ok=0;
    }
    printf("BF16 conv %d->%d k%d: max_abs_error=%g %s\n",ci,co,kh,max_error,ok?"PASS":"FAIL");
    if (dx) cuMemFree(dx);
    if (dw) cuMemFree(dw);
    if (db) cuMemFree(db);
    if (dy) cuMemFree(dy);
    free(x);free(w);free(b);free(out);
    return ok;
}

int main(void) {
    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    if (!r) return 1;
    int ok = r->cast_f32_to_bf16 != NULL;
    if (r->sm_version < 89)
        ok = ok && !r->gemm_fp8_mma && !r->quant_f32_to_fp8_e4m3 && !r->quant_bf16_to_fp8_e4m3;
    float input[] = {0.f, 1.f, -1.f, 1.00390625f, 1.01171875f, -1.00390625f, 0x1p-126f, 1000.f};
    uint16_t output[8] = {0};
    CUdeviceptr src = 0, dst = 0;
    int n = 8;
    if (ok) ok = cuMemAlloc(&src, sizeof(input)) == CUDA_SUCCESS &&
                 cuMemAlloc(&dst, sizeof(output)) == CUDA_SUCCESS;
    if (ok) ok = cuMemcpyHtoD(src, input, sizeof(input)) == CUDA_SUCCESS;
    void *args[] = {&src, &dst, &n};
    if (ok) ok = cuLaunchKernel(r->cast_f32_to_bf16, 1, 1, 1, 256, 1, 1, 0, r->stream, args, NULL) == CUDA_SUCCESS &&
                 cuStreamSynchronize(r->stream) == CUDA_SUCCESS &&
                 cuMemcpyDtoH(output, dst, sizeof(output)) == CUDA_SUCCESS;
    for (int i = 0; i < n && ok; i++)
        ok = output[i] == qimg_f32_to_bf16_bits(input[i]);
    printf("sm_%d: BF16 cast round-to-even and optional FP8 guards %s\n", r->sm_version, ok ? "PASS" : "FAIL");
    if (ok && r->sm_version>=80 && r->sm_version<120) {
        void *lib = dlopen("cuda/qimg21/libq21_cudnn_vae.so", RTLD_NOW|RTLD_LOCAL);
        conv_fn conv = lib ? (conv_fn)dlsym(lib,"q21_cudnn_vae_conv2d_ex") : NULL;
        ok = conv && check_conv(r,conv,4,96,8,3) && check_conv(r,conv,96,96,8,3) &&
             check_conv(r,conv,192,384,4,1);
        if (lib) dlclose(lib);
    }
    if (src) cuMemFree(src);
    if (dst) cuMemFree(dst);
    cuda_qimg_free(r);
    return ok ? 0 : 1;
}
