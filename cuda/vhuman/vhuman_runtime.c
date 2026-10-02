/* CUDA driver ownership and presentation, independent of tensor frameworks. */
#include "vhuman_runtime.h"
#include "../cuew.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct vh_cuda_runtime {
    CUdevice device;
    CUcontext context;
    CUstream stream;
    CUmodule module;
    CUfunction pixels;
    CUdeviceptr staging;
    void *host;
    size_t capacity;
    int owns_stream;
};
static const char source[] =
"extern \"C\" __global__ void pixels(const float *x,unsigned char *y,int n,float bg,int alpha) {\n"
" int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;\n"
" float a=fminf(1.f,fmaxf(0.f,x[4*i+3]));\n"
" for(int c=0;c<3;c++){float v=alpha?fminf(1.f,fmaxf(0.f,x[4*i+c]))/fmaxf(a,1.e-8f):x[4*i+c]+(1.f-x[4*i+3])*bg;\n"
" v=fminf(1.f,fmaxf(0.f,v));v=v<=.0031308f?12.92f*v:1.055f*powf(v,1.f/2.4f)-.055f;\n"
" y[i*(alpha?4:3)+c]=(unsigned char)__float2uint_rn(v*255.f);}\n"
" if(alpha)y[4*i+3]=(unsigned char)__float2uint_rn(a*255.f);}\n";

static char *compile(int major, int minor) {
    if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS) return NULL;
    nvrtcProgram program = NULL;
    if (nvrtcCreateProgram(&program, source, "vhuman_pixels.cu", 0, NULL, NULL)) return NULL;
    char arch[64]; snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);
    const char *options[] = {arch, "--fmad=false"};
    nvrtcResult result = nvrtcCompileProgram(program, 2, options);
    size_t size = 0; char *ptx = NULL;
    if (result != NVRTC_SUCCESS) {
        nvrtcGetProgramLogSize(program, &size);
        char *log = malloc(size + 1);
        if (log) { nvrtcGetProgramLog(program, log); fprintf(stderr, "%s\n", log); free(log); }
    } else if (nvrtcGetPTXSize(program, &size) == NVRTC_SUCCESS) {
        ptx = malloc(size);
        if (ptx && nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) { free(ptx); ptx = NULL; }
    }
    nvrtcDestroyProgram(&program);
    return ptx;
}
int vh_cuda_compile_probe(void) { char *ptx = compile(12, 0); int result = ptx ? 0 : -1; free(ptx); return result; }
static int enter(vh_cuda_runtime *r) { return r && cuCtxPushCurrent(r->context) == CUDA_SUCCESS ? 0 : -1; }
static int leave(int result) { CUcontext old; return cuCtxPopCurrent(&old) == CUDA_SUCCESS ? result : -1; }

vh_cuda_runtime *vh_cuda_open(int device, uintptr_t stream, int borrowed) {
    if (device < 0 || cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS || cuInit(0)) return NULL;
    vh_cuda_runtime *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    if (cuDeviceGet(&r->device, device) || cuDevicePrimaryCtxRetain(&r->context, r->device)) { free(r); return NULL; }
    if (enter(r)) { cuDevicePrimaryCtxRelease(r->device); free(r); return NULL; }
    r->owns_stream = borrowed < 0;
    if (r->owns_stream) {
        if (cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING)) goto failed;
    } else r->stream = (CUstream)stream;
    int major = 0, minor = 0;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device) ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device)) goto failed;
    char *ptx = compile(major, minor);
    if (!ptx) goto failed;
    CUresult loaded = cuModuleLoadData(&r->module, ptx); free(ptx);
    if (loaded || cuModuleGetFunction(&r->pixels, r->module, "pixels")) goto failed;
    leave(0); return r;
failed:
    leave(0); vh_cuda_close(r); return NULL;
}
void vh_cuda_close(vh_cuda_runtime *r) {
    if (!r) return;
    if (!enter(r)) {
        cuStreamSynchronize(r->stream);
        if (r->staging) cuMemFree(r->staging);
        if (r->host) cuMemFreeHost(r->host);
        if (r->module) cuModuleUnload(r->module);
        if (r->owns_stream && r->stream) cuStreamDestroy(r->stream);
        leave(0);
    }
    cuDevicePrimaryCtxRelease(r->device); free(r);
}
uintptr_t vh_cuda_stream(vh_cuda_runtime *r) { return r ? (uintptr_t)r->stream : 0; }
int vh_cuda_sync(vh_cuda_runtime *r) { if (enter(r)) return -1; return leave(cuStreamSynchronize(r->stream) ? -1 : 0); }
uintptr_t vh_cuda_record(vh_cuda_runtime *r) {
    if (enter(r)) return 0;
    CUevent event = NULL;
    if (cuEventCreate(&event, 0) || cuEventRecord(event, r->stream)) {
        if (event) cuEventDestroy(event);
        event = NULL;
    }
    leave(0); return (uintptr_t)event;
}
int vh_cuda_wait(vh_cuda_runtime *r, uintptr_t event) {
    if (!event || enter(r)) return -1;
    return leave(cuStreamWaitEvent(r->stream, (CUevent)event, 0) ? -1 : 0);
}
int vh_cuda_elapsed(vh_cuda_runtime *r, uintptr_t begin, uintptr_t end, float *ms) {
    if (!begin || !end || !ms || enter(r)) return -1;
    return leave(cuEventElapsedTime(ms, (CUevent)begin, (CUevent)end) ? -1 : 0);
}
int vh_cuda_event_sync(vh_cuda_runtime *r, uintptr_t event) {
    if (!event || enter(r)) return -1;
    return leave(cuEventSynchronize((CUevent)event) ? -1 : 0);
}
void vh_cuda_event_free(vh_cuda_runtime *r, uintptr_t event) {
    if (event && !enter(r)) { cuEventDestroy((CUevent)event); leave(0); }
}
int vh_cuda_download(vh_cuda_runtime *r, uintptr_t pointer, void *host, size_t bytes) {
    if (!pointer || !host || !bytes || enter(r)) return -1;
    int result = cuStreamSynchronize(r->stream) || cuMemcpyDtoH(host, pointer, bytes);
    return leave(result ? -1 : 0);
}
int vh_cuda_pixels(vh_cuda_runtime *r, uintptr_t rgba, int width, int height,
                   float bg, int alpha, unsigned char *host) {
    if (!rgba || !host || width < 1 || height < 1 || width > 8192 || height > 8192 ||
        !isfinite(bg) || bg < 0 || bg > 1 || (alpha != 0 && alpha != 1) || enter(r)) return -1;
    size_t bytes = (size_t)width * height * (alpha ? 4 : 3);
    if (bytes > r->capacity) {
        if (cuStreamSynchronize(r->stream)) return leave(-1);
        if (r->staging) cuMemFree(r->staging);
        if (r->host) cuMemFreeHost(r->host);
        r->staging = 0; r->host = NULL; r->capacity = 0;
        if (cuMemAlloc(&r->staging, bytes) || cuMemHostAlloc(&r->host, bytes, 0)) return leave(-1);
        r->capacity = bytes;
    }
    int n = width * height;
    void *args[] = {&rgba, &r->staging, &n, &bg, &alpha};
    if (cuLaunchKernel(r->pixels, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL) ||
        cuMemcpyDtoHAsync(r->host, r->staging, bytes, r->stream) || cuStreamSynchronize(r->stream)) return leave(-1);
    memcpy(host, r->host, bytes);
    return leave(0);
}
int vh_pixels_cpu(const float *rgba, size_t n, float bg, int alpha, unsigned char *host) {
    if (!rgba || !host || n > 8192u*8192u || !isfinite(bg) || bg < 0 || bg > 1 || (alpha != 0 && alpha != 1)) return -1;
    for (size_t i = 0; i < n*4; i++) if (!isfinite(rgba[i])) return -1;
    for (size_t i = 0; i < n; i++) {
        float a = fminf(1, fmaxf(0, rgba[4*i+3]));
        for (int c = 0; c < 3; c++) {
            float v = alpha ? fminf(1, fmaxf(0, rgba[4*i+c])) / fmaxf(a, 1.e-8f) : rgba[4*i+c] + (1-rgba[4*i+3])*bg;
            v = fminf(1, fmaxf(0, v));
            v = v <= .0031308f ? 12.92f*v : 1.055f*powf(v, 1.f/2.4f)-.055f;
            host[i*(alpha?4:3)+c] = (unsigned char)nearbyintf(v*255);
        }
        if (alpha) host[4*i+3] = (unsigned char)nearbyintf(a*255);
    }
    return 0;
}
