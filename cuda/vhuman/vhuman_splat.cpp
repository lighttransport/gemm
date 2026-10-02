/* Framework-free triangle-bound Gaussian inference. Host bins/sorts projected
 * splats; all deformation, projection and pixel blending execute on CUDA. */
#include "../cuew.h"
#include "vhuman_splat_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string last_error;
static void check(CUresult r) {
    if (r != CUDA_SUCCESS) throw std::runtime_error("CUDA driver error " + std::to_string(int(r)));
}
struct context_scope {
    explicit context_scope(CUcontext c) { check(cuCtxPushCurrent(c)); }
    ~context_scope() { CUcontext c; cuCtxPopCurrent(&c); }
};
struct buffer {
    CUdeviceptr p = 0;
    size_t capacity = 0;
    void reserve(size_t bytes) {
        if (capacity >= bytes) return;
        CUdeviceptr next = 0; check(cuMemAlloc(&next, bytes));
        if (p) cuMemFree(p);
        p = next; capacity = bytes;
    }
    void upload(const void *data, size_t bytes) { reserve(bytes); if (bytes) check(cuMemcpyHtoD(p, data, bytes)); }
    void clear() { if (p) cuMemFree(p); p = 0; capacity = 0; }
};
struct splat_renderer {
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUfunction deform = nullptr, project = nullptr, raster = nullptr;
    int n = 0, v = 0, controls = 0, trace = 0;
    buffer vertices, attachments, assets, coefficients, geometry, camera, projected, offsets, ids;
    std::vector<float> expression, projection;
    std::vector<int> order, tile_offsets, tile_ids, cursor;
    size_t frame_bytes = 0, peak_bytes = 0;
    size_t bytes() const {
        return vertices.capacity + attachments.capacity + assets.capacity + coefficients.capacity +
            geometry.capacity + camera.capacity + projected.capacity + offsets.capacity + ids.capacity + frame_bytes;
    }
    void peak() { peak_bytes = std::max(peak_bytes, bytes()); }
    ~splat_renderer() {
        if (!context) return;
        CUcontext previous;
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuStreamSynchronize(stream);
            for (auto p : {&vertices,&attachments,&assets,&coefficients,&geometry,&camera,&projected,&offsets,&ids}) p->clear();
            if (module) cuModuleUnload(module);
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }
};
static std::vector<char> compile_source(int major, int minor) {
    if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS) throw std::runtime_error("NVRTC unavailable");
    nvrtcProgram program = nullptr;
    if (nvrtcCreateProgram(&program, vh_splat_source, "vhuman_splat.cu", 0, nullptr, nullptr))
        throw std::runtime_error("NVRTC create failed");
    std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
    const char *options[] = {arch.c_str(), "--fmad=false", "--std=c++14"};
    auto result = nvrtcCompileProgram(program, 3, options);
    size_t size = 0;
    if (result != NVRTC_SUCCESS) {
        nvrtcGetProgramLogSize(program, &size);
        std::vector<char> log(size + 1); nvrtcGetProgramLog(program, log.data());
        nvrtcDestroyProgram(&program); throw std::runtime_error(log.data());
    }
    nvrtcGetPTXSize(program, &size);
    std::vector<char> ptx(size);
    result = nvrtcGetPTX(program, ptx.data()); nvrtcDestroyProgram(&program);
    if (result != NVRTC_SUCCESS) throw std::runtime_error("NVRTC PTX failed");
    return ptx;
}
extern "C" const char *vhs_error() { return last_error.c_str(); }
extern "C" int vhs_compile_probe() {
    try { compile_source(12, 0); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1; }
}
extern "C" void vhs_close(splat_renderer *r) { delete r; }
extern "C" splat_renderer *vhs_open(int device, uintptr_t stream, int n, int v, int controls,
    int trace, const int *attachments, const float *assets, const float *expression) {
    splat_renderer *r = nullptr;
    try {
        if (n < 1 || n > 200000 || v < 3 || v > 2000000 || controls < 1 || controls > 1024 ||
            !attachments || !assets || !expression || (trace != 0 && trace != 1))
            throw std::runtime_error("invalid renderer assets");
        for (int i = 0; i < 3*n; i++) if (attachments[i] < 0 || attachments[i] >= v)
            throw std::runtime_error("attachment index out of bounds");
        for (int i = 0; i < 41*n; i++) if (!std::isfinite(assets[i])) throw std::runtime_error("nonfinite asset");
        for (int i = 0; i < controls*8; i++) if (!std::isfinite(expression[i])) throw std::runtime_error("nonfinite expression matrix");
        if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) throw std::runtime_error("CUDA unavailable");
        check(cuInit(0)); r = new splat_renderer;
        check(cuDeviceGet(&r->device, device)); check(cuDevicePrimaryCtxRetain(&r->context, r->device));
        context_scope current(r->context);
        r->stream = reinterpret_cast<CUstream>(stream); r->n = n; r->v = v; r->controls = controls; r->trace = trace;
        r->expression.assign(expression, expression + controls*8);
        int major = 0, minor = 0;
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device));
        check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device));
        auto ptx = compile_source(major, minor);
        check(cuModuleLoadData(&r->module, ptx.data()));
        check(cuModuleGetFunction(&r->deform, r->module, "deform"));
        check(cuModuleGetFunction(&r->project, r->module, "project"));
        check(cuModuleGetFunction(&r->raster, r->module, "raster"));
        r->attachments.upload(attachments, size_t(n)*3*sizeof(int));
        r->assets.upload(assets, size_t(n)*41*sizeof(float));
        r->geometry.reserve(size_t(n)*16*sizeof(float)); r->projected.reserve(size_t(n)*12*sizeof(float));
        r->projection.resize(size_t(n)*12); r->order.resize(n); r->peak();
        return r;
    } catch (const std::exception &e) { last_error = e.what(); delete r; return nullptr; }
}
static void deform(splat_renderer *r, uintptr_t device_vertices, const float *host_vertices, const float *controls) {
    if (!device_vertices && !host_vertices) throw std::runtime_error("missing vertices");
    check(cuStreamSynchronize(r->stream));
    CUdeviceptr vp = device_vertices;
    if (!vp) {
        for (int i = 0; i < 3*r->v; i++) if (!std::isfinite(host_vertices[i])) throw std::runtime_error("nonfinite vertices");
        r->vertices.upload(host_vertices, size_t(r->v)*3*sizeof(float)); vp = r->vertices.p;
    }
    float coeff[8] = {};
    if (controls) for (int i = 0; i < r->controls; i++) {
        if (!std::isfinite(controls[i])) throw std::runtime_error("nonfinite controls");
        for (int j = 0; j < 8; j++) coeff[j] += controls[i]*r->expression[i*8+j];
    }
    for (float &c : coeff) c = std::tanh(c);
    r->coefficients.upload(coeff, sizeof(coeff));
    void *args[] = {&vp,&r->attachments.p,&r->assets.p,&r->coefficients.p,&r->n,&r->trace,&r->geometry.p};
    check(cuLaunchKernel(r->deform,(r->n+127)/128,1,1,128,1,1,0,r->stream,args,nullptr));
    r->peak();
}
extern "C" int vhs_deform(splat_renderer *r, uintptr_t vertices, const float *host, const float *controls, float *geometry) {
    try {
        if (!r || !geometry) throw std::runtime_error("invalid deformation output");
        context_scope current(r->context); deform(r,vertices,host,controls);
        check(cuStreamSynchronize(r->stream));
        check(cuMemcpyDtoH(geometry,r->geometry.p,size_t(r->n)*16*sizeof(float))); return 0;
    } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
extern "C" uintptr_t vhs_render(splat_renderer *r, uintptr_t vertices, const float *host, const float *controls,
    const float *camera, int w, int h) {
    CUdeviceptr output = 0;
    try {
        if (!r || !camera || w < 1 || h < 1 || w > 4096 || h > 4096) throw std::runtime_error("invalid render arguments");
        context_scope current(r->context);
        for (int i=0;i<25;i++) if (!std::isfinite(camera[i])) throw std::runtime_error("nonfinite camera");
        if (camera[16]<=0 || camera[20]<=0) throw std::runtime_error("invalid focal length");
        deform(r,vertices,host,controls); r->camera.upload(camera,25*sizeof(float));
        void *args[] = {&r->geometry.p,&r->camera.p,&r->n,&w,&h,&r->projected.p};
        check(cuLaunchKernel(r->project,(r->n+127)/128,1,1,128,1,1,0,r->stream,args,nullptr));
        check(cuStreamSynchronize(r->stream));
        check(cuMemcpyDtoH(r->projection.data(),r->projected.p,r->projection.size()*sizeof(float)));
        int tw=(w+15)/16,th=(h+15)/16,tiles=tw*th;
        r->tile_offsets.assign(tiles+1,0); std::iota(r->order.begin(),r->order.end(),0);
        std::stable_sort(r->order.begin(),r->order.end(),[&](int a,int b){return r->projection[a*12+2]<r->projection[b*12+2];});
        auto visit = [&](int i, auto append) {
            const float *p=r->projection.data()+12*i;
            if (p[10]<=0 || p[11]<=0) return;
            int x0=std::clamp(int(std::floor((p[0]-p[10])/16)),0,tw);
            int y0=std::clamp(int(std::floor((p[1]-p[11])/16)),0,th);
            int x1=std::clamp(int(std::ceil((p[0]+p[10])/16)),0,tw);
            int y1=std::clamp(int(std::ceil((p[1]+p[11])/16)),0,th);
            for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++)append(y*tw+x,i);
        };
        size_t count=0;
        for(int i:r->order) visit(i,[&](int tile,int){
            if (++count > 8000000) throw std::runtime_error("Gaussian tile overlap budget exceeded");
            r->tile_offsets[tile+1]++;
        });
        std::partial_sum(r->tile_offsets.begin(),r->tile_offsets.end(),r->tile_offsets.begin());
        r->tile_ids.resize(count); r->cursor=r->tile_offsets;
        for(int i:r->order) visit(i,[&](int tile,int id){r->tile_ids[r->cursor[tile]++]=id;});
        r->offsets.upload(r->tile_offsets.data(),r->tile_offsets.size()*sizeof(int));
        r->ids.upload(r->tile_ids.data(),r->tile_ids.size()*sizeof(int));
        size_t bytes=size_t(w)*h*4*sizeof(float); check(cuMemAlloc(&output,bytes));
        void *raster_args[]={&r->projected.p,&r->offsets.p,&r->ids.p,&w,&h,&output};
        check(cuLaunchKernel(r->raster,tiles,1,1,256,1,1,0,r->stream,raster_args,nullptr));
        r->frame_bytes+=bytes; r->peak(); return output;
    } catch(const std::exception &e) {
        last_error=e.what();
        if(output && r) { context_scope current(r->context); cuStreamSynchronize(r->stream); cuMemFree(output); }
        return 0;
    }
}
extern "C" void vhs_free_frame(splat_renderer *r, uintptr_t pointer, size_t bytes) {
    if (!r || !pointer) return;
    try { context_scope current(r->context); check(cuStreamSynchronize(r->stream)); check(cuMemFree(pointer)); r->frame_bytes-=bytes; }
    catch(const std::exception &e) { last_error=e.what(); }
}
extern "C" size_t vhs_memory(splat_renderer *r, int peak) { return r ? (peak ? r->peak_bytes : r->bytes()) : 0; }
extern "C" void vhs_reset_peak(splat_renderer *r) { if (r) r->peak_bytes=r->bytes(); }
