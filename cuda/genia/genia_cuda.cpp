#include "genia_cuda.h"
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../cuda_runner_common.h"
#pragma GCC diagnostic pop
#include "genia_kernels.inc"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

static thread_local std::string last_error;
static void check(CUresult rc)
{
    if (rc == CUDA_SUCCESS) return;
    const char *message = "CUDA operation failed";
    cuGetErrorString(rc, &message);
    throw std::runtime_error(message);
}

struct genia_cuda {
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction visibility = nullptr, fuse = nullptr, tweedie = nullptr, norm = nullptr, guidance = nullptr;
    CUfunction project = nullptr, raster = nullptr, project_backward = nullptr;
    size_t budget = 0, live = 0, peak = 0;
};

struct scope {
    explicit scope(genia_cuda *ctx) { if (!ctx) throw std::runtime_error("null GenIA context"); check(cuCtxPushCurrent(ctx->context)); }
    ~scope() { CUcontext popped; cuCtxPopCurrent(&popped); }
};

struct buffer {
    genia_cuda *ctx; CUdeviceptr pointer = 0; size_t bytes;
    buffer(genia_cuda *c, size_t n, const void *data = nullptr): ctx(c), bytes(n) {
        if (!n || n > ctx->budget-ctx->live) throw std::runtime_error("GenIA device allocation budget exceeded");
        check(cuMemAlloc(&pointer, n));
        ctx->live += n; if (ctx->live > ctx->peak) ctx->peak = ctx->live;
        if (data) {
            CUresult rc = cuMemcpyHtoD(pointer, data, n);
            if (rc != CUDA_SUCCESS) { cuMemFree(pointer); ctx->live -= bytes; pointer = 0; check(rc); }
        }
    }
    ~buffer() { if (pointer) { cuMemFree(pointer); ctx->live -= bytes; } }
    buffer(const buffer &) = delete;
    void download(void *data) { if (!data) throw std::runtime_error("null GenIA output"); check(cuMemcpyDtoH(data, pointer, bytes)); }
};

static size_t elements(int a, int b = 1, int c = 1)
{
    if (a <= 0 || b <= 0 || c <= 0) throw std::runtime_error("GenIA dimensions must be positive");
    size_t result = (size_t)a;
    if (result > std::numeric_limits<size_t>::max()/b) throw std::runtime_error("GenIA size overflow");
    result *= b;
    if (result > std::numeric_limits<size_t>::max()/c/sizeof(float)) throw std::runtime_error("GenIA size overflow");
    return result*c;
}

static void finite(const float *values, size_t count)
{
    if (!values) throw std::runtime_error("null GenIA input");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite GenIA input");
}

static void launch(CUfunction fn, int count, void **args, int threads = 128)
{
    check(cuLaunchKernel(fn, (unsigned)(((size_t)count+threads-1)/threads), 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
}

extern "C" const char *genia_cuda_error(void) { return last_error.c_str(); }
extern "C" size_t genia_cuda_peak_bytes(const genia_cuda *ctx) { return ctx ? ctx->peak : 0; }

extern "C" genia_cuda *genia_cuda_open(int device, size_t budget)
{
    last_error.clear(); genia_cuda *ctx = nullptr; CUcontext previous = nullptr;
    try {
        if (device < 0 || !budget) throw std::runtime_error("invalid GenIA device/budget");
        if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != 0) throw std::runtime_error("CUDA/NVRTC unavailable");
        check(cuInit(0)); check(cuCtxGetCurrent(&previous));
        CUdevice dev; check(cuDeviceGet(&dev, device));
        ctx = new genia_cuda; ctx->budget = budget;
        check(cuCtxCreate(&ctx->context, 0, dev));
        /* DDA tie-breaking is geometric: fused multiply-add can move an entry
         * point across a cell boundary relative to the upstream FP64 oracle. */
        if (cu_compile_kernels_ex(&ctx->module, dev, genia_kernel_source, "genia.cu", 0, "genia", CU_COMPILE_NO_FMAD) < 0)
            throw std::runtime_error("GenIA kernel compilation failed");
        check(cuModuleGetFunction(&ctx->visibility, ctx->module, "genia_visible_kernel"));
        check(cuModuleGetFunction(&ctx->fuse, ctx->module, "genia_fuse_kernel"));
        check(cuModuleGetFunction(&ctx->tweedie, ctx->module, "genia_tweedie_kernel"));
        check(cuModuleGetFunction(&ctx->norm, ctx->module, "genia_norm_kernel"));
        check(cuModuleGetFunction(&ctx->guidance, ctx->module, "genia_guidance_kernel"));
        check(cuModuleGetFunction(&ctx->project, ctx->module, "genia_project_kernel"));
        check(cuModuleGetFunction(&ctx->raster, ctx->module, "genia_raster_kernel"));
        check(cuModuleGetFunction(&ctx->project_backward, ctx->module, "genia_project_backward_kernel"));
        check(cuCtxSetCurrent(previous)); return ctx;
    } catch (const std::exception &error) {
        last_error = error.what();
        if (ctx) {
            if (ctx->module) cuModuleUnload(ctx->module);
            if (ctx->context) cuCtxDestroy(ctx->context);
            delete ctx;
        }
        if (cuCtxSetCurrent) cuCtxSetCurrent(previous);
        return nullptr;
    }
}

extern "C" void genia_cuda_close(genia_cuda *ctx)
{
    if (!ctx) return;
    CUcontext previous = nullptr;
    if (cuCtxGetCurrent(&previous) != CUDA_SUCCESS) return;
    if (cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) return;
    if (ctx->module) cuModuleUnload(ctx->module);
    cuCtxDestroy(ctx->context); delete ctx;
    cuCtxSetCurrent(previous);
}

#define GENIA_BEGIN last_error.clear(); try {
#define GENIA_END return 0; } catch (const std::exception &error) { last_error = error.what(); return -1; }

extern "C" int genia_cuda_render(genia_cuda *ctx, const float *rows, int count,
    const float camera[12], const float k[9], int width, int height, const float background[3],
    const float *rgba_gradient, const float *depth_gradient, float *rgba, float *depth,
    float *row_gradient, float *camera_gradient)
{
    GENIA_BEGIN
    size_t n = elements(count, 17), pixels = elements(width, height);
    if (pixels > 4000000 || width > 16384 || height > 16384)
        throw std::runtime_error("GenIA render exceeds 4M pixels");
    finite(rows, n); finite(camera, 12); finite(k, 9); finite(background, 3);
    if (k[0] <= 0 || k[4] <= 0 || k[1] != 0 || k[3] != 0 || k[6] != 0 || k[7] != 0 || k[8] != 1)
        throw std::runtime_error("GenIA render requires positive pinhole intrinsics without skew");
    if (rgba_gradient) finite(rgba_gradient, pixels*4);
    if (depth_gradient) finite(depth_gradient, pixels);
    bool backward = row_gradient || camera_gradient;
    scope active(ctx);
    buffer input(ctx, n*sizeof(float), rows), cam(ctx, 12*sizeof(float), camera),
        intrinsics(ctx, 9*sizeof(float), k), bg(ctx, 3*sizeof(float), background),
        projected(ctx, (size_t)count*12*sizeof(double));
    CUdeviceptr jacobian_pointer = 0;
    size_t jacobian_bytes = backward ? (size_t)count*260*sizeof(double) : sizeof(double);
    buffer jacobian(ctx, jacobian_bytes);
    if (backward) { jacobian_pointer = jacobian.pointer; check(cuMemsetD8(jacobian.pointer, 0, jacobian_bytes)); }
    void *project_args[] = {&input.pointer, &count, &cam.pointer, &intrinsics.pointer, &projected.pointer, &jacobian_pointer};
    launch(ctx->project, count, project_args, 32);
    std::vector<double> host_projection((size_t)count*12);
    projected.download(host_projection.data());
    int tw = (width+15)/16, th = (height+15)/16, tile_count = tw*th;
    std::vector<std::vector<int>> tiles(tile_count);
    size_t overlaps = 0;
    for (int i = 0; i < count; ++i) {
        const double *q = host_projection.data()+12*i;
        if (!(q[2] > .01) || q[10] <= 0 || q[11] <= 0) continue;
        double left = std::max(0., std::min((double)width, q[0]-q[10]));
        double right = std::max(0., std::min((double)width, q[0]+q[10]));
        double top = std::max(0., std::min((double)height, q[1]-q[11]));
        double bottom = std::max(0., std::min((double)height, q[1]+q[11]));
        if (right <= left || bottom <= top) continue;
        int x0 = (int)std::floor(left/16), x1 = std::min(tw, (int)std::ceil(right/16));
        int y0 = (int)std::floor(top/16), y1 = std::min(th, (int)std::ceil(bottom/16));
        overlaps += (size_t)(x1-x0)*(y1-y0);
        if (overlaps > 8000000) throw std::runtime_error("GenIA tile overlap limit exceeded");
        for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) tiles[y*tw+x].push_back(i);
    }
    std::vector<int> offsets(tile_count+1), ids;
    ids.reserve(overlaps);
    for (int tile = 0; tile < tile_count; ++tile) {
        auto &bucket = tiles[tile];
        std::sort(bucket.begin(), bucket.end(), [&](int a, int b) {
            double za = host_projection[12*a+2], zb = host_projection[12*b+2];
            return za == zb ? a < b : za < zb;
        });
        ids.insert(ids.end(), bucket.begin(), bucket.end()); offsets[tile+1] = (int)ids.size();
    }
    if (ids.empty()) ids.push_back(0);
    buffer tile_offsets(ctx, offsets.size()*sizeof(int), offsets.data()), tile_ids(ctx, ids.size()*sizeof(int), ids.data()),
        image(ctx, pixels*4*sizeof(float)), image_depth(ctx, pixels*sizeof(float)),
        g_rgba(ctx, rgba_gradient ? pixels*4*sizeof(float) : sizeof(float), rgba_gradient),
        g_depth(ctx, depth_gradient ? pixels*sizeof(float) : sizeof(float), depth_gradient),
        g_projection(ctx, backward ? (size_t)count*10*sizeof(double) : sizeof(double));
    CUdeviceptr rgba_grad_ptr = rgba_gradient ? g_rgba.pointer : 0, depth_grad_ptr = depth_gradient ? g_depth.pointer : 0,
        g_projected_ptr = backward ? g_projection.pointer : 0;
    if (backward) check(cuMemsetD8(g_projection.pointer, 0, (size_t)count*10*sizeof(double)));
    void *raster_args[] = {&projected.pointer, &tile_offsets.pointer, &tile_ids.pointer, &width, &height,
        &bg.pointer, &rgba_grad_ptr, &depth_grad_ptr, &image.pointer, &image_depth.pointer, &g_projected_ptr};
    check(cuLaunchKernel(ctx->raster, tile_count, 1, 1, 256, 1, 1, 0, nullptr, raster_args, nullptr));
    if (rgba) image.download(rgba);
    if (depth) image_depth.download(depth);
    if (backward) {
        buffer g_rows(ctx, row_gradient ? n*sizeof(float) : sizeof(float)), g_camera(ctx, 12*sizeof(double));
        check(cuMemsetD8(g_camera.pointer, 0, 12*sizeof(double)));
        CUdeviceptr g_rows_ptr = row_gradient ? g_rows.pointer : 0, g_camera_ptr = camera_gradient ? g_camera.pointer : 0;
        void *backward_args[] = {&jacobian.pointer, &g_projection.pointer, &count, &g_rows_ptr, &g_camera_ptr};
        launch(ctx->project_backward, count, backward_args);
        if (row_gradient) g_rows.download(row_gradient);
        if (camera_gradient) {
            double result[12]; g_camera.download(result);
            for (int i = 0; i < 12; ++i) camera_gradient[i] = (float)result[i];
        }
    }
    if (!rgba && !depth && !backward) check(cuCtxSynchronize());
    GENIA_END
}

extern "C" int genia_cuda_visibility(genia_cuda *ctx, const int32_t *xyz, int count,
    const float camera[3], const uint8_t *occupancy, int grid, float tolerance, uint8_t *visible)
{
    GENIA_BEGIN
    elements(count, 3); finite(camera, 3);
    if (!xyz || !occupancy || !visible || grid < 1 || grid > 512 || !std::isfinite(tolerance) || tolerance < 0)
        throw std::runtime_error("invalid GenIA visibility inputs");
    for (size_t i = 0; i < (size_t)count*3; ++i)
        if (xyz[i] < 0 || xyz[i] >= grid) throw std::runtime_error("voxel outside occupancy grid");
    scope active(ctx);
    buffer coords(ctx, (size_t)count*3*sizeof(int32_t), xyz), cam(ctx, 3*sizeof(float), camera),
        occ(ctx, (size_t)grid*grid*grid, occupancy), out(ctx, count);
    void *args[] = {&coords.pointer, &count, &cam.pointer, &occ.pointer, &grid, &tolerance, &out.pointer};
    launch(ctx->visibility, count, args); out.download(visible);
    GENIA_END
}

extern "C" int genia_cuda_fuse(genia_cuda *ctx, const float *scores, const float *velocities,
    int views, int count, int channels, float temperature, float floor_weight, float *output)
{
    GENIA_BEGIN
    if (views > 16 || !std::isfinite(temperature) || !std::isfinite(floor_weight) || floor_weight < 0)
        throw std::runtime_error("invalid GenIA fusion inputs");
    size_t n = elements(views, count, channels); finite(velocities, n); finite(scores, elements(views, count));
    scope active(ctx);
    buffer s(ctx, (size_t)views*count*sizeof(float), scores), v(ctx, n*sizeof(float), velocities),
        out(ctx, (size_t)count*channels*sizeof(float));
    void *args[] = {&s.pointer, &v.pointer, &views, &count, &channels, &temperature, &floor_weight, &out.pointer};
    launch(ctx->fuse, count, args); out.download(output);
    GENIA_END
}

extern "C" int genia_cuda_tweedie(genia_cuda *ctx, const float *latent, const float *velocity,
    int count, float time, float *clean)
{
    GENIA_BEGIN
    size_t n = elements(count); finite(latent, n); finite(velocity, n);
    if (!std::isfinite(time) || time < 0 || time > 1) throw std::runtime_error("invalid GenIA time");
    scope active(ctx);
    buffer x(ctx, n*sizeof(float), latent), v(ctx, n*sizeof(float), velocity), out(ctx, n*sizeof(float));
    void *args[] = {&x.pointer, &v.pointer, &count, &time, &out.pointer};
    launch(ctx->tweedie, count, args); out.download(clean);
    GENIA_END
}

extern "C" int genia_cuda_guidance(genia_cuda *ctx, const float *velocity, const float *gradient,
    int count, float weight, int normalize, float *guided)
{
    GENIA_BEGIN
    size_t n = elements(count); finite(velocity, n); finite(gradient, n);
    if (!std::isfinite(weight) || weight < 0 || (normalize != 0 && normalize != 1))
        throw std::runtime_error("invalid GenIA guidance options");
    scope active(ctx);
    buffer v(ctx, n*sizeof(float), velocity), g(ctx, n*sizeof(float), gradient), out(ctx, n*sizeof(float)), norm(ctx, 2*sizeof(double));
    void *norm_args[] = {&v.pointer, &g.pointer, &count, &norm.pointer};
    check(cuLaunchKernel(ctx->norm, 1, 1, 1, 256, 1, 1, 0, nullptr, norm_args, nullptr));
    void *args[] = {&v.pointer, &g.pointer, &count, &weight, &normalize, &norm.pointer, &out.pointer};
    launch(ctx->guidance, count, args); out.download(guided);
    GENIA_END
}
