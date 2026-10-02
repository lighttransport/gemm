/* Focused MHR CUDA context. No SAM3D network or vendor GEMM dependency. */
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../gemm/cuda_gemm_f32_kernels.h"

typedef struct {
    CUcontext context;
    CUmodule module;
    CUfunction matvec, blend;
    CUdeviceptr weights, vectors[2], base, input, output;
    size_t resident_bytes;
} mhr_cuda;

static const char mhr_cuda_source[] =
    "extern \"C\" {\n"
    CUDA_GEMV_F32_SRC
    "__global__ void mhr_blend(float *out, const float *c, const float *v,"
    " const float *base, int n, int width) {\n"
    " int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= width) return;\n"
    " float sum = base ? base[i] : 0.0f;\n"
    " for (int k=0; k<n; ++k) sum += c[k] * v[(size_t)k*width+i];\n"
    " out[i] = sum;\n}\n}\n";

static void mhr_cuda_free(mhr_cuda *c)
{
    if (c->context) {
        cuCtxSetCurrent(c->context);
        CU_FREE(c->weights); CU_FREE(c->vectors[0]); CU_FREE(c->vectors[1]);
        CU_FREE(c->base); CU_FREE(c->input); CU_FREE(c->output);
        if (c->module) cuModuleUnload(c->module);
        cuCtxDestroy(c->context);
    }
    memset(c, 0, sizeof(*c));
}

static int mhr_cuda_upload(CUdeviceptr *dst, const void *src, size_t bytes)
{
    CU_CHECK(cuMemAlloc(dst, bytes));
    CU_CHECK(cuMemcpyHtoD(*dst, src, bytes));
    return 0;
}

static int mhr_cuda_pc(void *user, const float *h, float *out)
{
    mhr_cuda *c = user;
    int k = S3DM_N_PC_H, n = S3DM_N_VERTS * 3;
    CU_CHECK(cuMemcpyHtoD(c->input, h, (size_t)k * sizeof(float)));
    void *args[] = {&c->output, &c->weights, &c->input, &k, &n};
    CU_CHECK(cuLaunchKernel(c->matvec, n, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CU_CHECK(cuMemcpyDtoH(out, c->output, (size_t)n * sizeof(float)));
    return 0;
}

static int mhr_cuda_blend(void *user, int which, const float *coeffs, int n, float *out)
{
    mhr_cuda *c = user;
    int width = S3DM_N_VERTS * 3;
    CUdeviceptr base = which == 0 ? c->base : 0;
    CU_CHECK(cuMemcpyHtoD(c->input, coeffs, (size_t)n * sizeof(float)));
    void *args[] = {&c->output, &c->input, &c->vectors[which], &base, &n, &width};
    CU_CHECK(cuLaunchKernel(c->blend, (width+255)/256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CU_CHECK(cuMemcpyDtoH(out, c->output, (size_t)width * sizeof(float)));
    return 0;
}

static int mhr_cuda_init(mhr_cuda *c, sam3d_body_mhr_assets *a, int device_index)
{
    CUdevice device;
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return -1;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device, device_index));
    CU_CHECK(cuCtxCreate(&c->context, 0, device));
    if (cu_compile_kernels_ex(&c->module, device, mhr_cuda_source,
                             "mhr_decode.cu", 0, "mhr_decode", 0)) return -1;
    CU_CHECK(cuModuleGetFunction(&c->matvec, c->module, "mhr_matvec_f32"));
    CU_CHECK(cuModuleGetFunction(&c->blend, c->module, "mhr_blend"));
    size_t width = S3DM_N_VERTS * 3 * sizeof(float);
    if (mhr_cuda_upload(&c->weights, a->pc_linear_weight.data, width*S3DM_N_PC_H) ||
        mhr_cuda_upload(&c->vectors[0], a->blend_shape_vectors.data, width*S3DM_N_SHAPE) ||
        mhr_cuda_upload(&c->vectors[1], a->face_shape_vectors.data, width*S3DM_N_FACE) ||
        mhr_cuda_upload(&c->base, a->blend_base_shape.data, width)) return -1;
    CU_CHECK(cuMemAlloc(&c->input, S3DM_N_PC_H * sizeof(float)));
    CU_CHECK(cuMemAlloc(&c->output, width));
    c->resident_bytes = width*(S3DM_N_PC_H+S3DM_N_SHAPE+S3DM_N_FACE+2)
                      + S3DM_N_PC_H*sizeof(float);
    a->pc_matvec_user = c; a->pc_matvec_fn = mhr_cuda_pc;
    a->blend_user = c; a->blend_combine_fn = mhr_cuda_blend;
    return 0;
}
