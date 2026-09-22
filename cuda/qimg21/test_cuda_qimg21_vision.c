/* Native Qwen3-VL vision front end.  This first-stage validator covers patch
 * projection plus learned-position interpolation before the 27 vision blocks. */
#define main q21_denoise_main
#include "test_cuda_qimg21_native.c"
#undef main

static const char *vision_front_src =
"extern \"C\" {\n"
"__device__ float rb(float x){unsigned u=__float_as_uint(x);return __uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000);}\n"
"__global__ void round_values(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=rb(x[i]);}\n"
"__global__ void add_pos(float*x,const float*p,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=rb(x[i]+p[i]);}\n"
"}\n";

static float bf16_host(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static CUdeviceptr upload_bf16_raw(const qimg21_shards *shards, const char *name) {
    int idx;
    st_context *st = find_tensor(shards, name, &idx);
    if (!st || strcmp(safetensors_dtype(st, idx), "BF16")) return 0;
    size_t bytes = safetensors_nbytes(st, idx);
    CUdeviceptr result = checked_cuMemAlloc(bytes);
    if (!result || cuMemcpyHtoD(result, safetensors_data(st, idx), bytes) ||
        cuCtxSynchronize()) {
        free_d(&result);
        return 0;
    }
    return result;
}

static int build_position_embedding(const qimg21_shards *shards, int h, int w, float *out) {
    int idx;
    st_context *st = find_tensor(shards, "model.visual.pos_embed.weight", &idx);
    if (!st || strcmp(safetensors_dtype(st, idx), "BF16") ||
        safetensors_nbytes(st, idx) != (size_t)48 * 48 * 1152 * 2) return -1;
    const uint16_t *table = safetensors_data(st, idx);
    for (int token = 0; token < h * w; token++) {
        int in_col = token % 2;
        int in_row = (token / 2) % 2;
        int block_col = (token / 4) % (w / 2);
        int block_row = token / (4 * (w / 2));
        int row = block_row * 2 + in_row;
        int col = block_col * 2 + in_col;
        float yf = h == 1 ? 0.0f : (float)row * 47.0f / (float)(h - 1);
        float xf = w == 1 ? 0.0f : (float)col * 47.0f / (float)(w - 1);
        int y0 = (int)floorf(yf), x0 = (int)floorf(xf);
        int y1 = y0 < 47 ? y0 + 1 : y0, x1 = x0 < 47 ? x0 + 1 : x0;
        float wy = yf - y0, wx = xf - x0;
        for (int d = 0; d < 1152; d++) {
            float a = bf16_host(table[((y0 * 48 + x0) * 1152) + d]);
            float b = bf16_host(table[((y0 * 48 + x1) * 1152) + d]);
            float c = bf16_host(table[((y1 * 48 + x0) * 1152) + d]);
            float e = bf16_host(table[((y1 * 48 + x1) * 1152) + d]);
            out[(size_t)token * 1152 + d] =
                a * (1.0f - wy) * (1.0f - wx) + b * (1.0f - wy) * wx +
                c * wy * (1.0f - wx) + e * wy * wx;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model = NULL, *pixels = NULL, *out = NULL, *patch_out = NULL;
    int h = 0, w = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--pixel-values") && i + 1 < argc) pixels = argv[++i];
        else if (!strcmp(argv[i], "--grid-height") && i + 1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grid-width") && i + 1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--patch-out") && i + 1 < argc) patch_out = argv[++i];
        else return 2;
    }
    if (!model || !pixels || !out || h < 1 || w < 1 || h % 2 || w % 2 || h * w > 4096) {
        fprintf(stderr, "usage: %s --model DIR --pixel-values PATCHES.npy --grid-height H --grid-width W --out BLOCK_INPUT.npy\n", argv[0]);
        return 2;
    }
    npy_f32 input = {0};
    if (npy_read_f32(pixels, &input) || input.ndim != 2 ||
        input.shape[0] != (size_t)h * w || input.shape[1] != 1536) return 1;
    qimg21_shards shards = {{0}, 0};
    char path[2048];
    for (int i = 1; i <= 4; i++) {
        snprintf(path, sizeof(path), "%s/text_encoder/model-%05d-of-00004.safetensors", model, i);
        shards.st[shards.n] = safetensors_open(path);
        if (!shards.st[shards.n]) goto fail;
        shards.n++;
    }
    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    CUmodule module = NULL;
    CUfunction add_pos = NULL, round_values = NULL;
    CUdeviceptr x = 0, bf = 0, projected = 0, weight = 0, bias = 0, pos = 0;
    float *host_pos = NULL, *host_out = NULL;
    int rc = 1, n = h * w, count = n * 1152;
    if (!r || cu_compile_kernels(&module, r->device, vision_front_src,
                                  "qimg21_vision_front.cu", 1, "qimg21_vision_front") < 0 ||
        cuModuleGetFunction(&add_pos, module, "add_pos") ||
        cuModuleGetFunction(&round_values, module, "round_values")) goto done;
    x = checked_cuMemAlloc((size_t)n * 1536 * 4);
    bf = checked_cuMemAlloc((size_t)n * 1536 * 2);
    if (!x || !bf || cuMemcpyHtoD(x, input.data, (size_t)n * 1536 * 4) || cuCtxSynchronize() ||
        launch_cast(r, bf, x, n * 1536)) goto done;
    free_d(&x);
    x = checked_cuMemAlloc((size_t)count * 4);
    projected = checked_cuMemAlloc((size_t)count * 2);
    weight = upload_bf16_raw(&shards, "model.visual.patch_embed.proj.weight");
    bias = upload_f32(&shards, "model.visual.patch_embed.proj.bias");
    pos = checked_cuMemAlloc((size_t)count * 4);
    host_pos = malloc((size_t)count * 4);
    host_out = malloc((size_t)count * 4);
    if (!x || !projected || !weight || !bias || !pos || !host_pos || !host_out ||
        cuCtxSynchronize() ||
        build_position_embedding(&shards, h, w, host_pos) ||
        cuMemcpyHtoD(pos, host_pos, (size_t)count * 4) ||
        cublasew_gemm_bf16_bf16_bf16_rowmajor_nt(r->cublaslt_ctx, projected, weight,
                                                  bf, n, 1152, 1536) ||
        cuCtxSynchronize()) goto done;
    void *bias_args[] = {&x, &projected, &bias, &(int){1152}, &n};
    if (cuLaunchKernel(r->bf16_to_f32_add_bias, (count + 255) / 256, 1, 1,
        256, 1, 1, 0, r->stream, bias_args, NULL)) goto done;
    void *round_args[] = {&x, &count};
    if (cuLaunchKernel(round_values, (count + 255) / 256, 1, 1, 256, 1, 1, 0,
                       r->stream, round_args, NULL) || cuStreamSynchronize(r->stream)) goto done;
    if (patch_out && (cuMemcpyDtoH(host_out, x, (size_t)count * 4) ||
                      npy_write_f32(patch_out, host_out, (size_t)count, n, 1152))) goto done;
    void *pos_args[] = {&x, &pos, &count};
    if (cuLaunchKernel(add_pos, (count + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, pos_args, NULL) ||
        cuStreamSynchronize(r->stream) || cuMemcpyDtoH(host_out, x, (size_t)count * 4) ||
        npy_write_f32(out, host_out, (size_t)count, n, 1152)) goto done;
    rc = 0;
done:
    free(host_pos); free(host_out);
    free_d(&x); free_d(&bf); free_d(&projected); free_d(&weight); free_d(&bias); free_d(&pos);
    if (module) cuModuleUnload(module);
    if (r) cuda_qimg_free(r);
    for (int i = 0; i < shards.n; i++) safetensors_close(shards.st[i]);
    npy_free(&input);
    return rc;
fail:
    for (int i = 0; i < shards.n; i++) safetensors_close(shards.st[i]);
    npy_free(&input);
    return 1;
}
