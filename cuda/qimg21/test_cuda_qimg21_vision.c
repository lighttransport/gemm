/* Native Qwen3-VL vision front end.  This first-stage validator covers patch
 * projection plus learned-position interpolation before the 27 vision blocks. */
#define main q21_denoise_main
#include "test_cuda_qimg21_native.c"
#undef main

typedef int (*q21_cutlass_vision_attention_fn)(float *, const void *, int, int, int, CUstream);

static const char *vision_front_src =
"extern \"C\" {\n"
"__device__ float rb(float x){unsigned u=__float_as_uint(x);return __uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000);}\n"
"__global__ void add_pos(float*x,const float*p,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=rb(x[i]+p[i]);}\n"
"__global__ void layer_norm(float*y,const float*x,const float*w,const float*b,int d){int r=blockIdx.x,t=threadIdx.x;__shared__ float m,iv;if(t==0){float s=0,q=0;for(int i=0;i<d;i++){float v=x[r*d+i];s+=v;q+=v*v;}m=s/d;iv=rsqrtf(q/d-m*m+1e-6f);}__syncthreads();for(int i=t;i<d;i+=256)y[r*d+i]=rb((x[r*d+i]-m)*iv*w[i]+b[i]);}\n"
"__global__ void linear_epilogue(float*y,const float*x,const float*b,int d,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*d)y[i]=rb(x[i]+b[i%d]);}\n"
"__global__ void vision_rope(float*qkv,int n,int gh,int gw){int t=blockIdx.x,h=blockIdx.y,j=threadIdx.x;if(t>=n||h>=16||j>=36)return;int ic=t%2,ir=(t/2)%2,bc=(t/4)%(gw/2),br=t/(4*(gw/2));int row=br*2+ir,col=bc*2+ic,coord=j<18?row:col,k=j%18;float inv=1.f/powf(10000.f,(float)(2*k)/36.f),a=coord*inv,c=cosf(a),s=sinf(a);for(int z=0;z<2;z++){int base=t*3456+z*1152+h*72;float u=qkv[base+j],v=qkv[base+j+36];qkv[base+j]=rb(u*c-v*s);qkv[base+j+36]=rb(v*c+u*s);}}\n"
"__global__ void vision_attn(float*o,const float*qkv,int n){int q=blockIdx.x,h=blockIdx.y,l=threadIdx.x;float qr[3]={0},acc[3]={0};for(int e=0;e<3;e++){int d=l+32*e;if(d<72)qr[e]=qkv[q*3456+h*72+d];}float mx=-1e30f;for(int k=0;k<n;k++){float z=0;for(int e=0;e<3;e++){int d=l+32*e;if(d<72)z+=qr[e]*qkv[k*3456+1152+h*72+d];}for(int s=16;s;s>>=1)z+=__shfl_xor_sync(0xffffffff,z,s);mx=fmaxf(mx,z*0.11785113019775793f);}float den=0;for(int k=0;k<n;k++){float z=0;for(int e=0;e<3;e++){int d=l+32*e;if(d<72)z+=qr[e]*qkv[k*3456+1152+h*72+d];}for(int s=16;s;s>>=1)z+=__shfl_xor_sync(0xffffffff,z,s);float p=expf(z*0.11785113019775793f-mx);den+=p;for(int e=0;e<3;e++){int d=l+32*e;if(d<72)acc[e]+=p*qkv[k*3456+2304+h*72+d];}}for(int e=0;e<3;e++){int d=l+32*e;if(d<72)o[q*1152+h*72+d]=rb(acc[e]/den);}}\n"
"__global__ void residual(float*x,const float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=rb(x[i]+y[i]);}\n"
"__global__ void gelu_tanh(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=rb(.5f*v*(1.f+tanhf(.7978845608028654f*(v+.044715f*v*v*v))));}}\n"
"__global__ void gelu_exact(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=rb(.5f*v*(1.f+erff(v*.7071067811865475f)));}}\n"
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

static int vision_linear(cuda_qimg_runner *r, CUfunction epilogue,
                         const qimg21_shards *shards, const char *base,
                         CUdeviceptr out, CUdeviceptr in, int rows, int no, int ni) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", base);
    CUdeviceptr weight = upload_bf16_raw(shards, name);
    snprintf(name, sizeof(name), "%s.bias", base);
    CUdeviceptr bias = upload_f32(shards, name);
    CUdeviceptr in_bf = checked_cuMemAlloc((size_t)rows * ni * 2);
    CUdeviceptr result = checked_cuMemAlloc((size_t)rows * no * 4);
    int rc = 1;
    if (!weight || !bias || !in_bf || !result || launch_cast(r, in_bf, in, rows * ni) ||
        cublasew_gemm_bf16_bf16_f32_rowmajor_nt(r->cublaslt_ctx, result, weight,
                                                 in_bf, rows, no, ni) ||
        cuCtxSynchronize()) goto done;
    void *args[] = {&out, &result, &bias, &no, &rows};
    if (cuLaunchKernel(epilogue, (rows * no + 255) / 256, 1, 1, 256, 1, 1, 0,
                       r->stream, args, NULL) || cuStreamSynchronize(r->stream)) goto done;
    rc = 0;
done:
    free_d(&weight); free_d(&bias); free_d(&in_bf); free_d(&result);
    return rc;
}

static int dump_vision(const char *directory, const char *name, CUdeviceptr data,
                       size_t count, int rows, int cols) {
    if (!directory) return 0;
    float *host = malloc(count * sizeof(float));
    char path[2048];
    int length = snprintf(path, sizeof(path), "%s/%s.npy", directory, name);
    int rc = !host || length < 0 || length >= (int)sizeof(path) || cuCtxSynchronize() ||
             cuMemcpyDtoH(host, data, count * sizeof(float)) ||
             npy_write_f32(path, host, count, rows, cols);
    free(host);
    return rc;
}

static int run_merger(cuda_qimg_runner *r, CUfunction layer_norm,
                      CUfunction linear_epilogue, CUfunction gelu_exact,
                      const qimg21_shards *shards, CUdeviceptr x, int tokens,
                      const char *base, int postshuffle, const char *output) {
    int rows = tokens / 4, norm_dim = postshuffle ? 4608 : 1152;
    CUdeviceptr norm = checked_cuMemAlloc((size_t)tokens * 1152 * 4);
    CUdeviceptr hidden = checked_cuMemAlloc((size_t)rows * 4608 * 4);
    CUdeviceptr merged = checked_cuMemAlloc((size_t)rows * 4096 * 4);
    CUdeviceptr nw = 0, nb = 0;
    float *host = NULL;
    char name[256];
    int rc = 1;
    snprintf(name, sizeof(name), "%s.norm.weight", base); nw = upload_f32(shards, name);
    snprintf(name, sizeof(name), "%s.norm.bias", base); nb = upload_f32(shards, name);
    if (!norm || !hidden || !merged || !nw || !nb || cuCtxSynchronize()) goto done;
    int norm_rows = postshuffle ? rows : tokens;
    void *ln[] = {&norm, &x, &nw, &nb, &norm_dim};
    if (cuLaunchKernel(layer_norm, norm_rows, 1, 1, 256, 1, 1, 0, r->stream, ln, NULL) ||
        cuStreamSynchronize(r->stream)) goto done;
    free_d(&nw); free_d(&nb);
    snprintf(name, sizeof(name), "%s.linear_fc1", base);
    if (vision_linear(r, linear_epilogue, shards, name, hidden, norm, rows, 4608, 4608)) goto done;
    int hidden_count = rows * 4608;
    void *ga[] = {&hidden, &hidden_count};
    if (cuLaunchKernel(gelu_exact, (hidden_count + 255) / 256, 1, 1, 256, 1, 1, 0,
                       r->stream, ga, NULL)) goto done;
    snprintf(name, sizeof(name), "%s.linear_fc2", base);
    if (vision_linear(r, linear_epilogue, shards, name, merged, hidden, rows, 4096, 4608)) goto done;
    host = malloc((size_t)rows * 4096 * 4);
    if (!host || cuCtxSynchronize() || cuMemcpyDtoH(host, merged, (size_t)rows * 4096 * 4) ||
        npy_write_f32(output, host, (size_t)rows * 4096, rows, 4096)) goto done;
    rc = 0;
done:
    free(host); free_d(&norm); free_d(&hidden); free_d(&merged); free_d(&nw); free_d(&nb);
    return rc;
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
            float value = a * ((1.0f - wy) * (1.0f - wx));
            value += b * ((1.0f - wy) * wx);
            value += c * (wy * (1.0f - wx));
            value += e * (wy * wx);
            out[(size_t)token * 1152 + d] = qimg21_round_bf16_host(value);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model = NULL, *pixels = NULL, *hidden = NULL, *out = NULL;
    const char *patch_out = NULL, *dump_dir = NULL, *merged_out = NULL, *deepstack_dir = NULL;
    const char *attention_mode = "cutlass";
    int h = 0, w = 0, max_blocks = 0, block_index = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--pixel-values") && i + 1 < argc) pixels = argv[++i];
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = argv[++i];
        else if (!strcmp(argv[i], "--grid-height") && i + 1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grid-width") && i + 1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--patch-out") && i + 1 < argc) patch_out = argv[++i];
        else if (!strcmp(argv[i], "--max-blocks") && i + 1 < argc) max_blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block-index") && i + 1 < argc) block_index = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--merged-out") && i + 1 < argc) merged_out = argv[++i];
        else if (!strcmp(argv[i], "--deepstack-dir") && i + 1 < argc) deepstack_dir = argv[++i];
        else if (!strcmp(argv[i], "--attention") && i + 1 < argc) attention_mode = argv[++i];
        else return 2;
    }
    if (!model || (!!pixels == !!hidden) || !out || h < 1 || w < 1 || h % 2 || w % 2 ||
        h * w > 4096 || max_blocks < 0 || block_index < 0 || block_index > 26 ||
        block_index + max_blocks > 27 || (hidden && max_blocks < 1) ||
        (strcmp(attention_mode, "math") && strcmp(attention_mode, "cutlass"))) {
        fprintf(stderr, "usage: %s --model DIR (--pixel-values PATCHES.npy | --hidden BLOCK_INPUT.npy) "
                        "--grid-height H --grid-width W [--block-index N --max-blocks N] --out OUTPUT.npy\n", argv[0]);
        return 2;
    }
    npy_f32 input = {0};
    if (npy_read_f32(pixels ? pixels : hidden, &input) || input.ndim != 2 ||
        input.shape[0] != (size_t)h * w || input.shape[1] != (size_t)(pixels ? 1536 : 1152)) return 1;
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
    CUfunction add_pos = NULL, layer_norm = NULL;
    CUfunction linear_epilogue = NULL, vision_rope = NULL, vision_attn = NULL;
    CUfunction residual = NULL, gelu_tanh = NULL, gelu_exact = NULL;
    CUdeviceptr x = 0, bf = 0, projected = 0, weight = 0, bias = 0, pos = 0;
    CUdeviceptr norm = 0, qkv = 0, qkv_bf = 0, att = 0, tmp = 0, mlp = 0;
    void *cutlass_plugin = NULL;
    q21_cutlass_vision_attention_fn cutlass_attention = NULL;
    float *host_pos = NULL, *host_out = NULL;
    int rc = 1, n = h * w, count = n * 1152;
    if (!r || cu_compile_kernels(&module, r->device, vision_front_src,
                                  "qimg21_vision_front.cu", 1, "qimg21_vision_front") < 0 ||
        cuModuleGetFunction(&add_pos, module, "add_pos") ||
        cuModuleGetFunction(&layer_norm, module, "layer_norm") ||
        cuModuleGetFunction(&linear_epilogue, module, "linear_epilogue") ||
        cuModuleGetFunction(&vision_rope, module, "vision_rope") ||
        cuModuleGetFunction(&vision_attn, module, "vision_attn") ||
        cuModuleGetFunction(&residual, module, "residual") ||
        cuModuleGetFunction(&gelu_tanh, module, "gelu_tanh") ||
        cuModuleGetFunction(&gelu_exact, module, "gelu_exact")) goto done;
    if (!strcmp(attention_mode, "cutlass")) {
        cutlass_plugin = dlopen("cuda/qimg21/libq21_cutlass_attention.so", RTLD_NOW | RTLD_LOCAL);
        if (!cutlass_plugin || !(cutlass_attention = (q21_cutlass_vision_attention_fn)
              dlsym(cutlass_plugin, "q21_cutlass_vision_attention"))) {
            fprintf(stderr, "vision: CUTLASS attention plugin unavailable\n");
            goto done;
        }
    }
    if (hidden) {
        x = checked_cuMemAlloc((size_t)count * 4);
        host_out = malloc((size_t)count * 4);
        if (!x || !host_out || cuMemcpyHtoD(x, input.data, (size_t)count * 4) ||
            cuCtxSynchronize()) goto done;
        goto blocks_ready;
    }
    x = checked_cuMemAlloc((size_t)n * 1536 * 4);
    bf = checked_cuMemAlloc((size_t)n * 1536 * 2);
    if (!x || !bf || cuMemcpyHtoD(x, input.data, (size_t)n * 1536 * 4) || cuCtxSynchronize() ||
        launch_cast(r, bf, x, n * 1536)) goto done;
    free_d(&x);
    x = checked_cuMemAlloc((size_t)count * 4);
    projected = checked_cuMemAlloc((size_t)count * 4);
    weight = upload_bf16_raw(&shards, "model.visual.patch_embed.proj.weight");
    bias = upload_f32(&shards, "model.visual.patch_embed.proj.bias");
    pos = checked_cuMemAlloc((size_t)count * 4);
    host_pos = malloc((size_t)count * 4);
    host_out = malloc((size_t)count * 4);
    if (!x || !projected || !weight || !bias || !pos || !host_pos || !host_out ||
        cuCtxSynchronize() ||
        build_position_embedding(&shards, h, w, host_pos) ||
        cuMemcpyHtoD(pos, host_pos, (size_t)count * 4) ||
        cublasew_gemm_bf16_bf16_f32_rowmajor_nt(r->cublaslt_ctx, projected, weight,
                                                 bf, n, 1152, 1536) ||
        cuCtxSynchronize()) goto done;
    void *bias_args[] = {&x, &projected, &bias, &(int){1152}, &n};
    if (cuLaunchKernel(linear_epilogue, (count + 255) / 256, 1, 1, 256, 1, 1, 0,
                       r->stream, bias_args, NULL) || cuStreamSynchronize(r->stream)) goto done;
    if (patch_out && (cuMemcpyDtoH(host_out, x, (size_t)count * 4) ||
                      npy_write_f32(patch_out, host_out, (size_t)count, n, 1152))) goto done;
    void *pos_args[] = {&x, &pos, &count};
    if (cuLaunchKernel(add_pos, (count + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, pos_args, NULL) ||
        cuStreamSynchronize(r->stream)) goto done;
blocks_ready:
    norm = checked_cuMemAlloc((size_t)count * 4);
    qkv = checked_cuMemAlloc((size_t)n * 3456 * 4);
    qkv_bf = checked_cuMemAlloc((size_t)n * 3456 * 2);
    att = checked_cuMemAlloc((size_t)count * 4);
    tmp = checked_cuMemAlloc((size_t)count * 4);
    mlp = checked_cuMemAlloc((size_t)n * 4304 * 4);
    if (max_blocks && (!norm || !qkv || !qkv_bf || !att || !tmp || !mlp)) goto done;
    for (int block = block_index; block < block_index + max_blocks; block++) {
        char name[256];
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.weight", block);
        CUdeviceptr nw = upload_f32(&shards, name);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.bias", block);
        CUdeviceptr nb = upload_f32(&shards, name);
        if (!nw || !nb || cuCtxSynchronize()) { free_d(&nw); free_d(&nb); goto done; }
        void *ln1[] = {&norm, &x, &nw, &nb, &(int){1152}};
        if (cuLaunchKernel(layer_norm, n, 1, 1, 256, 1, 1, 0, r->stream, ln1, NULL) ||
            cuStreamSynchronize(r->stream)) { free_d(&nw); free_d(&nb); goto done; }
        free_d(&nw); free_d(&nb);
        if (block == block_index && dump_vision(dump_dir, "norm1", norm, (size_t)count, n, 1152)) goto done;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv", block);
        if (vision_linear(r, linear_epilogue, &shards, name, qkv, norm, n, 3456, 1152)) goto done;
        if (block == block_index && dump_vision(dump_dir, "qkv", qkv, (size_t)n * 3456, n, 3456)) goto done;
        void *rope_args[] = {&qkv, &n, &h, &w};
        if (cuLaunchKernel(vision_rope, n, 16, 1, 36, 1, 1, 0, r->stream, rope_args, NULL)) goto done;
        if (cutlass_attention) {
            if (launch_cast(r, qkv_bf, qkv, n * 3456) ||
                cutlass_attention((float *)(uintptr_t)att, (const void *)(uintptr_t)qkv_bf,
                                  n, 16, 72, r->stream) || cuStreamSynchronize(r->stream)) goto done;
        } else {
            void *att_args[] = {&att, &qkv, &n};
            if (cuLaunchKernel(vision_attn, n, 16, 1, 32, 1, 1, 0, r->stream, att_args, NULL) ||
                cuStreamSynchronize(r->stream)) goto done;
        }
        if (block == block_index && dump_vision(dump_dir, "attn", att, (size_t)count, n, 1152)) goto done;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj", block);
        if (vision_linear(r, linear_epilogue, &shards, name, tmp, att, n, 1152, 1152)) goto done;
        if (block == block_index && dump_vision(dump_dir, "attn_proj", tmp, (size_t)count, n, 1152)) goto done;
        void *res1[] = {&x, &tmp, &count};
        if (cuLaunchKernel(residual, (count + 255) / 256, 1, 1, 256, 1, 1, 0,
                           r->stream, res1, NULL)) goto done;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.weight", block);
        nw = upload_f32(&shards, name);
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.bias", block);
        nb = upload_f32(&shards, name);
        if (!nw || !nb || cuCtxSynchronize()) { free_d(&nw); free_d(&nb); goto done; }
        void *ln2[] = {&norm, &x, &nw, &nb, &(int){1152}};
        if (cuLaunchKernel(layer_norm, n, 1, 1, 256, 1, 1, 0, r->stream, ln2, NULL) ||
            cuStreamSynchronize(r->stream)) { free_d(&nw); free_d(&nb); goto done; }
        free_d(&nw); free_d(&nb);
        if (block == block_index && dump_vision(dump_dir, "norm2", norm, (size_t)count, n, 1152)) goto done;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc1", block);
        if (vision_linear(r, linear_epilogue, &shards, name, mlp, norm, n, 4304, 1152)) goto done;
        if (block == block_index && dump_vision(dump_dir, "mlp_fc1", mlp, (size_t)n * 4304, n, 4304)) goto done;
        void *gelu_args[] = {&mlp, &(int){n * 4304}};
        if (cuLaunchKernel(gelu_tanh, (n * 4304 + 255) / 256, 1, 1, 256, 1, 1, 0,
                           r->stream, gelu_args, NULL)) goto done;
        if (block == block_index && dump_vision(dump_dir, "mlp_gelu", mlp, (size_t)n * 4304, n, 4304)) goto done;
        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.linear_fc2", block);
        if (vision_linear(r, linear_epilogue, &shards, name, tmp, mlp, n, 1152, 4304)) goto done;
        if (block == block_index && dump_vision(dump_dir, "mlp_fc2", tmp, (size_t)count, n, 1152)) goto done;
        if (cuLaunchKernel(residual, (count + 255) / 256, 1, 1, 256, 1, 1, 0,
                           r->stream, res1, NULL) || cuStreamSynchronize(r->stream)) goto done;
        if (deepstack_dir && (block == 8 || block == 16 || block == 24)) {
            int merger = block == 8 ? 0 : block == 16 ? 1 : 2;
            char base[256], destination[2048];
            snprintf(base, sizeof(base), "model.visual.deepstack_merger_list.%d", merger);
            snprintf(destination, sizeof(destination), "%s/deepstack_%d.npy", deepstack_dir, merger);
            if (run_merger(r, layer_norm, linear_epilogue, gelu_exact, &shards, x, n,
                           base, 1, destination)) goto done;
        }
        fprintf(stderr, "vision: block %d/27\n", block + 1);
    }
    if (merged_out && run_merger(r, layer_norm, linear_epilogue, gelu_exact, &shards, x, n,
                                 "model.visual.merger", 0, merged_out)) goto done;
    if (cuMemcpyDtoH(host_out, x, (size_t)count * 4) ||
        npy_write_f32(out, host_out, (size_t)count, n, 1152)) goto done;
    rc = 0;
done:
    free(host_pos); free(host_out);
    free_d(&x); free_d(&bf); free_d(&projected); free_d(&weight); free_d(&bias); free_d(&pos);
    free_d(&norm); free_d(&qkv); free_d(&qkv_bf); free_d(&att); free_d(&tmp); free_d(&mlp);
    if (cutlass_plugin) dlclose(cutlass_plugin);
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
