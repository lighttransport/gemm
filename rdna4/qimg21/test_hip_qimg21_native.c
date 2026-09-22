/*
 * Native Qwen-Image 2.1 transformer bring-up.
 *
 * This is deliberately a small, inspectable C/HIP path. It uses the
 * repository's dynamic HIP/HIPRTC loader and gfx12 BF16 WMMA GEMM kernels,
 * while owning the Qwen-Image 2.1 block math (zero-centred RMSNorm, Ada modulation,
 * 3-axis RoPE, block-causal attention, SwiGLU and the residual gates).
 *
 * The first native milestone consumes the text-encoder output from a .npy
 * fixture.  Keeping tokenisation/text/VAE out of this executable makes the
 * transformer kernel comparison deterministic and fits the 16 GB RDNA4 card.
 * The Python reference runner produces the fixture.
 */

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "quant_weights.h"
#include "hip_qimg21_runner.h"

/* Map the original Qwen 2.1 driver-shaped calls to HIP before the editing
 * helper is included; that helper compiles its own runtime kernels. */
#define cu_compile_kernels hip_compile_kernels
#define cuFuncSetAttribute hipFuncSetAttribute
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleUnload hipModuleUnload

#include <errno.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "edit_runtime.h"
#include "mma64_kernels.h"
#include "norm_vector_kernels.h"
#include "rope_table_kernels.h"

typedef struct {
    st_context *st[4];
    int n;
} qimg21_shards;

typedef struct {
    float *data;
    size_t n;
    int ndim;
    size_t shape[4];
} npy_f32;

static void npy_free(npy_f32 *a) { free(a->data); memset(a, 0, sizeof(*a)); }

/* Minimal little-endian, C-order F32 .npy reader.  The fixtures are produced
 * by numpy.save; rejecting everything else avoids silently transposing a
 * prompt embedding. */
static int npy_read_f32(const char *path, npy_f32 *out) {
    FILE *fp = fopen(path, "rb");
    char magic[6], header[65536];
    uint8_t ver[2];
    uint16_t h16 = 0;
    uint32_t h32 = 0;
    size_t hlen, pos, n = 1;
    memset(out, 0, sizeof(*out));
    if (!fp) { fprintf(stderr, "native: cannot open %s: %s\n", path, strerror(errno)); return -1; }
    if (fread(magic, 1, 6, fp) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0 ||
        fread(ver, 1, 2, fp) != 2) { fclose(fp); return -1; }
    if (ver[0] == 1) {
        if (fread(&h16, 2, 1, fp) != 1) { fclose(fp); return -1; }
        hlen = h16;
    } else if (ver[0] == 2 || ver[0] == 3) {
        if (fread(&h32, 4, 1, fp) != 1) { fclose(fp); return -1; }
        hlen = h32;
    } else { fclose(fp); return -1; }
    if (hlen == 0 || hlen >= sizeof(header) || fread(header, 1, hlen, fp) != hlen) {
        fclose(fp); return -1;
    }
    header[hlen] = 0;
    if (!strstr(header, "'descr': '<f4'") && !strstr(header, "\"descr\": \"<f4\"")) {
        fprintf(stderr, "native: %s is not little-endian F32\n", path); fclose(fp); return -1;
    }
    if (strstr(header, "fortran_order': True") || strstr(header, "fortran_order\": True")) {
        fprintf(stderr, "native: Fortran-order fixture is unsupported: %s\n", path); fclose(fp); return -1;
    }
    char *shape = strstr(header, "shape");
    if (!shape) { fclose(fp); return -1; }
    shape = strchr(shape, '(');
    if (!shape) { fclose(fp); return -1; }
    pos = (size_t)(shape - header) + 1;
    out->ndim = 0;
    while (pos < hlen && header[pos] != ')') {
        char *end;
        unsigned long long v;
        while (pos < hlen && (header[pos] == ' ' || header[pos] == ',')) pos++;
        if (header[pos] == ')') break;
        v = strtoull(header + pos, &end, 10);
        if (end == header + pos || out->ndim >= 4 || v == 0) { fclose(fp); return -1; }
        out->shape[out->ndim++] = (size_t)v;
        n *= (size_t)v;
        pos = (size_t)(end - header);
        while (pos < hlen && header[pos] != ',' && header[pos] != ')') pos++;
    }
    out->data = (float *)malloc(n * sizeof(float));
    if (!out->data || fread(out->data, sizeof(float), n, fp) != n) {
        fclose(fp); npy_free(out); return -1;
    }
    fclose(fp); out->n = n; return 0;
}

static int npy_write_f32(const char *path, const float *x, size_t n, int d0, int d1) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "native: cannot write %s: %s\n", path, strerror(errno)); return -1; }
    char hdr[256];
    int len = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", d0, d1);
    int padded = ((len + 10 + 63) / 64) * 64 - 10;
    if (padded >= (int)sizeof(hdr)) { fclose(fp); return -1; }
    /* numpy v1 header: magic+version+uint16 length, then 16-byte aligned dict. */
    char body[256];
    memset(body, ' ', (size_t)padded);
    memcpy(body, hdr, (size_t)len);
    body[padded - 1] = '\n';
    uint16_t h = (uint16_t)padded;
    int ok = fwrite("\x93NUMPY\x01\x00", 1, 8, fp) == 8 &&
             fwrite(&h, 2, 1, fp) == 1 &&
             fwrite(body, 1, (size_t)padded, fp) == (size_t)padded &&
             fwrite(x, sizeof(float), n, fp) == n ? 0 : -1;
    /* Buffered writes can succeed even when the final flush fails (ENOSPC). */
    if (fclose(fp) != 0) ok = -1;
    if (ok) fprintf(stderr, "native: failed writing %s; output may be incomplete\n", path);
    return ok;
}

/* Optional full-tensor stage dumps used to close the native/PyTorch parity
 * loop.  They are opt-in so normal runs do not copy 100+ MiB of activations
 * back to the host. */
static const char *qimg21_stage_dir;
static int qimg21_stage_block = 0;
static int qimg21_stage_all_blocks;
static int qimg21_stage_error;
static const char *qimg21_replay_hidden;
static const char *qimg21_replay_attention;

static void dump_stage(const char *label, CUdeviceptr d, size_t n, int d0, int d1) {
    if (!qimg21_stage_dir) return;
    /* Bound diagnostics for editing sequences: a full block dump can exceed
     * a gigabyte. Exact comma-separated labels avoid copying unused stages. */
    const char *selected=getenv("QIMG21_STAGE_KEYS");
    if(selected) {
        int found=0;
        size_t length=strlen(label);
        for(const char *p=selected;*p;) {
            const char *end=strchr(p,',');
            size_t size=end?(size_t)(end-p):strlen(p);
            if(size==length && !memcmp(p,label,length)){found=1;break;}
            if(!end)break;
            p=end+1;
        }
        if(!found)return;
    }
    char path[1024];
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) { qimg21_stage_error = 1; return; }
    if (cuMemcpyDtoH(h, d, n * sizeof(float)) == CUDA_SUCCESS) {
        snprintf(path, sizeof(path), "%s/%s.npy", qimg21_stage_dir, label);
        if (npy_write_f32(path, h, n, d0, d1)) qimg21_stage_error = 1;
    } else qimg21_stage_error = 1;
    free(h);
}

/* FlowMatchEulerDiscreteScheduler schedule used by the local qimg-21
 * scheduler_config.json.  The pipeline supplies the transformer t as sigma
 * in [0,1] (the scheduler's stored timestep is sigma*1000). */
static void qimg21_flow_sigmas(int steps, int image_tokens, float *sigmas) {
    const float base_seq = 256.0f, max_seq = 8192.0f;
    const float base_shift = 0.5f, max_shift = 0.9f;
    const float mu = image_tokens * (max_shift - base_shift) /
                     (max_seq - base_seq) + base_shift -
                     (max_shift - base_shift) / (max_seq - base_seq) * base_seq;
    const float emu = expf(mu);
    for (int i = 0; i < steps; i++) {
        /* Pipeline input is linspace(1, 1/steps, steps), not linspace(1,0).
         * Terminal stretching happens after the nonlinear dynamic shift. */
        float u = 1.0f - (float)i / (float)steps;
        sigmas[i] = emu / (emu + (1.0f / u - 1.0f));
    }
    /* With one inference step the scheduler has no interior endpoint to
     * stretch; it is simply the t=1 denoiser call followed by sigma=0. */
    if (steps == 1) {
        sigmas[0] = 1.0f;
        sigmas[1] = 0.0f;
        return;
    }
    /* shift_terminal=0.02: stretch so the last requested sigma is .02. */
    const float scale = (1.0f - sigmas[steps - 1]) / (1.0f - 0.02f);
    for (int i = 0; i < steps; i++) sigmas[i] = 1.0f - (1.0f - sigmas[i]) / scale;
    sigmas[steps] = 0.0f;
}

static st_context *find_tensor(const qimg21_shards *s, const char *name, int *idx) {
    for (int i = 0; i < s->n; i++) {
        int k = safetensors_find(s->st[i], name);
        if (k >= 0) { if (idx) *idx = k; return s->st[i]; }
    }
    return NULL;
}

static const char *qimg21_quantized_transformer;
static int qimg21_quantize_on_load;
static int qimg21_int8_tensor_core;
static int qimg21_int8_bf16_tail_blocks;
static int qimg21_int8_force_bf16;
static CUdeviceptr qimg21_int8_input_f32;
static size_t qimg21_int8_input_f32_bytes;
static unsigned long long qimg21_int8_mma_calls;

static CUdeviceptr upload_bf16(const qimg21_shards *s, const char *name) {
    int idx; st_context *st = find_tensor(s, name, &idx);
    if (!st) { fprintf(stderr, "native: missing tensor %s\n", name); return 0; }
    if ((qimg21_quantized_transformer || qimg21_quantize_on_load) && safetensors_ndims(st, idx) == 2) {
        const uint64_t *shape = safetensors_shape(st, idx);
        char path[2048];
        uint16_t *data;
        if (qimg21_int8_tensor_core && !qimg21_int8_force_bf16) {
            int len = snprintf(path, sizeof(path), "%s/%s.safetensors", qimg21_quantized_transformer, name);
            size_t bytes = 0;
            void *fat = len < 0 || len >= (int)sizeof(path) ? NULL :
                        q21_read_int8_fat(path, shape[0], shape[1], &bytes);
            if (!fat) { fprintf(stderr, "native: invalid/missing INT8 matrix %s\n", name); return 0; }
            CUdeviceptr d = checked_cuMemAlloc(bytes);
            if (d && cuMemcpyHtoD(d, fat, bytes) != CUDA_SUCCESS) { cuMemFree(d); d = 0; }
            free(fat);
            return d;
        } else if (qimg21_quantize_on_load) data = q21_quantize_matrix_on_load(st, idx);
        else {
            int len = snprintf(path, sizeof(path), "%s/%s.safetensors", qimg21_quantized_transformer, name);
            if (len < 0 || len >= (int)sizeof(path)) return 0;
            data = q21_read_int8_matrix(path, shape[0], shape[1]);
        }
        if (!data) { fprintf(stderr, "native: invalid/missing INT8 matrix %s\n", name); return 0; }
        size_t bytes = (size_t)shape[0] * shape[1] * 2;
        CUdeviceptr d = checked_cuMemAlloc(bytes);
        if (d && cuMemcpyHtoD(d, data, bytes) != CUDA_SUCCESS) { cuMemFree(d); d = 0; }
        free(data);
        return d;
    }
    const char *dt = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    size_t n = nbytes / (!strcmp(dt, "F32") ? sizeof(float) : sizeof(uint16_t));
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    if (!strcmp(dt, "BF16")) {
        /* Safetensors exposes a stable mmap view. Copy it directly instead
         * of allocating and touching a second host copy of every matrix. */
        CUdeviceptr d = checked_cuMemAlloc(nbytes);
        if (d && cuMemcpyHtoD(d, src, nbytes) != CUDA_SUCCESS) {
            cuMemFree(d);
            d = 0;
        }
        return d;
    }
    uint16_t *tmp = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!tmp) return 0;
    if (!strcmp(dt, "F32")) {
        const float *f = (const float *)src;
        for (size_t i = 0; i < n; i++) tmp[i] = qimg_f32_to_bf16_rne(f[i]);
    } else { fprintf(stderr, "native: %s dtype %s is not BF16/F32\n", name, dt); free(tmp); return 0; }
    CUdeviceptr d = checked_cuMemAlloc(n * 2);
    if (d && cuMemcpyHtoD(d, tmp, n * 2) != CUDA_SUCCESS) {
        cuMemFree(d);
        d = 0;
    }
    free(tmp); return d;
}

static CUdeviceptr upload_f32(const qimg21_shards *s, const char *name) {
    int idx; st_context *st = find_tensor(s, name, &idx);
    if (!st) { fprintf(stderr, "native: missing tensor %s\n", name); return 0; }
    const char *dt = safetensors_dtype(st, idx);
    size_t n = safetensors_nbytes(st, idx) / (strcmp(dt, "F32") == 0 ? 4 : 2);
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return 0;
    if (!strcmp(dt, "F32")) memcpy(tmp, safetensors_data(st, idx), n * 4);
    else if (!strcmp(dt, "BF16")) {
        const uint16_t *b = (const uint16_t *)safetensors_data(st, idx);
        for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)b[i] << 16; memcpy(&tmp[i], &u, 4); }
    } else { fprintf(stderr, "native: %s dtype %s is not F32/BF16\n", name, dt); free(tmp); return 0; }
    CUdeviceptr d = checked_cuMemAlloc(n * 4);
    if (d) cuMemcpyHtoD(d, tmp, n * 4);
    free(tmp); return d;
}

static void free_d(CUdeviceptr *p) { if (*p) { cuMemFree(*p); *p = 0; } }

static float qimg21_round_bf16_host(float x) {
    uint32_t bits=(uint32_t)qimg_f32_to_bf16_rne(x)<<16;
    float y; memcpy(&y,&bits,sizeof(y)); return y;
}

/* FlowMatch Euler uses a BF16 model output and a scalar F32 dt. PyTorch
 * retains BF16 for that product, adds to the F32-upcast sample, then casts
 * the result back to the prediction dtype. Preserve both rounding points. */
static void qimg21_euler_bf16(float *sample, const float *prediction,
                            size_t n, float sigma, float next_sigma) {
    const float dt=next_sigma-sigma;
    for(size_t i=0;i<n;i++) {
        float update=qimg21_round_bf16_host(dt*qimg21_round_bf16_host(prediction[i]));
        sample[i]=qimg21_round_bf16_host(sample[i]+update);
    }
}

/* Model-specific kernels.  GEMMs are cuBLAS BF16; these kernels are the
 * precision-sensitive pieces that are not delegated to a framework. */
static const char *qimg21_src =
"#ifndef __shfl_xor_sync\n#define __shfl_xor_sync(mask,val,delta) __shfl_xor(val,delta)\n#endif\n"
"extern \"C\" {\n"
"__global__ void zero_rms(float* y,const float* x,const float* w,int N,int D,float eps){int t=blockIdx.x, i=threadIdx.x; extern __shared__ float s[]; float z=0; for(int j=i;j<D;j+=blockDim.x){float v=x[t*D+j];z+=v*v;} s[i]=z; __syncthreads(); for(int q=blockDim.x/2;q;q>>=1){if(i<q)s[i]+=s[i+q];__syncthreads();} float inv=rsqrtf(s[0]/D+eps); for(int j=i;j<D;j+=blockDim.x)y[t*D+j]=x[t*D+j]*inv*(w[j]+1.f);}\n"
"__global__ void gelu_tanh(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));}}\n"
"__global__ void gelu_tanh_ordered(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i],cube=v*v*v,inner=0.7978845608f*fmaf(0.044715f,cube,v);float t=(float)tanh((double)inner);x[i]=(.5f*v)*(1.f+t);}}\n"
"__global__ void silu(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=v/(1.f+expf(-v));}}\n"
"__global__ void round_bf16(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned u=__float_as_uint(x[i]);unsigned l=(u>>16)&1u;u=(u+0x7fffu+l)&0xffff0000u;x[i]=__uint_as_float(u);}}\n"
"__global__ void mul_silu(float*y,const float*a,const float*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=a[i];y[i]=(v/(1.f+expf(-v)))*b[i];}}\n"
"__global__ void mul_silu_bf16(float*y,const float*a,const float*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=a[i];float z=v/(1.f+expf(-v));unsigned u=__float_as_uint(z),l=(u>>16)&1u;z=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float w=z*b[i];u=__float_as_uint(w);l=(u>>16)&1u;y[i]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}}\n"
"__global__ void mod_ln(float*y,const float*x,const float*m,int N,int D,int prefix,int which){int t=blockIdx.x,i=threadIdx.x;extern __shared__ float s[];float z=0;for(int j=i;j<D;j+=blockDim.x)z+=x[t*D+j];s[i]=z;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)s[i]+=s[i+q];__syncthreads();}float mean=s[0]/D;__syncthreads();z=0;for(int j=i;j<D;j+=blockDim.x){float d=x[t*D+j]-mean;z+=d*d;}s[i]=z;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)s[i]+=s[i+q];__syncthreads();}float inv=rsqrtf(s[0]/D+1e-6f);int row=t<prefix?1:0;int base=which*2*D+row*4*D;for(int j=i;j<D;j+=blockDim.x){float n=(x[t*D+j]-mean)*inv;float scale=m[base+j];y[t*D+j]=n*(1.f+scale);}}\n"
"__global__ void mod_ln_precise(float*y,const float*x,const float*m,int N,int D,int prefix,int which){int t=blockIdx.x,i=threadIdx.x;extern __shared__ double sd[];double z=0;for(int j=i;j<D;j+=blockDim.x)z+=(double)x[t*D+j];sd[i]=z;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)sd[i]+=sd[i+q];__syncthreads();}double mean=sd[0]/(double)D;__syncthreads();z=0;for(int j=i;j<D;j+=blockDim.x){double d=(double)x[t*D+j]-mean;z+=d*d;}sd[i]=z;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)sd[i]+=sd[i+q];__syncthreads();}float inv=rsqrtf((float)(sd[0]/(double)D+1e-6));int row=t<prefix?1:0;int base=which*2*D+row*4*D;for(int j=i;j<D;j+=blockDim.x){float n=(float)(((double)x[t*D+j]-mean)*(double)inv);y[t*D+j]=n*(1.f+m[base+j]);}}\n"
"__global__ void mod_ln_bf16(float*y,const float*x,const float*m,int N,int D,int prefix,int which){int t=blockIdx.x,i=threadIdx.x;extern __shared__ float s3[];float z=0;for(int j=i;j<D;j+=blockDim.x)z+=x[t*D+j];s3[i]=z;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)s3[i]+=s3[i+q];__syncthreads();}float mean=s3[0]/D;__syncthreads();z=0;for(int j=i;j<D;j+=blockDim.x){float d=x[t*D+j]-mean;z+=d*d;}s3[i]=z;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)s3[i]+=s3[i+q];__syncthreads();}float inv=rsqrtf(s3[0]/D+1e-6f);int row=t<prefix?1:0;int base=which*2*D+row*4*D;for(int j=i;j<D;j+=blockDim.x){float n=(x[t*D+j]-mean)*inv;unsigned u=__float_as_uint(n),l=(u>>16)&1u;n=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float f=1.f+m[base+j];u=__float_as_uint(f);l=(u>>16)&1u;f=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float v=n*f;u=__float_as_uint(v);l=(u>>16)&1u;y[t*D+j]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}}\n"
"__global__ void gate_res(float*x,const float*y,const float*m,int N,int D,int prefix,int which){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*D)return;int t=i/D,j=i%D,row=t<prefix?1:0;int base=row*4*D+which*2*D+D;x[i]+=tanhf(m[base+j])*y[i];}\n"
"__global__ void gate_res_bf16(float*x,const float*y,const float*m,int N,int D,int prefix,int which){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*D)return;int t=i/D,j=i%D,row=t<prefix?1:0;int base=row*4*D+which*2*D+D;float g=tanhf(m[base+j]);unsigned u=__float_as_uint(g),l=(u>>16)&1u;g=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float p=g*y[i];u=__float_as_uint(p);l=(u>>16)&1u;p=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float z=x[i]+p;u=__float_as_uint(z);l=(u>>16)&1u;x[i]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}\n"
"__global__ void qk_rope(float*q,float*k,const float*qw,const float*kw,int N,int D,int nh,int hd,int prefix,int ih,int iw){int t=blockIdx.x,h=blockIdx.y,j=threadIdx.x;if(t>=N||h>=nh)return;__shared__ float sq[128],sk[128];float aq=0,ak=0;for(int z=j;z<hd;z+=blockDim.x){float v=q[t*D+h*hd+z];aq+=v*v;v=k[t*D+h*hd+z];ak+=v*v;}sq[j]=aq;sk[j]=ak;__syncthreads();for(int z=64;z;z>>=1){if(j<z){sq[j]+=sq[j+z];sk[j]+=sk[j+z];}__syncthreads();}if(j&1)return;float iq=rsqrtf(sq[0]/hd+1e-6f),ik=rsqrtf(sk[0]/hd+1e-6f);int axis,off,pos;if(j<16){axis=16;off=0;pos=t<prefix?t:prefix;}else if(j<72){axis=56;off=16;pos=t<prefix?t:-(ih-ih/2)+(t-prefix)/iw;}else{axis=56;off=72;pos=t<prefix?t:-(iw-iw/2)+(t-prefix)%iw;}int pair=(j-off)&~1;float ang=(float)pos*exp2f(-log2f(10000.f)*(float)(pair)/(float)axis);float c=cosf(ang),sn=sinf(ang);int d0=h*hd+off+pair,d1=d0+1;float x0=q[t*D+d0]*iq*qw[off+pair],x1=q[t*D+d1]*iq*qw[off+pair+1];float y0=k[t*D+d0]*ik*kw[off+pair],y1=k[t*D+d1]*ik*kw[off+pair+1];q[t*D+d0]=x0*c-x1*sn;q[t*D+d1]=x0*sn+x1*c;k[t*D+d0]=y0*c-y1*sn;k[t*D+d1]=y0*sn+y1*c;}\n"
"__device__ float q21_rne(float x){unsigned u=__float_as_uint(x),l=(u>>16)&1u;return __uint_as_float((u+0x7fffu+l)&0xffff0000u);}\n"
/* CPU replay of the observed sm_120 fixture most closely matches 64-key
 * tiles visited backwards with unnormalized probabilities rounded to BF16. Keep
 * this experimental path explicit until full-denoiser parity is measured. */
"__global__ void masked_attn_reverse64(float*o,const float*q,const float*k,const float*v,int N,int P,int nh,int hd){\n"
" int h=blockIdx.x,qi=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x&31,D=nh*hd; if(qi>=N||h>=nh)return;\n"
" float qr[4],acc[4]={0,0,0,0}; for(int e=0;e<4;e++)qr[e]=q[qi*D+h*hd+lane+32*e];\n"
" int limit=qi<P?qi+1:N; float mx=-1e30f,den=0;\n"
" for(int b=((limit-1)/64)*64;b>=0;b-=64){float nm=mx; int end=min(b+64,limit);\n"
"  for(int t=b;t<end;t++){float dot=0;for(int e=0;e<4;e++)dot+=qr[e]*k[t*D+h*hd+lane+32*e];\n"
"   for(int z=16;z;z>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,z);nm=fmaxf(nm,dot*rsqrtf((float)hd));}\n"
"  float alpha=expf(mx-nm);den*=alpha;for(int e=0;e<4;e++)acc[e]*=alpha;\n"
"  for(int t=b;t<end;t++){float dot=0;for(int e=0;e<4;e++)dot+=qr[e]*k[t*D+h*hd+lane+32*e];\n"
"   for(int z=16;z;z>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,z);float p=expf(dot*rsqrtf((float)hd)-nm);den+=p;p=q21_rne(p);\n"
"   for(int e=0;e<4;e++)acc[e]+=p*v[t*D+h*hd+lane+32*e];}mx=nm;}\n"
" for(int e=0;e<4;e++)o[qi*D+h*hd+lane+32*e]=acc[e]/den;\n"
"}\n"
"__global__ void qk_rope_bf16(float*q,float*k,const float*qw,const float*kw,int N,int D,int nh,int hd,int prefix,int ih,int iw){int t=blockIdx.x,h=blockIdx.y,j=threadIdx.x;if(t>=N||h>=nh)return;__shared__ float sq6[128],sk6[128];float aq=0,ak=0;for(int z=j;z<hd;z+=blockDim.x){float v=q[t*D+h*hd+z];aq+=v*v;v=k[t*D+h*hd+z];ak+=v*v;}sq6[j]=aq;sk6[j]=ak;__syncthreads();for(int z=64;z;z>>=1){if(j<z){sq6[j]+=sq6[j+z];sk6[j]+=sk6[j+z];}__syncthreads();}if(j&1)return;float iq=rsqrtf(sq6[0]/hd+1e-6f),ik=rsqrtf(sk6[0]/hd+1e-6f);int axis,off,pos;if(j<16){axis=16;off=0;pos=t<prefix?t:prefix;}else if(j<72){axis=56;off=16;pos=t<prefix?t:-(ih-ih/2)+(t-prefix)/iw;}else{axis=56;off=72;pos=t<prefix?t:-(iw-iw/2)+(t-prefix)%iw;}int pair=(j-off)&~1;float ang=(float)pos*exp2f(-log2f(10000.f)*(float)(pair)/(float)axis);float c=cosf(ang),sn=sinf(ang);int d0=h*hd+off+pair,d1=d0+1;float x0=q21_rne(q[t*D+d0]*iq)*qw[off+pair],x1=q21_rne(q[t*D+d1]*iq)*qw[off+pair+1];float y0=q21_rne(k[t*D+d0]*ik)*kw[off+pair],y1=q21_rne(k[t*D+d1]*ik)*kw[off+pair+1];unsigned u=__float_as_uint(x0),l=(u>>16)&1u;x0=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(x1);l=(u>>16)&1u;x1=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(y0);l=(u>>16)&1u;y0=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(y1);l=(u>>16)&1u;y1=__uint_as_float((u+0x7fffu+l)&0xffff0000u);q[t*D+d0]=x0*c-x1*sn;q[t*D+d1]=x0*sn+x1*c;k[t*D+d0]=y0*c-y1*sn;k[t*D+d1]=y0*sn+y1*c;}\n"
"__global__ void masked_attn(float*o,const float*q,const float*k,const float*v,int N,int P,int nh,int hd){int h=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x&31,qi=blockIdx.y*4+warp;if(h>=nh)return;int D=nh*hd;float qr[4],or_[4];for(int e=0;e<4;e++){int d=lane*4+e;qr[e]=(qi<N&&d<hd)?q[qi*D+h*hd+d]:0;or_[e]=0;}float mi=-1e30f,li=0;extern __shared__ float sm[];float*sk=sm,*sv=sm+32*128;for(int b=0;b<N;b+=32){for(int z=threadIdx.x;z<32*128;z+=128){int kk=z/128,d=z%128,t=b+kk;sk[z]=(t<N)?k[t*D+h*hd+d]:0;sv[z]=(t<N)?v[t*D+h*hd+d]:0;}__syncthreads();for(int kk=0;kk<32;kk++){int kt=b+kk;bool allow=kt<N&&qi<N&&(qi>=P||kt<=qi);if(!allow)continue;float dot=0;for(int e=0;e<4;e++)dot+=qr[e]*sk[kk*128+lane*4+e];for(int z=16;z;z>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,z);float score=dot*rsqrtf((float)hd),nm=fmaxf(mi,score),a=expf(mi-nm),p=expf(score-nm);li=li*a+p;for(int e=0;e<4;e++)or_[e]=or_[e]*a+p*sv[kk*128+lane*4+e];mi=nm;}__syncthreads();}if(qi<N){float il=li>0?1.f/li:0;for(int e=0;e<4;e++){int d=lane*4+e;if(d<hd)o[qi*D+h*hd+d]=or_[e]*il;}}}\n"
"__global__ void masked_attn_precise(float*o,const float*q,const float*k,const float*v,int N,int P,int nh,int hd){int h=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x&31,qi=blockIdx.y*4+warp;if(h>=nh)return;int D=nh*hd;float qr[4],or_[4];for(int e=0;e<4;e++){int d=lane*4+e;qr[e]=(qi<N&&d<hd)?q[qi*D+h*hd+d]:0;or_[e]=0;}extern __shared__ float sm[];float*sk=sm,*sv=sm+32*128;float mx=-1e30f;for(int b=0;b<N;b+=32){for(int z=threadIdx.x;z<32*128;z+=128){int kk=z/128,d=z%128,t=b+kk;sk[z]=(t<N)?k[t*D+h*hd+d]:0;sv[z]=(t<N)?v[t*D+h*hd+d]:0;}__syncthreads();for(int kk=0;kk<32;kk++){int kt=b+kk;if(!(kt<N&&qi<N&&(qi>=P||kt<=qi)))continue;float dot=0;for(int e=0;e<4;e++)dot+=qr[e]*sk[kk*128+lane*4+e];for(int z=16;z;z>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,z);mx=fmaxf(mx,dot*rsqrtf((float)hd));}__syncthreads();}for(int z=16;z;z>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,z));float sum=0;for(int b=0;b<N;b+=32){for(int z=threadIdx.x;z<32*128;z+=128){int kk=z/128,d=z%128,t=b+kk;sk[z]=(t<N)?k[t*D+h*hd+d]:0;sv[z]=(t<N)?v[t*D+h*hd+d]:0;}__syncthreads();for(int kk=0;kk<32;kk++){int kt=b+kk;if(!(kt<N&&qi<N&&(qi>=P||kt<=qi)))continue;float dot=0;for(int e=0;e<4;e++)dot+=qr[e]*sk[kk*128+lane*4+e];for(int z=16;z;z>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,z);float p=expf(dot*rsqrtf((float)hd)-mx);sum+=p;for(int e=0;e<4;e++)or_[e]+=p*sv[kk*128+lane*4+e];}__syncthreads();}if(qi<N){float il=sum>0?1.f/sum:0;for(int e=0;e<4;e++){int d=lane*4+e;if(d<hd)o[qi*D+h*hd+d]=or_[e]*il;}}}\n"
"__global__ void final_ln(float*y,const float*x,const float*s,int N,int D,int prefix){int t=blockIdx.x,i=threadIdx.x;extern __shared__ float z[];float a=0;for(int j=i;j<D;j+=blockDim.x)a+=x[t*D+j];z[i]=a;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)z[i]+=z[i+q];__syncthreads();}float mu=z[0]/D;__syncthreads();a=0;for(int j=i;j<D;j+=blockDim.x){float d=x[t*D+j]-mu;a+=d*d;}z[i]=a;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)z[i]+=z[i+q];__syncthreads();}float iv=rsqrtf(z[0]/D+1e-6f);int row=t<prefix?1:0;for(int j=i;j<D;j+=blockDim.x)y[t*D+j]=(x[t*D+j]-mu)*iv*(1.f+s[row*D+j]);}\n"
"__global__ void final_ln_bf16(float*y,const float*x,const float*s,int N,int D,int prefix){int t=blockIdx.x,i=threadIdx.x;extern __shared__ float z4[];float a=0;for(int j=i;j<D;j+=blockDim.x)a+=x[t*D+j];z4[i]=a;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)z4[i]+=z4[i+q];__syncthreads();}float mu=z4[0]/D;__syncthreads();a=0;for(int j=i;j<D;j+=blockDim.x){float d=x[t*D+j]-mu;a+=d*d;}z4[i]=a;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)z4[i]+=z4[i+q];__syncthreads();}float iv=rsqrtf(z4[0]/D+1e-6f);int row=t<prefix?1:0;for(int j=i;j<D;j+=blockDim.x){float n=(x[t*D+j]-mu)*iv;unsigned u=__float_as_uint(n),l=(u>>16)&1u;n=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float f=1.f+s[row*D+j];u=__float_as_uint(f);l=(u>>16)&1u;f=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float v=n*f;u=__float_as_uint(v);l=(u>>16)&1u;y[t*D+j]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}}\n"
"__global__ void final_ln_precise(float*y,const float*x,const float*s,int N,int D,int prefix){int t=blockIdx.x,i=threadIdx.x;extern __shared__ double z5[];double a=0;for(int j=i;j<D;j+=blockDim.x)a+=(double)x[t*D+j];z5[i]=a;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)z5[i]+=z5[i+q];__syncthreads();}double mu=z5[0]/(double)D;__syncthreads();a=0;for(int j=i;j<D;j+=blockDim.x){double d=(double)x[t*D+j]-mu;a+=d*d;}z5[i]=a;__syncthreads();for(int q=blockDim.x/2;q;q>>=1){if(i<q)z5[i]+=z5[i+q];__syncthreads();}float iv=rsqrtf((float)(z5[0]/(double)D+1e-6));int row=t<prefix?1:0;for(int j=i;j<D;j+=blockDim.x){float n=(float)(((double)x[t*D+j]-mu)*(double)iv);unsigned u=__float_as_uint(n),l=(u>>16)&1u;n=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float f=1.f+s[row*D+j];u=__float_as_uint(f);l=(u>>16)&1u;f=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float v=n*f;u=__float_as_uint(v);l=(u>>16)&1u;y[t*D+j]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}}\n"
"__global__ void final_ln_welford(float*y,const float*x,const float*s,int N,int D,int prefix){int t=blockIdx.x,i=threadIdx.x;extern __shared__ float z[];float mean=0,m2=0,n=0;for(int j=i;j<D;j+=blockDim.x){float v=x[t*D+j];float nn=n+1.f;float d=v-mean;mean+=d/nn;m2+=d*(v-mean);n=nn;}z[i]=mean;z[256+i]=m2;z[512+i]=n;__syncthreads();for(int q=128;q;q>>=1){if(i<q){float n1=z[512+i],n2=z[512+i+q],nt=n1+n2,d=z[i+q]-z[i];z[i]=(n1*z[i]+n2*z[i+q])/nt;z[256+i]+=z[256+i+q]+d*d*n1*n2/nt;z[512+i]=nt;}__syncthreads();}float inv=rsqrtf(z[256]/D+1e-6f);int row=t<prefix?1:0;for(int j=i;j<D;j+=blockDim.x){float nn=(x[t*D+j]-z[0])*inv;unsigned u=__float_as_uint(nn),l=(u>>16)&1u;nn=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float f=1.f+s[row*D+j];u=__float_as_uint(f);l=(u>>16)&1u;f=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float v=nn*f;u=__float_as_uint(v);l=(u>>16)&1u;y[t*D+j]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}}\n"
"__global__ void mod_ln_welford(float*y,const float*x,const float*m,int N,int D,int prefix,int which){int t=blockIdx.x,i=threadIdx.x;extern __shared__ float z[];float mean=0,m2=0,n=0;for(int j=i;j<D;j+=blockDim.x){float v=x[t*D+j],nn=n+1.f,d=v-mean;mean+=d/nn;m2+=d*(v-mean);n=nn;}z[i]=mean;z[256+i]=m2;z[512+i]=n;__syncthreads();for(int q=128;q;q>>=1){if(i<q){float n1=z[512+i],n2=z[512+i+q],nt=n1+n2,d=z[i+q]-z[i];z[i]=(n1*z[i]+n2*z[i+q])/nt;z[256+i]+=z[256+i+q]+d*d*n1*n2/nt;z[512+i]=nt;}__syncthreads();}float inv=rsqrtf(z[256]/D+1e-6f),mean0=z[0];int row=t<prefix?1:0,base=which*2*D+row*4*D;for(int j=i;j<D;j+=blockDim.x){float nn=(x[t*D+j]-mean0)*inv;unsigned u=__float_as_uint(nn),l=(u>>16)&1u;nn=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float f=1.f+m[base+j];u=__float_as_uint(f);l=(u>>16)&1u;f=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float v=nn*f;u=__float_as_uint(v);l=(u>>16)&1u;y[t*D+j]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}}\n"
"__global__ void gate_res_bf16_precise(float*x,const float*y,const float*m,int N,int D,int prefix,int which){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*D)return;int t=i/D,j=i%D,row=t<prefix?1:0;int base=row*4*D+which*2*D+D;float g=(float)tanh((double)m[base+j]);unsigned u=__float_as_uint(g),l=(u>>16)&1u;g=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float p=g*y[i];u=__float_as_uint(p);l=(u>>16)&1u;p=__uint_as_float((u+0x7fffu+l)&0xffff0000u);float z=x[i]+p;u=__float_as_uint(z);l=(u>>16)&1u;x[i]=__uint_as_float((u+0x7fffu+l)&0xffff0000u);}\n"
"__global__ void silu_precise(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){double v=(double)x[i];x[i]=(float)(v/(1.0+exp(-v)));}}\n"
"__global__ void proj_bf16(float*y,const unsigned short*w,const unsigned short*x,int N,int O,int K){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=N*O)return;int t=z/O,o=z%O;float sum=0;for(int j=K-1;j>=0;j--){float xv=__uint_as_float((unsigned)x[t*K+j]<<16);float wv=__uint_as_float((unsigned)w[o*K+j]<<16);sum+=xv*wv;}y[z]=sum;}\n"
"}\n";

typedef struct {
    CUmodule mod;
    CUfunction zero_rms, gelu, silu, round_bf16, mul_silu, mod_ln, mod_ln_precise, gate_res, qk_rope, attn, final_ln, proj;
    CUfunction mma_attention;
    CUfunction table_rope, mma_text;
    int norm_threads;
} qimg21_kernels;

static int qimg21_attention_reverse64;
static int qimg21_attention_mma64;
static int qimg21_use_wmma = 1;
static int qimg21_norm_vector;
static int qimg21_host_rope;
static const char *qimg21_rope_base_path;
typedef int (*qimg21_cutlass_attention_fn)(float *, const void *, const void *,
                                           const void *, int, int, int, int, void *);
static qimg21_cutlass_attention_fn qimg21_cutlass_attention;
typedef int (*qimg21_cutlass_workspace_release_fn)(void);
typedef unsigned (*qimg21_cutlass_workspace_allocations_fn)(void);
static qimg21_cutlass_workspace_release_fn qimg21_cutlass_workspace_release;
static qimg21_cutlass_workspace_allocations_fn qimg21_cutlass_workspace_allocations;
typedef int (*qimg21_exact_rope_fn)(float *, float *, const float *, const float *,
                                    const float *, int, int, void *);
static qimg21_exact_rope_fn qimg21_exact_rope;

static int get_kernel(qimg21_kernels *k, CUmodule m) {
    k->mod = m;
    k->mma_attention = NULL;
    k->table_rope = NULL;
    k->mma_text = NULL;
    k->norm_threads = 256;
    return cuModuleGetFunction(&k->zero_rms, m, "zero_rms") ||
           cuModuleGetFunction(&k->gelu, m, "gelu_tanh_ordered") ||
           cuModuleGetFunction(&k->silu, m, "silu") ||
           cuModuleGetFunction(&k->round_bf16, m, "round_bf16") ||
           cuModuleGetFunction(&k->mul_silu, m, "mul_silu_bf16") ||
           cuModuleGetFunction(&k->mod_ln, m, "mod_ln_bf16") ||
           cuModuleGetFunction(&k->mod_ln_precise, m, "mod_ln_precise") ||
           cuModuleGetFunction(&k->gate_res, m, "gate_res_bf16") ||
           cuModuleGetFunction(&k->qk_rope, m, "qk_rope_bf16") ||
           cuModuleGetFunction(&k->attn, m, qimg21_attention_reverse64 ? "masked_attn_reverse64" : "masked_attn_precise") ||
           cuModuleGetFunction(&k->final_ln, m, "final_ln_precise") ||
           cuModuleGetFunction(&k->proj, m, "proj_bf16");
}

static int launch_cast(cuda_qimg_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *a[] = { &src, &dst, &n };
    CUresult rc = cuLaunchKernel(r->cast_f32_to_bf16, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
    if (rc == CUDA_SUCCESS) rc = cuCtxSynchronize();
    if (rc != CUDA_SUCCESS) fprintf(stderr, "native: cast launch failed rc=%d n=%d\n", rc, n);
    return (int)rc;
}

static int gemm(cuda_qimg_runner *r, CUdeviceptr y, CUdeviceptr w, CUdeviceptr x,
                int nt, int no, int ni) {
    void *args[] = {&y, &w, &x, &no, &ni, &nt};
    CUfunction fn = qimg21_use_wmma && r->gemm_bf16_wmma ? r->gemm_bf16_wmma : r->gemm_bf16;
    int wmma = fn == r->gemm_bf16_wmma;
    unsigned gx = wmma ? (unsigned)((no + 127) / 128) : (unsigned)((no + 15) / 16);
    unsigned gy = wmma ? (unsigned)((nt + 127) / 128) : (unsigned)((nt + 15) / 16);
    CUresult rc = cuLaunchKernel(fn, gx, gy, 1, wmma ? 256 : 16,
                                 wmma ? 1 : 16, 1, 0,
                                 r->stream, args, NULL);
    if (rc == CUDA_SUCCESS) rc = cuCtxSynchronize();
    if (rc != CUDA_SUCCESS) fprintf(stderr, "native: GEMM launch failed rc=%d nt=%d no=%d ni=%d wmma=%d\n", rc, nt, no, ni, wmma);
    return (int)rc;
}

static int launch_vec(CUfunction f, CUstream st, int n, CUdeviceptr x) {
    void *a[] = { &x, &n };
    CUresult rc = cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, st, a, NULL);
    if (rc == CUDA_SUCCESS) rc = cuCtxSynchronize();
    if (rc != CUDA_SUCCESS) fprintf(stderr, "native: vector launch failed rc=%d n=%d\n", rc, n);
    return (int)rc;
}

static void probe(cuda_qimg_runner *r, const char *label, CUdeviceptr d, int n) {
    if (r->verbose < 2) return;
    int m = n < 4096 ? n : 4096, bad = 0;
    float *h = (float *)malloc((size_t)m * sizeof(float));
    if (!h) return;
    cuStreamSynchronize(r->stream);
    if (cuMemcpyDtoH(h, d, (size_t)m * sizeof(float)) == CUDA_SUCCESS) {
        for (int i = 0; i < m; i++) if (!isfinite(h[i])) bad++;
        fprintf(stderr, "native: %-12s first=%+.5e bad=%d/%d\n", label, h[0], bad, m);
        if (getenv("QIMG21_PROBE_VALUES")) {
            int show = m < 8 ? m : 8;
            fprintf(stderr, "native: %-12s values", label);
            for (int i = 0; i < show; i++) fprintf(stderr, " %+.5e", h[i]);
            fputc('\n', stderr);
        }
    }
    free(h);
}

static int native_step(cuda_qimg_runner *r, qimg21_kernels *k, const qimg21_shards *s,
                       const float *prompt, int nt, const float *latent, int ni,
                       int ih, int iw, float timestep, float *out, const q21_edit_context *edit) {
    int D=4096, HD=128, NH=32, N=edit?edit->layout.n:nt+ni;
    int prefix=edit?edit->layout.prefix:nt, nout=N-prefix;
    CUdeviceptr txt=0,img=0,hidden=0,tmp=0,tmp2=0,bf=0,q=0,kk=0,v=0,att=0,mlp0=0,mlp1=0,mod=0,temb=0,time0=0,timebf=0,scale=0;
    CUdeviceptr wt_norm=0,wt_in=0,wt_out=0,wi=0,w_t1=0,w_t2=0,w_mod=0,w_img=0,w_proj=0;
    int result = -1;
    CUdeviceptr rope_table=0;
    #define A(p,bytes) do { (p)=checked_cuMemAlloc(bytes); if(!(p)) goto fail; } while(0)
    if(k->table_rope) {
        float *table=malloc((size_t)N*128*sizeof(float));if(!table)goto fail;
        npy_f32 rope_base={0};
        if(qimg21_rope_base_path && (npy_read_f32(qimg21_rope_base_path,&rope_base) ||
           rope_base.ndim!=2 || rope_base.shape[0]!=9216 || rope_base.shape[1]!=128)) {
            fprintf(stderr,"native: invalid exact RoPE frequency table\n");free(table);goto fail;
        }
        for(int t=0;t<N;t++)for(int j=0;j<128;j+=2) {
            int axis=j<16?0:(j<72?1:2),dim=axis==0?16:56,off=axis==0?0:(axis==1?16:72);
            int pos=edit?edit->layout.position[t*3+axis]:(t<nt?t:(axis==0?nt:
                      (axis==1?-(ih-ih/2)+(t-nt)/iw:-(iw-iw/2)+(t-nt)%iw)));
            if(rope_base.data) {
                int row=pos>=0?pos:8192+pos+1024;
                if(row<0 || row>=9216) {npy_free(&rope_base);free(table);goto fail;}
                table[t*128+j]=rope_base.data[(size_t)row*128+j];
                table[t*128+j+1]=rope_base.data[(size_t)row*128+j+1];
            } else {
                float inverse=1.f/powf(10000.f,(float)(j-off)/(float)dim),angle=(float)pos*inverse;
                table[t*128+j]=cosf(angle);table[t*128+j+1]=sinf(angle);
            }
        }
        npy_free(&rope_base);
        rope_table=checked_cuMemAlloc((size_t)N*128*sizeof(float));
        if(qimg21_stage_dir && (qimg21_stage_block < 0 || qimg21_stage_block < 32)) {
            char table_path[2048];
            snprintf(table_path,sizeof(table_path),"%s/rope_table.npy",qimg21_stage_dir);
            npy_write_f32(table_path,table,(size_t)N*128,N,128);
        }
        int error=!rope_table || cuMemcpyHtoD(rope_table,table,(size_t)N*128*sizeof(float)) || cuCtxSynchronize();
        free(table);if(error)goto fail;
    }
    A(txt,(size_t)nt*D*4); A(img,(size_t)ni*64*4); A(hidden,(size_t)N*D*4);
    /* tmp/bf are also the N*12288 SwiGLU activation hand-off buffers. */
    A(tmp,(size_t)N*12288*4); A(tmp2,(size_t)N*D*4); A(bf,(size_t)N*12288*2);
    A(q,(size_t)N*D*4); A(kk,(size_t)N*D*4); A(v,(size_t)N*D*4); A(att,(size_t)N*D*4); A(mlp0,(size_t)N*12288*4); A(mlp1,(size_t)N*12288*4);
    A(temb,2*D*4); A(time0,512*4); A(timebf,512*2); A(mod,2*16384*4); A(scale,2*D*4);
    cuMemcpyHtoD(txt,prompt,(size_t)nt*D*4); cuMemcpyHtoD(img,latent,(size_t)ni*64*4);
    dump_stage("txt_input", txt, (size_t)nt * D, nt, D);
    dump_stage("img_input", img, (size_t)ni * 64, ni, 64);
    wt_norm=upload_f32(s,"txt_in.text_norm.weight"); wt_in=upload_bf16(s,"txt_in.in_layer.weight"); wt_out=upload_bf16(s,"txt_in.out_layer.weight"); wi=upload_bf16(s,"img_in.weight");
    w_t1=upload_bf16(s,"time_text_embed.timestep_embedder.linear_1.weight"); w_t2=upload_bf16(s,"time_text_embed.timestep_embedder.linear_2.weight"); w_mod=upload_bf16(s,"modulation.1.weight");
    if(!wt_norm||!wt_in||!wt_out||!wi||!w_t1||!w_t2||!w_mod)goto fail;
    {void *a[]={&tmp,&txt,&wt_norm,&nt,&D,&(float){1e-6f}};
     CUresult zrc=cuLaunchKernel(k->zero_rms,nt,1,1,256,1,1,256*sizeof(float),r->stream,a,NULL);
     if(zrc==CUDA_SUCCESS) zrc=cuCtxSynchronize();
     if(zrc!=CUDA_SUCCESS){const char *es="?";if(hipGetErrorString)hipGetErrorString(zrc,&es);fprintf(stderr,"native: initial RMS kernel failed rc=%d (%s)\n",zrc,es);goto fail;}}
    if(launch_cast(r,bf,tmp,nt*D)!=CUDA_SUCCESS||gemm(r,txt,wt_in,bf,nt,D,D)!=0)goto fail;
    /* PyTorch's BF16 Linear returns BF16 before GELU. The GEMM helper
     * accumulates into F32, so preserve that explicit activation boundary. */
    if(launch_vec(k->round_bf16,r->stream,nt*D,txt)!=CUDA_SUCCESS)goto fail;
    if(launch_vec(k->gelu,r->stream,nt*D,txt)!=CUDA_SUCCESS||launch_cast(r,bf,txt,nt*D)!=CUDA_SUCCESS||gemm(r,tmp,wt_out,bf,nt,D,D)!=0||launch_vec(k->round_bf16,r->stream,nt*D,tmp)!=CUDA_SUCCESS)goto fail;
    probe(r,"txt_proj",tmp,nt*D); dump_stage("txt_proj",tmp,(size_t)nt*D,nt,D);
    /* Save the completed text projection before reusing tmp for img_in.  The
     * preceding txt buffer is only GELU(in_layer(...)); the model consumes
     * out_layer(GELU(in_layer(...))). */
    cuMemcpyDtoD(edit?txt:hidden, tmp, (size_t)nt*D*4); cuCtxSynchronize();
    if(launch_cast(r,bf,img,ni*64)!=CUDA_SUCCESS||gemm(r,tmp,wi,bf,ni,D,64)!=0||launch_vec(k->round_bf16,r->stream,ni*D,tmp)!=CUDA_SUCCESS)goto fail;
    probe(r,"img_proj",tmp,ni*D); dump_stage("img_proj",tmp,(size_t)ni*D,ni,D);
    if(edit) {
        void *args[]={&hidden,&txt,&tmp,(void *)&edit->text_index,(void *)&edit->image_index,(void *)&N,(void *)&D};
        if(cuLaunchKernel(edit->scatter,(N*D+255)/256,1,1,256,1,1,0,r->stream,args,NULL))goto fail;
    } else cuMemcpyDtoD(hidden + (size_t)nt*D*4, tmp, (size_t)ni*D*4);
    cuCtxSynchronize();
    probe(r,"hidden",hidden,N*D); dump_stage("hidden0",hidden,(size_t)N*D,N,D);
    /* Match the model's [real timestep, zero timestep] two-row GEMMs.
     * Single-row GEMV dispatch has different reduction/rounding behavior. */
    float te[512]={0};
    for(int i=0;i<128;i++){
        float f=expf(-logf(10000.f)*(float)i/128.f),a=timestep*1000.f*f;
        te[i]=cosf(a);te[128+i]=sinf(a);te[256+i]=1.0f;
    }
    cuMemcpyHtoD(time0,te,sizeof(te));
    if(launch_cast(r,timebf,time0,512)!=CUDA_SUCCESS ||
       gemm(r,temb,w_t1,timebf,2,D,256)!=0 ||
       launch_vec(k->round_bf16,r->stream,2*D,temb)!=CUDA_SUCCESS) goto fail;
    dump_stage("time1",temb,2u*D,2,D);
    if(launch_vec(k->silu,r->stream,2*D,temb)!=CUDA_SUCCESS ||
       launch_vec(k->round_bf16,r->stream,2*D,temb)!=CUDA_SUCCESS ||
       launch_cast(r,bf,temb,2*D)!=CUDA_SUCCESS ||
       gemm(r,temb,w_t2,bf,2,D,D)!=0 ||
       launch_vec(k->round_bf16,r->stream,2*D,temb)!=CUDA_SUCCESS) goto fail;
    dump_stage("time2",temb,2u*D,2,D);
    cuMemcpyDtoD(tmp2,temb,(size_t)2*D*4); cuCtxSynchronize();
    if(launch_vec(k->silu,r->stream,2*D,tmp2)!=CUDA_SUCCESS ||
       launch_vec(k->round_bf16,r->stream,2*D,tmp2)!=CUDA_SUCCESS ||
       launch_cast(r,bf,tmp2,2*D)!=CUDA_SUCCESS ||
       gemm(r,mod,w_mod,bf,2,16384,D)!=0 ||
       launch_vec(k->round_bf16,r->stream,2*16384,mod)!=CUDA_SUCCESS) goto fail;
    probe(r,"mod",mod,2*16384); dump_stage("mod",mod,2u*16384u,2,16384);
    probe(r,"mod_zero",mod+(size_t)16384*4,16384);
    int profile = getenv("QIMG21_PROFILE") != NULL;
    double profile_upload = 0.0, profile_compute = 0.0, profile_release = 0.0;
    for(int bidx=0;bidx<32;bidx++){
        struct timespec prof_start, prof_uploaded, prof_computed, prof_released;
        if(profile)clock_gettime(CLOCK_MONOTONIC,&prof_start);
        qimg21_int8_force_bf16 = qimg21_int8_tensor_core &&
                                 bidx >= 32 - qimg21_int8_bf16_tail_blocks;
        if(qimg21_replay_hidden && bidx<qimg21_stage_block)continue;
        if(qimg21_replay_hidden && bidx==qimg21_stage_block) {
            npy_f32 replay={0};
            if(npy_read_f32(qimg21_replay_hidden,&replay))goto fail;
            int valid=(replay.ndim==2 && replay.shape[0]==(size_t)N && replay.shape[1]==(size_t)D) ||
                      (replay.ndim==3 && replay.shape[0]==1 && replay.shape[1]==(size_t)N && replay.shape[2]==(size_t)D);
            for(size_t i=0;valid && i<replay.n;i++)
                if(!isfinite(replay.data[i]) || replay.data[i]!=qimg21_round_bf16_host(replay.data[i]))valid=0;
            int error=!valid || cuMemcpyHtoD(hidden,replay.data,replay.n*sizeof(float)) || cuCtxSynchronize();
            npy_free(&replay);
            if(error){fprintf(stderr,"native: invalid or failed hidden-state replay\n");goto fail;}
            fprintf(stderr,"native: DIAGNOSTIC ONLY: replaying block %d from external hidden state\n",bidx);
            dump_stage("replay_hidden",hidden,(size_t)N*D,N,D);
        }
        char nm[128];
        snprintf(nm,sizeof(nm),"transformer_blocks.%d.attn.to_q.weight",bidx);CUdeviceptr wq=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.attn.to_k.weight",bidx);CUdeviceptr wk=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.attn.to_v.weight",bidx);CUdeviceptr wv=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.attn.to_out.0.weight",bidx);CUdeviceptr wo=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.img_mlp.gate_layer.weight",bidx);CUdeviceptr wg=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.img_mlp.proj.weight",bidx);CUdeviceptr wp=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.img_mlp.out.weight",bidx);CUdeviceptr wmlpo=upload_bf16(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.attn.norm_q.weight",bidx);CUdeviceptr wqn=upload_f32(s,nm);snprintf(nm,sizeof(nm),"transformer_blocks.%d.attn.norm_k.weight",bidx);CUdeviceptr wkn=upload_f32(s,nm);
        if(!wq||!wk||!wv||!wo||!wg||!wp||!wmlpo||!wqn||!wkn)goto fail_block;
        if(profile)clock_gettime(CLOCK_MONOTONIC,&prof_uploaded);
        void *a1[]={&tmp,&hidden,&mod,&N,&D,&prefix,&(int){0}}; cuLaunchKernel(k->mod_ln,N,1,1,k->norm_threads,1,1,256*sizeof(float),r->stream,a1,NULL); cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*D,tmp)!=CUDA_SUCCESS) goto fail_block; probe(r,"mod_ln",tmp,N*D); if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("mod_ln",tmp,(size_t)N*D,N,D); if(launch_cast(r,bf,tmp,N*D)!=CUDA_SUCCESS)goto fail_block;
        if(gemm(r,q,wq,bf,N,D,D)!=0||gemm(r,kk,wk,bf,N,D,D)!=0||gemm(r,v,wv,bf,N,D,D)!=0||launch_vec(k->round_bf16,r->stream,N*D,q)!=CUDA_SUCCESS||launch_vec(k->round_bf16,r->stream,N*D,kk)!=CUDA_SUCCESS||launch_vec(k->round_bf16,r->stream,N*D,v)!=CUDA_SUCCESS)goto fail_block;
        probe(r,"q",q,N*D); probe(r,"v",v,N*D); if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) { dump_stage("q",q,(size_t)N*D,N,D); dump_stage("k",kk,(size_t)N*D,N,D); dump_stage("v",v,(size_t)N*D,N,D); }
        if(qimg21_exact_rope && qimg21_rope_base_path) {
            if(qimg21_exact_rope((float *)(uintptr_t)q,(float *)(uintptr_t)kk,
               (const float *)(uintptr_t)wqn,(const float *)(uintptr_t)wkn,
               (const float *)(uintptr_t)rope_table,N,NH,(void *)r->stream))goto fail_block;
        }
        else if(k->table_rope) {void *ar[]={&q,&kk,&wqn,&wkn,(void *)&N,(void *)&D,(void *)&NH,(void *)&HD,&nt,&ih,&iw,&rope_table};if(cuLaunchKernel(k->table_rope,N,NH,1,HD,1,1,0,r->stream,ar,NULL))goto fail_block;}
        else if(edit) { void *ar[]={&q,&kk,&wqn,&wkn,(void *)&edit->position,(void *)&N,(void *)&NH}; if(cuLaunchKernel(edit->rope,N,NH,1,HD,1,1,0,r->stream,ar,NULL))goto fail_block; }
        else { void *ar[]={&q,&kk,&wqn,&wkn,&N,&D,&NH,&HD,&nt,&ih,&iw}; cuLaunchKernel(k->qk_rope,N,NH,1,HD,1,1,0,r->stream,ar,NULL); }
        cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*D,q)!=CUDA_SUCCESS||launch_vec(k->round_bf16,r->stream,N*D,kk)!=CUDA_SUCCESS) goto fail_block; probe(r,"rope_q",q,N*D);
        if (qimg21_stage_dir && (qimg21_stage_block < 0 || bidx == qimg21_stage_block)) {
            dump_stage("rope_q",q,(size_t)N*D,N,D);
            dump_stage("rope_k",kk,(size_t)N*D,N,D);
        }
        if(qimg21_replay_attention) {
            npy_f32 replay={0};
            if(npy_read_f32(qimg21_replay_attention,&replay))goto fail_block;
            int valid=(replay.ndim==2 && replay.shape[0]==(size_t)N && replay.shape[1]==(size_t)D) ||
                      (replay.ndim==3 && replay.shape[0]==1 && replay.shape[1]==(size_t)N && replay.shape[2]==(size_t)D);
            for(size_t i=0;valid && i<replay.n;i++)
                if(!isfinite(replay.data[i]) || replay.data[i]!=qimg21_round_bf16_host(replay.data[i]))valid=0;
            int error=!valid || cuMemcpyHtoD(att,replay.data,replay.n*sizeof(float)) || cuCtxSynchronize();
            npy_free(&replay);
            if(error){fprintf(stderr,"native: invalid or failed attention-state replay\n");goto fail_block;}
            fprintf(stderr,"native: DIAGNOSTIC ONLY: injecting attention before output projection\n");
        }
        else if(k->mma_attention || qimg21_cutlass_attention) {
            /* Reuse the 3*D BF16 MLP hand-off allocation for Q/K/V. */
            CUdeviceptr qb=bf,kb=bf+(size_t)N*D*2,vb=bf+(size_t)N*D*4;
            if(launch_cast(r,qb,q,N*D) || launch_cast(r,kb,kk,N*D) || launch_cast(r,vb,v,N*D))goto fail_block;
            for(int start=0;start<N;) {
                int end=start+1;
                if(edit) {
                    if(edit->layout.image_id[start]>=0)
                        while(end<N && edit->layout.image_id[end]==edit->layout.image_id[start])end++;
                } else if(start>=nt)end=N;
                int nq=end-start,nkv=end;
                CUdeviceptr sq=qb+(size_t)start*D*2,so=att+(size_t)start*D*4;
                void *aa[]={&so,&sq,&kb,&vb,&nq,&nkv,(void *)&NH,(void *)&HD};
                int is_text=edit?edit->layout.image_id[start]<0:start<nt;
                CUfunction attention=is_text && k->mma_text?k->mma_text:k->mma_attention;
                if(qimg21_cutlass_attention) {
                    int error=qimg21_cutlass_attention((float *)(uintptr_t)so,
                        (const void *)(uintptr_t)sq,(const void *)(uintptr_t)kb,
                        (const void *)(uintptr_t)vb,nq,nkv,NH,HD,(void *)r->stream);
                    if(error || cuCtxSynchronize())goto fail_block;
                } else if(cuLaunchKernel(attention,NH,(nq+63)/64,1,128,1,1,
                                         4*64*136*2,r->stream,aa,NULL) ||
                          cuCtxSynchronize())goto fail_block;
                start=end;
            }
        }
        else if(edit) { void *aa[]={&att,&q,&kk,&v,(void *)&edit->image_id,(void *)&N,(void *)&NH}; if(cuLaunchKernel(edit->attention,NH,N,1,32,1,1,0,r->stream,aa,NULL))goto fail_block; }
        else { void *aa[]={&att,&q,&kk,&v,&N,&nt,&NH,&HD};cuLaunchKernel(k->attn,NH,(N+3)/4,1,128,1,1,2*32*128*sizeof(float),r->stream,aa,NULL); }
        cuCtxSynchronize(); if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("attn_pre_round",att,(size_t)N*D,N,D); if (launch_vec(k->round_bf16,r->stream,N*D,att)!=CUDA_SUCCESS) goto fail_block; probe(r,"attn",att,N*D); if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("attn_raw",att,(size_t)N*D,N,D); if(launch_cast(r,bf,att,N*D)!=CUDA_SUCCESS||gemm(r,tmp,wo,bf,N,D,D)!=0||launch_vec(k->round_bf16,r->stream,N*D,tmp)!=CUDA_SUCCESS)goto fail_block; probe(r,"attn_out",tmp,N*D); if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("attn_out",tmp,(size_t)N*D,N,D);
        void *ag[]={&hidden,&tmp,&mod,&N,&D,&prefix,&(int){0}};cuLaunchKernel(k->gate_res,(N*D+255)/256,1,1,256,1,1,0,r->stream,ag,NULL);cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*D,hidden)!=CUDA_SUCCESS) goto fail_block; if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("post_attn_hidden",hidden,(size_t)N*D,N,D);
        void *a2[]={&tmp,&hidden,&mod,&N,&D,&prefix,&(int){1}};cuLaunchKernel(k->mod_ln,N,1,1,k->norm_threads,1,1,256*sizeof(float),r->stream,a2,NULL);cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*D,tmp)!=CUDA_SUCCESS) goto fail_block; if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("mod_ln2",tmp,(size_t)N*D,N,D); if(launch_cast(r,bf,tmp,N*D)!=CUDA_SUCCESS)goto fail_block;
        if(gemm(r,mlp0,wg,bf,N,12288,D)!=0||gemm(r,mlp1,wp,bf,N,12288,D)!=0||launch_vec(k->round_bf16,r->stream,N*12288,mlp0)!=CUDA_SUCCESS||launch_vec(k->round_bf16,r->stream,N*12288,mlp1)!=CUDA_SUCCESS)goto fail_block;
        if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) { dump_stage("mlp_gate",mlp0,(size_t)N*12288,N,12288); dump_stage("mlp_proj",mlp1,(size_t)N*12288,N,12288); } void *am[]={&tmp, &mlp0,&mlp1,&(int){N*12288}};cuLaunchKernel(k->mul_silu,(N*12288+255)/256,1,1,256,1,1,0,r->stream,am,NULL);cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*12288,tmp)!=CUDA_SUCCESS) goto fail_block; if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("mlp_act",tmp,(size_t)N*12288,N,12288); if(launch_cast(r,bf,tmp,N*12288)!=CUDA_SUCCESS||gemm(r,tmp,wmlpo,bf,N,D,12288)!=0||launch_vec(k->round_bf16,r->stream,N*D,tmp)!=CUDA_SUCCESS)goto fail_block; if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) dump_stage("mlp_out",tmp,(size_t)N*D,N,D); void *ag2[]={&hidden,&tmp,&mod,&N,&D,&prefix,&(int){1}};cuLaunchKernel(k->gate_res,(N*D+255)/256,1,1,256,1,1,0,r->stream,ag2,NULL);cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*D,hidden)!=CUDA_SUCCESS) goto fail_block;
        /* The block owns streamed weight allocations.  Synchronize before
         * releasing them; this also makes the custom-kernel/cuBLAS hand-off
         * deterministic on drivers that do not fully order external-stream
         * work behind a cuBLAS call. */
        cuCtxSynchronize();
        if(profile)clock_gettime(CLOCK_MONOTONIC,&prof_computed);
        probe(r,"block",hidden,N*D); if (qimg21_stage_block < 0 || bidx == qimg21_stage_block) { char label[32]; snprintf(label,sizeof(label),"block_%02d",bidx); dump_stage(label,hidden,(size_t)N*D,N,D); } if(qimg21_stage_all_blocks){char label[48];snprintf(label,sizeof(label),"block_%02d_target",bidx);dump_stage(label,hidden+(size_t)prefix*D*4,(size_t)nout*D,nout,D);} free_d(&wq);free_d(&wk);free_d(&wv);free_d(&wo);free_d(&wg);free_d(&wp);free_d(&wmlpo);free_d(&wqn);free_d(&wkn);
        if(profile){
            clock_gettime(CLOCK_MONOTONIC,&prof_released);
            double up=(prof_uploaded.tv_sec-prof_start.tv_sec)+1e-9*(prof_uploaded.tv_nsec-prof_start.tv_nsec);
            double run=(prof_computed.tv_sec-prof_uploaded.tv_sec)+1e-9*(prof_computed.tv_nsec-prof_uploaded.tv_nsec);
            double release=(prof_released.tv_sec-prof_computed.tv_sec)+1e-9*(prof_released.tv_nsec-prof_computed.tv_nsec);
            profile_upload+=up;profile_compute+=run;profile_release+=release;
            fprintf(stderr,"qimg21-profile: block=%02d upload=%.4f compute=%.4f release=%.4f\n",bidx,up,run,release);
        }
        if(qimg21_replay_hidden){result=3;goto fail;} /* Never emit a model prediction from injected state. */
        continue;
fail_block: free_d(&wq);free_d(&wk);free_d(&wv);free_d(&wo);free_d(&wg);free_d(&wp);free_d(&wmlpo);free_d(&wqn);free_d(&wkn);goto fail;
    }
    if(profile)fprintf(stderr,"qimg21-profile: totals upload=%.4f compute=%.4f release=%.4f\n",
                       profile_upload,profile_compute,profile_release);
    qimg21_int8_force_bf16 = 0;
    w_img=upload_bf16(s,"norm_out.linear.weight");w_proj=upload_bf16(s,"proj_out.weight");if(!w_img||!w_proj)goto fail;
    /* Preserve both real and zero timestep rows through final modulation. */
    if(launch_vec(k->silu,r->stream,2*D,temb)!=CUDA_SUCCESS||launch_vec(k->round_bf16,r->stream,2*D,temb)!=CUDA_SUCCESS||launch_cast(r,bf,temb,2*D)!=CUDA_SUCCESS||gemm(r,scale,w_img,bf,2,D,D)!=0||launch_vec(k->round_bf16,r->stream,2*D,scale)!=CUDA_SUCCESS)goto fail;
    dump_stage("final_hidden",hidden,(size_t)N*D,N,D);
    dump_stage("final_scale",scale,2u*D,2,D);
    {void *a[]={&tmp,&hidden,&scale,&N,&D,&prefix};cuLaunchKernel(k->final_ln,N,1,1,k->norm_threads,1,1,256*sizeof(double),r->stream,a,NULL);cuCtxSynchronize(); if (launch_vec(k->round_bf16,r->stream,N*D,tmp)!=CUDA_SUCCESS) goto fail; dump_stage("final_ln",tmp,(size_t)N*D,N,D);}
    {CUdeviceptr dout=checked_cuMemAlloc((size_t)nout*64*4);if(!dout)goto fail;if(launch_cast(r,bf,tmp+(size_t)prefix*D*4,nout*D)!=CUDA_SUCCESS||gemm(r,dout,w_proj,bf,nout,64,D)!=0||launch_vec(k->round_bf16,r->stream,nout*64,dout)!=CUDA_SUCCESS){free_d(&dout);goto fail;}cuCtxSynchronize();cuMemcpyDtoH(out,dout,(size_t)nout*64*4); dump_stage("out",dout,(size_t)nout*64,nout,64);free_d(&dout);}
    result = 0;
    goto done;
fail:
    if(result==3)fprintf(stderr,"native: diagnostic block replay complete; no prediction emitted (status 3)\n");
    else fprintf(stderr,"native: transformer step failed\n");
done:
    free_d(&rope_table);
    free_d(&txt);free_d(&img);free_d(&hidden);free_d(&tmp);free_d(&tmp2);free_d(&bf);free_d(&q);free_d(&kk);free_d(&v);free_d(&att);free_d(&mlp0);free_d(&mlp1);free_d(&mod);free_d(&temb);free_d(&time0);free_d(&timebf);free_d(&scale);free_d(&wt_norm);free_d(&wt_in);free_d(&wt_out);free_d(&wi);free_d(&w_t1);free_d(&w_t2);free_d(&w_mod);free_d(&w_img);free_d(&w_proj);
    #undef A
    return result;
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt_path = NULL, *latent_path = NULL;
    const char *negative_prompt_path = NULL;
    const char *editing_layout_path = NULL, *condition_path = NULL;
    const char *negative_editing_layout_path = NULL;
    const char *cutlass_plugin_path = NULL;
    const char *out_path = "native_latents.npy", *dump_dir = NULL, *pred_dir = NULL;
    int ih = 16, iw = 16, steps = 1, verbose = 1;
    float guidance_scale = 1.0f;
    float manual_t = -1.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--rope") && i+1<argc) {
            const char *mode=argv[++i];
            if(!strcmp(mode,"host-table"))qimg21_host_rope=1;
            else if(!strcmp(mode,"host-table-vector4"))qimg21_host_rope=2;
            else if(!strcmp(mode,"host-table-exact")){qimg21_host_rope=2;qimg21_rope_base_path="cuda/qimg21/qwen21_rope_freqs.npy";}
            else if(!strcmp(mode,"default"))qimg21_host_rope=0;
            else return 2;
        }
        else if (!strcmp(argv[i], "--normalization") && i + 1 < argc) {
            const char *mode=argv[++i];
            if(!strcmp(mode,"vector4"))qimg21_norm_vector=1;
            else if(!strcmp(mode,"default"))qimg21_norm_vector=0;
            else return 2;
        }
        else if (!strcmp(argv[i], "--quantized-transformer") && i + 1 < argc) qimg21_quantized_transformer = argv[++i];
        else if (!strcmp(argv[i], "--quantize-on-load") && i + 1 < argc) {
            if (strcmp(argv[++i], "int8-row")) return 2;
            qimg21_quantize_on_load = 1;
        }
        else if (!strcmp(argv[i], "--int8-tensor-core")) qimg21_int8_tensor_core = 1;
        else if (!strcmp(argv[i], "--int8-bf16-tail-blocks") && i + 1 < argc)
            qimg21_int8_bf16_tail_blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--attention") && i + 1 < argc) {
            const char *mode = argv[++i];
            qimg21_attention_mma64=0;
            if (!strcmp(mode, "reverse64")) qimg21_attention_reverse64 = 1;
            else if (!strcmp(mode, "wmma")) { qimg21_use_wmma = 1; qimg21_attention_reverse64 = 0; }
            else if (!strcmp(mode, "math")) qimg21_attention_reverse64 = 0;
            else if (!strcmp(mode, "mma64")) {qimg21_attention_mma64=1;qimg21_attention_reverse64=0;}
            else if (!strcmp(mode, "mma64-flash")) {qimg21_attention_mma64=2;qimg21_attention_reverse64=0;}
            else if (!strcmp(mode, "mma64-mixed")) {qimg21_attention_mma64=3;qimg21_attention_reverse64=0;}
            else if (!strcmp(mode, "mma64-forward-flash")) {qimg21_attention_mma64=4;qimg21_attention_reverse64=0;}
            else if (!strcmp(mode, "mma128-efficient")) {qimg21_attention_mma64=5;qimg21_attention_reverse64=0;}
            else if (!strcmp(mode, "cutlass-efficient")) {cutlass_plugin_path="cuda/qimg21/libq21_cutlass_attention.so";qimg21_attention_reverse64=0;}
            else { fprintf(stderr, "native: unsupported attention mode\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--prompt-embeds") && i + 1 < argc) prompt_path = argv[++i];
        else if (!strcmp(argv[i], "--cutlass-plugin") && i + 1 < argc) cutlass_plugin_path = argv[++i];
        else if (!strcmp(argv[i], "--rope-table-base") && i + 1 < argc) qimg21_rope_base_path = argv[++i];
        else if (!strcmp(argv[i], "--negative-prompt-embeds") && i + 1 < argc) negative_prompt_path = argv[++i];
        else if (!strcmp(argv[i], "--guidance-scale") && i + 1 < argc) guidance_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--latents") && i + 1 < argc) latent_path = argv[++i];
        else if (!strcmp(argv[i], "--editing-layout") && i + 1 < argc) editing_layout_path = argv[++i];
        else if (!strcmp(argv[i], "--negative-editing-layout") && i + 1 < argc) negative_editing_layout_path = argv[++i];
        else if (!strcmp(argv[i], "--condition-latents") && i + 1 < argc) condition_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--pred-dir") && i + 1 < argc) pred_dir = argv[++i];
        else if (!strcmp(argv[i], "--height-tokens") && i + 1 < argc) ih = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width-tokens") && i + 1 < argc) iw = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timestep") && i + 1 < argc) manual_t = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 2;
        else if (!strcmp(argv[i], "--quiet")) verbose = 0;
        else {
            fprintf(stderr, "usage: %s --model DIR --prompt-embeds E.npy --latents L.npy "
                    "[--negative-prompt-embeds NEG.npy --guidance-scale S] "
                    "[--int8-tensor-core --int8-bf16-tail-blocks N] "
                    "[--editing-layout layout.txt --condition-latents C.npy] "
                    "[--negative-editing-layout negative_layout.txt] "
                    "[--steps N --dump-dir DIR --pred-dir DIR --height-tokens 16 --width-tokens 16 "
                    "--timestep .5 --out O.npy --verbose]\n", argv[0]);
            return 2;
        }
    }
    if (steps < 1 || steps > 100 || (manual_t >= 0.0f && steps != 1)) return 2;
    if (!!editing_layout_path != !!condition_path ||
        (editing_layout_path && (!!negative_prompt_path != !!negative_editing_layout_path)) ||
        (negative_editing_layout_path && !editing_layout_path)) {
        fprintf(stderr,"native: editing requires layout plus condition latents; editing CFG also requires a negative layout and embeds\n");
        return 2;
    }
    if (qimg21_quantize_on_load && qimg21_quantized_transformer) {
        fprintf(stderr,"native: choose a quantized package or quantize-on-load, not both\n"); return 2;
    }
    if (qimg21_int8_tensor_core && !qimg21_quantized_transformer) {
        fprintf(stderr,"native: --int8-tensor-core requires --quantized-transformer\n"); return 2;
    }
    if (qimg21_int8_bf16_tail_blocks < 0 || qimg21_int8_bf16_tail_blocks > 32) {
        fprintf(stderr,"native: --int8-bf16-tail-blocks must be in [0,32]\n"); return 2;
    }
    if (qimg21_quantize_on_load) fprintf(stderr,"native: row-INT8 quantization on load; BF16 compute, no exported copy\n");
    if (qimg21_quantized_transformer) {
        char path[2048], format[64];
        int len = snprintf(path, sizeof(path), "%s/format.txt", qimg21_quantized_transformer);
        if (len < 0 || len >= (int)sizeof(path)) return 2;
        FILE *fp = fopen(path, "r");
        int valid = fp && fgets(format, sizeof(format), fp) && !strcmp(format, "qimg21-int8-row-v1\n");
        if (fp) fclose(fp);
        if (!valid) { fprintf(stderr, "native: incomplete/unsupported quantized package\n"); return 2; }
        fprintf(stderr, qimg21_int8_tensor_core ?
                "native: row-INT8 W8A8 tensor-core compute enabled\n" :
                "native: row-INT8 weights, BF16 dequantized compute (MRE <= 0.10 validated)\n");
    }
    qimg21_stage_dir = getenv("QIMG21_STAGE_DIR");
    qimg21_stage_all_blocks = getenv("QIMG21_STAGE_ALL_BLOCKS") != NULL;
    qimg21_replay_hidden = getenv("QIMG21_REPLAY_HIDDEN");
    qimg21_replay_attention = getenv("QIMG21_REPLAY_ATTENTION");
    if(qimg21_replay_attention && !qimg21_replay_hidden) {
        fprintf(stderr,"native: attention replay requires guarded hidden replay\n");
        return 2;
    }
    if(qimg21_replay_hidden) {
        const char *block=getenv("QIMG21_STAGE_BLOCK");char *end=NULL;
        long number=block?strtol(block,&end,10):-1;
        if(!qimg21_stage_dir || !block || end==block || *end || number<0 || number>=32 ||
           steps!=1 || !isfinite(manual_t) || manual_t<0 || manual_t>1 || negative_prompt_path) {
            fprintf(stderr,"native: hidden replay requires stage directory, block 0..31, one manual-timestep step and no CFG\n");
            return 2;
        }
    }
    if (qimg21_stage_dir) {
        const char *b = getenv("QIMG21_STAGE_BLOCK");
        if (b) qimg21_stage_block = atoi(b);
        mkdir(qimg21_stage_dir, 0755);
    }
    if(!model||!prompt_path||!latent_path||ih<1||iw<1||ih>1024||iw>1024)return 2;
    npy_f32 pe,neg,la; memset(&neg,0,sizeof(neg));
    if(npy_read_f32(prompt_path,&pe)!=0||npy_read_f32(latent_path,&la)!=0)return 1;
    if (negative_prompt_path && npy_read_f32(negative_prompt_path, &neg) != 0) { npy_free(&pe); npy_free(&la); return 1; }
    int nt=(pe.ndim==3&&pe.shape[0]==1)?(int)pe.shape[1]:(pe.ndim==2?(int)pe.shape[0]:0), ni=(la.ndim==3&&la.shape[0]==1)?(int)la.shape[1]:(la.ndim==2?(int)la.shape[0]:0);
    int nnt=(neg.ndim==3&&neg.shape[0]==1)?(int)neg.shape[1]:(neg.ndim==2?(int)neg.shape[0]:0);
    if(nt<=0||pe.shape[pe.ndim-1]!=4096||ni!=ih*iw||la.shape[la.ndim-1]!=64||
       (negative_prompt_path && (nnt<=0 || neg.shape[neg.ndim-1]!=4096))){
        fprintf(stderr,"native: expected embeds [T,4096], optional negative embeds [U,4096], and latents [N,64]\n");
        npy_free(&pe); npy_free(&neg); npy_free(&la); return 1;
    }
    if (negative_prompt_path && guidance_scale <= 1.0f) { fprintf(stderr,"native: guidance-scale must be > 1 with negative embeds\n"); npy_free(&pe); npy_free(&neg); npy_free(&la); return 1; }
    npy_f32 condition={0};
    float *packed=NULL;
    int nc=0;
    if(condition_path) {
        if(npy_read_f32(condition_path,&condition))return 1;
        nc=(condition.ndim==3&&condition.shape[0]==1)?(int)condition.shape[1]:
           (condition.ndim==2?(int)condition.shape[0]:0);
        if(nc<1 || nc>1048576-ni || condition.shape[condition.ndim-1]!=64)return 2;
        for(size_t j=0;j<(size_t)nc*64;j++)if(!isfinite(condition.data[j]))return 2;
        for(size_t j=0;j<(size_t)ni*64;j++)if(!isfinite(la.data[j]))return 2;
        for(size_t j=0;j<(size_t)nt*4096;j++)if(!isfinite(pe.data[j]))return 2;
        for(size_t j=0;j<(size_t)nnt*4096;j++)if(!isfinite(neg.data[j]))return 2;
        packed=malloc((size_t)(nc+ni)*64*sizeof(float));
        if(!packed)return 1;
        memcpy(packed,condition.data,(size_t)nc*64*sizeof(float));
    }
    const float *p=pe.data; cuda_qimg_runner*r=cuda_qimg_init(0,verbose);if(!r)return 1;
    if(qimg21_int8_tensor_core) {
        if(!r->gemm_int8_s32 || !r->quant_act_perrow_int8 || !r->dequant_int32_to_bf16) {
            fprintf(stderr,"native: INT8 tensor-core kernels unavailable\n");cuda_qimg_free(r);return 1;
        }
        r->use_int8=1;
    }
    void *cutlass_plugin=NULL;
    if(cutlass_plugin_path) {
        cutlass_plugin=dlopen(cutlass_plugin_path,RTLD_NOW|RTLD_LOCAL);
        if(!cutlass_plugin || !(qimg21_cutlass_attention=(qimg21_cutlass_attention_fn)
             dlsym(cutlass_plugin,"q21_cutlass_attention"))) {
            fprintf(stderr,"native: cannot load CUTLASS attention plugin %s: %s\n",
                    cutlass_plugin_path,dlerror());
            if(cutlass_plugin)dlclose(cutlass_plugin);
            cuda_qimg_free(r);return 1;
        }
        qimg21_exact_rope=(qimg21_exact_rope_fn)dlsym(cutlass_plugin,"q21_exact_qk_rope");
        qimg21_cutlass_workspace_release=(qimg21_cutlass_workspace_release_fn)
            dlsym(cutlass_plugin,"q21_cutlass_workspace_release");
        qimg21_cutlass_workspace_allocations=(qimg21_cutlass_workspace_allocations_fn)
            dlsym(cutlass_plugin,"q21_cutlass_workspace_allocations");
        fprintf(stderr,"native: exact CUTLASS efficient attention enabled\n");
    }
    q21_edit_context edit={0};
    q21_edit_context negative_edit={0};
    if(editing_layout_path && q21_edit_init(&edit,r,editing_layout_path,nt,nc+ni,ih,iw,qimg21_attention_reverse64)) {
        cuda_qimg_free(r);free(packed);npy_free(&condition);return 1;
    }
    if(negative_editing_layout_path && q21_edit_init(&negative_edit,r,negative_editing_layout_path,
                                                  nnt,nc+ni,ih,iw,qimg21_attention_reverse64)) {
        q21_edit_free(&edit);cuda_qimg_free(r);free(packed);npy_free(&condition);return 1;
    }
    qimg21_shards s={{0},0};char path[1024];for(int i=1;i<=2;i++){snprintf(path,sizeof(path),"%s/transformer/diffusion_pytorch_model-%05d-of-00002.safetensors",model,i);s.st[s.n]=safetensors_open(path);if(!s.st[s.n]){fprintf(stderr,"native: cannot open %s\n",path);cuda_qimg_free(r);return 1;}fprintf(stderr,"native: opened shard %d (%d tensors)\n",i,s.st[s.n]->n_tensors);s.n++;}
    qimg21_kernels k;CUmodule m;if(cu_compile_kernels(&m,r->device,qimg21_src,"qimg21_native.cu",verbose,"qimg21_native")<0||get_kernel(&k,m)!=0){fprintf(stderr,"native: custom kernel compile failed\n");return 1;}fprintf(stderr,"native: custom kernels ready\n");
    CUmodule mma_module=NULL;
    CUmodule norm_module=NULL;
    CUmodule rope_module=NULL;
    if(qimg21_host_rope &&
       (cu_compile_kernels(&rope_module,r->device,q21_rope_table_src,"qimg21_rope_table.cu",verbose,"qimg21_rope_table")<0 ||
        cuModuleGetFunction(&k.table_rope,rope_module,qimg21_host_rope==2?
                            "qk_rope_table_vector4":"qk_rope_table")))return 1;
    if(qimg21_norm_vector) {
        if(cu_compile_kernels(&norm_module,r->device,q21_norm_vector_src,"qimg21_norm_vector.cu",verbose,"qimg21_norm_vector")<0 ||
           cuModuleGetFunction(&k.mod_ln,norm_module,"mod_ln_vector") ||
           cuModuleGetFunction(&k.final_ln,norm_module,"final_ln_vector"))return 1;
        k.norm_threads=128;
    }
    CUmodule text_mma_module=NULL;
    if(qimg21_attention_mma64) {
        size_t length=strlen(q21_mma64_src)+256;
        char *source=malloc(length);
        if(!source)return 1;
        snprintf(source,length,"#define Q21_FLASH_SOFTMAX %d\n#define Q21_FORWARD_KEYS %d\n#define Q21_BKV %d\n#define Q21_SINGLE_BUFFER %d\n#define Q21_PRE_SCALE_SCORES %d\n%s",
                 qimg21_attention_mma64>=2, qimg21_attention_mma64>=4,
                 qimg21_attention_mma64==5?128:64, qimg21_attention_mma64==5,
                 qimg21_attention_mma64==5, q21_mma64_src);
        int compiled=cu_compile_kernels(&mma_module,r->device,source,"qimg21_mma64.cu",verbose,"qimg21_mma64");
        free(source);
        if(compiled<0 || cuModuleGetFunction(&k.mma_attention,mma_module,"q21_flash_reverse64") ||
           cuFuncSetAttribute(k.mma_attention,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,4*64*136*2))return 1;
    }
    if(qimg21_attention_mma64==3) {
        size_t length=strlen(q21_mma64_src)+128;
        char *source=malloc(length);
        if(!source)return 1;
        snprintf(source,length,"#define Q21_FORWARD_KEYS 1\n#define Q21_FLASH_SOFTMAX 0\n%s",q21_mma64_src);
        int compiled=cu_compile_kernels(&text_mma_module,r->device,source,"qimg21_mma_text.cu",verbose,"qimg21_mma_text");
        free(source);
        if(compiled<0 || cuModuleGetFunction(&k.mma_text,text_mma_module,"q21_flash_reverse64") ||
           cuFuncSetAttribute(k.mma_text,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,4*64*136*2))return 1;
    }
    if (dump_dir) mkdir(dump_dir, 0755);
    if (pred_dir) mkdir(pred_dir, 0755);
    float *pred = (float *)malloc((size_t)ni * 64 * sizeof(float));
    float *neg_pred = negative_prompt_path ? (float *)malloc((size_t)ni * 64 * sizeof(float)) : NULL;
    float *sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    if (!pred || (negative_prompt_path && !neg_pred) || !sigmas) { free(pred); free(neg_pred); free(sigmas); npy_free(&pe); npy_free(&neg); npy_free(&la); return 1; }
    if (manual_t >= 0.0f) {
        for (int i = 0; i <= steps; i++) sigmas[i] = (i == 0) ? manual_t : 0.0f;
    } else qimg21_flow_sigmas(steps, ni, sigmas);
    int rc = 0;
    for (int i = 0; i < steps; i++) {
        fprintf(stderr, "native: step %d/%d sigma=%.7f\n", i + 1, steps, sigmas[i]);
        /* The pipeline casts scheduler timestep (sigma*1000) to BF16,
         * then divides by 1000 in BF16 before calling the transformer. */
        float model_t=manual_t>=0.0f ? manual_t :
            qimg21_round_bf16_host(qimg21_round_bf16_host(sigmas[i]*1000.0f)/1000.0f);
        if(packed)memcpy(packed+(size_t)nc*64,la.data,(size_t)ni*64*sizeof(float));
        rc = native_step(r, &k, &s, p, nt, packed?packed:la.data, nc+ni, ih, iw, model_t, pred,
                         editing_layout_path?&edit:NULL);
        if (rc == 0 && negative_prompt_path) {
            rc = native_step(r, &k, &s, neg.data, nnt, packed?packed:la.data, nc+ni, ih, iw, model_t,
                             neg_pred, negative_editing_layout_path?&negative_edit:NULL);
            if (rc == 0) for (size_t j = 0; j < (size_t)ni * 64; j++) {
                float difference=qimg21_round_bf16_host(pred[j]-neg_pred[j]);
                float guided=qimg21_round_bf16_host(guidance_scale*difference);
                pred[j]=qimg21_round_bf16_host(neg_pred[j]+guided);
            }
        }
        if (qimg21_stage_error) { fprintf(stderr, "native: stage capture failed\n"); rc = 1; }
        if (rc != 0) break;
        if (pred_dir) {
            char pred_path[1024];
            snprintf(pred_path, sizeof(pred_path), "%s/pred_%03d.npy", pred_dir, i);
            if (npy_write_f32(pred_path, pred, (size_t)ni * 64, ni, 64)) { rc = 1; break; }
        }
        qimg21_euler_bf16(la.data,pred,(size_t)ni*64,sigmas[i],sigmas[i+1]);
        if (dump_dir) {
            char step_path[1024];
            snprintf(step_path, sizeof(step_path), "%s/step_%03d.npy", dump_dir, i);
            if (npy_write_f32(step_path, la.data, (size_t)ni * 64, ni, 64)) { rc = 1; break; }
        }
    }
    if (rc == 0) {
        if (npy_write_f32(out_path, la.data, (size_t)ni * 64, ni, 64)) rc = 1;
        else fprintf(stderr, "native: wrote %s (%d tokens x 64, %d steps)\n", out_path, ni, steps);
    }
    free(pred); free(neg_pred); free(sigmas); cuModuleUnload(m); for(int i=0;i<s.n;i++)safetensors_close(s.st[i]);
    q21_edit_free(&edit);q21_edit_free(&negative_edit);free(packed);npy_free(&condition);
    if(mma_module)cuModuleUnload(mma_module);
    if(text_mma_module)cuModuleUnload(text_mma_module);
    if(norm_module)cuModuleUnload(norm_module);
    if(rope_module)cuModuleUnload(rope_module);
    if(cutlass_plugin) {
        if(qimg21_cutlass_workspace_allocations)
            fprintf(stderr,"native: CUTLASS workspace allocations=%u\n",
                    qimg21_cutlass_workspace_allocations());
        if(qimg21_cutlass_workspace_release)qimg21_cutlass_workspace_release();
        dlclose(cutlass_plugin);
    }
    if(qimg21_int8_input_f32) {
        cuMemFree(qimg21_int8_input_f32);
        qimg21_int8_input_f32=0;qimg21_int8_input_f32_bytes=0;
    }
    if(qimg21_int8_tensor_core)
        fprintf(stderr,"native: custom INT8 MMA GEMM calls=%llu\n",qimg21_int8_mma_calls);
    cuda_qimg_free(r); npy_free(&pe); npy_free(&neg); npy_free(&la); return rc;
}
