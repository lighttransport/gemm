/* Repeated, verified FP8 comparison using the existing PTX and cuBLAS paths. */
#define _POSIX_C_SOURCE 200809L
#include "../int4/bench_common.h"
#include "../cublasew.h"
#include "fp8_types.h"
#include "fp8_gemm_kernels.h"
#include "../gemm/cuda_gemm_ptx_kernels.h"
#include <errno.h>
#include <limits.h>
#include <dlfcn.h>

typedef struct { const char *name; int m, n, k; } shape;
static const shape shapes[] = {
    {"square_1k",1024,1024,1024}, {"square_2k",2048,2048,2048},
    {"square_4k",4096,4096,4096}, {"square_8k",8192,8192,8192},
    {"mm0",1024,4608,4608}, {"mm2",1024,1152,4608},
    {"qkv",512,4096,4096}, {"ffn_up",512,11008,4096}, {"ffn_down",512,4096,11008},
    {"small_m1",1,4096,4096}, {"small_m16",16,4096,4096}, {"small_m64",64,4096,4096}
};
typedef struct {
    const char *shape_name, *kernel, *csv;
    int m, n, k, warmup, iters, rounds, full_check;
} options;
typedef struct {
    const char *name;
    int format, kind, tile; /* kind: original PTX / NVRTC v5,v7 / cuBLAS FP8 / BF16 */
    CUmodule module;
    CUfunction fn;
} kernel;
static kernel kernels[] = {
    {"tc32_e4m3",0,0,32,0,0}, {"tc64_e4m3",0,0,64,0,0}, {"tc128_e4m3",0,0,128,0,0},
    {"tc32_e5m2",1,0,32,0,0}, {"tc64_e5m2",1,0,64,0,0}, {"tc128_e5m2",1,0,128,0,0},
    {"ptx_v5_e4m3",0,1,5,0,0}, {"ptx_v7_e4m3",0,1,7,0,0},
    {"cublas_e4m3",0,2,0,0,0}, {"cublas_bf16",0,3,0,0,0}
};
enum { NUM_KERNELS = sizeof(kernels) / sizeof(kernels[0]) };

static int selected(const options *o, const kernel *k) {
    return !strcmp(o->kernel,"all") || !strcmp(o->kernel,k->name) ||
        (!strcmp(o->kernel,"legacy") && k->kind == 0) ||
        (!strcmp(o->kernel,"ptx") && k->kind == 1) ||
        (!strcmp(o->kernel,"cublas") && k->kind == 2) ||
        (!strcmp(o->kernel,"bf16") && k->kind == 3);
}

static void build_kernels(const options *o) {
    const char *ptx[] = {ptx_gemm_tc_e4m3,ptx_gemm_tc64_e4m3,ptx_gemm_tc128_e4m3,
        ptx_gemm_tc_e5m2,ptx_gemm_tc64_e5m2,ptx_gemm_tc128_e5m2};
    const char *names[] = {"gemm_fp8_tc_e4m3","gemm_fp8_tc64_e4m3","gemm_fp8_tc128_e4m3",
        "gemm_fp8_tc_e5m2","gemm_fp8_tc64_e5m2","gemm_fp8_tc128_e5m2"};
    for (int i = 0; i < NUM_KERNELS; i++) {
        kernel *k = &kernels[i];
        if (!selected(o,k) || k->kind >= 2) continue;
        if (!k->kind) {
            CUDA(cuModuleLoadData(&k->module,ptx[i]));
            CUDA(cuModuleGetFunction(&k->fn,k->module,names[i]));
        } else {
            REQUIRE(cuewInit(CUEW_INIT_NVRTC) == CUEW_SUCCESS);
            const char *src = k->tile == 5 ? k_gemm_fp8_v5_src : k_gemm_fp8_v7_src;
            const char *name = k->tile == 5 ? "gemm_fp8_v5" : "gemm_fp8_v7";
            const char *flags[] = {"--gpu-architecture=sm_120a","--std=c++17"};
            nvrtcProgram p;
            REQUIRE(nvrtcCreateProgram(&p,src,name,0,NULL,NULL) == NVRTC_SUCCESS);
            nvrtcResult status = nvrtcCompileProgram(p,2,flags);
            if (status != NVRTC_SUCCESS) {
                size_t n; nvrtcGetProgramLogSize(p,&n);
                char *log = host_alloc(n + 1); nvrtcGetProgramLog(p,log); log[n] = 0;
                fprintf(stderr,"%s: %s\n",name,log); free(log);
            }
            REQUIRE(status == NVRTC_SUCCESS);
            size_t n; REQUIRE(nvrtcGetCUBINSize(p,&n) == NVRTC_SUCCESS);
            char *blob = host_alloc(n); REQUIRE(nvrtcGetCUBIN(p,blob) == NVRTC_SUCCESS);
            CUDA(cuModuleLoadData(&k->module,blob)); free(blob);
            REQUIRE(nvrtcDestroyProgram(&p) == NVRTC_SUCCESS);
            CUDA(cuModuleGetFunction(&k->fn,k->module,name));
        }
    }
    /* Keep the scalar fallback strings shared with the original runner too. */
    (void)ptx_gemm_naive_e4m3; (void)ptx_gemm_naive_e5m2;
}

typedef struct {
    shape s;
    CUdeviceptr a[2], b[2], bt[2], ah, bh, output, one;
    cublasew_context *blas;
} buffers;

static int run_kernel(const kernel *k, buffers *b) {
    int m = b->s.m, n = b->s.n, depth = b->s.k;
    int f = k->format;
    if (k->kind == 2)
        return !b->blas ? -1 : cublasew_gemm_fp8_e4m3_f32out_rowmajor_nt(b->blas,
            b->output,b->b[0],b->a[0],b->one,b->one,m,n,depth);
    if (k->kind == 3)
        return !b->blas ? -1 : cublasew_gemm_bf16_bf16_f32_rowmajor_nt(b->blas,
            b->output,b->bh,b->ah,m,n,depth);
    if (!k->kind) {
        void *args[] = {&b->a[f],&b->bt[f],&b->output,&m,&n,&depth};
        CUDA(cuLaunchKernel(k->fn,n / k->tile,m / k->tile,1,
            k->tile == 32 ? 128 : 512,1,1,0,NULL,args,NULL));
    } else {
        int gx = (n + 127) / 128, gy = (m + 63) / 64;
        if (k->tile == 7) { gx = (gx + 3) & ~3; gy = (gy + 3) & ~3; }
        void *args[] = {&b->output,&b->a[0],&b->b[0],&m,&n,&depth};
        CUDA(cuLaunchKernel(k->fn,gx,gy,1,256,1,1,24576,NULL,args,NULL));
    }
    return 0;
}

static float time_kernel(const kernel *k, buffers *b, int iters) {
    CUevent begin, end; CUDA(cuEventCreate(&begin,0)); CUDA(cuEventCreate(&end,0));
    CUDA(cuEventRecord(begin,NULL));
    for (int i = 0; i < iters; i++) REQUIRE(!run_kernel(k,b));
    CUDA(cuEventRecord(end,NULL)); CUDA(cuEventSynchronize(end));
    float ms; CUDA(cuEventElapsedTime(&ms,begin,end));
    CUDA(cuEventDestroy(begin)); CUDA(cuEventDestroy(end)); return ms / iters;
}

static double verify(buffers *b, const size_t *indices, const double *ref,
                     int samples, double *max_error) {
    double num = 0, den = 0; *max_error = 0;
    for (int i = 0; i < samples; i++) {
        float got; CUDA(cuMemcpyDtoH(&got,b->output + indices[i] * 4,4));
        REQUIRE(isfinite(got));
        double e = got - ref[i]; num += e * e; den += ref[i] * ref[i];
        *max_error = fmax(*max_error,fabs(e));
    }
    double rel = sqrt(num / fmax(den,1e-30));
    if (rel > 1e-4) fprintf(stderr,"verification rel_L2=%g max_error=%g\n",rel,*max_error);
    REQUIRE(rel <= 1e-4); return rel;
}

static int compare_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y);
}

static void benchmark(const shape *s, const options *o, FILE *csv) {
    size_t na = (size_t)s->m * s->k, nb = (size_t)s->n * s->k, nc = (size_t)s->m * s->n;
    float *src_a = host_alloc(na * 4), *src_b = host_alloc(nb * 4);
    uint32_t rng = 42;
    for (size_t i = 0; i < na; i++) src_a[i] = ((random_u32(&rng) & 0xffff) / 65535.0f - 0.5f) * 4;
    for (size_t i = 0; i < nb; i++) src_b[i] = ((random_u32(&rng) & 0xffff) / 65535.0f - 0.5f) * 4;
    buffers b = {0}; b.s = *s;
    b.output = device_alloc(nc * 4); b.one = device_alloc(4);
    float one = 1; CUDA(cuMemcpyHtoD(b.one,&one,4));
    /* A fresh cache per shape keeps the scale pointers alive through all launches. */
    if (cublasewInit() || cublasewCreate(&b.blas,NULL)) b.blas = NULL;
    int samples = o->full_check ? (int)nc : (nc < 64 ? (int)nc : 64);
    REQUIRE(!o->full_check || nc <= 65536);
    size_t *indices = host_alloc((size_t)samples * sizeof(size_t));
    double *refs[2], quant[2] = {0,0};
    for (int i = 0; i < samples; i++) indices[i] = samples == 1 ? 0 : (size_t)i * (nc - 1) / (samples - 1);
    for (int f = 0; f < 2; f++) {
        uint8_t *a = host_alloc(na), *w = host_alloc(nb), *wt = host_alloc(nb);
        float lut[256];
        for (int i = 0; i < 256; i++) lut[i] = f ? fp8_e5m2_to_float(i) : fp8_e4m3_to_float(i);
        for (size_t i = 0; i < na; i++) a[i] = f ? float_to_fp8_e5m2(src_a[i]) : float_to_fp8_e4m3(src_a[i]);
        for (int n = 0; n < s->n; n++) for (int k = 0; k < s->k; k++) {
            size_t i = (size_t)n * s->k + k;
            w[i] = f ? float_to_fp8_e5m2(src_b[i]) : float_to_fp8_e4m3(src_b[i]);
            wt[(size_t)k * s->n + n] = w[i];
        }
        b.a[f] = device_alloc(na); b.b[f] = device_alloc(nb); b.bt[f] = device_alloc(nb);
        CUDA(cuMemcpyHtoD(b.a[f],a,na)); CUDA(cuMemcpyHtoD(b.b[f],w,nb)); CUDA(cuMemcpyHtoD(b.bt[f],wt,nb));
        refs[f] = host_alloc((size_t)samples * sizeof(double));
        double qnum = 0, qden = 0;
        for (int i = 0; i < samples; i++) {
            size_t ar = indices[i] / s->n * s->k, br = indices[i] % s->n * s->k;
            double ref = 0, original = 0;
            for (int k = 0; k < s->k; k++) {
                ref += (double)lut[a[ar + k]] * lut[w[br + k]];
                original += (double)src_a[ar + k] * src_b[br + k];
            }
            refs[f][i] = ref; qnum += (ref - original) * (ref - original); qden += original * original;
        }
        quant[f] = sqrt(qnum / fmax(qden,1e-30));
        if (!f) {
            uint16_t *half = host_alloc((na > nb ? na : nb) * 2);
            for (size_t i = 0; i < na; i++) { float v = lut[a[i]]; uint32_t u; memcpy(&u,&v,4); half[i] = u >> 16; }
            b.ah = device_alloc(na * 2); CUDA(cuMemcpyHtoD(b.ah,half,na * 2));
            for (size_t i = 0; i < nb; i++) { float v = lut[w[i]]; uint32_t u; memcpy(&u,&v,4); half[i] = u >> 16; }
            b.bh = device_alloc(nb * 2); CUDA(cuMemcpyHtoD(b.bh,half,nb * 2)); free(half);
        }
        free(a); free(w); free(wt);
    }
    free(src_a); free(src_b);
    int active[NUM_KERNELS] = {0};
    float times[NUM_KERNELS][101]; double rel[NUM_KERNELS], max_error[NUM_KERNELS];
    for (int i = 0; i < NUM_KERNELS; i++) {
        kernel *k = &kernels[i];
        if (!selected(o,k)) continue;
        if (!k->kind && (s->m % k->tile || s->n % k->tile)) {
            printf("%-12s %-17s SKIP (dimension alignment)\n",s->name,k->name); continue;
        }
        CUDA(cuMemsetD32(b.output,0x7fc00000,nc));
        if (run_kernel(k,&b)) { printf("%-12s %-17s SKIP (library unsupported)\n",s->name,k->name); continue; }
        CUDA(cuCtxSynchronize());
        rel[i] = verify(&b,indices,refs[k->format],samples,&max_error[i]); active[i] = 1;
        for (int j = 0; j < o->warmup; j++) REQUIRE(!run_kernel(k,&b));
        CUDA(cuCtxSynchronize());
    }
    for (int round = 0; round < o->rounds; round++) for (int j = 0; j < NUM_KERNELS; j++) {
        int i = (round + j) % NUM_KERNELS;
        if (active[i]) times[i][round] = time_kernel(&kernels[i],&b,o->iters);
    }
    for (int i = 0; i < NUM_KERNELS; i++) if (active[i]) {
        kernel *k = &kernels[i];
        REQUIRE(!run_kernel(k,&b)); CUDA(cuCtxSynchronize());
        rel[i] = verify(&b,indices,refs[k->format],samples,&max_error[i]);
        qsort(times[i],o->rounds,sizeof(float),compare_float);
        float median = times[i][o->rounds / 2];
        double tops = 2.0 * s->m * s->n * s->k / (median * 1e9);
        printf("%-12s %-17s %9.4f ms %7.2f TOp/s range=[%.4f,%.4f] PASS rel_L2=%.2g quant_L2=%.4g\n",
            s->name,k->name,median,tops,times[i][0],times[i][o->rounds-1],rel[i],quant[k->format]);
        if (csv) {
            fprintf(csv,"%s,%d,%d,%d,%s,packed,%d,%d,%d,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%d,PASS\n",
                s->name,s->m,s->n,s->k,k->name,o->warmup,o->iters,o->rounds,median,
                times[i][0],times[i][o->rounds-1],tops,rel[i],max_error[i],quant[k->format],samples);
            fflush(csv);
        }
    }
    if (b.blas) cublasewDestroy(b.blas);
    for (int f = 0; f < 2; f++) {
        free(refs[f]); CUDA(cuMemFree(b.a[f])); CUDA(cuMemFree(b.b[f])); CUDA(cuMemFree(b.bt[f]));
    }
    free(indices); CUDA(cuMemFree(b.ah)); CUDA(cuMemFree(b.bh)); CUDA(cuMemFree(b.output)); CUDA(cuMemFree(b.one));
}

static int positive_int(const char *s) {
    char *end; errno = 0; long v = strtol(s,&end,10);
    REQUIRE(!errno && *s && !*end && v > 0 && v <= INT_MAX); return (int)v;
}

int main(int argc, char **argv) {
    setvbuf(stdout,NULL,_IOLBF,0);
    options o = {"square_1k","all",NULL,0,0,0,5,100,5,0};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--help")) {
            puts("bench_fp8 [--shape NAME|all | --m M --n N --k K] [--kernel all|legacy|ptx|cublas|bf16|NAME]\n"
                 "          [--warmup N --iters N --rounds N] [--csv PATH] [--full-check]\n"
                 "K must be a multiple of 64; original tiles require aligned M,N.\n"
                 "Packed GEMM only: CPU quantization, uploads, and compilation are outside timing."); return 0;
        }
        if (!strcmp(argv[i],"--full-check")) { o.full_check = 1; continue; }
        REQUIRE(i + 1 < argc); const char *key = argv[i++], *v = argv[i];
        if (!strcmp(key,"--shape")) o.shape_name = v;
        else if (!strcmp(key,"--kernel")) o.kernel = v;
        else if (!strcmp(key,"--csv")) o.csv = v;
        else if (!strcmp(key,"--m")) o.m = positive_int(v);
        else if (!strcmp(key,"--n")) o.n = positive_int(v);
        else if (!strcmp(key,"--k")) o.k = positive_int(v);
        else if (!strcmp(key,"--warmup")) o.warmup = positive_int(v);
        else if (!strcmp(key,"--iters")) o.iters = positive_int(v);
        else if (!strcmp(key,"--rounds")) o.rounds = positive_int(v);
        else { fprintf(stderr,"unknown option %s\n",key); return 1; }
    }
    REQUIRE(o.rounds <= 101 && (o.rounds & 1));
    REQUIRE((!o.m && !o.n && !o.k) || (o.m && o.n && o.k));
    REQUIRE(o.m <= 16384 && o.n <= 16384 && o.k <= 16384 && o.k % 64 == 0);
    REQUIRE(!o.full_check || (o.m && (size_t)o.m * o.n <= 65536));
    int count = 0; for (int i = 0; i < NUM_KERNELS; i++) count += selected(&o,&kernels[i]); REQUIRE(count);
    int found = o.m || !strcmp(o.shape_name,"all");
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) found |= !strcmp(o.shape_name,shapes[i].name);
    REQUIRE(found);
    CUcontext context = start_cuda(); CUdevice dev; CUDA(cuCtxGetDevice(&dev));
    char name[256]; CUDA(cuDeviceGetName(name,sizeof(name),dev)); int driver; CUDA(cuDriverGetVersion(&driver));
    printf("%s; driver API=%d; FP32 accumulation/output; unit per-tensor FP8 scales\n",name,driver);
    FILE *csv = NULL;
    if (o.csv) {
        csv = fopen(o.csv,"w"); REQUIRE(csv);
        fputs("shape,m,n,k,kernel,mode,warmup,iters,rounds,median_ms,min_ms,max_ms,tops,rel_l2,max_error,quant_l2,samples,status\n",csv);
        char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s.metadata.txt",o.csv) < (int)sizeof(path));
        FILE *meta = fopen(path,"w"); REQUIRE(meta);
        for (int i = 0; i < argc; i++) fprintf(meta,"%s%s",i ? " " : "",argv[i]);
        fprintf(meta,"\nGPU=%s\ndriver_api=%d\n",name,driver);
        if (cuewInit(CUEW_INIT_NVRTC) == CUEW_SUCCESS) { int a,b; nvrtcVersion(&a,&b); fprintf(meta,"NVRTC=%d.%d\n",a,b); }
        void *lib = dlopen("libcublas.so.13",RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("libcublas.so.12",RTLD_NOW | RTLD_LOCAL);
        if (lib) {
            int (*prop)(int,int *); void *sym = dlsym(lib,"cublasGetProperty"); memcpy(&prop,&sym,sizeof(sym));
            if (prop) { int a,b,c; if (!prop(0,&a) && !prop(1,&b) && !prop(2,&c)) fprintf(meta,"cuBLAS=%d.%d.%d\n",a,b,c); }
            dlclose(lib);
        }
        FILE *info = popen("nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,clocks.current.sm,clocks.max.sm --format=csv,noheader","r");
        if (info) { char line[512]; while (fgets(line,sizeof(line),info)) fputs(line,meta); pclose(info); }
        fclose(meta);
    }
    build_kernels(&o);
    if (o.m) { shape s = {"custom",o.m,o.n,o.k}; benchmark(&s,&o,csv); }
    else for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++)
        if (!strcmp(o.shape_name,"all") || !strcmp(o.shape_name,shapes[i].name)) benchmark(&shapes[i],&o,csv);
    for (int i = 0; i < NUM_KERNELS; i++) if (kernels[i].module) CUDA(cuModuleUnload(kernels[i].module));
    if (csv) REQUIRE(!fclose(csv));
    CUDA(cuCtxDestroy(context)); return 0;
}
