#define _POSIX_C_SOURCE 200809L
#include "bench_common.h"
#include "fp4_legacy.h"
#include "../cublasew.h"
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>

typedef struct { const char *name; int m, n, k; } shape;
static const shape shapes[] = {
    {"square_1k",1024,1024,1024}, {"square_2k",2048,2048,2048},
    {"square_4k",4096,4096,4096}, {"square_8k",8192,8192,8192},
    {"mm0",1024,4608,4608}, {"mm2",1024,1152,4608},
    {"qkv",512,4096,4096}, {"ffn_up",512,11008,4096}, {"ffn_down",512,4096,11008},
    {"small_m1",1,4096,4096}, {"small_m16",16,4096,4096}, {"small_m64",64,4096,4096}
};
typedef struct {
    const char *mode, *backend, *shape_name, *csv;
    int m, n, k, warmup, iters, rounds, variant, unit, probe_blocks, probe_repeats, legacy_fp4;
    char dir[4096];
} options;

/* Local cuBLAS INT8 baseline, using the official GemmEx ABI. No shared loader changes. */
typedef struct {
    void *library, *handle;
    int (*create)(void **), (*destroy)(void *), (*version)(void *, int *);
    int (*gemm)(void *, int, int, int, int, int, const void *, const void *, int, int,
                const void *, int, int, const void *, void *, int, int, int, int);
} int8_blas;

static int int8_blas_init(int8_blas *b) {
    memset(b, 0, sizeof(*b));
    b->library = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    if (!b->library) b->library = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!b->library) return -1;
#define SYMBOL(field, symbol) do { void *p = dlsym(b->library, symbol); if (!p) goto fail; memcpy(&b->field, &p, sizeof(p)); } while (0)
    SYMBOL(create, "cublasCreate_v2"); SYMBOL(destroy, "cublasDestroy_v2");
    SYMBOL(version, "cublasGetVersion_v2"); SYMBOL(gemm, "cublasGemmEx");
#undef SYMBOL
    if (b->create(&b->handle)) goto fail;
    return 0;
fail:
    dlclose(b->library); memset(b, 0, sizeof(*b)); return -1;
}

typedef struct {
    int4_context *ctx;
    cublasew_context *bf16;
    int8_blas *i8;
    fp4_legacy_context *legacy;
    int4_matrix a, b;
    CUdeviceptr src_a, src_b, output, other_a, other_b;
    int kind, format, variant, mode; /* kind: custom/BF16/INT8/legacy FP4; mode: packed/activation/full */
    const char *name;
    double golden[64], original[64];
    size_t indices[64];
    int samples;
    double rel_l2, max_error, quant_l2;
    int4_quant_stats stats;
    float times[101];
} job;

static int run_job(job *j) {
    if (j->kind == 3) {
        CUDA(fp4_legacy_gemm(j->legacy, j->a, j->b, j->output, j->variant, NULL));
        return 0;
    }
    if (j->kind == 1)
        return cublasew_gemm_bf16_bf16_f32_rowmajor_nt(j->bf16, j->output, j->other_b, j->other_a,
                                                       j->a.rows, j->b.rows, j->a.kp);
    if (j->kind == 2) {
        int alpha = 1, beta = 0;
        return j->i8->gemm(j->i8->handle, 1, 0, j->b.rows, j->a.rows, j->a.kp,
            &alpha, (void *)(uintptr_t)j->other_b, 3, j->a.kp,
            (void *)(uintptr_t)j->other_a, 3, j->a.kp, &beta, (void *)(uintptr_t)j->output,
            10, j->b.rows, 72, -1); /* CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I */
    }
    if (j->mode) CUDA(int4_quantize_f32(j->ctx, j->src_a, j->a, j->format, 0, NULL));
    if (j->mode == 2) CUDA(int4_quantize_f32(j->ctx, j->src_b, j->b, j->format, 0, NULL));
    CUDA(int4_gemm(j->ctx, j->a, j->b, j->output, j->format ? 3 : 0, j->variant, NULL));
    return 0;
}

static float time_job(job *j, int iters) {
    CUevent a, b; CUDA(cuEventCreate(&a, 0)); CUDA(cuEventCreate(&b, 0));
    CUDA(cuEventRecord(a, NULL));
    for (int i = 0; i < iters; i++) REQUIRE(!run_job(j));
    CUDA(cuEventRecord(b, NULL)); CUDA(cuEventSynchronize(b));
    float ms; CUDA(cuEventElapsedTime(&ms, a, b));
    CUDA(cuEventDestroy(a)); CUDA(cuEventDestroy(b)); return ms / iters;
}

static void reference_samples(job *j, const uint8_t *ac, const uint8_t *as,
    const uint8_t *bc, const uint8_t *bs, const float *src_a, const float *src_b) {
    size_t count = (size_t)j->a.rows * j->b.rows;
    j->samples = count < 64 ? (int)count : 64;
    for (int s = 0; s < j->samples; s++) {
        size_t index = j->samples == 1 ? 0 : (size_t)s * (count - 1) / (j->samples - 1);
        int m = (int)(index / j->b.rows), n = (int)(index % j->b.rows);
        double ref = 0, original = 0;
        for (int k = 0; k < j->a.k; k++) {
            ref += (double)int4_decode_cpu(ac, as, m, k, j->a.kp, j->format) *
                int4_decode_cpu(bc, bs, n, k, j->b.kp, j->format);
            original += (double)src_a[(size_t)m * j->a.k + k] * src_b[(size_t)n * j->b.k + k];
        }
        j->indices[s] = index; j->golden[s] = ref; j->original[s] = original;
    }
}

static void verify_job(job *j) {
    double num = 0, den = 0, qnum = 0, qden = 0, max_error = 0;
    for (int s = 0; s < j->samples; s++) {
        double got;
        if (j->kind == 2) { int v; CUDA(cuMemcpyDtoH(&v, j->output + j->indices[s] * 4, 4)); got = v; }
        else { float v; CUDA(cuMemcpyDtoH(&v, j->output + j->indices[s] * 4, 4)); got = v; }
        REQUIRE(isfinite(got));
        double e = got - j->golden[s], qe = j->golden[s] - j->original[s];
        num += e * e; den += j->golden[s] * j->golden[s]; max_error = fmax(max_error, fabs(e));
        qnum += qe * qe; qden += j->original[s] * j->original[s];
    }
    j->rel_l2 = den ? sqrt(num / den) : sqrt(num);
    j->quant_l2 = qden ? sqrt(qnum / qden) : sqrt(qnum);
    j->max_error = max_error;
    if (j->rel_l2 > 1e-4) fprintf(stderr, "%s variant=%d rel_L2=%g max_error=%g\n", j->name, j->variant, j->rel_l2, max_error);
    REQUIRE(j->rel_l2 <= 1e-4);
}

static int compare_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y);
}

static void report(FILE *csv, const shape *s, const char *backend, const char *mode,
    const char *variant, const options *o, float *times, double rel, double max_error,
    double quant, int4_quant_stats stats) {
    qsort(times, o->rounds, sizeof(float), compare_float);
    float median = times[o->rounds / 2], min = times[0], max = times[o->rounds - 1];
    double tops = 2.0 * s->m * s->n * s->k / (median * 1e9);
    printf("%-12s %-6s %-10s %-14s %9.4f ms %8.2f TOp/s range=[%.4f,%.4f] verify=PASS rel_L2=%.2g quant_L2=%.4g\n",
        s->name, backend, mode, variant, median, tops, min, max, rel, quant);
    if (csv) {
        fprintf(csv, "%s,%d,%d,%d,%s,%s,%s,%s,%d,%d,%d,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g,%llu,%llu,PASS\n",
            s->name,s->m,s->n,s->k,backend,mode,variant,o->unit ? "unit" : "k16",o->warmup,o->iters,o->rounds,
            median,min,max,tops,rel,max_error,quant,
            (unsigned long long)stats.clipped_values,(unsigned long long)stats.clamped_scales);
        fflush(csv);
    }
}

static void run_probes(int4_context *ctx, options *o, FILE *csv) {
    CUfunction funcs[] = {ctx->probe[0],ctx->probe[1],ctx->probe[2],ctx->probe[3],ctx->probe_bf16,ctx->probe_int8,ctx->probe_s4};
    const char *names[] = {"e2_e2","e0_e2","e2_e0","e0_e0","bf16","int8","s4"};
    int depths[] = {64,64,64,64,16,32,64};
    float times[7][101];
    size_t count = (size_t)o->probe_blocks * 32 * 32;
    CUdeviceptr output = device_alloc(count * 4);
    uint32_t *sample = host_alloc(count * 4);
    unsigned operand = 0;
    void *args[] = {&output, &o->probe_repeats, &operand};
    for (int f = 0; f < 7; f++) {
        operand = f == 4 ? 0x3f803f80 : f == 5 ? 0x01010101 : 0x11111111;
        for (int w = 0; w < o->warmup; w++)
            CUDA(cuLaunchKernel(funcs[f],o->probe_blocks,1,1,32,1,1,0,NULL,args,NULL));
    }
    for (int r = 0; r < o->rounds; r++) for (int k = 0; k < 7; k++) {
        int f = (r + k) % 7;
        operand = f == 4 ? 0x3f803f80 : f == 5 ? 0x01010101 : 0x11111111;
        times[f][r] = launch_time(funcs[f], o->probe_blocks, 32, args, o->iters);
        /* All independent accumulator outputs must survive compilation and agree. */
        CUDA(cuMemcpyDtoH(sample, output, count * 4));
        float expected = (float)(o->probe_repeats * (f == 0 || f == 4 ? 16 : f == 1 || f == 2 || f == 5 ? 32 : 64));
        for (size_t i = 0; i < count; i++) {
            if (f >= 5) REQUIRE((int32_t)sample[i] == (int32_t)expected);
            else { float value; memcpy(&value, &sample[i], 4); REQUIRE(value == expected); }
        }
    }
    options probe_options = *o;
    probe_options.unit = 1;
    for (int f = 0; f < 7; f++) {
        shape s = {"instruction",o->probe_blocks * 16,8,depths[f] * o->probe_repeats * 8};
        report(csv,&s,names[f],"probe","8_accumulators",&probe_options,times[f],0,0,0,(int4_quant_stats){0,0});
    }
    CUDA(cuMemFree(output)); free(sample);
}

static void matrix_benchmark(int4_context *ctx, cublasew_context *bf16, int8_blas *i8, fp4_legacy_context *legacy,
                              const shape *s, const options *o, FILE *csv) {
    int M = s->m, N = s->n, K = s->k, kp = int4_padded_k(K);
    size_t na = (size_t)M * K, nb = (size_t)N * K;
    float *src_a = host_alloc(na * 4), *src_b = host_alloc(nb * 4);
    uint32_t seed = 42;
    for (size_t i = 0; i < na + nb; i++) {
        uint32_t r = random_u32(&seed);
        float x = o->unit ? (float)((int)(r % 15) - 7) : ((int)(r % 65537) - 32768) / 16384.0f;
        if (i < na) src_a[i] = x; else src_b[i - na] = x;
    }
    CUdeviceptr da = device_alloc(na * 4), db = device_alloc(nb * 4), output = device_alloc((size_t)M * N * 4);
    CUDA(cuMemcpyHtoD(da, src_a, na * 4)); CUDA(cuMemcpyHtoD(db, src_b, nb * 4));
    job jobs[11]; int nj = 0;
    int4_matrix matrices[2][2];
    CUdeviceptr other_a[2] = {0,0}, other_b[2] = {0,0};
    for (int format = 0; format < 2; format++) {
        int4_matrix a = matrix_alloc(M, K), b = matrix_alloc(N, K);
        matrices[format][0] = a; matrices[format][1] = b;
        size_t ac = int4_codes_size(M,K), bc = int4_codes_size(N,K), as = int4_scales_size(M,K), bs = int4_scales_size(N,K);
        uint8_t *ah = host_alloc(ac), *bh = host_alloc(bc), *ash = host_alloc(as), *bsh = host_alloc(bs);
        int4_quant_stats sta = {0,0}, stb = {0,0};
        if (o->unit) {
            memset(ah,0,ac); memset(bh,0,bc); memset(ash,0x38,as); memset(bsh,0x38,bs);
            for (int r = 0; r < M; r++) for (int k = 0; k < K; k++) {
                float x = src_a[(size_t)r*K+k];
                ah[(size_t)r*(kp/2)+k/2] |= int4_code_encode(x,1,format) << (4*(k&1));
                if (!format && fabsf(x) > 6) sta.clipped_values++;
            }
            for (int r = 0; r < N; r++) for (int k = 0; k < K; k++) {
                float x = src_b[(size_t)r*K+k];
                bh[(size_t)r*(kp/2)+k/2] |= int4_code_encode(x,1,format) << (4*(k&1));
                if (!format && fabsf(x) > 6) stb.clipped_values++;
            }
        } else {
            REQUIRE(!int4_quantize_cpu(src_a,M,K,ah,ash,format,&sta));
            REQUIRE(!int4_quantize_cpu(src_b,N,K,bh,bsh,format,&stb));
        }
        CUDA(cuMemcpyHtoD(a.codes,ah,ac)); CUDA(cuMemcpyHtoD(b.codes,bh,bc));
        CUDA(cuMemcpyHtoD(a.scales,ash,as)); CUDA(cuMemcpyHtoD(b.scales,bsh,bs));
        job base = {.ctx=ctx,.bf16=bf16,.i8=i8,.legacy=legacy,.a=a,.b=b,.src_a=da,.src_b=db,.output=output,
                    .format=format,.name=format?"e0m3":"e2m1",.stats={sta.clipped_values+stb.clipped_values,sta.clamped_scales+stb.clamped_scales}};
        reference_samples(&base,ah,ash,bh,bsh,src_a,src_b);
        if (!strcmp(o->backend,"all") || !strcmp(o->backend,base.name)) {
            float best = INFINITY;
            for (int v = 0; v < INT4_VARIANTS; v++) {
                if (o->variant >= 0 && o->variant != v) continue;
                base.variant = v; REQUIRE(!run_job(&base)); verify_job(&base);
                float ms = time_job(&base, 3);
                if (ms < best) { best = ms; base.times[0] = (float)v; }
            }
            base.variant = (int)base.times[0];
            for (int mode = 0; mode < 3; mode++) {
                const char *name = mode == 0 ? "packed" : mode == 1 ? "activation" : "full";
                if (strcmp(o->mode,"all") && strcmp(o->mode,name)) continue;
                if (o->unit && mode) continue;
                base.mode = mode; jobs[nj++] = base;
            }
        }
        if (format == 0 && legacy) {
            for (int v = 0; v < 3; v++) {
                if (!fp4_legacy_supported(a, b, v)) {
                    printf("# SKIP %s %s: original kernel requires even N; reference also requires M%%16=N%%8=0\n",
                        s->name, fp4_legacy_names[v]);
                    continue;
                }
                job j = base; j.kind = 3; j.mode = 0; j.variant = v; j.name = fp4_legacy_names[v];
                REQUIRE(!run_job(&j)); verify_job(&j); jobs[nj++] = j;
            }
        }
        if (format == 1) {
            for (int kind = 1; kind <= 2; kind++) {
                const char *name = kind == 1 ? "bf16" : "int8";
                if (strcmp(o->backend,"all") && strcmp(o->backend,name)) continue;
                if (strcmp(o->mode,"all") && strcmp(o->mode,"packed")) continue;
                if ((kind == 1 && !bf16) || (kind == 2 && (!o->unit || !i8->handle))) {
                    printf("# SKIP %s %s: %s\n",s->name,name,kind==2&&!o->unit?"K16 scales require a scaled reduction; use --scales unit":"cuBLAS unavailable"); continue;
                }
                size_t bytes = kind == 1 ? 2 : 1;
                other_a[kind-1] = device_alloc((size_t)M*kp*bytes); other_b[kind-1] = device_alloc((size_t)N*kp*bytes);
                int4_matrix operands[] = {a,b}; CUdeviceptr targets[] = {other_a[kind-1],other_b[kind-1]};
                for (int p = 0; p < 2; p++) {
                    int4_matrix mat = operands[p]; void *args_bf[] = {&mat.codes,&mat.scales,&targets[p],&mat.rows,&kp,&format};
                    void *args_i8[] = {&mat.codes,&targets[p],&mat.rows,&kp};
                    size_t blocks = ((size_t)mat.rows*kp+255)/256;
                    CUDA(cuLaunchKernel(kind==1?ctx->dequant_bf16:ctx->unpack_int8,(unsigned)blocks,1,1,256,1,1,0,NULL,kind==1?args_bf:args_i8,NULL));
                }
                job j = base; j.kind = kind; j.mode = 0; j.variant = 0; j.name = name;
                j.other_a = targets[0]; j.other_b = targets[1];
                int status = run_job(&j);
                if (status) printf("# SKIP %s %s: cuBLAS status %d\n",s->name,name,status);
                else { verify_job(&j); jobs[nj++] = j; }
            }
        }
        free(ah);free(bh);free(ash);free(bsh);
    }
    for (int j = 0; j < nj; j++) for (int w = 0; w < o->warmup; w++) REQUIRE(!run_job(&jobs[j]));
    for (int r = 0; r < o->rounds; r++) for (int k = 0; k < nj; k++) {
        int j = (k+r)%nj; jobs[j].times[r] = time_job(&jobs[j],o->iters);
    }
    for (int j = 0; j < nj; j++) {
        job *p = &jobs[j]; REQUIRE(!run_job(p)); verify_job(p);
        report(csv,s,p->name,p->mode==0?"packed":p->mode==1?"activation":"full",
            p->kind==3?fp4_legacy_names[p->variant]:p->kind?"cublas":int4_variant_names[p->variant],
            o,p->times,p->rel_l2,p->max_error,p->quant_l2,p->stats);
    }
    for (int f = 0; f < 2; f++) { matrix_free(matrices[f][0]); matrix_free(matrices[f][1]); }
    for (int i = 0; i < 2; i++) { if(other_a[i]) CUDA(cuMemFree(other_a[i])); if(other_b[i]) CUDA(cuMemFree(other_b[i])); }
    CUDA(cuMemFree(da));CUDA(cuMemFree(db));CUDA(cuMemFree(output)); free(src_a);free(src_b);
}

static int integer_arg(const char *text, int allow_zero) {
    errno = 0; char *end; long n = strtol(text,&end,10);
    if (errno || end == text || *end || n < (allow_zero?0:1) || n > INT_MAX) { fprintf(stderr,"invalid integer: %s\n",text); exit(1); }
    return (int)n;
}

static void usage(void) {
    puts("bench_int4 [--shape NAME|all] [--m M --n N --k K]\n"
         "  --mode packed|activation|full|all|probe  --backend all|e0m3|fp4|e2m1|bf16|int8\n"
         "  --scales k16|unit  --variant auto|0..8  --warmup 5 --iters 100 --rounds 5\n"
         "  --csv FILE --cubin-dir DIR --probe-blocks 4096 --probe-repeats 512\n"
         "  --legacy-fp4 0|1  add the actual cuda/fp4 packed-GEMM kernels (NVRTC)\n"
         "Unit scales support packed/probe timing only. Shapes: square_1k/2k/4k/8k,\n"
         "mm0, mm2, qkv, ffn_up, ffn_down, small_m1, small_m16, small_m64.");
}

int main(int argc, char **argv) {
    options o = {.mode="packed",.backend="all",.shape_name="square_1k",.warmup=5,.iters=100,.rounds=5,
                  .variant=-1,.probe_blocks=4096,.probe_repeats=512};
    default_cubin_dir(o.dir,sizeof(o.dir));
    for (int i = 1; i < argc; i++) {
        const char *key = argv[i];
        if (!strcmp(key,"--help")) {usage();return 0;}
        if (i+1 == argc) {usage();return 1;}
        const char *value = argv[++i];
        if (!strcmp(key,"--mode")) o.mode=value;
        else if (!strcmp(key,"--backend")) o.backend=!strcmp(value,"fp4")?"e2m1":value;
        else if (!strcmp(key,"--shape")) o.shape_name=value;
        else if (!strcmp(key,"--csv")) o.csv=value;
        else if (!strcmp(key,"--cubin-dir")) { REQUIRE(strlen(value)<sizeof(o.dir));strcpy(o.dir,value); }
        else if (!strcmp(key,"--m")) o.m=integer_arg(value,0);
        else if (!strcmp(key,"--n")) o.n=integer_arg(value,0);
        else if (!strcmp(key,"--k")) o.k=integer_arg(value,0);
        else if (!strcmp(key,"--warmup")) o.warmup=integer_arg(value,1);
        else if (!strcmp(key,"--iters")) o.iters=integer_arg(value,0);
        else if (!strcmp(key,"--rounds")) {o.rounds=integer_arg(value,0); REQUIRE(o.rounds<=101 && (o.rounds&1));}
        else if (!strcmp(key,"--legacy-fp4")) {o.legacy_fp4=integer_arg(value,1); REQUIRE(o.legacy_fp4<=1);}
        else if (!strcmp(key,"--probe-blocks")) {o.probe_blocks=integer_arg(value,0);REQUIRE(o.probe_blocks<=131071);}
        else if (!strcmp(key,"--probe-repeats")) {o.probe_repeats=integer_arg(value,0);REQUIRE(o.probe_repeats<=32767);}
        else if (!strcmp(key,"--variant")) {if(strcmp(value,"auto")) {o.variant=integer_arg(value,1);REQUIRE(o.variant<INT4_VARIANTS);}}
        else if (!strcmp(key,"--scales")) {REQUIRE(!strcmp(value,"unit")||!strcmp(value,"k16"));o.unit=!strcmp(value,"unit");}
        else {fprintf(stderr,"unknown argument: %s\n",key);return 1;}
    }
    REQUIRE(!strcmp(o.mode,"packed")||!strcmp(o.mode,"activation")||!strcmp(o.mode,"full")||!strcmp(o.mode,"all")||!strcmp(o.mode,"probe"));
    REQUIRE(!strcmp(o.backend,"all")||!strcmp(o.backend,"e0m3")||!strcmp(o.backend,"e2m1")||!strcmp(o.backend,"bf16")||!strcmp(o.backend,"int8"));
    REQUIRE(!o.unit||!strcmp(o.mode,"packed")||!strcmp(o.mode,"all")||!strcmp(o.mode,"probe"));
    if (o.legacy_fp4 && strcmp(o.mode,"packed") && strcmp(o.mode,"all")) {
        fprintf(stderr,"--legacy-fp4 requires --mode packed or all\n"); return 1;
    }
    if ((!strcmp(o.backend,"bf16") || !strcmp(o.backend,"int8")) &&
        (!strcmp(o.mode,"activation") || !strcmp(o.mode,"full"))) {
        fprintf(stderr,"cuBLAS baselines support --mode packed, all, or probe\n"); return 1;
    }
    REQUIRE(!(o.m||o.n||o.k)||(o.m&&o.n&&o.k)); REQUIRE(!o.k||int4_padded_k(o.k));
    int shape_found = !strcmp(o.shape_name,"all");
    for (unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) if(!strcmp(o.shape_name,shapes[s].name)) shape_found=1;
    REQUIRE(shape_found||o.m);
    FILE *csv = o.csv ? fopen(o.csv,"w") : NULL; REQUIRE(!o.csv||csv);
    if(csv) fputs("shape,m,n,k,backend,mode,variant,scales,warmup,iters,rounds,median_ms,min_ms,max_ms,dense_tops,rel_l2,max_abs_error,quant_rel_l2,clipped_values,clamped_scales,verification\n",csv);
    CUcontext cuda=start_cuda();int4_context ctx;REQUIRE(!int4_init(&ctx,o.dir));
    char name[256];int driver=0;size_t free_bytes,total_bytes;
    CUDA(cuDeviceGetName(name,sizeof(name),0));CUDA(cuDriverGetVersion(&driver));CUDA(cuMemGetInfo(&free_bytes,&total_bytes));
    printf("# GPU=%s driver_api=%d memory=%zu MiB free=%zu MiB\n",name,driver,total_bytes>>20,free_bytes>>20);
    int8_blas i8; int8_blas_init(&i8);int version=0;if(i8.handle)i8.version(i8.handle,&version);
    fp4_legacy_context legacy = {0};
    int nvrtc_major = 0, nvrtc_minor = 0;
    if (o.legacy_fp4) {
        REQUIRE(!fp4_legacy_init(&legacy)); nvrtcVersion(&nvrtc_major, &nvrtc_minor);
        printf("# Reusing cuda/fp4 reference/sync/cp.async sources; NVRTC=%d.%d; compilation excluded from timing\n",
            nvrtc_major, nvrtc_minor);
    }
    printf("# cuBLAS=%d CUBIN_dir=%s; timing excludes allocation/transfer/tuning; validation=64 CPU FP64 samples\n",version,o.dir);
    if(o.csv) {
        char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s.metadata.txt",o.csv)<(int)sizeof(path));
        FILE *meta=fopen(path,"w");REQUIRE(meta);
        fprintf(meta,"GPU=%s\ndriver_api=%d\ncuBLAS=%d\nmemory_bytes=%zu\n",name,driver,version,total_bytes);
        if (o.legacy_fp4) fprintf(meta,"legacy_FP4=reference,sync,cp.async\nNVRTC=%d.%d\n",nvrtc_major,nvrtc_minor);
        fputs("command=",meta);for(int i=0;i<argc;i++)fprintf(meta,"%s%s",i?" ":"",argv[i]);fputc('\n',meta);
        FILE *info=popen("nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,clocks.current.sm,clocks.max.sm --format=csv,noheader","r");
        if(info){char line[1024];while(fgets(line,sizeof(line),info))fputs(line,meta);pclose(info);}fclose(meta);
    }
    cublasew_context *bf16=NULL;if(!cublasewInit())cublasewCreate(&bf16,NULL);
    if(!strcmp(o.mode,"probe"))run_probes(&ctx,&o,csv);
    else if(o.m){shape s={"custom",o.m,o.n,o.k};matrix_benchmark(&ctx,bf16,&i8,o.legacy_fp4?&legacy:NULL,&s,&o,csv);}
    else for(unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++)
        if(!strcmp(o.shape_name,"all")||!strcmp(o.shape_name,shapes[s].name))matrix_benchmark(&ctx,bf16,&i8,o.legacy_fp4?&legacy:NULL,&shapes[s],&o,csv);
    fp4_legacy_destroy(&legacy);
    if (bf16) cublasewDestroy(bf16);
    if (i8.handle) i8.destroy(i8.handle);
    if (i8.library) dlclose(i8.library);
    int4_destroy(&ctx);CUDA(cuCtxDestroy(cuda));if(csv)fclose(csv);return 0;
}
