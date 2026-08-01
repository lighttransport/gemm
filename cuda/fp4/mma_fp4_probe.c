/*
 * Raw mma.sync issue-rate probe for sm_120 (consumer Blackwell, e.g. RTX 5060 Ti)
 *
 * Definitive silicon-datapath test for narrow tensor-core formats: do they
 * assemble and run, and -- crucially -- at what *issue rate* (G mma/s)?
 *
 *   - ptxas only emits the SASS tensor-core op if the *target arch* supports it.
 *     So if e.g. `mma.sync...e2m1...` assembles at sm_120a the format is exposed;
 *     if ptxas rejects it ("not supported on target") it is not.
 *   - Assembling is necessary but NOT sufficient for "HW-accelerated": ptxas may
 *     *lower* an unsupported mma to an emulation (e.g. INT4 s4 -> 2x IMMA.s8 +
 *     integer unpacking). The mma issue rate exposes that -- native ops issue at
 *     the full tensor-core rate, emulated ones far slower.
 *   - Garbage operands: this measures issue throughput, not a correct GEMM.
 *
 * Each format is compiled as a SEPARATE NVRTC program at --gpu-architecture=sm_120a
 * (the repo's shared cu_compile_kernels hardcodes plain sm_120, which will NOT
 * assemble the block-scale FP4 mma), so one format failing does not kill the others.
 *
 * MMA shapes / ops per warp-level instruction (2*M*N*K):
 *   BF16  m16n8k16 :  4096   (f32 accum)
 *   FP8   m16n8k32 :  8192   (e4m3, f32 accum, no block scale)
 *   FP4   m16n8k64 : 16384   (e2m1, f32 accum, block_scale: NVFP4 vec16/ue4m3, MXFP4 vec32/ue8m0)
 *   INT8  m16n8k32 :  8192   (s8,  s32 accum)
 *   INT4  m16n8k64 : 16384   (s4,  s32 accum)  -- emulated on sm_120 (see README)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "cuew.h"

#ifndef NACC
#define NACC  16      /* independent accumulator tiles per lane (hide mma latency) */
#endif

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { const char* s; cuGetErrorString(err, &s); \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, s); exit(1); } } while(0)

/* The per-format inner MMA (one warp-level instruction, accumulate in place).
 * %0..%3 = D/C (f32x4, +f), %4..%7 = A (b32x4), %8..%9 = B (b32x2),
 * and for FP4: %10 = scale-A (b32), %11 = scale-B (b32); the {byte-id,thread-id}
 * selectors are literal {0,0}. */
/* NB: these bodies are copied VERBATIM into the generated CUDA source (not via
 * printf), so use single '%' for asm operand refs. */
static const char* MMA_BF16 =
  "    asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \"\n"
  "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
  "      : \"+f\"(d##J##0),\"+f\"(d##J##1),\"+f\"(d##J##2),\"+f\"(d##J##3)\n"
  "      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1));\n";

static const char* MMA_FP8 =
  "    asm volatile(\"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \"\n"
  "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
  "      : \"+f\"(d##J##0),\"+f\"(d##J##1),\"+f\"(d##J##2),\"+f\"(d##J##3)\n"
  "      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1));\n";

static const char* MMA_NVFP4 =
  "    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
  "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
  "      : \"+f\"(d##J##0),\"+f\"(d##J##1),\"+f\"(d##J##2),\"+f\"(d##J##3)\n"
  "      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n";

static const char* MMA_MXFP4 =
  "    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 \"\n"
  "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
  "      : \"+f\"(d##J##0),\"+f\"(d##J##1),\"+f\"(d##J##2),\"+f\"(d##J##3)\n"
  "      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n";

/* Integer tensor-core mma: s32 accumulator, s8/s4 packed inputs. */
static const char* MMA_INT8 =
  "    asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \"\n"
  "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
  "      : \"+r\"(d##J##0),\"+r\"(d##J##1),\"+r\"(d##J##2),\"+r\"(d##J##3)\n"
  "      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1));\n";

static const char* MMA_INT4 =
  "    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 \"\n"
  "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
  "      : \"+r\"(d##J##0),\"+r\"(d##J##1),\"+r\"(d##J##2),\"+r\"(d##J##3)\n"
  "      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1));\n";

/* Build the full CUDA kernel source for one format.  is_int selects an s32
 * accumulator (integer mma, "+r") vs an f32 accumulator (float mma, "+f"). */
static char* build_source(const char* mma_body, int is_int) {
    const char* ty   = is_int ? "int" : "float";
    const char* zero = is_int ? "0"   : "0.f";
    char* buf = (char*)malloc(1 << 16);
    int n = 0;
    n += sprintf(buf + n,
        "extern \"C\" __global__ void probe(float* out, int iters, unsigned seed) {\n"
        "  unsigned t = threadIdx.x + blockIdx.x * blockDim.x + seed;\n"
        "  unsigned a0=t*2654435761u+1u, a1=a0*2654435761u+1u, a2=a1*2654435761u+1u, a3=a2*2654435761u+1u;\n"
        "  unsigned b0=a3*2654435761u+1u, b1=b0*2654435761u+1u;\n"
        "  unsigned sfa=0x38383838u + (t & 1u), sfb=0x7f7f7f7fu + (t & 1u);\n");
    for (int j = 0; j < NACC; j++)
        n += sprintf(buf + n, "  %s d%d0=%s,d%d1=%s,d%d2=%s,d%d3=%s;\n", ty, j, zero, j, zero, j, zero, j, zero);
    n += sprintf(buf + n, "  (void)sfa;(void)sfb;\n  for (int i = 0; i < iters; i++) {\n");
    /* expand the per-tile mma body, substituting the tile index J */
    for (int j = 0; j < NACC; j++) {
        char tile[2048];
        const char* p = mma_body; char* q = tile;
        while (*p) {
            if (p[0]=='#'&&p[1]=='#'&&p[2]=='J'&&p[3]=='#'&&p[4]=='#') { q += sprintf(q, "%d", j); p += 5; }
            else *q++ = *p++;
        }
        *q = 0;
        n += sprintf(buf + n, "%s", tile);
    }
    n += sprintf(buf + n, "  }\n  %s s=%s;\n", ty, zero);
    for (int j = 0; j < NACC; j++)
        n += sprintf(buf + n, "  s += d%d0+d%d1+d%d2+d%d3;\n", j, j, j, j);
    n += sprintf(buf + n, "  out[t] = (float)s;\n}\n");
    return buf;
}

/* Compile one source at sm_120a. Returns module, or NULL (prints ptxas log). */
static CUmodule compile_sm120a(const char* src, const char* tag) {
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, tag, 0, NULL, NULL) != NVRTC_SUCCESS) return NULL;
    const char* opts[] = { "--gpu-architecture=sm_120a" };
    nvrtcResult r = nvrtcCompileProgram(prog, 1, opts);
    if (r != NVRTC_SUCCESS) {
        size_t lsz = 0; nvrtcGetProgramLogSize(prog, &lsz);
        char* log = (char*)malloc(lsz + 1); nvrtcGetProgramLog(prog, log); log[lsz] = 0;
        fprintf(stderr, "  [%s] NVRTC/ptxas REJECTED (r=%d):\n%s\n", tag, (int)r, log);
        free(log); nvrtcDestroyProgram(&prog); return NULL;
    }
    CUmodule mod = NULL; CUresult e;
    size_t blobsz = 0;
    if (nvrtcGetCUBINSize && nvrtcGetCUBIN &&
        nvrtcGetCUBINSize(prog, &blobsz) == NVRTC_SUCCESS && blobsz > 0) {
        char* blob = (char*)malloc(blobsz); nvrtcGetCUBIN(prog, blob);
        nvrtcDestroyProgram(&prog);
        e = cuModuleLoadData(&mod, blob); free(blob);
    } else {
        size_t psz = 0; nvrtcGetPTXSize(prog, &psz);
        char* ptx = (char*)malloc(psz); nvrtcGetPTX(prog, ptx);
        nvrtcDestroyProgram(&prog);
        e = cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL); free(ptx);
    }
    if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s);
        fprintf(stderr, "  [%s] module load failed: %s\n", tag, s); return NULL; }
    return mod;
}

typedef struct { const char* name; const char* body; double flops_per_mma; int is_int; } fmt_t;

int main(int argc, char** argv) {
    int iters = 20000, blocks_per_sm = 8;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bpsm") && i + 1 < argc) blocks_per_sm = atoi(argv[++i]);
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed (need CUDA + NVRTC)\n"); return 1; }
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));
    int major=0, minor=0, smcount=0, clk=0; char name[256]={0};
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&smcount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    cuDeviceGetName(name, sizeof(name), dev);
    printf("Device: %s  (SM %d.%d, %d SMs, %d MHz)\n", name, major, minor, smcount, clk/1000);
    { int vmaj=0, vmin=0; nvrtcVersion(&vmaj, &vmin);
      printf("NVRTC %d.%d; compiling each format at --gpu-architecture=sm_120a\n", vmaj, vmin); }

    int threads = 256;
    int blocks  = smcount * blocks_per_sm;
    long warps  = (long)blocks * threads / 32;
    long total_mmas = warps * (long)iters * NACC;
    printf("grid: %d blocks x %d threads, %ld warps; iters=%d, NACC=%d => %ld mma/format\n\n",
           blocks, threads, warps, iters, NACC, total_mmas);

    fmt_t fmts[] = {
        { "BF16",  MMA_BF16,   4096.0, 0 },
        { "FP8",   MMA_FP8,    8192.0, 0 },
        { "NVFP4", MMA_NVFP4, 16384.0, 0 },
        { "MXFP4", MMA_MXFP4, 16384.0, 0 },
        { "INT8",  MMA_INT8,   8192.0, 1 },   /* s32.s8.s8.s32  m16n8k32 */
        { "INT4",  MMA_INT4,  16384.0, 1 },   /* s32.s4.s4.s32  m16n8k64 */
    };
    int nf = (int)(sizeof(fmts)/sizeof(fmts[0]));

    CUdeviceptr d_out; CHECK_CUDA(cuMemAlloc(&d_out, (size_t)blocks * threads * sizeof(float)));

    double tflops[8], gmmas[8]; int ok[8];
    for (int f = 0; f < nf; f++) {
        ok[f] = 0; tflops[f] = 0; gmmas[f] = 0;
        char* src = build_source(fmts[f].body, fmts[f].is_int);
        CUmodule mod = compile_sm120a(src, fmts[f].name);
        free(src);
        if (!mod) { printf("  [%-6s] NOT AVAILABLE (see ptxas log above)\n", fmts[f].name); continue; }
        CUfunction fn; if (cuModuleGetFunction(&fn, mod, "probe") != CUDA_SUCCESS) {
            printf("  [%-6s] kernel symbol missing\n", fmts[f].name); cuModuleUnload(mod); continue; }

        unsigned seed = 12345u;
        void* args[] = { &d_out, &iters, &seed };
        /* warmup */
        CHECK_CUDA(cuLaunchKernel(fn, blocks,1,1, threads,1,1, 0,0, args, NULL));
        CHECK_CUDA(cuCtxSynchronize());
        CUevent e0, e1; CHECK_CUDA(cuEventCreate(&e0,0)); CHECK_CUDA(cuEventCreate(&e1,0));
        CHECK_CUDA(cuEventRecord(e0,0));
        CHECK_CUDA(cuLaunchKernel(fn, blocks,1,1, threads,1,1, 0,0, args, NULL));
        CHECK_CUDA(cuEventRecord(e1,0));
        CHECK_CUDA(cuEventSynchronize(e1));
        float ms=0; CHECK_CUDA(cuEventElapsedTime(&ms, e0, e1));
        tflops[f] = (total_mmas * fmts[f].flops_per_mma / (ms/1000.0)) / 1e12;
        gmmas[f]  = total_mmas / (ms/1000.0) / 1e9;
        ok[f] = 1;
        printf("  [%-6s] %.3f ms   %7.1f TOP/s   (%.2f G mma/s)\n",
               fmts[f].name, ms, tflops[f], gmmas[f]);
        cuEventDestroy(e0); cuEventDestroy(e1); cuModuleUnload(mod);
    }

    /* summary */
    double bf16 = ok[0] ? tflops[0] : 0, fp8 = ok[1] ? tflops[1] : 0;
    printf("\n=== Summary (raw mma.sync throughput, sm_120a) ===\n");
    printf("(float rows are TFLOP/s, INT rows are TOPS -- same 1e12 ops/s unit)\n");
    printf("%-7s %-14s %12s %10s %10s\n", "format", "status", "TOP/s", "vs BF16", "vs FP8");
    for (int f = 0; f < nf; f++) {
        if (!ok[f]) { printf("%-7s %-14s %12s %10s %10s\n", fmts[f].name, "ptxas-reject", "-","-","-"); continue; }
        char rb[16]="-", rf[16]="-";
        if (bf16>0) snprintf(rb,sizeof(rb),"%.2fx",tflops[f]/bf16);
        if (fp8 >0) snprintf(rf,sizeof(rf),"%.2fx",tflops[f]/fp8);
        printf("%-7s %-14s %12.1f %10s %10s\n", fmts[f].name, "ran", tflops[f], rb, rf);
    }
    /* INT4 verdict: a native datapath issues at ~the INT8 mma rate; an emulated
     * one (ptxas lowering s4 -> 2x IMMA.s8 + ALU unpacking) issues far slower. */
    int i_int4 = -1, i_int8 = -1;
    for (int f = 0; f < nf; f++) {
        if (!strcmp(fmts[f].name, "INT4")) i_int4 = f;
        if (!strcmp(fmts[f].name, "INT8")) i_int8 = f;
    }

    printf("\nInterpretation:\n");
    printf("  A format that ASSEMBLES + runs at sm_120a is exposed; whether it is truly\n");
    printf("  HW-accelerated shows in the mma issue rate (G mma/s), not just that it ran.\n");
    printf("  FP4 (e2m1): NVFP4 ~ MXFP4 => same native datapath, only scale granularity\n");
    printf("              differs (vec16/ue4m3 vs vec32/ue8m0); ~2x FP8 issue rate.\n");
    if (i_int4 >= 0 && ok[i_int4] && i_int8 >= 0 && ok[i_int8]) {
        double r = gmmas[i_int4] / gmmas[i_int8];
        if (r > 0.7)
            printf("  INT4 (s4):  issues at %.0f%% of the INT8 rate => NATIVE 4-bit integer tensor core.\n", r*100);
        else
            printf("  INT4 (s4):  ASSEMBLES but issues at only %.0f%% of the INT8 rate => SOFTWARE-\n"
                   "              EMULATED (ptxas lowers s4 mma to ~2x IMMA.s8 + integer unpacking).\n"
                   "              Available, but NOT hardware-accelerated -- prefer INT8 or FP4.\n", r*100);
    }

    cuMemFree(d_out); cuCtxDestroy(ctx);
    return 0;
}
