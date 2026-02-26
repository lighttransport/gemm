/*
 * test_mma_probe.c - Probe the actual MMA m16n8k16 fragment layout on sm_120
 *
 * Strategy: load each A register with a unique "tag" value and set B=identity,
 * then read back D to see which A register ended up where.
 *
 * A fragment has 4 registers (a0..a3), each f16x2.
 * We tag: a0={1.0, 2.0}, a1={3.0, 4.0}, a2={5.0, 6.0}, a3={7.0, 8.0}
 * for each thread, then see what D produces.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../cuew.h"

static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    uint32_t x = u.i;
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int32_t exp = ((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) {
        if (exp < -24) return sign;
        mant |= 0x800000;
        mant >>= (-1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    return sign | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13);
}

/*
 * Kernel: each thread loads its own tagged A values,
 * B = identity (only gid-th column is 1.0 at matching K positions),
 * then stores all 4 D registers + lane info to output.
 */
static const char *probe_kernel_src =
"typedef unsigned short half_raw;\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* Output: per-thread [lane, d0, d1, d2, d3] */\n"
"__global__ void mma_probe(float *out, const half_raw *W, int n_in) {\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"\n"
"    /* Load B from W (identity) */\n"
"    int bc = gid;\n"
"    const half_raw *wp = W + bc * n_in;\n"
"    unsigned int b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"    unsigned int b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"\n"
"    /* Tag A registers with unique per-thread values */\n"
"    /* a0 = {tag_lo, tag_hi} where tag encodes register# */\n"
"    /* Use values that let us trace which register contributed */\n"
"    float tag_base = (float)(lane * 100);\n"
"    unsigned int a0, a1, a2, a3;\n"
"    /* a0: should map to A[gid][tid4*2..tid4*2+1] (rows 0-7, cols 0-7) */\n"
"    asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a0) : \"f\"(tag_base + 1.0f), \"f\"(tag_base + 2.0f));\n"
"    /* a1: should map to A[gid][tid4*2+8..tid4*2+9] (rows 0-7, cols 8-15) */\n"
"    asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a1) : \"f\"(tag_base + 3.0f), \"f\"(tag_base + 4.0f));\n"
"    /* a2: should map to A[gid+8][tid4*2..tid4*2+1] (rows 8-15, cols 0-7) */\n"
"    asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a2) : \"f\"(tag_base + 5.0f), \"f\"(tag_base + 6.0f));\n"
"    /* a3: should map to A[gid+8][tid4*2+8..tid4*2+9] (rows 8-15, cols 8-15) */\n"
"    asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a3) : \"f\"(tag_base + 7.0f), \"f\"(tag_base + 8.0f));\n"
"\n"
"    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;\n"
"    asm volatile(\n"
"        \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"        \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"        : \"=f\"(d0), \"=f\"(d1), \"=f\"(d2), \"=f\"(d3)\n"
"        : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"          \"r\"(b0), \"r\"(b1),\n"
"          \"f\"(d0), \"f\"(d1), \"f\"(d2), \"f\"(d3)\n"
"    );\n"
"\n"
"    /* Store results: 5 floats per lane */\n"
"    out[lane * 5 + 0] = (float)lane;\n"
"    out[lane * 5 + 1] = d0;\n"
"    out[lane * 5 + 2] = d1;\n"
"    out[lane * 5 + 3] = d2;\n"
"    out[lane * 5 + 4] = d3;\n"
"}\n"
"\n"
"} /* extern C */\n";

int main(void) {
    printf("=== MMA Fragment Probe (sm_120) ===\n");
    printf("A regs tagged: a0={base+1,base+2} a1={base+3,base+4} a2={base+5,base+6} a3={base+7,base+8}\n");
    printf("B = identity[8,16], D should reveal which A reg maps where\n\n");

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0);

    CUdevice device; CUcontext ctx;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&ctx, 0, device);

    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    printf("GPU: sm_%d%d\n", major, minor);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, probe_kernel_src, "probe.cu", 0, NULL, NULL);
    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    const char *opts[] = { arch };
    nvrtcResult nres = nvrtcCompileProgram(prog, 1, opts);
    if (nres != NVRTC_SUCCESS) {
        size_t log_sz; nvrtcGetProgramLogSize(prog, &log_sz);
        char *log = (char *)malloc(log_sz);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "NVRTC error:\n%s\n", log);
        free(log); return 1;
    }

    size_t ptx_sz; nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = (char *)malloc(ptx_sz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule module;
    cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    free(ptx);

    CUfunction fn;
    cuModuleGetFunction(&fn, module, "mma_probe");

    /* W = identity[8,16]: W[n][k] = (n==k) ? 1 : 0 */
    int N = 8, K = 16;
    uint16_t h_W[8 * 16];
    memset(h_W, 0, sizeof(h_W));
    for (int n = 0; n < 8; n++)
        h_W[n * 16 + n] = f32_to_f16(1.0f);

    CUdeviceptr d_W, d_out;
    cuMemAlloc(&d_W, N * K * sizeof(uint16_t));
    cuMemAlloc(&d_out, 32 * 5 * sizeof(float));
    cuMemcpyHtoD(d_W, h_W, N * K * sizeof(uint16_t));
    cuMemsetD32(d_out, 0, 32 * 5);

    int n_in = K;
    void *args[] = { &d_out, &d_W, &n_in };
    cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, args, NULL);
    cuCtxSynchronize();

    float h_out[32 * 5];
    cuMemcpyDtoH(h_out, d_out, 32 * 5 * sizeof(float));

    /* Print results */
    printf("\nPer-thread MMA output (D fragment):\n");
    printf("Expected mapping (pre-Blackwell):\n");
    printf("  d0 = D[gid][tid4*2]     d1 = D[gid][tid4*2+1]\n");
    printf("  d2 = D[gid+8][tid4*2]   d3 = D[gid+8][tid4*2+1]\n");
    printf("With B=identity, D[row][col] = A[row][col], so:\n");
    printf("  d0 = A[gid][tid4*2]       -> from a0.lo (base+1)\n");
    printf("  d1 = A[gid][tid4*2+1]     -> from a0.hi (base+2)\n");
    printf("  d2 = A[gid+8][tid4*2]     -> from a2.lo (base+5)\n");
    printf("  d3 = A[gid+8][tid4*2+1]   -> from a2.hi (base+6)\n\n");

    printf("lane gid tid4 |   d0 (expected base+1)   d1 (expected base+2)   d2 (expected base+5)   d3 (expected base+6)\n");
    printf("------------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < 32; i++) {
        int lane = (int)h_out[i * 5 + 0];
        int gid = lane / 4;
        int tid4 = lane % 4;
        float base = (float)(lane * 100);
        float d0 = h_out[i * 5 + 1];
        float d1 = h_out[i * 5 + 2];
        float d2 = h_out[i * 5 + 3];
        float d3 = h_out[i * 5 + 4];

        /* Determine which A register the value came from */
        const char *src_d0 = "???", *src_d1 = "???", *src_d2 = "???", *src_d3 = "???";
        float diff;
        diff = d0 - base; if (fabsf(diff - 1) < 0.5f) src_d0 = "a0.lo"; else if (fabsf(diff - 2) < 0.5f) src_d0 = "a0.hi";
        else if (fabsf(diff - 3) < 0.5f) src_d0 = "a1.lo"; else if (fabsf(diff - 4) < 0.5f) src_d0 = "a1.hi";
        else if (fabsf(diff - 5) < 0.5f) src_d0 = "a2.lo"; else if (fabsf(diff - 6) < 0.5f) src_d0 = "a2.hi";
        else if (fabsf(diff - 7) < 0.5f) src_d0 = "a3.lo"; else if (fabsf(diff - 8) < 0.5f) src_d0 = "a3.hi";

        diff = d1 - base; if (fabsf(diff - 1) < 0.5f) src_d1 = "a0.lo"; else if (fabsf(diff - 2) < 0.5f) src_d1 = "a0.hi";
        else if (fabsf(diff - 3) < 0.5f) src_d1 = "a1.lo"; else if (fabsf(diff - 4) < 0.5f) src_d1 = "a1.hi";
        else if (fabsf(diff - 5) < 0.5f) src_d1 = "a2.lo"; else if (fabsf(diff - 6) < 0.5f) src_d1 = "a2.hi";
        else if (fabsf(diff - 7) < 0.5f) src_d1 = "a3.lo"; else if (fabsf(diff - 8) < 0.5f) src_d1 = "a3.hi";

        diff = d2 - base; if (fabsf(diff - 1) < 0.5f) src_d2 = "a0.lo"; else if (fabsf(diff - 2) < 0.5f) src_d2 = "a0.hi";
        else if (fabsf(diff - 3) < 0.5f) src_d2 = "a1.lo"; else if (fabsf(diff - 4) < 0.5f) src_d2 = "a1.hi";
        else if (fabsf(diff - 5) < 0.5f) src_d2 = "a2.lo"; else if (fabsf(diff - 6) < 0.5f) src_d2 = "a2.hi";
        else if (fabsf(diff - 7) < 0.5f) src_d2 = "a3.lo"; else if (fabsf(diff - 8) < 0.5f) src_d2 = "a3.hi";

        diff = d3 - base; if (fabsf(diff - 1) < 0.5f) src_d3 = "a0.lo"; else if (fabsf(diff - 2) < 0.5f) src_d3 = "a0.hi";
        else if (fabsf(diff - 3) < 0.5f) src_d3 = "a1.lo"; else if (fabsf(diff - 4) < 0.5f) src_d3 = "a1.hi";
        else if (fabsf(diff - 5) < 0.5f) src_d3 = "a2.lo"; else if (fabsf(diff - 6) < 0.5f) src_d3 = "a2.hi";
        else if (fabsf(diff - 7) < 0.5f) src_d3 = "a3.lo"; else if (fabsf(diff - 8) < 0.5f) src_d3 = "a3.hi";

        printf(" %2d  %d    %d  | %7.0f %-6s  %7.0f %-6s  %7.0f %-6s  %7.0f %-6s\n",
               lane, gid, tid4, d0, src_d0, d1, src_d1, d2, src_d2, d3, src_d3);
    }

    cuMemFree(d_W); cuMemFree(d_out);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);
    return 0;
}
