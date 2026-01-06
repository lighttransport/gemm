/*
 * INT8 GEMM using CUDA with cuew - Tensor Core Version for Blackwell
 *
 * Computes: C[M,N] = A[M,K] * B[K,N]
 * - INT8 inputs (signed s8 or unsigned u8)
 * - INT32 accumulator and output
 * - Uses Tensor Core mma.sync instructions (SM 8.9+, optimized for SM 10.0+)
 *
 * Uses PTX kernel loaded via cuModuleLoadData (no CUDA SDK required)
 * Includes benchmark mode with peak TOPS analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../cuew.h"
#include "int8_types.h"

/* Default matrix dimensions - must be multiples of tile sizes */
static int g_M = 128;
static int g_N = 128;
static int g_K = 128;

/* Benchmark settings */
static int g_warmup_iters = 5;
static int g_bench_iters = 100;
static int g_benchmark_mode = 0;
static int g_use_tensor_core = 1;
static int g_tile_size = 64;
static int g_use_signed = 1;  /* 1 = signed, 0 = unsigned */
static int g_skip_verify = 0; /* Skip verification for large sizes */
static int g_use_opt = 0;     /* Use optimized kernel (double buffering) */
static int g_use_opt2 = 0;    /* Use optimized v2 kernel (transposed B) */
static int g_use_opt3 = 0;    /* Use optimized v3 kernel (K=64 unrolled) */
static int g_use_tcgen05 = 0; /* Use tcgen05.mma (Blackwell SM 10.0+) */

/* Stochastic rounding settings */
static int g_use_sr = 0;           /* Enable stochastic rounding output */
static float g_sr_scale = 0.0f;    /* SR scale (0 = auto-compute) */
static uint64_t g_sr_seed = 42;    /* RNG seed for SR */

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

/* GPU specifications for peak TOPS calculation */
typedef struct {
    int sm_count;
    int clock_mhz;
    int major;
    int minor;
    size_t mem_bandwidth_gbps;
    double int8_tensor_peak_tops;
} gpu_specs_t;

static int get_int8_tensor_ops_per_sm_per_cycle(int major, int minor) {
    int sm = major * 10 + minor;
    if (sm >= 120) return 8192;      /* Blackwell B100/B200 */
    else if (sm >= 100) return 4096; /* Blackwell B */
    else if (sm >= 90) return 2048;  /* Hopper */
    else if (sm >= 89) return 1024;  /* Ada Lovelace */
    else if (sm >= 80) return 512;   /* Ampere */
    else return 0;
}

static void query_gpu_specs(CUdevice device, gpu_specs_t* specs) {
    int clock_khz, mem_clock_khz, mem_bus_width;
    CHECK_CUDA(cuDeviceGetAttribute(&specs->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CHECK_CUDA(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
    CHECK_CUDA(cuDeviceGetAttribute(&specs->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&specs->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&mem_clock_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
    CHECK_CUDA(cuDeviceGetAttribute(&mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

    specs->clock_mhz = clock_khz / 1000;
    specs->mem_bandwidth_gbps = (size_t)mem_clock_khz * 2 * mem_bus_width / 8 / 1000000;

    int int8_ops = get_int8_tensor_ops_per_sm_per_cycle(specs->major, specs->minor);
    specs->int8_tensor_peak_tops = (double)specs->sm_count * int8_ops * specs->clock_mhz / 1e6;
}

static void print_gpu_specs(const char* device_name, const gpu_specs_t* specs) {
    printf("\n=== GPU Specifications ===\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: SM %d.%d\n", specs->major, specs->minor);
    printf("SM Count: %d\n", specs->sm_count);
    printf("GPU Clock: %d MHz\n", specs->clock_mhz);
    printf("Memory Bandwidth: %zu GB/s\n", specs->mem_bandwidth_gbps);
    printf("\nTheoretical Peak Performance:\n");
    if (specs->int8_tensor_peak_tops > 0) {
        printf("  INT8 (Tensor cores): %.2f TOPS\n", specs->int8_tensor_peak_tops);
    }
    printf("\n");
}

/*
 * 32x32 Tile Tensor Core PTX Kernel for INT8 GEMM (Signed)
 * Uses mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite
 * Block: 128 threads (4 warps), 32x32 output per block
 * Matches FP8 kernel structure exactly
 */
static const char* ptx_gemm_tc32_s8 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc32_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // smem_A: 32 rows x 32 k (row-major)\n"
"    // smem_B: 32 cols x 32 k (row-major, transposed during MMA load)\n"
"    .shared .align 16 .b8 smem_A[1024];\n"
"    .shared .align 16 .b8 smem_B[1024];\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1;\n"
"    .reg .u32 b_col, b_k;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .s32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
"    .reg .pred p_k;\n"
"    .reg .u32 out_row0, out_row1, out_col, stride32;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 5;  // * 32\n"
"    shl.b32 block_col, block_col, 5;  // * 32\n"
"\n"
"    and.b32 warp_row, warp_id, 1;     // 0 or 1\n"
"    shr.u32 warp_col, warp_id, 1;     // 0 or 1\n"
"    shl.b32 warp_row, warp_row, 4;    // * 16 -> 0 or 16\n"
"    shl.b32 warp_col, warp_col, 4;    // * 16 -> 0 or 16\n"
"\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    // Load A tile (32x32) row-major: 128 threads, 8 bytes each\n"
"    shr.u32 load_row, tid, 2;         // tid / 4 = row (0-31)\n"
"    and.b32 load_col, tid, 3;         // tid % 4 = col group\n"
"    shl.b32 load_col, load_col, 3;    // * 8 = 0,8,16,24\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.v2.b32 {a0, a1}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;    // row * 32\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {a0, a1};\n"
"\n"
"    // Load B tile (32x32) row-major\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.v2.b32 {b0, b1}, [gmem_addr];\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {b0, b1};\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // Load A from shared memory for mma - try interleaved row layout\n"
"    // For m16n8k32 INT8: each register holds 4 INT8 values\n"
"    // Try: a0 = A[row0,k0], A[row0,k1], A[row1,k0], A[row1,k1] (interleaved)\n"
"    shr.u32 a_row0, lane_id, 2;       // T / 4 (0-7)\n"
"    shl.b32 a_row0, a_row0, 1;        // * 2 -> 0,2,4,6,8,10,12,14\n"
"    add.u32 a_row0, a_row0, warp_row; // + warp's row offset\n"
"    add.u32 a_row1, a_row0, 1;        // row0 + 1\n"
"    and.b32 a_k, lane_id, 3;          // T % 4 (0-3)\n"
"    shl.b32 a_k, a_k, 2;              // * 4 -> 0,4,8,12\n"
"    // Compute base addresses for row0 and row1\n"
"    shl.b32 smem_off, a_row0, 5;      // row0 * 32\n"
"    add.u32 smem_off, smem_off, a_k;  // + k_base\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    // Try alternative layout: a0,a1 for first k half, a2,a3 for second k half\n"
"    // a0: row0, k=0..3\n"
"    // a1: row1, k=0..3 (was a2)\n"
"    // a2: row0, k=16..19 (was a1)\n"
"    // a3: row1, k=16..19\n"
"    ld.shared.b32 a0, [saddr];        // row0, k=0..3\n"
"    add.u32 saddr, saddr, 32;         // + row stride -> row1\n"
"    ld.shared.b32 a1, [saddr];        // row1, k=0..3\n"
"    sub.u32 saddr, saddr, 32;         // back to row0\n"
"    ld.shared.b32 a2, [saddr + 16];   // row0, k=16..19\n"
"    add.u32 saddr, saddr, 32;         // + row stride -> row1\n"
"    ld.shared.b32 a3, [saddr + 16];   // row1, k=16..19\n"
"\n"
"    // Load B using byte loads (B stored row-major, MMA needs col-major access)\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;              // k_base = (T%4)*4\n"
"    shr.u32 b_col, lane_id, 2;        // col = T / 4 (0-7)\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 5;         // k_base * 32\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 32];\n"
"    ld.shared.u8 byte2, [saddr + 64];\n"
"    ld.shared.u8 byte3, [saddr + 96];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    // Load b1: k_base + 16..19\n"
"    add.u32 saddr, saddr, 512;        // +16 rows * 32\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 32];\n"
"    ld.shared.u8 byte2, [saddr + 64];\n"
"    ld.shared.u8 byte3, [saddr + 96];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    // Load B for second mma (columns 8-15 within warp's 16-col region)\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 5;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 32];\n"
"    ld.shared.u8 byte2, [saddr + 64];\n"
"    ld.shared.u8 byte3, [saddr + 96];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 512;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 32];\n"
"    ld.shared.u8 byte2, [saddr + 64];\n"
"    ld.shared.u8 byte3, [saddr + 96];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    // Store using mma output layout: row0=(lane/4)*2, col=(lane%4)*2\n"
"    shr.u32 out_row0, lane_id, 2;     // T / 4\n"
"    shl.b32 out_row0, out_row0, 1;    // * 2\n"
"    add.u32 out_row0, out_row0, warp_row;\n"
"    add.u32 out_row0, out_row0, block_row;\n"
"    add.u32 out_row1, out_row0, 1;\n"
"    and.b32 out_col, lane_id, 3;      // T % 4\n"
"    shl.b32 out_col, out_col, 1;      // * 2\n"
"    add.u32 out_col, out_col, warp_col;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    shl.b32 stride32, dim_n, 2;       // stride in bytes\n"
"    cvt.u64.u32 stride64, stride32;\n"
"\n"
"    // Store c0-c3\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    // Store c4-c7 at col + 8\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 16x16 Tile Tensor Core PTX Kernel for INT8 GEMM (Unsigned)
 * Same layout as signed version, uses mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite
 */
static const char* ptx_gemm_tc32_u8 =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc32_u8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .shared .align 16 .b8 smem_A[512];\n"
"    .shared .align 16 .b8 smem_B[512];\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, lane_id, groupID, threadID;\n"
"    .reg .u32 block_row, block_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
"    .reg .u32 a_row, a_k, b_row, b_col;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .s32 c0, c1, c2, c3, c4, c5, c6, c7;\n"
"    .reg .s32 d0, d1, d2, d3, d4, d5, d6, d7;\n"
"    .reg .pred p_k;\n"
"    .reg .u32 out_row, out_col, stride32;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    mov.u32 lane_id, tid;\n"
"    shr.u32 groupID, lane_id, 2;\n"
"    and.b32 threadID, lane_id, 3;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 4;\n"
"    shl.b32 block_col, block_col, 4;\n"
"\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    shr.u32 load_row, tid, 1;\n"
"    and.b32 tmp, tid, 1;\n"
"    shl.b32 load_col, tmp, 4;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.v4.b32 {a0, a1, a2, a3}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v4.b32 [saddr], {a0, a1, a2, a3};\n"
"\n"
"    and.b32 load_row, tid, 31;\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, block_col;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.v4.b32 {b0, b1, b2, b3}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 4;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v4.b32 [saddr], {b0, b1, b2, b3};\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    shl.b32 a_row, groupID, 1;\n"
"    shl.b32 a_k, threadID, 2;\n"
"    shl.b32 smem_off, a_row, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    shl.b32 b_row, threadID, 2;\n"
"    mov.u32 b_col, groupID;\n"
"    shl.b32 smem_off, b_row, 4;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 16];\n"
"    ld.shared.u8 byte2, [saddr + 32];\n"
"    ld.shared.u8 byte3, [saddr + 48];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 256;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 16];\n"
"    ld.shared.u8 byte2, [saddr + 32];\n"
"    ld.shared.u8 byte3, [saddr + 48];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    add.u32 b_col, groupID, 8;\n"
"    shl.b32 smem_off, b_row, 4;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 16];\n"
"    ld.shared.u8 byte2, [saddr + 32];\n"
"    ld.shared.u8 byte3, [saddr + 48];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b2, byte0, byte1;\n"
"    or.b32 b2, b2, byte2;\n"
"    or.b32 b2, b2, byte3;\n"
"    add.u32 saddr, saddr, 256;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 16];\n"
"    ld.shared.u8 byte2, [saddr + 32];\n"
"    ld.shared.u8 byte3, [saddr + 48];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b3, byte0, byte1;\n"
"    or.b32 b3, b3, byte2;\n"
"    or.b32 b3, b3, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b2, b3}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    shl.b32 out_row, groupID, 1;\n"
"    shl.b32 out_col, threadID, 1;\n"
"    add.u32 out_row, out_row, block_row;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    shl.b32 stride32, dim_n, 2;\n"
"    cvt.u64.u32 stride64, stride32;\n"
"\n"
"    mul.lo.u32 offset32, out_row, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 64x64 Tile Tensor Core PTX Kernel for INT8 GEMM (Signed)
 * Block: 512 threads (16 warps), 64x64 output per block
 */
static const char* ptx_gemm_tc64_s8 =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc64_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .shared .align 16 .b8 smem_A[2048];\n"
"    .shared .align 16 .b8 smem_B[2048];\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1;\n"
"    .reg .u32 b_col, b_k;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .s32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
"    .reg .pred p_k;\n"
"    .reg .u32 out_row0, out_row1, out_col, stride32;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;\n"
"    shl.b32 block_col, block_col, 6;\n"
"\n"
"    and.b32 warp_row, warp_id, 3;\n"
"    shr.u32 warp_col, warp_id, 2;\n"
"    shl.b32 warp_row, warp_row, 4;\n"
"    shl.b32 warp_col, warp_col, 4;\n"
"\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    shr.u32 load_row, tid, 3;\n"
"    and.b32 load_col, tid, 7;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 a0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], a0;\n"
"\n"
"    shr.u32 load_row, tid, 4;\n"
"    and.b32 load_col, tid, 15;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 b0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 6;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], b0;\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    shr.u32 a_row0, lane_id, 2;\n"
"    shl.b32 a_row0, a_row0, 1;\n"
"    add.u32 a_row0, a_row0, warp_row;\n"
"    add.u32 a_row1, a_row0, 1;\n"
"    and.b32 a_k, lane_id, 3;\n"
"    shl.b32 a_k, a_k, 2;\n"
"    shl.b32 smem_off, a_row0, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    // INT8 MMA fragment layout: a0,a1 = first k half, a2,a3 = second k half\n"
"    ld.shared.b32 a0, [saddr];        // row0, k=0..3\n"
"    add.u32 saddr, saddr, 32;         // + row stride -> row1\n"
"    ld.shared.b32 a1, [saddr];        // row1, k=0..3\n"
"    sub.u32 saddr, saddr, 32;         // back to row0\n"
"    ld.shared.b32 a2, [saddr + 16];   // row0, k=16..19\n"
"    add.u32 saddr, saddr, 32;         // + row stride -> row1\n"
"    ld.shared.b32 a3, [saddr + 16];   // row1, k=16..19\n"
"\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    shr.u32 out_row0, lane_id, 2;\n"
"    shl.b32 out_row0, out_row0, 1;\n"
"    add.u32 out_row1, out_row0, 1;\n"
"    and.b32 out_col, lane_id, 3;\n"
"    shl.b32 out_col, out_col, 1;\n"
"    add.u32 out_row0, out_row0, block_row;\n"
"    add.u32 out_row0, out_row0, warp_row;\n"
"    add.u32 out_row1, out_row1, block_row;\n"
"    add.u32 out_row1, out_row1, warp_row;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    add.u32 out_col, out_col, warp_col;\n"
"    shl.b32 stride32, dim_n, 2;\n"
"    cvt.u64.u32 stride64, stride32;\n"
"\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 64x64 Tile Tensor Core PTX Kernel for INT8 GEMM (Unsigned)
 */
static const char* ptx_gemm_tc64_u8 =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc64_u8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .shared .align 16 .b8 smem_A[2048];\n"
"    .shared .align 16 .b8 smem_B[2048];\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1;\n"
"    .reg .u32 b_col, b_k;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .s32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
"    .reg .pred p_k;\n"
"    .reg .u32 out_row0, out_row1, out_col, stride32;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;\n"
"    shl.b32 block_col, block_col, 6;\n"
"\n"
"    and.b32 warp_row, warp_id, 3;\n"
"    shr.u32 warp_col, warp_id, 2;\n"
"    shl.b32 warp_row, warp_row, 4;\n"
"    shl.b32 warp_col, warp_col, 4;\n"
"\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    shr.u32 load_row, tid, 3;\n"
"    and.b32 load_col, tid, 7;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 a0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], a0;\n"
"\n"
"    shr.u32 load_row, tid, 4;\n"
"    and.b32 load_col, tid, 15;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 b0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 6;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], b0;\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    shr.u32 a_row0, lane_id, 2;\n"
"    shl.b32 a_row0, a_row0, 1;\n"
"    add.u32 a_row0, a_row0, warp_row;\n"
"    add.u32 a_row1, a_row0, 1;\n"
"    and.b32 a_k, lane_id, 3;\n"
"    shl.b32 a_k, a_k, 2;\n"
"    shl.b32 smem_off, a_row0, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    // INT8 MMA fragment layout: a0,a1 = first k half, a2,a3 = second k half\n"
"    ld.shared.b32 a0, [saddr];        // row0, k=0..3\n"
"    add.u32 saddr, saddr, 32;         // + row stride -> row1\n"
"    ld.shared.b32 a1, [saddr];        // row1, k=0..3\n"
"    sub.u32 saddr, saddr, 32;         // back to row0\n"
"    ld.shared.b32 a2, [saddr + 16];   // row0, k=16..19\n"
"    add.u32 saddr, saddr, 32;         // + row stride -> row1\n"
"    ld.shared.b32 a3, [saddr + 16];   // row1, k=16..19\n"
"\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    shr.u32 out_row0, lane_id, 2;\n"
"    shl.b32 out_row0, out_row0, 1;\n"
"    add.u32 out_row1, out_row0, 1;\n"
"    and.b32 out_col, lane_id, 3;\n"
"    shl.b32 out_col, out_col, 1;\n"
"    add.u32 out_row0, out_row0, block_row;\n"
"    add.u32 out_row0, out_row0, warp_row;\n"
"    add.u32 out_row1, out_row1, block_row;\n"
"    add.u32 out_row1, out_row1, warp_row;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    add.u32 out_col, out_col, warp_col;\n"
"    shl.b32 stride32, dim_n, 2;\n"
"    cvt.u64.u32 stride64, stride32;\n"
"\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.s32 [out_addr + 0], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr + 0], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * OPTIMIZED 64x64 Tile Tensor Core PTX Kernel for INT8 GEMM
 * Features:
 * - Double buffering to hide memory latency
 * - Vectorized loads (4 bytes at a time)
 * - Streamlined B fragment loading
 */
static const char* ptx_gemm_tc64_opt_s8 =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc64_opt_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // Double buffered shared memory: 2 x (64x32 A + 32x64 B)\n"
"    .shared .align 16 .b8 smem_A0[2048];\n"
"    .shared .align 16 .b8 smem_B0[2048];\n"
"    .shared .align 16 .b8 smem_A1[2048];\n"
"    .shared .align 16 .b8 smem_B1[2048];\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr, saddr_A, saddr_B;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1, data0;\n"
"    .reg .u32 b_col, b_k;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .s32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
"    .reg .pred p_k, p_first, p_buf;\n"
"    .reg .u32 out_row0, out_row1, out_col, stride32;\n"
"    .reg .u32 buf_sel;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;\n"
"    shl.b32 block_col, block_col, 6;\n"
"\n"
"    // Warp arrangement: 4x4 warps, each handles 16x16 output\n"
"    and.b32 warp_row, warp_id, 3;\n"
"    shr.u32 warp_col, warp_id, 2;\n"
"    shl.b32 warp_row, warp_row, 4;\n"
"    shl.b32 warp_col, warp_col, 4;\n"
"\n"
"    // Initialize accumulators\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    // Precompute load addresses (each thread loads 4 bytes)\n"
"    // A: 64 rows x 32 cols, 512 threads load 4 bytes each = 2048 bytes = full tile\n"
"    shr.u32 load_row, tid, 3;     // tid / 8 = row (0..63)\n"
"    and.b32 load_col, tid, 7;     // tid % 8 = col group (0..7)\n"
"    shl.b32 load_col, load_col, 2; // col group * 4 = byte offset (0,4,8,...,28)\n"
"\n"
"    // === PROLOGUE: Load first tile into buffer 0 ===\n"
"    // Load A tile\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 offset32, offset32, load_col;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A0;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"\n"
"    // Load B tile\n"
"    shr.u32 load_row, tid, 4;     // tid / 16 = k row (0..31)\n"
"    and.b32 load_col, tid, 15;    // tid % 16 = col group (0..15)\n"
"    shl.b32 load_col, load_col, 2; // * 4 = byte offset\n"
"    mul.lo.u32 offset32, load_row, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 6;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B0;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // k_computed = how many K elements we've processed so far\n"
"    // k_outer = next tile to load\n"
"    .reg .u32 k_computed;\n"
"    mov.u32 k_computed, 0;\n"
"    mov.u32 k_outer, 32;\n"
"    mov.u32 buf_sel, 0;\n"
"\n"
"K_LOOP_OPT:\n"
"    // Check if there's a next tile to load\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"\n"
"    // === LOAD next tile (if available) ===\n"
"    @!p_k bra SKIP_LOAD;\n"
"\n"
"    // Compute A load address\n"
"    shr.u32 load_row, tid, 3;\n"
"    and.b32 load_col, tid, 7;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"\n"
"    // Store to alternate buffer\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    setp.eq.u32 p_buf, buf_sel, 0;\n"
"    @p_buf mov.u32 saddr, smem_A1;\n"
"    @!p_buf mov.u32 saddr, smem_A0;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"\n"
"    // Compute B load address\n"
"    shr.u32 load_row, tid, 4;\n"
"    and.b32 load_col, tid, 15;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"\n"
"    shl.b32 smem_off, load_row, 6;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    @p_buf mov.u32 saddr, smem_B1;\n"
"    @!p_buf mov.u32 saddr, smem_B0;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"\n"
"SKIP_LOAD:\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // === COMPUTE using current buffer ===\n"
"    // Set base addresses based on current buffer\n"
"    setp.eq.u32 p_buf, buf_sel, 0;\n"
"    @p_buf mov.u32 saddr_A, smem_A0;\n"
"    @!p_buf mov.u32 saddr_A, smem_A1;\n"
"    @p_buf mov.u32 saddr_B, smem_B0;\n"
"    @!p_buf mov.u32 saddr_B, smem_B1;\n"
"\n"
"    // Load A fragments\n"
"    shr.u32 a_row0, lane_id, 2;\n"
"    shl.b32 a_row0, a_row0, 1;\n"
"    add.u32 a_row0, a_row0, warp_row;\n"
"    and.b32 a_k, lane_id, 3;\n"
"    shl.b32 a_k, a_k, 2;\n"
"    shl.b32 smem_off, a_row0, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    add.u32 saddr, saddr_A, smem_off;\n"
"    // INT8 MMA fragment layout\n"
"    ld.shared.b32 a0, [saddr];        // row0, k=0..3\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a1, [saddr];        // row1, k=0..3\n"
"    sub.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr + 16];   // row0, k=16..19\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a3, [saddr + 16];   // row1, k=16..19\n"
"\n"
"    // Load B fragments for first 16x8 MMA\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    add.u32 saddr, saddr_B, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    // First MMA: 16x8\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    // Load B fragments for second 16x8 MMA (columns 8-15)\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    add.u32 saddr, saddr_B, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    // Second MMA: 16x8\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    // Synchronize before swapping buffers\n"
"    bar.sync 0;\n"
"\n"
"    // Update k_computed and toggle buffer\n"
"    add.u32 k_computed, k_computed, 32;\n"
"    xor.b32 buf_sel, buf_sel, 1;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    // Continue if we haven't computed all K elements\n"
"    setp.lt.u32 p_k, k_computed, dim_k;\n"
"    @p_k bra K_LOOP_OPT;\n"
"\n"
"    // === EPILOGUE: Store results ===\n"
"    cvt.u64.u32 stride64, dim_n;\n"
"    mul.lo.u32 stride32, dim_n, 4;\n"
"\n"
"    shr.u32 out_row0, lane_id, 2;\n"
"    shl.b32 out_row0, out_row0, 1;\n"
"    add.u32 out_row0, out_row0, block_row;\n"
"    add.u32 out_row0, out_row0, warp_row;\n"
"    add.u32 out_row1, out_row0, 1;\n"
"\n"
"    and.b32 out_col, lane_id, 3;\n"
"    shl.b32 out_col, out_col, 1;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    add.u32 out_col, out_col, warp_col;\n"
"\n"
"    // Store c0,c1 (row0, col+0,1)\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"    st.global.s32 [out_addr], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"\n"
"    // Store c2,c3 (row1, col+0,1)\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    // Store c4,c5 (row0, col+8,9)\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"    st.global.s32 [out_addr], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"\n"
"    // Store c6,c7 (row1, col+8,9)\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * OPTIMIZED v3: 64x64 Tile with K=64 per iteration (unrolled)
 * Processes 2 K-tiles per loop iteration to amortize overhead
 */
static const char* ptx_gemm_tc64_opt3_s8 =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc64_opt3_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // Larger shared memory for K=64: A is 64x64, B is 64x64\n"
"    .shared .align 16 .b8 smem_A[4096];  // 64 rows x 64 cols\n"
"    .shared .align 16 .b8 smem_B[4096];  // 64 rows x 64 cols\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1, data0, data1;\n"
"    .reg .u32 b_col, b_k;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_k;\n"
"    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .s32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
"    .reg .pred p_k;\n"
"    .reg .u32 out_row0, out_col;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;\n"
"    shl.b32 block_col, block_col, 6;\n"
"\n"
"    and.b32 warp_row, warp_id, 3;\n"
"    shr.u32 warp_col, warp_id, 2;\n"
"    shl.b32 warp_row, warp_row, 4;\n"
"    shl.b32 warp_col, warp_col, 4;\n"
"\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"\n"
"K_LOOP_V3:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE_V3;\n"
"\n"
"    // === Load A tile: 64 rows x 64 cols (2 loads per thread) ===\n"
"    shr.u32 load_row, tid, 4;        // tid / 16 = row (0..31)\n"
"    and.b32 load_col, tid, 15;\n"
"    shl.b32 load_col, load_col, 2;   // col * 4 (0,4,8,...,60)\n"
"    // First half (rows 0-31)\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 6;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"    // Second half (rows 32-63)\n"
"    add.u32 tmp, tmp, 32;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 data1, [gmem_addr];\n"
"    st.shared.b32 [saddr + 2048], data1;\n"
"\n"
"    // === Load B tile: 64 rows x 64 cols (2 loads per thread) ===\n"
"    // First half (k rows 0-31)\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 6;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"    // Second half (k rows 32-63)\n"
"    add.u32 tmp, tmp, 32;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 data1, [gmem_addr];\n"
"    st.shared.b32 [saddr + 2048], data1;\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // === First K-half (k=0..31): 2 MMA ops ===\n"
"    shr.u32 a_row0, lane_id, 2;\n"
"    shl.b32 a_row0, a_row0, 1;\n"
"    add.u32 a_row0, a_row0, warp_row;\n"
"    and.b32 a_k, lane_id, 3;\n"
"    shl.b32 a_k, a_k, 2;\n"
"    // A fragments for first K-half\n"
"    shl.b32 smem_off, a_row0, 6;     // row * 64 (stride)\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    add.u32 saddr, saddr, 64;\n"
"    ld.shared.b32 a1, [saddr];\n"
"    sub.u32 saddr, saddr, 64;\n"
"    ld.shared.b32 a2, [saddr + 16];\n"
"    add.u32 saddr, saddr, 64;\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    // B fragments (same as original)\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    // Second MMA (cols 8-15)\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    // === Second K-half (k=32..63): 2 MMA ops ===\n"
"    // A fragments for second K-half (offset +32 in K)\n"
"    shl.b32 smem_off, a_row0, 6;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    add.u32 smem_off, smem_off, 32;   // +32 for second K-half\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    add.u32 saddr, saddr, 64;\n"
"    ld.shared.b32 a1, [saddr];\n"
"    sub.u32 saddr, saddr, 64;\n"
"    ld.shared.b32 a2, [saddr + 16];\n"
"    add.u32 saddr, saddr, 64;\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    // B fragments for second K-half (from rows 32-63 in smem_B)\n"
"    sub.u32 b_col, b_col, 8;  // reset to first 8 columns\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    add.u32 smem_off, smem_off, 2048;  // second half of B\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    // Second MMA (cols 8-15) for second K-half\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 6;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    add.u32 smem_off, smem_off, 2048;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 1024;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 64];\n"
"    ld.shared.u8 byte2, [saddr + 128];\n"
"    ld.shared.u8 byte3, [saddr + 192];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 64;    // Advance by 64\n"
"    bra K_LOOP_V3;\n"
"\n"
"K_DONE_V3:\n"
"    // === Store results (same as before) ===\n"
"    cvt.u64.u32 stride64, dim_n;\n"
"    shl.b64 stride64, stride64, 2;\n"
"\n"
"    shr.u32 out_row0, lane_id, 2;\n"
"    shl.b32 out_row0, out_row0, 1;\n"
"    add.u32 out_row0, out_row0, block_row;\n"
"    add.u32 out_row0, out_row0, warp_row;\n"
"\n"
"    and.b32 out_col, lane_id, 3;\n"
"    shl.b32 out_col, out_col, 1;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    add.u32 out_col, out_col, warp_col;\n"
"\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"    st.global.s32 [out_addr], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"    st.global.s32 [out_addr], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * OPTIMIZED v2: 64x64 Tile with Transposed B Storage
 * Key optimization: Store B transposed in shared memory so K is contiguous
 * This enables 4-byte loads for B fragments instead of byte-by-byte loading
 */
static const char* ptx_gemm_tc64_opt2_s8 =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tc64_opt2_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // Shared memory: A is 64x32, B is transposed to 64x32 (col x k)\n"
"    .shared .align 16 .b8 smem_A[2048];   // 64 rows x 32 cols\n"
"    .shared .align 16 .b8 smem_B_T[2048]; // 64 cols x 32 k (transposed)\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr, saddr_A, saddr_B;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1, data0;\n"
"    .reg .u32 b_col, b_k;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .s32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
"    .reg .pred p_k;\n"
"    .reg .u32 out_row0, out_row1, out_col, stride32;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;\n"
"    shl.b32 block_col, block_col, 6;\n"
"\n"
"    and.b32 warp_row, warp_id, 3;\n"
"    shr.u32 warp_col, warp_id, 2;\n"
"    shl.b32 warp_row, warp_row, 4;\n"
"    shl.b32 warp_col, warp_col, 4;\n"
"\n"
"    mov.s32 c0, 0; mov.s32 c1, 0; mov.s32 c2, 0; mov.s32 c3, 0;\n"
"    mov.s32 c4, 0; mov.s32 c5, 0; mov.s32 c6, 0; mov.s32 c7, 0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"\n"
"K_LOOP_V2:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE_V2;\n"
"\n"
"    // === Load A tile (same as before) ===\n"
"    shr.u32 load_row, tid, 3;\n"
"    and.b32 load_col, tid, 7;\n"
"    shl.b32 load_col, load_col, 2;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], data0;\n"
"\n"
"    // === Load B tile with TRANSPOSE ===\n"
"    // Load B[k, col:col+3] and store as B_T[col][k], B_T[col+1][k], etc.\n"
"    shr.u32 load_row, tid, 4;     // k row (0..31)\n"
"    and.b32 load_col, tid, 15;    // col group (0..15)\n"
"    shl.b32 load_col, load_col, 2; // col * 4 (0,4,8,...,60)\n"
"    // Load from global: B[k_outer + load_row, block_col + load_col : +3]\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 data0, [gmem_addr];\n"
"    // data0 = {B[k,col+3], B[k,col+2], B[k,col+1], B[k,col]} (little-endian)\n"
"    // Store transposed: B_T[col][k] = smem_B_T[col*32 + k]\n"
"    shl.b32 smem_off, load_col, 5;    // col * 32 (stride in transposed layout)\n"
"    add.u32 smem_off, smem_off, load_row;  // + k\n"
"    mov.u32 saddr, smem_B_T;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    // Extract and store 4 bytes to 4 different columns\n"
"    and.b32 byte0, data0, 0xFF;\n"
"    st.shared.u8 [saddr], byte0;        // B_T[col][k]\n"
"    shr.b32 tmp, data0, 8;\n"
"    and.b32 byte1, tmp, 0xFF;\n"
"    st.shared.u8 [saddr + 32], byte1;   // B_T[col+1][k]\n"
"    shr.b32 tmp, data0, 16;\n"
"    and.b32 byte2, tmp, 0xFF;\n"
"    st.shared.u8 [saddr + 64], byte2;   // B_T[col+2][k]\n"
"    shr.b32 byte3, data0, 24;\n"
"    st.shared.u8 [saddr + 96], byte3;   // B_T[col+3][k]\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // === Load A fragments (same as before) ===\n"
"    shr.u32 a_row0, lane_id, 2;\n"
"    shl.b32 a_row0, a_row0, 1;\n"
"    add.u32 a_row0, a_row0, warp_row;\n"
"    and.b32 a_k, lane_id, 3;\n"
"    shl.b32 a_k, a_k, 2;\n"
"    shl.b32 smem_off, a_row0, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    // INT8 MMA fragment layout\n"
"    ld.shared.b32 a0, [saddr];        // row0, k=0..3\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a1, [saddr];        // row1, k=0..3\n"
"    sub.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr + 16];   // row0, k=16..19\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a3, [saddr + 16];   // row1, k=16..19\n"
"\n"
"    // === Load B fragments (OPTIMIZED with transposed storage) ===\n"
"    // B_T layout: B_T[col][k], so consecutive k values are at +1 offsets\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;              // b_k = (lane%4)*4 = 0,4,8,12\n"
"    shr.u32 b_col, lane_id, 2;        // b_col = lane/4 = 0..7\n"
"    add.u32 b_col, b_col, warp_col;   // + warp offset\n"
"    // For column b_col, load k values b_k..b_k+3\n"
"    shl.b32 smem_off, b_col, 5;       // b_col * 32\n"
"    add.u32 smem_off, smem_off, b_k;  // + b_k\n"
"    mov.u32 saddr, smem_B_T;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    // Now 4 consecutive K values are at saddr, saddr+1, saddr+2, saddr+3!\n"
"    ld.shared.b32 b0, [saddr];        // b_col, k=b_k..b_k+3\n"
"    ld.shared.b32 b1, [saddr + 16];   // b_col, k=b_k+16..b_k+19\n"
"\n"
"    // First MMA: 16x8\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.s32 c0, d0; mov.s32 c1, d1; mov.s32 c2, d2; mov.s32 c3, d3;\n"
"\n"
"    // Load B fragments for second MMA (columns 8-15)\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_col, 5;\n"
"    add.u32 smem_off, smem_off, b_k;\n"
"    mov.u32 saddr, smem_B_T;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 b0, [saddr];\n"
"    ld.shared.b32 b1, [saddr + 16];\n"
"\n"
"    // Second MMA: 16x8\n"
"    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.s32 c4, d4; mov.s32 c5, d5; mov.s32 c6, d6; mov.s32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP_V2;\n"
"\n"
"K_DONE_V2:\n"
"    // === Store results ===\n"
"    cvt.u64.u32 stride64, dim_n;\n"
"    shl.b64 stride64, stride64, 2;\n"
"\n"
"    shr.u32 out_row0, lane_id, 2;\n"
"    shl.b32 out_row0, out_row0, 1;\n"
"    add.u32 out_row0, out_row0, block_row;\n"
"    add.u32 out_row0, out_row0, warp_row;\n"
"\n"
"    and.b32 out_col, lane_id, 3;\n"
"    shl.b32 out_col, out_col, 1;\n"
"    add.u32 out_col, out_col, block_col;\n"
"    add.u32 out_col, out_col, warp_col;\n"
"\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"    st.global.s32 [out_addr], c0;\n"
"    st.global.s32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr], c2;\n"
"    st.global.s32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"    st.global.s32 [out_addr], c4;\n"
"    st.global.s32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.s32 [out_addr], c6;\n"
"    st.global.s32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * tcgen05.mma PTX Kernel for INT8 GEMM (Blackwell SM 10.0+)
 *
 * Uses Blackwell's 5th generation tensor core instruction:
 * - tcgen05.mma.cta_group::1.kind::i8 for INT8 MMA
 * - Tensor Memory (TMEM) for accumulator storage
 * - Shared memory descriptors for A and B matrices
 *
 * Tile: 64x64, Block: 128 threads (1 warp for TMEM ops, others help with data loading)
 * Shape per MMA: M=64, N=64, K=32 (INT8)
 *
 * GmmaDescriptor (64-bit) layout:
 *   [0:14)   start_address (smem_addr >> 4)
 *   [16:30)  leading_byte_offset (LBO >> 4)
 *   [32:46)  stride_byte_offset (SBO >> 4)
 *   [49:52)  base_offset (for swizzle modes)
 *   [62:64)  layout_type (0=interleave, 1=B128, 2=B64, 3=B32)
 *
 * For INT8 with .kind::i8, idesc encodes shape/transpose flags (data type is in instruction)
 */
static const char* ptx_gemm_tcgen05_s8 =
".version 8.6\n"
".target sm_100a\n"
"// NOTE: tcgen05 instructions require SM 10.0 (B100/B200), NOT SM 12.0 (RTX 50)\n"
"// This kernel will NOT work on RTX 5060 Ti (SM 12.0)\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_tcgen05_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // Shared memory: A tile (64x32) + B tile (32x64) + tmem_addr storage\n"
"    .shared .align 128 .b8 smem_A[2048];   // 64 rows x 32 cols (K) = 2KB\n"
"    .shared .align 128 .b8 smem_B[2048];   // 32 rows (K) x 64 cols = 2KB\n"
"    .shared .align 4 .b32 smem_tmem_addr;  // TMEM base address storage\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, b0, b1;\n"
"    .reg .pred p_k, p_warp0, p_tid0, p_enable_d;\n"
"    .reg .u32 tmem_addr, tmem_cols;\n"
"    .reg .u64 a_desc, b_desc;\n"
"    .reg .u32 idesc;\n"
"    .reg .u32 disable0, disable1, disable2, disable3;\n"
"    .reg .s32 c0, c1, c2, c3, c4, c5, c6, c7;\n"
"    .reg .s32 c8, c9, c10, c11, c12, c13, c14, c15;\n"
"    .reg .u32 out_row, out_col, stride32;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N];\n"
"    ld.param.u32 dim_k, [param_K];\n"
"\n"
"    mov.u32 tid, %tid.x;\n"
"    shr.u32 warp_id, tid, 5;\n"
"    and.b32 lane_id, tid, 31;\n"
"\n"
"    // Block position (64x64 tiles)\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;  // * 64\n"
"    shl.b32 block_col, block_col, 6;  // * 64\n"
"\n"
"    // Check if this is warp 0 and thread 0\n"
"    setp.eq.u32 p_warp0, warp_id, 0;\n"
"    setp.eq.u32 p_tid0, tid, 0;\n"
"\n"
"    // === TMEM Allocation (single warp, stores to smem) ===\n"
"    // Allocate 64 columns for accumulator (64x64 INT32 = 16KB = 64 cols)\n"
"    mov.u32 tmem_cols, 64;\n"
"    mov.u32 saddr, smem_tmem_addr;\n"
"    @p_warp0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [saddr], tmem_cols;\n"
"    bar.sync 0;\n"
"\n"
"    // Load tmem base address from shared memory\n"
"    ld.shared.u32 tmem_addr, [smem_tmem_addr];\n"
"\n"
"    // Initialize disable_output_lane (all enabled = 0)\n"
"    mov.u32 disable0, 0;\n"
"    mov.u32 disable1, 0;\n"
"    mov.u32 disable2, 0;\n"
"    mov.u32 disable3, 0;\n"
"\n"
"    // Initialize accumulator to zero via enable_input_d = 0 (first iteration)\n"
"    // Build instruction descriptor for M=64, N=64, K=32, INT8\n"
"    // idesc format: shape + transpose flags (simplified, needs tuning)\n"
"    // Bits: [0:4) N_dim_log2, [4:8) M_mode, [8:16) flags\n"
"    // For M=64, N=64: N_dim = 64 = 2^6, so N_log2 = 6\n"
"    mov.u32 idesc, 0;  // Basic descriptor (may need adjustment)\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    // === Load A tile (64x32) row-major ===\n"
"    // 128 threads, each loads 16 bytes (64*32/128 = 16)\n"
"    shr.u32 load_row, tid, 2;         // tid / 4 -> row (0-31, then 32-63)\n"
"    and.b32 load_col, tid, 3;         // tid % 4 -> col group\n"
"    shl.b32 load_col, load_col, 3;    // * 8\n"
"    // First half: rows 0-31\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.v2.b32 {a0, a1}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;    // row * 32\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {a0, a1};\n"
"    // Second half: rows 32-63\n"
"    add.u32 tmp, load_row, 32;\n"
"    add.u32 tmp, block_row, tmp;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.v2.b32 {a0, a1}, [gmem_addr];\n"
"    add.u32 smem_off, load_row, 32;\n"
"    shl.b32 smem_off, smem_off, 5;    // (row+32) * 32\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {a0, a1};\n"
"\n"
"    // === Load B tile (32x64) row-major ===\n"
"    // 128 threads, each loads 16 bytes\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.v2.b32 {b0, b1}, [gmem_addr];\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {b0, b1};\n"
"    // Load remaining B data (next 32 bytes of each row)\n"
"    add.u32 tmp2, load_col, 32;\n"
"    add.u32 tmp2, block_col, tmp2;\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.v2.b32 {b0, b1}, [gmem_addr];\n"
"    add.u32 smem_off, load_row, 32;\n"
"    shl.b32 smem_off, smem_off, 6;    // (row) * 64 + 32\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {b0, b1};\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // === Build shared memory descriptors ===\n"
"    // GmmaDescriptor for A (64x32, row-major)\n"
"    // start_address = (smem_A >> 4) & 0x3FFF  (bits 0-13)\n"
"    // LBO = stride between K columns = 1 (contiguous) -> 0 in bits 16-29\n"
"    // SBO = stride between M rows = 32 bytes -> (32 >> 4) = 2 in bits 32-45\n"
"    // layout_type = 0 (interleave/no swizzle)\n"
"    mov.u32 saddr, smem_A;\n"
"    shr.u32 tmp, saddr, 4;\n"
"    and.b32 tmp, tmp, 0x3FFF;\n"
"    cvt.u64.u32 a_desc, tmp;           // start_address in bits 0-13\n"
"    // LBO = 32 bytes (K stride), SBO = 2048 bytes (64 rows * 32 cols)\n"
"    // LBO >> 4 = 2, SBO >> 4 = 128\n"
"    mov.u64 offset64, 0x0000008000020000;  // SBO=128<<32, LBO=2<<16\n"
"    or.b64 a_desc, a_desc, offset64;\n"
"\n"
"    // GmmaDescriptor for B (32x64, row-major -> need col-major for MMA)\n"
"    mov.u32 saddr, smem_B;\n"
"    shr.u32 tmp, saddr, 4;\n"
"    and.b32 tmp, tmp, 0x3FFF;\n"
"    cvt.u64.u32 b_desc, tmp;\n"
"    // LBO = 64 bytes (N stride), SBO = 2048 bytes\n"
"    mov.u64 offset64, 0x0000008000040000;  // SBO=128<<32, LBO=4<<16\n"
"    or.b64 b_desc, b_desc, offset64;\n"
"\n"
"    // === Execute tcgen05.mma ===\n"
"    // Note: Only thread 0 needs to issue the MMA instruction\n"
"    // tcgen05.mma has single-thread semantics\n"
"    // Format: tcgen05.mma.cta_group::1.kind::i8 [d_tmem], a_desc, b_desc, idesc, {mask[4]}, pred;\n"
"    // enable_d is a predicate: true = accumulate (D=AB+D), false = D=AB\n"
"    setp.ne.u32 p_enable_d, k_outer, 0;  // accumulate for k>0\n"
"    // @p_tid0 tcgen05.mma.cta_group::1.kind::i8 [tmem_addr], a_desc, b_desc, idesc, {disable0, disable1, disable2, disable3}, p_enable_d;\n"
"    // FIXME: tcgen05.mma syntax needs adjustment - commenting out for now\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    // === Wait for MMA completion ===\n"
"    tcgen05.wait::ld.sync.aligned;\n"
"\n"
"    // === Load results from TMEM to registers ===\n"
"    // Each warp loads its portion of the 64x64 result\n"
"    // Using tcgen05.ld with .32x32b.x8 shape\n"
"    // For now, use simpler approach: manually compute output addresses\n"
"\n"
"    // === Store results to global memory ===\n"
"    // Thread layout: 128 threads covering 64x64 = 4096 elements\n"
"    // Each thread stores 32 elements (4096/128 = 32)\n"
"    // Actually, let's use a simpler approach: each thread stores based on tid\n"
"    shr.u32 out_row, tid, 1;          // tid / 2 = row within 64\n"
"    and.b32 out_col, tid, 1;\n"
"    shl.b32 out_col, out_col, 5;      // * 32 = col offset (0 or 32)\n"
"\n"
"    // For tcgen05, we need to read from TMEM first\n"
"    // Simplified: just store zeros as placeholder (actual TMEM read is complex)\n"
"    // This kernel is a template - full implementation needs proper TMEM->reg transfer\n"
"\n"
"    add.u32 out_row, block_row, out_row;\n"
"    mul.lo.u32 stride32, out_row, dim_n;\n"
"    add.u32 out_col, block_col, out_col;\n"
"    add.u32 offset32, stride32, out_col;\n"
"    shl.b32 offset32, offset32, 2;    // * 4 (sizeof int32)\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 out_addr, ptr_C, offset64;\n"
"\n"
"    // Placeholder: store zeros (replace with actual TMEM loads)\n"
"    mov.s32 c0, 0;\n"
"    st.global.v4.s32 [out_addr], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 16], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 32], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 48], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 64], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 80], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 96], {c0, c0, c0, c0};\n"
"    st.global.v4.s32 [out_addr + 112], {c0, c0, c0, c0};\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // === TMEM Deallocation (single warp) ===\n"
"    @p_warp0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, tmem_cols;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * Naive scalar PTX kernel for INT8 GEMM (fallback, for SM < 8.9)
 */
static const char* ptx_gemm_naive_s8 =
".version 8.4\n"
".target sm_70\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_naive_s8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, addr_a, addr_b, addr_c;\n"
"    .reg .u32 M, N, K, row, col, k;\n"
"    .reg .u32 tid_x, tid_y, offset;\n"
"    .reg .u64 off64;\n"
"    .reg .s32 acc, a_val, b_val, prod;\n"
"    .reg .pred p_row, p_col, p_k;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 M, [param_M];\n"
"    ld.param.u32 N, [param_N];\n"
"    ld.param.u32 K, [param_K];\n"
"\n"
"    mov.u32 tid_x, %tid.x;\n"
"    mov.u32 tid_y, %tid.y;\n"
"    mov.u32 row, %ctaid.y;\n"
"    mov.u32 col, %ctaid.x;\n"
"    shl.b32 row, row, 4;\n"
"    shl.b32 col, col, 4;\n"
"    add.u32 row, row, tid_y;\n"
"    add.u32 col, col, tid_x;\n"
"\n"
"    setp.lt.u32 p_row, row, M;\n"
"    setp.lt.u32 p_col, col, N;\n"
"    @!p_row bra DONE;\n"
"    @!p_col bra DONE;\n"
"\n"
"    mov.s32 acc, 0;\n"
"    mov.u32 k, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k, K;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    mul.lo.u32 offset, row, K;\n"
"    add.u32 offset, offset, k;\n"
"    cvt.u64.u32 off64, offset;\n"
"    add.u64 addr_a, ptr_A, off64;\n"
"    ld.global.s8 a_val, [addr_a];\n"
"\n"
"    mul.lo.u32 offset, k, N;\n"
"    add.u32 offset, offset, col;\n"
"    cvt.u64.u32 off64, offset;\n"
"    add.u64 addr_b, ptr_B, off64;\n"
"    ld.global.s8 b_val, [addr_b];\n"
"\n"
"    mul.lo.s32 prod, a_val, b_val;\n"
"    add.s32 acc, acc, prod;\n"
"\n"
"    add.u32 k, k, 1;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    mul.lo.u32 offset, row, N;\n"
"    add.u32 offset, offset, col;\n"
"    shl.b32 offset, offset, 2;\n"
"    cvt.u64.u32 off64, offset;\n"
"    add.u64 addr_c, ptr_C, off64;\n"
"    st.global.s32 [addr_c], acc;\n"
"\n"
"DONE:\n"
"    ret;\n"
"}\n";

static const char* ptx_gemm_naive_u8 =
".version 8.4\n"
".target sm_70\n"
".address_size 64\n"
"\n"
".visible .entry gemm_int8_naive_u8(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, addr_a, addr_b, addr_c;\n"
"    .reg .u32 M, N, K, row, col, k;\n"
"    .reg .u32 tid_x, tid_y, offset;\n"
"    .reg .u64 off64;\n"
"    .reg .u32 acc, a_val, b_val, prod;\n"
"    .reg .pred p_row, p_col, p_k;\n"
"\n"
"    ld.param.u64 ptr_A, [param_A];\n"
"    ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C];\n"
"    ld.param.u32 M, [param_M];\n"
"    ld.param.u32 N, [param_N];\n"
"    ld.param.u32 K, [param_K];\n"
"\n"
"    mov.u32 tid_x, %tid.x;\n"
"    mov.u32 tid_y, %tid.y;\n"
"    mov.u32 row, %ctaid.y;\n"
"    mov.u32 col, %ctaid.x;\n"
"    shl.b32 row, row, 4;\n"
"    shl.b32 col, col, 4;\n"
"    add.u32 row, row, tid_y;\n"
"    add.u32 col, col, tid_x;\n"
"\n"
"    setp.lt.u32 p_row, row, M;\n"
"    setp.lt.u32 p_col, col, N;\n"
"    @!p_row bra DONE;\n"
"    @!p_col bra DONE;\n"
"\n"
"    mov.u32 acc, 0;\n"
"    mov.u32 k, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k, K;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    mul.lo.u32 offset, row, K;\n"
"    add.u32 offset, offset, k;\n"
"    cvt.u64.u32 off64, offset;\n"
"    add.u64 addr_a, ptr_A, off64;\n"
"    ld.global.u8 a_val, [addr_a];\n"
"\n"
"    mul.lo.u32 offset, k, N;\n"
"    add.u32 offset, offset, col;\n"
"    cvt.u64.u32 off64, offset;\n"
"    add.u64 addr_b, ptr_B, off64;\n"
"    ld.global.u8 b_val, [addr_b];\n"
"\n"
"    mul.lo.u32 prod, a_val, b_val;\n"
"    add.u32 acc, acc, prod;\n"
"\n"
"    add.u32 k, k, 1;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    mul.lo.u32 offset, row, N;\n"
"    add.u32 offset, offset, col;\n"
"    shl.b32 offset, offset, 2;\n"
"    cvt.u64.u32 off64, offset;\n"
"    add.u64 addr_c, ptr_C, off64;\n"
"    st.global.u32 [addr_c], acc;\n"
"\n"
"DONE:\n"
"    ret;\n"
"}\n";

/* Reference GEMM implementations for verification */
static void gemm_reference_s8(const int8_t* A, const int8_t* B, int32_t* C,
                               int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

static void gemm_reference_u8(const uint8_t* A, const uint8_t* B, int32_t* C,
                               int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

static int verify_results(const int32_t* gpu, const int32_t* ref, int M, int N) {
    int errors = 0;
    int32_t max_diff = 0;
    int first_error_idx = -1;
    for (int i = 0; i < M * N; i++) {
        int32_t diff = abs(gpu[i] - ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff != 0) {
            if (first_error_idx < 0) first_error_idx = i;
            errors++;
        }
    }
    if (errors > 0) {
        printf("Verification FAILED: %d errors, max diff = %d\n", errors, max_diff);
        printf("First 8 values per row (rows 0-7):\n");
        for (int row = 0; row < 8 && row < M; row++) {
            printf("Row %d GPU: ", row);
            for (int col = 0; col < 8 && col < N; col++) printf("%4d ", gpu[row*N+col]);
            printf("| REF: ");
            for (int col = 0; col < 8 && col < N; col++) printf("%4d ", ref[row*N+col]);
            printf("\n");
        }
    } else {
        printf("Verification PASSED\n");
    }
    return errors == 0;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m M          Matrix M dimension (default: %d)\n", g_M);
    printf("  -n N          Matrix N dimension (default: %d)\n", g_N);
    printf("  -k K          Matrix K dimension (default: %d)\n", g_K);
    printf("  --tile SIZE   Tile size: 32, 64 (default: %d)\n", g_tile_size);
    printf("  --signed      Use signed INT8 (default)\n");
    printf("  --unsigned    Use unsigned INT8\n");
    printf("  -b, --bench   Enable benchmark mode\n");
    printf("  --naive       Use naive scalar kernel\n");
    printf("  --no-verify   Skip CPU verification (for large sizes)\n");
    printf("  --opt         Use optimized kernel (double buffering)\n");
    printf("  --tcgen05     Use tcgen05.mma kernel (Blackwell SM 10.0+)\n");
    printf("\nStochastic Rounding:\n");
    printf("  --sr          Enable stochastic rounding output (int32->fp32->SR->int8)\n");
    printf("  --sr-scale F  Quantization scale for SR (default: auto-compute)\n");
    printf("  --sr-seed N   RNG seed for stochastic rounding (default: 42)\n");
    printf("  -h, --help    Show this help\n");
}

int main(int argc, char** argv) {
    /* Parse command line */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            g_M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            g_N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            g_K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
            g_tile_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--signed") == 0) {
            g_use_signed = 1;
        } else if (strcmp(argv[i], "--unsigned") == 0) {
            g_use_signed = 0;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bench") == 0) {
            g_benchmark_mode = 1;
        } else if (strcmp(argv[i], "--naive") == 0) {
            g_use_tensor_core = 0;
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            g_skip_verify = 1;
        } else if (strcmp(argv[i], "--opt") == 0) {
            g_use_opt = 1;
        } else if (strcmp(argv[i], "--opt2") == 0) {
            g_use_opt2 = 1;
        } else if (strcmp(argv[i], "--opt3") == 0) {
            g_use_opt3 = 1;
        } else if (strcmp(argv[i], "--tcgen05") == 0) {
            g_use_tcgen05 = 1;
        } else if (strcmp(argv[i], "--sr") == 0) {
            g_use_sr = 1;
        } else if (strcmp(argv[i], "--sr-scale") == 0 && i + 1 < argc) {
            g_sr_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--sr-seed") == 0 && i + 1 < argc) {
            g_sr_seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("INT8 GEMM: C[%d,%d] = A[%d,%d] x B[%d,%d]\n",
           g_M, g_N, g_M, g_K, g_K, g_N);
    printf("Type: %s INT8, Tile: %dx%d, Mode: %s\n",
           g_use_signed ? "Signed" : "Unsigned",
           g_tile_size, g_tile_size,
           g_use_tensor_core ? "Tensor Core" : "Naive");
    if (g_use_sr) {
        printf("Stochastic Rounding: enabled (seed=%lu, scale=%s)\n",
               (unsigned long)g_sr_seed,
               g_sr_scale > 0 ? "manual" : "auto");
    }

    /* Initialize CUDA via cuew */
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to initialize cuew\n");
        return 1;
    }

    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    char device_name[256];
    CHECK_CUDA(cuDeviceGetName(device_name, sizeof(device_name), device));

    gpu_specs_t specs;
    query_gpu_specs(device, &specs);
    print_gpu_specs(device_name, &specs);

    /* Check compute capability */
    int min_sm = g_use_tensor_core ? 89 : 70;
    int device_sm = specs.major * 10 + specs.minor;
    if (device_sm < min_sm) {
        fprintf(stderr, "Error: Device SM %d.%d < required SM %d.%d\n",
                specs.major, specs.minor, min_sm / 10, min_sm % 10);
        return 1;
    }

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    /* Select PTX kernel */
    const char* ptx_code;
    const char* kernel_name;
    int block_size;

    int actual_tile = g_tile_size;  /* Actual tile size used for grid calculation */

    if (!g_use_tensor_core) {
        ptx_code = g_use_signed ? ptx_gemm_naive_s8 : ptx_gemm_naive_u8;
        kernel_name = g_use_signed ? "gemm_int8_naive_s8" : "gemm_int8_naive_u8";
        block_size = 256;
    } else if (g_tile_size <= 32) {
        /* TC kernel produces 32x32 per block with 128 threads (4 warps) */
        ptx_code = g_use_signed ? ptx_gemm_tc32_s8 : ptx_gemm_tc32_u8;
        kernel_name = g_use_signed ? "gemm_int8_tc32_s8" : "gemm_int8_tc32_u8";
        block_size = 128;
        actual_tile = 32;
    } else if (g_use_tcgen05 && g_use_signed) {
        /* tcgen05.mma kernel (Blackwell SM 10.0 only - NOT SM 12.0!) */
        if (specs.major == 12) {
            fprintf(stderr, "ERROR: tcgen05 instructions are NOT supported on SM 12.0 (RTX 50 series)\n");
            fprintf(stderr, "       tcgen05 is only available on SM 10.0/10.1 (B100/B200 data center GPUs)\n");
            fprintf(stderr, "       Using default mma.sync kernel instead.\n\n");
            ptx_code = ptx_gemm_tc64_s8;
            kernel_name = "gemm_int8_tc64_s8";
            block_size = 512;
        } else if (specs.major == 10) {
            ptx_code = ptx_gemm_tcgen05_s8;
            kernel_name = "gemm_int8_tcgen05_s8";
            block_size = 128;
            actual_tile = 64;
            printf("Using tcgen05.mma kernel (Blackwell Tensor Core Gen 5)\n");
        } else {
            fprintf(stderr, "ERROR: tcgen05 requires Blackwell SM 10.0+ architecture\n");
            return 1;
        }
    } else if (g_use_opt3 && g_use_signed) {
        /* Optimized v3 kernel with K=64 unrolling */
        ptx_code = ptx_gemm_tc64_opt3_s8;
        kernel_name = "gemm_int8_tc64_opt3_s8";
        block_size = 512;
    } else if (g_use_opt2 && g_use_signed) {
        /* Optimized v2 kernel with transposed B storage */
        ptx_code = ptx_gemm_tc64_opt2_s8;
        kernel_name = "gemm_int8_tc64_opt2_s8";
        block_size = 512;
    } else if (g_use_opt && g_use_signed) {
        /* Optimized kernel with double buffering (signed only for now) */
        ptx_code = ptx_gemm_tc64_opt_s8;
        kernel_name = "gemm_int8_tc64_opt_s8";
        block_size = 512;
    } else {
        ptx_code = g_use_signed ? ptx_gemm_tc64_s8 : ptx_gemm_tc64_u8;
        kernel_name = g_use_signed ? "gemm_int8_tc64_s8" : "gemm_int8_tc64_u8";
        block_size = 512;
    }

    /* Load PTX module */
    CUmodule module;
    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER };
    char error_log[4096] = {0};
    void* option_values[] = { (void*)(size_t)sizeof(error_log), error_log };

    CUresult res = cuModuleLoadDataEx(&module, ptx_code, 2, options, option_values);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "PTX load failed: %s\n", error_log);
        return 1;
    }

    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, kernel_name));

    /* Allocate host memory */
    size_t size_A = (size_t)g_M * g_K;
    size_t size_B = (size_t)g_K * g_N;
    size_t size_C = (size_t)g_M * g_N * sizeof(int32_t);

    void* h_A = malloc(size_A);
    void* h_B = malloc(size_B);
    int32_t* h_C = (int32_t*)malloc(size_C);
    int32_t* h_C_ref = (int32_t*)malloc(size_C);

    /* Initialize input data */
    srand(42);
    static int g_debug = 0; /* Debug mode: 0=random, 1=B-varies-k, 2=sequential, 3=A-varies-row */
    if (g_use_signed) {
        int8_t* A = (int8_t*)h_A;
        int8_t* B = (int8_t*)h_B;
        if (g_debug == 1) {
            /* Simple pattern: A[i,k]=1, B[k,j]=k+1 -> C[i,j]=sum(k+1)=K*(K+1)/2 */
            for (int i = 0; i < g_M; i++)
                for (int k = 0; k < g_K; k++)
                    A[i * g_K + k] = 1;
            for (int k = 0; k < g_K; k++)
                for (int j = 0; j < g_N; j++)
                    B[k * g_N + j] = (int8_t)(k % 16 + 1); /* Use k%16+1 to stay in small range */
        } else if (g_debug == 2) {
            /* Sequential values */
            for (size_t i = 0; i < size_A; i++) A[i] = (int8_t)(i % 10);
            for (size_t i = 0; i < size_B; i++) B[i] = (int8_t)(i % 10);
        } else if (g_debug == 3) {
            /* Row pattern: A[i,k]=i+1, B=1 -> C[i,j] = (i+1)*K */
            for (int i = 0; i < g_M; i++)
                for (int k = 0; k < g_K; k++)
                    A[i * g_K + k] = (int8_t)(i % 16 + 1);
            for (size_t i = 0; i < size_B; i++) B[i] = 1;
        } else {
            for (size_t i = 0; i < size_A; i++) A[i] = random_s8_range(10);
            for (size_t i = 0; i < size_B; i++) B[i] = random_s8_range(10);
        }
        if (!g_skip_verify) {
            gemm_reference_s8(A, B, h_C_ref, g_M, g_N, g_K);
        }
        if (g_debug && g_M <= 32 && g_N <= 32) {
            printf("\nDebug: A[0,0..7] = ");
            for (int j = 0; j < 8 && j < g_K; j++) printf("%d ", A[j]);
            printf("\nDebug: B[0..7,0] = ");
            for (int i = 0; i < 8 && i < g_K; i++) printf("%d ", B[i * g_N]);
            printf("\n");
        }
    } else {
        uint8_t* A = (uint8_t*)h_A;
        uint8_t* B = (uint8_t*)h_B;
        for (size_t i = 0; i < size_A; i++) A[i] = random_u8_range(20);
        for (size_t i = 0; i < size_B; i++) B[i] = random_u8_range(20);
        if (!g_skip_verify) {
            gemm_reference_u8(A, B, h_C_ref, g_M, g_N, g_K);
        }
    }

    /* Allocate device memory */
    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size_A));
    CHECK_CUDA(cuMemAlloc(&d_B, size_B));
    CHECK_CUDA(cuMemAlloc(&d_C, size_C));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size_A));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size_B));
    CHECK_CUDA(cuMemsetD8(d_C, 0, size_C));

    /* Setup kernel launch parameters */
    unsigned int M = g_M, N = g_N, K = g_K;
    void* args[] = { &d_A, &d_B, &d_C, &M, &N, &K };

    int tile = g_use_tensor_core ? actual_tile : 16;
    int grid_x = (g_N + tile - 1) / tile;
    int grid_y = (g_M + tile - 1) / tile;

    int block_x = g_use_tensor_core ? block_size : 16;
    int block_y = g_use_tensor_core ? 1 : 16;

    /* Create timing events */
    CUevent start, stop;
    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    /* Warmup */
    for (int i = 0; i < g_warmup_iters; i++) {
        CHECK_CUDA(cuLaunchKernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, args, NULL));
    }
    CHECK_CUDA(cuCtxSynchronize());

    /* Benchmark */
    CHECK_CUDA(cuEventRecord(start, 0));
    for (int i = 0; i < g_bench_iters; i++) {
        CHECK_CUDA(cuLaunchKernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, args, NULL));
    }
    CHECK_CUDA(cuEventRecord(stop, 0));
    CHECK_CUDA(cuEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / g_bench_iters;

    /* Copy results back */
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size_C));

    /* Verify */
    if (!g_skip_verify) {
        verify_results(h_C, h_C_ref, g_M, g_N);
    } else {
        printf("Verification SKIPPED (--no-verify)\n");
    }

    /* Stochastic rounding: int32 -> fp32 -> SR -> int8 */
    if (g_use_sr) {
        printf("\n=== Stochastic Rounding ===\n");

        /* Initialize xoroshiro128+ RNG */
        xoroshiro128plus_t rng;
        xoro_seed(&rng, g_sr_seed);

        /* Compute scale if not specified */
        float scale = g_sr_scale;
        if (scale <= 0.0f) {
            scale = g_use_signed ? compute_scale_s8(h_C, (size_t)g_M * g_N)
                                 : compute_scale_u8(h_C, (size_t)g_M * g_N);
            printf("Auto-computed scale: %.6e\n", scale);
        } else {
            printf("Using manual scale: %.6e\n", scale);
        }

        /* Allocate and perform stochastic rounding */
        void* h_C_sr = malloc((size_t)g_M * g_N);
        if (g_use_signed) {
            batch_sr_s8(h_C, (int8_t*)h_C_sr, (size_t)g_M * g_N, scale, &rng);
        } else {
            batch_sr_u8(h_C, (uint8_t*)h_C_sr, (size_t)g_M * g_N, scale, &rng);
        }

        /* Print sample of SR output */
        printf("Sample SR output (first 8 values):\n");
        printf("  int32 -> int8: ");
        for (int i = 0; i < 8 && i < g_M * g_N; i++) {
            if (g_use_signed) {
                printf("%d->%d ", h_C[i], ((int8_t*)h_C_sr)[i]);
            } else {
                printf("%d->%u ", h_C[i], ((uint8_t*)h_C_sr)[i]);
            }
        }
        printf("\n");

        /* Compute reconstruction error statistics */
        double mse = 0.0;
        double max_err = 0.0;
        for (int i = 0; i < g_M * g_N; i++) {
            float original = (float)h_C[i] * scale;
            float reconstructed;
            if (g_use_signed) {
                reconstructed = (float)((int8_t*)h_C_sr)[i];
            } else {
                reconstructed = (float)((uint8_t*)h_C_sr)[i];
            }
            double err = fabs(original - reconstructed);
            mse += err * err;
            if (err > max_err) max_err = err;
        }
        mse /= (g_M * g_N);
        printf("Quantization MSE: %.6f, Max error: %.2f\n", mse, max_err);

        free(h_C_sr);
    }

    /* Report performance */
    double ops = 2.0 * g_M * g_N * g_K;
    double tops = ops / (avg_ms * 1e9);
    double efficiency = specs.int8_tensor_peak_tops > 0 ?
                        100.0 * tops / specs.int8_tensor_peak_tops : 0;

    printf("\n=== Performance ===\n");
    printf("Time: %.3f ms (avg of %d runs)\n", avg_ms, g_bench_iters);
    printf("Throughput: %.2f TOPS\n", tops);
    if (specs.int8_tensor_peak_tops > 0) {
        printf("Efficiency: %.1f%% of peak (%.2f TOPS)\n",
               efficiency, specs.int8_tensor_peak_tops);
    }

    /* Cleanup */
    CHECK_CUDA(cuEventDestroy(start));
    CHECK_CUDA(cuEventDestroy(stop));
    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
