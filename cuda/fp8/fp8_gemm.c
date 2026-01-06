/*
 * FP8 GEMM using CUDA with cuew - Tensor Core Version
 *
 * Computes: C[M,N] = A[M,K] * B[K,N]
 * - FP8 inputs (E4M3 or E5M2)
 * - FP32 accumulator
 * - Uses Tensor Core mma.sync instructions (SM 8.9+)
 *
 * Uses PTX kernel loaded via cuModuleLoadData (no CUDA SDK required)
 * Includes benchmark mode with peak FLOPS analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../cuew.h"
#include "fp8_types.h"

/* Default matrix dimensions - must be multiples of tile sizes */
static int g_M = 128;
static int g_N = 128;
static int g_K = 128;

/* Benchmark settings */
static int g_warmup_iters = 5;
static int g_bench_iters = 100;
static int g_benchmark_mode = 0;
static int g_use_tensor_core = 1;  /* Use tensor cores by default */
static int g_tile_size = 64;      /* Tile size: 32, 64, or 128 */

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

/* GPU specifications for peak FLOPS calculation */
typedef struct {
    int sm_count;
    int clock_mhz;
    int major;
    int minor;
    size_t mem_bandwidth_gbps;
    double fp32_peak_tflops;
    double fp8_tensor_peak_tflops;
} gpu_specs_t;

static int get_fp8_tensor_ops_per_sm_per_cycle(int major, int minor) {
    int sm = major * 10 + minor;
    if (sm >= 120) return 4096;      /* Blackwell */
    else if (sm >= 90) return 2048;  /* Hopper */
    else if (sm >= 89) return 1024;  /* Ada Lovelace */
    else return 0;
}

static int get_fp32_ops_per_sm_per_cycle(int major, int minor) {
    int sm = major * 10 + minor;
    if (sm >= 90) return 256;
    else return 128;
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

    int fp32_ops = get_fp32_ops_per_sm_per_cycle(specs->major, specs->minor);
    int fp8_ops = get_fp8_tensor_ops_per_sm_per_cycle(specs->major, specs->minor);
    specs->fp32_peak_tflops = (double)specs->sm_count * fp32_ops * specs->clock_mhz / 1e6;
    specs->fp8_tensor_peak_tflops = (double)specs->sm_count * fp8_ops * specs->clock_mhz / 1e6;
}

static void print_gpu_specs(const char* device_name, const gpu_specs_t* specs) {
    printf("\n=== GPU Specifications ===\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: SM %d.%d\n", specs->major, specs->minor);
    printf("SM Count: %d\n", specs->sm_count);
    printf("GPU Clock: %d MHz\n", specs->clock_mhz);
    printf("Memory Bandwidth: %zu GB/s\n", specs->mem_bandwidth_gbps);
    printf("\nTheoretical Peak Performance:\n");
    printf("  FP32 (CUDA cores): %.2f TFLOPS\n", specs->fp32_peak_tflops);
    if (specs->fp8_tensor_peak_tflops > 0) {
        printf("  FP8 (Tensor cores): %.2f TFLOPS\n", specs->fp8_tensor_peak_tflops);
    }
    printf("\n");
}

/*
 * Tensor Core PTX Kernel for FP8 GEMM (E4M3)
 * Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 * Block: 128 threads (4 warps), 32x32 output per block
 * Tiles: A=32x32 (row-major), B=32x32 (transposed/col-major in smem)
 */
static const char* ptx_gemm_tc_e4m3 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_tc_e4m3(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // smem_A: 32 rows x 32 k (row-major)\n"
"    // smem_B: 32 cols x 32 k (col-major, transposed for aligned MMA access)\n"
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
"    .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .f32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
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
"    mov.f32 c0, 0.0; mov.f32 c1, 0.0; mov.f32 c2, 0.0; mov.f32 c3, 0.0;\n"
"    mov.f32 c4, 0.0; mov.f32 c5, 0.0; mov.f32 c6, 0.0; mov.f32 c7, 0.0;\n"
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
"    // Load A from shared memory for mma - correct fragment layout\n"
"    // Thread T: row0 = (T/4)*2, row1 = row0+1, k_base = (T%4)*4\n"
"    // a0 = A[row0][k_base:k_base+3], a1 = A[row0][k_base+16:k_base+19]\n"
"    // a2 = A[row1][k_base:k_base+3], a3 = A[row1][k_base+16:k_base+19]\n"
"    shr.u32 a_row0, lane_id, 2;       // T / 4 (0-7)\n"
"    shl.b32 a_row0, a_row0, 1;        // * 2 -> 0,2,4,6,8,10,12,14\n"
"    add.u32 a_row0, a_row0, warp_row; // + warp's row offset\n"
"    add.u32 a_row1, a_row0, 1;        // row0 + 1\n"
"    and.b32 a_k, lane_id, 3;          // T % 4 (0-3)\n"
"    shl.b32 a_k, a_k, 2;              // * 4 -> 0,4,8,12\n"
"    // Load a0: A[row0][k_base:k_base+3] - 4 consecutive bytes\n"
"    shl.b32 smem_off, a_row0, 5;      // row0 * 32\n"
"    add.u32 smem_off, smem_off, a_k;  // + k_base\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    // Load a1: A[row0][k_base+16:k_base+19]\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    // Load a2: A[row1][k_base:k_base+3]\n"
"    add.u32 saddr, saddr, 32;         // + row stride\n"
"    ld.shared.b32 a2, [saddr];\n"
"    // Load a3: A[row1][k_base+16:k_base+19]\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.f32 c0, d0; mov.f32 c1, d1; mov.f32 c2, d2; mov.f32 c3, d3;\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.f32 c4, d4; mov.f32 c5, d5; mov.f32 c6, d6; mov.f32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    // Store using mma output layout: row0=(lane/4)*2, col=(lane%4)*2\n"
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
"    st.global.f32 [out_addr + 0], c0;\n"
"    st.global.f32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c2;\n"
"    st.global.f32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c4;\n"
"    st.global.f32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c6;\n"
"    st.global.f32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 64x64 Tile Tensor Core PTX Kernel for FP8 GEMM (E4M3)
 * Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 * Block: 512 threads (16 warps), 64x64 output per block
 * Warps arranged in 4x4 grid, each warp produces 16x16
 * Tiles: A=64x32 (row-major), B=32x64 (row-major)
 */
static const char* ptx_gemm_tc64_e4m3 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_tc64_e4m3(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    // smem_A: 64 rows x 32 k = 2048 bytes\n"
"    // smem_B: 32 k x 64 cols = 2048 bytes\n"
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
"    .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .f32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
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
"    shr.u32 warp_id, tid, 5;       // tid / 32\n"
"    and.b32 lane_id, tid, 31;      // tid % 32\n"
"\n"
"    mov.u32 block_row, %ctaid.y;\n"
"    mov.u32 block_col, %ctaid.x;\n"
"    shl.b32 block_row, block_row, 6;  // * 64\n"
"    shl.b32 block_col, block_col, 6;  // * 64\n"
"\n"
"    // 16 warps in 4x4 grid, each producing 16x16\n"
"    and.b32 warp_row, warp_id, 3;     // warp_id % 4\n"
"    shr.u32 warp_col, warp_id, 2;     // warp_id / 4\n"
"    shl.b32 warp_row, warp_row, 4;    // * 16\n"
"    shl.b32 warp_col, warp_col, 4;    // * 16\n"
"\n"
"    mov.f32 c0, 0.0; mov.f32 c1, 0.0; mov.f32 c2, 0.0; mov.f32 c3, 0.0;\n"
"    mov.f32 c4, 0.0; mov.f32 c5, 0.0; mov.f32 c6, 0.0; mov.f32 c7, 0.0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    // Load A tile (64x32): 512 threads, 4 bytes each = 2048 bytes\n"
"    // Each thread loads 1 word (4 bytes)\n"
"    shr.u32 load_row, tid, 3;         // tid / 8 = row (0-63)\n"
"    and.b32 load_col, tid, 7;         // tid % 8 = col group\n"
"    shl.b32 load_col, load_col, 2;    // * 4 = 0,4,8,12,16,20,24,28\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.b32 a0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;    // row * 32\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], a0;\n"
"\n"
"    // Load B tile (32x64): 512 threads, 4 bytes each = 2048 bytes\n"
"    shr.u32 load_row, tid, 4;         // tid / 16 = row (0-31)\n"
"    and.b32 load_col, tid, 15;        // tid % 16 = col group\n"
"    shl.b32 load_col, load_col, 2;    // * 4 = 0,4,8,...,60\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.b32 b0, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 6;    // row * 64\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.b32 [saddr], b0;\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // Load A fragment: row0=(lane/4)*2, row1=row0+1, k=(lane%4)*4\n"
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
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    // Load B fragment using byte loads (B stored row-major, 64-col stride)\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 6;         // k_base * 64\n"
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
"    add.u32 saddr, saddr, 1024;       // +16 rows * 64\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.f32 c0, d0; mov.f32 c1, d1; mov.f32 c2, d2; mov.f32 c3, d3;\n"
"\n"
"    // Second MMA for columns 8-15\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.f32 c4, d4; mov.f32 c5, d5; mov.f32 c6, d6; mov.f32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    // Store: row0=(lane/4)*2, col=(lane%4)*2\n"
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
"    st.global.f32 [out_addr + 0], c0;\n"
"    st.global.f32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c2;\n"
"    st.global.f32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c4;\n"
"    st.global.f32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c6;\n"
"    st.global.f32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 64x64 Tile Tensor Core PTX Kernel for FP8 GEMM (E5M2)
 */
static const char* ptx_gemm_tc64_e5m2 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_tc64_e5m2(\n"
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
"    .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .f32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
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
"    mov.f32 c0, 0.0; mov.f32 c1, 0.0; mov.f32 c2, 0.0; mov.f32 c3, 0.0;\n"
"    mov.f32 c4, 0.0; mov.f32 c5, 0.0; mov.f32 c6, 0.0; mov.f32 c7, 0.0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    // Load A tile (64x32)\n"
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
"    // Load B tile (32x64)\n"
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
"    // Load A fragment\n"
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
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    // Load B fragment\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.f32 c0, d0; mov.f32 c1, d1; mov.f32 c2, d2; mov.f32 c3, d3;\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.f32 c4, d4; mov.f32 c5, d5; mov.f32 c6, d6; mov.f32 c7, d7;\n"
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
"    st.global.f32 [out_addr + 0], c0;\n"
"    st.global.f32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c2;\n"
"    st.global.f32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c4;\n"
"    st.global.f32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c6;\n"
"    st.global.f32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 128x128 Tile Tensor Core PTX Kernel for FP8 GEMM (E4M3)
 * Block: 512 threads (16 warps), 128x128 output per block
 * Each warp produces 32x32 via 4 MMAs (2x2 grid of 16x16)
 * Tiles: A=128x32, B=32x128 (8KB total shared memory)
 */
static const char* ptx_gemm_tc128_e4m3 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_tc128_e4m3(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .shared .align 16 .b8 smem_A[4096];\n"  // 128x32
"    .shared .align 16 .b8 smem_B[4096];\n"  // 32x128
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1;\n"
"    .reg .u32 b_col, b_k, mma_row, mma_col;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    // 16 accumulators for 4 MMAs (2x2 grid of 16x16)\n"
"    .reg .f32 c0, c1, c2, c3, c4, c5, c6, c7;\n"
"    .reg .f32 c8, c9, c10, c11, c12, c13, c14, c15;\n"
"    .reg .f32 d0, d1, d2, d3;\n"
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
"    shl.b32 block_row, block_row, 7;  // * 128\n"
"    shl.b32 block_col, block_col, 7;  // * 128\n"
"\n"
"    // 16 warps in 4x4 grid, each producing 32x32\n"
"    and.b32 warp_row, warp_id, 3;     // warp_id % 4\n"
"    shr.u32 warp_col, warp_id, 2;     // warp_id / 4\n"
"    shl.b32 warp_row, warp_row, 5;    // * 32\n"
"    shl.b32 warp_col, warp_col, 5;    // * 32\n"
"\n"
"    // Zero 16 accumulators\n"
"    mov.f32 c0, 0.0; mov.f32 c1, 0.0; mov.f32 c2, 0.0; mov.f32 c3, 0.0;\n"
"    mov.f32 c4, 0.0; mov.f32 c5, 0.0; mov.f32 c6, 0.0; mov.f32 c7, 0.0;\n"
"    mov.f32 c8, 0.0; mov.f32 c9, 0.0; mov.f32 c10, 0.0; mov.f32 c11, 0.0;\n"
"    mov.f32 c12, 0.0; mov.f32 c13, 0.0; mov.f32 c14, 0.0; mov.f32 c15, 0.0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    // Load A tile (128x32): 512 threads, 8 bytes each\n"
"    shr.u32 load_row, tid, 2;         // tid / 4 = row (0-127)\n"
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
"    // Load B tile (32x128): 512 threads, 8 bytes each\n"
"    shr.u32 load_row, tid, 4;         // tid / 16 = row (0-31)\n"
"    and.b32 load_col, tid, 15;        // tid % 16 = col group\n"
"    shl.b32 load_col, load_col, 3;    // * 8 = 0,8,...,120\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.v2.b32 {b0, b1}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 7;    // row * 128\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {b0, b1};\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // MMA 0,0: rows 0-15, cols 0-7 of warp's 32x32\n"
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
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    // B fragment for cols 0-7\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.f32 c0, d0; mov.f32 c1, d1; mov.f32 c2, d2; mov.f32 c3, d3;\n"
"\n"
"    // MMA 0,1: rows 0-15, cols 8-15\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.f32 c4, d0; mov.f32 c5, d1; mov.f32 c6, d2; mov.f32 c7, d3;\n"
"\n"
"    // MMA 1,0: rows 16-31, cols 0-7\n"
"    add.u32 a_row0, a_row0, 16;\n"
"    add.u32 a_row1, a_row0, 1;\n"
"    shl.b32 smem_off, a_row0, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    sub.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c8, c9, c10, c11};\n"
"    mov.f32 c8, d0; mov.f32 c9, d1; mov.f32 c10, d2; mov.f32 c11, d3;\n"
"\n"
"    // MMA 1,1: rows 16-31, cols 8-15\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c12, c13, c14, c15};\n"
"    mov.f32 c12, d0; mov.f32 c13, d1; mov.f32 c14, d2; mov.f32 c15, d3;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    // Store 4 16x16 results in 2x2 pattern\n"
"    shr.u32 out_row0, lane_id, 2;\n"
"    shl.b32 out_row0, out_row0, 1;\n"
"    add.u32 out_row1, out_row0, 1;\n"
"    and.b32 out_col, lane_id, 3;\n"
"    shl.b32 out_col, out_col, 1;\n"
"    shl.b32 stride32, dim_n, 2;\n"
"    cvt.u64.u32 stride64, stride32;\n"
"\n"
"    // MMA 0,0 result\n"
"    add.u32 tmp, out_row0, block_row;\n"
"    add.u32 tmp, tmp, warp_row;\n"
"    add.u32 tmp2, out_col, block_col;\n"
"    add.u32 tmp2, tmp2, warp_col;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c0;\n"
"    st.global.f32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c2;\n"
"    st.global.f32 [out_addr + 4], c3;\n"
"\n"
"    // MMA 0,1 result (+8 cols)\n"
"    add.u32 tmp2, tmp2, 8;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c4;\n"
"    st.global.f32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c6;\n"
"    st.global.f32 [out_addr + 4], c7;\n"
"\n"
"    // MMA 1,0 result (+16 rows)\n"
"    add.u32 tmp, tmp, 16;\n"
"    sub.u32 tmp2, tmp2, 8;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c8;\n"
"    st.global.f32 [out_addr + 4], c9;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c10;\n"
"    st.global.f32 [out_addr + 4], c11;\n"
"\n"
"    // MMA 1,1 result (+16 rows, +8 cols)\n"
"    add.u32 tmp2, tmp2, 8;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c12;\n"
"    st.global.f32 [out_addr + 4], c13;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c14;\n"
"    st.global.f32 [out_addr + 4], c15;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * 128x128 Tile Tensor Core PTX Kernel for FP8 GEMM (E5M2)
 */
static const char* ptx_gemm_tc128_e5m2 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_tc128_e5m2(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
"    .shared .align 16 .b8 smem_A[4096];\n"
"    .shared .align 16 .b8 smem_B[4096];\n"
"\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
"    .reg .u32 dim_m, dim_n, dim_k;\n"
"    .reg .u32 tid, warp_id, lane_id;\n"
"    .reg .u32 block_row, block_col, warp_row, warp_col;\n"
"    .reg .u32 k_outer, offset32, tmp, tmp2;\n"
"    .reg .u64 offset64, stride64;\n"
"    .reg .u32 load_row, load_col, smem_off, saddr;\n"
"    .reg .u32 a0, a1, a2, a3, b0, b1;\n"
"    .reg .u32 b_col, b_k, mma_row, mma_col;\n"
"    .reg .u32 byte0, byte1, byte2, byte3;\n"
"    .reg .u32 a_row0, a_row1, a_k;\n"
"    .reg .f32 c0, c1, c2, c3, c4, c5, c6, c7;\n"
"    .reg .f32 c8, c9, c10, c11, c12, c13, c14, c15;\n"
"    .reg .f32 d0, d1, d2, d3;\n"
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
"    shl.b32 block_row, block_row, 7;\n"
"    shl.b32 block_col, block_col, 7;\n"
"\n"
"    and.b32 warp_row, warp_id, 3;\n"
"    shr.u32 warp_col, warp_id, 2;\n"
"    shl.b32 warp_row, warp_row, 5;\n"
"    shl.b32 warp_col, warp_col, 5;\n"
"\n"
"    mov.f32 c0, 0.0; mov.f32 c1, 0.0; mov.f32 c2, 0.0; mov.f32 c3, 0.0;\n"
"    mov.f32 c4, 0.0; mov.f32 c5, 0.0; mov.f32 c6, 0.0; mov.f32 c7, 0.0;\n"
"    mov.f32 c8, 0.0; mov.f32 c9, 0.0; mov.f32 c10, 0.0; mov.f32 c11, 0.0;\n"
"    mov.f32 c12, 0.0; mov.f32 c13, 0.0; mov.f32 c14, 0.0; mov.f32 c15, 0.0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    shr.u32 load_row, tid, 2;\n"
"    and.b32 load_col, tid, 3;\n"
"    shl.b32 load_col, load_col, 3;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.v2.b32 {a0, a1}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {a0, a1};\n"
"\n"
"    shr.u32 load_row, tid, 4;\n"
"    and.b32 load_col, tid, 15;\n"
"    shl.b32 load_col, load_col, 3;\n"
"    add.u32 tmp, k_outer, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 tmp2, block_col, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_B, offset64;\n"
"    ld.global.v2.b32 {b0, b1}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 7;\n"
"    add.u32 smem_off, smem_off, load_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    st.shared.v2.b32 [saddr], {b0, b1};\n"
"\n"
"    bar.sync 0;\n"
"\n"
"    // MMA 0,0\n"
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
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.f32 c0, d0; mov.f32 c1, d1; mov.f32 c2, d2; mov.f32 c3, d3;\n"
"\n"
"    // MMA 0,1\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.f32 c4, d0; mov.f32 c5, d1; mov.f32 c6, d2; mov.f32 c7, d3;\n"
"\n"
"    // MMA 1,0\n"
"    add.u32 a_row0, a_row0, 16;\n"
"    add.u32 a_row1, a_row0, 1;\n"
"    shl.b32 smem_off, a_row0, 5;\n"
"    add.u32 smem_off, smem_off, a_k;\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    add.u32 saddr, saddr, 32;\n"
"    ld.shared.b32 a2, [saddr];\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    sub.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c8, c9, c10, c11};\n"
"    mov.f32 c8, d0; mov.f32 c9, d1; mov.f32 c10, d2; mov.f32 c11, d3;\n"
"\n"
"    // MMA 1,1\n"
"    add.u32 b_col, b_col, 8;\n"
"    shl.b32 smem_off, b_k, 7;\n"
"    add.u32 smem_off, smem_off, b_col;\n"
"    mov.u32 saddr, smem_B;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b0, byte0, byte1;\n"
"    or.b32 b0, b0, byte2;\n"
"    or.b32 b0, b0, byte3;\n"
"    add.u32 saddr, saddr, 2048;\n"
"    ld.shared.u8 byte0, [saddr];\n"
"    ld.shared.u8 byte1, [saddr + 128];\n"
"    ld.shared.u8 byte2, [saddr + 256];\n"
"    ld.shared.u8 byte3, [saddr + 384];\n"
"    shl.b32 byte1, byte1, 8;\n"
"    shl.b32 byte2, byte2, 16;\n"
"    shl.b32 byte3, byte3, 24;\n"
"    or.b32 b1, byte0, byte1;\n"
"    or.b32 b1, b1, byte2;\n"
"    or.b32 b1, b1, byte3;\n"
"\n"
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c12, c13, c14, c15};\n"
"    mov.f32 c12, d0; mov.f32 c13, d1; mov.f32 c14, d2; mov.f32 c15, d3;\n"
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
"    shl.b32 stride32, dim_n, 2;\n"
"    cvt.u64.u32 stride64, stride32;\n"
"\n"
"    add.u32 tmp, out_row0, block_row;\n"
"    add.u32 tmp, tmp, warp_row;\n"
"    add.u32 tmp2, out_col, block_col;\n"
"    add.u32 tmp2, tmp2, warp_col;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c0;\n"
"    st.global.f32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c2;\n"
"    st.global.f32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 tmp2, tmp2, 8;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c4;\n"
"    st.global.f32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c6;\n"
"    st.global.f32 [out_addr + 4], c7;\n"
"\n"
"    add.u32 tmp, tmp, 16;\n"
"    sub.u32 tmp2, tmp2, 8;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c8;\n"
"    st.global.f32 [out_addr + 4], c9;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c10;\n"
"    st.global.f32 [out_addr + 4], c11;\n"
"\n"
"    add.u32 tmp2, tmp2, 8;\n"
"    mul.lo.u32 offset32, tmp, dim_n;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c12;\n"
"    st.global.f32 [out_addr + 4], c13;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c14;\n"
"    st.global.f32 [out_addr + 4], c15;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * Tensor Core PTX Kernel for FP8 GEMM (E5M2)
 * Same structure as E4M3, uses e5m2 type in mma instruction
 * Block: 128 threads (4 warps), 32x32 output per block
 * Tiles: A=32x32 (row-major), B=32x32 (row-major, read with byte loads)
 */
static const char* ptx_gemm_tc_e5m2 =
".version 8.4\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_tc_e5m2(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
")\n"
"{\n"
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
"    .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
"    .reg .f32 c4, c5, c6, c7, d4, d5, d6, d7;\n"
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
"    mov.f32 c0, 0.0; mov.f32 c1, 0.0; mov.f32 c2, 0.0; mov.f32 c3, 0.0;\n"
"    mov.f32 c4, 0.0; mov.f32 c5, 0.0; mov.f32 c6, 0.0; mov.f32 c7, 0.0;\n"
"\n"
"    mov.u32 k_outer, 0;\n"
"K_LOOP:\n"
"    setp.lt.u32 p_k, k_outer, dim_k;\n"
"    @!p_k bra K_DONE;\n"
"\n"
"    // Load A tile (32x32) row-major\n"
"    shr.u32 load_row, tid, 2;\n"
"    and.b32 load_col, tid, 3;\n"
"    shl.b32 load_col, load_col, 3;\n"
"    add.u32 tmp, block_row, load_row;\n"
"    mul.lo.u32 offset32, tmp, dim_k;\n"
"    add.u32 tmp2, k_outer, load_col;\n"
"    add.u32 offset32, offset32, tmp2;\n"
"    cvt.u64.u32 offset64, offset32;\n"
"    add.u64 gmem_addr, ptr_A, offset64;\n"
"    ld.global.v2.b32 {a0, a1}, [gmem_addr];\n"
"    shl.b32 smem_off, load_row, 5;\n"
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
"    // Load A from shared memory for mma - correct fragment layout\n"
"    // Thread T: row0 = (T/4)*2, row1 = row0+1, k_base = (T%4)*4\n"
"    // a0 = A[row0][k_base:k_base+3], a1 = A[row0][k_base+16:k_base+19]\n"
"    // a2 = A[row1][k_base:k_base+3], a3 = A[row1][k_base+16:k_base+19]\n"
"    shr.u32 a_row0, lane_id, 2;       // T / 4 (0-7)\n"
"    shl.b32 a_row0, a_row0, 1;        // * 2 -> 0,2,4,6,8,10,12,14\n"
"    add.u32 a_row0, a_row0, warp_row; // + warp's row offset\n"
"    add.u32 a_row1, a_row0, 1;        // row0 + 1\n"
"    and.b32 a_k, lane_id, 3;          // T % 4 (0-3)\n"
"    shl.b32 a_k, a_k, 2;              // * 4 -> 0,4,8,12\n"
"    // Load a0: A[row0][k_base:k_base+3] - 4 consecutive bytes\n"
"    shl.b32 smem_off, a_row0, 5;      // row0 * 32\n"
"    add.u32 smem_off, smem_off, a_k;  // + k_base\n"
"    mov.u32 saddr, smem_A;\n"
"    add.u32 saddr, saddr, smem_off;\n"
"    ld.shared.b32 a0, [saddr];\n"
"    // Load a1: A[row0][k_base+16:k_base+19]\n"
"    ld.shared.b32 a1, [saddr + 16];\n"
"    // Load a2: A[row1][k_base:k_base+3]\n"
"    add.u32 saddr, saddr, 32;         // + row stride\n"
"    ld.shared.b32 a2, [saddr];\n"
"    // Load a3: A[row1][k_base+16:k_base+19]\n"
"    ld.shared.b32 a3, [saddr + 16];\n"
"\n"
"    // Load B using byte loads\n"
"    and.b32 tmp, lane_id, 3;\n"
"    shl.b32 b_k, tmp, 2;\n"
"    shr.u32 b_col, lane_id, 2;\n"
"    add.u32 b_col, b_col, warp_col;\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};\n"
"    mov.f32 c0, d0; mov.f32 c1, d1; mov.f32 c2, d2; mov.f32 c3, d3;\n"
"\n"
"    // Load B for second mma (columns 8-15)\n"
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
"    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
"        {d4, d5, d6, d7}, {a0, a1, a2, a3}, {b0, b1}, {c4, c5, c6, c7};\n"
"    mov.f32 c4, d4; mov.f32 c5, d5; mov.f32 c6, d6; mov.f32 c7, d7;\n"
"\n"
"    bar.sync 0;\n"
"    add.u32 k_outer, k_outer, 32;\n"
"    bra K_LOOP;\n"
"\n"
"K_DONE:\n"
"    // Store using mma output layout\n"
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
"    st.global.f32 [out_addr + 0], c0;\n"
"    st.global.f32 [out_addr + 4], c1;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c2;\n"
"    st.global.f32 [out_addr + 4], c3;\n"
"\n"
"    add.u32 out_col, out_col, 8;\n"
"    mul.lo.u32 offset32, out_row0, dim_n;\n"
"    add.u32 offset32, offset32, out_col;\n"
"    shl.b32 offset32, offset32, 2;\n"
"    cvt.u64.u32 out_addr, offset32;\n"
"    add.u64 out_addr, ptr_C, out_addr;\n"
"    st.global.f32 [out_addr + 0], c4;\n"
"    st.global.f32 [out_addr + 4], c5;\n"
"    add.u64 out_addr, out_addr, stride64;\n"
"    st.global.f32 [out_addr + 0], c6;\n"
"    st.global.f32 [out_addr + 4], c7;\n"
"\n"
"    ret;\n"
"}\n";

/*
 * Naive scalar kernel (fallback for older GPUs or debugging)
 */
static const char* ptx_gemm_naive_e4m3 =
".version 8.0\n"
".target sm_80\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_naive_e4m3(\n"
"    .param .u64 param_A, .param .u64 param_B, .param .u64 param_C,\n"
"    .param .u32 param_M, .param .u32 param_N, .param .u32 param_K)\n"
"{\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, addr, offset64;\n"
"    .reg .u32 row, col, kk, dim_m, dim_n, dim_k, a_byte, b_byte, offset32;\n"
"    .reg .u32 a_sign, a_exp, a_mant, a_bits, b_sign, b_exp, b_mant, b_bits;\n"
"    .reg .u32 ctaid_x, ctaid_y, ntid_x, ntid_y, tid_x, tid_y;\n"
"    .reg .f32 acc, a_val, b_val;\n"
"    .reg .pred p_loop, p_a_zero, p_b_zero;\n"
"    ld.param.u64 ptr_A, [param_A]; ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C]; ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N]; ld.param.u32 dim_k, [param_K];\n"
"    mov.u32 ctaid_x, %ctaid.x; mov.u32 ctaid_y, %ctaid.y;\n"
"    mov.u32 ntid_x, %ntid.x; mov.u32 ntid_y, %ntid.y;\n"
"    mov.u32 tid_x, %tid.x; mov.u32 tid_y, %tid.y;\n"
"    mul.lo.u32 row, ctaid_y, ntid_y; add.u32 row, row, tid_y;\n"
"    mul.lo.u32 col, ctaid_x, ntid_x; add.u32 col, col, tid_x;\n"
"    mov.f32 acc, 0.0; mov.u32 kk, 0;\n"
"LOOP:\n"
"    setp.lt.u32 p_loop, kk, dim_k; @!p_loop bra DONE;\n"
"    mul.lo.u32 offset32, row, dim_k; add.u32 offset32, offset32, kk;\n"
"    cvt.u64.u32 offset64, offset32; add.u64 addr, ptr_A, offset64;\n"
"    ld.global.u8 a_byte, [addr];\n"
"    mul.lo.u32 offset32, kk, dim_n; add.u32 offset32, offset32, col;\n"
"    cvt.u64.u32 offset64, offset32; add.u64 addr, ptr_B, offset64;\n"
"    ld.global.u8 b_byte, [addr];\n"
"    shr.u32 a_sign, a_byte, 7; shr.u32 a_exp, a_byte, 3;\n"
"    and.b32 a_exp, a_exp, 15; and.b32 a_mant, a_byte, 7;\n"
"    or.b32 a_bits, a_exp, a_mant; setp.eq.u32 p_a_zero, a_bits, 0;\n"
"    add.u32 a_bits, a_exp, 120; shl.b32 a_bits, a_bits, 23;\n"
"    shl.b32 a_mant, a_mant, 20; or.b32 a_bits, a_bits, a_mant;\n"
"    shl.b32 a_sign, a_sign, 31; or.b32 a_bits, a_bits, a_sign;\n"
"    mov.b32 a_val, a_bits; @p_a_zero mov.f32 a_val, 0.0;\n"
"    shr.u32 b_sign, b_byte, 7; shr.u32 b_exp, b_byte, 3;\n"
"    and.b32 b_exp, b_exp, 15; and.b32 b_mant, b_byte, 7;\n"
"    or.b32 b_bits, b_exp, b_mant; setp.eq.u32 p_b_zero, b_bits, 0;\n"
"    add.u32 b_bits, b_exp, 120; shl.b32 b_bits, b_bits, 23;\n"
"    shl.b32 b_mant, b_mant, 20; or.b32 b_bits, b_bits, b_mant;\n"
"    shl.b32 b_sign, b_sign, 31; or.b32 b_bits, b_bits, b_sign;\n"
"    mov.b32 b_val, b_bits; @p_b_zero mov.f32 b_val, 0.0;\n"
"    fma.rn.f32 acc, a_val, b_val, acc; add.u32 kk, kk, 1; bra LOOP;\n"
"DONE:\n"
"    mul.lo.u32 offset32, row, dim_n; add.u32 offset32, offset32, col;\n"
"    cvt.u64.u32 offset64, offset32; shl.b64 offset64, offset64, 2;\n"
"    add.u64 addr, ptr_C, offset64; st.global.f32 [addr], acc;\n"
"    ret;\n"
"}\n";

static const char* ptx_gemm_naive_e5m2 =
".version 8.0\n"
".target sm_80\n"
".address_size 64\n"
"\n"
".visible .entry gemm_fp8_naive_e5m2(\n"
"    .param .u64 param_A, .param .u64 param_B, .param .u64 param_C,\n"
"    .param .u32 param_M, .param .u32 param_N, .param .u32 param_K)\n"
"{\n"
"    .reg .u64 ptr_A, ptr_B, ptr_C, addr, offset64;\n"
"    .reg .u32 row, col, kk, dim_m, dim_n, dim_k, a_byte, b_byte, offset32;\n"
"    .reg .u32 a_sign, a_exp, a_mant, a_bits, b_sign, b_exp, b_mant, b_bits;\n"
"    .reg .u32 ctaid_x, ctaid_y, ntid_x, ntid_y, tid_x, tid_y;\n"
"    .reg .f32 acc, a_val, b_val;\n"
"    .reg .pred p_loop, p_a_zero, p_b_zero;\n"
"    ld.param.u64 ptr_A, [param_A]; ld.param.u64 ptr_B, [param_B];\n"
"    ld.param.u64 ptr_C, [param_C]; ld.param.u32 dim_m, [param_M];\n"
"    ld.param.u32 dim_n, [param_N]; ld.param.u32 dim_k, [param_K];\n"
"    mov.u32 ctaid_x, %ctaid.x; mov.u32 ctaid_y, %ctaid.y;\n"
"    mov.u32 ntid_x, %ntid.x; mov.u32 ntid_y, %ntid.y;\n"
"    mov.u32 tid_x, %tid.x; mov.u32 tid_y, %tid.y;\n"
"    mul.lo.u32 row, ctaid_y, ntid_y; add.u32 row, row, tid_y;\n"
"    mul.lo.u32 col, ctaid_x, ntid_x; add.u32 col, col, tid_x;\n"
"    mov.f32 acc, 0.0; mov.u32 kk, 0;\n"
"LOOP:\n"
"    setp.lt.u32 p_loop, kk, dim_k; @!p_loop bra DONE;\n"
"    mul.lo.u32 offset32, row, dim_k; add.u32 offset32, offset32, kk;\n"
"    cvt.u64.u32 offset64, offset32; add.u64 addr, ptr_A, offset64;\n"
"    ld.global.u8 a_byte, [addr];\n"
"    mul.lo.u32 offset32, kk, dim_n; add.u32 offset32, offset32, col;\n"
"    cvt.u64.u32 offset64, offset32; add.u64 addr, ptr_B, offset64;\n"
"    ld.global.u8 b_byte, [addr];\n"
"    shr.u32 a_sign, a_byte, 7; shr.u32 a_exp, a_byte, 2;\n"
"    and.b32 a_exp, a_exp, 31; and.b32 a_mant, a_byte, 3;\n"
"    or.b32 a_bits, a_exp, a_mant; setp.eq.u32 p_a_zero, a_bits, 0;\n"
"    add.u32 a_bits, a_exp, 112; shl.b32 a_bits, a_bits, 23;\n"
"    shl.b32 a_mant, a_mant, 21; or.b32 a_bits, a_bits, a_mant;\n"
"    shl.b32 a_sign, a_sign, 31; or.b32 a_bits, a_bits, a_sign;\n"
"    mov.b32 a_val, a_bits; @p_a_zero mov.f32 a_val, 0.0;\n"
"    shr.u32 b_sign, b_byte, 7; shr.u32 b_exp, b_byte, 2;\n"
"    and.b32 b_exp, b_exp, 31; and.b32 b_mant, b_byte, 3;\n"
"    or.b32 b_bits, b_exp, b_mant; setp.eq.u32 p_b_zero, b_bits, 0;\n"
"    add.u32 b_bits, b_exp, 112; shl.b32 b_bits, b_bits, 23;\n"
"    shl.b32 b_mant, b_mant, 21; or.b32 b_bits, b_bits, b_mant;\n"
"    shl.b32 b_sign, b_sign, 31; or.b32 b_bits, b_bits, b_sign;\n"
"    mov.b32 b_val, b_bits; @p_b_zero mov.f32 b_val, 0.0;\n"
"    fma.rn.f32 acc, a_val, b_val, acc; add.u32 kk, kk, 1; bra LOOP;\n"
"DONE:\n"
"    mul.lo.u32 offset32, row, dim_n; add.u32 offset32, offset32, col;\n"
"    cvt.u64.u32 offset64, offset32; shl.b64 offset64, offset64, 2;\n"
"    add.u64 addr, ptr_C, offset64; st.global.f32 [addr], acc;\n"
"    ret;\n"
"}\n";

/* Reference CPU implementations */
static void gemm_reference_e4m3(const fp8_e4m3_t* A, const fp8_e4m3_t* B, float* C,
                                 int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                acc += fp8_e4m3_to_float(A[i * k + kk]) * fp8_e4m3_to_float(B[kk * n + j]);
            }
            C[i * n + j] = acc;
        }
    }
}

static void gemm_reference_e5m2(const fp8_e5m2_t* A, const fp8_e5m2_t* B, float* C,
                                 int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                acc += fp8_e5m2_to_float(A[i * k + kk]) * fp8_e5m2_to_float(B[kk * n + j]);
            }
            C[i * n + j] = acc;
        }
    }
}

typedef struct {
    double elapsed_ms;
    double gflops;
    double fp32_efficiency;
    double fp8_tensor_efficiency;
    int verified;
} benchmark_result_t;

static benchmark_result_t run_gemm_benchmark(
    const char* ptx_code, const char* kernel_name,
    const uint8_t* A, const uint8_t* B, float* C_ref,
    int m, int n, int k,
    int use_tc, /* use tensor core kernel */
    const gpu_specs_t* specs,
    const char* format_name)
{
    benchmark_result_t result = {0};
    CUmodule module;
    CUfunction kernel_func;
    CUdeviceptr d_A, d_B, d_C;
    CUevent start, stop;
    float* h_C;

    printf("\n=== %s %s (%s) ===\n",
           g_benchmark_mode ? "Benchmarking" : "Testing",
           format_name,
           use_tc ? "Tensor Core" : "Naive");
    printf("Matrix size: %d x %d x %d\n", m, n, k);

    /* Load PTX */
    char error_log[4096] = {0};
    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void* option_values[] = { (void*)error_log, (void*)(size_t)sizeof(error_log) };

    CUresult res = cuModuleLoadDataEx(&module, ptx_code, 2, options, option_values);
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "Failed to load PTX: %s\n", errStr);
        if (error_log[0]) fprintf(stderr, "JIT Error: %s\n", error_log);
        return result;
    }

    CHECK_CUDA(cuModuleGetFunction(&kernel_func, module, kernel_name));

    /* Allocate device memory */
    CHECK_CUDA(cuMemAlloc(&d_A, (size_t)m * k));
    CHECK_CUDA(cuMemAlloc(&d_B, (size_t)k * n));
    CHECK_CUDA(cuMemAlloc(&d_C, (size_t)m * n * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_A, A, (size_t)m * k));
    CHECK_CUDA(cuMemcpyHtoD(d_B, B, (size_t)k * n));
    CHECK_CUDA(cuMemsetD8(d_C, 0, (size_t)m * n * sizeof(float)));

    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    unsigned int m_val = m, n_val = n, k_val = k;
    void* args[] = { &d_A, &d_B, &d_C, &m_val, &n_val, &k_val };

    unsigned int blockDimX, blockDimY, gridDimX, gridDimY;
    if (use_tc) {
        /* Tensor core block/grid configuration based on tile size */
        if (g_tile_size == 128) {
            /* 512 threads (16 warps), 128x128 output per block */
            blockDimX = 512; blockDimY = 1;
            gridDimX = (n + 127) / 128;
            gridDimY = (m + 127) / 128;
        } else if (g_tile_size == 64) {
            /* 512 threads (16 warps), 64x64 output per block */
            blockDimX = 512; blockDimY = 1;
            gridDimX = (n + 63) / 64;
            gridDimY = (m + 63) / 64;
        } else {
            /* 128 threads (4 warps), 32x32 output per block */
            blockDimX = 128; blockDimY = 1;
            gridDimX = (n + 31) / 32;
            gridDimY = (m + 31) / 32;
        }
    } else {
        /* Naive: 16x16 threads/block */
        blockDimX = 16; blockDimY = 16;
        gridDimX = (n + 15) / 16;
        gridDimY = (m + 15) / 16;
    }

    /* Warmup */
    for (int i = 0; i < g_warmup_iters; i++) {
        CHECK_CUDA(cuLaunchKernel(kernel_func, gridDimX, gridDimY, 1,
                                  blockDimX, blockDimY, 1, 0, NULL, args, NULL));
    }
    CHECK_CUDA(cuCtxSynchronize());

    /* Benchmark */
    CHECK_CUDA(cuEventRecord(start, NULL));
    for (int i = 0; i < g_bench_iters; i++) {
        CHECK_CUDA(cuLaunchKernel(kernel_func, gridDimX, gridDimY, 1,
                                  blockDimX, blockDimY, 1, 0, NULL, args, NULL));
    }
    CHECK_CUDA(cuEventRecord(stop, NULL));
    CHECK_CUDA(cuEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
    result.elapsed_ms = elapsed_ms / g_bench_iters;

    double flops = 2.0 * m * n * k;
    result.gflops = flops / (result.elapsed_ms * 1e6);
    result.fp32_efficiency = result.gflops / (specs->fp32_peak_tflops * 1000.0) * 100.0;
    if (specs->fp8_tensor_peak_tflops > 0) {
        result.fp8_tensor_efficiency = result.gflops / (specs->fp8_tensor_peak_tflops * 1000.0) * 100.0;
    }

    /* Verify */
    h_C = (float*)malloc((size_t)m * n * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, (size_t)m * n * sizeof(float)));

    /* Tolerance scales with sqrt(K) due to FP8 accumulation error
     * Base tolerance ~0.1 per 32-element accumulation, scales as sqrt(K/32) */
    float tolerance = use_tc ? 0.1f * sqrtf((float)k) : 0.01f;
    float max_error = 0.0f;
    int errors = 0;
    for (int i = 0; i < m * n; i++) {
        float diff = fabsf(h_C[i] - C_ref[i]);
        if (diff > max_error) max_error = diff;
        if (diff > tolerance) errors++;
    }
    result.verified = (errors == 0);

    printf("Time: %.4f ms, Throughput: %.2f GFLOPS\n", result.elapsed_ms, result.gflops);
    printf("Efficiency: %.2f%% of FP32, %.4f%% of FP8-TC peak\n",
           result.fp32_efficiency, result.fp8_tensor_efficiency);
    printf("Verification: max_error=%.6f, %s\n", max_error, result.verified ? "PASSED" : "FAILED");

    free(h_C);
    CHECK_CUDA(cuEventDestroy(start));
    CHECK_CUDA(cuEventDestroy(stop));
    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    CHECK_CUDA(cuModuleUnload(module));

    return result;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m <size>     M dimension (default: %d)\n", g_M);
    printf("  -n <size>     N dimension (default: %d)\n", g_N);
    printf("  -k <size>     K dimension (default: %d)\n", g_K);
    printf("  --tile <size> Tile size: 32, 64, or 128 (default: %d)\n", g_tile_size);
    printf("  -b, --bench   Benchmark mode\n");
    printf("  -t, --tc      Use Tensor Core (default)\n");
    printf("  --naive       Use naive scalar kernel\n");
    printf("  -i <iters>    Benchmark iterations (default: %d)\n", g_bench_iters);
    printf("  -h, --help    Show this help\n");
}

int main(int argc, char** argv) {
    CUdevice device;
    CUcontext context;
    int deviceCount;
    char deviceName[256];
    gpu_specs_t specs;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) g_M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) g_N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) g_K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) g_bench_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) g_tile_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bench") == 0) g_benchmark_mode = 1;
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tc") == 0) g_use_tensor_core = 1;
        else if (strcmp(argv[i], "--naive") == 0) g_use_tensor_core = 0;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    /* Validate tile size */
    if (g_tile_size != 32 && g_tile_size != 64 && g_tile_size != 128) {
        fprintf(stderr, "Invalid tile size %d (must be 32, 64, or 128)\n", g_tile_size);
        return 1;
    }

    /* Adjust dimensions for tensor core requirements */
    if (g_use_tensor_core) {
        g_M = ((g_M + g_tile_size - 1) / g_tile_size) * g_tile_size;
        g_N = ((g_N + g_tile_size - 1) / g_tile_size) * g_tile_size;
        g_K = ((g_K + 31) / 32) * 32;  /* K always processed in 32-element chunks */
    }

    printf("FP8 GEMM %s (%s, tile=%d)\n", g_benchmark_mode ? "Benchmark" : "Test",
           g_use_tensor_core ? "Tensor Core" : "Naive", g_tile_size);
    printf("==========================================\n");

    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to initialize cuew\n"); return 1;
    }

    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA devices\n"); return 1; }

    CHECK_CUDA(cuDeviceGet(&device, 0));
    CHECK_CUDA(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    query_gpu_specs(device, &specs);
    print_gpu_specs(deviceName, &specs);

    if (g_use_tensor_core && specs.major * 10 + specs.minor < 89) {
        printf("Warning: GPU doesn't support FP8 tensor cores, falling back to naive\n");
        g_use_tensor_core = 0;
    }

    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    size_t size_A = (size_t)g_M * g_K;
    size_t size_B = (size_t)g_K * g_N;
    size_t size_C = (size_t)g_M * g_N;

    fp8_e4m3_t* A_e4m3 = (fp8_e4m3_t*)malloc(size_A);
    fp8_e4m3_t* B_e4m3 = (fp8_e4m3_t*)malloc(size_B);
    fp8_e5m2_t* A_e5m2 = (fp8_e5m2_t*)malloc(size_A);
    fp8_e5m2_t* B_e5m2 = (fp8_e5m2_t*)malloc(size_B);
    float* C_ref = (float*)malloc(size_C * sizeof(float));

    printf("Initializing data (%dx%dx%d)...\n", g_M, g_N, g_K);
    srand(42);
    for (size_t i = 0; i < size_A; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        A_e4m3[i] = float_to_fp8_e4m3(val);
        A_e5m2[i] = float_to_fp8_e5m2(val);
    }
    for (size_t i = 0; i < size_B; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        B_e4m3[i] = float_to_fp8_e4m3(val);
        B_e5m2[i] = float_to_fp8_e5m2(val);
    }

    const char *ptx_e4m3, *ptx_e5m2, *name_e4m3, *name_e5m2;
    if (g_use_tensor_core) {
        if (g_tile_size == 128) {
            ptx_e4m3 = ptx_gemm_tc128_e4m3; name_e4m3 = "gemm_fp8_tc128_e4m3";
            ptx_e5m2 = ptx_gemm_tc128_e5m2; name_e5m2 = "gemm_fp8_tc128_e5m2";
        } else if (g_tile_size == 64) {
            ptx_e4m3 = ptx_gemm_tc64_e4m3; name_e4m3 = "gemm_fp8_tc64_e4m3";
            ptx_e5m2 = ptx_gemm_tc64_e5m2; name_e5m2 = "gemm_fp8_tc64_e5m2";
        } else {
            ptx_e4m3 = ptx_gemm_tc_e4m3; name_e4m3 = "gemm_fp8_tc_e4m3";
            ptx_e5m2 = ptx_gemm_tc_e5m2; name_e5m2 = "gemm_fp8_tc_e5m2";
        }
    } else {
        ptx_e4m3 = ptx_gemm_naive_e4m3; name_e4m3 = "gemm_fp8_naive_e4m3";
        ptx_e5m2 = ptx_gemm_naive_e5m2; name_e5m2 = "gemm_fp8_naive_e5m2";
    }

    printf("\nComputing CPU reference...\n");
    gemm_reference_e4m3(A_e4m3, B_e4m3, C_ref, g_M, g_N, g_K);
    benchmark_result_t r1 = run_gemm_benchmark(ptx_e4m3, name_e4m3, A_e4m3, B_e4m3, C_ref,
                                                g_M, g_N, g_K, g_use_tensor_core, &specs, "FP8 E4M3");

    gemm_reference_e5m2(A_e5m2, B_e5m2, C_ref, g_M, g_N, g_K);
    benchmark_result_t r2 = run_gemm_benchmark(ptx_e5m2, name_e5m2, A_e5m2, B_e5m2, C_ref,
                                                g_M, g_N, g_K, g_use_tensor_core, &specs, "FP8 E5M2");

    printf("\n==========================================\n");
    printf("Summary (%dx%dx%d, %s)\n", g_M, g_N, g_K, g_use_tensor_core ? "Tensor Core" : "Naive");
    printf("==========================================\n");
    printf("Format  Time(ms)   GFLOPS   FP32%%   FP8-TC%%  Verified\n");
    printf("------- --------  --------  ------  -------  --------\n");
    printf("E4M3    %8.4f  %8.2f  %5.2f%%  %6.3f%%  %s\n",
           r1.elapsed_ms, r1.gflops, r1.fp32_efficiency, r1.fp8_tensor_efficiency,
           r1.verified ? "PASS" : "FAIL");
    printf("E5M2    %8.4f  %8.2f  %5.2f%%  %6.3f%%  %s\n",
           r2.elapsed_ms, r2.gflops, r2.fp32_efficiency, r2.fp8_tensor_efficiency,
           r2.verified ? "PASS" : "FAIL");
    printf("\nPeak: FP32=%.2f TFLOPS, FP8-TC=%.2f TFLOPS\n",
           specs.fp32_peak_tflops, specs.fp8_tensor_peak_tflops);

    free(A_e4m3); free(B_e4m3); free(A_e5m2); free(B_e5m2); free(C_ref);
    CHECK_CUDA(cuCtxDestroy(context));
    printf("\nDone.\n");
    return 0;
}
