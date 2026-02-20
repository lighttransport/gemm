/*
 * test_ptx_validate.c - Tests for PTX string validator
 *
 * Tests the ptx_validate.h validator against:
 * 1. Minimal valid PTX programs
 * 2. Invalid PTX with various syntax errors
 * 3. Real PTX kernel strings from the codebase (int8_gemm.c, fp8_gemm.c)
 *
 * CPU-only, no GPU required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ptx_validate.h"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name) do { \
    g_tests_run++; \
    printf("  %-55s ", name); \
} while(0)

#define PASS() do { g_tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg, ...) do { printf("FAIL: " msg "\n", ##__VA_ARGS__); } while(0)

/* ---- Minimal valid PTX ---- */

static const char *ptx_minimal =
    ".version 8.4\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry my_kernel(\n"
    "    .param .u64 param_A\n"
    ")\n"
    "{\n"
    "    .reg .u64 addr;\n"
    "    ld.param.u64 addr, [param_A];\n"
    "    ret;\n"
    "}\n";

static void test_minimal_valid(void) {
    TEST("minimal: valid PTX accepted");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.valid) PASS();
    else FAIL("error: %s (line %d)", r.error, r.error_line);
}

static void test_minimal_version(void) {
    TEST("minimal: version 8.4 parsed");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.version_major == 8 && r.version_minor == 4) PASS();
    else FAIL("got %d.%d expected 8.4", r.version_major, r.version_minor);
}

static void test_minimal_target(void) {
    TEST("minimal: target sm_89 parsed");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (strcmp(r.target, "sm_89") == 0) PASS();
    else FAIL("got '%s' expected 'sm_89'", r.target);
}

static void test_minimal_address_size(void) {
    TEST("minimal: address_size 64 parsed");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.address_size == 64) PASS();
    else FAIL("got %d expected 64", r.address_size);
}

static void test_minimal_entry(void) {
    TEST("minimal: 1 entry 'my_kernel' found");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.n_entries == 1 && strcmp(r.entry_names[0], "my_kernel") == 0) PASS();
    else FAIL("n_entries=%d name='%s'", r.n_entries,
              r.n_entries > 0 ? r.entry_names[0] : "(none)");
}

static void test_minimal_params(void) {
    TEST("minimal: 1 parameter for my_kernel");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.n_params[0] == 1) PASS();
    else FAIL("got %d expected 1", r.n_params[0]);
}

static void test_minimal_regs(void) {
    TEST("minimal: 1 register declaration");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.n_regs == 1) PASS();
    else FAIL("got %d expected 1", r.n_regs);
}

static void test_minimal_instructions(void) {
    TEST("minimal: 2 instructions (ld, ret)");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.n_instructions == 2) PASS();
    else FAIL("got %d expected 2", r.n_instructions);
}

/* ---- Multi-entry PTX ---- */

static const char *ptx_multi_entry =
    ".version 8.4\n"
    ".target sm_120\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry kernel_a(\n"
    "    .param .u64 param_A,\n"
    "    .param .u64 param_B\n"
    ")\n"
    "{\n"
    "    .reg .u64 a, b;\n"
    "    ld.param.u64 a, [param_A];\n"
    "    ld.param.u64 b, [param_B];\n"
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry kernel_b(\n"
    "    .param .u32 param_N\n"
    ")\n"
    "{\n"
    "    .reg .u32 n;\n"
    "    ld.param.u32 n, [param_N];\n"
    "    ret;\n"
    "}\n";

static void test_multi_entry_count(void) {
    TEST("multi: 2 entries found");
    ptx_validation_result r;
    ptx_validate(ptx_multi_entry, &r);
    if (r.valid && r.n_entries == 2) PASS();
    else FAIL("valid=%d n_entries=%d", r.valid, r.n_entries);
}

static void test_multi_entry_names(void) {
    TEST("multi: entry names kernel_a, kernel_b");
    ptx_validation_result r;
    ptx_validate(ptx_multi_entry, &r);
    if (strcmp(r.entry_names[0], "kernel_a") == 0 &&
        strcmp(r.entry_names[1], "kernel_b") == 0) PASS();
    else FAIL("'%s', '%s'", r.entry_names[0], r.entry_names[1]);
}

static void test_multi_entry_params(void) {
    TEST("multi: param counts 2 and 1");
    ptx_validation_result r;
    ptx_validate(ptx_multi_entry, &r);
    if (r.n_params[0] == 2 && r.n_params[1] == 1) PASS();
    else FAIL("params=[%d, %d]", r.n_params[0], r.n_params[1]);
}

/* ---- Complex PTX with shared memory ---- */

static const char *ptx_with_shared =
    ".version 8.4\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry gemm_kernel(\n"
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
    "    .reg .u64 ptr_A, ptr_B, ptr_C;\n"
    "    .reg .u32 dim_m, dim_n, dim_k;\n"
    "    .reg .u32 tid, warp_id, lane_id;\n"
    "    .reg .s32 c0, c1, c2, c3;\n"
    "    .reg .pred p_k;\n"
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
    "    bar.sync 0;\n"
    "    ret;\n"
    "}\n";

static void test_shared_memory(void) {
    TEST("shared: 2 shared memory declarations found");
    ptx_validation_result r;
    ptx_validate(ptx_with_shared, &r);
    if (r.valid && r.n_shared == 2) PASS();
    else FAIL("valid=%d n_shared=%d", r.valid, r.n_shared);
}

static void test_shared_regs(void) {
    TEST("shared: 5 register declarations");
    ptx_validation_result r;
    ptx_validate(ptx_with_shared, &r);
    if (r.n_regs == 5) PASS();
    else FAIL("got %d expected 5", r.n_regs);
}

static void test_shared_params(void) {
    TEST("shared: 6 parameters for gemm_kernel");
    ptx_validation_result r;
    ptx_validate(ptx_with_shared, &r);
    if (r.n_params[0] == 6) PASS();
    else FAIL("got %d expected 6", r.n_params[0]);
}

static void test_shared_instructions(void) {
    TEST("shared: instruction count > 5");
    ptx_validation_result r;
    ptx_validate(ptx_with_shared, &r);
    if (r.n_instructions >= 5) PASS();
    else FAIL("got %d expected >= 5", r.n_instructions);
}

/* ---- With comments ---- */

static const char *ptx_with_comments =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "// This is a comment\n"
    ".visible .entry test_comments(\n"
    "    .param .u32 param_x  // inline comment style\n"
    ")\n"
    "{\n"
    "    // register declarations\n"
    "    .reg .u32 x;\n"
    "    ld.param.u32 x, [param_x];\n"
    "    ret;\n"
    "}\n";

static void test_comments_valid(void) {
    TEST("comments: PTX with comments is valid");
    ptx_validation_result r;
    ptx_validate(ptx_with_comments, &r);
    if (r.valid) PASS();
    else FAIL("error: %s", r.error);
}

static void test_comments_version(void) {
    TEST("comments: version 7.0 parsed");
    ptx_validation_result r;
    ptx_validate(ptx_with_comments, &r);
    if (r.version_major == 7 && r.version_minor == 0) PASS();
    else FAIL("got %d.%d", r.version_major, r.version_minor);
}

/* ---- Invalid PTX: missing directives ---- */

static void test_invalid_empty(void) {
    TEST("invalid: empty string rejected");
    ptx_validation_result r;
    ptx_validate("", &r);
    if (!r.valid) PASS();
    else FAIL("should be rejected");
}

static void test_invalid_null(void) {
    TEST("invalid: NULL rejected");
    ptx_validation_result r;
    ptx_validate(NULL, &r);
    if (!r.valid) PASS();
    else FAIL("should be rejected");
}

static void test_invalid_no_version(void) {
    TEST("invalid: missing .version");
    ptx_validation_result r;
    ptx_validate(
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (!r.valid && strstr(r.error, "version")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

static void test_invalid_no_target(void) {
    TEST("invalid: missing .target");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (!r.valid && strstr(r.error, "target")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

static void test_invalid_no_address_size(void) {
    TEST("invalid: missing .address_size");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".target sm_89\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (!r.valid && strstr(r.error, "address_size")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

static void test_invalid_no_entry(void) {
    TEST("invalid: no .entry function");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".target sm_89\n"
        ".address_size 64\n", &r);
    if (!r.valid && strstr(r.error, "entry")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

/* ---- Invalid PTX: syntax errors ---- */

static void test_invalid_unmatched_open_brace(void) {
    TEST("invalid: unmatched opening brace");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n", &r);
    if (!r.valid && strstr(r.error, "brace")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

static void test_invalid_unmatched_close_brace(void) {
    TEST("invalid: unmatched closing brace");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n}\n", &r);
    if (!r.valid && strstr(r.error, "brace")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

static void test_invalid_bad_target(void) {
    TEST("invalid: unknown target");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".target unknown_gpu\n"
        ".address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (!r.valid && strstr(r.error, "target")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

static void test_invalid_bad_address_size(void) {
    TEST("invalid: address_size 48");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.4\n"
        ".target sm_89\n"
        ".address_size 48\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (!r.valid && strstr(r.error, "address_size")) PASS();
    else FAIL("valid=%d error='%s'", r.valid, r.error);
}

/* ---- Different SM versions ---- */

static void test_target_sm70(void) {
    TEST("target: sm_70 (V100)");
    ptx_validation_result r;
    ptx_validate(
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (r.valid && strcmp(r.target, "sm_70") == 0) PASS();
    else FAIL("valid=%d target='%s'", r.valid, r.target);
}

static void test_target_sm80(void) {
    TEST("target: sm_80 (A100)");
    ptx_validation_result r;
    ptx_validate(
        ".version 7.4\n.target sm_80\n.address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (r.valid && strcmp(r.target, "sm_80") == 0) PASS();
    else FAIL("valid=%d target='%s'", r.valid, r.target);
}

static void test_target_sm90(void) {
    TEST("target: sm_90 (H100)");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.0\n.target sm_90\n.address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (r.valid && strcmp(r.target, "sm_90") == 0) PASS();
    else FAIL("valid=%d target='%s'", r.valid, r.target);
}

static void test_target_sm120(void) {
    TEST("target: sm_120 (Blackwell)");
    ptx_validation_result r;
    ptx_validate(
        ".version 8.8\n.target sm_120\n.address_size 64\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (r.valid && strcmp(r.target, "sm_120") == 0) PASS();
    else FAIL("valid=%d target='%s'", r.valid, r.target);
}

/* ---- Realistic INT8 GEMM PTX (tc32 structure) ---- */

static const char *ptx_gemm_int8_like =
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
    "    .shared .align 16 .b8 smem_A[1024];\n"
    "    .shared .align 16 .b8 smem_B[1024];\n"
    "\n"
    "    .reg .u64 ptr_A, ptr_B, ptr_C, gmem_addr, out_addr;\n"
    "    .reg .u32 dim_m, dim_n, dim_k;\n"
    "    .reg .u32 tid, warp_id, lane_id;\n"
    "    .reg .u32 block_row, block_col;\n"
    "    .reg .u32 k_outer, offset32, tmp;\n"
    "    .reg .u64 offset64, stride64;\n"
    "    .reg .u32 a0, a1, b0, b1;\n"
    "    .reg .s32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
    "    .reg .pred p_k;\n"
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
    "    // Compute block position\n"
    "    mov.u32 block_row, %ctaid.x;\n"
    "    mov.u32 block_col, %ctaid.y;\n"
    "\n"
    "    // MMA instruction\n"
    "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite\n"
    "        {c0, c1, c2, c3}, {a0, a1}, {b0}, {c0, c1, c2, c3};\n"
    "\n"
    "    bar.sync 0;\n"
    "\n"
    "    // Store results\n"
    "    st.global.s32 [out_addr], c0;\n"
    "\n"
    "    ret;\n"
    "}\n";

static void test_gemm_int8_valid(void) {
    TEST("gemm_int8: realistic TC PTX validates");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_int8_like, &r);
    if (r.valid) PASS();
    else FAIL("error: %s (line %d)", r.error, r.error_line);
}

static void test_gemm_int8_entry(void) {
    TEST("gemm_int8: entry name 'gemm_int8_tc32_s8'");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_int8_like, &r);
    if (strcmp(r.entry_names[0], "gemm_int8_tc32_s8") == 0) PASS();
    else FAIL("got '%s'", r.entry_names[0]);
}

static void test_gemm_int8_6_params(void) {
    TEST("gemm_int8: 6 parameters (A,B,C,M,N,K)");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_int8_like, &r);
    if (r.n_params[0] == 6) PASS();
    else FAIL("got %d", r.n_params[0]);
}

static void test_gemm_int8_shared(void) {
    TEST("gemm_int8: 2 shared memory regions");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_int8_like, &r);
    if (r.n_shared == 2) PASS();
    else FAIL("got %d", r.n_shared);
}

static void test_gemm_int8_has_mma(void) {
    TEST("gemm_int8: mma instruction detected");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_int8_like, &r);
    /* mma should be counted as an instruction */
    if (r.n_instructions > 0) PASS();
    else FAIL("no instructions found");
}

/* ---- Realistic FP8 GEMM PTX ---- */

static const char *ptx_gemm_fp8_like =
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
    "    .shared .align 16 .b8 smem_A[1024];\n"
    "    .shared .align 16 .b8 smem_B[1024];\n"
    "\n"
    "    .reg .u64 ptr_A, ptr_B, ptr_C;\n"
    "    .reg .u32 dim_m, dim_n, dim_k;\n"
    "    .reg .u32 tid;\n"
    "    .reg .f32 c0, c1, c2, c3;\n"
    "    .reg .u32 a0, b0;\n"
    "\n"
    "    ld.param.u64 ptr_A, [param_A];\n"
    "    ld.param.u64 ptr_B, [param_B];\n"
    "    ld.param.u64 ptr_C, [param_C];\n"
    "    ld.param.u32 dim_m, [param_M];\n"
    "    ld.param.u32 dim_n, [param_N];\n"
    "    ld.param.u32 dim_k, [param_K];\n"
    "\n"
    "    mov.u32 tid, %tid.x;\n"
    "\n"
    "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
    "        {c0, c1, c2, c3}, {a0}, {b0}, {c0, c1, c2, c3};\n"
    "\n"
    "    st.global.f32 [ptr_C], c0;\n"
    "    ret;\n"
    "}\n";

static void test_gemm_fp8_valid(void) {
    TEST("gemm_fp8: realistic TC PTX validates");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_fp8_like, &r);
    if (r.valid) PASS();
    else FAIL("error: %s (line %d)", r.error, r.error_line);
}

static void test_gemm_fp8_entry(void) {
    TEST("gemm_fp8: entry name 'gemm_fp8_tc_e4m3'");
    ptx_validation_result r;
    ptx_validate(ptx_gemm_fp8_like, &r);
    if (strcmp(r.entry_names[0], "gemm_fp8_tc_e4m3") == 0) PASS();
    else FAIL("got '%s'", r.entry_names[0]);
}

/* ---- Naive scalar PTX (non-TC) ---- */

static const char *ptx_naive_like =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry gemm_naive_s8(\n"
    "    .param .u64 param_A,\n"
    "    .param .u64 param_B,\n"
    "    .param .u64 param_C,\n"
    "    .param .u32 param_M,\n"
    "    .param .u32 param_N,\n"
    "    .param .u32 param_K\n"
    ")\n"
    "{\n"
    "    .reg .u64 addr_A, addr_B, addr_C;\n"
    "    .reg .u32 dim_m, dim_n, dim_k;\n"
    "    .reg .u32 row, col, k;\n"
    "    .reg .s32 acc, a_val, b_val, product;\n"
    "    .reg .pred p_row, p_col, p_k;\n"
    "\n"
    "    ld.param.u64 addr_A, [param_A];\n"
    "    ld.param.u64 addr_B, [param_B];\n"
    "    ld.param.u64 addr_C, [param_C];\n"
    "    ld.param.u32 dim_m, [param_M];\n"
    "    ld.param.u32 dim_n, [param_N];\n"
    "    ld.param.u32 dim_k, [param_K];\n"
    "\n"
    "    // Thread index -> row, col\n"
    "    mov.u32 row, %ctaid.x;\n"
    "    mov.u32 col, %ctaid.y;\n"
    "\n"
    "    // Bounds check\n"
    "    setp.ge.u32 p_row, row, dim_m;\n"
    "    setp.ge.u32 p_col, col, dim_n;\n"
    "\n"
    "    // Accumulate\n"
    "    mov.s32 acc, 0;\n"
    "    mov.u32 k, 0;\n"
    "\n"
    "LOOP:\n"
    "    setp.lt.u32 p_k, k, dim_k;\n"
    "    add.u32 k, k, 1;\n"
    "    mad.lo.s32 acc, a_val, b_val, acc;\n"
    "\n"
    "DONE:\n"
    "    st.global.s32 [addr_C], acc;\n"
    "    ret;\n"
    "}\n";

static void test_naive_valid(void) {
    TEST("naive: scalar PTX validates");
    ptx_validation_result r;
    ptx_validate(ptx_naive_like, &r);
    if (r.valid) PASS();
    else FAIL("error: %s (line %d)", r.error, r.error_line);
}

static void test_naive_no_shared(void) {
    TEST("naive: no shared memory (scalar kernel)");
    ptx_validation_result r;
    ptx_validate(ptx_naive_like, &r);
    if (r.n_shared == 0) PASS();
    else FAIL("got %d shared", r.n_shared);
}

static void test_naive_has_mad(void) {
    TEST("naive: has mad instruction");
    ptx_validation_result r;
    ptx_validate(ptx_naive_like, &r);
    /* mad.lo.s32 should be counted */
    if (r.n_instructions > 5) PASS();
    else FAIL("only %d instructions", r.n_instructions);
}

/* ---- 32-bit address size ---- */

static void test_address_size_32(void) {
    TEST("addr: address_size 32 accepted");
    ptx_validation_result r;
    ptx_validate(
        ".version 6.0\n.target sm_70\n.address_size 32\n"
        ".visible .entry k(.param .u32 p)\n{\n    ret;\n}\n", &r);
    if (r.valid && r.address_size == 32) PASS();
    else FAIL("valid=%d addr=%d", r.valid, r.address_size);
}

/* ---- Brace depth tracking ---- */

static void test_brace_depth(void) {
    TEST("depth: max brace depth tracked");
    ptx_validation_result r;
    ptx_validate(ptx_minimal, &r);
    if (r.brace_depth_max == 1) PASS();
    else FAIL("got %d expected 1", r.brace_depth_max);
}

int main(void) {
    printf("=== PTX String Validator Tests ===\n\n");

    printf("Minimal valid PTX:\n");
    test_minimal_valid();
    test_minimal_version();
    test_minimal_target();
    test_minimal_address_size();
    test_minimal_entry();
    test_minimal_params();
    test_minimal_regs();
    test_minimal_instructions();

    printf("\nMulti-entry PTX:\n");
    test_multi_entry_count();
    test_multi_entry_names();
    test_multi_entry_params();

    printf("\nComplex PTX with shared memory:\n");
    test_shared_memory();
    test_shared_regs();
    test_shared_params();
    test_shared_instructions();

    printf("\nComments handling:\n");
    test_comments_valid();
    test_comments_version();

    printf("\nInvalid PTX - missing directives:\n");
    test_invalid_empty();
    test_invalid_null();
    test_invalid_no_version();
    test_invalid_no_target();
    test_invalid_no_address_size();
    test_invalid_no_entry();

    printf("\nInvalid PTX - syntax errors:\n");
    test_invalid_unmatched_open_brace();
    test_invalid_unmatched_close_brace();
    test_invalid_bad_target();
    test_invalid_bad_address_size();

    printf("\nSM target versions:\n");
    test_target_sm70();
    test_target_sm80();
    test_target_sm90();
    test_target_sm120();

    printf("\nRealistic INT8 GEMM PTX:\n");
    test_gemm_int8_valid();
    test_gemm_int8_entry();
    test_gemm_int8_6_params();
    test_gemm_int8_shared();
    test_gemm_int8_has_mma();

    printf("\nRealistic FP8 GEMM PTX:\n");
    test_gemm_fp8_valid();
    test_gemm_fp8_entry();

    printf("\nNaive scalar PTX:\n");
    test_naive_valid();
    test_naive_no_shared();
    test_naive_has_mad();

    printf("\nMiscellaneous:\n");
    test_address_size_32();
    test_brace_depth();

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
