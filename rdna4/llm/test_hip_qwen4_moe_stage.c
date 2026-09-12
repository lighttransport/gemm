/* Small, model-free GPU test of the actual staged dispatcher and upload paths.
 * Include its translation unit to test internal functions without a public API. */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "hip_llm_runner.c"
#include <assert.h>
#include <stddef.h>

#define REQUIRE(call) do { if ((call) != 0) { \
    fprintf(stderr, "staging test failed at line %d: %s\n", __LINE__, #call); exit(1); \
} } while (0)
#ifndef QWEN4_STAGE_TEST_DIM
#define QWEN4_STAGE_TEST_DIM 256
#endif
enum { TEST_NE = 11, TEST_DIM = QWEN4_STAGE_TEST_DIM, TEST_TASKS = TEST_NE * 4 };

static void *test_alloc(size_t bytes) {
    void *p = NULL; REQUIRE(hipMalloc(&p, bytes)); return p;
}
static void test_weights(unsigned char *p, size_t bytes, int type, unsigned seed) {
    for (size_t i = 0; i < bytes; ++i) {
        seed = seed * 1664525u + 1013904223u; p[i] = (unsigned char)(seed >> 24);
    }
    size_t block = type == GGML_TYPE_Q8_0 ? 34 :
                   type == GGML_TYPE_Q5_1 ? 24 :
                   type == GGML_TYPE_Q4_K ? 144 : type == GGML_TYPE_Q5_K ? 176 : 210;
    for (size_t i = 0; i < bytes; i += block) {
        uint16_t scale = 0x1800; /* finite, small, nonzero */
        size_t offset = type == GGML_TYPE_Q6_K ? 208 : 0;
        memcpy(p + i + offset, &scale, 2);
        if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q5_1)
            memcpy(p + i + 2, &scale, 2);
    }
}
static void upload_expert(hip_llm_runner *r, hip_layer *cl, int e, int slot) {
    REQUIRE(hipStreamSynchronize(r->stream));
    REQUIRE(hllm_cache_copy_buffer_on(r,
        (char *)cl->moe_cache_gate + (size_t)slot * cl->moe_cache_stride_gate,
        (char *)cl->moe_gate_exps_host + (size_t)e * cl->moe_exp_stride_gu,
        cl->moe_gate_exps_type, TEST_DIM, TEST_DIM, cl->moe_exp_stride_gu,
        cl->moe_cache_stride_gate, r->qwen4_prefill_q8_stage, r->moe_q8_stage_bytes, r->stream));
    REQUIRE(hllm_cache_copy_buffer_on(r,
        (char *)cl->moe_cache_up + (size_t)slot * cl->moe_cache_stride_up,
        (char *)cl->moe_up_exps_host + (size_t)e * cl->moe_exp_stride_gu,
        cl->moe_up_exps_type, TEST_DIM, TEST_DIM, cl->moe_exp_stride_gu,
        cl->moe_cache_stride_up, r->qwen4_prefill_q8_stage, r->moe_q8_stage_bytes, r->stream));
    REQUIRE(hllm_cache_copy_buffer_on(r,
        (char *)cl->moe_cache_down + (size_t)slot * cl->moe_cache_stride_down,
        (char *)cl->moe_down_exps_host + (size_t)e * cl->moe_exp_stride_d,
        cl->moe_down_exps_type, TEST_DIM, TEST_DIM, cl->moe_exp_stride_d,
        cl->moe_cache_stride_down, r->qwen4_prefill_q8_stage, r->moe_q8_stage_bytes, r->stream));
    REQUIRE(qwen4_cache_map_set(r, cl->d_moe_cache_map, -1, e, slot));
    REQUIRE(hipStreamSynchronize(r->stream));
    cl->moe_cache_ids[slot] = e;
}
static thipMemcpyAsync saved_copy;
static int fail_copy_count;
static hipError_t HIPAPI fail_copy(void *dst, const void *src, size_t n,
                                   hipMemcpyKind kind, hipStream_t stream) {
    if (--fail_copy_count == 0) return hipErrorInvalidValue;
    return saved_copy(dst, src, n, kind, stream);
}
static void run_case(hip_llm_runner *r, int gu, int down, int registered) {
    hip_layer cl = {0};
    cl.moe_gate_exps_type = cl.moe_up_exps_type = gu;
    cl.moe_down_exps_type = down;
    cl.moe_exp_rows_gu = cl.moe_exp_cols_gu = cl.moe_exp_rows_d = cl.moe_exp_cols_d = TEST_DIM;
    cl.moe_exp_stride_gu = dequant_row_size(gu, TEST_DIM) * TEST_DIM;
    cl.moe_exp_stride_d = dequant_row_size(down, TEST_DIM) * TEST_DIM;
    cl.moe_cache_stride_gate = cl.moe_cache_stride_up = cl.moe_exp_stride_gu;
    cl.moe_cache_stride_down = down == GGML_TYPE_Q8_0 ? (size_t)TEST_DIM * (TEST_DIM / 32) * 36 : cl.moe_exp_stride_d;
    size_t gu_bytes = TEST_NE * cl.moe_exp_stride_gu, down_bytes = TEST_NE * cl.moe_exp_stride_d;
    unsigned char *gate = malloc(gu_bytes), *up = malloc(gu_bytes), *dw = malloc(down_bytes);
    assert(gate && up && dw);
    test_weights(gate, gu_bytes, gu, 1); test_weights(up, gu_bytes, gu, 2);
    test_weights(dw, down_bytes, down, 3);
    if (registered) {
        REQUIRE(hipHostRegister(gate, gu_bytes, 0)); REQUIRE(hipHostRegister(up, gu_bytes, 0));
        REQUIRE(hipHostRegister(dw, down_bytes, 0));
    }
    cl.moe_gate_exps_host = gate; cl.moe_up_exps_host = up; cl.moe_down_exps_host = dw;
    cl.moe_cache_gate = test_alloc(TEST_NE * cl.moe_cache_stride_gate);
    cl.moe_cache_up = test_alloc(TEST_NE * cl.moe_cache_stride_up);
    cl.moe_cache_down = test_alloc(TEST_NE * cl.moe_cache_stride_down);
    cl.d_moe_cache_map = test_alloc(TEST_NE * sizeof(int));
    int ids[TEST_NE], offs[TEST_NE+1] = {0}; uint32_t scores[TEST_NE] = {0};
    cl.moe_cache_ids = ids; cl.moe_cache_slots = TEST_NE; cl.moe_prefill_score = scores;
    for (int e = 0; e < TEST_NE; ++e) { ids[e] = -1; offs[e+1] = offs[e] + 1 + e % 4; }
    size_t out_bytes = (size_t)offs[TEST_NE] * TEST_DIM * sizeof(float);
    float *reference = malloc(out_bytes), *actual = malloc(out_bytes); assert(reference && actual);
    /* Serial expert oracle: each copy and compute completes before slot reuse. */
    for (int e = 0; e < TEST_NE; ++e) {
        upload_expert(r, &cl, e, 0);
        int count = offs[e+1] - offs[e], tasks[8];
        for (int i = 0; i < count; ++i) { tasks[i] = e; tasks[count+i] = offs[e]+i; }
        REQUIRE(hipMemcpyAsync(r->d_router_logits_batch, tasks, (size_t)2*count*sizeof(int), hipMemcpyHostToDevice, r->stream));
        REQUIRE(launch_qwen4_experts_grouped(r, &cl, TEST_NE, TEST_DIM, TEST_DIM, count));
        REQUIRE(hipStreamSynchronize(r->stream));
    }
    REQUIRE(hipMemcpy(reference, r->d_moe_eout, out_bytes, hipMemcpyDeviceToHost));
    double magnitude = 0;
    for (size_t i = 0; i < out_bytes / sizeof(float); ++i) { assert(isfinite(reference[i])); magnitude += fabs(reference[i]); }
    assert(magnitude > 0);
    for (int overlap = 0; overlap < 2; ++overlap) for (int promote = 0; promote < 2; ++promote) {
        r->moe_copy_pipeline = overlap;
        setenv("LLM_QWEN4_STAGE_PROMOTE", promote ? "1" : "0", 1);
        for (int resident_case = 0; resident_case < 3; ++resident_case) {
            int residents = resident_case == 0 ? 0 : resident_case == 1 ? 3 : TEST_NE;
            for (int repeat = 0; repeat < 4; ++repeat) {
                REQUIRE(hipMemset(cl.d_moe_cache_map, 0xff, TEST_NE * sizeof(int)));
                for (int e = 0; e < TEST_NE; ++e) ids[e] = -1;
                for (int e = 0; e < residents; ++e) upload_expert(r, &cl, e, e);
                REQUIRE(hipMemset(r->d_moe_eout, 0xff, out_bytes));
                REQUIRE(forward_qwen4_moe_staged(r, &cl, offs, TEST_NE, TEST_DIM, TEST_DIM));
                REQUIRE(hipMemcpy(actual, r->d_moe_eout, out_bytes, hipMemcpyDeviceToHost));
                if (memcmp(reference, actual, out_bytes)) {
                    fprintf(stderr, "staging mismatch gu=%d down=%d pinned=%d overlap=%d promotion=%d residents=%d repeat=%d\n",
                            gu, down, registered, overlap, promote, residents, repeat); exit(1);
                }
                if (repeat == 2) REQUIRE(qwen4_prefill_copies_drain(r));
            }
        }
    }
    /* Mid-wave upload errors are fatal, never a successful partial/fallback result. */
    for (int e = 0; e < TEST_NE; ++e) ids[e] = -1;
    saved_copy = hipMemcpyAsync; fail_copy_count = 2; hipMemcpyAsync = fail_copy;
    assert(forward_qwen4_moe_staged(r, &cl, offs, TEST_NE, TEST_DIM, TEST_DIM) == -1);
    hipMemcpyAsync = saved_copy;
    assert(r->qwen4_forward_error); r->qwen4_forward_error = 0;
    REQUIRE(qwen4_prefill_copies_drain(r));
    for (int e = 0; e < TEST_NE; ++e) ids[e] = -1;
    REQUIRE(forward_qwen4_moe_staged(r, &cl, offs, TEST_NE, TEST_DIM, TEST_DIM));
    REQUIRE(hipMemcpy(actual, r->d_moe_eout, out_bytes, hipMemcpyDeviceToHost));
    assert(!memcmp(reference, actual, out_bytes));
    if (registered) { REQUIRE(hipHostUnregister(gate)); REQUIRE(hipHostUnregister(up)); REQUIRE(hipHostUnregister(dw)); }
    free(gate); free(up); free(dw); free(reference); free(actual);
    REQUIRE(hipFree(cl.moe_cache_gate)); REQUIRE(hipFree(cl.moe_cache_up));
    REQUIRE(hipFree(cl.moe_cache_down)); REQUIRE(hipFree(cl.d_moe_cache_map));
}
int main(void) {
    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) { fprintf(stderr, "SKIP: HIP device unavailable\n"); return 77; }
    r->is_qwen4exp = 1; r->n_experts = TEST_NE; r->n_embd = r->expert_ff = TEST_DIM;
    r->qwen4_prefill_staging = 1; r->qwen4_stage_slots = 2;
    r->qwen4_stage_task_capacity = TEST_TASKS;
    r->qwen4_stage_stride_gate = r->qwen4_stage_stride_up = (TEST_DIM / 256) * 176 * TEST_DIM;
    r->qwen4_stage_stride_down = (TEST_DIM / 32) * 36 * TEST_DIM;
    r->qwen4_stage_q8_bytes = r->moe_q8_stage_bytes = (TEST_DIM / 32) * 34 * TEST_DIM;
    REQUIRE(qwen4_prefill_copies_init(r));
    for (int b = 0; b < 2; ++b)
        REQUIRE(qwen4_moe_bank_init(&r->qwen4_stage_bank[b], 2,
            r->qwen4_stage_stride_gate, r->qwen4_stage_stride_up, r->qwen4_stage_stride_down,
            r->qwen4_stage_q8_bytes, (TEST_NE + 2*TEST_TASKS) * sizeof(int)));
    r->d_moe_eout = test_alloc((size_t)TEST_TASKS * TEST_DIM * sizeof(float));
    r->d_moe_eg = test_alloc((size_t)TEST_TASKS * TEST_DIM * sizeof(float));
    r->d_moe_gather_in = test_alloc((size_t)TEST_TASKS * TEST_DIM * sizeof(float));
    r->d_router_logits_batch = test_alloc(2 * TEST_TASKS * sizeof(int));
    float input[TEST_TASKS * TEST_DIM];
    for (int i = 0; i < TEST_TASKS * TEST_DIM; ++i) input[i] = (float)((i*17)%31 - 15) / 128;
    REQUIRE(hipMemcpy(r->d_moe_gather_in, input, sizeof(input), hipMemcpyHostToDevice));
    int downs[] = {GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K};
    for (int gu = 0; gu < 2; ++gu) for (int d = 0; d < 3; ++d) for (int pin = 0; pin < 2; ++pin)
        run_case(r, gu ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K, downs[d], pin);
    hip_llm_free(r);
    puts("Qwen4 GPU staging: serial parity, residency, waves, promotion, pinning, reset, errors: PASS");
    return 0;
}
