"""CPU-only mocks exercising production MoE copy-resource lifecycle code."""
from pathlib import Path
import os
import subprocess
import tempfile

source = Path(__file__).with_name("hip_llm_runner.c").read_text()
start = source.index('    const char *pipeline_env = getenv("LLM_MOE_COPY_PIPELINE");')
end = source.index("    if (verbose >= 1)", start)
init = source[start:end]
start = source.index("void hip_llm_set_decode_mode(")
end = source.index("int hip_llm_n_embd(", start)
setter = source[start:end]
start = source.index("static int qwen4_prefill_copies_drain(")
end = source.index("static int qwen4_stage_metadata(", start)
drain = source[start:end]
program = r'''
#include <assert.h>
#include <stdlib.h>
#include <string.h>
typedef int hipError_t;
enum { hipSuccess = 0 };
typedef struct { int uploaded, consumed; } qwen4_moe_fence;
typedef struct { qwen4_moe_fence fence; } qwen4_moe_bank;
static void qwen4_moe_fence_reset(qwen4_moe_fence *f) { f->uploaded = f->consumed = 0; }
typedef struct {
    int *moe_cache_ids;
    int moe_cache_slots;
    int *d_moe_cache_map;
    int moe_pending_slot, moe_pending_expert;
} hip_layer;
typedef struct {
    int moe_copy_pipeline, moe_copy_stream, moe_copy_ready[128], moe_compute_done[128];
    int decode_mode, n_layers, is_qwen4exp, n_experts, stream;
    hip_layer *layers;
    int qwen4_prefill_copy_stream, qwen4_forward_error;
    qwen4_moe_fence qwen4_prefill_cache_fence[128];
    qwen4_moe_bank qwen4_stage_bank[2];
} hip_llm_runner;
static int syncs, events;
static const int *pending_src;
static int *pending_dst;
static size_t pending_bytes;
enum { hipStreamNonBlocking = 1, hipEventDisableTiming = 2 };
enum { hipMemcpyHostToDevice = 1 };
static int hipStreamCreateWithFlags(int *stream, int flags) {
    assert(flags == hipStreamNonBlocking); *stream = 1; return 0;
}
static int hipEventCreateWithFlags(int *event, int flags) {
    assert(flags == hipEventDisableTiming); *event = ++events; return 0;
}
static int hipStreamSynchronize(int stream) {
    assert(stream); ++syncs;
    if (pending_src) {
        memcpy(pending_dst, pending_src, pending_bytes);
        pending_src = NULL;
    }
    return 0;
}
static int hipMemcpyAsync(int *dst, const int *src, size_t n, int kind, int stream) {
    assert(kind == hipMemcpyHostToDevice && stream);
    assert(!pending_src);
    pending_src = src; pending_dst = dst; pending_bytes = n;
    return 0;
}
static void checked_free(void *p) { assert(p != pending_src); free(p); }
#define free checked_free
#define CHECK_HIP_NULL(call) do { if ((call) != 0) return -1; } while (0)
static int init_copies(hip_llm_runner *r) {
''' + init + r'''
    return 0;
}
''' + drain + setter + r'''
int main(void) {
    for (int mask = 0; mask < 8; ++mask) {
        setenv("LLM_MOE_COPY_PIPELINE", mask & 1 ? "1" : "0", 1);
        setenv("LLM_MOE_CPU_DECODE_MISSES", mask & 2 ? "1" : "0", 1);
        setenv("LLM_QWEN4_DELAYED_CACHE", mask & 4 ? "1" : "0", 1);
        hip_llm_runner r = {0}; events = 0;
        assert(init_copies(&r) == 0);
        assert(r.moe_copy_pipeline == (mask & 1));
        assert(!!r.moe_copy_stream == !!mask);
        assert(events == (mask ? 256 : 0));
    }
    int ids[] = {-2, 7};
    int map[8] = {0};
    hip_layer layer = {ids, 2, map, 0, 42};
    hip_llm_runner r = {0};
    r.moe_copy_stream = 1; r.layers = &layer; r.n_layers = 1;
    r.is_qwen4exp = 1; r.n_experts = 8; r.stream = 1;
    hip_llm_set_decode_mode(&r, 1);
    assert(r.decode_mode == 1 && syncs == 3 && ids[0] == -2);
    assert(map[7] == 1 && map[0] == -1);
    hip_llm_set_decode_mode(&r, 0);
    assert(!r.decode_mode && syncs == 4 && ids[0] == 42 && ids[1] == 7);
    assert(layer.moe_pending_slot == -1 && layer.moe_pending_expert == -1);
    hip_llm_set_decode_mode(&r, 0);
    assert(ids[0] == 42);
    r.qwen4_prefill_copy_stream = 2;
    r.qwen4_stage_bank[0].fence.uploaded = 1;
    r.qwen4_prefill_cache_fence[127].consumed = 1;
    int before = syncs;
    hip_llm_set_decode_mode(&r, 0);
    assert(syncs == before + 3);
    assert(!r.qwen4_stage_bank[0].fence.uploaded);
    assert(!r.qwen4_prefill_cache_fence[127].consumed);
    hip_llm_set_decode_mode(NULL, 0);
    return 0;
}
'''
tmp_root = Path(__file__).resolve().parent / "tmp"
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="qwen-copy-lifecycle-", dir=tmp_root) as directory:
    path = Path(directory)
    (path / "test.c").write_text(program)
    subprocess.run(["cc", "-Wall", "-Wextra", "-Werror", str(path / "test.c"),
                    "-o", str(path / "test")], check=True,
                   env=os.environ | {"TMPDIR": str(path)})
    subprocess.run([str(path / "test")], check=True)
print("MoE copy resources and decode/prefill transition: PASS")
