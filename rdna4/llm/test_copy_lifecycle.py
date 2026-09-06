"""CPU-only mocks exercising production MoE copy-resource lifecycle code."""
from pathlib import Path
import subprocess
import tempfile

source = Path(__file__).with_name("hip_llm_runner.c").read_text()
start = source.index('    const char *pipeline_env = getenv("LLM_MOE_COPY_PIPELINE");')
end = source.index("    if (verbose >= 1)", start)
init = source[start:end]
start = source.index("void hip_llm_set_decode_mode(")
end = source.index("int hip_llm_n_embd(", start)
setter = source[start:end]
program = r'''
#include <assert.h>
#include <stdlib.h>
#include <string.h>
typedef struct { int *moe_cache_ids; int moe_pending_slot, moe_pending_expert; } hip_layer;
typedef struct {
    int moe_copy_pipeline, moe_copy_stream, moe_copy_ready[128], moe_compute_done[128];
    int decode_mode, n_layers;
    hip_layer *layers;
} hip_llm_runner;
static int syncs, events;
enum { hipStreamNonBlocking = 1, hipEventDisableTiming = 2 };
static int hipStreamCreateWithFlags(int *stream, int flags) {
    assert(flags == hipStreamNonBlocking); *stream = 1; return 0;
}
static int hipEventCreateWithFlags(int *event, int flags) {
    assert(flags == hipEventDisableTiming); *event = ++events; return 0;
}
static int hipStreamSynchronize(int stream) { assert(stream); ++syncs; return 0; }
#define CHECK_HIP_NULL(call) do { if ((call) != 0) return -1; } while (0)
static int init_copies(hip_llm_runner *r) {
''' + init + r'''
    return 0;
}
''' + setter + r'''
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
    hip_layer layer = {ids, 0, 42};
    hip_llm_runner r = {0};
    r.moe_copy_stream = 1; r.layers = &layer; r.n_layers = 1;
    hip_llm_set_decode_mode(&r, 1);
    assert(r.decode_mode == 1 && syncs == 0 && ids[0] == -2);
    hip_llm_set_decode_mode(&r, 0);
    assert(!r.decode_mode && syncs == 1 && ids[0] == 42 && ids[1] == 7);
    assert(layer.moe_pending_slot == -1 && layer.moe_pending_expert == -1);
    hip_llm_set_decode_mode(&r, 0);
    assert(ids[0] == 42);
    hip_llm_set_decode_mode(NULL, 0);
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="qwen-copy-lifecycle-") as directory:
    path = Path(directory)
    (path / "test.c").write_text(program)
    subprocess.run(["cc", "-Wall", "-Wextra", "-Werror", str(path / "test.c"),
                    "-o", str(path / "test")], check=True)
    subprocess.run([str(path / "test")], check=True)
print("MoE copy resources and decode/prefill transition: PASS")
