"""Check production verifier launch argument layouts with a recording HIP stub."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent
source = (root / "hip_llm_runner.c").read_text()
start = source.index("static inline void launch_attn_verify_native_q8(")
end = source.index("static int launch_attn_prefill_native_q8(", start)
function = source[start:end]
fields = sorted(set(re.findall(r"r->(\w+)", function)))
program = r'''
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <initializer_list>
using hipFunction_t = int;
struct hip_llm_runner {
''' + "\n".join("    int " + name + ";" for name in fields) + r'''
};
static hip_llm_runner runner;
static void *expected_gate, *expected_positions;
static int expected_queries, expected_width, calls;
static void record(int fn, int gx, int gy, int, int bx, int, int,
                   size_t, int, void **args) {
    if (fn < 100) return; // attention split launch
    ++calls;
    assert(gx == runner.n_heads && bx == 256);
    assert(gy == (expected_queries + expected_width - 1) / expected_width);
    int i = 3;
    if (expected_gate) {
        assert(fn == 100 + expected_width);
        assert(*static_cast<void **>(args[i++]) == expected_gate);
    } else assert(fn == 200 + expected_width);
    assert(*static_cast<void **>(args[i++]) == expected_positions);
    assert(*static_cast<int *>(args[i++]) == runner.n_heads);
    assert(*static_cast<int *>(args[i++]) == runner.q8_attention_nsm);
    assert(*static_cast<int *>(args[i++]) == 11);
    assert(*static_cast<int *>(args[i++]) == 0);
    assert(*static_cast<int *>(args[i++]) == expected_queries);
}
#define LAUNCH record
''' + function + r'''
int main() {
    unsetenv("LLM_QWEN35_VERIFY_FUSED_SPLIT_COMBINE");
    runner.n_heads = 24;
    runner.q8_attention_nsm = 64;
    runner.q8_attention_max_splits = 128;
    runner.fn_q8_attention_decode = 1;
    runner.fn_q8_attention_decode_reuse8 = 2;
    runner.fn_q8_attention_combine_verify4_gate = 104;
    runner.fn_q8_attention_combine_verify8_gate = 108;
    runner.fn_q8_attention_combine_verify16_gate = 116;
    runner.fn_q8_attention_combine_verify4 = 204;
    runner.fn_q8_attention_combine_verify8 = 208;
    runner.fn_q8_attention_combine_verify16 = 216;
    int gate_data, positions;
    expected_positions = &positions;
    int cases = 0;
    for (int mode : {1, 2, 3})
    for (int queries : {2, 4, 7, 8, 9, 16})
    for (bool gated : {false, true}) {
        char value[] = {char('0' + mode), 0};
        setenv("LLM_QWEN35_VERIFY_COMBINE_GROUPED", value, 1);
        expected_gate = gated ? &gate_data : nullptr;
        expected_queries = queries;
        expected_width = queries <= 4 ? 4 : queries <= 8 ?
            (mode >= 2 ? 8 : 4) : (mode >= 3 ? 16 : mode >= 2 ? 8 : 4);
        calls = 0;
        launch_attn_verify_native_q8(&runner, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            expected_positions, expected_gate, queries, 1);
        assert(calls == 1);
        ++cases;
    }
    std::printf("Verifier grouped launch: %d cases PASS\n", cases);
}
'''
tmp_root = root / "tmp"
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="verifier-launch-", dir=tmp_root) as directory:
    path = Path(directory)
    (path / "test.cpp").write_text(program)
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(path / "test.cpp"), "-o", str(path / "test")], check=True,
                   env=os.environ | {"TMPDIR": str(path)})
    subprocess.run([str(path / "test")], check=True)
