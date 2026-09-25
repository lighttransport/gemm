"""Check native Q8 decode keeps combine in graphs captured at short positions."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent
source = (root / 'hip_llm_runner.c').read_text()
start = source.index('static inline void launch_attn_decode_native_q8(')
end = source.index('static inline void launch_attn_verify_native_q8(', start)
function = source[start:end]
fields = sorted(set(re.findall(r'r->(\w+)', function)) |
                {'requested_qwen35_decode_graph'})
program = r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
struct hip_llm_runner {
''' + '\n'.join('    int ' + name + ';' for name in fields) + r'''
};
static int decode_calls, combine_calls;
static void record(int fn, int, int, int, int, int, int, size_t, int, void **) {
    if (fn == 4) ++combine_calls;
    else { assert(fn >= 1 && fn <= 3); ++decode_calls; }
}
#define LAUNCH record
''' + function + r'''
int main() {
    int cases=0;
    for (int variant : {1,2,3})
    for (int graph : {0,1})
    for (int position : {0,1,254,255,256,257,4096,16384,65536}) {
        hip_llm_runner r{};
        r.n_heads=24; r.n_kv_heads=4; r.q8_attention_nsm=64;
        r.q8_attention_max_splits=128;
        r.fn_q8_attention_decode=1;
        r.fn_q8_attention_decode_gqa3=variant>=2?2:0;
        r.fn_q8_attention_decode_gqa3_reuse=variant==3?3:0;
        r.fn_q8_attention_combine=4;
        r.requested_qwen35_decode_graph=graph;
        r.cur_position=position;
        decode_calls=combine_calls=0;
        launch_attn_decode_native_q8(&r,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);
        assert(decode_calls==1);
        assert(combine_calls==(graph || position>=256));
        ++cases;
    }
    std::printf("Decode attention graph/short-context launch: %d cases PASS\n",cases);
}
'''
tmp_root = root / 'tmp'
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='decode-attention-launch-', dir=tmp_root) as directory:
    path = Path(directory)
    (path/'test.cpp').write_text(program)
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    str(path/'test.cpp'), '-o', str(path/'test')], check=True,
                   env=os.environ | {'TMPDIR': str(path)})
    subprocess.run([str(path/'test')], check=True)
