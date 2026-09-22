"""CPU failure-injection tests for production DFlash overlap setup and fences."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
source = (root / 'qwen35_dflash2.h').read_text()
start = source.index('static void hllm_dflash_overlap_workspace_free(')
end = source.index('static void hllm_qwen35_dflash2_free(', start)
functions = source[start:end]
program = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#undef NULL
#define NULL 0
using hipError_t = int;
constexpr int hipSuccess=0, hipStreamNonBlocking=1, hipEventDisableTiming=2;
constexpr int HLLM_DFLASH_LAYERS=5, HLLM_DFLASH_MAX_BLOCK=8;
constexpr int HLLM_DFLASH_KV_HEADS=8, HLLM_DFLASH_HEAD_DIM=128;
struct hllm_dflash_layer { void *inject_k_bf16, *inject_v_bf16, *inject_kv_bf16; };
struct hllm_qwen35_dflash2 {
    void *fc_bf16, *features_bf16, *x_bf16;
    hllm_dflash_layer layers[5];
    int inject_disabled, inject_stream, target_ready, inject_done;
    void *inject_x, *inject_x_bf16, *inject_norm, *inject_k, *inject_v, *inject_kv;
    int inject_capacity, inject_pending;
};
struct hip_llm_runner { int n_embd, stream, qwen4_forward_error; hllm_qwen35_dflash2 *qwen35_dflash2; };
static int calls, fail_at, allocations, handles, waits, syncs, fail_wait, fail_sync;
static int last_synced=-1;
static int next() { return ++calls==fail_at; }
static int hipMalloc(void **p,size_t n) {
    if(next()) return 1;
    *p=std::malloc(n); assert(*p); ++allocations; return 0;
}
static int hipFree(void *p) { if(p) { std::free(p); --allocations; } return 0; }
static int hipStreamCreateWithFlags(int *p,int) { if(next()) return 1; *p=++handles+10; return 0; }
static int hipEventCreateWithFlags(int *p,int) { return hipStreamCreateWithFlags(p,0); }
static int hipStreamDestroy(int) { --handles; return 0; }
static int hipEventDestroy(int) { --handles; return 0; }
static int hipStreamWaitEvent(int,int,int) { ++waits; return fail_wait; }
static int hipStreamSynchronize(int stream) { ++syncs; last_synced=stream; return stream==fail_sync; }
''' + functions + r'''
static void release(hllm_qwen35_dflash2 &d) {
    hllm_dflash_overlap_workspace_free(&d);
    if(d.inject_stream) hipStreamDestroy(d.inject_stream);
    if(d.target_ready) hipEventDestroy(d.target_ready);
    if(d.inject_done) hipEventDestroy(d.inject_done);
    assert(!allocations && !handles);
}
int main() {
    int sentinel=0, cases=0;
    for(int fused=0;fused<2;++fused) {
        int operations=fused?7:8;
        for(int failure=0;failure<=operations;++failure) {
            hllm_qwen35_dflash2 d{};
            d.fc_bf16=d.features_bf16=d.x_bf16=&sentinel;
            for(auto &l:d.layers) {
                if(fused) l.inject_kv_bf16=&sentinel;
                else l.inject_k_bf16=l.inject_v_bf16=&sentinel;
            }
            hip_llm_runner r{5120,1,0,&d};
            calls=0;fail_at=failure;
            int rc=hllm_dflash_overlap_init(&r,&d);
            assert((rc!=0)==(failure!=0));
            int previous=calls;
            if(failure) {
                assert(d.inject_disabled);
                assert(hllm_dflash_overlap_init(&r,&d)==-1 && calls==previous);
            } else {
                assert(d.inject_capacity==8);
                assert(hllm_dflash_overlap_init(&r,&d)==0 && calls==previous);
            }
            release(d); ++cases;
        }
    }
    for(int stream=0;stream<=1;++stream)
    for(int error=0;error<3;++error) {
        hllm_qwen35_dflash2 d{}; d.inject_pending=1; d.inject_stream=9; d.inject_done=8;
        hip_llm_runner r{5120,stream,0,&d};
        waits=syncs=0; fail_wait=error==1; fail_sync=error==2?stream:-1;
        assert(hllm_qwen35_dflash2_overlap_pending(&r));
        int rc=hllm_qwen35_dflash2_overlap_wait_reset(&r);
        assert((rc!=0)==(error!=0));
        assert(r.qwen4_forward_error==(error!=0));
        assert(!hllm_qwen35_dflash2_overlap_pending(&r));
        assert(waits==1 && syncs==(error==2?2:1));
        assert(last_synced==(error?9:stream));
        ++cases;
    }
    std::printf("DFlash overlap lifecycle: %d cases PASS\n",cases);
}
'''
tmp_root = root / 'tmp'
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='dflash-lifecycle-', dir=tmp_root) as directory:
    path = Path(directory)
    (path/'test.cpp').write_text(program)
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    str(path/'test.cpp'), '-o', str(path/'test')], check=True,
                   env=os.environ | {'TMPDIR': str(path)})
    subprocess.run([str(path/'test')], check=True)
