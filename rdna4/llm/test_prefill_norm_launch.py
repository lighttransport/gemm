"""Check the prefill-only reference RMSNorm dispatch and HIP argument layout."""
import os
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parent
source = (root/'hip_llm_runner.c').read_text()
start = source.index('static inline void launch_qwen35_prefill_rmsnorm(')
end = source.index('static inline void launch_matvec(', start)
program = r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
struct hip_llm_runner {
    int fn_qwen35_prefill_rmsnorm_reference, is_hybrid, is_qwen4exp, stream;
};
static int calls, fallback, width, rows, stride;
static void *expected_out, *expected_x, *expected_w;
static void launch_rmsnorm_batch(hip_llm_runner *, void *out, void *x, void *w,
                               int n, int m, int s, float eps) {
    assert(out==expected_out && x==expected_x && w==expected_w);
    assert(n==width && m==rows && s==stride && eps==1e-6f);
    ++fallback;
}
static void record(int fn,int gx,int gy,int gz,int bx,int by,int bz,size_t sm,int stream,void **a) {
    assert(fn==7 && gx==rows && gy==1 && gz==1);
    assert(bx==(width<1024?256:1024) && by==1 && bz==1 && !sm && stream==9);
    assert(*static_cast<void **>(a[0])==expected_out);
    assert(*static_cast<void **>(a[1])==expected_x);
    assert(*static_cast<void **>(a[2])==expected_w);
    assert(*static_cast<int *>(a[3])==width);
    assert(*static_cast<int *>(a[4])==stride);
    assert(*static_cast<float *>(a[5])==1e-6f);
    ++calls;
}
#define LAUNCH record
''' + source[start:end] + r'''
int main() {
    int data[3], cases=0;
    for (int enabled:{0,1}) for(int hybrid:{0,1}) for(int qwen4:{0,1})
    for(int n:{128,5120}) for(int m:{1,189,512}) for(int padding:{0,32})
    for(bool inplace:{false,true}) {
        hip_llm_runner r{enabled?7:0,hybrid,qwen4,9};
        expected_x=&data[0]; expected_out=inplace?expected_x:&data[1]; expected_w=&data[2];
        width=n; rows=m; stride=n+padding; calls=fallback=0;
        launch_qwen35_prefill_rmsnorm(&r,expected_out,expected_x,expected_w,n,m,stride,1e-6f);
        assert(calls==(enabled && hybrid && !qwen4));
        assert(calls+fallback==1); ++cases;
    }
    std::printf("Prefill reference norm dispatch: %d cases PASS\n",cases);
}
'''
tmp=root/'tmp'; tmp.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='prefill-norm-launch-',dir=tmp) as directory:
    p=Path(directory);(p/'test.cpp').write_text(program)
    subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(p/'test.cpp'),'-o',str(p/'test')],check=True,env=os.environ|{'TMPDIR':str(p)})
    subprocess.run([str(p/'test')],check=True)
