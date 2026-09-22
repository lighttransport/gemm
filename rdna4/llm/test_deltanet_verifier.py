"""GPU bit-parity test of every verifier output and rollback checkpoint."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
source = (root / 'hip_llm_runner.c').read_text()
start = source.index('"__global__ void deltanet_step_batch_gda_verify_f32(')
end = source.index('"/* ---- 25. gated_rmsnorm_silu_f32', start)
kernels = ''.join(ast.literal_eval(line) for line in source[start:end].splitlines())
program = r'''
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#define DLN_BATCH_MAX_DSTATE 256
''' + kernels + r'''
#define HIP(call) do { hipError_t e=(call); if(e!=hipSuccess) { \
    std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e));std::exit(1); } } while(0)
struct Buffer {
    float *p; size_t n;
    explicit Buffer(size_t count):n(count) { HIP(hipMalloc(&p,n*sizeof(float))); }
    ~Buffer() { HIP(hipFree(p)); }
    void upload(const std::vector<float>& v) { HIP(hipMemcpy(p,v.data(),n*sizeof(float),hipMemcpyHostToDevice)); }
    std::vector<float> read() { std::vector<float> v(n); HIP(hipMemcpy(v.data(),p,n*sizeof(float),hipMemcpyDeviceToHost));return v; }
};
static void exact(const std::vector<float>& a,const std::vector<float>& b,const char *label) {
    if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))) {
        std::fprintf(stderr,"FAIL %s\n",label);std::exit(3);
    }
    for(float x:b) {
        uint32_t bits; std::memcpy(&bits,&x,sizeof(bits));
        if((bits & 0x7f800000u)==0x7f800000u) std::exit(4);
    }
}
int main() {
    hipDeviceProp_t props; HIP(hipGetDeviceProperties(&props,0));
    if(std::strncmp(props.gcnArchName,"gfx1201",7)) return 2;
    uint32_t rng=42;
    auto random=[&]() { rng=1664525*rng+1013904223;return rng; };
    auto fill=[&](Buffer &b,float bias,float factor) {
        std::vector<float> v(b.n);
        for(float &x:v) x=bias+factor*(int(random()%4096)-2048)/2048.0f;
        b.upload(v);return v;
    };
    int cases=0;
    for(int heads:{1,48}) for(int rows:{1,4,5,8,16}) for(int pad:{0,17}) {
        int stride=heads*128+pad;
        size_t state_size=size_t(heads)*128*128;
        Buffer state(state_size),q(rows*heads*128),k(q.n),v(rows*stride);
        Buffer alpha(rows*heads),beta(alpha.n);
        auto original=fill(state,0,0.05f);
        fill(q,0,0.05f);fill(k,0,0.05f);fill(v,0,0.05f);
        fill(alpha,0.5f,0.4f);fill(beta,0.5f,0.4f);
        Buffer checkpoints(rows*state_size+32),out(q.n+32);
        std::vector<float> sentinel_cp(checkpoints.n,-999),sentinel_out(out.n,-999);
        checkpoints.upload(sentinel_cp);out.upload(sentinel_out);
        deltanet_step_batch_gda_verify_f32<<<dim3(heads,1,32),dim3(32,4)>>>(
            state.p,checkpoints.p,out.p,q.p,k.p,v.p,alpha.p,beta.p,heads,128,stride,rows);
        HIP(hipGetLastError());HIP(hipDeviceSynchronize());
        auto ref_cp=checkpoints.read(),ref_out=out.read();
        exact(original,state.read(),"reference mutated canonical state");
        checkpoints.upload(sentinel_cp);out.upload(sentinel_out);
        deltanet_step_batch_gda_verify_128_f32<<<dim3(heads,1,32),dim3(32,4)>>>(
            state.p,checkpoints.p,out.p,q.p,k.p,v.p,alpha.p,beta.p,heads,128,stride,rows);
        HIP(hipGetLastError());HIP(hipDeviceSynchronize());
        exact(ref_cp,checkpoints.read(),"rollback checkpoints");
        exact(ref_out,out.read(),"output");exact(original,state.read(),"canonical state");
        for(size_t i=rows*state_size;i<ref_cp.size();++i) if(ref_cp[i]!=-999) return 5;
        for(size_t i=q.n;i<ref_out.size();++i) if(ref_out[i]!=-999) return 6;
        ++cases;
    }
    std::printf("DeltaNet verifier: %d GPU cases, outputs/checkpoints/state bit-identical PASS\n",cases);
}
'''
tmp_root = root / 'tmp'
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='deltanet-verifier-', dir=tmp_root) as directory:
    path = Path(directory)
    (path/'test.hip').write_text(program)
    rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm/core-10.0'))
    subprocess.run([str(rocm/'bin/hipcc'), '-O3', '-ffast-math', '--offload-arch=gfx1201',
                    str(path/'test.hip'), '-Wl,-rpath,'+str(rocm/'lib'), '-o', str(path/'test')],
                   check=True, env=os.environ | {'TMPDIR': str(path)})
    subprocess.run([str(path/'test')], check=True)
