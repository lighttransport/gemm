"""GPU bit-parity test for the eight-row Q4_K draft specialization."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
source = (root / 'qwen35_matvec_q2k.hip').read_text()
start = source.index('extern "C" __global__ void qwen35_matvec_q4k_q81_multi8(')
end = source.index('/* K=4 DFlash2 windows', start)
program = r'''
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
''' + source[start:end] + r'''
#define HIP(call) do { hipError_t e=(call); if(e!=hipSuccess) { \
    std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
    std::exit(1); } } while(0)
template<class T> struct Buffer {
    T *p;
    explicit Buffer(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
    ~Buffer() { HIP(hipFree(p)); }
    void upload(const std::vector<T>& v) { HIP(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
};
int main() {
    hipDeviceProp_t props; HIP(hipGetDeviceProperties(&props,0));
    if(std::strncmp(props.gcnArchName,"gfx1201",7)) return 2;
    uint32_t rng=42;
    auto random=[&]() { rng=1664525*rng+1013904223;return rng; };
    int cases=0;
    for(int cols: {256,5120,17408})
    for(int rows: {1,7,8,17,1024}) {
        std::vector<unsigned char> weights(size_t(rows)*(cols/256)*144);
        for(auto &x:weights) x=random()>>24;
        for(size_t i=0;i<weights.size();i+=144) {
            _Float16 scale[2]={_Float16((1+random()%16)/4096.0f),
                               _Float16((1+random()%16)/4096.0f)};
            std::memcpy(weights.data()+i,scale,sizeof(scale));
        }
        std::vector<signed char> input(8*cols);
        for(auto &x:input) x=int(random()%255)-127;
        std::vector<float> scales(8*cols/32);
        for(auto &x:scales) x=float(_Float16((1+random()%32)/1024.0f));
        Buffer<unsigned char> w(weights.size());w.upload(weights);
        Buffer<signed char> q(input.size());q.upload(input);
        Buffer<float> s(scales.size());s.upload(scales);
        constexpr int guard=32;
        std::vector<float> reference(8*rows+guard,-999),actual(reference);
        Buffer<float> output(reference.size());output.upload(reference);
        qwen35_matvec_q4k_q81_multi8<<<(rows+7)/8,256>>>(output.p,w.p,q.p,s.p,rows,cols,8);
        HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
        HIP(hipMemcpy(reference.data(),output.p,reference.size()*sizeof(float),hipMemcpyDeviceToHost));
        output.upload(actual);
        qwen35_matvec_q4k_q81_fixed8<<<(rows+7)/8,256>>>(output.p,w.p,q.p,s.p,rows,cols,8);
        HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
        HIP(hipMemcpy(actual.data(),output.p,actual.size()*sizeof(float),hipMemcpyDeviceToHost));
        for(float x:actual) if(!std::isfinite(x)) return 3;
        if(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float))) {
            std::fprintf(stderr,"FAIL rows=%d cols=%d\n",rows,cols); return 4;
        }
        for(size_t i=8*rows;i<actual.size();++i) if(actual[i]!=-999) return 5;
        ++cases;
    }
    std::printf("DFlash fixed8: %d GPU shapes bit-identical, guards intact PASS\n",cases);
}
'''
tmp_root = root / 'tmp'
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='dflash-fixed8-', dir=tmp_root) as directory:
    path = Path(directory)
    (path/'test.hip').write_text(program)
    rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm/core-10.0'))
    subprocess.run([str(rocm/'bin/hipcc'), '-O3', '--offload-arch=gfx1201',
                    str(path/'test.hip'), '-Wl,-rpath,'+str(rocm/'lib'), '-o', str(path/'test')],
                   check=True, env=os.environ | {'TMPDIR': str(path)})
    subprocess.run([str(path/'test')], check=True)
