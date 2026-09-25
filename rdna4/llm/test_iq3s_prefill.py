"""GPU parity for IQ3_S prefill row tiling against the scalar native kernel."""
import os
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parent
program = r'''
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "qwen35_matvec_iq.hip"
#define HIP(call) do { auto e=(call); if(e!=hipSuccess) {std::fprintf(stderr,"%s\n",hipGetErrorString(e));std::exit(2);} }while(0)
template<class T> struct Buffer {
    T *p;
    explicit Buffer(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
    ~Buffer() { HIP(hipFree(p)); }
    void upload(const std::vector<T>& v) { HIP(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
};
int main() {
    hipDeviceProp_t props; HIP(hipGetDeviceProperties(&props,0));
    if(std::strncmp(props.gcnArchName,"gfx1201",7)) return 2;
    uint32_t state=42;
    auto rng=[&]() { state=1664525*state+1013904223;return state; };
    int cases=0;
    for(int cols:{256,5120}) for(int rows:{5,48}) for(int count:{1,7,8,9,189,512}) {
        std::vector<unsigned char> w(size_t(rows)*(cols/256)*110);
        for(auto &v:w) v=rng()>>24;
        for(size_t i=0;i<w.size();i+=110) {uint16_t d=0x2000+(rng()%1024); std::memcpy(w.data()+i,&d,2);}
        std::vector<signed char> q(size_t(count)*cols);
        for(auto &v:q) v=int(rng()%255)-127;
        std::vector<float> s(size_t(count)*(cols/32));
        for(auto &v:s) v=(1+rng()%1000)/65536.0f;
        Buffer<unsigned char> dw(w.size()); Buffer<signed char> dq(q.size()); Buffer<float> ds(s.size());
        Buffer<float> da(size_t(count)*rows), db(size_t(count)*rows);
        dw.upload(w);dq.upload(q);ds.upload(s);
        HIP(hipMemset(da.p,0xff,size_t(count)*rows*4));HIP(hipMemset(db.p,0xff,size_t(count)*rows*4));
        qwen35_matvec_iq3s<<<dim3((rows+3)/4,count),128>>>(da.p,dw.p,dq.p,ds.p,rows,cols);
        qwen35_matvec_iq3s_prefill8<<<dim3((rows+3)/4,(count+7)/8),128>>>(db.p,dw.p,dq.p,ds.p,rows,cols,count);
        HIP(hipGetLastError());HIP(hipDeviceSynchronize());
        std::vector<float>a(size_t(count)*rows),b(a.size());
        HIP(hipMemcpy(a.data(),da.p,a.size()*4,hipMemcpyDeviceToHost));
        HIP(hipMemcpy(b.data(),db.p,b.size()*4,hipMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();++i) if(!std::isfinite(a[i]) || std::memcmp(&a[i],&b[i],4)) {
            std::fprintf(stderr,"FAIL rows=%d cols=%d count=%d i=%zu %.9g %.9g\n",rows,cols,count,i,a[i],b[i]);return 1;
        }
        ++cases;
    }
    std::printf("IQ3_S prefill tiling: %d GPU cases bit-identical PASS\n",cases);
}
'''
tmp=root/'tmp';tmp.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='iq3s-prefill-',dir=tmp) as directory:
    p=Path(directory);(p/'test.hip').write_text(program)
    rocm=Path(os.environ.get('ROCM_PATH','/opt/rocm/core-10.0'))
    subprocess.run([str(rocm/'bin/hipcc'),'-O3','--offload-arch=gfx1201','-I'+str(root),str(p/'test.hip'),'-Wl,-rpath,'+str(rocm/'lib'),'-o',str(p/'test')],check=True,env=os.environ|{'TMPDIR':str(p)})
    subprocess.run([str(p/'test')],check=True)
