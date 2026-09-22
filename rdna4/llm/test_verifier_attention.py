"""GPU bit-parity/timing check of production Q8 verifier attention at 4K–64K.

Runs kernels directly, independent of DFlash's >=32K target-only fallback.
Requires ROCm gfx1201. This checks attention, not full-model state semantics.
"""
import ast
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
gate = ''.join(ast.literal_eval(line.rstrip(';')) for line in
               (root / 'qwen35_attention_q8_gate.inc').read_text().splitlines()
               if line.startswith('"'))
program = r'''
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "qwen35_attention_q8.hip"
''' + gate + r'''
#define HIP(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); } } while (0)
template<class T> struct Buffer {
    T *p;
    explicit Buffer(size_t n) { HIP(hipMalloc(&p, n * sizeof(T))); }
    ~Buffer() { HIP(hipFree(p)); }
    void upload(const std::vector<T>& v) {
        HIP(hipMemcpy(p, v.data(), v.size()*sizeof(T), hipMemcpyHostToDevice));
    }
};
int main() {
    hipDeviceProp_t props; HIP(hipGetDeviceProperties(&props, 0));
    if (std::strncmp(props.gcnArchName, "gfx1201", 7)) return 2;
    constexpr int heads=24, kvheads=4, maxq=16, capacity=65536+maxq, splits=128;
    uint32_t rng=42;
    auto random = [&]() { rng=1664525*rng+1013904223; return rng; };
    std::vector<signed char> codes(size_t(capacity)*kvheads*256);
    Buffer<signed char> k(codes.size()), v(codes.size());
    for (auto &x: codes) x = int(random()%255)-127;
    k.upload(codes);
    for (auto &x: codes) x = int(random()%255)-127;
    v.upload(codes);
    std::vector<q8_half> scales(size_t(capacity)*kvheads*8);
    for (auto &x: scales) x = q8_half((1+random()%16)/2048.0f);
    Buffer<q8_half> ks(scales.size()), vs(scales.size());
    ks.upload(scales); vs.upload(scales);
    std::vector<float> host(size_t(maxq)*heads*256);
    Buffer<float> q(host.size()), gate(host.size()), out(host.size());
    for (auto &x: host) x = (int(random()%4096)-2048)/1024.0f;
    q.upload(host);
    for (auto &x: host) x = (int(random()%4096)-2048)/1024.0f;
    gate.upload(host);
    Buffer<float> parts(size_t(maxq)*heads*splits*256);
    Buffer<float2> meta(size_t(maxq)*heads*splits);
    Buffer<int> positions(maxq);
    hipEvent_t begin, end; HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
    int cases=0;
    for (int depth: {4096,16384,65536})
    for (int rows: {4,5,8,16}) {
        std::vector<int> pos(rows);
        for (int i=0;i<rows;++i) pos[i]=depth+i;
        positions.upload(pos);
        std::vector<float> reference(size_t(rows)*heads*256), actual(reference.size());
        for (int mode=0;mode<3;++mode) {
            if (mode==2 && rows!=5 && rows!=8) continue;
            auto launch = [&]() {
                if (mode==2) {
                    if (rows==5)
                        qwen35_attention_q8_decode_reuse_fixed5<<<dim3(1,splits,heads),dim3(32,4)>>>(
                            out.p,parts.p,meta.p,q.p,k.p,v.p,ks.p,vs.p,positions.p,
                            heads,kvheads,props.multiProcessorCount,11,0,rows,-1);
                    else qwen35_attention_q8_decode_reuse_fixed8<<<dim3(1,splits,heads),dim3(32,4)>>>(
                        out.p,parts.p,meta.p,q.p,k.p,v.p,ks.p,vs.p,positions.p,
                        heads,kvheads,props.multiProcessorCount,11,0,rows,-1);
                    qwen35_attention_q8_combine_gate<<<dim3(heads,rows),256,splits*2*sizeof(float)>>>(
                        out.p,parts.p,meta.p,gate.p,positions.p,heads,props.multiProcessorCount,11,0);
                } else {
                    if (rows<=8)
                        qwen35_attention_q8_decode_reuse8<<<dim3(1,splits,heads),dim3(32,4)>>>(
                            out.p,parts.p,meta.p,q.p,k.p,v.p,ks.p,vs.p,positions.p,
                            heads,kvheads,props.multiProcessorCount,11,0,rows,-1);
                    else
                        qwen35_attention_q8_decode<<<dim3(rows,splits,heads),dim3(32,4)>>>(
                            out.p,parts.p,meta.p,q.p,k.p,v.p,ks.p,vs.p,positions.p,
                            heads,kvheads,props.multiProcessorCount,11,0,rows,-1);
                    if (mode==0)
                        qwen35_attention_q8_combine_gate<<<dim3(heads,rows),256,splits*2*sizeof(float)>>>(
                            out.p,parts.p,meta.p,gate.p,positions.p,heads,props.multiProcessorCount,11,0);
                    else if (rows==4)
                        qwen35_attention_q8_combine_verify4_gate<<<heads,256>>>(
                            out.p,parts.p,meta.p,gate.p,positions.p,heads,props.multiProcessorCount,11,0,rows);
                    else if (rows<=8)
                        qwen35_attention_q8_combine_verify8_gate<<<heads,256>>>(
                            out.p,parts.p,meta.p,gate.p,positions.p,heads,props.multiProcessorCount,11,0,rows);
                    else
                        qwen35_attention_q8_combine_verify16_gate<<<heads,256>>>(
                            out.p,parts.p,meta.p,gate.p,positions.p,heads,props.multiProcessorCount,11,0,rows);
                }
                HIP(hipGetLastError());
            };
            HIP(hipMemset(out.p,0xff,host.size()*sizeof(float)));
            launch(); HIP(hipDeviceSynchronize());
            HIP(hipMemcpy(actual.data(),out.p,actual.size()*sizeof(float),hipMemcpyDeviceToHost));
            for (float x: actual) if (!std::isfinite(x)) return 3;
            if (!mode) reference=actual;
            else if (std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(float))) {
                size_t different=0; float error=0;
                for (size_t i=0;i<actual.size();++i) {
                    different += std::memcmp(&actual[i],&reference[i],sizeof(float))!=0;
                    error=std::fmax(error,std::fabs(actual[i]-reference[i]));
                }
                std::fprintf(stderr,"FAIL depth=%d rows=%d mode=%d different=%zu max_abs=%g\n",
                             depth,rows,mode,different,error);
                return 4;
            }
            HIP(hipEventRecord(begin));
            for (int repeat=0;repeat<10;++repeat) launch();
            HIP(hipEventRecord(end)); HIP(hipEventSynchronize(end));
            float ms; HIP(hipEventElapsedTime(&ms,begin,end));
            std::printf("PASS depth=%d rows=%d mode=%d ms=%.6f\n",depth,rows,mode,ms/10);
            std::fflush(stdout); ++cases;
        }
    }
    HIP(hipEventDestroy(begin)); HIP(hipEventDestroy(end));
    std::printf("Verifier attention: %d GPU cases bit-identical PASS\n",cases);
}
'''
tmp_root = root / 'tmp'
tmp_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='verifier-attention-', dir=tmp_root) as directory:
    path = Path(directory)
    source = path / 'test.hip'
    source.write_text(program)
    compiler = Path(os.environ.get('ROCM_PATH', '/opt/rocm/core-10.0')) / 'bin/hipcc'
    subprocess.run([str(compiler), '-O3', '--offload-arch=gfx1201', '-I'+str(root),
                    str(source), '-Wl,-rpath,'+str(compiler.parent.parent/'lib'), '-o', str(path/'test')], check=True,
                   env=os.environ | {'TMPDIR': str(path)})
    subprocess.run([str(path/'test')], check=True)
