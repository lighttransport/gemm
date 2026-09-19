#!/usr/bin/env python3
"""Compare Q2_K/IQ MMVQ and activation quantization with pinned HIP kernels."""
import argparse
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hipcc", default="/opt/rocm/core-10.0/bin/hipcc")
    args = parser.parse_args()
    args.llama = args.llama.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    cuda = args.llama / "ggml/src/ggml-cuda"
    mmvq = (cuda / "mmvq.cu").read_text()
    # Extract the original generic template and its compile-time dispatch,
    # without instantiating the full backend's unrelated host entry points.
    program = mmvq[:mmvq.index("static __host__ mmvq_parameter_table_id")]
    program += mmvq[mmvq.index("static constexpr __host__ __device__ int calc_nwarps("):
                    mmvq.index("template <ggml_type type, int c_rows_per_block")]
    quant = (cuda / "quantize.cu").read_text()
    program += quant[quant.index("static __global__ void quantize_q8_1("):
                     quant.index("__device__ __forceinline__ uint8_t compute_e8m0_scale")]
    program += (Path(__file__).parent / "qwen35_matvec_q2k.hip").read_text()
    iq = Path(__file__).parent / "qwen35_matvec_iq.hip"
    program += re.sub(r'#include "([^"]+)"',
                      lambda match: (iq.parent / match[1]).read_text(), iq.read_text())
    program += r'''
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,hipGetErrorString(e));return 2;} }while(0)
int main() {
    std::mt19937 rng(7319);
    size_t quant_checked=0, matrix_checked=0;
    for(int kind : {0,1,2,3})
    for(int cols : {256,512,4096,5120,6144,17408})
    for(int rows : {1,48,129,5120})
    for(int pattern : {0,1,2}) {
        int nb=cols/256;
        int bs=kind==0?84:kind==1?82:kind==2?98:110;
        const char *name=kind==0?"Q2_K":kind==1?"IQ2_S":kind==2?"IQ3_XXS":"IQ3_S";
        std::vector<unsigned char> w((size_t)rows*nb*bs);
        for(auto &v:w) v=rng();
        for(int i=0;i<rows*nb;++i) {
            half d=__float2half((rng()%1000+1)*.0001f);
            half m=__float2half((rng()%1000+1)*.0001f);
            memcpy(w.data()+i*bs+(kind==0?80:0),&d,2);
            if(kind==0) memcpy(w.data()+i*bs+82,&m,2);
        }
        std::vector<float> x(cols),sc(cols/32),a(rows),b(rows);
        for(int i=0;i<cols;++i) {
            x[i]=pattern==1?0.0f:(int(rng()%20000)-10000)*.001f;
            if(pattern==2) x[i]=(i%32==0)?127.0f:((int(rng()%255)-127)+.5f);
        }
        std::vector<signed char> q(cols);
        std::vector<block_q8_1> rq(cols/32);
        unsigned char *dw; float *dx,*ds,*da,*db; signed char *dq; block_q8_1 *drq;
        CHECK(hipMalloc(&dw,w.size()));CHECK(hipMalloc(&dx,cols*4));
        CHECK(hipMalloc(&ds,cols/32*4));CHECK(hipMalloc(&dq,cols));
        CHECK(hipMalloc(&drq,rq.size()*sizeof(block_q8_1)));
        CHECK(hipMalloc(&da,rows*4));CHECK(hipMalloc(&db,rows*4));
        CHECK(hipMemcpy(dw,w.data(),w.size(),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx,x.data(),cols*4,hipMemcpyHostToDevice));
        uint3 one=init_fastdiv_values(1);
        qwen35_quantize_q81<<<cols/32,32>>>(dq,ds,dx,cols);
        quantize_q8_1<<<cols/256,256>>>(dx,drq,cols,cols,cols,cols,cols,1,one);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(q.data(),dq,cols,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(sc.data(),ds,sc.size()*4,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(rq.data(),drq,rq.size()*sizeof(block_q8_1),hipMemcpyDeviceToHost));
        for(int i=0;i<cols;++i) {
            float s=__low2float(rq[i/32].ds);
            if(q[i]!=rq[i/32].qs[i%32] || memcmp(&sc[i/32],&s,4)) {
                fprintf(stderr,"quant mismatch cols=%d pattern=%d i=%d\n",cols,pattern,i);return 1;
            }
            ++quant_checked;
        }
        auto run=[&](bool reference, int threads=256) {
#define REF(TYPE,WARPS) mul_mat_vec_q<TYPE,1,false><<<rows,dim3(32,WARPS)>>>( \
                dw,drq,nullptr,{},db,cols,one,nb,cols/32,rows,one,rows*nb,cols/32,rows, \
                one,rows*nb,cols/32,rows,0)
            if(reference) {
                if(kind==0) { REF(GGML_TYPE_Q2_K,8); }
                else if(kind==1) { REF(GGML_TYPE_IQ2_S,1); }
                else if(kind==2) { REF(GGML_TYPE_IQ3_XXS,1); }
                else { REF(GGML_TYPE_IQ3_S,1); }
            } else if(kind==0) qwen35_matvec_q2k<<<rows,256>>>(da,dw,dq,ds,rows,cols);
            else if(kind==1) qwen35_matvec_iq2s<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else if(kind==2) qwen35_matvec_iq3xxs<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else qwen35_matvec_iq3s<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
#undef REF
        };
        run(true);
        for(int threads : {32,64,128,256}) {
        if(kind==0 && threads!=256) continue;
        run(false,threads);CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(a.data(),da,rows*4,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(),db,rows*4,hipMemcpyDeviceToHost));
        for(int i=0;i<rows;++i) {
            if(memcmp(&a[i],&b[i],4) || !std::isfinite(a[i])) {
                fprintf(stderr,"%s mismatch cols=%d rows=%d pattern=%d threads=%d i=%d %.9g %.9g\n",
                    name,cols,rows,pattern,threads,i,a[i],b[i]);return 1;
            }
            ++matrix_checked;
        }
        if(rows==5120 && pattern==0 && (cols==5120 || cols==17408)) {
            for(bool reference : {true,false}) {
                if(reference && threads!=256) continue;
                hipEvent_t start,stop;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
                CHECK(hipEventRecord(start));
                for(int i=0;i<200;++i) run(reference,threads);
                CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
                float elapsed;CHECK(hipEventElapsedTime(&elapsed,start,stop));
                printf("%s %s rows=%d cols=%d threads=%d %.3f us\n",reference?"llama":"ours",name,rows,cols,
                    reference?(kind==0?256:32):threads,elapsed*5);
                CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
            }
        }
        }
        CHECK(hipFree(dw));CHECK(hipFree(dx));CHECK(hipFree(ds));CHECK(hipFree(dq));
        CHECK(hipFree(drq));CHECK(hipFree(da));CHECK(hipFree(db));
    }
    printf("PASS: %zu bitwise activation quantization and %zu Q2_K/IQ output comparisons\n",quant_checked,matrix_checked);
}
'''
    source = args.out / "test.cu"
    source.write_text(program)
    subprocess.run([args.hipcc, "-O3", "-std=c++17", "--offload-arch=gfx1201", "-DGGML_USE_HIP",
                    "-I" + str(args.llama / "ggml/include"), "-I" + str(args.llama / "ggml/src"),
                    "-I" + str(cuda), str(source), "-L" + str(args.llama.parent / "build/bin"),
                    "-Wl,-rpath," + str(args.llama.parent / "build/bin"), "-lggml-base",
                    "-o", str(args.out / "test")], check=True,
                   env=dict(os.environ, TMPDIR=str(args.out.resolve())))


if __name__ == "__main__":
    main()
