#!/usr/bin/env python3
"""Compare Q2_K/IQ MMVQ and activation quantization with pinned HIP kernels."""
import argparse
import ast
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
    runner = Path(__file__).with_name("hip_llm_runner.c").read_text()
    kernels = "".join(ast.literal_eval(x) for x in
        re.findall(r'^\s*("(?:[^"\\]|\\.)*")', runner, re.M))
    start = kernels.index("__global__ void qwen4_argmax(")
    end = kernels.index("{", start) + 1
    depth = 1
    while depth:
        depth += (kernels[end] == "{") - (kernels[end] == "}")
        end += 1
    program += kernels[start:end]
    program += r'''
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,hipGetErrorString(e));return 2;} }while(0)
int main() {
    std::mt19937 rng(7319);
    size_t quant_checked=0, matrix_checked=0;
    for(int kind : {0,1,2,3,4,5,6})
    for(int cols : {256,512,4096,5120,6144,17408})
    for(int rows : {1,48,129,5120,17408})
    for(int pattern : {0,1,2}) {
        if(kind!=0 && rows==17408 && !(cols==5120 && pattern==0)) continue;
        int nb=cols/256;
        const int block_sizes[] = {84,82,98,110,66,74,136};
        const char *names[] = {"Q2_K","IQ2_S","IQ3_XXS","IQ3_S","IQ2_XXS","IQ2_XS","IQ4_XS"};
        int bs=block_sizes[kind];
        const char *name=names[kind];
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
                else if(kind==3) { REF(GGML_TYPE_IQ3_S,1); }
                else if(kind==4) { REF(GGML_TYPE_IQ2_XXS,1); }
                else if(kind==5) { REF(GGML_TYPE_IQ2_XS,1); }
                else { REF(GGML_TYPE_IQ4_XS,8); }
            } else if(kind==0) {
                if(threads==512) qwen35_matvec_q2k<<<rows,256>>>(da,dw,dq,ds,rows,cols);
                else qwen35_matvec_q2k_rows<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            }
            else if(kind==1) qwen35_matvec_iq2s<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else if(kind==2) qwen35_matvec_iq3xxs<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else if(kind==3) qwen35_matvec_iq3s<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else if(kind==4) qwen35_matvec_iq2xxs<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else if(kind==5) qwen35_matvec_iq2xs<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
            else qwen35_matvec_iq4xs<<<(rows+threads/32-1)/(threads/32),threads>>>(da,dw,dq,ds,rows,cols);
#undef REF
        };
        run(true);
        for(int threads : {32,64,128,256,512,1024}) {
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
        if((rows==5120 || rows==17408) && pattern==0 && (cols==5120 || cols==17408)) {
            for(bool reference : {true,false}) {
                if(reference && threads!=256) continue;
                hipEvent_t start,stop;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
                CHECK(hipEventRecord(start));
                for(int i=0;i<200;++i) run(reference,threads);
                CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
                float elapsed;CHECK(hipEventElapsedTime(&elapsed,start,stop));
                printf("%s %s rows=%d cols=%d threads=%d %.3f us\n",
                    reference?"llama":kind==0 && threads==512?"ours-eight-warps":"ours",name,rows,cols,
                    reference?((kind==0 || kind==6)?256:32):kind==0 && threads==512?256:threads,elapsed*5);
                CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
            }
        }
        }

        if(kind<=5 && pattern==0) {
            signed char *mq; float *ms,*mo,*mr;
            CHECK(hipMalloc(&mq,4*cols));CHECK(hipMalloc(&ms,4*cols/32*4));
            CHECK(hipMalloc(&mo,4*rows*4));CHECK(hipMalloc(&mr,4*rows*4));
            std::vector<signed char> hq(4*cols);std::vector<float> hs(4*cols/32);
            for(int m=0;m<4;++m) {
                for(int j=0;j<cols;++j) hq[m*cols+j]=q[(j+32*m)%cols];
                for(int j=0;j<cols/32;++j) hs[m*cols/32+j]=sc[(j+m)%(cols/32)];
            }
            CHECK(hipMemcpy(mq,hq.data(),hq.size(),hipMemcpyHostToDevice));
            CHECK(hipMemcpy(ms,hs.data(),hs.size()*4,hipMemcpyHostToDevice));
            int multi_kind=kind==4?0:kind==5?1:kind==1?2:kind==2?3:4;
            auto multi=[&](bool reference,int count) {
                if(reference) for(int m=0;m<count;++m) {
                    auto *o=mr+(size_t)m*rows; auto *q=mq+(size_t)m*cols; auto *sc=ms+(size_t)m*cols/32;
                    if(kind==0) qwen35_matvec_q2k<<<rows,256>>>(o,dw,q,sc,rows,cols);
                    if(kind==1) qwen35_matvec_iq2s<<<(rows+7)/8,256>>>(o,dw,q,sc,rows,cols);
                    if(kind==2) qwen35_matvec_iq3xxs<<<(rows+7)/8,256>>>(o,dw,q,sc,rows,cols);
                    if(kind==3) qwen35_matvec_iq3s<<<(rows+7)/8,256>>>(o,dw,q,sc,rows,cols);
                    if(kind==4) qwen35_matvec_iq2xxs<<<(rows+7)/8,256>>>(o,dw,q,sc,rows,cols);
                    if(kind==5) qwen35_matvec_iq2xs<<<(rows+7)/8,256>>>(o,dw,q,sc,rows,cols);
                } else if(kind==0) qwen35_matvec_q2k_multi4<<<(rows+3)/4,128>>>(mo,dw,mq,ms,rows,cols,count);
                else qwen35_matvec_iq_multi4<<<(rows+3)/4,128>>>(mo,dw,mq,ms,rows,cols,multi_kind,count);
            };
            for(int count : {2,3,4}) {
                multi(true,count);multi(false,count);
                CHECK(hipDeviceSynchronize());
                std::vector<float> ao(count*rows),ar(count*rows);
                CHECK(hipMemcpy(ao.data(),mo,ao.size()*4,hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(ar.data(),mr,ar.size()*4,hipMemcpyDeviceToHost));
                if(memcmp(ao.data(),ar.data(),ao.size()*4)) {
                    fprintf(stderr,"multi4 mismatch kind=%d rows=%d cols=%d count=%d\n",kind,rows,cols,count);return 1;
                }
                matrix_checked+=ao.size();
                if(count==4 && rows==5120 && (cols==5120 || cols==17408)) {
                    for(bool reference : {true,false}) {
                        hipEvent_t start,stop; CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
                        CHECK(hipEventRecord(start));
                        for(int rep=0;rep<100;++rep) multi(reference,count);
                        CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));
                        float ms;CHECK(hipEventElapsedTime(&ms,start,stop));
                        printf("multi4 %s %s rows=%d cols=%d %.3f us/window\n",name,reference?"scalar":"shared",rows,cols,ms*10);
                        CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
                    }
                }

            }
            CHECK(hipFree(mq));CHECK(hipFree(ms));CHECK(hipFree(mo));CHECK(hipFree(mr));
        }
        CHECK(hipFree(dw));CHECK(hipFree(dx));CHECK(hipFree(ds));CHECK(hipFree(dq));
        CHECK(hipFree(drq));CHECK(hipFree(da));CHECK(hipFree(db));
    }

    for(int n : {1,31,255,256,257,4095,4096,4097,248320,1048576})
    for(int pattern : {0,1,2,3}) {
        std::vector<float> x(n);
        for(auto &v:x) v=pattern==1?0.0f:pattern==2?-3.402823466e38f:pattern==3?-INFINITY:(int(rng()%20000)-10000)*.001f;
        int expected=0;
        for(int i=1;i<n;++i)if(x[i]>x[expected])expected=i;
        float *dx,*scores;int *indices,*a,*b;int groups=(n+4095)/4096;
        CHECK(hipMalloc(&dx,n*4));CHECK(hipMalloc(&scores,groups*4));CHECK(hipMalloc(&indices,groups*4));
        CHECK(hipMalloc(&a,4));CHECK(hipMalloc(&b,4));CHECK(hipMemcpy(dx,x.data(),n*4,hipMemcpyHostToDevice));
        auto run=[&](bool reference){
            if(reference) qwen4_argmax<<<1,256>>>(dx,n,a);
            else {qwen35_argmax_parts<<<groups,256>>>(dx,n,scores,indices);qwen35_argmax_finish<<<1,256>>>(scores,indices,groups,b);}
        };
        run(true);run(false);CHECK(hipDeviceSynchronize());int ha,hb;
        CHECK(hipMemcpy(&ha,a,4,hipMemcpyDeviceToHost));CHECK(hipMemcpy(&hb,b,4,hipMemcpyDeviceToHost));
        if(ha!=hb || hb!=expected){fprintf(stderr,"argmax mismatch n=%d pattern=%d %d %d %d\n",n,pattern,ha,hb,expected);return 1;}
        if(n==248320 && pattern==0) for(bool reference : {true,false}) {
            hipEvent_t start,stop;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));
            CHECK(hipEventRecord(start));for(int i=0;i<100;++i)run(reference);
            CHECK(hipEventRecord(stop));CHECK(hipEventSynchronize(stop));float ms;CHECK(hipEventElapsedTime(&ms,start,stop));
            printf("argmax %s %.3f us\n",reference?"single":"parallel",ms*10);
            CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));
        }
        CHECK(hipFree(dx));CHECK(hipFree(scores));CHECK(hipFree(indices));CHECK(hipFree(a));CHECK(hipFree(b));
    }
    printf("PASS: 40 argmax shapes/patterns, including ties and negative infinity\n");
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
