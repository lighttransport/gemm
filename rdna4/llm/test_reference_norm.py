"""Compile the runner RMSNorm and the pinned llama.cpp kernel on identical inputs."""
import argparse
import ast
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hipcc", default="/opt/rocm/core-10.0/bin/hipcc")
    args = p.parse_args()
    args.llama = args.llama.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    ours = "".join(ast.literal_eval(s.rstrip(";")) for s in
                   (here / "qwen35_reference_math.h").read_text().splitlines() if s.startswith('"'))
    reference = (args.llama / "ggml/src/ggml-cuda/norm.cu").read_text()
    start = reference.rfind("template", 0, reference.index("static __global__ void rms_norm_f32("))
    end = reference.index("template <int block_size>\nstatic __global__ void rms_norm_back_f32", start)
    conv = (args.llama / "ggml/src/ggml-cuda/ssm-conv.cu").read_text()
    conv = conv[conv.index("template <bool apply_silu"):conv.index("template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t")]
    program = '#include "common.cuh"\n#include "unary.cuh"\n#include <vector>\n#include <random>\n#include <cstdio>\n#include <cstring>\n'
    program += ours + reference[start:end] + conv + r'''
#define CHECK(x) do { auto error=(x); if(error!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(error));return 2;} }while(0)
int main() {
    size_t checked=0;
    std::mt19937 rng(7319);
    for(bool weighted : {false,true})
    for(int n : {128,256,1024,5120,6144,17408}) for(int rows : {1,7,512,513}) {
        std::vector<float> x(n*rows),w(n),a(n*rows),b(n*rows);
        for(auto &v:x) v=(int(rng()%20000)-10000)*.001f;
        for(auto &v:w) v=(int(rng()%20000)-10000)*.001f;
        float *dx,*dw,*da,*db;
        CHECK(hipMalloc(&dx,x.size()*4));CHECK(hipMalloc(&dw,w.size()*4));
        CHECK(hipMalloc(&da,a.size()*4));CHECK(hipMalloc(&db,b.size()*4));
        CHECK(hipMemcpy(dx,x.data(),x.size()*4,hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dw,w.data(),w.size()*4,hipMemcpyHostToDevice));
        int threads=n<1024?256:1024;
        qwen35_rmsnorm_reference<<<rows,threads>>>(da,dx,weighted?dw:nullptr,n,n,1e-6f);
        uint3 nc=init_fastdiv_values(n), one=init_fastdiv_values(1);
        if(n<1024 && weighted) rms_norm_f32<256,true><<<rows,256,128>>>(dx,db,n,n,n*rows,n*rows,1e-6f,dw,0,0,0,nc,one,one,one);
        else if(n<1024) rms_norm_f32<256,false><<<rows,256,128>>>(dx,db,n,n,n*rows,n*rows,1e-6f,nullptr,0,0,0,nc,one,one,one);
        else if(weighted) rms_norm_f32<1024,true><<<rows,1024,128>>>(dx,db,n,n,n*rows,n*rows,1e-6f,dw,0,0,0,nc,one,one,one);
        else rms_norm_f32<1024,false><<<rows,1024,128>>>(dx,db,n,n,n*rows,n*rows,1e-6f,nullptr,0,0,0,nc,one,one,one);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(a.data(),da,a.size()*4,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(),db,b.size()*4,hipMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();++i){
            if(memcmp(&a[i],&b[i],4)){fprintf(stderr,"mismatch n=%d rows=%d i=%zu %.9g %.9g\n",n,rows,i,a[i],b[i]);return 1;}
            ++checked;
        }
        CHECK(hipFree(dx));CHECK(hipFree(dw));CHECK(hipFree(da));CHECK(hipFree(db));
    }
    printf("PASS: %zu bitwise RMSNorm comparisons\n",checked);
    checked=0;
    for(int channels : {128,10240}) for(int rows : {1,4,33,513}) {
        std::vector<float> x(channels*rows),state(channels*3),packed(channels*(rows+3)),w(channels*4),a(x.size()),b(x.size());
        for(auto &v:x) v=(int(rng()%20000)-10000)*.001f;
        for(auto &v:state) v=(int(rng()%20000)-10000)*.001f;
        for(auto &v:w) v=(int(rng()%20000)-10000)*.001f;
        for(int c=0;c<channels;++c) {
            for(int j=0;j<3;++j) packed[c*(rows+3)+j]=state[j*channels+c];
            for(int j=0;j<rows;++j) packed[c*(rows+3)+j+3]=x[j*channels+c];
        }
        float *dx,*ds,*dp,*dw,*da,*db;
        CHECK(hipMalloc(&dx,x.size()*4));CHECK(hipMalloc(&ds,state.size()*4));
        CHECK(hipMalloc(&dp,packed.size()*4));CHECK(hipMalloc(&dw,w.size()*4));
        CHECK(hipMalloc(&da,a.size()*4));CHECK(hipMalloc(&db,b.size()*4));
        CHECK(hipMemcpy(dx,x.data(),x.size()*4,hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ds,state.data(),state.size()*4,hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dp,packed.data(),packed.size()*4,hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dw,w.data(),w.size()*4,hipMemcpyHostToDevice));
        for(int row=0;row<rows;++row)
            qwen35_conv_reference<<<(channels+255)/256,256>>>(da+row*channels,ds,dx+row*channels,dw,channels,4);
        ssm_conv_f32<true,128,4><<<dim3(1,channels/128),128>>>(dp,dw,nullptr,4,(rows+3)*4,channels*(rows+3)*4,16,
            db,4,channels*4,channels*rows*4,rows);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(a.data(),da,a.size()*4,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(),db,b.size()*4,hipMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();++i) {
            if(memcmp(&a[i],&b[i],4)){fprintf(stderr,"conv mismatch channels=%d rows=%d i=%zu %.9g %.9g\n",channels,rows,i,a[i],b[i]);return 1;}
            ++checked;
        }
        CHECK(hipFree(dx));CHECK(hipFree(ds));CHECK(hipFree(dp));CHECK(hipFree(dw));CHECK(hipFree(da));CHECK(hipFree(db));
    }
    printf("PASS: %zu bitwise convolution/SiLU comparisons\n",checked);
}
'''
    source = args.out / "test.cu"
    source.write_text(program)
    env = dict(os.environ, TMPDIR=str(args.out.resolve()))
    subprocess.run([args.hipcc, "-O3", "--offload-arch=gfx1201", "-DGGML_USE_HIP",
                    "-I" + str(args.llama / "ggml/include"),
                    "-I" + str(args.llama / "ggml/src"),
                    "-I" + str(args.llama / "ggml/src/ggml-cuda"),
                    str(source), "-L" + str(args.llama.parent / "build/bin"),
                    "-Wl,-rpath," + str(args.llama.parent / "build/bin"), "-lggml-base",
                    "-o", str(args.out / "test")], check=True, env=env)


if __name__ == "__main__":
    main()
