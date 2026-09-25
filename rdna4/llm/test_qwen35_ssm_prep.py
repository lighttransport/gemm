#!/usr/bin/env python3
"""Build a bitwise check of fused SSM preparation against the scalar chain."""
import argparse
import ast
import os
from pathlib import Path
import re
import subprocess

PROGRAM = r'''
#define CK(x) do{auto e=(x);if(e){printf("HIP error %d line %d\n",e,__LINE__);return 2;}}while(0)
int main(){
 std::mt19937 rng(7193);size_t checked=0;
 for(int ds:{128,256})for(int ng:{4,16}){
  int dt=48,ck=4,qd=2*ng*ds+dt*ds;float eps=1e-6f;
  std::vector<int> sizes={qd,(ck-1)*qd,qd,qd*ck,dt,dt,dt,dt,dt*ds,dt*ds};
  float *a[10],*b[10];std::vector<float> h;
  for(int k=0;k<10;++k){h.resize(sizes[k]);for(auto &x:h)x=(int(rng()%20001)-10000)*.0001f;
   CK(hipMalloc(&a[k],sizes[k]*4));CK(hipMalloc(&b[k],sizes[k]*4));
   CK(hipMemcpy(a[k],h.data(),sizes[k]*4,hipMemcpyHostToDevice));CK(hipMemcpy(b[k],h.data(),sizes[k]*4,hipMemcpyHostToDevice));}
  for(int step=0;step<16;++step){
   softplus_mul_f32<<<1,256>>>(a[4],a[4],a[6],a[7],dt);sigmoid_inplace_f32<<<1,256>>>(a[5],dt);
   conv1d_depthwise_silu_f32<<<(qd+255)/256,256>>>(a[0],a[1],a[2],a[3],qd,ck);
   l2_norm_heads_f32<<<ng,ds,ds*4>>>(a[0],ng,ds,eps);
   l2_norm_heads_f32<<<ng,ds,ds*4>>>(a[0]+ng*ds,ng,ds,eps);
   repeat_tile_f32<<<(dt*ds+255)/256,256>>>(a[8],a[0],dt,ds,ng);
   repeat_tile_f32<<<(dt*ds+255)/256,256>>>(a[9],a[0]+ng*ds,dt,ds,ng);
   ssm_prep_f32<<<ng,256>>>(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],qd,ck,ds,ng,dt,eps);
   CK(hipDeviceSynchronize());
   for(int k:{0,1,4,5,8,9}){std::vector<float>x(sizes[k]),y(sizes[k]);
    CK(hipMemcpy(x.data(),a[k],sizes[k]*4,hipMemcpyDeviceToHost));CK(hipMemcpy(y.data(),b[k],sizes[k]*4,hipMemcpyDeviceToHost));
    for(int i=0;i<sizes[k];++i)if(memcmp(&x[i],&y[i],4)){printf("mismatch ds=%d ng=%d step=%d field=%d i=%d %.9g %.9g\n",ds,ng,step,k,i,x[i],y[i]);return 1;}
    checked+=sizes[k];}
  }
  for(int k=0;k<10;++k){CK(hipFree(a[k]));CK(hipFree(b[k]));}
 }

 printf("PASS %zu fused SSM prep bitwise values\n",checked);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hipcc", default="/opt/rocm/core-10.0/bin/hipcc")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("hip_llm_runner.c").read_text()
    source = "".join(ast.literal_eval(x) for x in
        re.findall(r'^\s*("(?:[^"\\]|\\.)*")', runner, re.M))

    def extract(name):
        start = source.index("__global__ void " + name + "(")
        end = source.index("{", start) + 1
        depth = 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end] + "\n"

    kernels = ("softplus_mul_f32", "sigmoid_inplace_f32", "conv1d_depthwise_silu_f32",
               "l2_norm_heads_f32", "repeat_tile_f32", "ssm_prep_f32")
    headers = "#include <hip/hip_runtime.h>\n#include <vector>\n#include <random>\n#include <cstdio>\n#include <cstring>\n"
    path = args.out / "test.cu"
    path.write_text(headers + "".join(extract(k) for k in kernels) + PROGRAM)
    subprocess.run([args.hipcc, "-O3", "-ffast-math", "--offload-arch=gfx1201",
                    str(path), "-o", str(args.out / "test")], check=True,
                   env=dict(os.environ, TMPDIR=str(args.out.resolve())))


if __name__ == "__main__":
    main()
