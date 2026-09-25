#!/usr/bin/env python3
"""Check fixed-shape GDA against generic order under runner fast-math flags."""
import argparse
import ast
import os
from pathlib import Path
import re
import subprocess


PROGRAM = r'''
#define CK(x) do { auto e=(x); if(e){printf("HIP error %d line %d\n",e,__LINE__);return 2;} } while(0)
int main() {
    std::mt19937 rng(20260920);
    size_t checked = 0;
    constexpr int dt = 48, ds = 128;
    for (int contract : {0, 1})
    for (int M : {1, 2, 7, 8}) for (int pattern : {0, 1, 2}) {
        size_t state_n = (size_t)dt*ds*ds, row_n = (size_t)M*dt*ds;
        std::vector<float> state(state_n), q(row_n), k(row_n), v(row_n);
        std::vector<float> alpha((size_t)M*dt), beta((size_t)M*dt);
        for (size_t i=0; i<state_n; ++i)
            state[i] = pattern == 1 ? 0.0f : (int(rng()%20001)-10000)*0.00001f;
        for (size_t i=0; i<row_n; ++i) {
            q[i] = pattern == 2 ? ((i&1) ? -0.125f : 0.125f) :
                (int(rng()%20001)-10000)*0.00002f;
            k[i] = (int(rng()%20001)-10000)*0.00002f;
            v[i] = (int(rng()%20001)-10000)*0.00002f;
        }
        for (size_t i=0; i<alpha.size(); ++i) {
            alpha[i] = contract ? 0.5f + (rng()%5001)*0.0001f :
                                  -int(rng()%10001)*0.00001f;
            beta[i] = (rng()%10001)*0.0001f;
        }
        float *sa,*sb,*qa,*ka,*va,*aa,*ba,*oa,*ob;
        CK(hipMalloc(&sa,state_n*4)); CK(hipMalloc(&sb,state_n*4));
        CK(hipMalloc(&qa,row_n*4)); CK(hipMalloc(&ka,row_n*4));
        CK(hipMalloc(&va,row_n*4)); CK(hipMalloc(&aa,alpha.size()*4));
        CK(hipMalloc(&ba,beta.size()*4)); CK(hipMalloc(&oa,row_n*4));
        CK(hipMalloc(&ob,row_n*4));
        CK(hipMemcpy(sa,state.data(),state_n*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(sb,state.data(),state_n*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(qa,q.data(),row_n*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(ka,k.data(),row_n*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(va,v.data(),row_n*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(aa,alpha.data(),alpha.size()*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(ba,beta.data(),beta.size()*4,hipMemcpyHostToDevice));
        /* The generic kernel's zero flag consumes raw alpha and calls expf.
         * The fixed scalar kernel deliberately retains the original launcher
         * contract: its non-zero flag performs that same expf operation. */
        int stride=dt*ds;
        int generic_contract = contract ? 1 : 0;
        int fixed_contract = contract ? 2 : 1;
        dim3 grid(dt,1,(ds+3)/4), block(32,4,1);
        deltanet_step_batch_gda_ref_f32<<<grid,block>>>(
            sa,oa,qa,ka,va,aa,ba,dt,ds,stride,M,generic_contract);
        deltanet_step_batch_gda_ref_128_f32<<<grid,block>>>(
            sb,ob,qa,ka,va,aa,ba,dt,ds,stride,M,fixed_contract);
        CK(hipDeviceSynchronize());
        std::vector<float> ha(state_n),hb(state_n),ya(row_n),yb(row_n);
        CK(hipMemcpy(ha.data(),sa,state_n*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(hb.data(),sb,state_n*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(ya.data(),oa,row_n*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(yb.data(),ob,row_n*4,hipMemcpyDeviceToHost));
        for (size_t i=0; i<state_n; ++i) if (memcmp(&ha[i],&hb[i],4)) {
            printf("state mismatch contract=%d M=%d pattern=%d i=%zu %.9g %.9g\n",contract,M,pattern,i,ha[i],hb[i]); return 1;
        }
        for (size_t i=0; i<row_n; ++i) if (memcmp(&ya[i],&yb[i],4)) {
            printf("output mismatch contract=%d M=%d pattern=%d i=%zu %.9g %.9g\n",contract,M,pattern,i,ya[i],yb[i]); return 1;
        }
        checked += state_n + row_n;
        CK(hipFree(sa)); CK(hipFree(sb)); CK(hipFree(qa)); CK(hipFree(ka));
        CK(hipFree(va)); CK(hipFree(aa)); CK(hipFree(ba)); CK(hipFree(oa)); CK(hipFree(ob));
    }
    printf("PASS: %zu bitwise GDA state/output comparisons\n",checked);
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

    headers = ("#include <hip/hip_runtime.h>\n#include <vector>\n#include <random>\n"
               "#include <cstdio>\n#include <cstring>\n")
    headers += "#define DLN_BATCH_MAX_DSTATE 128\n"
    kernels = ("deltanet_step_batch_gda_ref_f32",
               "deltanet_step_batch_gda_ref_128_f32")
    path = args.out / "test.cu"
    path.write_text(headers + "".join(extract(k) for k in kernels) + PROGRAM)
    subprocess.run([args.hipcc, "-O3", "-ffast-math", "--offload-arch=gfx1201", str(path),
                    "-o", str(args.out / "test")], check=True,
                   env=dict(os.environ, TMPDIR=str(args.out.resolve())))


if __name__ == "__main__":
    main()
