/*
 * moe_gpu_kernels.cu — Simple GPU kernels for MoE decode.
 * No device globals needed — all grid data from NVRTC module.
 *
 * Compile: nvcc -cubin -arch=sm_120 -o moe_gpu_kernels.cubin moe_gpu_kernels.cu
 * Load at runtime via cuModuleLoadData (works because no __device__ vars).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
 * GPU top-k: pick top n_used from n_experts, write indices + sigmoid weights.
 * Launched: grid = 1, block = 128
 */
extern "C" __global__ void moe_topk_gpu(
    int * __restrict__ idx_out,
    float * __restrict__ wgt_out,
    const float * __restrict__ logits,
    int n_experts, int n_used)
{
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    float tv[8]; int ti[8];
    for (int i = 0; i < n_used; i++) { tv[i] = -1e38f; ti[i] = -1; }
    int es = warp * (n_experts / 4);
    int ee = es + (n_experts / 4);
    if (warp == 3) ee = n_experts;
    for (int e = es + lane; e < ee; e += 32) {
        float v = logits[e];
        if (v > tv[n_used-1]) {
            tv[n_used-1] = v; ti[n_used-1] = e;
            for (int i = n_used-2; i >= 0 && tv[i+1] > tv[i]; i--) {
                float xv = tv[i]; int xi = ti[i];
                tv[i] = tv[i+1]; ti[i] = ti[i+1];
                tv[i+1] = xv; ti[i+1] = xi;
            }
        }
    }

    __shared__ float slv[32];
    __shared__ int sli[32];
    int wp = warp * 8;
    for (int i = lane; i < 8; i += 32) { slv[wp+i] = tv[i]; sli[wp+i] = ti[i]; }
    __syncthreads();

    if (warp == 0 && lane < n_used) {
        float fv[8]; int fi[8];
        for (int i = 0; i < n_used; i++) { fv[i] = slv[i]; fi[i] = sli[i]; }
        for (int w = 1; w < 4; w++) {
            int off = w * 8;
            for (int k = 0; k < n_used; k++) {
                int idx = sli[off + k];
                if (idx < 0) continue;
                float v = slv[off + k];
                if (v > fv[n_used-1]) {
                    fv[n_used-1] = v; fi[n_used-1] = idx;
                    for (int ii = n_used-2; ii >= 0 && fv[ii+1] > fv[ii]; ii--) {
                        float xv = fv[ii]; int xi = fi[ii];
                        fv[ii] = fv[ii+1]; fi[ii] = fi[ii+1];
                        fv[ii+1] = xv; fi[ii+1] = xi;
                    }
                }
            }
        }
        idx_out[lane] = fi[lane];
        wgt_out[lane] = 1.0f / (1.0f + expf(-fv[lane]));
    }
}

/*
 * GPU shared gate: compute sigmoid(dot(xb, gate_w)), write to float[1].
 * Launched: grid = 1, block = 128
 */
extern "C" __global__ void moe_shared_gate_gpu(
    float * __restrict__ gate_out,
    const float * __restrict__ xb,
    const float * __restrict__ gate_w,
    int n_embd)
{
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += 128)
        s += gate_w[i] * xb[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) *gate_out = 1.0f / (1.0f + expf(-s));
}
