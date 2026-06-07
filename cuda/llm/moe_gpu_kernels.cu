/*
 * moe_gpu_kernels.cu — MoE GPU kernels for 35B-A3B decode.
 * Compile: nvcc -cubin -arch=sm_120 -o moe_gpu_kernels.cubin moe_gpu_kernels.cu
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* Device tables (static so each cubin has its own copy) */
__device__ static const unsigned char grid_ksigns[128] = {
#include "../../common/ksigns_iq2xs.inc"
};
__device__ static const unsigned long long grid_iq2s[1024] = {
#include "../../common/iq2s_grid.inc"
};
__device__ static const unsigned int grid_iq3[256] = {
#include "../../common/iq3xxs_grid.inc"
};

/*
 * moe_topk_gpu: GPU top-k. grid=1, block=128.
 */
extern "C" __global__ void moe_topk_gpu(
    int *idx_out, float *wgt_out,
    const float *logits, int n_experts, int n_used)
{
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    float tv[8]; int ti[8];
    for (int i = 0; i < n_used; i++) { tv[i] = -1e38f; ti[i] = -1; }
    int es = warp * (n_experts / 4), ee = es + (n_experts / 4);
    if (warp == 3) ee = n_experts;
    for (int e = es + lane; e < ee; e += 32) {
        float v = logits[e];
        if (v > tv[n_used-1]) {
            tv[n_used-1] = v; ti[n_used-1] = e;
            for (int i = n_used-2; i >= 0 && tv[i+1] > tv[i]; i--) {
                float xv = tv[i]; int xi = ti[i];
                tv[i] = tv[i+1]; ti[i] = ti[i+1]; tv[i+1] = xv; ti[i+1] = xi;
            }
        }
    }
    __shared__ float slv[32]; __shared__ int sli[32];
    for (int i = lane; i < 8; i += 32) { slv[warp*8+i] = tv[i]; sli[warp*8+i] = ti[i]; }
    __syncthreads();
    if (warp == 0 && lane < n_used) {
        float fv[8]; int fi[8];
        for (int i = 0; i < n_used; i++) { fv[i] = slv[i]; fi[i] = sli[i]; }
        for (int w = 1; w < 4; w++) {
            int off = w * 8;
            for (int k = 0; k < n_used; k++) {
                int idx = sli[off + k]; if (idx < 0) continue;
                float v = slv[off + k];
                if (v > fv[n_used-1]) {
                    fv[n_used-1] = v; fi[n_used-1] = idx;
                    for (int ii = n_used-2; ii >= 0 && fv[ii+1] > fv[ii]; ii--) {
                        float xv = fv[ii]; int xi = fi[ii];
                        fv[ii] = fv[ii+1]; fi[ii] = fi[ii+1]; fv[ii+1] = xv; fi[ii+1] = xi;
                    }
                }
            }
        }
        idx_out[lane] = fi[lane];
        wgt_out[lane] = 1.0f / (1.0f + expf(-fv[lane]));
    }
}

/*
 * moe_shared_gate_gpu: compute sigmoid(dot(xb, gate_w)). grid=1, block=128.
 */
extern "C" __global__ void moe_shared_gate_gpu(
    float *gate_out, const float *xb, const float *gate_w, int n_embd)
{
    float s = 0;
    for (int i = threadIdx.x; i < n_embd; i += 128) s += gate_w[i] * xb[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) *gate_out = 1.0f / (1.0f + expf(-s));
}

/*
 * moe_expert_fused: one expert's full FFN: gate(IQ2_S) + up(IQ2_S) + SiLU + down(IQ3_S).
 * grid = (n_used, 1, 1), block = 128. Each block processes one expert.
 * Device tables passed as params (device globals don't link in cubins via cuModuleLoadData).
 */
extern "C" __global__ void moe_expert_fused(
    float *accum,
    const float *xb,
    const int *expert_idx,
    const float *expert_wgt,
    const unsigned char *gate_exps,
    const unsigned char *up_exps,
    const unsigned char *down_exps,
    int n_embd, int expert_ff,
    size_t stride_gu, size_t stride_d,
    const unsigned char *grid_ksigns,
    const unsigned long long *grid_iq2s,
    const unsigned int *grid_iq3)
{
    int ei = blockIdx.x;
    int eidx = expert_idx[ei];
    float wgt = expert_wgt[ei];
    if (eidx < 0) return;

    __shared__ float sx[2048];
    __shared__ float sg[512];
    __shared__ float su[512];

    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

    /* DEBUG: write grid[0] to verify parameter-passing works */
    if (tid == 0) {
        accum[0] = (float)(unsigned char)(grid_iq2s[0] & 0xff);
        accum[1] = (float)(unsigned char)(grid_ksigns[0]);
    }
    return;

    for (int i = tid; i < n_embd; i += 128) sx[i] = xb[i];
    __syncthreads();

    /* Gate IQ2_S */
    {
        int nb = n_embd / 256, rb = nb * 82;
        for (int i = tid; i < expert_ff; i += 128) sg[i] = 0.0f;
        __syncthreads();
        for (int r = warp; r < expert_ff; r += 4) {
            const unsigned char *rp = gate_exps + (size_t)eidx * stride_gu + (size_t)r * rb;
            float s = 0;
            for (int b = lane; b < nb; b += 32) {
                const unsigned char *bp = rp + (size_t)b * 82;
                float d = __half2float(*(const __half *)bp);
                const unsigned char *qs = bp+2, *qh = bp+66, *sc = bp+74, *sgp = bp+34;
                const float *xx = sx + b * 256;
                float p = 0; int yi = 0;
                for (int ib = 0; ib < 8; ib++) {
                    float da = d * (0.5f + (sc[ib] & 0xf)) * 0.25f;
                    float db = d * (0.5f + (sc[ib] >> 4)) * 0.25f;
                    for (int lx = 0; lx < 4; lx++) {
                        float dl = (lx < 2) ? da : db;
                        int gid = qs[lx] | ((qh[ib] << (8-2*lx)) & 0x300);
                        const unsigned char *grd = (const unsigned char *)&grid_iq2s[gid];
                        unsigned char sn = sgp[lx];
                        for (int j = 0; j < 8; j++)
                            p += (dl * (float)grd[j] * ((sn & (1<<j))?-1.0f:1.0f)) * xx[yi++];
                    }
                    qs += 4; sgp += 4;
                }
                s += p;
            }
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
            if (lane == 0) sg[r] = s;
        }
        __syncthreads();
    }

    /* Up IQ2_S */
    {
        int nb = n_embd / 256, rb = nb * 82;
        for (int i = tid; i < expert_ff; i += 128) su[i] = 0.0f;
        __syncthreads();
        for (int r = warp; r < expert_ff; r += 4) {
            const unsigned char *rp = up_exps + (size_t)eidx * stride_gu + (size_t)r * rb;
            float s = 0;
            for (int b = lane; b < nb; b += 32) {
                const unsigned char *bp = rp + (size_t)b * 82;
                float d = __half2float(*(const __half *)bp);
                const unsigned char *qs = bp+2, *qh = bp+66, *sc = bp+74, *sgp = bp+34;
                const float *xx = sx + b * 256;
                float p = 0; int yi = 0;
                for (int ib = 0; ib < 8; ib++) {
                    float da = d * (0.5f + (sc[ib] & 0xf)) * 0.25f;
                    float db = d * (0.5f + (sc[ib] >> 4)) * 0.25f;
                    for (int lx = 0; lx < 4; lx++) {
                        float dl = (lx < 2) ? da : db;
                        int gid = qs[lx] | ((qh[ib] << (8-2*lx)) & 0x300);
                        const unsigned char *grd = (const unsigned char *)&grid_iq2s[gid];
                        unsigned char sn = sgp[lx];
                        for (int j = 0; j < 8; j++)
                            p += (dl * (float)grd[j] * ((sn & (1<<j))?-1.0f:1.0f)) * xx[yi++];
                    }
                    qs += 4; sgp += 4;
                }
                s += p;
            }
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
            if (lane == 0) su[r] = s;
        }
        __syncthreads();
    }

    for (int i = tid; i < expert_ff; i += 128)
        sg[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];
    __syncthreads();

    /* Down IQ3_S */
    {
        int nb = expert_ff / 256, rb = nb * 110;
        for (int r = warp; r < n_embd; r += 4) {
            const unsigned char *rp = down_exps + (size_t)eidx * stride_d + (size_t)r * rb;
            float s = 0;
            for (int b = lane; b < nb; b += 32) {
                const unsigned char *bp = rp + (size_t)b * 110;
                float d = __half2float(*(const __half *)bp);
                const unsigned char *qs = bp+2, *ss = qs+64;
                const float *gg = sg + b * 256;
                float p = 0; int yi = 0;
                for (int ib = 0; ib < 8; ib++) {
                    unsigned aux32;
                    memcpy(&aux32, ss + 4*ib, 4);
                    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
                    for (int lx = 0; lx < 4; lx++) {
                        unsigned char sb = (unsigned char)grid_ksigns[(aux32>>(7*lx))&127];
                        unsigned g1 = grid_iq3[qs[2*lx+0]];
                        unsigned g2 = grid_iq3[qs[2*lx+1]];
                        for (int j = 0; j < 4; j++) {
                            float w0 = db*(float)(unsigned char)(g1>>(8*j))*((sb&(1<<j))?-1.0f:1.0f);
                            float w1 = db*(float)(unsigned char)(g2>>(8*j))*((sb&(1<<(j+4)))?-1.0f:1.0f);
                            p += w0*gg[yi+j] + w1*gg[yi+j+4];
                        }
                        yi += 8;
                    }
                    qs += 8;
                }
                s += p;
            }
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
            if (lane == 0) atomicAdd(&accum[r], wgt * s);
        }
    }
}
