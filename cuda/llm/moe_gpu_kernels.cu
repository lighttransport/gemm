/*
 * moe_gpu_kernels.cu — MoE GPU kernels.
 * AOT: nvcc -cubin -arch=sm_120 -o moe_gpu_kernels.cubin moe_gpu_kernels.cu
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* ---- GPU Top-k ---- */
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

/* ---- GPU Shared Gate ---- */
extern "C" __global__ void moe_shared_gate_gpu(
    float *gate_out, const float *xb, const float *gate_w, int n_embd)
{
    float s = 0;
    for (int i = threadIdx.x; i < n_embd; i += 128) s += gate_w[i] * xb[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) *gate_out = 1.0f / (1.0f + expf(-s));
}

/*
 * moe_iq2s_tc: single-token IQ2_S TC matvec for ALL experts in one launch.
 * grid = (expert_ff/16, n_used, 1) = (32, 8, 1) = 256 blocks
 * block = 32 threads (1 warp)
 * Each block: processes 16 rows for one expert.
 * grid_iq2s: device pointer to IQ2_S grid table (1024 × ULL)
 */
extern "C" __global__ void moe_iq2s_tc(
    float *output,           // [n_used * expert_ff]
    const float *xb,         // [n_embd]
    const unsigned char *weights_base,  // packed 3D: [stride * n_experts]
    const int *expert_idx,   // [n_used]
    const unsigned long long *grid_iq2s,
    int n_used, int expert_ff, int n_embd,
    size_t stride_gu)
{
    int ei = blockIdx.y;
    int r0 = blockIdx.x * 16;
    int eidx = expert_idx[ei];
    if (eidx < 0 || r0 >= expert_ff) return;

    int lane = threadIdx.x;
    float *out = output + (size_t)ei * expert_ff;
    int nb = n_embd / 256, rb = nb * 82;

    float rc[4] = {0,0,0,0};
    for (int k = 0; k < n_embd; k += 16) {
        int a_row = r0 + lane / 2;
        int col_k = k + (lane % 2) * 8;
        int bi = col_k / 256;
        int ib32 = (col_k % 256) / 32;
        int tl = ((k + (lane%2)*8) % 32) / 8;
        const unsigned char *bp = weights_base + (size_t)eidx * stride_gu
                                + (size_t)a_row * rb + bi * 82;
        float d = __half2float(*(const __half *)bp);
        float db0 = d * (0.5f + (bp[74+ib32] & 0xf)) * 0.25f;
        float db1 = d * (0.5f + (bp[74+ib32] >>  4)) * 0.25f;
        float dl = (tl < 2) ? db0 : db1;
        int gidx = bp[2+ib32*4+tl] | ((bp[66+ib32] << (8-2*tl)) & 0x300);
        unsigned long long gv = grid_iq2s[gidx];
        unsigned char sgn = bp[34+ib32*4+tl];
        half w16[8];
        for (int j = 0; j < 8; j++) {
            float wv = dl * (float)(unsigned char)(gv >> (8*j));
            if ((sgn >> j) & 1) wv = -wv;
            w16[j] = __float2half(wv);
        }
        unsigned ra[4];
        ra[0] = *(const unsigned *)&w16[0];
        ra[1] = *(const unsigned *)&w16[2];
        ra[2] = *(const unsigned *)&w16[4];
        ra[3] = *(const unsigned *)&w16[6];

        half b16[4];
        int bk = ((lane % 8) / 2) * 2;
        int kk = k + bk;
        b16[0] = __float2half(xb[kk]);
        b16[1] = __float2half(xb[kk+1]);
        b16[2] = __float2half(xb[kk]);
        b16[3] = __float2half(xb[kk+1]);
        unsigned rb2[2];
        rb2[0] = *(const unsigned *)&b16[0];
        rb2[1] = *(const unsigned *)&b16[2];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0, %1, %2, %3},"
            " {%4, %5, %6, %7},"
            " {%8, %9},"
            " {%10, %11, %12, %13};"
            : "+f"(rc[0]), "+f"(rc[1]), "+f"(rc[2]), "+f"(rc[3])
            : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
              "r"(rb2[0]), "r"(rb2[1]),
              "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3]));
    }
    int m0 = lane / 4, m1 = m0 + 8;
    if ((lane & 3) == 0) { out[r0 + m0] = rc[0]; out[r0 + m1] = rc[2]; }
}
