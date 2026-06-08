/*
 * moe_gpu_kernels.cu — MoE GPU kernels for 35B-A3B decode.
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
 * moe_iq2s_tc: single-token IQ2_S TC matvec for ALL experts.
 * grid = (expert_ff/16, n_used, 1) = (32, 8, 1) = 256 blocks, 32 threads/block.
 */
extern "C" __global__ void moe_iq2s_tc(
    float *output, const float *xb,
    const unsigned char *weights_base, const int *expert_idx,
    const unsigned long long *grid_iq2s,
    int n_used, int expert_ff, int n_embd, size_t stride_gu)
{
    int ei = blockIdx.y, r0 = blockIdx.x * 16;
    int eidx = expert_idx[ei];
    if (eidx < 0 || r0 >= expert_ff) return;

    int lane = threadIdx.x;
    float *out = output + (size_t)ei * expert_ff;
    int nb = n_embd / 256, rb = nb * 82;

    float rc[4] = {0,0,0,0};
    for (int k = 0; k < n_embd; k += 16) {
        int a_row = r0 + lane / 2, col_k = k + (lane % 2) * 8;
        int bi = col_k / 256, ib32 = (col_k % 256) / 32;
        int tl = ((k + (lane%2)*8) % 32) / 8;
        const unsigned char *bp = weights_base + (size_t)eidx * stride_gu
                                + (size_t)a_row * rb + bi * 82;
        float d = __half2float(*(const __half *)bp);
        float db0 = d * (0.5f + (bp[74+ib32] & 0xf)) * 0.25f;
        float db1 = d * (0.5f + (bp[74+ib32] >> 4)) * 0.25f;
        int gidx = bp[2+ib32*4+tl] | ((bp[66+ib32] << (8-2*tl)) & 0x300);
        unsigned long long gv = grid_iq2s[gidx];
        unsigned char sgn = bp[34+ib32*4+tl];
        float dl = (tl < 2) ? db0 : db1;
        half w16[8];
        for (int j = 0; j < 8; j++) {
            float wv = dl * (float)(unsigned char)(gv >> (8*j));
            w16[j] = __float2half((sgn >> j) & 1 ? -wv : wv);
        }
        unsigned ra[4];
        ra[0] = *(const unsigned *)&w16[0]; ra[1] = *(const unsigned *)&w16[2];
        ra[2] = *(const unsigned *)&w16[4]; ra[3] = *(const unsigned *)&w16[6];

        half b16[4];
        int bk = ((lane % 8) / 2) * 2, kk = k + bk;
        b16[0] = __float2half(xb[kk]); b16[1] = __float2half(xb[kk+1]);
        b16[2] = __float2half(xb[kk]); b16[3] = __float2half(xb[kk+1]);
        unsigned rb2[2];
        rb2[0] = *(const unsigned *)&b16[0]; rb2[1] = *(const unsigned *)&b16[2];

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
            : "r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),
              "r"(rb2[0]),"r"(rb2[1]),
              "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
    }
    int m0 = lane / 4, m1 = m0 + 8;
    if ((lane & 3) == 0) { out[r0 + m0] = rc[0]; out[r0 + m1] = rc[2]; }
}

/*
 * moe_iq3s_tc: single-token IQ3_S TC matvec for ALL experts.
 * grid = (n_embd/16, n_used, 1) = (128, 8, 1) = 1024 blocks, 32 threads/block.
 */
extern "C" __global__ void moe_iq3s_tc(
    float *output, const float *input,
    const unsigned char *weights_base, const int *expert_idx,
    const unsigned int *grid_iq3s,
    int n_used, int n_embd, int expert_ff, size_t stride_d)
{
    int ei = blockIdx.y, r0 = blockIdx.x * 16;
    int eidx = expert_idx[ei];
    if (eidx < 0 || r0 >= n_embd) return;

    int lane = threadIdx.x;
    float *out = output + (size_t)ei * n_embd;
    int nb = expert_ff / 256, rb = nb * 110;
    const float *xx = input + (size_t)ei * expert_ff;

    float rc[4] = {0,0,0,0};
    for (int k = 0; k < expert_ff; k += 16) {
        int a_row = r0 + lane / 2, col_k = k + (lane % 2) * 8;
        int bi = col_k / 256;
        const unsigned char *bp = weights_base + (size_t)eidx * stride_d
                                + (size_t)a_row * rb + bi * 110;
        float d = __half2float(*(const __half *)bp);
        const unsigned char *qs0 = bp + 2;            /* qs base */
        const unsigned char *qh_base = bp + 2 + 64;    /* qh (8 bytes) */
        const unsigned char *signs_base = bp + 2 + 64 + 8; /* signs (32 bytes) */
        const unsigned char *scales = bp + 2 + 64 + 8 + 32; /* scales (4 bytes) */

        /* Determine position within the 256-value block */
        int pos_in_block = col_k % 256;
        int ib32 = pos_in_block / 32;      /* 0..7: which 32-col group */
        int tl = (pos_in_block % 32) / 8;  /* 0..3: which 8-col sub-group */

        /* IQ3_S layout per 256-value block:
         * qs: 64 bytes starting at bp+2, accessed as 4-byte-advancing by ib32
         * qh_base: 8 bytes at bp+66
         * signs: 32 bytes at bp+74
         * scales: 4 bytes at bp+106 */

        /* For sub-group tl within ib32:
         * Grid indices at qs_base[ib32*8 + tl*2 + 0] and qs_base[ib32*8 + tl*2 + 1]
         * qh byte at qh_base[ib32], contributes bit 8 via (8-2*tl) shift
         * Sign byte at signs_base[ib32*4 + tl]
         * Scale nibble: ib32/2 gives byte index, ib32%2 gives high/low nibble */

        int sc_byte = ib32 / 2;
        int nib_shift = (ib32 % 2) * 4; /* ib32 even→low nibble(db1), odd→high nibble(db2) */
        float db = d * (float)(1 + 2 * ((scales[sc_byte] >> nib_shift) & 0xf));

        /* Each ib32 uses 8 qs bytes (4 lx × 2 bytes each). */
        int qs_off = ib32 * 8 + tl * 2;
        int g1 = qs0[qs_off + 0] | ((qh_base[ib32] << (8 - 2*tl)) & 256);
        int g2 = qs0[qs_off + 1] | ((qh_base[ib32] << (7 - 2*tl)) & 256);

        const unsigned char *grd1 = (const unsigned char *)&grid_iq3s[g1];
        const unsigned char *grd2 = (const unsigned char *)&grid_iq3s[g2];
        /* Signs: each ib32 uses 4 bytes (tl=0..3, one byte each) */
        unsigned char sgn = signs_base[ib32 * 4 + tl];

        half w16[8];
        for (int j = 0; j < 4; j++) {
            float w0 = db * (float)grd1[j];
            float w1 = db * (float)grd2[j];
            w16[j]     = __float2half((sgn & (1<<j))    ? -w0 : w0);
            w16[j + 4] = __float2half((sgn & (1<<(j+4))) ? -w1 : w1);
        }
        unsigned ra[4];
        ra[0] = *(const unsigned *)&w16[0]; ra[1] = *(const unsigned *)&w16[2];
        ra[2] = *(const unsigned *)&w16[4]; ra[3] = *(const unsigned *)&w16[6];

        half b16[4];
        int bk = ((lane % 8) / 2) * 2, kk = k + bk;
        b16[0] = __float2half(xx[kk]);   b16[1] = __float2half(xx[kk+1]);
        b16[2] = __float2half(xx[kk]);   b16[3] = __float2half(xx[kk+1]);
        unsigned rb2[2];
        rb2[0] = *(const unsigned *)&b16[0]; rb2[1] = *(const unsigned *)&b16[2];

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
            : "r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),
              "r"(rb2[0]),"r"(rb2[1]),
              "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
    }
    int m0 = lane / 4, m1 = m0 + 8;
    if ((lane & 3) == 0) { out[r0 + m0] = rc[0]; out[r0 + m1] = rc[2]; }
}
