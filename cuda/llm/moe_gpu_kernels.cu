/*
 * moe_gpu_kernels.cu — MoE GPU kernels.
 * AOT: nvcc -cubin -arch=sm_120 -o moe_gpu_kernels.cubin moe_gpu_kernels.cu
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

extern "C" __global__ void moe_topk_gpu(int *idx_out, float *wgt_out, const float *logits, int n_experts, int n_used, int n_tokens) {
    int token = blockIdx.x;
    if (token >= n_tokens) return;
    logits += (size_t)token * n_experts;
    idx_out += (size_t)token * n_used;
    wgt_out += (size_t)token * n_used;
    int tid = threadIdx.x;
    float tv[8]; int ti[8];
    for (int i = 0; i < 8; i++) { tv[i] = -1e38f; ti[i] = -1; }
    for (int e = tid; e < n_experts; e += blockDim.x) {
        float v = logits[e];
        if (!(v == v)) v = -FLT_MAX;
        if (v > tv[n_used-1] || (v == tv[n_used-1] && (ti[n_used-1] < 0 || e < ti[n_used-1]))) {
            tv[n_used-1] = v; ti[n_used-1] = e;
            for (int i = n_used-2; i >= 0 &&
                 (tv[i+1] > tv[i] || (tv[i+1] == tv[i] && ti[i+1] >= 0 && (ti[i] < 0 || ti[i+1] < ti[i]))); i--) {
                float xv = tv[i]; int xi = ti[i];
                tv[i] = tv[i+1]; ti[i] = ti[i+1]; tv[i+1] = xv; ti[i+1] = xi;
            }
        }
    }
    __shared__ float slv[1024];
    __shared__ int sli[1024];
    int base = tid * 8;
    for (int i = 0; i < 8; i++) {
        slv[base + i] = tv[i];
        sli[base + i] = ti[i];
    }
    __syncthreads();
    if (tid == 0) {
        float fv[8]; int fi[8];
        for (int i = 0; i < 8; i++) { fv[i] = -1e38f; fi[i] = -1; }
        int n_cand = blockDim.x * 8;
        for (int c = 0; c < n_cand; c++) {
            int idx = sli[c];
            if (idx < 0) continue;
            float v = slv[c];
            if (v > fv[n_used-1] || (v == fv[n_used-1] && (fi[n_used-1] < 0 || idx < fi[n_used-1]))) {
                fv[n_used-1] = v; fi[n_used-1] = idx;
                for (int ii = n_used-2; ii >= 0 &&
                     (fv[ii+1] > fv[ii] || (fv[ii+1] == fv[ii] && fi[ii+1] >= 0 && (fi[ii] < 0 || fi[ii+1] < fi[ii]))); ii--) {
                    float xv = fv[ii]; int xi = fi[ii];
                    fv[ii] = fv[ii+1]; fi[ii] = fi[ii+1]; fv[ii+1] = xv; fi[ii+1] = xi;
                }
            }
        }
        float max_v = fv[0];
        for (int i = 1; i < n_used; i++)
            if (fv[i] > max_v) max_v = fv[i];
        float sum = 0.0f;
        for (int i = 0; i < n_used; i++)
            sum += expf(fv[i] - max_v);
        for (int i = 0; i < n_used; i++) {
            idx_out[i] = fi[i];
            wgt_out[i] = expf(fv[i] - max_v) / sum;
        }
    }
}

extern "C" __global__ void moe_shared_gate_gpu(float *go, const float *xb, const float *gw, int ne) {
    __shared__ float warp_sums[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float s = 0; for (int i = threadIdx.x; i < ne; i += 128) s += gw[i] * xb[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) warp_sums[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < 4) ? warp_sums[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) *go = 1.0f / (1.0f + expf(-s));
    }
}

/* F16 TC matvec (shared expert) */
extern "C" __global__ void moe_f16_tc(float *out, const half *w, const float *x, int od, int id) {
    int r0 = blockIdx.x * 16; if (r0 >= od) return;
    int l = threadIdx.x; float rc[4] = {0,0,0,0};
    for (int k = 0; k < id; k += 16) {
        int row = r0 + l/2, ck = k + (l%2)*8;
        half w16[8]; for (int j = 0; j < 8; j++) w16[j] = w[(size_t)row * id + ck + j];
        unsigned ra[4]; ra[0]=*(const unsigned*)&w16[0];ra[1]=*(const unsigned*)&w16[2];
        ra[2]=*(const unsigned*)&w16[4];ra[3]=*(const unsigned*)&w16[6];
        int bk = ((l%8)/2)*2, kk = k+bk;
        half b16[4];b16[0]=__float2half(x[kk]);b16[1]=__float2half(x[kk+1]);
        b16[2]=__float2half(x[kk]);b16[3]=__float2half(x[kk+1]);
        unsigned rb2[2];rb2[0]=*(const unsigned*)&b16[0];rb2[1]=*(const unsigned*)&b16[2];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            :"+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
            :"r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb2[0]),"r"(rb2[1]),
             "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
    }
    int m0=l/4,m1=m0+8;if((l&3)==0){out[r0+m0]=rc[0];out[r0+m1]=rc[2];}
}

/*
 * moe_prefill_fused: fused MoE FFN for batched prefill.
 * grid = n_tokens, block = 128 (4 warps). One token per block.
 * Processes all 8 experts with IQ2_S TC gate/up + IQ3_S scalar down.
 * Grid tables and ksigns passed as parameters.
 */
extern "C" __global__ void moe_prefill_fused(
    float *output,              // [n_tokens * n_embd] — output accumulated here
    const float *input,         // [n_tokens * n_embd] — FFN input (post-rmsnorm)
    const float *router_logits, // [n_tokens * n_experts]
    const unsigned char *gate_exps,
    const unsigned char *up_exps,
    const unsigned char *down_exps,
    int n_tokens, int n_experts, int n_used,
    int n_embd, int expert_ff,
    size_t stride_gu, size_t stride_d,
    const unsigned long long *grid_iq2s,
    const unsigned int *grid_iq3s,
    const unsigned char *grid_ksigns)
{
    int token = blockIdx.x;
    if (token >= n_tokens) return;

    __shared__ float sx[2048];
    __shared__ float sg[512];
    __shared__ float su[512];
    __shared__ float sa[2048];
    __shared__ int sidx[8];
    __shared__ float sw[8];

    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

    /* Load input to shared */
    const float *xb = input + (size_t)token * n_embd;
    for (int i = tid; i < n_embd; i += 128) sx[i] = xb[i];
    __syncthreads();

    /* GPU top-k (same as existing moe_fused_ffn) */
    float lv[8]; int li[8];
    for (int i = 0; i < n_used; i++) { lv[i] = -1e38f; li[i] = -1; }
    const float *rl = router_logits + (size_t)token * n_experts;
    for (int e = tid; e < n_experts; e += 128) {
        float v = rl[e];
        if (v > lv[n_used-1]) {
            lv[n_used-1] = v; li[n_used-1] = e;
            for (int i = n_used-2; i >= 0 && lv[i+1] > lv[i]; i--) {
                float xv = lv[i]; int xi = li[i];
                lv[i] = lv[i+1]; li[i] = li[i+1]; lv[i+1] = xv; li[i+1] = xi;
            }
        }
    }
    for (int rr = 16; rr > 0; rr >>= 1) {
        if (tid + rr < 128) {
            for (int kk = 0; kk < n_used; kk++) {
                if (lv[kk] > lv[n_used-1]) {
                    lv[n_used-1] = lv[kk]; li[n_used-1] = li[kk];
                    for (int ii = n_used-2; ii >= 0 && lv[ii+1] > lv[ii]; ii--) {
                        float xv = lv[ii]; int xi = li[ii];
                        lv[ii] = lv[ii+1]; li[ii] = li[ii+1]; lv[ii+1] = xv; li[ii+1] = xi;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (tid < n_used) { sidx[tid] = li[tid]; sw[tid] = 1.0f/(1.0f+expf(-lv[tid])); }
    __syncthreads();

    for (int i = tid; i < n_embd; i += 128) sa[i] = 0.0f;
    __syncthreads();

    /* Expert loop */
    for (int ei = 0; ei < n_used; ei++) {
        int eidx = sidx[ei];
        float wgt = sw[ei];
        for (int i = tid; i < expert_ff; i += 128) { sg[i] = 0.0f; su[i] = 0.0f; }
        __syncthreads();

        /* Gate IQ2_S TC — each warp covers expert_ff/4 = 128 rows */
        {
            int nb = n_embd / 256, rb = nb * 82;
            for (int mg = 0; mg < 8 && (warp*128 + mg*16) < expert_ff; mg++) {
                int rs = warp * 128 + mg * 16;
                float rc[4] = {0,0,0,0};
                for (int k = 0; k < n_embd; k += 16) {
                    int a_row = rs + lane/2, ck = k + (lane%2)*8;
                    int bi = ck/256, ib = (ck%256)/32, tl = ((k+(lane%2)*8)%32)/8;
                    const unsigned char *bp = gate_exps + (size_t)eidx * stride_gu
                                           + (size_t)a_row * rb + bi * 82;
                    float d = __half2float(*(const __half *)bp);
                    float da = d*(.5f+(bp[74+ib]&0xf))*.25f, db = d*(.5f+(bp[74+ib]>>4))*.25f;
                    int gid = bp[2+ib*4+tl]|((bp[66+ib]<<(8-2*tl))&0x300);
                    unsigned long long gv = grid_iq2s[gid];
                    unsigned char sn = bp[34+ib*4+tl];
                    float dl = (tl<2)?da:db;
                    half w16[8];
                    for (int j = 0; j < 8; j++) {
                        float wv = dl*(float)(unsigned char)(gv>>(8*j));
                        w16[j] = __float2half((sn>>j)&1?-wv:wv);
                    }
                    unsigned ra[4];ra[0]=*(const unsigned*)&w16[0];ra[1]=*(const unsigned*)&w16[2];
                    ra[2]=*(const unsigned*)&w16[4];ra[3]=*(const unsigned*)&w16[6];
                    int bk = ((lane%8)/2)*2, kk = k+bk;
                    half b16[4];b16[0]=__float2half(sx[kk]);b16[1]=__float2half(sx[kk+1]);
                    b16[2]=__float2half(sx[kk]);b16[3]=__float2half(sx[kk+1]);
                    unsigned rb2[2];rb2[0]=*(const unsigned*)&b16[0];rb2[1]=*(const unsigned*)&b16[2];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
                        :"r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb2[0]),"r"(rb2[1]),
                         "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
                }
                int m0=lane/4,m1=m0+8;if((lane&3)==0){sg[rs+m0]=rc[0];sg[rs+m1]=rc[2];}
            }
        }
        __syncthreads();

        /* Up IQ2_S TC */
        {
            int nb = n_embd / 256, rb = nb * 82;
            for (int mg = 0; mg < 8 && (warp*128 + mg*16) < expert_ff; mg++) {
                int rs = warp * 128 + mg * 16;
                float rc[4] = {0,0,0,0};
                for (int k = 0; k < n_embd; k += 16) {
                    int a_row = rs + lane/2, ck = k + (lane%2)*8;
                    int bi = ck/256, ib = (ck%256)/32, tl = ((k+(lane%2)*8)%32)/8;
                    const unsigned char *bp = up_exps + (size_t)eidx * stride_gu
                                           + (size_t)a_row * rb + bi * 82;
                    float d = __half2float(*(const __half *)bp);
                    float da = d*(.5f+(bp[74+ib]&0xf))*.25f, db = d*(.5f+(bp[74+ib]>>4))*.25f;
                    int gid = bp[2+ib*4+tl]|((bp[66+ib]<<(8-2*tl))&0x300);
                    unsigned long long gv = grid_iq2s[gid];
                    unsigned char sn = bp[34+ib*4+tl];
                    float dl = (tl<2)?da:db;
                    half w16[8];
                    for (int j = 0; j < 8; j++) {
                        float wv = dl*(float)(unsigned char)(gv>>(8*j));
                        w16[j] = __float2half((sn>>j)&1?-wv:wv);
                    }
                    unsigned ra[4];ra[0]=*(const unsigned*)&w16[0];ra[1]=*(const unsigned*)&w16[2];
                    ra[2]=*(const unsigned*)&w16[4];ra[3]=*(const unsigned*)&w16[6];
                    int bk = ((lane%8)/2)*2, kk = k+bk;
                    half b16[4];b16[0]=__float2half(sx[kk]);b16[1]=__float2half(sx[kk+1]);
                    b16[2]=__float2half(sx[kk]);b16[3]=__float2half(sx[kk+1]);
                    unsigned rb2[2];rb2[0]=*(const unsigned*)&b16[0];rb2[1]=*(const unsigned*)&b16[2];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
                        :"r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb2[0]),"r"(rb2[1]),
                         "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
                }
                int m0=lane/4,m1=m0+8;if((lane&3)==0){su[rs+m0]=rc[0];su[rs+m1]=rc[2];}
            }
        }
        __syncthreads();

        for (int i = tid; i < expert_ff; i += 128)
            sg[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];
        __syncthreads();

        /* Down IQ3_S scalar */
        {
            int nb = expert_ff / 256, rb = nb * 110;
            for (int r = warp; r < n_embd; r += 4) {
                const unsigned char *rp = down_exps + (size_t)eidx * stride_d + (size_t)r * rb;
                float s = 0;
                for (int b = lane; b < nb; b += 32) {
                    const unsigned char *bp = rp + (size_t)b * 110;
                    float d = __half2float(*(const __half *)bp);
                    const unsigned char *qs = bp+2, *qh = bp+66, *sgnb = bp+74, *scl = bp+106;
                    const float *gg = sg + b * 256;
                    float p = 0; int yi = 0;
                    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                        float db1 = d*(float)(1+2*(scl[ib32/2]&0xf));
                        float db2 = d*(float)(1+2*(scl[ib32/2]>>4));
                        for (int lx = 0; lx < 4; lx++) {
                            int g1 = qs[2*lx+0]|((qh[ib32]<<(8-2*lx))&256);
                            int g2 = qs[2*lx+1]|((qh[ib32]<<(7-2*lx))&256);
                            const unsigned char *gr1 = (const unsigned char*)&grid_iq3s[g1];
                            const unsigned char *gr2 = (const unsigned char*)&grid_iq3s[g2];
                            unsigned char sb = sgnb[ib32/2*4+lx];
                            for (int j = 0; j < 4; j++) {
                                float w0 = db1*(float)gr1[j]*((sb&(1<<j))?-1.0f:1.0f);
                                float w1 = db1*(float)gr2[j]*((sb&(1<<(j+4)))?-1.0f:1.0f);
                                p += w0*gg[yi+j] + w1*gg[yi+j+4];
                            }
                            yi += 8;
                        }
                        qs += 8;
                        for (int lx = 0; lx < 4; lx++) {
                            int g1 = qs[2*lx+0]|((qh[ib32+1]<<(8-2*lx))&256);
                            int g2 = qs[2*lx+1]|((qh[ib32+1]<<(7-2*lx))&256);
                            const unsigned char *gr1 = (const unsigned char*)&grid_iq3s[g1];
                            const unsigned char *gr2 = (const unsigned char*)&grid_iq3s[g2];
                            unsigned char sb = sgnb[ib32/2*4+4+lx];
                            for (int j = 0; j < 4; j++) {
                                float w0 = db2*(float)gr1[j]*((sb&(1<<j))?-1.0f:1.0f);
                                float w1 = db2*(float)gr2[j]*((sb&(1<<(j+4)))?-1.0f:1.0f);
                                p += w0*gg[yi+j] + w1*gg[yi+j+4];
                            }
                            yi += 8;
                        }
                        qs += 8;
                    }
                    s += p;
                }
                for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
                if (lane == 0) sa[r] += wgt * s;
            }
        }
        __syncthreads();
    }

    /* Shared expert */
    {
        float gs = 0; for (int i = lane; i < n_embd; i += 32) gs += 0; /* no shared gate in prefill */
        for (int i = tid; i < n_embd; i += 128)
            output[(size_t)token * n_embd + i] = sa[i];
    }
}

/* Accumulator kernel */
extern "C" __global__ void moe_accum_gpu(float *accum, const float *down, const float *wgt, int nu, int ne) {
    int i = blockIdx.x*256+threadIdx.x; if (i>=ne) return;
    float s = 0; for (int e=0;e<nu;e++) s+=wgt[e]*down[(size_t)e*ne+i];
    accum[i]+=s;
}

/*
 * moe_prefill_q4k: fused MoE FFN for batched prefill with Q4_K weights.
 * grid = n_tokens, block = 128 (4 warps). One token per block.
 * All 8 experts processed in a single kernel launch.
 * Q4_K dequant inline (144 bytes per 256-element block).
 */
extern "C" __global__ void moe_prefill_q4k(
    float *output,              // [n_tokens * n_embd]
    const float *input,         // [n_tokens * n_embd]
    const float *router_logits, // [n_tokens * n_experts]
    const unsigned char *gate_exps,
    const unsigned char *up_exps,
    const unsigned char *down_exps,
    int n_tokens, int n_experts, int n_used,
    int n_embd, int expert_ff,
    size_t stride_gu, size_t stride_d)
{
    int token = blockIdx.x;
    if (token >= n_tokens) return;

    __shared__ float sx[2048];
    __shared__ float sg[512];
    __shared__ float su[512];
    __shared__ float sa[2048];
    __shared__ int sidx[8];
    __shared__ float sw[8];

    int tid = threadIdx.x;

    /* Load input to shared memory */
    const float *xb = input + (size_t)token * n_embd;
    for (int i = tid; i < n_embd; i += 128) sx[i] = xb[i];
    __syncthreads();

    /* GPU top-k */
    float lv[8]; int li[8];
    for (int i = 0; i < n_used; i++) { lv[i] = -1e38f; li[i] = -1; }
    const float *rl = router_logits + (size_t)token * n_experts;
    for (int e = tid; e < n_experts; e += 128) {
        float v = rl[e];
        if (v > lv[n_used-1]) {
            lv[n_used-1] = v; li[n_used-1] = e;
            for (int i = n_used-2; i >= 0 && lv[i+1] > lv[i]; i--) {
                float xv = lv[i]; int xi = li[i];
                lv[i] = lv[i+1]; li[i] = li[i+1]; lv[i+1] = xv; li[i+1] = xi;
            }
        }
    }
    for (int rr = 16; rr > 0; rr >>= 1) {
        if (tid + rr < 128) {
            for (int kk = 0; kk < n_used; kk++) {
                if (lv[kk] > lv[n_used-1]) {
                    lv[n_used-1] = lv[kk]; li[n_used-1] = li[kk];
                    for (int ii = n_used-2; ii >= 0 && lv[ii+1] > lv[ii]; ii--) {
                        float xv = lv[ii]; int xi = li[ii];
                        lv[ii] = lv[ii+1]; li[ii] = li[ii+1]; lv[ii+1] = xv; li[ii+1] = xi;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (tid < n_used) { sidx[tid] = li[tid]; sw[tid] = 1.0f/(1.0f+expf(-lv[tid])); }
    __syncthreads();

    for (int i = tid; i < n_embd; i += 128) sa[i] = 0.0f;
    __syncthreads();

    int nb_gu = n_embd / 256, rb_gu = nb_gu * 144;
    int nb_d = expert_ff / 256, rb_d = nb_d * 144;

    /* Expert loop */
    for (int ei = 0; ei < n_used; ei++) {
        int eidx = sidx[ei];
        float wgt = sw[ei];

        /* Gate: Q4_K matvec, 512 rows */
        for (int r = tid; r < expert_ff; r += 128) {
            const unsigned char *rp = gate_exps + (size_t)eidx * stride_gu + (size_t)r * rb_gu;
            float s = 0;
            for (int b = 0; b < nb_gu; b++) {
                const unsigned char *bp = rp + (size_t)b * 144;
                float d = __half2float(*(const __half *)bp);
                float dmin = __half2float(*(const __half *)(bp + 2));
                const unsigned char *sc = bp + 4;
                const unsigned char *qs = bp + 16;
                const float *xx = sx + b * 256;
                float p = 0; int yi = 0;
                for (int j = 0; j < 4; j++) {
                    int is = j * 2;
                    float sv0f, mv0f, sv1f, mv1f;
                    if (is < 4) {
                        sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
                        sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
                    } else {
                        sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
                        mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
                        sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
                        mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
                    }
                    float d1 = d * sv0f, m1 = dmin * mv0f;
                    float d2 = d * sv1f, m2 = dmin * mv1f;
                    const unsigned char *q = qs + j * 32;
                    for (int l = 0; l < 32; l++)
                        p += (d1 * (q[l] & 0xF) - m1) * xx[yi++];
                    for (int l = 0; l < 32; l++)
                        p += (d2 * (q[l] >> 4) - m2) * xx[yi++];
                }
                s += p;
            }
            sg[r] = s;
        }
        __syncthreads();

        /* Up: Q4_K matvec, 512 rows */
        for (int r = tid; r < expert_ff; r += 128) {
            const unsigned char *rp = up_exps + (size_t)eidx * stride_gu + (size_t)r * rb_gu;
            float s = 0;
            for (int b = 0; b < nb_gu; b++) {
                const unsigned char *bp = rp + (size_t)b * 144;
                float d = __half2float(*(const __half *)bp);
                float dmin = __half2float(*(const __half *)(bp + 2));
                const unsigned char *sc = bp + 4;
                const unsigned char *qs = bp + 16;
                const float *xx = sx + b * 256;
                float p = 0; int yi = 0;
                for (int j = 0; j < 4; j++) {
                    int is = j * 2;
                    float sv0f, mv0f, sv1f, mv1f;
                    if (is < 4) {
                        sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
                        sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
                    } else {
                        sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
                        mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
                        sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
                        mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
                    }
                    float d1 = d * sv0f, m1 = dmin * mv0f;
                    float d2 = d * sv1f, m2 = dmin * mv1f;
                    const unsigned char *q = qs + j * 32;
                    for (int l = 0; l < 32; l++)
                        p += (d1 * (q[l] & 0xF) - m1) * xx[yi++];
                    for (int l = 0; l < 32; l++)
                        p += (d2 * (q[l] >> 4) - m2) * xx[yi++];
                }
                s += p;
            }
            su[r] = s;
        }
        __syncthreads();

        for (int i = tid; i < expert_ff; i += 128)
            sg[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];
        __syncthreads();

        /* Down: Q4_K matvec, 2048 rows */
        for (int r = tid; r < n_embd; r += 128) {
            const unsigned char *rp = down_exps + (size_t)eidx * stride_d + (size_t)r * rb_d;
            float s = 0;
            for (int b = 0; b < nb_d; b++) {
                const unsigned char *bp = rp + (size_t)b * 144;
                float d = __half2float(*(const __half *)bp);
                float dmin = __half2float(*(const __half *)(bp + 2));
                const unsigned char *sc = bp + 4;
                const unsigned char *qs = bp + 16;
                const float *gg = sg + b * 256;
                float p = 0; int yi = 0;
                for (int j = 0; j < 4; j++) {
                    int is = j * 2;
                    float sv0f, mv0f, sv1f, mv1f;
                    if (is < 4) {
                        sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
                        sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
                    } else {
                        sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
                        mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
                        sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
                        mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
                    }
                    float d1 = d * sv0f, m1 = dmin * mv0f;
                    float d2 = d * sv1f, m2 = dmin * mv1f;
                    const unsigned char *q = qs + j * 32;
                    for (int l = 0; l < 32; l++)
                        p += (d1 * (q[l] & 0xF) - m1) * gg[yi++];
                    for (int l = 0; l < 32; l++)
                        p += (d2 * (q[l] >> 4) - m2) * gg[yi++];
                }
                s += p;
            }
            sa[r] += wgt * s;
        }
        __syncthreads();
    }

    for (int i = tid; i < n_embd; i += 128)
        output[(size_t)token * n_embd + i] += sa[i];
}

/*
 * dequant_q4_K_to_f16: Convert Q4_K quantized weights to FP16.
 * grid = (rows * blocks_per_row + 127) / 128, block = 128.
 * Each thread processes 1 super-block (256 elements) and uses F16x2
 * vectorized stores to write 2 F16 outputs per 32-bit store, cutting
 * store count and instructions roughly in half vs per-F16 stores.
 */
extern "C" __global__ void dequant_q4_K_to_f16(
    half *out,                 // [rows * cols] FP16 output
    const unsigned char *mat,  // [rows * row_bytes] Q4_K input
    int rows, int cols)
{
    int nb = cols / 256, rb = nb * 144;
    int bid = blockIdx.x * 128 + threadIdx.x;
    int total_blocks = rows * nb;
    if (bid >= total_blocks) return;

    int row = bid / nb;
    int bk = bid % nb;

    const unsigned char *bp = mat + (size_t)row * rb + (size_t)bk * 144;
    half2 *out_blk2 = (half2 *)(out + (size_t)row * cols + (size_t)bk * 256);

    float d = __half2float(*(const __half *)bp);
    float dmin = __half2float(*(const __half *)(bp + 2));
    const unsigned char *sc = bp + 4;
    const unsigned char *qs = bp + 16;

    for (int j = 0; j < 4; j++) {
        int is = j * 2;
        float sv0f, mv0f, sv1f, mv1f;
        if (is < 4) {
            sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
            sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
        } else {
            sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
            mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
            sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
            mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
        }
        float d1 = d * sv0f, m1 = dmin * mv0f;
        float d2 = d * sv1f, m2 = dmin * mv1f;
        __half2 d1_2 = __float2half2_rn(d1);
        __half2 d2_2 = __float2half2_rn(d2);
        __half2 m1_2 = __float2half2_rn(m1);
        __half2 m2_2 = __float2half2_rn(m2);
        const unsigned char *q = qs + j * 32;
        #pragma unroll
        for (int l = 0; l < 32; l += 2) {
            unsigned char q0 = q[l], q1 = q[l+1];
            __half2 v0 = __floats2half2_rn((float)(q0 & 0xF), (float)(q1 & 0xF));
            __half2 v1 = __floats2half2_rn((float)(q0 >> 4),  (float)(q1 >> 4));
            v0 = __hsub2(__hmul2(d1_2, v0), m1_2);
            v1 = __hsub2(__hmul2(d2_2, v1), m2_2);
            out_blk2[j*32 + (l >> 1)]     = v0;
            out_blk2[j*32 + 16 + (l >> 1)] = v1;
        }
    }
}

/*
 * dequant_q4_0_to_f16: Q4_0 weight matrix -> FP16 row-major [rows, cols].
 * Mirrors dequant_q4_K_to_f16 for the gemma4 dequant->F16->cuBLAS batched-prefill
 * path. Q4_0 GPU block = 18 bytes: half d @0, qs[16] @2; weight j =
 * (qs[j]&0xF - 8)*d for j<16, (qs[j]>>4 - 8)*d for j+16 (matches matvec_q4_0_f32).
 *
 * ONE THREAD PER OUTPUT ELEMENT (not per block). Consecutive threads write
 * consecutive F16 outputs -> fully coalesced 64-byte stores per warp, and the
 * 32 threads of a block-region share the same scale `d` (broadcast read). This
 * is the dequant_q6_K_to_f16 layout; the old per-block kernel issued strided
 * 2-byte stores 64 bytes apart across a warp (uncoalesced). Launched 1D with
 * grid=(rows*cols+255)/256, block=256; rows*cols <= 256MB/2 < 2^31 (lm_head is
 * excluded from this path), so a uint32 element index is safe.
 */
extern "C" __global__ void dequant_q4_0_to_f16(
    half *out,                 // [rows * cols] FP16 output
    const unsigned char *mat,  // [rows * row_bytes] Q4_0 input
    int rows, int cols)
{
    unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = (unsigned)rows * (unsigned)cols;
    if (e >= total) return;

    unsigned ucols = (unsigned)cols;
    unsigned nb = ucols >> 5;            // cols/32 (cols is a multiple of 32)
    unsigned row = e / ucols;
    unsigned col = e - row * ucols;
    unsigned bk = col >> 5;              // block within row
    unsigned local = col & 31;           // 0..31

    const unsigned char *bp = mat + ((size_t)row * nb + bk) * 18;
    float d = __half2float(*(const __half *)bp);
    unsigned char q = bp[2 + (local & 15)];
    int nib = (local < 16) ? (int)(q & 0x0F) : (int)(q >> 4);
    out[e] = __float2half((float)(nib - 8) * d);
}

/*
 * moe_expert_fused_q4k: Fused per-expert Q4_K gate+up+SiLU+down for batched tokens.
 * One block per expert. Processes all n_e tokens sequentially within the block.
 * Eliminates all per-expert kernel launch overhead (gather, convert, dequant, cuBLAS, scatter).
 *
 * Grid: n_unique_experts (one block per expert with tokens)
 * Block: 128 threads (4 warps)
 */
extern "C" __global__ void moe_expert_fused_q4k(
    float *output,              // [n_tokens, n_embd] — accumulate onto existing values
    const float *input,         // [n_tokens, n_embd] — FFN input (post-rmsnorm)
    const int *perm,            // [n_e] — token indices for this expert
    const float *tw,            // [n_e] — sigmoid weights per token
    const unsigned char *gate_w, // Q4_K gate weight for this expert
    const unsigned char *up_w,   // Q4_K up weight for this expert
    const unsigned char *down_w, // Q4_K down weight for this expert
    int n_e, int n_embd, int expert_ff,
    size_t stride_gu, size_t stride_d)
{
    __shared__ float sx[2048];   /* input for current token */
    __shared__ float sg[512];    /* gate + SiLU output */
    __shared__ float su[512];    /* up output */
    int tid = threadIdx.x;
    int nb_gu = n_embd / 256, rb_gu = nb_gu * 144;
    int nb_d = expert_ff / 256, rb_d = nb_d * 144;

    for (int ti = 0; ti < n_e; ti++) {
        int tok = perm[ti];
        float wgt = tw[ti];
        const float *xb = input + (size_t)tok * n_embd;

        /* Load input */
        for (int i = tid; i < n_embd; i += 128) sx[i] = xb[i];
        __syncthreads();

        /* Gate: Q4_K matvec, 512 rows */
        for (int r = tid; r < expert_ff; r += 128) {
            const unsigned char *rp = gate_w + (size_t)r * rb_gu;
            float s = 0;
            for (int b = 0; b < nb_gu; b++) {
                const unsigned char *bp = rp + (size_t)b * 144;
                float d = __half2float(*(const __half *)bp);
                float dmin = __half2float(*(const __half *)(bp + 2));
                const unsigned char *sc = bp + 4, *qs = bp + 16;
                const float *xx = sx + b * 256;
                float p = 0; int yi = 0;
                for (int j = 0; j < 4; j++) {
                    int is = j * 2;
                    float sv0f, mv0f, sv1f, mv1f;
                    if (is < 4) {
                        sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
                        sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
                    } else {
                        sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
                        mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
                        sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
                        mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
                    }
                    float d1 = d * sv0f, m1 = dmin * mv0f;
                    float d2 = d * sv1f, m2 = dmin * mv1f;
                    const unsigned char *q = qs + j * 32;
                    for (int l = 0; l < 32; l++)
                        p += (d1 * (q[l] & 0xF) - m1) * xx[yi++];
                    for (int l = 0; l < 32; l++)
                        p += (d2 * (q[l] >> 4) - m2) * xx[yi++];
                }
                s += p;
            }
            sg[r] = s;
        }
        __syncthreads();

        /* Up: Q4_K matvec, 512 rows */
        for (int r = tid; r < expert_ff; r += 128) {
            const unsigned char *rp = up_w + (size_t)r * rb_gu;
            float s = 0;
            for (int b = 0; b < nb_gu; b++) {
                const unsigned char *bp = rp + (size_t)b * 144;
                float d = __half2float(*(const __half *)bp);
                float dmin = __half2float(*(const __half *)(bp + 2));
                const unsigned char *sc = bp + 4, *qs = bp + 16;
                const float *xx = sx + b * 256;
                float p = 0; int yi = 0;
                for (int j = 0; j < 4; j++) {
                    int is = j * 2;
                    float sv0f, mv0f, sv1f, mv1f;
                    if (is < 4) {
                        sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
                        sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
                    } else {
                        sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
                        mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
                        sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
                        mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
                    }
                    float d1 = d * sv0f, m1 = dmin * mv0f;
                    float d2 = d * sv1f, m2 = dmin * mv1f;
                    const unsigned char *q = qs + j * 32;
                    for (int l = 0; l < 32; l++)
                        p += (d1 * (q[l] & 0xF) - m1) * xx[yi++];
                    for (int l = 0; l < 32; l++)
                        p += (d2 * (q[l] >> 4) - m2) * xx[yi++];
                }
                s += p;
            }
            su[r] = s;
        }
        __syncthreads();

        for (int i = tid; i < expert_ff; i += 128)
            sg[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];
        __syncthreads();

        /* Down: Q4_K matvec, 2048 rows */
        for (int r = tid; r < n_embd; r += 128) {
            const unsigned char *rp = down_w + (size_t)r * rb_d;
            float s = 0;
            for (int b = 0; b < nb_d; b++) {
                const unsigned char *bp = rp + (size_t)b * 144;
                float d = __half2float(*(const __half *)bp);
                float dmin = __half2float(*(const __half *)(bp + 2));
                const unsigned char *sc = bp + 4, *qs = bp + 16;
                const float *gg = sg + b * 256;
                float p = 0; int yi = 0;
                for (int j = 0; j < 4; j++) {
                    int is = j * 2;
                    float sv0f, mv0f, sv1f, mv1f;
                    if (is < 4) {
                        sv0f = (float)(sc[is] & 63); mv0f = (float)(sc[is+4] & 63);
                        sv1f = (float)(sc[is+1] & 63); mv1f = (float)(sc[is+1+4] & 63);
                    } else {
                        sv0f = (float)((sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4));
                        mv0f = (float)((sc[is+4] >> 4) | ((sc[is] >> 6) << 4));
                        sv1f = (float)((sc[is+1+4] & 0xF) | ((sc[is+1-4] >> 6) << 4));
                        mv1f = (float)((sc[is+1+4] >> 4) | ((sc[is+1] >> 6) << 4));
                    }
                    float d1 = d * sv0f, m1 = dmin * mv0f;
                    float d2 = d * sv1f, m2 = dmin * mv1f;
                    const unsigned char *q = qs + j * 32;
                    for (int l = 0; l < 32; l++)
                        p += (d1 * (q[l] & 0xF) - m1) * gg[yi++];
                    for (int l = 0; l < 32; l++)
                        p += (d2 * (q[l] >> 4) - m2) * gg[yi++];
                }
                s += p;
            }
            atomicAdd(&output[(size_t)tok * n_embd + r], wgt * s);
        }
        __syncthreads();
    }
}

/* ---- dequant_q8_0_to_f16: Q8_0 weight matrix -> F32 (F16 in name for backwards compat) ---- */
extern "C" __global__ void dequant_q8_0_to_f16(
    float *dst,
    const unsigned char *mat,
    int rows, int cols)
{
    int bid = blockIdx.x * 32 + threadIdx.x;
    int nb = cols / 32, rb = nb * 36;
    int total_blocks = rows * nb;
    if (bid >= total_blocks) return;

    int row = bid / nb;
    int bk = bid % nb;

    const unsigned char *bp = mat + (size_t)row * rb + (size_t)bk * 36;
    float *d = dst + (size_t)row * cols + (size_t)bk * 32;

    float scale = __half2float(*(const __half *)bp);
    const signed char *qs = (const signed char *)(bp + 4);

    for (int i = 0; i < 32; i++) {
        d[i] = scale * (float)qs[i];
    }
}

/* ---- dequant_q8_0_to_f16_h: Q8_0 weight matrix -> real FP16 (half) ----
 * Enables tensor-core F16 GEMM for Q8_0 QKV projections (vs F32 SIMT GEMM).
 * GPU layout is 36 bytes/block (4-byte-aligned f16 scale + 32 int8). One warp
 * per 32-element block, 8 blocks per 256-thread CUDA block; coalesced writes. */
extern "C" __global__ void dequant_q8_0_to_f16_h(
    half *out,
    const unsigned char *mat,
    int rows, int cols)
{
    int nb = cols / 32, rb = nb * 36;
    int total = rows * nb;
    int sb = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (sb >= total) return;
    int lane = threadIdx.x & 31;
    int row = sb / nb, bk = sb % nb;
    const unsigned char *bp = mat + (size_t)row * rb + (size_t)bk * 36;
    half *o = out + (size_t)row * cols + (size_t)bk * 32;
    float scale = __half2float(*(const __half *)bp);
    const signed char *qs = (const signed char *)(bp + 4);
    o[lane] = __float2half(scale * (float)qs[lane]);
}

/* ---- dequant_q6_K_to_f16: Q6_K weight matrix -> FP16 ----
 * Q6_K super-block = 256 elements in 210 bytes:
 *   ql[128] (low 4 bits), qh[64] (high 2 bits), sc[16] (int8 scales), d (f16).
 * One thread per 256-element block. Output layout matches matvec_q6_K_f32 so
 * the dequanted F16 weight feeds cuBLAS F16 GEMM with identical math. */
#define DQ6_SB_PER_BLOCK 4
extern "C" __global__ void dequant_q6_K_to_f16(
    half *out,
    const unsigned char *mat,
    int rows, int cols)
{
    /* Each CUDA block (256 threads) processes DQ6_SB_PER_BLOCK consecutive
     * super-blocks; warp w (of 8) -> super-block (w/2), 64 threads decode its 256
     * elements (4 each). More work per block amortizes launch/scheduling and
     * keeps writes coalesced (consecutive output elements -> consecutive lanes). */
    int nb = cols / 256, rb = nb * 210;
    int total_sb = rows * nb;
    int sb_base = blockIdx.x * DQ6_SB_PER_BLOCK;
    int tpb = blockDim.x / DQ6_SB_PER_BLOCK;   /* threads per super-block = 64 */
    int local = threadIdx.x / tpb;             /* which of the 4 super-blocks */
    int lane = threadIdx.x % tpb;              /* 0..63 */
    int sb = sb_base + local;

    __shared__ unsigned char ss[DQ6_SB_PER_BLOCK][210];
    if (sb < total_sb) {
        int row = sb / nb, bk = sb % nb;
        const unsigned char *bp = mat + (size_t)row * rb + (size_t)bk * 210;
        for (int i = lane; i < 210; i += tpb) ss[local][i] = bp[i];
    }
    __syncthreads();
    if (sb >= total_sb) return;

    int row = sb / nb, bk = sb % nb;
    half *o = out + (size_t)row * cols + (size_t)bk * 256;
    const unsigned char *s = ss[local];
    float d = __half2float(*(const __half *)(s + 208));

    for (int e = lane; e < 256; e += tpb) {
    int half_i = e >> 7;                 /* 0 or 1 */
    int r = e & 127;                     /* 0..127 */
    int grp = r >> 5;                    /* 0..3 */
    int l = r & 31;                      /* 0..31 */
    const unsigned char *ql = s + half_i * 64;
    const unsigned char *qh = s + 128 + half_i * 32;
    const signed char *sc = (const signed char *)(s + 192) + half_i * 8;
    int is = l >> 4;                     /* 0 or 1 */
    int qval, scv;
    if (grp == 0)      { qval = (int)((ql[l]    & 0xF) | (((qh[l]>>0)&3)<<4)); scv = sc[is+0]; }
    else if (grp == 1) { qval = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)); scv = sc[is+2]; }
    else if (grp == 2) { qval = (int)((ql[l]    >> 4)  | (((qh[l]>>4)&3)<<4)); scv = sc[is+4]; }
    else               { qval = (int)((ql[l+32] >> 4)  | (((qh[l]>>6)&3)<<4)); scv = sc[is+6]; }
    o[e] = __float2half(d * scv * (qval - 32));
    }
}
