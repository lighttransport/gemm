/*
 * moe_gpu_kernels.cu — MoE GPU kernels for 35B-A3B decode.
 * AOT: nvcc -cubin -arch=sm_120 -o moe_gpu_kernels.cubin moe_gpu_kernels.cu
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void moe_topk_gpu(int *idx_out, float *wgt_out, const float *logits, int n_experts, int n_used) {
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
        idx_out[lane] = fi[lane]; wgt_out[lane] = 1.0f / (1.0f + expf(-fv[lane]));
    }
}

extern "C" __global__ void moe_shared_gate_gpu(float *go, const float *xb, const float *gw, int ne) {
    float s = 0;
    for (int i = threadIdx.x; i < ne; i += 128) s += gw[i] * xb[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) *go = 1.0f / (1.0f + expf(-s));
}

/* moe_f16_tc: general F16 TC matvec. grid=(out_dim/16,1), block=32 */
extern "C" __global__ void moe_f16_tc(float *out, const half *w, const float *x, int od, int id) {
    int r0 = blockIdx.x * 16; if (r0 >= od) return;
    int l = threadIdx.x;
    float rc[4] = {0,0,0,0};
    for (int k = 0; k < id; k += 16) {
        int row = r0 + l/2, ck = k + (l%2)*8;
        half w16[8];
        for (int j = 0; j < 8; j++) w16[j] = w[(size_t)row * id + ck + j];
        unsigned ra[4]; ra[0]=*(const unsigned*)&w16[0];ra[1]=*(const unsigned*)&w16[2];
        ra[2]=*(const unsigned*)&w16[4];ra[3]=*(const unsigned*)&w16[6];
        int bk = ((l%8)/2)*2, kk = k+bk;
        half b16[4];
        b16[0]=__float2half(x[kk]);b16[1]=__float2half(x[kk+1]);
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

/* IQ2_S TC matvec. grid=(32,8), block=32 */
extern "C" __global__ void moe_iq2s_tc(float *out, const float *xb,
    const unsigned char *w, const int *ei, const unsigned long long *g2,
    int nu, int ef, int ne, size_t sg)
{
    int ex = blockIdx.y, r0 = blockIdx.x * 16; int eidx = ei[ex]; if (eidx<0||r0>=ef) return;
    int l = threadIdx.x; float *o = out + (size_t)ex * ef; int nb = ne/256, rb = nb*82;
    float rc[4] = {0,0,0,0};
    for (int k = 0; k < ne; k += 16) {
        int row = r0 + l/2, ck = k + (l%2)*8, bi = ck/256, ib = (ck%256)/32, tl = ((k+(l%2)*8)%32)/8;
        const unsigned char *bp = w + (size_t)eidx*sg + (size_t)row*rb + bi*82;
        float d = __half2float(*(const __half *)bp);
        float da = d*(.5f+(bp[74+ib]&0xf))*.25f, db = d*(.5f+(bp[74+ib]>>4))*.25f;
        int gid = bp[2+ib*4+tl]|((bp[66+ib]<<(8-2*tl))&0x300);
        unsigned long long gv = g2[gid]; unsigned char sn = bp[34+ib*4+tl];
        float dl = (tl<2)?da:db; half w16[8];
        for (int j = 0; j < 8; j++) {
            float wv = dl*(float)(unsigned char)(gv>>(8*j));
            w16[j] = __float2half((sn>>j)&1?-wv:wv);
        }
        unsigned ra[4];ra[0]=*(const unsigned*)&w16[0];ra[1]=*(const unsigned*)&w16[2];
        ra[2]=*(const unsigned*)&w16[4];ra[3]=*(const unsigned*)&w16[6];
        int bk = ((l%8)/2)*2, kk = k+bk;
        half b16[4];b16[0]=__float2half(xb[kk]);b16[1]=__float2half(xb[kk+1]);
        b16[2]=__float2half(xb[kk]);b16[3]=__float2half(xb[kk+1]);
        unsigned rb2[2];rb2[0]=*(const unsigned*)&b16[0];rb2[1]=*(const unsigned*)&b16[2];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            :"+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
            :"r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb2[0]),"r"(rb2[1]),
             "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
    }
    int m0=l/4,m1=m0+8;if((l&3)==0){o[r0+m0]=rc[0];o[r0+m1]=rc[2];}
}

/* IQ3_S TC down matvec. grid=(128,8), block=32 */
extern "C" __global__ void moe_iq3s_tc(float *out, const float *in,
    const unsigned char *w, const int *ei, const unsigned int *g3,
    int nu, int ne, int ef, size_t sd)
{
    int ex = blockIdx.y, r0 = blockIdx.x * 16; int eidx = ei[ex]; if (eidx<0||r0>=ne) return;
    int l = threadIdx.x; float *o = out + (size_t)ex * ne;
    const float *xx = in + (size_t)ex * ef; int nb = ef/256, rb = nb*110;
    float rc[4] = {0,0,0,0};
    for (int k = 0; k < ef; k += 16) {
        int row = r0 + l/2, ck = k + (l%2)*8;
        unsigned char *bp = (unsigned char*)w + (size_t)eidx*sd + (size_t)row*rb + (ck/256)*110;
        float d = __half2float(*(const __half *)bp);
        const unsigned char *qs0 = bp+2, *qh = bp+66, *sgnb = bp+74, *scl = bp+106;
        int ib32 = (ck%256)/32, tl = ((ck%256)%32)/8;
        int sc = ib32/2, ns = (ib32%2)*4;
        float db = d*(float)(1+2*((scl[sc]>>ns)&0xf));
        int qo = ib32*8+tl*2;
        int g1 = qs0[qo+0]|((qh[ib32]<<(8-2*tl))&256);
        int g2 = qs0[qo+1]|((qh[ib32]<<(7-2*tl))&256);
        const unsigned char *gr1 = (const unsigned char *)&g3[g1];
        const unsigned char *gr2 = (const unsigned char *)&g3[g2];
        unsigned char sb = sgnb[ib32*4+tl];
        half w16[8];
        for (int j = 0; j < 4; j++) {
            float w0 = db*(float)gr1[j], w1 = db*(float)gr2[j];
            w16[j]   = __float2half((sb&(1<<j))?-w0:w0);
            w16[j+4] = __float2half((sb&(1<<(j+4)))?-w1:w1);
        }
        unsigned ra[4];ra[0]=*(const unsigned*)&w16[0];ra[1]=*(const unsigned*)&w16[2];
        ra[2]=*(const unsigned*)&w16[4];ra[3]=*(const unsigned*)&w16[6];
        int bk = ((l%8)/2)*2, kk = k+bk;
        half b16[4];b16[0]=__float2half(xx[kk]);b16[1]=__float2half(xx[kk+1]);
        b16[2]=__float2half(xx[kk]);b16[3]=__float2half(xx[kk+1]);
        unsigned rb2[2];rb2[0]=*(const unsigned*)&b16[0];rb2[1]=*(const unsigned*)&b16[2];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            :"+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
            :"r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb2[0]),"r"(rb2[1]),
             "f"(rc[0]),"f"(rc[1]),"f"(rc[2]),"f"(rc[3]));
    }
    int m0=l/4,m1=m0+8;if((l&3)==0){o[r0+m0]=rc[0];o[r0+m1]=rc[2];}
}

/* Fused accumulate for all 8 experts */
extern "C" __global__ void moe_accum_gpu(float *accum, const float *down, const float *wgt, int nu, int ne) {
    int i = blockIdx.x*256+threadIdx.x; if (i>=ne) return;
    float s = 0; for (int e=0;e<nu;e++) s+=wgt[e]*down[(size_t)e*ne+i];
    accum[i]+=s;
}
