/* Rectangular BF16 WMMA attention adapted from the validated Trellis kernel. SPDX-License-Identifier: MIT */
#include <hip/hip_runtime.h>
#define T2_SWZ(v,m) __shfl_xor(v,m)
__device__ unsigned short f2b(float x){unsigned u=__float_as_uint(x);return (u+0x7fff+((u>>16)&1))>>16;}
__global__ void q21_fast_attention(unsigned short *__restrict__ out,
    const unsigned short *__restrict__ Q, const unsigned short *__restrict__ K,
    const unsigned short *__restrict__ V,
    int q_len, int kv_len, int qs, int ks, int os, int n_heads, int mask, float scale) {
    (void)n_heads;
    int h = blockIdx.x;
    int cta_q0 = blockIdx.y * 64;
    int tid = threadIdx.x;
    int wave = tid >> 5;
    int lane = tid & 31;
    int half = lane >> 4;
    int idx  = lane & 15;
    /* Padded strides to break LDS bank conflicts (see flash_attn_sa_wmma_bc32). */
    enum { K_STRIDE = 132, V_STRIDE = 18 };
    __shared__ short smK[16*K_STRIDE];
    __shared__ short smVt[128*V_STRIDE];
    __shared__ short smQ[4*16*128];
    __shared__ short smP[4*16*16];
    short *myQ = smQ + wave * (16*128);
    short *myP = smP + wave * (16*16);
    typedef float float8 __attribute__((ext_vector_type(8)));
    typedef short bf16x8 __attribute__((ext_vector_type(8)));
    int wq0 = cta_q0 + wave * 16;
    for (int it = 0; it < 64; it++) {
        int e = lane * 64 + it;
        int r = e >> 7;
        int d = e & 127;
        int qi = wq0 + r;
        myQ[r*128+d] = qi<q_len ? Q[(long)qi*qs+h*128+d] : 0;
    }
    float8 O0 = {0,0,0,0,0,0,0,0};
    float8 O1 = O0, O2 = O0, O3 = O0, O4 = O0, O5 = O0, O6 = O0, O7 = O0;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.f; }
    int n_kv_tiles = (kv_len + 15) / 16;
    for (int kv = 0; kv < n_kv_tiles; kv++) {
        int kv_base = kv * 16;
        for (int it = 0; it < 16; it++) {
            int e = tid * 16 + it;
            int r = e >> 7;
            int d = e & 127;
            int kvi = kv_base + r;
            smK[r*K_STRIDE + d] = kvi<kv_len ? K[(long)kvi*ks + h*128 + d] : 0;
        }
        for (int it = 0; it < 16; it++) {
            int e = tid * 16 + it;
            int r = e >> 7;
            int d = e & 127;
            int kvi = kv_base + r;
            smVt[d*V_STRIDE + r] = kvi<kv_len ? V[(long)kvi*ks + h*128 + d] : 0;
        }
        __syncthreads();
        float8 S = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < 8; kt++) {
            int k_off = kt * 16 + half * 8;
            bf16x8 qa, kb;
            for (int i = 0; i < 8; i++) {
                qa[i] = myQ[idx*128 + k_off + i];
                kb[i] = smK[idx*K_STRIDE + k_off + i];
            }
            S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qa, kb, S);
        }
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float s = S[i] * scale;
            int kvi_col = kv_base + idx;
            if (kvi_col >= kv_len || (mask && kvi_col > wq0+half*8+i+(mask==2 ? kv_len-q_len : 0))) s = -1e30f;
            S[i] = s;
            float m = s;
            m = fmaxf(m, T2_SWZ(m, 1));
            m = fmaxf(m, T2_SWZ(m, 2));
            m = fmaxf(m, T2_SWZ(m, 4));
            m = fmaxf(m, T2_SWZ(m, 8));
            row_max[i] = m;
        }
        float alpha[8];
        for (int i = 0; i < 8; i++) {
            float m_new = fmaxf(m_i[i], row_max[i]);
            alpha[i] = __expf(m_i[i] - m_new);
            float p = __expf(S[i] - m_new);
            S[i] = p;
            float rs = p;
            rs += T2_SWZ(rs, 1);
            rs += T2_SWZ(rs, 2);
            rs += T2_SWZ(rs, 4);
            rs += T2_SWZ(rs, 8);
            l_i[i] = l_i[i] * alpha[i] + rs;
            m_i[i] = m_new;
        }
        float8 *Os[8] = {&O0,&O1,&O2,&O3,&O4,&O5,&O6,&O7};
        for (int t = 0; t < 8; t++)
            for (int i = 0; i < 8; i++) (*Os[t])[i] *= alpha[i];
        for (int i = 0; i < 8; i++) {
            int row = half*8 + i;
            float pv = S[i];
            unsigned int b; memcpy(&b, &pv, 4);
            myP[row*16 + idx] = (short)(b >> 16);
        }
        __builtin_amdgcn_s_waitcnt(0);
        bf16x8 pa;
        for (int i = 0; i < 8; i++) pa[i] = myP[idx*16 + half*8 + i];
        for (int t = 0; t < 8; t++) {
            int hd_base = t * 16;
            bf16x8 vb;
            for (int i = 0; i < 8; i++) vb[i] = smVt[(hd_base + idx)*V_STRIDE + half*8 + i];
            *Os[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pa, vb, *Os[t]);
        }
        __syncthreads();
    }
    float8 *Os[8] = {&O0,&O1,&O2,&O3,&O4,&O5,&O6,&O7};
    float inv_l[8];
    for (int i = 0; i < 8; i++)
        inv_l[i] = (l_i[i] > 0.f) ? (1.f / l_i[i]) : 0.f;
    for (int t = 0; t < 8; t++) {
        int col = t*16 + idx;
        for (int i = 0; i < 8; i++) {
            int qi = wq0 + half*8 + i;
            if (qi < q_len)
                out[(long)qi*os + h*128 + col] = f2b((*Os[t])[i] * inv_l[i]);
        }
    }
}

extern "C" int q21f_attention(void*out,const void*q,const void*k,const void*v,int nq,int nk,int heads,int qs,int ks,int os,int mask,void*s) {
 if(!out||!q||!k||!v||nq<1||nk<1||heads<1||qs<heads*128||ks<heads*128||os<heads*128||mask<0||mask>2)return hipErrorInvalidValue;
 q21_fast_attention<<<dim3(heads,(nq+63)/64),128,0,(hipStream_t)s>>>((unsigned short*)out,(const unsigned short*)q,(const unsigned short*)k,(const unsigned short*)v,nq,nk,qs,ks,os,heads,mask,0.08838834764831845f);
 return hipGetLastError();
}
extern "C" int q21f_flash_attention(void*out,const void*q,const void*k,const void*v,int nq,int nk,int heads,int qs,int ks,int os,int mask,void*s) {
 return q21f_attention(out,q,k,v,nq,nk,heads,qs,ks,os,mask,s);
}
