/* gfx12 WMMA kernels adapted from the native Qwen runners. SPDX-License-Identifier: MIT */
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstdint>
typedef int wm_i32x8 __attribute__((ext_vector_type(8)));
typedef unsigned wm_u32x2 __attribute__((ext_vector_type(2)));
__device__ unsigned wm_pack4_i8(signed char a,signed char b,signed char c,signed char d) { return (unsigned char)a|((unsigned)(unsigned char)b<<8)|((unsigned)(unsigned char)c<<16)|((unsigned)(unsigned char)d<<24); }
__device__ unsigned short f2b(float x) { unsigned u=__float_as_uint(x); return (u+0x7fff+((u>>16)&1))>>16; }
extern "C" __global__ void q21_gemm_bf16_wmma(unsigned short*Y,const unsigned short*W,const unsigned short*X,int no,int ni,int nt,int ldx,int ldy){int tid=threadIdx.x,wave=tid>>5,lane=tid&31,wM=wave&1,wN=wave>>1,half=lane>>4,idx=lane&15;int m0=blockIdx.y*128,n0=blockIdx.x*128;__shared__ short A[128*16],B[128*16];typedef float float8 __attribute__((ext_vector_type(8)));typedef short bf16x8 __attribute__((ext_vector_type(8)));float8 c00={0,0,0,0,0,0,0,0},c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00;for(int k=0;k<ni;k+=16){for(int it=0;it<8;it++){int e=tid*8+it,r=e>>4,q=e&15,rr=m0+r,kk=k+q;A[r*16+q]=(rr<nt&&kk<ni)?X[rr*ldx+kk]:0;}for(int it=0;it<8;it++){int e=tid*8+it,r=e>>4,q=e&15,cc=n0+r,kk=k+q;B[r*16+q]=(cc<no&&kk<ni)?W[cc*ni+kk]:0;}__syncthreads();int am=wM*64,bm=wN*32;bf16x8 a0,a1,a2,a3,b0,b1;for(int i=0;i<8;i++){a0[i]=A[(am+idx)*16+half*8+i];a1[i]=A[(am+16+idx)*16+half*8+i];a2[i]=A[(am+32+idx)*16+half*8+i];a3[i]=A[(am+48+idx)*16+half*8+i];b0[i]=B[(bm+idx)*16+half*8+i];b1[i]=B[(bm+16+idx)*16+half*8+i];}c00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,c00);c01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,c01);c10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,c10);c11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,c11);c20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,c20);c21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,c21);c30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,c30);c31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,c31);__syncthreads();}float8*cs[8]={&c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31};int ms[8]={0,0,16,16,32,32,48,48},ns[8]={0,16,0,16,0,16,0,16};for(int t=0;t<8;t++){int col=n0+wN*32+ns[t]+idx;if(col>=no)continue;float8 v=*cs[t];for(int i=0;i<8;i++){int row=m0+wM*64+ms[t]+half*8+i;if(row<nt)Y[row*ldy+col]=f2b(v[i]);}}}

__global__ void gemm_w8a8_wmma(unsigned short *Y, const signed char *Wq, const float *wscale,
        const signed char *Xq, const float *xscale, const float *bias, int n_out, int n_in, int n_tok, int ldy) {
    int tid=threadIdx.x, wave_id=tid>>5, lane=tid&31, wM=wave_id&1, wN=wave_id>>1;
    int half=lane>>4, idx=lane&15, k_off=half*8, cta_m0=blockIdx.y*128, cta_n0=blockIdx.x*128;
    __shared__ signed char smA[128*16]; __shared__ signed char smB[128*16];
    wm_i32x8 z={0,0,0,0,0,0,0,0}; wm_i32x8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;
    for (int k=0;k<n_in;k+=16){
        for (int it=0;it<8;it++){int e=tid*8+it,er=e>>4,ek=e&15,row=cta_m0+er,kp=k+ek; smA[e]=(row<n_tok&&kp<n_in)?Xq[(long)row*n_in+kp]:(signed char)0;}
        for (int it=0;it<8;it++){int e=tid*8+it,er=e>>4,ek=e&15,col=cta_n0+er,kp=k+ek; smB[e]=(col<n_out&&kp<n_in)?Wq[(long)col*n_in+kp]:(signed char)0;}
        __syncthreads();
        int a_base=wM*64,b_base=wN*32; wm_u32x2 a0,a1,a2,a3,b0,b1;
        a0[0]=wm_pack4_i8(smA[(a_base+0+idx)*16+k_off+0],smA[(a_base+0+idx)*16+k_off+1],smA[(a_base+0+idx)*16+k_off+2],smA[(a_base+0+idx)*16+k_off+3]); a0[1]=wm_pack4_i8(smA[(a_base+0+idx)*16+k_off+4],smA[(a_base+0+idx)*16+k_off+5],smA[(a_base+0+idx)*16+k_off+6],smA[(a_base+0+idx)*16+k_off+7]);
        a1[0]=wm_pack4_i8(smA[(a_base+16+idx)*16+k_off+0],smA[(a_base+16+idx)*16+k_off+1],smA[(a_base+16+idx)*16+k_off+2],smA[(a_base+16+idx)*16+k_off+3]); a1[1]=wm_pack4_i8(smA[(a_base+16+idx)*16+k_off+4],smA[(a_base+16+idx)*16+k_off+5],smA[(a_base+16+idx)*16+k_off+6],smA[(a_base+16+idx)*16+k_off+7]);
        a2[0]=wm_pack4_i8(smA[(a_base+32+idx)*16+k_off+0],smA[(a_base+32+idx)*16+k_off+1],smA[(a_base+32+idx)*16+k_off+2],smA[(a_base+32+idx)*16+k_off+3]); a2[1]=wm_pack4_i8(smA[(a_base+32+idx)*16+k_off+4],smA[(a_base+32+idx)*16+k_off+5],smA[(a_base+32+idx)*16+k_off+6],smA[(a_base+32+idx)*16+k_off+7]);
        a3[0]=wm_pack4_i8(smA[(a_base+48+idx)*16+k_off+0],smA[(a_base+48+idx)*16+k_off+1],smA[(a_base+48+idx)*16+k_off+2],smA[(a_base+48+idx)*16+k_off+3]); a3[1]=wm_pack4_i8(smA[(a_base+48+idx)*16+k_off+4],smA[(a_base+48+idx)*16+k_off+5],smA[(a_base+48+idx)*16+k_off+6],smA[(a_base+48+idx)*16+k_off+7]);
        b0[0]=wm_pack4_i8(smB[(b_base+0+idx)*16+k_off+0],smB[(b_base+0+idx)*16+k_off+1],smB[(b_base+0+idx)*16+k_off+2],smB[(b_base+0+idx)*16+k_off+3]); b0[1]=wm_pack4_i8(smB[(b_base+0+idx)*16+k_off+4],smB[(b_base+0+idx)*16+k_off+5],smB[(b_base+0+idx)*16+k_off+6],smB[(b_base+0+idx)*16+k_off+7]);
        b1[0]=wm_pack4_i8(smB[(b_base+16+idx)*16+k_off+0],smB[(b_base+16+idx)*16+k_off+1],smB[(b_base+16+idx)*16+k_off+2],smB[(b_base+16+idx)*16+k_off+3]); b1[1]=wm_pack4_i8(smB[(b_base+16+idx)*16+k_off+4],smB[(b_base+16+idx)*16+k_off+5],smB[(b_base+16+idx)*16+k_off+6],smB[(b_base+16+idx)*16+k_off+7]);
        cv00=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a0,true,b0,cv00,false); cv01=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a0,true,b1,cv01,false); cv10=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a1,true,b0,cv10,false); cv11=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a1,true,b1,cv11,false);
        cv20=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a2,true,b0,cv20,false); cv21=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a2,true,b1,cv21,false); cv30=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a3,true,b0,cv30,false); cv31=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a3,true,b1,cv31,false);
        __syncthreads();
    }
    int wm0=cta_m0+wM*64,wn0=cta_n0+wN*32; wm_i32x8 *ac[8]={&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31}; int ms[8]={0,0,16,16,32,32,48,48},ns[8]={0,16,0,16,0,16,0,16};
    for(int t=0;t<8;t++){int col=wn0+ns[t]+idx; if(col>=n_out)continue; float wsc=wscale[col],bv=bias?bias[col]:0.f; wm_i32x8 acc=*ac[t]; for(int i=0;i<8;i++){int row=wm0+ms[t]+half*8+i; if(row<n_tok)Y[(long)row*ldy+col]=f2b((float)acc[i]*xscale[row]*wsc+bv);}}
}

/* Tiled W8A8 GEMM for gfx12 (RDNA4). Same math and output as gemm_w8a8_wmma (exact INT32 accumulation, identical
 * epilogue), restructured for memory throughput: 128x128 block tile, BK=64, 16-byte global loads, padded LDS rows
 * (80 B) and a register-staged double buffer so the next K tile is fetched while WMMAs run on the current one.
 * Fragment mapping per 16x16x16 WMMA is unchanged: lane&15 selects the row, lane>>4 the 8-byte K half. */
#define TW_BM 128
#define TW_BN 128
#define TW_BK 64
#define TW_LD 80
__global__ __launch_bounds__(256) void gemm_w8a8_wmma_tiled(unsigned short *Y, const signed char *Wq, const float *wscale,
        const signed char *Xq, const float *xscale, const float *bias, int n_out, int n_in, int n_tok, int ldy) {
    __shared__ __attribute__((aligned(16))) signed char As[2][TW_BM * TW_LD];
    __shared__ __attribute__((aligned(16))) signed char Bs[2][TW_BN * TW_LD];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wM = wave & 1, wN = wave >> 1;
    const int half = lane >> 4, idx = lane & 15;
    const int m0 = blockIdx.y * TW_BM, n0 = blockIdx.x * TW_BN;
    /* each thread moves two 16-byte chunks of A and two of B per K tile: chunk c -> row c>>2, column (c&3)*16 */
    int4 ra[2], rb[2];
    auto load = [&](int k) {
        for (int i = 0; i < 2; i++) {
            int c = tid + i * 256, r = c >> 2, col = (c & 3) * 16;
            int row = m0 + r;
            ra[i] = row < n_tok ? *(const int4 *)(Xq + (long)row * n_in + k + col) : make_int4(0, 0, 0, 0);
            int orow = n0 + r;
            rb[i] = orow < n_out ? *(const int4 *)(Wq + (long)orow * n_in + k + col) : make_int4(0, 0, 0, 0);
        }
    };
    auto store = [&](int buf) {
        for (int i = 0; i < 2; i++) {
            int c = tid + i * 256, r = c >> 2, col = (c & 3) * 16;
            *(int4 *)(&As[buf][r * TW_LD + col]) = ra[i];
            *(int4 *)(&Bs[buf][r * TW_LD + col]) = rb[i];
        }
    };
    wm_i32x8 acc[4][2];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) acc[i][j] = (wm_i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    load(0); store(0); __syncthreads();
    int buf = 0;
    for (int k = 0; k < n_in; k += TW_BK) {
        const bool more = k + TW_BK < n_in;
        if (more) load(k + TW_BK);
        const signed char *A = As[buf], *B = Bs[buf];
#pragma unroll
        for (int ks = 0; ks < TW_BK; ks += 16) {
            wm_u32x2 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = *(const wm_u32x2 *)(A + (wM * 64 + i * 16 + idx) * TW_LD + ks + half * 8);
#pragma unroll
            for (int j = 0; j < 2; j++) b[j] = *(const wm_u32x2 *)(B + (wN * 32 + j * 16 + idx) * TW_LD + ks + half * 8);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a[i], true, b[j], acc[i][j], false);
        }
        if (more) { store(buf ^ 1); __syncthreads(); buf ^= 1; }
    }
    const int wm0 = m0 + wM * 64, wn0 = n0 + wN * 32;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) {
            int col = wn0 + j * 16 + idx;
            if (col >= n_out) continue;
            float wsc = wscale[col], bv = bias ? bias[col] : 0.f;
            wm_i32x8 v = acc[i][j];
            for (int e = 0; e < 8; e++) {
                int row = wm0 + i * 16 + half * 8 + e;
                if (row < n_tok) Y[(long)row * ldy + col] = f2b((float)v[e] * xscale[row] * wsc + bv);
            }
        }
}

extern "C" int q21f_bf16_gemm(void*y,int ldy,const void*w,const void*x,int ldx,int m,int n,int k,void*s) {
 q21_gemm_bf16_wmma<<<dim3((n+127)/128,(m+127)/128),256,0,(hipStream_t)s>>>((unsigned short*)y,(const unsigned short*)w,(const unsigned short*)x,n,k,m,ldx,ldy);
 return hipGetLastError();
}
static int q21f_i8_legacy = -1;
extern "C" int q21f_i8_gemm(void*y,int ldy,const void*x,const float*xs,const void*w,const float*ws,int m,int n,int k,int,void*s) {
 if (q21f_i8_legacy < 0) { const char *e = getenv("Q21F_I8_LEGACY"); q21f_i8_legacy = e && *e == '1'; }
 /* The tiled kernel needs 16-byte aligned rows and whole 64-wide K tiles; otherwise keep the reference kernel. */
 if (!q21f_i8_legacy && k % TW_BK == 0 && ((uintptr_t)x & 15) == 0 && ((uintptr_t)w & 15) == 0)
  gemm_w8a8_wmma_tiled<<<dim3((n+TW_BN-1)/TW_BN,(m+TW_BM-1)/TW_BM),256,0,(hipStream_t)s>>>((unsigned short*)y,(const signed char*)w,ws,(const signed char*)x,xs,nullptr,n,k,m,ldy);
 else
  gemm_w8a8_wmma<<<dim3((n+127)/128,(m+127)/128),256,0,(hipStream_t)s>>>((unsigned short*)y,(const signed char*)w,ws,(const signed char*)x,xs,nullptr,n,k,m,ldy);
 return hipGetLastError();
}
/* Diagnostic entry points for the standalone parity/throughput test (bench_fast_gemm.hip). */
extern "C" int q21f_i8_gemm_reference(void*y,int ldy,const void*x,const float*xs,const void*w,const float*ws,int m,int n,int k,void*s) {
 gemm_w8a8_wmma<<<dim3((n+127)/128,(m+127)/128),256,0,(hipStream_t)s>>>((unsigned short*)y,(const signed char*)w,ws,(const signed char*)x,xs,nullptr,n,k,m,ldy);
 return hipGetLastError();
}
extern "C" int q21f_i8_gemm_tiled(void*y,int ldy,const void*x,const float*xs,const void*w,const float*ws,int m,int n,int k,void*s) {
 if (k % TW_BK) return -1;
 gemm_w8a8_wmma_tiled<<<dim3((n+TW_BN-1)/TW_BN,(m+TW_BM-1)/TW_BM),256,0,(hipStream_t)s>>>((unsigned short*)y,(const signed char*)w,ws,(const signed char*)x,xs,nullptr,n,k,m,ldy);
 return hipGetLastError();
}
