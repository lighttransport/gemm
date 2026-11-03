#include "wmma_modular.hpp"

namespace ozaki2_wmma {
using i2 = int __attribute__((ext_vector_type(2)));
using i8 = int __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));

template<bool fp8>
__global__ void product_kernel(int m, int n, int k, const uint8_t *a,
                               size_t lda, size_t sa, const uint8_t *b,
                               size_t ldb, size_t sb, void *out, size_t ldc,
                               size_t sc, bool accumulate) {
    int lane = int(threadIdx.x), col = int(blockIdx.x) * 16 + (lane & 15);
    int row_base = int(blockIdx.y) * 16, half = lane >> 4;
    if (a) a += size_t(blockIdx.z) * sa;
    if (b) b += size_t(blockIdx.z) * sb;
    i8 ic = {}; f8 fc = {};
    if (accumulate) {
        for (int t = 0; t < 8; ++t) {
            int row = row_base + half * 8 + t;
            if (row < m && col < n) {
                size_t p = size_t(blockIdx.z) * sc + size_t(col) * ldc + row;
                if constexpr (fp8) fc[t] = static_cast<float *>(out)[p];
                else ic[t] = static_cast<int32_t *>(out)[p];
            }
        }
    }
    for (int kk = 0; kk < k; kk += 16) {
        int arow = row_base + (lane & 15);
        i2 av = {}, bv = {};
        for (int t = 0; t < 8; ++t) {
            int z = kk + half * 8 + t;
            uint32_t x = (arow < m && z < k) ? a[size_t(arow) * lda + z] : 0;
            uint32_t y = (col < n && z < k) ? b[size_t(col) * ldb + z] : 0;
            av[t / 4] = int(uint32_t(av[t / 4]) | (x << (8 * (t % 4))));
            bv[t / 4] = int(uint32_t(bv[t / 4]) | (y << (8 * (t % 4))));
        }
        if constexpr (fp8)
            fc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, fc);
        else
            ic = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, true, bv, ic, false);
    }
    for (int t = 0; t < 8; ++t) {
        int row = row_base + half * 8 + t;
        if (row < m && col < n) {
            size_t p = size_t(blockIdx.z) * sc + size_t(col) * ldc + row;
            if constexpr (fp8) static_cast<float *>(out)[p] = fc[t];
            else static_cast<int32_t *>(out)[p] = ic[t];
        }
    }
}

template<bool fp8>
__global__ __launch_bounds__(128) void product_kernel64(int m, int n, int k,
        const uint8_t *a, size_t lda, size_t sa, const uint8_t *b, size_t ldb,
        size_t sb, void *out, size_t ldc, size_t sc, bool accumulate) {
    __shared__ uint8_t as[64][32], bs[64][32];
    int tid=int(threadIdx.x), lane=tid&31, wave=tid>>5, idx=lane&15, half=lane>>4;
    int wm=wave&1, wn=wave>>1, row0=int(blockIdx.y)*64, col0=int(blockIdx.x)*64;
    if(a)a+=size_t(blockIdx.z)*sa;
    if(b)b+=size_t(blockIdx.z)*sb;
    i8 ic[2][2]={}; f8 fc[2][2]={};
    if(accumulate) {
        for(int u=0;u<2;u++)for(int v=0;v<2;v++)for(int t=0;t<8;t++) {
            int row=row0+wm*32+u*16+half*8+t,col=col0+wn*32+v*16+idx;
            if(row<m&&col<n) {
                size_t p=size_t(blockIdx.z)*sc+size_t(col)*ldc+row;
                if constexpr(fp8)fc[u][v][t]=static_cast<float *>(out)[p];
                else ic[u][v][t]=static_cast<int32_t *>(out)[p];
            }
        }
    }
    for(int kk=0;kk<k;kk+=32) {
        for(int q=tid;q<2048;q+=128) {
            int r=q/32,z=kk+q%32;
            as[r][q%32]=(row0+r<m&&z<k)?a[size_t(row0+r)*lda+z]:0;
            bs[r][q%32]=(col0+r<n&&z<k)?b[size_t(col0+r)*ldb+z]:0;
        }
        __syncthreads();
        for(int h=0;h<32;h+=16) {
            i2 av[2]={},bv[2]={};
            for(int u=0;u<2;u++)for(int t=0;t<8;t++) {
                int z=h+half*8+t, shift=8*(t%4);
                uint32_t x=as[wm*32+u*16+idx][z],y=bs[wn*32+u*16+idx][z];
                av[u][t/4]=int(uint32_t(av[u][t/4])|(x<<shift));
                bv[u][t/4]=int(uint32_t(bv[u][t/4])|(y<<shift));
            }
            for(int u=0;u<2;u++)for(int v=0;v<2;v++) {
                if constexpr(fp8)fc[u][v]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av[u],bv[v],fc[u][v]);
                else ic[u][v]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,av[u],true,bv[v],ic[u][v],false);
            }
        }
        __syncthreads();
    }
    for(int u=0;u<2;u++)for(int v=0;v<2;v++)for(int t=0;t<8;t++) {
        int row=row0+wm*32+u*16+half*8+t,col=col0+wn*32+v*16+idx;
        if(row<m&&col<n) {
            size_t p=size_t(blockIdx.z)*sc+size_t(col)*ldc+row;
            if constexpr(fp8)static_cast<float *>(out)[p]=fc[u][v][t];
            else static_cast<int32_t *>(out)[p]=ic[u][v][t];
        }
    }
}

// Variant bits select M=64/128, N=64/128, K=32/64, waves=4/8,
// and single/double LDS buffers. One wave owns a rectangular group of
// 16x16 WMMA fragments; all fragments keep exact integer accumulators.
static thread_local int selected_variant = -1;
void set_variant(int variant) { selected_variant = variant; }

template<bool fp8, int variant>
__global__ void product_variant_kernel(int m,int n,int k,const uint8_t *a,size_t lda,size_t sa,
                                        const uint8_t *b,size_t ldb,size_t sb,void *out,
                                        size_t ldc,size_t sc,bool accumulate) {
    constexpr int BM=(variant&1)?128:64, BN=(variant&2)?128:64, BK=(variant&4)?64:32;
    constexpr int WM=(variant&8)?4:2, WN=2, AM=BM/(WM*16), AN=BN/(WN*16);
    constexpr int BUFS=(variant&16)?2:1, THREADS=(variant&8)?256:128;
    __shared__ uint8_t as[BUFS][BM][BK],bs[BUFS][BN][BK];
    int tid=int(threadIdx.x),lane=tid&31,wave=tid>>5,idx=lane&15,half=lane>>4;
    int wm=wave%WM,wn=wave/WM,row0=int(blockIdx.y)*BM,col0=int(blockIdx.x)*BN;
    if(a)a+=size_t(blockIdx.z)*sa;if(b)b+=size_t(blockIdx.z)*sb;
    i8 ic[AM][AN]={};f8 fc[AM][AN]={};
    if(accumulate) {
#pragma unroll
        for(int u=0;u<AM;u++)for(int v=0;v<AN;v++)for(int t=0;t<8;t++) {
            int row=row0+wm*(AM*16)+u*16+half*8+t,col=col0+wn*(AN*16)+v*16+idx;
            if(row<m&&col<n){size_t p=size_t(blockIdx.z)*sc+size_t(col)*ldc+row;
                if constexpr(fp8)fc[u][v][t]=static_cast<float *>(out)[p];else ic[u][v][t]=static_cast<int32_t *>(out)[p];}
        }
    }
    constexpr int NEXT_A=(BM*BK)/THREADS, NEXT_B=(BN*BK)/THREADS;
    if constexpr(BUFS==2) {
        for(int q=tid;q<BM*BK;q+=THREADS){int r=q/BK,z=q%BK;as[0][r][z]=(row0+r<m&&z<k)?a[size_t(row0+r)*lda+z]:0;}
        for(int q=tid;q<BN*BK;q+=THREADS){int r=q/BK,z=q%BK;bs[0][r][z]=(col0+r<n&&z<k)?b[size_t(col0+r)*ldb+z]:0;}
        __syncthreads();
    }
    for(int kk=0;kk<k;kk+=BK){
        int buffer=(kk/BK)%BUFS;
        if constexpr(BUFS==1) {
            for(int q=tid;q<BM*BK;q+=THREADS){int r=q/BK,z=kk+q%BK;as[0][r][q%BK]=(row0+r<m&&z<k)?a[size_t(row0+r)*lda+z]:0;}
            for(int q=tid;q<BN*BK;q+=THREADS){int r=q/BK,z=kk+q%BK;bs[0][r][q%BK]=(col0+r<n&&z<k)?b[size_t(col0+r)*ldb+z]:0;}
            __syncthreads();
        }
        // Prefetch the next K tile into registers before issuing the current
        // WMMA operations. The alternate LDS buffer is filled afterwards.
        uint32_t next_a[BUFS==2?NEXT_A:1],next_b[BUFS==2?NEXT_B:1];
        if constexpr(BUFS==2) {
            if(kk+BK<k) {
#pragma unroll
                for(int t=0;t<NEXT_A;t++){int q=tid+t*THREADS,r=q/BK,z=kk+BK+q%BK;next_a[t]=(row0+r<m&&z<k)?a[size_t(row0+r)*lda+z]:0;}
#pragma unroll
                for(int t=0;t<NEXT_B;t++){int q=tid+t*THREADS,r=q/BK,z=kk+BK+q%BK;next_b[t]=(col0+r<n&&z<k)?b[size_t(col0+r)*ldb+z]:0;}
            }
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for(int h=0;h<BK;h+=16){
            i2 av[AM]={},bv[AN]={};
#pragma unroll
            for(int u=0;u<AM;u++)for(int t=0;t<8;t++){int z=h+half*8+t,shift=8*(t%4);uint32_t x=as[buffer][wm*(AM*16)+u*16+idx][z];av[u][t/4]=int(uint32_t(av[u][t/4])|(x<<shift));}
#pragma unroll
            for(int v=0;v<AN;v++)for(int t=0;t<8;t++){int z=h+half*8+t,shift=8*(t%4);uint32_t x=bs[buffer][wn*(AN*16)+v*16+idx][z];bv[v][t/4]=int(uint32_t(bv[v][t/4])|(x<<shift));}
#pragma unroll
            for(int u=0;u<AM;u++)for(int v=0;v<AN;v++) {
                if constexpr(fp8)fc[u][v]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av[u],bv[v],fc[u][v]);
                else ic[u][v]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,av[u],true,bv[v],ic[u][v],false);
            }
        }
        if constexpr(BUFS==2) {
            if(kk+BK<k) {
                int next_buffer=buffer^1;
#pragma unroll
                for(int t=0;t<NEXT_A;t++){int q=tid+t*THREADS;as[next_buffer][q/BK][q%BK]=uint8_t(next_a[t]);}
#pragma unroll
                for(int t=0;t<NEXT_B;t++){int q=tid+t*THREADS;bs[next_buffer][q/BK][q%BK]=uint8_t(next_b[t]);}
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int u=0;u<AM;u++)for(int v=0;v<AN;v++)for(int t=0;t<8;t++) {
        int row=row0+wm*(AM*16)+u*16+half*8+t,col=col0+wn*(AN*16)+v*16+idx;
        if(row<m&&col<n){size_t p=size_t(blockIdx.z)*sc+size_t(col)*ldc+row;
            if constexpr(fp8)static_cast<float *>(out)[p]=fc[u][v][t];else static_cast<int32_t *>(out)[p]=ic[u][v][t];}
    }
}

template<bool fp8>
static void launch_variant(int v,dim3 grid,dim3 block,hipStream_t stream,int m,int n,int k,
                           const uint8_t *a,size_t lda,size_t sa,const uint8_t *b,size_t ldb,size_t sb,
                           void *c,size_t ldc,size_t sc,bool accumulate) {
#define LAUNCH(V) case V: hipLaunchKernelGGL((product_variant_kernel<fp8,V>),grid,block,0,stream,m,n,k,a,lda,sa,b,ldb,sb,c,ldc,sc,accumulate);break
    switch(v){
        LAUNCH(0);LAUNCH(1);LAUNCH(2);LAUNCH(3);LAUNCH(4);LAUNCH(5);LAUNCH(6);LAUNCH(7);
        LAUNCH(8);LAUNCH(9);LAUNCH(10);LAUNCH(11);LAUNCH(12);LAUNCH(13);LAUNCH(14);LAUNCH(15);
        LAUNCH(16);LAUNCH(17);LAUNCH(18);LAUNCH(19);LAUNCH(20);LAUNCH(21);LAUNCH(22);LAUNCH(23);
        LAUNCH(24);LAUNCH(25);LAUNCH(26);LAUNCH(27);LAUNCH(28);LAUNCH(29);LAUNCH(30);LAUNCH(31);
    }
#undef LAUNCH
}

hipError_t product(backend type, hipStream_t stream, int m, int n, int k,
                   const uint8_t *a, size_t lda, size_t sa,
                   const uint8_t *b, size_t ldb, size_t sb,
                   void *c, size_t ldc, size_t sc, int batches, bool accumulate) {
    if ((type != backend::int8 && type != backend::fp8) || m < 0 || n < 0 || k < 0 ||
        batches < 0 || k > (type == backend::fp8 ? max_k_fp8 : max_k_int8) ||
        lda < size_t(k) || ldb < size_t(k) || ldc < size_t(m)) return hipErrorInvalidValue;
    if (!m || !n || !batches) return hipSuccess;
    if (!c || (k && (!a || !b)) || batches > 65535) return hipErrorInvalidValue;
    if (selected_variant >= 0 && selected_variant < 32 && m >= 32 && n >= 32) {
        int bm=(selected_variant&1)?128:64,bn=(selected_variant&2)?128:64;
        dim3 grid((n+bn-1)/bn,(m+bm-1)/bm,batches),block((selected_variant&8)?256:128);
        if(type==backend::fp8)launch_variant<true>(selected_variant,grid,block,stream,m,n,k,a,lda,sa,b,ldb,sb,c,ldc,sc,accumulate);
        else launch_variant<false>(selected_variant,grid,block,stream,m,n,k,a,lda,sa,b,ldb,sb,c,ldc,sc,accumulate);
        return hipGetLastError();
    }
    if (m >= 32 && n >= 32) {
        dim3 grid((n + 63) / 64, (m + 63) / 64, batches);
        if (type == backend::fp8)
            hipLaunchKernelGGL((product_kernel64<true>), grid, dim3(128), 0, stream,
                               m,n,k,a,lda,sa,b,ldb,sb,c,ldc,sc,accumulate);
        else
            hipLaunchKernelGGL((product_kernel64<false>), grid, dim3(128), 0, stream,
                               m,n,k,a,lda,sa,b,ldb,sb,c,ldc,sc,accumulate);
        return hipGetLastError();
    }
    dim3 grid((n + 15) / 16, (m + 15) / 16, batches);
    if (type == backend::fp8)
        hipLaunchKernelGGL((product_kernel<true>), grid, dim3(32), 0, stream,
                           m, n, k, a, lda, sa, b, ldb, sb, c, ldc, sc, accumulate);
    else
        hipLaunchKernelGGL((product_kernel<false>), grid, dim3(32), 0, stream,
                           m, n, k, a, lda, sa, b, ldb, sb, c, ldc, sc, accumulate);
    return hipGetLastError();
}
}
