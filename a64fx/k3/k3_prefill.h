#ifndef K3_PREFILL_H
#define K3_PREFILL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "k3_dense.h"

#define K3_PREFILL_MR 8
#define K3_PREFILL_NR 48
#define K3_PREFILL_PV_PREFIX 64
#ifndef K3_PREFILL_BATCH_PANEL
#define K3_PREFILL_BATCH_PANEL 256
#endif

extern void micro_kernel_bf16B_8x3_unroll4_pv(
    const float *a_packed, const uint16_t *b_packed, float *c,
    int64_t k, int64_t unused, int64_t ldc_bytes);

static inline size_t k3_prefill_bf16_packed_bytes(int rows, int cols) {
    size_t nblocks=(size_t)(rows+K3_PREFILL_NR-1)/K3_PREFILL_NR;
    size_t kround=(size_t)(cols+3)&~(size_t)3;
    return K3_PREFILL_PV_PREFIX+nblocks*kround*K3_PREFILL_NR*sizeof(uint16_t);
}

static inline size_t k3_prefill_bf16_scratch_bytes(int batch, int cols) {
    int panel=batch>K3_PREFILL_BATCH_PANEL?K3_PREFILL_BATCH_PANEL:batch;
    size_t mblocks=(size_t)(panel+K3_PREFILL_MR-1)/K3_PREFILL_MR;
    size_t kround=(size_t)(cols+3)&~(size_t)3;
    return mblocks*kround*K3_PREFILL_MR*sizeof(float);
}

/* Directly pack row-major W[N,K] into the pair-interleaved Kx48 layout.
 * This avoids constructing a full W^T staging buffer. */
static inline int k3_prefill_pack_bf16_pv(uint16_t *packed, size_t packed_bytes,
        const k3_bf16_matrix *matrix, int threads) {
    if(!packed||!matrix||!matrix->weight||matrix->rows<1||matrix->cols<1||
       matrix->cols%4||packed_bytes<k3_prefill_bf16_packed_bytes(matrix->rows,matrix->cols)||
       threads<1)return-1;
    memset(packed,0,K3_PREFILL_PV_PREFIX);
    uint16_t *body=(uint16_t*)((uint8_t*)packed+K3_PREFILL_PV_PREFIX);
    int nblocks=(matrix->rows+K3_PREFILL_NR-1)/K3_PREFILL_NR;
    int kround=(matrix->cols+3)&~3;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for(int nb=0;nb<nblocks;++nb){
        int n0=nb*K3_PREFILL_NR,ncount=matrix->rows-n0;
        if(ncount>K3_PREFILL_NR)ncount=K3_PREFILL_NR;
        uint16_t *dst=body+(size_t)nb*kround*K3_PREFILL_NR;
        for(int kp=0;kp<kround;kp+=2)for(int c=0;c<3;++c){
            uint16_t *chunk=dst+(size_t)(kp/2)*(K3_PREFILL_NR*2)+c*32;
            for(int i=0;i<16;++i){int n=c*16+i;uint16_t v0=0,v1=0;
                if(n<ncount&&kp<matrix->cols)v0=matrix->weight[(size_t)(n0+n)*matrix->cols+kp];
                if(n<ncount&&kp+1<matrix->cols)v1=matrix->weight[(size_t)(n0+n)*matrix->cols+kp+1];
                chunk[2*i]=v0;chunk[2*i+1]=v1;}}
    }
    return 0;
}

/* Pack from full_bf16_pv_repack's 8-row pair-plane representation.  Complete
 * groups of eight rows are interleaved; a short final group remains row-major. */
static inline int k3_prefill_pack_bf16_decode_pv(uint16_t *packed,
        size_t packed_bytes,const k3_bf16_matrix *matrix,int threads) {
    if(!packed||!matrix||!matrix->weight||matrix->rows<1||matrix->cols<1||
       matrix->cols%4||packed_bytes<k3_prefill_bf16_packed_bytes(matrix->rows,matrix->cols)||
       threads<1)return-1;
    memset(packed,0,K3_PREFILL_PV_PREFIX);
    uint16_t *body=(uint16_t*)((uint8_t*)packed+K3_PREFILL_PV_PREFIX);
    int nblocks=(matrix->rows+K3_PREFILL_NR-1)/K3_PREFILL_NR;
    int kround=(matrix->cols+3)&~3;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for(int nb=0;nb<nblocks;++nb){
        int n0=nb*K3_PREFILL_NR,ncount=matrix->rows-n0;
        if(ncount>K3_PREFILL_NR)ncount=K3_PREFILL_NR;
        uint16_t *dst=body+(size_t)nb*kround*K3_PREFILL_NR;
        for(int kp=0;kp<kround;kp+=2)for(int c=0;c<3;++c){
            uint16_t *chunk=dst+(size_t)(kp/2)*(K3_PREFILL_NR*2)+c*32;
            for(int i=0;i<16;++i){int n=c*16+i;uint16_t v0=0,v1=0;
                if(n<ncount){int row=n0+n,group=row&~7,within=row&7;
                    const uint16_t *src;
                    if(group+8<=matrix->rows)
                        src=matrix->weight+(size_t)group*matrix->cols+
                            (size_t)(within>>1)*2*matrix->cols+(within&1);
                    else src=matrix->weight+(size_t)row*matrix->cols;
                    if(kp<matrix->cols)v0=src[(size_t)2*kp];
                    if(kp+1<matrix->cols)v1=src[(size_t)2*(kp+1)];
                }
                chunk[2*i]=v0;chunk[2*i+1]=v1;}}
    }
    return 0;
}

static inline void k3_prefill_pack_activation_block(float *dst,
        const float *src,int batch,int cols,int mb,int kround){
    int m0=mb*K3_PREFILL_MR,mcount=batch-m0;
    if(mcount>K3_PREFILL_MR)mcount=K3_PREFILL_MR;
    for(int k=0;k<cols;++k){int m=0;for(;m<mcount;++m)
        dst[(size_t)k*K3_PREFILL_MR+m]=src[(size_t)(m0+m)*cols+k];
        for(;m<K3_PREFILL_MR;++m)dst[(size_t)k*K3_PREFILL_MR+m]=0.0f;}
    for(int k=cols;k<kround;++k)for(int m=0;m<K3_PREFILL_MR;++m)
        dst[(size_t)k*K3_PREFILL_MR+m]=0.0f;
}

/* C[M,N] = X[M,K] * W[N,K]^T. The caller owns both persistent packed
 * weights and activation scratch, so this path performs no allocation. */
static inline int k3_prefill_gemm_bf16_pv(float *out,const float *x,int batch,
        const k3_bf16_matrix *matrix,const uint16_t *packed,
        float *scratch,size_t scratch_bytes,int threads){
    if(!out||!x||!matrix||!packed||!scratch||batch<1||threads<1||
       matrix->rows<1||matrix->cols<1||matrix->cols%4||
       scratch_bytes<k3_prefill_bf16_scratch_bytes(batch,matrix->cols))return-1;
    int kround=(matrix->cols+3)&~3;
    int nblocks=(matrix->rows+K3_PREFILL_NR-1)/K3_PREFILL_NR;
    const uint16_t *body=(const uint16_t*)((const uint8_t*)packed+K3_PREFILL_PV_PREFIX);
    for(int token0=0;token0<batch;token0+=K3_PREFILL_BATCH_PANEL){int panel=batch-token0;
    if(panel>K3_PREFILL_BATCH_PANEL)panel=K3_PREFILL_BATCH_PANEL;int mblocks=(panel+K3_PREFILL_MR-1)/K3_PREFILL_MR;
    const float *panel_x=x+(size_t)token0*matrix->cols;
    float *panel_out=out+(size_t)token0*matrix->rows;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
#pragma omp for schedule(static)
#endif
    for(int mb=0;mb<mblocks;++mb)k3_prefill_pack_activation_block(
        scratch+(size_t)mb*kround*K3_PREFILL_MR,panel_x,panel,matrix->cols,mb,kround);
#if defined(_OPENMP)
#pragma omp for collapse(2) schedule(static)
#endif
    for(int mb=0;mb<mblocks;++mb)for(int nb=0;nb<nblocks;++nb){
        int m0=mb*K3_PREFILL_MR,n0=nb*K3_PREFILL_NR;
        int mc=panel-m0,nc=matrix->rows-n0;if(mc>K3_PREFILL_MR)mc=K3_PREFILL_MR;
        if(nc>K3_PREFILL_NR)nc=K3_PREFILL_NR;
        const float *at=scratch+(size_t)mb*kround*K3_PREFILL_MR;
        const uint16_t *bt=body+(size_t)nb*kround*K3_PREFILL_NR;
        if(mc==K3_PREFILL_MR&&nc==K3_PREFILL_NR)
            micro_kernel_bf16B_8x3_unroll4_pv(at,bt,panel_out+(size_t)m0*matrix->rows+n0,
                kround,0,(int64_t)matrix->rows*sizeof(float));
        else{float tmp[K3_PREFILL_MR*K3_PREFILL_NR] __attribute__((aligned(256)));
            micro_kernel_bf16B_8x3_unroll4_pv(at,bt,tmp,kround,0,
                K3_PREFILL_NR*sizeof(float));
            for(int m=0;m<mc;++m)memcpy(panel_out+(size_t)(m0+m)*matrix->rows+n0,
                tmp+(size_t)m*K3_PREFILL_NR,(size_t)nc*sizeof(float));}
    }
#if defined(_OPENMP)
    }
#endif
    }
    return 0;
}

/* Multi-output variant for projections sharing the same activation matrix.
 * The activation panel is packed once, then one persistent team walks the
 * output tensors.  This is used by MoE routed-down/shared gate/shared-up. */
static inline int k3_prefill_gemm_bf16_pv_many(float *const *outs,
        const k3_bf16_matrix *const *matrices,
        const uint16_t *const *packed, int count, const float *x, int batch,
        float *scratch, size_t scratch_bytes, int threads) {
    if (!outs || !matrices || !packed || count < 1 || !x || !scratch ||
        batch < 1 || threads < 1) return -1;
    int cols = matrices[0]->cols;
    if (cols < 1 || scratch_bytes < k3_prefill_bf16_scratch_bytes(batch, cols))
        return -1;
    for (int p = 0; p < count; ++p)
        if (!matrices[p] || !packed[p] || matrices[p]->cols != cols ||
            matrices[p]->rows < 1 || matrices[p]->cols % 4) return -1;
    for (int token0 = 0; token0 < batch; token0 += K3_PREFILL_BATCH_PANEL) {
        int panel = batch - token0;
        if (panel > K3_PREFILL_BATCH_PANEL) panel = K3_PREFILL_BATCH_PANEL;
        int mblocks = (panel + K3_PREFILL_MR - 1) / K3_PREFILL_MR;
#if defined(_OPENMP)
        omp_set_num_threads(threads);
#pragma omp parallel
        {
#pragma omp for schedule(static)
#endif
        for (int mb = 0; mb < mblocks; ++mb)
            k3_prefill_pack_activation_block(
                scratch + (size_t)mb * ((cols + 3) & ~3) * K3_PREFILL_MR,
                x + (size_t)token0 * cols, panel, cols, mb, (cols + 3) & ~3);
#if defined(_OPENMP)
#pragma omp for collapse(3) schedule(static)
#endif
        for (int p = 0; p < count; ++p)
            for (int mb = 0; mb < mblocks; ++mb)
                for (int nb = 0; nb < (matrices[p]->rows + K3_PREFILL_NR - 1) / K3_PREFILL_NR;
                     ++nb) {
                    int rows = matrices[p]->rows;
                    int m0 = mb * K3_PREFILL_MR, n0 = nb * K3_PREFILL_NR;
                    int mc = panel - m0, nc = rows - n0;
                    if (mc > K3_PREFILL_MR) mc = K3_PREFILL_MR;
                    if (nc > K3_PREFILL_NR) nc = K3_PREFILL_NR;
                    const float *at = scratch +
                        (size_t)mb * ((cols + 3) & ~3) * K3_PREFILL_MR;
                    const uint16_t *bt = packed[p] + K3_PREFILL_PV_PREFIX +
                        (size_t)nb * ((cols + 3) & ~3) * K3_PREFILL_NR;
                    float *out = outs[p] + (size_t)token0 * rows;
                    if (mc == K3_PREFILL_MR && nc == K3_PREFILL_NR)
                        micro_kernel_bf16B_8x3_unroll4_pv(at, bt,
                            out + (size_t)m0 * rows + n0, (cols + 3) & ~3, 0,
                            (int64_t)rows * sizeof(float));
                    else {
                        float tmp[K3_PREFILL_MR * K3_PREFILL_NR]
                            __attribute__((aligned(256)));
                        micro_kernel_bf16B_8x3_unroll4_pv(at, bt, tmp,
                            (cols + 3) & ~3, 0,
                            K3_PREFILL_NR * sizeof(float));
                        for (int m = 0; m < mc; ++m)
                            memcpy(out + (size_t)(m0 + m) * rows + n0,
                                tmp + (size_t)m * K3_PREFILL_NR,
                                (size_t)nc * sizeof(float));
                    }
                }
#if defined(_OPENMP)
        }
#endif
    }
    return 0;
}

#endif
