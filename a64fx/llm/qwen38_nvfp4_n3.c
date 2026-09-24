/* Exact three-token Qwen3.8-27B NVFP4 projections on 512-bit SVE.
 * Fixed model shapes and strides remove branches from the decode loop. */
#include <arm_sve.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double q38_n3_profile_ms[7];
static unsigned long q38_n3_profile_count[7];
static const char *q38_n3_profile_name[7] = {
    "unknown", "ffn_gate", "ffn_down", "ssm_qkv",
    "ssm_gate", "ssm_out", "attn_q"
};
static double q38_n3_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
static void q38_n3_profile_report(void) {
    for (int i = 1; i <= 6; i++)
        fprintf(stderr, "qwen38: n3_shape %s calls=%lu total=%.1f avg=%.3f ms\n",
                q38_n3_profile_name[i], q38_n3_profile_count[i],
                q38_n3_profile_ms[i],
                q38_n3_profile_count[i] ?
                    q38_n3_profile_ms[i] / q38_n3_profile_count[i] : 0.0);
}
typedef struct { uint8_t d[8], qs[64]; } q38_n3_subblock;
typedef struct { q38_n3_subblock s[4]; } q38_n3_block;
typedef struct { float d[8]; uint8_t qs[64]; } q38_n3_packed_subblock;
typedef struct { q38_n3_packed_subblock s[4]; } q38_n3_packed_block;
typedef struct { int n_cols; void *data; } q38_n3_matrix;
_Static_assert(sizeof(q38_n3_block) == 288, "NVFP4 eight-row tile layout");
_Static_assert(sizeof(q38_n3_packed_block) == 384,
               "NVFP4 predecoded-scale tile layout");
/* Match the MXFP4 codebook and UE4M3 scale table in transformer.h. */
static const float q38_n3_codes[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static inline float q38_n3_scale(uint8_t x) {
    static const float scale[256] = {
        0x0.0p+0f, 0x1.0000000000000p-10f, 0x1.0000000000000p-9f, 0x1.8000000000000p-9f, 0x1.0000000000000p-8f, 0x1.4000000000000p-8f, 0x1.8000000000000p-8f, 0x1.c000000000000p-8f,
        0x1.0000000000000p-7f, 0x1.2000000000000p-7f, 0x1.4000000000000p-7f, 0x1.6000000000000p-7f, 0x1.8000000000000p-7f, 0x1.a000000000000p-7f, 0x1.c000000000000p-7f, 0x1.e000000000000p-7f,
        0x1.0000000000000p-6f, 0x1.2000000000000p-6f, 0x1.4000000000000p-6f, 0x1.6000000000000p-6f, 0x1.8000000000000p-6f, 0x1.a000000000000p-6f, 0x1.c000000000000p-6f, 0x1.e000000000000p-6f,
        0x1.0000000000000p-5f, 0x1.2000000000000p-5f, 0x1.4000000000000p-5f, 0x1.6000000000000p-5f, 0x1.8000000000000p-5f, 0x1.a000000000000p-5f, 0x1.c000000000000p-5f, 0x1.e000000000000p-5f,
        0x1.0000000000000p-4f, 0x1.2000000000000p-4f, 0x1.4000000000000p-4f, 0x1.6000000000000p-4f, 0x1.8000000000000p-4f, 0x1.a000000000000p-4f, 0x1.c000000000000p-4f, 0x1.e000000000000p-4f,
        0x1.0000000000000p-3f, 0x1.2000000000000p-3f, 0x1.4000000000000p-3f, 0x1.6000000000000p-3f, 0x1.8000000000000p-3f, 0x1.a000000000000p-3f, 0x1.c000000000000p-3f, 0x1.e000000000000p-3f,
        0x1.0000000000000p-2f, 0x1.2000000000000p-2f, 0x1.4000000000000p-2f, 0x1.6000000000000p-2f, 0x1.8000000000000p-2f, 0x1.a000000000000p-2f, 0x1.c000000000000p-2f, 0x1.e000000000000p-2f,
        0x1.0000000000000p-1f, 0x1.2000000000000p-1f, 0x1.4000000000000p-1f, 0x1.6000000000000p-1f, 0x1.8000000000000p-1f, 0x1.a000000000000p-1f, 0x1.c000000000000p-1f, 0x1.e000000000000p-1f,
        0x1.0000000000000p+0f, 0x1.2000000000000p+0f, 0x1.4000000000000p+0f, 0x1.6000000000000p+0f, 0x1.8000000000000p+0f, 0x1.a000000000000p+0f, 0x1.c000000000000p+0f, 0x1.e000000000000p+0f,
        0x1.0000000000000p+1f, 0x1.2000000000000p+1f, 0x1.4000000000000p+1f, 0x1.6000000000000p+1f, 0x1.8000000000000p+1f, 0x1.a000000000000p+1f, 0x1.c000000000000p+1f, 0x1.e000000000000p+1f,
        0x1.0000000000000p+2f, 0x1.2000000000000p+2f, 0x1.4000000000000p+2f, 0x1.6000000000000p+2f, 0x1.8000000000000p+2f, 0x1.a000000000000p+2f, 0x1.c000000000000p+2f, 0x1.e000000000000p+2f,
        0x1.0000000000000p+3f, 0x1.2000000000000p+3f, 0x1.4000000000000p+3f, 0x1.6000000000000p+3f, 0x1.8000000000000p+3f, 0x1.a000000000000p+3f, 0x1.c000000000000p+3f, 0x1.e000000000000p+3f,
        0x1.0000000000000p+4f, 0x1.2000000000000p+4f, 0x1.4000000000000p+4f, 0x1.6000000000000p+4f, 0x1.8000000000000p+4f, 0x1.a000000000000p+4f, 0x1.c000000000000p+4f, 0x1.e000000000000p+4f,
        0x1.0000000000000p+5f, 0x1.2000000000000p+5f, 0x1.4000000000000p+5f, 0x1.6000000000000p+5f, 0x1.8000000000000p+5f, 0x1.a000000000000p+5f, 0x1.c000000000000p+5f, 0x1.e000000000000p+5f,
        0x1.0000000000000p+6f, 0x1.2000000000000p+6f, 0x1.4000000000000p+6f, 0x1.6000000000000p+6f, 0x1.8000000000000p+6f, 0x1.a000000000000p+6f, 0x1.c000000000000p+6f, 0x1.e000000000000p+6f,
        0x1.0000000000000p+7f, 0x1.2000000000000p+7f, 0x1.4000000000000p+7f, 0x1.6000000000000p+7f, 0x1.8000000000000p+7f, 0x1.a000000000000p+7f, 0x1.c000000000000p+7f, 0x0.0p+0f,
        0x0.0p+0f, 0x1.0000000000000p-10f, 0x1.0000000000000p-9f, 0x1.8000000000000p-9f, 0x1.0000000000000p-8f, 0x1.4000000000000p-8f, 0x1.8000000000000p-8f, 0x1.c000000000000p-8f,
        0x1.0000000000000p-7f, 0x1.2000000000000p-7f, 0x1.4000000000000p-7f, 0x1.6000000000000p-7f, 0x1.8000000000000p-7f, 0x1.a000000000000p-7f, 0x1.c000000000000p-7f, 0x1.e000000000000p-7f,
        0x1.0000000000000p-6f, 0x1.2000000000000p-6f, 0x1.4000000000000p-6f, 0x1.6000000000000p-6f, 0x1.8000000000000p-6f, 0x1.a000000000000p-6f, 0x1.c000000000000p-6f, 0x1.e000000000000p-6f,
        0x1.0000000000000p-5f, 0x1.2000000000000p-5f, 0x1.4000000000000p-5f, 0x1.6000000000000p-5f, 0x1.8000000000000p-5f, 0x1.a000000000000p-5f, 0x1.c000000000000p-5f, 0x1.e000000000000p-5f,
        0x1.0000000000000p-4f, 0x1.2000000000000p-4f, 0x1.4000000000000p-4f, 0x1.6000000000000p-4f, 0x1.8000000000000p-4f, 0x1.a000000000000p-4f, 0x1.c000000000000p-4f, 0x1.e000000000000p-4f,
        0x1.0000000000000p-3f, 0x1.2000000000000p-3f, 0x1.4000000000000p-3f, 0x1.6000000000000p-3f, 0x1.8000000000000p-3f, 0x1.a000000000000p-3f, 0x1.c000000000000p-3f, 0x1.e000000000000p-3f,
        0x1.0000000000000p-2f, 0x1.2000000000000p-2f, 0x1.4000000000000p-2f, 0x1.6000000000000p-2f, 0x1.8000000000000p-2f, 0x1.a000000000000p-2f, 0x1.c000000000000p-2f, 0x1.e000000000000p-2f,
        0x1.0000000000000p-1f, 0x1.2000000000000p-1f, 0x1.4000000000000p-1f, 0x1.6000000000000p-1f, 0x1.8000000000000p-1f, 0x1.a000000000000p-1f, 0x1.c000000000000p-1f, 0x1.e000000000000p-1f,
        0x1.0000000000000p+0f, 0x1.2000000000000p+0f, 0x1.4000000000000p+0f, 0x1.6000000000000p+0f, 0x1.8000000000000p+0f, 0x1.a000000000000p+0f, 0x1.c000000000000p+0f, 0x1.e000000000000p+0f,
        0x1.0000000000000p+1f, 0x1.2000000000000p+1f, 0x1.4000000000000p+1f, 0x1.6000000000000p+1f, 0x1.8000000000000p+1f, 0x1.a000000000000p+1f, 0x1.c000000000000p+1f, 0x1.e000000000000p+1f,
        0x1.0000000000000p+2f, 0x1.2000000000000p+2f, 0x1.4000000000000p+2f, 0x1.6000000000000p+2f, 0x1.8000000000000p+2f, 0x1.a000000000000p+2f, 0x1.c000000000000p+2f, 0x1.e000000000000p+2f,
        0x1.0000000000000p+3f, 0x1.2000000000000p+3f, 0x1.4000000000000p+3f, 0x1.6000000000000p+3f, 0x1.8000000000000p+3f, 0x1.a000000000000p+3f, 0x1.c000000000000p+3f, 0x1.e000000000000p+3f,
        0x1.0000000000000p+4f, 0x1.2000000000000p+4f, 0x1.4000000000000p+4f, 0x1.6000000000000p+4f, 0x1.8000000000000p+4f, 0x1.a000000000000p+4f, 0x1.c000000000000p+4f, 0x1.e000000000000p+4f,
        0x1.0000000000000p+5f, 0x1.2000000000000p+5f, 0x1.4000000000000p+5f, 0x1.6000000000000p+5f, 0x1.8000000000000p+5f, 0x1.a000000000000p+5f, 0x1.c000000000000p+5f, 0x1.e000000000000p+5f,
        0x1.0000000000000p+6f, 0x1.2000000000000p+6f, 0x1.4000000000000p+6f, 0x1.6000000000000p+6f, 0x1.8000000000000p+6f, 0x1.a000000000000p+6f, 0x1.c000000000000p+6f, 0x1.e000000000000p+6f,
        0x1.0000000000000p+7f, 0x1.2000000000000p+7f, 0x1.4000000000000p+7f, 0x1.6000000000000p+7f, 0x1.8000000000000p+7f, 0x1.a000000000000p+7f, 0x1.c000000000000p+7f, 0x1.e000000000000p+7f
    };
    return scale[x];
}
static inline __attribute__((always_inline)) void q38_nvfp4_exact_n3_rows(float *y, const q38_n3_matrix *mat,
        const float *x, int n, int ys, int xs, int first, int last) {
    n = 3;
    const int nb = mat->n_cols / 64;
    const q38_n3_block *base = (const q38_n3_block *)mat->data;
    const svbool_t pg = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const svuint32_t repeat8 = svand_n_u32_x(pg, svindex_u32(0, 1), 7);
    const svfloat32_t lut = svld1(pg, q38_n3_codes);
    for (int row = first; row < last; row += 8) {
        const int tile = row / 8, r0 = row % 8;
        svfloat32_t a00=svdup_f32(0), a01=a00, a02=a00, a03=a00;
        svfloat32_t a10=a00, a11=a00, a12=a00, a13=a00;
        svfloat32_t a20=a00, a21=a00, a22=a00, a23=a00;
        svfloat32_t a30=a00, a31=a00, a32=a00, a33=a00;
        const q38_n3_block *w = base + (size_t)tile * nb;
        for (int ib = 0; ib < nb; ib++)
            for (int s = 0; s < 4; s++) {
                const q38_n3_subblock *p = &w[ib].s[s];
#define Q38_NVFP4_BATCH_PAIR(R,A0,A1,A2,A3) do { \
                    int rr=r0+2*(R); \
                    svuint32_t z=svld1ub_u32(pg,p->qs+rr*8); \
                    svfloat32_t d=svsel_f32(p8, \
                        svdup_f32(q38_n3_scale(p->d[rr])), \
                        svdup_f32(q38_n3_scale(p->d[rr+1]))); \
                    svfloat32_t lo=svmul_x(pg, \
                        svtbl_f32(lut,svand_n_u32_x(pg,z,15)),d); \
                    svfloat32_t hi=svmul_x(pg, \
                        svtbl_f32(lut,svlsr_n_u32_x(pg,z,4)),d); \
                    const float *x0=x+ib*64+s*16; \
                    svfloat32_t xl=svtbl_f32(svld1(p8,x0),repeat8); \
                    svfloat32_t xh=svtbl_f32(svld1(p8,x0+8),repeat8); \
                    (A0)=svmla_x(pg,(A0),lo,xl); \
                    (A0)=svmla_x(pg,(A0),hi,xh); \
                    if(n>1){const float *xt=x0+xs; \
                        xl=svtbl_f32(svld1(p8,xt),repeat8); \
                        xh=svtbl_f32(svld1(p8,xt+8),repeat8); \
                        (A1)=svmla_x(pg,(A1),lo,xl); \
                        (A1)=svmla_x(pg,(A1),hi,xh);} \
                    if(n>2){const float *xt=x0+(size_t)2*xs; \
                        xl=svtbl_f32(svld1(p8,xt),repeat8); \
                        xh=svtbl_f32(svld1(p8,xt+8),repeat8); \
                        (A2)=svmla_x(pg,(A2),lo,xl); \
                        (A2)=svmla_x(pg,(A2),hi,xh);} \
                    if(n>3){const float *xt=x0+(size_t)3*xs; \
                        xl=svtbl_f32(svld1(p8,xt),repeat8); \
                        xh=svtbl_f32(svld1(p8,xt+8),repeat8); \
                        (A3)=svmla_x(pg,(A3),lo,xl); \
                        (A3)=svmla_x(pg,(A3),hi,xh);} \
                }while(0)
                Q38_NVFP4_BATCH_PAIR(0,a00,a01,a02,a03);
                Q38_NVFP4_BATCH_PAIR(1,a10,a11,a12,a13);
                Q38_NVFP4_BATCH_PAIR(2,a20,a21,a22,a23);
                Q38_NVFP4_BATCH_PAIR(3,a30,a31,a32,a33);
#undef Q38_NVFP4_BATCH_PAIR
            }
#define Q38_NVFP4_BATCH_PAIR_STORE(R,A0,A1,A2,A3) do { \
            int rr=row+2*(R); \
            if(rr<last){ \
                y[rr]=svaddv_f32(p8,A0); \
                y[rr+1]=svaddv_f32(p8,svext_f32(A0,A0,8)); \
                if(n>1){y[(size_t)ys+rr]=svaddv_f32(p8,A1); \
                    y[(size_t)ys+rr+1]=svaddv_f32(p8,svext_f32(A1,A1,8));} \
                if(n>2){y[(size_t)2*ys+rr]=svaddv_f32(p8,A2); \
                    y[(size_t)2*ys+rr+1]=svaddv_f32(p8,svext_f32(A2,A2,8));} \
                if(n>3){y[(size_t)3*ys+rr]=svaddv_f32(p8,A3); \
                    y[(size_t)3*ys+rr+1]=svaddv_f32(p8,svext_f32(A3,A3,8));} \
            } \
        }while(0)
        Q38_NVFP4_BATCH_PAIR_STORE(0,a00,a01,a02,a03);
        Q38_NVFP4_BATCH_PAIR_STORE(1,a10,a11,a12,a13);
        Q38_NVFP4_BATCH_PAIR_STORE(2,a20,a21,a22,a23);
        Q38_NVFP4_BATCH_PAIR_STORE(3,a30,a31,a32,a33);
#undef Q38_NVFP4_BATCH_PAIR_STORE
    }
}

/* Predecoded FP32 scales are produced while repacking the staged GGUF into
 * four CMG-local HBM arenas. Reuse each FP4 nibble decode for all three
 * verifier candidates, preserving the exact low/high FMA sequence. */
static void q38_nvfp4_packed_n3_rows(float *y, const void *weights,
        const float *x, int rows, int cols, int first, int last) {
    const int nb = cols / 64;
    const q38_n3_packed_block *base = (const q38_n3_packed_block *)weights;
    const svbool_t pg = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const svuint32_t repeat8 = svand_n_u32_x(pg, svindex_u32(0, 1), 7);
    const svfloat32_t lut = svld1(pg, q38_n3_codes);
    for (int row = first; row < last; row += 8) {
        svfloat32_t a00=svdup_f32(0), a01=a00, a02=a00;
        svfloat32_t a10=a00, a11=a00, a12=a00;
        svfloat32_t a20=a00, a21=a00, a22=a00;
        svfloat32_t a30=a00, a31=a00, a32=a00;
        const q38_n3_packed_block *w = base + (size_t)(row / 8) * nb;
        for (int ib = 0; ib < nb; ib++) {
            for (int s = 0; s < 4; s++) {
                const q38_n3_packed_subblock *p = &w[ib].s[s];
                const float *x0 = x + ib * 64 + s * 16;
#define Q38_PACKED_N3_PAIR(R,A0,A1,A2) do { \
                    int rr=2*(R); \
                    svuint32_t z=svld1ub_u32(pg,p->qs+rr*8); \
                    svfloat32_t d=svsel_f32(p8,svdup_f32(p->d[rr]), \
                                                  svdup_f32(p->d[rr+1])); \
                    svfloat32_t lo=svmul_x(pg, \
                        svtbl_f32(lut,svand_n_u32_x(pg,z,15)),d); \
                    svfloat32_t hi=svmul_x(pg, \
                        svtbl_f32(lut,svlsr_n_u32_x(pg,z,4)),d); \
                    for(int t=0;t<3;t++){ \
                        const float *xt=x0+(size_t)t*cols; \
                        svfloat32_t xl=svtbl_f32(svld1(p8,xt),repeat8); \
                        svfloat32_t xh=svtbl_f32(svld1(p8,xt+8),repeat8); \
                        if(t==0){(A0)=svmla_x(pg,(A0),lo,xl); \
                                 (A0)=svmla_x(pg,(A0),hi,xh);} \
                        else if(t==1){(A1)=svmla_x(pg,(A1),lo,xl); \
                                      (A1)=svmla_x(pg,(A1),hi,xh);} \
                        else {(A2)=svmla_x(pg,(A2),lo,xl); \
                              (A2)=svmla_x(pg,(A2),hi,xh);} \
                    } \
                }while(0)
                Q38_PACKED_N3_PAIR(0,a00,a01,a02);
                Q38_PACKED_N3_PAIR(1,a10,a11,a12);
                Q38_PACKED_N3_PAIR(2,a20,a21,a22);
                Q38_PACKED_N3_PAIR(3,a30,a31,a32);
#undef Q38_PACKED_N3_PAIR
            }
        }
#define Q38_PACKED_N3_STORE(R,A0,A1,A2) do { \
            int rr=row+2*(R); \
            if(rr<last){ \
                y[rr]=svaddv_f32(p8,A0); \
                y[rr+1]=svaddv_f32(p8,svext_f32(A0,A0,8)); \
                y[(size_t)rows+rr]=svaddv_f32(p8,A1); \
                y[(size_t)rows+rr+1]=svaddv_f32(p8,svext_f32(A1,A1,8)); \
                y[(size_t)2*rows+rr]=svaddv_f32(p8,A2); \
                y[(size_t)2*rows+rr+1]=svaddv_f32(p8,svext_f32(A2,A2,8)); \
            } \
        }while(0)
        Q38_PACKED_N3_STORE(0,a00,a01,a02);
        Q38_PACKED_N3_STORE(1,a10,a11,a12);
        Q38_PACKED_N3_STORE(2,a20,a21,a22);
        Q38_PACKED_N3_STORE(3,a30,a31,a32);
#undef Q38_PACKED_N3_STORE
    }
}

int q38_nvfp4_packed_n3_mt(float *y, const void *weights, const float *x,
                           int rows, int cols, int n_threads) {
    if (svcntw() != 16 || rows % 8 || cols % 64) return 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), team = omp_get_num_threads();
#else
        int tid = 0, team = 1;
        (void)n_threads;
#endif
        int tiles = rows / 8;
        int first = tiles * tid / team * 8;
        int last = tiles * (tid + 1) / team * 8;
        q38_nvfp4_packed_n3_rows(y, weights, x, rows, cols, first, last);
    }
    return 1;
}

#define Q38_N3_SHAPE(NAME, ROWS, COLS) \
static __attribute__((noinline)) void NAME(float *y, const void *weights, \
                                           const float *x, int first, int last) { \
    q38_n3_matrix mat = {COLS, (void *)weights}; \
    q38_nvfp4_exact_n3_rows(y, &mat, x, 3, ROWS, COLS, first, last); \
}
Q38_N3_SHAPE(q38_n3_ffn_gate, 17408, 5120)
Q38_N3_SHAPE(q38_n3_ffn_down, 5120, 17408)
Q38_N3_SHAPE(q38_n3_ssm_qkv, 10240, 5120)
Q38_N3_SHAPE(q38_n3_ssm_gate, 6144, 5120)
Q38_N3_SHAPE(q38_n3_ssm_out, 5120, 6144)
Q38_N3_SHAPE(q38_n3_attn_q, 12288, 5120)
#undef Q38_N3_SHAPE

int q38_nvfp4_exact_n3_mt(float *y, const void *weights, const float *x,
                            int rows, int cols, int n_threads) {
    static int profile = -1;
    if (profile < 0) {
        const char *env = getenv("Q38_N3_PROFILE");
        profile = env && atoi(env) != 0;
        if (profile) atexit(q38_n3_profile_report);
    }
    int kind = rows == 17408 && cols == 5120 ? 1 :
               rows == 5120 && cols == 17408 ? 2 :
               rows == 10240 && cols == 5120 ? 3 :
               rows == 6144 && cols == 5120 ? 4 :
               rows == 5120 && cols == 6144 ? 5 :
               rows == 12288 && cols == 5120 ? 6 : 0;
    if (!kind) return 0;
    double t0 = profile ? q38_n3_now_ms() : 0.0;
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), team = omp_get_num_threads();
#else
        int tid = 0, team = 1;
        (void)n_threads;
#endif
        int tiles = rows / 8;
        int first = tiles * tid / team * 8;
        int last = tiles * (tid + 1) / team * 8;
        switch (kind) {
            case 1: q38_n3_ffn_gate(y, weights, x, first, last); break;
            case 2: q38_n3_ffn_down(y, weights, x, first, last); break;
            case 3: q38_n3_ssm_qkv(y, weights, x, first, last); break;
            case 4: q38_n3_ssm_gate(y, weights, x, first, last); break;
            case 5: q38_n3_ssm_out(y, weights, x, first, last); break;
            case 6: q38_n3_attn_q(y, weights, x, first, last); break;
        }
    }
    if (profile) {
        q38_n3_profile_ms[kind] += q38_n3_now_ms() - t0;
        q38_n3_profile_count[kind]++;
    }
    return 1;
}
