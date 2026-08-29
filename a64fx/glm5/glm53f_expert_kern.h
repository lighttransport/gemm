#ifndef GLM53F_EXPERT_KERN_H
#define GLM53F_EXPERT_KERN_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *gate_up;
    const float *gate_up_scale;
    const uint8_t *down;
    const float *down_scale;
    int inter;
} glm53f_expert_part;

static inline float glm53f_fp8_e4m3_scalar(uint8_t q) {
    int sign = q >> 7, exponent = (q >> 3) & 15, mantissa = q & 7;
    float v;
    if (!exponent) v = (float)mantissa / 512.0f;
    else if (exponent == 15 && mantissa == 7) return NAN;
    else v = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -v : v;
}

static inline float glm53f_dot_fp8_block128(
        const uint8_t *w, const float *scale, const float *x, int cols) {
    double acc = 0;
    for (int c = 0; c < cols; ++c)
        acc += (double)glm53f_fp8_e4m3_scalar(w[c]) * scale[c / 128] * x[c];
    return (float)acc;
}

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t glm53f_fp8_e4m3_bits(svbool_t pg, const uint8_t *w, int c) {
    svuint32_t q = svld1ub_u32(pg, w + c);
    svuint32_t sign = svlsl_n_u32_x(pg, svand_n_u32_x(pg, q, 0x80), 24);
    svuint32_t exponent = svand_n_u32_x(pg, svlsr_n_u32_x(pg, q, 3), 15);
    svuint32_t mantissa = svand_n_u32_x(pg, q, 7);
    svuint32_t normal = svorr_u32_x(
        pg, sign, svlsl_n_u32_x(pg, svadd_n_u32_x(pg, exponent, 120), 23));
    normal = svorr_u32_x(pg, normal, svlsl_n_u32_x(pg, mantissa, 20));
    svfloat32_t subnormal = svmul_n_f32_x(
        pg, svcvt_f32_u32_x(pg, mantissa), 1.0f / 512.0f);
    svuint32_t bits = svsel_u32(
        svcmpne_n_u32(pg, exponent, 0), normal,
        svorr_u32_x(pg, svreinterpret_u32_f32(subnormal), sign));
    /* The staged checkpoint contract excludes E4M3 NaNs (0x7f/0xff).
     * Avoid two compares and a select in the decode-token inner loop. */
    return svreinterpret_f32_u32(bits);
}

static inline void glm53f_matvec_fp8_bits_8(
        float *dst, const uint8_t *w, const float *scale,
        const float *x, int cols) {
    svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0);
    svfloat32_t a2 = svdup_f32(0), a3 = svdup_f32(0);
    svfloat32_t a4 = svdup_f32(0), a5 = svdup_f32(0);
    svfloat32_t a6 = svdup_f32(0), a7 = svdup_f32(0);
    int vl = (int)svcntw();
    for (int b = 0; b < cols; b += 128) {
        int end = b + 128 < cols ? b + 128 : cols;
        int block = b / 128;
        for (int c = b; c < end; c += vl) {
            if (c + 64 < end) {
                __builtin_prefetch(x + c + 64, 0, 1);
                __builtin_prefetch(w + c + 64, 0, 0);
                __builtin_prefetch(w + (size_t)cols + c + 64, 0, 0);
            }
            svbool_t pg = svwhilelt_b32(c, end);
            svfloat32_t xv = svld1(pg, x + c);
            svfloat32_t xs = svmul_n_f32_x(pg, xv, scale[block]);
#define GLM53F_FP8_ROW(J, A) \
            A = svmla_x(pg, A, glm53f_fp8_e4m3_bits( \
                pg, w + (size_t)(J) * cols, c), xs)
            GLM53F_FP8_ROW(0, a0); GLM53F_FP8_ROW(1, a1);
            GLM53F_FP8_ROW(2, a2); GLM53F_FP8_ROW(3, a3);
            GLM53F_FP8_ROW(4, a4); GLM53F_FP8_ROW(5, a5);
            GLM53F_FP8_ROW(6, a6); GLM53F_FP8_ROW(7, a7);
#undef GLM53F_FP8_ROW
        }
    }
    svbool_t pt = svptrue_b32();
    dst[0] = svaddv_f32(pt, a0); dst[1] = svaddv_f32(pt, a1);
    dst[2] = svaddv_f32(pt, a2); dst[3] = svaddv_f32(pt, a3);
    dst[4] = svaddv_f32(pt, a4); dst[5] = svaddv_f32(pt, a5);
    dst[6] = svaddv_f32(pt, a6); dst[7] = svaddv_f32(pt, a7);
}

static inline void glm53f_mv_fp8_block128_bits(
        float *y, const uint8_t *w, const float *scale,
        const float *x, int rows, int cols) {
    int blocks = (cols + 127) / 128, n8 = rows / 8;
#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n8; ++bi) {
        int r = bi * 8;
        glm53f_matvec_fp8_bits_8(
            y + r, w + (size_t)r * cols,
            scale + (size_t)(r / 128) * blocks, x, cols);
    }
    for (int r = n8 * 8; r < rows; ++r)
        y[r] = glm53f_dot_fp8_block128(
            w + (size_t)r * cols, scale + (size_t)(r / 128) * blocks, x, cols);
}

/* Matrix x up-to-four-token microkernel.  Four output rows leave enough SVE
 * registers for four token accumulators per row while reusing each FP8 weight
 * vector across all tokens.  Inputs and outputs are token-major. */
static inline void glm53f_matvec_fp8_bits_4x4(
        float *dst, int dst_stride, const uint8_t *w, const float *scale,
        const float *x, int tokens, int cols) {
    svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
    svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
    int vl = (int)svcntw();
    for (int b = 0; b < cols; b += 128) {
        int end = b + 128 < cols ? b + 128 : cols;
        int block = b / 128;
        for (int c = b; c < end; c += vl) {
            svbool_t pg = svwhilelt_b32(c, end);
            svfloat32_t w0=glm53f_fp8_e4m3_bits(pg,w,c);
            svfloat32_t w1=glm53f_fp8_e4m3_bits(pg,w+(size_t)cols,c);
            svfloat32_t w2=glm53f_fp8_e4m3_bits(pg,w+(size_t)2*cols,c);
            svfloat32_t w3=glm53f_fp8_e4m3_bits(pg,w+(size_t)3*cols,c);
#define GLM53F_TOKEN4(T,A0,A1,A2,A3) do { \
    svfloat32_t xv=svmul_n_f32_x(pg,svld1(pg,x+(size_t)(T)*cols+c),scale[block]); \
    A0=svmla_x(pg,A0,w0,xv); A1=svmla_x(pg,A1,w1,xv); \
    A2=svmla_x(pg,A2,w2,xv); A3=svmla_x(pg,A3,w3,xv); \
} while(0)
            GLM53F_TOKEN4(0,a00,a01,a02,a03);
            if(tokens>1)GLM53F_TOKEN4(1,a10,a11,a12,a13);
            if(tokens>2)GLM53F_TOKEN4(2,a20,a21,a22,a23);
            if(tokens>3)GLM53F_TOKEN4(3,a30,a31,a32,a33);
#undef GLM53F_TOKEN4
        }
    }
    svbool_t pt = svptrue_b32();
#define GLM53F_STORE4(T,A0,A1,A2,A3) do { \
    dst[(size_t)(T)*dst_stride]=svaddv_f32(pt,A0); \
    dst[(size_t)(T)*dst_stride+1]=svaddv_f32(pt,A1); \
    dst[(size_t)(T)*dst_stride+2]=svaddv_f32(pt,A2); \
    dst[(size_t)(T)*dst_stride+3]=svaddv_f32(pt,A3); \
} while(0)
    GLM53F_STORE4(0,a00,a01,a02,a03);
    if(tokens>1)GLM53F_STORE4(1,a10,a11,a12,a13);
    if(tokens>2)GLM53F_STORE4(2,a20,a21,a22,a23);
    if(tokens>3)GLM53F_STORE4(3,a30,a31,a32,a33);
#undef GLM53F_STORE4
}

static inline void glm53f_mv_fp8_block128_bits_batch(
        float *y, const uint8_t *w, const float *scale, const float *x,
        int tokens, int rows, int cols) {
    if (tokens == 1) {
        glm53f_mv_fp8_block128_bits(y, w, scale, x, rows, cols);
        return;
    }
    if (tokens < 1 || tokens > 4) return;
    int blocks = (cols + 127) / 128, n4 = rows / 4;
#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n4; ++bi) {
        int r = bi * 4;
        glm53f_matvec_fp8_bits_4x4(y + r, rows,
            w + (size_t)r * cols, scale + (size_t)(r / 128) * blocks,
            x, tokens, cols);
    }
    for (int t = 0; t < tokens; ++t)
        for (int r = n4 * 4; r < rows; ++r)
            y[(size_t)t * rows + r] = glm53f_dot_fp8_block128(
                w + (size_t)r * cols, scale + (size_t)r * blocks,
                x + (size_t)t * cols, cols);
}

static inline float glm53f_dot_bf16_sve(
    const uint16_t *w, const float *x, int n);

/* BF16 matrix x up-to-four-token kernel.  This has the same token-major
 * contract as the FP8 batch kernel and reuses four BF16 weight rows across
 * all live verification positions. */
static inline void glm53f_matvec_bf16_4x4(
        float *dst, int dst_stride, const uint16_t *w,
        const float *x, int tokens, int cols) {
    svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
    svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
    int vl=(int)svcntw();
    for(int c=0;c<cols;c+=vl){svbool_t pg=svwhilelt_b32(c,cols);
        svfloat32_t w0=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+c),16));
        svfloat32_t w1=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+(size_t)cols+c),16));
        svfloat32_t w2=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+(size_t)2*cols+c),16));
        svfloat32_t w3=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+(size_t)3*cols+c),16));
#define GLM53F_BF16_TOKEN4(T,A0,A1,A2,A3) do { \
    svfloat32_t xv=svld1(pg,x+(size_t)(T)*cols+c); \
    A0=svmla_x(pg,A0,w0,xv); A1=svmla_x(pg,A1,w1,xv); \
    A2=svmla_x(pg,A2,w2,xv); A3=svmla_x(pg,A3,w3,xv); \
} while(0)
        GLM53F_BF16_TOKEN4(0,a00,a01,a02,a03);
        if(tokens>1)GLM53F_BF16_TOKEN4(1,a10,a11,a12,a13);
        if(tokens>2)GLM53F_BF16_TOKEN4(2,a20,a21,a22,a23);
        if(tokens>3)GLM53F_BF16_TOKEN4(3,a30,a31,a32,a33);
#undef GLM53F_BF16_TOKEN4
    }
    svbool_t pt=svptrue_b32();
#define GLM53F_BF16_STORE4(T,A0,A1,A2,A3) do { \
    dst[(size_t)(T)*dst_stride]=svaddv_f32(pt,A0); \
    dst[(size_t)(T)*dst_stride+1]=svaddv_f32(pt,A1); \
    dst[(size_t)(T)*dst_stride+2]=svaddv_f32(pt,A2); \
    dst[(size_t)(T)*dst_stride+3]=svaddv_f32(pt,A3); \
} while(0)
    GLM53F_BF16_STORE4(0,a00,a01,a02,a03);
    if(tokens>1)GLM53F_BF16_STORE4(1,a10,a11,a12,a13);
    if(tokens>2)GLM53F_BF16_STORE4(2,a20,a21,a22,a23);
    if(tokens>3)GLM53F_BF16_STORE4(3,a30,a31,a32,a33);
#undef GLM53F_BF16_STORE4
}

static inline void glm53f_mv_bf16_batch(
        float *y,const uint16_t*w,const float*x,int tokens,int rows,int cols){
    if(tokens<1||tokens>4)return;int n4=rows/4;
#pragma omp parallel for schedule(static)
    for(int bi=0;bi<n4;bi++){int r=bi*4;glm53f_matvec_bf16_4x4(
        y+r,rows,w+(size_t)r*cols,x,tokens,cols);}
    for(int t=0;t<tokens;t++)for(int r=n4*4;r<rows;r++)
        y[(size_t)t*rows+r]=glm53f_dot_bf16_sve(
            w+(size_t)r*cols,x+(size_t)t*cols,cols);
}

static inline void glm53f_mv_fp8_block128_bits_2(
        float *y0, const uint8_t *w0, const float *s0, int rows0,
        float *y1, const uint8_t *w1, const float *s1, int rows1,
        const float *x, int cols) {
    int blocks = (cols + 127) / 128, n0 = rows0 / 8, n1 = rows1 / 8;
#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n0 + n1; ++bi) {
        int second = bi >= n0, r = (second ? bi - n0 : bi) * 8;
        const uint8_t *w = second ? w1 : w0;
        const float *s = second ? s1 : s0;
        float *y = second ? y1 : y0;
        glm53f_matvec_fp8_bits_8(
            y + r, w + (size_t)r * cols,
            s + (size_t)(r / 128) * blocks, x, cols);
    }
    for (int r = n0 * 8; r < rows0; ++r)
        y0[r] = glm53f_dot_fp8_block128(
            w0 + (size_t)r * cols, s0 + (size_t)(r / 128) * blocks, x, cols);
    for (int r = n1 * 8; r < rows1; ++r)
        y1[r] = glm53f_dot_fp8_block128(
            w1 + (size_t)r * cols, s1 + (size_t)(r / 128) * blocks, x, cols);
}

static inline svfloat32_t glm53f_load_bf16_f32(svbool_t pg,
                                                const uint16_t *p) {
    svuint32_t u = svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16);
    return svreinterpret_f32_u32(u);
}

static inline float glm53f_dot_bf16_sve(const uint16_t *w,
                                         const float *x, int n) {
    svfloat32_t acc = svdup_f32(0.0f);
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        acc = svmla_x(pg, acc, glm53f_load_bf16_f32(pg, w + i),
                      svld1(pg, x + i));
    }
    return svaddv_f32(svptrue_b32(), acc);
}

/* Decode-optimized compressed-latent MLA. Parallelism is over attention
 * heads, keeping all per-head temporary state private and eliminating K/V
 * expansion for every selected cache entry. */
static inline int glm53f_mla_absorbed_sve(float *out, const float *query,
        const float *latent_cache, const uint16_t *kv_b, const int *selected,
        int n_selected, int heads, int key_dim, int value_dim, int latent_dim) {
    float *qlat = NULL, *vacc = NULL, *logit = NULL;
    if (n_selected <= 0) return -1;
    if (posix_memalign((void **)&qlat, 256, (size_t)heads * latent_dim * 4) ||
        posix_memalign((void **)&vacc, 256, (size_t)heads * latent_dim * 4) ||
        posix_memalign((void **)&logit, 256, (size_t)heads * n_selected * 4)) {
        free(logit); free(vacc); free(qlat); return -1;
    }
#pragma omp parallel for schedule(static)
    for (int h = 0; h < heads; ++h) {
        const uint16_t *wk = kv_b + (size_t)h * (key_dim + value_dim) * latent_dim;
        const uint16_t *wv = wk + (size_t)key_dim * latent_dim;
        float *qz = qlat + (size_t)h * latent_dim;
        float *vz = vacc + (size_t)h * latent_dim;
        float *ls = logit + (size_t)h * n_selected;
        int vl = (int)svcntw();
        for (int d = 0; d < latent_dim; d += vl) {
            svbool_t pg = svwhilelt_b32(d, latent_dim);
            svfloat32_t a = svdup_f32(0.0f);
            for (int j = 0; j < key_dim; ++j)
                a = svmla_n_f32_x(pg, a,
                    glm53f_load_bf16_f32(pg, wk + (size_t)j * latent_dim + d),
                    query[(size_t)h * key_dim + j]);
            svst1(pg, qz + d, svmul_n_f32_x(pg, a, 1.0f / sqrtf((float)key_dim)));
        }
        float mx = -INFINITY, sum = 0.0f;
        for (int p = 0; p < n_selected; ++p) {
            ls[p] = 0.0f;
            const float *z = latent_cache + (size_t)selected[p] * latent_dim;
            svfloat32_t a = svdup_f32(0.0f);
            for (int d = 0; d < latent_dim; d += vl) {
                svbool_t pg = svwhilelt_b32(d, latent_dim);
                a = svmla_x(pg, a, svld1(pg, qz + d), svld1(pg, z + d));
            }
            ls[p] = svaddv_f32(svptrue_b32(), a);
            if (ls[p] > mx) mx = ls[p];
        }
        for (int p = 0; p < n_selected; ++p) { ls[p] = expf(ls[p] - mx); sum += ls[p]; }
        memset(vz, 0, (size_t)latent_dim * 4);
        for (int p = 0; p < n_selected; ++p) {
            const float *z = latent_cache + (size_t)selected[p] * latent_dim;
            float a = ls[p] / sum;
            for (int d = 0; d < latent_dim; d += vl) {
                svbool_t pg = svwhilelt_b32(d, latent_dim);
                svst1(pg, vz + d, svmla_n_f32_x(pg, svld1(pg, vz + d), svld1(pg, z + d), a));
            }
        }
        for (int j = 0; j < value_dim; ++j)
            out[(size_t)h * value_dim + j] =
                glm53f_dot_bf16_sve(wv + (size_t)j * latent_dim, vz, latent_dim);
    }
    free(logit); free(vacc); free(qlat);
    return 0;
}

/* Run `batch` independent quarter experts concurrently. Threads are divided
 * into equal teams; batch=4 and 48 threads maps one 12-core team to each CMG. */
static inline void glm53f_expert_batch_bits(
        const glm53f_expert_part *part, int batch, const float *x,
        float *up, float *act, float *y) {
    enum { HIDDEN = 4096, INTER_STRIDE = 512, GATE_UP_STRIDE = 1024 };
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int total_units = 0, task = batch, lane = 0, lanes = 0, prefix = 0;
        for (int j = 0; j < batch; ++j) total_units += part[j].inter / 128;
        for (int j = 0; j < batch; ++j) {
            int units = part[j].inter / 128;
            int begin = nth * prefix / total_units;
            int end = nth * (prefix + units) / total_units;
            if (tid >= begin && tid < end) { task = j; lane = tid - begin; lanes = end - begin; }
            prefix += units;
        }
        if (task < batch) {
            int inter = part[task].inter, gate_up = 2 * inter;
            for (int bi = lane; bi < gate_up / 8; bi += lanes) {
                int r = bi * 8;
                glm53f_matvec_fp8_bits_8(
                    up + (size_t)task * GATE_UP_STRIDE + r,
                    part[task].gate_up + (size_t)r * HIDDEN,
                    part[task].gate_up_scale + (size_t)(r / 128) * (HIDDEN / 128),
                    x, HIDDEN);
            }
        }
#pragma omp barrier
        if (task < batch) {
            int inter = part[task].inter;
            for (int i = lane; i < inter; i += lanes) {
                float g = up[(size_t)task * GATE_UP_STRIDE + i];
                float u = up[(size_t)task * GATE_UP_STRIDE + inter + i];
                if (g > 10) g = 10;
                if (g < -100) g = -100;
                if (u > 10) u = 10;
                if (u < -10) u = -10;
                act[(size_t)task * INTER_STRIDE + i] = (g / (1 + expf(-g))) * u;
            }
        }
#pragma omp barrier
        if (task < batch) {
            int inter = part[task].inter;
            for (int bi = lane; bi < HIDDEN / 8; bi += lanes) {
                int r = bi * 8;
                glm53f_matvec_fp8_bits_8(
                    y + (size_t)task * HIDDEN + r,
                    part[task].down + (size_t)r * inter,
                    part[task].down_scale + (size_t)(r / 128) * (inter / 128),
                    act + (size_t)task * INTER_STRIDE, inter);
            }
        }
    }
}

/* Multi-position routed-expert execution.  Each token/part task shares one
 * OpenMP team; output aggregation remains in the caller's original route
 * order, so verification arithmetic is unchanged. */
static inline void glm53f_expert_tasks_bits(
        const glm53f_expert_part *parts, const int *counts, int tokens,
        const float *x, float *up, float *act, float *y) {
    enum { HIDDEN = 4096, INTER_STRIDE = 512, GATE_UP_STRIDE = 1024,
           MAX_PARTS = 9 };
    int total_tasks = 0;
    for (int t = 0; t < tokens; ++t)
        total_tasks += counts[t];
    if (total_tasks <= 0) return;
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int task_t = -1, task_k = -1, lane = 0, lanes = 1, prefix = 0;
        for (int t = 0; t < tokens; ++t)
            for (int k = 0; k < counts[t]; ++k) {
                /* Assign whole tasks first: unit-weight partitioning can
                 * produce zero-thread tasks when 128-row units < threads. */
                int begin = nth * prefix / total_tasks;
                int end = nth * (prefix + 1) / total_tasks;
                if (tid >= begin && tid < end) {
                    task_t = t; task_k = k; lane = tid - begin;
                    lanes = end - begin;
                }
                prefix++;
            }
        if (task_t >= 0) {
            const glm53f_expert_part *p = &parts[task_t * MAX_PARTS + task_k];
            size_t base = ((size_t)task_t * MAX_PARTS + task_k);
            for (int bi = lane; bi < (2 * p->inter) / 8; bi += lanes) {
                int r = bi * 8;
                glm53f_matvec_fp8_bits_8(
                    up + base * GATE_UP_STRIDE + r,
                    p->gate_up + (size_t)r * HIDDEN,
                    p->gate_up_scale + (size_t)(r / 128) * (HIDDEN / 128),
                    x + (size_t)task_t * HIDDEN, HIDDEN);
            }
        }
#pragma omp barrier
        if (task_t >= 0) {
            const glm53f_expert_part *p = &parts[task_t * MAX_PARTS + task_k];
            size_t base = ((size_t)task_t * MAX_PARTS + task_k);
            for (int i = lane; i < p->inter; i += lanes) {
                float g = up[base * GATE_UP_STRIDE + i];
                float u = up[base * GATE_UP_STRIDE + p->inter + i];
                if (g > 10) g = 10; if (g < -100) g = -100;
                if (u > 10) u = 10; if (u < -10) u = -10;
                act[base * INTER_STRIDE + i] = (g / (1 + expf(-g))) * u;
            }
        }
#pragma omp barrier
        if (task_t >= 0) {
            const glm53f_expert_part *p = &parts[task_t * MAX_PARTS + task_k];
            size_t base = ((size_t)task_t * MAX_PARTS + task_k);
            for (int bi = lane; bi < HIDDEN / 8; bi += lanes) {
                int r = bi * 8;
                glm53f_matvec_fp8_bits_8(
                    y + base * HIDDEN + r,
                    p->down + (size_t)r * p->inter,
                    p->down_scale + (size_t)(r / 128) * (p->inter / 128),
                    act + base * INTER_STRIDE, p->inter);
            }
        }
    }
}

/* One expert evaluated for up to four token vectors.  This is the guaranteed
 * reuse case for the shared expert in every MoE layer. */
static inline void glm53f_expert_tokens_bits(
        const glm53f_expert_part *part, int tokens, const float *x,
        float *up, float *act, float *y) {
    enum { HIDDEN = 4096 };
    int inter = part->inter, gate_up = 2 * inter;
    glm53f_mv_fp8_block128_bits_batch(up, part->gate_up,
        part->gate_up_scale, x, tokens, gate_up, HIDDEN);
#pragma omp parallel for schedule(static)
    for (int q = 0; q < tokens * inter; ++q) {
        int t = q / inter, i = q - t * inter;
        float g = up[(size_t)t * gate_up + i];
        float u = up[(size_t)t * gate_up + inter + i];
        if (g > 10) g = 10;
        if (g < -100) g = -100;
        if (u > 10) u = 10;
        if (u < -10) u = -10;
        act[(size_t)t * inter + i] = (g / (1 + expf(-g))) * u;
    }
    glm53f_mv_fp8_block128_bits_batch(y, part->down,
        part->down_scale, act, tokens, HIDDEN, inter);
}
#endif

#endif
