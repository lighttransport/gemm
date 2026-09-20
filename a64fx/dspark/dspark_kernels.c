#define _GNU_SOURCE
#include "dspark_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

float ds_bf16_to_f32(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t ds_f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float ds_decode_e2m1(uint8_t x) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    return table[x & 15u];
}

float ds_decode_e4m3(uint8_t x) {
    int sign = x >> 7;
    int exp = (x >> 3) & 15;
    int mant = x & 7;
    float v;
    if (exp == 0) v = ldexpf((float)mant, -9);
    else if (exp == 15 && mant == 7) return copysignf(NAN, sign ? -1.0f : 1.0f);
    else v = ldexpf(1.0f + (float)mant * 0.125f, exp - 7);
    return sign ? -v : v;
}

void *ds_anon_alloc(size_t bytes) {
    if (!bytes) return NULL;
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

void ds_anon_free(void *ptr, size_t bytes) {
    if (ptr && bytes) (void)munmap(ptr, bytes);
}

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t ds_load_bf16_lo(svbool_t pg, const uint16_t *p) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}

static void ds_gemm_bf16_sve(const uint16_t *w0, const uint16_t *w1,
                             size_t rows, size_t cols, const float *x,
                             size_t m, float *y0, float *y1, int threads) {
    size_t jobs = w1 ? 2 * rows : rows;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t job = 0; job < jobs; ++job) {
        size_t r = job < rows ? job : job - rows;
        const uint16_t *w = job < rows ? w0 : w1;
        float *y = job < rows ? y0 : y1;
        const uint16_t *wr = w + r * cols;
        svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
        svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
        svfloat32_t b0=svdup_f32(0),b1=svdup_f32(0),b2=svdup_f32(0),b3=svdup_f32(0);
        svfloat32_t b4=svdup_f32(0),b5=svdup_f32(0),b6=svdup_f32(0),b7=svdup_f32(0);
        size_t vl = svcntw();
        size_t k = 0;
        for (; k + vl < cols; k += 2 * vl) {
            svbool_t pg = svwhilelt_b32(k, cols);
            svbool_t pg1 = svwhilelt_b32(k + vl, cols);
            svfloat32_t wa = ds_load_bf16_lo(pg, wr + k);
            svfloat32_t wb = ds_load_bf16_lo(pg1, wr + k + vl);
            #define DS_FMA2(A,B,I) do { if (m > (I)) { \
                (A)=svmla_f32_m(pg,(A),wa,svld1_f32(pg,x+(size_t)(I)*cols+k)); \
                (B)=svmla_f32_m(pg1,(B),wb,svld1_f32(pg1,x+(size_t)(I)*cols+k+vl)); \
            } } while (0)
            DS_FMA2(a0,b0,0); DS_FMA2(a1,b1,1); DS_FMA2(a2,b2,2); DS_FMA2(a3,b3,3);
            DS_FMA2(a4,b4,4); DS_FMA2(a5,b5,5); DS_FMA2(a6,b6,6); DS_FMA2(a7,b7,7);
            #undef DS_FMA2
        }
        if (k < cols) {
            svbool_t pg = svwhilelt_b32(k, cols);
            svfloat32_t vw = ds_load_bf16_lo(pg, wr + k);
            #define DS_FMA(A,I) do { if (m > (I)) (A)=svmla_f32_m(pg,(A),vw,svld1_f32(pg,x+(size_t)(I)*cols+k)); } while (0)
            DS_FMA(a0,0); DS_FMA(a1,1); DS_FMA(a2,2); DS_FMA(a3,3);
            DS_FMA(a4,4); DS_FMA(a5,5); DS_FMA(a6,6); DS_FMA(a7,7);
            #undef DS_FMA
        }
        svbool_t all=svptrue_b32();
        if(m>0)y[r]=svaddv_f32(all,svadd_f32_x(all,a0,b0)); if(m>1)y[rows+r]=svaddv_f32(all,svadd_f32_x(all,a1,b1));
        if(m>2)y[2*rows+r]=svaddv_f32(all,svadd_f32_x(all,a2,b2)); if(m>3)y[3*rows+r]=svaddv_f32(all,svadd_f32_x(all,a3,b3));
        if(m>4)y[4*rows+r]=svaddv_f32(all,svadd_f32_x(all,a4,b4)); if(m>5)y[5*rows+r]=svaddv_f32(all,svadd_f32_x(all,a5,b5));
        if(m>6)y[6*rows+r]=svaddv_f32(all,svadd_f32_x(all,a6,b6)); if(m>7)y[7*rows+r]=svaddv_f32(all,svadd_f32_x(all,a7,b7));
    }
}
#endif

void ds_gemm_bf16(const dspark_model *model, const uint16_t *w,
                  size_t rows, size_t cols, const float *x, size_t m,
                  float *y) {
    if (!model || !w || !x || !y || !m || m > 8) return;
#if defined(__ARM_FEATURE_SVE)
    if (model->backend == DSPARK_BACKEND_SVE) {
        ds_gemm_bf16_sve(w, NULL, rows, cols, x, m, y, NULL, model->threads);
        return;
    }
#endif
    #pragma omp parallel for num_threads(model->threads) schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t *wr = w + r * cols;
        for (size_t j = 0; j < m; ++j) {
            float sum = 0.0f;
            const float *xr = x + j * cols;
            for (size_t k = 0; k < cols; ++k)
                sum = fmaf(ds_bf16_to_f32(wr[k]), xr[k], sum);
            y[j * rows + r] = sum;
        }
    }
}

void ds_gemm_bf16_pair(const dspark_model *model, const uint16_t *w0,
                       const uint16_t *w1, size_t rows, size_t cols,
                       const float *x, size_t m, float *y0, float *y1) {
    if (!model || !w0 || !w1 || !x || !y0 || !y1 || !m || m > 8) return;
#if defined(__ARM_FEATURE_SVE)
    if (model->backend == DSPARK_BACKEND_SVE) {
        ds_gemm_bf16_sve(w0, w1, rows, cols, x, m, y0, y1,
                         model->threads);
        return;
    }
#endif
    ds_gemm_bf16(model, w0, rows, cols, x, m, y0);
    ds_gemm_bf16(model, w1, rows, cols, x, m, y1);
}

void ds_rmsnorm(const uint16_t *weight, const float *x, float *y,
                size_t rows, size_t cols, float eps, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        const float *xr = x + r * cols;
        float *yr = y + r * cols;
        double ss = 0.0;
        for (size_t i = 0; i < cols; ++i) ss += (double)xr[i] * xr[i];
        float scale = 1.0f / sqrtf((float)(ss / (double)cols) + eps);
        for (size_t i = 0; i < cols; ++i)
            yr[i] = xr[i] * scale * (weight ? ds_bf16_to_f32(weight[i]) : 1.0f);
    }
}

void ds_head_rmsnorm(const uint16_t *weight, float *x, size_t rows,
                     size_t heads, size_t head_dim, float eps, int threads) {
    size_t jobs = rows * heads;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t job = 0; job < jobs; ++job) {
        float *v = x + job * head_dim;
        double ss = 0.0;
        for (size_t i = 0; i < head_dim; ++i) ss += (double)v[i] * v[i];
        float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (size_t i = 0; i < head_dim; ++i)
            v[i] *= scale * ds_bf16_to_f32(weight[i]);
    }
}

void ds_apply_rope(const dspark_model *model, float *x, size_t rows,
                   size_t heads, size_t position0, int threads) {
    size_t jobs = rows * heads;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t job = 0; job < jobs; ++job) {
        size_t row = job / heads;
        float *v = x + job * DS_HEAD_DIM;
        float pos = (float)(position0 + row);
        for (size_t i = 0; i < DS_HEAD_DIM / 2; ++i) {
            float angle = pos * model->rope_inv_freq[i];
            float c = cosf(angle) * model->rope_attention_factor;
            float s = sinf(angle) * model->rope_attention_factor;
            float a = v[i], b = v[i + DS_HEAD_DIM / 2];
            v[i] = a * c - b * s;
            v[i + DS_HEAD_DIM / 2] = b * c + a * s;
        }
    }
}

float ds_dot_bf16(const float *x, const uint16_t *y, size_t n,
                  dspark_backend backend) {
#if defined(__ARM_FEATURE_SVE)
    if (backend == DSPARK_BACKEND_SVE) {
        svfloat32_t sum0 = svdup_f32(0.0f), sum1 = svdup_f32(0.0f);
        size_t vl = svcntw();
        size_t i = 0;
        for (; i + vl < n; i += 2 * vl) {
            svbool_t pg = svwhilelt_b32(i, n);
            svbool_t pg1 = svwhilelt_b32(i + vl, n);
            sum0 = svmla_f32_m(pg, sum0, svld1_f32(pg, x + i), ds_load_bf16_lo(pg, y + i));
            sum1 = svmla_f32_m(pg1, sum1, svld1_f32(pg1, x + i + vl), ds_load_bf16_lo(pg1, y + i + vl));
        }
        if (i < n) {
            svbool_t pg = svwhilelt_b32(i, n);
            sum0=svmla_f32_m(pg,sum0,svld1_f32(pg,x+i),ds_load_bf16_lo(pg,y+i));
        }
        svbool_t all=svptrue_b32();
        return svaddv_f32(all,svadd_f32_x(all,sum0,sum1));
    }
#else
    (void)backend;
#endif
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum = fmaf(x[i], ds_bf16_to_f32(y[i]), sum);
    return sum;
}

static inline const uint16_t *ds_attention_row(const uint16_t *cache,
                                                const uint16_t *block,
                                                size_t context, size_t at,
                                                size_t kv_head) {
    size_t row = at < context ? at : at - context;
    const uint16_t *base = at < context ? cache : block;
    return base + (row * DS_KV_HEADS + kv_head) * DS_HEAD_DIM;
}

float ds_attention_scores_bf16(const float *q, const uint16_t *kc,
                               const uint16_t *kn, size_t context,
                               size_t total, size_t kv_head, float scale,
                               float *scores, dspark_backend backend) {
    float mx = -INFINITY;
#if defined(__ARM_FEATURE_SVE)
    if (backend == DSPARK_BACKEND_SVE) {
        svbool_t all = svptrue_b32();
        size_t vl = svcntw();
        for (size_t at = 0; at < total; ++at) {
            const uint16_t *kp=ds_attention_row(kc,kn,context,at,kv_head);
            svfloat32_t a=svdup_f32(0),b=svdup_f32(0);size_t d=0;
            for(;d+vl<DS_HEAD_DIM;d+=2*vl){
                svbool_t p0=svwhilelt_b32(d,(size_t)DS_HEAD_DIM);
                svbool_t p1=svwhilelt_b32(d+vl,(size_t)DS_HEAD_DIM);
                a=svmla_f32_m(p0,a,svld1_f32(p0,q+d),ds_load_bf16_lo(p0,kp+d));
                b=svmla_f32_m(p1,b,svld1_f32(p1,q+d+vl),ds_load_bf16_lo(p1,kp+d+vl));
            }
            if(d<DS_HEAD_DIM){svbool_t p=svwhilelt_b32(d,(size_t)DS_HEAD_DIM);a=svmla_f32_m(p,a,svld1_f32(p,q+d),ds_load_bf16_lo(p,kp+d));}
            float z=svaddv_f32(all,svadd_f32_x(all,a,b))*scale;scores[at]=z;if(z>mx)mx=z;
        }
        return mx;
    }
#endif
    for(size_t at=0;at<total;at++){
        const uint16_t*kp=ds_attention_row(kc,kn,context,at,kv_head);
        float z=ds_dot_bf16(q,kp,DS_HEAD_DIM,backend)*scale;scores[at]=z;if(z>mx)mx=z;
    }
    return mx;
}

void ds_attention_values_bf16(const float *scores, const uint16_t *vc,
                              const uint16_t *vn, size_t context,
                              size_t total, size_t kv_head, float norm,
                              float *out, dspark_backend backend) {
#if defined(__ARM_FEATURE_SVE)
    if (backend == DSPARK_BACKEND_SVE && svcntw() == 16) {
        svbool_t pg=svptrue_b32();
        svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
        svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
        for(size_t at=0;at<total;at++){
            const uint16_t*vp=ds_attention_row(vc,vn,context,at,kv_head);
            float w=scores[at]*norm;
            a0=svmla_n_f32_x(pg,a0,ds_load_bf16_lo(pg,vp+0),w);
            a1=svmla_n_f32_x(pg,a1,ds_load_bf16_lo(pg,vp+16),w);
            a2=svmla_n_f32_x(pg,a2,ds_load_bf16_lo(pg,vp+32),w);
            a3=svmla_n_f32_x(pg,a3,ds_load_bf16_lo(pg,vp+48),w);
            a4=svmla_n_f32_x(pg,a4,ds_load_bf16_lo(pg,vp+64),w);
            a5=svmla_n_f32_x(pg,a5,ds_load_bf16_lo(pg,vp+80),w);
            a6=svmla_n_f32_x(pg,a6,ds_load_bf16_lo(pg,vp+96),w);
            a7=svmla_n_f32_x(pg,a7,ds_load_bf16_lo(pg,vp+112),w);
        }
        svst1_f32(pg,out+0,a0);svst1_f32(pg,out+16,a1);svst1_f32(pg,out+32,a2);svst1_f32(pg,out+48,a3);
        svst1_f32(pg,out+64,a4);svst1_f32(pg,out+80,a5);svst1_f32(pg,out+96,a6);svst1_f32(pg,out+112,a7);
        return;
    }
#endif
    memset(out,0,DS_HEAD_DIM*sizeof(float));
    for(size_t at=0;at<total;at++){
        float w=scores[at]*norm;const uint16_t*vp=ds_attention_row(vc,vn,context,at,kv_head);
        for(size_t d=0;d<DS_HEAD_DIM;d++)out[d]=fmaf(w,ds_bf16_to_f32(vp[d]),out[d]);
    }
}

static void ds_nvfp4_gemm_scalar(const ds_nvfp4_matrix *w, const float *x,
                                 size_t m, float *y, int threads) {
    size_t panels = (w->n + 15) / 16;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t p = 0; p < panels; ++p) {
        float acc[8][16] = {{0}};
        for (size_t g = 0; g < w->groups; ++g) {
            const uint8_t *sp = w->scales + (p * w->groups + g) * 16;
            for (size_t q = 0; q < 8; ++q) {
                const uint8_t *cp = w->codes + ((p * w->groups + g) * 8 + q) * 16;
                size_t k0 = g * 16 + q * 2;
                for (size_t lane = 0; lane < 16; ++lane) {
                    size_t col = p * 16 + lane;
                    if (col >= w->n) continue;
                    float sc = ds_decode_e4m3(sp[lane]) * w->global_scale;
                    float a = ds_decode_e2m1(cp[lane] & 15u) * sc;
                    float b = ds_decode_e2m1(cp[lane] >> 4) * sc;
                    for (size_t r = 0; r < m; ++r)
                        acc[r][lane] = fmaf(a, x[r*w->k+k0],
                                            fmaf(b, x[r*w->k+k0+1], acc[r][lane]));
                }
            }
        }
        for (size_t r = 0; r < m; ++r)
            for (size_t lane = 0; lane < 16 && p*16+lane < w->n; ++lane)
                y[r*w->n+p*16+lane] = acc[r][lane];
    }
}

#if defined(__ARM_FEATURE_SVE)
static const float ds_fp4_table[16] = {
    0,.5f,1,1.5f,2,3,4,6,-0,-.5f,-1,-1.5f,-2,-3,-4,-6
};
static float ds_fp8_table[256];
static int ds_fp8_ready;

static void ds_init_fp8_table(void) {
    if (ds_fp8_ready) return;
    #pragma omp critical(dspark_fp8_table)
    {
        if (!ds_fp8_ready) {
            for (int i=0;i<256;i++) ds_fp8_table[i]=ds_decode_e4m3((uint8_t)i);
            ds_fp8_ready=1;
        }
    }
}

static void ds_nvfp4_gemm7_sve(const ds_nvfp4_matrix *w, const float *x,
                               float *y, int threads) {
    size_t panels = w->n / 16;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t p = 0; p < panels; ++p) {
        svbool_t pg=svptrue_b32();
        svfloat32_t tab=svld1_f32(pg,ds_fp4_table);
        svfloat32_t z0=svdup_f32(0),z1=svdup_f32(0),z2=svdup_f32(0),z3=svdup_f32(0);
        svfloat32_t z4=svdup_f32(0),z5=svdup_f32(0),z6=svdup_f32(0);
        svfloat32_t u0=svdup_f32(0),u1=svdup_f32(0),u2=svdup_f32(0),u3=svdup_f32(0);
        svfloat32_t u4=svdup_f32(0),u5=svdup_f32(0),u6=svdup_f32(0);
        for(size_t g=0;g<w->groups;++g){
            const uint8_t*sp=w->scales+(p*w->groups+g)*16;
            svuint32_t si=svld1ub_u32(pg,sp);
            svfloat32_t scale=svmul_n_f32_x(pg,svld1_gather_u32index_f32(pg,ds_fp8_table,si),w->global_scale);
            for(size_t q=0;q<8;++q){
                svuint32_t raw=svld1ub_u32(pg,w->codes+((p*w->groups+g)*8+q)*16);
                svfloat32_t a=svmul_f32_x(pg,svtbl_f32(tab,svand_n_u32_x(pg,raw,15)),scale);
                svfloat32_t b=svmul_f32_x(pg,svtbl_f32(tab,svlsr_n_u32_x(pg,raw,4)),scale);
                size_t k=g*16+q*2;
                #define DS_NV_ROW(Z,R) do { (Z)=svmla_n_f32_x(pg,(Z),a,x[(R)*w->k+k]); (Z)=svmla_n_f32_x(pg,(Z),b,x[(R)*w->k+k+1]); } while(0)
                if(q&1){DS_NV_ROW(u0,0);DS_NV_ROW(u1,1);DS_NV_ROW(u2,2);DS_NV_ROW(u3,3);DS_NV_ROW(u4,4);DS_NV_ROW(u5,5);DS_NV_ROW(u6,6);}
                else{DS_NV_ROW(z0,0);DS_NV_ROW(z1,1);DS_NV_ROW(z2,2);DS_NV_ROW(z3,3);DS_NV_ROW(z4,4);DS_NV_ROW(z5,5);DS_NV_ROW(z6,6);}
                #undef DS_NV_ROW
            }
        }
        svst1_f32(pg,y+0*w->n+p*16,svadd_f32_x(pg,z0,u0));svst1_f32(pg,y+1*w->n+p*16,svadd_f32_x(pg,z1,u1));
        svst1_f32(pg,y+2*w->n+p*16,svadd_f32_x(pg,z2,u2));svst1_f32(pg,y+3*w->n+p*16,svadd_f32_x(pg,z3,u3));
        svst1_f32(pg,y+4*w->n+p*16,svadd_f32_x(pg,z4,u4));svst1_f32(pg,y+5*w->n+p*16,svadd_f32_x(pg,z5,u5));
        svst1_f32(pg,y+6*w->n+p*16,svadd_f32_x(pg,z6,u6));
    }
}
#endif

void ds_nvfp4_gemm(const dspark_model *model, const ds_nvfp4_matrix *w,
                   const float *x, size_t m, float *y) {
#if defined(__ARM_FEATURE_SVE)
    if (model->backend == DSPARK_BACKEND_SVE && m == 7 && !(w->n & 15)) {
        ds_init_fp8_table();
        ds_nvfp4_gemm7_sve(w, x, y, model->threads);
        return;
    }
#endif
    ds_nvfp4_gemm_scalar(w, x, m, y, model->threads);
}
