/* Load-time row-scaled INT8 with [rows/64][K/4][64][4] SDOT packing.
 * Checkpoint scales are folded into the quantizer. No token-time FP8 decode.
 * Rows are padded to 64 and columns to 128 by callers (model shapes already
 * satisfy this contract). One activation scale per vector, FP32 outputs.
 */
#ifndef GLM53F_INT8_H
#define GLM53F_INT8_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

enum { GLM53F_I8_ROWS = 64, GLM53F_I8_TYPE = -8 };

static inline float glm53f_i8_fp8(uint8_t q) {
    int e = (q >> 3) & 15, m = q & 7;
    float v = e ? ldexpf(1.0f + m * 0.125f, e - 7) : m / 512.0f;
    return (q & 127) == 127 ? NAN : (q & 128) ? -v : v;
}

static inline int8_t glm53f_i8_round(float v) {
    float q = roundf(v);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

static inline int glm53f_i8_quantize_x(int8_t *q, float *scale,
                                      const float *x, int cols) {
    float mx = 0;
    for (int c = 0; c < cols; ++c) {
        if (!isfinite(x[c])) return -1;
        if (fabsf(x[c]) > mx) mx = fabsf(x[c]);
    }
    *scale = mx / 127.0f;
    float inv = mx > 0 ? 127.0f / mx : 0;
    for (int c = 0; c < cols; ++c)
        q[c] = glm53f_i8_round(isfinite(inv) ? x[c] * inv : (x[c] / mx) * 127.0f);
    return 0;
}

/* Convert a 64-row tile in place. Workspace is 64*cols bytes per worker.
 * The source FP8 scale grid remains live until all tiles have been converted.
 * `row0` is the original matrix row offset used for the 128-row scale grid.
 */
static inline int glm53f_i8_pack_fp8_tile(uint8_t *tile, float *row_scale,
        const float *block_scale, int row0, int cols, int8_t *scratch) {
    if (cols < 4 || cols % 128 || row0 % 64) return -1;
    int blocks = cols / 128;
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = glm53f_i8_fp8((uint8_t)i);
    for (int r = 0; r < 64; ++r) {
        const float *s = block_scale + (size_t)((row0 + r) / 128) * blocks;
        float mx = 0;
        for (int c = 0; c < cols; ++c) {
            float v = lut[tile[(size_t)r * cols + c]] * s[c / 128];
            if (!isfinite(v)) return -1;
            if (fabsf(v) > mx) mx = fabsf(v);
        }
        row_scale[r] = mx / 127.0f;
        float inv = mx > 0 ? 127.0f / mx : 0;
        for (int c = 0; c < cols; ++c) {
            float v = lut[tile[(size_t)r * cols + c]] * s[c / 128];
            scratch[(size_t)(c / 4) * 256 + r * 4 + c % 4] =
                glm53f_i8_round(isfinite(inv) ? v * inv : (v / mx) * 127.0f);
        }
    }
    memcpy(tile, scratch, (size_t)64 * cols);
    return 0;
}

static inline int glm53f_i8_pack_bf16_tile(int8_t *tile, float *row_scale,
        const uint16_t *source, int cols) {
    if (cols < 128 || cols % 128) return -1;
    for (int r = 0; r < 64; ++r) {
        float mx = 0;
        for (int c = 0; c < cols; ++c) {
            uint32_t bits = (uint32_t)source[(size_t)r * cols + c] << 16;
            float v; memcpy(&v, &bits, sizeof(v));
            if (!isfinite(v)) return -1;
            if (fabsf(v) > mx) mx = fabsf(v);
        }
        row_scale[r] = mx / 127.0f;
        float inv = mx > 0 ? 127.0f / mx : 0;
        for (int c = 0; c < cols; ++c) {
            uint32_t bits = (uint32_t)source[(size_t)r * cols + c] << 16;
            float v; memcpy(&v, &bits, sizeof(v));
            tile[(size_t)(c / 4) * 256 + r * 4 + c % 4] =
                glm53f_i8_round(isfinite(inv) ? v * inv : (v / mx) * 127.0f);
        }
    }
    return 0;
}

static inline void glm53f_i8_dot64(float *out, const int8_t *w,
        const float *scale, const int8_t *x, float xscale, int cols) {
#if defined(__ARM_FEATURE_SVE)
    /* A64FX has 16 FP32 lanes. Each lane produces one output row, avoiding
     * horizontal reductions and streaming one contiguous packed tile. */
    if (svcntw() == 16) {
        svint32_t a0 = svdup_s32(0), a1 = svdup_s32(0), a2 = svdup_s32(0), a3 = svdup_s32(0);
        svint32_t b0 = svdup_s32(0), b1 = svdup_s32(0), b2 = svdup_s32(0), b3 = svdup_s32(0);
        svint32_t c0 = svdup_s32(0), c1 = svdup_s32(0), c2 = svdup_s32(0), c3 = svdup_s32(0);
        svint32_t d0 = svdup_s32(0), d1 = svdup_s32(0), d2 = svdup_s32(0), d3 = svdup_s32(0);
        svbool_t pb = svptrue_b8(), ps = svptrue_b32();
        /* Independent chains cover SDOT latency. Model K is a multiple of
         * 128, so sixteen-column unrolling has no tail. */
        for (int c = 0; c < cols; c += 16) {
#define GLM53F_I8_STEP(O,A0,A1,A2,A3) do { \
            int32_t word; memcpy(&word, x + c + (O), sizeof(word)); \
            svint8_t xv = svreinterpret_s8_s32(svdup_s32(word)); \
            const int8_t *p = w + (size_t)(c + (O)) * 64; \
            A0 = svdot_s32(A0, svld1_s8(pb, p), xv); \
            A1 = svdot_s32(A1, svld1_s8(pb, p + 64), xv); \
            A2 = svdot_s32(A2, svld1_s8(pb, p + 128), xv); \
            A3 = svdot_s32(A3, svld1_s8(pb, p + 192), xv); \
        } while (0)
            GLM53F_I8_STEP(0,a0,a1,a2,a3); GLM53F_I8_STEP(4,b0,b1,b2,b3);
            GLM53F_I8_STEP(8,c0,c1,c2,c3); GLM53F_I8_STEP(12,d0,d1,d2,d3);
#undef GLM53F_I8_STEP
        }
#define GLM53F_I8_MERGE(A,B,C,D) A=svadd_s32_x(ps,svadd_s32_x(ps,A,B),svadd_s32_x(ps,C,D))
        GLM53F_I8_MERGE(a0,b0,c0,d0); GLM53F_I8_MERGE(a1,b1,c1,d1);
        GLM53F_I8_MERGE(a2,b2,c2,d2); GLM53F_I8_MERGE(a3,b3,c3,d3);
#undef GLM53F_I8_MERGE
#define GLM53F_I8_STORE(N,A) svst1(ps, out + (N), svmul_f32_x(ps, \
        svcvt_f32_s32_x(ps, A), svmul_n_f32_x(ps, svld1(ps, scale + (N)), xscale)))
        GLM53F_I8_STORE(0, a0); GLM53F_I8_STORE(16, a1);
        GLM53F_I8_STORE(32, a2); GLM53F_I8_STORE(48, a3);
#undef GLM53F_I8_STORE
        return;
    }
#endif
    for (int r = 0; r < 64; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < cols; ++c)
            sum += w[(size_t)(c / 4) * 256 + r * 4 + c % 4] * (int)x[c];
        out[r] = (float)sum * (scale[r] * xscale);
    }
}

/* One quarter of a packed 64-row tile. Finer work units keep 47 workers busy
 * for narrow gate/QKV projections and avoid the 64/47 down-projection tail.
 * `w` points to row-within-tile * 4; column groups still have a 256-byte stride. */
static inline void glm53f_i8_dot16(float *out, const int8_t *w,
        const float *scale, const int8_t *x, float xscale, int cols) {
#if defined(__ARM_FEATURE_SVE)
    if (svcntw() == 16) {
        svint32_t a0=svdup_s32(0), a1=svdup_s32(0), a2=svdup_s32(0), a3=svdup_s32(0);
        svint32_t b0=svdup_s32(0), b1=svdup_s32(0), b2=svdup_s32(0), b3=svdup_s32(0);
        svint32_t c0=svdup_s32(0), c1=svdup_s32(0), c2=svdup_s32(0), c3=svdup_s32(0);
        svint32_t d0=svdup_s32(0), d1=svdup_s32(0), d2=svdup_s32(0), d3=svdup_s32(0);
        svbool_t pb=svptrue_b8(), ps=svptrue_b32();
        for (int c=0; c<cols; c+=64) {
#define GLM53F_I8_16_STEP(O,A) do { \
            int32_t word; memcpy(&word,x+c+(O),4); \
            A=svdot_s32(A,svld1_s8(pb,w+(size_t)(c+(O))*64),svreinterpret_s8_s32(svdup_s32(word))); \
        } while(0)
            GLM53F_I8_16_STEP(0,a0); GLM53F_I8_16_STEP(4,a1); GLM53F_I8_16_STEP(8,a2); GLM53F_I8_16_STEP(12,a3);
            GLM53F_I8_16_STEP(16,b0); GLM53F_I8_16_STEP(20,b1); GLM53F_I8_16_STEP(24,b2); GLM53F_I8_16_STEP(28,b3);
            GLM53F_I8_16_STEP(32,c0); GLM53F_I8_16_STEP(36,c1); GLM53F_I8_16_STEP(40,c2); GLM53F_I8_16_STEP(44,c3);
            GLM53F_I8_16_STEP(48,d0); GLM53F_I8_16_STEP(52,d1); GLM53F_I8_16_STEP(56,d2); GLM53F_I8_16_STEP(60,d3);
#undef GLM53F_I8_16_STEP
        }
#define GLM53F_I8_16_MERGE(A,B,C,D) A=svadd_s32_x(ps,svadd_s32_x(ps,A,B),svadd_s32_x(ps,C,D))
        GLM53F_I8_16_MERGE(a0,a1,a2,a3); GLM53F_I8_16_MERGE(b0,b1,b2,b3);
        GLM53F_I8_16_MERGE(c0,c1,c2,c3); GLM53F_I8_16_MERGE(d0,d1,d2,d3);
        GLM53F_I8_16_MERGE(a0,b0,c0,d0);
#undef GLM53F_I8_16_MERGE
        svst1(ps,out,svmul_f32_x(ps,svcvt_f32_s32_x(ps,a0),svmul_n_f32_x(ps,svld1(ps,scale),xscale)));
        return;
    }
#endif
    for (int r=0; r<16; ++r) {
        int32_t sum=0;
        for (int c=0; c<cols; ++c) sum+=w[(size_t)(c/4)*256+r*4+c%4]*(int)x[c];
        out[r]=(float)sum*(scale[r]*xscale);
    }
}

/* Token-matrix form of dot16; retain the on-load packed weight layout and
 * per-token activation scales. Two K chains for eight tokens, or four chains
 * for four tokens, provide sixteen independent SDOT accumulators. Integer
 * reassociation is exact (model K <= 4096, far below signed int32 overflow).
 * No horizontal float reduction or changed dequantization expression. */
static inline void glm53f_i8_dot16_batch(float *out, size_t out_stride,
        const int8_t *w, const float *scale, const int8_t *x, size_t x_stride,
        const float *xscale, int tokens, int cols) {
    int base = 0;
#if defined(__ARM_FEATURE_SVE)
    if (svcntw() == 16) {
        svbool_t pb = svptrue_b8(), ps = svptrue_b32();
        for (; base + 8 <= tokens; base += 8) {
            svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
            svint32_t b0=a0,b1=a0,b2=a0,b3=a0,b4=a0,b5=a0,b6=a0,b7=a0;
            for (int c = 0; c < cols; c += 8) {
#define GLM53F_I8_BT(T,A,O) do { \
                int32_t word; memcpy(&word, x + (size_t)(base+(T))*x_stride+c+(O), 4); \
                A = svdot_s32(A, weight, svreinterpret_s8_s32(svdup_s32(word))); \
            } while (0)
                svint8_t weight = svld1_s8(pb, w + (size_t)c * 64);
                GLM53F_I8_BT(0,a0,0); GLM53F_I8_BT(1,a1,0);
                GLM53F_I8_BT(2,a2,0); GLM53F_I8_BT(3,a3,0);
                GLM53F_I8_BT(4,a4,0); GLM53F_I8_BT(5,a5,0);
                GLM53F_I8_BT(6,a6,0); GLM53F_I8_BT(7,a7,0);
                weight = svld1_s8(pb, w + (size_t)(c + 4) * 64);
                GLM53F_I8_BT(0,b0,4); GLM53F_I8_BT(1,b1,4);
                GLM53F_I8_BT(2,b2,4); GLM53F_I8_BT(3,b3,4);
                GLM53F_I8_BT(4,b4,4); GLM53F_I8_BT(5,b5,4);
                GLM53F_I8_BT(6,b6,4); GLM53F_I8_BT(7,b7,4);
#undef GLM53F_I8_BT
            }
#define GLM53F_I8_BSTORE(T,A,B) svst1(ps, out + (size_t)(base+(T))*out_stride, \
                svmul_f32_x(ps, svcvt_f32_s32_x(ps, svadd_s32_x(ps,A,B)), \
                    svmul_n_f32_x(ps,svld1(ps,scale),xscale[base+(T)])))
            GLM53F_I8_BSTORE(0,a0,b0); GLM53F_I8_BSTORE(1,a1,b1);
            GLM53F_I8_BSTORE(2,a2,b2); GLM53F_I8_BSTORE(3,a3,b3);
            GLM53F_I8_BSTORE(4,a4,b4); GLM53F_I8_BSTORE(5,a5,b5);
            GLM53F_I8_BSTORE(6,a6,b6); GLM53F_I8_BSTORE(7,a7,b7);
#undef GLM53F_I8_BSTORE
        }
        for (; base + 4 <= tokens; base += 4) {
            svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0;
            svint32_t b0=a0,b1=a0,b2=a0,b3=a0,c0=a0,c1=a0,c2=a0,c3=a0;
            svint32_t d0=a0,d1=a0,d2=a0,d3=a0;
            for (int c = 0; c < cols; c += 16) {
#define GLM53F_I8_BT(T,A,O) do { \
                int32_t word; memcpy(&word, x + (size_t)(base+(T))*x_stride+c+(O), 4); \
                A = svdot_s32(A, weight, svreinterpret_s8_s32(svdup_s32(word))); \
            } while (0)
#define GLM53F_I8_BSTEP(O,A,B,C,D) do { \
                svint8_t weight = svld1_s8(pb,w+(size_t)(c+(O))*64); \
                GLM53F_I8_BT(0,A,O); GLM53F_I8_BT(1,B,O); \
                GLM53F_I8_BT(2,C,O); GLM53F_I8_BT(3,D,O); \
            } while (0)
                GLM53F_I8_BSTEP(0,a0,a1,a2,a3); GLM53F_I8_BSTEP(4,b0,b1,b2,b3);
                GLM53F_I8_BSTEP(8,c0,c1,c2,c3); GLM53F_I8_BSTEP(12,d0,d1,d2,d3);
#undef GLM53F_I8_BSTEP
#undef GLM53F_I8_BT
            }
#define GLM53F_I8_BSTORE(T,A,B,C,D) svst1(ps,out+(size_t)(base+(T))*out_stride, \
                svmul_f32_x(ps,svcvt_f32_s32_x(ps,svadd_s32_x(ps, \
                    svadd_s32_x(ps,A,B),svadd_s32_x(ps,C,D))), \
                    svmul_n_f32_x(ps,svld1(ps,scale),xscale[base+(T)])))
            GLM53F_I8_BSTORE(0,a0,b0,c0,d0); GLM53F_I8_BSTORE(1,a1,b1,c1,d1);
            GLM53F_I8_BSTORE(2,a2,b2,c2,d2); GLM53F_I8_BSTORE(3,a3,b3,c3,d3);
#undef GLM53F_I8_BSTORE
        }
    }
#endif
    for (; base < tokens; ++base)
        glm53f_i8_dot16(out + (size_t)base * out_stride, w, scale,
                        x + (size_t)base * x_stride, xscale[base], cols);
}

/* Sixteen independent tokens reuse each packed weight vector. Integer sums
 * are exact at K<=4096 (including -128 inputs); the existing 8/4/1 tails and
 * post-dot scale multiplication order are retained. */
static inline void glm53f_i8_dot16_batch16(float *out, size_t out_stride,
        const int8_t *w, const float *scale, const int8_t *x, size_t x_stride,
        const float *xscale, int tokens, int cols) {
    int base = 0;
#if defined(__ARM_FEATURE_SVE)
    if (svcntw() == 16) {
        svbool_t pb = svptrue_b8(), ps = svptrue_b32();
        for (; base + 16 <= tokens; base += 16) {
            svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
            svint32_t a8=a0,a9=a0,a10=a0,a11=a0,a12=a0,a13=a0,a14=a0,a15=a0;
            for (int c = 0; c < cols; c += 4) {
                svint8_t weight = svld1_s8(pb, w + (size_t)c * 64);
#define GLM53F_I8_T16(T) do { \
                int32_t word; memcpy(&word, x + (size_t)(base+(T))*x_stride+c, 4); \
                a##T = svdot_s32(a##T, weight, svreinterpret_s8_s32(svdup_s32(word))); \
            } while (0)
                GLM53F_I8_T16(0); GLM53F_I8_T16(1); GLM53F_I8_T16(2); GLM53F_I8_T16(3);
                GLM53F_I8_T16(4); GLM53F_I8_T16(5); GLM53F_I8_T16(6); GLM53F_I8_T16(7);
                GLM53F_I8_T16(8); GLM53F_I8_T16(9); GLM53F_I8_T16(10); GLM53F_I8_T16(11);
                GLM53F_I8_T16(12); GLM53F_I8_T16(13); GLM53F_I8_T16(14); GLM53F_I8_T16(15);
#undef GLM53F_I8_T16
            }
#define GLM53F_I8_T16(T) svst1(ps, out + (size_t)(base+(T))*out_stride, \
                svmul_f32_x(ps, svcvt_f32_s32_x(ps,a##T), \
                    svmul_n_f32_x(ps,svld1(ps,scale),xscale[base+(T)])))
            GLM53F_I8_T16(0); GLM53F_I8_T16(1); GLM53F_I8_T16(2); GLM53F_I8_T16(3);
            GLM53F_I8_T16(4); GLM53F_I8_T16(5); GLM53F_I8_T16(6); GLM53F_I8_T16(7);
            GLM53F_I8_T16(8); GLM53F_I8_T16(9); GLM53F_I8_T16(10); GLM53F_I8_T16(11);
            GLM53F_I8_T16(12); GLM53F_I8_T16(13); GLM53F_I8_T16(14); GLM53F_I8_T16(15);
#undef GLM53F_I8_T16
        }
    }
#endif
    if (base < tokens)
        glm53f_i8_dot16_batch(out + (size_t)base * out_stride, out_stride, w, scale,
            x + (size_t)base * x_stride, x_stride, xscale + base, tokens - base, cols);
}

/* Row-major control used to compare the packed stream with eight independent
 * weight streams. Both accumulate exactly the same integer dot products. */
static inline void glm53f_i8_dot8_rows(float *out, const int8_t *w,
        const float *scale, const int8_t *x, float xscale, int cols) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t a0=svdup_s32(0), a1=svdup_s32(0), a2=svdup_s32(0), a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0), a5=svdup_s32(0), a6=svdup_s32(0), a7=svdup_s32(0);
    for (int c=0; c<cols; c+=(int)svcntb()) {
        svbool_t p=svwhilelt_b8(c,cols);
        svint8_t xv=svld1_s8(p,x+c);
#define GLM53F_I8_ROW(R,A) A=svdot_s32(A,svld1_s8(p,w+(size_t)(R)*cols+c),xv)
        GLM53F_I8_ROW(0,a0); GLM53F_I8_ROW(1,a1); GLM53F_I8_ROW(2,a2); GLM53F_I8_ROW(3,a3);
        GLM53F_I8_ROW(4,a4); GLM53F_I8_ROW(5,a5); GLM53F_I8_ROW(6,a6); GLM53F_I8_ROW(7,a7);
#undef GLM53F_I8_ROW
    }
#define GLM53F_I8_SUM(R,A) out[R]=(float)svaddv_s32(svptrue_b32(),A)*(scale[R]*xscale)
    GLM53F_I8_SUM(0,a0); GLM53F_I8_SUM(1,a1); GLM53F_I8_SUM(2,a2); GLM53F_I8_SUM(3,a3);
    GLM53F_I8_SUM(4,a4); GLM53F_I8_SUM(5,a5); GLM53F_I8_SUM(6,a6); GLM53F_I8_SUM(7,a7);
#undef GLM53F_I8_SUM
#else
    for(int r=0;r<8;r++){int sum=0;for(int c=0;c<cols;c++)sum+=w[(size_t)r*cols+c]*(int)x[c];out[r]=sum*(scale[r]*xscale);}
#endif
}

#endif
