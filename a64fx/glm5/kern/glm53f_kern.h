/* GLM-5.3-Flash decode kernel cores for A64FX.
 *
 * Each core computes output rows [r0, r1) of one matrix-vector product and
 * contains no OpenMP and no libc calls, so the identical object can be timed
 * by the pthread harness (clair sim-accuracy/glm53f), replayed under QLAIR,
 * and wrapped by the production OpenMP runner.  The v0 cores are verbatim
 * copies of the production inner loops in ../glm53f_iq_bridge.c and
 * ../glm53f_target_head_12n.c; optimized variants get a new suffix. */
#ifndef GLM53F_KERN_H
#define GLM53F_KERN_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF K-quant blocks (layouts from common/ggml_dequant.h). */
typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} gk_block_q4_K;
typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} gk_block_q5_K;
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} gk_block_q6_K;
/* Q8_K-style activation block used by the K-quant kernels (glm5_iq_q8_block). */
typedef struct {
    float d;
    int8_t q[256];
} gk_q8k_block;

/* Prepared activation: both llama.cpp contracts, filled by the caller. */
typedef struct {
    int columns;
    const gk_q8k_block *q8k; /* columns/256 blocks, K-quant weights */
    const int8_t *xq;        /* Q8_0 values, contiguous (Q8_0R weights) */
    const float *xpat;       /* per 64 values: 8 lanes d[2k], 8 lanes d[2k+1] */
    const float *xd;         /* per 32-value block: Q8_0 scale as f32 */
    const int32_t *q8k_bsum32; /* per 32-value block: sum of q8k[].q (Q4_KP16) */
    const float *xf;         /* F32 activation (head) */
} gk_act;

/* One matrix-vector product: y[r] = W[r,:] . x for r in [r0, r1). */
typedef struct {
    const uint8_t *w;   /* row 0 of the matrix (row-major, row_bytes apart) */
    size_t row_bytes;
    int rows, columns;
    const gk_act *a;
    float *y;
} gk_mv;

/* Row sizes of the supported formats. */
size_t gk_row_bytes_q8_0r(int columns); /* int8 q[columns], float d[columns/32] */
size_t gk_row_bytes_q4_k(int columns);
size_t gk_row_bytes_q5_k(int columns);
size_t gk_row_bytes_q6_k(int columns);
size_t gk_row_bytes_f32(int columns);

/* v0: production loops.  Q8_0R processes four-row groups: r0 must be a
 * multiple of 4 unless r0 == r1. */
void gk_q8_0r_v0(const gk_mv *m, int r0, int r1);
void gk_q4_k_v0(const gk_mv *m, int r0, int r1);
void gk_q5_k_v0(const gk_mv *m, int r0, int r1);
void gk_q6_k_v0(const gk_mv *m, int r0, int r1);
void gk_f32_v0(const gk_mv *m, int r0, int r1);

/* v1: lossless 16-row panel repack (glm53f_kern_q8r16.c).  The matrix
 * pointer addresses panel 0; panel p holds rows 16p..16p+15 and occupies
 * gk_panel_bytes_q8_0r16() bytes.  r0 must be a multiple of 16. */
size_t gk_panel_bytes_q8_0r16(int columns);
void gk_pack_q8_0r16(uint8_t *dst, const uint8_t *rows, size_t row_bytes,
                     int nrows, int columns);
void gk_q8_0r16_v1(const gk_mv *m, int r0, int r1);
void gk_q8_0r16_v2(const gk_mv *m, int r0, int r1);
void gk_q8_0r16_v3(const gk_mv *m, int r0, int r1);
void gk_q8_0r16_v3pf(const gk_mv *m, int r0, int r1, int pf_bytes);
void gk_q8_0r16_v3pf4k(const gk_mv *m, int r0, int r1);
void gk_q8_0r16_v3pf16k(const gk_mv *m, int r0, int r1);
void gk_q8_0r16_v3pf64k(const gk_mv *m, int r0, int r1);

/* v1: lossless 16-row Q4_K panel repack (glm53f_kern_q4k16.c).  Row-size
 * accounting is per panel/16: 152 bytes per 256 columns. */
size_t gk_row_bytes_q4_kp16(int columns);
void gk_pack_q4_kp16(uint8_t *dst, const uint8_t *rows, size_t row_bytes,
                     int nrows, int columns);
void gk_q4_kp16_v1(const gk_mv *m, int r0, int r1);
void gk_q4_kp16_v1pf(const gk_mv *m, int r0, int r1);

#ifdef __cplusplus
}
#endif
#endif
