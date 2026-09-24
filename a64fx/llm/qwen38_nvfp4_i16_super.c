/* Experimental three-token NVFP4 projection with INT16 activations.
 * The compact source remains resident for serial/draft paths. */
#include <arm_sve.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { Q38_SUPER_MAX = 64 };
typedef struct { uint8_t d[8], qs[64]; } q38_super_subblock;
typedef struct { q38_super_subblock s[4]; } q38_super_block;
typedef struct { uint32_t k; float w; } q38_super_rare;
typedef struct {
    const void *source;
    int rows, cols;
    int8_t *coeff;
    int32_t *weight_sum;
    size_t *rare_offsets;
    q38_super_rare *rare;
    size_t rare_count;
} q38_super_matrix;
static q38_super_matrix q38_super_matrices[Q38_SUPER_MAX];
static int q38_super_count;
static const int8_t q38_super_code[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static float q38_super_scale(uint8_t raw) {
    int x = raw & 127;
    if (x == 0 || x == 127) return 0.0f;
    int exponent = x >> 3, mantissa = x & 7;
    return exponent ? ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 8)
                    : ldexpf((float)mantissa, -10);
}

void q38_nvfp4_i16_super_release(void) {
    for (int i = 0; i < q38_super_count; i++) {
        free(q38_super_matrices[i].coeff);
        free(q38_super_matrices[i].weight_sum);
        free(q38_super_matrices[i].rare_offsets);
        free(q38_super_matrices[i].rare);
    }
    memset(q38_super_matrices, 0, sizeof(q38_super_matrices));
    q38_super_count = 0;
}

int q38_nvfp4_i16_super_register(const void *source, int rows, int cols) {
    if (!source || rows <= 0 || rows % 64 || cols <= 0 || cols % 64 ||
        q38_super_count >= Q38_SUPER_MAX) return -1;
    q38_super_matrix m = {0};
    m.source = source;
    m.rows = rows;
    m.cols = cols;
    size_t values = (size_t)rows * (size_t)cols;
    size_t coeff_bytes = values;
    if (posix_memalign((void **)&m.coeff, 2u * 1024u * 1024u, coeff_bytes) ||
        posix_memalign((void **)&m.weight_sum, 256, (size_t)rows * sizeof(int32_t)))
        goto fail;
    m.rare_offsets = calloc((size_t)rows + 1, sizeof(size_t));
    if (!m.rare_offsets) goto fail;
    const q38_super_block *src = (const q38_super_block *)source;
    int nb = cols / 64;
    for (int row = 0; row < rows; row++) {
        size_t count = 0;
        for (int ib = 0; ib < nb; ib++)
            for (int s = 0; s < 4; s++)
                if (src[(size_t)(row / 8) * nb + ib].s[s].d[row % 8] > 10)
                    count += 16;
        m.rare_offsets[row + 1] = m.rare_offsets[row] + count;
    }
    m.rare_count = m.rare_offsets[rows];
    if (m.rare_count) {
        m.rare = malloc(m.rare_count * sizeof(*m.rare));
        if (!m.rare) goto fail;
    }
    size_t *cursor = malloc((size_t)rows * sizeof(*cursor));
    if (!cursor) goto fail;
    memcpy(cursor, m.rare_offsets, (size_t)rows * sizeof(*cursor));
    memset(m.weight_sum, 0, (size_t)rows * sizeof(int32_t));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(48)
#endif
    for (int nt = 0; nt < rows / 64; nt++) {
        for (int k = 0; k < cols; k += 4)
            for (int g = 0; g < 4; g++)
                for (int r = 0; r < 16; r++)
                    for (int j = 0; j < 4; j++) {
                        int row = nt * 64 + g * 16 + r;
                        int kk = k + j, ib = kk / 64;
                        int s = (kk % 64) / 16, v = kk % 16;
                        const q38_super_subblock *p =
                            &src[(size_t)(row / 8) * nb + ib].s[s];
                        uint8_t z = p->qs[(row % 8) * 8 + v % 8];
                        int code = q38_super_code[v < 8 ? z & 15 : z >> 4];
                        uint8_t d = p->d[row % 8];
                        int8_t coefficient = 0;
                        if (d <= 10) {
                            coefficient = (int8_t)(code * (int)d);
                            m.weight_sum[row] += coefficient;
                        } else {
                            size_t pos = cursor[row]++;
                            m.rare[pos].k = (uint32_t)kk;
                            m.rare[pos].w = (float)code * q38_super_scale(d);
                        }
                        m.coeff[(size_t)nt * cols * 64 + (size_t)k * 64 +
                                (size_t)g * 64 + (size_t)r * 4 + j] = coefficient;
                    }
    }
    free(cursor);
    q38_super_matrices[q38_super_count++] = m;
    fprintf(stderr, "qwen38: i16 supertile rows=%d cols=%d sidecar=%.3fGB rare=%zu\n",
            rows, cols, coeff_bytes / 1e9, m.rare_count);
    return 0;
fail:
    free(m.coeff);
    free(m.weight_sum);
    free(m.rare_offsets);
    free(m.rare);
    return -1;
}

static int q38_super_quantize(int8_t *low, int8_t *high, float scale[3],
                                const float *x, int cols) {
    for (int c = 0; c < 3; c++) {
        float maxabs = 0.0f;
        for (int k = 0; k < cols; k++) {
            float a = fabsf(x[(size_t)c * cols + k]);
            if (!isfinite(a)) return -1;
            if (a > maxabs) maxabs = a;
        }
        scale[c] = maxabs / 32767.0f / 1024.0f;
        float inv = maxabs > 0.0f ? 32767.0f / maxabs : 0.0f;
        for (int k = 0; k < cols; k++) {
            int q = (int)lrintf(x[(size_t)c * cols + k] * inv);
            if (q > 32767) q = 32767;
            if (q < -32767) q = -32767;
            low[(size_t)c * cols + k] = (int8_t)((q & 255) - 128);
            high[(size_t)c * cols + k] = (int8_t)((q - (q & 255)) / 256);
        }
    }
    return 0;
}

static void q38_super_rows(float *y, const q38_super_matrix *m,
                            const int8_t *low, const int8_t *high,
                            const float scale[3], const float *x,
                            int first, int last) {
    const int cols = m->cols, rows = m->rows;
    const svbool_t pb = svptrue_b8(), pi = svptrue_b32();
    for (int nt = first; nt < last; nt++) {
        int64_t sums[3][64] = {{0}};
        const int8_t *wp = m->coeff + (size_t)nt * cols * 64;
        /* At most 256 products enter an INT32 lane before widening. Even
         * 256 * 120 * 32767 is below INT32_MAX; full-K accumulation is not. */
        for (int kb = 0; kb < cols; kb += 256) {
            svint32_t a00=svdup_s32(0),a01=a00,a02=a00;
            svint32_t a10=a00,a11=a00,a12=a00;
            svint32_t a20=a00,a21=a00,a22=a00;
            svint32_t a30=a00,a31=a00,a32=a00;
            int kend = kb + 256 < cols ? kb + 256 : cols;
            for (int k = kb; k < kend; k += 4) {
            uint32_t l0,l1,l2,h0,h1,h2;
            memcpy(&l0, low + k, 4);
            memcpy(&l1, low + cols + k, 4);
            memcpy(&l2, low + 2 * cols + k, 4);
            memcpy(&h0, high + k, 4);
            memcpy(&h1, high + cols + k, 4);
            memcpy(&h2, high + 2 * cols + k, 4);
            svint8_t v0=svreinterpret_s8_u32(svdup_n_u32(l0));
            svint8_t v1=svreinterpret_s8_u32(svdup_n_u32(l1));
            svint8_t v2=svreinterpret_s8_u32(svdup_n_u32(l2));
            svint8_t u0=svreinterpret_s8_u32(svdup_n_u32(h0));
            svint8_t u1=svreinterpret_s8_u32(svdup_n_u32(h1));
            svint8_t u2=svreinterpret_s8_u32(svdup_n_u32(h2));
            const int8_t *p=wp+(size_t)k*64;
#define Q38_SUPER_GROUP(G,A0,A1,A2) do { \
            svint8_t z=svld1_s8(pb,p+(G)*64); \
            svint32_t t0=svdot_s32(svdup_s32(0),z,u0); \
            svint32_t t1=svdot_s32(svdup_s32(0),z,u1); \
            svint32_t t2=svdot_s32(svdup_s32(0),z,u2); \
            (A0)=svadd_s32_x(pi,svdot_s32((A0),z,v0),svlsl_n_s32_x(pi,t0,8)); \
            (A1)=svadd_s32_x(pi,svdot_s32((A1),z,v1),svlsl_n_s32_x(pi,t1,8)); \
            (A2)=svadd_s32_x(pi,svdot_s32((A2),z,v2),svlsl_n_s32_x(pi,t2,8)); \
        } while (0)
            Q38_SUPER_GROUP(0,a00,a01,a02);
            Q38_SUPER_GROUP(1,a10,a11,a12);
            Q38_SUPER_GROUP(2,a20,a21,a22);
            Q38_SUPER_GROUP(3,a30,a31,a32);
#undef Q38_SUPER_GROUP
            }
        int32_t z0[16],z1[16],z2[16];
#define Q38_SUPER_STORE(G,A0,A1,A2) do { \
            svst1_s32(pi,z0,(A0));svst1_s32(pi,z1,(A1));svst1_s32(pi,z2,(A2)); \
            for(int r=0;r<16;r++){ \
                int lane=(G)*16+r; \
                sums[0][lane]+=z0[r]; \
                sums[1][lane]+=z1[r]; \
                sums[2][lane]+=z2[r]; \
            } \
        } while (0)
        Q38_SUPER_STORE(0,a00,a01,a02);
        Q38_SUPER_STORE(1,a10,a11,a12);
        Q38_SUPER_STORE(2,a20,a21,a22);
        Q38_SUPER_STORE(3,a30,a31,a32);
#undef Q38_SUPER_STORE
        }
        for (int lane = 0; lane < 64; lane++) {
            int row = nt * 64 + lane;
            int64_t correction = 128ll * m->weight_sum[row];
            for (int c = 0; c < 3; c++)
                y[(size_t)c * rows + row] =
                    (float)(sums[c][lane] + correction) * scale[c];
        }
        for (int row = nt * 64; row < (nt + 1) * 64; row++)
            for (size_t ri = m->rare_offsets[row]; ri < m->rare_offsets[row + 1]; ri++) {
                uint32_t k = m->rare[ri].k;
                float w = m->rare[ri].w;
                for (int c = 0; c < 3; c++)
                    y[(size_t)c * rows + row] += w * x[(size_t)c * cols + k];
            }
    }
}

int q38_nvfp4_i16_super_mt(float *y, const void *source, const float *x,
                             int rows, int cols, int n_threads) {
    const q38_super_matrix *m = NULL;
    for (int i = 0; i < q38_super_count; i++)
        if (q38_super_matrices[i].source == source &&
            q38_super_matrices[i].rows == rows && q38_super_matrices[i].cols == cols)
            m = &q38_super_matrices[i];
    if (!m) return 0;
    int8_t *digits = malloc((size_t)6 * cols);
    if (!digits) return 0;
    float scale[3];
    if (q38_super_quantize(digits, digits + (size_t)3 * cols, scale, x, cols)) {
        free(digits);
        return 0;
    }
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
        q38_super_rows(y, m, digits, digits + (size_t)3 * cols, scale, x,
                       (rows / 64) * tid / team, (rows / 64) * (tid + 1) / team);
    }
    free(digits);
    return 1;
}
