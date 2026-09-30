/* Lossless conversion of Q8_0-family GGUF-derived matrices to the int8 panel64 layout (sb = 32) consumed by
 * gk_gemm_panel64 / gk_gemm_tile6x4p_asm.  Supported sources: GGML Q8_0 (34-byte blocks), NATIVE_Q8_0R (int8 row
 * followed by float scales) and NATIVE_Q8_0R16 (16-row panels, 576 B per 32 columns).  rows must be a multiple of 64. */
#ifndef GLM53F_Q80_PANEL64_H
#define GLM53F_Q80_PANEL64_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_iq_bridge.h"
#include "kern/glm53f_kern.h"

static inline int glm53f_q80_family(int type) {
    return type == GLM53F_GGML_Q8_0 || type == GLM53F_NATIVE_Q8_0R || type == GLM53F_NATIVE_Q8_0R16;
}

/* Returns a malloc'd (256-aligned) buffer of rows/64 panels, or NULL. */
static inline uint8_t *glm53f_q80_to_panel64(int type, const uint8_t *src, int rows, int cols) {
    if (!glm53f_q80_family(type) || rows % 64 || cols % 32) return NULL;
    const size_t pbytes = gk_panel64_bytes(32, cols);
    uint8_t *dst = NULL;
    if (posix_memalign((void **)&dst, 256, (size_t)(rows / 64) * pbytes)) return NULL;
    memset(dst, 0, (size_t)(rows / 64) * pbytes);
    const int nb = cols / 32;
    const size_t rb_r = (size_t)cols + (size_t)nb * 4, rb_q = (size_t)nb * 34;
    int8_t *q = (int8_t *)malloc((size_t)64 * cols);
    float *sc = (float *)malloc((size_t)64 * nb * sizeof(float));
    if (!q || !sc) { free(q); free(sc); free(dst); return NULL; }
    for (int panel = 0; panel < rows / 64; ++panel) {
        for (int r = 0; r < 64; ++r) {
            const int row = panel * 64 + r;
            for (int b = 0; b < nb; ++b) {
                if (type == GLM53F_NATIVE_Q8_0R) {
                    memcpy(q + (size_t)r * cols + b * 32, src + (size_t)row * rb_r + b * 32, 32);
                    memcpy(&sc[(size_t)r * nb + b], src + (size_t)row * rb_r + cols + (size_t)b * 4, 4);
                } else if (type == GLM53F_GGML_Q8_0) {
                    const uint8_t *blk = src + (size_t)row * rb_q + (size_t)b * 34;
                    _Float16 h; memcpy(&h, blk, 2);
                    sc[(size_t)r * nb + b] = (float)h;
                    memcpy(q + (size_t)r * cols + b * 32, blk + 2, 32);
                } else {
                    const uint8_t *blk = src + (size_t)(row / 16) * ((size_t)nb * 576) + (size_t)b * 576;
                    const int rr = row % 16;
                    for (int j = 0; j < 8; ++j) memcpy(q + (size_t)r * cols + b * 32 + 4 * j, blk + j * 64 + rr * 4, 4);
                    memcpy(&sc[(size_t)r * nb + b], blk + 512 + rr * 4, 4);
                }
            }
        }
        gk_pack_panel64(32, dst + (size_t)panel * pbytes, q, sc, 64, cols);
    }
    free(q); free(sc);
    return dst;
}
#endif
