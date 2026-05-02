#include "turboquant_cpu.h"
#include "turboquant_cpu_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

const float tq3_centroids[8] = {
    -0.1900314689f, -0.1187858144f, -0.0668221228f, -0.0216554848f,
     0.0216554848f,  0.0668221228f,  0.1187858144f,  0.1900314689f,
};

const float tq3_thresholds[7] = {
    -0.1544086412f, -0.0928039686f, -0.0442388038f, 0.0f,
     0.0442388038f,  0.0928039686f,  0.1544086412f,
};

static const float tq3_inv_sqrt_128 = 0.08838834764831845f;

uint64_t tq3_splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float tq3_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

uint16_t tq3_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127;
    uint32_t mant = x & 0x7fffffu;
    if (exp > 15) return (uint16_t)(sign | 0x7c00u);
    if (exp < -14) {
        if (exp < -24) return sign;
        mant |= 0x800000u;
        mant >>= (uint32_t)(-1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }
    return (uint16_t)(sign | ((uint16_t)(exp + 15) << 10) | (uint16_t)(mant >> 13));
}

size_t tq3_num_blocks(int n) {
    if (n <= 0 || (n % TQ3_BLOCK_SIZE) != 0) return 0;
    return (size_t)n / TQ3_BLOCK_SIZE;
}

size_t tq3_row_bytes(int n) {
    size_t nb = tq3_num_blocks(n);
    return nb ? nb * sizeof(tq3_block) : 0;
}

static uint8_t sign_bit(uint64_t seed, uint64_t block_index, int plane, int i) {
    uint64_t s = seed ^ (0xd1b54a32d192ed03ull * (block_index + 1));
    s ^= 0x9e3779b97f4a7c15ull * (uint64_t)(plane + 17);
    s ^= (uint64_t)i * 0xbf58476d1ce4e5b9ull;
    return (uint8_t)(tq3_splitmix64(&s) >> 63);
}

static void fwht128(float v[TQ3_BLOCK_SIZE]) {
    for (int h = 1; h < TQ3_BLOCK_SIZE; h <<= 1) {
        for (int i = 0; i < TQ3_BLOCK_SIZE; i += h << 1) {
            for (int j = 0; j < h; j++) {
                float a = v[i + j];
                float b = v[i + j + h];
                v[i + j] = a + b;
                v[i + j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) v[i] *= tq3_inv_sqrt_128;
}

void tq3_forward_rotate(float y[TQ3_BLOCK_SIZE], const float x[TQ3_BLOCK_SIZE],
                        uint64_t seed, uint64_t block_index) {
    float tmp[TQ3_BLOCK_SIZE];
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        tmp[i] = sign_bit(seed, block_index, 0, i) ? -x[i] : x[i];
    }
    fwht128(tmp);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        y[i] = sign_bit(seed, block_index, 1, i) ? -tmp[i] : tmp[i];
    }
}

void tq3_inverse_rotate(float x[TQ3_BLOCK_SIZE], const float y[TQ3_BLOCK_SIZE],
                        uint64_t seed, uint64_t block_index) {
    float tmp[TQ3_BLOCK_SIZE];
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        tmp[i] = sign_bit(seed, block_index, 1, i) ? -y[i] : y[i];
    }
    fwht128(tmp);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        x[i] = sign_bit(seed, block_index, 0, i) ? -tmp[i] : tmp[i];
    }
}

void tq3_pack_indices(tq3_block *blk, const uint8_t idx[TQ3_BLOCK_SIZE]) {
    memset(blk->qs, 0, sizeof(blk->qs));
    int bit = 0;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++, bit += 3) {
        uint32_t v = (uint32_t)(idx[i] & 7u);
        int byte = bit >> 3;
        int shift = bit & 7;
        uint32_t w = v << shift;
        blk->qs[byte] |= (uint8_t)w;
        if (shift > 5) blk->qs[byte + 1] |= (uint8_t)(w >> 8);
    }
}

void tq3_unpack_indices(uint8_t idx[TQ3_BLOCK_SIZE], const tq3_block *blk) {
    int bit = 0;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++, bit += 3) {
        int byte = bit >> 3;
        int shift = bit & 7;
        uint32_t w = blk->qs[byte];
        if (byte + 1 < TQ3_PACKED_BYTES) w |= (uint32_t)blk->qs[byte + 1] << 8;
        idx[i] = (uint8_t)((w >> shift) & 7u);
    }
}

static uint8_t nearest_centroid(float v) {
    uint8_t idx = 0;
    idx += (uint8_t)(v > tq3_thresholds[0]);
    idx += (uint8_t)(v > tq3_thresholds[1]);
    idx += (uint8_t)(v > tq3_thresholds[2]);
    idx += (uint8_t)(v > tq3_thresholds[3]);
    idx += (uint8_t)(v > tq3_thresholds[4]);
    idx += (uint8_t)(v > tq3_thresholds[5]);
    idx += (uint8_t)(v > tq3_thresholds[6]);
    return idx;
}

int tq3_quantize_row_f32(tq3_block *dst, const float *src, int n, uint64_t seed) {
    size_t nb = tq3_num_blocks(n);
    if (!dst || !src || nb == 0) return -1;
    for (size_t b = 0; b < nb; b++) {
        const float *x = src + b * TQ3_BLOCK_SIZE;
        float norm2 = 0.0f;
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) norm2 += x[i] * x[i];
        if (norm2 <= 0.0f) {
            dst[b].scale_f16 = 0;
            dst[b].reserved = 0;
            memset(dst[b].qs, 0, sizeof(dst[b].qs));
            continue;
        }

        float inv_norm = 1.0f / sqrtf(norm2);
        float normalized[TQ3_BLOCK_SIZE];
        float rotated[TQ3_BLOCK_SIZE];
        uint8_t idx[TQ3_BLOCK_SIZE];
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) normalized[i] = x[i] * inv_norm;
        tq3_forward_rotate(rotated, normalized, seed, b);

        float recon_norm2 = 0.0f;
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
            idx[i] = nearest_centroid(rotated[i]);
            float c = tq3_centroids[idx[i]];
            recon_norm2 += c * c;
        }
        float scale = sqrtf(norm2) / fmaxf(sqrtf(recon_norm2), 1.0e-20f);
        dst[b].scale_f16 = tq3_f32_to_f16(scale);
        dst[b].reserved = 0;
        tq3_pack_indices(&dst[b], idx);
    }
    return 0;
}

int tq3_dequantize_row_f32(float *dst, const tq3_block *src, int n, uint64_t seed) {
    size_t nb = tq3_num_blocks(n);
    if (!dst || !src || nb == 0) return -1;
    for (size_t b = 0; b < nb; b++) {
        uint8_t idx[TQ3_BLOCK_SIZE];
        float rotated[TQ3_BLOCK_SIZE];
        float scale = tq3_f16_to_f32(src[b].scale_f16);
        tq3_unpack_indices(idx, &src[b]);
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) rotated[i] = tq3_centroids[idx[i]] * scale;
        tq3_inverse_rotate(dst + b * TQ3_BLOCK_SIZE, rotated, seed, b);
    }
    return 0;
}

float tq3_dot_block_scalar(const tq3_block *blk, const float *x,
                           uint64_t seed, uint64_t block_index) {
    uint8_t idx[TQ3_BLOCK_SIZE];
    float xrot[TQ3_BLOCK_SIZE];
    float scale = tq3_f16_to_f32(blk->scale_f16);
    tq3_unpack_indices(idx, blk);
    tq3_forward_rotate(xrot, x, seed, block_index);
    float acc = 0.0f;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        acc += tq3_centroids[idx[i]] * scale * xrot[i];
    }
    return acc;
}

float tq3_dot_row_f32(const tq3_block *qrow, const float *x, int n, uint64_t seed) {
    size_t nb = tq3_num_blocks(n);
    if (!qrow || !x || nb == 0) return 0.0f;
    float acc = 0.0f;
    if (tq3_x86_has_avx2()) {
        for (size_t b = 0; b < nb; b++) {
            acc += tq3_dot_block_avx2(&qrow[b], x + b * TQ3_BLOCK_SIZE, seed, b);
        }
    } else if (tq3_x86_has_sse2()) {
        for (size_t b = 0; b < nb; b++) {
            acc += tq3_dot_block_sse2(&qrow[b], x + b * TQ3_BLOCK_SIZE, seed, b);
        }
    } else {
        for (size_t b = 0; b < nb; b++) {
            acc += tq3_dot_block_scalar(&qrow[b], x + b * TQ3_BLOCK_SIZE, seed, b);
        }
    }
    return acc;
}

const char *tq3_cpu_backend(void) {
    if (tq3_x86_has_avx2()) return "avx2";
    if (tq3_x86_has_sse2()) return "sse2";
    return "scalar";
}
