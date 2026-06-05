/* Host-side fp16/bf16 weight cache helpers for SAM3D CUDA DiT runners. */
#ifndef CUDA_SAM3D_MMA_CACHE_H_
#define CUDA_SAM3D_MMA_CACHE_H_

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../cuew.h"

enum {
    CS3D_MMA_PREC_F16  = 1,
    CS3D_MMA_PREC_BF16 = 2,
};

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t precision;
    uint32_t n_elem;
    uint32_t reserved;
} cs3d_mma_cache_header;

static int cs3d_mma_precision_id(const char *precision)
{
    if (precision && !strcmp(precision, "bf16")) return CS3D_MMA_PREC_BF16;
    if (precision && !strcmp(precision, "fp16")) return CS3D_MMA_PREC_F16;
    return 0;
}

static uint16_t cs3d_mma_f32_to_bf16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t r = 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t)((x + r) >> 16);
}

static uint16_t cs3d_mma_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007fffffu;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;

    if (exp >= 31) {
        if ((x & 0x7fffffffu) > 0x7f800000u)
            return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_m = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_m & 1u))) half_m++;
        return (uint16_t)(sign | half_m);
    }

    uint32_t half_m = mant >> 13;
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) {
        half_m++;
        if (half_m == 0x400u) {
            half_m = 0;
            exp++;
            if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_m);
}

static int cs3d_mma_cache_root(const char *cache_dir, char *out, size_t out_sz)
{
    if (!cache_dir || !cache_dir[0] || !out || out_sz == 0) return -1;
    if (getenv("SAM3D_MMA_WEIGHT_CACHE") &&
        getenv("SAM3D_MMA_WEIGHT_CACHE")[0] == '0')
        return -1;
    int n = snprintf(out, out_sz, "%s/cuda_mma_cache", cache_dir);
    if (n < 0 || (size_t)n >= out_sz) return -1;
    if (mkdir(out, 0775) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int cs3d_mma_cache_path(const char *cache_dir, const char *tag,
                               int precision_id, char *out, size_t out_sz)
{
    char root[1024];
    if (cs3d_mma_cache_root(cache_dir, root, sizeof(root)) != 0) return -1;
    const char *p = (precision_id == CS3D_MMA_PREC_BF16) ? "bf16" : "fp16";
    int n = snprintf(out, out_sz, "%s/%s.%s.u16.bin", root, tag, p);
    return (n >= 0 && (size_t)n < out_sz) ? 0 : -1;
}

static uint16_t *cs3d_mma_cache_read(const char *path, int precision_id, int n_elem)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    cs3d_mma_cache_header h;
    if (fread(&h, sizeof(h), 1, fp) != 1 ||
        memcmp(h.magic, "CS3DMMA", 7) != 0 ||
        h.version != 1 ||
        h.precision != (uint32_t)precision_id ||
        h.n_elem != (uint32_t)n_elem) {
        fclose(fp);
        return NULL;
    }
    uint16_t *buf = (uint16_t *)malloc((size_t)n_elem * sizeof(uint16_t));
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t nr = fread(buf, sizeof(uint16_t), (size_t)n_elem, fp);
    fclose(fp);
    if (nr != (size_t)n_elem) {
        free(buf);
        return NULL;
    }
    return buf;
}

static void cs3d_mma_cache_write(const char *path, int precision_id,
                                 const uint16_t *buf, int n_elem)
{
    if (!path || !buf || n_elem <= 0) return;
    char tmp[1200];
    int n = snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    if (n < 0 || (size_t)n >= sizeof(tmp)) return;
    FILE *fp = fopen(tmp, "wb");
    if (!fp) return;
    cs3d_mma_cache_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "CS3DMMA", 7);
    h.version = 1;
    h.precision = (uint32_t)precision_id;
    h.n_elem = (uint32_t)n_elem;
    int ok = fwrite(&h, sizeof(h), 1, fp) == 1 &&
             fwrite(buf, sizeof(uint16_t), (size_t)n_elem, fp) == (size_t)n_elem;
    fclose(fp);
    if (ok) rename(tmp, path);
    else unlink(tmp);
}

static uint16_t *cs3d_mma_convert_u16(const float *src, int n_elem,
                                      int precision_id)
{
    if (!src || n_elem <= 0) return NULL;
    uint16_t *buf = (uint16_t *)malloc((size_t)n_elem * sizeof(uint16_t));
    if (!buf) return NULL;
    if (precision_id == CS3D_MMA_PREC_BF16) {
        for (int i = 0; i < n_elem; i++) buf[i] = cs3d_mma_f32_to_bf16(src[i]);
    } else {
        for (int i = 0; i < n_elem; i++) buf[i] = cs3d_mma_f32_to_f16(src[i]);
    }
    return buf;
}

static int cs3d_mma_upload_weight_u16(const char *cache_dir,
                                      const char *tag,
                                      const char *precision,
                                      const float *fallback_f32,
                                      int n_elem,
                                      CUdeviceptr *out_d,
                                      size_t *out_bytes,
                                      int verbose)
{
    if (!out_d) return -1;
    *out_d = 0;
    int precision_id = cs3d_mma_precision_id(precision);
    if (!precision_id || n_elem <= 0) return 0;

    char path[1200];
    uint16_t *buf = NULL;
    int have_path = (cs3d_mma_cache_path(cache_dir, tag, precision_id,
                                         path, sizeof(path)) == 0);
    if (have_path) buf = cs3d_mma_cache_read(path, precision_id, n_elem);
    if (!buf) {
        buf = cs3d_mma_convert_u16(fallback_f32, n_elem, precision_id);
        if (!buf) return -1;
        if (have_path) cs3d_mma_cache_write(path, precision_id, buf, n_elem);
    } else if (verbose >= 2) {
        fprintf(stderr, "sam3d_mma_cache: loaded %s\n", path);
    }

    size_t nb = (size_t)n_elem * sizeof(uint16_t);
    CUdeviceptr d = cu_upload_raw(buf, nb);
    free(buf);
    if (!d) return -1;
    *out_d = d;
    if (out_bytes) *out_bytes += nb;
    return 0;
}

#endif /* CUDA_SAM3D_MMA_CACHE_H_ */
