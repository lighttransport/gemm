/* Compact, bounded boundary trace for GLM-5.3F diagnosis.
 *
 * The writer emits one record at a time and never owns more than the current
 * boundary vector.  Full values are optional; callers can emit only the
 * digest/statistics for long prompt runs.
 */
#ifndef GLM53F_TRACE_H
#define GLM53F_TRACE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define GLM53F_TRACE_MAGIC UINT64_C(0x474c4d3533464254) /* GLM53FBT */
#define GLM53F_TRACE_VERSION 1u

enum glm53f_trace_flags {
    GLM53F_TRACE_VALUES = 1u << 0,
    GLM53F_TRACE_ROUTER = 1u << 1,
    GLM53F_TRACE_STATE = 1u << 2
};

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t layer;
    uint32_t token;
    uint32_t boundary;
    uint32_t dtype;       /* 1 = f32, 2 = bf16, 3 = i32 */
    uint32_t count;
    uint64_t payload_bytes;
    uint64_t digest;
    double sum_sq;
    float min_value;
    float max_value;
} glm53f_trace_record;

static inline uint64_t glm53f_trace_hash(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    while (bytes--) { h ^= *p++; h *= UINT64_C(1099511628211); }
    return h;
}

static inline int glm53f_trace_open(FILE **out, const char *path,
                                    uint32_t flags, uint32_t hidden) {
    uint64_t header[4] = { GLM53F_TRACE_MAGIC, GLM53F_TRACE_VERSION,
                           flags, hidden };
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(header, sizeof(header), 1, f) != 1) {
        if (f) fclose(f);
        return -1;
    }
    *out = f;
    return 0;
}

static inline int glm53f_trace_f32(FILE *f, uint32_t layer, uint32_t token,
                                   uint32_t boundary, const float *v,
                                   uint32_t count, int values) {
    glm53f_trace_record r;
    memset(&r, 0, sizeof(r));
    r.magic = GLM53F_TRACE_MAGIC;
    r.version = GLM53F_TRACE_VERSION;
    r.flags = values ? GLM53F_TRACE_VALUES : 0;
    r.layer = layer; r.token = token; r.boundary = boundary;
    r.dtype = 1; r.count = count;
    r.payload_bytes = values ? (uint64_t)count * sizeof(float) : 0;
    r.digest = glm53f_trace_hash(v, (size_t)count * sizeof(float));
    r.min_value = count ? v[0] : 0.0f;
    r.max_value = r.min_value;
    for (uint32_t i = 0; i < count; ++i) {
        if (!isfinite(v[i])) return -1;
        r.sum_sq += (double)v[i] * v[i];
        if (v[i] < r.min_value) r.min_value = v[i];
        if (v[i] > r.max_value) r.max_value = v[i];
    }
    if (fwrite(&r, sizeof(r), 1, f) != 1) return -1;
    if (values && fwrite(v, sizeof(float), count, f) != count) return -1;
    return 0;
}

static inline int glm53f_trace_close(FILE *f) {
    if (!f) return -1;
    return fflush(f) || fclose(f) ? -1 : 0;
}

#endif
