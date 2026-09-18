#ifndef GLM53F_STAGE_ARTIFACT_H
#define GLM53F_STAGE_ARTIFACT_H

/* Portable, bounded stage artifacts.  Payloads are little-endian native
 * float32/int32 streams; the JSONL manifest is the interchange metadata. */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GLM53F_STAGE_ARTIFACT_VERSION 1u

typedef struct {
    char root[4096];
    char manifest_path[4096];
    FILE *manifest;
    int failed;
} glm53f_stage_artifact;

static inline uint64_t glm53f_stage_hash(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *) data;
    uint64_t h = UINT64_C(1469598103934665603);
    while (bytes--) { h ^= *p++; h *= UINT64_C(1099511628211); }
    return h;
}

static inline int glm53f_stage_mkdir(const char *path) {
    char tmp[4096];
    size_t n = strlen(path);
    if (!n || n >= sizeof(tmp)) return -1;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
        *p = '/';
    }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}

static inline int glm53f_stage_artifact_open(glm53f_stage_artifact *a,
        const char *root, const char *producer, const char *model,
        const char *prompt, unsigned layer, unsigned token_start,
        unsigned token_count) {
    if (!a || !root || !producer || !model || !prompt) return -1;
    memset(a, 0, sizeof(*a));
    if (snprintf(a->root, sizeof(a->root), "%s", root) >= (int) sizeof(a->root) ||
        glm53f_stage_mkdir(a->root) ||
        snprintf(a->manifest_path, sizeof(a->manifest_path), "%s/manifest.jsonl", a->root) >= (int) sizeof(a->manifest_path))
        return -1;
    /* A directory is one immutable run.  Re-running into it must not leave
     * duplicate JSONL records that make a later comparison ambiguous. */
    a->manifest = fopen(a->manifest_path, "w");
    if (!a->manifest) return -1;
    if (fprintf(a->manifest,
            "{\"kind\":\"header\",\"version\":%u,\"producer\":\"%s\","
            "\"model\":\"%s\",\"prompt\":\"%s\",\"layer\":%u,"
            "\"token_start\":%u,\"token_count\":%u}\n",
            GLM53F_STAGE_ARTIFACT_VERSION, producer, model, prompt, layer,
            token_start, token_count) < 0) a->failed = 1;
    return a->failed ? -1 : 0;
}

static inline int glm53f_stage_write_record(glm53f_stage_artifact *a,
        const char *stage, const char *name, const char *dtype,
        const void *data, size_t elem_size, size_t count,
        const size_t *shape, size_t nshape) {
    if (!a || !a->manifest || !stage || !name || !dtype || !data || !count ||
        nshape > 8) return -1;
    char leaf[256], path[4096];
    size_t used = 0;
    for (const char *p = name; *p && used + 1 < sizeof(leaf); ++p)
        leaf[used++] = (*p == '/' || *p == ' ' || *p == '\t') ? '_' : *p;
    leaf[used] = '\0';
    if (snprintf(path, sizeof(path), "%s/%s_%s.%s", a->root, stage, leaf, dtype) >= (int) sizeof(path))
        return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = fwrite(data, elem_size, count, f) == count && !fflush(f) &&
             !fsync(fileno(f)) && !fclose(f);
    if (!ok) { fclose(f); a->failed = 1; return -1; }
    double ss = 0.0;
    float lo = 0.0f, hi = 0.0f;
    if (!strcmp(dtype, "f32")) {
        const float *v = (const float *) data;
        lo = hi = v[0];
        for (size_t i = 0; i < count; ++i) {
            if (!isfinite(v[i])) { a->failed = 1; return -1; }
            ss += (double) v[i] * v[i];
            if (v[i] < lo) lo = v[i];
            if (v[i] > hi) hi = v[i];
        }
    }
    uint64_t hash = glm53f_stage_hash(data, elem_size * count);
    if (fprintf(a->manifest, "{\"kind\":\"tensor\",\"stage\":\"%s\","
            "\"name\":\"%s\",\"dtype\":\"%s\",\"payload\":\"%s\","
            "\"count\":%zu,\"bytes\":%zu,\"hash_fnv1a\":\"%016llx\","
            "\"rms\":%.9g,\"min\":%.9g,\"max\":%.9g,\"shape\":[",
            stage, name, dtype, strrchr(path, '/') + 1, count, elem_size * count,
            (unsigned long long) hash, count ? sqrt(ss / count) : 0.0,
            (double) lo, (double) hi) < 0) a->failed = 1;
    for (size_t i = 0; i < nshape; ++i)
        if (fprintf(a->manifest, "%s%zu", i ? "," : "", shape[i]) < 0) a->failed = 1;
    if (fprintf(a->manifest, "]}\n") < 0 || fflush(a->manifest) || fsync(fileno(a->manifest)))
        a->failed = 1;
    return a->failed ? -1 : 0;
}

static inline int glm53f_stage_write_f32(glm53f_stage_artifact *a,
        const char *stage, const char *name, const float *data,
        size_t count, const size_t *shape, size_t nshape) {
    return glm53f_stage_write_record(a, stage, name, "f32", data, sizeof(float),
                                     count, shape, nshape);
}

static inline int glm53f_stage_write_i32(glm53f_stage_artifact *a,
        const char *stage, const char *name, const int32_t *data,
        size_t count, const size_t *shape, size_t nshape) {
    return glm53f_stage_write_record(a, stage, name, "i32", data, sizeof(int32_t),
                                     count, shape, nshape);
}

static inline int glm53f_stage_artifact_close(glm53f_stage_artifact *a) {
    if (!a || !a->manifest) return -1;
    int rc = fflush(a->manifest) || fsync(fileno(a->manifest)) || fclose(a->manifest);
    a->manifest = NULL;
    return rc || a->failed ? -1 : 0;
}

#endif
