/*
 * tensor_dump.c - VLMD writer + reader.
 */
#include "tensor_dump.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

size_t vlmd_dtype_size(int dtype) {
    switch (dtype) {
        case VLMD_F32:  return 4;
        case VLMD_BF16: return 2;
        case VLMD_F16:  return 2;
        case VLMD_I32:  return 4;
        default:        return 0;
    }
}

size_t vlmd_numel(const vlmd_header *h) {
    if (!h || h->ndim == 0) return 0;
    size_t n = 1;
    for (uint32_t i = 0; i < h->ndim; i++) n *= (size_t)h->dims[i];
    return n;
}

static int mkdir_p(const char *path) {
    if (!path || !*path) return -1;
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    /* Single-level mkdir is enough for our use (caller passes a flat dir). */
    if (mkdir(path, 0755) == 0) return 0;
    return errno == EEXIST ? 0 : -1;
}

int vlmd_writer_open(vlmd_writer *w, const char *dir) {
    if (!w) return -1;
    memset(w, 0, sizeof(*w));
    if (!dir || !*dir) {
        w->enabled = 0;
        return 0;
    }
    if (mkdir_p(dir) != 0) {
        fprintf(stderr, "vlmd: cannot create dir '%s': %s\n", dir, strerror(errno));
        return -1;
    }
    strncpy(w->dir, dir, sizeof(w->dir) - 1);
    char manpath[600];
    snprintf(manpath, sizeof(manpath), "%s/manifest.txt", w->dir);
    w->manifest = fopen(manpath, "w");
    if (!w->manifest) {
        fprintf(stderr, "vlmd: cannot open manifest '%s': %s\n", manpath, strerror(errno));
        return -1;
    }
    fprintf(w->manifest, "# filename name layer dtype ndim dims...\n");
    w->enabled = 1;
    return 0;
}

void vlmd_writer_close(vlmd_writer *w) {
    if (!w) return;
    if (w->manifest) { fclose(w->manifest); w->manifest = NULL; }
    w->enabled = 0;
}

static const char *dtype_str(int dtype) {
    switch (dtype) {
        case VLMD_F32:  return "f32";
        case VLMD_BF16: return "bf16";
        case VLMD_F16:  return "f16";
        case VLMD_I32:  return "i32";
        default:        return "?";
    }
}

int vlmd_dump(vlmd_writer *w,
              const char *name,
              int layer,
              int dtype,
              int ndim,
              const uint32_t *dims,
              const void *data)
{
    if (!w || !w->enabled) return 0;
    if (!name || !dims || !data) return -1;
    if (ndim < 1 || ndim > VLMD_NDIM_MAX) return -1;
    size_t esz = vlmd_dtype_size(dtype);
    if (esz == 0) return -1;

    size_t numel = 1;
    for (int i = 0; i < ndim; i++) numel *= (size_t)dims[i];
    size_t bytes = numel * esz;

    char fname[VLMD_NAME_MAX + 32];
    if (layer >= 0) snprintf(fname, sizeof(fname), "L%02d_%s.vlmd", layer, name);
    else            snprintf(fname, sizeof(fname), "%s.vlmd", name);

    char fpath[1024];
    snprintf(fpath, sizeof(fpath), "%s/%s", w->dir, fname);
    FILE *fp = fopen(fpath, "wb");
    if (!fp) {
        fprintf(stderr, "vlmd: cannot open '%s' for write: %s\n", fpath, strerror(errno));
        return -1;
    }

    uint8_t hdr[VLMD_HDR_BYTES];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr + 0, VLMD_MAGIC, 4);
    uint32_t version = VLMD_VERSION;
    uint32_t dt = (uint32_t)dtype;
    uint32_t nd = (uint32_t)ndim;
    memcpy(hdr + 4,  &version, 4);
    memcpy(hdr + 8,  &dt,      4);
    memcpy(hdr + 12, &nd,      4);
    uint32_t dvals[VLMD_NDIM_MAX] = {0};
    for (int i = 0; i < ndim; i++) dvals[i] = dims[i];
    memcpy(hdr + 16, dvals, sizeof(dvals));        /* 32 bytes */
    /* name @ offset 16 + 32 = 48 */
    strncpy((char *)(hdr + 48), name, VLMD_NAME_MAX - 1);
    /* reserved[32] @ offset 48 + 64 = 112 (already zeroed) */

    if (fwrite(hdr, 1, VLMD_HDR_BYTES, fp) != VLMD_HDR_BYTES) goto write_err;
    if (bytes > 0 && fwrite(data, 1, bytes, fp) != bytes) goto write_err;
    fclose(fp);

    if (w->manifest) {
        fprintf(w->manifest, "%s %s %d %s %d",
                fname, name, layer, dtype_str(dtype), ndim);
        for (int i = 0; i < ndim; i++) fprintf(w->manifest, " %u", dims[i]);
        fprintf(w->manifest, "\n");
        fflush(w->manifest);
    }
    return 0;

write_err:
    fprintf(stderr, "vlmd: write failed for '%s': %s\n", fpath, strerror(errno));
    fclose(fp);
    return -1;
}

int vlmd_dump_f32_2d(vlmd_writer *w, const char *name, int layer,
                     int rows, int cols, const float *data) {
    uint32_t d[2] = { (uint32_t)rows, (uint32_t)cols };
    return vlmd_dump(w, name, layer, VLMD_F32, 2, d, data);
}

int vlmd_dump_f32_3d(vlmd_writer *w, const char *name, int layer,
                     int d0, int d1, int d2, const float *data) {
    uint32_t d[3] = { (uint32_t)d0, (uint32_t)d1, (uint32_t)d2 };
    return vlmd_dump(w, name, layer, VLMD_F32, 3, d, data);
}

int vlmd_dump_f32_4d(vlmd_writer *w, const char *name, int layer,
                     int d0, int d1, int d2, int d3, const float *data) {
    uint32_t d[4] = { (uint32_t)d0, (uint32_t)d1, (uint32_t)d2, (uint32_t)d3 };
    return vlmd_dump(w, name, layer, VLMD_F32, 4, d, data);
}

int vlmd_read(const char *path,
              vlmd_header *hdr,
              void **out_data,
              size_t *out_bytes)
{
    if (out_data) *out_data = NULL;
    if (out_bytes) *out_bytes = 0;
    if (!path || !hdr) return -1;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "vlmd: cannot open '%s': %s\n", path, strerror(errno));
        return -1;
    }

    uint8_t buf[VLMD_HDR_BYTES];
    if (fread(buf, 1, VLMD_HDR_BYTES, fp) != VLMD_HDR_BYTES) {
        fprintf(stderr, "vlmd: short header in '%s'\n", path);
        fclose(fp);
        return -1;
    }

    memset(hdr, 0, sizeof(*hdr));
    memcpy(hdr->magic, buf + 0, 4);
    memcpy(&hdr->version, buf + 4,  4);
    memcpy(&hdr->dtype,   buf + 8,  4);
    memcpy(&hdr->ndim,    buf + 12, 4);
    memcpy(hdr->dims,     buf + 16, 32);
    memcpy(hdr->name,     buf + 48, VLMD_NAME_MAX);
    memcpy(hdr->reserved, buf + 112, 32);

    if (memcmp(hdr->magic, VLMD_MAGIC, 4) != 0) {
        fprintf(stderr, "vlmd: bad magic in '%s'\n", path);
        fclose(fp);
        return -1;
    }
    if (hdr->ndim < 1 || hdr->ndim > VLMD_NDIM_MAX) {
        fprintf(stderr, "vlmd: bad ndim=%u in '%s'\n", hdr->ndim, path);
        fclose(fp);
        return -1;
    }

    size_t numel = vlmd_numel(hdr);
    size_t esz   = vlmd_dtype_size((int)hdr->dtype);
    size_t bytes = numel * esz;

    if (out_data) {
        *out_data = malloc(bytes ? bytes : 1);
        if (!*out_data) { fclose(fp); return -1; }
        if (bytes > 0 && fread(*out_data, 1, bytes, fp) != bytes) {
            fprintf(stderr, "vlmd: short data in '%s'\n", path);
            free(*out_data); *out_data = NULL;
            fclose(fp);
            return -1;
        }
        if (out_bytes) *out_bytes = bytes;
    }
    fclose(fp);
    return 0;
}
