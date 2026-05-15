/*
 * tensor_dump.h - per-stage tensor dump format for vlm runner validation.
 *
 * Layout (little-endian, packed):
 *   char   magic[4]   = "VLMD"
 *   u32    version    = 1
 *   u32    dtype      (0=f32, 1=bf16, 2=f16, 3=i32)
 *   u32    ndim       (1..8)
 *   u32    dims[8]    (unused trailing slots = 0)
 *   char   name[64]   (NUL-padded)
 *   u8     reserved[32]
 *   raw element bytes (ndim product * sizeof(dtype))
 *
 * Header is fixed size = 4 + 4 + 4 + 4 + 32 + 64 + 32 = 144 bytes.
 *
 * Sidecar manifest: <dir>/manifest.txt
 *   one line per tensor: "<filename> <name> <layer> <dtype> <ndim> d0 d1 ..."
 */
#ifndef TENSOR_DUMP_H
#define TENSOR_DUMP_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VLMD_MAGIC      "VLMD"
#define VLMD_VERSION    1u
#define VLMD_HDR_BYTES  144
#define VLMD_NAME_MAX   64
#define VLMD_NDIM_MAX   8

enum vlmd_dtype {
    VLMD_F32  = 0,
    VLMD_BF16 = 1,
    VLMD_F16  = 2,
    VLMD_I32  = 3,
};

typedef struct vlmd_writer {
    char  dir[512];
    FILE *manifest;
    int   enabled;
} vlmd_writer;

/* dir == NULL or empty disables dumping (all calls are no-ops). */
int  vlmd_writer_open (vlmd_writer *w, const char *dir);
void vlmd_writer_close(vlmd_writer *w);

/* Write a single tensor. `layer` = -1 for stage-global tensors. dtype is the
 * dtype enum; element count is product(dims[0..ndim-1]). Returns 0 on success. */
int  vlmd_dump(vlmd_writer *w,
               const char *name,
               int         layer,
               int         dtype,
               int         ndim,
               const uint32_t *dims,
               const void *data);

/* Convenience helpers (most common shapes / dtypes). */
int vlmd_dump_f32_2d(vlmd_writer *w, const char *name, int layer,
                     int rows, int cols, const float *data);
int vlmd_dump_f32_3d(vlmd_writer *w, const char *name, int layer,
                     int d0, int d1, int d2, const float *data);
int vlmd_dump_f32_4d(vlmd_writer *w, const char *name, int layer,
                     int d0, int d1, int d2, int d3, const float *data);

/* ── Reader API (used by tensor_diff and any consumer) ── */

typedef struct vlmd_header {
    char     magic[4];
    uint32_t version;
    uint32_t dtype;
    uint32_t ndim;
    uint32_t dims[VLMD_NDIM_MAX];
    char     name[VLMD_NAME_MAX];
    uint8_t  reserved[32];
} vlmd_header;

/* Reads header from `path` and (optionally) the raw bytes into a malloc'd buffer.
 * If `out_data` is non-NULL the caller must free(*out_data). On error returns
 * non-zero and leaves *out_data NULL. */
int vlmd_read(const char *path,
              vlmd_header *hdr,
              void **out_data,
              size_t *out_bytes);

size_t vlmd_dtype_size(int dtype);
size_t vlmd_numel(const vlmd_header *h);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_DUMP_H */
