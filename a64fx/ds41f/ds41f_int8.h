#ifndef DS41F_INT8_H
#define DS41F_INT8_H
#include <stddef.h>
#include <stdint.h>

/* Four rows, K blocks, then 32-column chunks containing four contiguous
 * 32-byte rows. Scales are per row and K block; padding rows are zero. */
typedef struct {
    size_t rows, cols, block, bytes;
    int8_t *weight;
    float *scale;
} ds41f_int8;

int ds41f_int8_from_fp8(ds41f_int8 *out, const uint8_t *weight,
                      const uint8_t *scale, size_t rows, size_t cols, size_t block);
void ds41f_int8_free(ds41f_int8 *q);
int ds41f_int8_matvec(float *out, const ds41f_int8 *q, const float *x,
                     size_t group_rows, int reference);
#endif
