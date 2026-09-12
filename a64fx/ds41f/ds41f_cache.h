#ifndef DS41F_CACHE_H
#define DS41F_CACHE_H
#include <stddef.h>
#include <stdint.h>
/* Set before starting workers; the scalar control remains the default. */
void ds41f_set_cache_sve(int enabled);
int ds41f_get_cache_sve(void);
/* Packed row: adjacent E2M1 nibbles then scale bytes. */
int ds41f_fp4_pack(uint8_t *out,const float *x,size_t dim,size_t group,int e4scale);
int ds41f_fp4_unpack(float *out,const uint8_t *row,size_t dim,size_t group,int e4scale);
/* Return top-k finite scores (ties lower position), sorted by position. */
size_t ds41f_select_topk(const float *scores,size_t n,size_t k,int *indices);
#endif
