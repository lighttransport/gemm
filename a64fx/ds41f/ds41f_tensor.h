#ifndef DS41F_TENSOR_H
#define DS41F_TENSOR_H
#include <stddef.h>
/* Load an exact-sized staged tensor into anonymous memory, dropping source
 * cache after each bounded chunk. The caller owns the returned allocation. */
int ds41f_tensor_load(const char *directory, const char *name, size_t bytes,
                      void **data);
/* Pair with tensor_free_local using the same bytes/fresh_pages values.
 * Fresh anonymous allocations bypass the platform malloc pool on Linux LP64. */
int ds41f_tensor_load_local(const char *directory,const char *name,size_t bytes,
                           void **data,int fresh_pages);
void ds41f_tensor_free_local(void *data,size_t bytes,int fresh_pages);
#endif
