#ifndef DS41F_TENSOR_H
#define DS41F_TENSOR_H
#include <stddef.h>
/* Load an exact-sized staged tensor into anonymous memory, dropping source
 * cache after each bounded chunk. The caller owns the returned allocation. */
int ds41f_tensor_load(const char *directory, const char *name, size_t bytes,
                      void **data);
#endif
