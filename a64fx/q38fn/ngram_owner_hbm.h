/* Rank-owned resident HBM store for Qwen3.8 packed n-gram shards. */
#ifndef Q38FN_NGRAM_OWNER_HBM_H
#define Q38FN_NGRAM_OWNER_HBM_H

#include "../../common/glm53f_safetensors.h"
#include "../../common/q38fn_arch.h"
#include <stdint.h>

typedef struct q38fn_ngram_owner_hbm q38fn_ngram_owner_hbm;

int q38fn_ngram_owner_hbm_open(q38fn_ngram_owner_hbm **out,
                               const char *model_dir, uint32_t rank,
                               uint32_t nranks);
void q38fn_ngram_owner_hbm_close(q38fn_ngram_owner_hbm *h);
int q38fn_ngram_owner_hbm_read(void *opaque, uint32_t shard,
                               uint64_t first_row, uint32_t rows, void *dst);

/* Validate the resident image without involving uTofu.  Returns a stable
 * sample checksum and catches rank/shard placement errors before transport
 * diagnostics are attempted. */
int q38fn_ngram_owner_hbm_validate(const q38fn_ngram_owner_hbm *h,
                                   uint64_t *checksum, uint32_t *samples);

#endif
