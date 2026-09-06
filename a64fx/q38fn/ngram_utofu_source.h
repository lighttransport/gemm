/* Fugaku uTofu source for the Qwen3.8 n-gram pipeline.
 *
 * One rank owns shard % nranks.  Remote pipeline workers issue bounded span
 * reads through registered request/response slots; one service thread per
 * peer on the owner reads its resident shard and returns the span with one
 * uTofu Put.
 * Client requests and service responses use independent VCQs so their TCQ
 * progress cannot consume one another's completion notices.
 * This file intentionally depends on the Fugaku uTofu ABI and is not part of
 * the host-portable q38fn pipeline.
 */
#ifndef Q38FN_NGRAM_UTOFU_SOURCE_H
#define Q38FN_NGRAM_UTOFU_SOURCE_H

#include "../../common/q38fn_ngram_transport.h"
#include <stddef.h>
#include <stdint.h>

#define Q38FN_UTOFU_MAX_RANKS 12
#define Q38FN_UTOFU_MAX_SPAN 256
#define Q38FN_UTOFU_SLOTS 4 /* bounded per-owner credit slots */
#define Q38FN_UTOFU_MAX_SERVICE_THREADS \
    (Q38FN_UTOFU_MAX_RANKS * Q38FN_UTOFU_SLOTS)

typedef int (*q38fn_utofu_read_fn)(void *opaque, uint32_t shard,
                                   uint64_t first_row, uint32_t rows,
                                   void *dst);

typedef struct q38fn_utofu_source q38fn_utofu_source;

int q38fn_utofu_source_init(q38fn_utofu_source **out, const char *topology,
                            uint32_t rank, uint32_t nranks,
                            q38fn_utofu_read_fn read_local, void *read_opaque,
                            q38fn_ngram_source *sources, uint32_t source_count);
void q38fn_utofu_source_destroy(q38fn_utofu_source *s);

#endif
