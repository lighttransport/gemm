/* Adapter boundary for remote n-gram owners.
 *
 * This header has no utofu.h dependency. A Fugaku adapter can register one
 * request/response slab per peer, submit a span with uTofu Put, poll its TCQ,
 * and expose completed bytes through fetch_span. The pipeline's persistent
 * workers provide bounded parallelism above it.
 */
#ifndef Q38FN_NGRAM_TRANSPORT_H
#define Q38FN_NGRAM_TRANSPORT_H

#include "q38fn_ngram_pipeline.h"

typedef int (*q38fn_ngram_transport_fetch_fn)(void *opaque, uint32_t owner,
                                              uint64_t first_row, uint32_t rows,
                                              void *dst);
typedef struct {
    q38fn_ngram_transport_fetch_fn fetch_span;
    void *opaque;
    uint32_t owner;
} q38fn_ngram_transport_source;

static inline int q38fn_ngram_transport_read_span(void *opaque,
                                                   uint64_t first_row,
                                                   uint32_t rows, void *dst) {
    q38fn_ngram_transport_source *s = (q38fn_ngram_transport_source *)opaque;
    return s && s->fetch_span ? s->fetch_span(s->opaque, s->owner, first_row, rows, dst) : -1;
}

static inline q38fn_ngram_source q38fn_ngram_transport_as_source(
    q38fn_ngram_transport_source *s) {
    q38fn_ngram_source out = { q38fn_ngram_transport_read_span, s, s ? s->owner : 0 };
    return out;
}

#endif
