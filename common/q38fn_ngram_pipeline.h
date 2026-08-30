/* Bounded asynchronous Qwen3.8 n-gram row pipeline.
 *
 * The core is transport neutral: a source may be a local safetensors file,
 * an mmap/HBM cache, or a uTofu owner.  Requests are deduplicated before they
 * enter the queue and each worker coalesces nearby rows into bounded spans.
 */
#ifndef Q38FN_NGRAM_PIPELINE_H
#define Q38FN_NGRAM_PIPELINE_H

#include "q38fn_arch.h"
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <pthread.h>

#define Q38FN_NGRAM_MAX_HEADS Q38FN_NGRAM_HEADS
#define Q38FN_NGRAM_MAX_TOKEN_WINDOW 16
#define Q38FN_NGRAM_MAX_BATCH \
    (Q38FN_NGRAM_MAX_HEADS * Q38FN_NGRAM_MAX_TOKEN_WINDOW)

typedef int (*q38fn_ngram_read_span_fn)(void *opaque, uint64_t first_row,
                                        uint32_t rows, void *dst);

typedef struct {
    q38fn_ngram_read_span_fn read_span;
    void *opaque;
    uint32_t owner;
} q38fn_ngram_source;

typedef struct {
    uint64_t row;
    uint16_t logical;
    uint16_t unique;
} q38fn_ngram_request;

typedef struct {
    uint64_t submitted, completed, logical_rows, unique_rows;
    uint64_t spans, physical_bytes, useful_bytes, deduplicated_rows;
    uint64_t local_rows, remote_rows, wait_ns;
    uint64_t cache_hits, cache_misses;
    uint64_t direct_spans, reorder_spans;
} q38fn_ngram_stats;

typedef struct q38fn_ngram_pipeline q38fn_ngram_pipeline;

typedef struct {
    uint32_t slot;
    uint64_t generation;
} q38fn_ngram_ticket;

int q38fn_ngram_pipeline_init(q38fn_ngram_pipeline **out,
                              const q38fn_ngram_source *sources,
                              uint32_t source_count, uint32_t queue_depth,
                              uint32_t workers, uint32_t max_span_rows,
                              uint32_t max_gap_rows);
int q38fn_ngram_pipeline_init_ex(q38fn_ngram_pipeline **out,
                                 const q38fn_ngram_source *sources,
                                 uint32_t source_count, uint32_t queue_depth,
                                 uint32_t workers, uint32_t max_span_rows,
                                 uint32_t max_gap_rows, uint32_t cache_rows);
void q38fn_ngram_pipeline_destroy(q38fn_ngram_pipeline *p);
/* When enabled, cache only rows whose source owner differs from local_owner.
 * This avoids adding tag/probe overhead to already-local page-cache reads. */
void q38fn_ngram_set_cache_remote_only(q38fn_ngram_pipeline *p, uint32_t local_owner);

/* Submit one arbitrary logical request batch. rows are global packed-table
 * row IDs. The returned ticket remains valid until wait releases its slot. */
int q38fn_ngram_submit(q38fn_ngram_pipeline *p, const uint64_t *rows,
                       uint32_t count, q38fn_ngram_ticket *ticket);
int q38fn_ngram_submit_token(q38fn_ngram_pipeline *p, uint64_t current,
                             uint64_t previous, uint64_t previous2,
                             q38fn_ngram_ticket *ticket);
/* Submit up to 16 consecutive token contexts as one deduplicated batch.
 * out_rows passed to wait is count * Q38FN_NGRAM_MAX_HEADS rows, in token
 * order. This is the preferred interface when memory-level parallelism is
 * available across multiple next-token contexts. */
int q38fn_ngram_submit_token_window(q38fn_ngram_pipeline *p,
                                    const uint64_t *current,
                                    const uint64_t *previous,
                                    const uint64_t *previous2,
                                    uint32_t count,
                                    q38fn_ngram_ticket *ticket);
int q38fn_ngram_wait(q38fn_ngram_pipeline *p, q38fn_ngram_ticket ticket,
                     void *out_rows, size_t out_bytes);
int q38fn_ngram_poll(q38fn_ngram_pipeline *p, q38fn_ngram_ticket ticket);
void q38fn_ngram_get_stats(const q38fn_ngram_pipeline *p,
                           q38fn_ngram_stats *stats);

/* A local file source helper. base is the byte offset of row zero. */
typedef struct { int fd; off_t base; } q38fn_ngram_fd_source;
int q38fn_ngram_fd_read_span(void *opaque, uint64_t first_row,
                             uint32_t rows, void *dst);

#endif
