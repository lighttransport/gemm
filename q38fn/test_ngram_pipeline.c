#include "../common/q38fn_ngram_pipeline.h"
#include "../common/q38fn_ngram_transport.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>

static int fake_read(void *opaque, uint64_t first, uint32_t rows, void *dst) {
    (void)opaque;
    uint16_t *p = (uint16_t *)dst;
    for (uint32_t r = 0; r < rows; ++r)
        for (int i = 0; i < Q38FN_NGRAM_HEAD_DIM; ++i)
            p[(size_t)r * Q38FN_NGRAM_HEAD_DIM + i] = (uint16_t)(first + r + i);
    return 0;
}

typedef struct {
    q38fn_ngram_pipeline *pipeline;
    uint64_t base;
    int rc;
} producer_arg;

static void *producer_main(void *opaque)
{
    producer_arg *a = (producer_arg *)opaque;
    uint16_t out[8 * Q38FN_NGRAM_HEAD_DIM];
    for (uint32_t n = 0; n < 16; ++n) {
        uint64_t rows[8];
        for (uint32_t i = 0; i < 8; ++i) rows[i] = a->base + n * 16 + i;
        q38fn_ngram_ticket ticket;
        a->rc = q38fn_ngram_submit(a->pipeline, rows, 8, &ticket);
        if (a->rc) return NULL;
        a->rc = q38fn_ngram_wait(a->pipeline, ticket, out, sizeof out);
        if (a->rc) return NULL;
        for (uint32_t i = 0; i < 8; ++i)
            if (out[i * Q38FN_NGRAM_HEAD_DIM] != (uint16_t)rows[i]) {
                a->rc = EPROTO;
                return NULL;
            }
    }
    return NULL;
}

int main(void) {
    q38fn_ngram_source src[Q38FN_NGRAM_SHARDS];
    memset(src, 0, sizeof(src));
    for (int i = 0; i < Q38FN_NGRAM_SHARDS; ++i) src[i].read_span = fake_read;
    q38fn_ngram_pipeline *p = NULL;
    if (q38fn_ngram_pipeline_init_ex(&p, src, Q38FN_NGRAM_SHARDS, 8, 8, 8, 1, 32) != 0)
        return 1;
    uint64_t rows[] = {0, 1, 1, 3, 4, 7, 7, 8};
    uint16_t out[Q38FN_NGRAM_MAX_BATCH * Q38FN_NGRAM_HEAD_DIM];
    q38fn_ngram_ticket ticket;
    int rc = q38fn_ngram_submit(p, rows, (uint32_t)(sizeof(rows) / sizeof(rows[0])), &ticket);
    if (!rc) rc = q38fn_ngram_wait(p, ticket, out, sizeof(out));
    for (size_t r = 0; !rc && r < sizeof(rows) / sizeof(rows[0]); ++r)
        if (out[r * Q38FN_NGRAM_HEAD_DIM] != (uint16_t)rows[r]) rc = 2;
    q38fn_ngram_stats st;
    q38fn_ngram_get_stats(p, &st);
    if (!rc && (st.logical_rows != 8 || st.unique_rows != 6 || st.deduplicated_rows != 2 ||
                st.cache_hits != 0 || st.cache_misses != 6 || st.spans == 0)) rc = 3;
    if (!rc && q38fn_ngram_submit(p, rows, (uint32_t)(sizeof(rows) / sizeof(rows[0])), &ticket) != 0) rc = 4;
    if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 5;
    q38fn_ngram_get_stats(p, &st);
    if (!rc && (st.logical_rows != 16 || st.unique_rows != 12 || st.cache_hits != 6 ||
                st.cache_misses != 6)) {
        fprintf(stderr, "cache stats: logical=%llu unique=%llu hits=%llu misses=%llu\\n",
                (unsigned long long)st.logical_rows, (unsigned long long)st.unique_rows,
                (unsigned long long)st.cache_hits, (unsigned long long)st.cache_misses);
        rc = 6;
    }
    {
        uint64_t mixed[] = {0, 1, 9, 10, 3, 11, 7, 12};
        if (!rc && q38fn_ngram_submit(p, mixed,
                                      (uint32_t)(sizeof(mixed) / sizeof(mixed[0])),
                                      &ticket) != 0) rc = 14;
        if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 15;
        for (size_t i = 0; !rc && i < sizeof(mixed) / sizeof(mixed[0]); ++i)
            if (out[i * Q38FN_NGRAM_HEAD_DIM] != (uint16_t)mixed[i]) rc = 16;
    }
    q38fn_ngram_set_cache_remote_only(p, 0);
    if (!rc && q38fn_ngram_submit(p, rows, (uint32_t)(sizeof(rows) / sizeof(rows[0])), &ticket) != 0) rc = 7;
    if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 8;
    q38fn_ngram_get_stats(p, &st);
    if (!rc && (st.cache_hits != 10 || st.cache_misses != 16)) rc = 9;
    if (!rc && q38fn_ngram_submit_token(p, 1, 2, 3, &ticket) != 0) rc = 10;
    if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 11;
    {
        uint64_t current[Q38FN_NGRAM_MAX_TOKEN_WINDOW];
        uint64_t previous[Q38FN_NGRAM_MAX_TOKEN_WINDOW];
        uint64_t previous2[Q38FN_NGRAM_MAX_TOKEN_WINDOW];
        for (int i = 0; i < Q38FN_NGRAM_MAX_TOKEN_WINDOW; ++i) {
            current[i] = (uint64_t)i + 40;
            previous[i] = 20;
            previous2[i] = 30;
        }
        if (!rc && q38fn_ngram_submit_token_window(p, current, previous,
                                                    previous2, Q38FN_NGRAM_MAX_TOKEN_WINDOW,
                                                    &ticket) != 0)
            rc = 17;
        if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 18;
        /* fake_read encodes the row in lane zero, so every restored logical
         * row can be checked without reproducing the transport internals. */
        for (int i = 0; !rc && i < Q38FN_NGRAM_MAX_TOKEN_WINDOW; ++i) {
            uint64_t expected[Q38FN_NGRAM_HEADS];
            q38fn_ngram_rows(current[i], previous[i], previous2[i], expected);
            for (int h = 0; h < Q38FN_NGRAM_HEADS; ++h)
                if (out[(size_t)(i * Q38FN_NGRAM_HEADS + h) * Q38FN_NGRAM_HEAD_DIM] !=
                    (uint16_t)(expected[h] % Q38FN_NGRAM_ROWS_PER_SHARD)) rc = 19;
        }
    }
    {
        uint64_t max_rows[Q38FN_NGRAM_MAX_BATCH];
        for (uint32_t i = 0; i < Q38FN_NGRAM_MAX_BATCH; ++i)
            max_rows[i] = ((uint64_t)i * UINT64_C(104729) + 17) %
                          Q38FN_NGRAM_ROWS_PER_SHARD;
        if (!rc && q38fn_ngram_submit(p, max_rows, Q38FN_NGRAM_MAX_BATCH,
                                      &ticket) != 0) rc = 24;
        if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 25;
        for (uint32_t i = 0; !rc && i < Q38FN_NGRAM_MAX_BATCH; ++i)
            if (out[(size_t)i * Q38FN_NGRAM_HEAD_DIM] != (uint16_t)max_rows[i])
                rc = 26;
    }
    q38fn_ngram_ticket pending[8];
    for (int i = 0; !rc && i < 8; ++i)
        if (q38fn_ngram_submit_token(p, (uint64_t)i + 10, 20, 30, &pending[i]) != 0) rc = 12;
    for (int i = 0; !rc && i < 8; ++i)
        if (q38fn_ngram_wait(p, pending[i], out, sizeof(out)) != 0) rc = 13;
    {
        /* Exercise the direct destination path: sorted, unique and
         * contiguous rows need no scratch-to-result reorder copy. */
        uint64_t contiguous[] = {100, 101, 102, 103};
        if (!rc && q38fn_ngram_submit(p, contiguous, 4, &ticket) != 0) rc = 20;
        if (!rc && q38fn_ngram_wait(p, ticket, out, sizeof(out)) != 0) rc = 21;
        for (int i = 0; !rc && i < 4; ++i)
            if (out[(size_t)i * Q38FN_NGRAM_HEAD_DIM] != (uint16_t)contiguous[i]) rc = 22;
        q38fn_ngram_get_stats(p, &st);
        if (!rc && (st.direct_spans == 0 || st.reorder_spans == 0)) rc = 23;
    }
    {
        producer_arg args[2] = {{p, 10000, 0}, {p, 20000, 0}};
        pthread_t producers[2];
        int created = 0;
        if (!rc) {
            if (pthread_create(&producers[0], NULL, producer_main, &args[0]))
                rc = 27;
            else
                created = 1;
        }
        if (!rc) {
            if (pthread_create(&producers[1], NULL, producer_main, &args[1]))
                rc = 27;
            else
                created = 2;
        }
        for (int i = 0; i < created; ++i)
            pthread_join(producers[i], NULL);
        if (!rc) {
            if (args[0].rc || args[1].rc) rc = args[0].rc ? args[0].rc : args[1].rc;
        }
    }
    q38fn_ngram_pipeline_destroy(p);
    if (rc) fprintf(stderr, "ngram pipeline test failed: %d\n", rc);
    else printf("ngram pipeline test passed: unique=%llu spans=%llu physical=%llu\n",
                (unsigned long long)st.unique_rows, (unsigned long long)st.spans,
                (unsigned long long)st.physical_bytes);
    return rc;
}
