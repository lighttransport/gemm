/* Measure rank-owned staged-shard loading without starting uTofu. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "ngram_owner_hbm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const char *model = argc > 1 ? argv[1] : NULL;
    unsigned rank = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 0;
    unsigned nranks = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 128;
    unsigned read_rows = argc > 4 ? (unsigned)strtoul(argv[4], NULL, 10) : 0;
    unsigned read_iters = argc > 5 ? (unsigned)strtoul(argv[5], NULL, 10) : 10000;
    if (!model || nranks == 0 || rank >= nranks) {
        fprintf(stderr, "usage: %s MODEL_DIR [rank=0] [nranks=128] [read_rows=0] [read_iters=10000]\n", argv[0]);
        return 2;
    }
    double start = now_sec();
    q38fn_ngram_owner_hbm *owner = NULL;
    int rc = q38fn_ngram_owner_hbm_open(&owner, model, rank, nranks);
    double elapsed = now_sec() - start;
    printf("Q38FN_OWNER_LOAD rank=%u nranks=%u rc=%d seconds=%.6f\n",
           rank, nranks, rc, elapsed);
    if (!rc && read_rows) {
        if (read_rows > 32) read_rows = 32;
        size_t bytes = (size_t)read_rows * Q38FN_NGRAM_ROW_BYTES;
        uint16_t *dst = NULL;
        if (posix_memalign((void **)&dst, 256, bytes) != 0) rc = 12;
        uint64_t checksum = 0;
        double read_start = now_sec();
        for (unsigned i = 0; !rc && i < read_iters; ++i) {
            uint64_t first = ((uint64_t)i * 7919u) %
                (Q38FN_NGRAM_ROWS_PER_SHARD - read_rows);
            rc = q38fn_ngram_owner_hbm_read(owner, rank, first, read_rows, dst);
            if (!rc) checksum += dst[(size_t)(i % read_rows) *
                                     Q38FN_NGRAM_HEAD_DIM];
        }
        double read_elapsed = now_sec() - read_start;
        printf("Q38FN_OWNER_READ rows=%u iters=%u rc=%d seconds=%.6f GB_s=%.3f checksum=%llu\n",
               read_rows, read_iters, rc, read_elapsed,
               rc || !read_elapsed ? 0.0 :
               (double)bytes * read_iters / read_elapsed / 1e9,
               (unsigned long long)checksum);
        free(dst);
    }
    q38fn_ngram_owner_hbm_close(owner);
    return rc ? 1 : 0;
}
