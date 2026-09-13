#define _GNU_SOURCE
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s LOCAL_BASE RANK\n", argv[0]);
        return 2;
    }
    int rank = atoi(argv[2]);
    char directory[4096];
    if (rank < 0 || rank >= Q38FN_TP_RANKS ||
        snprintf(directory, sizeof(directory), "%s/rank-%02d", argv[1], rank) >=
            (int)sizeof(directory)) return 2;
    q38fn_tp_blob blob;
    if (q38fn_tp_blob_open(&blob, directory, 0)) return 1;
    size_t q5_bytes = 0, q5_tensors = 0, ngram_q5 = 0;
    for (int i = 0; i < blob.n_entries; ++i) {
        const q38fn_tp_blob_entry *entry = &blob.entries[i];
        if (entry->q5_data) {
            q5_bytes += entry->q5_bytes;
            q5_tensors++;
            ngram_q5 += strstr(entry->name, "ngram_embedding.shard_") != NULL;
        }
    }
    printf("Q38FN_TP_BLOB_PROBE rank=%d tensors=%d blob_bytes=%zu "
           "q5_tensors=%zu q5_bytes=%zu ngram_q5=%zu\n",
           rank, blob.n_entries, blob.bytes, q5_tensors, q5_bytes, ngram_q5);
    q38fn_tp_blob_close(&blob);
    return q5_tensors ? 0 : 1;
}
