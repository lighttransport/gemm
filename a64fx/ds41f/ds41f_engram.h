#ifndef DS41F_ENGRAM_H
#define DS41F_ENGRAM_H

#include <stdint.h>

#define DS41F_ENGRAM_LAYERS 2
#define DS41F_ENGRAM_HEADS 8
#define DS41F_ENGRAM_HASH_COLS 24
#define DS41F_ENGRAM_DIM 256
#define DS41F_ENGRAM_MAX_RANKS 12

typedef struct {
    uint64_t rows;
    uint32_t rank, ranks;
    uint64_t first;
    uint64_t owned_rows;
    int weight_fd, scale_fd;
    uint64_t lookups, local_rows, remote_rows;
} ds41f_engram_table;

typedef struct {
    ds41f_engram_table table[DS41F_ENGRAM_LAYERS];
    uint64_t token_map[129280];
    uint64_t multipliers[DS41F_ENGRAM_LAYERS][4];
    uint64_t primes[DS41F_ENGRAM_LAYERS][3][DS41F_ENGRAM_HEADS];
    uint32_t history[4];
    uint32_t history_len;
} ds41f_engram;

int ds41f_engram_open(ds41f_engram *e, const char *stage, uint32_t rank,
                      uint32_t ranks);
void ds41f_engram_close(ds41f_engram *e);
int ds41f_engram_hash_ids(ds41f_engram *e, uint32_t token,
                          uint64_t ids[DS41F_ENGRAM_LAYERS][DS41F_ENGRAM_HASH_COLS]);
int ds41f_engram_read_local(ds41f_engram *e, int layer, uint64_t row,
                            uint16_t *out);

#endif
