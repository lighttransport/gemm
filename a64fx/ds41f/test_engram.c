#include "ds41f_engram.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv)
{
    ds41f_engram *e = calloc(1, sizeof *e);
    if (!e) return 1;
    e->token_map[2] = 5;
    e->token_map[100] = 7;
    e->token_map[101] = 11;
    for (int l = 0; l < 2; ++l) {
        for (int j = 0; j < 4; ++j) e->multipliers[l][j] = 2*j+1;
        for (int n = 0; n < 3; ++n)
            for (int h = 0; h < 8; ++h) e->primes[l][n][h] = 101+2*(n*8+h);
    }
    uint64_t ids[2][24];
    for (int step = 0; step < 2; ++step) {
        if (ds41f_engram_hash_ids(e,100+step,ids)) return 1;
        uint64_t tokens[4] = {step ? 11 : 7, step ? 7 : 5,5,5};
        for (int l = 0; l < 2; ++l) {
            uint64_t offset = 0;
            for (int n = 0; n < 3; ++n) {
                uint64_t hash = 0;
                for (int j = 0; j < n+2; ++j) hash ^= tokens[j]*(2*j+1);
                for (int h = 0; h < 8; ++h) {
                    uint64_t prime = e->primes[l][n][h];
                    if (ids[l][n*8+h] != offset+hash%prime) return 1;
                    offset += prime;
                }
            }
        }
    }
    if (ds41f_engram_hash_ids(e,129280,ids) != EINVAL) return 1;
    free(e);
    puts("DS41F_ENGRAM PASS compressed_history padding bucket_offsets bounds");
    if (argc==3) {
        int rank=atoi(argv[2]);
        e=calloc(1,sizeof *e);
        if (!e || ds41f_engram_open(e,argv[1],rank,12)) return 1;
        if (ds41f_engram_hash_ids(e,100,ids)) return 1;
        for (int l=0;l<2;++l) {
            for (int j=0;j<24;++j) if (ids[l][j]>=e->table[l].rows) return 1;
            uint16_t row[256];
            if (ds41f_engram_read_local(e,l,e->table[l].first,row)) return 1;
            for (int j=0;j<256;++j) if ((row[j]&0x7f80)==0x7f80) return 1;
        }
        ds41f_engram_close(e);free(e);
        printf("STAGED_ENGRAM PASS rank=%d metadata hash_bounds local_rows_finite\n",rank);
    }
    return 0;
}
