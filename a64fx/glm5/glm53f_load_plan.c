/* GLM-5.3F 12-rank load and long-context memory planner.
 *
 * This is intentionally payload-free.  It opens safetensors headers, assigns
 * every tensor to an EP/TP ownership rule, and proves the per-rank HBM budget
 * before the distributed loader allocates or reads multi-gigabyte weights.
 */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_arch.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t expert, tp, replicated, kv, index, linear_state, scratch, os;
    unsigned expert_tensors, tp_tensors, replicated_tensors;
} rank_plan;

static int expert_id(const char *name) {
    const char *p = strstr(name, ".mlp.experts.");
    char *end;
    long id;
    if (!p) return -1;
    p += strlen(".mlp.experts.");
    errno = 0;
    id = strtol(p, &end, 10);
    return errno || end == p || *end != '.' || id < 0 || id >= 288 ? -1 : (int)id;
}

static uint64_t split_bytes(const st_tensor_info *t, int rank, int nranks) {
    uint64_t rows, r0, r1;
    if (t->n_dims < 2 || t->shape[0] == 0) return t->nbytes;
    rows = t->shape[0];
    r0 = rows * (uint64_t)rank / (uint64_t)nranks;
    r1 = rows * (uint64_t)(rank + 1) / (uint64_t)nranks;
    return (uint64_t)t->nbytes * (r1 - r0) / rows;
}

static int is_replicated(const char *name, const st_tensor_info *t) {
    /* Norms, biases, and small routing vectors are cheaper and simpler to
     * replicate.  All matrix payloads, including embeddings/head/shared
     * experts and attention projections, are tensor-parallel row shards. */
    (void)name;
    return t->n_dims < 2 || t->nbytes < 1024 * 1024;
}

static double gib(uint64_t n) { return (double)n / 1073741824.0; }

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : NULL;
    int nranks = argc > 2 ? atoi(argv[2]) : 12;
    uint64_t ctx = argc > 3 ? strtoull(argv[3], NULL, 0) : 262144;
    double scratch_gib = argc > 4 ? atof(argv[4]) : 2.0;
    double os_gib = argc > 5 ? atof(argv[5]) : 1.0;
    const uint64_t hbm = 32ull << 30;
    const glm53f_arch a = glm53f_arch_default();
    glm53f_st_context *st;
    rank_plan *p;
    int i, r, bad = 0;
    if (!dir || nranks < 1 || nranks > 384 || ctx < 1 || ctx > (uint64_t)a.max_position_embeddings) {
        fprintf(stderr, "usage: %s MODEL_DIR [ranks=12] [ctx=262144] [scratch_GiB=2] [os_GiB=1]\n", argv[0]);
        return 2;
    }
    st = glm53f_st_open(dir);
    if (!st || glm53f_st_validate_contract(st, 1) != 0) {
        fprintf(stderr, "GLM53F_LOAD_PLAN checkpoint contract failed\n");
        glm53f_st_close(st);
        return 1;
    }
    p = (rank_plan *)calloc((size_t)nranks, sizeof(*p));
    if (!p) { glm53f_st_close(st); return 1; }
    for (i = 0; i < st->n_entries; ++i) {
        const glm53f_st_entry *e = &st->entries[i];
        const st_tensor_info *t = &st->shards[e->shard].st->tensors[e->tensor];
        int ex = expert_id(e->name);
        if (ex >= 0) {
            r = ex % nranks;
            p[r].expert += t->nbytes;
            p[r].expert_tensors++;
        } else if (is_replicated(e->name, t)) {
            for (r = 0; r < nranks; ++r) {
                p[r].replicated += t->nbytes;
                p[r].replicated_tensors++;
            }
        } else {
            for (r = 0; r < nranks; ++r) {
                p[r].tp += split_bytes(t, r, nranks);
                p[r].tp_tensors++;
            }
        }
    }
    for (r = 0; r < nranks; ++r) {
        uint64_t slots = (ctx + (uint64_t)nranks - 1) / (uint64_t)nranks;
        p[r].kv = slots * 11ull * (uint64_t)a.kv_lora_rank * 2ull;
        p[r].index = slots * 11ull * (uint64_t)a.index_head_dim * 2ull;
        p[r].linear_state = 34ull * (uint64_t)a.n_attention_heads *
                            (uint64_t)a.linear_head_dim * (uint64_t)a.linear_head_dim * 4ull;
        p[r].scratch = (uint64_t)(scratch_gib * 1073741824.0);
        p[r].os = (uint64_t)(os_gib * 1073741824.0);
    }
    printf("GLM53F_LOAD_PLAN ranks=%d ctx=%" PRIu64 " hbm=%.2f_GiB/rank scratch=%.2f os=%.2f\n",
           nranks, ctx, gib(hbm), scratch_gib, os_gib);
    printf("ownership=experts:e%%ranks; matrices:row-TP; vectors/small(<1MiB):replicated; cache:block-CP\n");
    for (r = 0; r < nranks; ++r) {
        uint64_t weights = p[r].expert + p[r].tp + p[r].replicated;
        uint64_t total = weights + p[r].kv + p[r].index + p[r].linear_state + p[r].scratch + p[r].os;
        int fit = total <= hbm;
        printf("rank=%d weights=%.3f expert=%.3f tp=%.3f repl=%.3f kv=%.3f index=%.3f "
               "linear=%.3f reserve=%.3f total=%.3f headroom=%.3f fit=%s\n",
               r, gib(weights), gib(p[r].expert), gib(p[r].tp), gib(p[r].replicated),
               gib(p[r].kv), gib(p[r].index), gib(p[r].linear_state),
               gib(p[r].scratch + p[r].os), gib(total), gib(hbm) - gib(total), fit ? "YES" : "NO");
        bad |= !fit;
    }
    printf("SENTINEL glm53f_load_plan=%s\n", bad ? "NOFIT" : "FIT");
    free(p);
    glm53f_st_close(st);
    return bad ? 3 : 0;
}
