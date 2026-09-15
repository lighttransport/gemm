#ifndef GLM53F_PREFILL_H
#define GLM53F_PREFILL_H
#include <stdlib.h>
#include <string.h>
/* The outer layer tile is independent of verification snapshots, arithmetic
 * panels, and collective payloads. Attention scratch stays bounded at 32. */
enum { GLM53F_PREFILL_MAX_TOKENS = 512, GLM53F_PREFILL_V5_TOKENS = 256,
       GLM53F_PREFILL_ATTN_TOKENS = 32 };
enum { GLM53F_PREFILL_LEGACY = 0, GLM53F_PREFILL_V5 = 1, GLM53F_PREFILL_FAST = 2 };
enum { GLM53F_PREFILL_COMM = 1, GLM53F_PREFILL_RECURRENCE = 2,
       GLM53F_PREFILL_GEMM = 4, GLM53F_PREFILL_EXPERT16 = 8,
       GLM53F_PREFILL_MLA_REG = 16, GLM53F_PREFILL_FAST_ALL = 31,
       GLM53F_PREFILL_FAST_DEFAULT = GLM53F_PREFILL_COMM | GLM53F_PREFILL_RECURRENCE | GLM53F_PREFILL_EXPERT16 };
typedef struct {
    int mode;
    int slab_tokens;             /* 4, 8, 16 or 32; independent of arithmetic. */
    unsigned features;           /* Diagnostic ablation; zero keeps v5 kernels. */
    float *gemm_arena;            /* Internal model-owned scratch; callers use NULL. */
    int collective;              /* 0=default, 1=MPI, 2=ring, 3=tree, 4=packed tree. */
} glm53f_prefill_config;

/* Shared argument parser. Returns 1 if consumed, 0 if unknown, -1 if invalid. */
static inline int glm53f_prefill_option(glm53f_prefill_config *c,
                                       int argc, char **argv, int *index) {
    const char *key = argv[*index];
    if (strcmp(key, "--prefill-mode") && strcmp(key, "--prefill-slab") &&
        strcmp(key, "--prefill-features") && strcmp(key, "--prefill-collective")) return 0;
    if (*index + 1 >= argc) return -1;
    const char *value = argv[++*index];
    if (!strcmp(key, "--prefill-mode")) {
        if (!strcmp(value, "v5")) c->mode = GLM53F_PREFILL_V5;
        else if (!strcmp(value, "fast")) c->mode = GLM53F_PREFILL_FAST;
        else return -1;
    } else if (!strcmp(key, "--prefill-collective")) {
        if (!strcmp(value, "utofu")) c->collective = 0;
        else if (!strcmp(value, "mpi-rsag")) c->collective = 1;
        else if (!strcmp(value, "ring")) c->collective = 2;
        else if (!strcmp(value, "tree-rsag")) c->collective = 3;
        else if (!strcmp(value, "tree-packed")) c->collective = 4;
        else return -1;
    } else {
        char *end;
        long n = strtol(value, &end, 10);
        if (!*value || *end || n < 0 || n > 32) return -1;
        if (!strcmp(key, "--prefill-slab")) {
            if (n != 4 && n != 8 && n != 16 && n != 32) return -1;
            c->slab_tokens = (int)n;
        } else {
            if (n > GLM53F_PREFILL_FAST_ALL) return -1;
            c->features = (unsigned)n;
        }
    }
    return 1;
}
#endif
