#ifndef GLM53F_RUNTIME_H
#define GLM53F_RUNTIME_H
#include <stdlib.h>
#include <string.h>

/* Startup-only switches shared by generation, benchmarking and verification.
 * Parse before model creation: scratch and communication ownership depend on them. */
static inline int glm53f_runtime_option(int argc, char **argv, int *index) {
    const char *key = argv[*index], *env;
    if (!strcmp(key, "--verify-kernel")) env = "GLM53F_VERIFY_GROUPED";
    else if (!strcmp(key, "--decode-executor")) env = "GLM53F_DECODE_EXECUTOR";
    else if (!strcmp(key, "--router-kernel")) env = "GLM53F_ROUTER_FUSE";
    else if (!strcmp(key, "--collective-owner")) env = "GLM53F_COMM_OWNER";
    else if (!strcmp(key, "--moe-combine-kernel")) env = "GLM53F_MOE_COMBINE";
    else if (!strcmp(key, "--index-kernel")) env = "GLM53F_INDEX_HEADS";
    else if (!strcmp(key, "--pool-selector")) env = "GLM53F_POOL_PARTITION_4K";
    else if (!strcmp(key, "--q8-row-kernel")) env = "GLM53F_NATIVE_Q8_ROWS8";
    else if (!strcmp(key, "--q8-prefill-kernel")) env = "GLM53F_NATIVE_Q8_TILE2X8";
    else if (!strcmp(key, "--mla-projection-kernel")) env = "GLM53F_MLA_FUSED_PROJECTION";
    else if (!strcmp(key, "--mla-kernel")) env = "GLM53F_MLA_REGISTERS";
    else return 0;
    if (*index + 1 >= argc) return -1;
    const char *value = argv[++*index];
    int enabled;
    if ((!strcmp(key, "--q8-row-kernel") && !strcmp(value, "rows4")) ||
        (!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile4x4")) ||
        (!strcmp(key, "--pool-selector") && !strcmp(value, "heap")) || !strcmp(value, "legacy")) enabled = 0;
    else if ((!strcmp(key, "--q8-row-kernel") && !strcmp(value, "rows8")) ||
             (!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile2x8")) ||
             (!strcmp(key, "--pool-selector") && !strcmp(value, "partition4k")) ||
             (!strcmp(key, "--verify-kernel") && !strcmp(value, "grouped")) ||
             (!strcmp(key, "--collective-owner") && !strcmp(value, "serialized")) ||
             (!strcmp(key, "--router-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--mla-projection-kernel") && !strcmp(value, "fused")) ||
             (!strcmp(key, "--moe-combine-kernel") && !strcmp(value, "vector")) ||
             (!strcmp(key, "--index-kernel") && !strcmp(value, "heads")) ||
             (!strcmp(key, "--mla-kernel") && !strcmp(value, "registers")) ||
             (!strcmp(key, "--decode-executor") && !strcmp(value, "persistent"))) enabled = 1;
    else if ((!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile4x4-asm")) ||
             (!strcmp(key, "--moe-combine-kernel") && !strcmp(value, "overlap")) ||
             (!strcmp(key, "--index-kernel") && !strcmp(value, "keys4")) ||
             (!strcmp(key, "--mla-kernel") && !strcmp(value, "values"))) enabled = 2;
    else if ((!strcmp(key, "--q8-prefill-kernel") && !strcmp(value, "tile2x8-asm")) ||
             (!strcmp(key, "--mla-kernel") && !strcmp(value, "fp16-cache")) ||
             (!strcmp(key, "--index-kernel") && !strcmp(value, "replicated-heads"))) enabled = 3;
    else if (!strcmp(key, "--index-kernel") && !strcmp(value, "replicated-keys4")) enabled = 4;
    else return -1;
    return setenv(env, enabled == 4 ? "4" : enabled == 3 ? "3" : enabled == 2 ? "2" : enabled ? "1" : "0", 1) ? -1 : 1;
}
#endif
