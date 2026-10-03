#include "glm53f_prefill.h"
#include <stdio.h>

static int parse(const char *key, const char *value, glm53f_prefill_config *c) {
    char *argv[] = {(char *)key, (char *)value};
    int index = 0;
    return glm53f_prefill_option(c, value ? 2 : 1, argv, &index);
}

int main(void) {
    glm53f_prefill_config c = {GLM53F_PREFILL_LEGACY, 32, GLM53F_PREFILL_FAST_DEFAULT, NULL, 0};
    int failed = 0;
    failed |= parse("--verify-head-kernel", "shared", &c) != 1 || strcmp(getenv("GLM53F_HEAD_VERIFY_SHARED"), "1");
    failed |= parse("--verify-head-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_HEAD_VERIFY_SHARED"), "0");
    failed |= parse("--verify-head-kernel", "bad", &c) != -1;
    failed |= parse("--verify-head-kernel", NULL, &c) != -1;
    failed |= parse("--embedding-batch-kernel", "packed", &c) != 1 || strcmp(getenv("GLM53F_EMBED_BATCH_PACKED"), "1");
    failed |= parse("--embedding-batch-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_EMBED_BATCH_PACKED"), "0");
    failed |= parse("--embedding-batch-kernel", "bad", &c) != -1;
    failed |= parse("--embedding-batch-kernel", NULL, &c) != -1;
    const char *dense_tiles[] = {"4", "16", "32", "64"};
    for (int i = 0; i < 4; ++i)
        failed |= parse("--dense-prefill-tile", dense_tiles[i], &c) != 1 ||
                  strcmp(getenv("GLM53F_DENSE_PREFILL_TILE"), dense_tiles[i]);
    const char *bad_dense_tiles[] = {"0", "8", "128", "16junk", "-1", ""};
    for (int i = 0; i < 6; ++i)
        failed |= parse("--dense-prefill-tile", bad_dense_tiles[i], &c) != -1;
    failed |= parse("--dense-prefill-tile", NULL, &c) != -1;
    failed |= parse("--mla-softmax-kernel", "parallel", &c) != 1 || strcmp(getenv("GLM53F_MLA_PARALLEL_SOFTMAX"), "1");
    failed |= parse("--mla-softmax-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MLA_PARALLEL_SOFTMAX"), "0");
    failed |= parse("--mla-softmax-kernel", "fused", &c) != -1;
    failed |= parse("--mla-softmax-kernel", NULL, &c) != -1;
    failed |= parse("--kda-decode-kernel", "columns", &c) != 1 || strcmp(getenv("GLM53F_KDA_DECODE_COLUMNS"), "1");
    failed |= parse("--kda-decode-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_KDA_DECODE_COLUMNS"), "0");
    failed |= parse("--kda-decode-kernel", "team", &c) != -1;
    failed |= parse("--kda-decode-kernel", NULL, &c) != -1;
    failed |= parse("--moe-scale-kernel", "words", &c) != 1 || strcmp(getenv("GLM53F_IQ_SCALE_WORDS"), "1");
    failed |= parse("--moe-scale-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_IQ_SCALE_WORDS"), "0");
    failed |= parse("--moe-scale-kernel", "columns", &c) != -1;
    failed |= parse("--moe-scale-kernel", NULL, &c) != -1;
    failed |= parse("--kda-prefill-kernel", "columns", &c) != 1 || strcmp(getenv("GLM53F_KDA_PREFILL_COLUMNS"), "1");
    failed |= parse("--kda-prefill-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_KDA_PREFILL_COLUMNS"), "0");
    failed |= parse("--kda-prefill-kernel", "team", &c) != -1;
    failed |= parse("--kda-prefill-kernel", NULL, &c) != -1;
    failed |= parse("--mhc-verify-kernel", "team", &c) != 1 || strcmp(getenv("GLM53F_MHC_BATCH_TEAM"), "1");
    failed |= parse("--mhc-verify-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MHC_BATCH_TEAM"), "0");
    failed |= parse("--mhc-verify-kernel", "fused-sync", &c) != -1;
    failed |= parse("--mhc-verify-kernel", NULL, &c) != -1;
    failed |= parse("--moe-prefill-layout", "padded", &c) != 1 || strcmp(getenv("GLM53F_MOE_GU_PAD"), "1");
    failed |= parse("--moe-prefill-layout", "tight", &c) != 1 || strcmp(getenv("GLM53F_MOE_GU_PAD"), "0");
    failed |= parse("--moe-prefill-layout", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MOE_GU_PAD"), "0");
    failed |= parse("--moe-prefill-layout", "panels", &c) != -1;
    failed |= parse("--moe-prefill-layout", NULL, &c) != -1;
    failed |= parse("--mhc-kernel", "fused-sync", &c) != 1 || strcmp(getenv("GLM53F_MHC_FUSED_SYNC"), "1");
    failed |= parse("--mhc-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MHC_FUSED_SYNC"), "0");
    failed |= parse("--mhc-kernel", "distributed", &c) != -1;
    failed |= parse("--mhc-kernel", NULL, &c) != -1;
    failed |= parse("--mla-projection-kernel", "fused", &c) != 1 || strcmp(getenv("GLM53F_MLA_FUSED_PROJECTION"), "1");
    failed |= parse("--mla-projection-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MLA_FUSED_PROJECTION"), "0");
    failed |= parse("--mla-projection-kernel", "tile4x4-asm", &c) != -1;
    failed |= parse("--q8-row-kernel", "rows8", &c) != 1 || strcmp(getenv("GLM53F_NATIVE_Q8_ROWS8"), "1");
    failed |= parse("--q8-row-kernel", "rows4", &c) != 1 || strcmp(getenv("GLM53F_NATIVE_Q8_ROWS8"), "0");
    failed |= parse("--q8-prefill-kernel", "tile2x8", &c) != 1 || strcmp(getenv("GLM53F_NATIVE_Q8_TILE2X8"), "1");
    failed |= parse("--q8-prefill-kernel", "tile4x4-asm", &c) != 1 || strcmp(getenv("GLM53F_NATIVE_Q8_TILE2X8"), "2");
    failed |= parse("--q8-prefill-kernel", "tile2x8-asm", &c) != 1 || strcmp(getenv("GLM53F_NATIVE_Q8_TILE2X8"), "3");
    failed |= parse("--q8-prefill-kernel", "tile4x4", &c) != 1 || strcmp(getenv("GLM53F_NATIVE_Q8_TILE2X8"), "0");
    failed |= parse("--q8-row-kernel", "tile2x8", &c) != -1;
    failed |= parse("--q8-prefill-kernel", "rows8", &c) != -1;
    failed |= glm53f_prefill_collective_count(8049, 4096) != GLM53F_PREFILL_ATTN_TOKENS * 4096;
    failed |= glm53f_prefill_collective_count(16387, 4096) != GLM53F_PREFILL_ATTN_TOKENS * 4096;
    failed |= glm53f_prefill_collective_count(16388, 4096) != GLM53F_PREFILL_ATTN_TOKENS * 4097;
    failed |= glm53f_prefill_collective_count(32196, 4096) != GLM53F_PREFILL_ATTN_TOKENS * 8049;
    failed |= glm53f_prefill_collective_count(-1, 4096) != -1;
    failed |= glm53f_prefill_collective_count(0, 0) != -1;
    failed |= glm53f_prefill_collective_count(INT_MAX, 4096) != -1;
    failed |= glm53f_prefill_collective_count(0, INT_MAX) != -1;
    failed |= parse("--prefill-mode", "v5", &c) != 1 || c.mode != GLM53F_PREFILL_V5;
    failed |= parse("--prefill-mode", "fast", &c) != 1 || c.mode != GLM53F_PREFILL_FAST;
    failed |= parse("--prefill-mode", "unknown", &c) != -1;
    failed |= parse("--prefill-mode", NULL, &c) != -1;
    const char *slabs[] = {"4", "8", "16", "32"};
    for (int i = 0; i < 4; ++i)
        failed |= parse("--prefill-slab", slabs[i], &c) != 1 || c.slab_tokens != (4 << i);
    const char *bad[] = {"0", "3", "5", "31", "33", "256", "512", "-1", "16junk", ""};
    for (unsigned i = 0; i < sizeof(bad)/sizeof(bad[0]); ++i)
        failed |= parse("--prefill-slab", bad[i], &c) != -1;
    const char *collectives[] = {"utofu", "mpi-rsag", "ring", "tree-rsag", "tree-packed", "mtni"};
    for (int i = 0; i < 6; ++i)
        failed |= parse("--prefill-collective", collectives[i], &c) != 1 || c.collective != i;
    failed |= parse("--prefill-collective", "unknown", &c) != -1;
    failed |= parse("--prefill-features", "0", &c) != 1 || c.features != 0;
    failed |= parse("--prefill-features", "31", &c) != 1 || c.features != GLM53F_PREFILL_FAST_ALL;
    failed |= parse("--prefill-features", "32", &c) != -1;
    failed |= parse("--unknown", "1", &c) != 0;
    failed |= parse("--verify-kernel", "grouped", &c) != 1 || strcmp(getenv("GLM53F_VERIFY_GROUPED"), "1");
    failed |= parse("--verify-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_VERIFY_GROUPED"), "0");
    failed |= parse("--verify-kernel", "invalid", &c) != -1;
    failed |= parse("--collective-owner", "serialized", &c) != 1 || strcmp(getenv("GLM53F_COMM_OWNER"), "1");
    failed |= parse("--collective-owner", "legacy", &c) != 1 || strcmp(getenv("GLM53F_COMM_OWNER"), "0");
    failed |= parse("--collective-owner", NULL, &c) != -1;
    failed |= parse("--router-kernel", "fused", &c) != 1 || strcmp(getenv("GLM53F_ROUTER_FUSE"), "1");
    failed |= parse("--decode-executor", "persistent", &c) != 1 || strcmp(getenv("GLM53F_DECODE_EXECUTOR"), "1");
    failed |= parse("--decode-executor", "legacy", &c) != 1 || strcmp(getenv("GLM53F_DECODE_EXECUTOR"), "0");
    failed |= parse("--decode-executor", "invalid", &c) != -1;
    failed |= parse("--moe-combine-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MOE_COMBINE"), "0");
    failed |= parse("--moe-combine-kernel", "vector", &c) != 1 || strcmp(getenv("GLM53F_MOE_COMBINE"), "1");
    failed |= parse("--moe-combine-kernel", "overlap", &c) != 1 || strcmp(getenv("GLM53F_MOE_COMBINE"), "2");
    failed |= parse("--moe-combine-kernel", "invalid", &c) != -1;
    failed |= parse("--pool-selector", "partition4k", &c) != 1 || strcmp(getenv("GLM53F_POOL_PARTITION_4K"), "1");
    failed |= parse("--pool-selector", "heap", &c) != 1 || strcmp(getenv("GLM53F_POOL_PARTITION_4K"), "0");
    failed |= parse("--pool-selector", "invalid", &c) != -1;
    failed |= parse("--index-kernel", "heads", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "1");
    failed |= parse("--index-kernel", "keys4", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "2");
    failed |= parse("--index-kernel", "replicated-heads", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "3");
    failed |= parse("--index-kernel", "replicated-keys4", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "4");
    failed |= parse("--index-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "0");
    failed |= parse("--index-kernel", "invalid", &c) != -1;
    failed |= parse("--mla-kernel", "registers", &c) != 1 || strcmp(getenv("GLM53F_MLA_REGISTERS"), "1");
    failed |= parse("--mla-kernel", "values", &c) != 1 || strcmp(getenv("GLM53F_MLA_REGISTERS"), "2");
    failed |= parse("--mla-kernel", "fp16-cache", &c) != 1 || strcmp(getenv("GLM53F_MLA_REGISTERS"), "3");
    failed |= parse("--mla-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_MLA_REGISTERS"), "0");
    failed |= parse("--mla-kernel", "invalid", &c) != -1;
    printf("PREFILL_CONFIG %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
