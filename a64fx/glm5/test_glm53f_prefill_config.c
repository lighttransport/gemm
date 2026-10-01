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
    failed |= parse("--index-kernel", "heads", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "1");
    failed |= parse("--index-kernel", "legacy", &c) != 1 || strcmp(getenv("GLM53F_INDEX_HEADS"), "0");
    failed |= parse("--index-kernel", "invalid", &c) != -1;
    printf("PREFILL_CONFIG %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
