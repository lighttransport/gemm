#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"

#include <sys/stat.h>

static int write_fixture(const char *dir) {
    char path[4160];
    const char *header = "{\"target.weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[0,16]},"
        "\"draft.weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[16,32]}}";
    const uint16_t checkpoint[] = {1,2,3,4,5,6,7,8,21,22,23,24,25,26,27,28};
    const uint16_t compact[] = {101,102,103,104,105,106,107,108,202,203,206,207};
    uint64_t bytes = (strlen(header) + 7) & ~(uint64_t)7;
    snprintf(path, sizeof(path), "%s/tiny.safetensors", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int failed = fwrite(&bytes, sizeof(bytes), 1, f) != 1 ||
        fwrite(header, 1, strlen(header), f) != strlen(header);
    for (size_t i = strlen(header); i < bytes; ++i) failed |= fputc(' ', f) == EOF;
    failed |= fwrite(checkpoint, sizeof(checkpoint), 1, f) != 1;
    failed |= fclose(f) != 0;
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    f = fopen(path, "w");
    if (!f) return -1;
    failed |= fputs("{\"weight_map\":{\"target.weight\":\"tiny.safetensors\","
        "\"draft.weight\":\"tiny.safetensors\"}}", f) == EOF;
    failed |= fclose(f) != 0;
    snprintf(path, sizeof(path), "%s/rank04.core.blob", dir);
    f = fopen(path, "wb");
    if (!f) return -1;
    failed |= fwrite(compact, sizeof(compact), 1, f) != 1;
    failed |= fclose(f) != 0;
    snprintf(path, sizeof(path), "%s/rank04.core.manifest", dir);
    f = fopen(path, "w");
    if (!f) return -1;
    failed |= fputs("R target.weight 0 16 0\nC target.weight 8 2 4 16\n", f) == EOF;
    failed |= fclose(f) != 0;
    return failed ? -1 : 0;
}

static int check_tensor(glm53f_st_context *ctx, const char *name,
        const uint16_t *row, const uint16_t *columns, int required, int *cases) {
    uint16_t out[8];
    for (int column = 0; column < 2; ++column) {
        memset(out, 0xa5, sizeof(out));
        int rc = column ? glm53f_st_read_columns(ctx, name, 8, 2, 4, out) :
            glm53f_st_read(ctx, name, 0, out, sizeof(out));
        ++*cases;
        int failed = required ? rc == 0 : rc != 0 ||
            memcmp(out, column ? columns : row, column ? 8 : sizeof(out));
        if (failed) fprintf(stderr, "REPACK_POLICY_FAIL tensor=%s columns=%d required=%d rc=%d\n",
                            name, column, required, rc);
        if (failed) return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[2], "0") && strcmp(argv[2], "1"))) return 2;
    char dir[4096], path[4160];
    if (snprintf(dir, sizeof(dir), "%s/glm53f-repack.XXXXXX", argv[1]) >= (int)sizeof(dir) ||
        !mkdtemp(dir)) return 2;
    int failed = write_fixture(dir), cases = 0;
    failed |= setenv("GLM53F_REPACK_DIR", dir, 1) || setenv("PMIX_RANK", "4", 1) ||
        setenv("GLM53F_REPACK_REQUIRE", argv[2], 1);
    unsetenv("GLM53F_ST_PAYLOAD_DIR");
    unsetenv("GLM53F_REPACK_TRACE_DIR");
    unsetenv("GLM53F_REPACK_TRACE_ONLY");
    const uint16_t target[] = {101,102,103,104,105,106,107,108};
    const uint16_t target_columns[] = {202,203,206,207};
    const uint16_t draft[] = {21,22,23,24,25,26,27,28};
    const uint16_t draft_columns[] = {22,23,26,27};
    glm53f_st_context *ctx = failed ? NULL : glm53f_st_open(dir);
    if (!ctx) failed = 1;
    if (ctx) {
        failed |= check_tensor(ctx, "target.weight", target, target_columns, 0, &cases);
        failed |= check_tensor(ctx, "draft.weight", draft, draft_columns, atoi(argv[2]), &cases);
        /* Target setup initialized the cache. Draft construction temporarily
         * allows missing core entries, then target loading becomes strict. */
        const char *policies[] = {"0", "1", "0", "1"};
        for (size_t i = 0; i < sizeof(policies) / sizeof(policies[0]); ++i) {
            failed |= setenv("GLM53F_REPACK_REQUIRE", policies[i], 1);
            failed |= check_tensor(ctx, "draft.weight", draft, draft_columns, atoi(policies[i]), &cases);
            failed |= check_tensor(ctx, "target.weight", target, target_columns, 0, &cases);
        }
        glm53f_st_close(ctx);
    }
    const char *files[] = {"tiny.safetensors", "model.safetensors.index.json",
                           "rank04.core.blob", "rank04.core.manifest"};
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
        if (snprintf(path, sizeof(path), "%s/%s", dir, files[i]) >= (int)sizeof(path) ||
            unlink(path)) failed = 1;
    }
    if (rmdir(dir)) failed = 1;
    printf("GLM53F_REPACK_POLICY initial_required=%s cases=%d checkpoint_fallback_and_strict_restore=%s %s\n",
           argv[2], cases, failed ? "FAIL" : "BIT_EXACT", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
