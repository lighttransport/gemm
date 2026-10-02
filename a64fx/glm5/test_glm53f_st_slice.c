#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include <assert.h>
#include <sys/stat.h>

typedef struct { int calls, closed; unsigned char bias; } fixture;
static int read_slice(void *context, const char *kind, const char *name,
        size_t a, size_t b, size_t c, void *dst, size_t bytes) {
    fixture *f = context;
    assert(!strcmp(name, "matrix"));
    ++f->calls;
    if (kind[0] == 'R') {
        assert(bytes == b && !c);
        for (size_t i = 0; i < bytes; ++i) ((unsigned char *)dst)[i] = (unsigned char)(a + i + f->bias);
    } else {
        assert(kind[0] == 'C' && bytes == 3 * c);
        for (size_t i = 0; i < bytes; ++i) ((unsigned char *)dst)[i] = (unsigned char)((i / c) * a + b + i % c + f->bias);
    }
    return 0;
}
static void close_slice(void *context) { ++((fixture *)context)->closed; }
int main(int argc, char **argv) {
    assert(argc == 2);
    char dir[4096], path[4096];
    assert(snprintf(dir, sizeof(dir), "%s/slice-XXXXXX", argv[1]) < (int)sizeof(dir));
    assert(mkdtemp(dir));
    assert(snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir) < (int)sizeof(path));
    FILE *f = fopen(path, "w"); assert(f);
    assert(fputs("{\"weight_map\":{\"matrix\":\"part.safetensors\"}}", f) >= 0); assert(!fclose(f));
    assert(snprintf(path, sizeof(path), "%s/part.safetensors", dir) < (int)sizeof(path));
    const char *header = "{\"matrix\":{\"dtype\":\"U8\",\"shape\":[3,8],\"data_offsets\":[0,24]}}";
    uint64_t length = strlen(header);
    f = fopen(path, "wb"); assert(f);
    assert(fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, length, f) == length);
    unsigned char data[24], result[24];
    for (int i = 0; i < 24; ++i) data[i] = (unsigned char)i;
    assert(fwrite(data, 1, 24, f) == 24 && !fclose(f));
    glm53f_st_context *ordinary = glm53f_st_open(dir), *owned = glm53f_st_open(dir);
    assert(ordinary && owned);
    fixture state = {0, 0, 100};
    owned->slice_reader = read_slice; owned->slice_close = close_slice; owned->slice_context = &state;
    for (int pass = 0; pass < 3; ++pass) {
        assert(!glm53f_st_read(ordinary, "matrix", 4, result, 12));
        assert(!memcmp(result, data + 4, 12));
        assert(!glm53f_st_read(owned, "matrix", 4, result, 12));
        for (int i = 0; i < 12; ++i) assert(result[i] == i + 104);
        assert(!glm53f_st_read_columns(ordinary, "matrix", 8, 2, 3, result));
        for (int i = 0; i < 9; ++i) assert(result[i] == (i / 3) * 8 + 2 + i % 3);
        assert(!glm53f_st_read_columns(owned, "matrix", 8, 2, 3, result));
        for (int i = 0; i < 9; ++i) assert(result[i] == (i / 3) * 8 + 102 + i % 3);
    }
    assert(state.calls == 6);
    assert(glm53f_st_read(owned, "matrix", 23, result, 2) == -1);
    assert(glm53f_st_read(owned, "missing", 0, result, 1) == -1);
    assert(glm53f_st_read_columns(owned, "matrix", 8, 7, 2, result) == -1);
    assert(glm53f_st_read_columns(owned, "matrix", 7, 0, 1, result) == -1);
    assert(state.calls == 6);
    glm53f_st_close(owned); assert(state.closed == 1);
    assert(!glm53f_st_read(ordinary, "matrix", 0, result, 24) && !memcmp(result, data, 24));
    glm53f_st_close(ordinary);
    assert(!unlink(path));
    assert(snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir) < (int)sizeof(path));
    assert(!unlink(path) && !rmdir(dir));
    puts("GLM53F_ST_SLICE_PASS context_isolation=1 legacy_reads=7 owned_reads=6 invalid_rejected=4 close=1");
    return 0;
}
