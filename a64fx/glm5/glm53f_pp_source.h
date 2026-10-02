#ifndef GLM53F_PP_SOURCE_H
#define GLM53F_PP_SOURCE_H
#include <stdint.h>
#include <sys/stat.h>
/* Requires gguf_loader.h. This is a reuse stamp, not a content digest;
 * tensor payload hashes are independently checked during resident loading. */
static inline int glm53f_pp_source_stamp(const gguf_context *g,
        const char *path, uint64_t *identity) {
    if (!g || !path || !identity || !g->n_shards) return -1;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        hash ^= *p; hash *= UINT64_C(1099511628211);
    }
    for (int i = 0; i < g->n_shards; ++i) {
        struct stat st;
        if (fstat(g->shards[i]->fd, &st)) return -1;
        const uint64_t fields[] = {(uint64_t)st.st_size,
            (uint64_t)st.st_mtim.tv_sec, (uint64_t)st.st_mtim.tv_nsec,
            g->shards[i]->n_tensors, (uint64_t)g->shards[i]->data_offset};
        for (size_t j = 0; j < sizeof(fields) / sizeof(fields[0]); ++j)
            for (int byte = 0; byte < 8; ++byte) {
                hash ^= (fields[j] >> (8 * byte)) & 255;
                hash *= UINT64_C(1099511628211);
            }
    }
    *identity = hash; return 0;
}
#endif
