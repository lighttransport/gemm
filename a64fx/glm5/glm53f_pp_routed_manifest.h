#ifndef GLM53F_PP_ROUTED_MANIFEST_H
#define GLM53F_PP_ROUTED_MANIFEST_H
#include "glm53f_pp_manifest.h"
#include "glm53f_iq_bridge.h"
#include <stdlib.h>
#include <stdint.h>
/* Validate contiguous native entries, exactly one owned part per expert,
 * source ownership and complete byte coverage. With data, check every payload
 * and the complete-image digest before inference can reference offsets. */
static inline int glm53f_pp_routed_validate(const char *manifest, const glm53f_dist *d,
        int first, int end, uint64_t image_bytes, const unsigned char *data) {
    if (first < 3 || end > 45 || first >= end ||
        glm53f_pp_manifest_check(manifest, "ROUTED", d, first, end)) return -1;
    FILE *f = fopen(manifest, "r"); if (!f) return -1;
    unsigned char *seen = calloc((size_t)(end - first) * 288 * 2, 1);
    if (!seen) { fclose(f); return -1; }
    char line[2048], type_name[32], name[256];
    uint64_t offset = 0, complete_hash = UINT64_C(1469598103934665603);
    size_t pending_bytes = 0; unsigned long long pending_offset = 0;
    int count = 0, pending = 0, complete = 0, failed = 0;
    while (fgets(line, sizeof(line), f) && !failed) {
        unsigned long long off, bytes, hash; int nd, rows, cols, part, expert, layer, named_expert;
        char suffix[128];
        if (line[0] != '#') {
            if (pending || sscanf(line, "%llu %31s %d %d %d part=%d source_expert=%d %255s",
                    &off, type_name, &nd, &rows, &cols, &part, &expert, name) != 8 ||
                sscanf(name, "model.language_model.layers.%d.mlp.experts.%d.%127s", &layer, &named_expert, suffix) != 3 ||
                layer < first || layer >= end || expert < 0 || expert >= 288 || expert != named_expert ||
                part < 0 || part >= 4 || (expert + part) % 4 != d->map.tp_rank || nd != 2) { failed = 1; break; }
            int type = !strcmp(type_name, "Q4_K") ? GLM53F_GGML_Q4_K :
                !strcmp(type_name, "Q5_K") ? GLM53F_GGML_Q5_K :
                !strcmp(type_name, "Q6_K") ? GLM53F_GGML_Q6_K :
                !strcmp(type_name, "IQ2_XS") ? GLM53F_GGML_IQ2_XS :
                !strcmp(type_name, "IQ3_XXS") ? GLM53F_GGML_IQ3_XXS :
                !strcmp(type_name, "IQ4_XS") ? GLM53F_GGML_IQ4_XS : -1;
            int kind = !strcmp(suffix, "gate_up_fused.weight") ? 0 : !strcmp(suffix, "down_proj.weight") ? 1 : -1;
            if (type < 0 || kind < 0 || rows != (kind ? 4096 : 1024) || cols != (kind ? 512 : 4096)) { failed = 1; break; }
            size_t index = ((size_t)(layer - first) * 288 + expert) * 2 + kind;
            size_t rb = glm53f_native_row_size(type, cols);
            if (seen[index] || !rb || (size_t)rows > SIZE_MAX / rb || off != offset) { failed = 1; break; }
            seen[index] = 1; pending_bytes = (size_t)rows * rb; pending_offset = off; pending = 1;
            if (offset > image_bytes || pending_bytes > image_bytes - offset) { failed = 1; break; }
        } else if (sscanf(line, "# PAYLOAD offset=%llu bytes=%llu fnv1a=%llx", &off, &bytes, &hash) == 3) {
            if (!pending || off != pending_offset || bytes != pending_bytes) { failed = 1; break; }
            if (data) {
                uint64_t actual = UINT64_C(1469598103934665603);
                for (size_t i = 0; i < pending_bytes; ++i) {
                    actual ^= data[offset + i]; actual *= UINT64_C(1099511628211);
                    complete_hash ^= data[offset + i]; complete_hash *= UINT64_C(1099511628211);
                }
                if (actual != hash) { failed = 1; break; }
            }
            offset += pending_bytes; pending = 0; ++count;
        } else if (sscanf(line, "# COMPLETE bytes=%llu fnv1a=%llx", &bytes, &hash) == 2) {
            if (complete || pending || bytes != image_bytes || offset != image_bytes || (data && hash != complete_hash)) failed = 1;
            complete = 1;
        }
    }
    failed |= ferror(f) || !complete || pending || count != (end - first) * 288 * 2;
    failed |= fclose(f) != 0; free(seen); return failed ? -1 : 0;
}
#endif
