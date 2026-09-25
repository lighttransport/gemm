/* Convert the node-local shared-expert image from GGUF into its existing
 * block-128 FP8 production layout.  The shared safetensors source is never
 * modified; this utility accepts only a /local stage directory. */
#define main glm53f_q2_core_patch_embedded_main
#include "glm53f_q2_core_patch.c"
#undef main

typedef struct {
    uint64_t offset;
    int rows, cols, begin, global;
} stage_entry;

static int find_stage_entry(const char *manifest, const char *name,
                            stage_entry *out) {
    FILE *f = fopen(manifest, "r");
    char line[2048], got[1024], dtype[32], axis[32];
    unsigned long long offset;
    int nd, rows, cols, begin, global;
    if (!f) return -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%llu %31s %d %d %d %31s begin=%d global=%d %1023s",
                   &offset, dtype, &nd, &rows, &cols, axis,
                   &begin, &global, got) != 9 || strcmp(got, name)) continue;
        *out = (stage_entry){offset, rows, cols, begin, global};
        fclose(f);
        return 0;
    }
    fclose(f); errno = ENOENT; return -1;
}

static int patch_shared_layer(int blob_fd, const char *manifest,
                              const gguf_context *g, int layer) {
    char gate_name[768], scale_name[768], down_name[768], dscale_name[768];
    char gguf_name[256];
    stage_entry gate, scale, down, dscale;
    snprintf(gate_name, sizeof(gate_name),
             "model.language_model.layers.%d.mlp.shared_experts.gate_up_fused.weight", layer);
    snprintf(scale_name, sizeof(scale_name),
             "model.language_model.layers.%d.mlp.shared_experts.gate_up_fused.weight_scale_inv", layer);
    snprintf(down_name, sizeof(down_name),
             "model.language_model.layers.%d.mlp.shared_experts.down_proj.weight", layer);
    snprintf(dscale_name, sizeof(dscale_name),
             "model.language_model.layers.%d.mlp.shared_experts.down_proj.weight_scale_inv", layer);
    if (find_stage_entry(manifest, gate_name, &gate) ||
        find_stage_entry(manifest, scale_name, &scale) ||
        find_stage_entry(manifest, down_name, &down) ||
        find_stage_entry(manifest, dscale_name, &dscale)) return -1;
    int rows = gate.rows / 2;
    if (gate.rows != 2 * rows || gate.cols != HIDDEN || gate.begin % BLOCK ||
        rows % BLOCK || scale.rows != 2 * (rows / BLOCK) ||
        scale.cols != HIDDEN / BLOCK || down.rows != HIDDEN ||
        down.cols != rows || down.begin != gate.begin ||
        dscale.rows != HIDDEN / BLOCK || dscale.cols != rows / BLOCK)
        return -1;

    image_entry weight = {'R', "", 0, 0, 0, gate.offset};
    image_entry weights_scale = {'R', "", 0, 0, 0, scale.offset};
    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_shexp.weight", layer);
    tensor_ref source = find_tensor(g, gguf_name);
    if (!source.info || quantize_dense_slice(blob_fd, &weight, &weights_scale,
            &source, (uint64_t)gate.begin, (uint64_t)rows, 0, HIDDEN)) return -1;

    weight.blob = gate.offset + (uint64_t)rows * HIDDEN;
    weights_scale.blob = scale.offset +
        (uint64_t)(rows / BLOCK) * (HIDDEN / BLOCK) * sizeof(float);
    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up_shexp.weight", layer);
    source = find_tensor(g, gguf_name);
    if (!source.info || quantize_dense_slice(blob_fd, &weight, &weights_scale,
            &source, (uint64_t)gate.begin, (uint64_t)rows, 0, HIDDEN)) return -1;

    weight.blob = down.offset;
    weights_scale.blob = dscale.offset;
    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down_shexp.weight", layer);
    source = find_tensor(g, gguf_name);
    if (!source.info || quantize_dense_slice(blob_fd, &weight, &weights_scale,
            &source, 0, HIDDEN, (uint64_t)down.begin, (uint64_t)rows)) return -1;
    return 0;
}

int main(int argc, char **argv) {
    char manifest[4096], blob[4096];
    int rank = -1, blob_fd = -1, rc = 1;
    gguf_context *g = NULL;
    if (argc != 4 || strncmp(argv[2], "/local/", 7) ||
        (rank = atoi(argv[3])) < 0 || rank >= RANKS) {
        fprintf(stderr, "usage: %s MODEL-00001-of-00004.gguf /local/SHARED_DIR RANK\n",
                argv[0]);
        return 2;
    }
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", argv[2], rank);
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", argv[2], rank);
    if (!(g = gguf_open_multi(argv[1], 3)) || (blob_fd = open(blob, O_RDWR)) < 0)
        goto done;
    for (int layer = 3; layer < 45; ++layer) {
        if (patch_shared_layer(blob_fd, manifest, g, layer)) goto done;
        printf("GLM53F_Q2_SHARED_PATCH rank=%d layer=%d\n", rank, layer);
        fflush(stdout);
    }
    if (fsync(blob_fd)) goto done;
    printf("SENTINEL glm53f_q2_shared_patch=OK rank=%d layers=3:45\n", rank);
    rc = 0;
done:
    if (rc) fprintf(stderr, "glm53f_q2_shared_patch failed rank=%d: %s\n",
                    rank, errno ? strerror(errno) : "contract failure");
    if (blob_fd >= 0) close(blob_fd);
    if (g) gguf_close(g);
    return rc;
}
