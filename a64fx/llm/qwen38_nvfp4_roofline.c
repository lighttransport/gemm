/* Metadata-only byte and operation inventory for one-token Qwen3.8 decode. */
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *name;
    uint64_t tensors;
    uint64_t source_bytes;
    uint64_t packed_bytes;
    double operations;
} roof_stage;

enum {
    ROOF_ATTN_QKV,
    ROOF_ATTN_OUT,
    ROOF_SSM_IN,
    ROOF_SSM_OUT,
    ROOF_FFN_GATEUP,
    ROOF_FFN_DOWN,
    ROOF_HEAD,
    ROOF_NEXTN,
    ROOF_OTHER,
    ROOF_STAGE_COUNT
};

static int stage_for_name(const char *name) {
    if (!strcmp(name, "output.weight")) return ROOF_HEAD;
    if (!strncmp(name, "blk.64.", 7) || !strncmp(name, "nextn.", 6))
        return ROOF_NEXTN;
    if (!strncmp(name, "blk.", 4)) {
        if (strstr(name, ".attn_qkv.weight") ||
            strstr(name, ".attn_gate.weight") ||
            strstr(name, ".ssm_alpha.weight") ||
            strstr(name, ".ssm_beta.weight")) return ROOF_SSM_IN;
        if (strstr(name, ".ssm_out.weight")) return ROOF_SSM_OUT;
        if (strstr(name, ".attn_q.weight") ||
            strstr(name, ".attn_k.weight") ||
            strstr(name, ".attn_v.weight")) return ROOF_ATTN_QKV;
        if (strstr(name, ".attn_output.weight")) return ROOF_ATTN_OUT;
        if (strstr(name, ".ffn_gate.weight") ||
            strstr(name, ".ffn_up.weight")) return ROOF_FFN_GATEUP;
        if (strstr(name, ".ffn_down.weight")) return ROOF_FFN_DOWN;
    }
    return ROOF_OTHER;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    gguf_context *g = gguf_open_multi(argv[1], 3);
    if (!g) return 1;
    roof_stage stages[ROOF_STAGE_COUNT] = {
        {"attn_qkv", 0, 0, 0, 0}, {"attn_out", 0, 0, 0, 0},
        {"ssm_in", 0, 0, 0, 0}, {"ssm_out", 0, 0, 0, 0},
        {"ffn_gateup", 0, 0, 0, 0}, {"ffn_down", 0, 0, 0, 0},
        {"lm_head", 0, 0, 0, 0}, {"nextn", 0, 0, 0, 0},
        {"other", 0, 0, 0, 0}
    };
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_info *ti = &g->tensors[i];
        int stage = stage_for_name(ti->name.str);
        roof_stage *s = &stages[stage];
        size_t source = gguf_tensor_size(g, (int)i);
        size_t packed = source;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) elements *= ti->dims[d];
        if (ti->type == GGML_TYPE_NVFP4 && ti->n_dims >= 2 && ti->dims[0] &&
            ti->dims[0] % 64 == 0 && elements / ti->dims[0] % 8 == 0)
            packed = source * 4 / 3;
        s->tensors++;
        s->source_bytes += source;
        s->packed_bytes += packed;
        if (stage != ROOF_OTHER && ti->n_dims >= 2)
            s->operations += 2.0 * (double)elements;
    }
    puts("stage,tensors,source_gb,packed_gb,giga_ops,gbps_for_40_source,gbps_for_40_packed");
    for (int j = 0; j < ROOF_STAGE_COUNT; j++) {
        const roof_stage *s = &stages[j];
        printf("%s,%llu,%.6f,%.6f,%.6f,%.3f,%.3f\n", s->name,
               (unsigned long long)s->tensors, s->source_bytes / 1e9,
               s->packed_bytes / 1e9, s->operations / 1e9,
               s->source_bytes * 40.0 / 1e9,
               s->packed_bytes * 40.0 / 1e9);
    }
    uint64_t source = 0, packed = 0, tensors = 0;
    double operations = 0;
    for (int j = 0; j <= ROOF_HEAD; j++) {
        source += stages[j].source_bytes;
        packed += stages[j].packed_bytes;
        tensors += stages[j].tensors;
        operations += stages[j].operations;
    }
    printf("trunk_total,%llu,%.6f,%.6f,%.6f,%.3f,%.3f\n",
           (unsigned long long)tensors, source / 1e9, packed / 1e9,
           operations / 1e9, source * 40.0 / 1e9, packed * 40.0 / 1e9);
    gguf_close(g);
    return 0;
}
