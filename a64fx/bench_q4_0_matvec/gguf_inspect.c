/* Dump gguf tensors (name, shape, type) + all metadata KV. For MTP arch RE. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [tensor_substr]\n", argv[0]); return 1; }
    const char *filt = argc > 2 ? argv[2] : NULL;
    gguf_context *g = gguf_open(argv[1], 1);
    if (!g) { fprintf(stderr, "open fail\n"); return 1; }
    printf("=== METADATA (%llu kv) ===\n", (unsigned long long)g->n_kv);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const char *k = g->kv[i].key.str;
        if (!k) continue;
        if (strstr(k, "tokenizer")) continue;  /* skip huge token arrays */
        gguf_kv *kv = &g->kv[i];
        printf("  %-50s ", k);
        switch (kv->type) {  /* 0u8 1i8 2u16 3i16 4u32 5i32 6f32 7bool 8str 10u64 11i64 12f64 */
            case 0: printf("%u\n", kv->value.u8); break;
            case 1: printf("%d\n", kv->value.i8); break;
            case 2: printf("%u\n", kv->value.u16); break;
            case 3: printf("%d\n", kv->value.i16); break;
            case 4: printf("%u\n", kv->value.u32); break;
            case 5: printf("%d\n", kv->value.i32); break;
            case 6: printf("%g\n", kv->value.f32); break;
            case 7: printf("%u (bool)\n", kv->value.b); break;
            case 8: printf("\"%s\"\n", kv->value.str.str ? kv->value.str.str : ""); break;
            case 10: printf("%llu\n", (unsigned long long)kv->value.u64); break;
            case 11: printf("%lld\n", (long long)kv->value.i64); break;
            case 12: printf("%g\n", kv->value.f64); break;
            default: printf("(type %u)\n", kv->type); break;
        }
    }
    printf("=== %llu TENSORS ===\n", (unsigned long long)g->n_tensors);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *nm = g->tensors[i].name.str;
        if (filt && (!nm || !strstr(nm, filt))) continue;
        gguf_tensor_info *t = &g->tensors[i];
        printf("  %-44s [", nm ? nm : "?");
        for (uint32_t d = 0; d < t->n_dims; d++)
            printf("%llu%s", (unsigned long long)t->dims[d], d + 1 < t->n_dims ? "," : "");
        printf("] type=%s\n", ggml_type_name(t->type));
    }
    return 0;
}
