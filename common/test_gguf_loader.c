#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#define DEFAULT_SNIP_N 10

static void print_arr_elem(const void *data, uint32_t type, uint64_t idx) {
    switch (type) {
        case GGUF_TYPE_UINT8:   printf("%" PRIu8,  ((const uint8_t *)data)[idx]);  break;
        case GGUF_TYPE_INT8:    printf("%" PRId8,  ((const int8_t *)data)[idx]);   break;
        case GGUF_TYPE_UINT16:  printf("%" PRIu16, ((const uint16_t *)data)[idx]); break;
        case GGUF_TYPE_INT16:   printf("%" PRId16, ((const int16_t *)data)[idx]);  break;
        case GGUF_TYPE_UINT32:  printf("%" PRIu32, ((const uint32_t *)data)[idx]); break;
        case GGUF_TYPE_INT32:   printf("%" PRId32, ((const int32_t *)data)[idx]);  break;
        case GGUF_TYPE_FLOAT32: printf("%g",       ((const float *)data)[idx]);    break;
        case GGUF_TYPE_BOOL:    printf("%s", ((const uint8_t *)data)[idx] ? "true" : "false"); break;
        case GGUF_TYPE_UINT64:  printf("%" PRIu64, ((const uint64_t *)data)[idx]); break;
        case GGUF_TYPE_INT64:   printf("%" PRId64, ((const int64_t *)data)[idx]);  break;
        case GGUF_TYPE_FLOAT64: printf("%g",       ((const double *)data)[idx]);   break;
        case GGUF_TYPE_STRING: {
            const gguf_str *s = &((const gguf_str *)data)[idx];
            printf("\"%.60s%s\"", s->str, s->len > 60 ? "..." : "");
            break;
        }
        default: printf("?"); break;
    }
}

static void print_array_snipped(const void *data, uint32_t elem_type, uint64_t n, int snip_n) {
    if (snip_n <= 0) snip_n = DEFAULT_SNIP_N;
    uint64_t sn = (uint64_t)snip_n;

    printf("{");
    if (n <= sn * 2) {
        /* small enough to print all */
        for (uint64_t i = 0; i < n; i++) {
            if (i) printf(", ");
            print_arr_elem(data, elem_type, i);
        }
    } else {
        /* first N */
        for (uint64_t i = 0; i < sn; i++) {
            if (i) printf(", ");
            print_arr_elem(data, elem_type, i);
        }
        printf(", ... (%" PRIu64 " items omitted) ..., ", n - sn * 2);
        /* last N */
        for (uint64_t i = n - sn; i < n; i++) {
            if (i != n - sn) printf(", ");
            print_arr_elem(data, elem_type, i);
        }
    }
    printf("}");
}

static void print_kv(const gguf_kv *kv, int snip_n) {
    printf("  %-50s [%-7s] = ", kv->key.str, gguf_type_name(kv->type));
    switch (kv->type) {
        case GGUF_TYPE_UINT8:   printf("%" PRIu8,  kv->value.u8);  break;
        case GGUF_TYPE_INT8:    printf("%" PRId8,  kv->value.i8);  break;
        case GGUF_TYPE_UINT16:  printf("%" PRIu16, kv->value.u16); break;
        case GGUF_TYPE_INT16:   printf("%" PRId16, kv->value.i16); break;
        case GGUF_TYPE_UINT32:  printf("%" PRIu32, kv->value.u32); break;
        case GGUF_TYPE_INT32:   printf("%" PRId32, kv->value.i32); break;
        case GGUF_TYPE_FLOAT32: printf("%g",       kv->value.f32); break;
        case GGUF_TYPE_BOOL:    printf("%s", kv->value.b ? "true" : "false"); break;
        case GGUF_TYPE_UINT64:  printf("%" PRIu64, kv->value.u64); break;
        case GGUF_TYPE_INT64:   printf("%" PRId64, kv->value.i64); break;
        case GGUF_TYPE_FLOAT64: printf("%g",       kv->value.f64); break;
        case GGUF_TYPE_STRING:
            printf("\"%.100s%s\"", kv->value.str.str,
                   kv->value.str.len > 100 ? "..." : "");
            break;
        case GGUF_TYPE_ARRAY:
            printf("[%s x %" PRIu64 "] ", gguf_type_name(kv->value.arr.type), kv->value.arr.n);
            if (kv->value.arr.n > 0)
                print_array_snipped(kv->value.arr.data, kv->value.arr.type, kv->value.arr.n, snip_n);
            break;
        default: printf("?"); break;
    }
    printf("\n");
}

int main(int argc, char **argv) {
    const char *path = "/mnt/disk1/models/qwen3-vl-embedding-8b-q4_k_m.gguf";
    int snip_n = DEFAULT_SNIP_N;

    /* parse args: [path] [-n snip_count] */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            snip_n = atoi(argv[++i]);
            if (snip_n <= 0) snip_n = DEFAULT_SNIP_N;
        } else {
            path = argv[i];
        }
    }

    printf("Loading: %s\n", path);
    printf("Array snip: first/last %d items\n", snip_n);
    gguf_context *ctx = gguf_open(path, 1);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF file\n"); return 1; }

    printf("Version:   %u\n", ctx->version);
    printf("KV pairs:  %" PRIu64 "\n", ctx->n_kv);
    printf("Tensors:   %" PRIu64 "\n", ctx->n_tensors);
    printf("Alignment: %u\n", ctx->alignment);
    printf("Data off:  %zu\n", ctx->data_offset);
    printf("\n--- KV pairs ---\n");

    for (uint64_t i = 0; i < ctx->n_kv; i++)
        print_kv(&ctx->kv[i], snip_n);

    printf("\n--- Tensors (first 20) ---\n");
    uint64_t show = ctx->n_tensors < 20 ? ctx->n_tensors : 20;
    for (uint64_t i = 0; i < show; i++) {
        const gguf_tensor_info *ti = &ctx->tensors[i];
        printf("  [%3" PRIu64 "] %-60s %-6s [", i, ti->name.str, ggml_type_name(ti->type));
        for (uint32_t d = 0; d < ti->n_dims; d++)
            printf("%s%" PRIu64, d ? "," : "", ti->dims[d]);
        printf("]  size=%zu  data=%s\n",
               gguf_tensor_size(ctx, (int)i),
               gguf_tensor_data(ctx, (int)i) ? "OK" : "NULL");
    }

    gguf_close(ctx);
    printf("\nDone.\n");
    return 0;
}
