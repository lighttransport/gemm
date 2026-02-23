/*
 * test_safetensors.c - Test safetensors loader and JSON parser
 *
 * Build:
 *   gcc -O2 -o test_safetensors test_safetensors.c -lm
 *
 * Usage:
 *   ./test_safetensors ../models/da3-small/model.safetensors [../models/da3-small/config.json]
 */
#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void test_json_parser(const char *config_path) {
    printf("--- JSON parser test: %s ---\n", config_path);
    FILE *f = fopen(config_path, "rb");
    if (!f) { printf("  (skipped, file not found)\n\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz);
    size_t nr = fread(buf, 1, sz, f);
    (void)nr;
    fclose(f);

    json_val *root = json_parse(buf, (int)sz);
    assert(root && root->type == JSON_OBJECT);
    printf("  root keys: %d\n", root->obj.count);

    json_val *name = json_obj_get(root, "model_name");
    assert(name && name->type == JSON_STRING);
    printf("  model_name: \"%s\"\n", name->str.ptr);

    json_val *config = json_obj_get(root, "config");
    assert(config && config->type == JSON_OBJECT);

    json_val *head = json_obj_get(config, "head");
    assert(head && head->type == JSON_OBJECT);
    json_val *dim_in = json_obj_get(head, "dim_in");
    assert(dim_in && dim_in->type == JSON_NUMBER);
    printf("  config.head.dim_in: %d\n", (int)dim_in->num);

    json_val *out_channels = json_obj_get(head, "out_channels");
    assert(out_channels && out_channels->type == JSON_ARRAY && out_channels->arr.count == 4);
    printf("  config.head.out_channels: [");
    for (int i = 0; i < out_channels->arr.count; i++)
        printf("%s%d", i ? ", " : "", (int)out_channels->arr.items[i].num);
    printf("]\n");

    json_val *net = json_obj_get(config, "net");
    assert(net && net->type == JSON_OBJECT);
    json_val *cat = json_obj_get(net, "cat_token");
    assert(cat && cat->type == JSON_TRUE);
    printf("  config.net.cat_token: true\n");

    json_free(root);
    free(buf);
    printf("  JSON parser: OK\n\n");
}

static void test_dtype_size(void) {
    printf("--- dtype_size test ---\n");
    assert(safetensors_dtype_size("F32") == 4);
    assert(safetensors_dtype_size("F16") == 2);
    assert(safetensors_dtype_size("BF16") == 2);
    assert(safetensors_dtype_size("F64") == 8);
    assert(safetensors_dtype_size("I8") == 1);
    assert(safetensors_dtype_size("I32") == 4);
    assert(safetensors_dtype_size("BOOL") == 1);
    assert(safetensors_dtype_size("UNKNOWN") == 0);
    printf("  dtype_size: OK\n\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.safetensors> [config.json]\n", argv[0]);
        return 1;
    }

    test_dtype_size();

    /* JSON parser test */
    if (argc >= 3) {
        test_json_parser(argv[2]);
    } else {
        /* Try config.json in same directory as model */
        char config_path[512];
        const char *slash = strrchr(argv[1], '/');
        if (slash) {
            int dir_len = (int)(slash - argv[1]);
            snprintf(config_path, sizeof(config_path), "%.*s/config.json", dir_len, argv[1]);
        } else {
            snprintf(config_path, sizeof(config_path), "config.json");
        }
        test_json_parser(config_path);
    }

    /* Safetensors loader test */
    printf("--- safetensors loader test: %s ---\n", argv[1]);
    st_context *ctx = safetensors_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to open safetensors\n"); return 1; }

    printf("  n_tensors: %d\n", ctx->n_tensors);
    printf("  data section at: %p\n", (void *)ctx->data);

    /* Print first 10 tensors */
    int show = ctx->n_tensors < 10 ? ctx->n_tensors : 10;
    printf("\n  First %d tensors:\n", show);
    for (int i = 0; i < show; i++) {
        printf("    [%3d] %-60s  dtype=%-4s  shape=[",
               i, safetensors_name(ctx, i), safetensors_dtype(ctx, i));
        int nd = safetensors_ndims(ctx, i);
        const uint64_t *shape = safetensors_shape(ctx, i);
        for (int d = 0; d < nd; d++)
            printf("%s%lu", d ? "," : "", (unsigned long)shape[d]);
        printf("]  nbytes=%zu\n", safetensors_nbytes(ctx, i));
    }

    /* Verify tensor data consistency */
    size_t total_bytes = 0;
    for (int i = 0; i < ctx->n_tensors; i++) {
        total_bytes += safetensors_nbytes(ctx, i);
        /* Check nbytes matches shape * dtype_size */
        size_t ds = safetensors_dtype_size(safetensors_dtype(ctx, i));
        if (ds > 0) {
            int nd = safetensors_ndims(ctx, i);
            const uint64_t *shape = safetensors_shape(ctx, i);
            size_t expected = ds;
            for (int d = 0; d < nd; d++) expected *= shape[d];
            if (expected != safetensors_nbytes(ctx, i)) {
                fprintf(stderr, "  ERROR: tensor %d (%s) nbytes=%zu expected=%zu\n",
                        i, safetensors_name(ctx, i), safetensors_nbytes(ctx, i), expected);
                safetensors_close(ctx);
                return 1;
            }
        }
    }
    printf("\n  total tensor bytes: %zu (%.1f MB)\n", total_bytes, total_bytes / (1024.0 * 1024.0));

    /* Find test */
    const char *test_name = "model.backbone.pretrained.blocks.0.attn.proj.bias";
    int idx = safetensors_find(ctx, test_name);
    printf("\n  find(\"%s\") = %d\n", test_name, idx);
    assert(idx >= 0);
    printf("    dtype=%s ndims=%d nbytes=%zu\n",
           safetensors_dtype(ctx, idx), safetensors_ndims(ctx, idx), safetensors_nbytes(ctx, idx));

    /* Peek at first few float values */
    float *fdata = (float *)safetensors_data(ctx, idx);
    printf("    first 4 values: [%.6f, %.6f, %.6f, %.6f]\n",
           fdata[0], fdata[1], fdata[2], fdata[3]);

    /* Negative find test */
    assert(safetensors_find(ctx, "nonexistent.tensor.name") == -1);
    printf("  find(nonexistent) = -1: OK\n");

    safetensors_close(ctx);
    printf("\n  safetensors loader: OK\n");
    printf("\nAll tests passed.\n");
    return 0;
}
