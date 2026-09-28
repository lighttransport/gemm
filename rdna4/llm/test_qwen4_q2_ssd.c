#define _GNU_SOURCE
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "qwen4_ple_disk.h"

#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    block_q2_0 block = { .d = 0x3c00 };
    float values[64];
    for (int i = 0; i < 16; ++i) block.qs[i] = 0xe4;
    if (dequant_row(GGML_TYPE_Q2_0, &block, values, 64)) return 1;
    for (int i = 0; i < 64; ++i)
        if (values[i] != (float)((i % 4) - 1)) return 1;
    if (argc == 1) {
        puts("Q2_0 dequant OK");
        return 0;
    }

    gguf_context *gguf = gguf_open(argv[1], 1);
    if (!gguf) return 1;
    int index = -1;
    for (int i = 0; i < (int)gguf->n_tensors; ++i)
        if (!strcmp(gguf->tensors[i].name.str, "per_layer_token_embd.weight")) index = i;
    if (index < 0) return 1;
    const gguf_tensor_info *tensor = &gguf->tensors[index];
    int dim = (int)tensor->dims[0];
    qwen4_ple_disk *disk = qwen4_ple_disk_open(gguf, tensor, dim, 16);
    if (!disk) return 1;
    float *actual = (float *)calloc((size_t)16 * dim, sizeof(float));
    float *expected = (float *)calloc((size_t)dim, sizeof(float));
    const unsigned char *mapped = (const unsigned char *)gguf_tensor_data(gguf, index);
    if (!actual || !expected || !mapped) return 1;
    uint64_t rows[16];
    int ok = 1;
    for (int pass = 0; pass < 2; ++pass) {
        for (int h = 0; h < 16; ++h)
            rows[h] = ((uint64_t)h * 131071 + (uint64_t)pass * 7919) % disk->row_count;
        if (qwen4_ple_disk_gather(disk, rows, actual)) { ok = 0; break; }
        for (int h = 0; h < 16; ++h) {
            if (dequant_row(tensor->type, mapped + rows[h] * disk->row_bytes,
                            expected, dim) ||
                memcmp(expected, actual + (size_t)h * dim,
                       (size_t)dim * sizeof(float))) ok = 0;
        }
    }
    rows[0] = disk->row_count;
    if (qwen4_ple_disk_gather(disk, rows, actual) == 0) ok = 0;
    printf("Q2_0 dequant and PLE SSD rows: %s\n", ok ? "OK" : "FAIL");
    free(expected);
    free(actual);
    qwen4_ple_disk_close(disk);
    gguf_close(gguf);
    return ok ? 0 : 1;
}
