/* CPU-only exercise of the exact native quantized-weight loader. */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "quant_weights.h"

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) return 2;
    size_t rows = strtoull(argv[2], NULL, 10), cols = strtoull(argv[3], NULL, 10);
    uint16_t *out = NULL;
    void *fat = NULL;
    size_t output_bytes = rows * cols * 2;
    if (argc == 6 && !strcmp(argv[5], "--quantize")) {
        st_context *st = safetensors_open(argv[1]);
        if (!st) return 1;
        int index = safetensors_find(st, "weight");
        if (index >= 0 && safetensors_ndims(st, index) == 2 &&
            safetensors_shape(st, index)[0] == rows && safetensors_shape(st, index)[1] == cols)
            out = q21_quantize_matrix_on_load(st, index);
        safetensors_close(st);
    } else if (argc == 6 && !strcmp(argv[5], "--fat")) {
        fat = q21_read_int8_fat(argv[1], rows, cols, &output_bytes);
    } else if (argc == 5) out = q21_read_int8_matrix(argv[1], rows, cols);
    if (!out && !fat) return 1;
    FILE *fp = fopen(argv[4], "wb");
    if (!fp) { free(out); free(fat); return 1; }
    int rc = fwrite(fat ? fat : (void *)out, 1, output_bytes, fp) == output_bytes ? 0 : 1;
    if (fclose(fp)) rc = 1;
    free(out);
    free(fat);
    return rc;
}
