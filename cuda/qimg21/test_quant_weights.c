/* CPU-only exercise of the exact native quantized-weight loader. */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "quant_weights.h"

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    size_t rows = strtoull(argv[2], NULL, 10), cols = strtoull(argv[3], NULL, 10);
    uint16_t *out = q21_read_int8_matrix(argv[1], rows, cols);
    if (!out) return 1;
    FILE *fp = fopen(argv[4], "wb");
    if (!fp) { free(out); return 1; }
    int rc = fwrite(out, 2, rows * cols, fp) == rows * cols ? 0 : 1;
    if (fclose(fp)) rc = 1;
    free(out);
    return rc;
}
