#define _GNU_SOURCE
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#include "qwen38_lowbit_model.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void test_model(int format, int arithmetic) {
    const int rows = 35, cols = format == Q38_LB_NVFP4 ? 128 : 65;
    size_t rb = format == Q38_LB_NVFP4 ? (size_t)cols / 64 * 36 : (size_t)cols * 2;
    size_t bytes = rows * rb, total = bytes + 32;
    unsigned char *source = calloc(1, total);
    float *fp = calloc((size_t)rows * cols, sizeof(float));
    float *row = malloc((size_t)cols * sizeof(float));
    float *x = malloc((size_t)cols * sizeof(float));
    size_t packed_bytes = q38_lowbit_bytes(format, rows, cols);
    void *packed = malloc(packed_bytes);
    CHECK(source && fp && row && x && packed);
    for (int r = 0; r < rows; r++) for (int k = 0; k < cols; k++) {
        float f = q38_fp6_decode((uint8_t)(k % 64)) * (float)(1 << (r % 5));
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint16_t bf = (uint16_t)(bits >> 16);
        fp[(size_t)r * cols + k] = f;
        if (format == Q38_LB_FP6_E2M3) memcpy(source + (size_t)r * rb + k * 2, &bf, 2);
    }
    if (format == Q38_LB_NVFP4) {
        for (size_t b = 0; b < bytes / 36; b++) {
            for (int s = 0; s < 4; s++) source[b * 36 + s] = (uint8_t)(b + s);
            for (int j = 4; j < 36; j++) source[b * 36 + j] = (uint8_t)(b * 13 + j);
        }
        CHECK(q38_lowbit_pack_nvfp4(packed, packed_bytes, source, rb, rows, cols));
    } else CHECK(q38_lowbit_pack_fp6(packed, packed_bytes, fp, cols, rows, cols));
    for (int i = 0; i < 32; i++) source[bytes + i] = (unsigned char)i;
    char path[4096];
    const char *scratch = getenv("TMPDIR");
    CHECK(scratch && *scratch);
    CHECK(snprintf(path, sizeof(path), "%s/lowbit-model-XXXXXX", scratch) < (int)sizeof(path));
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(write(fd, source, total) == (ssize_t)total);
    CHECK(!unlink(path));
    gguf_tensor_info tensors[2] = {0};
    tensors[0].name.str = "blk.0.ffn_gate.weight";
    tensors[0].n_dims = 2; tensors[0].dims[0] = cols; tensors[0].dims[1] = rows;
    tensors[0].type = format == Q38_LB_NVFP4 ? GGML_TYPE_NVFP4 : GGML_TYPE_BF16;
    tensors[1].name.str = "blk.0.attn_norm.weight";
    tensors[1].n_dims = 1; tensors[1].dims[0] = 8; tensors[1].type = GGML_TYPE_F32;
    tensors[1].offset = bytes;
    gguf_context g = {0};
    g.n_tensors = 2; g.tensors = tensors; g.fd = fd; g.use_mmap = 1;
    g.data = source; g.data_size = total;
    CHECK(!q38_lowbit_model_load(&g, format, arithmetic, 0, SIZE_MAX));
    q38_lowbit_model *m = q38_lowbit_model_load(&g, format, arithmetic, 0, 0);
    CHECK(m);
    const q38_lowbit_matrix *mat = q38_lowbit_model_tensor(&g, 0);
    CHECK(mat && !q38_lowbit_model_tensor(&g, 1));
    CHECK(gguf_tensor_data(&g, 1) != source + bytes);
    CHECK(!memcmp(gguf_tensor_data(&g, 1), source + bytes, 32));
    for (int r = 0; r < rows; r++) {
        CHECK(q38_lowbit_matrix_row(row, mat, r));
        CHECK(q38_lowbit_dequant_row(fp + (size_t)r * cols, packed, format, rows, cols, r));
        CHECK(!memcmp(row, fp + (size_t)r * cols, cols * sizeof(float)));
    }
    float gold[35], got[35];
    size_t count = ((size_t)cols + 15) / 16;
    q38_lowbit_act *act = malloc(count * sizeof(*act));
    CHECK(act);
    for (int pass = 0; pass < 2; pass++) {
        /* Deliberately reuse the same input address with changed contents. */
        for (int k = 0; k < cols; k++) x[k] = (float)((k * 17 + pass * 19) % 31 - 15) * .03125f;
        if (arithmetic) {
            CHECK(q38_lowbit_prepare(act, count, x, cols, arithmetic));
            CHECK(q38_lowbit_dot(gold, packed, format, act, arithmetic, rows, cols));
        } else CHECK(q38_lowbit_reference(gold, packed, format, x, rows, cols));
        CHECK(q38_lowbit_matrix_begin(mat, x));
        if (arithmetic) CHECK(!q38_lowbit_matrix_begin(mat, x));
        for (int r = 0; r < rows; r++) got[r] = NAN;
        CHECK(q38_lowbit_matrix_rows(got, mat, x, 0, 3));
        CHECK(q38_lowbit_matrix_rows(got, mat, x, 3, 19));
        CHECK(q38_lowbit_matrix_rows(got, mat, x, 19, rows));
        q38_lowbit_matrix_end(mat);
        for (int r = 0; r < rows; r++) CHECK(isfinite(got[r]) && fabsf(gold[r] - got[r]) <= 2e-5f * (1 + fabsf(gold[r])));
        CHECK(!q38_lowbit_matrix_rows(got, mat, x, -1, rows));
    }
    char image_path[4096];
    CHECK(snprintf(image_path, sizeof(image_path), "%s/lowbit-image-XXXXXX", scratch) < (int)sizeof(image_path));
    int image_fd = mkstemp(image_path);
    CHECK(image_fd >= 0 && !close(image_fd));
    CHECK(q38_lowbit_model_save_image(m, image_path));
    q38_lowbit_model_free(m);
    m = q38_lowbit_model_load_image(&g, format, arithmetic, 0, 0, image_path);
    CHECK(m);
    mat = q38_lowbit_model_tensor(&g, 0);
    for (int r = 0; r < rows; r++) {
        CHECK(q38_lowbit_matrix_row(row, mat, r));
        CHECK(!memcmp(row, fp + (size_t)r * cols, cols * sizeof(float)));
    }
    CHECK(!memcmp(gguf_tensor_data(&g, 1), source + bytes, 32));
    q38_lowbit_model_free(m);
    /* A changed inventory or payload must fail closed, with restored pointers. */
    tensors[0].name.str = "blk.1.ffn_gate.weight";
    CHECK(!q38_lowbit_model_load_image(&g, format, arithmetic, 0, 0, image_path));
    tensors[0].name.str = "blk.0.ffn_gate.weight";
    image_fd = open(image_path, O_RDWR);
    CHECK(image_fd >= 0);
    unsigned char corrupt;
    CHECK(pread(image_fd, &corrupt, 1, 320) == 1);
    corrupt ^= 1;
    CHECK(pwrite(image_fd, &corrupt, 1, 320) == 1);
    CHECK(!q38_lowbit_model_load_image(&g, format, arithmetic, 0, 0, image_path));
    corrupt ^= 1;
    CHECK(pwrite(image_fd, &corrupt, 1, 320) == 1);
    CHECK(!ftruncate(image_fd, 320));
    CHECK(!q38_lowbit_model_load_image(&g, format, arithmetic, 0, 0, image_path));
    CHECK(!close(image_fd) && !unlink(image_path));
    CHECK(!q38_lowbit_model_tensor(&g, 0));
    CHECK(gguf_tensor_data(&g, 1) == source + bytes);
    CHECK(!ftruncate(fd, (off_t)total - 1));
    CHECK(!q38_lowbit_model_load(&g, format, arithmetic, 0, 0));
    free(g.tensor_data); close(fd);
    free(act); free(source); free(fp); free(row); free(x); free(packed);
    printf("bounded model format=%d arithmetic=%d PASS\n", format, arithmetic);
}

int main(void) {
    for (int format = 1; format <= 2; format++)
        for (int arithmetic = 0; arithmetic <= 16; arithmetic += 8) test_model(format, arithmetic);
    puts("bounded lowbit model PASS");
    return 0;
}
