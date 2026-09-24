/* Load-time FP4 repack and fused exact N=3 projection, runnable under SVE QEMU. */
#include <omp.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"
#include "qwen38_nvfp4_pack.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

extern int q38_nvfp4_packed_n3_mt(float *, const void *, const float *,
                                   int, int, int);

int main(void) {
#if !defined(__ARM_FEATURE_SVE)
    return 77;
#else
    enum { ROWS = 16, COLS = 128, NB = COLS / 64 };
    block_nvfp4 raw[ROWS * NB];
    tf_nvfp4_tiled_block tiled[ROWS / 8 * NB];
    tf_nvfp4_packed_block packed[ROWS / 8 * NB];
    const uint8_t scales[] = {0, 1, 2, 3, 10, 11, 15, 32, 80, 127};
    for (int row = 0; row < ROWS; row++) {
        for (int ib = 0; ib < NB; ib++) {
            block_nvfp4 *q = &raw[row * NB + ib];
            for (int s = 0; s < 4; s++) {
                q->d[s] = scales[(row + ib + s) % 10];
                for (int j = 0; j < 8; j++)
                    q->qs[s * 8 + j] = (uint8_t)(((row + j + s) & 15) |
                                   (((row * 3 + j * 5 + ib) & 15) << 4));
            }
        }
    }
    for (int tile = 0; tile < ROWS / 8; tile++) {
        q38_nvfp4_pack_tile(packed + tile * NB, raw + tile * 8 * NB, NB);
        for (int ib = 0; ib < NB; ib++) {
            for (int s = 0; s < 4; s++) {
                tf_nvfp4_tiled_subblock *p = &tiled[tile * NB + ib].s[s];
                for (int row = 0; row < 8; row++) {
                    const block_nvfp4 *q = &raw[tile * 8 * NB + row * NB + ib];
                    p->d[row] = q->d[s];
                    memcpy(p->qs + row * 8, q->qs + s * 8, 8);
                }
            }
        }
    }
    gguf_tensor_info infos[2] = {0};
    infos[0].name.str = "blk.0.ffn_gate.weight";
    infos[1].name.str = "blk.64.ffn_gate.weight";
    for (int i = 0; i < 2; i++) {
        infos[i].n_dims = 2;
        infos[i].dims[0] = 64;
        infos[i].dims[1] = 8;
        infos[i].type = GGML_TYPE_NVFP4;
        infos[i].offset = (uint64_t)i * 288;
    }
    gguf_context fake = {0};
    fake.n_tensors = 2;
    fake.tensors = infos;
    fake.data = (uint8_t *)raw;
    fake.data_alloc_size = 1024;
    fake.alignment = 32;
    q38_nvfp4_layout *layout = NULL;
    size_t layout_bytes = 0;
    int packed_count = 0;
    if (q38_nvfp4_plan(&fake, &layout, &layout_bytes, &packed_count) ||
        packed_count != 1 || layout_bytes != 672 ||
        !layout[0].packed || layout[1].packed || layout[1].new_off != 384) {
        fprintf(stderr, "FAIL packed layout plan or NextN exclusion\n");
        free(layout);
        return 1;
    }
    int fd = memfd_create("q38-packed-test", 0);
    uint8_t resident[1024] = {0};
    if (fd < 0 || write(fd, raw, 576) != 576) {
        fprintf(stderr, "FAIL packed source fixture\n");
        free(layout);
        if (fd >= 0) close(fd);
        return 1;
    }
    fake.fd = fd;
    fake.data = resident;
    q38_nvfp4_pack_tile((tf_nvfp4_packed_block *)resident, raw, 1);
    memcpy(resident + layout[1].new_off, raw + 8, 288);
    if (q38_nvfp4_verify_packed_tiles(&fake, layout, 2)) {
        fprintf(stderr, "FAIL packed source comparison\n");
        free(layout);
        close(fd);
        return 1;
    }
    close(fd);
    free(layout);
    qtensor pm = {0};
    pm.type = GGML_TYPE_NVFP4;
    pm.n_rows = ROWS; pm.n_cols = COLS;
    pm.nvfp4_packed = 1; pm.data = (uint8_t *)packed;
    float x[3 * COLS], expected[3 * ROWS], got[3 * ROWS];
    for (int trial = 0; trial < 8; trial++) {
        for (int i = 0; i < 3 * COLS; i++)
            x[i] = (float)((i * 29 + trial * 17) % 101 - 50) * 0.03125f;
        for (int t = 0; t < 3; t++)
            for (int tile = 0; tile < ROWS / 8; tile++)
                tf_nvfp4_tiled_dot8(expected + t * ROWS + tile * 8,
                                      tiled + tile * NB, x + t * COLS, NB);
        if (!q38_nvfp4_packed_n3_mt(got, packed, x, ROWS, COLS, 1)) return 1;
        for (int i = 0; i < 3 * ROWS; i++)
            if (memcmp(&got[i], &expected[i], sizeof(float))) {
                fprintf(stderr, "FAIL packed N3 trial=%d output=%d ref=%a got=%a\n",
                        trial, i, expected[i], got[i]);
                return 1;
            }
        for (int t = 0; t < 3; t++) {
            tf_nvfp4_packed_exact_matvec_rows(got + t * ROWS,
                                                &pm, x + t * COLS, 0, ROWS);
            for (int i = 0; i < ROWS; i++)
                if (memcmp(&got[t * ROWS + i], &expected[t * ROWS + i],
                           sizeof(float))) {
                    fprintf(stderr, "FAIL packed N1 trial=%d token=%d row=%d\n",
                            trial, t, i);
                    return 1;
                }
        }
    }
    printf("PASS packed_exact outputs=%d sve_bytes=%zu\n", 8 * 3 * ROWS,
           (size_t)svcntb());
    return 0;
#endif
}
