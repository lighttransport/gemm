/* Synthetic correctness checks for the shared Q5R/IQ4R decode layouts. */
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kquant_decode_cache.h"
#include "qwen38_kquant_attach.h"

enum { TEST_ROWS = 16, TEST_COLS = 512 };

static uint32_t test_random(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_weights(block_q5_K *q5, block_iq4_xs *iq4) {
    uint32_t state = 0x53f38a64u;
    size_t blocks = (size_t)TEST_ROWS * (TEST_COLS / 256);
    for (size_t b = 0; b < blocks; b++) {
        q5[b].d = ggml_fp32_to_fp16(0.0008f + 0.0001f * (float)(b % 7));
        q5[b].dmin = ggml_fp32_to_fp16(0.0003f + 0.00005f * (float)(b % 5));
        for (size_t i = 0; i < sizeof(q5[b].scales); i++)
            q5[b].scales[i] = (uint8_t)test_random(&state);
        for (size_t i = 0; i < sizeof(q5[b].qh); i++)
            q5[b].qh[i] = (uint8_t)test_random(&state);
        for (size_t i = 0; i < sizeof(q5[b].qs); i++)
            q5[b].qs[i] = (uint8_t)test_random(&state);

        iq4[b].d = ggml_fp32_to_fp16(0.0007f + 0.0001f * (float)(b % 9));
        iq4[b].scales_h = (uint16_t)test_random(&state);
        for (size_t i = 0; i < sizeof(iq4[b].scales_l); i++)
            iq4[b].scales_l[i] = (uint8_t)test_random(&state);
        for (size_t i = 0; i < sizeof(iq4[b].qs); i++)
            iq4[b].qs[i] = (uint8_t)test_random(&state);
    }
}

static void fill_activation(float *x, int pattern) {
    uint32_t state = 0x1234abcdu;
    for (int i = 0; i < TEST_COLS; i++) {
        if (pattern == 0)
            x[i] = 0.75f * sinf((float)i * 0.031f) + 0.2f * cosf((float)i * 0.17f);
        else if (pattern == 1)
            x[i] = i % 31 == 0 ? sinf((float)i * 0.13f) : 0.0f;
        else if (pattern == 2)
            x[i] = (i & 1 ? -1.0f : 1.0f) * ldexpf(0.75f, i % 17 - 8);
        else
            x[i] = ((float)(test_random(&state) >> 8) *
                    (2.0f / 16777215.0f) - 1.0f);
    }
}

static int compare(const float *got, const float *want, float rel_limit,
                   int require_bit_exact, double *nrmse_out, double *max_out) {
    double err2 = 0.0, norm2 = 0.0, max_abs = 0.0;
    for (int r = 0; r < TEST_ROWS; r++) {
        double e = (double)got[r] - want[r];
        double a = fabs(e);
        err2 += e * e;
        norm2 += (double)want[r] * want[r];
        if (a > max_abs) max_abs = a;
    }
    double nrmse = sqrt(err2 / (norm2 + 1e-30));
    *nrmse_out = nrmse;
    *max_out = max_abs;
    if (require_bit_exact && memcmp(got, want, TEST_ROWS * sizeof(*got))) return -1;
    return nrmse <= rel_limit ? 0 : -1;
}

static void reference_a8(float *dst, uint32_t type, const void *weights,
                         const kquant_cache_a8_block *qx) {
    size_t row_bytes = tf_row_bytes(type, TEST_COLS);
    float dequant[TEST_COLS];
    for (int row = 0; row < TEST_ROWS; row++) {
        dequant_row(type, (const uint8_t *)weights + (size_t)row * row_bytes,
                    dequant, TEST_COLS);
        float sum = 0.0f;
        for (int block = 0; block < TEST_COLS / 256; block++)
            for (int group = 0; group < 8; group++)
                for (int k = 0; k < 32; k++) {
                    int col = block * 256 + group * 32 + k;
                    sum += dequant[col] * (float)qx[block].q[group * 32 + k] *
                           qx[block].d[group];
                }
        dst[row] = sum;
    }
}

static void compact_a8(float *q5_dst, float *iq4_dst,
                       const block_q5_K *q5, const block_iq4_xs *iq4,
                       const float *x) {
    const int nb = TEST_COLS / 256;
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), threads = omp_get_num_threads();
        int row0 = TEST_ROWS * tid / threads;
        int row1 = TEST_ROWS * (tid + 1) / threads;
        tf_kquant_a8_block qx[TEST_COLS / 256];
        tf_kquant_quant_a8(qx, x, TEST_COLS);
        int row = row0;
        for (; row + 3 < row1; row += 4) {
            const block_q5_K *w = q5 + (size_t)row * nb;
            tf_q5_k_a8_dot4_sve(q5_dst + row, w, w + nb, w + 2 * nb,
                                w + 3 * nb, qx, nb);
        }
        for (; row < row1; row++)
            q5_dst[row] = tf_q5_k_a8_dot_sve(
                q5 + (size_t)row * nb, qx, nb);
        for (row = row0; row < row1; row++) {
            iq4_dst[row] = tf_iq4_xs_a8_dot_sve(
                iq4 + (size_t)row * nb, qx, nb);
        }
    }
}

static int test_selective_materialize(void) {
    enum { PAYLOAD_BYTES = 4096 };
    const char *scratch = getenv("TMPDIR");
    if (!scratch || !*scratch) scratch = ".";
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path),
                          "%s/q38kc-materialize-%ld", scratch, (long)getpid());
    if (length < 0 || (size_t)length >= sizeof(path)) return -1;

    q38kc_header *header = calloc(1, sizeof(*header));
    uint8_t *q5_payload = malloc(PAYLOAD_BYTES);
    uint8_t *iq4_payload = malloc(PAYLOAD_BYTES);
    if (!header || !q5_payload || !iq4_payload) return -1;
    memcpy(header->magic, Q38KC_MAGIC, sizeof(header->magic));
    header->n_entries = 2;
    header->entries[0].source_type = GGML_TYPE_Q5_K;
    header->entries[0].local_rows = TF_KQUANT_CACHE_ROWS;
    header->entries[0].file_offset = Q38KC_HEADER_BYTES;
    header->entries[0].byte_length = PAYLOAD_BYTES;
    header->entries[1].source_type = GGML_TYPE_IQ4_XS;
    header->entries[1].local_rows = TF_KQUANT_CACHE_ROWS;
    header->entries[1].file_offset = Q38KC_HEADER_BYTES + PAYLOAD_BYTES;
    header->entries[1].byte_length = PAYLOAD_BYTES;
    memset(q5_payload, 0x5a, PAYLOAD_BYTES);
    memset(iq4_payload, 0x49, PAYLOAD_BYTES);
    size_t file_bytes = Q38KC_HEADER_BYTES + 2 * PAYLOAD_BYTES;

    int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
    int failed = fd < 0 || ftruncate(fd, (off_t)file_bytes) ||
        pwrite(fd, header, sizeof(*header), 0) != (ssize_t)sizeof(*header) ||
        pwrite(fd, q5_payload, PAYLOAD_BYTES,
               (off_t)header->entries[0].file_offset) != PAYLOAD_BYTES ||
        pwrite(fd, iq4_payload, PAYLOAD_BYTES,
               (off_t)header->entries[1].file_offset) != PAYLOAD_BYTES;
    void *mapping = MAP_FAILED;
    if (!failed)
        mapping = mmap(NULL, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fd >= 0) close(fd);
    if (mapping == MAP_FAILED) failed = 1;

    q38kc_model_cache cache = {0};
    transformer_model model = {0};
    char error[256] = {0};
    if (!failed) {
        cache.loaded.header = header;
        cache.loaded.mapping = mapping;
        cache.loaded.mapping_bytes = file_bytes;
        if (q38kc_model_materialize(&cache, &model, path,
                                    error, sizeof(error))) {
            fprintf(stderr, "selective materialize failed: %s\n", error);
            failed = 1;
        }
    }
    if (!failed) {
        const uint8_t *materialized = (const uint8_t *)cache.loaded.mapping;
        const uint8_t *q5 = materialized + header->entries[0].file_offset;
        const uint8_t *iq4 = materialized + header->entries[1].file_offset;
        for (int i = 0; i < PAYLOAD_BYTES; i++)
            if (q5[i] || iq4[i] != 0x49) {
                fprintf(stderr,
                        "selective payload mismatch at byte %d q5=%02x iq4=%02x\n",
                        i, q5[i], iq4[i]);
                failed = 1;
                break;
            }
        if (cache.materialized_entries != 1 ||
            cache.materialized_bytes != Q38KC_HEADER_BYTES + PAYLOAD_BYTES) {
            fprintf(stderr,
                    "selective accounting mismatch entries=%u bytes=%llu\n",
                    cache.materialized_entries,
                    (unsigned long long)cache.materialized_bytes);
            failed = 1;
        }
    }

    if (cache.loaded.mapping && cache.loaded.mapping != MAP_FAILED)
        q38kc_unload(&cache.loaded);
    else {
        if (mapping != MAP_FAILED) munmap(mapping, file_bytes);
        free(header);
    }
    unlink(path);
    free(iq4_payload);
    free(q5_payload);
    return failed ? -1 : 0;
}

int main(void) {
    const int nb = TEST_COLS / 256;
    size_t compact_blocks = (size_t)TEST_ROWS * nb;
    block_q5_K *q5 = aligned_alloc(256, compact_blocks * sizeof(*q5));
    block_iq4_xs *iq4 = aligned_alloc(256, compact_blocks * sizeof(*iq4));
    size_t q5r_size = packed_q5r_bytes(TEST_ROWS, TEST_COLS);
    size_t iq4r_size = packed_iq4r_bytes(TEST_ROWS, TEST_COLS);
    uint8_t *q5r = aligned_alloc(256, (q5r_size + 255) & ~(size_t)255);
    uint8_t *iq4r = aligned_alloc(256, (iq4r_size + 255) & ~(size_t)255);
    float *x = aligned_alloc(256, TEST_COLS * sizeof(*x));
    float native_q5[TEST_ROWS], packed_q5[TEST_ROWS];
    float native_iq4[TEST_ROWS], packed_iq4[TEST_ROWS];
    float compact_q5_a8[TEST_ROWS], compact_iq4_a8[TEST_ROWS];
    float ranged_q5[TEST_ROWS], ranged_iq4[TEST_ROWS];
    float owned_q5[TEST_ROWS], owned_iq4[TEST_ROWS];
    float tail_q5[TEST_ROWS], tail_reference[TEST_ROWS];
    if (!q5 || !iq4 || !q5r || !iq4r || !x) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    if (test_selective_materialize()) return 1;
    if (packed_q5r_bytes(7, TEST_COLS) || packed_iq4r_bytes(TEST_ROWS, 255) ||
        !q5r_size || !iq4r_size) {
        fprintf(stderr, "dimension validation failed\n");
        return 1;
    }
    fill_weights(q5, iq4);
    if (pack_q5r(q5r, q5, TEST_ROWS, TEST_COLS) ||
        pack_iq4r(iq4r, iq4, TEST_ROWS, TEST_COLS)) {
        fprintf(stderr, "pack failed\n");
        return 1;
    }
    for (int pattern = 0; pattern < 4; pattern++) {
        fill_activation(x, pattern);
        kquant_cache_a8_block qx[TEST_COLS / 256];
        kquant_cache_quant_a8(qx, x, TEST_COLS);
        reference_a8(native_q5, GGML_TYPE_Q5_K, q5, qx);
        reference_a8(native_iq4, GGML_TYPE_IQ4_XS, iq4, qx);
        compact_a8(compact_q5_a8, compact_iq4_a8, q5, iq4, x);
        if (run_packed_q5r(packed_q5, q5r, x, TEST_ROWS, TEST_COLS) ||
            run_packed_iq4r(packed_iq4, iq4r, x, TEST_ROWS, TEST_COLS)) {
            fprintf(stderr, "matvec failed\n");
            return 1;
        }
        qtensor tq5 = {.data = q5, .type = GGML_TYPE_Q5_K,
                       .n_rows = TEST_ROWS, .n_cols = TEST_COLS,
                       .kquant_cache = q5r,
                       .kquant_cache_format = Q38KC_FORMAT_Q5R};
        qtensor tiq4 = {.data = iq4, .type = GGML_TYPE_IQ4_XS,
                        .n_rows = TEST_ROWS, .n_cols = TEST_COLS,
                        .kquant_cache = iq4r,
                        .kquant_cache_format = Q38KC_FORMAT_IQ4R};
        memset(ranged_q5, 0, sizeof(ranged_q5));
        memset(ranged_iq4, 0, sizeof(ranged_iq4));
        const int cuts[] = {0, 3, 11, TEST_ROWS};
        for (int cut = 0; cut < 3; cut++) {
            if (!tf_kquant_cache_rows(ranged_q5, &tq5, x,
                                      cuts[cut], cuts[cut + 1]) ||
                !tf_kquant_cache_rows(ranged_iq4, &tiq4, x,
                                      cuts[cut], cuts[cut + 1])) {
                fprintf(stderr, "ranged runtime dispatch failed\n");
                return 1;
            }
        }
        if (memcmp(ranged_q5, packed_q5, sizeof(ranged_q5)) ||
            memcmp(ranged_iq4, packed_iq4, sizeof(ranged_iq4))) {
            fprintf(stderr, "unaligned ranged runtime mismatch pattern=%d\n", pattern);
            return 1;
        }
        for (int row = 0; row < TEST_ROWS; row++) {
            owned_q5[row] = NAN;
            owned_iq4[row] = NAN;
        }
        for (int tid = 0; tid < 3; tid++) {
            tf_thread_matvec(owned_q5, &tq5, x, TEST_ROWS, tid, 3);
            tf_thread_matvec(owned_iq4, &tiq4, x, TEST_ROWS, tid, 3);
        }
        if (memcmp(owned_q5, packed_q5, sizeof(owned_q5)) ||
            memcmp(owned_iq4, packed_iq4, sizeof(owned_iq4))) {
            fprintf(stderr, "persistent ownership mismatch pattern=%d\n", pattern);
            return 1;
        }
        qtensor compact_q5 = tq5;
        compact_q5.kquant_cache = NULL;
        compact_q5.kquant_cache_format = 0;
        for (int row = 0; row < TEST_ROWS; row++) {
            tail_q5[row] = 1234567.0f;
            tail_reference[row] = 1234567.0f;
        }
        tf_matvec_qtensor_rows(tail_reference, &compact_q5, x, 0,
                               TEST_ROWS - 1);
        for (int tid = 0; tid < 3; tid++)
            tf_thread_matvec(tail_q5, &tq5, x, TEST_ROWS - 1, tid, 3);
        if (memcmp(tail_q5, tail_reference,
                   (TEST_ROWS - 1) * sizeof(tail_q5[0])) ||
            tail_q5[TEST_ROWS - 1] != 1234567.0f) {
            int bad_row = -1;
            for (int row = 0; row < TEST_ROWS - 1; row++)
                if (memcmp(&tail_q5[row], &tail_reference[row], sizeof(float))) {
                    bad_row = row;
                    break;
                }
            fprintf(stderr, "persistent compact-tail fallback mismatch pattern=%d row=%d got=%g want=%g tail=%g\n",
                    pattern, bad_row,
                    bad_row >= 0 ? tail_q5[bad_row] : 0.0f,
                    bad_row >= 0 ? tail_reference[bad_row] : 0.0f,
                    tail_q5[TEST_ROWS - 1]);
            return 1;
        }
        double q5_nrmse, q5_max, iq4_nrmse, iq4_max;
        double q5_compact_nrmse, q5_compact_max;
        double iq4_compact_nrmse, iq4_compact_max;
        if (compare(packed_q5, native_q5, 2e-5f, 0, &q5_nrmse, &q5_max) ||
            compare(packed_iq4, native_iq4, 2e-5f, 0, &iq4_nrmse, &iq4_max) ||
            compare(packed_q5, compact_q5_a8, 2e-5f, 0,
                    &q5_compact_nrmse, &q5_compact_max) ||
            compare(packed_iq4, compact_iq4_a8, 0.0f, 1,
                    &iq4_compact_nrmse, &iq4_compact_max)) {
            fprintf(stderr, "pattern=%d mismatch q5_nrmse=%.3g q5_max=%.3g "
                    "iq4_nrmse=%.3g iq4_max=%.3g q5_compact_nrmse=%.3g "
                    "q5_compact_max=%.3g iq4_compact_nrmse=%.3g "
                    "iq4_compact_max=%.3g\n",
                    pattern, q5_nrmse, q5_max, iq4_nrmse, iq4_max,
                    q5_compact_nrmse, q5_compact_max,
                    iq4_compact_nrmse, iq4_compact_max);
            return 1;
        }
        printf("pattern=%d q5_nrmse=%.3g q5_max=%.3g "
               "iq4_nrmse=%.3g iq4_max=%.3g q5_compact_nrmse=%.3g "
               "q5_compact_max=%.3g iq4_compact_exact=1\n",
               pattern, q5_nrmse, q5_max, iq4_nrmse, iq4_max,
               q5_compact_nrmse, q5_compact_max);
    }
    qtensor bad = {.type = GGML_TYPE_Q5_K, .n_rows = TEST_ROWS,
                   .n_cols = TEST_COLS, .kquant_cache = q5r,
                   .kquant_cache_format = Q38KC_FORMAT_IQ4R};
    if (tf_kquant_cache_rows(ranged_q5, &bad, x, 0, TEST_ROWS)) {
        fprintf(stderr, "invalid runtime format accepted\n");
        return 1;
    }
    transformer_model attach_model = {0};
    transformer_layer attach_layer = {0};
    attach_model.n_layers = 1;
    attach_model.layers = &attach_layer;
    attach_layer.ffn_gate = (qtensor){.data = q5, .type = GGML_TYPE_Q5_K,
        .n_rows = TEST_ROWS, .n_cols = TEST_COLS};
    q38kc_header attach_header = {0};
    attach_header.n_entries = 1;
    snprintf(attach_header.entries[0].name,
             sizeof(attach_header.entries[0].name),
             "blk.0.ffn_gate.weight");
    attach_header.entries[0].source_type = GGML_TYPE_Q5_K;
    attach_header.entries[0].cache_format = Q38KC_FORMAT_Q5R;
    attach_header.entries[0].local_rows = TEST_ROWS;
    attach_header.entries[0].local_cols = TEST_COLS;
    q38kc_model_cache attach_cache = {0};
    attach_cache.loaded.header = &attach_header;
    attach_cache.loaded.mapping = q5r;
    attach_cache.loaded.mapping_bytes = q5r_size;
    char attach_error[128] = {0};
    if (q38kc_model_attach(&attach_cache, &attach_model,
                           attach_error, sizeof(attach_error)) ||
        attach_layer.ffn_gate.kquant_cache ||
        attach_layer.ffn_gate.kquant_cache_format ||
        attach_cache.attached_entries) {
        fprintf(stderr, "default Q5 skip failed: %s\n", attach_error);
        return 1;
    }
    attach_cache.enable_q5 = 1;
    if (q38kc_model_attach(&attach_cache, &attach_model,
                           attach_error, sizeof(attach_error)) ||
        attach_layer.ffn_gate.kquant_cache != q5r ||
        attach_layer.ffn_gate.kquant_cache_format != Q38KC_FORMAT_Q5R ||
        attach_cache.attached_entries != 1) {
        fprintf(stderr, "model cache attach failed: %s\n", attach_error);
        return 1;
    }
    q38kc_model_detach(&attach_cache, &attach_model);
    if (attach_layer.ffn_gate.kquant_cache ||
        attach_layer.ffn_gate.kquant_cache_format ||
        attach_cache.attached_entries) {
        fprintf(stderr, "model cache detach failed\n");
        return 1;
    }
    printf("SENTINEL qwen38_kquant_cache=OK layout_version=%d q5r_bytes=%zu iq4r_bytes=%zu ownership_threads=3 tail_fallback=1 q5_default_skip=1 selective_materialize=1 attach_detach=1\n",
           TF_KQUANT_CACHE_LAYOUT_VERSION, q5r_size, iq4r_size);
    free(x);
    free(iq4r);
    free(q5r);
    free(iq4);
    free(q5);
    return 0;
}
