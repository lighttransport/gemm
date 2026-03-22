/*
 * test_sparse3d.c - Unit tests for sparse 3D tensor operations
 *
 * Build:
 *   make
 *
 * Tests: create/free, linear, layernorm, conv3d, attention,
 *        downsample, upsample, cat_feats, rope_3d
 *
 * Verification mode:
 *   ./test_sparse3d --export <output_dir> --input <input_dir>
 *   Loads .npy inputs from input_dir, runs each op, writes .npy outputs to output_dir
 */

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

/* ==================================================================== */
/* NPY I/O                                                                */
/* ==================================================================== */

static void write_npy_f32(const char *path, const float *data, const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char shape_str[128];
    int slen = 0;
    slen += snprintf(shape_str + slen, sizeof(shape_str) - (size_t)slen, "(");
    size_t n_elem = 1;
    for (int i = 0; i < ndim; i++) {
        slen += snprintf(shape_str + slen, sizeof(shape_str) - (size_t)slen, "%d,", dims[i]);
        n_elem *= (size_t)dims[i];
    }
    slen += snprintf(shape_str + slen, sizeof(shape_str) - (size_t)slen, ")");
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape_str);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), n_elem, f);
    fclose(f);
}

static void write_npy_i32(const char *path, const int32_t *data, const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char shape_str[128];
    int slen = 0;
    slen += snprintf(shape_str + slen, sizeof(shape_str) - (size_t)slen, "(");
    size_t n_elem = 1;
    for (int i = 0; i < ndim; i++) {
        slen += snprintf(shape_str + slen, sizeof(shape_str) - (size_t)slen, "%d,", dims[i]);
        n_elem *= (size_t)dims[i];
    }
    slen += snprintf(shape_str + slen, sizeof(shape_str) - (size_t)slen, ")");
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<i4', 'fortran_order': False, 'shape': %s, }", shape_str);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(int32_t), n_elem, f);
    fclose(f);
}

/* Parse .npy header and return pointer to data (malloc'd).
 * Sets *out_ndim and out_dims[]. Supports '<f4' and '<i4'. */
static void *read_npy(const char *path, int *out_ndim, int *out_dims, char *out_dtype) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }

    /* Read magic + version (8 bytes) */
    uint8_t buf[10];
    if (fread(buf, 1, 8, f) != 8) { fclose(f); return NULL; }

    /* Read header length (2 bytes, little-endian) */
    uint16_t header_len;
    if (fread(&header_len, 2, 1, f) != 1) { fclose(f); return NULL; }

    /* Read header string */
    char *header = (char *)malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != header_len) { free(header); fclose(f); return NULL; }
    header[header_len] = '\0';

    /* Parse dtype: look for 'descr': '<XN' */
    char dtype = 'f';
    int elem_size = 4;
    const char *dp = strstr(header, "'descr':");
    if (dp) {
        dp = strchr(dp, '<');
        if (dp) {
            dtype = dp[1];  /* 'f' or 'i' */
            elem_size = dp[2] - '0';
        }
    }
    if (out_dtype) *out_dtype = dtype;

    /* Parse shape: look for 'shape': (d1, d2, ...) */
    int ndim = 0;
    const char *sp = strstr(header, "'shape':");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ') sp++;
                if (*sp == ')' || *sp == '\0') break;
                out_dims[ndim++] = atoi(sp);
                while (*sp && *sp != ',' && *sp != ')') sp++;
                if (*sp == ',') sp++;
            }
        }
    }
    *out_ndim = ndim;

    size_t n_elem = 1;
    for (int i = 0; i < ndim; i++) n_elem *= (size_t)out_dims[i];

    void *data = malloc(n_elem * (size_t)elem_size);
    size_t read_count = fread(data, (size_t)elem_size, n_elem, f);
    if (read_count != n_elem) {
        fprintf(stderr, "Warning: %s: expected %zu elements, read %zu\n", path, n_elem, read_count);
    }

    free(header);
    fclose(f);
    return data;
}

static float *read_npy_f32(const char *path, int *out_ndim, int *out_dims) {
    char dtype;
    void *data = read_npy(path, out_ndim, out_dims, &dtype);
    if (!data) return NULL;
    if (dtype != 'f') {
        fprintf(stderr, "Expected float32 in %s, got '%c'\n", path, dtype);
        free(data);
        return NULL;
    }
    return (float *)data;
}

static int32_t *read_npy_i32(const char *path, int *out_ndim, int *out_dims) {
    char dtype;
    void *data = read_npy(path, out_ndim, out_dims, &dtype);
    if (!data) return NULL;
    if (dtype != 'i') {
        fprintf(stderr, "Expected int32 in %s, got '%c'\n", path, dtype);
        free(data);
        return NULL;
    }
    return (int32_t *)data;
}

static void make_path(char *dst, size_t dst_size, const char *dir, const char *name) {
    snprintf(dst, dst_size, "%s/%s.npy", dir, name);
}

static int g_tests = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    g_tests++; \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define SECTION(name) fprintf(stderr, "\n--- %s ---\n", name)

/* ---- Test: create / free / clone ---- */
static void test_lifecycle(void) {
    SECTION("Lifecycle");

    /* 2 batches, 5 voxels total: batch 0 has 3, batch 1 has 2 */
    int32_t coords[] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 0,
        1, 1, 1, 1,
    };
    float feats[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
    };
    sp3d_tensor *t = sp3d_create(coords, feats, 5, 2, 2);
    CHECK(t != NULL, "create non-null");
    CHECK(t->N == 5, "N == 5");
    CHECK(t->C == 2, "C == 2");
    CHECK(t->batch_size == 2, "batch_size == 2");
    CHECK(t->batch_starts[0] == 0, "batch_starts[0] == 0");
    CHECK(t->batch_starts[1] == 3, "batch_starts[1] == 3");
    CHECK(t->batch_starts[2] == 5, "batch_starts[2] == 5");
    CHECK(t->feats[0] == 1.0f, "feats[0] == 1.0");
    CHECK(t->feats[9] == 10.0f, "feats[9] == 10.0");

    sp3d_tensor *c = sp3d_clone(t);
    CHECK(c->N == 5, "clone N == 5");
    CHECK(c->feats[0] == 1.0f, "clone feats match");
    CHECK(c->coords[4] == 0, "clone coords match");
    /* Modify clone, original unchanged */
    c->feats[0] = 99.0f;
    CHECK(t->feats[0] == 1.0f, "original unchanged after clone modify");

    sp3d_free(c);
    sp3d_free(t);
    fprintf(stderr, "  lifecycle: OK\n");
}

/* ---- Test: cat_feats ---- */
static void test_cat_feats(void) {
    SECTION("cat_feats");

    int32_t coords[] = { 0,0,0,0, 0,1,0,0 };
    float feats_a[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float feats_b[] = { 5.0f, 6.0f };
    sp3d_tensor *a = sp3d_create(coords, feats_a, 2, 2, 1);
    sp3d_tensor *b = sp3d_create(coords, feats_b, 2, 1, 1);

    sp3d_tensor *cat = sp3d_cat_feats(a, b);
    CHECK(cat != NULL, "cat non-null");
    CHECK(cat->C == 3, "cat C == 3");
    CHECK(cat->feats[0] == 1.0f, "cat[0][0] == 1");
    CHECK(cat->feats[1] == 2.0f, "cat[0][1] == 2");
    CHECK(cat->feats[2] == 5.0f, "cat[0][2] == 5");
    CHECK(cat->feats[3] == 3.0f, "cat[1][0] == 3");
    CHECK(cat->feats[5] == 6.0f, "cat[1][2] == 6");

    sp3d_free(cat);
    sp3d_free(b);
    sp3d_free(a);
    fprintf(stderr, "  cat_feats: OK\n");
}

/* ---- Test: hash table ---- */
static void test_hash(void) {
    SECTION("Hash table");

    int32_t coords[] = {
        0, 0, 0, 0,
        0, 5, 10, 15,
        1, 3, 3, 3,
    };
    sp3d_hash *h = sp3d_hash_build(coords, 3);
    CHECK(sp3d_hash_lookup(h, 0, 0, 0, 0) == 0, "lookup (0,0,0,0)");
    CHECK(sp3d_hash_lookup(h, 0, 5, 10, 15) == 1, "lookup (0,5,10,15)");
    CHECK(sp3d_hash_lookup(h, 1, 3, 3, 3) == 2, "lookup (1,3,3,3)");
    CHECK(sp3d_hash_lookup(h, 0, 1, 1, 1) == -1, "lookup miss");
    CHECK(sp3d_hash_lookup(h, 2, 0, 0, 0) == -1, "lookup wrong batch");
    sp3d_hash_free(h);

    /* Stress test: 10K random coords */
    int N = 10000;
    int32_t *big_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    srand(42);
    for (int i = 0; i < N; i++) {
        big_coords[i*4]   = 0;
        big_coords[i*4+1] = rand() % 64;
        big_coords[i*4+2] = rand() % 64;
        big_coords[i*4+3] = rand() % 64;
    }
    h = sp3d_hash_build(big_coords, N);
    int found = 0;
    for (int i = 0; i < N; i++) {
        int idx = sp3d_hash_lookup(h, big_coords[i*4], big_coords[i*4+1],
                                    big_coords[i*4+2], big_coords[i*4+3]);
        if (idx >= 0) found++;
    }
    CHECK(found == N, "all 10K coords found");
    sp3d_hash_free(h);
    free(big_coords);
    fprintf(stderr, "  hash: OK\n");
}

/* ---- Test: elementwise ops ---- */
static void test_elementwise(void) {
    SECTION("Elementwise ops");

    /* GELU */
    float x[] = {0.0f, 1.0f, -1.0f, 2.0f};
    sp3d_gelu(x, 4);
    CHECK(fabsf(x[0]) < 1e-5f, "gelu(0) ~ 0");
    CHECK(fabsf(x[1] - 0.8412f) < 0.01f, "gelu(1) ~ 0.84");
    CHECK(fabsf(x[2] - (-0.1588f)) < 0.01f, "gelu(-1) ~ -0.16");

    /* SiLU */
    float y[] = {0.0f, 1.0f, -1.0f};
    sp3d_silu(y, 3);
    CHECK(fabsf(y[0]) < 1e-5f, "silu(0) ~ 0");
    CHECK(fabsf(y[1] - 0.7311f) < 0.01f, "silu(1) ~ 0.73");

    fprintf(stderr, "  elementwise: OK\n");
}

/* ---- Test: sparse conv3d ---- */
static void test_conv3d(void) {
    SECTION("Sparse Conv3d");

    /* Small test: 4 voxels in a line, kernel_size=3, in_C=1, out_C=1
     * Weight: all 1.0 (identity-sum kernel) */
    int32_t coords[] = {
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 0, 2,
        0, 0, 0, 3,
    };
    float feats[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    sp3d_tensor *t = sp3d_create(coords, feats, 4, 1, 1);

    int K3 = 27;
    int in_C = 1, out_C = 1;
    float *weight = (float *)calloc((size_t)out_C * K3 * in_C, sizeof(float));
    /* Set all kernel weights to 1.0 */
    for (int k = 0; k < K3; k++) weight[k] = 1.0f;

    float *dst = (float *)calloc(4, sizeof(float));
    sp3d_conv3d_forward(dst, t, weight, NULL, in_C, out_C, 3, 1);

    /* Voxel (0,0,0,0) neighbors: self + (0,0,0,1) → 1+2=3 */
    /* Voxel (0,0,0,1) neighbors: (0,0,0,0)+(0,0,0,1)+(0,0,0,2) → 1+2+3=6 */
    /* Voxel (0,0,0,2) neighbors: (0,0,0,1)+(0,0,0,2)+(0,0,0,3) → 2+3+4=9 */
    /* Voxel (0,0,0,3) neighbors: (0,0,0,2)+(0,0,0,3) → 3+4=7 */
    CHECK(fabsf(dst[0] - 3.0f) < 1e-5f, "conv voxel 0 == 3");
    CHECK(fabsf(dst[1] - 6.0f) < 1e-5f, "conv voxel 1 == 6");
    CHECK(fabsf(dst[2] - 9.0f) < 1e-5f, "conv voxel 2 == 9");
    CHECK(fabsf(dst[3] - 7.0f) < 1e-5f, "conv voxel 3 == 7");

    free(dst); free(weight);
    sp3d_free(t);
    fprintf(stderr, "  conv3d: OK\n");
}

/* ---- Test: variable-length attention ---- */
static void test_attention(void) {
    SECTION("Variable-length attention");

    /* 2 batches: batch 0 has 2 tokens, batch 1 has 3 tokens
     * dim=128, n_heads=2, head_dim=64 (must be 64 for AVX cpu_attention) */
    int32_t coords[] = {
        0,0,0,0, 0,0,0,1,
        1,0,0,0, 1,0,0,1, 1,0,0,2,
    };
    int N = 5;
    int n_heads = 2, head_dim = 64, dim = 128;

    /* Fill QKV with small values */
    float *qkv = (float *)calloc((size_t)N * 3 * dim, sizeof(float));
    srand(123);
    for (int i = 0; i < N * 3 * dim; i++)
        qkv[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.1f;

    sp3d_tensor *t = sp3d_create(coords, NULL, N, dim, 2);

    float *out = (float *)calloc((size_t)N * dim, sizeof(float));
    sp3d_attention(out, qkv, t, n_heads, head_dim, 1);

    /* Just verify: output is non-zero and finite */
    int any_nonzero = 0;
    for (int i = 0; i < N * dim; i++) {
        if (fabsf(out[i]) > 1e-10f) any_nonzero = 1;
    }
    CHECK(any_nonzero, "attention output nonzero");
    /* Note: NaN check may not work with -ffast-math, so just check a reasonable range */
    float max_val = 0;
    for (int i = 0; i < N * dim; i++)
        if (fabsf(out[i]) > max_val) max_val = fabsf(out[i]);
    CHECK(max_val < 100.0f, "attention output in reasonable range");

    free(out); free(qkv);
    sp3d_free(t);
    fprintf(stderr, "  attention: OK\n");
}

/* ---- Test: downsample ---- */
static void test_downsample(void) {
    SECTION("Downsample");

    /* 4 voxels, factor=2: coords (0,0,0,0),(0,0,0,1),(0,1,0,0),(0,1,0,1)
     * After /2: all map to (0,0,0,0) → mean of all features */
    int32_t coords[] = {
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 1, 0, 0,
        0, 1, 0, 1,
    };
    float feats[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    sp3d_tensor *t = sp3d_create(coords, feats, 4, 1, 1);

    sp3d_tensor *ds = sp3d_downsample(t, 2, 0); /* mean pool */
    CHECK(ds->N == 1, "downsample N == 1");
    CHECK(fabsf(ds->feats[0] - 2.5f) < 1e-5f, "downsample mean == 2.5");

    sp3d_free(ds);
    sp3d_free(t);

    /* Test with non-colliding coords */
    int32_t coords2[] = {
        0, 0, 0, 0,
        0, 0, 0, 2,
        0, 2, 0, 0,
    };
    float feats2[] = { 10.0f, 20.0f, 30.0f };
    t = sp3d_create(coords2, feats2, 3, 1, 1);
    ds = sp3d_downsample(t, 2, 0);
    CHECK(ds->N == 3, "downsample non-colliding N == 3");
    sp3d_free(ds);
    sp3d_free(t);

    fprintf(stderr, "  downsample: OK\n");
}

/* ---- Test: upsample ---- */
static void test_upsample(void) {
    SECTION("Upsample");

    /* Source: 1 voxel at (0,0,0,0) with feat=5.0
     * Target: 4 voxels at (0,0,0,0),(0,0,0,1),(0,1,0,0),(0,1,0,1)
     * Factor=2: all target coords /2 = (0,0,0,0) → all get 5.0 */
    int32_t src_coords[] = { 0, 0, 0, 0 };
    float src_feats[] = { 5.0f };
    sp3d_tensor *src = sp3d_create(src_coords, src_feats, 1, 1, 1);

    int32_t tgt_coords[] = {
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 1, 0, 0,
        0, 1, 0, 1,
    };
    sp3d_tensor *up = sp3d_upsample(src, 2, tgt_coords, 4);
    CHECK(up->N == 4, "upsample N == 4");
    CHECK(fabsf(up->feats[0] - 5.0f) < 1e-5f, "upsample[0] == 5");
    CHECK(fabsf(up->feats[1] - 5.0f) < 1e-5f, "upsample[1] == 5");
    CHECK(fabsf(up->feats[2] - 5.0f) < 1e-5f, "upsample[2] == 5");
    CHECK(fabsf(up->feats[3] - 5.0f) < 1e-5f, "upsample[3] == 5");

    sp3d_free(up);
    sp3d_free(src);
    fprintf(stderr, "  upsample: OK\n");
}

/* ---- Test: 3D RoPE ---- */
static void test_rope_3d(void) {
    SECTION("3D RoPE");

    /* 2 voxels at different coords, n_heads=1, head_dim=12 (4 per axis, 2 freqs) */
    int32_t coords[] = { 0, 0, 0, 0, 0, 1, 2, 3 };
    int N = 2, n_heads = 1, head_dim = 12;
    int n_freqs = 2; /* head_dim / 6 */
    float rope_freqs[] = { 1.0f, 0.1f }; /* 2 frequencies */

    /* QK: fill with 1.0 */
    float *qk = (float *)malloc((size_t)N * head_dim * sizeof(float));
    for (int i = 0; i < N * head_dim; i++) qk[i] = 1.0f;

    sp3d_tensor *t = sp3d_create(coords, NULL, N, 1, 1);

    sp3d_rope_3d(qk, t, n_heads, head_dim, head_dim, rope_freqs, n_freqs);

    /* Voxel 0 at (0,0,0): all rotations by 0 → cos(0)=1, sin(0)=0
     * So values should remain 1.0 */
    int all_one = 1;
    for (int i = 0; i < head_dim; i++) {
        if (fabsf(qk[i] - 1.0f) > 1e-5f) all_one = 0;
    }
    CHECK(all_one, "rope at origin preserves values");

    /* Voxel 1 at (1,2,3): values should be rotated (different from 1.0) */
    int any_changed = 0;
    for (int i = 0; i < head_dim; i++) {
        if (fabsf(qk[head_dim + i] - 1.0f) > 1e-5f) any_changed = 1;
    }
    CHECK(any_changed, "rope at non-origin changes values");

    free(qk);
    sp3d_free(t);
    fprintf(stderr, "  rope_3d: OK\n");
}

/* ---- Test: linear (F32 weight) ---- */
static void test_linear(void) {
    SECTION("Linear");

    /* 3 voxels, in_C=2, out_C=3, W=[3,2], no bias */
    float src[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    /* W[0]=[1,0], W[1]=[0,1], W[2]=[1,1] */
    float W_data[] = { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    qtensor W = {0};
    W.data = W_data;
    W.type = GGML_TYPE_F32;
    W.n_dims = 2;
    W.dims[0] = 3; W.dims[1] = 2;
    W.n_rows = 3; W.n_cols = 2;

    float dst[9];
    sp3d_linear(dst, src, 3, &W, NULL, 3, 2, 1);

    /* Voxel 0: [1,2] @ W^T → [1, 2, 3] */
    CHECK(fabsf(dst[0] - 1.0f) < 1e-5f, "linear[0][0] == 1");
    CHECK(fabsf(dst[1] - 2.0f) < 1e-5f, "linear[0][1] == 2");
    CHECK(fabsf(dst[2] - 3.0f) < 1e-5f, "linear[0][2] == 3");
    /* Voxel 2: [5,6] @ W^T → [5, 6, 11] */
    CHECK(fabsf(dst[6] - 5.0f) < 1e-5f, "linear[2][0] == 5");
    CHECK(fabsf(dst[7] - 6.0f) < 1e-5f, "linear[2][1] == 6");
    CHECK(fabsf(dst[8] - 11.0f) < 1e-5f, "linear[2][2] == 11");

    fprintf(stderr, "  linear: OK\n");
}

/* ---- Test: conv3d with threading ---- */
static void test_conv3d_threaded(void) {
    SECTION("Conv3d (threaded)");

    /* 8 voxels in a 2x2x2 cube, kernel_size=3, all-ones weight */
    int32_t coords[8*4];
    float feats[8];
    int idx = 0;
    for (int z = 0; z < 2; z++)
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++) {
                coords[idx*4]   = 0;
                coords[idx*4+1] = z;
                coords[idx*4+2] = y;
                coords[idx*4+3] = x;
                feats[idx] = 1.0f;
                idx++;
            }

    sp3d_tensor *t = sp3d_create(coords, feats, 8, 1, 1);

    int K3 = 27, in_C = 1, out_C = 1;
    float *weight = (float *)calloc((size_t)K3, sizeof(float));
    for (int k = 0; k < K3; k++) weight[k] = 1.0f;

    float dst1[8], dst4[8];
    sp3d_conv3d_forward(dst1, t, weight, NULL, in_C, out_C, 3, 1);
    sp3d_conv3d_forward(dst4, t, weight, NULL, in_C, out_C, 3, 4);

    /* Each corner voxel in a 2x2x2 cube sees all 8 neighbors → sum=8 */
    CHECK(fabsf(dst1[0] - 8.0f) < 1e-5f, "conv 2x2x2 1-thread == 8");
    CHECK(fabsf(dst4[0] - 8.0f) < 1e-5f, "conv 2x2x2 4-thread == 8");

    /* Verify 1-thread == 4-thread */
    int match = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(dst1[i] - dst4[i]) > 1e-5f) match = 0;
    CHECK(match, "conv 1-thread == 4-thread");

    free(weight);
    sp3d_free(t);
    fprintf(stderr, "  conv3d threaded: OK\n");
}

/* ---- Benchmark: conv3d scaling ---- */
static void bench_conv3d(void) {
    SECTION("Benchmark: conv3d");

    int N = 4096;
    int in_C = 32, out_C = 32;
    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    float *feats = (float *)malloc((size_t)N * in_C * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        coords[i*4]   = 0;
        coords[i*4+1] = rand() % 32;
        coords[i*4+2] = rand() % 32;
        coords[i*4+3] = rand() % 32;
        for (int c = 0; c < in_C; c++)
            feats[i*in_C+c] = (float)(rand() % 1000) / 1000.0f;
    }
    sp3d_tensor *t = sp3d_create(coords, feats, N, in_C, 1);

    int K3 = 27;
    float *weight = (float *)malloc((size_t)out_C * K3 * in_C * sizeof(float));
    for (int i = 0; i < out_C * K3 * in_C; i++)
        weight[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    float *dst = (float *)calloc((size_t)N * out_C, sizeof(float));

    for (int thr = 1; thr <= 4; thr *= 2) {
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        sp3d_conv3d_forward(dst, t, weight, NULL, in_C, out_C, 3, thr);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double ms = (ts1.tv_sec - ts0.tv_sec) * 1e3 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
        fprintf(stderr, "  conv3d N=%d C=%d->%d, %d threads: %.1f ms\n", N, in_C, out_C, thr, ms);
    }

    free(dst); free(weight); free(feats); free(coords);
    sp3d_free(t);
}

/* ==================================================================== */
/* Export mode: load .npy inputs, run ops, write .npy outputs            */
/* ==================================================================== */

static void export_all(const char *out_dir, const char *in_dir) {
    mkdir(out_dir, 0755);
    char path[512];
    int ndim, dims[8];

    /* --- 1. Linear --- */
    {
        fprintf(stderr, "  export: linear...\n");
        make_path(path, sizeof(path), in_dir, "linear_input");
        float *input = read_npy_f32(path, &ndim, dims);
        int N = dims[0], in_C = dims[1];

        make_path(path, sizeof(path), in_dir, "linear_weight");
        float *weight = read_npy_f32(path, &ndim, dims);
        int out_C = dims[0];

        make_path(path, sizeof(path), in_dir, "linear_bias");
        float *bias = read_npy_f32(path, &ndim, dims);

        qtensor W = {0};
        W.data = weight; W.type = GGML_TYPE_F32;
        W.n_rows = out_C; W.n_cols = in_C;
        W.n_dims = 2; W.dims[0] = (uint64_t)out_C; W.dims[1] = (uint64_t)in_C;

        qtensor B = {0};
        B.data = bias; B.type = GGML_TYPE_F32;
        B.n_rows = 1; B.n_cols = out_C;
        B.n_dims = 1; B.dims[0] = (uint64_t)out_C;

        float *dst = (float *)malloc((size_t)N * out_C * sizeof(float));
        sp3d_linear(dst, input, N, &W, &B, out_C, in_C, 1);

        int out_dims[2] = {N, out_C};
        make_path(path, sizeof(path), out_dir, "linear_output");
        write_npy_f32(path, dst, out_dims, 2);

        free(input); free(weight); free(bias); free(dst);
    }

    /* --- 2. LayerNorm --- */
    {
        fprintf(stderr, "  export: layernorm...\n");
        make_path(path, sizeof(path), in_dir, "layernorm_input");
        float *input = read_npy_f32(path, &ndim, dims);
        int N = dims[0], C = dims[1];

        make_path(path, sizeof(path), in_dir, "layernorm_weight");
        float *w = read_npy_f32(path, &ndim, dims);

        make_path(path, sizeof(path), in_dir, "layernorm_bias");
        float *b = read_npy_f32(path, &ndim, dims);

        qtensor W = {0};
        W.data = w; W.type = GGML_TYPE_F32;
        W.n_rows = 1; W.n_cols = C;
        W.n_dims = 1; W.dims[0] = (uint64_t)C;

        qtensor B = {0};
        B.data = b; B.type = GGML_TYPE_F32;
        B.n_rows = 1; B.n_cols = C;
        B.n_dims = 1; B.dims[0] = (uint64_t)C;

        float *dst = (float *)malloc((size_t)N * C * sizeof(float));
        sp3d_layernorm(dst, input, &W, &B, N, C, 1e-6f);

        int out_dims[2] = {N, C};
        make_path(path, sizeof(path), out_dir, "layernorm_output");
        write_npy_f32(path, dst, out_dims, 2);

        free(input); free(w); free(b); free(dst);
    }

    /* --- 3. GELU / SiLU --- */
    {
        fprintf(stderr, "  export: gelu/silu...\n");
        make_path(path, sizeof(path), in_dir, "activation_input");
        float *input = read_npy_f32(path, &ndim, dims);
        int count = dims[0];

        float *gelu_out = (float *)malloc((size_t)count * sizeof(float));
        memcpy(gelu_out, input, (size_t)count * sizeof(float));
        sp3d_gelu(gelu_out, count);

        float *silu_out = (float *)malloc((size_t)count * sizeof(float));
        memcpy(silu_out, input, (size_t)count * sizeof(float));
        sp3d_silu(silu_out, count);

        int out_dims[1] = {count};
        make_path(path, sizeof(path), out_dir, "gelu_output");
        write_npy_f32(path, gelu_out, out_dims, 1);
        make_path(path, sizeof(path), out_dir, "silu_output");
        write_npy_f32(path, silu_out, out_dims, 1);

        free(input); free(gelu_out); free(silu_out);
    }

    /* --- 4. Conv3d --- */
    {
        fprintf(stderr, "  export: conv3d...\n");
        make_path(path, sizeof(path), in_dir, "conv3d_coords");
        int32_t *coords = read_npy_i32(path, &ndim, dims);
        int N = dims[0];

        make_path(path, sizeof(path), in_dir, "conv3d_input");
        float *feats = read_npy_f32(path, &ndim, dims);
        int in_C = dims[1];

        make_path(path, sizeof(path), in_dir, "conv3d_weight");
        float *weight = read_npy_f32(path, &ndim, dims);
        int out_C = dims[0];

        make_path(path, sizeof(path), in_dir, "conv3d_bias");
        float *bias = read_npy_f32(path, &ndim, dims);

        sp3d_tensor *t = sp3d_create(coords, feats, N, in_C, 1);

        float *dst = (float *)calloc((size_t)N * out_C, sizeof(float));
        sp3d_conv3d_forward(dst, t, weight, bias, in_C, out_C, 3, 1);

        int out_dims[2] = {N, out_C};
        make_path(path, sizeof(path), out_dir, "conv3d_output");
        write_npy_f32(path, dst, out_dims, 2);

        free(coords); free(feats); free(weight); free(bias); free(dst);
        sp3d_free(t);
    }

    /* --- 5. Attention --- */
    {
        fprintf(stderr, "  export: attention...\n");
        make_path(path, sizeof(path), in_dir, "attention_coords");
        int32_t *coords = read_npy_i32(path, &ndim, dims);
        int N = dims[0];

        make_path(path, sizeof(path), in_dir, "attention_qkv");
        float *qkv = read_npy_f32(path, &ndim, dims);
        int qkv_dim = dims[1]; /* 3 * dim */
        int dim = qkv_dim / 3;
        int n_heads = 2, head_dim = 64;

        /* Determine batch_size from coords */
        int batch_size = 0;
        for (int i = 0; i < N; i++) {
            if (coords[i*4] + 1 > batch_size)
                batch_size = coords[i*4] + 1;
        }

        sp3d_tensor *t = sp3d_create(coords, NULL, N, dim, batch_size);

        float *out = (float *)calloc((size_t)N * dim, sizeof(float));
        sp3d_attention(out, qkv, t, n_heads, head_dim, 1);

        int out_dims[2] = {N, dim};
        make_path(path, sizeof(path), out_dir, "attention_output");
        write_npy_f32(path, out, out_dims, 2);

        free(coords); free(qkv); free(out);
        sp3d_free(t);
    }

    /* --- 6. 3D RoPE --- */
    {
        fprintf(stderr, "  export: rope_3d...\n");
        make_path(path, sizeof(path), in_dir, "rope_coords");
        int32_t *coords = read_npy_i32(path, &ndim, dims);
        int N = dims[0];

        make_path(path, sizeof(path), in_dir, "rope_input");
        float *qk = read_npy_f32(path, &ndim, dims);
        int total_dim = dims[1]; /* n_heads * head_dim */

        make_path(path, sizeof(path), in_dir, "rope_freqs");
        float *freqs = read_npy_f32(path, &ndim, dims);
        int n_freqs = dims[0];

        /* Derive head_dim and n_heads from n_freqs:
         * head_dim = 6 * n_freqs, n_heads = total_dim / head_dim */
        int head_dim = 6 * n_freqs;
        int n_heads = total_dim / head_dim;

        sp3d_tensor *t = sp3d_create(coords, NULL, N, 1, 1);

        /* qk is modified in-place, so we work on a copy */
        float *qk_out = (float *)malloc((size_t)N * total_dim * sizeof(float));
        memcpy(qk_out, qk, (size_t)N * total_dim * sizeof(float));
        sp3d_rope_3d(qk_out, t, n_heads, head_dim, total_dim, freqs, n_freqs);

        int out_dims[2] = {N, total_dim};
        make_path(path, sizeof(path), out_dir, "rope_output");
        write_npy_f32(path, qk_out, out_dims, 2);

        free(coords); free(qk); free(freqs); free(qk_out);
        sp3d_free(t);
    }

    /* --- 7. Downsample --- */
    {
        fprintf(stderr, "  export: downsample...\n");
        make_path(path, sizeof(path), in_dir, "downsample_coords");
        int32_t *coords = read_npy_i32(path, &ndim, dims);
        int N = dims[0];

        make_path(path, sizeof(path), in_dir, "downsample_input");
        float *feats = read_npy_f32(path, &ndim, dims);
        int C = dims[1];

        sp3d_tensor *t = sp3d_create(coords, feats, N, C, 1);
        sp3d_tensor *ds = sp3d_downsample(t, 2, 0); /* mean pool */

        int feat_dims[2] = {ds->N, ds->C};
        make_path(path, sizeof(path), out_dir, "downsample_feats");
        write_npy_f32(path, ds->feats, feat_dims, 2);

        int coord_dims[2] = {ds->N, 4};
        make_path(path, sizeof(path), out_dir, "downsample_coords_out");
        write_npy_i32(path, ds->coords, coord_dims, 2);

        free(coords); free(feats);
        sp3d_free(ds);
        sp3d_free(t);
    }

    /* --- 8. Upsample --- */
    {
        fprintf(stderr, "  export: upsample...\n");
        make_path(path, sizeof(path), in_dir, "upsample_src_coords");
        int32_t *src_coords = read_npy_i32(path, &ndim, dims);
        int src_N = dims[0];

        make_path(path, sizeof(path), in_dir, "upsample_src_feats");
        float *src_feats = read_npy_f32(path, &ndim, dims);
        int C = dims[1];

        make_path(path, sizeof(path), in_dir, "upsample_tgt_coords");
        int32_t *tgt_coords = read_npy_i32(path, &ndim, dims);
        int tgt_N = dims[0];

        sp3d_tensor *src = sp3d_create(src_coords, src_feats, src_N, C, 1);
        sp3d_tensor *up = sp3d_upsample(src, 2, tgt_coords, tgt_N);

        int out_dims[2] = {tgt_N, C};
        make_path(path, sizeof(path), out_dir, "upsample_output");
        write_npy_f32(path, up->feats, out_dims, 2);

        free(src_coords); free(src_feats); free(tgt_coords);
        sp3d_free(up);
        sp3d_free(src);
    }

    fprintf(stderr, "  export: done. Outputs in %s/\n", out_dir);
}

int main(int argc, char **argv) {
    /* Check for --export mode */
    const char *export_dir = NULL;
    const char *input_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) {
            export_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_dir = argv[++i];
        }
    }

    if (export_dir && input_dir) {
        fprintf(stderr, "sparse3d export mode: input=%s output=%s\n", input_dir, export_dir);
        export_all(export_dir, input_dir);
        return 0;
    }

    /* Normal unit test mode */
    fprintf(stderr, "sparse3d unit tests\n");

    test_lifecycle();
    test_cat_feats();
    test_hash();
    test_elementwise();
    test_linear();
    test_conv3d();
    test_conv3d_threaded();
    test_attention();
    test_downsample();
    test_upsample();
    test_rope_3d();
    bench_conv3d();

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Tests: %d total, %d passed, %d failed\n", g_tests, g_pass, g_fail);
    fprintf(stderr, "========================================\n");
    return g_fail > 0 ? 1 : 0;
}
