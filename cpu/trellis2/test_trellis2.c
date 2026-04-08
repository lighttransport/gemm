/*
 * test_trellis2.c - Test harness for TRELLIS.2 Stage 1 (Structure Generation)
 *
 * Usage:
 *   ./test_trellis2 full <dinov3.st> <stage1.st> <decoder.st> <image> [-t N] [-o out.npy]
 *   ./test_trellis2 stage1 <stage1.st> <features.npy> [-t N] [-o structure.npy]
 *   ./test_trellis2 decode <decoder.st> <structure.npy> [-t N] [-o occupancy.npy]
 *   ./test_trellis2 encode <dinov3.st> <image> [-t N] [-o features.npy]
 *
 * Modes:
 *   full    - End-to-end: image -> DINOv3 -> Stage1 DiT -> Decoder -> occupancy
 *   stage1  - Stage1 only: DINOv3 features (.npy) -> DiT -> latent
 *   decode  - Decoder only: latent (.npy) -> occupancy
 *   encode  - DINOv3 only: image -> features
 *
 * Build:
 *   make
 */

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define DINOV3_IMPLEMENTATION
#include "../../common/dinov3.h"

#define T2DIT_IMPLEMENTATION
#include "../../common/trellis2_dit.h"

#define T2_STAGE1_IMPLEMENTATION
#include "../../common/trellis2_stage1.h"

#define T2_SS_DEC_IMPLEMENTATION
#include "../../common/trellis2_ss_decoder.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ---- NumPy .npy I/O ---- */

static void write_npy_f32(const char *path, const float *data,
                            const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char shape_str[256];
    int slen = 0;
    slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, "(");
    size_t n_elem = 1;
    for (int i = 0; i < ndim; i++) {
        slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, "%d,", dims[i]);
        n_elem *= (size_t)dims[i];
    }
    slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, ")");
    char header[512];
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
    fprintf(stderr, "Wrote %s (", path);
    for (int i = 0; i < ndim; i++) fprintf(stderr, "%s%d", i ? "x" : "", dims[i]);
    fprintf(stderr, ", float32)\n");
}

/* Read a float32 .npy file. Returns data pointer (caller frees).
 * Sets *ndim and dims[]. */
static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    /* Skip magic (6) + version (2) */
    fseek(f, 8, SEEK_SET);
    uint16_t header_len;
    if (fread(&header_len, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *header = (char *)malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != header_len) {
        free(header); fclose(f); return NULL;
    }
    header[header_len] = '\0';

    /* Parse shape from header: 'shape': (d0, d1, ...) */
    *ndim = 0;
    char *sp = strstr(header, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ' || *sp == ',') sp++;
                if (*sp == ')') break;
                dims[*ndim] = (int)strtol(sp, &sp, 10);
                (*ndim)++;
                if (*ndim >= 8) break;
            }
        }
    }

    size_t n_elem = 1;
    for (int i = 0; i < *ndim; i++) n_elem *= (size_t)dims[i];

    float *data = (float *)malloc(n_elem * sizeof(float));
    size_t got = fread(data, sizeof(float), n_elem, f);
    fclose(f);
    free(header);

    if (got != n_elem) {
        fprintf(stderr, "Warning: read %zu of %zu elements from %s\n", got, n_elem, path);
    }

    fprintf(stderr, "Read %s: (", path);
    for (int i = 0; i < *ndim; i++) fprintf(stderr, "%s%d", i ? ", " : "", dims[i]);
    fprintf(stderr, "), %zu elements\n", n_elem);
    return data;
}

static void print_stats(const char *label, const float *data, int n) {
    if (n <= 0) return;
    float min_v = data[0], max_v = data[0];
    double sum = 0;
    for (int i = 0; i < n; i++) {
        float v = data[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
    }
    fprintf(stderr, "%s: min=%.4f, max=%.4f, mean=%.6f\n",
            label, min_v, max_v, sum / n);
}

/* ---- Mode: encode ---- */

static int mode_encode(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: encode <dinov3.st> <image> [-t N] [-o features.npy]\n");
        return 1;
    }
    const char *model_path = argv[0];
    const char *image_path = argv[1];
    int n_threads = 4;
    const char *out_path = "dinov3_features.npy";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
    }

    fprintf(stderr, "Loading DINOv3 model...\n");
    dinov3_model *m = dinov3_load_safetensors(model_path);
    if (!m) return 1;

    int img_w, img_h, channels;
    uint8_t *img = stbi_load(image_path, &img_w, &img_h, &channels, 3);
    if (!img) { fprintf(stderr, "Failed to load image: %s\n", image_path); dinov3_free(m); return 1; }
    fprintf(stderr, "Image: %dx%d\n", img_w, img_h);

    dinov3_result result = dinov3_encode(m, img, img_w, img_h, n_threads);
    free(img);

    if (result.features) {
        int dims[2] = {result.n_tokens, result.dim};
        write_npy_f32(out_path, result.features, dims, 2);
        print_stats("features", result.features, result.n_tokens * result.dim);
        dinov3_result_free(&result);
    }
    dinov3_free(m);
    return 0;
}

/* ---- Mode: stage1 ---- */

static int mode_stage1(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: stage1 <stage1.st> <features.npy> [-t N] [-o structure.npy] [-s seed]\n");
        return 1;
    }
    const char *model_path = argv[0];
    const char *features_path = argv[1];
    int n_threads = 4;
    const char *out_path = "stage1_latent.npy";
    uint64_t seed = 42;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
    }

    /* Load features */
    int ndim, dims[8];
    float *features = read_npy_f32(features_path, &ndim, dims);
    if (!features) return 1;
    int n_cond = dims[0];
    (void)ndim;  /* cond_dim auto-detected by model */

    /* Load Stage 1 model */
    fprintf(stderr, "Loading Stage 1 model...\n");
    t2_stage1 *s = t2_stage1_load(model_path);
    if (!s) { free(features); return 1; }

    /* Sample */
    float *latent = t2_stage1_sample(s, features, n_cond, n_threads, seed);
    free(features);

    if (latent) {
        /* Output as [8, 16, 16, 16] */
        int out_dims[4] = {s->dit->in_channels, s->dit->grid_size,
                           s->dit->grid_size, s->dit->grid_size};
        write_npy_f32(out_path, latent, out_dims, 4);
        print_stats("latent", latent, s->dit->n_tokens * s->dit->in_channels);
        free(latent);
    }
    t2_stage1_free(s);
    return 0;
}

/* ---- Mode: decode ---- */

static int mode_decode(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: decode <decoder.st> <structure.npy> [-t N] [-o occupancy.npy]\n");
        return 1;
    }
    const char *model_path = argv[0];
    const char *latent_path = argv[1];
    int n_threads = 4;
    const char *out_path = "occupancy.npy";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
    }

    /* Load latent */
    int ndim, dims[8];
    float *latent = read_npy_f32(latent_path, &ndim, dims);
    if (!latent) return 1;

    /* Load decoder */
    fprintf(stderr, "Loading structure decoder...\n");
    t2_ss_dec *d = t2_ss_dec_load(model_path);
    if (!d) { free(latent); return 1; }

    /* Forward */
    float *occupancy = t2_ss_dec_forward(d, latent, n_threads);
    free(latent);

    if (occupancy) {
        int out_dims[3] = {64, 64, 64};
        write_npy_f32(out_path, occupancy, out_dims, 3);
        print_stats("occupancy", occupancy, 64 * 64 * 64);

        /* Count occupied voxels at threshold 0.0 */
        int count = 0;
        for (int i = 0; i < 64 * 64 * 64; i++)
            if (occupancy[i] > 0.0f) count++;
        fprintf(stderr, "Occupied voxels (logit > 0): %d / %d (%.1f%%)\n",
                count, 64*64*64, 100.0f * count / (64*64*64));
        free(occupancy);
    }
    t2_ss_dec_free(d);
    return 0;
}

/* ---- Mode: full ---- */

static int mode_full(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: full <dinov3.st> <stage1.st> <decoder.st> <image> "
                "[-t N] [-o occupancy.npy] [-s seed]\n");
        return 1;
    }
    const char *dinov3_path = argv[0];
    const char *stage1_path = argv[1];
    const char *decoder_path = argv[2];
    const char *image_path = argv[3];
    int n_threads = 4;
    const char *out_path = "occupancy.npy";
    uint64_t seed = 42;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
    }

    /* Step 1: DINOv3 encode */
    fprintf(stderr, "=== Step 1: DINOv3 Encoding ===\n");
    dinov3_model *dm = dinov3_load_safetensors(dinov3_path);
    if (!dm) return 1;

    int img_w, img_h, channels;
    uint8_t *img = stbi_load(image_path, &img_w, &img_h, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        dinov3_free(dm);
        return 1;
    }
    fprintf(stderr, "Image: %dx%d\n", img_w, img_h);

    dinov3_result dr = dinov3_encode(dm, img, img_w, img_h, n_threads);
    free(img);
    dinov3_free(dm);
    if (!dr.features) { fprintf(stderr, "DINOv3 encoding failed\n"); return 1; }
    print_stats("DINOv3 features", dr.features, dr.n_tokens * dr.dim);

    /* Save intermediate features */
    {
        int dims[2] = {dr.n_tokens, dr.dim};
        write_npy_f32("stage1_cond.npy", dr.features, dims, 2);
    }

    /* Step 2: Stage 1 sampling */
    fprintf(stderr, "\n=== Step 2: Stage 1 Flow Sampling ===\n");
    t2_stage1 *s = t2_stage1_load(stage1_path);
    if (!s) { dinov3_result_free(&dr); return 1; }

    float *latent = t2_stage1_sample(s, dr.features, dr.n_tokens, n_threads, seed);
    dinov3_result_free(&dr);
    if (!latent) { t2_stage1_free(s); return 1; }
    print_stats("latent", latent, s->dit->n_tokens * s->dit->in_channels);

    /* Save intermediate latent */
    {
        int dims[4] = {s->dit->in_channels, s->dit->grid_size,
                       s->dit->grid_size, s->dit->grid_size};
        write_npy_f32("stage1_latent.npy", latent, dims, 4);
    }
    t2_stage1_free(s);

    /* Step 3: Decode */
    fprintf(stderr, "\n=== Step 3: Structure Decoding ===\n");
    t2_ss_dec *dec = t2_ss_dec_load(decoder_path);
    if (!dec) { free(latent); return 1; }

    float *occupancy = t2_ss_dec_forward(dec, latent, n_threads);
    free(latent);
    t2_ss_dec_free(dec);

    if (occupancy) {
        int out_dims[3] = {64, 64, 64};
        write_npy_f32(out_path, occupancy, out_dims, 3);
        print_stats("occupancy", occupancy, 64 * 64 * 64);

        int count = 0;
        for (int i = 0; i < 64 * 64 * 64; i++)
            if (occupancy[i] > 0.0f) count++;
        fprintf(stderr, "Occupied voxels (logit > 0): %d / %d (%.1f%%)\n",
                count, 64*64*64, 100.0f * count / (64*64*64));
        free(occupancy);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}

/* ==================================================================== */
/* Marching Cubes — .obj mesh export from occupancy grid                 */
/* ==================================================================== */

/* Standard marching cubes edge table: for each of 256 cube configs,
 * a 12-bit mask of which edges have surface intersections. */
static const int mc_edge_table[256] = {
    0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000
};

/* Triangle table: for each of 256 configs, up to 5 triangles (15 edge indices), -1 terminated. */
static const int mc_tri_table[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
    {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
    {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
    {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
    {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
    {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
    {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
    {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
    {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
    {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
    {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
    {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
    {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
    {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
    {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
    {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
    {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
    {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
    {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
    {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
    {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
    {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
    {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
    {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
    {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
    {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
    {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
    {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
    {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
    {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
    {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
    {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
    {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
    {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
    {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
    {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
    {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
    {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
    {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
    {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
    {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
    {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
    {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
    {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
    {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
    {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
    {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
    {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
    {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
    {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
    {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
    {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
    {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
    {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
    {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
    {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
    {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
    {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
    {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
    {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
    {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
    {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
    {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
    {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
    {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
    {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
    {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
    {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
    {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
    {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
    {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
    {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
    {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
    {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
    {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
    {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
    {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
    {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
    {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
    {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
    {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
    {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
    {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
    {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
    {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
    {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
    {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
    {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
    {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
    {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
    {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
    {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
    {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
    {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
    {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
    {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
    {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
    {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
    {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
    {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
    {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
    {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
    {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
    {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
    {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
    {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
    {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
    {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
    {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
    {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
    {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
    {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
    {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
    {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
    {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
    {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
    {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
    {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
    {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
    {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
    {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
    {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
    {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
    {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
    {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
    {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
    {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
    {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
    {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
    {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
    {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
    {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
    {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
    {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
    {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
    {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
    {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
    {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
    {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
    {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
    {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
    {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
    {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

/* Edge endpoints: edge i connects corner mc_edge_conn[i][0] to mc_edge_conn[i][1] */
static const int mc_edge_conn[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
};

/* Corner offsets in (z,y,x): corner i is at (dz,dy,dx) */
static const int mc_corner[8][3] = {
    {0,0,0},{0,0,1},{0,1,1},{0,1,0},{1,0,0},{1,0,1},{1,1,1},{1,1,0}
};

static float mc_interp(float v0, float v1, float iso) {
    if (fabsf(v0 - v1) < 1e-10f) return 0.5f;
    return (iso - v0) / (v1 - v0);
}

/* Run marching cubes on a [nz, ny, nx] grid and write .obj file.
 * Grid values > iso are considered "inside". */
static int write_obj_marching_cubes(const char *path, const float *grid,
                                      int nz, int ny, int nx, float iso) {
    /* Pass 1: count vertices and faces */
    /* Pass 2: emit them */
    /* We use a vertex dedup hash to avoid duplicate vertices on shared edges. */

    /* Allocate generous buffers (worst case: ~5 tris per cell) */
    int max_verts = nz * ny * nx * 3;
    int max_faces = nz * ny * nx * 5;
    float *verts = (float *)malloc((size_t)max_verts * 3 * sizeof(float));
    int *faces = (int *)malloc((size_t)max_faces * 3 * sizeof(int));
    int n_verts = 0, n_faces = 0;

    /* Edge-vertex index cache: for dedup, cache vertex index per edge.
     * Key: encode (z, y, x, edge_dir) -> vertex index.
     * edge_dir: 0=x-edge, 1=y-edge, 2=z-edge at corner (z,y,x).
     * We only need edges along 3 directions from each grid point. */
    /* Use a flat array: [nz+1][ny+1][nx+1][3], value = vertex index or -1 */
    int *edge_cache = (int *)malloc((size_t)(nz+1) * (ny+1) * (nx+1) * 3 * sizeof(int));
    memset(edge_cache, -1, (size_t)(nz+1) * (ny+1) * (nx+1) * 3 * sizeof(int));

    #define EC_IDX(z,y,x,d) (((z)*(ny+1)*(nx+1) + (y)*(nx+1) + (x))*3 + (d))

    /* Map MC edge index to (corner_offset_z, corner_offset_y, corner_offset_x, direction) */
    /* Edge 0: (0,0,0)-(0,0,1) -> base(0,0,0), dir=0(x) */
    /* Edge 1: (0,0,1)-(0,1,1) -> base(0,0,1), dir=1(y) */
    /* Edge 2: (0,1,0)-(0,1,1) -> base(0,1,0), dir=0(x) */
    /* Edge 3: (0,0,0)-(0,1,0) -> base(0,0,0), dir=1(y) */
    /* Edge 4: (1,0,0)-(1,0,1) -> base(1,0,0), dir=0(x) */
    /* Edge 5: (1,0,1)-(1,1,1) -> base(1,0,1), dir=1(y) */
    /* Edge 6: (1,1,0)-(1,1,1) -> base(1,1,0), dir=0(x) */
    /* Edge 7: (1,0,0)-(1,1,0) -> base(1,0,0), dir=1(y) */
    /* Edge 8: (0,0,0)-(1,0,0) -> base(0,0,0), dir=2(z) */
    /* Edge 9: (0,0,1)-(1,0,1) -> base(0,0,1), dir=2(z) */
    /* Edge10: (0,1,1)-(1,1,1) -> base(0,1,1), dir=2(z) */
    /* Edge11: (0,1,0)-(1,1,0) -> base(0,1,0), dir=2(z) */
    static const int edge_to_base_dir[12][4] = {
        /* dz, dy, dx, dir */
        {0,0,0, 0}, {0,0,1, 1}, {0,1,0, 0}, {0,0,0, 1},
        {1,0,0, 0}, {1,0,1, 1}, {1,1,0, 0}, {1,0,0, 1},
        {0,0,0, 2}, {0,0,1, 2}, {0,1,1, 2}, {0,1,0, 2}
    };

    for (int cz = 0; cz < nz - 1; cz++) {
        for (int cy = 0; cy < ny - 1; cy++) {
            for (int cx = 0; cx < nx - 1; cx++) {
                /* Read 8 corner values */
                float val[8];
                for (int c = 0; c < 8; c++) {
                    int gz = cz + mc_corner[c][0];
                    int gy = cy + mc_corner[c][1];
                    int gx = cx + mc_corner[c][2];
                    val[c] = grid[gz * ny * nx + gy * nx + gx];
                }

                /* Compute cube index */
                int cube_idx = 0;
                for (int c = 0; c < 8; c++)
                    if (val[c] > iso) cube_idx |= (1 << c);

                if (mc_edge_table[cube_idx] == 0) continue;

                /* Compute vertex indices for intersected edges */
                int vert_idx[12];
                for (int e = 0; e < 12; e++) {
                    if (!(mc_edge_table[cube_idx] & (1 << e))) {
                        vert_idx[e] = -1;
                        continue;
                    }
                    int bz = cz + edge_to_base_dir[e][0];
                    int by = cy + edge_to_base_dir[e][1];
                    int bx = cx + edge_to_base_dir[e][2];
                    int dir = edge_to_base_dir[e][3];
                    int ci = EC_IDX(bz, by, bx, dir);

                    if (edge_cache[ci] >= 0) {
                        vert_idx[e] = edge_cache[ci];
                    } else {
                        /* Interpolate vertex position */
                        int c0 = mc_edge_conn[e][0], c1 = mc_edge_conn[e][1];
                        float t = mc_interp(val[c0], val[c1], iso);
                        float vz = (float)cz + (float)mc_corner[c0][0] + t * (float)(mc_corner[c1][0] - mc_corner[c0][0]);
                        float vy = (float)cy + (float)mc_corner[c0][1] + t * (float)(mc_corner[c1][1] - mc_corner[c0][1]);
                        float vx = (float)cx + (float)mc_corner[c0][2] + t * (float)(mc_corner[c1][2] - mc_corner[c0][2]);

                        /* Normalize to [0, 1] */
                        vz /= (float)(nz - 1);
                        vy /= (float)(ny - 1);
                        vx /= (float)(nx - 1);

                        if (n_verts >= max_verts) {
                            max_verts *= 2;
                            verts = (float *)realloc(verts, (size_t)max_verts * 3 * sizeof(float));
                        }
                        verts[n_verts * 3 + 0] = vx;
                        verts[n_verts * 3 + 1] = vy;
                        verts[n_verts * 3 + 2] = vz;
                        edge_cache[ci] = n_verts;
                        vert_idx[e] = n_verts;
                        n_verts++;
                    }
                }

                /* Emit triangles */
                for (int i = 0; mc_tri_table[cube_idx][i] != -1; i += 3) {
                    if (n_faces >= max_faces) {
                        max_faces *= 2;
                        faces = (int *)realloc(faces, (size_t)max_faces * 3 * sizeof(int));
                    }
                    faces[n_faces * 3 + 0] = vert_idx[mc_tri_table[cube_idx][i]];
                    faces[n_faces * 3 + 1] = vert_idx[mc_tri_table[cube_idx][i + 1]];
                    faces[n_faces * 3 + 2] = vert_idx[mc_tri_table[cube_idx][i + 2]];
                    n_faces++;
                }
            }
        }
    }

    free(edge_cache);

    /* Write .obj */
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Cannot write %s\n", path);
        free(verts); free(faces);
        return -1;
    }
    fprintf(f, "# TRELLIS.2 Stage 1 marching cubes mesh\n");
    fprintf(f, "# %d vertices, %d faces\n", n_verts, n_faces);
    for (int i = 0; i < n_verts; i++)
        fprintf(f, "v %f %f %f\n", verts[i*3], verts[i*3+1], verts[i*3+2]);
    for (int i = 0; i < n_faces; i++)
        fprintf(f, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);

    fprintf(stderr, "Wrote %s (%d vertices, %d faces)\n", path, n_verts, n_faces);
    free(verts); free(faces);
    return 0;

    #undef EC_IDX
}

/* ---- Mode: mesh ---- */

static int mode_mesh(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr, "Usage: mesh <occupancy.npy> [-o output.obj] [-t threshold]\n");
        return 1;
    }
    const char *input_path = argv[0];
    const char *out_path = "output.obj";
    float threshold = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) threshold = (float)atof(argv[++i]);
    }

    int ndim, dims[8];
    float *data = read_npy_f32(input_path, &ndim, dims);
    if (!data) return 1;

    int nz = 64, ny = 64, nx = 64;
    if (ndim == 3) { nz = dims[0]; ny = dims[1]; nx = dims[2]; }
    else if (ndim == 4) { nz = dims[1]; ny = dims[2]; nx = dims[3]; }

    print_stats("occupancy", data, nz * ny * nx);
    int count = 0;
    for (int i = 0; i < nz * ny * nx; i++)
        if (data[i] > threshold) count++;
    fprintf(stderr, "Occupied voxels (logit > %.1f): %d / %d (%.1f%%)\n",
            threshold, count, nz*ny*nx, 100.0f * count / (nz*ny*nx));

    write_obj_marching_cubes(out_path, data, nz, ny, nx, threshold);
    free(data);
    return 0;
}


/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "TRELLIS.2 Stage 1 - Structure Generation\n\n");
        fprintf(stderr, "Usage: %s <mode> [args...]\n\n", argv[0]);
        fprintf(stderr, "Modes:\n");
        fprintf(stderr, "  full    <dinov3.st> <stage1.st> <decoder.st> <image> [-t N] [-o out.npy] [-s seed]\n");
        fprintf(stderr, "  stage1  <stage1.st> <features.npy> [-t N] [-o structure.npy] [-s seed]\n");
        fprintf(stderr, "  decode  <decoder.st> <structure.npy> [-t N] [-o occupancy.npy]\n");
        fprintf(stderr, "  encode  <dinov3.st> <image> [-t N] [-o features.npy]\n");
        fprintf(stderr, "  mesh    <occupancy.npy> [-o output.obj] [-t threshold]\n");
        return 1;
    }

    const char *mode = argv[1];
    if (strcmp(mode, "full") == 0)
        return mode_full(argc - 2, argv + 2);
    else if (strcmp(mode, "stage1") == 0)
        return mode_stage1(argc - 2, argv + 2);
    else if (strcmp(mode, "decode") == 0)
        return mode_decode(argc - 2, argv + 2);
    else if (strcmp(mode, "encode") == 0)
        return mode_encode(argc - 2, argv + 2);
    else if (strcmp(mode, "mesh") == 0)
        return mode_mesh(argc - 2, argv + 2);
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
}
