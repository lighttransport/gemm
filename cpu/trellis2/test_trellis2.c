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
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
}
