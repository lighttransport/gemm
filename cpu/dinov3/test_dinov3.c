/*
 * test_dinov3.c - Test harness for DINOv3 ViT-L/16 vision encoder
 *
 * Usage:
 *   ./test_dinov3 model.safetensors [-t threads] [-i input.ppm] [-o output.npy]
 *
 * Loads a DINOv3 model from safetensors, encodes an image, and outputs
 * the feature tokens as a NumPy .npy file for verification against PyTorch.
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

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* Write a float32 array as NumPy .npy format (v1.0) */
static void write_npy_f32_nd(const char *path, const float *data, const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char shape_str[128];
    int slen = 0;
    slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, "(");
    size_t n_elem = 1;
    for (int i = 0; i < ndim; i++) {
        slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, "%d,", dims[i]);
        n_elem *= (size_t)dims[i];
    }
    slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, ")");
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
    fprintf(stderr, "Wrote %s (", path);
    for (int i = 0; i < ndim; i++) fprintf(stderr, "%s%d", i ? "x" : "", dims[i]);
    fprintf(stderr, ", float32)\n");
}

/* Generate a synthetic gradient image as uint8 RGB [h][w][3] */
static uint8_t *generate_gradient(int width, int height) {
    uint8_t *img = (uint8_t *)malloc((size_t)width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            img[idx + 0] = (uint8_t)(255 * x / (width - 1));   /* R */
            img[idx + 1] = (uint8_t)(255 * y / (height - 1));  /* G */
            img[idx + 2] = 128;                                  /* B */
        }
    }
    return img;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.safetensors [-t threads] [-i input.ppm] [-o output.npy]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *input_path = NULL;
    const char *output_path = "dinov3_output.npy";
    int n_threads = 4;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    fprintf(stderr, "Loading DINOv3 model from %s...\n", model_path);
    dinov3_model *m = dinov3_load_safetensors(model_path);
    if (!m) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Load or generate image */
    uint8_t *img;
    int img_w, img_h;
    if (input_path) {
        int channels;
        img = stbi_load(input_path, &img_w, &img_h, &channels, 3);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", input_path);
            dinov3_free(m);
            return 1;
        }
        fprintf(stderr, "Loaded image: %dx%d\n", img_w, img_h);
    } else {
        img_w = 512; img_h = 512;
        img = generate_gradient(img_w, img_h);
        fprintf(stderr, "Using synthetic 512x512 gradient image\n");
    }

    /* Encode */
    fprintf(stderr, "Encoding with %d threads...\n", n_threads);
    dinov3_result result = dinov3_encode(m, img, img_w, img_h, n_threads);

    if (result.features) {
        fprintf(stderr, "Output: %d tokens x %d dim\n", result.n_tokens, result.dim);

        /* Write features as .npy */
        int dims[2] = {result.n_tokens, result.dim};
        write_npy_f32_nd(output_path, result.features, dims, 2);

        /* Print statistics */
        float min_v = result.features[0], max_v = result.features[0];
        double sum = 0;
        int total = result.n_tokens * result.dim;
        for (int i = 0; i < total; i++) {
            float v = result.features[i];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum += v;
        }
        fprintf(stderr, "Feature stats: min=%.4f, max=%.4f, mean=%.6f\n",
                min_v, max_v, sum / total);

        /* Print first few values of CLS token */
        fprintf(stderr, "CLS token [0:8]: ");
        for (int i = 0; i < 8 && i < result.dim; i++)
            fprintf(stderr, "%.4f ", result.features[i]);
        fprintf(stderr, "\n");

        dinov3_result_free(&result);
    } else {
        fprintf(stderr, "Encoding failed\n");
    }

    free(img);
    dinov3_free(m);
    return 0;
}
