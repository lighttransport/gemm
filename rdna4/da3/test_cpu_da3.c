/*
 * test_cpu_da3.c - Run DA3 depth estimation on CPU only, save as .npy
 * Usage: ./test_cpu_da3 <model.safetensors> -i image.jpg -o depth.npy
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"

#define DEPTH_ANYTHING3_IMPLEMENTATION
#include "../../common/depth_anything3.h"

/* Write float32 array as NumPy .npy (2D) */
static void write_npy_f32(const char *path, const float *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", h, w);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)w * h, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%d, float32)\n", path, w, h);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.safetensors> -i image.jpg [-o depth.npy] [-t threads]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *input_path = NULL;
    const char *output_path = "/tmp/da3_cpu_depth.npy";
    int n_threads = 4;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) n_threads = atoi(argv[++i]);
    }

    /* Load model */
    fprintf(stderr, "Loading model: %s\n", model_path);
    /* Derive config.json path from model path */
    char config_path[512];
    const char *slash = strrchr(model_path, '/');
    if (slash) {
        int dir_len = (int)(slash - model_path);
        snprintf(config_path, sizeof(config_path), "%.*s/config.json", dir_len, model_path);
    } else {
        snprintf(config_path, sizeof(config_path), "config.json");
    }
    da3_model *m = da3_load_safetensors(model_path, config_path);
    if (!m) { fprintf(stderr, "Failed to load model\n"); return 1; }

    fprintf(stderr, "  dim=%d, heads=%d, blocks=%d, features=%d\n",
            m->dim, m->n_heads, m->n_blocks, m->head_features);
    fprintf(stderr, "  feature_layers=[%d,%d,%d,%d]\n",
            m->feature_layers[0], m->feature_layers[1],
            m->feature_layers[2], m->feature_layers[3]);

    /* Load image */
    if (!input_path) { fprintf(stderr, "No input image (-i)\n"); da3_free(m); return 1; }
    int img_w, img_h, img_c;
    uint8_t *img = stbi_load(input_path, &img_w, &img_h, &img_c, 3);
    if (!img) { fprintf(stderr, "Failed to load %s\n", input_path); da3_free(m); return 1; }
    fprintf(stderr, "  Image: %dx%d\n", img_w, img_h);

    /* Run inference */
    fprintf(stderr, "Running CPU inference (%d threads)...\n", n_threads);
    da3_result result = da3_predict(m, img, img_w, img_h, n_threads);
    stbi_image_free(img);

    if (!result.depth) { fprintf(stderr, "No depth output\n"); da3_free(m); return 1; }

    int npix = result.width * result.height;
    float mn = result.depth[0], mx = result.depth[0], sum = 0;
    for (int i = 0; i < npix; i++) {
        if (result.depth[i] < mn) mn = result.depth[i];
        if (result.depth[i] > mx) mx = result.depth[i];
        sum += result.depth[i];
    }
    fprintf(stderr, "  Output: %dx%d\n", result.width, result.height);
    fprintf(stderr, "  depth: min=%.4f max=%.4f mean=%.6f\n", mn, mx, sum/npix);

    if (result.confidence) {
        float cmn = result.confidence[0], cmx = result.confidence[0];
        for (int i = 0; i < npix; i++) {
            if (result.confidence[i] < cmn) cmn = result.confidence[i];
            if (result.confidence[i] > cmx) cmx = result.confidence[i];
        }
        fprintf(stderr, "  conf:  min=%.4f max=%.4f\n", cmn, cmx);
    }

    write_npy_f32(output_path, result.depth, result.width, result.height);

    da3_result_free(&result);
    da3_free(m);
    fprintf(stderr, "Done.\n");
    return 0;
}
