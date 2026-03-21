/*
 * test_hy3d.c - CPU test harness for Hunyuan3D-2.1
 *
 * Usage:
 *   ./test_hy3d <conditioner.safetensors> <model.safetensors> <vae.safetensors>
 *               [-i image.ppm] [-o output.obj]
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#define HUNYUAN3D_IMPLEMENTATION
#include "../../common/hunyuan3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Simple PPM reader (P6 binary) */
static uint8_t *read_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char hdr[4];
    if (!fgets(hdr, sizeof(hdr), f) || hdr[0] != 'P' || hdr[1] != '6') {
        fclose(f); return NULL;
    }
    int c;
    while ((c = fgetc(f)) == '#') while (fgetc(f) != '\n');
    ungetc(c, f);
    int maxval;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) { fclose(f); return NULL; }
    fgetc(f);
    size_t sz = (size_t)(*w) * (*h) * 3;
    uint8_t *data = (uint8_t *)malloc(sz);
    if (fread(data, 1, sz, f) != sz) { free(data); fclose(f); return NULL; }
    fclose(f);
    return data;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Hunyuan3D-2.1 CPU reference runner\n"
            "\n"
            "Usage: %s <conditioner.safetensors> <model.safetensors> <vae.safetensors>\n"
            "       [-i image.ppm] [-o output.obj]\n",
            argv[0]);
        return 1;
    }

    const char *cond_path = argv[1];
    const char *model_path = argv[2];
    const char *vae_path = argv[3];
    const char *img_path = NULL;
    const char *obj_path = "output.obj";

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc) img_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) obj_path = argv[++i];
    }

    fprintf(stderr, "Initializing CPU Hunyuan3D...\n");
    hy3d_context *ctx = hy3d_init();
    if (!ctx) { fprintf(stderr, "Failed to init\n"); return 1; }

    fprintf(stderr, "Loading weights...\n");
    if (hy3d_load_weights(ctx, cond_path, model_path, vae_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        hy3d_free(ctx);
        return 1;
    }

    if (img_path) {
        int w, h;
        uint8_t *rgb = read_ppm(img_path, &w, &h);
        if (!rgb) {
            fprintf(stderr, "Cannot read %s\n", img_path);
            hy3d_free(ctx);
            return 1;
        }

        fprintf(stderr, "Running inference on %s (%dx%d)...\n", img_path, w, h);
        hy3d_mesh_result mesh = hy3d_predict(ctx, rgb, w, h, 30, 7.5f, 256, 42);
        free(rgb);

        if (mesh.n_verts > 0) {
            mc_mesh mc_m;
            mc_m.vertices = mesh.vertices;
            mc_m.triangles = mesh.triangles;
            mc_m.n_verts = mesh.n_verts;
            mc_m.n_tris = mesh.n_tris;
            mc_write_obj(obj_path, &mc_m);
            fprintf(stderr, "Wrote %s (%d verts, %d tris)\n",
                    obj_path, mesh.n_verts, mesh.n_tris);
            hy3d_mesh_result_free(&mesh);
        }
    } else {
        fprintf(stderr, "No input image. Weight loading test passed.\n");
    }

    hy3d_free(ctx);
    fprintf(stderr, "Done.\n");
    return 0;
}
