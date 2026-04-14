/*
 * test_cuda_hy3d.c - Test harness for CUDA Hunyuan3D-2.1 runner
 *
 * Usage:
 *   ./test_cuda_hy3d <conditioner.safetensors> <model.safetensors> <vae.safetensors>
 *                    [-i image.ppm] [-o output.obj] [--ply output.ply]
 *                    [-s steps] [-g guidance] [--grid res] [--seed N] [-d device]
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "cuda_hy3d_runner.h"
#include "../../common/marching_cubes.h"

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
            "Hunyuan3D-2.1 CUDA runner (NVRTC-based, no nvcc needed)\n"
            "\n"
            "Usage: %s <conditioner.safetensors> <model.safetensors> <vae.safetensors>\n"
            "       [-i image.ppm] [-o output.obj] [--ply output.ply]\n"
            "       [-s steps] [-g guidance] [--grid res] [--seed N] [-d device]\n"
            "\n"
            "Pipeline: Image -> DINOv2 -> DiT diffusion -> ShapeVAE -> Marching Cubes -> Mesh\n",
            argv[0]);
        return 1;
    }

    const char *cond_path = argv[1];
    const char *model_path = argv[2];
    const char *vae_path = argv[3];
    const char *img_path = NULL;
    const char *obj_path = "output.obj";
    const char *ply_path = NULL;
    int n_steps = 30;
    float guidance = 7.5f;
    int grid_res = 256;
    uint32_t seed = 42;
    int device_id = 0;
    int verbose = 1;
    int dump_every = 0;
    int dump_grid = 48;
    const char *dump_prefix = NULL;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc) img_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) obj_path = argv[++i];
        else if (strcmp(argv[i], "--ply") == 0 && i+1 < argc) ply_path = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-g") == 0 && i+1 < argc) guidance = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--grid") == 0 && i+1 < argc) grid_res = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 && i+1 < argc) verbose = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-every") == 0 && i+1 < argc) dump_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-grid") == 0 && i+1 < argc) dump_grid = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-prefix") == 0 && i+1 < argc) dump_prefix = argv[++i];
    }

    fprintf(stderr, "Initializing CUDA Hunyuan3D runner...\n");
    cuda_hy3d_runner *r = cuda_hy3d_init(device_id, verbose);
    if (!r) { fprintf(stderr, "Failed to init CUDA\n"); return 1; }

    if (dump_every > 0) {
        const char *pfx = dump_prefix ? dump_prefix : "hy3d_step";
        cuda_hy3d_set_dump(r, dump_every, dump_grid, pfx);
        fprintf(stderr, "Per-step dump: every=%d grid=%d prefix=%s_NNN.obj\n",
                dump_every, dump_grid, pfx);
    }

    fprintf(stderr, "Loading weights...\n");
    if (cuda_hy3d_load_weights(r, cond_path, model_path, vae_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        cuda_hy3d_free(r);
        return 1;
    }

    if (img_path) {
        int w, h;
        uint8_t *rgb = read_ppm(img_path, &w, &h);
        if (!rgb) {
            fprintf(stderr, "Cannot read %s\n", img_path);
            cuda_hy3d_free(r);
            return 1;
        }

        fprintf(stderr, "Running inference on %s (%dx%d)...\n", img_path, w, h);
        fprintf(stderr, "  steps=%d, guidance=%.1f, grid=%d, seed=%u\n",
                n_steps, guidance, grid_res, seed);

        hy3d_mesh mesh = cuda_hy3d_predict(r, rgb, w, h, n_steps, guidance, grid_res, seed);
        free(rgb);

        if (mesh.n_verts > 0) {
            /* Write OBJ */
            mc_mesh mc_m;
            mc_m.vertices = mesh.vertices;
            mc_m.triangles = mesh.triangles;
            mc_m.n_verts = mesh.n_verts;
            mc_m.n_tris = mesh.n_tris;

            if (mc_write_obj(obj_path, &mc_m) == 0) {
                fprintf(stderr, "Wrote OBJ to %s (%d verts, %d tris)\n",
                        obj_path, mesh.n_verts, mesh.n_tris);
            }

            /* Write PLY if requested */
            if (ply_path) {
                if (mc_write_ply(ply_path, &mc_m) == 0) {
                    fprintf(stderr, "Wrote PLY to %s\n", ply_path);
                }
            }

            cuda_hy3d_mesh_free(&mesh);
        } else {
            fprintf(stderr, "No mesh generated (empty result)\n");
        }
    } else {
        fprintf(stderr, "No input image specified (-i). Weight loading test passed.\n");
    }

    cuda_hy3d_free(r);
    fprintf(stderr, "Done.\n");
    return 0;
}
