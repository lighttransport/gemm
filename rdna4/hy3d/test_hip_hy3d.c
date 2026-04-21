/*
 * test_hip_hy3d.c - Test harness for HIP Hunyuan3D-2.1 runner
 *
 * Usage:
 *   ./test_hip_hy3d <conditioner.safetensors> <model.safetensors> <vae.safetensors>
 *                    [-i image.ppm] [-o output.obj] [--ply output.ply]
 *                    [-s steps] [-g guidance] [--grid res] [--seed N] [-d device]
 *                    [--dump-latent-steps "1,15,30"] [--dump-latent-prefix prefix]
 *                    [--dump-velocity-steps "1,15,30"] [--dump-velocity-prefix prefix]
 *                    [--init-latents path.npy]
 *                    [--init-context path.npy]
 *                    [--init-trace-dir trace_dir]
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "hip_hy3d_runner.h"
#include "../../common/marching_cubes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void ensure_parent_dir_for_prefix(const char *prefix) {
    if (!prefix) return;
    const char *slash = strrchr(prefix, '/');
    if (!slash) return;
    size_t n = (size_t)(slash - prefix);
    if (n == 0 || n >= 900) return;
    char dir[1024];
    memcpy(dir, prefix, n);
    dir[n] = '\0';
    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", dir);
    (void)!system(cmd);
}

static int make_trace_file_path(char *dst, size_t dst_sz,
                                const char *trace_dir, const char *name) {
    if (!dst || dst_sz == 0 || !trace_dir || !*trace_dir || !name || !*name) return -1;
    int n = snprintf(dst, dst_sz, "%s/%s", trace_dir, name);
    if (n < 0 || (size_t)n >= dst_sz) return -1;
    return 0;
}

/* Minimal .npy reader for f32 latents.
 * Supports [4096,64] or [B,4096,64] and returns the first batch item. */
static float *read_npy_latents(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return NULL; }
    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t hdr_len;
    if (fread(&hdr_len, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc((size_t)hdr_len + 1);
    if (!hdr) { fclose(f); return NULL; }
    if (fread(hdr, 1, hdr_len, f) != hdr_len) { free(hdr); fclose(f); return NULL; }
    hdr[hdr_len] = '\0';

    int dims[4] = {0};
    int ndims = 0;
    char *sp = strstr(hdr, "'shape': (");
    if (!sp) { free(hdr); fclose(f); return NULL; }
    sp += 10;
    int total = 1;
    while (*sp && *sp != ')') {
        if (*sp >= '0' && *sp <= '9') {
            int d = (int)strtol(sp, &sp, 10);
            if (ndims < 4) dims[ndims] = d;
            ndims++;
            total *= d;
        } else sp++;
    }
    free(hdr);

    if (ndims < 2) { fclose(f); return NULL; }
    float *all = (float *)malloc((size_t)total * sizeof(float));
    if (!all) { fclose(f); return NULL; }
    if ((int)fread(all, sizeof(float), (size_t)total, f) != total) {
        free(all); fclose(f); return NULL;
    }
    fclose(f);

    const int latent_n = 4096 * 64;
    if (ndims == 2 && dims[0] == 4096 && dims[1] == 64) {
        *n_out = latent_n;
        return all;
    }
    if (ndims == 3 && dims[1] == 4096 && dims[2] == 64 && dims[0] >= 1) {
        float *out = (float *)malloc((size_t)latent_n * sizeof(float));
        if (!out) { free(all); return NULL; }
        memcpy(out, all, (size_t)latent_n * sizeof(float));
        free(all);
        *n_out = latent_n;
        return out;
    }
    free(all);
    return NULL;
}

/* Read context override from .npy.
 * Supports [1370,1024] or [2,1370,1024]. For 3D case:
 *   - batch 0 => cond, batch 1 => uncond */
static int read_npy_contexts(const char *path, float **cond_out, float **uncond_out, int *n_out) {
    *cond_out = NULL;
    *uncond_out = NULL;
    *n_out = 0;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return -1; }
    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return -1; }
    uint16_t hdr_len;
    if (fread(&hdr_len, 2, 1, f) != 1) { fclose(f); return -1; }
    char *hdr = (char *)malloc((size_t)hdr_len + 1);
    if (!hdr) { fclose(f); return -1; }
    if (fread(hdr, 1, hdr_len, f) != hdr_len) { free(hdr); fclose(f); return -1; }
    hdr[hdr_len] = '\0';

    int dims[4] = {0};
    int ndims = 0;
    char *sp = strstr(hdr, "'shape': (");
    if (!sp) { free(hdr); fclose(f); return -1; }
    sp += 10;
    int total = 1;
    while (*sp && *sp != ')') {
        if (*sp >= '0' && *sp <= '9') {
            int d = (int)strtol(sp, &sp, 10);
            if (ndims < 4) dims[ndims] = d;
            ndims++;
            total *= d;
        } else sp++;
    }
    free(hdr);

    float *all = (float *)malloc((size_t)total * sizeof(float));
    if (!all) { fclose(f); return -1; }
    if ((int)fread(all, sizeof(float), (size_t)total, f) != total) {
        free(all); fclose(f); return -1;
    }
    fclose(f);

    const int ctx_n = 1370 * 1024;
    if (ndims == 2 && dims[0] == 1370 && dims[1] == 1024) {
        *cond_out = all;
        *uncond_out = NULL;
        *n_out = ctx_n;
        return 0;
    }
    if (ndims == 3 && dims[1] == 1370 && dims[2] == 1024 && dims[0] >= 1) {
        float *cond = (float *)malloc((size_t)ctx_n * sizeof(float));
        if (!cond) { free(all); return -1; }
        memcpy(cond, all, (size_t)ctx_n * sizeof(float));
        float *uncond = NULL;
        if (dims[0] >= 2) {
            uncond = (float *)malloc((size_t)ctx_n * sizeof(float));
            if (!uncond) { free(cond); free(all); return -1; }
            memcpy(uncond, all + ctx_n, (size_t)ctx_n * sizeof(float));
        }
        free(all);
        *cond_out = cond;
        *uncond_out = uncond;
        *n_out = ctx_n;
        return 0;
    }

    free(all);
    return -1;
}

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
            "Hunyuan3D-2.1 HIP runner (RDNA4)\n"
            "\n"
            "Usage: %s <conditioner.safetensors> <model.safetensors> <vae.safetensors>\n"
            "       [-i image.ppm] [-o output.obj] [--ply output.ply]\n"
            "       [-s steps] [-g guidance] [--grid res] [--seed N] [-d device]\n"
            "       [--dump-latent-steps \"1,15,30\"] [--dump-latent-prefix prefix]\n"
            "       [--dump-velocity-steps \"1,15,30\"] [--dump-velocity-prefix prefix]\n"
            "       [--init-latents path.npy]\n"
            "       [--init-context path.npy]\n"
            "       [--init-trace-dir trace_dir]\n"
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
    const char *latent_dump_steps = NULL;
    const char *latent_dump_prefix = "dit_latent_step";
    const char *velocity_dump_steps = NULL;
    const char *velocity_dump_prefix = "dit_velocity_step";
    const char *init_latents_path = NULL;
    const char *init_context_path = NULL;
    const char *init_trace_dir = NULL;

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
        else if (strcmp(argv[i], "--dump-latent-steps") == 0 && i+1 < argc) latent_dump_steps = argv[++i];
        else if (strcmp(argv[i], "--dump-latent-prefix") == 0 && i+1 < argc) latent_dump_prefix = argv[++i];
        else if (strcmp(argv[i], "--dump-velocity-steps") == 0 && i+1 < argc) velocity_dump_steps = argv[++i];
        else if (strcmp(argv[i], "--dump-velocity-prefix") == 0 && i+1 < argc) velocity_dump_prefix = argv[++i];
        else if (strcmp(argv[i], "--init-latents") == 0 && i+1 < argc) init_latents_path = argv[++i];
        else if (strcmp(argv[i], "--init-context") == 0 && i+1 < argc) init_context_path = argv[++i];
        else if (strcmp(argv[i], "--init-trace-dir") == 0 && i+1 < argc) init_trace_dir = argv[++i];
    }

    char trace_latents_path[1024];
    char trace_context_path[1024];
    if (init_trace_dir && *init_trace_dir) {
        if (!init_latents_path) {
            if (make_trace_file_path(trace_latents_path, sizeof(trace_latents_path),
                                     init_trace_dir, "04_dit_latents_step0.npy") != 0) {
                fprintf(stderr, "Invalid --init-trace-dir path: %s\n", init_trace_dir);
                return 1;
            }
            init_latents_path = trace_latents_path;
        }
        if (!init_context_path) {
            if (make_trace_file_path(trace_context_path, sizeof(trace_context_path),
                                     init_trace_dir, "03_dit_context_cfg.npy") != 0) {
                fprintf(stderr, "Invalid --init-trace-dir path: %s\n", init_trace_dir);
                return 1;
            }
            init_context_path = trace_context_path;
        }
    }

    fprintf(stderr, "Initializing HIP Hunyuan3D runner...\n");
    hip_hy3d_runner *r = hip_hy3d_init(device_id, verbose);
    if (!r) { fprintf(stderr, "Failed to init HIP\n"); return 1; }

    if (latent_dump_steps && *latent_dump_steps) {
        ensure_parent_dir_for_prefix(latent_dump_prefix);
        hip_hy3d_set_latent_dump(r, latent_dump_steps, latent_dump_prefix);
        fprintf(stderr, "Latent dump: steps=%s prefix=%s_NNN.npy\n",
                latent_dump_steps, latent_dump_prefix);
    }
    if (velocity_dump_steps && *velocity_dump_steps) {
        ensure_parent_dir_for_prefix(velocity_dump_prefix);
        hip_hy3d_set_velocity_dump(r, velocity_dump_steps, velocity_dump_prefix);
        fprintf(stderr, "Velocity dump: steps=%s prefix=%s_NNN.npy\n",
                velocity_dump_steps, velocity_dump_prefix);
    }
    if (init_latents_path) {
        int n_lat = 0;
        float *lat = read_npy_latents(init_latents_path, &n_lat);
        if (!lat) {
            fprintf(stderr, "Cannot read init latents from %s\n", init_latents_path);
            hip_hy3d_free(r);
            return 1;
        }
        if (hip_hy3d_set_init_latents(r, lat, n_lat) != 0) {
            fprintf(stderr, "Invalid init latents in %s\n", init_latents_path);
            free(lat);
            hip_hy3d_free(r);
            return 1;
        }
        fprintf(stderr, "Using init latents from %s\n", init_latents_path);
        free(lat);
    }
    if (init_context_path) {
        float *cond = NULL, *uncond = NULL;
        int n_ctx = 0;
        if (read_npy_contexts(init_context_path, &cond, &uncond, &n_ctx) != 0) {
            fprintf(stderr, "Cannot read init context from %s\n", init_context_path);
            hip_hy3d_free(r);
            return 1;
        }
        if (hip_hy3d_set_init_contexts(r, cond, uncond, n_ctx) != 0) {
            fprintf(stderr, "Invalid init context in %s\n", init_context_path);
            free(cond);
            free(uncond);
            hip_hy3d_free(r);
            return 1;
        }
        fprintf(stderr, "Using init context from %s\n", init_context_path);
        free(cond);
        free(uncond);
    }

    fprintf(stderr, "Loading weights...\n");
    if (hip_hy3d_load_weights(r, cond_path, model_path, vae_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        hip_hy3d_free(r);
        return 1;
    }

    if (img_path) {
        int w, h;
        uint8_t *rgb = read_ppm(img_path, &w, &h);
        if (!rgb) {
            fprintf(stderr, "Cannot read %s\n", img_path);
            hip_hy3d_free(r);
            return 1;
        }

        fprintf(stderr, "Running inference on %s (%dx%d)...\n", img_path, w, h);
        fprintf(stderr, "  steps=%d, guidance=%.1f, grid=%d, seed=%u\n",
                n_steps, guidance, grid_res, seed);

        hy3d_mesh mesh = hip_hy3d_predict(r, rgb, w, h, n_steps, guidance, grid_res, seed);
        free(rgb);

        if (mesh.n_verts > 0) {
            mc_mesh mc_m;
            mc_m.vertices = mesh.vertices;
            mc_m.triangles = mesh.triangles;
            mc_m.n_verts = mesh.n_verts;
            mc_m.n_tris = mesh.n_tris;

            if (mc_write_obj(obj_path, &mc_m) == 0) {
                fprintf(stderr, "Wrote OBJ to %s (%d verts, %d tris)\n",
                        obj_path, mesh.n_verts, mesh.n_tris);
            }

            if (ply_path) {
                if (mc_write_ply(ply_path, &mc_m) == 0) {
                    fprintf(stderr, "Wrote PLY to %s\n", ply_path);
                }
            }

            hip_hy3d_mesh_free(&mesh);
        } else {
            fprintf(stderr, "No mesh generated (empty result)\n");
        }
    } else {
        fprintf(stderr, "No input image specified (-i). Weight loading test passed.\n");
    }

    hip_hy3d_free(r);
    fprintf(stderr, "Done.\n");
    return 0;
}
