/*
 * verify_vae.c - Verify CUDA ShapeVAE decoder against PyTorch reference
 *
 * Usage:
 *   ./verify_vae <vae.safetensors> [--ref-dir ref/output/] [--grid-res 8]
 */
#include "cuda_hy3d_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static float *read_npy_f32(const char *path, int dims[4], int *ndims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    char magic[6]; fread(magic, 1, 6, f);
    uint8_t ver[2]; fread(ver, 1, 2, f);
    uint16_t hdr_len; fread(&hdr_len, 2, 1, f);
    char *hdr = (char *)malloc(hdr_len + 1);
    fread(hdr, 1, hdr_len, f); hdr[hdr_len] = '\0';
    char *sp = strstr(hdr, "'shape': (");
    if (!sp) { free(hdr); fclose(f); return NULL; }
    sp += 10;
    *ndims = 0; int total = 1;
    while (*sp && *sp != ')') {
        if (*sp >= '0' && *sp <= '9') {
            int d = (int)strtol(sp, &sp, 10);
            dims[*ndims] = d; (*ndims)++; total *= d;
        } else sp++;
    }
    free(hdr);
    float *data = (float *)malloc((size_t)total * sizeof(float));
    size_t nr = fread(data, sizeof(float), (size_t)total, f);
    fclose(f);
    if ((int)nr != total) { free(data); return NULL; }
    return data;
}

static void write_npy_f32_3d(const char *path, const float *data, int a, int b, int c) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0}; fwrite(version, 1, 2, f);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }", a, b, c);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)a * b * c, f);
    fclose(f);
}

static void compare_f32(const char *name, const float *ref, const float *test,
                        int n, float atol, float rtol) {
    float max_abs = 0, mean_abs = 0;
    int worst = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(ref[i] - test[i]);
        mean_abs += d;
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    mean_abs /= (float)n;
    int ok = max_abs <= (atol + rtol * fmaxf(fabsf(ref[worst]), 1e-8f));
    fprintf(stderr, "  %s: %s  max=%.2e mean=%.2e (n=%d)\n",
            name, ok ? "OK" : "FAIL", max_abs, mean_abs, n);
    if (!ok)
        fprintf(stderr, "    worst@%d: ref=%.6f cuda=%.6f\n",
                worst, ref[worst], test[worst]);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <vae.safetensors> [--ref-dir dir] [--out-dir dir] [--grid-res N]\n",
            argv[0]);
        return 1;
    }

    const char *vae_path = argv[1];
    const char *ref_dir = "ref/output";
    const char *out_dir = "cuda_output";
    int grid_res = 8;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--ref-dir") == 0 && i+1 < argc) ref_dir = argv[++i];
        else if (strcmp(argv[i], "--out-dir") == 0 && i+1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--grid-res") == 0 && i+1 < argc) grid_res = atoi(argv[++i]);
    }

    { char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir); system(cmd); }

    /* Load reference latents */
    char path[512];
    snprintf(path, sizeof(path), "%s/vae_input_latents.npy", ref_dir);
    int dims[4], ndims;
    float *latents = read_npy_f32(path, dims, &ndims);
    if (!latents) {
        fprintf(stderr, "Cannot load %s\nGenerate reference first.\n", path);
        return 1;
    }
    fprintf(stderr, "Loaded ref latents: [%d, %d]\n", dims[0], dims[1]);

    /* Load reference SDF grid */
    snprintf(path, sizeof(path), "%s/vae_sdf_grid.npy", ref_dir);
    int sdims[4], sndims;
    float *ref_sdf = read_npy_f32(path, sdims, &sndims);
    int ref_grid_res = sdims[0];

    /* Init CUDA */
    fprintf(stderr, "\nInitializing CUDA...\n");
    cuda_hy3d_runner *r = cuda_hy3d_init(0, 1);
    if (!r) { free(latents); free(ref_sdf); return 1; }

    if (cuda_hy3d_load_weights(r, NULL, NULL, vae_path) != 0) {
        fprintf(stderr, "Failed to load VAE weights\n");
        cuda_hy3d_free(r); free(latents); free(ref_sdf); return 1;
    }

    /* Run VAE */
    fprintf(stderr, "\nRunning ShapeVAE on GPU (grid %d^3)...\n", grid_res);
    int total_pts = grid_res * grid_res * grid_res;
    float *cuda_sdf = (float *)calloc((size_t)total_pts, sizeof(float));

    if (cuda_hy3d_run_vae(r, latents, grid_res, cuda_sdf) != 0) {
        fprintf(stderr, "VAE forward pass failed\n");
        cuda_hy3d_free(r); free(latents); free(ref_sdf); free(cuda_sdf); return 1;
    }

    /* Stats */
    fprintf(stderr, "\nCUDA SDF stats:\n");
    float mn = cuda_sdf[0], mx = cuda_sdf[0], sm = 0;
    for (int i = 0; i < total_pts; i++) {
        if (cuda_sdf[i] < mn) mn = cuda_sdf[i];
        if (cuda_sdf[i] > mx) mx = cuda_sdf[i];
        sm += cuda_sdf[i];
    }
    fprintf(stderr, "  min=%.6f max=%.6f mean=%.6f\n", mn, mx, sm / total_pts);

    /* Compare if grid sizes match */
    if (ref_sdf && ref_grid_res == grid_res) {
        fprintf(stderr, "\nComparison:\n");
        compare_f32("vae_sdf_grid", ref_sdf, cuda_sdf, total_pts, 1e-2f, 1e-2f);
    } else if (ref_sdf) {
        fprintf(stderr, "\nSKIP comparison: ref grid=%d^3, cuda grid=%d^3 (mismatch)\n",
                ref_grid_res, grid_res);
    }

    /* Save */
    snprintf(path, sizeof(path), "%s/vae_sdf_grid.npy", out_dir);
    write_npy_f32_3d(path, cuda_sdf, grid_res, grid_res, grid_res);
    fprintf(stderr, "Saved to %s\n", path);

    free(latents); free(ref_sdf); free(cuda_sdf);
    cuda_hy3d_free(r);
    return 0;
}
