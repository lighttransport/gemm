/*
 * verify_dinov2.c - Verify HIP DINOv2 encoder against PyTorch reference
 *
 * Usage:
 *   ./verify_dinov2 <conditioner.safetensors> [--ref-dir ref/output/] [--out-dir hip_output/]
 */
#include "hip_hy3d_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ---- Minimal .npy reader/writer ---- */

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
    *ndims = 0;
    int total = 1;
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

static void write_npy_f32_2d(const char *path, const float *data, int r, int c) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0}; fwrite(version, 1, 2, f);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", r, c);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)r * c, f);
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
        fprintf(stderr, "    worst@%d: ref=%.6f hip=%.6f\n",
                worst, ref[worst], test[worst]);
}

static void print_stats(const char *name, const float *data, int n) {
    float mn = data[0], mx = data[0], sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    fprintf(stderr, "  %s: min=%.6f max=%.6f mean=%.6f\n", name, mn, mx, sum / n);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <conditioner.safetensors> [--ref-dir dir] [--out-dir dir]\n\n"
            "Verify DINOv2 HIP implementation against PyTorch reference.\n"
            "Generate reference first:\n"
            "  cd ref && uv run python dump_dinov2.py --ckpt <ckpt>\n",
            argv[0]);
        return 1;
    }

    const char *cond_path = argv[1];
    const char *ref_dir = "ref/output";
    const char *out_dir = "hip_output";

    int use_f32 = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--ref-dir") == 0 && i+1 < argc) ref_dir = argv[++i];
        else if (strcmp(argv[i], "--out-dir") == 0 && i+1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--f32") == 0) use_f32 = 1;
    }

    { char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir); (void)!system(cmd); }

    /* 1. Load reference input */
    char path[512];
    snprintf(path, sizeof(path), "%s/dinov2_input.npy", ref_dir);
    int dims[4], ndims;
    float *input = read_npy_f32(path, dims, &ndims);
    if (!input) {
        fprintf(stderr, "Cannot load %s\nGenerate reference outputs first.\n", path);
        return 1;
    }
    fprintf(stderr, "Loaded ref input: [%d, %d, %d]\n", dims[0], dims[1], dims[2]);

    /* 2. Load reference output */
    snprintf(path, sizeof(path), "%s/dinov2_output.npy", ref_dir);
    int odims[4], ondims;
    float *ref_output = read_npy_f32(path, odims, &ondims);
    if (!ref_output) {
        fprintf(stderr, "Cannot load reference output: %s\n", path);
        free(input); return 1;
    }
    fprintf(stderr, "Loaded ref output: [%d, %d]\n", odims[0], odims[1]);

    /* 3. Init HIP and load DINOv2 weights */
    fprintf(stderr, "\nInitializing HIP...\n");
    hip_hy3d_runner *r = hip_hy3d_init(0, 1);
    if (!r) { free(input); free(ref_output); return 1; }
    if (use_f32) hip_hy3d_set_f32_gemm(r, 1);

    if (hip_hy3d_load_weights(r, cond_path, NULL, NULL) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        hip_hy3d_free(r); free(input); free(ref_output);
        return 1;
    }

    /* 4. Run DINOv2 on GPU */
    fprintf(stderr, "\nRunning DINOv2 forward pass on GPU...\n");
    int out_n = 1370 * 1024;
    float *hip_output = (float *)malloc((size_t)out_n * sizeof(float));

    if (hip_hy3d_run_dinov2(r, input, hip_output) != 0) {
        fprintf(stderr, "DINOv2 forward pass failed\n");
        hip_hy3d_free(r); free(input); free(ref_output); free(hip_output);
        return 1;
    }

    /* 5. Print stats */
    fprintf(stderr, "\nResults:\n");
    print_stats("ref ", ref_output, out_n);
    print_stats("hip ", hip_output, out_n);

    /* 6. Compare */
    fprintf(stderr, "\nComparison (atol=0.15, rtol=0.05 for F16 weights, 24 layers):\n");
    compare_f32("dinov2_output", ref_output, hip_output, out_n, 0.15f, 0.05f);

    /* 7. Save HIP output */
    snprintf(path, sizeof(path), "%s/dinov2_output.npy", out_dir);
    write_npy_f32_2d(path, hip_output, 1370, 1024);
    fprintf(stderr, "\nSaved HIP output to %s\n", path);

    free(input);
    free(ref_output);
    free(hip_output);
    hip_hy3d_free(r);
    return 0;
}
