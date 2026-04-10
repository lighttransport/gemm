/*
 * test_vulkan_trellis2.cc - Test Vulkan TRELLIS.2 Stages 1-3
 *
 * Usage:
 *   ./test_vulkan_trellis2 <stage1.st> <decoder.st> <features.npy> [options]
 *
 * Runs Stage 1 DiT + decoder on Vulkan GPU, exports mesh via marching cubes.
 * Stage 2/3 require additional --stage2 / --stage3 flags with weight paths.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "vulkan_trellis2_runner.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <vector>

/* ---- .npy reader ---- */
static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return nullptr; }
    fseek(f, 8, SEEK_SET);
    uint16_t header_len;
    if (fread(&header_len, 2, 1, f) != 1) { fclose(f); return nullptr; }
    char *header = (char *)malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != (size_t)header_len) { free(header); fclose(f); return nullptr; }
    header[header_len] = '\0';
    *ndim = 0;
    char *sp = strstr(header, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') { while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break; dims[*ndim] = (int)strtol(sp, &sp, 10); (*ndim)++; if (*ndim >= 8) break; }}}
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    float *data = (float *)malloc(n * sizeof(float));
    size_t nr = fread(data, sizeof(float), n, f);
    (void)nr;
    fclose(f); free(header);
    fprintf(stderr, "Read %s: (", path);
    for (int i = 0; i < *ndim; i++) fprintf(stderr, "%s%d", i ? "," : "", dims[i]);
    fprintf(stderr, ") = %zu elements\n", n);
    return data;
}

/* ---- .npy writer ---- */
static void write_npy_f32(const char *path, const float *data, const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char shape[256]; int sl = 0;
    sl += snprintf(shape + sl, sizeof(shape) - sl, "(");
    size_t n = 1;
    for (int i = 0; i < ndim; i++) { sl += snprintf(shape + sl, sizeof(shape) - sl, "%d,", dims[i]); n *= dims[i]; }
    sl += snprintf(shape + sl, sizeof(shape) - sl, ")");
    char hdr[512];
    int hl = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape);
    int total = 10 + hl + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t hlen = (uint16_t)(hl + pad + 1);
    fwrite(&hlen, 2, 1, f); fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f); fputc('\n', f);
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "Wrote %s\n", path);
}

/* Simple xoshiro256** */
static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
struct rng_state { uint64_t s[4]; };
static uint64_t rng_next(rng_state *r) {
    uint64_t *s = r->s; uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl64(s[3], 45);
    return result;
}
static float rng_randn(rng_state *r) {
    double u1 = ((double)(rng_next(r) >> 11) + 0.5) / (double)(1ULL << 53);
    double u2 = ((double)(rng_next(r) >> 11) + 0.5) / (double)(1ULL << 53);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}
static float rescale_t(float t, float rt) { return t * rt / (1.0f + (rt - 1.0f) * t); }

/* ---- OBJ writer ---- */
static void write_obj(const char *path, const float *verts, int nv,
                       const int *tris, int nt) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < nv; i++)
        fprintf(f, "v %f %f %f\n", verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
    for (int i = 0; i < nt; i++)
        fprintf(f, "f %d %d %d\n", tris[i * 3] + 1, tris[i * 3 + 1] + 1, tris[i * 3 + 2] + 1);
    fclose(f);
    fprintf(stderr, "Wrote %s (%d verts, %d tris)\n", path, nv, nt);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <stage1.st> <decoder.st> <features.npy> [options]\n\n"
            "Options:\n"
            "  -s <seed>        Random seed (default: 42)\n"
            "  -n <steps>       Euler steps (default: 12)\n"
            "  -g <cfg_scale>   CFG guidance scale (default: 7.5)\n"
            "  -r <rescale>     CFG rescale factor (default: 0.7)\n"
            "  -o <output.obj>  Output mesh path (default: output.obj)\n"
            "  --noise <path>   Load initial noise from .npy\n"
            "  --npy <path>     Save latent/occupancy as .npy\n"
            "  --dit-only       Only run DiT (no decoder)\n"
            "  --decode-only <latent.npy>  Only run decoder\n"
            "  --encode <dinov3.st> <image.npy>  Run DINOv3 encoder only\n"
            "  --stage2 <path>  Stage 2 shape flow DiT weights\n"
            "  --stage3 <path>  Stage 3 texture flow DiT weights\n"
            "  -v               Verbose\n",
            argv[0]);
        return 1;
    }

    const char *stage1_path = argv[1];
    const char *decoder_path = argv[2];
    const char *features_path = argv[3];
    const char *output_path = "output.obj";
    const char *noise_path = nullptr;
    const char *npy_path = nullptr;
    const char *decode_only_path = nullptr;
    const char *encode_dinov3_path = nullptr;
    const char *encode_image_path = nullptr;
    const char *stage2_path = nullptr;
    const char *stage3_path = nullptr;
    const char *shape_dec_path = nullptr;
    const char *tex_dec_path = nullptr;
    uint32_t seed = 42;
    int n_steps = 12;
    float cfg_scale = 7.5f;
    float cfg_rescale = 0.7f;
    int verbose = 0;
    bool dit_only = false;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) cfg_rescale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--noise") == 0 && i + 1 < argc) noise_path = argv[++i];
        else if (strcmp(argv[i], "--npy") == 0 && i + 1 < argc) npy_path = argv[++i];
        else if (strcmp(argv[i], "--dit-only") == 0) dit_only = true;
        else if (strcmp(argv[i], "--decode-only") == 0 && i + 1 < argc) decode_only_path = argv[++i];
        else if (strcmp(argv[i], "--encode") == 0 && i + 2 < argc) { encode_dinov3_path = argv[++i]; encode_image_path = argv[++i]; }
        else if (strcmp(argv[i], "--stage2") == 0 && i + 1 < argc) stage2_path = argv[++i];
        else if (strcmp(argv[i], "--stage3") == 0 && i + 1 < argc) stage3_path = argv[++i];
        else if (strcmp(argv[i], "--shape-dec") == 0 && i + 1 < argc) shape_dec_path = argv[++i];
        else if (strcmp(argv[i], "--tex-dec") == 0 && i + 1 < argc) tex_dec_path = argv[++i];
        else if (strcmp(argv[i], "-v") == 0) verbose = 1;
    }

    /* Initialize */
    vulkan_trellis2_runner *r = vulkan_trellis2_init(0, verbose);
    if (!r) { fprintf(stderr, "Failed to init Vulkan runner\n"); return 1; }

    /* Encode-only mode */
    if (encode_dinov3_path) {
        if (vulkan_trellis2_load_weights(r, encode_dinov3_path, nullptr, nullptr) != 0) {
            fprintf(stderr, "Failed to load DINOv3 weights\n");
            vulkan_trellis2_free(r);
            return 1;
        }
        int ndim, dims[8];
        float *image = read_npy_f32(encode_image_path, &ndim, dims);
        if (!image) { vulkan_trellis2_free(r); return 1; }

        std::vector<float> features(1029 * 1024);
        vulkan_trellis2_run_dinov3(r, image, features.data());

        fprintf(stderr, "DINOv3 features: [%.6f, %.6f, %.6f, %.6f]\n",
                features[0], features[1], features[2], features[3]);

        if (npy_path) {
            int od[] = {1029, 1024};
            write_npy_f32(npy_path, features.data(), od, 2);
        }
        free(image);
        vulkan_trellis2_free(r);
        return 0;
    }

    /* Load weights */
    if (vulkan_trellis2_load_weights(r, nullptr, stage1_path, decoder_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        vulkan_trellis2_free(r);
        return 1;
    }
    if (stage2_path) {
        if (vulkan_trellis2_load_stage2(r, stage2_path) != 0) {
            fprintf(stderr, "Failed to load Stage 2 weights\n");
            vulkan_trellis2_free(r); return 1;
        }
    }
    if (stage3_path) {
        if (vulkan_trellis2_load_stage3(r, stage3_path) != 0) {
            fprintf(stderr, "Failed to load Stage 3 weights\n");
            vulkan_trellis2_free(r); return 1;
        }
    }

    /* Decode-only mode */
    if (decode_only_path) {
        int ndim, dims[8];
        float *latent = read_npy_f32(decode_only_path, &ndim, dims);
        if (!latent) return 1;

        std::vector<float> occ(64 * 64 * 64);
        vulkan_trellis2_run_decoder(r, latent, occ.data());

        /* Count occupied */
        int n_occ = 0;
        for (int i = 0; i < 64 * 64 * 64; i++)
            if (occ[i] > 0.0f) n_occ++;
        fprintf(stderr, "Occupied: %d / %d (%.1f%%)\n", n_occ, 64*64*64,
                100.0f * n_occ / (64*64*64));

        if (npy_path) {
            int od[] = {1, 64, 64, 64};
            write_npy_f32(npy_path, occ.data(), od, 4);
        }

        free(latent);
        vulkan_trellis2_free(r);
        return 0;
    }

    /* Load features */
    int ndim, dims[8];
    float *features = read_npy_f32(features_path, &ndim, dims);
    if (!features) { vulkan_trellis2_free(r); return 1; }

    /* Load or generate noise */
    float *noise = nullptr;
    if (noise_path) {
        int nn, nd[8];
        noise = read_npy_f32(noise_path, &nn, nd);
    }

    if (dit_only) {
        /* Single-step DiT test */
        float t_start = 1.0f;
        float t_cur = rescale_t(t_start, 5.0f);

        /* Generate noise if not loaded */
        std::vector<float> x(8 * 4096);
        if (noise) {
            memcpy(x.data(), noise, x.size() * sizeof(float));
        } else {
            rng_state rng;
            rng.s[0] = seed; rng.s[1] = seed ^ 0x1234567890abcdefULL;
            rng.s[2] = seed ^ 0xfedcba0987654321ULL; rng.s[3] = seed ^ 0xdeadbeefcafebabeULL;
            for (int i = 0; i < 100; i++) rng_next(&rng);
            for (size_t i = 0; i < x.size(); i++) x[i] = rng_randn(&rng);
        }

        std::vector<float> v_out(8 * 4096);
        vulkan_trellis2_run_dit(r, x.data(), t_cur, features, v_out.data());

        if (npy_path) {
            int od[] = {8, 16, 16, 16};
            write_npy_f32(npy_path, v_out.data(), od, 4);
        }

        fprintf(stderr, "DiT output: v[:4]=%.6f %.6f %.6f %.6f\n",
                v_out[0], v_out[1], v_out[2], v_out[3]);
    } else {
        /* Full pipeline */
        std::vector<float> occ(64 * 64 * 64);

        vulkan_trellis2_run_stage1(r, features, noise, occ.data(),
                                    n_steps, cfg_scale, cfg_rescale, seed);

        /* Count occupied */
        int n_occ = 0;
        for (int i = 0; i < 64 * 64 * 64; i++)
            if (occ[i] > 0.0f) n_occ++;
        fprintf(stderr, "Occupied: %d / %d (%.1f%%)\n", n_occ, 64*64*64,
                100.0f * n_occ / (64*64*64));

        if (npy_path) {
            int od[] = {1, 64, 64, 64};
            write_npy_f32(npy_path, occ.data(), od, 4);
        }

        /* Marching cubes */
        mc_mesh mesh = mc_marching_cubes(occ.data(), 64, 64, 64, 0.0f, nullptr);
        fprintf(stderr, "Marching cubes: %d verts, %d tris\n", mesh.n_verts, mesh.n_tris);

        if (mesh.n_tris > 0) {
            mc_write_obj(output_path, &mesh);
        }
        mc_mesh_free(&mesh);

        /* === Stage 2: Shape flow sampling === */
        if (stage2_path) {
            fprintf(stderr, "\n=== Stage 2: Shape Flow Sampling (%d steps) ===\n", n_steps);

            /* Allocate max possible (all 64^3 occupied, unlikely) */
            std::vector<float> shape_latent(n_occ * 32);
            std::vector<int32_t> sparse_coords(n_occ * 4);
            int N_sparse = 0;

            vulkan_trellis2_run_stage2(r, features, occ.data(),
                                        shape_latent.data(), sparse_coords.data(), &N_sparse,
                                        n_steps, cfg_scale, cfg_rescale, seed);

            fprintf(stderr, "Stage 2 output: N=%d, shape_latent[:4]=%.4f %.4f %.4f %.4f\n",
                    N_sparse, shape_latent[0], shape_latent[1], shape_latent[2], shape_latent[3]);

            if (npy_path) {
                char s2path[512];
                snprintf(s2path, sizeof(s2path), "%s_stage2.npy", npy_path);
                int od[] = {N_sparse, 32};
                write_npy_f32(s2path, shape_latent.data(), od, 2);
            }

            /* === Shape Decoder === */
            if (shape_dec_path) {
                fprintf(stderr, "\n=== Shape Decoder (CPU) ===\n");

                /* Max output: ~16× input (4 C2S stages, each ~2× expansion) */
                int max_out = N_sparse * 16;
                std::vector<float> dec_feats(max_out * 7);
                std::vector<int32_t> dec_coords(max_out * 4);
                int dec_N = 0;

                vulkan_trellis2_run_shape_decoder(r, shape_dec_path,
                    shape_latent.data(), sparse_coords.data(), N_sparse,
                    dec_feats.data(), dec_coords.data(), &dec_N);

                fprintf(stderr, "Shape decoder output: %d voxels, 7 channels\n", dec_N);
                if (dec_N > 0) {
                    fprintf(stderr, "  feats[:4]=%.4f %.4f %.4f %.4f\n",
                            dec_feats[0], dec_feats[1], dec_feats[2], dec_feats[3]);
                }

                if (npy_path) {
                    char dpath[512];
                    snprintf(dpath, sizeof(dpath), "%s_shape_dec.npy", npy_path);
                    int od[] = {dec_N, 7};
                    write_npy_f32(dpath, dec_feats.data(), od, 2);
                }
            }

            /* === Stage 3: Texture flow sampling === */
            if (stage3_path) {
                fprintf(stderr, "\n=== Stage 3: Texture Flow Sampling (%d steps, no CFG) ===\n", n_steps);

                std::vector<float> tex_latent(N_sparse * 32);
                vulkan_trellis2_run_stage3(r, features, shape_latent.data(),
                                            sparse_coords.data(), N_sparse,
                                            tex_latent.data(), n_steps, seed);

                fprintf(stderr, "Stage 3 output: tex_latent[:4]=%.4f %.4f %.4f %.4f\n",
                        tex_latent[0], tex_latent[1], tex_latent[2], tex_latent[3]);

                if (npy_path) {
                    char s3path[512];
                    snprintf(s3path, sizeof(s3path), "%s_stage3.npy", npy_path);
                    int od[] = {N_sparse, 32};
                    write_npy_f32(s3path, tex_latent.data(), od, 2);
                }

                /* === Texture Decoder === */
                if (tex_dec_path) {
                    fprintf(stderr, "\n=== Texture Decoder (GPU ConvNeXt + CPU C2S) ===\n");

                    int max_out = N_sparse * 16;
                    std::vector<float> tex_feats(max_out * 6);
                    std::vector<int32_t> tex_coords(max_out * 4);
                    int tex_N = 0;

                    vulkan_trellis2_run_shape_decoder(r, tex_dec_path,
                        tex_latent.data(), sparse_coords.data(), N_sparse,
                        tex_feats.data(), tex_coords.data(), &tex_N);

                    fprintf(stderr, "Texture decoder output: %d voxels, 6 channels\n", tex_N);

                    if (npy_path) {
                        char tpath[512];
                        snprintf(tpath, sizeof(tpath), "%s_tex_dec.npy", npy_path);
                        int od[] = {tex_N, 6};
                        write_npy_f32(tpath, tex_feats.data(), od, 2);
                    }
                }
            }
        }
    }

    free(features);
    if (noise) free(noise);
    vulkan_trellis2_free(r);

    return 0;
}
