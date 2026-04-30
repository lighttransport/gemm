/* test_cuda_sam3d — CLI for the CUDA SAM 3D Objects runner.
 *
 * Phase 7a: end-to-end image → splat.ply. Mirrors the CPU runner CLI in
 * cpu/sam3d/test_sam3d.c so a single fujisan.jpg + pointmap.npy
 * invocation drives the full six-stage CUDA pipeline.
 *
 * Usage:
 *   test_cuda_sam3d [--safetensors-dir DIR] [--pipeline-yaml YAML]
 *                   <image.png> <mask.png>
 *                   [--pointmap pmap.npy] [--slat-ref <dir>]
 *                   [--seed N] [--steps N] [--slat-steps N]
 *                   [--cfg F] [-o splat.ply]
 *                   [--mesh-out mesh.obj|mesh.ply|mesh.glb]
 *                   [--mesh-source occupancy|slat]
 *                   [--mesh-decoder mesh.safetensors]
 *                   [--mesh-iso F] [--mesh-only] [--mesh-threads N]
 *                   [--mesh-texture-size N]
 *                   [--mesh-texture-mode xatlas|grid]
 *                   [--mesh-texture-color auto|decoder|image]
 *                   [--moge] [--moge-out pmap.npy]
 *                   [--moge-python PY] [--moge-script SCRIPT]
 *                   [--moge-model MODEL] [--moge-device cpu|cuda]
 *                   [--device N] [--precision fp16|bf16|fp32] [-v|-vv]
 *
 * --slat-ref bypasses upstream stages by loading
 * slat_dit_out_{coords,feats}.npy (or slat_gs_in_*) and runs only
 * SLAT-GS-decode + PLY write.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define GS_PLY_WRITER_IMPLEMENTATION
#include "../../common/gs_ply_writer.h"

#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINY_GLTF_IMPLEMENTATION
#include "../../common/tiny_gltf.h"

#define SAM3D_MESH_DECODER_IMPLEMENTATION
#include "../../common/sam3d_mesh_decoder.h"

#include "../../common/sam3d_xatlas.h"
#include "../../common/npy_io.h"
#include "cuda_sam3d_runner.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    MESH_TEXTURE_COLOR_AUTO = 0,
    MESH_TEXTURE_COLOR_DECODER = 1,
    MESH_TEXTURE_COLOR_IMAGE = 2
};

typedef struct {
    float p[3];
    float rgb[3];
    int next;
} image_texel_point;

static int has_suffix(const char *s, const char *suffix)
{
    if (!s || !suffix) return 0;
    size_t ns = strlen(s), nx = strlen(suffix);
    return ns >= nx && strcmp(s + ns - nx, suffix) == 0;
}

static int write_mesh_file_rgb(const mc_mesh *mesh, const float *rgb,
                               const char *path, int texture_size,
                               int texture_use_xatlas)
{
    if (has_suffix(path, ".glb")) {
        if (rgb && texture_size > 0) {
            if (texture_use_xatlas) {
                sam3d_xatlas_mesh xm = {0};
                if (sam3d_xatlas_generate(mesh->vertices, mesh->n_verts,
                                          mesh->triangles, mesh->n_tris,
                                          texture_size, &xm) == 0) {
                    float *xrgb = (float *)malloc((size_t)xm.n_vertices *
                                                  3 * sizeof(float));
                    if (xrgb) {
                        for (int i = 0; i < xm.n_vertices; i++) {
                            uint32_t src = xm.xrefs[i];
                            xrgb[i * 3 + 0] = rgb[(size_t)src * 3 + 0];
                            xrgb[i * 3 + 1] = rgb[(size_t)src * 3 + 1];
                            xrgb[i * 3 + 2] = rgb[(size_t)src * 3 + 2];
                        }
                        int rc = tinygltf_write_glb_mesh_rgb_texture_indexed(
                            path, xm.positions, xrgb, xm.uvs, xm.n_vertices,
                            xm.indices, xm.n_indices, texture_size);
                        free(xrgb);
                        if (rc == 0) {
                            fprintf(stderr,
                                    "[test_cuda_sam3d] xatlas texture atlas: "
                                    "%d verts %d tris %d charts %dx%d\n",
                                    xm.n_vertices, xm.n_indices / 3,
                                    xm.chart_count, xm.atlas_width,
                                    xm.atlas_height);
                            sam3d_xatlas_free(&xm);
                            return 0;
                        }
                    }
                    sam3d_xatlas_free(&xm);
                }
                fprintf(stderr,
                        "[test_cuda_sam3d] xatlas texture bake failed; "
                        "falling back to triangle-grid atlas\n");
            }
            return tinygltf_write_glb_mesh_rgb_texture(path, mesh->vertices,
                                                       rgb, mesh->n_verts,
                                                       mesh->triangles,
                                                       mesh->n_tris,
                                                       texture_size);
        }
        return rgb ? tinygltf_write_glb_mesh_rgb(path, mesh->vertices, rgb,
                                                 mesh->n_verts,
                                                 mesh->triangles,
                                                 mesh->n_tris)
                   : tinygltf_write_glb_mesh(path, mesh->vertices,
                                             mesh->n_verts, mesh->triangles,
                                             mesh->n_tris);
    } else if (has_suffix(path, ".ply")) {
        return mc_write_ply(path, mesh);
    }
    return mc_write_obj(path, mesh);
}

static int write_mesh_file(const mc_mesh *mesh, const char *path)
{
    return write_mesh_file_rgb(mesh, NULL, path, 0, 1);
}

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float *project_image_rgb_to_mesh(const mc_mesh *mesh,
                                        const float *fallback_rgb,
                                        const uint8_t *pixels,
                                        int iw, int ih,
                                        const uint8_t *mask,
                                        int mw, int mh,
                                        const float *pmap,
                                        int pw, int ph,
                                        int *out_projected,
                                        int *out_points)
{
    if (out_projected) *out_projected = 0;
    if (out_points) *out_points = 0;
    if (!mesh || mesh->n_verts <= 0 || !mesh->vertices || !pixels ||
        iw <= 0 || ih <= 0 || !pmap || pw <= 0 || ph <= 0) {
        return NULL;
    }

    float mesh_min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float mesh_max[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX };
    for (int i = 0; i < mesh->n_verts; i++) {
        const float *p = mesh->vertices + (size_t)i * 3;
        for (int k = 0; k < 3; k++) {
            if (p[k] < mesh_min[k]) mesh_min[k] = p[k];
            if (p[k] > mesh_max[k]) mesh_max[k] = p[k];
        }
    }

    float pmin[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float pmax[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX };
    int n_pts = 0;
    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            if (mask && mw > 0 && mh > 0) {
                int mx = clampi((int)(((float)x + 0.5f) * (float)mw / (float)pw), 0, mw - 1);
                int my = clampi((int)(((float)y + 0.5f) * (float)mh / (float)ph), 0, mh - 1);
                if (mask[(size_t)my * mw + mx] == 0) continue;
            }
            const float *q = pmap + ((size_t)y * pw + x) * 3;
            if (!isfinite(q[0]) || !isfinite(q[1]) || !isfinite(q[2])) continue;
            for (int k = 0; k < 3; k++) {
                if (q[k] < pmin[k]) pmin[k] = q[k];
                if (q[k] > pmax[k]) pmax[k] = q[k];
            }
            n_pts++;
        }
    }
    if (n_pts <= 0) return NULL;

    image_texel_point *pts =
        (image_texel_point *)malloc((size_t)n_pts * sizeof(*pts));
    if (!pts) return NULL;

    int grid = 32;
    int nb = grid * grid * grid;
    int *head = (int *)malloc((size_t)nb * sizeof(int));
    if (!head) { free(pts); return NULL; }
    for (int i = 0; i < nb; i++) head[i] = -1;

    int pi = 0;
    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            if (mask && mw > 0 && mh > 0) {
                int mx = clampi((int)(((float)x + 0.5f) * (float)mw / (float)pw), 0, mw - 1);
                int my = clampi((int)(((float)y + 0.5f) * (float)mh / (float)ph), 0, mh - 1);
                if (mask[(size_t)my * mw + mx] == 0) continue;
            }
            const float *q = pmap + ((size_t)y * pw + x) * 3;
            if (!isfinite(q[0]) || !isfinite(q[1]) || !isfinite(q[2])) continue;

            image_texel_point *tp = &pts[pi];
            int cell[3];
            for (int k = 0; k < 3; k++) {
                float denom = pmax[k] - pmin[k];
                float t = (denom > 1e-12f) ? ((q[k] - pmin[k]) / denom) : 0.5f;
                if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
                tp->p[k] = mesh_min[k] + t * (mesh_max[k] - mesh_min[k]);
                cell[k] = clampi((int)(t * (float)grid), 0, grid - 1);
            }

            int ix = clampi((int)(((float)x + 0.5f) * (float)iw / (float)pw), 0, iw - 1);
            int iy = clampi((int)(((float)y + 0.5f) * (float)ih / (float)ph), 0, ih - 1);
            const uint8_t *rgba = pixels + ((size_t)iy * iw + ix) * 4;
            tp->rgb[0] = (float)rgba[0] / 255.0f;
            tp->rgb[1] = (float)rgba[1] / 255.0f;
            tp->rgb[2] = (float)rgba[2] / 255.0f;

            int h = (cell[0] * grid + cell[1]) * grid + cell[2];
            tp->next = head[h];
            head[h] = pi;
            pi++;
        }
    }
    n_pts = pi;
    if (out_points) *out_points = n_pts;
    if (n_pts <= 0) { free(head); free(pts); return NULL; }

    float *out = (float *)malloc((size_t)mesh->n_verts * 3 * sizeof(float));
    if (!out) { free(head); free(pts); return NULL; }
    if (fallback_rgb) {
        memcpy(out, fallback_rgb, (size_t)mesh->n_verts * 3 * sizeof(float));
    } else {
        for (int i = 0; i < mesh->n_verts * 3; i++) out[i] = 0.8f;
    }

    int projected = 0;
    int max_search_radius = grid - 1;
    for (int vi = 0; vi < mesh->n_verts; vi++) {
        const float *vp = mesh->vertices + (size_t)vi * 3;
        int vc[3];
        for (int k = 0; k < 3; k++) {
            float denom = mesh_max[k] - mesh_min[k];
            float t = (denom > 1e-12f) ? ((vp[k] - mesh_min[k]) / denom) : 0.5f;
            if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
            vc[k] = clampi((int)(t * (float)grid), 0, grid - 1);
        }
        float best_d2 = FLT_MAX;
        int best = -1;
        for (int r = 0; r <= max_search_radius && best < 0; r++) {
            for (int dz = -r; dz <= r; dz++) {
                int cz = vc[0] + dz;
                if (cz < 0 || cz >= grid) continue;
                for (int dy = -r; dy <= r; dy++) {
                    int cy = vc[1] + dy;
                    if (cy < 0 || cy >= grid) continue;
                    for (int dx = -r; dx <= r; dx++) {
                        int cx = vc[2] + dx;
                        if (cx < 0 || cx >= grid) continue;
                        int h = (cz * grid + cy) * grid + cx;
                        for (int pidx = head[h]; pidx >= 0; pidx = pts[pidx].next) {
                            float d0 = vp[0] - pts[pidx].p[0];
                            float d1 = vp[1] - pts[pidx].p[1];
                            float d2 = vp[2] - pts[pidx].p[2];
                            float dd = d0*d0 + d1*d1 + d2*d2;
                            if (dd < best_d2) {
                                best_d2 = dd;
                                best = pidx;
                            }
                        }
                    }
                }
            }
        }
        if (best >= 0) {
            out[(size_t)vi * 3 + 0] = pts[best].rgb[0];
            out[(size_t)vi * 3 + 1] = pts[best].rgb[1];
            out[(size_t)vi * 3 + 2] = pts[best].rgb[2];
            projected++;
        }
    }

    if (out_projected) *out_projected = projected;
    free(head);
    free(pts);
    return out;
}

static int write_occupancy_mesh(cuda_sam3d_ctx *ctx, const char *path,
                                float iso)
{
    if (!ctx || !path) return -1;
    int dims[3] = {0, 0, 0};
    if (cuda_sam3d_get_occupancy(ctx, NULL, dims) != 0 ||
        dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1) {
        fprintf(stderr, "[test_cuda_sam3d] mesh export needs occupancy\n");
        return -1;
    }
    size_t n = (size_t)dims[0] * dims[1] * dims[2];
    float *occ = (float *)malloc(n * sizeof(float));
    if (!occ) return -1;
    if (cuda_sam3d_get_occupancy(ctx, occ, NULL) != 0) {
        free(occ);
        return -1;
    }
    float bounds[6] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    mc_mesh mesh = mc_marching_cubes(occ, dims[0], dims[1], dims[2],
                                     iso, bounds);
    free(occ);
    if (mesh.n_verts <= 0 || mesh.n_tris <= 0) {
        fprintf(stderr, "[test_cuda_sam3d] mesh extraction produced no faces at iso=%g\n",
                iso);
        mc_mesh_free(&mesh);
        return -1;
    }

    int rc = write_mesh_file(&mesh, path);
    if (rc == 0) {
        fprintf(stderr, "[test_cuda_sam3d] wrote mesh %d verts %d tris to %s\n",
                mesh.n_verts, mesh.n_tris, path);
    } else {
        fprintf(stderr, "[test_cuda_sam3d] mesh write failed: %s\n", path);
    }
    mc_mesh_free(&mesh);
    return rc;
}

static int write_learned_mesh(cuda_sam3d_ctx *ctx, const char *decoder_path,
                              const char *path, int n_threads,
                              int texture_size, int texture_use_xatlas,
                              int texture_color_mode,
                              const uint8_t *pixels, int iw, int ih,
                              const uint8_t *mask, int mw, int mh,
                              const float *pmap, int pw, int ph)
{
    if (!ctx || !decoder_path || !path) return -1;
    int N = 0, C = 0;
    if (cuda_sam3d_get_slat_tokens(ctx, NULL, NULL, &N, &C) != 0 ||
        N <= 0 || C != 8) {
        fprintf(stderr, "[test_cuda_sam3d] learned mesh export needs SLAT tokens (N=%d C=%d)\n",
                N, C);
        return -1;
    }
    float *feats = (float *)malloc((size_t)N * C * sizeof(float));
    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    if (!feats || !coords) { free(feats); free(coords); return -1; }
    if (cuda_sam3d_get_slat_tokens(ctx, feats, coords, NULL, NULL) != 0) {
        free(feats); free(coords); return -1;
    }
    sam3d_mesh_decoder_model *dec =
        sam3d_mesh_decoder_load_safetensors(decoder_path);
    if (!dec) {
        fprintf(stderr, "[test_cuda_sam3d] cannot load mesh decoder: %s\n",
                decoder_path);
        free(feats); free(coords); return -1;
    }
    mc_mesh mesh = {0};
    float *rgb = NULL;
    int rc = sam3d_mesh_decoder_decode_rgb(dec, coords, feats, N, C, &mesh,
                                           &rgb, n_threads);
    sam3d_mesh_decoder_free(dec);
    free(feats); free(coords);
    if (rc != 0 || mesh.n_verts <= 0 || mesh.n_tris <= 0) {
        fprintf(stderr, "[test_cuda_sam3d] learned mesh extraction failed\n");
        mc_mesh_free(&mesh);
        free(rgb);
        return -1;
    }
    float *write_rgb = rgb;
    float *projected_rgb = NULL;
    if (texture_color_mode != MESH_TEXTURE_COLOR_DECODER) {
        int projected = 0, source_points = 0;
        projected_rgb = project_image_rgb_to_mesh(&mesh, rgb, pixels, iw, ih,
                                                  mask, mw, mh, pmap, pw, ph,
                                                  &projected, &source_points);
        if (projected_rgb && projected > 0) {
            write_rgb = projected_rgb;
            fprintf(stderr,
                    "[test_cuda_sam3d] source-image mesh colors: "
                    "%d/%d vertices from %d pointmap texels%s\n",
                    projected, mesh.n_verts, source_points,
                    texture_color_mode == MESH_TEXTURE_COLOR_AUTO ?
                        " (auto)" : "");
        } else if (texture_color_mode == MESH_TEXTURE_COLOR_IMAGE) {
            fprintf(stderr,
                    "[test_cuda_sam3d] source-image mesh colors unavailable; "
                    "falling back to decoder RGB\n");
        }
    }

    rc = write_mesh_file_rgb(&mesh, write_rgb, path, texture_size,
                             texture_use_xatlas);
    if (rc == 0) {
        fprintf(stderr, "[test_cuda_sam3d] wrote learned mesh %d verts %d tris%s%s to %s\n",
                mesh.n_verts, mesh.n_tris, write_rgb ? " with vertex RGB" : "",
                (write_rgb && texture_size > 0 && has_suffix(path, ".glb")) ?
                    " baked texture" : "",
                path);
    } else {
        fprintf(stderr, "[test_cuda_sam3d] learned mesh write failed: %s\n", path);
    }
    mc_mesh_free(&mesh);
    free(projected_rgb);
    free(rgb);
    return rc;
}

static int file_exists(const char *path)
{
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) return 0;
    fclose(f);
    return 1;
}

static const char *first_existing_path(const char **paths)
{
    for (int i = 0; paths && paths[i]; i++) {
        if (file_exists(paths[i])) return paths[i];
    }
    return NULL;
}

static int run_moge_pointmap(const char *python, const char *script,
                             const char *image, const char *out,
                             const char *model, const char *device)
{
    if (!python || !script || !image || !out || !model || !device) return -1;
    fprintf(stderr, "[test_cuda_sam3d] generating MoGe pointmap: %s\n", out);
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }
    if (pid == 0) {
        char *const argv[] = {
            (char *)python,
            (char *)script,
            (char *)"--image", (char *)image,
            (char *)"--out", (char *)out,
            (char *)"--model", (char *)model,
            (char *)"--device", (char *)device,
            NULL
        };
        execvp(python, argv);
        perror("execvp");
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "[test_cuda_sam3d] MoGe pointmap failed (status=%d)\n",
                status);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    cuda_sam3d_config cfg = {0};
    cfg.device_ordinal = 0;
    cfg.verbose        = 0;
    cfg.precision      = "fp16";
    cfg.seed           = 42;
    cfg.ss_steps       = 2;
    cfg.slat_steps     = 12;
    cfg.cfg_scale      = 2.0f;

    const char *image_path    = NULL;
    const char *mask_path     = NULL;
    const char *pointmap_path = NULL;
    const char *slat_ref_dir  = NULL;
    const char *out_path      = "splat.ply";
    const char *mesh_out_path = NULL;
    const char *mesh_dec_path = NULL;
    float mesh_iso = 0.0f;
    int mesh_only = 0;
    int mesh_source_slat = 0;
    int mesh_threads = 4;
    int mesh_texture_size = 0;
    int mesh_texture_use_xatlas = 1;
    int mesh_texture_color_mode = MESH_TEXTURE_COLOR_AUTO;
    int use_moge = 0;
    const char *moge_python = NULL;
    const char *moge_script = NULL;
    const char *moge_model = NULL;
    const char *moge_device = "cpu";
    const char *moge_out_path = NULL;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) cfg.safetensors_dir = argv[++i];
        else if (!strcmp(a, "--pipeline-yaml")   && i+1 < argc) cfg.pipeline_yaml   = argv[++i];
        else if (!strcmp(a, "--device")          && i+1 < argc) cfg.device_ordinal  = atoi(argv[++i]);
        else if (!strcmp(a, "--precision")       && i+1 < argc) cfg.precision       = argv[++i];
        else if (!strcmp(a, "--pointmap")        && i+1 < argc) pointmap_path       = argv[++i];
        else if (!strcmp(a, "--slat-ref")        && i+1 < argc) slat_ref_dir        = argv[++i];
        else if (!strcmp(a, "--seed")            && i+1 < argc) cfg.seed            = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--steps")           && i+1 < argc) cfg.ss_steps        = atoi(argv[++i]);
        else if (!strcmp(a, "--slat-steps")      && i+1 < argc) cfg.slat_steps      = atoi(argv[++i]);
        else if (!strcmp(a, "--cfg")             && i+1 < argc) cfg.cfg_scale       = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-o")                && i+1 < argc) out_path            = argv[++i];
        else if (!strcmp(a, "--mesh-out")        && i+1 < argc) mesh_out_path       = argv[++i];
        else if (!strcmp(a, "--mesh-decoder")    && i+1 < argc) { mesh_dec_path     = argv[++i]; mesh_source_slat = 1; }
        else if (!strcmp(a, "--mesh-source")     && i+1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "slat")) mesh_source_slat = 1;
            else if (!strcmp(v, "occupancy")) mesh_source_slat = 0;
            else { fprintf(stderr, "unknown --mesh-source: %s\n", v); return 2; }
        }
        else if (!strcmp(a, "--mesh-iso")        && i+1 < argc) mesh_iso            = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--mesh-only"))                     mesh_only           = 1;
        else if (!strcmp(a, "--mesh-threads")    && i+1 < argc) mesh_threads        = atoi(argv[++i]);
        else if (!strcmp(a, "--mesh-texture-size") && i+1 < argc) mesh_texture_size = atoi(argv[++i]);
        else if (!strcmp(a, "--mesh-texture-mode") && i+1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "xatlas")) mesh_texture_use_xatlas = 1;
            else if (!strcmp(v, "grid")) mesh_texture_use_xatlas = 0;
            else { fprintf(stderr, "unknown --mesh-texture-mode: %s\n", v); return 2; }
        }
        else if (!strcmp(a, "--mesh-texture-color") && i+1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) mesh_texture_color_mode = MESH_TEXTURE_COLOR_AUTO;
            else if (!strcmp(v, "decoder")) mesh_texture_color_mode = MESH_TEXTURE_COLOR_DECODER;
            else if (!strcmp(v, "image")) mesh_texture_color_mode = MESH_TEXTURE_COLOR_IMAGE;
            else { fprintf(stderr, "unknown --mesh-texture-color: %s\n", v); return 2; }
        }
        else if (!strcmp(a, "--moge"))                          use_moge            = 1;
        else if (!strcmp(a, "--moge-python")     && i+1 < argc) { moge_python       = argv[++i]; use_moge = 1; }
        else if (!strcmp(a, "--moge-script")     && i+1 < argc) { moge_script       = argv[++i]; use_moge = 1; }
        else if (!strcmp(a, "--moge-model")      && i+1 < argc) { moge_model        = argv[++i]; use_moge = 1; }
        else if (!strcmp(a, "--moge-device")     && i+1 < argc) { moge_device       = argv[++i]; use_moge = 1; }
        else if (!strcmp(a, "--moge-out")        && i+1 < argc) { moge_out_path     = argv[++i]; use_moge = 1; }
        else if (!strcmp(a, "-v"))                              cfg.verbose         = 1;
        else if (!strcmp(a, "-vv"))                             cfg.verbose         = 2;
        else if (a[0] != '-') {
            if      (positional == 0) image_path = a;
            else if (positional == 1) mask_path  = a;
            positional++;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            fprintf(stderr,
                    "Usage: %s [--safetensors-dir DIR | --pipeline-yaml YAML] "
                    "<image.png> <mask.png> "
                    "[--pointmap pmap.npy] [--slat-ref <dir>] "
                    "[--seed N] [--steps N] [--slat-steps N] [--cfg F] "
                    "[-o splat.ply] [--mesh-out mesh.obj|mesh.ply|mesh.glb] "
                    "[--mesh-source occupancy|slat] [--mesh-decoder mesh.st] "
                    "[--mesh-iso F] [--mesh-only] [--mesh-threads N] "
                    "[--mesh-texture-size N] "
                    "[--mesh-texture-mode xatlas|grid] "
                    "[--mesh-texture-color auto|decoder|image] "
                    "[--moge] [--moge-out pmap.npy] "
                    "[--moge-python PY] [--moge-script SCRIPT] "
                    "[--moge-model MODEL] [--moge-device cpu|cuda] "
                    "[--device N] [--precision fp16|bf16|fp32] "
                    "[-v|-vv]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!mesh_out_path && has_suffix(out_path, ".glb")) {
        mesh_out_path = out_path;
        mesh_only = 1;
    }
    char default_mesh_dec[1024];
    if (!mesh_dec_path && cfg.safetensors_dir) {
        snprintf(default_mesh_dec, sizeof(default_mesh_dec),
                 "%s/sam3d_slat_mesh_decoder.safetensors",
                 cfg.safetensors_dir);
        if (file_exists(default_mesh_dec)) {
            mesh_dec_path = default_mesh_dec;
            mesh_source_slat = 1;
        }
    }
    if (mesh_threads <= 0) mesh_threads = 1;
    if (mesh_texture_size < 0) mesh_texture_size = 0;

    const char *moge_python_candidates[] = {
        "ref/sam3d/.venv/bin/python",
        "../../ref/sam3d/.venv/bin/python",
        NULL
    };
    const char *moge_script_candidates[] = {
        "ref/sam3d/moge_pointmap.py",
        "../../ref/sam3d/moge_pointmap.py",
        NULL
    };
    if (!moge_python) {
        const char *p = first_existing_path(moge_python_candidates);
        moge_python = p ? p : "python3";
    }
    if (!moge_script) moge_script = first_existing_path(moge_script_candidates);
    if (!moge_model) {
        moge_model = file_exists("/mnt/disk01/models/moge-vitl/model.pt") ?
            "/mnt/disk01/models/moge-vitl/model.pt" : "Ruicheng/moge-vitl";
    }

    if (!cfg.safetensors_dir && !cfg.pipeline_yaml) {
        fprintf(stderr,
                "[test_cuda_sam3d] need --safetensors-dir or --pipeline-yaml.\n"
                "  Default layout: $MODELS/sam3d/safetensors/ next to "
                "$MODELS/sam3d/checkpoints/pipeline.yaml\n");
        return 2;
    }
    if (!slat_ref_dir && (!image_path || !mask_path)) {
        fprintf(stderr,
                "[test_cuda_sam3d] need <image.png> <mask.png> "
                "(or --slat-ref <dir> to bypass upstream).\n");
        return 2;
    }

    /* Load image (force 4 channels, RGBA). */
    int iw = 0, ih = 0, ichan = 0;
    uint8_t *pixels = NULL;
    if (image_path) {
        pixels = stbi_load(image_path, &iw, &ih, &ichan, 4);
        if (!pixels) { fprintf(stderr, "cannot decode %s\n", image_path); return 3; }
    }

    /* Load mask (grayscale). */
    int mw = 0, mh = 0, mchan = 0;
    uint8_t *mpix = NULL;
    if (mask_path) {
        mpix = stbi_load(mask_path, &mw, &mh, &mchan, 1);
        if (!mpix) {
            fprintf(stderr, "cannot decode %s\n", mask_path);
            stbi_image_free(pixels); return 3;
        }
    }

    char generated_pointmap_path[1024] = {0};
    int generated_pointmap_temp = 0;
    if (!pointmap_path && use_moge) {
        if (!image_path) {
            fprintf(stderr, "[test_cuda_sam3d] --moge needs an input image\n");
            stbi_image_free(pixels); stbi_image_free(mpix);
            return 4;
        }
        if (!moge_script) {
            fprintf(stderr,
                    "[test_cuda_sam3d] cannot find ref/sam3d/moge_pointmap.py; "
                    "pass --moge-script\n");
            stbi_image_free(pixels); stbi_image_free(mpix);
            return 4;
        }
        if (moge_out_path) {
            pointmap_path = moge_out_path;
        } else {
            snprintf(generated_pointmap_path, sizeof(generated_pointmap_path),
                     "/tmp/sam3d_moge_pointmap_%ld.npy", (long)getpid());
            pointmap_path = generated_pointmap_path;
            generated_pointmap_temp = 1;
        }
        if (run_moge_pointmap(moge_python, moge_script, image_path,
                              pointmap_path, moge_model, moge_device) != 0) {
            stbi_image_free(pixels); stbi_image_free(mpix);
            return 4;
        }
    }

    /* Optional pointmap. Use --moge to generate one with the Python MoGe helper. */
    float *pmap = NULL;
    int pmap_dims[8] = {0}, pmap_ndim = 0, pmap_is_f32 = 0;
    if (pointmap_path) {
        pmap = (float *)npy_load(pointmap_path, &pmap_ndim, pmap_dims, &pmap_is_f32);
        if (!pmap || !pmap_is_f32 || pmap_ndim != 3 || pmap_dims[2] != 3) {
            fprintf(stderr, "pointmap must be (H, W, 3) f32; got ndim=%d\n", pmap_ndim);
            free(pmap); stbi_image_free(pixels); stbi_image_free(mpix);
            return 4;
        }
    }

    cuda_sam3d_ctx *ctx = cuda_sam3d_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "[test_cuda_sam3d] cuda_sam3d_create failed\n");
        stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }

    if (pixels && cuda_sam3d_set_image_rgba(ctx, pixels, iw, ih) != 0) {
        fprintf(stderr, "set image failed\n");
        cuda_sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }
    if (mpix && cuda_sam3d_set_mask(ctx, mpix, mw, mh) != 0) {
        fprintf(stderr, "set mask failed\n");
        cuda_sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }
    if (pmap && cuda_sam3d_set_pointmap(ctx, pmap, pmap_dims[1], pmap_dims[0]) != 0) {
        fprintf(stderr, "set pointmap failed\n");
        cuda_sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }

    int rc = 0;
    int32_t *slat_coords = NULL;
    float   *slat_feats  = NULL;

    if (slat_ref_dir) {
        char pbuf[1536];
        int nd = 0, dims[8] = {0}, is_f32 = 0;
        const char *names_c[2] = { "slat_gs_in_coords.npy", "slat_dit_out_coords.npy" };
        const char *names_f[2] = { "slat_gs_in_feats.npy",  "slat_dit_out_feats.npy"  };
        for (int k = 0; k < 2 && !slat_coords; k++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", slat_ref_dir, names_c[k]);
            slat_coords = (int32_t *)npy_load(pbuf, &nd, dims, &is_f32);
        }
        if (!slat_coords || nd != 2 || dims[1] != 4) {
            fprintf(stderr, "--slat-ref: cannot read slat_*_coords.npy from %s\n",
                    slat_ref_dir);
            rc = 6; goto cleanup;
        }
        int N = dims[0];
        for (int k = 0; k < 2 && !slat_feats; k++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", slat_ref_dir, names_f[k]);
            slat_feats = (float *)npy_load(pbuf, &nd, dims, &is_f32);
        }
        if (!slat_feats || !is_f32 || nd != 2 || dims[0] != N) {
            fprintf(stderr, "--slat-ref: cannot read slat_*_feats.npy from %s\n",
                    slat_ref_dir);
            rc = 6; goto cleanup;
        }
        int C = dims[1];
        if (cuda_sam3d_debug_override_slat(ctx, slat_feats, slat_coords, N, C) != 0) {
            fprintf(stderr, "--slat-ref: override failed\n");
            rc = 7; goto cleanup;
        }
        fprintf(stderr, "[test_cuda_sam3d] --slat-ref: loaded N=%d C=%d\n", N, C);
        if (mesh_out_path && mesh_source_slat) {
            if (!mesh_dec_path) {
                fprintf(stderr, "[test_cuda_sam3d] --mesh-source slat needs --mesh-decoder\n");
                rc = 10; goto cleanup;
            }
            if (write_learned_mesh(ctx, mesh_dec_path, mesh_out_path,
                                   mesh_threads, mesh_texture_size,
                                   mesh_texture_use_xatlas,
                                   mesh_texture_color_mode,
                                   pixels, iw, ih, mpix, mw, mh,
                                   pmap, pmap_dims[1], pmap_dims[0]) != 0) {
                rc = 10; goto cleanup;
            }
            if (mesh_only) goto cleanup;
        }
    } else {
        if ((rc = cuda_sam3d_run_dinov2(ctx))     != 0) { fprintf(stderr, "dinov2 rc=%d\n",     rc); goto cleanup; }
        if ((rc = cuda_sam3d_run_cond_fuser(ctx)) != 0) { fprintf(stderr, "cond_fuser rc=%d\n", rc); goto cleanup; }
        if ((rc = cuda_sam3d_run_ss_dit(ctx))     != 0) { fprintf(stderr, "ss_dit rc=%d\n",     rc); goto cleanup; }
        if ((rc = cuda_sam3d_run_ss_decode(ctx))  != 0) { fprintf(stderr, "ss_decode rc=%d\n",  rc); goto cleanup; }
        if (mesh_out_path && !mesh_source_slat) {
            if (write_occupancy_mesh(ctx, mesh_out_path, mesh_iso) != 0) {
                rc = 10; goto cleanup;
            }
            if (mesh_only) goto cleanup;
        }
        if ((rc = cuda_sam3d_run_slat_dit(ctx))   != 0) { fprintf(stderr, "slat_dit rc=%d\n",   rc); goto cleanup; }
        if (mesh_out_path && mesh_source_slat) {
            if (!mesh_dec_path) {
                fprintf(stderr, "[test_cuda_sam3d] --mesh-source slat needs --mesh-decoder\n");
                rc = 10; goto cleanup;
            }
            if (write_learned_mesh(ctx, mesh_dec_path, mesh_out_path,
                                   mesh_threads, mesh_texture_size,
                                   mesh_texture_use_xatlas,
                                   mesh_texture_color_mode,
                                   pixels, iw, ih, mpix, mw, mh,
                                   pmap, pmap_dims[1], pmap_dims[0]) != 0) {
                rc = 10; goto cleanup;
            }
            if (mesh_only) goto cleanup;
        }
    }

    if (mesh_only) {
        fprintf(stderr, "[test_cuda_sam3d] --mesh-only needs --mesh-out\n");
        rc = 10; goto cleanup;
    }
    if ((rc = cuda_sam3d_run_slat_gs_decode(ctx)) != 0) {
        fprintf(stderr, "slat_gs_decode rc=%d\n", rc); goto cleanup;
    }

    int n_gauss = 0;
    if (cuda_sam3d_get_gaussians(ctx, NULL, &n_gauss) != 0 || n_gauss <= 0) {
        fprintf(stderr, "no gaussians produced\n"); rc = 8; goto cleanup;
    }
    float *gaussians = (float *)malloc((size_t)n_gauss * CUDA_SAM3D_GS_STRIDE * sizeof(float));
    if (!gaussians || cuda_sam3d_get_gaussians(ctx, gaussians, NULL) != 0) {
        fprintf(stderr, "get_gaussians failed\n"); free(gaussians); rc = 8; goto cleanup;
    }

    /* INRIA-PLY rows: x y z  nx ny nz  f_dc(3)  opacity_logit
     *                 scale_log(3)  rot(4) — slice into per-channel
     * arrays for gs_ply_write. */
    float *xyz  = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *f_dc = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *op   = (float *)malloc((size_t)n_gauss     * sizeof(float));
    float *scl  = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *rot  = (float *)malloc((size_t)n_gauss * 4 * sizeof(float));
    for (int i = 0; i < n_gauss; i++) {
        const float *row = gaussians + (size_t)i * CUDA_SAM3D_GS_STRIDE;
        xyz [i*3+0] = row[0];  xyz [i*3+1] = row[1];  xyz [i*3+2] = row[2];
        f_dc[i*3+0] = row[6];  f_dc[i*3+1] = row[7];  f_dc[i*3+2] = row[8];
        op[i] = row[9];
        scl [i*3+0] = row[10]; scl [i*3+1] = row[11]; scl [i*3+2] = row[12];
        rot [i*4+0] = row[13]; rot [i*4+1] = row[14];
        rot [i*4+2] = row[15]; rot [i*4+3] = row[16];
    }
    if (gs_ply_write(out_path, n_gauss, xyz, NULL, f_dc, op, scl, rot) != 0) {
        fprintf(stderr, "gs_ply_write failed\n"); rc = 9;
    } else {
        fprintf(stderr, "[test_cuda_sam3d] wrote %d gaussians to %s\n", n_gauss, out_path);
    }
    free(xyz); free(f_dc); free(op); free(scl); free(rot);
    free(gaussians);

cleanup:
    cuda_sam3d_destroy(ctx);
    stbi_image_free(pixels);
    stbi_image_free(mpix);
    free(pmap);
    free(slat_coords); free(slat_feats);
    if (generated_pointmap_temp && generated_pointmap_path[0]) {
        unlink(generated_pointmap_path);
    }
    return rc;
}
