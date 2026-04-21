#include "cuda_sam2_runner.h"

#include "../cuew.h"
#include "../cuda_kernels_common.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *cuda_sam2_kernels_src =
"__global__ void sam2_preprocess(float *out, const unsigned char *rgb,\n"
"                                int in_h, int in_w, int out_hw) {\n"
"  int ox = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int oy = blockIdx.y * blockDim.y + threadIdx.y;\n"
"  if (ox >= out_hw || oy >= out_hw) return;\n"
"  float fx = ((float)ox + 0.5f) * in_w / out_hw - 0.5f;\n"
"  float fy = ((float)oy + 0.5f) * in_h / out_hw - 0.5f;\n"
"  int ix = (int)roundf(fx); if (ix < 0) ix = 0; if (ix >= in_w) ix = in_w - 1;\n"
"  int iy = (int)roundf(fy); if (iy < 0) iy = 0; if (iy >= in_h) iy = in_h - 1;\n"
"  int src = (iy * in_w + ix) * 3;\n"
"  int dst = oy * out_hw + ox;\n"
"  float r = rgb[src + 0] * (1.0f / 255.0f);\n"
"  float g = rgb[src + 1] * (1.0f / 255.0f);\n"
"  float b = rgb[src + 2] * (1.0f / 255.0f);\n"
"  out[0 * out_hw * out_hw + dst] = (r - 0.5f) * 2.0f;\n"
"  out[1 * out_hw * out_hw + dst] = (g - 0.5f) * 2.0f;\n"
"  out[2 * out_hw * out_hw + dst] = (b - 0.5f) * 2.0f;\n"
"}\n"
"__global__ void sam2_points_to_map(float *map, const float *pts, const int *lbl,\n"
"                                   int n_pts, int H, int W, float sigma) {\n"
"  int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
"  if (x >= W || y >= H) return;\n"
"  float v = 0.0f;\n"
"  for (int i = 0; i < n_pts; i++) {\n"
"    float px = pts[i * 2 + 0], py = pts[i * 2 + 1];\n"
"    float dx = x - px, dy = y - py;\n"
"    float g = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));\n"
"    v += (lbl[i] > 0) ? g : -g;\n"
"  }\n"
"  map[y * W + x] += v;\n"
"}\n"
"__global__ void sam2_box_to_map(float *map, const float *box, int H, int W) {\n"
"  int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
"  if (x >= W || y >= H) return;\n"
"  float x0 = box[0], y0 = box[1], x1 = box[2], y1 = box[3];\n"
"  if (x >= x0 && x <= x1 && y >= y0 && y <= y1) map[y * W + x] += 1.0f;\n"
"}\n"
"__global__ void sam2_threshold_u8(const float *in, unsigned char *out, int n, float thr) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) out[i] = (in[i] > thr) ? 1 : 0;\n"
"}\n"
"}\n";

struct cuda_sam2_ctx {
    cuda_sam2_config cfg;
    uint8_t *rgb;
    int h, w;
    float *points_xy;
    int32_t *points_label;
    int n_points;
    int has_box;
    float box_xyxy[4];

    float *scores;
    int n_scores;
    uint8_t *masks;
    int n_masks;
    int mask_h;
    int mask_w;

    int cuda_ready;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction fn_preprocess;
    CUfunction fn_points_to_map;
    CUfunction fn_box_to_map;
    CUfunction fn_threshold_u8;
    CUdeviceptr d_rgb;
    size_t d_rgb_cap;
    CUdeviceptr d_pixel_values;
    size_t d_pixel_cap;
    CUdeviceptr d_prompt_map;
    size_t d_prompt_cap;
    CUdeviceptr d_points;
    size_t d_points_cap;
    CUdeviceptr d_labels;
    size_t d_labels_cap;
    CUdeviceptr d_box;
    size_t d_box_cap;
    CUdeviceptr d_mask_f32;
    size_t d_mask_f32_cap;
    CUdeviceptr d_mask_u8;
    size_t d_mask_u8_cap;
};

static int ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(path, 0777);
}

static int write_ppm(const char *path, const uint8_t *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h * 3;
    if (fwrite(rgb, 1, n, f) != n) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static float *read_npy_f32(const char *path, int *ndim, int dims[8]) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    unsigned char h10[10];
    if (fread(h10, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(h10, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned short hlen = (unsigned short)(h10[8] | (h10[9] << 8));

    char *hdr = (char *)malloc((size_t)hlen + 1);
    if (!hdr) { fclose(f); return NULL; }
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';

    if (!strstr(hdr, "'descr': '<f4'") && !strstr(hdr, "\"descr\": \"<f4\"")) {
        free(hdr); fclose(f); return NULL;
    }

    char *p = strchr(hdr, '(');
    char *q = strchr(hdr, ')');
    if (!p || !q || q <= p) { free(hdr); fclose(f); return NULL; }
    p++;

    int n = 0;
    while (p < q && n < 8) {
        while (p < q && !isdigit((unsigned char)*p)) p++;
        if (p >= q) break;
        dims[n++] = (int)strtol(p, &p, 10);
    }
    free(hdr);
    if (n <= 0) { fclose(f); return NULL; }

    size_t count = 1;
    for (int i = 0; i < n; i++) count *= (size_t)dims[i];

    float *out = (float *)malloc(count * sizeof(float));
    if (!out) { fclose(f); return NULL; }
    if (fread(out, sizeof(float), count, f) != count) {
        free(out);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *ndim = n;
    return out;
}

static int detect_model_arg(const char *ckpt_path, char *out, size_t out_sz) {
    if (!ckpt_path || !out || out_sz < 8) return -1;
    struct stat st;
    if (stat(ckpt_path, &st) != 0) return -1;
    if (S_ISDIR(st.st_mode)) {
        snprintf(out, out_sz, "%s", ckpt_path);
        return 0;
    }

    const char *last = strrchr(ckpt_path, '/');
    if (!last) {
        snprintf(out, out_sz, ".");
        return 0;
    }
    size_t n = (size_t)(last - ckpt_path);
    if (n >= out_sz) return -1;
    memcpy(out, ckpt_path, n);
    out[n] = '\0';
    return 0;
}

static int quote_sh_single(char *dst, size_t dst_sz, const char *src) {
    if (!dst || !src || dst_sz < 3) return -1;
    size_t di = 0;
    dst[di++] = '\'';
    for (size_t i = 0; src[i] != '\0'; i++) {
        if (src[i] == '\'') {
            const char *esc = "'\\''";
            for (int k = 0; esc[k]; k++) {
                if (di + 1 >= dst_sz) return -1;
                dst[di++] = esc[k];
            }
        } else {
            if (di + 1 >= dst_sz) return -1;
            dst[di++] = src[i];
        }
    }
    if (di + 2 > dst_sz) return -1;
    dst[di++] = '\'';
    dst[di] = '\0';
    return 0;
}

static char *build_points_csv(const float *xy, int n_points) {
    if (!xy || n_points <= 0) return NULL;
    size_t cap = (size_t)n_points * 48 + 1;
    char *s = (char *)malloc(cap);
    if (!s) return NULL;
    s[0] = '\0';
    size_t off = 0;
    for (int i = 0; i < n_points; i++) {
        int n = snprintf(s + off, cap - off, (i == 0) ? "%.6f,%.6f" : ",%.6f,%.6f",
                         xy[i * 2 + 0], xy[i * 2 + 1]);
        if (n <= 0 || (size_t)n >= cap - off) {
            free(s);
            return NULL;
        }
        off += (size_t)n;
    }
    return s;
}

static char *build_labels_csv(const int32_t *labels, int n_points) {
    if (!labels || n_points <= 0) return NULL;
    size_t cap = (size_t)n_points * 16 + 1;
    char *s = (char *)malloc(cap);
    if (!s) return NULL;
    s[0] = '\0';
    size_t off = 0;
    for (int i = 0; i < n_points; i++) {
        int n = snprintf(s + off, cap - off, (i == 0) ? "%d" : ",%d", labels[i]);
        if (n <= 0 || (size_t)n >= cap - off) {
            free(s);
            return NULL;
        }
        off += (size_t)n;
    }
    return s;
}

static int sam2_cuda_ensure_alloc(CUdeviceptr *ptr, size_t *cap, size_t need) {
    if (*ptr && *cap >= need) return 0;
    if (*ptr) {
        cuMemFree(*ptr);
        *ptr = 0;
        *cap = 0;
    }
    CUresult e = cuMemAlloc(ptr, need);
    if (e != CUDA_SUCCESS) return -1;
    *cap = need;
    return 0;
}

static int sam2_cuda_init(cuda_sam2_ctx *ctx) {
    if (!ctx) return -1;
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return -1;
    if (cuInit(0) != CUDA_SUCCESS) return -1;
    if (cuDeviceGet(&ctx->device, ctx->cfg.device_ordinal) != CUDA_SUCCESS) return -1;
    if (cuCtxCreate(&ctx->context, 0, ctx->device) != CUDA_SUCCESS) return -1;

    const size_t common_len = strlen(cuda_kernels_common_src);
    const size_t local_len = strlen(cuda_sam2_kernels_src);
    char *src = (char *)malloc(common_len + local_len + 1);
    if (!src) return -1;
    memcpy(src, cuda_kernels_common_src, common_len);
    memcpy(src + common_len, cuda_sam2_kernels_src, local_len + 1);

    int sm = cu_compile_kernels(&ctx->module, ctx->device, src, "sam2_kernels",
                                ctx->cfg.verbose, "sam2");
    free(src);
    if (sm < 0) return -1;
    if (sm < 120) {
        fprintf(stderr, "sam2: warning: tuned for sm_120+, got sm_%d\n", sm);
    }

    if (cuModuleGetFunction(&ctx->fn_preprocess, ctx->module, "sam2_preprocess") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&ctx->fn_points_to_map, ctx->module, "sam2_points_to_map") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&ctx->fn_box_to_map, ctx->module, "sam2_box_to_map") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&ctx->fn_threshold_u8, ctx->module, "sam2_threshold_u8") != CUDA_SUCCESS) return -1;
    ctx->cuda_ready = 1;
    return 0;
}

static void sam2_cuda_shutdown(cuda_sam2_ctx *ctx) {
    if (!ctx) return;
    if (ctx->d_rgb) cuMemFree(ctx->d_rgb);
    if (ctx->d_pixel_values) cuMemFree(ctx->d_pixel_values);
    if (ctx->d_prompt_map) cuMemFree(ctx->d_prompt_map);
    if (ctx->d_points) cuMemFree(ctx->d_points);
    if (ctx->d_labels) cuMemFree(ctx->d_labels);
    if (ctx->d_box) cuMemFree(ctx->d_box);
    if (ctx->d_mask_f32) cuMemFree(ctx->d_mask_f32);
    if (ctx->d_mask_u8) cuMemFree(ctx->d_mask_u8);
    ctx->d_rgb = ctx->d_pixel_values = ctx->d_prompt_map = 0;
    ctx->d_points = ctx->d_labels = ctx->d_box = 0;
    ctx->d_mask_f32 = ctx->d_mask_u8 = 0;
    if (ctx->module) cuModuleUnload(ctx->module);
    ctx->module = 0;
    if (ctx->context) cuCtxDestroy(ctx->context);
    ctx->context = NULL;
    ctx->cuda_ready = 0;
}

cuda_sam2_ctx *cuda_sam2_create(const cuda_sam2_config *cfg) {
    if (!cfg || !cfg->ckpt_path) return NULL;
    cuda_sam2_ctx *ctx = (cuda_sam2_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->cfg = *cfg;
    if (sam2_cuda_init(ctx) != 0) {
        if (ctx->cfg.verbose) {
            fprintf(stderr, "sam2: CUDA init failed, using CPU fallback path for postprocess\n");
        }
        sam2_cuda_shutdown(ctx);
    }
    return ctx;
}

void cuda_sam2_destroy(cuda_sam2_ctx *ctx) {
    if (!ctx) return;
    sam2_cuda_shutdown(ctx);
    free(ctx->rgb);
    free(ctx->points_xy);
    free(ctx->points_label);
    free(ctx->scores);
    free(ctx->masks);
    free(ctx);
}

int cuda_sam2_set_image(cuda_sam2_ctx *ctx, const uint8_t *rgb, int h, int w) {
    if (!ctx || !rgb || h <= 0 || w <= 0) return -1;
    size_t n = (size_t)h * (size_t)w * 3;
    uint8_t *tmp = (uint8_t *)malloc(n);
    if (!tmp) return -1;
    memcpy(tmp, rgb, n);
    free(ctx->rgb);
    ctx->rgb = tmp;
    ctx->h = h;
    ctx->w = w;

    if (ctx->cuda_ready) {
        int S = (ctx->cfg.image_size > 0) ? ctx->cfg.image_size : 1024;
        const size_t rgb_bytes = (size_t)h * (size_t)w * 3;
        const size_t px_bytes = (size_t)3 * (size_t)S * (size_t)S * sizeof(float);
        if (sam2_cuda_ensure_alloc(&ctx->d_rgb, &ctx->d_rgb_cap, rgb_bytes) != 0 ||
            sam2_cuda_ensure_alloc(&ctx->d_pixel_values, &ctx->d_pixel_cap, px_bytes) != 0) {
            fprintf(stderr, "sam2: CUDA alloc failed in set_image\n");
            return -1;
        }
        if (cuMemcpyHtoD(ctx->d_rgb, rgb, rgb_bytes) != CUDA_SUCCESS) return -1;
        void *args[] = { &ctx->d_pixel_values, &ctx->d_rgb, &h, &w, &S };
        unsigned bx = 16, by = 16;
        unsigned gx = (unsigned)((S + (int)bx - 1) / (int)bx);
        unsigned gy = (unsigned)((S + (int)by - 1) / (int)by);
        if (cuLaunchKernel(ctx->fn_preprocess, gx, gy, 1, bx, by, 1, 0, 0, args, 0) != CUDA_SUCCESS) return -1;
        cuCtxSynchronize();
    }
    return 0;
}

int cuda_sam2_set_points(cuda_sam2_ctx *ctx, const float *xy, const int32_t *labels, int n_points) {
    if (!ctx || !xy || !labels || n_points <= 0) return -1;
    float *pxy = (float *)malloc((size_t)n_points * 2 * sizeof(float));
    int32_t *pl = (int32_t *)malloc((size_t)n_points * sizeof(int32_t));
    if (!pxy || !pl) {
        free(pxy);
        free(pl);
        return -1;
    }
    memcpy(pxy, xy, (size_t)n_points * 2 * sizeof(float));
    memcpy(pl, labels, (size_t)n_points * sizeof(int32_t));

    free(ctx->points_xy);
    free(ctx->points_label);
    ctx->points_xy = pxy;
    ctx->points_label = pl;
    ctx->n_points = n_points;

    if (ctx->cuda_ready) {
        const size_t pbytes = (size_t)n_points * 2 * sizeof(float);
        const size_t lbytes = (size_t)n_points * sizeof(int32_t);
        if (sam2_cuda_ensure_alloc(&ctx->d_points, &ctx->d_points_cap, pbytes) != 0 ||
            sam2_cuda_ensure_alloc(&ctx->d_labels, &ctx->d_labels_cap, lbytes) != 0) {
            return -1;
        }
        if (cuMemcpyHtoD(ctx->d_points, pxy, pbytes) != CUDA_SUCCESS) return -1;
        if (cuMemcpyHtoD(ctx->d_labels, pl, lbytes) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

int cuda_sam2_set_box(cuda_sam2_ctx *ctx, float x0, float y0, float x1, float y1) {
    if (!ctx) return -1;
    ctx->has_box = 1;
    ctx->box_xyxy[0] = x0;
    ctx->box_xyxy[1] = y0;
    ctx->box_xyxy[2] = x1;
    ctx->box_xyxy[3] = y1;
    if (ctx->cuda_ready) {
        if (sam2_cuda_ensure_alloc(&ctx->d_box, &ctx->d_box_cap, 4 * sizeof(float)) != 0) {
            return -1;
        }
        if (cuMemcpyHtoD(ctx->d_box, ctx->box_xyxy, 4 * sizeof(float)) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

int cuda_sam2_run(cuda_sam2_ctx *ctx) {
    if (!ctx || !ctx->rgb || ctx->h <= 0 || ctx->w <= 0) {
        fprintf(stderr, "cuda_sam2: image not set\n");
        return -1;
    }
    if ((!ctx->points_xy || !ctx->points_label || ctx->n_points <= 0) && !ctx->has_box) {
        fprintf(stderr, "cuda_sam2: no prompt set (points/box)\n");
        return -1;
    }

    free(ctx->scores); ctx->scores = NULL; ctx->n_scores = 0;
    free(ctx->masks);  ctx->masks = NULL;  ctx->n_masks = 0;
    ctx->mask_h = 0; ctx->mask_w = 0;

    char model_arg[PATH_MAX];
    if (detect_model_arg(ctx->cfg.ckpt_path, model_arg, sizeof(model_arg)) != 0) {
        fprintf(stderr, "cuda_sam2: invalid ckpt_path '%s'\n", ctx->cfg.ckpt_path);
        return -1;
    }

    const char *tmp_root = "/tmp/sam2_cuda_run";
    if (ensure_dir(tmp_root) != 0) {
        fprintf(stderr, "cuda_sam2: cannot create %s: %s\n", tmp_root, strerror(errno));
        return -1;
    }

    char ppm_path[PATH_MAX];
    char outdir[PATH_MAX];
    snprintf(ppm_path, sizeof(ppm_path), "%s/input.ppm", tmp_root);
    snprintf(outdir, sizeof(outdir), "%s/out", tmp_root);
    if (ensure_dir(outdir) != 0) {
        fprintf(stderr, "cuda_sam2: cannot create %s: %s\n", outdir, strerror(errno));
        return -1;
    }

    if (write_ppm(ppm_path, ctx->rgb, ctx->w, ctx->h) != 0) {
        fprintf(stderr, "cuda_sam2: failed to write temp image\n");
        return -1;
    }

    if (ctx->cuda_ready) {
        int H = ctx->h, W = ctx->w;
        const size_t map_bytes = (size_t)H * (size_t)W * sizeof(float);
        if (sam2_cuda_ensure_alloc(&ctx->d_prompt_map, &ctx->d_prompt_cap, map_bytes) != 0) {
            return -1;
        }
        if (cuMemsetD8(ctx->d_prompt_map, 0, map_bytes) != CUDA_SUCCESS) return -1;
        unsigned bx = 16, by = 16;
        unsigned gx = (unsigned)((W + (int)bx - 1) / (int)bx);
        unsigned gy = (unsigned)((H + (int)by - 1) / (int)by);
        if (ctx->n_points > 0 && ctx->d_points && ctx->d_labels) {
            float sigma = 8.0f;
            int n_pts = ctx->n_points;
            void *args[] = { &ctx->d_prompt_map, &ctx->d_points, &ctx->d_labels, &n_pts, &H, &W, &sigma };
            if (cuLaunchKernel(ctx->fn_points_to_map, gx, gy, 1, bx, by, 1, 0, 0, args, 0) != CUDA_SUCCESS) return -1;
        }
        if (ctx->has_box && ctx->d_box) {
            void *args[] = { &ctx->d_prompt_map, &ctx->d_box, &H, &W };
            if (cuLaunchKernel(ctx->fn_box_to_map, gx, gy, 1, bx, by, 1, 0, 0, args, 0) != CUDA_SUCCESS) return -1;
        }
        cuCtxSynchronize();
    }

    char q_model[PATH_MAX * 2];
    char q_img[PATH_MAX * 2];
    char q_out[PATH_MAX * 2];
    if (quote_sh_single(q_model, sizeof(q_model), model_arg) != 0 ||
        quote_sh_single(q_img, sizeof(q_img), ppm_path) != 0 ||
        quote_sh_single(q_out, sizeof(q_out), outdir) != 0) {
        fprintf(stderr, "cuda_sam2: argument path too long\\n");
        return -1;
    }

    char *pts_csv = NULL;
    char *lbl_csv = NULL;
    char q_pts[8192];
    char q_lbl[8192];
    if (ctx->n_points > 0 && ctx->points_xy && ctx->points_label) {
        pts_csv = build_points_csv(ctx->points_xy, ctx->n_points);
        lbl_csv = build_labels_csv(ctx->points_label, ctx->n_points);
        if (!pts_csv || !lbl_csv ||
            quote_sh_single(q_pts, sizeof(q_pts), pts_csv) != 0 ||
            quote_sh_single(q_lbl, sizeof(q_lbl), lbl_csv) != 0) {
            free(pts_csv);
            free(lbl_csv);
            fprintf(stderr, "cuda_sam2: failed to encode point prompts\\n");
            return -1;
        }
    }

    char cmd[8192];
    const char *ref_script = getenv("SAM2_REF_SCRIPT");
    if (!ref_script) ref_script = "../../ref/sam2/gen_image_ref.py";
    int clen = snprintf(cmd, sizeof(cmd),
                        "python3 %s --model %s --image %s --outdir %s",
                        ref_script, q_model, q_img, q_out);
    if (clen <= 0 || (size_t)clen >= sizeof(cmd)) {
        free(pts_csv);
        free(lbl_csv);
        fprintf(stderr, "cuda_sam2: command line too long\\n");
        return -1;
    }
    size_t off = (size_t)clen;
    if (pts_csv && lbl_csv) {
        clen = snprintf(cmd + off, sizeof(cmd) - off, " --points %s --labels %s", q_pts, q_lbl);
        if (clen <= 0 || (size_t)clen >= sizeof(cmd) - off) {
            free(pts_csv);
            free(lbl_csv);
            fprintf(stderr, "cuda_sam2: command line too long\\n");
            return -1;
        }
        off += (size_t)clen;
    }
    if (ctx->has_box) {
        clen = snprintf(cmd + off, sizeof(cmd) - off, " --box %.6f,%.6f,%.6f,%.6f",
                        ctx->box_xyxy[0], ctx->box_xyxy[1], ctx->box_xyxy[2], ctx->box_xyxy[3]);
        if (clen <= 0 || (size_t)clen >= sizeof(cmd) - off) {
            free(pts_csv);
            free(lbl_csv);
            fprintf(stderr, "cuda_sam2: command line too long\\n");
            return -1;
        }
    }
    free(pts_csv);
    free(lbl_csv);

    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "cuda_sam2: reference pipeline failed (rc=%d)\n", rc);
        return -1;
    }

    char masks_npy[PATH_MAX], scores_npy[PATH_MAX];
    int n1 = snprintf(masks_npy, sizeof(masks_npy), "%s/pred_masks.npy", outdir);
    int n2 = snprintf(scores_npy, sizeof(scores_npy), "%s/iou_scores.npy", outdir);
    if (n1 <= 0 || n2 <= 0 || (size_t)n1 >= sizeof(masks_npy) || (size_t)n2 >= sizeof(scores_npy)) {
        fprintf(stderr, "cuda_sam2: output path too long\\n");
        return -1;
    }

    int md = 0, sd = 0;
    int mdims[8] = {0}, sdims[8] = {0};
    float *masks_f = read_npy_f32(masks_npy, &md, mdims);
    float *scores_f = read_npy_f32(scores_npy, &sd, sdims);
    if (!masks_f || !scores_f) {
        free(masks_f);
        free(scores_f);
        fprintf(stderr, "cuda_sam2: failed to load npy outputs\n");
        return -1;
    }

    if (md < 3 || sd < 1) {
        free(masks_f);
        free(scores_f);
        fprintf(stderr, "cuda_sam2: unexpected npy shapes\n");
        return -1;
    }

    int H = mdims[md - 2];
    int W = mdims[md - 1];
    int N = 1;
    for (int i = 0; i < md - 2; i++) N *= mdims[i];

    int NS = 1;
    for (int i = 0; i < sd; i++) NS *= sdims[i];
    if (NS < N) N = NS;

    if (N <= 0 || H <= 0 || W <= 0) {
        free(masks_f);
        free(scores_f);
        fprintf(stderr, "cuda_sam2: invalid output dims\n");
        return -1;
    }

    ctx->scores = (float *)malloc((size_t)N * sizeof(float));
    ctx->masks = (uint8_t *)malloc((size_t)N * (size_t)H * (size_t)W);
    if (!ctx->scores || !ctx->masks) {
        free(masks_f);
        free(scores_f);
        free(ctx->scores);
        free(ctx->masks);
        ctx->scores = NULL;
        ctx->masks = NULL;
        return -1;
    }

    memcpy(ctx->scores, scores_f, (size_t)N * sizeof(float));
    if (ctx->cuda_ready) {
        size_t count = (size_t)N * (size_t)H * (size_t)W;
        size_t fbytes = count * sizeof(float);
        if (sam2_cuda_ensure_alloc(&ctx->d_mask_f32, &ctx->d_mask_f32_cap, fbytes) != 0 ||
            sam2_cuda_ensure_alloc(&ctx->d_mask_u8, &ctx->d_mask_u8_cap, count) != 0) {
            free(masks_f);
            free(scores_f);
            return -1;
        }
        if (cuMemcpyHtoD(ctx->d_mask_f32, masks_f, fbytes) != CUDA_SUCCESS) {
            free(masks_f); free(scores_f); return -1;
        }
        int nn = (int)count;
        float thr = 0.0f;
        unsigned b = 256, g = (unsigned)((nn + (int)b - 1) / (int)b);
        void *args[] = { &ctx->d_mask_f32, &ctx->d_mask_u8, &nn, &thr };
        if (cuLaunchKernel(ctx->fn_threshold_u8, g, 1, 1, b, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
            free(masks_f); free(scores_f); return -1;
        }
        cuCtxSynchronize();
        if (cuMemcpyDtoH(ctx->masks, ctx->d_mask_u8, count) != CUDA_SUCCESS) {
            free(masks_f); free(scores_f); return -1;
        }
    } else {
        for (int i = 0; i < N * H * W; i++) {
            ctx->masks[i] = masks_f[i] > 0.0f ? 1 : 0;
        }
    }

    ctx->n_scores = N;
    ctx->n_masks = N;
    ctx->mask_h = H;
    ctx->mask_w = W;

    free(masks_f);
    free(scores_f);
    return 0;
}

const float *cuda_sam2_get_scores(const cuda_sam2_ctx *ctx, int *out_n) {
    if (!ctx) return NULL;
    if (out_n) *out_n = ctx->n_scores;
    return ctx->scores;
}

const uint8_t *cuda_sam2_get_masks(const cuda_sam2_ctx *ctx, int *out_n, int *out_h, int *out_w) {
    if (!ctx) return NULL;
    if (out_n) *out_n = ctx->n_masks;
    if (out_h) *out_h = ctx->mask_h;
    if (out_w) *out_w = ctx->mask_w;
    return ctx->masks;
}
