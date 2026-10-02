/* Swin V1 Large inference for the RMBG-2.0 backbone (FP32, batch one).
 * The caller supplies row-major GEMM and allocation/error handling through
 * SWIN_LINEAR / SWIN_MATMUL. No framework or vendor BLAS dependency.
 * Features are caller-owned CHW arrays. Weights stay in the mapped checkpoint.
 */
#ifndef SWIN_V1_H
#define SWIN_V1_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "safetensors.h"

typedef struct {
    const float *norm1_w, *norm1_b, *qkv_w, *qkv_b, *relative_bias;
    const float *proj_w, *proj_b, *norm2_w, *norm2_b;
    const float *fc1_w, *fc1_b, *fc2_w, *fc2_b;
} swin_block;
typedef struct {
    st_context *weights;
    const float *patch_w, *patch_b, *patch_norm_w, *patch_norm_b;
    swin_block blocks[4][18];
    const float *merge_w[3], *merge_norm_w[3], *merge_norm_b[3];
    const float *out_norm_w[4], *out_norm_b[4];
} swin_model;
typedef struct { float *data; int h, w, c; } swin_feature;
static const int swin_depths[4] = {2, 2, 18, 2};

static const float *swin_weight(st_context *st, const char *name, int rank,
                                int a, int b, int c, int d)
{
    int i = safetensors_find(st, name), dims[4] = {a, b, c, d};
    size_t count = 1;
    if (i < 0 || strcmp(safetensors_dtype(st, i), "F32") ||
        safetensors_ndims(st, i) != rank) goto bad;
    for (int j = 0; j < rank; j++) {
        if (safetensors_shape(st, i)[j] != (uint64_t)dims[j]) goto bad;
        count *= dims[j];
    }
    if (safetensors_nbytes(st, i) != count * sizeof(float)) goto bad;
    return safetensors_data(st, i);
bad:
    fprintf(stderr, "swin: missing or incompatible FP32 tensor: %s\n", name);
    return NULL;
}

static swin_model *swin_load(const char *path)
{
    swin_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->weights = safetensors_open(path);
    if (!m->weights) { free(m); return NULL; }
    char name[256];
#define SWIN_GET(dst, key, rank, a, b, c, d) do { \
    (dst) = swin_weight(m->weights, key, rank, a, b, c, d); \
    if (!(dst)) goto bad; \
} while (0)
    SWIN_GET(m->patch_w, "bb.patch_embed.proj.weight", 4, 192, 3, 4, 4);
    SWIN_GET(m->patch_b, "bb.patch_embed.proj.bias", 1, 192, 0, 0, 0);
    SWIN_GET(m->patch_norm_w, "bb.patch_embed.norm.weight", 1, 192, 0, 0, 0);
    SWIN_GET(m->patch_norm_b, "bb.patch_embed.norm.bias", 1, 192, 0, 0, 0);
    for (int s = 0; s < 4; s++) {
        int c = 192 << s;
        snprintf(name, sizeof(name), "bb.norm%d.weight", s);
        SWIN_GET(m->out_norm_w[s], name, 1, c, 0, 0, 0);
        snprintf(name, sizeof(name), "bb.norm%d.bias", s);
        SWIN_GET(m->out_norm_b[s], name, 1, c, 0, 0, 0);
        for (int j = 0; j < swin_depths[s]; j++) {
            swin_block *b = &m->blocks[s][j];
#define BLOCK_GET(field, suffix, rank, a, bdim) do { \
    snprintf(name, sizeof(name), "bb.layers.%d.blocks.%d.%s", s, j, suffix); \
    SWIN_GET(b->field, name, rank, a, bdim, 0, 0); \
} while (0)
            BLOCK_GET(norm1_w, "norm1.weight", 1, c, 0);
            BLOCK_GET(norm1_b, "norm1.bias", 1, c, 0);
            BLOCK_GET(qkv_w, "attn.qkv.weight", 2, 3*c, c);
            BLOCK_GET(qkv_b, "attn.qkv.bias", 1, 3*c, 0);
            BLOCK_GET(relative_bias, "attn.relative_position_bias_table", 2, 529, c/32);
            BLOCK_GET(proj_w, "attn.proj.weight", 2, c, c);
            BLOCK_GET(proj_b, "attn.proj.bias", 1, c, 0);
            BLOCK_GET(norm2_w, "norm2.weight", 1, c, 0);
            BLOCK_GET(norm2_b, "norm2.bias", 1, c, 0);
            BLOCK_GET(fc1_w, "mlp.fc1.weight", 2, 4*c, c);
            BLOCK_GET(fc1_b, "mlp.fc1.bias", 1, 4*c, 0);
            BLOCK_GET(fc2_w, "mlp.fc2.weight", 2, c, 4*c);
            BLOCK_GET(fc2_b, "mlp.fc2.bias", 1, c, 0);
#undef BLOCK_GET
            /* The trained bias lookup must have the standard V1 indexing. */
            snprintf(name, sizeof(name), "bb.layers.%d.blocks.%d.attn.relative_position_index", s, j);
            int i = safetensors_find(m->weights, name);
            if (i < 0 || strcmp(safetensors_dtype(m->weights, i), "I64") ||
                safetensors_ndims(m->weights, i) != 2 ||
                safetensors_shape(m->weights, i)[0] != 144 ||
                safetensors_shape(m->weights, i)[1] != 144 ||
                safetensors_nbytes(m->weights, i) != 144*144*sizeof(int64_t)) goto bad;
            const int64_t *indices = safetensors_data(m->weights, i);
            for (int q = 0; q < 144; q++) for (int k = 0; k < 144; k++)
                if (indices[q*144+k] != (q/12-k/12+11)*23+q%12-k%12+11) goto bad;
        }
        if (s < 3) {
            snprintf(name, sizeof(name), "bb.layers.%d.downsample.reduction.weight", s);
            SWIN_GET(m->merge_w[s], name, 2, 2*c, 4*c, 0, 0);
            snprintf(name, sizeof(name), "bb.layers.%d.downsample.norm.weight", s);
            SWIN_GET(m->merge_norm_w[s], name, 1, 4*c, 0, 0, 0);
            snprintf(name, sizeof(name), "bb.layers.%d.downsample.norm.bias", s);
            SWIN_GET(m->merge_norm_b[s], name, 1, 4*c, 0, 0, 0);
        }
    }
#undef SWIN_GET
    return m;
bad:
    fprintf(stderr, "swin: checkpoint is not the supported RMBG-2.0 Swin-L\n");
    safetensors_close(m->weights); free(m); return NULL;
}

static void swin_free(swin_model *m)
{
    if (m) { safetensors_close(m->weights); free(m); }
}

static void swin_norm(float *out, const float *x, const float *w, const float *b, int n, int c)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const float *p = x + (size_t)i*c;
        double mean = 0, var = 0;
        for (int j = 0; j < c; j++) mean += p[j];
        mean /= c;
        for (int j = 0; j < c; j++) { double v = p[j]-mean; var += v*v; }
        double r = 1.0/sqrt(var/c+1e-5);
        for (int j = 0; j < c; j++) out[(size_t)i*c+j] = (float)((p[j]-mean)*r*w[j]+b[j]);
    }
}

/* Region IDs reproduce the upstream three-by-three shifted-window mask.
 * Padding tokens are NOT masked: upstream attends to their zero input. */
static int swin_region(int y, int x, int hp, int wp)
{
    int ry = y < hp-12 ? 0 : y < hp-6 ? 1 : 2;
    int rx = x < wp-12 ? 0 : x < wp-6 ? 1 : 2;
    return 3*ry+rx;
}

static void swin_attention(float *out, const float *qkv, const float *bias,
                           int hp, int wp, int c, int shift)
{
    int windows = hp/12*(wp/12), heads = c/32;
    #pragma omp parallel
    {
        float q[144*32], kt[32*144], v[144*32], scores[144*144], av[144*32];
        #pragma omp for schedule(static)
        for (int job = 0; job < windows*heads; job++) {
            int win = job/heads, head = job%heads;
            const float *p = qkv+(size_t)win*144*3*c+head*32;
            for (int i = 0; i < 144; i++) for (int d = 0; d < 32; d++) {
                q[i*32+d] = p[(size_t)i*3*c+d]*(1.0f/sqrtf(32));
                kt[d*144+i] = p[(size_t)i*3*c+c+d];
                v[i*32+d] = p[(size_t)i*3*c+2*c+d];
            }
            SWIN_MATMUL(scores, q, kt, 144, 144, 32);
            int wy = win/(wp/12)*12, wx = win%(wp/12)*12;
            for (int i = 0; i < 144; i++) {
                float maximum = -INFINITY, sum = 0;
                for (int j = 0; j < 144; j++) {
                    int rel = (i/12-j/12+11)*23+i%12-j%12+11;
                    float val = scores[i*144+j]+bias[rel*heads+head];
                    if (shift && swin_region(wy+i/12, wx+i%12, hp, wp) !=
                                 swin_region(wy+j/12, wx+j%12, hp, wp)) val -= 100;
                    scores[i*144+j] = val;
                    if (val > maximum) maximum = val;
                }
                for (int j = 0; j < 144; j++) {
                    float val = expf(scores[i*144+j]-maximum);
                    scores[i*144+j] = val; sum += val;
                }
                for (int j = 0; j < 144; j++) scores[i*144+j] /= sum;
            }
            SWIN_MATMUL(av, scores, v, 144, 32, 144);
            for (int i = 0; i < 144; i++)
                memcpy(out+((size_t)win*144+i)*c+head*32, av+i*32, 32*sizeof(float));
        }
    }
}

static void swin_run_block(float *x, int h, int w, int c, const swin_block *b, int shift)
{
    int hp = (h+11)/12*12, wp = (w+11)/12*12, n = h*w, padded = hp*wp;
    float *norm = SWIN_ALLOC((size_t)n*c*sizeof(float));
    float *windows = SWIN_ALLOC((size_t)padded*c*sizeof(float));
    float *qkv = SWIN_ALLOC((size_t)padded*3*c*sizeof(float));
    float *attn = SWIN_ALLOC((size_t)padded*c*sizeof(float));
    swin_norm(norm, x, b->norm1_w, b->norm1_b, n, c);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < hp; y++) for (int col = 0; col < wp; col++) {
        int sy = (y+shift)%hp, sx = (col+shift)%wp;
        int off = ((y/12)*(wp/12)+col/12)*144+(y%12)*12+col%12;
        float *dst = windows+(size_t)off*c;
        if (sy < h && sx < w) memcpy(dst, norm+((size_t)sy*w+sx)*c, c*sizeof(float));
        else memset(dst, 0, c*sizeof(float));
    }
    SWIN_LINEAR(qkv, b->qkv_w, b->qkv_b, windows, padded, 3*c, c);
    swin_attention(attn, qkv, b->relative_bias, hp, wp, c, shift);
    free(qkv);
    SWIN_LINEAR(windows, b->proj_w, b->proj_b, attn, padded, c, c);
    free(attn);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++) for (int col = 0; col < w; col++) {
        int sy = (y-shift+hp)%hp, sx = (col-shift+wp)%wp;
        int off = (sy/12*(wp/12)+sx/12)*144+sy%12*12+sx%12;
        for (int d = 0; d < c; d++) x[((size_t)y*w+col)*c+d] += windows[(size_t)off*c+d];
    }
    free(windows);
    swin_norm(norm, x, b->norm2_w, b->norm2_b, n, c);
    float *hidden = SWIN_ALLOC((size_t)n*4*c*sizeof(float));
    SWIN_LINEAR(hidden, b->fc1_w, b->fc1_b, norm, n, 4*c, c);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)n*4*c; i++)
        hidden[i] = (float)(0.5*hidden[i]*(1+erf(hidden[i]*0.7071067811865475244)));
    SWIN_LINEAR(norm, b->fc2_w, b->fc2_b, hidden, n, c, 4*c);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)n*c; i++) x[i] += norm[i];
    free(hidden); free(norm);
}

static void swin_predict(const swin_model *m, const float *chw, int ih, int iw,
                         swin_feature features[4])
{
    int h = (ih+3)/4, w = (iw+3)/4, c = 192;
    float *patches = SWIN_ALLOC((size_t)h*w*48*sizeof(float));
    float *x = SWIN_ALLOC((size_t)h*w*c*sizeof(float));
    for (int y = 0; y < h; y++) for (int col = 0; col < w; col++)
        for (int d = 0; d < 3; d++) for (int ky = 0; ky < 4; ky++) for (int kx = 0; kx < 4; kx++) {
            int sy = y*4+ky, sx = col*4+kx;
            patches[((size_t)y*w+col)*48+d*16+ky*4+kx] =
                sy < ih && sx < iw ? chw[((size_t)d*ih+sy)*iw+sx] : 0;
        }
    SWIN_LINEAR(x, m->patch_w, m->patch_b, patches, h*w, c, 48);
    free(patches);
    swin_norm(x, x, m->patch_norm_w, m->patch_norm_b, h*w, c);
    for (int s = 0; s < 4; s++) {
        for (int j = 0; j < swin_depths[s]; j++) {
            swin_run_block(x, h, w, c, &m->blocks[s][j], j%2 ? 6 : 0);
#ifdef SWIN_TRACE
            SWIN_TRACE(s, j, x, h, w, c);
#endif
        }
        float *norm = SWIN_ALLOC((size_t)h*w*c*sizeof(float));
        swin_norm(norm, x, m->out_norm_w[s], m->out_norm_b[s], h*w, c);
        features[s] = (swin_feature){SWIN_ALLOC((size_t)h*w*c*sizeof(float)), h, w, c};
        for (int i = 0; i < h*w; i++) for (int d = 0; d < c; d++)
            features[s].data[(size_t)d*h*w+i] = norm[(size_t)i*c+d];
        free(norm);
        if (s < 3) {
            int nh = (h+1)/2, nw = (w+1)/2;
            float *merged = SWIN_ALLOC((size_t)nh*nw*4*c*sizeof(float));
            for (int y = 0; y < nh; y++) for (int col = 0; col < nw; col++)
                for (int k = 0; k < 4; k++) {
                    int sy = y*2+k%2, sx = col*2+k/2;
                    float *dst = merged+((size_t)y*nw+col)*4*c+k*c;
                    if (sy < h && sx < w) memcpy(dst, x+((size_t)sy*w+sx)*c, c*sizeof(float));
                    else memset(dst, 0, c*sizeof(float));
                }
            free(x);
            swin_norm(merged, merged, m->merge_norm_w[s], m->merge_norm_b[s], nh*nw, 4*c);
            x = SWIN_ALLOC((size_t)nh*nw*2*c*sizeof(float));
            SWIN_LINEAR(x, m->merge_w[s], NULL, merged, nh*nw, 2*c, 4*c);
            free(merged); h = nh; w = nw; c *= 2;
        }
    }
    free(x);
}
#endif
