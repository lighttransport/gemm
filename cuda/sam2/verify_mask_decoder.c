/*
 * verify_mask_decoder.c — CPU reference verifier for the SAM2 mask decoder.
 *
 * Pipeline (single point prompt, multimask_output=True):
 *   1. Build output tokens: [obj_score, iou, mask_tokens(4)] + sparse(2) = (8, 256)
 *   2. image_embed += dense_prompt   (1,256,64,64)
 *   3. Two-way transformer: 2 blocks + final attention
 *   4. Upscale: convT + feat_s1 + LN + GELU → convT + feat_s0 + GELU → (32,256,256)
 *   5. Hypernetwork MLPs on mask_tokens → (4, 32); masks = hyper @ upscaled
 *   6. IoU head: sigmoid(MLP(iou_token)) → (4,)
 *   7. multimask: drop index 0 → (3, 256, 256) and (3,)
 *
 * Verified against /tmp/sam2_trace/{md_low_res_masks.npy, md_iou_scores.npy}.
 */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_npy_f32(const char *path, int dims[6], int *ndims) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    uint8_t h10[10]; if (fread(h10, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(h10, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint16_t hlen = (uint16_t)(h10[8] | (h10[9] << 8));
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';
    if (!strstr(hdr, "'descr': '<f4'")) { free(hdr); fclose(f); return NULL; }
    char *p = strchr(hdr, '('), *q = strchr(hdr, ')');
    p++; int n = 0;
    while (p < q && n < 6) {
        while (p < q && (*p < '0' || *p > '9')) p++;
        if (p >= q) break;
        dims[n++] = (int)strtol(p, &p, 10);
    }
    free(hdr);
    size_t cnt = 1; for (int i = 0; i < n; i++) cnt *= (size_t)dims[i];
    float *x = (float *)malloc(cnt * sizeof(float));
    if (fread(x, sizeof(float), cnt, f) != cnt) { free(x); fclose(f); return NULL; }
    fclose(f); *ndims = n; return x;
}

static float *T(st_context *st, const char *name) {
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); exit(4); }
    if (strcmp(safetensors_dtype(st, i), "F32")) { fprintf(stderr, "not f32: %s\n", name); exit(4); }
    return (float *)safetensors_data(st, i);
}

static void diff(const char *name, const float *a, const float *b, size_t n) {
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d > mxd) mxd = d; mad += d; }
    mad /= (double)n;
    fprintf(stderr, "  %-18s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
}

/* y[t, o] = b[o] + sum_i W[o, i] * x[t, i]     for t in [0, N) */
static void linear(float *y, const float *x, const float *W, const float *b,
                   int N, int din, int dout) {
    for (int t = 0; t < N; t++) {
        for (int o = 0; o < dout; o++) {
            float acc = b ? b[o] : 0.f;
            const float *wo = W + (size_t)o*din;
            const float *xt = x + (size_t)t*din;
            for (int i = 0; i < din; i++) acc += wo[i] * xt[i];
            y[(size_t)t*dout + o] = acc;
        }
    }
}

/* In-place LayerNorm over last dim (size C), weight+bias shape (C,). */
static void layer_norm(float *x, const float *w, const float *b, int N, int C, float eps) {
    for (int t = 0; t < N; t++) {
        float *xt = x + (size_t)t*C;
        double s = 0, ss = 0;
        for (int c = 0; c < C; c++) s += xt[c];
        float mean = (float)(s / C);
        for (int c = 0; c < C; c++) { float d = xt[c] - mean; ss += d*d; }
        float inv = 1.f / sqrtf((float)(ss / C) + eps);
        for (int c = 0; c < C; c++) xt[c] = (xt[c] - mean) * inv * w[c] + b[c];
    }
}

/* Multi-head attention. Inputs Q (nq, D), K/V (nk, D).
 * q_proj/k_proj/v_proj weights (intD, D), biases (intD,); o_proj (D, intD), bias (D,).
 * Returns out in provided buffer (nq, D). tmp_* are scratch of sufficient size.
 *   Q/K: (intD); scores (nh, nq, nk); attn (nq, intD).
 */
static void attention(
    float *out,
    const float *q_in, const float *k_in, const float *v_in,
    int nq, int nk, int D, int intD, int nh,
    const float *Wq, const float *Bq,
    const float *Wk, const float *Bk,
    const float *Wv, const float *Bv,
    const float *Wo, const float *Bo,
    float *Qp, float *Kp, float *Vp, float *scores, float *attn_out)
{
    int hd = intD / nh;
    float scale = 1.f / sqrtf((float)hd);

    linear(Qp, q_in, Wq, Bq, nq, D, intD);
    linear(Kp, k_in, Wk, Bk, nk, D, intD);
    linear(Vp, v_in, Wv, Bv, nk, D, intD);

    /* scores[h, i, j] = sum_d Q[i, h*hd+d] * K[j, h*hd+d] * scale */
    for (int h = 0; h < nh; h++) {
        for (int i = 0; i < nq; i++) {
            float *sr = scores + ((size_t)h*nq + i)*nk;
            const float *qi = Qp + (size_t)i*intD + h*hd;
            /* softmax(max, sum) inline */
            float mx = -1e30f;
            for (int j = 0; j < nk; j++) {
                const float *kj = Kp + (size_t)j*intD + h*hd;
                float s = 0.f;
                for (int d = 0; d < hd; d++) s += qi[d] * kj[d];
                s *= scale;
                sr[j] = s;
                if (s > mx) mx = s;
            }
            float sm = 0.f;
            for (int j = 0; j < nk; j++) { sr[j] = expf(sr[j] - mx); sm += sr[j]; }
            float inv = 1.f / sm;
            for (int j = 0; j < nk; j++) sr[j] *= inv;
        }
    }

    /* attn_out[i, h*hd+d] = sum_j scores[h,i,j] * V[j, h*hd+d] */
    for (int i = 0; i < nq; i++)
        for (int h = 0; h < nh; h++)
            for (int d = 0; d < hd; d++) {
                float acc = 0.f;
                for (int j = 0; j < nk; j++)
                    acc += scores[((size_t)h*nq + i)*nk + j] * Vp[(size_t)j*intD + h*hd + d];
                attn_out[(size_t)i*intD + h*hd + d] = acc;
            }

    linear(out, attn_out, Wo, Bo, nq, intD, D);
}

static inline float gelu(float x) {
    /* PyTorch nn.GELU default: exact erf form: 0.5*x*(1+erf(x/sqrt(2))) */
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
}

/* Sam2FeedForward: proj_in + relu + (L-2 middle layers + relu) + proj_out [+ sigmoid].
 * Sizes: input_dim → hidden_dim → ... → hidden_dim → output_dim. num_layers >= 2.
 */
static void ffwd(float *out, const float *x, int N,
                 int din, int dh, int dout, int num_layers, int sigmoid_out,
                 const float *W_in, const float *B_in,
                 const float *W_mid, const float *B_mid,   /* concat (L-2) layers: [(dh,dh) per layer, dh per bias] */
                 const float *W_out, const float *B_out)
{
    float *h1 = (float *)malloc((size_t)N*dh*sizeof(float));
    float *h2 = (float *)malloc((size_t)N*dh*sizeof(float));
    linear(h1, x, W_in, B_in, N, din, dh);
    for (int i = 0; i < N*dh; i++) h1[i] = h1[i] > 0 ? h1[i] : 0; /* relu */
    for (int L = 0; L < num_layers - 2; L++) {
        const float *Wl = W_mid + (size_t)L*dh*dh;
        const float *Bl = B_mid + (size_t)L*dh;
        linear(h2, h1, Wl, Bl, N, dh, dh);
        for (int i = 0; i < N*dh; i++) h2[i] = h2[i] > 0 ? h2[i] : 0;
        float *tmp = h1; h1 = h2; h2 = tmp;
    }
    linear(out, h1, W_out, B_out, N, dh, dout);
    if (sigmoid_out) for (int i = 0; i < N*dout; i++) out[i] = 1.f / (1.f + expf(-out[i]));
    free(h1); free(h2);
}

/* ConvTranspose2d with kernel=stride=2 (non-overlapping).
 * in: BCHW (B, Ci, H, W). W_kern: (Ci, Co, 2, 2). bias: (Co,). out: (B, Co, 2H, 2W).
 */
static void conv_transpose2x(float *out, const float *in, const float *W, const float *b,
                             int B, int Ci, int Co, int H, int Wdim) {
    int Ho = 2*H, Wo = 2*Wdim;
    for (int bi = 0; bi < B; bi++)
        for (int co = 0; co < Co; co++)
            for (int oi = 0; oi < Ho; oi++)
                for (int oj = 0; oj < Wo; oj++) {
                    int i = oi/2, j = oj/2, dy = oi%2, dx = oj%2;
                    float acc = b[co];
                    for (int ci = 0; ci < Ci; ci++) {
                        /* W[ci, co, dy, dx] */
                        float w = W[(((size_t)ci*Co + co)*2 + dy)*2 + dx];
                        acc += in[(((size_t)bi*Ci + ci)*H + i)*Wdim + j] * w;
                    }
                    out[(((size_t)bi*Co + co)*Ho + oi)*Wo + oj] = acc;
                }
}

/* In-place LayerNorm over channel dim on BCHW tensor (Sam2LayerNorm channels_first). */
static void layer_norm_chw(float *x, const float *w, const float *b, int B, int C, int H, int W, float eps) {
    int spatial = H*W;
    for (int bi = 0; bi < B; bi++) {
        float *xb = x + (size_t)bi*C*spatial;
        for (int p = 0; p < spatial; p++) {
            double s = 0, ss = 0;
            for (int c = 0; c < C; c++) s += xb[(size_t)c*spatial + p];
            float mean = (float)(s / C);
            for (int c = 0; c < C; c++) { float d = xb[(size_t)c*spatial + p] - mean; ss += d*d; }
            float inv = 1.f / sqrtf((float)(ss / C) + eps);
            for (int c = 0; c < C; c++) {
                float v = (xb[(size_t)c*spatial + p] - mean) * inv;
                xb[(size_t)c*spatial + p] = v * w[c] + b[c];
            }
        }
    }
}

/* ---- Model constants ---- */
#define HD 256            /* hidden_size */
#define NH 8              /* num_attention_heads */
#define SA_INT 256        /* self-attn internal_dim (ds=1) */
#define CA_INT 128        /* cross-attn internal_dim (ds=2) */
#define NL 2              /* num_hidden_layers (two-way blocks) */
#define MD 2048           /* mlp_dim */
#define NMASK 4           /* num_mask_tokens */
#define H_IM 64
#define W_IM 64
#define N_IM (H_IM*W_IM) /* 4096 */

static st_context *ST;

static void load_attn(const char *prefix, int intD,
                      float **Wq, float **Bq, float **Wk, float **Bk,
                      float **Wv, float **Bv, float **Wo, float **Bo) {
    char n[256];
    #define L(v,s) do{ snprintf(n,sizeof(n),"%s.%s",prefix,s); *v = T(ST,n); }while(0)
    L(Wq,"q_proj.weight"); L(Bq,"q_proj.bias");
    L(Wk,"k_proj.weight"); L(Bk,"k_proj.bias");
    L(Wv,"v_proj.weight"); L(Bv,"v_proj.bias");
    L(Wo,"o_proj.weight"); L(Bo,"o_proj.bias");
    (void)intD;
    #undef L
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.safetensors> <refdir>\n", argv[0]); return 1; }
    const char *ckpt = argv[1]; const char *refdir = argv[2]; char path[1024];

    ST = safetensors_open(ckpt);
    if (!ST) { fprintf(stderr, "safetensors_open failed\n"); return 3; }

    /* ---------- Load trace inputs ---------- */
    int d[6], nd;
    snprintf(path, sizeof(path), "%s/md_image_embed.npy", refdir);
    float *img_embed = read_npy_f32(path, d, &nd);                 /* (1,256,64,64) */
    snprintf(path, sizeof(path), "%s/md_image_pe.npy", refdir);
    float *img_pe = read_npy_f32(path, d, &nd);                    /* (1,256,64,64) */
    snprintf(path, sizeof(path), "%s/md_high_res_0.npy", refdir);
    float *hr0 = read_npy_f32(path, d, &nd);                       /* (1,32,256,256) */
    snprintf(path, sizeof(path), "%s/md_high_res_1.npy", refdir);
    float *hr1 = read_npy_f32(path, d, &nd);                       /* (1,64,128,128) */
    snprintf(path, sizeof(path), "%s/prompt_sparse.npy", refdir);
    float *sparse = read_npy_f32(path, d, &nd);                    /* (1,1,2,256) */
    snprintf(path, sizeof(path), "%s/prompt_dense.npy", refdir);
    float *dense = read_npy_f32(path, d, &nd);                     /* (1,256,64,64) */
    if (!img_embed || !img_pe || !hr0 || !hr1 || !sparse || !dense) {
        fprintf(stderr, "failed to load one or more trace tensors\n"); return 2;
    }

    /* ---------- Build tokens (8, 256) ---------- */
    float *obj_t  = T(ST, "mask_decoder.obj_score_token.weight");   /* (1,256) */
    float *iou_t  = T(ST, "mask_decoder.iou_token.weight");          /* (1,256) */
    float *mask_t = T(ST, "mask_decoder.mask_tokens.weight");        /* (4,256) */

    int N_TOK = 8;
    float *queries = (float *)calloc((size_t)N_TOK*HD, sizeof(float));
    memcpy(queries + 0*HD, obj_t, HD*4);
    memcpy(queries + 1*HD, iou_t, HD*4);
    memcpy(queries + 2*HD, mask_t, 4*HD*4);
    /* sparse is (1,1,2,256) — contiguous */
    memcpy(queries + 6*HD, sparse, 2*HD*4);

    /* ---------- image_embed += dense_prompt ---------- */
    size_t spatial = (size_t)HD*N_IM;
    float *keys = (float *)malloc(spatial*sizeof(float));
    float *pe_k = (float *)malloc(spatial*sizeof(float));
    /* Flatten BCHW -> (N_IM, 256). image_embed + dense, transposed. */
    for (int c = 0; c < HD; c++)
        for (int p = 0; p < N_IM; p++) {
            keys[(size_t)p*HD + c] = img_embed[c*N_IM + p] + dense[c*N_IM + p];
            pe_k[(size_t)p*HD + c] = img_pe[c*N_IM + p];
        }

    /* Save ORIGINAL tokens as query_point_embedding (used across all layers for PE add). */
    float *point_pe = (float *)malloc((size_t)N_TOK*HD*sizeof(float));
    memcpy(point_pe, queries, (size_t)N_TOK*HD*sizeof(float));

    /* ---------- Two-way transformer ---------- */
    /* Scratch for attention ops. Max needs for cross-attn (nq=8 or 4096, nk=4096 or 8). */
    int MAX_N = N_IM; /* 4096 */
    float *Qp   = (float *)malloc((size_t)MAX_N*SA_INT*sizeof(float));
    float *Kp   = (float *)malloc((size_t)MAX_N*SA_INT*sizeof(float));
    float *Vp   = (float *)malloc((size_t)MAX_N*SA_INT*sizeof(float));
    float *scrs = (float *)malloc((size_t)NH*N_TOK*N_IM*sizeof(float));
    float *aout = (float *)malloc((size_t)MAX_N*SA_INT*sizeof(float));
    float *qbuf = (float *)malloc((size_t)N_TOK*HD*sizeof(float));
    float *abuf = (float *)malloc((size_t)MAX_N*HD*sizeof(float));
    float *mlpH = (float *)malloc((size_t)N_TOK*MD*sizeof(float));

    const float LN_EPS = 1e-5f;

    for (int layer = 0; layer < NL; layer++) {
        char base[128]; snprintf(base, sizeof(base), "mask_decoder.transformer.layers.%d", layer);
        char name[256];

        /* ---- Self-attn ---- */
        float *Wq, *Bq, *Wk, *Bk, *Wv, *Bv, *Wo, *Bo;
        snprintf(name, sizeof(name), "%s.self_attn", base);
        load_attn(name, SA_INT, &Wq, &Bq, &Wk, &Bk, &Wv, &Bv, &Wo, &Bo);
        if (layer == 0) {
            /* skip_first_layer_pe: queries = self_attn(q=queries, k=queries, v=queries) [replace, no residual] */
            attention(abuf, queries, queries, queries, N_TOK, N_TOK, HD, SA_INT, NH,
                      Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, Qp, Kp, Vp, scrs, aout);
            memcpy(queries, abuf, (size_t)N_TOK*HD*4);
        } else {
            /* q = queries + point_pe; self_attn(q,q,queries); queries += attn_out */
            for (int i = 0; i < N_TOK*HD; i++) qbuf[i] = queries[i] + point_pe[i];
            attention(abuf, qbuf, qbuf, queries, N_TOK, N_TOK, HD, SA_INT, NH,
                      Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, Qp, Kp, Vp, scrs, aout);
            for (int i = 0; i < N_TOK*HD; i++) queries[i] += abuf[i];
        }
        /* LN1 */
        snprintf(name, sizeof(name), "%s.layer_norm1.weight", base); float *ln1w = T(ST, name);
        snprintf(name, sizeof(name), "%s.layer_norm1.bias",   base); float *ln1b = T(ST, name);
        layer_norm(queries, ln1w, ln1b, N_TOK, HD, LN_EPS);

        /* ---- Cross-attn token → image ---- */
        snprintf(name, sizeof(name), "%s.cross_attn_token_to_image", base);
        load_attn(name, CA_INT, &Wq, &Bq, &Wk, &Bk, &Wv, &Bv, &Wo, &Bo);
        /* q = queries + point_pe; k = keys + key_pe; v = keys */
        float *qq = (float *)malloc((size_t)N_TOK*HD*sizeof(float));
        float *kk = (float *)malloc((size_t)N_IM*HD*sizeof(float));
        for (int i = 0; i < N_TOK*HD; i++) qq[i] = queries[i] + point_pe[i];
        for (int i = 0; i < N_IM*HD; i++)  kk[i] = keys[i] + pe_k[i];
        attention(abuf, qq, kk, keys, N_TOK, N_IM, HD, CA_INT, NH,
                  Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, Qp, Kp, Vp, scrs, aout);
        for (int i = 0; i < N_TOK*HD; i++) queries[i] += abuf[i];
        /* LN2 */
        snprintf(name, sizeof(name), "%s.layer_norm2.weight", base); float *ln2w = T(ST, name);
        snprintf(name, sizeof(name), "%s.layer_norm2.bias",   base); float *ln2b = T(ST, name);
        layer_norm(queries, ln2w, ln2b, N_TOK, HD, LN_EPS);

        /* ---- MLP (Sam2FeedForward with 2 layers = proj_in + relu + proj_out) ---- */
        snprintf(name, sizeof(name), "%s.mlp.proj_in.weight", base);  float *Wmi = T(ST, name);
        snprintf(name, sizeof(name), "%s.mlp.proj_in.bias",   base);  float *Bmi = T(ST, name);
        snprintf(name, sizeof(name), "%s.mlp.proj_out.weight", base); float *Wmo = T(ST, name);
        snprintf(name, sizeof(name), "%s.mlp.proj_out.bias",   base); float *Bmo = T(ST, name);
        linear(mlpH, queries, Wmi, Bmi, N_TOK, HD, MD);
        for (int i = 0; i < N_TOK*MD; i++) mlpH[i] = mlpH[i] > 0 ? mlpH[i] : 0; /* relu */
        linear(qbuf, mlpH, Wmo, Bmo, N_TOK, MD, HD);
        for (int i = 0; i < N_TOK*HD; i++) queries[i] += qbuf[i];
        /* LN3 */
        snprintf(name, sizeof(name), "%s.layer_norm3.weight", base); float *ln3w = T(ST, name);
        snprintf(name, sizeof(name), "%s.layer_norm3.bias",   base); float *ln3b = T(ST, name);
        layer_norm(queries, ln3w, ln3b, N_TOK, HD, LN_EPS);

        /* ---- Cross-attn image → token: q=keys+pe_k, k=queries+point_pe, v=queries ---- */
        snprintf(name, sizeof(name), "%s.cross_attn_image_to_token", base);
        load_attn(name, CA_INT, &Wq, &Bq, &Wk, &Bk, &Wv, &Bv, &Wo, &Bo);
        for (int i = 0; i < N_TOK*HD; i++) qq[i] = queries[i] + point_pe[i];
        for (int i = 0; i < N_IM*HD; i++)  kk[i] = keys[i] + pe_k[i];
        attention(abuf, kk, qq, queries, N_IM, N_TOK, HD, CA_INT, NH,
                  Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, Qp, Kp, Vp, scrs, aout);
        for (int i = 0; i < N_IM*HD; i++) keys[i] += abuf[i];
        /* LN4 on keys */
        snprintf(name, sizeof(name), "%s.layer_norm4.weight", base); float *ln4w = T(ST, name);
        snprintf(name, sizeof(name), "%s.layer_norm4.bias",   base); float *ln4b = T(ST, name);
        layer_norm(keys, ln4w, ln4b, N_IM, HD, LN_EPS);

        free(qq); free(kk);
    }

    /* ---- Final attention token → image ---- */
    {
        float *Wq, *Bq, *Wk, *Bk, *Wv, *Bv, *Wo, *Bo;
        load_attn("mask_decoder.transformer.final_attn_token_to_image", CA_INT,
                  &Wq, &Bq, &Wk, &Bk, &Wv, &Bv, &Wo, &Bo);
        float *qq = (float *)malloc((size_t)N_TOK*HD*sizeof(float));
        float *kk = (float *)malloc((size_t)N_IM*HD*sizeof(float));
        for (int i = 0; i < N_TOK*HD; i++) qq[i] = queries[i] + point_pe[i];
        for (int i = 0; i < N_IM*HD; i++)  kk[i] = keys[i] + pe_k[i];
        attention(abuf, qq, kk, keys, N_TOK, N_IM, HD, CA_INT, NH,
                  Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, Qp, Kp, Vp, scrs, aout);
        for (int i = 0; i < N_TOK*HD; i++) queries[i] += abuf[i];
        float *lnw = T(ST, "mask_decoder.transformer.layer_norm_final_attn.weight");
        float *lnb = T(ST, "mask_decoder.transformer.layer_norm_final_attn.bias");
        layer_norm(queries, lnw, lnb, N_TOK, HD, LN_EPS);
        free(qq); free(kk);
    }

    /* ---- IoU prediction head ---- */
    float *iou_tok_out = queries + 1*HD;  /* (HD,) */
    float *iou_pred = (float *)malloc(NMASK*sizeof(float));
    {
        float *Wi = T(ST, "mask_decoder.iou_prediction_head.proj_in.weight");
        float *Bi = T(ST, "mask_decoder.iou_prediction_head.proj_in.bias");
        float *Wm = T(ST, "mask_decoder.iou_prediction_head.layers.0.weight");
        float *Bm = T(ST, "mask_decoder.iou_prediction_head.layers.0.bias");
        float *Wo = T(ST, "mask_decoder.iou_prediction_head.proj_out.weight");
        float *Bo = T(ST, "mask_decoder.iou_prediction_head.proj_out.bias");
        ffwd(iou_pred, iou_tok_out, 1, HD, HD, NMASK, 3, 1, Wi, Bi, Wm, Bm, Wo, Bo);
    }

    /* ---- Upscale ---- */
    /* keys back to BCHW (1,256,64,64) */
    float *key_chw = (float *)malloc(spatial*sizeof(float));
    for (int c = 0; c < HD; c++)
        for (int p = 0; p < N_IM; p++) key_chw[c*N_IM + p] = keys[(size_t)p*HD + c];

    /* upscale_conv1: (256,64,2,2) + bias(64) stride 2 → (1,64,128,128) */
    float *Wc1 = T(ST, "mask_decoder.upscale_conv1.weight");
    float *Bc1 = T(ST, "mask_decoder.upscale_conv1.bias");
    float *up1 = (float *)malloc((size_t)64*128*128*sizeof(float));
    conv_transpose2x(up1, key_chw, Wc1, Bc1, 1, 256, 64, 64, 64);
    /* + feat_s1 (1,64,128,128) */
    for (size_t i = 0; i < (size_t)64*128*128; i++) up1[i] += hr1[i];
    /* LN then GELU */
    float *lnw = T(ST, "mask_decoder.upscale_layer_norm.weight");
    float *lnb = T(ST, "mask_decoder.upscale_layer_norm.bias");
    layer_norm_chw(up1, lnw, lnb, 1, 64, 128, 128, 1e-6f);
    for (size_t i = 0; i < (size_t)64*128*128; i++) up1[i] = gelu(up1[i]);

    /* upscale_conv2: (64,32,2,2) → (1,32,256,256) */
    float *Wc2 = T(ST, "mask_decoder.upscale_conv2.weight");
    float *Bc2 = T(ST, "mask_decoder.upscale_conv2.bias");
    float *up2 = (float *)malloc((size_t)32*256*256*sizeof(float));
    conv_transpose2x(up2, up1, Wc2, Bc2, 1, 64, 32, 128, 128);
    /* + feat_s0 + GELU */
    for (size_t i = 0; i < (size_t)32*256*256; i++) up2[i] = gelu(up2[i] + hr0[i]);

    /* ---- Hypernetwork MLPs on mask_tokens[2..5] ---- */
    float hyper[NMASK*32];
    for (int m = 0; m < NMASK; m++) {
        char base[128]; snprintf(base, sizeof(base), "mask_decoder.output_hypernetworks_mlps.%d", m);
        char n[256];
        #define GT(v,s) do{ snprintf(n,sizeof(n),"%s.%s",base,s); v = T(ST,n); }while(0)
        float *Wi, *Bi, *Wm, *Bm, *Wo, *Bo;
        GT(Wi,"proj_in.weight"); GT(Bi,"proj_in.bias");
        GT(Wm,"layers.0.weight"); GT(Bm,"layers.0.bias");
        GT(Wo,"proj_out.weight"); GT(Bo,"proj_out.bias");
        #undef GT
        ffwd(&hyper[m*32], queries + (2+m)*HD, 1, HD, HD, 32, 3, 0, Wi, Bi, Wm, Bm, Wo, Bo);
    }

    /* ---- masks = hyper @ upscaled ---- */
    /* hyper: (NMASK, 32). upscaled_flat: (32, 256*256). masks: (NMASK, 256*256). */
    int HW = 256*256;
    float *masks = (float *)malloc((size_t)NMASK*HW*sizeof(float));
    for (int m = 0; m < NMASK; m++) {
        for (int p = 0; p < HW; p++) {
            float acc = 0.f;
            for (int c = 0; c < 32; c++) acc += hyper[m*32 + c] * up2[c*HW + p];
            masks[m*HW + p] = acc;
        }
    }

    /* ---- multimask slice (drop index 0) ---- */
    /* Reference md_low_res_masks.npy: (1,1,3,256,256); md_iou_scores: (1,1,3). */
    snprintf(path, sizeof(path), "%s/md_low_res_masks.npy", refdir);
    float *ref_masks = read_npy_f32(path, d, &nd);
    snprintf(path, sizeof(path), "%s/md_iou_scores.npy", refdir);
    float *ref_iou = read_npy_f32(path, d, &nd);
    diff("md_iou_scores", &iou_pred[1], ref_iou, 3);
    diff("md_low_res_masks", &masks[1*HW], ref_masks, (size_t)3*HW);

    /* cleanup intentionally partial */
    free(img_embed); free(img_pe); free(hr0); free(hr1); free(sparse); free(dense);
    free(queries); free(keys); free(pe_k);
    free(Qp); free(Kp); free(Vp); free(scrs); free(aout); free(qbuf); free(abuf); free(mlpH);
    free(iou_pred); free(key_chw); free(up1); free(up2); free(masks);
    if (ref_masks) free(ref_masks); if (ref_iou) free(ref_iou);
    safetensors_close(ST);
    return 0;
}
