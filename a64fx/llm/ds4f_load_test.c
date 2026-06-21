/* ds4f_load_test.c — validate ds4f_load_real() against the staged rank00 blob.
 *
 * Loads a FEW layers (DS4F_TEST_LAYERS, default 1) so the arena stays small
 * enough to run on a shared login node, then checks that the loaded weights are
 * (a) the right dtype/shape (the loader aborts otherwise) and (b) dequant to the
 * SAME VALUES the upstream DeepSeek conventions imply:
 *
 *   - MXFP4 experts: re-open the raw blob, dequant expert-0 w1 the DeepSeek way
 *     (float4_e2m1fn_x2 sequential nibbles, e2m1 table max 6, plain E8M0) and
 *     compare against the kernel-convention dequant of the LOADED arena bytes
 *     (repacked (j,j+16), 2x table, E8M0-1). They must match -> proves the
 *     nibble repack + scale decrement are exact.
 *   - FP8 dense (wq_a): dequant a row via the e4m3fn LUT; all finite, |v|<=448.
 *   - norms / hc / sink: finite.
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE \
 *       -I../../common -o build/ds4f_load_test ds4f_load_test.c -lm -lpthread
 * Run:
 *   DS4F_TEST_LAYERS=1 DS4F_STAGE_DIR=/local/ds4f ./build/ds4f_load_test
 */
#include "ds4f.h"

/* standard OCP e2m1 (float4_e2m1fn) value table — the DeepSeek convert.py table */
static const float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

/* DeepSeek dequant of logical element (r,k) from RAW staged expert bytes:
 * sequential nibbles, e2m1 table, scale = 2^(e8m0-127). cols = logical K. */
static float deepseek_val(const uint8_t *w, const uint8_t *s, int r, int k, int cols) {
    size_t rb = (size_t)cols / 2, sb = (size_t)cols / 32;
    uint8_t byte = w[r * rb + (size_t)(k >> 1)];
    uint8_t nib  = (k & 1) ? (byte >> 4) : (byte & 0xf);
    uint8_t e    = s[r * sb + (size_t)(k >> 5)];
    return FP4_TABLE[nib] * ggml_e8m0_to_fp32(e);
}

/* kernel-convention dequant of (r,k) from the LOADED arena bytes:
 * within a 32-block, element p=k&31 lives in byte j=p&15 (low if p<16, high if
 * p>=16); 2x table {0,1,2,...}; scale = 2^(e8m0-127). */
static float kernel_val(const uint8_t *w, const uint8_t *s, int r, int k, int cols) {
    size_t rb = (size_t)cols / 2, sb = (size_t)cols / 32;
    int b = k >> 5, p = k & 31, j = p & 15;
    uint8_t byte = w[r * rb + (size_t)b * 16 + (size_t)j];
    uint8_t nib  = (p >= 16) ? (byte >> 4) : (byte & 0xf);
    uint8_t e    = s[r * sb + (size_t)b];
    return ds4f_kvalues_mxfp4_f32[nib] * ggml_e8m0_to_fp32(e);
}

int main(void) {
    ds4f_config c = ds4f_default_config();
    const char *el = getenv("DS4F_TEST_LAYERS"); int nl = (el && *el) ? atoi(el) : 1;
    if (nl < 1) nl = 1; if (nl > c.n_layers) nl = c.n_layers;
    c.n_layers = nl;
    int ep_rank = 0, ep_size = 11;
    const char *dir = getenv("DS4F_STAGE_DIR"); if (!dir || !*dir) dir = "/local/ds4f";

    printf("=== ds4f_load_test: %d layer(s), rank %d/%d, dir %s ===\n", nl, ep_rank, ep_size, dir);
    ds4f_model *m = ds4f_load_real(c, ep_rank, ep_size, dir, 12, 4);
    if (!m) { fprintf(stderr, "FAIL: ds4f_load_real returned NULL\n"); return 1; }

    int fails = 0;

    /* (1) out_norm + a couple of norms finite */
    {
        int bad = 0;
        for (int i = 0; i < c.hidden; i++) if (!isfinite(ds4f_bf16(m->out_norm[i]))) bad++;
        for (int i = 0; i < c.q_lora; i++) if (!isfinite(ds4f_bf16(m->layers[0].q_norm[i]))) bad++;
        printf("[norms]   out_norm+q_norm non-finite: %d  %s\n", bad, bad ? "FAIL" : "ok");
        fails += !!bad;
    }

    /* (2) FP8 dense wq_a row 0: dequant via the e4m3fn LUT, all finite, |v|<=448 */
    {
        ds4f_tensor *t = &m->layers[0].wq_a;
        const uint8_t *w = (const uint8_t *)t->w; int K = t->cols;
        int sbc = (K + 127) / 128, bad = 0; float mx = 0.f;
        for (int k = 0; k < K; k++) {
            uint32_t bits = m->fp8_lut[w[k]];           /* row 0 */
            float v; memcpy(&v, &bits, 4);
            float s = ggml_e8m0_to_fp32(t->scale[(k >> 7)]); /* row-block 0, col-block k/128 */
            (void)sbc; float dv = v * s;
            if (!isfinite(dv)) bad++;
            if (fabsf(v) > mx) mx = fabsf(v);
        }
        printf("[fp8]     wq_a row0: non-finite %d, max|e4m3fn| %.1f (<=448)  %s\n",
               bad, mx, (bad || mx > 448.001f) ? "FAIL" : "ok");
        fails += (bad || mx > 448.001f);
    }

    /* (3) MXFP4 expert-0 w1: prove the loaded arena == DeepSeek dequant of raw blob */
    {
        ds4f_blob B;
        if (ds4f_blob_open(&B, dir, ep_rank) != 0) { fprintf(stderr, "FAIL: reopen blob\n"); return 1; }
        const ds4f_mani_ent *we = ds4f_mani_find(&B, "layers.0.ffn.experts.0.w1.weight");
        const ds4f_mani_ent *se = ds4f_mani_find(&B, "layers.0.ffn.experts.0.w1.scale");
        if (!we || !se) { fprintf(stderr, "FAIL: expert0 w1 not in manifest\n"); ds4f_blob_close(&B); return 1; }
        const uint8_t *raw_w = B.blob + we->off, *raw_s = B.blob + se->off;
        ds4f_tensor *t = &m->layers[0].ex_w1[0];
        const uint8_t *ar_w = (const uint8_t *)t->w, *ar_s = (const uint8_t *)t->scale;
        int rows = t->rows, cols = t->cols;        /* logical [2048, 4096] */
        double maxerr = 0.0; int bad = 0, checked = 0;
        for (int r = 0; r < rows; r += 257) {       /* stride to sample ~8 rows */
            for (int k = 0; k < cols; k += 37) {    /* sample columns across all 32-blocks */
                float vd = deepseek_val(raw_w, raw_s, r, k, cols);
                float vk = kernel_val(ar_w, ar_s, r, k, cols);
                double err = fabs((double)vd - (double)vk);
                if (err > maxerr) maxerr = err;
                if (err > 1e-6) bad++;
                checked++;
            }
        }
        printf("[mxfp4]   expert0 w1: checked %d, repack max|deepseek-kernel| %.3e, mism %d  %s\n",
               checked, maxerr, bad, bad ? "FAIL" : "ok");
        fails += !!bad;
        ds4f_blob_close(&B);
    }

    /* (4) hc + sink finite */
    {
        int bad = 0, mix = (2 + c.hc_mult) * c.hc_mult;
        for (int i = 0; i < mix; i++) if (!isfinite(m->layers[0].hc_attn_base[i])) bad++;
        for (int i = 0; i < c.n_heads; i++) if (!isfinite(m->layers[0].attn_sink[i])) bad++;
        printf("[hc/sink] non-finite: %d  %s\n", bad, bad ? "FAIL" : "ok");
        fails += !!bad;
    }

    printf("=== %s ===\n", fails ? "FAIL" : "PASS");
    ds4f_free(m);
    return fails ? 1 : 0;
}
