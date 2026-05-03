/*
 * verify_decoder_layer — diff each TransformerDecoderLayer's forward
 * against the reference dumps in REFDIR. Runs the C layer forward
 * with the reference (x_in, context_in, x_pe_in, context_pe_in) for
 * each layer and compares the output x to decoder_layer{i}_out__tokens.
 *
 * Feeding the ref inputs at each layer isolates per-layer numerical
 * drift — the upstream PromptableDecoder runs an inter-layer
 * keypoint_token_update_fn between layers that modifies x/x_pe for
 * the next layer's input (step 4f), so verifying end-to-end would
 * conflate layer math with that update path.
 *
 * Usage:
 *   verify_decoder_layer --safetensors-dir <dir> --refdir <dir>
 *                        [--threshold F] [--layer N] [-t N] [-v]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static float *load_or_die(const char *refdir, const char *name, int want_ndim,
                          const int *want_dims)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd, dims[8];
    float *d = (float *)npy_load(path, &nd, dims, NULL);
    if (!d) {
        fprintf(stderr, "[verify_decoder_layer] missing %s\n", path);
        return NULL;
    }
    if (nd != want_ndim) {
        fprintf(stderr, "[verify_decoder_layer] %s: rank=%d want %d\n", name, nd, want_ndim);
        free(d); return NULL;
    }
    for (int i = 0; i < want_ndim; i++) {
        if (want_dims[i] > 0 && dims[i] != want_dims[i]) {
            fprintf(stderr, "[verify_decoder_layer] %s: dim[%d]=%d want %d\n",
                    name, i, dims[i], want_dims[i]);
            free(d); return NULL;
        }
    }
    return d;
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void resolve_variant_path(const char *dir, const char *bucket,
                                 const char *tag, char *out, size_t out_sz)
{
    snprintf(out, out_sz, "%s/sam3d_body_%s_%s.safetensors",
             dir, tag, bucket);
    if (file_exists(out)) return;
    snprintf(out, out_sz, "%s/sam3d_body_%s.safetensors", dir, bucket);
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    const char *backbone = "dinov3";
    /* Budget for per-layer diff vs fp32 upstream. Self+cross attn +
     * FFN add up — empirical fp32 floor observed around 5e-4 max_abs
     * on layer 5 (deepest accumulation). Set with 4x headroom. */
    float threshold = 2e-3f;
    int one_layer = -1, n_threads = 1, verbose = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--layer")           && i+1 < argc) one_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            backbone = argv[++i];
            if (strcmp(backbone, "dinov3") && strcmp(backbone, "vith")) {
                fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n",
                        backbone);
                return 2;
            }
        }
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [--layer N] [--backbone dinov3|vith] "
                "[-t N] [-v]\n", argv[0]);
        return 2;
    }
    (void)verbose;

    char path[1024];
    resolve_variant_path(sft_dir, "decoder", backbone, path, sizeof(path));
    char mhr_path[1024];
    resolve_variant_path(sft_dir, "mhr_head", backbone, mhr_path, sizeof(mhr_path));
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(path, mhr_path);
    if (!m) return 5;

    const int N_Q = 145, D = m->dim, DC = m->kv_dim;
    int nd0 = 0, dims0[8] = {0};
    char p0[1024];
    snprintf(p0, sizeof(p0), "%s/decoder_layer0_in__context.npy", refdir);
    void *tmp0 = npy_load(p0, &nd0, dims0, NULL);
    if (!tmp0 || nd0 != 3 || dims0[2] != DC) {
        fprintf(stderr, "[verify_decoder_layer] bad %s\n", p0);
        free(tmp0); sam3d_body_decoder_free(m); return 6;
    }
    const int N_C = dims0[1];
    free(tmp0);
    float *x_out = (float *)malloc((size_t)N_Q * D * sizeof(float));
    if (!x_out) { sam3d_body_decoder_free(m); return 6; }

    int rc_total = 0;
    int lo = (one_layer >= 0) ? one_layer : 0;
    int hi = (one_layer >= 0) ? one_layer + 1 : m->n_layers;

    for (int li = lo; li < hi; li++) {
        char name[64];
        int xdims[3]  = {1, N_Q, D};
        int cdims[3]  = {1, N_C, DC};

        snprintf(name, sizeof(name), "decoder_layer%d_in__x", li);
        float *x_in = load_or_die(refdir, name, 3, xdims);
        snprintf(name, sizeof(name), "decoder_layer%d_in__context", li);
        float *c_in = load_or_die(refdir, name, 3, cdims);
        snprintf(name, sizeof(name), "decoder_layer%d_in__x_pe", li);
        float *xpe  = load_or_die(refdir, name, 3, xdims);
        snprintf(name, sizeof(name), "decoder_layer%d_in__context_pe", li);
        float *cpe  = load_or_die(refdir, name, 3, cdims);
        snprintf(name, sizeof(name), "decoder_layer%d_out__tokens", li);
        float *x_ref = load_or_die(refdir, name, 3, xdims);
        if (!x_in || !c_in || !xpe || !cpe || !x_ref) {
            free(x_in); free(c_in); free(xpe); free(cpe); free(x_ref);
            rc_total = 3; break;
        }

        int rc = sam3d_body_decoder_layer_forward(
            m, li, x_in, c_in, xpe, cpe, N_Q, N_C, n_threads, x_out, NULL);
        if (rc != SAM3D_BODY_DECODER_E_OK) {
            fprintf(stderr, "[verify_decoder_layer] layer %d: forward rc=%d\n", li, rc);
            free(x_in); free(c_in); free(xpe); free(cpe); free(x_ref);
            rc_total = 7; break;
        }

        double sum = 0.0; float mx = 0.0f; size_t mx_i = 0;
        size_t nel = (size_t)N_Q * D;
        for (size_t i = 0; i < nel; i++) {
            float d = fabsf(x_out[i] - x_ref[i]);
            if (d > mx) { mx = d; mx_i = i; }
            sum += d;
        }
        double mean = sum / (double)nel;
        int tok = (int)(mx_i / D);
        int dim = (int)(mx_i % D);
        float mean_gate = threshold * 0.15f;
        int layer_fail = (mx >= threshold || mean >= mean_gate);
        fprintf(stderr, "[verify_decoder_layer] layer %d  max_abs=%.6e (tok=%d dim=%d)  "
                        "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e) %s\n",
                li, mx, tok, dim, mean, threshold, mean_gate,
                layer_fail ? "FAIL" : "OK");
        if (layer_fail) rc_total = 1;

        free(x_in); free(c_in); free(xpe); free(cpe); free(x_ref);
    }

    free(x_out);
    sam3d_body_decoder_free(m);
    return rc_total;
}
