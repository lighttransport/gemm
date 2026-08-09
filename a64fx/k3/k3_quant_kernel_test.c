#define _GNU_SOURCE
#include "k3_quant.h"

#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_state = 0x31415926u;
static uint32_t rnd_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static int check_type(const char *name, int type, int cols) {
    const int rows = 17;
    const size_t rb = k3_quant_row_bytes(type, cols);
    uint8_t *w = (uint8_t *)calloc((size_t)rows, rb);
    float *x = (float *)malloc((size_t)cols * sizeof(*x));
    float *ref = (float *)malloc((size_t)rows * sizeof(*ref));
    float *got = (float *)malloc((size_t)rows * sizeof(*got));
    if (!w || !x || !ref || !got) return 2;
    for (int i = 0; i < cols; ++i)
        x[i] = ((float)(rnd_u32() & 0xffffu) / 32768.0f - 1.0f) * 0.2f;
    for (int r = 0; r < rows; ++r) {
        uint8_t *row = w + (size_t)r * rb;
        for (size_t i = 0; i < rb; ++i) row[i] = (uint8_t)rnd_u32();
        /* Random bytes are valid structurally, but keep every FP16 scale
         * finite and positive so the numerical check is meaningful. */
        if (type == K3_Q_Q8_0) {
            for (int b = 0; b < cols / 32; ++b) {
                uint16_t d = 0x2e66;
                memcpy(row + (size_t)b * 34, &d, sizeof d);
            }
        } else {
            for (int b = 0; b < cols / 256; ++b) {
                uint16_t d = 0x2e66;
                memcpy(row + (size_t)b * (rb / (size_t)(cols / 256)),
                       &d, sizeof d);
            }
        }
        ref[r] = k3_quant_dot_row_ref(row, type, x, cols);
    }
    k3_quant_matrix m = {w, type, rows, cols, rb};
    /* The reference matvec must stay bit-faithful to the row reference.  The
     * default entry point now takes an SDOT kernel, so it is held to the
     * activation-quantization tolerance instead. */
    int rc = k3_quant_matvec_ref(got, &m, x, 48);
    double se = 0.0, sr = 0.0, mx = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = (double)got[r] - ref[r];
        se += d * d; sr += (double)ref[r] * ref[r];
        if (fabs(d) > mx) mx = fabs(d);
    }
    double rel = sqrt(se / (sr + 1e-30));
    printf("[%s ref] rows=%d cols=%d rel_l2=%.3e max_abs=%.3e %s\n",
           name, rows, cols, rel, mx, rc == 0 && rel < 1e-6 ? "OK" : "FAIL");
    int bad = rc || rel >= 1e-6;

    /* Default path: W8A8 quantizes the activation to 7 bits + sign, so ~1e-2
     * relative is the floor here regardless of how exact the weights are. */
    int drc = k3_quant_matvec(got, &m, x, 48);
    double dse = 0.0, dsr = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = (double)got[r] - ref[r];
        dse += d * d; dsr += (double)ref[r] * ref[r];
    }
    double drel = sqrt(dse / (dsr + 1e-30));
    printf("[%s default] rel_l2=%.3e %s\n", name, drel,
           drc == 0 && drel < 0.02 ? "OK" : "FAIL");
    bad |= drc || drel >= 0.02;

    if (type == K3_Q_Q8_0 || (type >= K3_Q_IQ1_S && type <= K3_Q_IQ3_XXS)) {
        for (int mode = K3_QUANT_SVE_A16; mode <= K3_QUANT_SVE_Q8; ++mode) {
            int mrc = k3_quant_matvec_mode(got, &m, x, 48, mode);
            double mse = 0.0, mse_ref = 0.0;
            for (int r = 0; r < rows; ++r) {
                double d = (double)got[r] - ref[r];
                mse += d * d; mse_ref += (double)ref[r] * ref[r];
            }
            double mrel = sqrt(mse / (mse_ref + 1e-30));
            printf("[%s %s] rel_l2=%.3e %s\n", name,
                   mode == K3_QUANT_SVE_A16 ? "a16" : "q8", mrel,
                   mrc == 0 && mrel < 0.08 ? "OK" : "FAIL");
            bad |= mrc || mrel >= 0.08;
        }
    }
    if (type == K3_Q_Q8_0) {
        enum { QBATCH = 4 };
        float *qx = malloc((size_t)QBATCH * cols * sizeof(*qx));
        float *qgot = malloc((size_t)QBATCH * rows * sizeof(*qgot));
        float *qone = malloc((size_t)QBATCH * rows * sizeof(*qone));
        k3_quant_workspace qws[QBATCH] = {{0}};
        int qrc = !qx || !qgot || !qone;
        for (int b = 0; b < QBATCH && !qrc; ++b) {
            for (int c = 0; c < cols; ++c)
                qx[(size_t)b * cols + c] = x[c] * (1.0f + 0.125f * b);
            qrc |= k3_quant_workspace_prepare(&qws[b], cols, K3_QUANT_SVE_Q8);
            if (!qrc) {
                k3_quant_prepare_a16(&qws[b], qx + (size_t)b * cols, cols);
                qws[b].scale_a16 = qws[b].scale;
                qws[b].a16_ready = 1;
            }
        }
        if (!qrc) qrc = k3_quant_q8_0_matvec_batch(qgot, rows, &m,
                                                    qws, QBATCH, 48);
        for (int b = 0; b < QBATCH && !qrc; ++b)
            qrc |= k3_quant_matvec_ws(qone + (size_t)b * rows, &m,
                                      qx + (size_t)b * cols, &qws[b],
                                      48, K3_QUANT_SVE_Q8);
        int exact = !qrc && !memcmp(qgot, qone,
                                    sizeof(*qgot) * QBATCH * rows);
        printf("[%s q8-batch4] batch=%d exact=%s %s\n", name, QBATCH,
               exact ? "yes" : "no", exact ? "OK" : "FAIL");
        bad |= !exact;
        for (int b = 0; b < QBATCH; ++b) k3_quant_workspace_free(&qws[b]);
        free(qx); free(qgot); free(qone);
    }
    if (type == K3_Q_IQ1_S || type == K3_Q_IQ2_XS) {
        const int prow = 16;
        uint8_t *pw = calloc((size_t)prow, rb);
        float *pout = malloc((size_t)prow * sizeof(*pout));
        float *pref = malloc((size_t)prow * sizeof(*pref));
        k3_quant_packed packed = {0};
        k3_quant_packed packed4 = {0};
        k3_quant_workspace pws = {0};
        if (!pw || !pout || !pref) {
            bad = 1;
        } else {
            for (int r = 0; r < prow; ++r) {
                uint8_t *row = pw + (size_t)r * rb;
                memcpy(row, w + (size_t)r * rb, rb);
                pref[r] = k3_quant_dot_row_ref(row, type, x, cols);
            }
            if (k3_quant_pack_iq_rows16(&packed,
                    &(k3_quant_matrix){pw, type, prow, cols, rb}) ||
                k3_quant_workspace_prepare(&pws, cols, K3_QUANT_SVE_Q8)) {
                bad = 1;
                goto packed_done;
            }
            k3_quant_prepare_q8(&pws, x, cols);
            int prc = k3_quant_matvec_packed_ws(pout,
                &(k3_quant_matrix){pw, type, prow, cols, rb}, &packed, &pws);
            double pse = 0.0, psr = 0.0;
            double bse = 0.0, bdot = 0.0, bnorm = 0.0;
            double pdot = 0.0, pnorm = 0.0;
            for (int r = 0; r < prow; ++r) {
                double d = (double)pout[r] - pref[r];
                pse += d * d; psr += (double)pref[r] * pref[r];
                double bd = (double)got[r] - pref[r];
                bse += bd * bd;
                bdot += (double)got[r] * pref[r];
                bnorm += (double)got[r] * got[r];
                pdot += (double)pout[r] * pref[r];
                pnorm += (double)pout[r] * pout[r];
            }
            double prel = sqrt(pse / (psr + 1e-30));
            double brel = sqrt(bse / (psr + 1e-30));
            double bcos = bdot / sqrt((bnorm + 1e-30) * (psr + 1e-30));
            double pcos = pdot / sqrt((pnorm + 1e-30) * (psr + 1e-30));
            int quality = prc == 0 && prel <= 1.05 * brel &&
                          pcos >= bcos - 1e-5;
            printf("[%s packed-q8] rel_l2=%.3e baseline=%.3e ratio=%.3f "
                   "cos=%.8f baseline_cos=%.8f %s\n", name, prel, brel,
                   prel / (brel + 1e-30), pcos, bcos,
                   quality ? "OK" : "FAIL");
            bad |= !quality;
            {
                enum { PBATCH = 3 };
                float *bx = malloc((size_t)PBATCH * cols * sizeof(*bx));
                float *bout = malloc((size_t)PBATCH * prow * sizeof(*bout));
                float *bone = malloc((size_t)PBATCH * prow * sizeof(*bone));
                k3_quant_workspace bws[PBATCH] = {{0}};
                int brc = !bx || !bout || !bone;
                for (int b = 0; b < PBATCH && !brc; ++b) {
                    for (int c = 0; c < cols; ++c)
                        bx[(size_t)b * cols + c] = x[c] * (1.0f + 0.125f * b);
                    brc |= k3_quant_workspace_prepare(&bws[b], cols,
                                                       K3_QUANT_SVE_Q8);
                    if (!brc) k3_quant_prepare_q8(&bws[b],
                                                  bx + (size_t)b * cols, cols);
                }
                if (!brc) brc = k3_quant_matvec_packed_batch(
                    bout, prow, &(k3_quant_matrix){pw, type, prow, cols, rb},
                    &packed, bws, PBATCH);
                for (int b = 0; b < PBATCH && !brc; ++b)
                    brc |= k3_quant_matvec_packed_ws(
                        bone + (size_t)b * prow,
                        &(k3_quant_matrix){pw, type, prow, cols, rb},
                        &packed, &bws[b]);
                int exact = !brc && !memcmp(bout, bone, sizeof(*bout) * PBATCH * prow);
                printf("[%s packed-batch] batch=%d exact=%s %s\n", name,
                       PBATCH, exact ? "yes" : "no", exact ? "OK" : "FAIL");
                bad |= !exact;
                for (int b = 0; b < PBATCH; ++b)
                    k3_quant_workspace_free(&bws[b]);
                free(bx); free(bout); free(bone);
            }
            if (type != K3_Q_IQ1_S) {
                /* IQ2_XS grid magnitudes exceed signed-nibble range. */
            } else if (!k3_quant_pack_iq_rows16_nibble(&packed4,
                    &(k3_quant_matrix){pw, type, prow, cols, rb})) {
                int nrc = k3_quant_matvec_packed_ws(pout,
                    &(k3_quant_matrix){pw, type, prow, cols, rb},
                    &packed4, &pws);
                double nse = 0.0, nsr = 0.0;
                for (int r = 0; r < prow; ++r) {
                    double d = (double)pout[r] - pref[r];
                    nse += d * d; nsr += (double)pref[r] * pref[r];
                }
                double nrel = sqrt(nse / (nsr + 1e-30));
                printf("[%s packed-iq4] rel_l2=%.3e %s\n", name, nrel,
                       nrc == 0 && nrel < 0.08 ? "OK" : "FAIL");
                bad |= nrc || nrel >= 0.08;
            } else {
                bad = 1;
            }
        }
packed_done:
        k3_quant_workspace_free(&pws);
        k3_quant_packed_free(&packed);
        k3_quant_packed_free(&packed4);
        free(pw); free(pout); free(pref);
    }
    free(w); free(x); free(ref); free(got);
    return bad;
}

static int check_expert_pool_view(void) {
    const int cols = 256, rows = 16, experts = 2;
    size_t rb = k3_quant_row_bytes(K3_Q_Q8_0, cols);
    uint8_t *w = calloc((size_t)experts * rows, rb);
    float *x = malloc((size_t)cols * sizeof(*x));
    float *got = malloc((size_t)rows * sizeof(*got));
    float *ref = malloc((size_t)rows * sizeof(*ref));
    if (!w || !x || !got || !ref) return 1;
    for (int c = 0; c < cols; ++c) x[c] = (float)((c % 19) - 9) * 0.01f;
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < rows; ++r) {
            uint8_t *row = w + (size_t)(e * rows + r) * rb;
            for (int b = 0; b < cols / 32; ++b) {
                uint16_t d = 0x2e66;
                memcpy(row + (size_t)b * 34, &d, sizeof d);
                for (int j = 0; j < 32; ++j)
                    row[(size_t)b * 34 + 2 + j] = (uint8_t)(e + r + j);
            }
        }
        k3_quant_matrix m = {w + (size_t)e * rows * rb, K3_Q_Q8_0,
                             rows, cols, rb};
        for (int r = 0; r < rows; ++r)
            ref[r] = k3_quant_dot_row_ref(m.data + (size_t)r * rb,
                                          K3_Q_Q8_0, x, cols);
        if (k3_quant_matvec_expert3d(got, w, K3_Q_Q8_0, cols, rows,
                                     experts, e, x, 1, K3_QUANT_REFERENCE)) {
            free(w); free(x); free(got); free(ref); return 1;
        }
        for (int r = 0; r < rows; ++r)
            if (fabsf(got[r] - ref[r]) > 1e-6f) {
                free(w); free(x); free(got); free(ref); return 1;
            }
    }
    free(w); free(x); free(got); free(ref);
    puts("[expert-3d-view] Q8_0 plane selection OK");
    return 0;
}

static int check_expert_tp_slices(void) {
    const int cols = 512, rows = 64, experts = 2;
    const int row_start = 16, row_count = 32;
    const int col_start = 256, col_count = 256;
    size_t rb = k3_quant_row_bytes(K3_Q_Q8_0, cols);
    uint8_t *w = calloc((size_t)experts * rows, rb);
    float *x = malloc((size_t)cols * sizeof(*x));
    float *full = malloc((size_t)rows * sizeof(*full));
    float *part = malloc((size_t)rows * sizeof(*part));
    float *row_part = malloc((size_t)row_count * sizeof(*row_part));
    if (!w || !x || !full || !part || !row_part) return 1;
    for (int c = 0; c < cols; ++c) x[c] = (float)((c % 23) - 11) * 0.013f;
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < rows; ++r) {
            uint8_t *row = w + ((size_t)e * rows + r) * rb;
            for (int b = 0; b < cols / 32; ++b) {
                uint16_t d = 0x2e66;
                memcpy(row + (size_t)b * 34, &d, sizeof d);
                for (int j = 0; j < 32; ++j)
                    row[(size_t)b * 34 + 2 + j] =
                        (uint8_t)(e * 7 + r + j);
            }
        }
    }
    int bad = k3_quant_matvec_expert3d(full, w, K3_Q_Q8_0, cols, rows,
                                       experts, 1, x, 1, K3_QUANT_REFERENCE);
    bad |= k3_quant_matvec_expert3d_rowslice(
        row_part, w, K3_Q_Q8_0, cols, rows, experts, 1,
        row_start, row_count, x, 1, K3_QUANT_REFERENCE);
    for (int r = 0; r < row_count; ++r)
        if (fabsf(row_part[r] - full[row_start + r]) > 1e-6f) bad = 1;

    memset(part, 0, (size_t)rows * sizeof(*part));
    bad |= k3_quant_matvec_expert3d_colslice(
        part, w, K3_Q_Q8_0, cols, rows, experts, 1,
        0, col_start, x, 1, K3_QUANT_REFERENCE);
    bad |= k3_quant_matvec_expert3d_colslice(
        part, w, K3_Q_Q8_0, cols, rows, experts, 1,
        col_start, col_count, x + col_start, 1, K3_QUANT_REFERENCE);
    /* The second call writes rather than accumulates; compute both halves
     * separately and add them to make the ownership reconstruction explicit. */
    float *left = calloc((size_t)rows, sizeof(*left));
    float *right = calloc((size_t)rows, sizeof(*right));
    if (!left || !right) bad = 1;
    else {
        bad |= k3_quant_matvec_expert3d_colslice(left, w, K3_Q_Q8_0,
            cols, rows, experts, 1, 0, col_start, x, 1, K3_QUANT_REFERENCE);
        bad |= k3_quant_matvec_expert3d_colslice(right, w, K3_Q_Q8_0,
            cols, rows, experts, 1, col_start, col_count, x + col_start,
            1, K3_QUANT_REFERENCE);
        for (int r = 0; r < rows; ++r)
            if (fabsf(left[r] + right[r] - full[r]) > 1e-6f) bad = 1;
    }
    free(left); free(right); free(w); free(x); free(full); free(part); free(row_part);
    puts(bad ? "[expert-tp-slices] FAIL" :
         "[expert-tp-slices] row/column reconstruction OK");
    return bad;
}

int main(void) {
    int bad = 0;
    bad |= check_type("Q8_0", K3_Q_Q8_0, 2048);
    bad |= check_type("IQ1_S", K3_Q_IQ1_S, 2048);
    bad |= check_type("IQ2_XS", K3_Q_IQ2_XS, 2048);
    bad |= check_type("IQ2_XXS", K3_Q_IQ2_XXS, 2048);
    bad |= check_type("IQ3_XXS", K3_Q_IQ3_XXS, 2048);
    bad |= check_expert_pool_view();
    bad |= check_expert_tp_slices();
    puts(bad ? "K3 quantized kernel tests: FAIL" :
         "K3 quantized kernel tests: PASS");
    return bad ? 1 : 0;
}
