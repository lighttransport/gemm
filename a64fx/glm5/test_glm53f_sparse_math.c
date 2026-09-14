/* Native A64FX checks for exact top-k and dimension-parallel sparse MLA. */
#define GLM53F_SPARSE_NO_MAIN
#include "glm53f_sparse_layer_12n.c"

static uint32_t rng = 17;
static float random_float(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 8) % 20001 - 10000) * 0.0001f;
}

int main(void) {
    enum { COUNT = 10923, HEADS = 6, TOKENS = 2051 };
    cp_candidate *v = malloc(COUNT * sizeof(*v)), *reference = malloc(COUNT * sizeof(*v));
    if (!v || !reference) return 2;
    int failed = 0;
    for (int mode = 0; mode < 4; ++mode) for (int k = 0; k <= 512; k += 1 + k / 7) {
        for (int i = 0; i < COUNT; ++i) {
            float score = mode == 0 ? 0 : mode == 1 ? (float)(i % 13) : random_float();
            if (mode == 3 && i % 19 == 0) score = -INFINITY;
            v[i] = (cp_candidate){score, COUNT - 1 - i};
        }
        memcpy(reference, v, COUNT * sizeof(*v));
        cp_top_exact(v, COUNT, k);
        qsort(reference, COUNT, sizeof(*reference), cp_candidate_cmp);
        if (memcmp(v, reference, k * sizeof(*v))) failed = 1;
    }
    printf("CP_TOP_EXACT %s\n", failed ? "FAIL" : "PASS");
    free(reference); free(v);
    glm53f_sparse_context_12n codec = {0};
    codec.latent_bf16 = 1;
    codec.cp_latent = a256(3 * LAT * sizeof(uint16_t));
    float source[LAT], decoded[LAT];
    int codec_failed = 0;
    for (int slot = 0; slot < 3; ++slot) {
        for (int d = 0; d < LAT; ++d) source[d] = random_float();
        cp_store_latent(&codec, slot, source);
        cp_read_latent(decoded, &codec, slot);
        for (int d = 0; d < LAT; ++d) {
            uint32_t bits = (uint32_t)((uint16_t *)codec.cp_latent)[slot * LAT + d] << 16;
            float expected; memcpy(&expected, &bits, sizeof(expected));
            if (memcmp(&expected, decoded + d, sizeof(expected)) ||
                fabsf(decoded[d] - source[d]) > fabsf(source[d]) / 256.0f + 1e-30f) codec_failed = 1;
        }
    }
    printf("CP_BF16_CODEC %s\n", codec_failed ? "FAIL" : "PASS");
    failed |= codec_failed;
    free(codec.cp_latent);
    float *q = a256((size_t)HEADS * KD * 4), *z = a256((size_t)TOKENS * LAT * 4);
    uint16_t *w = a256((size_t)HEADS * (KD + VD) * LAT * 2);
    float *expected = a256((size_t)HEADS * VD * 4), *actual = a256((size_t)HEADS * VD * 4);
    float *ql = a256((size_t)HEADS * LAT * 4), *log = a256((size_t)HEADS * TOKENS * 4);
    float *va = a256((size_t)HEADS * LAT * 4);
    int selected[TOKENS];
    for (int i = 0; i < HEADS * KD; ++i) q[i] = random_float();
    for (int i = 0; i < TOKENS * LAT; ++i) z[i] = random_float();
    for (int i = 0; i < HEADS * (KD + VD) * LAT; ++i) {
        float x = random_float() * 0.03f; uint32_t bits;
        memcpy(&bits, &x, 4); w[i] = (uint16_t)(bits >> 16);
    }
    for (int i = 0; i < TOKENS; ++i) selected[i] = i;
    const int lengths[] = {1, 7, 128, 2048, 2051};
    for (int nh = 5; nh <= HEADS; ++nh) for (int j = 0; j < 5; ++j) {
        int nt = lengths[j], mismatch = 0;
        double ref_best = 1e30, candidate_best = 1e30;
        for (int rep = -1; rep < 4; ++rep) {
            double begin = omp_get_wtime();
            mla_heads(expected, q, z, w, selected, nt, nh);
            double elapsed = omp_get_wtime() - begin;
            if (rep >= 0 && elapsed < ref_best) ref_best = elapsed;
            begin = omp_get_wtime();
            mla_heads_exact_parallel(actual, q, z, w, nt, nh, ql, log, va);
            elapsed = omp_get_wtime() - begin;
            if (rep >= 0 && elapsed < candidate_best) candidate_best = elapsed;
            mismatch |= memcmp(expected, actual, (size_t)nh * VD * 4) != 0;
        }
        printf("MLA_EXACT heads=%d selected=%d bit_mismatch=%d reference_ms=%.6f candidate_ms=%.6f %s\n",
            nh, nt, mismatch, ref_best * 1e3, candidate_best * 1e3, mismatch ? "FAIL" : "PASS");
        failed |= mismatch;
    }
    float *packed = a256((size_t)TOKENS * LAT * sizeof(float));
    float *part = a256((size_t)HEADS * 8 * LAT * sizeof(float));
    for (int t = 0; t < TOKENS; ++t) {
        selected[t] = (t * 37) % TOKENS;
        memcpy(packed + (size_t)t * LAT, z + (size_t)selected[t] * LAT, LAT * sizeof(float));
    }
    for (int nh = 5; nh <= HEADS; ++nh) for (int j = 0; j < 5; ++j) {
        int nt = lengths[j];
        mla_heads_sharded(expected, q, packed, w, nt, nh, ql, log, part, va);
#pragma omp parallel for schedule(static)
        for (int h = 0; h < nh; ++h)
            mla_one_sharded_indexed(actual + h * VD, q + h * KD, z,
                w + (size_t)h * (KD + VD) * LAT, selected, nt);
        int mismatch = memcmp(expected, actual, (size_t)nh * VD * sizeof(float)) != 0;
        failed |= mismatch;
        printf("MLA_SHARDED_INDEXED heads=%d selected=%d exact=%d\n", nh, nt, !mismatch);
    }
    free(part); free(packed);
    free(va); free(log); free(ql); free(actual); free(expected); free(w); free(z); free(q);
    return failed;
}
