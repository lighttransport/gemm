/* verify_decoder.c — Standalone decoder verification.
 * Usage: ./verify_decoder <decoder.st> <latent.npy> <ref_occ.npy> */
#include "cuda_trellis2_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t hl;
    if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != (size_t)hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = 0;
    *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp==' '||*sp==',') sp++;
            if (*sp==')') break;
            dims[*ndim]=(int)strtol(sp,&sp,10); (*ndim)++;
            if(*ndim>=8) break;
        }}}
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= dims[i];
    float *data = malloc(n * sizeof(float));
    if (fread(data, sizeof(float), n, f) != n) { free(data); free(hdr); fclose(f); return NULL; }
    fclose(f); free(hdr);
    return data;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <decoder.st> <latent.npy> <ref_occ.npy>\n", argv[0]);
        return 1;
    }

    /* Init CUDA */
    cuda_trellis2_runner *r = cuda_trellis2_init(0, 2);
    if (!r) return 1;

    /* Load only decoder weights */
    if (cuda_trellis2_load_weights(r, NULL, NULL, argv[1]) != 0) {
        cuda_trellis2_free(r); return 1;
    }

    /* Load latent */
    int nd, dd[8];
    float *latent = read_npy_f32(argv[2], &nd, dd);
    if (!latent) { cuda_trellis2_free(r); return 1; }
    fprintf(stderr, "Latent loaded: (");
    for (int i = 0; i < nd; i++) fprintf(stderr, "%s%d", i?",":"", dd[i]);
    fprintf(stderr, ")\n");

    /* Run decoder */
    float *occ = malloc(64 * 64 * 64 * sizeof(float));
    cuda_trellis2_run_decoder(r, latent, occ);

    /* Load reference */
    float *ref = read_npy_f32(argv[3], &nd, dd);

    /* Compare */
    double sum_r = 0, sum_c = 0, sum_r2 = 0, sum_c2 = 0, sum_rc = 0, sum_d2 = 0;
    float min_r = ref[0], max_r = ref[0], min_c = occ[0], max_c = occ[0];
    int N = 64*64*64;
    for (int i = 0; i < N; i++) {
        sum_r += ref[i]; sum_c += occ[i];
        sum_r2 += (double)ref[i]*ref[i]; sum_c2 += (double)occ[i]*occ[i];
        sum_rc += (double)ref[i]*occ[i];
        sum_d2 += (double)(ref[i]-occ[i])*(ref[i]-occ[i]);
        if (ref[i] < min_r) min_r = ref[i];
        if (ref[i] > max_r) max_r = ref[i];
        if (occ[i] < min_c) min_c = occ[i];
        if (occ[i] > max_c) max_c = occ[i];
    }
    double mr = sum_r/N, mc = sum_c/N;
    double corr = (sum_rc/N - mr*mc) / sqrt((sum_r2/N-mr*mr)*(sum_c2/N-mc*mc));
    double rel_l2 = sqrt(sum_d2) / sqrt(sum_r2);

    int occ_r = 0, occ_c = 0;
    for (int i = 0; i < N; i++) { if (ref[i]>0) occ_r++; if (occ[i]>0) occ_c++; }

    fprintf(stderr, "Ref:  range=[%.2f, %.2f], occupied=%d (%.1f%%)\n",
            min_r, max_r, occ_r, 100.0*occ_r/N);
    fprintf(stderr, "CUDA: range=[%.2f, %.2f], occupied=%d (%.1f%%)\n",
            min_c, max_c, occ_c, 100.0*occ_c/N);
    fprintf(stderr, "Correlation: %.8f\n", corr);
    fprintf(stderr, "Rel L2: %.8f\n", rel_l2);
    fprintf(stderr, "Ref[:4]:  %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    fprintf(stderr, "CUDA[:4]: %.4f %.4f %.4f %.4f\n", occ[0], occ[1], occ[2], occ[3]);

    free(latent); free(occ); free(ref);
    cuda_trellis2_free(r);
    return (corr > 0.99) ? 0 : 1;
}
