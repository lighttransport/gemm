/* verify_dinov3.c — Verify CUDA DINOv3 against PyTorch reference features.
 * Usage: ./verify_dinov3 <dinov3.st> <image_norm.npy> <ref_features.npy> */
#include "cuda_trellis2_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl; fread(&hl, 2, 1, f);
    char *hdr = malloc(hl+1); fread(hdr, 1, hl, f); hdr[hl]=0;
    *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp!=')') { while (*sp==' '||*sp==',') sp++;
            if (*sp==')') break; dims[*ndim]=(int)strtol(sp,&sp,10); (*ndim)++; }}}
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= dims[i];
    float *data = malloc(n * sizeof(float));
    fread(data, sizeof(float), n, f);
    fclose(f); free(hdr);
    return data;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <dinov3.st> <image_norm.npy> <ref_features.npy>\n", argv[0]);
        return 1;
    }

    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;
    if (cuda_trellis2_load_weights(r, argv[1], NULL, NULL) != 0) return 1;

    int nd, dd[8];
    float *image = read_npy_f32(argv[2], &nd, dd);
    float *ref = read_npy_f32(argv[3], &nd, dd);

    float *features = (float *)malloc(1029 * 1024 * sizeof(float));
    cuda_trellis2_run_dinov3(r, image, features);

    /* Compare */
    double sr = 0, sc = 0, sr2 = 0, sc2 = 0, src2 = 0;
    int N = 1029 * 1024;
    for (int i = 0; i < N; i++) {
        sr += ref[i]; sc += features[i];
        sr2 += (double)ref[i]*ref[i]; sc2 += (double)features[i]*features[i];
        src2 += (double)ref[i]*features[i];
    }
    double mr = sr/N, mc = sc/N;
    double corr = (src2/N - mr*mc) / sqrt((sr2/N-mr*mr)*(sc2/N-mc*mc));

    fprintf(stderr, "Ref:  std=%.4f, [:4]=%.4f %.4f %.4f %.4f\n",
            sqrt(sr2/N - mr*mr), ref[0], ref[1], ref[2], ref[3]);
    fprintf(stderr, "CUDA: std=%.4f, [:4]=%.4f %.4f %.4f %.4f\n",
            sqrt(sc2/N - mc*mc), features[0], features[1], features[2], features[3]);
    fprintf(stderr, "Correlation: %.8f\n", corr);

    free(image); free(ref); free(features);
    cuda_trellis2_free(r);
    return (corr > 0.99) ? 0 : 1;
}
