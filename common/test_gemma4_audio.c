/* test_gemma4_audio.c — validate the CPU Conformer audio encoder vs torch oracle.
 *
 *   cc -O2 -o test_gemma4_audio common/test_gemma4_audio.c -lm
 *   G4A_DUMP_TOWER=/tmp/our_tower.bin ./test_gemma4_audio <model_dir> \
 *       /tmp/mel.bin 1499 /tmp/ref_tower.bin /tmp/ref_soft.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#define GEMMA4_AUDIO_IMPLEMENTATION
#include "gemma4_audio_encoder.h"

static float *read_bin(const char *path, size_t n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    float *b = (float *)malloc(n * sizeof(float));
    size_t got = fread(b, sizeof(float), n, fp); fclose(fp);
    if (got != n) { fprintf(stderr, "%s: read %zu want %zu\n", path, got, n); free(b); return NULL; }
    return b;
}
static void cmp(const char *tag, const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0, mae = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
        double d = fabs((double)a[i] - b[i]); mae += d; if (d > mx) mx = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb) + 1e-12);
    double rel = sqrt((na + nb - 2 * dot) / (nb + 1e-12));
    printf("%-12s cos=%.6f rel_L2=%.6f mae=%.6f maxabs=%.6f\n", tag, cos, rel, mae / n, mx);
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <model_dir> <mel.bin> <n_frames> <ref_tower.bin> <ref_soft.bin>\n", argv[0]); return 1; }
    const char *model = argv[1], *melp = argv[2];
    int nf = atoi(argv[3]);
    const char *reftp = argv[4], *refsp = argv[5];

    float *mel = read_bin(melp, (size_t)nf * 128);
    if (!mel) return 1;

    fprintf(stderr, "loading audio_tower from %s ...\n", model);
    g4a_model *m = g4a_load_safetensors(model);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }

    int ntok = 0, dim = 0;
    float *soft = g4a_encode(m, mel, nf, &ntok, &dim);
    fprintf(stderr, "encoded: %d tokens x %d dim\n", ntok, dim);

    float *ref_soft = read_bin(refsp, (size_t)ntok * dim);
    if (ref_soft) cmp("soft", soft, ref_soft, (size_t)ntok * dim);

    const char *dt = getenv("G4A_DUMP_TOWER");
    if (dt) {
        float *our = read_bin(dt, (size_t)ntok * dim);
        float *ref = read_bin(reftp, (size_t)ntok * dim);
        if (our && ref) cmp("tower", our, ref, (size_t)ntok * dim);
        free(our); free(ref);
    }

    free(mel); free(soft); free(ref_soft); g4a_free(m);
    return 0;
}
