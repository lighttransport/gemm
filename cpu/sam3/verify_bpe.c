/* Verify CLIP BPE tokenizer produces ids matching HF CLIPTokenizer.
 * Usage: verify_bpe --vocab <vocab.json> --merges <merges.txt>
 *        [--text "cat"] [--refdir /tmp/sam3_ref_cat]
 */
#include "sam3_clip_bpe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_npy(const char *path, int *n, size_t *esz) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl; if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = '\0';
    *esz = strstr(hdr, "<i8") ? 8 : (strstr(hdr, "<f4") ? 4 : 4);
    char *sp = strstr(hdr, "shape"); int dims[4] = {0}; int nd = 0;
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[nd++] = (int)strtol(sp, &sp, 10);
            if (nd >= 4) break;
        } } }
    size_t tot = 1; for (int i = 0; i < nd; i++) tot *= (size_t)dims[i];
    *n = (int)tot;
    void *d = malloc(tot * *esz);
    if (fread(d, *esz, tot, f) != tot) { free(d); fclose(f); free(hdr); return NULL; }
    fclose(f); free(hdr);
    return d;
}

int main(int argc, char **argv) {
    const char *vocab = "/mnt/disk1/models/clip-bpe/vocab.json";
    const char *merges = "/mnt/disk1/models/clip-bpe/merges.txt";
    const char *text = "cat";
    const char *refdir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vocab") && i+1 < argc) vocab = argv[++i];
        else if (!strcmp(argv[i], "--merges") && i+1 < argc) merges = argv[++i];
        else if (!strcmp(argv[i], "--text") && i+1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
    }
    sam3_clip_bpe *t = sam3_clip_bpe_load(vocab, merges);
    if (!t) { fprintf(stderr, "load failed\n"); return 1; }

    int32_t ids[32], mask[32];
    int n = sam3_clip_bpe_encode(t, text, 32, ids, mask);
    fprintf(stderr, "text: %s -> %d valid tokens\n", text, n);
    fprintf(stderr, "ids : ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%d ", ids[i]);
    fprintf(stderr, "\nmask: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%d ", mask[i]);
    fprintf(stderr, "\n");

    if (refdir) {
        char path[512]; int nn; size_t esz;
        snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
        int64_t *rids = (int64_t *)read_npy(path, &nn, &esz);
        snprintf(path, sizeof(path), "%s/input_attention_mask.npy", refdir);
        int64_t *rmask = (int64_t *)read_npy(path, &nn, &esz);
        if (rids && rmask) {
            int ok = 1;
            for (int i = 0; i < 32; i++) {
                if (ids[i] != rids[i] || mask[i] != rmask[i]) {
                    fprintf(stderr, "DIFF@%d ours=(%d,%d) ref=(%lld,%lld)\n",
                            i, ids[i], mask[i],
                            (long long)rids[i], (long long)rmask[i]);
                    ok = 0;
                }
            }
            fprintf(stderr, "match: %s\n", ok ? "OK" : "FAIL");
        }
        free(rids); free(rmask);
    }

    sam3_clip_bpe_free(t);
    return 0;
}
