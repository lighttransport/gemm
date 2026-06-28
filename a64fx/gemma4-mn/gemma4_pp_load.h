/* gemma4_pp_load.h - load ONE pipeline stage's weights from its per-rank blob+manifest.
 *
 * Opens the SHARED gguf for METADATA ONLY (mmap = lazy, no 24GB read), then redirects
 * the tensor data backing to the local rank<RR>.blob: rewrites each owned tensor's
 * offset from the manifest and mangles non-owned tensor names so tf_find_tensor() skips
 * them. With TF_PP_L0/L1 set, transformer_load() loads only this stage's layers (+ the
 * embed/head/global tensors this rank owns) -> per-node ~24/N GB, not 24 GB.
 *
 * Requires transformer.h + gguf_loader.h already included by the TU.
 */
#ifndef GEMMA4_PP_LOAD_H
#define GEMMA4_PP_LOAD_H
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

typedef struct { char name[160]; uint64_t off; } g4pp_ent;

/* returns a model with only layers [*L0,*L1) populated, or NULL on error. */
static transformer_model *gemma4_pp_load(const char *gguf_path, const char *blob_path,
                                         const char *manifest_path, int rank, int nranks,
                                         int max_seq, int *out_L0, int *out_L1) {
    gguf_context *g = gguf_open(gguf_path, 1);          /* metadata (lazy mmap) */
    if (!g) { fprintf(stderr, "gemma4_pp_load: open gguf %s failed\n", gguf_path); return NULL; }
    int ki = gguf_find_key_internal(g, "gemma4.block_count");
    if (ki < 0) { fprintf(stderr, "gemma4_pp_load: no gemma4.block_count\n"); return NULL; }
    int n_layers = (int)g->kv[ki].value.u32;
    int L0 = (int)((long)rank * n_layers / nranks);
    int L1 = (int)((long)(rank + 1) * n_layers / nranks);

    /* mmap the rank's blob (read-only); becomes the tensor data backing */
    int bfd = open(blob_path, O_RDONLY);
    if (bfd < 0) { fprintf(stderr, "gemma4_pp_load: open blob %s: %s\n", blob_path, strerror(errno)); return NULL; }
    struct stat st; if (fstat(bfd, &st) != 0) { close(bfd); return NULL; }
    void *blob = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, bfd, 0);
    if (blob == MAP_FAILED) { fprintf(stderr, "gemma4_pp_load: mmap blob failed\n"); close(bfd); return NULL; }
#ifdef MADV_HUGEPAGE
    madvise(blob, (size_t)st.st_size, MADV_HUGEPAGE);
#endif

    /* parse manifest: name -> blob offset */
    FILE *mf = fopen(manifest_path, "r");
    if (!mf) { fprintf(stderr, "gemma4_pp_load: open manifest %s\n", manifest_path); return NULL; }
    g4pp_ent *ents = (g4pp_ent *)malloc(sizeof(g4pp_ent) * (g->n_tensors + 8));
    int ne = 0; char line[512];
    while (fgets(line, sizeof line, mf)) {
        if (line[0] == '#') continue;
        unsigned long long off; char nm[160] = {0};
        /* line: <off> <nbytes> <dtype> <ndims> <dims...> <name> ; name is the LAST token */
        if (sscanf(line, "%llu", &off) != 1) continue;
        char *last = NULL, *tok = strtok(line, " \t\n");
        while (tok) { last = tok; tok = strtok(NULL, " \t\n"); }
        if (!last) continue;
        snprintf(nm, sizeof nm, "%s", last);
        snprintf(ents[ne].name, sizeof ents[ne].name, "%s", nm);
        ents[ne].off = (uint64_t)off; ne++;
    }
    fclose(mf);

    /* redirect: owned tensor -> blob offset; non-owned -> mangle name so find fails */
    g->data = (uint8_t *)blob;     /* gguf_tensor_data returns g->data + tensor.offset */
    g->data_offset = 0;
    int n_owned = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *name = g->tensors[i].name.str;
        uint64_t boff = 0; int found = 0;
        for (int e = 0; e < ne; e++) if (!strcmp(ents[e].name, name)) { boff = ents[e].off; found = 1; break; }
        if (found) { g->tensors[i].offset = boff; n_owned++; }
        else if (g->tensors[i].name.str) g->tensors[i].name.str[0] = '\0';  /* hide from tf_find_tensor */
    }
    free(ents);
    fprintf(stderr, "gemma4_pp_load rank %d/%d: layers [%d,%d), %d owned tensors, blob %.2f GB\n",
            rank, nranks, L0, L1, n_owned, st.st_size / 1e9);

    char b0[16], b1[16]; snprintf(b0, sizeof b0, "%d", L0); snprintf(b1, sizeof b1, "%d", L1);
    setenv("TF_PP_L0", b0, 1); setenv("TF_PP_L1", b1, 1);
    transformer_model *m = transformer_load(g, max_seq);
    unsetenv("TF_PP_L0"); unsetenv("TF_PP_L1");
    if (!m) { fprintf(stderr, "gemma4_pp_load: transformer_load failed\n"); return NULL; }
    transformer_free_unused_kv(m, L0, L1);
    if (out_L0) *out_L0 = L0;
    if (out_L1) *out_L1 = L1;
    return m;
}
#endif
