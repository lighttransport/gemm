/* gemma4_tp_load.h - load this rank's TENSOR-PARALLEL shard from its rank<RR>.blob.
 *
 * Unlike PP (which loads a layer subset of FULL tensors), TP loads EVERY tensor of EVERY
 * layer but each is ROW/COL-sliced (or replicated) per the stager (gemma4_stage.c "tp").
 * The manifest carries each tensor's LOCAL (sliced) dims; we override the gguf tensor dims
 * with them so tf_load_tensor() builds qtensors with local n_rows/n_cols, while the model
 * CONFIG (n_ff, n_embd, n_heads) still comes from the global metadata KEYS (so buffers are
 * full-width and the forward's tp_rank/tp_size logic slices correctly). All BF16 => dense
 * slices => the standard SVE matvec works; row-parallel (o-proj/down) matvecs yield partial
 * sums the forward allreduce-SUMs.
 *
 * Requires transformer.h + gguf_loader.h already included by the TU.
 */
#ifndef GEMMA4_TP_LOAD_H
#define GEMMA4_TP_LOAD_H
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

typedef struct { char name[160]; uint64_t off; uint32_t ndims; uint64_t dims[4]; } g4tp_ent;

/* returns a fully-populated TP-sharded model (call transformer_set_tp after), or NULL. */
static transformer_model *gemma4_tp_load(const char *gguf_path, const char *blob_path,
                                         const char *manifest_path, int rank, int nranks,
                                         int max_seq) {
    gguf_context *g = gguf_open(gguf_path, 1);          /* metadata (lazy mmap) */
    if (!g) { fprintf(stderr, "gemma4_tp_load: open gguf %s failed\n", gguf_path); return NULL; }

    /* ANON-load the blob (NOT mmap): resident + NUMA-interleaved, no per-token re-fault. */
    int bfd = open(blob_path, O_RDONLY);
    if (bfd < 0) { fprintf(stderr, "gemma4_tp_load: open blob %s: %s\n", blob_path, strerror(errno)); return NULL; }
    struct stat st; if (fstat(bfd, &st) != 0) { close(bfd); return NULL; }
    size_t blob_sz = (size_t)st.st_size;
#if defined(__linux__) && defined(SYS_set_mempolicy)
    { unsigned long nodemask = 0xFFUL; syscall(SYS_set_mempolicy, 3 /*MPOL_INTERLEAVE*/, &nodemask, 8UL); }
#endif
    void *blob = NULL;
    if (posix_memalign(&blob, 2*1024*1024, blob_sz) != 0 || !blob) { fprintf(stderr, "gemma4_tp_load: blob alloc %.1f GB failed\n", blob_sz/1e9); close(bfd); return NULL; }
    {   size_t off = 0;
        while (off < blob_sz) {
            size_t chunk = blob_sz - off; if (chunk > 256u*1024*1024) chunk = 256u*1024*1024;
            ssize_t r = pread(bfd, (uint8_t*)blob + off, chunk, (off_t)off);
            if (r <= 0) { fprintf(stderr, "gemma4_tp_load: blob pread failed at %zu\n", off); close(bfd); return NULL; }
#if defined(POSIX_FADV_DONTNEED)
            posix_fadvise(bfd, (off_t)off, (size_t)r, POSIX_FADV_DONTNEED);
#endif
            off += (size_t)r;
        }
    }
    close(bfd);
#if defined(__linux__) && defined(SYS_set_mempolicy)
    syscall(SYS_set_mempolicy, 0 /*MPOL_DEFAULT*/, NULL, 0UL);
#endif

    /* parse manifest: name -> (blob offset, local dims) */
    FILE *mf = fopen(manifest_path, "r");
    if (!mf) { fprintf(stderr, "gemma4_tp_load: open manifest %s\n", manifest_path); return NULL; }
    g4tp_ent *ents = (g4tp_ent *)calloc(g->n_tensors + 8, sizeof(g4tp_ent));
    int ne = 0; char line[512];
    while (fgets(line, sizeof line, mf)) {
        if (line[0] == '#') continue;
        char *tok[32]; int nt = 0;
        for (char *p = strtok(line, " \t\n"); p && nt < 32; p = strtok(NULL, " \t\n")) tok[nt++] = p;
        if (nt < 5) continue;                       /* off nbytes dtype ndims dims... name */
        g4tp_ent *e = &ents[ne];
        e->off   = strtoull(tok[0], NULL, 10);
        e->ndims = (uint32_t)strtoul(tok[3], NULL, 10);
        if (e->ndims > 4) e->ndims = 4;
        for (uint32_t d = 0; d < e->ndims && (4 + (int)d) < nt - 1; d++)
            e->dims[d] = strtoull(tok[4 + d], NULL, 10);
        snprintf(e->name, sizeof e->name, "%s", tok[nt - 1]);   /* name is the LAST token */
        ne++;
    }
    fclose(mf);

    /* redirect every tensor to its blob slice + OVERRIDE dims with the local sliced dims */
    g->data = (uint8_t *)blob; g->data_offset = 0;
    int n_set = 0, n_miss = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *name = g->tensors[i].name.str;
        g4tp_ent *e = NULL;
        for (int k = 0; k < ne; k++) if (!strcmp(ents[k].name, name)) { e = &ents[k]; break; }
        if (!e) { n_miss++; if (g->tensors[i].name.str) g->tensors[i].name.str[0] = '\0'; continue; }
        g->tensors[i].offset = e->off;
        for (uint32_t d = 0; d < e->ndims; d++) g->tensors[i].dims[d] = e->dims[d];  /* LOCAL dims */
        n_set++;
    }
    free(ents);
    fprintf(stderr, "gemma4_tp_load rank %d/%d: %d tensors set, %d hidden, blob %.2f GB\n",
            rank, nranks, n_set, n_miss, blob_sz / 1e9);

    transformer_model *m = transformer_load(g, max_seq);
    if (!m) { fprintf(stderr, "gemma4_tp_load: transformer_load failed\n"); return NULL; }
    return m;   /* caller: transformer_set_tp(m, rank, nranks, cb, &comm) */
}
#endif
