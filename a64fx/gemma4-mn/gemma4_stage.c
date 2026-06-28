/* gemma4_stage.c - per-rank SHARDED staging of the single-file Gemma-4 12B BF16 gguf
 * to node-local /local, for multinode (PP first; TP mode later).
 *
 * Each rank stages ONLY the tensors it owns (PP: its layer range + embed/head/globals)
 * into a packed blob + text manifest, so per-node staging is ~24/N GB instead of 24 GB.
 * Memory-safe: streams the source gguf via pread in <=256 MB sub-chunks and
 * posix_fadvise(DONTNEED) BOTH the source range and the dest blob every <=1 GiB written
 * (a plain copy balloons dirty cache -> kswapd thrash -> the node hangs).
 *
 * Output (matches the ds4f_stage manifest format the loader consumes):
 *   <out_dir>/rank<RR>.blob       packed weights, 256 B aligned per tensor
 *   <out_dir>/rank<RR>.manifest   header + one line/tensor: <off> <nbytes> <dtype> <ndims> <dims...> <name>
 *
 * Usage: gemma4_stage <model.gguf> <out_dir> <rank> <nranks> [pp|tp]
 *   (tp mode not yet implemented -> falls back to full-tensor PP keep.)
 * Env: GEMMA4_STAGE_FLUSH_GB (default 1) -- writeback+drop every N GiB.
 *
 * Build (native A64FX): fcc -Nclang -O2 -D_GNU_SOURCE -I../../common gemma4_stage.c -o gemma4_stage
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define ALIGN 256u
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static int envi(const char *k, int d){ const char *e=getenv(k); return e?atoi(e):d; }

/* layer index from "blk.<L>.<...>", or -1 if not a per-layer tensor */
static long blk_layer(const char *name){
    if (strncmp(name, "blk.", 4) != 0) return -1;
    return strtol(name + 4, NULL, 10);
}

/* PP ownership: rank owns layers [L0,L1); rank 0 owns token_embd; last rank owns
 * output_norm + output (lm_head). Small non-layer globals (rope_freqs, etc.) are
 * replicated on every rank (tiny). Returns 1 to keep. */
static int keep_pp(const char *name, long L0, long L1, int rank, int nranks){
    long l = blk_layer(name);
    if (l >= 0) return (l >= L0 && l < L1);
    /* token_embd: rank 0 for the input embedding AND the last rank for the lm_head
     * (gemma4 12B has NO separate output.weight -> the head is TIED to token_embd). */
    if (!strcmp(name, "token_embd.weight"))  return rank == 0 || rank == nranks - 1;
    if (!strcmp(name, "output_norm.weight")) return rank == nranks - 1;
    if (!strcmp(name, "output.weight"))      return rank == nranks - 1;  /* untied models */
    return 1;   /* small global -> replicate */
}

static int write_range(int dfd, int sfd, off_t src_off, uint64_t nbytes, uint64_t *dst_off){
    /* align dest, then pread/write in <=256 MB sub-chunks */
    uint64_t a = (*dst_off + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    if (a != *dst_off && lseek(dfd, (off_t)a, SEEK_SET) < 0) return -1;
    static char *buf = NULL; static size_t cap = 0;
    size_t chunk = 256u*1024*1024;
    if (cap < chunk) { free(buf); buf = malloc(chunk); cap = buf ? chunk : 0; if (!buf) return -1; }
    uint64_t done = 0;
    while (done < nbytes) {
        size_t n = nbytes - done; if (n > chunk) n = chunk;
        ssize_t r = pread(sfd, buf, n, src_off + (off_t)done);
        if (r <= 0) return -1;
        ssize_t w = 0; while (w < r) { ssize_t x = write(dfd, buf + w, (size_t)(r - w)); if (x <= 0) return -1; w += x; }
        posix_fadvise(sfd, src_off + (off_t)done, (size_t)r, POSIX_FADV_DONTNEED);  /* drop source cache */
        done += (uint64_t)r;
    }
    *dst_off = a + nbytes;
    return 0;
}

int main(int argc, char **argv){
    if (argc < 5) { fprintf(stderr, "usage: %s <model.gguf> <out_dir> <rank> <nranks> [pp|tp]\n", argv[0]); return 1; }
    const char *gguf_path = argv[1], *out_dir = argv[2];
    int rank = atoi(argv[3]), nranks = atoi(argv[4]);
    const char *mode = argc > 5 ? argv[5] : "pp";
    if (rank < 0 || rank >= nranks) { fprintf(stderr, "bad rank\n"); return 1; }
    if (strcmp(mode, "pp") != 0) fprintf(stderr, "gemma4_stage: only pp mode implemented; staging full tensors\n");

    gguf_context *g = gguf_open(gguf_path, 1);   /* mmap = metadata only, data lazy */
    if (!g) { fprintf(stderr, "gemma4_stage: cannot open %s\n", gguf_path); return 1; }
    int ki = gguf_find_key_internal(g, "gemma4.block_count");
    if (ki < 0) { fprintf(stderr, "gemma4_stage: no gemma4.block_count in metadata\n"); return 1; }
    int n_layers = (int)g->kv[ki].value.u32;
    /* PP layer split: rank r owns [r*L/N, (r+1)*L/N) */
    long L0 = (long)rank * n_layers / nranks;
    long L1 = (long)(rank + 1) * n_layers / nranks;
    fprintf(stderr, "gemma4_stage rank %d/%d: %d layers, owns [%ld,%ld)\n", rank, nranks, n_layers, L0, L1);

    int sfd = open(gguf_path, O_RDONLY);
    if (sfd < 0) { perror("open src"); return 2; }
    char blob_path[1100], mani_path[1100];
    snprintf(blob_path, sizeof blob_path, "%s/rank%02d.blob", out_dir, rank);
    snprintf(mani_path, sizeof mani_path, "%s/rank%02d.manifest", out_dir, rank);
    int bfd = open(blob_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (bfd < 0) { fprintf(stderr, "cannot create %s: %s\n", blob_path, strerror(errno)); return 2; }
    FILE *mf = fopen(mani_path, "w");
    if (!mf) { fprintf(stderr, "cannot create %s\n", mani_path); return 2; }
    long hdr_pos = ftell(mf);
    fprintf(mf, "# GEMMA4MANIFEST rank=%02d nranks=%02d n_tensors=%-12d blob_bytes=%-18lld\n", rank, nranks, 0, 0LL);

    uint64_t flush_bytes = (uint64_t)envi("GEMMA4_STAGE_FLUSH_GB", 1) * 1024ull*1024*1024;
    uint64_t off = 0, last_sync = 0; long long n_kept = 0; double t0 = now_sec();

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *name = gguf_tensor_name(g, (int)i);
        if (!name || !keep_pp(name, L0, L1, rank, nranks)) continue;
        size_t nb = gguf_tensor_size(g, (int)i);
        off_t src_off = (off_t)(g->data_offset + g->tensors[i].offset);
        uint64_t aligned = (off + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
        if (write_range(bfd, sfd, src_off, nb, &off) != 0) { fprintf(stderr, "write failed on %s: %s\n", name, strerror(errno)); return 2; }
        gguf_tensor_info *t = &g->tensors[i];
        fprintf(mf, "%llu %zu %s %u", (unsigned long long)aligned, nb, ggml_type_name(t->type), t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) fprintf(mf, " %llu", (unsigned long long)t->dims[d]);
        fprintf(mf, " %s\n", name);
        n_kept++;
        if (off - last_sync >= flush_bytes) { fdatasync(bfd); posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED); last_sync = off; }
    }

    fseek(mf, hdr_pos, SEEK_SET);
    fprintf(mf, "# GEMMA4MANIFEST rank=%02d nranks=%02d n_tensors=%-12lld blob_bytes=%-18llu\n", rank, nranks, n_kept, (unsigned long long)off);
    fclose(mf);
    fdatasync(bfd); posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
    if (close(bfd) < 0) { perror("close blob"); return 2; }
    close(sfd);
    double el = now_sec() - t0, gb = off / 1e9;
    fprintf(stderr, "gemma4_stage rank %d: kept %lld tensors, %.2f GB in %.1fs (%.2f GB/s)\n  -> %s\n  -> %s\n",
            rank, n_kept, gb, el, el > 0 ? gb/el : 0, blob_path, mani_path);
    printf("STAGE rank=%d nranks=%d kept=%lld blob_GB=%.3f sec=%.1f DONE\n", rank, nranks, n_kept, gb, el);
    return 0;
}
