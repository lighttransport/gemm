/* gemma4_stage.c - per-rank SHARDED staging of the single-file Gemma-4 12B BF16 gguf
 * to node-local /local, for multinode (PP layer-split OR TP tensor-slice).
 *
 * PP mode: rank stages only its layer range (+embed/head/globals) as FULL tensors.
 * TP mode: rank stages a ROW or COL slice of every weight (Megatron sharding), so the
 *   forward uses the standard matvec on a dense local tensor (col-slice => partial sums
 *   + allreduce-SUM). The model is ALL BF16 (2 B/elem, no block quant) => element-wise
 *   slicing is exact and dense-packable. See TP_DESIGN.md for the per-tensor table.
 *
 * Memory-safe: streams the source gguf via pread in <=256 MB sub-chunks and
 * posix_fadvise(DONTNEED) BOTH the source range and the dest blob every <=1 GiB written
 * (a plain copy balloons dirty cache -> kswapd thrash -> the node hangs).
 *
 * Output (matches the ds4f_stage manifest format the loader consumes):
 *   <out_dir>/rank<RR>.blob       packed weights, 256 B aligned per tensor
 *   <out_dir>/rank<RR>.manifest   header + one line/tensor: <off> <nbytes> <dtype> <ndims> <dims...> <name>
 *     (TP: <dims...> are the LOCAL sliced dims so the loader builds the right qtensor.)
 *
 * Usage: gemma4_stage <model.gguf> <out_dir> <rank> <nranks> [pp|tp]
 * Env: GEMMA4_STAGE_FLUSH_GB (default 1) -- writeback+drop every N GiB.
 *
 * Build (native A64FX): fcc -Nclang -O2 -D_GNU_SOURCE -I../../common gemma4_stage.c -o gemma4_stage
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
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
static int name_has(const char *name, const char *suffix){
    size_t ln = strlen(name), ls = strlen(suffix);
    return ln >= ls && strcmp(name + ln - ls, suffix) == 0;
}

/* ---- PP ownership ---- rank owns layers [L0,L1); rank 0 owns token_embd; last rank
 * owns output_norm + output (lm_head). Small non-layer globals replicated. */
static int keep_pp(const char *name, long L0, long L1, int rank, int nranks){
    long l = blk_layer(name);
    if (l >= 0) return (l >= L0 && l < L1);
    if (!strcmp(name, "token_embd.weight"))  return rank == 0 || rank == nranks - 1;
    if (!strcmp(name, "output_norm.weight")) return rank == nranks - 1;
    if (!strcmp(name, "output.weight"))      return rank == nranks - 1;  /* untied models */
    return 1;   /* small global -> replicate */
}

/* ---- TP sharding ---- axis: 0=slice cols(dims[0],contraction), 1=slice rows(dims[1],output),
 * -1=replicate (full). dims[0]=n_cols (innermost/contiguous), dims[1]=n_rows. */
enum { TP_REPL = -1, TP_COL = 0, TP_ROW = 1 };
/* split [0,total) into nranks pieces, return [*lo,*hi) for rank (sizes differ by <=1) */
static void split_even(long total, int rank, int nranks, long *lo, long *hi){
    *lo = (long)rank * total / nranks;
    *hi = (long)(rank + 1) * total / nranks;
}
/* return axis for a TP tensor and fill [*lo,*hi) on that axis; n_heads needed for head-split.
 * ROW-slice attn_q + ffn_gate/up; COL-slice attn_output + ffn_down; REPLICATE the rest
 * (attn_k/v, token_embd [embed needs full; lm_head vocab-slices at compute], norms, ple). */
static int classify_tp(const char *name, uint64_t d0, uint64_t d1,
                       int rank, int nranks, int n_heads, long *lo, long *hi){
    if (blk_layer(name) < 0) return TP_REPL;          /* globals (token_embd, norms, rope) */
    if (name_has(name, "attn_q.weight")) {            /* ROW-slice by heads (rows=q_dim) */
        long hd = (long)d1 / n_heads, h0, h1; split_even(n_heads, rank, nranks, &h0, &h1);
        *lo = h0 * hd; *hi = h1 * hd; return TP_ROW;
    }
    if (name_has(name, "attn_output.weight")) {       /* COL-slice by heads (cols=q_dim) */
        long hd = (long)d0 / n_heads, h0, h1; split_even(n_heads, rank, nranks, &h0, &h1);
        *lo = h0 * hd; *hi = h1 * hd; return TP_COL;
    }
    if (name_has(name, "ffn_gate.weight") || name_has(name, "ffn_up.weight")) {
        split_even((long)d1, rank, nranks, lo, hi); return TP_ROW;   /* rows = n_ff */
    }
    if (name_has(name, "ffn_down.weight")) {
        split_even((long)d0, rank, nranks, lo, hi); return TP_COL;   /* cols = n_ff */
    }
    return TP_REPL;   /* attn_k, attn_v, all norms, ple_*, etc. */
}

/* contiguous source byte range -> dest, pread/write <=256MB, fadvise(DONTNEED) source. */
static int write_bytes(int dfd, int sfd, off_t src_off, uint64_t nbytes){
    static char *buf = NULL; static size_t cap = 0;
    size_t chunk = 256u*1024*1024;
    if (cap < chunk) { free(buf); buf = malloc(chunk); cap = buf ? chunk : 0; if (!buf) return -1; }
    uint64_t done = 0;
    while (done < nbytes) {
        size_t n = nbytes - done; if (n > chunk) n = chunk;
        ssize_t r = pread(sfd, buf, n, src_off + (off_t)done);
        if (r <= 0) return -1;
        ssize_t w = 0; while (w < r) { ssize_t x = write(dfd, buf + w, (size_t)(r - w)); if (x <= 0) return -1; w += x; }
        posix_fadvise(sfd, src_off + (off_t)done, (size_t)r, POSIX_FADV_DONTNEED);
        done += (uint64_t)r;
    }
    return 0;
}
/* align dest to ALIGN, return aligned start; caller writes nbytes then bumps off. */
static int dest_align(int dfd, uint64_t *off, uint64_t *aligned){
    uint64_t a = (*off + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    if (a != *off && lseek(dfd, (off_t)a, SEEK_SET) < 0) return -1;
    *aligned = a; return 0;
}
/* FULL tensor copy (PP, or TP replicate). */
static int write_full(int dfd, int sfd, off_t src_off, uint64_t nbytes, uint64_t *off, uint64_t *aligned){
    if (dest_align(dfd, off, aligned) != 0) return -1;
    if (write_bytes(dfd, sfd, src_off, nbytes) != 0) return -1;
    *off = *aligned + nbytes; return 0;
}
/* ROW-slice: rows [lo,hi) of a [n_rows=d1, row_bytes] tensor => contiguous sub-range. */
static int write_rowslice(int dfd, int sfd, off_t src_off, uint64_t row_bytes,
                          long lo, long hi, uint64_t *off, uint64_t *aligned, uint64_t *out_bytes){
    uint64_t nb = (uint64_t)(hi - lo) * row_bytes;
    if (dest_align(dfd, off, aligned) != 0) return -1;
    if (write_bytes(dfd, sfd, src_off + (off_t)((uint64_t)lo * row_bytes), nb) != 0) return -1;
    *off = *aligned + nb; *out_bytes = nb; return 0;
}
/* COL-slice: cols [lo,hi) of every row => strided gather into a dense [n_rows, hi-lo] blob.
 * esz = element bytes (BF16=2). Streams per-row; small per-row reads but memory-flat. */
static int write_colslice(int dfd, int sfd, off_t src_off, uint64_t n_rows, uint64_t n_cols,
                          uint64_t esz, long lo, long hi, uint64_t *off, uint64_t *aligned, uint64_t *out_bytes){
    if (dest_align(dfd, off, aligned) != 0) return -1;
    uint64_t row_bytes = n_cols * esz, loc_bytes = (uint64_t)(hi - lo) * esz;
    /* batch rows through a bounded buffer to amortize syscalls (<=64MB) */
    uint64_t rows_per = (64ull*1024*1024) / (loc_bytes ? loc_bytes : 1); if (rows_per < 1) rows_per = 1;
    static char *sbuf = NULL, *dbuf = NULL; static size_t scap = 0, dcap = 0;
    size_t need_s = (size_t)(rows_per * row_bytes), need_d = (size_t)(rows_per * loc_bytes);
    if (scap < need_s) { free(sbuf); sbuf = malloc(need_s); scap = sbuf ? need_s : 0; if (!sbuf) return -1; }
    if (dcap < need_d) { free(dbuf); dbuf = malloc(need_d); dcap = dbuf ? need_d : 0; if (!dbuf) return -1; }
    uint64_t written = 0;
    for (uint64_t r0 = 0; r0 < n_rows; r0 += rows_per) {
        uint64_t rc = n_rows - r0; if (rc > rows_per) rc = rows_per;
        ssize_t rd = pread(sfd, sbuf, (size_t)(rc * row_bytes), src_off + (off_t)(r0 * row_bytes));
        if (rd < (ssize_t)(rc * row_bytes)) return -1;
        for (uint64_t i = 0; i < rc; i++)
            memcpy(dbuf + i * loc_bytes, sbuf + i * row_bytes + (uint64_t)lo * esz, loc_bytes);
        uint64_t db = rc * loc_bytes; ssize_t w = 0;
        while (w < (ssize_t)db) { ssize_t x = write(dfd, dbuf + w, (size_t)(db - (uint64_t)w)); if (x <= 0) return -1; w += x; }
        posix_fadvise(sfd, src_off + (off_t)(r0 * row_bytes), (size_t)(rc * row_bytes), POSIX_FADV_DONTNEED);
        written += db;
    }
    *off = *aligned + written; *out_bytes = written; return 0;
}

int main(int argc, char **argv){
    if (argc < 5) { fprintf(stderr, "usage: %s <model.gguf> <out_dir> <rank> <nranks> [pp|tp]\n", argv[0]); return 1; }
    const char *gguf_path = argv[1], *out_dir = argv[2];
    int rank = atoi(argv[3]), nranks = atoi(argv[4]);
    const char *mode = argc > 5 ? argv[5] : "pp";
    int is_tp = (strcmp(mode, "tp") == 0);
    if (!is_tp && strcmp(mode, "pp") != 0) { fprintf(stderr, "gemma4_stage: mode must be pp|tp\n"); return 1; }
    if (rank < 0 || rank >= nranks) { fprintf(stderr, "bad rank\n"); return 1; }

    gguf_context *g = gguf_open(gguf_path, 1);   /* mmap = metadata only, data lazy */
    if (!g) { fprintf(stderr, "gemma4_stage: cannot open %s\n", gguf_path); return 1; }
    int ki = gguf_find_key_internal(g, "gemma4.block_count");
    if (ki < 0) { fprintf(stderr, "gemma4_stage: no gemma4.block_count in metadata\n"); return 1; }
    int n_layers = (int)g->kv[ki].value.u32;
    int n_heads = 16;   /* gemma4-12b */
    int khi = gguf_find_key_internal(g, "gemma4.attention.head_count");
    if (khi >= 0) n_heads = (int)g->kv[khi].value.u32;
    long L0 = (long)rank * n_layers / nranks, L1 = (long)(rank + 1) * n_layers / nranks;
    if (is_tp) fprintf(stderr, "gemma4_stage rank %d/%d: TP slice, %d layers, n_heads=%d\n", rank, nranks, n_layers, n_heads);
    else       fprintf(stderr, "gemma4_stage rank %d/%d: PP, %d layers, owns [%ld,%ld)\n", rank, nranks, n_layers, L0, L1);

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
    fprintf(mf, "# GEMMA4MANIFEST rank=%02d nranks=%02d mode=%s n_tensors=%-12d blob_bytes=%-18lld\n", rank, nranks, mode, 0, 0LL);

    uint64_t flush_bytes = (uint64_t)envi("GEMMA4_STAGE_FLUSH_GB", 1) * 1024ull*1024*1024;
    uint64_t off = 0, last_sync = 0; long long n_kept = 0; double t0 = now_sec();

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *name = gguf_tensor_name(g, (int)i);
        if (!name) continue;
        if (!is_tp && !keep_pp(name, L0, L1, rank, nranks)) continue;
        gguf_tensor_info *t = &g->tensors[i];
        size_t nb = gguf_tensor_size(g, (int)i);
        off_t src_off = (off_t)(g->data_offset + t->offset);
        uint64_t aligned = 0, out_bytes = nb;
        uint64_t d0 = t->n_dims > 0 ? t->dims[0] : 1, d1 = t->n_dims > 1 ? t->dims[1] : 1;
        int axis = TP_REPL; long lo = 0, hi = 0;
        if (is_tp) axis = classify_tp(name, d0, d1, rank, nranks, n_heads, &lo, &hi);

        uint64_t out_d0 = d0, out_d1 = d1;   /* local dims written to manifest */
        if (axis == TP_ROW) {                /* slice rows (dims[1]); row_bytes = d0*esz */
            uint64_t row_bytes = nb / d1;
            if (write_rowslice(bfd, sfd, src_off, row_bytes, lo, hi, &off, &aligned, &out_bytes) != 0) goto wfail;
            out_d1 = (uint64_t)(hi - lo);
        } else if (axis == TP_COL) {         /* slice cols (dims[0]) */
            uint64_t esz = nb / (d0 * d1);
            if (write_colslice(bfd, sfd, src_off, d1, d0, esz, lo, hi, &off, &aligned, &out_bytes) != 0) goto wfail;
            out_d0 = (uint64_t)(hi - lo);
        } else {                             /* replicate / PP full */
            if (write_full(bfd, sfd, src_off, nb, &off, &aligned) != 0) goto wfail;
        }

        fprintf(mf, "%llu %llu %s %u", (unsigned long long)aligned, (unsigned long long)out_bytes,
                ggml_type_name(t->type), t->n_dims);
        if (t->n_dims > 0) fprintf(mf, " %llu", (unsigned long long)out_d0);
        if (t->n_dims > 1) fprintf(mf, " %llu", (unsigned long long)out_d1);
        for (uint32_t d = 2; d < t->n_dims; d++) fprintf(mf, " %llu", (unsigned long long)t->dims[d]);
        fprintf(mf, " %s\n", name);
        n_kept++;
        if (off - last_sync >= flush_bytes) { fdatasync(bfd); posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED); last_sync = off; }
        continue;
    wfail:
        fprintf(stderr, "write failed on %s: %s\n", name, strerror(errno)); return 2;
    }

    fseek(mf, hdr_pos, SEEK_SET);
    fprintf(mf, "# GEMMA4MANIFEST rank=%02d nranks=%02d mode=%s n_tensors=%-12lld blob_bytes=%-18llu\n", rank, nranks, mode, n_kept, (unsigned long long)off);
    fclose(mf);
    fdatasync(bfd); posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
    if (close(bfd) < 0) { perror("close blob"); return 2; }
    close(sfd);
    double el = now_sec() - t0, gb = off / 1e9;
    fprintf(stderr, "gemma4_stage rank %d: kept %lld tensors, %.2f GB in %.1fs (%.2f GB/s)\n  -> %s\n  -> %s\n",
            rank, n_kept, gb, el, el > 0 ? gb/el : 0, blob_path, mani_path);
    printf("STAGE %s rank=%d nranks=%d kept=%lld blob_GB=%.3f sec=%.1f DONE\n", mode, rank, nranks, n_kept, gb, el);
    return 0;
}
