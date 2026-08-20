/* Rank-local BF16 tensor-parallel stage builder for Qwen3.8-27B.
 *
 * The source split GGUF remains on the shared filesystem.  Each MPI rank writes
 * only its final TP tensor slices to its node-local /local filesystem.  The
 * output is consumed by tp_runner through TP_STAGE_DIR and never contains the
 * large token embedding or the optional NextN layer. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "qwen38_tp_stage.h"

static void die(const char *s) { perror(s); exit(1); }
static long env_rank(void) {
    const char *names[] = {"PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        const char *v = getenv(names[i]); if (v && *v) return strtol(v, NULL, 10);
    }
    return -1;
}
static long env_size(void) {
    const char *names[] = {"PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "PJM_MPI_PROC"};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        const char *v = getenv(names[i]); if (v && *v) return strtol(v, NULL, 10);
    }
    return -1;
}
static void tp_range(int n, int parts, int rank, int *lo, int *hi) {
    int base = n / parts, rem = n % parts;
    *lo = rank * base + (rank < rem ? rank : rem);
    *hi = *lo + base + (rank < rem);
}
static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }
static int mkdir_p(const char *path) {
    char tmp[PATH_MAX]; size_t n = strlen(path);
    if (!n || n >= sizeof(tmp)) return -1;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; p++) if (*p == '/') {
        *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/';
    }
    return (mkdir(tmp, 0755) && errno != EEXIST) ? -1 : 0;
}

static int write_all_at(int fd, const void *buf, size_t n, uint64_t off) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n) {
        ssize_t w = pwrite(fd, p, n, (off_t)off);
        if (w <= 0) return -1;
        p += w; n -= (size_t)w; off += (uint64_t)w;
    }
    return 0;
}
static int read_all_at(int fd, void *buf, size_t n, uint64_t off) {
    uint8_t *p = (uint8_t *)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r; off += (uint64_t)r;
    }
    return 0;
}

static int copy_contiguous(int sfd, uint64_t soff, int dfd, uint64_t doff,
                           uint64_t bytes, uint64_t *hash) {
    const size_t cap = 64u * 1024u * 1024u;
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, cap) != 0) return -1;
    uint64_t done = 0;
    while (done < bytes) {
        size_t n = (size_t)(bytes - done); if (n > cap) n = cap;
        if (read_all_at(sfd, buf, n, soff + done) || write_all_at(dfd, buf, n, doff + done)) {
            free(buf); return -1;
        }
        *hash = q38tp_hash_update(*hash, buf, n);
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(sfd, (off_t)(soff + done), n, POSIX_FADV_DONTNEED);
#endif
        done += n;
    }
    free(buf);
    return 0;
}

static int copy_columns(int sfd, uint64_t soff, int dfd, uint64_t doff,
                        uint32_t rows, uint64_t src_rb, uint64_t byte0,
                        uint64_t dst_rb, uint64_t *hash) {
    const size_t cap = 64u * 1024u * 1024u;
    size_t rows_per = src_rb ? cap / (size_t)src_rb : 0; if (!rows_per) rows_per = 1;
    uint8_t *src = NULL, *dst = NULL;
    if (posix_memalign((void **)&src, 4096, rows_per * (size_t)src_rb) != 0) return -1;
    if (posix_memalign((void **)&dst, 4096, rows_per * (size_t)dst_rb) != 0) { free(src); return -1; }
    for (uint32_t r0 = 0; r0 < rows; ) {
        uint32_t nr = rows - r0; if (nr > rows_per) nr = (uint32_t)rows_per;
        size_t sn = (size_t)nr * (size_t)src_rb, dn = (size_t)nr * (size_t)dst_rb;
        if (read_all_at(sfd, src, sn, soff + (uint64_t)r0 * src_rb)) { free(src); free(dst); return -1; }
        for (uint32_t r = 0; r < nr; r++)
            memcpy(dst + (size_t)r * dst_rb, src + (size_t)r * src_rb + byte0, (size_t)dst_rb);
        if (write_all_at(dfd, dst, dn, doff + (uint64_t)r0 * dst_rb)) { free(src); free(dst); return -1; }
        *hash = q38tp_hash_update(*hash, dst, dn);
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(sfd, (off_t)(soff + (uint64_t)r0 * src_rb), sn, POSIX_FADV_DONTNEED);
#endif
        r0 += nr;
    }
    free(src); free(dst); return 0;
}

static int parse_block_name(const char *name, int *layer, const char **suffix) {
    int n = 0;
    if (sscanf(name, "blk.%d.%n", layer, &n) == 1 && n > 0) { *suffix = name + n; return 1; }
    return 0;
}
static int tensor_index(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(gguf_tensor_name(g, (int)i), name)) return (int)i;
    return -1;
}

static int make_entry(const gguf_context *g, int ti, int rank, int size,
                      q38tp_entry *e) {
    const char *name = gguf_tensor_name(g, ti), *suf = NULL;
    int l = -1, kind = 0, r0 = 0, r1 = 0, c0 = 0, c1 = 0, qk = 0;
    const gguf_tensor_info *t = &g->tensors[ti];
    if (t->n_dims != 2 || t->type != GGML_TYPE_BF16) return 0;
    int cols = (int)t->dims[0], rows = (int)t->dims[1];
    if (!strcmp(name, "output.weight")) {
        kind = Q38TP_SLICE_ROWS; tp_range(rows, size, rank, &r0, &r1); c1 = cols;
    } else if (parse_block_name(name, &l, &suf) && l >= 0 && l < 64) {
        int attn = ((l + 1) % 4 == 0);
        if (!strcmp(suf, "ffn_gate.weight") || !strcmp(suf, "ffn_up.weight")) {
            kind = Q38TP_SLICE_ROWS; tp_range(rows, size, rank, &r0, &r1); c1 = cols;
        } else if (!strcmp(suf, "ffn_down.weight")) {
            kind = Q38TP_SLICE_COLS; r1 = rows; tp_range(cols, size, rank, &c0, &c1);
        } else if (attn && (!strcmp(suf, "attn_q.weight") || !strcmp(suf, "attn_k.weight") || !strcmp(suf, "attn_v.weight"))) {
            kind = Q38TP_SLICE_ROWS; tp_range(rows, size, rank, &r0, &r1); c1 = cols;
        } else if (attn && !strcmp(suf, "attn_output.weight")) {
            kind = Q38TP_SLICE_COLS; r1 = rows; tp_range(cols, size, rank, &c0, &c1);
        } else if (!attn && (!strcmp(suf, "attn_gate.weight") || !strcmp(suf, "ssm_alpha.weight") || !strcmp(suf, "ssm_beta.weight"))) {
            kind = Q38TP_SLICE_ROWS; tp_range(rows, size, rank, &r0, &r1); c1 = cols;
        } else if (!attn && !strcmp(suf, "attn_qkv.weight")) {
            kind = Q38TP_SLICE_SSM_ROWS; qk = 4096;
            int v0, v1; tp_range(rows - qk, size, rank, &v0, &v1);
            r0 = qk + v0; r1 = qk + v1; c1 = cols;
        } else if (!attn && !strcmp(suf, "ssm_out.weight")) {
            kind = Q38TP_SLICE_COLS; r1 = rows; tp_range(cols, size, rank, &c0, &c1);
        }
    }
    if (!kind) return 0;
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->type = t->type; e->kind = (uint32_t)kind;
    e->source_rows = (uint32_t)rows; e->source_cols = (uint32_t)cols;
    e->row0 = (uint32_t)r0; e->row1 = (uint32_t)r1;
    e->col0 = (uint32_t)c0; e->col1 = (uint32_t)c1; e->qk_rows = (uint32_t)qk;
    e->local_rows = (uint32_t)(kind == Q38TP_SLICE_SSM_ROWS ? qk + r1 - r0 : r1 - r0);
    e->local_cols = (uint32_t)(kind == Q38TP_SLICE_COLS ? c1 - c0 : cols);
    e->byte_length = (uint64_t)e->local_rows * e->local_cols * 2u;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL-00001-of-N.gguf STAGE_DIR\n", argv[0]); return 2;
    }
    long rank = env_rank(), size = env_size();
    if (rank < 0 || (size != 2 && size != 4) || rank >= size) {
        fprintf(stderr, "qwen38_tp_stage: requires an mpiexec -np 2 or -np 4 launch (rank=%ld size=%ld)\n", rank, size); return 2;
    }
    gguf_context *g = gguf_open_multi(argv[1], 2);
    if (!g) { fprintf(stderr, "qwen38_tp_stage: cannot open %s\n", argv[1]); return 3; }
    q38tp_header *h = (q38tp_header *)calloc(1, sizeof(*h)); if (!h) die("calloc header");
    memcpy(h->magic, Q38TP_MAGIC, 8); h->version = Q38TP_VERSION; h->header_bytes = Q38TP_HEADER_BYTES;
    h->tp_rank = (uint32_t)rank; h->tp_size = (uint32_t)size; h->n_layers = 64; h->n_embd = 5120; h->n_vocab = 248320;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (h->n_entries >= Q38TP_MAX_ENTRIES) { fprintf(stderr, "too many entries\n"); return 4; }
        q38tp_entry e;
        if (make_entry(g, (int)i, (int)rank, (int)size, &e)) h->entries[h->n_entries++] = e;
    }
    if (h->n_entries != 497) {
        fprintf(stderr, "qwen38_tp_stage: expected 497 decode tensors, found %u\n", h->n_entries); return 4;
    }
    if (getenv("Q38TP_PLAN") && atoi(getenv("Q38TP_PLAN"))) {
        uint64_t planned = Q38TP_HEADER_BYTES;
        for (uint32_t i = 0; i < h->n_entries; i++) {
            planned = align_up(planned, 256);
            planned += h->entries[i].byte_length;
        }
        printf("qwen38_tp_stage plan rank=%ld/%ld entries=%u data=%.3fGB file=%.3fGB\n",
               rank, size, h->n_entries, (double)(planned-Q38TP_HEADER_BYTES)/1e9,
               (double)planned/1e9);
        free(h); gguf_close(g); return 0;
    }
    if (mkdir_p(argv[2])) die("mkdir stage");
    char final[PATH_MAX], partial[PATH_MAX];
    snprintf(final, sizeof(final), "%s/rank%02ld.blob", argv[2], rank);
    snprintf(partial, sizeof(partial), "%s.partial.%ld", final, (long)getpid());
    {
        q38tp_header old;
        int in = open(final, O_RDONLY);
        struct stat st;
        uint64_t planned = Q38TP_HEADER_BYTES;
        for (uint32_t i = 0; i < h->n_entries; i++) {
            planned = align_up(planned, 256); planned += h->entries[i].byte_length;
        }
        if (in >= 0 && fstat(in, &st) == 0 && pread(in, &old, sizeof(old), 0) == (ssize_t)sizeof(old) &&
            !memcmp(old.magic, Q38TP_MAGIC, 8) && old.version == Q38TP_VERSION &&
            old.tp_rank == (uint32_t)rank && old.tp_size == (uint32_t)size &&
            old.n_entries == h->n_entries && (uint64_t)st.st_size == planned) {
            close(in);
            printf("[%s] qwen38_tp_stage reuse rank=%ld entries=%u bytes=%llu path=%s\n",
                   getenv("HOSTNAME") ? getenv("HOSTNAME") : "node", rank, old.n_entries,
                   (unsigned long long)old.data_bytes, final);
            free(h); gguf_close(g); return 0;
        }
        if (in >= 0) close(in);
    }
    int out = open(partial, O_CREAT|O_TRUNC|O_RDWR, 0644); if (out < 0) die("open partial");
    uint64_t off = Q38TP_HEADER_BYTES;
    for (uint32_t i = 0; i < h->n_entries; i++) {
        q38tp_entry *e = &h->entries[i];
        int ti = tensor_index(g, e->name); if (ti < 0) { fprintf(stderr, "missing %s\n", e->name); return 5; }
        const gguf_tensor_info *t = &g->tensors[ti];
        uint64_t src_rb = (uint64_t)t->dims[0] * 2u;
        int sfd = g->tensor_fds[ti]; uint64_t soff = g->tensor_file_offsets[ti];
        off = align_up(off, 256); e->file_offset = off; uint64_t hash = 0;
        int rc = 0;
        if (e->kind == Q38TP_SLICE_ROWS) {
            uint64_t n = (uint64_t)(e->row1 - e->row0) * src_rb;
            rc = copy_contiguous(sfd, soff + (uint64_t)e->row0 * src_rb, out, off, n, &hash);
        } else if (e->kind == Q38TP_SLICE_COLS) {
            rc = copy_columns(sfd, soff, out, off, e->source_rows, src_rb,
                              (uint64_t)e->col0 * 2u, (uint64_t)e->local_cols * 2u, &hash);
        } else {
            uint64_t qbytes = (uint64_t)e->qk_rows * src_rb;
            rc = copy_contiguous(sfd, soff, out, off, qbytes, &hash);
            if (!rc) rc = copy_contiguous(sfd, soff + (uint64_t)e->row0 * src_rb, out,
                                          off + qbytes, (uint64_t)(e->row1-e->row0)*src_rb, &hash);
        }
        if (rc) { fprintf(stderr, "qwen38_tp_stage: copy failed %s\n", e->name); return 6; }
        e->checksum = hash; off += e->byte_length; h->source_bytes += gguf_tensor_size(g, ti);
        if ((i & 15u) == 15u) {
            if (fdatasync(out)) die("fdatasync");
#ifdef POSIX_FADV_DONTNEED
            posix_fadvise(out, 0, (off_t)off, POSIX_FADV_DONTNEED);
#endif
        }
        if (rank == 0 && ((i + 1) % 64 == 0 || i + 1 == h->n_entries))
            fprintf(stderr, "qwen38_tp_stage: %u/%u %.3f GB\n", i+1, h->n_entries, (double)(off-Q38TP_HEADER_BYTES)/1e9);
    }
    h->data_bytes = off - Q38TP_HEADER_BYTES;
    h->entries_checksum = q38tp_hash_update(0, h->entries, (size_t)h->n_entries * sizeof(h->entries[0]));
    if (write_all_at(out, h, sizeof(*h), 0) || fdatasync(out)) die("write header");
    close(out);
    if (rename(partial, final)) die("rename stage");
    {
        char manifest[PATH_MAX]; snprintf(manifest, sizeof(manifest), "%s/rank%02ld.manifest", argv[2], rank);
        FILE *mf = fopen(manifest, "w");
        if (mf) {
            fprintf(mf, "format=q38tp-v%u\nrank=%ld\nsize=%ld\nentries=%u\nbytes=%llu\nmodel=%s\n",
                    Q38TP_VERSION, rank, size, h->n_entries,
                    (unsigned long long)h->data_bytes, argv[1]);
            fclose(mf);
        }
    }
    printf("[%s] qwen38_tp_stage rank=%ld entries=%u bytes=%llu path=%s\n",
           getenv("HOSTNAME") ? getenv("HOSTNAME") : "node", rank, h->n_entries,
           (unsigned long long)h->data_bytes, final);
    free(h); gguf_close(g); return 0;
}
