/* ds4f_stage.c — shard the DeepSeek-V4-Flash safetensors load and stage each
 * EP rank's slice to that node's local scratch (/local).
 *
 * The 46-shard, ~155 GB model is loaded EP-style: every node holds the
 * REPLICATED dense tensors (attn/MLA, shared expert, router, indexer,
 * compressor, norms, mHC, embed, head) plus only the ROUTED experts it owns
 * (expert e is owned by rank e % ep_size). The MTP layer (mtp.0.*) is skipped.
 *
 * Ownership is decided from the tensor NAME alone, so no index.json is needed:
 *   - "mtp."            prefix          -> SKIP   (multi-token-predict layer)
 *   - "...ffn.experts.E..."             -> KEEP iff E % ep_size == ep_rank
 *     (note ".experts." has a leading dot, so "shared_experts" never matches)
 *   - everything else                   -> KEEP   (replicated dense)
 *
 * Each kept tensor's bytes are copied straight out of the shard's read-only
 * mmap into a packed, 256B-aligned blob on the local disk, plus a text
 * manifest the model loader consumes (tensor name -> dtype/shape/local offset).
 * Copies stream file->file from the mmap, so process RSS stays at a few MB
 * even while moving ~25 GB.
 *
 *   out_dir/rank<rr>.blob       packed weights (256B aligned per tensor)
 *   out_dir/rank<rr>.manifest   header line + one line per tensor:
 *       <local_off> <nbytes> <dtype> <ndims> <d0..dn> <name>
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 \
 *       -D_GNU_SOURCE -I../../common -o build/ds4f_stage ds4f_stage.c
 *
 * Run (single node, rank 0 of 11 -> /local/ds4f):
 *   DS4F_EP_RANK=0 DS4F_EP_SIZE=11 ./build/ds4f_stage
 *
 * Env:
 *   DS4F_MODEL       "ds4f" (default) or "ds4p" (DeepSeek-V4-Pro: flips the
 *                    model-dir/nshards defaults to ~/models/ds4p / 64)
 *   DS4F_MODEL_DIR   model dir (default $HOME/models/<DS4F_MODEL>)
 *   DS4F_STAGE_DIR   output dir (default /local/ds4f, fallback $HOME/tmp/ds4f)
 *   DS4F_EP_RANK     this node's EP rank (default: MPI rank env, else 0)
 *   DS4F_EP_SIZE     number of EP ranks (default 11)
 *   DS4F_NSHARDS     shard count (default 46; 64 for ds4p)
 *   DS4F_STAGE_LAYERS  stage only layers.L.* with L < N (0 = all; embed/head/
 *                    out-norm always staged).  Needed for layer-truncated
 *                    12-node DS4P tests: the full 61-layer per-rank blob
 *                    (~101 GB @11 ranks) exceeds the 87 GiB /local.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define ALIGN 256u

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int envi(const char *k, int def) {
    const char *e = getenv(k);
    return (e && *e) ? atoi(e) : def;
}

/* auto-detect MPI rank from whatever the launcher exports */
static int detect_rank(void) {
    const char *keys[] = { "DS4F_EP_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK",
                           "PMI_RANK", "MV2_COMM_WORLD_RANK", NULL };
    for (int i = 0; keys[i]; i++) {
        const char *e = getenv(keys[i]);
        if (e && *e) return atoi(e);
    }
    return 0;
}

/* parse the integer expert id from "...ffn.experts.<E>...."; -1 if not an
 * expert tensor.  ".experts." (leading dot) excludes "shared_experts". */
static long expert_id(const char *name) {
    const char *p = strstr(name, ".experts.");
    if (!p) return -1;
    p += 9; /* past ".experts." */
    if (*p < '0' || *p > '9') return -1;
    return strtol(p, NULL, 10);
}

enum { CLS_SKIP, CLS_DENSE, CLS_EXPERT };

static int classify(const char *name, int rank, int ep_size) {
    if (strncmp(name, "mtp.", 4) == 0) {                    /* MTP layer: skip unless DS4F_STAGE_MTP */
        static int smtp = -1;
        if (smtp < 0) { const char *e = getenv("DS4F_STAGE_MTP"); smtp = (e && atoi(e)) ? 1 : 0; }
        if (!smtp) return CLS_SKIP;
        /* else fall through: mtp.0.ffn.experts.N EP-sharded, rest dense (like a main layer) */
    }
    if (strncmp(name, "layers.", 7) == 0) {                 /* layer-truncated stage (DS4F_STAGE_LAYERS) */
        static int slay = -1;
        if (slay < 0) slay = envi("DS4F_STAGE_LAYERS", 0);
        if (slay > 0 && strtol(name + 7, NULL, 10) >= slay) return CLS_SKIP;
    }
    long e = expert_id(name);
    if (e >= 0) return (e % ep_size == rank) ? CLS_EXPERT : CLS_SKIP;
    return CLS_DENSE;                                       /* replicated */
}

/* write exactly n bytes (loop over partial writes) */
static int write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n) {
        ssize_t w = write(fd, p, n > (1u << 30) ? (1u << 30) : n);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += w; n -= (size_t)w;
    }
    return 0;
}

int main(void) {
    const char *home = getenv("HOME"); if (!home) home = ".";
    const char *mtag = getenv("DS4F_MODEL");
    int is_pro = (mtag && strcmp(mtag, "ds4p") == 0);
    char model_dir[1024], stage_dir[1024];
    {   const char *e = getenv("DS4F_MODEL_DIR");
        if (e && *e) snprintf(model_dir, sizeof model_dir, "%s", e);
        else snprintf(model_dir, sizeof model_dir, "%s/models/%s", home, is_pro ? "ds4p" : "ds4f"); }
    {   const char *e = getenv("DS4F_STAGE_DIR");
        if (e && *e) snprintf(stage_dir, sizeof stage_dir, "%s", e);
        else {
            /* prefer /local, fall back to $HOME/tmp */
            struct stat sb;
            if (stat("/local", &sb) == 0 && S_ISDIR(sb.st_mode))
                snprintf(stage_dir, sizeof stage_dir, "/local/ds4f");
            else
                snprintf(stage_dir, sizeof stage_dir, "%s/tmp/ds4f", home);
        } }

    int rank    = detect_rank();
    int ep_size = envi("DS4F_EP_SIZE", 11);
    int nshards = envi("DS4F_NSHARDS", is_pro ? 64 : 46);  /* real total (used in filename) */
    int slimit  = envi("DS4F_SHARD_LIMIT", 0);    /* cap iterations for smoke tests; 0 = all */
    int last    = (slimit > 0 && slimit < nshards) ? slimit : nshards;
    /* Bound the HBM /local page cache during staging. The ~22 GB blob's dirty
     * pages otherwise pile up in HBM (the "/local caching" that OOM-segfaulted
     * the 11-node stage). Force writeback + drop cache every DS4F_STAGE_FLUSH_GB
     * (default 2) GB written, and DONTNEED each source shard's clean mmap pages
     * after use -> peak staging HBM ~= flush_gb (dirty) + one shard (clean). */
    int flush_gb = envi("DS4F_STAGE_FLUSH_GB", 2);
    uint64_t flush_bytes = (uint64_t)(flush_gb > 0 ? flush_gb : 2) << 30;
    if (rank < 0 || rank >= ep_size) {
        fprintf(stderr, "ds4f_stage: bad rank %d for ep_size %d\n", rank, ep_size);
        return 2;
    }

    mkdir(stage_dir, 0755); /* ignore EEXIST */

    char blob_path[1100], mani_path[1100];
    snprintf(blob_path, sizeof blob_path, "%s/rank%02d.blob", stage_dir, rank);
    snprintf(mani_path, sizeof mani_path, "%s/rank%02d.manifest", stage_dir, rank);

    int bfd = open(blob_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (bfd < 0) { fprintf(stderr, "ds4f_stage: cannot create %s: %s\n", blob_path, strerror(errno)); return 2; }
    FILE *mf = fopen(mani_path, "w");
    if (!mf) { fprintf(stderr, "ds4f_stage: cannot create %s: %s\n", mani_path, strerror(errno)); close(bfd); return 2; }

    printf("ds4f_stage: rank %d/%d  model=%s  out=%s  shards=%d\n",
           rank, ep_size, model_dir, stage_dir, nshards);
    fflush(stdout);

    /* manifest header is rewritten at the end with the final totals; reserve a
     * fixed-width line so we can seek back and overwrite it in place. */
    long hdr_pos = ftell(mf);
    fprintf(mf, "# DS4FMANIFEST rank=%02d ep_size=%02d n_tensors=%-12d blob_bytes=%-18lld\n",
            rank, ep_size, 0, 0LL);

    uint64_t off = 0;                  /* current (aligned) blob offset */
    uint64_t last_sync = 0;            /* blob bytes already flushed + dropped */
    long long n_dense = 0, n_expert = 0;
    uint64_t b_dense = 0, b_expert = 0;
    double t0 = now_sec();

    for (int s = 1; s <= last; s++) {
        char shard[1200];
        snprintf(shard, sizeof shard, "%s/model-%05d-of-%05d.safetensors", model_dir, s, nshards);
        st_context *st = safetensors_open(shard);
        if (!st) { fprintf(stderr, "ds4f_stage: skip unreadable shard %s\n", shard); continue; }

        int kept = 0;
        for (int i = 0; i < st->n_tensors; i++) {
            const char *name = st->tensors[i].name;
            int cls = classify(name, rank, ep_size);
            if (cls == CLS_SKIP) continue;

            size_t nb = st->tensors[i].nbytes;
            /* align the destination offset to ALIGN (sparse seek over the gap) */
            uint64_t aligned = (off + (ALIGN - 1)) & ~(uint64_t)(ALIGN - 1);
            if (aligned != off) {
                if (lseek(bfd, (off_t)aligned, SEEK_SET) < 0) {
                    fprintf(stderr, "ds4f_stage: lseek failed: %s\n", strerror(errno));
                    goto fail;
                }
            }
            if (write_all(bfd, safetensors_data(st, i), nb) != 0) {
                fprintf(stderr, "ds4f_stage: write failed on %s: %s\n", name, strerror(errno));
                goto fail;
            }

            /* manifest line: off nbytes dtype ndims shape... name */
            st_tensor_info *t = &st->tensors[i];
            fprintf(mf, "%llu %zu %s %d", (unsigned long long)aligned, nb, t->dtype_str, t->n_dims);
            for (int d = 0; d < t->n_dims; d++) fprintf(mf, " %llu", (unsigned long long)t->shape[d]);
            fprintf(mf, " %s\n", name);

            off = aligned + nb;
            if (cls == CLS_EXPERT) { n_expert++; b_expert += nb; }
            else                   { n_dense++;  b_dense  += nb; }
            kept++;

            /* keep the blob's dirty page cache bounded in HBM */
            if (off - last_sync >= flush_bytes) {
                fdatasync(bfd);
                posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
                last_sync = off;
            }
        }
        /* drop this shard's source pages (clean, read-only) before the next */
        madvise(st->map_base, st->map_size, MADV_DONTNEED);
        safetensors_close(st);
        double el = now_sec() - t0;
        double gb = (b_dense + b_expert) / 1e9;
        printf("  shard %2d/%d  kept %4d  cum %5.1f GB  %5.1f s  %5.2f GB/s\n",
               s, nshards, kept, gb, el, el > 0 ? gb / el : 0.0);
        fflush(stdout);
    }

    double tel = now_sec() - t0;
    long long n_total = n_dense + n_expert;
    uint64_t b_total = b_dense + b_expert;

    /* rewrite the header line with final totals (fixed width keeps the offset) */
    fseek(mf, hdr_pos, SEEK_SET);
    fprintf(mf, "# DS4FMANIFEST rank=%02d ep_size=%02d n_tensors=%-12lld blob_bytes=%-18llu\n",
            rank, ep_size, n_total, (unsigned long long)b_total);
    fclose(mf);
    fdatasync(bfd);                                  /* flush the tail dirty pages */
    posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);   /* release the blob cache from HBM */
    if (close(bfd) < 0) { fprintf(stderr, "ds4f_stage: close blob failed: %s\n", strerror(errno)); return 2; }

    printf("\nrank %d done: %lld tensors (%lld dense / %lld expert)\n", rank, n_total, n_dense, n_expert);
    printf("  staged %.2f GB (dense %.2f + expert %.2f)  blob_size=%.2f GB\n",
           b_total / 1e9, b_dense / 1e9, b_expert / 1e9, off / 1e9);
    printf("  %.1f s  %.2f GB/s effective\n", tel, tel > 0 ? b_total / 1e9 / tel : 0.0);
    printf("  -> %s\n  -> %s\n", blob_path, mani_path);

    /* per-rank status line on the SHARED FS (mpiexec drops stdout and /local is
     * node-local) so a multinode launcher can confirm every rank finished. */
    {   const char *sdir = getenv("DS4F_STATUS_DIR");
        char sp[1200];
        snprintf(sp, sizeof sp, "%s/ds4f_stage_rank%02d.txt",
                 (sdir && *sdir) ? sdir : ".", rank);
        FILE *sf = fopen(sp, "w");
        if (sf) {
            fprintf(sf, "rank=%02d ep_size=%d tensors=%lld dense=%lld expert=%lld "
                        "staged_GB=%.3f blob_GB=%.3f sec=%.1f GBps=%.3f blob=%s DONE\n",
                    rank, ep_size, n_total, n_dense, n_expert,
                    b_total / 1e9, off / 1e9, tel, tel > 0 ? b_total / 1e9 / tel : 0.0,
                    blob_path);
            fclose(sf);
        }
    }
    return 0;

fail:
    fclose(mf);
    close(bfd);
    return 2;
}
