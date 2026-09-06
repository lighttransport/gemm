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
 * The ONE transform applied on the way through: an F32 "*.scale" is folded to the
 * E8M0 byte the loader/kernels expect (see f32_scale_to_e8m0). This is what lets the
 * SAME stager+loader serve both ds4f (Flash: F8_E8M0 scales, MXFP4 experts) and
 * ds4fbase (base: F32 scales, FP8-e4m3 experts) -- see DS4F_MODEL=ds4fbase.
 *
 *   out_dir/rank<rr>.blob       packed weights (256B aligned per tensor)
 *   out_dir/rank<rr>.manifest   header line + one line per tensor:
 *       <local_off> <nbytes> <dtype> <ndims> <d0..dn> <name>
 *
 * DS4F_STAGE_NOCOPY=1 writes the manifest ONLY, with offsets pointing into a
 * virtual concatenation of the ORIGINAL safetensors shards (recorded as
 * "#shard <idx> <vbase> <size> <path>" header lines). The loader reserves one
 * address range and MAP_FIXEDs each shard into it, so the rest of the load path
 * still sees a single flat blob and needs no changes. This exists because the
 * hetero/ds4f host has ~68 GB free disk and the ep_size=1 blob would be ~150 GB.
 * It requires every kept tensor to be usable as-is: no bake overlay, and no
 * F32->E8M0 scale fold (true for ds4f Flash, whose scales already ship F8_E8M0,
 * but NOT for ds4fbase -- the stager refuses no-copy in that case).
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

/* ---- F32 "ue8m0" scale -> E8M0 byte ------------------------------------------------
 * DeepSeek-V4-Flash ships every *.scale as F8_E8M0 (one byte = the biased exponent of a
 * power-of-2 scale). The BASE model (ds4fbase) ships the SAME scales as F32 -- same shape,
 * same values, 4x the bytes. config.json says scale_fmt="ue8m0" for both, and every value
 * checks out as an exact power of two (verified over dense/shared/indexer/expert scales:
 * sign=0, mantissa=0, log2 in [-12,-8]).
 *
 * So we normalize F32 scales to E8M0 HERE, at stage time. That is a lossless exponent
 * bit-extract (byte = the f32's biased-exponent field), and it means the loader and every
 * FP8 kernel keep consuming exactly the E8M0 layout they already do -- no downstream change.
 *
 * Exactness is load-bearing, so a non-pow2 value is a hard error, never a silent round. */
static int f32_scale_to_e8m0(const void *src, size_t nbytes, uint8_t *dst, const char *name) {
    if (nbytes % 4) {
        fprintf(stderr, "ds4f_stage: '%s' F32 scale nbytes %zu not a multiple of 4\n", name, nbytes);
        return -1;
    }
    size_t n = nbytes / 4;
    for (size_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, (const uint8_t *)src + i * 4, 4);
        /* pow2 <=> sign clear AND mantissa clear. (Also rejects 0, inf/NaN-with-mantissa.) */
        if (b & 0x807FFFFFu) {
            float f; memcpy(&f, &b, 4);
            fprintf(stderr, "ds4f_stage: '%s'[%zu] = %.9g (bits %08x) is NOT a power of two -- "
                            "the ue8m0 premise fails, refusing to lossily round to E8M0\n",
                    name, i, (double)f, b);
            return -1;
        }
        dst[i] = (uint8_t)((b >> 23) & 0xFFu);   /* biased exponent == the E8M0 byte */
    }
    return 0;
}

static int is_scale_name(const char *name) {
    size_t n = strlen(name);
    return n >= 6 && strcmp(name + n - 6, ".scale") == 0;
}

/* ---- BAKE OVERLAY (DS4F_BAKE_DIR + DS4F_DENSE=bf16pv|q8pv) --------------------------
 * ds4f_bake.c pre-packs the 8 dominant dense tensors per layer into the kernel-ready
 * BF16_PV / Q8_PV layouts under ~/models/<model>-fast/. When the overlay is on we source
 * those tensors from the baked blob instead of the safetensors, so the runtime mmaps the
 * final layout straight in -- NO bf16 promotion peak at load. Everything else (experts,
 * embed/head, norms, tb2) still comes from the safetensors, unchanged.
 *
 * The baked tensors' ".scale" siblings are DROPPED: Q8_PV carries its scales inline and
 * BF16_PV has none, so the loader never asks for them. Staging them would just waste blob. */
typedef struct { char name[192]; uint64_t off; size_t nbytes; int rows, cols; } bake_ent;
static bake_ent *g_bake = NULL;
static int       g_bake_n = 0;
static uint8_t  *g_bake_blob = MAP_FAILED;
static size_t    g_bake_blob_sz = 0;
static const char *g_bake_dtype = NULL;      /* "BF16_PV" | "Q8_PV" */

static int bake_load(const char *dir, const char *variant) {
    char mp[1200], bp[1200];
    snprintf(mp, sizeof mp, "%s/dense_%s.manifest", dir, variant);
    snprintf(bp, sizeof bp, "%s/dense_%s.blob",     dir, variant);
    FILE *f = fopen(mp, "r");
    if (!f) { fprintf(stderr, "ds4f_stage: cannot open bake manifest %s: %s\n", mp, strerror(errno)); return -1; }
    int cap = 512; g_bake = (bake_ent *)malloc((size_t)cap * sizeof(bake_ent));
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        bake_ent e; char dt[32]; int nd;
        if (sscanf(line, "%llu %zu %31s %d %d %d %191s",
                   (unsigned long long *)&e.off, &e.nbytes, dt, &nd, &e.rows, &e.cols, e.name) != 7) continue;
        if (nd != 2) continue;
        if (!g_bake_dtype) g_bake_dtype = strcmp(dt, "Q8_PV") == 0 ? "Q8_PV" : "BF16_PV";
        if (g_bake_n == cap) { cap *= 2; g_bake = (bake_ent *)realloc(g_bake, (size_t)cap * sizeof(bake_ent)); }
        g_bake[g_bake_n++] = e;
    }
    fclose(f);
    int bfd = open(bp, O_RDONLY);
    if (bfd < 0) { fprintf(stderr, "ds4f_stage: cannot open bake blob %s: %s\n", bp, strerror(errno)); return -1; }
    struct stat sb; if (fstat(bfd, &sb) != 0) { close(bfd); return -1; }
    g_bake_blob_sz = (size_t)sb.st_size;
    g_bake_blob = (uint8_t *)mmap(NULL, g_bake_blob_sz, PROT_READ, MAP_PRIVATE, bfd, 0);
    close(bfd);
    if (g_bake_blob == MAP_FAILED) { fprintf(stderr, "ds4f_stage: mmap bake blob failed: %s\n", strerror(errno)); return -1; }
    fprintf(stderr, "ds4f_stage: bake overlay ON — %d %s tensors from %s (%.2f GB)\n",
            g_bake_n, g_bake_dtype, dir, g_bake_blob_sz / 1e9);
    return 0;
}
static const bake_ent *bake_find(const char *name) {
    for (int i = 0; i < g_bake_n; i++) if (strcmp(g_bake[i].name, name) == 0) return &g_bake[i];
    return NULL;
}
/* is this a ".scale" whose ".weight" sibling is baked? -> drop it */
static int bake_drops_scale(const char *name) {
    if (!g_bake_n) return 0;
    size_t n = strlen(name);
    if (n < 6 || strcmp(name + n - 6, ".scale") != 0) return 0;
    char wn[256];
    snprintf(wn, sizeof wn, "%.*s.weight", (int)(n - 6), name);
    return bake_find(wn) != NULL;
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
    int nocopy = envi("DS4F_STAGE_NOCOPY", 0);
    int flush_gb = envi("DS4F_STAGE_FLUSH_GB", 2);
    uint64_t flush_bytes = (uint64_t)(flush_gb > 0 ? flush_gb : 2) << 30;

    /* bake overlay: DS4F_DENSE=bf16pv|q8pv sources the 8 dense tensors/layer from the baked
     * blob (ds4f_bake.c). Default fp8 = today's behavior, byte-for-byte unchanged. */
    {   const char *dv = getenv("DS4F_DENSE");
        if (dv && *dv && strcmp(dv, "fp8") != 0) {
            if (strcmp(dv, "q8pv") != 0 && strcmp(dv, "bf16pv") != 0) {
                fprintf(stderr, "ds4f_stage: DS4F_DENSE=%s unknown (want fp8|bf16pv|q8pv)\n", dv); return 2; }
            char bd[1024];
            const char *e = getenv("DS4F_BAKE_DIR");
            if (e && *e) snprintf(bd, sizeof bd, "%s", e);
            else snprintf(bd, sizeof bd, "%s/models/%s-fast", home, mtag ? mtag : "ds4f");
            if (bake_load(bd, dv) != 0) return 2;
        } }
    if (rank < 0 || rank >= ep_size) {
        fprintf(stderr, "ds4f_stage: bad rank %d for ep_size %d\n", rank, ep_size);
        return 2;
    }

    mkdir(stage_dir, 0755); /* ignore EEXIST */

    char blob_path[1100], mani_path[1100];
    snprintf(blob_path, sizeof blob_path, "%s/rank%02d.blob", stage_dir, rank);
    snprintf(mani_path, sizeof mani_path, "%s/rank%02d.manifest", stage_dir, rank);

    if (nocopy && g_bake_n) {
        fprintf(stderr, "ds4f_stage: DS4F_STAGE_NOCOPY is incompatible with DS4F_DENSE bake overlay "
                        "(baked tensors have no bytes in the original shards)\n");
        return 2;
    }
    int bfd = nocopy ? -1 : open(blob_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (!nocopy && bfd < 0) { fprintf(stderr, "ds4f_stage: cannot create %s: %s\n", blob_path, strerror(errno)); return 2; }
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

    /* NOCOPY: each shard is mapped whole, page-aligned, into one virtual range;
     * vbase is this shard's start in that range. Tensor offsets are then just
     * vbase + (data section start + in-file tensor offset). */
    const uint64_t PAGE = 4096;
    uint64_t vbase = 0;

    uint64_t off = 0;                  /* current (aligned) blob offset */
    uint64_t last_sync = 0;            /* blob bytes already flushed + dropped */
    long long n_dense = 0, n_expert = 0;
    uint64_t b_dense = 0, b_expert = 0;
    long long n_e8m0 = 0;              /* F32 scales folded to E8M0 (ds4fbase); 0 for ds4f */
    long long n_baked = 0;             /* dense tensors sourced from the bake overlay; 0 when off */
    static uint8_t e8m0_buf[1 << 16];  /* biggest scale is wq_b [256,8] = 2 K elems */
    double t0 = now_sec();

    for (int s = 1; s <= last; s++) {
        char shard[1200];
        snprintf(shard, sizeof shard, "%s/model-%05d-of-%05d.safetensors", model_dir, s, nshards);
        st_context *st = safetensors_open(shard);
        if (!st) { fprintf(stderr, "ds4f_stage: skip unreadable shard %s\n", shard); continue; }
        uint64_t shard_vbase = vbase, shard_dofs = 0;
        if (nocopy) {
            shard_dofs = (uint64_t)((const uint8_t *)st->data - (const uint8_t *)st->map_base);
            fprintf(mf, "#shard %d %llu %llu %s\n", s,
                    (unsigned long long)shard_vbase, (unsigned long long)st->map_size, shard);
            vbase += (st->map_size + PAGE - 1) & ~(PAGE - 1);
        }

        int kept = 0;
        for (int i = 0; i < st->n_tensors; i++) {
            const char *name = st->tensors[i].name;
            int cls = classify(name, rank, ep_size);
            if (cls == CLS_SKIP) continue;

            st_tensor_info *t = &st->tensors[i];
            size_t nb = t->nbytes;
            const void *src = safetensors_data(st, i);
            const char *dtype = t->dtype_str;
            int nd = t->n_dims; uint64_t shp[2];

            /* --- bake overlay: swap in the kernel-ready dense bytes, drop their scales --- */
            if (bake_drops_scale(name)) continue;                  /* inline (Q8) or absent (bf16-pv) */
            const bake_ent *be = g_bake_n ? bake_find(name) : NULL;
            if (be) {
                src = g_bake_blob + be->off; nb = be->nbytes; dtype = g_bake_dtype;
                nd = 2; shp[0] = (uint64_t)be->rows; shp[1] = (uint64_t)be->cols;  /* LOGICAL shape */
                n_baked++;
            }

            /* ds4fbase ships *.scale as F32; fold it to the E8M0 byte the loader expects.
             * (ds4f's scales are already F8_E8M0 -> this never fires, plain byte copy.) */
            if (nocopy && !be && strcmp(dtype, "F32") == 0 && is_scale_name(name)) {
                fprintf(stderr, "ds4f_stage: DS4F_STAGE_NOCOPY cannot serve '%s': its F32 scale needs "
                                "the E8M0 fold, so the bytes must be materialized (use ds4f Flash, "
                                "whose scales already ship F8_E8M0)\n", name);
                goto fail;
            }
            if (!be && strcmp(dtype, "F32") == 0 && is_scale_name(name)) {
                if (nb / 4 > sizeof e8m0_buf) {   /* scales are tiny (<=8 KB); guard anyway */
                    fprintf(stderr, "ds4f_stage: '%s' scale too large (%zu B) for the E8M0 buffer\n", name, nb);
                    goto fail;
                }
                if (f32_scale_to_e8m0(src, nb, e8m0_buf, name) != 0) goto fail;
                src = e8m0_buf; nb /= 4; dtype = "F8_E8M0"; n_e8m0++;
            }

            uint64_t aligned;
            if (nocopy) {
                aligned = shard_vbase + shard_dofs + (uint64_t)t->offset;   /* into the mapped shard */
            } else {
                /* align the destination offset to ALIGN (sparse seek over the gap) */
                aligned = (off + (ALIGN - 1)) & ~(uint64_t)(ALIGN - 1);
                if (aligned != off) {
                    if (lseek(bfd, (off_t)aligned, SEEK_SET) < 0) {
                        fprintf(stderr, "ds4f_stage: lseek failed: %s\n", strerror(errno));
                        goto fail;
                    }
                }
                if (write_all(bfd, src, nb) != 0) {
                    fprintf(stderr, "ds4f_stage: write failed on %s: %s\n", name, strerror(errno));
                    goto fail;
                }
            }

            /* manifest line: off nbytes dtype ndims shape... name
             * (shape is unchanged by the F32->E8M0 fold -- only the element size shrinks; a baked
             *  tensor records its LOGICAL [rows,cols] and lets the dtype imply the packing, so
             *  ds4f_wbytes() reproduces nbytes exactly on the load side) */
            const uint64_t *shape = be ? shp : t->shape;
            fprintf(mf, "%llu %zu %s %d", (unsigned long long)aligned, nb, dtype, nd);
            for (int d = 0; d < nd; d++) fprintf(mf, " %llu", (unsigned long long)shape[d]);
            fprintf(mf, " %s\n", name);

            off = nocopy ? off + nb : aligned + nb;
            if (cls == CLS_EXPERT) { n_expert++; b_expert += nb; }
            else                   { n_dense++;  b_dense  += nb; }
            kept++;

            /* keep the blob's dirty page cache bounded in HBM */
            if (!nocopy && off - last_sync >= flush_bytes) {
                fdatasync(bfd);
                posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
                last_sync = off;
            }
        }
        /* drop this shard's source pages (clean, read-only) before the next */
        if (!nocopy) madvise(st->map_base, st->map_size, MADV_DONTNEED);
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
    if (!nocopy) {
        fdatasync(bfd);                                  /* flush the tail dirty pages */
        posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);   /* release the blob cache from HBM */
        if (close(bfd) < 0) { fprintf(stderr, "ds4f_stage: close blob failed: %s\n", strerror(errno)); return 2; }
    }

    printf("\nrank %d done: %lld tensors (%lld dense / %lld expert)\n", rank, n_total, n_dense, n_expert);
    printf("  F32->E8M0 scales folded: %lld  (0 = ds4f, already E8M0)\n", n_e8m0);
    printf("  dense from BAKE overlay:  %lld %s  (0 = DS4F_DENSE=fp8, no overlay)\n",
           n_baked, g_bake_dtype ? g_bake_dtype : "-");
    if (nocopy)
        printf("  NOCOPY: %.2f GB referenced in place (dense %.2f + expert %.2f); no blob written\n",
               b_total / 1e9, b_dense / 1e9, b_expert / 1e9);
    else
        printf("  staged %.2f GB (dense %.2f + expert %.2f)  blob_size=%.2f GB\n",
               b_total / 1e9, b_dense / 1e9, b_expert / 1e9, off / 1e9);
    printf("  %.1f s  %.2f GB/s effective\n", tel, tel > 0 ? b_total / 1e9 / tel : 0.0);
    if (nocopy) printf("  -> %s (references %s/model-*.safetensors)\n", mani_path, model_dir);
    else        printf("  -> %s\n  -> %s\n", blob_path, mani_path);

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
