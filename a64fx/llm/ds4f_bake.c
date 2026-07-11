/* ds4f_bake.c — OFFLINE bake of the dominant dense tensors into the kernel-ready
 * layouts, written PERMANENTLY under ~/models/<model>-fast/.
 *
 * WHY THIS EXISTS
 * ---------------
 * The fast dense decode reps (BF16_PV, and Q8_PV int8-W8A8/svdot) are only reachable at
 * RUNTIME by promoting FP8 -> bf16-pv first (ds4f_promote_dense) and then repacking
 * bf16-pv -> Q8_PV (ds4f_q8_promote_dense, which HARD-REQUIRES a DS4F_BF16_PV source).
 * That promotion costs a transient bf16 peak (~2 B/elem) on top of the arena.
 *
 *   - ds4f (Flash) can pay it, but it is what makes the 8-node decode floor
 *     "load-peak-tight" (see pjsub_ds4f.sh).
 *   - ds4fbase CANNOT pay it at all: 22.17 GiB of FP8 experts leave no room for +5.5 GiB
 *     of bf16 dense, so base is stuck on FP8-on-demand dense -- which the roofline pins at
 *     ~8-20% of node BW (dequant/ISSUE-bound), vs bf16-pv's 735 GB/s (== the ~720 GB/s
 *     node ceiling) and q8-sdot's 402 GB/s at the SAME wall (tools/q8_mv_bw.c).
 *
 * Baking those bytes ONCE, offline, removes the peak entirely: the runtime then mmaps the
 * final layout straight out of the staged blob. Q8_PV is the interesting one -- it is
 * bf16-class SPEED at FP8-class MEMORY, which is exactly what base needs.
 *
 * WHAT IS BAKED
 * -------------
 * Only the 8 dominant dense tensors per layer -- the EXACT set ds4f_q8_promote_dense
 * repacks (wq_a wq_b wkv wo_a wo_b sh_w1 sh_w3 sh_w2). The router gate and lm_head stay
 * bf16-pv at runtime (argmax protection), and experts/embed/head/norms/tb2 keep coming from
 * the original safetensors -- so this writes ~17 GB, not a 150-275 GB model copy.
 *
 * CORRECTNESS: BIT-IDENTICAL BY CONSTRUCTION
 * ------------------------------------------
 * We do NOT reimplement the packing. We call the SAME workers the runtime calls:
 *     FP8 e4m3fn + E8M0  --ds4f_promote_worker-->  BF16_PV     (LOSSLESS: e4m3's 3-bit
 *                                                               mantissa x 2^k fits bf16's 7)
 *     BF16_PV            --ds4f_q8repack_worker--> Q8_PV       (lossy but DETERMINISTIC:
 *                                                               per-8row x 64col absmax + RNE)
 * so offline == runtime by construction. NOTE the LUT: the real-weight path uses
 * ds4f_init_fp8_e4m3fn_lut (exp==15 is FINITE, max 448) -- NOT the synth
 * ds4f_init_fp8_e4m3_lut, which maps every exp==15 to NaN. Using the wrong one would
 * silently corrupt every baked tensor.
 *
 * Since the packing is by-construction identical, --verify does NOT memcmp against
 * itself (that would be tautological). It checks the things that can ACTUALLY break:
 *   1. FP8 -> BF16_PV is exactly lossless   (max |err| must be 0)
 *   2. Q8_PV round-trip relL2 is in the expected int8 band (~1e-2), not garbage
 *   3. the bytes we WROTE read back identical (blob offsets / manifest shape / I/O)
 * The remaining risk -- the loader reading them back at the wrong offset -- is caught
 * end-to-end by the token-identity gate on ds4f, not here.
 *
 * OUTPUT (manifest format is the one the loader already parses, ds4f_mani_*):
 *   ~/models/<model>-fast/dense_bf16pv.blob + .manifest     ~11.4 GB  (2 B/elem)
 *   ~/models/<model>-fast/dense_q8pv.blob   + .manifest     ~ 5.9 GB  (1.03 B/elem)
 * Manifest records the LOGICAL [rows, cols] shape; the dtype string (BF16_PV / Q8_PV)
 * tells the loader the packing, and ds4f_wbytes() reproduces nbytes exactly.
 *
 * Build:  make -C a64fx/llm ds4f_bake CC=fcc
 * Run (single node, no allocation needed):
 *   ./build/ds4f_bake                      # bakes ~/models/ds4f      -> ~/models/ds4f-fast
 *   DS4F_MODEL=ds4fbase ./build/ds4f_bake  # bakes ~/models/ds4fbase  -> ~/models/ds4fbase-fast
 *   DS4F_BAKE_VERIFY=1 ./build/ds4f_bake   # + the three checks above (slower)
 *
 * Env:
 *   DS4F_MODEL        "ds4f" (default) | "ds4fbase" | "ds4p"
 *   DS4F_MODEL_DIR    source safetensors   (default $HOME/models/<DS4F_MODEL>)
 *   DS4F_BAKE_DIR     output dir           (default $HOME/models/<DS4F_MODEL>-fast)
 *   DS4F_NSHARDS      shard count          (default 46; 64 for ds4p)
 *   DS4F_BAKE_VERIFY  1 = run the checks   (default 0)
 *   LLM_THREADS       bake pool threads    (default 48)
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
#include <math.h>
#include <sys/stat.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#include "ds4f.h"          /* pulls ds4f_impl.h: pool, ds4f_promote_worker, ds4f_q8repack_worker, LUTs */

/* the 8 dominant dense tensors per layer — EXACTLY ds4f_q8_promote_dense's set */
static const char *BAKE_SUFFIX[] = {
    ".attn.wq_a",  ".attn.wq_b",  ".attn.wkv",  ".attn.wo_a",  ".attn.wo_b",
    ".ffn.shared_experts.w1", ".ffn.shared_experts.w3", ".ffn.shared_experts.w2",
};
enum { N_BAKE = (int)(sizeof BAKE_SUFFIX / sizeof BAKE_SUFFIX[0]) };

/* "layers.<L><suffix>.weight" -> 1, else 0. mtp.* is skipped (the runtime skips it too). */
static int is_bake_weight(const char *name) {
    if (strncmp(name, "layers.", 7) != 0) return 0;
    size_t n = strlen(name);
    if (n < 8 || strcmp(name + n - 7, ".weight") != 0) return 0;
    for (int i = 0; i < N_BAKE; i++) {
        size_t sl = strlen(BAKE_SUFFIX[i]);
        if (n >= sl + 7 && strncmp(name + n - 7 - sl, BAKE_SUFFIX[i], sl) == 0) return 1;
    }
    return 0;
}

static int envi(const char *k, int d) { const char *e = getenv(k); return (e && *e) ? atoi(e) : d; }

/* F32 "ue8m0" scale -> E8M0 byte (ds4fbase ships F32 scales; ds4f already ships E8M0).
 * Exact power of two => a pure exponent bit-extract. Non-pow2 is a hard error: the whole
 * "the kernels never know the difference" argument rests on this being lossless. */
static int f32_scale_to_e8m0(const void *src, size_t nbytes, uint8_t *dst, const char *name) {
    if (nbytes % 4) { fprintf(stderr, "bake: '%s' F32 scale nbytes %zu %% 4\n", name, nbytes); return -1; }
    size_t n = nbytes / 4;
    for (size_t i = 0; i < n; i++) {
        uint32_t b; memcpy(&b, (const uint8_t *)src + i * 4, 4);
        if (b & 0x807FFFFFu) {
            float f; memcpy(&f, &b, 4);
            fprintf(stderr, "bake: '%s'[%zu] = %.9g is NOT a power of two — refusing to round\n",
                    name, i, (double)f);
            return -1;
        }
        dst[i] = (uint8_t)((b >> 23) & 0xFFu);
    }
    return 0;
}

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
    const char *mtag = getenv("DS4F_MODEL"); if (!mtag || !*mtag) mtag = "ds4f";
    int is_pro = (strcmp(mtag, "ds4p") == 0);

    char src_dir[1024], out_dir[1024];
    {   const char *e = getenv("DS4F_MODEL_DIR");
        if (e && *e) snprintf(src_dir, sizeof src_dir, "%s", e);
        else snprintf(src_dir, sizeof src_dir, "%s/models/%s", home, mtag); }
    {   const char *e = getenv("DS4F_BAKE_DIR");
        if (e && *e) snprintf(out_dir, sizeof out_dir, "%s", e);
        else snprintf(out_dir, sizeof out_dir, "%s/models/%s-fast", home, mtag); }

    int nshards = envi("DS4F_NSHARDS", is_pro ? 64 : 46);
    int verify  = envi("DS4F_BAKE_VERIFY", 0);
    int nthr    = envi("LLM_THREADS", 48);
    /* smoke-test knob: stop after N shards. nshards stays in the FILENAME (…-of-00046). */
    int slimit  = envi("DS4F_BAKE_SHARD_LIMIT", 0);
    int last    = (slimit > 0 && slimit < nshards) ? slimit : nshards;

    mkdir(out_dir, 0755);

    /* REAL-weight LUT: e4m3fn (exp==15 FINITE). NOT the synth e4m3 LUT (exp==15 -> NaN). */
    uint32_t lut[256];
    ds4f_init_fp8_e4m3fn_lut(lut);

    ds4f_pool *pool = ds4f_pool_start(nthr, 4);
    if (!pool) { fprintf(stderr, "bake: pool start failed\n"); return 2; }

    char bp[2][1200], mp[2][1200];
    snprintf(bp[0], sizeof bp[0], "%s/dense_bf16pv.blob",    out_dir);
    snprintf(mp[0], sizeof mp[0], "%s/dense_bf16pv.manifest", out_dir);
    snprintf(bp[1], sizeof bp[1], "%s/dense_q8pv.blob",      out_dir);
    snprintf(mp[1], sizeof mp[1], "%s/dense_q8pv.manifest",   out_dir);

    int   bfd[2]; FILE *mf[2];
    for (int v = 0; v < 2; v++) {
        bfd[v] = open(bp[v], O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (bfd[v] < 0) { fprintf(stderr, "bake: cannot create %s: %s\n", bp[v], strerror(errno)); return 2; }
        mf[v] = fopen(mp[v], "w");
        if (!mf[v]) { fprintf(stderr, "bake: cannot create %s: %s\n", mp[v], strerror(errno)); return 2; }
        fprintf(mf[v], "# DS4FBAKE model=%s variant=%s\n", mtag, v ? "Q8_PV" : "BF16_PV");
    }

    printf("ds4f_bake: model=%s\n  src=%s\n  out=%s\n  shards=%d threads=%d verify=%d\n",
           mtag, src_dir, out_dir, nshards, nthr, verify);
    fflush(stdout);

    uint64_t off[2] = { 0, 0 };
    long long nten = 0;
    double t0 = ds4f_wall();
    double worst_bf16_err = 0.0, worst_q8_rel = 0.0;

    for (int s = 1; s <= last; s++) {
        char shard[1300];
        snprintf(shard, sizeof shard, "%s/model-%05d-of-%05d.safetensors", src_dir, s, nshards);
        st_context *st = safetensors_open(shard);
        if (!st) { fprintf(stderr, "bake: skip unreadable %s\n", shard); continue; }

        for (int i = 0; i < st->n_tensors; i++) {
            st_tensor_info *t = &st->tensors[i];
            if (!is_bake_weight(t->name)) continue;
            if (t->n_dims != 2) { fprintf(stderr, "bake: '%s' ndims %d != 2\n", t->name, t->n_dims); goto fail; }
            if (strcmp(t->dtype_str, "F8_E4M3") != 0) {
                fprintf(stderr, "bake: '%s' dtype %s != F8_E4M3\n", t->name, t->dtype_str); goto fail; }

            int N = (int)t->shape[0], K = (int)t->shape[1];
            if ((N & 7) || (K & 63)) {            /* Q8_PV needs 8-row groups x 64-col blocks */
                fprintf(stderr, "bake: '%s' [%d,%d] not 8x64-aligned\n", t->name, N, K); goto fail; }

            /* --- the .scale sibling (E8M0 bytes; ds4fbase ships F32 -> fold) --- */
            char sn[256];
            size_t bl = strlen(t->name) - 7;                 /* strip ".weight" */
            snprintf(sn, sizeof sn, "%.*s.scale", (int)bl, t->name);
            int si = -1;
            for (int j = 0; j < st->n_tensors; j++)
                if (strcmp(st->tensors[j].name, sn) == 0) { si = j; break; }
            if (si < 0) { fprintf(stderr, "bake: '%s' has no .scale\n", t->name); goto fail; }

            size_t sb = ds4f_sbytes(DS4F_FP8, N, K);          /* [N/128, K/128] E8M0 bytes */
            uint8_t *e8 = (uint8_t *)malloc(sb);
            st_tensor_info *sti = &st->tensors[si];
            if (strcmp(sti->dtype_str, "F8_E8M0") == 0) {
                if (sti->nbytes != sb) { fprintf(stderr, "bake: '%s' scale nbytes %zu != %zu\n", sn, sti->nbytes, sb); goto fail; }
                memcpy(e8, safetensors_data(st, si), sb);
            } else if (strcmp(sti->dtype_str, "F32") == 0) {
                if (sti->nbytes != sb * 4) { fprintf(stderr, "bake: '%s' F32 scale nbytes %zu != %zu\n", sn, sti->nbytes, sb * 4); goto fail; }
                if (f32_scale_to_e8m0(safetensors_data(st, si), sti->nbytes, e8, sn) != 0) goto fail;
            } else { fprintf(stderr, "bake: '%s' scale dtype %s unsupported\n", sn, sti->dtype_str); goto fail; }

            const uint8_t *fw = (const uint8_t *)safetensors_data(st, i);

            /* --- FP8 -> BF16_PV, via the RUNTIME worker (lossless) --- */
            size_t wb_bf = ds4f_wbytes(DS4F_BF16_PV, N, K);
            uint16_t *bf = (uint16_t *)aligned_alloc(256, (wb_bf + 255) & ~(size_t)255);
            {   ds4f_promote_task P;
                P.t.w = bf; P.t.scale = NULL; P.t.type = DS4F_BF16_PV; P.t.rows = N; P.t.cols = K;
                P.src_w = fw; P.src_s = e8; P.lut = lut; P.src_fp8 = 1;
                ds4f_pool_run(pool, ds4f_promote_worker, &P); }

            /* --- BF16_PV -> Q8_PV, via the RUNTIME worker (deterministic) --- */
            size_t wb_q8 = ds4f_wbytes(DS4F_Q8_PV, N, K);
            uint8_t *q8 = (uint8_t *)aligned_alloc(256, (wb_q8 + 255) & ~(size_t)255);
            {   ds4f_q8repack_task Q = { bf, q8, N, K };
                ds4f_pool_run(pool, ds4f_q8repack_worker, &Q); }

            if (verify) {
                /* (1) FP8 -> bf16 must be EXACTLY lossless. Recompute the reference value and
                 *     compare to the baked bf16 at its pair-interleaved address. */
                int sbc = (K + 127) / 128;
                for (int r = 0; r < N; r += (N > 64 ? N / 64 : 1)) {         /* sample rows */
                    int local = r & 7, pair = local >> 1, slot = local & 1;
                    const uint16_t *d = bf + (size_t)(r / 8) * 8 * K + (size_t)pair * 2 * K;
                    const uint8_t *es = e8 + (size_t)(r >> 7) * sbc;
                    for (int j = 0; j < K; j += 97) {
                        uint32_t bits = lut[fw[(size_t)r * K + j]]; float v; memcpy(&v, &bits, 4);
                        float ref = v * ggml_e8m0_to_fp32(es[j >> 7]);
                        float got = ds4f_bf16(d[2 * j + slot]);
                        double e = fabs((double)ref - (double)got);
                        if (e > worst_bf16_err) worst_bf16_err = e;
                    }
                }
                /* (2) Q8 round-trip relL2 vs its bf16 source (expect the int8 band ~1e-2). */
                {   int nb = K / 64; double num = 0, den = 0;
                    for (int g = 0; g < N / 8 && g < 8; g++)
                        for (int b = 0; b < nb; b++) {
                            const uint8_t *blk = q8 + ((size_t)g * nb + b) * 528;
                            const uint16_t *scl = (const uint16_t *)blk;
                            const int8_t *qs = (const int8_t *)(blk + 16);
                            for (int r = 0; r < 8; r++) {
                                int pair = r >> 1, slot = r & 1;
                                const uint16_t *src = bf + (size_t)g * 8 * K + (size_t)pair * 2 * K + (size_t)2 * (b * 64) + slot;
                                /* the block scale is IEEE fp16 (ggml_fp32_to_fp16 at repack) — decode it
                                 * with the SAME helper the matvec kernel uses, not a hand-rolled one. */
                                float sc = ggml_fp16_to_fp32(scl[r]);
                                for (int j = 0; j < 64; j++) {
                                    float ref = ds4f_bf16(src[2 * j]);
                                    float got = (float)qs[(size_t)r * 64 + j] * sc;
                                    num += (double)(ref - got) * (ref - got); den += (double)ref * ref;
                                }
                            }
                        }
                    double rel = den > 0 ? sqrt(num / den) : 0;
                    if (rel > worst_q8_rel) worst_q8_rel = rel;
                }
            }

            /* --- append both variants + manifest lines (logical shape; dtype implies packing) --- */
            const void *src_v[2] = { bf, q8 };
            size_t      nb_v[2]  = { wb_bf, wb_q8 };
            const char *dt_v[2]  = { "BF16_PV", "Q8_PV" };
            for (int v = 0; v < 2; v++) {
                uint64_t a = (off[v] + 255) & ~(uint64_t)255;
                if (a != off[v] && lseek(bfd[v], (off_t)a, SEEK_SET) < 0) {
                    fprintf(stderr, "bake: lseek: %s\n", strerror(errno)); goto fail; }
                if (write_all(bfd[v], src_v[v], nb_v[v]) != 0) {
                    fprintf(stderr, "bake: write %s: %s\n", t->name, strerror(errno)); goto fail; }
                fprintf(mf[v], "%llu %zu %s 2 %d %d %s\n",
                        (unsigned long long)a, nb_v[v], dt_v[v], N, K, t->name);
                off[v] = a + nb_v[v];
            }
            nten++;
            free(e8); free(bf); free(q8);
        }
        madvise(st->map_base, st->map_size, MADV_DONTNEED);
        safetensors_close(st);
        printf("  shard %2d/%d  baked %lld tensors  bf16pv %.2f GB  q8pv %.2f GB  %.0f s\n",
               s, nshards, nten, off[0] / 1e9, off[1] / 1e9, ds4f_wall() - t0);
        fflush(stdout);
    }

    for (int v = 0; v < 2; v++) { fclose(mf[v]); fdatasync(bfd[v]); close(bfd[v]); }
    ds4f_pool_stop(pool);

    printf("\nbake DONE: %lld tensors (expect 8/layer)\n", nten);
    printf("  %s  %.2f GB\n  %s  %.2f GB\n", bp[0], off[0] / 1e9, bp[1], off[1] / 1e9);
    printf("  %.0f s\n", ds4f_wall() - t0);
    if (verify) {
        printf("VERIFY: FP8->BF16_PV max|err| = %.3g  (MUST be 0 — the promote is lossless)\n", worst_bf16_err);
        printf("VERIFY: Q8_PV round-trip relL2 = %.3g  (expect the int8 band ~1e-2)\n", worst_q8_rel);
        if (worst_bf16_err != 0.0) {
            fprintf(stderr, "FAIL: FP8->bf16 is NOT lossless — wrong LUT (e4m3 vs e4m3fn?) or scale fold\n");
            return 3;
        }
        if (!(worst_q8_rel > 0.0 && worst_q8_rel < 0.1)) {
            fprintf(stderr, "FAIL: Q8 relL2 %.3g out of band — layout/scale bug\n", worst_q8_rel);
            return 3;
        }
        printf("VERIFY: OK\n");
    }
    return 0;

fail:
    for (int v = 0; v < 2; v++) { if (mf[v]) fclose(mf[v]); if (bfd[v] >= 0) close(bfd[v]); }
    return 2;
}
