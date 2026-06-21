/* tp_allreduce.h - MPI-free uTofu sum-all-reduce over a tensor-parallel group.
 *
 * Recursive-doubling (Rabenseifner, non-power-of-2 aware) all-reduce that
 * actually SUMS float buffers in-place -- the comm primitive for tensor-parallel
 * decode (one all-reduce per row-parallel projection, ~2-3 per layer). Latency
 * scales as ceil(log2 N) rounds, not the ring's N-1. Modeled on the dependency
 * chain in ring_attn_bench.c's TREE_ALLREDUCE, but with real reduction.
 *
 * Usage: the runner does the uTofu bootstrap (VCQ create, peer VCQ reconstruct
 * from topo) exactly as pp_runner.c does, then:
 *     tp_comm c;
 *     tp_comm_init(&c, vcq, peer_vcq, my_rank, nprocs, max_count, barrier_fn);
 *     ... tp_allreduce_sum(&c, buf, count);   // buf[0..count) := sum over ranks
 * The module registers its OWN comm region under TP_AR_STAG (separate from the
 * runner's data region), so it composes with an existing hidden-handoff region.
 *
 * Constraints: count <= max_count; every rank calls with the SAME count (true
 * for TP -- all ranks reduce the same projection output dim). nprocs <= TP_AR_MAXN.
 */
#ifndef TP_ALLREDUCE_H
#define TP_ALLREDUCE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <utofu.h>

#ifndef TP_AR_STAG
#define TP_AR_STAG 7                  /* steering tag for the all-reduce region  */
#endif
#define TP_AR_MAXN   256              /* max ranks in a TP/EP group (covers GLM5 192-node runs) */
#define TP_AR_NSTEP  9                /* recv slots: sid 0..ceil(log2 pof2(N))+1; 9 covers N<=256 */
#define TP_AR_LINE   256              /* A64FX cache line; each slot own-aligned  */
#ifndef TP_AR_TIMEOUT
#define TP_AR_TIMEOUT 60.0
#endif

typedef struct {
    utofu_vcq_hdl_t vcq;
    utofu_vcq_id_t  peer_vcq[TP_AR_MAXN];
    utofu_stadd_t   peer_base[TP_AR_MAXN];   /* peers' TP_AR_STAG region base     */
    utofu_stadd_t   base;                    /* my region base stadd              */
    char           *region;                  /* send slot + TP_AR_NSTEP recv slots*/
    size_t          slot;                    /* bytes per slot (payload+seq, aligned)*/
    int             my_rank, nprocs, max_count;
    /* precomputed recursive-doubling schedule */
    int             pof2, rem, nrounds, bcast_sid, newrank;
    int             use_bf16;                /* TP_AR_BF16=1: halve reduce payload */
    int             robust;                  /* TP_AR_ROBUST=1: drain-per-recv + civac */
    uint64_t        seq;                     /* monotonic call counter            */
} tp_comm;

static double tp_ar_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* bf16 round-trip helpers for the optional half-precision reduce payload.
 * f2bf16 uses round-to-nearest-even (matches the residual-stream bf16 weights).
 * The reduction is made SYMMETRIC (every exchange computes Rf(Rf(a)+Rf(b)), keeping
 * buf bf16-valued) so all ranks stay BITWISE-IDENTICAL — required for the
 * lockstep-argmax design where each rank independently argmaxes the same logits. */
static inline uint16_t tp_f2bf16(float f) {
    uint32_t x; memcpy(&x, &f, sizeof x);
    uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);   /* round-to-nearest-even */
    return (uint16_t)(r >> 16);
}
static inline float tp_bf162f(uint16_t b) {
    uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, sizeof f); return f;
}
static inline float tp_bf16_round(float f) { return tp_bf162f(tp_f2bf16(f)); }

/* Invalidate one cache line before reading an RDMA-written trailer (A64FX).
 * A purely passive volatile-read spin can keep hitting a stale cached copy of the
 * trailer line and never observe a Put that already landed in DRAM. `dc civac`
 * (clean+invalidate to PoC, allowed at EL0) drops the line so the next read
 * re-fetches from DRAM. SAFE ONLY because the recv slots are CPU-READ-ONLY after
 * the startup memset+clean baseline (tp_comm_init flushes the dirty zeros to DRAM
 * BEFORE registration), so a poll-time civac on a clean line is invalidate-only
 * and never writes a stale zero back over a landed Put. Borrowed from
 * a64fx/assetload/assetload_dist_bench.c (flag_inval, same uTofu collective). */
static inline void tp_ar_flag_inval(const volatile void *p) {
    __asm__ __volatile__("dc civac, %0" :: "r"(p) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");   /* wait for the DC op to complete */
}

/* slot s layout: [payload: max_count floats][seq trailer][... pad ...].
 * The sum trailer is at a fixed max_count*sizeof(float) offset, not after the
 * live payload. Calls may use different count values in one communicator
 * (batched prefill then scalar decode); a count-relative trailer can alias stale
 * payload bytes from a previous larger transfer and falsely complete a receive.
 * send slot = 0, recv slot for step sid = (1+sid). */
static inline size_t tp_ar_slot_off(const tp_comm *c, int s) { return (size_t)s * c->slot; }
static inline size_t tp_ar_trailer_off(const tp_comm *c) { return (size_t)c->max_count * sizeof(float); }

/* Drain (and discard) any pending receive-completion notices. On Tofu-D every
 * landed Put posts an RMT_PUT entry to the *receiver's* MRQ regardless of the
 * REMOTE_MRQ_NOTICE flag; this trailer-polling protocol never consumes them, so
 * across a long decode (~10^4+ all-reduces) the MRQ OVERFLOWS, faults the TNI,
 * and subsequent Puts silently stop landing → the receiver spins on its trailer
 * forever (seen as `tp_ar wait timeout want=N+1 got=N` after ~86 tokens). We
 * don't use the notices (completion is the in-memory seq trailer), so drain-all
 * and discard. Cheap: NOT_FOUND returns immediately when the MRQ is empty. */
static inline void tp_ar_drain_mrq(tp_comm *c) {
    struct utofu_mrq_notice nt;
    while (utofu_poll_mrq(c->vcq, 0, &nt) == UTOFU_SUCCESS) { /* discard */ }
}

/* Spin until recv-slot trailer `trl` reaches `tok`, then return. In ROBUST mode
 * this is the fix for the large-M dropped-Put deadlock (`want=N+1 got=N`):
 *  (1) drain our MRQ EACH spin AND once on completion — recursive doubling does
 *      send(drains)/recv(notice lands) per round, so the last round's recv notice
 *      leaks 1/reduce; over ~10^4 reduces the *receiver's* MRQ overflows, faults
 *      the TNI, and inbound Puts silently stop landing. Draining per-recv keeps it
 *      near-empty so the TNI never faults.
 *  (2) civac the trailer line each spin — defeats a stale cached trailer masking a
 *      delivered Put. Free on the fast path (loop body runs only while waiting).
 * Non-robust path is byte-for-byte the original passive spin. */
static inline void tp_ar_wait(tp_comm *c, volatile uint64_t *trl, uint64_t tok,
                              int sid, const char *what) {
    double t0 = tp_ar_now();
    while (*trl < tok) {
        if (c->robust) { tp_ar_drain_mrq(c); tp_ar_flag_inval(trl); }
        if (tp_ar_now() - t0 > TP_AR_TIMEOUT) {
            fprintf(stderr, "tp_ar: rank %d %s timeout sid=%d want=%lu got=%lu\n",
                    c->my_rank, what, sid, (unsigned long)tok, (unsigned long)*trl);
            exit(1);
        }
    }
    if (c->robust) tp_ar_drain_mrq(c);   /* consume THIS recv's RMT_PUT notice (no leak) */
}

/* one Put with BUSY-retry + local-completion drain (pp_runner idiom). */
static void tp_ar_put(tp_comm *c, int peer, utofu_stadd_t src, utofu_stadd_t dst, size_t len) {
    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    int rc; void *cb;
    for (;;) { rc = utofu_put(c->vcq, c->peer_vcq[peer], src, dst, len, 0, flags, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(c->vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) { fprintf(stderr, "tp_ar: utofu_put rc=%d\n", rc); exit(1); }
    do { rc = utofu_poll_tcq(c->vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) { fprintf(stderr, "tp_ar: poll_tcq rc=%d\n", rc); exit(1); }
    tp_ar_drain_mrq(c);   /* consume receiver-side RMT_PUT notices → no MRQ overflow */
}

/* copy buf into send slot, Put payload to peer's recv slot `sid`, then publish
 * the fixed-offset trailer. For full-width fp32 max_count sends, payload and
 * trailer are contiguous and travel in one Put; otherwise use a second 8-byte
 * Put so small decode reductions do not pay the batched-prefill payload size. */
static void tp_ar_send(tp_comm *c, int peer, int sid, const float *buf, int count, uint64_t tok) {
    char *sb = c->region + tp_ar_slot_off(c, 0);
    size_t pbytes;
    if (c->use_bf16) {
        uint16_t *d = (uint16_t *)sb;
        for (int i = 0; i < count; i++) d[i] = tp_f2bf16(buf[i]);
        pbytes = (size_t)count * sizeof(uint16_t);
    } else {
        memcpy(sb, buf, (size_t)count * sizeof(float));
        pbytes = (size_t)count * sizeof(float);
    }
    size_t tr_off = tp_ar_trailer_off(c);
    *(volatile uint64_t *)(sb + tr_off) = tok;
    utofu_stadd_t src = c->base + tp_ar_slot_off(c, 0);
    utofu_stadd_t dst = c->peer_base[peer] + tp_ar_slot_off(c, 1 + sid);
    if (!c->use_bf16 && count == c->max_count) {
        tp_ar_put(c, peer, src, dst, pbytes + 8);
    } else {
        tp_ar_put(c, peer, src, dst, pbytes);
        tp_ar_put(c, peer, src + tr_off, dst + tr_off, 8);
    }
}

/* wait for recv slot `sid` trailer to reach tok, then add its payload into buf.
 * bf16 mode: buf[i] = Rf(Rf(buf[i]) + recv_bf16[i]) — symmetric so both exchange
 * partners end bitwise-equal and buf stays bf16-valued (broadcast can be exact). */
static void tp_ar_recv_add(tp_comm *c, int sid, float *buf, int count, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "wait");
    if (c->use_bf16) {
        const uint16_t *r = (const uint16_t *)rb;
        for (int i = 0; i < count; i++)
            buf[i] = tp_bf16_round(tp_bf16_round(buf[i]) + tp_bf162f(r[i]));
    } else {
        const float *r = (const float *)rb;
        for (int i = 0; i < count; i++) buf[i] += r[i];
    }
}
/* wait for recv slot `sid` trailer, then element-wise MAX its payload into buf.
 * max is exact (the result is always one of the two inputs), so unlike the sum the
 * bf16 path needs no symmetric-round trick: both partners compute max(Rf(buf),Rf(recv))
 * over bf16-valued operands and end bitwise-equal; fp32 path is plain max. Deterministic
 * (assoc/comm) => identical on every rank regardless of fold order => lockstep-safe. */
static void tp_ar_recv_max(tp_comm *c, int sid, float *buf, int count, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "max");
    if (c->use_bf16) {
        const uint16_t *r = (const uint16_t *)rb;
        for (int i = 0; i < count; i++) { float v = tp_bf162f(r[i]), b = tp_bf16_round(buf[i]); buf[i] = v > b ? v : b; }
    } else {
        const float *r = (const float *)rb;
        for (int i = 0; i < count; i++) if (r[i] > buf[i]) buf[i] = r[i];
    }
}
/* same wait but overwrite (broadcast leg: receive the final reduced value). */
static void tp_ar_recv_copy(tp_comm *c, int sid, float *buf, int count, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "bcast");
    if (c->use_bf16) {
        const uint16_t *r = (const uint16_t *)rb;
        for (int i = 0; i < count; i++) buf[i] = tp_bf162f(r[i]);
    } else {
        memcpy(buf, rb, (size_t)count * sizeof(float));
    }
}

/* in-place sum-all-reduce of buf[0..count). All ranks must pass the same count. */
static void tp_allreduce_sum(tp_comm *c, float *buf, int count) {
    if (c->nprocs == 1) return;
    uint64_t tok = ++c->seq;
    int mr = c->my_rank, rem = c->rem;

    /* 1. pre-reduce fold: even of the lowest 2*rem ranks -> its odd partner. */
    if (mr < 2 * rem) {
        if (mr % 2 == 0) tp_ar_send(c, mr + 1, 0, buf, count, tok);   /* even sends, then idle */
        else             tp_ar_recv_add(c, 0, buf, count, tok);       /* odd folds it in */
    }

    /* 2. recursive doubling among the pof2 survivors. */
    if (c->newrank != -1) {
        for (int k = 0; k < c->nrounds; k++) {
            int pnr = c->newrank ^ (1 << k);
            int pr  = (pnr < rem) ? (pnr * 2 + 1) : (pnr + rem);
            tp_ar_send(c, pr, k + 1, buf, count, tok);
            tp_ar_recv_add(c, k + 1, buf, count, tok);
        }
    }

    /* 3. broadcast the result back to the folded-out even ranks. */
    if (mr < 2 * rem) {
        if (mr % 2 == 0) tp_ar_recv_copy(c, c->bcast_sid, buf, count, tok);
        else             tp_ar_send(c, mr - 1, c->bcast_sid, buf, count, tok);
    }
}

/* in-place MAX-all-reduce of buf[0..count). Same recursive-doubling schedule as
 * tp_allreduce_sum, reduction op = element-wise max (tp_ar_recv_max). Used by the
 * Phase-2 context-parallel online-softmax combine (global per-head max before the
 * exp rescale). Must be called in lockstep with the sum all-reduces (shares seq). */
static void tp_allreduce_max(tp_comm *c, float *buf, int count) {
    if (c->nprocs == 1) return;
    uint64_t tok = ++c->seq;
    int mr = c->my_rank, rem = c->rem;

    if (mr < 2 * rem) {                                   /* 1. pre-reduce fold even->odd */
        if (mr % 2 == 0) tp_ar_send(c, mr + 1, 0, buf, count, tok);
        else             tp_ar_recv_max(c, 0, buf, count, tok);
    }
    if (c->newrank != -1) {                               /* 2. recursive doubling */
        for (int k = 0; k < c->nrounds; k++) {
            int pnr = c->newrank ^ (1 << k);
            int pr  = (pnr < rem) ? (pnr * 2 + 1) : (pnr + rem);
            tp_ar_send(c, pr, k + 1, buf, count, tok);
            tp_ar_recv_max(c, k + 1, buf, count, tok);
        }
    }
    if (mr < 2 * rem) {                                   /* 3. broadcast to folded-out evens */
        if (mr % 2 == 0) tp_ar_recv_copy(c, c->bcast_sid, buf, count, tok);
        else             tp_ar_send(c, mr - 1, c->bcast_sid, buf, count, tok);
    }
}

/* Argmax send: copy 2-float payload (val + index-as-bits) into the send slot
 * and stamp the seq trailer at the DEDICATED max_count*4 offset (the same slot
 * the sum trailer uses) — NOT count*4. The 2-float payload never reaches that
 * offset, so the trailer is a dedicated, monotonic location that no payload byte
 * can alias. Send as TWO small Puts (16 B total): the 2-float payload, then the
 * far trailer — NOT the whole slot. With max_count enlarged for batched-prefill
 * sums the slot is megabytes, so a whole-slot argmax Put would cost MB/token of
 * decode comm. Payload-then-trailer ordering matches tp_ar_send, so the receiver
 * never observes the trailer advance before the payload lands. The receiver reads
 * only bytes 0..7 and the far trailer, so the untouched middle is irrelevant. */
static void tp_ar_send_argmax(tp_comm *c, int peer, int sid, const float *vi, uint64_t tok) {
    char *sb = c->region + tp_ar_slot_off(c, 0);
    memcpy(sb, vi, 2 * sizeof(float));
    size_t tr_off = (size_t)c->max_count * sizeof(float);
    *(volatile uint64_t *)(sb + tr_off) = tok;
    utofu_stadd_t src = c->base + tp_ar_slot_off(c, 0);
    utofu_stadd_t dst = c->peer_base[peer] + tp_ar_slot_off(c, 1 + sid);
    tp_ar_put(c, peer, src, dst, 2 * sizeof(float));
    tp_ar_put(c, peer, src + tr_off, dst + tr_off, 8);
}

/* wait for recv slot `sid` (trailer at the dedicated max_count*4 offset), then
 * argmax-combine its (val,idx) into buf[0..1]. Larger value wins; ties break to
 * the LOWER index so the reduction is associative+commutative → identical on
 * every rank regardless of fold order. */
static void tp_ar_recv_argmax(tp_comm *c, int sid, float *buf, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + (size_t)c->max_count * sizeof(float));
    tp_ar_wait(c, trl, tok, sid, "argmax");
    const float *r = (const float *)rb;
    int32_t oidx, cidx; memcpy(&oidx, &r[1], 4); memcpy(&cidx, &buf[1], 4);
    if (r[0] > buf[0] || (r[0] == buf[0] && oidx < cidx)) { buf[0] = r[0]; buf[1] = r[1]; }
}

/* Broadcast leg for argmax: like tp_ar_recv_copy but keyed off the dedicated
 * max_count*4 trailer (copies the final 2-float result). */
static void tp_ar_recv_copy_argmax(tp_comm *c, int sid, float *buf, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + (size_t)c->max_count * sizeof(float));
    tp_ar_wait(c, trl, tok, sid, "argmax bcast");
    memcpy(buf, rb, 2 * sizeof(float));
}

/* All-reduce-argmax over the TP group: every rank passes its local best logit
 * (*val) and the GLOBAL token index (*idx); on return both hold the group-wide
 * argmax. Same recursive-doubling schedule as tp_allreduce_sum but the reduction
 * op is max-with-index. The payload is always 2 floats, so ranks may own
 * differently-sized vocab shards. Must be called in lockstep with the sum
 * all-reduces (shares the monotonic seq counter). */
static void tp_allreduce_argmax(tp_comm *c, float *val, int32_t *idx) {
    if (c->nprocs == 1) return;
    float buf[2]; buf[0] = *val; memcpy(&buf[1], idx, 4);
    uint64_t tok = ++c->seq;
    int mr = c->my_rank, rem = c->rem;

    if (mr < 2 * rem) {
        if (mr % 2 == 0) tp_ar_send_argmax(c, mr + 1, 0, buf, tok);
        else             tp_ar_recv_argmax(c, 0, buf, tok);
    }
    if (c->newrank != -1) {
        for (int k = 0; k < c->nrounds; k++) {
            int pnr = c->newrank ^ (1 << k);
            int pr  = (pnr < rem) ? (pnr * 2 + 1) : (pnr + rem);
            tp_ar_send_argmax(c, pr, k + 1, buf, tok);
            tp_ar_recv_argmax(c, k + 1, buf, tok);
        }
    }
    if (mr < 2 * rem) {
        if (mr % 2 == 0) tp_ar_recv_copy_argmax(c, c->bcast_sid, buf, tok);
        else             tp_ar_send_argmax(c, mr - 1, c->bcast_sid, buf, tok);
    }
    *val = buf[0]; memcpy(idx, &buf[1], 4);
}

/* Register the comm region (TP_AR_STAG) and query peers. `barrier_fn` must
 * globally synchronize all ranks (so every region is registered before the
 * stadd queries). Returns 0 on success. */
static int tp_comm_init(tp_comm *c, utofu_vcq_hdl_t vcq, const utofu_vcq_id_t *peer_vcq,
                        int my_rank, int nprocs, int max_count, void (*barrier_fn)(void)) {
    if (nprocs > TP_AR_MAXN) { fprintf(stderr, "tp_ar: nprocs %d > %d\n", nprocs, TP_AR_MAXN); return -1; }
    memset(c, 0, sizeof *c);
    c->vcq = vcq; c->my_rank = my_rank; c->nprocs = nprocs; c->max_count = max_count;
    for (int r = 0; r < nprocs; r++) c->peer_vcq[r] = peer_vcq[r];

    c->slot = ((size_t)max_count * sizeof(float) + 8 + (TP_AR_LINE - 1)) & ~(size_t)(TP_AR_LINE - 1);
    size_t region_sz = (size_t)(1 + TP_AR_NSTEP) * c->slot;
    if (posix_memalign((void **)&c->region, TP_AR_LINE, region_sz) != 0) {
        fprintf(stderr, "tp_ar: posix_memalign failed\n"); return -1;
    }
    memset(c->region, 0, region_sz);
    /* Flush the dirty memset-zeros to DRAM BEFORE registration so the robust-path
     * poll-time `dc civac` (tp_ar_flag_inval) can only ever pull a landed Put down
     * from DRAM — never write a still-dirty stale zero back OVER a delivered Put.
     * Establishes the clean cache baseline the assetload collective relies on. */
    for (size_t off = 0; off < region_sz; off += TP_AR_LINE)
        __asm__ __volatile__("dc civac, %0" :: "r"(c->region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");

    int rc = utofu_reg_mem_with_stag(vcq, c->region, region_sz, TP_AR_STAG, 0, &c->base);
    if (rc != UTOFU_SUCCESS) { fprintf(stderr, "tp_ar: reg_mem rc=%d\n", rc); return -1; }

    if (barrier_fn) barrier_fn();        /* all regions registered before query */

    for (int r = 0; r < nprocs; r++) {
        if (r == my_rank) { c->peer_base[r] = c->base; continue; }
        rc = utofu_query_stadd(c->peer_vcq[r], TP_AR_STAG, &c->peer_base[r]);
        if (rc != UTOFU_SUCCESS) { fprintf(stderr, "tp_ar: query_stadd peer %d rc=%d\n", r, rc); return -1; }
    }

    /* recursive-doubling schedule */
    c->pof2 = 1; while (c->pof2 * 2 <= nprocs) c->pof2 *= 2;
    c->rem = nprocs - c->pof2;
    c->nrounds = 0; for (int x = 1; x < c->pof2; x <<= 1) c->nrounds++;
    c->bcast_sid = c->nrounds + 1;
    if (c->bcast_sid >= TP_AR_NSTEP) { fprintf(stderr, "tp_ar: too many steps for N=%d\n", nprocs); return -1; }
    if (my_rank < 2 * c->rem) c->newrank = (my_rank % 2 == 0) ? -1 : my_rank / 2;
    else                      c->newrank = my_rank - c->rem;
    c->seq = 0;
    c->use_bf16 = getenv("TP_AR_BF16") && atoi(getenv("TP_AR_BF16")) != 0;
    c->robust   = getenv("TP_AR_ROBUST") ? atoi(getenv("TP_AR_ROBUST")) : 1;  /* default ON */
    if (my_rank == 0)
        fprintf(stderr, "tp_ar: N=%d pof2=%d rem=%d rounds=%d payload=%s robust=%d\n",
                nprocs, c->pof2, c->rem, c->nrounds, c->use_bf16 ? "bf16" : "fp32", c->robust);
    return 0;
}

static void tp_comm_free(tp_comm *c) {
    if (c->region) { utofu_dereg_mem(c->vcq, c->base, 0); free(c->region); c->region = NULL; }
}

#endif /* TP_ALLREDUCE_H */
