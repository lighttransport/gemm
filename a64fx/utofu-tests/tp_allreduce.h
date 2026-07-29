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

#include <errno.h>
#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <utofu.h>

#ifndef TP_AR_STAG
#define TP_AR_STAG 7                  /* steering tag for the all-reduce region  */
#define TP_AR_STAG2 8                 /* second region for the 2D/hierarchical AR col sub-comm  */
#endif
#define TP_AR_MAXN   512              /* max ranks in a TP/EP group (covers GLM5 384-node runs) */
#define TP_AR_NSTEP  11               /* recv slots: sid 0..nrounds+1 (bcast); 11 covers N<=512 (384-node) */
#define TP_AR_LINE   256              /* A64FX cache line; each slot own-aligned  */
#ifndef TP_AR_TIMEOUT
#define TP_AR_TIMEOUT 60.0
#endif

typedef struct {
    int use_bf16, deterministic, robust;
    int a2a, a2a_max;
    int ack, ack_retx;
    double ack_rtt, timeout;
    unsigned long drop_n;
} tp_comm_config;

typedef struct {
    utofu_vcq_hdl_t vcq;
    utofu_vcq_id_t  peer_vcq[TP_AR_MAXN];
    utofu_stadd_t   peer_base[TP_AR_MAXN];   /* peers' TP_AR_STAG region base     */
    utofu_stadd_t   base;                    /* my region base stadd              */
    char           *region;                  /* send slot + TP_AR_NSTEP recv slots*/
    size_t          region_size;
    int             owns_region;
    size_t          slot;                    /* bytes per slot (payload+seq, aligned)*/
    int             stag;                    /* steering tag of THIS comm's region (sub-comms differ)*/
    int             my_rank, nprocs, max_count;
    /* precomputed recursive-doubling schedule */
    int             pof2, rem, nrounds, bcast_sid, newrank;
    int             use_bf16;                /* TP_AR_BF16=1: halve reduce payload */
    int             deterministic;           /* TP_AR_DETERMINISTIC=1: fixed-root reduce/broadcast */
    int             robust;                  /* TP_AR_ROBUST: 0=passive spin, 1=drain+civac per spin,
                                              * 2=LEAN decode path (amortized drain + civac every 64 spins) */
    uint64_t        seq;                     /* monotonic call counter            */
    /* --- TP_AR_A2A: direct all-to-all sum for small (decode-size) payloads --- */
    int             a2a;                     /* TP_AR_A2A=1: enable */
    int             a2a_max;                 /* max elems for the a2a path (TP_AR_A2A_MAX, clamped to max_count) */
    size_t          a2a_base, a2a_slot;      /* dedicated recv region: 2 generations x nprocs slots */
    /* --- TP_AR_ACK: ack/retransmit reliability prototype (default off) --- */
    int             ack;                     /* 1 = reliable send (bounded retransmit + ack) */
    int             ack_retx;                /* max retransmits before optimistic proceed (TP_AR_ACK_RETX) */
    double          ack_rtt;                 /* retransmit interval seconds (TP_AR_ACK_RTT) */
    double          timeout;
    size_t          ack_base;                /* byte offset of ack region (nprocs 8B slots + 1 scratch) */
    unsigned long   drop_n, put_ctr;         /* TP_AR_DROP=N: drop 1-in-N payload Puts (loss injection) */
    int             send_inflight;           /* fast contiguous Put awaiting local completion */
    /* one outstanding send awaiting confirmation. send() is NON-blocking (Put + stash here); it is
     * retransmitted from BOTH the recv-wait spin AND tp_ar_confirm() until the peer acks -- retransmit
     * during recv is essential: if both directions of a doubling pair drop, both ranks block in recv,
     * and only a recv-loop retransmit (not the after-recv confirm, never reached) breaks the deadlock.
     * The send slot is untouched between send and confirm, so a retransmit re-Puts the correct payload. */
    int             pend_peer, pend_sid, pend_contig, pend_active, pend_retx;
    size_t          pend_pbytes; uint64_t pend_tok; double pend_t0;
    jmp_buf         *failure_jmp;
    int             error_code;
    char            error_message[192];
} tp_comm;

static void tp_ar_fail(tp_comm *c,int code,const char *fmt,...) __attribute__((noreturn));
static void tp_ar_fail(tp_comm *c,int code,const char *fmt,...){
    va_list ap;va_start(ap,fmt);vsnprintf(c->error_message,sizeof c->error_message,fmt,ap);va_end(ap);
    c->error_code=code;if(c->failure_jmp)longjmp(*c->failure_jmp,1);
    fprintf(stderr,"tp_ar: %s\n",c->error_message);exit(1);
}

static const char *tp_comm_error(const tp_comm *c){
    return c&&c->error_message[0]?c->error_message:"no collective error recorded";
}

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
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
/* SVE forms of the bf16 payload loops — BIT-EXACT elementwise transcriptions of the
 * scalar formulas above (same integer round-to-nearest-even, no reassociation), so the
 * lockstep bitwise-identical invariant is preserved.  The scalar loops cost ~12 us per
 * 6144-float round on one core, which dominated the measured 29 us/round AR anatomy. */
static inline svuint32_t tp_bf16_rne_sve(svbool_t pg, svuint32_t x){   /* bits -> rounded bf16 (in low 16) */
    return svlsr_n_u32_x(pg, svadd_u32_x(pg, x,
               svadd_n_u32_x(pg, svand_n_u32_x(pg, svlsr_n_u32_x(pg, x, 16), 1u), 0x7fffu)), 16);
}
static inline svfloat32_t tp_bf16_round_sve(svbool_t pg, svfloat32_t f){
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, tp_bf16_rne_sve(pg, svreinterpret_u32_f32(f)), 16));
}
static inline svfloat32_t tp_bf16_ld_sve(svbool_t pg, const uint16_t*p){
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}
#endif

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

static void tp_ar_send_puts(tp_comm *c, int peer, int sid, size_t pbytes, int contiguous);  /* fwd decl */
static int tp_ar_put_nb(tp_comm *c,int peer,utofu_stadd_t src,utofu_stadd_t dst,size_t len);
/* Retransmit the one outstanding send if still unacked and ack_rtt elapsed. Called from BOTH the
 * recv-wait spin AND tp_ar_confirm so a send is re-driven even while this rank blocks in a recv --
 * essential when both directions of a doubling pair drop (both ranks block in recv; only a recv-loop
 * retransmit, not the never-reached after-recv confirm, breaks it). Returns 1 when the pending send is
 * confirmed/inactive/optimistically-abandoned, 0 while still outstanding. No-op when ack is off. */
static inline int tp_ar_service_pending(tp_comm *c) {
    if (!c->pend_active) return 1;
    volatile uint64_t *ackp = (volatile uint64_t *)(c->region + c->ack_base + (size_t)c->pend_peer * TP_AR_LINE);
    tp_ar_flag_inval(ackp);
    if (*ackp >= c->pend_tok) { c->pend_active = 0; return 1; }              /* peer confirmed receipt */
    if (tp_ar_now() - c->pend_t0 >= c->ack_rtt) {
        if (++c->pend_retx > c->ack_retx) { c->pend_active = 0; return 1; }  /* optimistic proceed */
        tp_ar_send_puts(c, c->pend_peer, c->pend_sid, c->pend_pbytes, c->pend_contig);   /* retransmit */
        c->pend_t0 = tp_ar_now();
    }
    return 0;
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
    unsigned long spins = 0;
    /* LEAN decode path (robust=2): the per-spin drain+civac+dsb is the dominant
     * per-round cost of the decode all-reduce (the "robustness tax"). Amortize:
     * drain the MRQ once at wait ENTRY (+ once on completion, below) — overflow
     * pressure is ~1 notice per recv, so per-wait draining keeps the queue near
     * empty without polling it inside the hot spin; civac the trailer line only
     * every eighth spin — bounded staleness without a
     * clean+invalidate+dsb on every iteration. Correctness envelope is the same
     * as robust=1 (nothing is skipped, only done less often); validated
     * bitwise vs robust=1 under the qlair sim and for 16K sequential reduces
     * on a real 12-node A64FX allocation. */
    if (c->robust >= 2) tp_ar_drain_mrq(c);
    while (*trl < tok) {
        if (c->robust == 1) { tp_ar_drain_mrq(c); tp_ar_flag_inval(trl); }
        else if (c->robust >= 2 && (spins & 7ul) == 7ul) tp_ar_flag_inval(trl);
        if (c->ack) tp_ar_service_pending(c);   /* re-drive my outstanding send while I block here */
        /* TP_AR_SPIN_DBG=1: report long spins UNBUFFERED (raw write; simulator-friendly —
         * under qlair the sim-time TP_AR_TIMEOUT is effectively unreachable). */
        if (((++spins & 0xFFFFFul) == 0) && getenv("TP_AR_SPIN_DBG")) {
            char b[128]; int n = snprintf(b, sizeof b,
                "tp_ar SPIN rank=%d %s sid=%d want=%lu got=%lu spins=%luM\n",
                c->my_rank, what, sid, (unsigned long)tok, (unsigned long)*trl, spins >> 20);
            if (n > 0) { ssize_t w = write(2, b, (size_t)n); (void)w; }
        }
        if(tp_ar_now()-t0>c->timeout)tp_ar_fail(c,ETIMEDOUT,
            "rank %d %s timeout sid=%d want=%lu got=%lu",c->my_rank,what,sid,
            (unsigned long)tok,(unsigned long)*trl);
    }
    if (c->robust) tp_ar_drain_mrq(c);   /* consume THIS recv's RMT_PUT notice (no leak) */
}

/* one Put with BUSY-retry + local-completion drain (pp_runner idiom). */
static void tp_ar_put(tp_comm *c, int peer, utofu_stadd_t src, utofu_stadd_t dst, size_t len) {
    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    int rc; void *cb;
    for (;;) { rc = utofu_put(c->vcq, c->peer_vcq[peer], src, dst, len, 0, flags, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(c->vcq, 0, &cb); }
    if(rc!=UTOFU_SUCCESS)tp_ar_fail(c,EIO,"utofu_put rc=%d",rc);
    do { rc = utofu_poll_tcq(c->vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
    if(rc!=UTOFU_SUCCESS)tp_ar_fail(c,EIO,"poll_tcq rc=%d",rc);
    tp_ar_drain_mrq(c);   /* consume receiver-side RMT_PUT notices → no MRQ overflow */
}

static void tp_ar_complete_sends(tp_comm *c){if(!c->send_inflight)return;void *cb;int rc;
    while(c->send_inflight>0){rc=utofu_poll_tcq(c->vcq,0,&cb);
        if(rc==UTOFU_SUCCESS)c->send_inflight--;
        else if(rc!=UTOFU_ERR_NOT_FOUND)tp_ar_fail(c,EIO,"send poll_tcq rc=%d",rc);}
    tp_ar_drain_mrq(c);
}

/* --- TP_AR_ACK reliability helpers --- */
/* Put the (already-filled) send slot to peer's recv[sid]. Optional TP_AR_DROP loss injection:
 * every drop_n-th call the payload+trailer "vanish" (skipped) to exercise the retransmit path. */
static void tp_ar_send_puts(tp_comm *c, int peer, int sid, size_t pbytes, int contiguous) {
    if (c->drop_n && (++c->put_ctr % c->drop_n) == 0) return;   /* simulate a lost message */
    size_t tr_off = tp_ar_trailer_off(c);
    utofu_stadd_t src = c->base + tp_ar_slot_off(c, 0);
    utofu_stadd_t dst = c->peer_base[peer] + tp_ar_slot_off(c, 1 + sid);
    if(contiguous&&!c->ack)
        c->send_inflight+=tp_ar_put_nb(c,peer,src,dst,pbytes+8);
    else if (contiguous) tp_ar_put(c, peer, src, dst, pbytes + 8);
    else { tp_ar_put(c, peer, src, dst, pbytes); tp_ar_put(c, peer, src + tr_off, dst + tr_off, 8); }
}
/* Receiver R -> sender S: Put R's ack tok into S's ack[R] slot (8 B). Never drop-injected. */
static void tp_ar_ack_send(tp_comm *c, int to_peer, uint64_t tok) {
    char *scr = c->region + c->ack_base + (size_t)c->nprocs * TP_AR_LINE;   /* ack-send scratch */
    *(volatile uint64_t *)scr = tok;
    utofu_stadd_t src = c->base + c->ack_base + (size_t)c->nprocs * TP_AR_LINE;
    utofu_stadd_t dst = c->peer_base[to_peer] + c->ack_base + (size_t)c->my_rank * TP_AR_LINE;
    tp_ar_put(c, to_peer, src, dst, 8);
}
/* copy buf into send slot, Put payload to peer's recv slot `sid`, then publish the fixed-offset
 * trailer. For full-width fp32 max_count sends, payload and trailer are contiguous and travel in one
 * Put; otherwise a second 8-byte Put so small decode reductions do not pay the prefill payload size.
 * NON-BLOCKING: under TP_AR_ACK the send only Puts + records the pending send; the paired recv acks
 * the peer, and the driver then calls tp_ar_confirm() (below) to await THIS send's ack + retransmit.
 * Blocking on the ack here would deadlock -- recursive doubling has both partners send before either
 * recvs, and recv is what emits the ack. */
static void tp_ar_send(tp_comm *c, int peer, int sid, const float *buf, int count, uint64_t tok) {
    char *sb = c->region + tp_ar_slot_off(c, 0);
    size_t pbytes;
    if (c->use_bf16) {
        uint16_t *d = (uint16_t *)sb;
#if defined(__ARM_FEATURE_SVE)
        { const int vl=(int)svcntw();
          for (int i = 0; i < count; i += vl) {
              svbool_t pg = svwhilelt_b32(i, count);
              svst1h_u32(pg, d + i, tp_bf16_rne_sve(pg, svld1_u32(pg, (const uint32_t*)(buf + i))));
          } }
#else
        for (int i = 0; i < count; i++) d[i] = tp_f2bf16(buf[i]);
#endif
        pbytes = (size_t)count * sizeof(uint16_t);
    } else {
        memcpy(sb, buf, (size_t)count * sizeof(float));
        pbytes = (size_t)count * sizeof(float);
    }
    *(volatile uint64_t *)(sb + tp_ar_trailer_off(c)) = tok;
    int contiguous = (!c->use_bf16 && count == c->max_count);
    tp_ar_send_puts(c, peer, sid, pbytes, contiguous);
    if (c->ack) { c->pend_peer = peer; c->pend_sid = sid; c->pend_pbytes = pbytes; c->pend_contig = contiguous;
                  c->pend_tok = tok; c->pend_active = 1; c->pend_retx = 0; c->pend_t0 = tp_ar_now(); }
}
/* Finish confirming the LAST send (call once after the paired recv): retransmit until the peer acks,
 * then proceed OPTIMISTICALLY after ack_retx tries (idempotent payload). The recv-wait loop already
 * services it too, so by here it is usually already acked. No-op when ack is off. */
static void tp_ar_confirm(tp_comm *c) {
    tp_ar_complete_sends(c);
    if (!c->ack) return;
    while (!tp_ar_service_pending(c)) { if (c->robust) tp_ar_drain_mrq(c); }
}

/* wait for recv slot `sid` trailer to reach tok, then add its payload into buf.
 * bf16 mode: buf[i] = Rf(Rf(buf[i]) + recv_bf16[i]) — symmetric so both exchange
 * partners end bitwise-equal and buf stays bf16-valued (broadcast can be exact). */
static void tp_ar_recv_add(tp_comm *c, int sid, int from, float *buf, int count, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "wait");
    if (c->use_bf16) {
        const uint16_t *r = (const uint16_t *)rb;
#if defined(__ARM_FEATURE_SVE)
        { const int vl=(int)svcntw();
          for (int i = 0; i < count; i += vl) {
              svbool_t pg = svwhilelt_b32(i, count);
              svfloat32_t b = tp_bf16_round_sve(pg, svld1(pg, buf + i));
              svfloat32_t v = tp_bf16_ld_sve(pg, r + i);
              svst1(pg, buf + i, tp_bf16_round_sve(pg, svadd_f32_x(pg, b, v)));
          } }
#else
        for (int i = 0; i < count; i++)
            buf[i] = tp_bf16_round(tp_bf16_round(buf[i]) + tp_bf162f(r[i]));
#endif
    } else {
        const float *r = (const float *)rb;
        for (int i = 0; i < count; i++) buf[i] += r[i];
    }
    if (c->ack) tp_ar_ack_send(c, from, tok);   /* ack AFTER reading rb (a retransmit can't corrupt it) */
}
/* wait for recv slot `sid` trailer, then element-wise MAX its payload into buf.
 * max is exact (the result is always one of the two inputs), so unlike the sum the
 * bf16 path needs no symmetric-round trick: both partners compute max(Rf(buf),Rf(recv))
 * over bf16-valued operands and end bitwise-equal; fp32 path is plain max. Deterministic
 * (assoc/comm) => identical on every rank regardless of fold order => lockstep-safe. */
static void tp_ar_recv_max(tp_comm *c, int sid, int from, float *buf, int count, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "max");
    if (c->use_bf16) {
        const uint16_t *r = (const uint16_t *)rb;
#if defined(__ARM_FEATURE_SVE)
        { const int vl=(int)svcntw();
          for (int i = 0; i < count; i += vl) {
              svbool_t pg = svwhilelt_b32(i, count);
              svfloat32_t v = tp_bf16_ld_sve(pg, r + i);
              svfloat32_t b = tp_bf16_round_sve(pg, svld1(pg, buf + i));
              /* exact `v > b ? v : b` ternary semantics (not fmax) */
              svst1(pg, buf + i, svsel_f32(svcmpgt_f32(pg, v, b), v, b));
          } }
#else
        for (int i = 0; i < count; i++) { float v = tp_bf162f(r[i]), b = tp_bf16_round(buf[i]); buf[i] = v > b ? v : b; }
#endif
    } else {
        const float *r = (const float *)rb;
        for (int i = 0; i < count; i++) if (r[i] > buf[i]) buf[i] = r[i];
    }
    if (c->ack) tp_ar_ack_send(c, from, tok);
}
/* same wait but overwrite (broadcast leg: receive the final reduced value). */
static void tp_ar_recv_copy(tp_comm *c, int sid, int from, float *buf, int count, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "bcast");
    if (c->use_bf16) {
        const uint16_t *r = (const uint16_t *)rb;
#if defined(__ARM_FEATURE_SVE)
        { const int vl=(int)svcntw();
          for (int i = 0; i < count; i += vl) {
              svbool_t pg = svwhilelt_b32(i, count);
              svst1(pg, buf + i, tp_bf16_ld_sve(pg, r + i));
          } }
#else
        for (int i = 0; i < count; i++) buf[i] = tp_bf162f(r[i]);
#endif
    } else {
        memcpy(buf, rb, (size_t)count * sizeof(float));
    }
    if (c->ack) tp_ar_ack_send(c, from, tok);
}

/* one Put WITHOUT waiting for local TCQ completion (pipelined injection); the caller
 * polls 'inflight' completions afterwards. BUSY-retry drains one TCQ entry to make room. */
static int tp_ar_put_nb(tp_comm *c, int peer, utofu_stadd_t src, utofu_stadd_t dst, size_t len) {
    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    int rc; void *cb;
    for (;;) { rc = utofu_put(c->vcq, c->peer_vcq[peer], src, dst, len, 0, flags, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(c->vcq, 0, &cb); }
    if(rc!=UTOFU_SUCCESS)tp_ar_fail(c,EIO,"utofu_put(nb) rc=%d",rc);
    return 1;
}
/* TP_AR_A2A sum: Put my payload to EVERY peer's a2a slot[gen][my_rank] (pipelined),
 * wait all N-1 trailers ONCE, then fold all N payloads in RANK ORDER. One detection
 * latency instead of ~ceil(log2 N)+2 sequential exchanges; every rank folds the same
 * buffers in the same order -> bitwise-identical across ranks (lockstep-safe), but the
 * fold order differs from recursive doubling -> reassoc vs the doubling path (coherent-
 * class; integer payloads, e.g. tp_ar_ack_test's, sum exactly -> bitwise-equal there).
 * BW cost x(N-1)/log2(N) -- enabled only for count <= a2a_max (decode-size payloads). */
static void tp_ar_sum_a2a(tp_comm *c, float *buf, int count, uint64_t tok) {
    int N = c->nprocs, me = c->my_rank;
    char *sb = c->region + tp_ar_slot_off(c, 0);            /* reuse the send slot */
    size_t pbytes = (size_t)count * sizeof(float);
    size_t tr = (size_t)c->a2a_max * sizeof(float);         /* a2a slots' fixed trailer offset */
    memcpy(sb, buf, pbytes);
    *(volatile uint64_t *)(sb + tr) = tok;                  /* fits: a2a_max <= max_count */
    int gen = (int)(tok & 1);
    int inflight = 0; void *cb; int rc;
    for (int d = 1; d < N; d++) {
        int peer = (me + d) % N;
        utofu_stadd_t src = c->base + tp_ar_slot_off(c, 0);
        utofu_stadd_t dst = c->peer_base[peer] + c->a2a_base + ((size_t)gen * N + me) * c->a2a_slot;
        inflight += tp_ar_put_nb(c, peer, src, dst, pbytes);           /* payload */
        inflight += tp_ar_put_nb(c, peer, src + tr, dst + tr, 8);      /* then trailer (in-order per pair) */
    }
    while (inflight > 0) {                                   /* reap local completions */
        rc = utofu_poll_tcq(c->vcq, 0, &cb);
        if (rc == UTOFU_SUCCESS) inflight--;
        else if(rc!=UTOFU_ERR_NOT_FOUND)tp_ar_fail(c,EIO,"a2a poll_tcq rc=%d",rc);
    }
    tp_ar_drain_mrq(c);
    for (int r = 0; r < N; r++) {                            /* fold in rank order */
        const float *pr;
        if (r == me) pr = (const float *)sb;
        else {
            char *rb = c->region + c->a2a_base + ((size_t)gen * N + r) * c->a2a_slot;
            volatile uint64_t *trl = (volatile uint64_t *)(rb + tr);
            tp_ar_wait(c, trl, tok, r, "a2a");
            pr = (const float *)rb;
        }
        if (r == 0) memcpy(buf, pr, pbytes);
        else        for (int i = 0; i < count; i++) buf[i] += pr[i];
    }
}

/* Deterministic fixed-root sum. Recursive doubling is faster, but each survivor folds
 * ranks in a different order, and floating/BF16 addition is not associative.  That lets
 * long-context logits diverge after many reductions even though every rank starts from
 * the same token.  Reduce to rank 0 with a fixed binomial tree, then broadcast the same
 * result over the reverse tree.  The normal fast path remains unchanged; this opt-in mode
 * is for quality/stability validation and costs extra synchronization. */
static void tp_allreduce_sum_deterministic(tp_comm *c, float *buf, int count, uint64_t tok) {
    int nr=0; for(int step=1;step<c->nprocs;step<<=1) nr++;
    if(2*nr > TP_AR_NSTEP-1){
        tp_ar_fail(c,EINVAL,"deterministic N=%d needs %d slots (max %d)",
                c->nprocs,2*nr,TP_AR_NSTEP-1);
    }
    int active=1;
    for(int k=0,step=1; k<nr && active; k++,step<<=1){
        int span=step<<1, lane=c->my_rank%span;
        if(lane>=step){
            tp_ar_send(c,c->my_rank-step,k,buf,count,tok);
            tp_ar_confirm(c); active=0;
        } else if(c->my_rank+step<c->nprocs){
            tp_ar_recv_add(c,k,c->my_rank+step,buf,count,tok);
        }
    }
    for(int k=nr-1,step=1<<(nr-1); k>=0; k--,step>>=1){
        int span=step<<1, lane=c->my_rank%span, sid=nr+(nr-1-k);
        if(lane<step){
            if(c->my_rank+step<c->nprocs){
                tp_ar_send(c,c->my_rank+step,sid,buf,count,tok);
                tp_ar_confirm(c);
            }
        } else {
            tp_ar_recv_copy(c,sid,c->my_rank-step,buf,count,tok);
        }
    }
}

/* in-place sum-all-reduce of buf[0..count). All ranks must pass the same count. */
static void tp_allreduce_sum(tp_comm *c, float *buf, int count) {
    if (c->nprocs == 1) return;
    uint64_t tok = ++c->seq;
    if (c->deterministic){
        tp_allreduce_sum_deterministic(c,buf,count,tok);
        return;
    }
    if (c->a2a && !c->ack && !c->use_bf16 && count <= c->a2a_max && c->nprocs >= 2) {
        tp_ar_sum_a2a(c, buf, count, tok);
        return;
    }
    int mr = c->my_rank, rem = c->rem;

    /* 1. pre-reduce fold: even of the lowest 2*rem ranks -> its odd partner. The odd's recv_add is its
     * FIRST op (not a send), so the even confirms its prefold send IMMEDIATELY -- deferring it past the
     * bcast recv would let a dropped prefold wedge the odd (stuck in recv) before the retransmit runs. */
    if (mr < 2 * rem) {
        if (mr % 2 == 0) { tp_ar_send(c, mr + 1, 0, buf, count, tok); tp_ar_confirm(c); }  /* even sends + confirms */
        else             tp_ar_recv_add(c, 0, mr - 1, buf, count, tok);   /* odd folds it in */
    }

    /* 2. recursive doubling among the pof2 survivors. */
    if (c->newrank != -1) {
        for (int k = 0; k < c->nrounds; k++) {
            int pnr = c->newrank ^ (1 << k);
            int pr  = (pnr < rem) ? (pnr * 2 + 1) : (pnr + rem);
            tp_ar_send(c, pr, k + 1, buf, count, tok);
            tp_ar_recv_add(c, k + 1, pr, buf, count, tok);
            tp_ar_confirm(c);                        /* await pr's ack of my send (pr just acked in recv) */
        }
    }

    /* 3. broadcast the result back to the folded-out even ranks. */
    if (mr < 2 * rem) {
        if (mr % 2 == 0) tp_ar_recv_copy(c, c->bcast_sid, mr + 1, buf, count, tok);   /* even: recv only (prefold already confirmed) */
        else { tp_ar_send(c, mr - 1, c->bcast_sid, buf, count, tok); tp_ar_confirm(c); } /* odd: confirm the bcast send */
    }
}

/* Fixed-root MAX companion.  The mathematical max is associative, but the BF16
 * transport rounds the local accumulator while folding; canonicalizing the tree
 * removes that order-dependent rounding from CP/MSA block selection. */
static void tp_allreduce_max_deterministic(tp_comm *c, float *buf, int count, uint64_t tok) {
    int nr=0; for(int step=1;step<c->nprocs;step<<=1) nr++;
    if(2*nr > TP_AR_NSTEP-1){
        tp_ar_fail(c,EINVAL,"deterministic MAX N=%d needs %d slots (max %d)",
                c->nprocs,2*nr,TP_AR_NSTEP-1);
    }
    int active=1;
    for(int k=0,step=1; k<nr && active; k++,step<<=1){
        int span=step<<1, lane=c->my_rank%span;
        if(lane>=step){
            tp_ar_send(c,c->my_rank-step,k,buf,count,tok);
            tp_ar_confirm(c); active=0;
        } else if(c->my_rank+step<c->nprocs){
            tp_ar_recv_max(c,k,c->my_rank+step,buf,count,tok);
        }
    }
    for(int k=nr-1,step=1<<(nr-1); k>=0; k--,step>>=1){
        int span=step<<1, lane=c->my_rank%span, sid=nr+(nr-1-k);
        if(lane<step){
            if(c->my_rank+step<c->nprocs){
                tp_ar_send(c,c->my_rank+step,sid,buf,count,tok);
                tp_ar_confirm(c);
            }
        } else {
            tp_ar_recv_copy(c,sid,c->my_rank-step,buf,count,tok);
        }
    }
}

/* in-place MAX-all-reduce of buf[0..count). Same recursive-doubling schedule as
 * tp_allreduce_sum, reduction op = element-wise max (tp_ar_recv_max). Used by the
 * Phase-2 context-parallel online-softmax combine (global per-head max before the
 * exp rescale). Must be called in lockstep with the sum all-reduces (shares seq). */
static void tp_allreduce_max(tp_comm *c, float *buf, int count) {
    if (c->nprocs == 1) return;
    uint64_t tok = ++c->seq;
    if (c->deterministic){
        tp_allreduce_max_deterministic(c,buf,count,tok);
        return;
    }
    int mr = c->my_rank, rem = c->rem;

    if (mr < 2 * rem) {                                   /* 1. pre-reduce fold even->odd */
        if (mr % 2 == 0) { tp_ar_send(c, mr + 1, 0, buf, count, tok); tp_ar_confirm(c); }
        else             tp_ar_recv_max(c, 0, mr - 1, buf, count, tok);
    }
    if (c->newrank != -1) {                               /* 2. recursive doubling */
        for (int k = 0; k < c->nrounds; k++) {
            int pnr = c->newrank ^ (1 << k);
            int pr  = (pnr < rem) ? (pnr * 2 + 1) : (pnr + rem);
            tp_ar_send(c, pr, k + 1, buf, count, tok);
            tp_ar_recv_max(c, k + 1, pr, buf, count, tok);
            tp_ar_confirm(c);
        }
    }
    if (mr < 2 * rem) {                                   /* 3. broadcast to folded-out evens */
        if (mr % 2 == 0) tp_ar_recv_copy(c, c->bcast_sid, mr + 1, buf, count, tok);
        else { tp_ar_send(c, mr - 1, c->bcast_sid, buf, count, tok); tp_ar_confirm(c); }
    }
}

static int tp_allreduce_checked(tp_comm *c,float *buf,int count,
        void (*operation)(tp_comm*,float*,int)){
    if(count<0||count>c->max_count){c->error_code=EINVAL;
        snprintf(c->error_message,sizeof c->error_message,"count %d exceeds max_count %d",count,c->max_count);return EINVAL;}
    jmp_buf failure;c->error_code=0;c->error_message[0]='\0';c->failure_jmp=&failure;
    if(setjmp(failure)){c->failure_jmp=NULL;return c->error_code?c->error_code:EIO;}
    operation(c,buf,count);c->failure_jmp=NULL;return 0;
}

static int tp_allreduce_sum_checked(tp_comm *c,float *buf,int count){
    return tp_allreduce_checked(c,buf,count,tp_allreduce_sum);
}

static int tp_allreduce_max_checked(tp_comm *c,float *buf,int count){
    return tp_allreduce_checked(c,buf,count,tp_allreduce_max);
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

/* Batched argmax: N independent (val, idx-as-float-bits) pairs reduced in ONE collective
 * (payload 2*n floats) instead of n tp_allreduce_argmax calls — the batched-decode head
 * merges all M streams' vocab-shard argmaxes with a single AR. Same recursive-doubling
 * schedule; combine = per-pair max-with-lower-index (assoc+comm => lockstep-safe).
 * Requires 2*n <= max_count. Shares the monotonic seq with the other collectives. */
static void tp_ar_recv_argmax_n(tp_comm *c, int sid, float *vi, int n, uint64_t tok) {
    char *rb = c->region + tp_ar_slot_off(c, 1 + sid);
    volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
    tp_ar_wait(c, trl, tok, sid, "argmaxn");
    const float *r = (const float *)rb;
    for (int k = 0; k < n; k++) {
        int32_t oidx, cidx; memcpy(&oidx, &r[2*k+1], 4); memcpy(&cidx, &vi[2*k+1], 4);
        if (r[2*k] > vi[2*k] || (r[2*k] == vi[2*k] && oidx < cidx)) { vi[2*k] = r[2*k]; vi[2*k+1] = r[2*k+1]; }
    }
}
static void tp_ar_send_argmax_n(tp_comm *c, int peer, int sid, const float *vi, int n, uint64_t tok) {
    char *sb = c->region + tp_ar_slot_off(c, 0);
    memcpy(sb, vi, (size_t)2 * n * sizeof(float));
    size_t tr_off = tp_ar_trailer_off(c);
    *(volatile uint64_t *)(sb + tr_off) = tok;
    utofu_stadd_t src = c->base + tp_ar_slot_off(c, 0);
    utofu_stadd_t dst = c->peer_base[peer] + tp_ar_slot_off(c, 1 + sid);
    tp_ar_put(c, peer, src, dst, (size_t)2 * n * sizeof(float));
    tp_ar_put(c, peer, src + tr_off, dst + tr_off, 8);
}
static void tp_allreduce_argmax_n(tp_comm *c, float *vi, int n) {
    if (c->nprocs == 1) return;
    if(2*n>c->max_count)tp_ar_fail(c,EINVAL,"argmax_n %d > max_count",n);
    uint64_t tok = ++c->seq;
    int mr = c->my_rank, rem = c->rem;
    if (mr < 2 * rem) {
        if (mr % 2 == 0) tp_ar_send_argmax_n(c, mr + 1, 0, vi, n, tok);
        else             tp_ar_recv_argmax_n(c, 0, vi, n, tok);
    }
    if (c->newrank != -1) {
        for (int k = 0; k < c->nrounds; k++) {
            int pnr = c->newrank ^ (1 << k);
            int pr  = (pnr < rem) ? (pnr * 2 + 1) : (pnr + rem);
            tp_ar_send_argmax_n(c, pr, k + 1, vi, n, tok);
            tp_ar_recv_argmax_n(c, k + 1, vi, n, tok);
        }
    }
    if (mr < 2 * rem) {
        if (mr % 2 == 0) {   /* broadcast leg: overwrite with the final pairs */
            char *rb = c->region + tp_ar_slot_off(c, 1 + c->bcast_sid);
            volatile uint64_t *trl = (volatile uint64_t *)(rb + tp_ar_trailer_off(c));
            tp_ar_wait(c, trl, tok, c->bcast_sid, "argmaxn bcast");
            memcpy(vi, rb, (size_t)2 * n * sizeof(float));
        } else tp_ar_send_argmax_n(c, mr - 1, c->bcast_sid, vi, n, tok);
    }
}

static tp_comm_config tp_comm_env_config(void) {
    tp_comm_config o = {0};
    o.use_bf16 = getenv("TP_AR_BF16") ? atoi(getenv("TP_AR_BF16")) : 0;
    o.deterministic = getenv("TP_AR_DETERMINISTIC") ? atoi(getenv("TP_AR_DETERMINISTIC")) : 0;
    o.robust = getenv("TP_AR_ROBUST") ? atoi(getenv("TP_AR_ROBUST")) : 1;
    o.a2a = getenv("TP_AR_A2A") ? atoi(getenv("TP_AR_A2A")) : 0;
    o.a2a_max = getenv("TP_AR_A2A_MAX") ? atoi(getenv("TP_AR_A2A_MAX")) : 8192;
    o.ack = getenv("TP_AR_ACK") ? atoi(getenv("TP_AR_ACK")) : 0;
    o.ack_retx = getenv("TP_AR_ACK_RETX") ? atoi(getenv("TP_AR_ACK_RETX")) : 64;
    o.ack_rtt = getenv("TP_AR_ACK_RTT") ? atof(getenv("TP_AR_ACK_RTT")) : 0.001;
    o.timeout = TP_AR_TIMEOUT;
    o.drop_n = getenv("TP_AR_DROP") ? strtoul(getenv("TP_AR_DROP"), NULL, 10) : 0;
    return o;
}

static size_t tp_comm_region_size(int nprocs, int max_count, const tp_comm_config *options) {
    tp_comm_config fallback = { .robust=1, .a2a_max=8192, .ack_retx=64, .ack_rtt=0.001, .timeout=TP_AR_TIMEOUT };
    const tp_comm_config *o = options ? options : &fallback;
    int a2a_max=o->a2a_max>0?o->a2a_max:8192;if(a2a_max>max_count)a2a_max=max_count;
    size_t slot=((size_t)max_count*sizeof(float)+8+(TP_AR_LINE-1))&~(size_t)(TP_AR_LINE-1);
    size_t bytes=(size_t)(1+TP_AR_NSTEP)*slot+(o->ack?(size_t)(nprocs+1)*TP_AR_LINE:0);
    size_t a2a_slot=((size_t)a2a_max*sizeof(float)+8+(TP_AR_LINE-1))&~(size_t)(TP_AR_LINE-1);
    if(o->a2a)bytes+=(size_t)2*nprocs*a2a_slot;return bytes;
}

/* Register an optionally caller-owned comm region and query peers. */
static int tp_comm_init_region_ex(tp_comm *c, utofu_vcq_hdl_t vcq,
                        const utofu_vcq_id_t *peer_vcq,int my_rank,int nprocs,
                        int max_count,void (*barrier_fn)(void),int stag,
                        const tp_comm_config *options,void *external_region,size_t external_size) {
    if (nprocs > TP_AR_MAXN) { fprintf(stderr, "tp_ar: nprocs %d > %d\n", nprocs, TP_AR_MAXN); return -1; }
    memset(c, 0, sizeof *c);
    c->vcq = vcq; c->my_rank = my_rank; c->nprocs = nprocs; c->max_count = max_count; c->stag = stag;
    for (int r = 0; r < nprocs; r++) c->peer_vcq[r] = peer_vcq[r];

    tp_comm_config env;if(!options){env=tp_comm_env_config();options=&env;}
    c->use_bf16=options->use_bf16;c->deterministic=options->deterministic;
    c->robust=options->robust;c->ack=options->ack;c->ack_retx=options->ack_retx;
    c->ack_rtt=options->ack_rtt;c->timeout=options->timeout>0?options->timeout:TP_AR_TIMEOUT;
    c->drop_n=options->drop_n;c->a2a=options->a2a;

    c->slot = ((size_t)max_count * sizeof(float) + 8 + (TP_AR_LINE - 1)) & ~(size_t)(TP_AR_LINE - 1);
    /* reliability prototype: an ack region of nprocs 8B slots (peer p writes its ack tok to ack[p])
     * plus one scratch slot the acking rank Puts FROM. Only allocated when TP_AR_ACK=1. */
    c->ack_base = (size_t)(1 + TP_AR_NSTEP) * c->slot;
    size_t region_sz = c->ack_base + (c->ack ? (size_t)(nprocs + 1) * TP_AR_LINE : 0);
    /* TP_AR_A2A recv region: 2 generations x nprocs slots sized for a2a_max elems (small decode
     * payloads only), appended after the ack region. Generation double-buffering (slot picked by
     * seq&1) keeps a rank one reduce ahead from overwriting a slot its slow peer hasn't read. */
    c->a2a_max = options->a2a_max>0?options->a2a_max:8192;
    if (c->a2a_max > max_count) c->a2a_max = max_count;
    c->a2a_slot = ((size_t)c->a2a_max * sizeof(float) + 8 + (TP_AR_LINE - 1)) & ~(size_t)(TP_AR_LINE - 1);
    c->a2a_base = region_sz;
    if (c->a2a) region_sz += (size_t)2 * nprocs * c->a2a_slot;
    c->region_size=region_sz;
    if(external_region){if(((uintptr_t)external_region&(TP_AR_LINE-1))||external_size<region_sz){
            fprintf(stderr,"tp_ar: external region invalid: ptr=%p supplied=%zu required=%zu\n",external_region,external_size,region_sz);return-1;}
        c->region=external_region;c->owns_region=0;
    }else{if(posix_memalign((void **)&c->region,TP_AR_LINE,region_sz)!=0){
            fprintf(stderr,"tp_ar: posix_memalign failed\n");return-1;}c->owns_region=1;}
    memset(c->region, 0, region_sz);
    /* Flush the dirty memset-zeros to DRAM BEFORE registration so the robust-path
     * poll-time `dc civac` (tp_ar_flag_inval) can only ever pull a landed Put down
     * from DRAM — never write a still-dirty stale zero back OVER a delivered Put.
     * Establishes the clean cache baseline the assetload collective relies on. */
    for (size_t off = 0; off < region_sz; off += TP_AR_LINE)
        __asm__ __volatile__("dc civac, %0" :: "r"(c->region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");

    int rc = utofu_reg_mem_with_stag(vcq, c->region, region_sz, stag, 0, &c->base);
    if (rc != UTOFU_SUCCESS) { fprintf(stderr, "tp_ar: reg_mem rc=%d\n", rc);
        if(c->owns_region)free(c->region);c->region=NULL;return -1; }

    if (barrier_fn) barrier_fn();        /* all regions registered before query */

    for (int r = 0; r < nprocs; r++) {
        if (r == my_rank) { c->peer_base[r] = c->base; continue; }
        rc = utofu_query_stadd(c->peer_vcq[r], stag, &c->peer_base[r]);
        if (rc != UTOFU_SUCCESS) { fprintf(stderr, "tp_ar: query_stadd peer %d rc=%d\n", r, rc);
            utofu_dereg_mem(c->vcq,c->base,0);if(c->owns_region)free(c->region);c->region=NULL;return -1; }
    }

    /* recursive-doubling schedule */
    c->pof2 = 1; while (c->pof2 * 2 <= nprocs) c->pof2 *= 2;
    c->rem = nprocs - c->pof2;
    c->nrounds = 0; for (int x = 1; x < c->pof2; x <<= 1) c->nrounds++;
    c->bcast_sid = c->nrounds + 1;
    if(c->bcast_sid>=TP_AR_NSTEP){fprintf(stderr,"tp_ar: too many steps for N=%d\n",nprocs);
        utofu_dereg_mem(c->vcq,c->base,0);if(c->owns_region)free(c->region);c->region=NULL;return -1;}
    if (my_rank < 2 * c->rem) c->newrank = (my_rank % 2 == 0) ? -1 : my_rank / 2;
    else                      c->newrank = my_rank - c->rem;
    c->seq = 0;
    /* retransmit interval: recovery latency is ~ack_rtt per lost Put. 1 ms is >> the µs-scale real
     * ack RTT + payload-reduce time (even a ~256 KB batched-prefill tile reduces in <~1 ms), so no
     * spurious retransmits, while giving ~20x faster loss recovery than the old 20 ms (validated 11n:
     * drop=50 119 -> 2244 reduce/s). ack_retx*ack_rtt = 64 ms optimistic-proceed budget. */
    c->put_ctr  = 0;
    if (my_rank == 0)
        fprintf(stderr, "tp_ar: N=%d pof2=%d rem=%d rounds=%d payload=%s deterministic=%d robust=%d ack=%d%s\n",
                nprocs, c->pof2, c->rem, c->nrounds, c->use_bf16 ? "bf16" : "fp32", c->deterministic, c->robust,
                c->ack, c->drop_n ? " DROP-INJECT" : "");
    return 0;
}

static int tp_comm_init_ex(tp_comm *c, utofu_vcq_hdl_t vcq,const utofu_vcq_id_t *peer_vcq,
                        int my_rank,int nprocs,int max_count,void (*barrier_fn)(void),int stag){
    return tp_comm_init_region_ex(c,vcq,peer_vcq,my_rank,nprocs,max_count,barrier_fn,stag,NULL,NULL,0);
}

static int tp_comm_init_external(tp_comm *c,utofu_vcq_hdl_t vcq,const utofu_vcq_id_t *peer_vcq,
                        int my_rank,int nprocs,int max_count,void (*barrier_fn)(void),
                        const tp_comm_config *options,void *region,size_t region_size){
    return tp_comm_init_region_ex(c,vcq,peer_vcq,my_rank,nprocs,max_count,barrier_fn,
        TP_AR_STAG,options,region,region_size);
}

/* back-compat: the single-region all-reduce over the whole group (TP_AR_STAG). */
static int tp_comm_init(tp_comm *c, utofu_vcq_hdl_t vcq, const utofu_vcq_id_t *peer_vcq,
                        int my_rank, int nprocs, int max_count, void (*barrier_fn)(void)) {
    return tp_comm_init_ex(c, vcq, peer_vcq, my_rank, nprocs, max_count, barrier_fn, TP_AR_STAG);
}

static void tp_comm_free(tp_comm *c) {
    if (c->region) { utofu_dereg_mem(c->vcq, c->base, 0);
        if(c->owns_region)free(c->region);c->region = NULL; }
}

/* ======================= hierarchical (2-level) all-reduce =======================
 * Factor the N ranks into A groups of B (A*B == N, rank r -> group g=r/B, pos b=r%B).
 * A flat recursive-double over N pairs partners 2^k apart in rank space -> up to
 * log2(N) *physically distant* torus hops.  The 2-level form runs two SMALL reduces:
 *   (1) row: all-reduce within my group {g*B .. g*B+B-1}  (B CONTIGUOUS ranks)
 *   (2) col: all-reduce across the A group-siblings {b, B+b, 2B+b, ...} (stride B)
 * so each level's partners are a contiguous ring / a single torus axis -> cheaper
 * per-hop latency, smaller per-level non-pof2 remainders, and the fast local row
 * reduce absorbs arrival skew before the global col step.  Round count is unchanged
 * (log2 A + log2 B == log2 N); the win is per-round cost + skew, not round count.
 * The two sub-comms register DISTINCT regions (TP_AR_STAG / TP_AR_STAG2) so they do
 * not collide, and reuse the full recursive-doubling / bf16 / robust machinery.
 * Pick A to match a real torus dimension (e.g. N=36 -> A=6,B=6; N=32 -> A=4,B=8). */
static int tp_comm_init_2d(tp_comm *row, tp_comm *col, utofu_vcq_hdl_t vcq,
                           const utofu_vcq_id_t *peer_vcq, int my_rank, int nprocs,
                           int A, int max_count, void (*barrier_fn)(void)) {
    if (A < 1 || nprocs % A != 0) { fprintf(stderr, "tp_ar_2d: A=%d does not divide N=%d\n", A, nprocs); return -1; }
    int B = nprocs / A;
    int g = my_rank / B, b = my_rank % B;
    utofu_vcq_id_t pv[TP_AR_MAXN];
    /* row sub-comm: the B contiguous ranks of my group; my row-rank is b (STAG). */
    for (int j = 0; j < B; j++) pv[j] = peer_vcq[g * B + j];
    if (tp_comm_init_ex(row, vcq, pv, b, B, max_count, barrier_fn, TP_AR_STAG) != 0) return -1;
    /* col sub-comm: my A group-siblings at stride B; my col-rank is g (STAG2).
     * A second global barrier (barrier_fn) ensures every rank has registered STAG2
     * before any col query_stadd -- both inits are called in lockstep by all ranks. */
    for (int i = 0; i < A; i++) pv[i] = peer_vcq[i * B + b];
    if (tp_comm_init_ex(col, vcq, pv, g, A, max_count, barrier_fn, TP_AR_STAG2) != 0) return -1;
    if (my_rank == 0) fprintf(stderr, "tp_ar_2d: N=%d = A(%d) x B(%d), rounds %d+%d = %d (flat would be %d)\n",
                              nprocs, A, B, col->nrounds, row->nrounds, col->nrounds + row->nrounds,
                              (int)(8 * sizeof(int) - __builtin_clz(nprocs - 1)));
    return 0;
}

/* Pool-backed form used by production runners.  The caller owns both regions;
 * keeping their allocation policy outside this transport avoids hidden mallocs
 * and lets A64FX runners preserve NUMA placement and 256-byte alignment. */
static int tp_comm_init_2d_external(tp_comm *row, tp_comm *col,
                           utofu_vcq_hdl_t vcq,
                           const utofu_vcq_id_t *peer_vcq,
                           int my_rank, int nprocs, int A, int max_count,
                           void (*barrier_fn)(void),
                           const tp_comm_config *options,
                           void *row_region, size_t row_region_size,
                           void *col_region, size_t col_region_size) {
    if (A < 1 || nprocs % A != 0) {
        fprintf(stderr, "tp_ar_2d: A=%d does not divide N=%d\n", A, nprocs);
        return -1;
    }
    int B=nprocs/A,g=my_rank/B,b=my_rank%B;
    utofu_vcq_id_t pv[TP_AR_MAXN];
    for(int j=0;j<B;++j)pv[j]=peer_vcq[g*B+j];
    if(tp_comm_init_region_ex(row,vcq,pv,b,B,max_count,barrier_fn,
            TP_AR_STAG,options,row_region,row_region_size))return-1;
    for(int i=0;i<A;++i)pv[i]=peer_vcq[i*B+b];
    if(tp_comm_init_region_ex(col,vcq,pv,g,A,max_count,barrier_fn,
            TP_AR_STAG2,options,col_region,col_region_size)){
        tp_comm_free(row);return-1;
    }
    if(my_rank==0)fprintf(stderr,
        "tp_ar_2d: external N=%d = A(%d) x B(%d), rounds %d+%d\n",
        nprocs,A,B,col->nrounds,row->nrounds);
    return 0;
}

/* in-place SUM-all-reduce of buf[0..count) via the 2-level schedule: reduce within the
 * group (row), then across groups (col).  After row, every rank in group g holds the
 * group-sum S_g; col then reduces {S_0..S_{A-1}} so every rank ends with the global sum.
 * Bit-reproducible across ranks (both sub-reduces are; the two-level order is fixed). */
static void tp_allreduce_sum_2d(tp_comm *row, tp_comm *col, float *buf, int count) {
    tp_allreduce_sum(row, buf, count);   /* within-group partial */
    tp_allreduce_sum(col, buf, count);   /* across-group -> global */
}

static int tp_allreduce_sum_2d_checked(tp_comm *row,tp_comm *col,
                                       float *buf,int count){
    int rc=tp_allreduce_sum_checked(row,buf,count);
    return rc?rc:tp_allreduce_sum_checked(col,buf,count);
}

static int tp_allreduce_max_2d_checked(tp_comm *row,tp_comm *col,
                                       float *buf,int count){
    int rc=tp_allreduce_max_checked(row,buf,count);
    return rc?rc:tp_allreduce_max_checked(col,buf,count);
}

static void tp_comm_free_2d(tp_comm *row, tp_comm *col) { tp_comm_free(row); tp_comm_free(col); }

#endif /* TP_ALLREDUCE_H */
