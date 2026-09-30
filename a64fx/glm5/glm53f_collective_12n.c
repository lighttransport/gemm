#define _GNU_SOURCE
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <utofu.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"
#include "glm53f_collective_12n.h"

enum { GLM53F_COLLECTIVE_MAX_NODES = 32 };
static tp_comm glm53f_comm;
static tp_comm glm53f_comm_row, glm53f_comm_col;
/* Small communicator sized for exactly one hidden vector (4096 floats).  The main communicator is sized for 32-token
 * prefill slabs, so a 4096-float decode reduce is never "count == max_count" and every send pays two blocking Puts
 * (payload, then trailer) plus an MRQ drain.  With max_count == count the trailer rides in the same Put. */
enum { GLM53F_SMALL_COUNT = 4096 };
static tp_comm glm53f_comm_small;
static int glm53f_small_active;
static utofu_vcq_hdl_t glm53f_vcq;
static int glm53f_utofu_active;
static int glm53f_utofu_2d;
static int glm53f_max_count;
static int glm53f_prefill_algorithm;

static void glm53f_mpi_barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }

static int glm53f_read_topology(const char *path,
        uint8_t coords[][TOFU_NCOORDS]) {
    FILE *file = fopen(path, "r");
    char line[256];
    int count = 0;
    if (!file) return -1;
    while (fgets(line, sizeof(line), file)) {
        unsigned rank, c[TOFU_NCOORDS];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1],
                &c[2], &c[3], &c[4], &c[5]) != 7 || rank != (unsigned)count ||
                count >= GLM53F_COLLECTIVE_MAX_NODES) {
            fclose(file);
            return -1;
        }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[count][k] = (uint8_t)c[k];
        count++;
    }
    fclose(file);
    return count;
}


/* ---- multi-TNI reduce-scatter + all-gather all-reduce (fp32, deterministic) --------------------------------
 * Each rank Puts shard d of its vector to rank d (11 Puts, distinct destinations round-robined over the TNIs), rank d
 * sums its shard in fixed rank order 0..N-1 and Puts the reduced shard back to every peer.  Payload and the 8-byte
 * sequence trailer travel in ONE Put (trailer directly behind the payload; the token is unique per call and carries a
 * magic prefix so stale payload bytes cannot alias it).  Slots need no double buffering: a peer cannot finish call n
 * (and start n+1) before it has received this rank's gather Put of call n, which is sent after this rank consumed its
 * scatter slots, and its call n+1 gather Put needs this rank's call n+1 scatter Put. */
enum { GLM53F_MTNI_MAXT = 6, GLM53F_MTNI_STAG = 9 };
static int glm53f_mtni_active, glm53f_mtni_nt, glm53f_mtni_max, glm53f_mtni_n;
static utofu_vcq_hdl_t glm53f_mtni_vcq[GLM53F_MTNI_MAXT];
static utofu_vcq_id_t glm53f_mtni_peer[GLM53F_MTNI_MAXT][GLM53F_COLLECTIVE_MAX_NODES];
static utofu_stadd_t glm53f_mtni_base[GLM53F_MTNI_MAXT], glm53f_mtni_pbase[GLM53F_MTNI_MAXT][GLM53F_COLLECTIVE_MAX_NODES];
static char *glm53f_mtni_region;
static size_t glm53f_mtni_slot, glm53f_mtni_rsize;
static uint64_t glm53f_mtni_seq;
/* region: send[d] (N) | recv[s] (N) | gsend (1) | grecv[s] (N)  -- each slot glm53f_mtni_slot bytes */
static inline size_t mtni_send_off(int d) { return (size_t)d * glm53f_mtni_slot; }
static inline size_t mtni_recv_off(int s) { return (size_t)(glm53f_mtni_n + s) * glm53f_mtni_slot; }
static inline size_t mtni_gsend_off(void) { return (size_t)(2 * glm53f_mtni_n) * glm53f_mtni_slot; }
static inline size_t mtni_grecv_off(int s) { return (size_t)(2 * glm53f_mtni_n + 1 + s) * glm53f_mtni_slot; }

static void mtni_drain_mrq(void) {
    struct utofu_mrq_notice nt;
    for (int k = 0; k < glm53f_mtni_nt; ++k) while (utofu_poll_mrq(glm53f_mtni_vcq[k], 0, &nt) == UTOFU_SUCCESS) {}
}
static int mtni_put(int k, int dst, size_t src_off, size_t dst_off, size_t len) {
    int rc; void *cb;
    for (;;) {
        rc = utofu_put(glm53f_mtni_vcq[k], glm53f_mtni_peer[k][dst], glm53f_mtni_base[k] + src_off,
                       glm53f_mtni_pbase[k][dst] + dst_off, len, 0, UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc != UTOFU_ERR_BUSY) break;
        utofu_poll_tcq(glm53f_mtni_vcq[k], 0, &cb);
    }
    return rc == UTOFU_SUCCESS ? 0 : -1;
}
static int mtni_drain_tcq(const int *issued) {
    void *cb;
    for (int k = 0; k < glm53f_mtni_nt; ++k)
        for (int j = 0; j < issued[k]; ++j) {
            int rc;
            do { rc = utofu_poll_tcq(glm53f_mtni_vcq[k], 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
            if (rc != UTOFU_SUCCESS) return -1;
        }
    return 0;
}
static inline void mtni_inval(const volatile void *p) {
    __asm__ __volatile__("dc civac, %0" :: "r"(p) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
}
/* Wait until every source slot (other than mine) carries `tok` in its trailer.  trailer_of(s) is the trailer offset
 * inside the slot of source s.  One pass invalidates all pending trailer lines, issues ONE dsb, then reads them. */
static int mtni_wait_all(size_t (*slot_off)(int), const size_t *trailer_of, uint64_t tok, int me, unsigned char *done_out) {
    const int n = glm53f_mtni_n;
    unsigned char done[GLM53F_COLLECTIVE_MAX_NODES];
    int pending = 0;
    for (int s = 0; s < n; ++s) { done[s] = (s == me); pending += (s != me); }
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long passes = 0;
    while (pending) {
        for (int s = 0; s < n; ++s)
            if (!done[s]) __asm__ __volatile__("dc civac, %0" :: "r"(glm53f_mtni_region + slot_off(s) + trailer_of[s]) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        for (int s = 0; s < n; ++s)
            if (!done[s] && *(volatile uint64_t *)(glm53f_mtni_region + slot_off(s) + trailer_of[s]) == tok) { done[s] = 1; --pending; }
        if ((++passes & 0x3FFF) == 0) {
            mtni_drain_mrq();
            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
            if ((t1.tv_sec - t0.tv_sec) > 30) return -1;
        }
    }
    (void)done_out;
    return 0;
}

static int glm53f_mtni_init(const uint8_t topology[][TOFU_NCOORDS], int rank, int ranks, int max_count,
                            const utofu_tni_id_t *tnis, size_t ntni) {
    int nt = (int)ntni < GLM53F_MTNI_MAXT ? (int)ntni : GLM53F_MTNI_MAXT;
    if (nt < 1 || ranks > GLM53F_COLLECTIVE_MAX_NODES) return -1;
    glm53f_mtni_n = ranks; glm53f_mtni_max = max_count;
    size_t shard = ((size_t)max_count / ranks + 1) * sizeof(float);
    glm53f_mtni_slot = (shard + 16 + 255) & ~(size_t)255;
    glm53f_mtni_rsize = (size_t)(3 * ranks + 1) * glm53f_mtni_slot;
    if (posix_memalign((void **)&glm53f_mtni_region, 256, glm53f_mtni_rsize)) return -1;
    memset(glm53f_mtni_region, 0, glm53f_mtni_rsize);
    for (size_t off = 0; off < glm53f_mtni_rsize; off += 256) __asm__ __volatile__("dc civac, %0" :: "r"(glm53f_mtni_region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    for (int k = 0; k < nt; ++k) {
        if (k == 0) glm53f_mtni_vcq[0] = glm53f_vcq; /* TNI 0 already has the collective's VCQ */
        else if (utofu_create_vcq_with_cmp_id(tnis[k], DEMO_CMP_ID, 0, &glm53f_mtni_vcq[k]) != UTOFU_SUCCESS) {
            if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "mtni: create_vcq tni %d failed\n", k);
            return -1;
        }
        if (utofu_reg_mem_with_stag(glm53f_mtni_vcq[k], glm53f_mtni_region, glm53f_mtni_rsize, GLM53F_MTNI_STAG, 0,
                                    &glm53f_mtni_base[k]) != UTOFU_SUCCESS) {
            if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "mtni: reg_mem tni %d failed\n", k);
            return -1;
        }
        for (int r = 0; r < ranks; ++r) {
            if (utofu_construct_vcq_id((uint8_t *)topology[r], tnis[k], DEMO_CQ_ID, DEMO_CMP_ID, &glm53f_mtni_peer[k][r]) != UTOFU_SUCCESS) return -1;
            utofu_set_vcq_id_path(&glm53f_mtni_peer[k][r], NULL);
        }
    }
    glm53f_mpi_barrier();
    for (int k = 0; k < nt; ++k)
        for (int r = 0; r < ranks; ++r) {
            if (r == rank) { glm53f_mtni_pbase[k][r] = glm53f_mtni_base[k]; continue; }
            if (utofu_query_stadd(glm53f_mtni_peer[k][r], GLM53F_MTNI_STAG, &glm53f_mtni_pbase[k][r]) != UTOFU_SUCCESS) return -1;
        }
    glm53f_mtni_nt = nt;
    glm53f_mtni_seq = 0;
    glm53f_mtni_active = 1;
    return 0;
}

/* Out-of-place or in-place sum all-reduce of count fp32 (count <= max_count, all ranks equal). Returns 0/-1. */
static int glm53f_mtni_allreduce(const float *input, float *output, int count) {
    const int n = glm53f_mtni_n, nt = glm53f_mtni_nt, me = glm53f_comm.my_rank;
    if (count < 1 || count > glm53f_mtni_max) return -1;
    const uint64_t tok = 0x5A5A5A5A00000000ull | (++glm53f_mtni_seq & 0xFFFFFFFFull);
    int off[GLM53F_COLLECTIVE_MAX_NODES + 1];
    off[0] = 0;
    for (int r = 0; r < n; ++r) off[r + 1] = off[r] + count / n + (r < count % n);
    int issued[GLM53F_MTNI_MAXT] = {0};
    mtni_drain_mrq();
    /* 1. scatter: shard d -> rank d */
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, len = off[d + 1] - off[d], k = idx % nt;
        const size_t pl = ((size_t)len * sizeof(float) + 7) & ~(size_t)7;
        char *sb = glm53f_mtni_region + mtni_send_off(d);
        memcpy(sb, input + off[d], (size_t)len * sizeof(float));
        *(volatile uint64_t *)(sb + pl) = tok;
        if (mtni_put(k, d, mtni_send_off(d), mtni_recv_off(me), pl + 8)) return -1;
        issued[k]++;
    }
    const int mylen = off[me + 1] - off[me];
    const size_t mypl = ((size_t)mylen * sizeof(float) + 7) & ~(size_t)7;
    size_t trl[GLM53F_COLLECTIVE_MAX_NODES];
    for (int s = 0; s < n; ++s) trl[s] = mypl;
    if (mtni_wait_all(mtni_recv_off, trl, tok, me, NULL)) return -1;
    /* 2. reduce my shard in fixed rank order into the gather-send slot (payload + trailer) */
    float *red = (float *)(glm53f_mtni_region + mtni_gsend_off());
#if defined(__ARM_FEATURE_SVE)
    for (int i = 0; i < mylen; i += 16) {
        const svbool_t pg = svwhilelt_b32(i, mylen);
        svfloat32_t acc = svdup_f32(0);
        for (int r = 0; r < n; ++r) {
            const float *src = r == me ? input + off[me] : (const float *)(glm53f_mtni_region + mtni_recv_off(r));
            acc = svadd_f32_x(pg, acc, svld1_f32(pg, src + i));
        }
        svst1_f32(pg, red + i, acc);
    }
#else
    for (int i = 0; i < mylen; ++i) {
        float acc = 0;
        for (int r = 0; r < n; ++r) acc += (r == me ? input + off[me] : (const float *)(glm53f_mtni_region + mtni_recv_off(r)))[i];
        red[i] = acc;
    }
#endif
    *(volatile uint64_t *)((char *)red + mypl) = tok;
    if (mtni_drain_tcq(issued)) return -1;
    memset(issued, 0, sizeof issued);
    /* 3. gather: my reduced shard -> every peer's grecv[me] */
    for (int j = 1, idx = 0; j < n; ++j, ++idx) {
        const int d = (me + j) % n, k = idx % nt;
        if (mtni_put(k, d, mtni_gsend_off(), mtni_grecv_off(me), mypl + 8)) return -1;
        issued[k]++;
    }
    memcpy(output + off[me], red, (size_t)mylen * sizeof(float));
    for (int s = 0; s < n; ++s) trl[s] = (((size_t)(off[s + 1] - off[s]) * sizeof(float)) + 7) & ~(size_t)7;
    if (mtni_wait_all(mtni_grecv_off, trl, tok, me, NULL)) return -1;
    for (int s = 0; s < n; ++s)
        if (s != me) memcpy(output + off[s], glm53f_mtni_region + mtni_grecv_off(s), (size_t)(off[s + 1] - off[s]) * sizeof(float));
    if (mtni_drain_tcq(issued)) return -1;
    mtni_drain_mrq();
    return 0;
}

int glm53f_collective_init_12n(const char *path, int max_count) {
    int rank, ranks, rc, topo_count = -1, physical_rank = -1;
    uint8_t topology[GLM53F_COLLECTIVE_MAX_NODES][TOFU_NCOORDS];
    uint8_t mine[TOFU_NCOORDS];
    utofu_tni_id_t *tnis = NULL;
    utofu_tni_id_t tni;
    size_t ntni = 0;
    utofu_vcq_id_t peers[GLM53F_COLLECTIVE_MAX_NODES];
    if (glm53f_utofu_active) return max_count <= glm53f_max_count ? 0 : -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (!path || ranks != 12 || max_count < 1 ||
            (topo_count = glm53f_read_topology(path, topology)) != ranks) {
        if (getenv("GLM53F_UTOFU_DEBUG"))
            fprintf(stderr, "GLM53F_UTOFU init preflight path=%s ranks=%d max=%d topo=%d\n",
                    path ? path : "(null)", ranks, max_count,
                    path ? topo_count : -1);
        return -1;
    }
    if (utofu_query_my_coords(mine) != UTOFU_SUCCESS) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU query coords failed rank=%d\n", rank);
        return -1;
    }
    for (int r = 0; r < ranks; r++)
        if (!memcmp(mine, topology[r], TOFU_NCOORDS)) physical_rank = r;
    if (physical_rank != rank) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU rank map failed rank=%d physical=%d\n", rank, physical_rank);
        return -1;
    }
    rc = utofu_get_onesided_tnis(&tnis, &ntni);
    if (rc != UTOFU_SUCCESS || ntni < 1) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU get tni failed rank=%d rc=%d ntni=%zu\n", rank, rc, ntni);
        free(tnis); return -1;
    }
    tni = tnis[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &glm53f_vcq);
    free(tnis);
    if (rc != UTOFU_SUCCESS) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU create vcq failed rank=%d rc=%d\n", rank, rc);
        return -1;
    }
    for (int r = 0; r < ranks; r++) {
        rc = utofu_construct_vcq_id(topology[r], tni, DEMO_CQ_ID, DEMO_CMP_ID,
                                    &peers[r]);
        if (rc != UTOFU_SUCCESS) {
            if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU construct peer failed rank=%d peer=%d rc=%d\n", rank, r, rc);
            return -1;
        }
        utofu_set_vcq_id_path(&peers[r], NULL);
    }
    if (getenv("GLM53F_UTOFU_2D")) {
        /* A=2 maps rank groups onto the allocation's 2x3x2 shape: six
         * contiguous ranks in each row group, then two group siblings. */
        if (tp_comm_init_2d(&glm53f_comm_row, &glm53f_comm_col,
                            glm53f_vcq, peers, rank, ranks, 2, max_count,
                            glm53f_mpi_barrier)) {
            if (getenv("GLM53F_UTOFU_DEBUG"))
                fprintf(stderr, "GLM53F_UTOFU tp_comm_init_2d failed rank=%d\n", rank);
            return -1;
        }
        glm53f_utofu_2d = 1;
    } else if (tp_comm_init(&glm53f_comm, glm53f_vcq, peers, rank, ranks,
                            max_count, glm53f_mpi_barrier)) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU tp_comm_init failed rank=%d\n", rank);
        return -1;
    }
    glm53f_utofu_active = 1;
    glm53f_max_count = max_count;
    if (!glm53f_utofu_2d && max_count > GLM53F_SMALL_COUNT && !(getenv("GLM53F_SMALL_COMM") && !atoi(getenv("GLM53F_SMALL_COMM")))) {
        if (!tp_comm_init_ex(&glm53f_comm_small, glm53f_vcq, peers, rank, ranks, GLM53F_SMALL_COUNT,
                             glm53f_mpi_barrier, TP_AR_STAG2)) glm53f_small_active = 1;
        else if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU small comm init failed rank=%d\n", rank);
    }
    if (!glm53f_utofu_2d && !(getenv("GLM53F_MTNI") && !atoi(getenv("GLM53F_MTNI")))) {
        utofu_tni_id_t *tl = NULL; size_t nl = 0;
        if (utofu_get_onesided_tnis(&tl, &nl) == UTOFU_SUCCESS) {
            if (glm53f_mtni_init((const uint8_t (*)[TOFU_NCOORDS])topology, rank, ranks, max_count, tl, nl))
                { glm53f_mtni_active = 0; if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU mtni init failed rank=%d\n", rank); }
            free(tl);
        }
    }
    if (!rank) fprintf(stderr, "GLM53F_COLLECTIVE mode=utofu max_count=%d mtni=%d(x%d) small=%d\n", max_count, glm53f_mtni_active, glm53f_mtni_nt, glm53f_small_active);
    return 0;
}

void glm53f_collective_free_12n(void) {
    glm53f_prefill_algorithm = 0;
    if (!glm53f_utofu_active) return;
    if (glm53f_mtni_active) { for (int k = 0; k < glm53f_mtni_nt; ++k) { utofu_dereg_mem(glm53f_mtni_vcq[k], glm53f_mtni_base[k], 0); if (k) utofu_free_vcq(glm53f_mtni_vcq[k]); } free(glm53f_mtni_region); glm53f_mtni_active = 0; }
    if (glm53f_small_active) { tp_comm_free(&glm53f_comm_small); glm53f_small_active = 0; }
    if (glm53f_utofu_2d) {
        tp_comm_free(&glm53f_comm_col);
        tp_comm_free(&glm53f_comm_row);
        glm53f_utofu_2d = 0;
    } else {
        tp_comm_free(&glm53f_comm);
    }
    utofu_free_vcq(glm53f_vcq);
    glm53f_utofu_active = 0;
    glm53f_max_count = 0;
    glm53f_prefill_algorithm = 0;
}

int glm53f_collective_is_utofu_12n(void) { return glm53f_utofu_active; }
int glm53f_collective_capacity_12n(void) {
    return glm53f_utofu_active ? glm53f_max_count : INT_MAX;
}

int glm53f_collective_prefill_algorithm_12n(int algorithm) {
    if (algorithm < 0 || algorithm > 5) return -1;
    if (algorithm == 5 && !glm53f_mtni_active) return -1;
    if (algorithm >= 2 && algorithm <= 4 && (!glm53f_utofu_active || glm53f_utofu_2d ||
        glm53f_comm.use_bf16 || glm53f_comm.nprocs != 12)) return -1;
    if (algorithm == 4 && (glm53f_comm.ack || glm53f_comm.drop_n)) return -1;
    glm53f_prefill_algorithm = algorithm;
    return 0;
}

/* Right-align a shortened payload against the FIXED sequence trailer. This
 * gives every tree stage one contiguous Put without moving the trailer into
 * a region that a previous larger payload could have overwritten. */
static void glm53f_tree_send(tp_comm *c, int peer, int sid, const float *buf,
                             int count, uint64_t token) {
    if (glm53f_prefill_algorithm != 4) { tp_ar_send(c,peer,sid,buf,count,token); return; }
    size_t bytes=(size_t)count*sizeof(float), trailer=tp_ar_trailer_off(c);
    size_t start=trailer-bytes;
    memcpy(c->region+start,buf,bytes);
    *(volatile uint64_t *)(c->region+trailer)=token;
    c->send_inflight += tp_ar_put_nb(c,peer,c->base+start,
        c->peer_base[peer]+tp_ar_slot_off(c,1+sid)+start,bytes+sizeof(uint64_t));
}

static void glm53f_tree_recv(tp_comm *c, int sid, int peer, float *buf,
                             int count, uint64_t token, int add) {
    if (glm53f_prefill_algorithm != 4) {
        if (add) tp_ar_recv_add(c,sid,peer,buf,count,token);
        else tp_ar_recv_copy(c,sid,peer,buf,count,token);
        return;
    }
    size_t bytes=(size_t)count*sizeof(float), trailer=tp_ar_trailer_off(c);
    char *slot=c->region+tp_ar_slot_off(c,1+sid);
    tp_ar_wait(c,(volatile uint64_t *)(slot+trailer),token,sid,"prefill packed tree");
    const float *received=(const float *)(slot+trailer-bytes);
    if (add) for (int i=0;i<count;++i) buf[i]+=received[i];
    else memcpy(buf,received,bytes);
}

/* Same FP32 tree as tp_allreduce_sum: prefold, XOR masks 1/2/4, unfold.
 * Halving in that order assigns bit-reversed chunks to the survivors, which
 * the reverse-mask allgather reconstructs. Reversing the reduction masks to
 * the usual 4/2/1 would change the numerical tree and is deliberately avoided. */
static int glm53f_prefill_tree_reduce_gather(float *buf, int count) {
    tp_comm *c = &glm53f_comm;
    uint64_t token = ++c->seq;
    int rank = c->my_rank, rem = c->rem;
    if (count < c->pof2 || 2*c->nrounds+1 >= TP_AR_NSTEP) return -1;
    if (rank < 2*rem) {
        if (!(rank & 1)) { glm53f_tree_send(c, rank+1, 0, buf, count, token); tp_ar_confirm(c); }
        else glm53f_tree_recv(c, 0, rank-1, buf, count, token, 1);
    }
    if (c->newrank != -1) {
        int lo=0, n=count, send_lo[TP_AR_NSTEP], send_n[TP_AR_NSTEP];
        int keep_lo[TP_AR_NSTEP], keep_n[TP_AR_NSTEP];
        for (int k=0;k<c->nrounds;++k) {
            int half=n/2, partner=c->newrank^(1<<k);
            int peer=partner<rem ? partner*2+1 : partner+rem;
            if (c->newrank & (1<<k)) {
                send_lo[k]=lo; send_n[k]=half;
                keep_lo[k]=lo+half; keep_n[k]=n-half;
            } else {
                send_lo[k]=lo+half; send_n[k]=n-half;
                keep_lo[k]=lo; keep_n[k]=half;
            }
            glm53f_tree_send(c,peer,k+1,buf+send_lo[k],send_n[k],token);
            glm53f_tree_recv(c,k+1,peer,buf+keep_lo[k],keep_n[k],token,1);
            tp_ar_confirm(c);
            lo=keep_lo[k]; n=keep_n[k];
        }
        for (int k=c->nrounds-1;k>=0;--k) {
            int partner=c->newrank^(1<<k);
            int peer=partner<rem ? partner*2+1 : partner+rem;
            int sid=2*c->nrounds-k;
            glm53f_tree_send(c,peer,sid,buf+keep_lo[k],keep_n[k],token);
            glm53f_tree_recv(c,sid,peer,buf+send_lo[k],send_n[k],token,0);
            tp_ar_confirm(c);
        }
    }
    int sid=2*c->nrounds+1;
    if (rank < 2*rem) {
        if (!(rank & 1)) glm53f_tree_recv(c,sid,rank+1,buf,count,token,0);
        else { glm53f_tree_send(c,rank-1,sid,buf,count,token); tp_ar_confirm(c); }
    }
    return 0;
}

static int glm53f_prefill_reduce_gather(const float *input, float *output,
                                       int count, int algorithm) {
    if (algorithm == 5) return glm53f_mtni_allreduce(input, output, count);
    int rank, ranks, sizes[GLM53F_COLLECTIVE_MAX_NODES], offsets[GLM53F_COLLECTIVE_MAX_NODES];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks > GLM53F_COLLECTIVE_MAX_NODES || count < ranks) return -1;
    int off = 0;
    for (int r = 0; r < ranks; ++r) {
        sizes[r] = count / ranks + (r < count % ranks);
        offsets[r] = off; off += sizes[r];
    }
    if (algorithm == 1) {
        /* Bounded stack scratch: at most 32*4096/12+1 FP32 elements. */
        float slice[32 * 4096 / 12 + 1];
        if (sizes[rank] > (int)(sizeof(slice) / sizeof(slice[0]))) return -1;
        if (MPI_Reduce_scatter(input, slice, sizes, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
            MPI_Allgatherv(slice, sizes[rank], MPI_FLOAT, output, sizes, offsets, MPI_FLOAT,
                           MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
        return 0;
    }
    tp_comm *c = &glm53f_comm;
    if (input != output) memcpy(output, input, (size_t)count * sizeof(float));
    if (algorithm >= 3) return glm53f_prefill_tree_reduce_gather(output, count);
    int next = (rank + 1) % ranks, prev = (rank + ranks - 1) % ranks;
    /* Eleven distinct receive slots prevent a fast ring neighbor overwriting
     * an unread chunk. The phase barriers are required before slot reuse and
     * before returning to the legacy decode collectives; do not remove them. */
    uint64_t token = ++c->seq;
    for (int step = 0; step < ranks - 1; ++step) {
        int send = (rank - step + ranks) % ranks;
        int recv = (rank - step - 1 + ranks) % ranks;
        tp_ar_send(c, next, step, output + offsets[send], sizes[send], token);
        tp_ar_recv_add(c, step, prev, output + offsets[recv], sizes[recv], token);
        tp_ar_confirm(c);
    }
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
    token = ++c->seq;
    for (int step = 0; step < ranks - 1; ++step) {
        int send = (rank - step + 1 + ranks) % ranks;
        int recv = (rank - step + ranks) % ranks;
        tp_ar_send(c, next, step, output + offsets[send], sizes[send], token);
        tp_ar_recv_copy(c, step, prev, output + offsets[recv], sizes[recv], token);
        tp_ar_confirm(c);
    }
    return MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
}

/* Large-message MPI_Allreduce (no uTofu buffer capacity limit).  At 128-512 tokens per call this is faster than the
 * reduce-scatter + allgatherv wrapper; at 32 tokens it is slower (see bench_glm53f_allreduce_12n). */
int glm53f_sum_allreduce_mpi_12n(const float *input, float *output, int count) {
    if (!input || !output || count < 1) return -1;
    return MPI_Allreduce(input == output ? MPI_IN_PLACE : input, output, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
}

int glm53f_sum_allreduce_slabs_12n(const float *input, float *output,
                                  int tokens, int width, int slab_tokens) {
    if (!input || !output || tokens < 1 || width < 1 || slab_tokens < 1) return -1;
    int available = glm53f_collective_capacity_12n() / width;
    if (available < 1) return -1;
    if (slab_tokens > available) slab_tokens = available;
    for (int t = 0; t < tokens;) {
        int n = tokens - t < slab_tokens ? tokens - t : slab_tokens;
        int rc = n > 5 && glm53f_prefill_algorithm ?
            glm53f_prefill_reduce_gather(input + (size_t)t * width,
                output + (size_t)t * width, n * width, glm53f_prefill_algorithm) :
            glm53f_sum_allreduce_12n(input + (size_t)t * width, output + (size_t)t * width, n * width);
        if (rc) return -1;
        t += n; /* Avoid overshooting/overflow on a final partial slab. */
    }
    return 0;
}

/* Asynchronous slab reduction on a dedicated (spare) core: the producer marks tokens ready while it keeps
 * computing; the helper reduces each slab as soon as it is complete. Only usable with the uTofu collectives
 * (multi-TNI / recursive doubling); the producer must not call any collective between begin and finish. */
static struct {
    pthread_t th; int started, width, slab, total, rc;
    const float *in; float *out;
    _Atomic int go, stop, ready, done;
} glm53f_async;
static int glm53f_async_pick_core(void) {
    const char *e = getenv("GLM53F_ASYNC_CORE");
    if (e && *e) return atoi(e);
    cpu_set_t set; int hi = -1;
    CPU_ZERO(&set);
    if (!sched_getaffinity(0, sizeof set, &set))
        for (int c = 0; c < 512; ++c) if (CPU_ISSET(c, &set)) hi = c;
    return hi;
}
static void *glm53f_async_main(void *arg) {
    (void)arg;
    int core = glm53f_async_pick_core();
    if (core >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    for (;;) {
        while (!atomic_load_explicit(&glm53f_async.go, memory_order_acquire) &&
               !atomic_load_explicit(&glm53f_async.stop, memory_order_acquire))
            __asm__ __volatile__("yield" ::: "memory");
        if (atomic_load_explicit(&glm53f_async.stop, memory_order_acquire)) break;
        atomic_store_explicit(&glm53f_async.go, 0, memory_order_relaxed);
        /* Snapshot the job: the producer may publish the next job the moment done reaches total. */
        const int total = glm53f_async.total, slab = glm53f_async.slab, width = glm53f_async.width;
        const float *in = glm53f_async.in; float *out = glm53f_async.out;
        int t = 0, rc = 0;
        while (t < total) {
            int n = total - t < slab ? total - t : slab;
            while (atomic_load_explicit(&glm53f_async.ready, memory_order_acquire) < t + n)
                __asm__ __volatile__("yield" ::: "memory");
            const size_t off = (size_t)t * width;
            if (!rc) rc = n > 5 && glm53f_prefill_algorithm ?
                glm53f_prefill_reduce_gather(in + off, out + off, n * width, glm53f_prefill_algorithm) :
                glm53f_sum_allreduce_12n(in + off, out + off, n * width);
            t += n;
            if (t == total) glm53f_async.rc = rc; /* before publishing completion */
            atomic_store_explicit(&glm53f_async.done, t, memory_order_release);
        }
    }
    return NULL;
}
int glm53f_async_available_12n(void) { return glm53f_utofu_active && glm53f_prefill_algorithm == 5; }
int glm53f_async_begin_12n(const float *input, float *output, int tokens, int width, int slab_tokens) {
    if (!glm53f_async_available_12n() || !input || !output || tokens < 1 || width < 1) return -1;
    int available = glm53f_collective_capacity_12n() / width;
    if (available < 1) return -1;
    if (!glm53f_async.started) {
        if (pthread_create(&glm53f_async.th, NULL, glm53f_async_main, NULL)) return -1;
        glm53f_async.started = 1;
    }
    glm53f_async.in = input; glm53f_async.out = output; glm53f_async.width = width;
    glm53f_async.total = tokens; glm53f_async.rc = 0;
    glm53f_async.slab = slab_tokens > available ? available : slab_tokens;
    atomic_store_explicit(&glm53f_async.ready, 0, memory_order_relaxed);
    atomic_store_explicit(&glm53f_async.done, 0, memory_order_relaxed);
    if (getenv("GLM53F_ASYNC_TRACE")) {
        static int calls; int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
        fprintf(stderr, "ASYNC begin rank=%d call=%d tokens=%d slab=%d\n", r, calls++, tokens, glm53f_async.slab);
    }
    atomic_store_explicit(&glm53f_async.go, 1, memory_order_release);
    return 0;
}
void glm53f_async_ready_12n(int tokens_ready) {
    atomic_store_explicit(&glm53f_async.ready, tokens_ready, memory_order_release);
}
int glm53f_async_finish_12n(void) {
    static int trace = -1, calls;
    if (trace < 0) trace = getenv("GLM53F_ASYNC_TRACE") != NULL;
    while (atomic_load_explicit(&glm53f_async.done, memory_order_acquire) < glm53f_async.total)
        __asm__ __volatile__("yield" ::: "memory");
    if (trace) { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); fprintf(stderr, "ASYNC finish rank=%d call=%d\n", r, calls++); }
    return glm53f_async.rc;
}

int glm53f_sum_allreduce_12n(const float *input, float *output, int count) {
    if (!input || !output || count < 1 || count > glm53f_collective_capacity_12n()) return -1;
    if (!glm53f_utofu_active) {
        /* Optional payload-sharded experiment.  Reduce-scatter computes each
         * rank's disjoint output slice, then an allgatherv reconstructs the
         * full vector.  The default MPI_Allreduce path remains unchanged. */
        if (getenv("GLM53F_SPLIT_AR")) {
            int rank, nr, rc[GLM53F_COLLECTIVE_MAX_NODES], ds[GLM53F_COLLECTIVE_MAX_NODES];
            static float *slice;
            static int slice_cap;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nr);
            if (nr > GLM53F_COLLECTIVE_MAX_NODES) return -1;
            int off = 0;
            for (int r = 0; r < nr; ++r) {
                rc[r] = count / nr + (r < count % nr);
                ds[r] = off;
                off += rc[r];
            }
            if (slice_cap < rc[rank]) {
                float *p = realloc(slice, (size_t)rc[rank] * sizeof(*slice));
                if (!p) return -1;
                slice = p;
                slice_cap = rc[rank];
            }
            if (MPI_Reduce_scatter(input, slice, rc, MPI_FLOAT, MPI_SUM,
                                   MPI_COMM_WORLD) != MPI_SUCCESS ||
                MPI_Allgatherv(slice, rc[rank], MPI_FLOAT, output, rc, ds,
                               MPI_FLOAT, MPI_COMM_WORLD) != MPI_SUCCESS)
                return -1;
            return 0;
        }
        if (input == output || getenv("GLM53F_MPI_INPLACE")) {
            if (input != output)
                memcpy(output, input, (size_t)count * sizeof(float));
            return MPI_Allreduce(MPI_IN_PLACE, output, count, MPI_FLOAT,
                                 MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
        }
        return MPI_Allreduce(input, output, count, MPI_FLOAT, MPI_SUM,
                             MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
    }
    if (input != output) memcpy(output, input, (size_t)count * sizeof(float));
    static int mtni_decode = -1;
    if (mtni_decode < 0) mtni_decode = getenv("GLM53F_MTNI_DECODE") ? atoi(getenv("GLM53F_MTNI_DECODE")) : 1;
    if (glm53f_mtni_active && mtni_decode && count <= glm53f_mtni_max && count >= glm53f_mtni_n)
        return glm53f_mtni_allreduce(output, output, count);
    if (glm53f_utofu_2d)
        tp_allreduce_sum_2d(&glm53f_comm_row, &glm53f_comm_col, output, count);
    else if (glm53f_small_active && count == GLM53F_SMALL_COUNT)
        tp_allreduce_sum(&glm53f_comm_small, output, count);
    else
        tp_allreduce_sum(&glm53f_comm, output, count);
    return 0;
}

int glm53f_sum_allreduce_prefill_12n(const float *input, float *output, int count) {
    if (!input || !output || count < 1 || count > glm53f_collective_capacity_12n()) return -1;
    if (count > 5 && glm53f_prefill_algorithm)
        return glm53f_prefill_reduce_gather(input, output, count, glm53f_prefill_algorithm);
    return glm53f_sum_allreduce_12n(input, output, count);
}
