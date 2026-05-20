/*
 * tofu_put_demo - MPI-free uTofu one-sided Put exchange across nodes.
 *
 * This binary makes ZERO MPI calls. It is placed on N nodes by `mpiexec -np N`
 * purely as a launcher; each process then:
 *   1. reads tofu_topo.txt (written by tofu_topo_helper) to learn every node's
 *      Tofu coordinates,
 *   2. identifies itself by matching its own coordinates,
 *   3. reconstructs its ring peer's VCQ ID from the peer's coordinates plus the
 *      conventional (tni, cq, cmp) indices in tofu_demo.h -- no exchange,
 *   4. Puts a small payload into the peer's buffer and validates what it
 *      receives from its other-side neighbor.
 *
 * Ring topology: peer = (rank + 1) % nprocs. Validated for 2..12 nodes; the
 * logic is N-agnostic (capped only by MAX_NODES). In a ring each node's recv
 * buffer has exactly one writer -- its (rank-1) neighbor -- so a single recv
 * slot suffices at any N. Assumes node topology is fixed for the job's duration.
 *
 * Build (NO -lmpi). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o tofu_put_demo tofu_put_demo.c -ltofucom
 *   (login/cross node: use fccpx instead of fcc)
 *
 * Run (inside a pjsub node=N allocation, after running tofu_topo_helper):
 *   mpiexec -np <N> ./tofu_put_demo
 */
#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utofu.h>

#include "tofu_demo.h"

#define MAX_NODES 256
#define SPIN_TIMEOUT 20000000000ULL  /* spin-loop guard so we never hang */

/* MPI launchers on Fugaku redirect rank stdout/stderr away from the terminal,
 * so every rank also logs to its own file (named by its coordinates) in cwd. */
static FILE *g_log = NULL;

static void logmsg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    if (g_log) { vfprintf(g_log, fmt, ap); fflush(g_log); }
    vfprintf(stdout, fmt, ap);
    fflush(stdout);
    va_end(ap);
}

static void die(const char *what, int rc)
{
    logmsg("FATAL: %s (rc=%d)\n", what, rc);
    exit(1);
}

/* Parse tofu_topo.txt into coords[][6]; returns node count, fills *out_n. */
static int read_topo(uint8_t coords[][TOFU_NCOORDS])
{
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) {
        perror("cannot open " TOPO_PATH);
        fprintf(stderr, "  (run tofu_topo_helper first)\n");
        exit(1);
    }
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        if (n >= MAX_NODES) {
            fprintf(stderr, "too many nodes in %s (max %d)\n", TOPO_PATH, MAX_NODES);
            exit(1);
        }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u",
                   &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7) {
            fprintf(stderr, "malformed line in %s: %s", TOPO_PATH, line);
            exit(1);
        }
        if ((int)r != n) {
            fprintf(stderr, "%s ranks not in order (expected %d, got %u)\n",
                    TOPO_PATH, n, r);
            exit(1);
        }
        for (int k = 0; k < TOFU_NCOORDS; k++)
            coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 2) {
        fprintf(stderr, "%s lists %d node(s); need >= 2\n", TOPO_PATH, n);
        exit(1);
    }
    return n;
}

int main(void)
{
    int rc;

    /* --- pick the one-sided TNI by the shared convention --- */
    utofu_tni_id_t *tni_ids = NULL;
    size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS || num_tnis <= DEMO_TNI_INDEX)
        die("utofu_get_onesided_tnis", rc);
    utofu_tni_id_t tni = tni_ids[DEMO_TNI_INDEX];
    free(tni_ids);

    /* --- learn my own coordinates and find my rank in the topology file --- */
    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS)
        die("utofu_query_my_coords", rc);

    {
        char name[64];
        snprintf(name, sizeof(name), "demo_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2],
                 my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }
    logmsg("coords=%u,%u,%u,%u,%u,%u tni=%u\n",
           my_coords[0], my_coords[1], my_coords[2],
           my_coords[3], my_coords[4], my_coords[5], (unsigned)tni);

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    int nprocs = read_topo(topo);

    int my_rank = -1;
    for (int r = 0; r < nprocs; r++) {
        if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) {
            if (my_rank != -1) {
                fprintf(stderr, "FATAL: my coords match >1 row in %s "
                        "(stale/foreign file?)\n", TOPO_PATH);
                exit(1);
            }
            my_rank = r;
        }
    }
    if (my_rank == -1) {
        fprintf(stderr, "FATAL: my coords (%u,%u,%u,%u,%u,%u) not in %s "
                "(stale/foreign file?)\n",
                my_coords[0], my_coords[1], my_coords[2],
                my_coords[3], my_coords[4], my_coords[5], TOPO_PATH);
        exit(1);
    }

    int peer_rank = (my_rank + 1) % nprocs;
    uint8_t peer_coords[TOFU_NCOORDS];
    memcpy(peer_coords, topo[peer_rank], TOFU_NCOORDS);
    logmsg("my_rank=%d peer_rank=%d nprocs=%d\n", my_rank, peer_rank, nprocs);

    /* --- create my VCQ with the conventional component id --- */
    utofu_vcq_hdl_t vcq;
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcq);
    if (rc != UTOFU_SUCCESS)
        die("utofu_create_vcq_with_cmp_id", rc);

    utofu_vcq_id_t my_vcq_real;
    rc = utofu_query_vcq_id(vcq, &my_vcq_real);
    if (rc != UTOFU_SUCCESS)
        die("utofu_query_vcq_id", rc);

    /* --- VCQ self-check: prove the (tni, cq, cmp) convention reproduces my own
     * VCQ ID. By symmetry the same construction applied to a peer's coords is
     * then correct. cq_id is the only system-assigned value; this makes a wrong
     * assumption abort loudly instead of corrupting remote memory. --- */
    {
        uint8_t got_coords[TOFU_NCOORDS];
        utofu_tni_id_t got_tni = 0;
        utofu_cq_id_t got_cq = 0;
        uint16_t extra = 0;
        rc = utofu_query_vcq_info(my_vcq_real, got_coords, &got_tni, &got_cq, &extra);
        if (rc != UTOFU_SUCCESS)
            die("utofu_query_vcq_info", rc);

        utofu_vcq_id_t my_vcq_conv;
        rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &my_vcq_conv);
        if (rc != UTOFU_SUCCESS)
            die("utofu_construct_vcq_id(self)", rc);

        /* Normalize path bits on both before comparing identity bits only. */
        utofu_vcq_id_t a = my_vcq_real, b = my_vcq_conv;
        utofu_set_vcq_id_path(&a, NULL);
        utofu_set_vcq_id_path(&b, NULL);
        logmsg("vcq self-check: real=0x%016lx norm=0x%016lx conv=0x%016lx "
               "(got_tni=%u got_cq=%u)\n",
               (unsigned long)my_vcq_real, (unsigned long)a, (unsigned long)b,
               (unsigned)got_tni, (unsigned)got_cq);
        if (a != b) {
            fprintf(stderr,
                "SELF-CHECK FAILED on rank %d: assumed indices do not reproduce "
                "my own VCQ ID.\n"
                "  real vcq_id=0x%016lx (tni=%u cq=%u)\n"
                "  conv vcq_id=0x%016lx (assumed tni=%u cq=%u cmp=%u)\n"
                "  -> peers would target the WRONG VCQ; aborting.\n",
                my_rank,
                (unsigned long)a, (unsigned)got_tni, (unsigned)got_cq,
                (unsigned long)b, (unsigned)tni, DEMO_CQ_ID, DEMO_CMP_ID);
            abort();
        }
    }

    /* --- register the communication region with the conventional stag ---
     * Cache-line aligned so .recv starts its own line (see tofu_demo.h). */
    struct demo_region region __attribute__((aligned(DEMO_CACHE_LINE)));
    memset((void *)&region, 0, sizeof(region));

    utofu_stadd_t base_stadd;
    rc = utofu_reg_mem_with_stag(vcq, &region, sizeof(region), DEMO_STAG, 0, &base_stadd);
    if (rc != UTOFU_SUCCESS)
        die("utofu_reg_mem_with_stag", rc);

    /* STADD self-check: query_stadd on my own VCQ must match what reg returned,
     * so deriving the remote STADD the same way is trustworthy. */
    {
        utofu_stadd_t chk;
        rc = utofu_query_stadd(my_vcq_real, DEMO_STAG, &chk);
        if (rc != UTOFU_SUCCESS)
            die("utofu_query_stadd(self)", rc);
        if (chk != base_stadd) {
            fprintf(stderr, "SELF-CHECK FAILED on rank %d: query_stadd=0x%lx != "
                    "reg stadd=0x%lx\n", my_rank,
                    (unsigned long)chk, (unsigned long)base_stadd);
            abort();
        }
    }

    /* --- reconstruct the peer's VCQ ID and remote receive STADD (no exchange) --- */
    utofu_vcq_id_t peer_vcq;
    rc = utofu_construct_vcq_id(peer_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &peer_vcq);
    if (rc != UTOFU_SUCCESS)
        die("utofu_construct_vcq_id(peer)", rc);
    utofu_set_vcq_id_path(&peer_vcq, NULL);  /* use the default communication path */

    utofu_stadd_t peer_base;
    rc = utofu_query_stadd(peer_vcq, DEMO_STAG, &peer_base);
    if (rc != UTOFU_SUCCESS)
        die("utofu_query_stadd(peer)", rc);

    utofu_stadd_t send_stadd      = base_stadd + offsetof(struct demo_region, send);
    utofu_stadd_t peer_recv_stadd = peer_base  + offsetof(struct demo_region, recv);
    logmsg("base_stadd=0x%lx peer_vcq=0x%016lx peer_base=0x%lx "
           "send_stadd=0x%lx peer_recv_stadd=0x%lx\n",
           (unsigned long)base_stadd, (unsigned long)peer_vcq,
           (unsigned long)peer_base, (unsigned long)send_stadd,
           (unsigned long)peer_recv_stadd);

    /* --- exchange + verify, with no barrier available ---
     * No barrier exists, so when we issue our first Put we cannot assume the
     * peer has finished registering its recv buffer; an early Put can be
     * dropped. We therefore (a) Put on EVERY iteration so dropped Puts are
     * retried, and (b) keep going until we have BOTH received the peer's
     * payload AND issued at least MIN_PUTS Puts. Receiving proves the peer is
     * up and registered; the extra Puts then guarantee a late-registering peer
     * still gets ours before we exit and deregister.
     *
     * Correctness of the recv sentinel: recv was zeroed by memset() BEFORE
     * registration (no Put could have landed earlier) and the CPU never writes
     * recv afterwards -- and recv sits on its own cache line (tofu_demo.h) so a
     * landed Put is never clobbered by a CPU-dirty line. We poll memory rather
     * than the remote MRQ, so repeated Puts leave no unread notices behind. */
    region.send.magic = (uint64_t)(DEMO_MAGIC_BASE | (unsigned)my_rank);
    memcpy(region.send.coords, my_coords, TOFU_NCOORDS);

    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    const int MIN_PUTS = 8;       /* min Puts before exit: covers the startup race */
    const int MAX_ATTEMPTS = 200; /* x 50 ms = 10 s hard cap so we never hang */
    void *cb;
    int attempts = 0, recv_done = 0;

    for (; attempts < MAX_ATTEMPTS; attempts++) {
        /* (re)issue the Put to the peer's recv buffer; drain TCQ on busy */
        for (;;) {
            rc = utofu_put(vcq, peer_vcq, send_stadd, peer_recv_stadd,
                           sizeof(struct msg), 0, flags, NULL);
            if (rc != UTOFU_ERR_BUSY)
                break;
            utofu_poll_tcq(vcq, 0, &cb);
        }
        if (rc != UTOFU_SUCCESS)
            die("utofu_put", rc);
        /* wait for this Put to leave the local TNI */
        do {
            rc = utofu_poll_tcq(vcq, 0, &cb);
        } while (rc == UTOFU_ERR_NOT_FOUND);
        if (rc != UTOFU_SUCCESS)
            die("utofu_poll_tcq", rc);

        if (region.recv.magic != 0)
            recv_done = 1;
        if (recv_done && attempts + 1 >= MIN_PUTS) {
            attempts++;
            break;
        }
        usleep(50000); /* 50 ms: let the peer register / respond */
    }
    if (!recv_done)
        die("exchange timeout (peer payload never arrived)", -1);
    logmsg("recv arrived (after %d put attempt(s)): magic=0x%lx\n",
           attempts, (unsigned long)region.recv.magic);

    /* validate payload: the sender is my (rank-1) neighbor in the ring */
    int from_rank = (my_rank - 1 + nprocs) % nprocs;
    uint64_t expect = (uint64_t)(DEMO_MAGIC_BASE | (unsigned)from_rank);
    if (region.recv.magic != expect ||
        memcmp((const void *)region.recv.coords, topo[from_rank], TOFU_NCOORDS) != 0) {
        logmsg("rank %d: VERIFY FAILED: got magic=0x%lx coords=(%u,%u,%u,%u,%u,%u), "
            "expected magic=0x%lx from rank %d\n",
            my_rank, (unsigned long)region.recv.magic,
            region.recv.coords[0], region.recv.coords[1], region.recv.coords[2],
            region.recv.coords[3], region.recv.coords[4], region.recv.coords[5],
            (unsigned long)expect, from_rank);
        abort();
    }

    logmsg("rank %d: received from peer rank %d OK (magic=0x%lx, coords=%u,%u,%u,%u,%u,%u)\n",
           my_rank, from_rank, (unsigned long)region.recv.magic,
           region.recv.coords[0], region.recv.coords[1], region.recv.coords[2],
           region.recv.coords[3], region.recv.coords[4], region.recv.coords[5]);

    utofu_dereg_mem(vcq, base_stadd, 0);
    utofu_free_vcq(vcq);
    return 0;
}
