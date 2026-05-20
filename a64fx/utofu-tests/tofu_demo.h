/*
 * Shared definitions for the MPI-free uTofu multi-node Put demo.
 *
 * Two programs share this header:
 *   tofu_topo_helper.c  - MPI + uTofu, writes the topology file once.
 *   tofu_put_demo.c     - pure uTofu (no MPI), reads the file and communicates.
 *
 * The whole design hinges on every node running the IDENTICAL binary and
 * creating its uTofu resources with the SAME conventional indices below, so a
 * peer's VCQ ID can be reconstructed from just its 6D Tofu coordinates via
 * utofu_construct_vcq_id() -- no runtime VCQ-ID/STADD exchange needed.
 */
#ifndef TOFU_DEMO_H
#define TOFU_DEMO_H

#include <stdint.h>

/* Conventional, fixed-by-agreement uTofu indices used on every node. */
#define DEMO_TNI_INDEX 0          /* index into utofu_get_onesided_tnis() list */
#define DEMO_CMP_ID    0          /* component id (we pick it, so it is fixed)  */
#define DEMO_CQ_ID     0          /* assumed cq id; validated by self-check     */
#define DEMO_STAG      1          /* steering tag -> predictable STADD (avoid 0)*/

#define DEMO_MAGIC_BASE 0xC0FFEE00u  /* payload magic = base | sender_rank */
#define TOPO_PATH       "tofu_topo.txt"

#define TOFU_NCOORDS 6            /* Tofu network coordinates: x,y,z,a,b,c */
#define DEMO_CACHE_LINE 256       /* A64FX cache line (uTofu caps cache_line_size) */

/* 16-byte, 8-byte-aligned payload exchanged between nodes. */
struct msg {
    uint8_t  coords[TOFU_NCOORDS];  /* sender's Tofu coordinates */
    uint8_t  pad[2];
    uint64_t magic;                 /* DEMO_MAGIC_BASE | sender_rank */
};

/*
 * One registered region per node, covered by a single stag. The peer Puts into
 * .recv; we fill .send locally then Put it into the peer's .recv.
 *
 * recv and send sit on SEPARATE cache lines. The CPU writes .send (dirtying its
 * line) and spin-reads .recv; the peer's RDMA write lands in .recv. If both
 * shared one cache line, the CPU-held dirty line would clobber the incoming
 * RDMA write -- so .recv must be a line the CPU only ever reads.
 */
struct demo_region {
    volatile struct msg recv __attribute__((aligned(DEMO_CACHE_LINE)));
    struct msg          send __attribute__((aligned(DEMO_CACHE_LINE)));
};

#endif /* TOFU_DEMO_H */
