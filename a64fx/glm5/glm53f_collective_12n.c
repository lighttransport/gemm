#define _GNU_SOURCE
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <utofu.h>
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"
#include "glm53f_collective_12n.h"

enum { GLM53F_COLLECTIVE_MAX_NODES = 32 };
static tp_comm glm53f_comm;
static tp_comm glm53f_comm_row, glm53f_comm_col;
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
    if (!rank) fprintf(stderr, "GLM53F_COLLECTIVE mode=utofu max_count=%d\n", max_count);
    return 0;
}

void glm53f_collective_free_12n(void) {
    glm53f_prefill_algorithm = 0;
    if (!glm53f_utofu_active) return;
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
    if (algorithm < 0 || algorithm > 4) return -1;
    if (algorithm >= 2 && (!glm53f_utofu_active || glm53f_utofu_2d ||
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
    if (glm53f_utofu_2d)
        tp_allreduce_sum_2d(&glm53f_comm_row, &glm53f_comm_col, output, count);
    else
        tp_allreduce_sum(&glm53f_comm, output, count);
    return 0;
}
