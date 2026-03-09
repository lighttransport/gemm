/*
 * comm.h - Lightweight collective communication library (no MPI dependency)
 *
 * TCP sockets for inter-process transport + POSIX shared memory for
 * same-node fast-path allreduce. Designed for distributed LLM inference.
 *
 * Supports: allreduce (sum), send/recv, broadcast, barrier, allgather, split.
 *
 * Usage:
 *   #define COMM_IMPLEMENTATION
 *   #include "comm.h"
 *
 * Launch N processes manually or with launch.sh:
 *   ./launch.sh 4 ./program <args>
 * Each process receives --comm-rank, --comm-nranks, --comm-addr, --comm-port.
 */
#ifndef COMM_H
#define COMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct comm_context comm_context;

/* Create a communicator. All ranks must call simultaneously.
 * rank: [0, world_size). world_size: total processes.
 * master_addr: IP/hostname of rank 0. master_port: TCP port for bootstrap.
 * Returns NULL on error. */
comm_context *comm_create(int rank, int world_size,
                           const char *master_addr, int master_port);

/* Point-to-point (blocking) */
void comm_send(comm_context *ctx, const void *buf, size_t bytes, int dst);
void comm_recv(comm_context *ctx, void *buf, size_t bytes, int src);

/* Allreduce sum of float buffer in-place. Uses shm if same-node, else TCP ring. */
void comm_allreduce_sum_f32(comm_context *ctx, float *buf, int count);

/* Broadcast buf from root to all. */
void comm_broadcast(comm_context *ctx, void *buf, size_t bytes, int root);

/* Allgather: each rank sends elem_bytes, recvbuf has world_size * elem_bytes. */
void comm_allgather(comm_context *ctx, const void *sendbuf, void *recvbuf, size_t elem_bytes);

/* Barrier. */
void comm_barrier(comm_context *ctx);

/* Split into sub-groups by color (ranks with same color form a group, ordered by key). */
comm_context *comm_split(comm_context *ctx, int color, int key);

/* Accessors */
int comm_rank(const comm_context *ctx);
int comm_size(const comm_context *ctx);

/* Cleanup */
void comm_destroy(comm_context *ctx);

/* Helper: parse --comm-rank, --comm-nranks, --comm-addr, --comm-port from argv.
 * Removes parsed args from argv, updates *argc. Returns 0 on success. */
int comm_parse_args(int *argc, char **argv,
                     int *rank, int *world_size,
                     char *master_addr, int addr_len, int *master_port);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef COMM_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <sched.h>

/* ---- Configuration ---- */
#define COMM_SHM_SLOT_SIZE  (1024 * 1024)   /* 1MB per rank slot for allreduce */
#define COMM_SHM_HDR_SIZE   128             /* barrier counters (cache-line aligned) */
#define COMM_MAX_HOSTNAME   64
#define COMM_CONNECT_RETRIES 50
#define COMM_CONNECT_DELAY_US 100000        /* 100ms between retries */

/* ---- Internal structures ---- */

typedef struct {
    int rank;
    int data_port;
    char hostname[COMM_MAX_HOSTNAME];
} comm_peer_info;

struct comm_context {
    int rank;
    int world_size;

    /* TCP connections (root context only) */
    int *peer_fds;          /* [world_size], -1 for self */
    int listen_fd;          /* data listen socket */
    comm_peer_info *peers;  /* [world_size] peer info */
    int master_port;

    /* Sub-group (non-root context) */
    comm_context *root;         /* root context for fd lookup (NULL if this is root) */
    int *group_to_global;       /* [world_size] sub-group rank → root rank */

    /* Shared memory */
    char shm_path[128];
    int shm_fd;
    void *shm_ptr;          /* mmap'd region: [hdr 128B][slot0][slot1]... */
    size_t shm_size;
    int use_shm;
    int barrier_sense;      /* local sense for sense-reversing barrier */
};

/* ---- TCP helpers ---- */

static int comm_tcp_send_exact(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            fprintf(stderr, "comm: tcp_send_exact failed: %s\n", strerror(errno));
            return -1;
        }
        sent += n;
    }
    return 0;
}

static int comm_tcp_recv_exact(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            fprintf(stderr, "comm: tcp_recv_exact failed: %s\n", strerror(errno));
            return -1;
        }
        got += n;
    }
    return 0;
}

static int comm_tcp_set_nodelay(int fd) {
    int flag = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
}

static int comm_tcp_listen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    if (listen(fd, 128) < 0) { close(fd); return -1; }
    return fd;
}

static int comm_tcp_get_port(int fd) {
    struct sockaddr_in addr;
    socklen_t alen = sizeof(addr);
    getsockname(fd, (struct sockaddr *)&addr, &alen);
    return ntohs(addr.sin_port);
}

static int comm_tcp_connect(const char *host, int port) {
    for (int attempt = 0; attempt < COMM_CONNECT_RETRIES; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
            /* Try hostname resolution */
            struct hostent *he = gethostbyname(host);
            if (he) memcpy(&addr.sin_addr, he->h_addr, he->h_length);
            else { close(fd); return -1; }
        }
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            comm_tcp_set_nodelay(fd);
            return fd;
        }
        close(fd);
        if (attempt < COMM_CONNECT_RETRIES - 1)
            usleep(COMM_CONNECT_DELAY_US);
    }
    fprintf(stderr, "comm: failed to connect to %s:%d after %d retries\n",
            host, port, COMM_CONNECT_RETRIES);
    return -1;
}

static int comm_tcp_accept(int listen_fd) {
    struct sockaddr_in addr;
    socklen_t alen = sizeof(addr);
    int fd;
    do {
        fd = accept(listen_fd, (struct sockaddr *)&addr, &alen);
    } while (fd < 0 && errno == EINTR);
    if (fd >= 0) comm_tcp_set_nodelay(fd);
    return fd;
}

/* ---- Shared memory helpers ---- */

static void comm_shm_barrier(comm_context *ctx) {
    volatile int *counter = (volatile int *)ctx->shm_ptr;
    volatile int *sense   = (volatile int *)((char *)ctx->shm_ptr + 64);
    int my_sense = !ctx->barrier_sense;

    if (__atomic_add_fetch(counter, 1, __ATOMIC_ACQ_REL) == ctx->world_size) {
        __atomic_store_n(counter, 0, __ATOMIC_RELAXED);
        __atomic_store_n(sense, my_sense, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(sense, __ATOMIC_ACQUIRE) != my_sense)
            sched_yield();
    }
    ctx->barrier_sense = my_sense;
}

static void comm_shm_allreduce_f32(comm_context *ctx, float *buf, int count) {
    char *base = (char *)ctx->shm_ptr;
    char *slots = base + COMM_SHM_HDR_SIZE;
    int slot_floats = COMM_SHM_SLOT_SIZE / (int)sizeof(float);

    /* Process in chunks that fit in one slot */
    for (int offset = 0; offset < count; offset += slot_floats) {
        int chunk = count - offset;
        if (chunk > slot_floats) chunk = slot_floats;
        size_t chunk_bytes = chunk * sizeof(float);

        /* Write local data to our slot */
        memcpy(slots + ctx->rank * COMM_SHM_SLOT_SIZE, buf + offset, chunk_bytes);

        /* Barrier: all ranks have written */
        comm_shm_barrier(ctx);

        /* Sum all slots into buf */
        memset(buf + offset, 0, chunk_bytes);
        for (int r = 0; r < ctx->world_size; r++) {
            const float *slot = (const float *)(slots + r * COMM_SHM_SLOT_SIZE);
            int i = 0;
#if defined(__AVX2__)
            for (; i + 7 < chunk; i += 8) {
                __m256 a = _mm256_loadu_ps(buf + offset + i);
                __m256 b = _mm256_loadu_ps(slot + i);
                _mm256_storeu_ps(buf + offset + i, _mm256_add_ps(a, b));
            }
#endif
            for (; i < chunk; i++)
                buf[offset + i] += slot[i];
        }

        /* Barrier: safe to reuse slots */
        comm_shm_barrier(ctx);
    }
}

static int comm_setup_shm(comm_context *ctx, int unique_id, int color) {
    size_t total = COMM_SHM_HDR_SIZE + (size_t)ctx->world_size * COMM_SHM_SLOT_SIZE;
    snprintf(ctx->shm_path, sizeof(ctx->shm_path),
             "/dev/shm/comm_%d_%d", unique_id, color);

    if (ctx->rank == 0) {
        ctx->shm_fd = open(ctx->shm_path, O_CREAT | O_RDWR | O_TRUNC, 0600);
        if (ctx->shm_fd < 0) return -1;
        if (ftruncate(ctx->shm_fd, total) < 0) { close(ctx->shm_fd); return -1; }
    }

    /* TCP barrier so rank 0 finishes creation before others open */
    comm_barrier(ctx);

    if (ctx->rank != 0) {
        ctx->shm_fd = open(ctx->shm_path, O_RDWR, 0600);
        if (ctx->shm_fd < 0) return -1;
    }

    ctx->shm_ptr = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED,
                          ctx->shm_fd, 0);
    if (ctx->shm_ptr == MAP_FAILED) { close(ctx->shm_fd); return -1; }
    ctx->shm_size = total;

    /* Initialize barrier counters */
    if (ctx->rank == 0) {
        memset(ctx->shm_ptr, 0, COMM_SHM_HDR_SIZE);
    }
    ctx->barrier_sense = 0;
    ctx->use_shm = 1;

    /* Barrier to ensure init is visible */
    /* Use TCP barrier since shm barrier needs initialized counters.
     * After this point, shm_barrier is safe to use. */
    {
        char dummy = 0;
        int fd0 = (ctx->root ? ctx->root : ctx)->peer_fds[
            ctx->group_to_global ? ctx->group_to_global[0] : 0];
        if (ctx->rank == 0) {
            for (int i = 1; i < ctx->world_size; i++) {
                int fd = (ctx->root ? ctx->root : ctx)->peer_fds[
                    ctx->group_to_global ? ctx->group_to_global[i] : i];
                comm_tcp_recv_exact(fd, &dummy, 1);
            }
            for (int i = 1; i < ctx->world_size; i++) {
                int fd = (ctx->root ? ctx->root : ctx)->peer_fds[
                    ctx->group_to_global ? ctx->group_to_global[i] : i];
                comm_tcp_send_exact(fd, &dummy, 1);
            }
        } else {
            int fd = (ctx->root ? ctx->root : ctx)->peer_fds[
                ctx->group_to_global ? ctx->group_to_global[0] : 0];
            comm_tcp_send_exact(fd, &dummy, 1);
            comm_tcp_recv_exact(fd, &dummy, 1);
        }
    }

    return 0;
}

/* ---- Peer fd lookup ---- */

static int comm_get_peer_fd(comm_context *ctx, int peer_rank) {
    comm_context *root = ctx->root ? ctx->root : ctx;
    int global = ctx->group_to_global ? ctx->group_to_global[peer_rank] : peer_rank;
    return root->peer_fds[global];
}

/* ---- Ring allreduce over TCP ---- */

static void comm_ring_allreduce_f32(comm_context *ctx, float *buf, int count) {
    int N = ctx->world_size;
    int rank = ctx->rank;
    if (N <= 1) return;

    int left_fd  = comm_get_peer_fd(ctx, (rank - 1 + N) % N);
    int right_fd = comm_get_peer_fd(ctx, (rank + 1) % N);

    /* Compute segment boundaries */
    int seg_size = count / N;
    int remainder = count % N;
    int *seg_start = (int *)alloca(N * sizeof(int));
    int *seg_count = (int *)alloca(N * sizeof(int));
    int off = 0;
    for (int i = 0; i < N; i++) {
        seg_start[i] = off;
        seg_count[i] = seg_size + (i < remainder ? 1 : 0);
        off += seg_count[i];
    }

    int max_seg = seg_size + 1;
    float *tmp = (float *)malloc(max_seg * sizeof(float));

    /* Reduce-scatter: N-1 rounds */
    for (int k = 0; k < N - 1; k++) {
        int send_seg = (rank - k + N) % N;
        int recv_seg = (rank - k - 1 + N) % N;
        int sc = seg_count[send_seg];
        int rc = seg_count[recv_seg];

        /* Alternate send/recv order to avoid deadlock */
        if (rank % 2 == 0) {
            comm_tcp_send_exact(right_fd, buf + seg_start[send_seg], sc * sizeof(float));
            comm_tcp_recv_exact(left_fd, tmp, rc * sizeof(float));
        } else {
            comm_tcp_recv_exact(left_fd, tmp, rc * sizeof(float));
            comm_tcp_send_exact(right_fd, buf + seg_start[send_seg], sc * sizeof(float));
        }

        /* Accumulate received data */
        float *dst = buf + seg_start[recv_seg];
        for (int i = 0; i < rc; i++) dst[i] += tmp[i];
    }

    /* Allgather: N-1 rounds */
    for (int k = 0; k < N - 1; k++) {
        int send_seg = (rank - k + 1 + N) % N;
        int recv_seg = (rank - k + N) % N;
        int sc = seg_count[send_seg];
        int rc = seg_count[recv_seg];

        if (rank % 2 == 0) {
            comm_tcp_send_exact(right_fd, buf + seg_start[send_seg], sc * sizeof(float));
            comm_tcp_recv_exact(left_fd, buf + seg_start[recv_seg], rc * sizeof(float));
        } else {
            comm_tcp_recv_exact(left_fd, buf + seg_start[recv_seg], rc * sizeof(float));
            comm_tcp_send_exact(right_fd, buf + seg_start[send_seg], sc * sizeof(float));
        }
    }

    free(tmp);
}

/* ---- Public API ---- */

int comm_rank(const comm_context *ctx) { return ctx->rank; }
int comm_size(const comm_context *ctx) { return ctx->world_size; }

comm_context *comm_create(int rank, int world_size,
                           const char *master_addr, int master_port) {
    comm_context *ctx = (comm_context *)calloc(1, sizeof(comm_context));
    ctx->rank = rank;
    ctx->world_size = world_size;
    ctx->master_port = master_port;
    ctx->shm_fd = -1;

    /* Create data listen socket on ephemeral port */
    ctx->listen_fd = comm_tcp_listen(0);
    if (ctx->listen_fd < 0) { free(ctx); return NULL; }
    int my_data_port = comm_tcp_get_port(ctx->listen_fd);

    /* Prepare my peer info */
    ctx->peers = (comm_peer_info *)calloc(world_size, sizeof(comm_peer_info));
    ctx->peers[rank].rank = rank;
    ctx->peers[rank].data_port = my_data_port;
    gethostname(ctx->peers[rank].hostname, COMM_MAX_HOSTNAME);

    /* ---- Bootstrap via rank 0 ---- */
    if (rank == 0) {
        int boot_fd = comm_tcp_listen(master_port);
        if (boot_fd < 0) {
            fprintf(stderr, "comm: rank 0 failed to listen on port %d\n", master_port);
            free(ctx->peers); close(ctx->listen_fd); free(ctx); return NULL;
        }

        /* Accept connections from ranks 1..N-1 */
        int *boot_fds = (int *)alloca((world_size - 1) * sizeof(int));
        for (int i = 0; i < world_size - 1; i++) {
            boot_fds[i] = comm_tcp_accept(boot_fd);
            if (boot_fds[i] < 0) {
                fprintf(stderr, "comm: rank 0 failed to accept bootstrap connection\n");
                close(boot_fd); free(ctx->peers); close(ctx->listen_fd); free(ctx);
                return NULL;
            }
            /* Receive peer info */
            comm_peer_info pi;
            comm_tcp_recv_exact(boot_fds[i], &pi, sizeof(pi));
            ctx->peers[pi.rank] = pi;
        }

        /* Send full peer table to each rank */
        for (int i = 0; i < world_size - 1; i++) {
            comm_tcp_send_exact(boot_fds[i], ctx->peers, world_size * sizeof(comm_peer_info));
            close(boot_fds[i]);
        }
        close(boot_fd);
    } else {
        /* Connect to rank 0's bootstrap port */
        int boot_fd = comm_tcp_connect(master_addr, master_port);
        if (boot_fd < 0) {
            fprintf(stderr, "comm: rank %d failed to connect to bootstrap %s:%d\n",
                    rank, master_addr, master_port);
            free(ctx->peers); close(ctx->listen_fd); free(ctx);
            return NULL;
        }
        comm_tcp_send_exact(boot_fd, &ctx->peers[rank], sizeof(comm_peer_info));
        comm_tcp_recv_exact(boot_fd, ctx->peers, world_size * sizeof(comm_peer_info));
        close(boot_fd);
    }

    /* ---- Establish full-mesh data connections ---- */
    ctx->peer_fds = (int *)calloc(world_size, sizeof(int));
    for (int i = 0; i < world_size; i++) ctx->peer_fds[i] = -1;

    /* Connect to higher-ranked peers */
    for (int j = rank + 1; j < world_size; j++) {
        ctx->peer_fds[j] = comm_tcp_connect(ctx->peers[j].hostname,
                                              ctx->peers[j].data_port);
        if (ctx->peer_fds[j] < 0) {
            fprintf(stderr, "comm: rank %d failed to connect to rank %d (%s:%d)\n",
                    rank, j, ctx->peers[j].hostname, ctx->peers[j].data_port);
            /* cleanup omitted for brevity */
            return NULL;
        }
        /* Identify ourselves */
        int my_rank = rank;
        comm_tcp_send_exact(ctx->peer_fds[j], &my_rank, sizeof(int));
    }

    /* Accept from lower-ranked peers */
    for (int i = 0; i < rank; i++) {
        int fd = comm_tcp_accept(ctx->listen_fd);
        if (fd < 0) {
            fprintf(stderr, "comm: rank %d failed to accept from peer\n", rank);
            return NULL;
        }
        int their_rank;
        comm_tcp_recv_exact(fd, &their_rank, sizeof(int));
        ctx->peer_fds[their_rank] = fd;
    }

    /* ---- Shared memory setup (if all on same host) ---- */
    {
        int same_host = 1;
        for (int i = 1; i < world_size; i++) {
            if (strcmp(ctx->peers[i].hostname, ctx->peers[0].hostname) != 0) {
                same_host = 0;
                break;
            }
        }
        if (same_host && world_size > 1) {
            if (comm_setup_shm(ctx, master_port, 0) < 0) {
                fprintf(stderr, "comm: rank %d shm setup failed, using TCP\n", rank);
                ctx->use_shm = 0;
            }
        }
    }

    if (rank == 0)
        fprintf(stderr, "comm: %d ranks connected (shm=%s)\n",
                world_size, ctx->use_shm ? "yes" : "no");

    return ctx;
}

void comm_send(comm_context *ctx, const void *buf, size_t bytes, int dst) {
    int fd = comm_get_peer_fd(ctx, dst);
    comm_tcp_send_exact(fd, buf, bytes);
}

void comm_recv(comm_context *ctx, void *buf, size_t bytes, int src) {
    int fd = comm_get_peer_fd(ctx, src);
    comm_tcp_recv_exact(fd, buf, bytes);
}

void comm_allreduce_sum_f32(comm_context *ctx, float *buf, int count) {
    if (ctx->world_size <= 1) return;
    if (ctx->use_shm) {
        comm_shm_allreduce_f32(ctx, buf, count);
    } else {
        comm_ring_allreduce_f32(ctx, buf, count);
    }
}

void comm_barrier(comm_context *ctx) {
    if (ctx->world_size <= 1) return;
    if (ctx->use_shm) {
        comm_shm_barrier(ctx);
        return;
    }
    /* TCP barrier: gather to rank 0, then scatter */
    char dummy = 0;
    if (ctx->rank == 0) {
        for (int i = 1; i < ctx->world_size; i++)
            comm_tcp_recv_exact(comm_get_peer_fd(ctx, i), &dummy, 1);
        for (int i = 1; i < ctx->world_size; i++)
            comm_tcp_send_exact(comm_get_peer_fd(ctx, i), &dummy, 1);
    } else {
        comm_tcp_send_exact(comm_get_peer_fd(ctx, 0), &dummy, 1);
        comm_tcp_recv_exact(comm_get_peer_fd(ctx, 0), &dummy, 1);
    }
}

void comm_broadcast(comm_context *ctx, void *buf, size_t bytes, int root) {
    if (ctx->world_size <= 1) return;
    if (root == ctx->rank) {
        for (int i = 0; i < ctx->world_size; i++) {
            if (i == root) continue;
            comm_send(ctx, buf, bytes, i);
        }
    } else {
        comm_recv(ctx, buf, bytes, root);
    }
}

void comm_allgather(comm_context *ctx, const void *sendbuf, void *recvbuf, size_t elem_bytes) {
    char *rbuf = (char *)recvbuf;
    memcpy(rbuf + ctx->rank * elem_bytes, sendbuf, elem_bytes);
    if (ctx->world_size <= 1) return;

    /* Gather to rank 0, then broadcast */
    if (ctx->rank == 0) {
        for (int i = 1; i < ctx->world_size; i++)
            comm_recv(ctx, rbuf + i * elem_bytes, elem_bytes, i);
        for (int i = 1; i < ctx->world_size; i++)
            comm_send(ctx, rbuf, ctx->world_size * elem_bytes, i);
    } else {
        comm_send(ctx, sendbuf, elem_bytes, 0);
        comm_recv(ctx, rbuf, ctx->world_size * elem_bytes, 0);
    }
}

comm_context *comm_split(comm_context *ctx, int color, int key) {
    /* Allgather {color, key, global_rank} */
    typedef struct { int color, key, global_rank; } split_info;
    split_info my_info = {color, key, ctx->rank};
    split_info *all = (split_info *)malloc(ctx->world_size * sizeof(split_info));
    comm_allgather(ctx, &my_info, all, sizeof(split_info));

    /* Find members of my color group */
    int group_size = 0;
    int *members = (int *)malloc(ctx->world_size * sizeof(int));
    int *keys    = (int *)malloc(ctx->world_size * sizeof(int));
    for (int i = 0; i < ctx->world_size; i++) {
        if (all[i].color == color) {
            members[group_size] = all[i].global_rank;
            keys[group_size] = all[i].key;
            group_size++;
        }
    }

    /* Sort by key (insertion sort, small N) */
    for (int i = 1; i < group_size; i++) {
        int km = keys[i], mm = members[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > km) {
            keys[j + 1] = keys[j];
            members[j + 1] = members[j];
            j--;
        }
        keys[j + 1] = km;
        members[j + 1] = mm;
    }

    /* Find my rank in the sub-group */
    int my_group_rank = -1;
    for (int i = 0; i < group_size; i++) {
        if (members[i] == ctx->rank) { my_group_rank = i; break; }
    }

    /* Create sub-context */
    comm_context *sub = (comm_context *)calloc(1, sizeof(comm_context));
    sub->rank = my_group_rank;
    sub->world_size = group_size;
    sub->root = ctx->root ? ctx->root : ctx;
    sub->group_to_global = members;
    sub->shm_fd = -1;
    sub->listen_fd = -1;

    /* Check if all sub-group members are on the same host */
    comm_context *root = sub->root;
    int same_host = 1;
    const char *host0 = root->peers[members[0]].hostname;
    for (int i = 1; i < group_size; i++) {
        if (strcmp(root->peers[members[i]].hostname, host0) != 0) {
            same_host = 0;
            break;
        }
    }

    /* Setup shared memory for sub-group if same-host */
    if (same_host && group_size > 1) {
        if (comm_setup_shm(sub, root->master_port, color + 1) < 0) {
            sub->use_shm = 0;
        }
    }

    free(keys);
    free(all);
    return sub;
}

void comm_destroy(comm_context *ctx) {
    if (!ctx) return;

    /* Cleanup shared memory */
    if (ctx->shm_ptr && ctx->shm_ptr != MAP_FAILED) {
        munmap(ctx->shm_ptr, ctx->shm_size);
    }
    if (ctx->shm_fd >= 0) close(ctx->shm_fd);
    if (ctx->shm_path[0] && ctx->rank == 0) {
        unlink(ctx->shm_path);
    }

    /* Cleanup TCP (root context only) */
    if (!ctx->root) {
        if (ctx->peer_fds) {
            for (int i = 0; i < ctx->world_size; i++) {
                if (ctx->peer_fds[i] >= 0) close(ctx->peer_fds[i]);
            }
            free(ctx->peer_fds);
        }
        if (ctx->listen_fd >= 0) close(ctx->listen_fd);
        free(ctx->peers);
    }

    free(ctx->group_to_global);
    free(ctx);
}

int comm_parse_args(int *argc, char **argv,
                     int *rank, int *world_size,
                     char *master_addr, int addr_len, int *master_port) {
    *rank = 0; *world_size = 1;
    strncpy(master_addr, "127.0.0.1", addr_len);
    *master_port = 29500;

    int out = 1;
    for (int i = 1; i < *argc; i++) {
        if (strcmp(argv[i], "--comm-rank") == 0 && i + 1 < *argc) {
            *rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--comm-nranks") == 0 && i + 1 < *argc) {
            *world_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--comm-addr") == 0 && i + 1 < *argc) {
            strncpy(master_addr, argv[++i], addr_len - 1);
        } else if (strcmp(argv[i], "--comm-port") == 0 && i + 1 < *argc) {
            *master_port = atoi(argv[++i]);
        } else {
            argv[out++] = argv[i];
        }
    }
    *argc = out;
    argv[out] = NULL;
    return 0;
}

#endif /* COMM_IMPLEMENTATION */
#endif /* COMM_H */
