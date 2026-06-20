#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int env_i(const char *name, int defval) {
    const char *s = getenv(name);
    return (s && *s) ? atoi(s) : defval;
}

static size_t env_z(const char *name, size_t defval) {
    const char *s = getenv(name);
    if (!s || !*s) return defval;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (end && (*end == 'k' || *end == 'K')) v *= 1024ull;
    if (end && (*end == 'm' || *end == 'M')) v *= 1024ull * 1024ull;
    return (size_t)v;
}

static double min_nonzero(double v, double cur) {
    return (cur == 0.0 || (v > 0.0 && v < cur)) ? v : cur;
}

static void fill_bytes(unsigned char *p, size_t n, int rank) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)((i * 131u + (unsigned)rank) & 255u);
}

static void bench_allreduce(int rank, int size, int iters) {
    double best = 0.0, sum = 0.0;
    for (int i = 0; i < iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        long long in = rank + i, out = 0;
        MPI_Allreduce(&in, &out, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        double dt = MPI_Wtime() - t0;
        sum += dt;
        best = min_nonzero(dt, best);
        if (out != ((long long)size * (size - 1) / 2 + (long long)i * size)) {
            fprintf(stderr, "rank %d: allreduce mismatch iter=%d out=%lld\n", rank, i, out);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    if (rank == 0) {
        printf("MPI_BENCH allreduce_i64 iters=%d best_us=%.3f avg_us=%.3f\n",
               iters, best * 1e6, (sum / iters) * 1e6);
    }
}

static void bench_bcast(int rank, int iters, size_t bytes) {
    unsigned char *buf = malloc(bytes ? bytes : 1);
    if (!buf) MPI_Abort(MPI_COMM_WORLD, 3);
    if (rank == 0) fill_bytes(buf, bytes, rank);
    double best = 0.0, sum = 0.0;
    for (int i = 0; i < iters; i++) {
        if (rank == 0) buf[0] = (unsigned char)i;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Bcast(buf, (int)bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        double dt = MPI_Wtime() - t0;
        sum += dt;
        best = min_nonzero(dt, best);
        if (bytes && buf[0] != (unsigned char)i) {
            fprintf(stderr, "rank %d: bcast mismatch iter=%d got=%u\n", rank, i, (unsigned)buf[0]);
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }
    if (rank == 0) {
        double gbps = bytes / best / 1e9;
        printf("MPI_BENCH bcast bytes=%zu iters=%d best_ms=%.3f avg_ms=%.3f best_GBps=%.3f\n",
               bytes, iters, best * 1e3, (sum / iters) * 1e3, gbps);
    }
    free(buf);
}

static void bench_alltoall(int rank, int size, int iters, size_t each_bytes) {
    size_t total = each_bytes * (size_t)size;
    unsigned char *send = malloc(total ? total : 1);
    unsigned char *recv = malloc(total ? total : 1);
    if (!send || !recv) MPI_Abort(MPI_COMM_WORLD, 5);
    fill_bytes(send, total, rank);
    double best = 0.0, sum = 0.0;
    for (int i = 0; i < iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Alltoall(send, (int)each_bytes, MPI_BYTE, recv, (int)each_bytes, MPI_BYTE, MPI_COMM_WORLD);
        double dt = MPI_Wtime() - t0;
        sum += dt;
        best = min_nonzero(dt, best);
    }
    if (rank == 0) {
        double rank_bytes = each_bytes * (size_t)size;
        printf("MPI_BENCH alltoall each_bytes=%zu rank_bytes=%zu iters=%d best_ms=%.3f avg_ms=%.3f per_rank_GBps=%.3f\n",
               each_bytes, rank_bytes, iters, best * 1e3, (sum / iters) * 1e3, rank_bytes / best / 1e9);
    }
    free(send);
    free(recv);
}

static void bench_ring_sendrecv(int rank, int size, int iters, size_t bytes) {
    unsigned char *send = malloc(bytes ? bytes : 1);
    unsigned char *recv = malloc(bytes ? bytes : 1);
    if (!send || !recv) MPI_Abort(MPI_COMM_WORLD, 6);
    fill_bytes(send, bytes, rank);
    int left = (rank + size - 1) % size;
    int right = (rank + 1) % size;
    double best = 0.0, sum = 0.0;
    for (int i = 0; i < iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Sendrecv(send, (int)bytes, MPI_BYTE, right, 100,
                     recv, (int)bytes, MPI_BYTE, left, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double dt = MPI_Wtime() - t0;
        sum += dt;
        best = min_nonzero(dt, best);
    }
    if (rank == 0) {
        printf("MPI_BENCH ring_sendrecv bytes=%zu iters=%d best_ms=%.3f avg_ms=%.3f per_rank_GBps=%.3f\n",
               bytes, iters, best * 1e3, (sum / iters) * 1e3, bytes / best / 1e9);
    }
    free(send);
    free(recv);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char pname[MPI_MAX_PROCESSOR_NAME];
    int plen = 0;
    MPI_Get_processor_name(pname, &plen);

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);

    int world_nodes = 0;
    int is_node_leader = (local_rank == 0) ? 1 : 0;
    MPI_Allreduce(&is_node_leader, &world_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI_PROBE size=%d nodes=%d ranks_per_node_minmax pending\n", size, world_nodes);
        printf("MPI_PROBE env PJM_JOBID=%s PJM_RSCGRP=%s PJM_NODE=%s\n",
               getenv("PJM_JOBID") ? getenv("PJM_JOBID") : "-",
               getenv("PJM_RSCGRP") ? getenv("PJM_RSCGRP") : "-",
               getenv("PJM_NODE") ? getenv("PJM_NODE") : "-");
    }

    int loc_min = local_size, loc_max = local_size;
    MPI_Allreduce(MPI_IN_PLACE, &loc_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &loc_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_PROBE ranks_per_node min=%d max=%d\n", loc_min, loc_max);

    for (int r = 0; r < size; r++) {
        if (r == rank && (rank < 8 || rank == size - 1 || local_rank == 0)) {
            printf("MPI_RANK rank=%d local_rank=%d local_size=%d host=%s\n",
                   rank, local_rank, local_size, pname);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int iters = env_i("MPI_PROBE_ITERS", 20);
    size_t bytes = env_z("MPI_PROBE_BYTES", 1u << 20);
    size_t each = env_z("MPI_PROBE_ALLTOALL_EACH", 4096);

    bench_allreduce(rank, size, iters);
    bench_bcast(rank, iters, bytes);
    bench_ring_sendrecv(rank, size, iters, bytes);
    bench_alltoall(rank, size, iters, each);

    if (rank == 0) printf("MPI_PROBE_DONE size=%d nodes=%d\n", size, world_nodes);
    MPI_Comm_free(&local);
    MPI_Finalize();
    return 0;
}
