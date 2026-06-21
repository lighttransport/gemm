/*
 * bench_mmap.c -- mmap()+sequential-read throughput benchmark
 *
 * Usage:
 *   bench_mmap <file> [mode=seq|seq8|page] [passes=2] [populate=0|1]
 *
 *   seq   : read 1 byte from every cache line (64B stride)
 *   seq8  : 8-byte XOR scan, contiguous (compiler may vectorize to SVE)
 *   page  : touch first byte of every 4 KiB page (faulting cost only)
 *
 * Reports wall time and bandwidth (file-size / dt). Cold pass measures
 * page-fault + read cost; warm pass measures cached touch cost.
 *
 * Build:   fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_mmap bench_mmap.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t scan_seq_cl(const uint8_t *p, size_t n) {
    /* Touch one byte per 64-byte cache line. */
    uint64_t acc = 0;
    for (size_t i = 0; i < n; i += 64) acc ^= p[i];
    return acc;
}

static uint64_t scan_seq8(const uint8_t *p, size_t n) {
    /* Contiguous 8-byte XOR scan. */
    uint64_t acc = 0;
    const uint64_t *q = (const uint64_t *)p;
    size_t n8 = n / 8;
    for (size_t i = 0; i < n8; i++) acc ^= q[i];
    return acc;
}

static uint64_t scan_page(const uint8_t *p, size_t n) {
    /* Touch first byte of every 4 KiB page. */
    uint64_t acc = 0;
    for (size_t i = 0; i < n; i += 4096) acc ^= p[i];
    return acc;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <file> [mode=seq|seq8|page] [passes=2] [populate=0|1]\n",
                argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const char *mode = (argc >= 3) ? argv[2] : "seq8";
    int passes = (argc >= 4) ? atoi(argv[3]) : 2;
    int populate = (argc >= 5) ? atoi(argv[4]) : 0;
    if (passes <= 0) passes = 1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const size_t fsize = (size_t)st.st_size;

    printf("file        : %s\n", path);
    printf("size        : %zu bytes (%.3f GiB)\n", fsize,
           (double)fsize / (1ull << 30));
    printf("mode        : %s\n", mode);
    printf("passes      : %d\n", passes);
    printf("populate    : %d\n\n", populate);

    int flags = MAP_PRIVATE;
    if (populate) flags |= MAP_POPULATE;
    /* Time the mmap itself (cheap, but report so callers can see). */
    double tm0 = now_sec();
    void *map = mmap(NULL, fsize, PROT_READ, flags, fd, 0);
    double tm1 = now_sec();
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    printf("mmap        : %.3f ms%s\n",
           (tm1 - tm0) * 1e3, populate ? " (MAP_POPULATE)" : "");

    /* Hint sequential access so the kernel readaheads more aggressively. */
    if (madvise(map, fsize, MADV_SEQUENTIAL) != 0) perror("madvise");

    for (int p = 0; p < passes; p++) {
        uint64_t acc = 0;
        double t0 = now_sec();
        if (strcmp(mode, "seq") == 0)       acc = scan_seq_cl((const uint8_t *)map, fsize);
        else if (strcmp(mode, "page") == 0) acc = scan_page  ((const uint8_t *)map, fsize);
        else                                 acc = scan_seq8 ((const uint8_t *)map, fsize);
        double t1 = now_sec();
        double dt = t1 - t0;
        double gib = (double)fsize / (1ull << 30);
        double gb  = (double)fsize / 1e9;
        printf("pass %d  %s  scan=%.3f GiB  time=%.3f s  bw=%.3f GB/s (%.3f GiB/s)  xor=%016llx\n",
               p, (p == 0) ? "cold" : "warm",
               gib, dt, gb / dt, gib / dt,
               (unsigned long long)acc);
        fflush(stdout);
    }

    munmap(map, fsize);
    close(fd);
    return 0;
}
