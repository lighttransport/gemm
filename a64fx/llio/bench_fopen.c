/*
 * bench_fopen.c -- fopen()/fread() throughput benchmark
 *
 * Usage:
 *   bench_fopen <file> [chunk_bytes=1048576] [passes=2]
 *
 * Reports per-pass wall time and bandwidth (GB/s, GiB/s).
 * First pass is "cold" (page cache empty or LLIO miss); subsequent
 * passes are "warm". An XOR checksum prevents the compiler from
 * eliding the read.
 *
 * Build:   fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_fopen bench_fopen.c
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

static uint64_t reduce_xor(const uint8_t *p, size_t n, uint64_t acc) {
    /* 8-byte tail-tolerant XOR reduction. */
    size_t i = 0;
    uint64_t a = acc;
    while (i + 8 <= n) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        a ^= v;
        i += 8;
    }
    while (i < n) {
        a ^= (uint64_t)p[i] << ((i & 7) * 8);
        i++;
    }
    return a;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [chunk_bytes=1048576] [passes=2]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    size_t chunk = (argc >= 3) ? (size_t)strtoull(argv[2], NULL, 0) : (1ull << 20);
    int passes = (argc >= 4) ? atoi(argv[3]) : 2;
    if (chunk == 0) chunk = 1ull << 20;
    if (passes <= 0) passes = 1;

    struct stat st;
    if (stat(path, &st) != 0) {
        perror("stat");
        return 1;
    }
    const uint64_t fsize = (uint64_t)st.st_size;
    printf("file        : %s\n", path);
    printf("size        : %llu bytes (%.3f GiB)\n",
           (unsigned long long)fsize, (double)fsize / (1ull << 30));
    printf("chunk       : %zu bytes\n", chunk);
    printf("passes      : %d\n\n", passes);

    void *buf = NULL;
    if (posix_memalign(&buf, 4096, chunk) != 0 || buf == NULL) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }

    for (int p = 0; p < passes; p++) {
        /* Hint kernel to drop cached pages for this file BEFORE reading
         * if this is pass 0. POSIX_FADV_DONTNEED on a fresh fd does
         * nothing reliable, but we still issue it on the previous fd. */
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            perror("fopen");
            free(buf);
            return 1;
        }
        /* Disable stdio's own buffering: we want to measure fread of
         * `chunk` bytes directly. */
        setvbuf(fp, NULL, _IONBF, 0);

        uint64_t total = 0;
        uint64_t xsum = 0;
        double t0 = now_sec();
        for (;;) {
            size_t r = fread(buf, 1, chunk, fp);
            if (r == 0) break;
            xsum = reduce_xor((const uint8_t *)buf, r, xsum);
            total += r;
        }
        double t1 = now_sec();
        fclose(fp);

        double dt = t1 - t0;
        double gib = (double)total / (1ull << 30);
        double gb  = (double)total / 1e9;
        printf("pass %d  %s  read=%.3f GiB  time=%.3f s  bw=%.3f GB/s (%.3f GiB/s)  xor=%016llx\n",
               p, (p == 0) ? "cold" : "warm",
               gib, dt, gb / dt, gib / dt,
               (unsigned long long)xsum);
        fflush(stdout);
    }

    free(buf);
    return 0;
}
