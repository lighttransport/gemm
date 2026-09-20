/* Validate a rank-local Q38KQC sidecar against its compact Q38TP source. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "qwen38_kquant_load.h"

static long env_value(const char *explicit_name, const char *const *names,
                      size_t count) {
    const char *value = getenv(explicit_name);
    if (value && *value) return strtol(value, NULL, 10);
    for (size_t i = 0; i < count; i++) {
        value = getenv(names[i]);
        if (value && *value) return strtol(value, NULL, 10);
    }
    return -1;
}

static long env_rank(void) {
    static const char *const names[] = {
        "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"
    };
    return env_value("Q38TP_RANK", names, sizeof(names) / sizeof(names[0]));
}

static long env_size(void) {
    static const char *const names[] = {
        "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
        "MV2_COMM_WORLD_SIZE", "PJM_MPI_PROC"
    };
    return env_value("Q38TP_SIZE", names, sizeof(names) / sizeof(names[0]));
}

static int read_all_at(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *p = (uint8_t *)buffer;
    while (bytes) {
        ssize_t got = pread(fd, p, bytes, (off_t)offset);
        if (got <= 0) return -1;
        p += got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
    return 0;
}

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s COMPACT_STAGE_DIR CACHE_STAGE_DIR\n", argv[0]);
        return 2;
    }
    long rank = env_rank(), size = env_size();
    if (rank < 0 || size < 2 || rank >= size) {
        fprintf(stderr, "qwen38_kquant_check: invalid rank/size %ld/%ld\n",
                rank, size);
        return 2;
    }
    char source_path[PATH_MAX], cache_path[PATH_MAX];
    snprintf(source_path, sizeof(source_path), "%s/rank%02ld.blob", argv[1], rank);
    snprintf(cache_path, sizeof(cache_path), "%s/rank%02ld.kquant", argv[2], rank);
    int source_fd = open(source_path, O_RDONLY);
    struct stat source_stat;
    q38tp_header *source = calloc(1, sizeof(*source));
    if (source_fd < 0 || !source || fstat(source_fd, &source_stat) ||
        source_stat.st_size < 0 ||
        read_all_at(source_fd, source, sizeof(*source), 0)) {
        fprintf(stderr, "qwen38_kquant_check: read %s: %s\n",
                source_path, strerror(errno));
        return 1;
    }
    close(source_fd);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    q38kc_loaded loaded = {0};
    char error[256] = {0};
    if (q38kc_load(&loaded, cache_path, source, (uint64_t)source_stat.st_size,
                   (uint32_t)rank, (uint32_t)size, error, sizeof(error))) {
        fprintf(stderr, "qwen38_kquant_check: %s\n", error);
        free(source);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SENTINEL qwen38_kquant_check=OK rank=%ld/%ld entries=%u "
           "file_bytes=%zu seconds=%.3f path=%s\n",
           rank, size, loaded.header->n_entries, loaded.mapping_bytes,
           elapsed_seconds(start, end), cache_path);
    q38kc_unload(&loaded);
    free(source);
    return 0;
}
