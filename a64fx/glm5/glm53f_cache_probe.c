#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../../common/glm53f_ref.h"

static long env_long(const char *name, long fallback) {
    const char *s = getenv(name); char *end; long x;
    if (!s || !*s) return fallback;
    errno = 0; x = strtol(s, &end, 10);
    return errno || *end ? fallback : x;
}
static uint64_t mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r"); char key[64], unit[16]; uint64_t kb, out = 0;
    if (!f) return 0;
    while (fscanf(f, "%63s %" SCNu64 " %15s", key, &kb, unit) == 3)
        if (!strcmp(key, "MemAvailable:")) { out = kb * 1024; break; }
    fclose(f); return out;
}
static unsigned long commit(unsigned char *p, size_t n) {
    volatile unsigned char *vp = p;
    size_t i, page = (size_t)sysconf(_SC_PAGESIZE);
    unsigned long sum = 0;
    for (i = 0; i < n; i += page) vp[i] = (unsigned char)(i / page);
    if (n) vp[n - 1] = 1;
    for (i = 0; i < n; i += page) sum += vp[i];
    return sum;
}
int main(int argc, char **argv) {
    size_t ctx = argc > 1 ? (size_t)strtoull(argv[1], NULL, 0) : 1048576;
    int ranks = argc > 2 ? atoi(argv[2]) : 12;
    int rank = (int)env_long("PMIX_RANK", env_long("PJM_MPI_RANK", 0));
    size_t slots = glm53f_cp_slots(ctx, ranks);
    size_t kv_n = slots * 11u * 512u * 2u;
    size_t idx_n = slots * 11u * 128u * 2u;
    size_t lin_n = 34u * 64u * 128u * 128u * 4u;
    unsigned char *kv, *idx, *lin;
    FILE *result = stdout;
    char result_path[4096];
    const char *result_dir = getenv("GLM53F_RESULT_DIR");
    uint64_t before = mem_available(), after;
    unsigned long checksum;
    if (ctx < 1 || ctx > 1048576 || ranks < 1 || rank < 0 || rank >= ranks) return 2;
    kv = (unsigned char *)malloc(kv_n); idx = (unsigned char *)malloc(idx_n); lin = (unsigned char *)malloc(lin_n);
    if (!kv || !idx || !lin) { fprintf(stderr, "rank=%d allocation failed\n", rank); return 1; }
    checksum = commit(kv, kv_n) + commit(idx, idx_n) + commit(lin, lin_n);
    after = mem_available();
    if (result_dir && *result_dir &&
        snprintf(result_path, sizeof(result_path), "%s/cache_%zu_rank%02d.log", result_dir, ctx, rank) < (int)sizeof(result_path)) {
        FILE *f = fopen(result_path, "w");
        if (f) result = f;
    }
    fprintf(result, "GLM53F_CACHE rank=%d ctx=%zu slots=%zu kv=%.3fGiB index=%.3fGiB linear=%.3fGiB "
           "avail_before=%.3fGiB avail_after=%.3fGiB checksum=%lu sentinel=%s\n", rank, ctx, slots,
           kv_n/1073741824.0, idx_n/1073741824.0, lin_n/1073741824.0,
           before/1073741824.0, after/1073741824.0, checksum, after >= (2ull<<30) ? "OK" : "LOWMEM");
    if (result != stdout) fclose(result);
    free(lin); free(idx); free(kv);
    return after >= (2ull << 30) ? 0 : 3;
}
