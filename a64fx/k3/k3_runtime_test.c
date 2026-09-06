#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "k3_runtime.h"

int main(void) {
    k3_pool pool;
    int fail = k3_pool_init(&pool, "runtime-test") != 0;
    unsigned char *a = k3_pool_alloc(&pool, 1);
    unsigned char *b = k3_pool_alloc(&pool, 257);
    uint32_t *z = k3_pool_calloc(&pool, 65, sizeof(*z));
    fail |= !a || !b || !z;
    fail |= a && ((uintptr_t)a % K3_POOL_ALIGNMENT) != 0;
    fail |= b && ((uintptr_t)b % K3_POOL_ALIGNMENT) != 0;
    fail |= z && ((uintptr_t)z % K3_POOL_ALIGNMENT) != 0;
    for (int i = 0; z && i < 65; ++i)
        fail |= z[i] != 0;

    void *old_b = b;
    fail |= k3_pool_free(&pool, b) != 0;
    b = k3_pool_alloc(&pool, 128);
    fail |= b != old_b;
    fail |= pool.peak_active_bytes < 1024;
    fail |= k3_pool_free(&pool, a) != 0;
    fail |= k3_pool_free(&pool, a) != EINVAL;
    fail |= k3_pool_free(&pool, b) != 0;
    fail |= k3_pool_free(&pool, z) != 0;
    k3_pool_trim(&pool);
    fail |= pool.active_bytes != 0 || pool.reserved_bytes != 0;

    size_t rss=0,hwm=0;int mem_rc=k3_process_memory_bytes(&rss,&hwm);
    int mem_ok=!mem_rc&&rss>0&&hwm>=rss;fail|=!mem_ok;
    printf("[runtime-memory] rss_MiB=%.2f hwm_MiB=%.2f %s\n",
           rss/1048576.0,hwm/1048576.0,mem_ok?"PASS":"FAIL");

    printf("[runtime-pool] alignment=%lu reuse=%s accounting=%s %s\n",
           K3_POOL_ALIGNMENT, b == old_b ? "yes" : "no",
           pool.active_bytes == 0 && pool.reserved_bytes == 0 ? "OK" : "FAIL",
           fail ? "FAIL" : "PASS");
    k3_pool_destroy(&pool);
    return fail ? 1 : 0;
}
