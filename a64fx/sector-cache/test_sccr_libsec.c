/*
 * A64FX Sector Cache Configuration via libsec.so (xos_sclib API)
 *
 * libsec.so opens /dev/xos_sec_normal, ioctls the kernel to enable
 * IMP_SCCR_CTRL_EL1.el0ae=1, then directly accesses IMP_SCCR_L1_EL0
 * (S3_3_C11_C8_2) and IMP_SCCR_VSCCR_L2_EL0 (S3_3_C15_C8_2).
 *
 * Build:
 *   fcc -Nclang -O2 -march=armv8.2-a+sve -o test_sccr_libsec test_sccr_libsec.c -lsec -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <signal.h>
#include <setjmp.h>
#include <arm_sve.h>

/* ===== SCCR register access ===== */

static inline uint64_t rd_sccr_l1(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, S3_3_C11_C8_2" : "=r"(v));
    return v;
}

static inline void wr_sccr_l1(uint64_t v) {
    __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(v));
    __asm__ volatile("isb" ::: "memory");
}

static inline uint64_t rd_vsccr_l2(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, S3_3_C15_C8_2" : "=r"(v));
    return v;
}

/* ===== SIGILL-safe probe ===== */

static sigjmp_buf jbuf;
static volatile int sigill_caught;

static void sigill_handler(int sig) {
    sigill_caught = 1;
    siglongjmp(jbuf, 1);
}

static int try_read_sccr_l1(uint64_t *out) {
    struct sigaction sa = {.sa_handler = sigill_handler};
    struct sigaction old;
    sigaction(SIGILL, &sa, &old);
    sigill_caught = 0;

    if (sigsetjmp(jbuf, 1) == 0) {
        *out = rd_sccr_l1();
        sigaction(SIGILL, &old, NULL);
        return 0; /* success */
    }
    sigaction(SIGILL, &old, NULL);
    return -1; /* SIGILL */
}

static int try_read_vsccr_l2(uint64_t *out) {
    struct sigaction sa = {.sa_handler = sigill_handler};
    struct sigaction old;
    sigaction(SIGILL, &sa, &old);
    sigill_caught = 0;

    if (sigsetjmp(jbuf, 1) == 0) {
        *out = rd_vsccr_l2();
        sigaction(SIGILL, &old, NULL);
        return 0;
    }
    sigaction(SIGILL, &old, NULL);
    return -1;
}

/* ===== Timer ===== */

static inline uint64_t rdtsc(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static inline uint64_t rdfreq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

static void flush_cache(void *ptr, size_t size) {
    char *p = (char *)ptr;
    for (size_t i = 0; i < size; i += 256)
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    __asm__ volatile("dsb ish" ::: "memory");
}

/* ===== xos_sclib API (from libsec.so) ===== */

/*
 * xos_sclib_init(struct { int fd; int mode; ... } *handle, int flag)
 * xos_sclib_set_l1_way(handle, struct { uint16_t sec0,sec1,sec2,sec3; } *ways)
 * xos_sclib_set_l2_way(handle, struct { uint16_t sec0,sec1; } *ways)
 * xos_sclib_get_cache_size(handle, struct { ... } *out)
 * xos_sclib_finalize(handle)
 */

typedef int (*fn_init_t)(void *handle, int flag);
typedef int (*fn_finalize_t)(void *handle);
typedef int (*fn_set_l1_way_t)(void *handle, void *ways);
typedef int (*fn_set_l2_way_t)(void *handle, void *ways);
typedef int (*fn_get_cache_size_t)(void *handle, void *out);
typedef int (*fn_get_cache_maxway_t)(void *handle, void *out);

static fn_init_t          p_init;
static fn_finalize_t      p_finalize;
static fn_set_l1_way_t    p_set_l1_way;
static fn_set_l2_way_t    p_set_l2_way;
static fn_get_cache_size_t  p_get_cache_size;
static fn_get_cache_maxway_t p_get_cache_maxway;

static int load_libsec(void) {
    void *h = dlopen("libsec.so", RTLD_NOW);
    if (!h) {
        printf("  dlopen(libsec.so): %s\n", dlerror());
        return -1;
    }
    p_init           = (fn_init_t)dlsym(h, "xos_sclib_init");
    p_finalize       = (fn_finalize_t)dlsym(h, "xos_sclib_finalize");
    p_set_l1_way     = (fn_set_l1_way_t)dlsym(h, "xos_sclib_set_l1_way");
    p_set_l2_way     = (fn_set_l2_way_t)dlsym(h, "xos_sclib_set_l2_way");
    p_get_cache_size = (fn_get_cache_size_t)dlsym(h, "xos_sclib_get_cache_size");
    p_get_cache_maxway = (fn_get_cache_maxway_t)dlsym(h, "xos_sclib_get_cache_maxway");

    if (!p_init || !p_finalize || !p_set_l1_way) {
        printf("  dlsym failed: init=%p finalize=%p set_l1=%p\n",
               p_init, p_finalize, p_set_l1_way);
        return -1;
    }
    printf("  libsec.so loaded: init=%p set_l1_way=%p set_l2_way=%p\n",
           p_init, p_set_l1_way, p_set_l2_way);
    return 0;
}

/* ===== Streaming benchmark ===== */

static __attribute__((noinline)) uint64_t stream_sum(float *data, size_t n, int iters) {
    svbool_t pg = svptrue_b32();
    volatile float sink = 0.0f;
    __asm__ volatile("" : "+r"(data));

    uint64_t start = rdtsc();
    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);
        for (size_t i = 0; i < n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &data[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }
        sink += svaddv_f32(pg, vsum);
    }
    uint64_t end = rdtsc();
    (void)sink;
    return end - start;
}

static void decode_l1_sccr(uint64_t val) {
    printf("    sec0=%lu sec1=%lu sec2=%lu sec3=%lu",
           val & 0x7, (val >> 4) & 0x7, (val >> 8) & 0x7, (val >> 12) & 0x7);
    /* upper bits [61:49] also decoded by libsec */
    uint64_t ub3 = (val >> 61) & 0x7;
    uint64_t ub2 = (val >> 57) & 0x7;
    uint64_t ub1 = (val >> 53) & 0x7;
    uint64_t ub0 = (val >> 49) & 0x7;
    if (ub3 | ub2 | ub1 | ub0)
        printf("  [upper: %lu,%lu,%lu,%lu]", ub0, ub1, ub2, ub3);
    printf("\n");
}

static void decode_l2_vsccr(uint64_t val) {
    printf("    sec_lo=%lu sec_hi=%lu", val & 0x1F, (val >> 8) & 0x1F);
    uint64_t ub1 = (val >> 51) & 0x1F;
    uint64_t ub0 = (val >> 59) & 0x1F;
    if (ub1 | ub0)
        printf("  [upper: %lu,%lu]", ub0, ub1);
    printf("\n");
}

int main(void) {
    printf("=== A64FX Sector Cache via libsec.so ===\n\n");

    /* Phase 0: Probe SCCR before libsec init */
    printf("--- Phase 0: SCCR access BEFORE libsec init ---\n");
    uint64_t l1_val, l2_val;
    int l1_ok = try_read_sccr_l1(&l1_val);
    int l2_ok = try_read_vsccr_l2(&l2_val);
    printf("  IMP_SCCR_L1_EL0:      %s", l1_ok == 0 ? "ACCESSIBLE" : "SIGILL");
    if (l1_ok == 0) { printf(" = 0x%016lx", l1_val); decode_l1_sccr(l1_val); }
    else printf("\n");
    printf("  IMP_SCCR_VSCCR_L2_EL0: %s", l2_ok == 0 ? "ACCESSIBLE" : "SIGILL");
    if (l2_ok == 0) { printf(" = 0x%016lx", l2_val); decode_l2_vsccr(l2_val); }
    else printf("\n");
    printf("\n");

    /* Phase 1: Load libsec.so and init */
    printf("--- Phase 1: Load libsec.so and init ---\n");
    if (load_libsec() != 0) {
        printf("  FAILED to load libsec.so\n");
        return 1;
    }

    /* xos_sclib_init needs a handle struct:
     * offset 0: fd (set to -1 to let it open /dev/xos_sec_normal)
     * offset 4: mode
     * The struct is ~1KB, zero-init it */
    char handle[1024];
    memset(handle, 0, sizeof(handle));
    /* fd = -1 means not yet opened, let init open it */
    *(int *)handle = -1;

    printf("  Calling xos_sclib_init...\n");
    int ret = p_init(handle, 0);
    printf("  xos_sclib_init returned: %d\n", ret);

    if (ret != 0) {
        printf("  INIT FAILED (ret=%d). Trying with flag=1...\n", ret);
        memset(handle, 0, sizeof(handle));
        *(int *)handle = -1;
        ret = p_init(handle, 1);
        printf("  xos_sclib_init(flag=1) returned: %d\n", ret);
    }
    printf("\n");

    /* Phase 2: Probe SCCR after libsec init */
    printf("--- Phase 2: SCCR access AFTER libsec init ---\n");
    l1_ok = try_read_sccr_l1(&l1_val);
    l2_ok = try_read_vsccr_l2(&l2_val);
    printf("  IMP_SCCR_L1_EL0:      %s", l1_ok == 0 ? "ACCESSIBLE" : "SIGILL");
    if (l1_ok == 0) { printf(" = 0x%016lx", l1_val); decode_l1_sccr(l1_val); }
    else printf("\n");
    printf("  IMP_SCCR_VSCCR_L2_EL0: %s", l2_ok == 0 ? "ACCESSIBLE" : "SIGILL");
    if (l2_ok == 0) { printf(" = 0x%016lx", l2_val); decode_l2_vsccr(l2_val); }
    else printf("\n");
    printf("\n");

    if (l1_ok != 0) {
        printf("  SCCR still not accessible after libsec init. Aborting.\n");
        p_finalize(handle);
        return 1;
    }

    /* Phase 3: Try set_l1_way with different configurations */
    printf("--- Phase 3: Configure L1 sector cache via xos_sclib_set_l1_way ---\n\n");

    /* struct for set_l1_way: 4 x uint16_t */
    struct { uint16_t sec0, sec1, sec2, sec3; } l1_ways;

    /* Config A: equal split (1,1,1,1) */
    l1_ways = (typeof(l1_ways)){1, 1, 1, 1};
    ret = p_set_l1_way(handle, &l1_ways);
    l1_val = rd_sccr_l1();
    printf("  set_l1_way(1,1,1,1) ret=%d  L1 reg=0x%016lx", ret, l1_val);
    decode_l1_sccr(l1_val);

    /* Config B: sector0 gets 4, rest get 0 */
    l1_ways = (typeof(l1_ways)){4, 0, 0, 0};
    ret = p_set_l1_way(handle, &l1_ways);
    l1_val = rd_sccr_l1();
    printf("  set_l1_way(4,0,0,0) ret=%d  L1 reg=0x%016lx", ret, l1_val);
    decode_l1_sccr(l1_val);

    /* Config C: sector0=2, sector1=2 (half-half) */
    l1_ways = (typeof(l1_ways)){2, 2, 0, 0};
    ret = p_set_l1_way(handle, &l1_ways);
    l1_val = rd_sccr_l1();
    printf("  set_l1_way(2,2,0,0) ret=%d  L1 reg=0x%016lx", ret, l1_val);
    decode_l1_sccr(l1_val);

    printf("\n");

    /* Phase 4: Benchmark with sector cache configs */
    printf("--- Phase 4: Streaming benchmark with sector cache ---\n\n");

    size_t buf_size = 256 * 1024;  /* 256 KB — fits in L1 (64KB) x4 */
    size_t n = buf_size / sizeof(float);
    float *data = NULL;
    posix_memalign((void **)&data, 256, buf_size);
    for (size_t i = 0; i < n; i++) data[i] = (float)(i % 100) / 10.0f;

    int iters = 500;
    uint64_t freq = rdfreq();

    /* Baseline: default (4,0,0,0) */
    l1_ways = (typeof(l1_ways)){4, 0, 0, 0};
    p_set_l1_way(handle, &l1_ways);
    flush_cache(data, buf_size);
    memory_fence();
    uint64_t cyc_base = stream_sum(data, n, iters);

    /* Half split: (2,2,0,0) */
    l1_ways = (typeof(l1_ways)){2, 2, 0, 0};
    p_set_l1_way(handle, &l1_ways);
    flush_cache(data, buf_size);
    memory_fence();
    uint64_t cyc_half = stream_sum(data, n, iters);

    /* Quarter: (1,1,1,1) */
    l1_ways = (typeof(l1_ways)){1, 1, 1, 1};
    p_set_l1_way(handle, &l1_ways);
    flush_cache(data, buf_size);
    memory_fence();
    uint64_t cyc_quarter = stream_sum(data, n, iters);

    /* Restore full */
    l1_ways = (typeof(l1_ways)){4, 0, 0, 0};
    p_set_l1_way(handle, &l1_ways);

    printf("  256KB × %d iters, timer freq=%lu Hz:\n\n", iters, freq);
    printf("  %-20s  %10s  %8s  %8s\n", "L1 Config", "Ticks", "GB/s", "Ratio");
    printf("  %-20s  %10s  %8s  %8s\n", "---------", "-----", "----", "-----");

    double bw_b = (double)buf_size * iters / (double)cyc_base * (double)freq / 1e9;
    double bw_h = (double)buf_size * iters / (double)cyc_half * (double)freq / 1e9;
    double bw_q = (double)buf_size * iters / (double)cyc_quarter * (double)freq / 1e9;

    printf("  %-20s  %10lu  %8.2f  %7.2fx\n", "full (4,0,0,0)", cyc_base, bw_b, 1.0);
    printf("  %-20s  %10lu  %8.2f  %7.2fx\n", "half (2,2,0,0)", cyc_half, bw_h,
           (double)cyc_base / cyc_half);
    printf("  %-20s  %10lu  %8.2f  %7.2fx\n", "quarter (1,1,1,1)", cyc_quarter, bw_q,
           (double)cyc_base / cyc_quarter);
    printf("\n");

    free(data);

    /* Cleanup */
    p_finalize(handle);
    printf("  xos_sclib_finalize done.\n\n");

    /* Phase 5: Verify SCCR access revoked */
    printf("--- Phase 5: SCCR access AFTER finalize ---\n");
    l1_ok = try_read_sccr_l1(&l1_val);
    printf("  IMP_SCCR_L1_EL0: %s\n", l1_ok == 0 ? "still ACCESSIBLE" : "SIGILL (revoked)");
    printf("\n");

    return 0;
}
