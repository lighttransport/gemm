/*
 * A64FX HPC Extension Register Probe
 *
 * Probes ALL implementation-specific system registers for A64FX HPC extensions
 * from EL0 (user space). Catches SIGILL to safely detect which registers the
 * Fugaku kernel exposes.
 *
 * ============================================================================
 * 1. SECTOR CACHE registers
 * ============================================================================
 *   IMP_SCCR_CTRL_EL1       S3_0_C11_C8_0   Read-only from EL0 (doc says)
 *     [63] el1ae, [62] el0ae
 *   IMP_SCCR_ASSIGN_EL1     S3_0_C11_C8_1   EL1+ only
 *     [3] mode, [2] assign, [1:0] default_sector
 *   IMP_SCCR_L1_EL0         S3_3_C11_C8_2   R/W from EL0 (if el0ae=1)
 *     [14:12] sec3, [10:8] sec2, [6:4] sec1, [2:0] sec0  (3-bit each)
 *   IMP_SCCR_SET0_L2_EL1    S3_0_C15_C8_2   EL1+ only
 *   IMP_SCCR_SET1_L2_EL1    S3_0_C15_C8_3   EL1+ only
 *   IMP_SCCR_VSCCR_L2_EL0   S3_3_C15_C8_2   R/W from EL0 (if el0ae=1)
 *     [12:8] sec_hi(5b), [4:0] sec_lo(5b)  — window into SET0 or SET1
 *
 * ============================================================================
 * 2. HARDWARE PREFETCH registers
 * ============================================================================
 *   IMP_PF_CTRL_EL1             S3_0_C11_C4_0   EL1 master prefetch control
 *   IMP_PF_STREAM_DETECT_CTRL_EL0  S3_3_C11_C4_0   EL0 stream detect config
 *     Controls number of HW prefetch streams, distances, L1/L2 enables
 *
 * ============================================================================
 * 3. TAG ADDRESS OVERRIDE registers
 * ============================================================================
 *   IMP_FJ_TAG_ADDRESS_CTRL_EL1  S3_0_C11_C2_0   EL1 tag override control
 *     Controls TBI, TBO, SCE, PFE — enables HPC tag address override
 *
 * ============================================================================
 * 4. HARDWARE BARRIER registers
 * ============================================================================
 *   EL1 control plane (kernel configures barrier groups at boot):
 *   IMP_BARRIER_CTRL_EL1         S3_0_C11_C12_0  EL1 barrier control
 *   IMP_BARRIER_BST_BIT_EL1      S3_0_C11_C12_4  EL1 BST base/mask config
 *   IMP_BARRIER_INIT_SYNC_BB0    S3_0_C11_C12_5  EL1 init sync
 *   IMP_BARRIER_INIT_SYNC_BB1    S3_0_C11_C12_6  EL1 init sync
 *   IMP_BARRIER_INIT_SYNC_BB2    S3_0_C11_C12_7  EL1 init sync
 *   IMP_BARRIER_ASSIGN_EL1       S3_0_C11_C15_0  EL1 barrier assignment
 *
 *   EL0 data plane (user-space barrier trigger):
 *   IMP_BARRIER_BST_EL0          S3_3_C15_C15_0  EL0 BST toggle-and-wait
 *     Bit [0]: barrier toggle. Always accessible from EL0 on Fugaku.
 *     Used by fjomplib (__mpc_obar) for #pragma omp barrier.
 *     Discovered by reverse-engineering libfj90i.so.1 (55 accesses).
 *
 *   NOTE: EL1 registers (S3_0_*) and EL0 BST (S3_3_C15_C15_0) are
 *   completely different register addresses, NOT endian-swapped.
 *
 * Build:
 *   fcc -Nclang -O2 -march=armv8.2-a+sve -o test_sccr_probe test_sccr_probe.c
 */

#include <stdio.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

/* ===== SIGILL-safe register access framework ===== */

static volatile sig_atomic_t got_fault = 0;
static sigjmp_buf fault_jmpbuf;

static void fault_handler(int sig) {
    got_fault = 1;
    siglongjmp(fault_jmpbuf, 1);
}

typedef uint64_t (*reg_read_fn)(void);
typedef void (*reg_write_fn)(uint64_t val);

static int safe_read(reg_read_fn fn, uint64_t *out) {
    struct sigaction sa, old_sa;
    sa.sa_handler = fault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old_sa);

    got_fault = 0;
    if (sigsetjmp(fault_jmpbuf, 1) == 0) {
        *out = fn();
    }

    sigaction(SIGILL, &old_sa, NULL);
    return !got_fault;
}

static int safe_write(reg_write_fn fn, uint64_t val) {
    struct sigaction sa, old_sa;
    sa.sa_handler = fault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old_sa);

    got_fault = 0;
    if (sigsetjmp(fault_jmpbuf, 1) == 0) {
        fn(val);
    }

    sigaction(SIGILL, &old_sa, NULL);
    return !got_fault;
}

/* ===== Register accessor functions ===== */
/* Each needs its own function — can't parametrize MSR/MRS encoding at runtime */

/* --- Sector Cache --- */

static uint64_t rd_sccr_ctrl(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C8_0" : "=r"(v)); return v;
}
static uint64_t rd_sccr_assign(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C8_1" : "=r"(v)); return v;
}
static uint64_t rd_sccr_l1(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_3_C11_C8_2" : "=r"(v)); return v;
}
static void wr_sccr_l1(uint64_t v) {
    __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(v));
    __asm__ volatile("isb" ::: "memory");
}
static uint64_t rd_sccr_set0_l2(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C15_C8_2" : "=r"(v)); return v;
}
static uint64_t rd_sccr_set1_l2(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C15_C8_3" : "=r"(v)); return v;
}
static uint64_t rd_vsccr_l2(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_3_C15_C8_2" : "=r"(v)); return v;
}
static void wr_vsccr_l2(uint64_t v) {
    __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(v));
    __asm__ volatile("isb" ::: "memory");
}

/* --- Hardware Prefetch --- */

static uint64_t rd_pf_ctrl(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C4_0" : "=r"(v)); return v;
}
static uint64_t rd_pf_stream_detect(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_3_C11_C4_0" : "=r"(v)); return v;
}
static void wr_pf_stream_detect(uint64_t v) {
    __asm__ volatile("msr S3_3_C11_C4_0, %0" :: "r"(v));
    __asm__ volatile("isb" ::: "memory");
}

/* --- Tag Address Override --- */

static uint64_t rd_tag_addr_ctrl(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C2_0" : "=r"(v)); return v;
}

/* --- Hardware Barrier --- */

static uint64_t rd_barrier_ctrl(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C12_0" : "=r"(v)); return v;
}
static uint64_t rd_barrier_bst(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C12_4" : "=r"(v)); return v;
}
static uint64_t rd_barrier_assign(void) {
    uint64_t v; __asm__ volatile("mrs %0, S3_0_C11_C15_0" : "=r"(v)); return v;
}

/* ===== Decode helpers ===== */

static void decode_sccr_ctrl(uint64_t val) {
    printf("    [63] el1ae=%d  (NS-EL1 R/W all SCCR: %s)\n",
           (int)((val >> 63) & 1), (val >> 63) & 1 ? "ENABLED" : "DISABLED");
    printf("    [62] el0ae=%d  (EL0 R/W L1/L2 SCCR: %s)\n",
           (int)((val >> 62) & 1), (val >> 62) & 1 ? "ENABLED" : "DISABLED");
}

static void decode_sccr_assign(uint64_t val) {
    printf("    [3]   mode=%d           (%s)\n",
           (int)((val >> 3) & 1),
           (val >> 3) & 1 ? "keep original sector ID" : "update sector ID on access");
    printf("    [2]   assign=%d         (VSCCR_L2 → %s)\n",
           (int)((val >> 2) & 1),
           (val >> 2) & 1 ? "SET1 (sectors 2,3)" : "SET0 (sectors 0,1)");
    printf("    [1:0] default_sector=%d\n", (int)(val & 3));
}

static void decode_sccr_l1(uint64_t val) {
    unsigned s0 = (val >> 0)  & 0x7;
    unsigned s1 = (val >> 4)  & 0x7;
    unsigned s2 = (val >> 8)  & 0x7;
    unsigned s3 = (val >> 12) & 0x7;
    printf("    sec0_max=%u  sec1_max=%u  sec2_max=%u  sec3_max=%u  (total=%u, L1D=4-way)\n",
           s0, s1, s2, s3, s0 + s1 + s2 + s3);
}

static void decode_sccr_l2(uint64_t val, const char* set_name) {
    unsigned lo = (val >> 0) & 0x1F;
    unsigned hi = (val >> 8) & 0x1F;
    printf("    %s: sec_lo_max=%u  sec_hi_max=%u  (total=%u, L2=16-way/14 usable)\n",
           set_name, lo, hi, lo + hi);
}

static void decode_pf_stream_detect(uint64_t val) {
    /* Known bit fields from A64FX spec (may vary by stepping) */
    printf("    raw value = 0x%016lx\n", val);
    printf("    (decode: consult A64FX Microarchitecture Manual for bit layout)\n");
    printf("    Typical fields: L1 streams, L2 streams, distances, enables\n");
}

static void decode_tag_addr_ctrl(uint64_t val) {
    printf("    raw value = 0x%016lx\n", val);
    printf("    Expected: TBI, TBO, SCE, PFE control bits\n");
}

/* ===== Probe one register ===== */

static void probe_read(const char* name, const char* encoding, const char* access,
                       reg_read_fn fn, void (*decode)(uint64_t)) {
    uint64_t val = 0;
    printf("  %-32s %s  (%s)\n", name, encoding, access);
    int ok = safe_read(fn, &val);
    if (ok) {
        printf("    READ OK: 0x%016lx\n", val);
        if (decode) decode(val);
    } else {
        printf("    READ FAILED (SIGILL)\n");
    }
}

static void probe_readwrite(const char* name, const char* encoding, const char* access,
                            reg_read_fn rfn, reg_write_fn wfn,
                            void (*decode)(uint64_t),
                            uint64_t test_val) {
    uint64_t val = 0;
    printf("  %-32s %s  (%s)\n", name, encoding, access);
    int rok = safe_read(rfn, &val);
    if (rok) {
        printf("    READ OK: 0x%016lx\n", val);
        if (decode) decode(val);

        /* Try write + readback */
        int wok = safe_write(wfn, test_val);
        if (wok) {
            uint64_t rb = 0;
            safe_read(rfn, &rb);
            printf("    WRITE OK: wrote 0x%016lx → readback 0x%016lx %s\n",
                   test_val, rb, rb == test_val ? "(MATCH)" : "(DIFFERS)");
            /* Restore */
            safe_write(wfn, val);
        } else {
            printf("    WRITE FAILED (SIGILL) — read-only at this EL\n");
        }
    } else {
        printf("    READ FAILED (SIGILL)\n");
    }
}

/* ===== Main ===== */

int main(int argc, char* argv[]) {
    int total = 0, accessible = 0;

    printf("=== A64FX HPC Extension Register Probe (EL0) ===\n\n");

    /* ---- Section 1: Sector Cache ---- */
    printf("━━━ 1. SECTOR CACHE ━━━\n\n");

    probe_read("IMP_SCCR_CTRL_EL1", "S3_0_C11_C8_0", "RO from EL0",
               rd_sccr_ctrl, decode_sccr_ctrl);
    { uint64_t v; total++; if (safe_read(rd_sccr_ctrl, &v)) accessible++; }
    printf("\n");

    probe_read("IMP_SCCR_ASSIGN_EL1", "S3_0_C11_C8_1", "EL1+ only",
               rd_sccr_assign, decode_sccr_assign);
    { uint64_t v; total++; if (safe_read(rd_sccr_assign, &v)) accessible++; }
    printf("\n");

    /* L1: test value = sec0=2, sec1=2, sec2=0, sec3=0 */
    probe_readwrite("IMP_SCCR_L1_EL0", "S3_3_C11_C8_2", "R/W EL0 (el0ae)",
                    rd_sccr_l1, wr_sccr_l1, decode_sccr_l1, 0x22ULL);
    { uint64_t v; total++; if (safe_read(rd_sccr_l1, &v)) accessible++; }
    printf("\n");

    probe_read("IMP_SCCR_SET0_L2_EL1", "S3_0_C15_C8_2", "EL1+ only",
               rd_sccr_set0_l2, NULL);
    { uint64_t v; total++; if (safe_read(rd_sccr_set0_l2, &v)) accessible++; }
    printf("\n");

    probe_read("IMP_SCCR_SET1_L2_EL1", "S3_0_C15_C8_3", "EL1+ only",
               rd_sccr_set1_l2, NULL);
    { uint64_t v; total++; if (safe_read(rd_sccr_set1_l2, &v)) accessible++; }
    printf("\n");

    /* L2 VSCCR: test value = lo=10, hi=4 */
    probe_readwrite("IMP_SCCR_VSCCR_L2_EL0", "S3_3_C15_C8_2", "R/W EL0 (el0ae)",
                    rd_vsccr_l2, wr_vsccr_l2, NULL, 0x40AULL);
    { uint64_t v; total++; if (safe_read(rd_vsccr_l2, &v)) accessible++; }
    printf("\n");

    /* ---- Section 2: Hardware Prefetch ---- */
    printf("━━━ 2. HARDWARE PREFETCH ━━━\n\n");

    probe_read("IMP_PF_CTRL_EL1", "S3_0_C11_C4_0", "EL1+ only",
               rd_pf_ctrl, NULL);
    { uint64_t v; total++; if (safe_read(rd_pf_ctrl, &v)) accessible++; }
    printf("\n");

    probe_readwrite("IMP_PF_STREAM_DETECT_CTRL_EL0", "S3_3_C11_C4_0", "R/W EL0",
                    rd_pf_stream_detect, wr_pf_stream_detect,
                    decode_pf_stream_detect, 0 /* don't actually change */);
    { uint64_t v; total++; if (safe_read(rd_pf_stream_detect, &v)) accessible++; }
    printf("\n");

    /* ---- Section 3: Tag Address Override ---- */
    printf("━━━ 3. TAG ADDRESS OVERRIDE ━━━\n\n");

    probe_read("IMP_FJ_TAG_ADDRESS_CTRL_EL1", "S3_0_C11_C2_0", "EL1+ only",
               rd_tag_addr_ctrl, decode_tag_addr_ctrl);
    { uint64_t v; total++; if (safe_read(rd_tag_addr_ctrl, &v)) accessible++; }
    printf("\n");

    /* ---- Section 4: Hardware Barrier ---- */
    printf("━━━ 4. HARDWARE BARRIER ━━━\n\n");

    probe_read("IMP_BARRIER_CTRL_EL1", "S3_0_C11_C12_0", "EL1+ only",
               rd_barrier_ctrl, NULL);
    { uint64_t v; total++; if (safe_read(rd_barrier_ctrl, &v)) accessible++; }
    printf("\n");

    probe_read("IMP_BARRIER_BST_BIT_EL1", "S3_0_C11_C12_4", "EL1+ only",
               rd_barrier_bst, NULL);
    { uint64_t v; total++; if (safe_read(rd_barrier_bst, &v)) accessible++; }
    printf("\n");

    probe_read("IMP_BARRIER_ASSIGN_EL1", "S3_0_C11_C15_0", "EL1+ only",
               rd_barrier_assign, NULL);
    { uint64_t v; total++; if (safe_read(rd_barrier_assign, &v)) accessible++; }
    printf("\n");

    /* ---- Summary ---- */
    printf("━━━ SUMMARY ━━━\n\n");
    printf("  Registers probed: %d\n", total);
    printf("  Accessible (EL0): %d\n\n", accessible);

    /* Re-probe key registers for summary table */
    struct {
        const char* name;
        reg_read_fn fn;
        const char* expected;
    } summary[] = {
        { "SCCR_CTRL",         rd_sccr_ctrl,         "RO from EL0" },
        { "SCCR_ASSIGN",       rd_sccr_assign,       "EL1 only" },
        { "SCCR_L1",           rd_sccr_l1,           "R/W EL0" },
        { "SCCR_SET0_L2",      rd_sccr_set0_l2,      "EL1 only" },
        { "SCCR_SET1_L2",      rd_sccr_set1_l2,      "EL1 only" },
        { "VSCCR_L2",          rd_vsccr_l2,          "R/W EL0" },
        { "PF_CTRL",           rd_pf_ctrl,           "EL1 only" },
        { "PF_STREAM_DETECT",  rd_pf_stream_detect,  "R/W EL0" },
        { "TAG_ADDR_CTRL",     rd_tag_addr_ctrl,     "EL1 only" },
        { "BARRIER_CTRL",      rd_barrier_ctrl,      "EL1 only" },
        { "BARRIER_BST",       rd_barrier_bst,       "EL1 only" },
        { "BARRIER_ASSIGN",    rd_barrier_assign,    "EL1 only" },
    };
    int n = sizeof(summary) / sizeof(summary[0]);

    printf("  %-22s  %-12s  %-12s  %s\n", "Register", "Expected", "Actual", "Value");
    printf("  %-22s  %-12s  %-12s  %s\n", "--------", "--------", "------", "-----");
    for (int i = 0; i < n; i++) {
        uint64_t v = 0;
        int ok = safe_read(summary[i].fn, &v);
        printf("  %-22s  %-12s  %-12s",
               summary[i].name,
               summary[i].expected,
               ok ? "ACCESSIBLE" : "TRAPPED");
        if (ok) printf("  0x%016lx", v);
        printf("\n");
    }
    printf("\n");

    if (accessible == 0) {
        printf("  No A64FX HPC extension registers are accessible from EL0.\n");
        printf("  The Fugaku kernel does not expose sector cache, prefetch,\n");
        printf("  tag override, or barrier registers to user space.\n");
    }

    return 0;
}
