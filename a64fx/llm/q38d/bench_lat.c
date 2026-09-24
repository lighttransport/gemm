/* Latency probe: one dependent chain per instruction type (8 per iteration). */
#include <stdint.h>
#include <stdio.h>
static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static float fbuf[64] __attribute__((aligned(256)));
#define S(x) #x
#define LAT(name, init, op)                                                        \
    static double name(long n) {                                                   \
        uint64_t t0 = ticks();                                                     \
        __asm__ volatile("ptrue p0.b\n ptrue p1.d\n ptrue p2.s\n"                   \
                         "ld1w {z30.s}, p2/z, [%1]\n ld1w {z31.s}, p2/z, [%1, #1, mul vl]\n" \
                         init "\n1:\n" op op op op op op op op                      \
                         "subs %0, %0, #1\n b.ne 1b\n"                             \
                         : "+r"(n) : "r"(fbuf)                                      \
                         : "memory", "cc", "x9", "z0", "z1", "z30", "z31", "p0", "p1", "p2"); \
        return (double)(ticks() - t0);                                             \
    }
LAT(l_add, "mov x9, #0", "add x9, x9, #1\n")
LAT(l_sdot, "dup z0.s, #0", "sdot z0.s, z30.b, z31.b\n")
LAT(l_tbl, "mov z0.d, z30.d", "tbl z0.b, {z0.b}, z31.b\n")
LAT(l_and, "mov z0.d, z30.d", "and z0.b, z0.b, #0xf\n")
LAT(l_lsr, "mov z0.d, z30.d", "lsr z0.b, z0.b, #1\n")
LAT(l_fmla, "fmov z0.s, #1.0", "fmla z0.s, p2/m, z30.s, z31.s\n")
LAT(l_scvtf, "mov z0.d, z30.d", "scvtf z0.s, p2/m, z0.s\n fcvtzs z0.s, p2/m, z0.s\n")
LAT(l_zip, "mov z0.d, z30.d", "zip1 z0.b, z0.b, z31.b\n")
LAT(l_add_v, "mov z0.d, z30.d", "add z0.s, z0.s, z31.s\n")
LAT(l_ldchain, "mov x9, %1", "ldr x9, [x9]\n")
int main(void) {
    for (int i = 0; i < 64; i++) fbuf[i] = 1.0f + i * 1e-3f;
    fbuf[0] = 0; /* pointer chase to itself: fbuf[0..1] = &fbuf */
    uint64_t self = (uint64_t)(uintptr_t)fbuf; __builtin_memcpy(fbuf, &self, 8);
    long n = 5000000;
    double add = l_add(n) / (8.0 * n); /* ticks per 1-cycle op */
    printf("ticks/cycle calibration: %.4f\n", add);
    struct { const char *nm; double (*f)(long); double div; } t[] = {
        {"sdot", l_sdot, 1}, {"tbl", l_tbl, 1}, {"and.imm", l_and, 1}, {"lsr.imm", l_lsr, 1},
        {"fmla", l_fmla, 1}, {"scvtf+fcvtzs", l_scvtf, 1}, {"zip1", l_zip, 1}, {"add.s", l_add_v, 1},
        {"ldr(L1)", l_ldchain, 1}};
    for (unsigned i = 0; i < sizeof(t)/sizeof(t[0]); i++)
        printf("%-14s latency %.2f cycles\n", t[i].nm, t[i].f(n) / (8.0 * n) / add);
    return 0;
}
