/* Instruction-mix throughput probe (cycles per loop iteration). */
#include <stdint.h>
#include <stdio.h>
static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static int8_t buf[4096] __attribute__((aligned(256)));
#define LOOP(name, body)                                                              \
    static double name(long n) {                                                      \
        uint64_t t0 = ticks(); long n0 = n;                                            \
        __asm__ volatile("ptrue p0.b\n ptrue p1.d\n ptrue p2.s\n dup z30.b, #1\n dup z31.b, #3\n dup z29.s, #0\n" \
                         "mov x9, #0\n mov x8, #0\n"                                   \
                         "1:\n" body "subs %0, %0, #1\n b.ne 1b\n"                    \
                         : "+r"(n) : "r"(buf) : "memory", "cc", "x8", "x9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", \
                           "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z29", "z30", "z31", "p0", "p1", "p2"); \
        return (double)(ticks() - t0) * 20.0 / n0;                                    \
    }
#define T4 "tbl z0.b, {z30.b}, z31.b\n tbl z1.b, {z30.b}, z31.b\n tbl z2.b, {z30.b}, z31.b\n tbl z3.b, {z30.b}, z31.b\n"
#define ADD8 "add z4.s, z30.s, z31.s\n add z5.s, z30.s, z31.s\n add z6.s, z30.s, z31.s\n add z7.s, z30.s, z31.s\n add z8.s, z30.s, z31.s\n add z9.s, z30.s, z31.s\n add z10.s, z30.s, z31.s\n add z11.s, z30.s, z31.s\n"
#define MS4 "movprfx z12, z29\n sdot z12.s, z30.b, z31.b\n movprfx z13, z29\n sdot z13.s, z30.b, z31.b\n movprfx z14, z29\n sdot z14.s, z30.b, z31.b\n movprfx z15, z29\n sdot z15.s, z30.b, z31.b\n"
#define LS4 "ld1rd {z12.d}, p1/z, [%1]\n ld1rd {z13.d}, p1/z, [%1, #8]\n ld1rd {z14.d}, p1/z, [%1, #16]\n ld1rd {z15.d}, p1/z, [%1, #24]\n sdot z12.s, z30.b, z31.b\n sdot z13.s, z30.b, z31.b\n sdot z14.s, z30.b, z31.b\n sdot z15.s, z30.b, z31.b\n"
#define SC8 "add x9, x9, #1\n add x8, x8, #1\n add x9, x9, #1\n add x8, x8, #1\n add x9, x9, #1\n add x8, x8, #1\n add x9, x9, #1\n add x8, x8, #1\n"
#define S8 "sdot z12.s, z30.b, z31.b\n sdot z13.s, z30.b, z31.b\n sdot z14.s, z30.b, z31.b\n sdot z15.s, z30.b, z31.b\n sdot z16.s, z30.b, z31.b\n sdot z17.s, z30.b, z31.b\n sdot z18.s, z30.b, z31.b\n sdot z19.s, z30.b, z31.b\n"
#define S16 S8 "sdot z20.s, z30.b, z31.b\n sdot z21.s, z30.b, z31.b\n sdot z22.s, z30.b, z31.b\n sdot z23.s, z30.b, z31.b\n sdot z0.s, z30.b, z31.b\n sdot z1.s, z30.b, z31.b\n sdot z2.s, z30.b, z31.b\n sdot z3.s, z30.b, z31.b\n"
LOOP(m_t4, T4)
LOOP(m_add8, ADD8)
LOOP(m_t4add8, T4 ADD8)
LOOP(m_ms4, MS4)
LOOP(m_ls4, LS4)
LOOP(m_t4ms4, T4 MS4)
LOOP(m_add8sc8, ADD8 SC8)
LOOP(m_s16, S16)
LOOP(m_t4ls4, T4 LS4)
int main(void) {
    long n = 10000000;
    struct { const char *nm; double (*f)(long); } t[] = {
        {"4 tbl", m_t4}, {"8 add.v", m_add8}, {"4 tbl + 8 add.v", m_t4add8}, {"4 movprfx+sdot", m_ms4},
        {"4 ld1rd + 4 sdot(acc on load)", m_ls4}, {"4 tbl + 4 movprfx+sdot", m_t4ms4},
        {"8 add.v + 8 scalar add", m_add8sc8}, {"16 sdot chained(16 chains)", m_s16}, {"4 tbl + 4 ld1rd+sdot", m_t4ls4}};
    for (unsigned i = 0; i < sizeof t / sizeof t[0]; i++) printf("%-34s %.2f cycles/iter\n", t[i].nm, t[i].f(n));
    return 0;
}
