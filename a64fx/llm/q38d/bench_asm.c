/* Single-core, L1-resident ablation of the FP4 A8 pair loop in assembly.
 * Reports core cycles per pair for several variants. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static uint8_t codes[160 * 128] __attribute__((aligned(256)));
static uint8_t scales[160 * 16] __attribute__((aligned(256)));
static int8_t actq[160 * 64] __attribute__((aligned(256)));
static float acts[160 * 2] __attribute__((aligned(256)));
static const int8_t lut[64] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

#define PROLOG \
    "ptrue p0.b\n ptrue p1.s\n ptrue p2.d\n ld1b {z31.b}, p0/z, [%[lut]]\n" \
    "dup z30.b, #15\n dup z29.s, #0\n dup z28.s, #0\n dup z27.s, #0\n"
/* one pair; A=register base for this pair's temporaries, OFF = pair offset k */
#define PAIR(k, ACC) \
    "ld1b {z0.b}, p0/z, [x10, #" #k "*2, mul vl]\n"                   \
    "ld1b {z1.b}, p0/z, [x10, #" #k "*2+1, mul vl]\n"                 \
    "lsr z2.b, z0.b, #4\n and z0.d, z0.d, z30.d\n"                    \
    "lsr z3.b, z1.b, #4\n and z1.d, z1.d, z30.d\n"                    \
    TBLS                                                              \
    "ld1rd {z4.d}, p2/z, [x12, #" #k "*32]\n"                         \
    "ld1rd {z5.d}, p2/z, [x12, #" #k "*32+8]\n"                       \
    "ld1rd {z6.d}, p2/z, [x12, #" #k "*32+16]\n"                      \
    "ld1rd {z7.d}, p2/z, [x12, #" #k "*32+24]\n"                      \
    "movprfx z8, z29\n sdot z8.s, z0.b, z4.b\n"                       \
    "movprfx z9, z29\n sdot z9.s, z2.b, z5.b\n"                       \
    "movprfx z10, z29\n sdot z10.s, z1.b, z6.b\n"                     \
    "movprfx z11, z29\n sdot z11.s, z3.b, z7.b\n"                     \
    "add z8.s, z8.s, z9.s\n add z10.s, z10.s, z11.s\n add z8.s, z8.s, z10.s\n" \
    "scvtf z8.s, p1/m, z8.s\n"                                        \
    SCALE(k)                                                          \
    "fmla " #ACC ".s, p1/m, z8.s, z12.s\n"
#define TBLS_ON "tbl z0.b, {z31.b}, z0.b\n tbl z2.b, {z31.b}, z2.b\n tbl z1.b, {z31.b}, z1.b\n tbl z3.b, {z31.b}, z3.b\n"
#define SCALE_ON(k) "ld1b {z12.s}, p1/z, [x11, #" #k ", mul vl]\n ld1rd {z13.d}, p2/z, [x13, #" #k "*8]\n" \
                    "lsl z12.s, z12.s, #20\n fmul z12.s, z12.s, z13.s\n"

#define KERNEL(name, TBLS_, SCALE_)                                                   \
static uint64_t name(long reps) {                                                     \
    uint64_t t0 = ticks();                                                           \
    for (long r = 0; r < reps; r++) {                                                \
        __asm__ volatile(PROLOG                                                      \
            "mov x10, %[c]\n mov x11, %[s]\n mov x12, %[q]\n mov x13, %[a]\n mov x14, #80\n" \
            "1:\n"                                                                   \
            BODY                                                                     \
            "add x10, x10, #256\n add x11, x11, #32\n add x12, x12, #64\n add x13, x13, #16\n" \
            "subs x14, x14, #1\n b.ne 1b\n"                                          \
            : : [c] "r"(codes), [s] "r"(scales), [q] "r"(actq), [a] "r"(acts), [lut] "r"(lut) \
            : "memory", "cc", "x10", "x11", "x12", "x13", "x14", "p0", "p1", "p2",   \
              "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13", \
              "z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31"); \
    }                                                                                \
    return ticks() - t0;                                                             \
}
#define TBLS TBLS_ON
#define SCALE(k) SCALE_ON(k)
#define BODY PAIR(0, z28) PAIR(1, z27)
KERNEL(k_full, 1, 1)
#undef TBLS
#define TBLS ""
KERNEL(k_notbl, 0, 1)
#undef TBLS
#define TBLS TBLS_ON
#undef SCALE
#define SCALE(k) "mov z12.d, z30.d\n"
KERNEL(k_noscale, 1, 0)
#undef SCALE
#define SCALE(k) SCALE_ON(k)
/* second register set for the odd pair to remove false WAR reuse */
#define PAIR2(k, ACC) \
    "ld1b {z14.b}, p0/z, [x10, #" #k "*2, mul vl]\n"                  \
    "ld1b {z15.b}, p0/z, [x10, #" #k "*2+1, mul vl]\n"                \
    "lsr z16.b, z14.b, #4\n and z14.d, z14.d, z30.d\n"                \
    "lsr z17.b, z15.b, #4\n and z15.d, z15.d, z30.d\n"                \
    "tbl z14.b, {z31.b}, z14.b\n tbl z16.b, {z31.b}, z16.b\n tbl z15.b, {z31.b}, z15.b\n tbl z17.b, {z31.b}, z17.b\n" \
    "ld1rd {z18.d}, p2/z, [x12, #" #k "*32]\n"                        \
    "ld1rd {z19.d}, p2/z, [x12, #" #k "*32+8]\n"                      \
    "ld1rd {z20.d}, p2/z, [x12, #" #k "*32+16]\n"                     \
    "ld1rd {z21.d}, p2/z, [x12, #" #k "*32+24]\n"                     \
    "movprfx z22, z29\n sdot z22.s, z14.b, z18.b\n"                   \
    "movprfx z23, z29\n sdot z23.s, z16.b, z19.b\n"                   \
    "movprfx z24, z29\n sdot z24.s, z15.b, z20.b\n"                   \
    "movprfx z25, z29\n sdot z25.s, z17.b, z21.b\n"                   \
    "add z22.s, z22.s, z23.s\n add z24.s, z24.s, z25.s\n add z22.s, z22.s, z24.s\n" \
    "scvtf z22.s, p1/m, z22.s\n"                                      \
    "ld1b {z26.s}, p1/z, [x11, #" #k ", mul vl]\n ld1rd {z13.d}, p2/z, [x13, #" #k "*8]\n" \
    "lsl z26.s, z26.s, #20\n fmul z26.s, z26.s, z13.s\n"              \
    "fmla " #ACC ".s, p1/m, z22.s, z26.s\n"
#undef BODY
#define BODY PAIR(0, z28) PAIR2(1, z27)
KERNEL(k_tworeg, 1, 1)
#undef BODY
#define BODY "ld1b {z0.b}, p0/z, [x10]\n ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n" \
    "tbl z4.b, {z31.b}, z0.b\n tbl z5.b, {z31.b}, z1.b\n tbl z6.b, {z31.b}, z2.b\n tbl z7.b, {z31.b}, z3.b\n" \
    "tbl z8.b, {z31.b}, z0.b\n tbl z9.b, {z31.b}, z1.b\n tbl z10.b, {z31.b}, z2.b\n tbl z11.b, {z31.b}, z3.b\n"
KERNEL(k_tblonly, 1, 0)
#undef BODY
#define BODY "ld1rd {z4.d}, p2/z, [x12]\n ld1rd {z5.d}, p2/z, [x12, #8]\n ld1rd {z6.d}, p2/z, [x12, #16]\n ld1rd {z7.d}, p2/z, [x12, #24]\n" \
    "movprfx z8, z29\n sdot z8.s, z0.b, z4.b\n movprfx z9, z29\n sdot z9.s, z2.b, z5.b\n movprfx z10, z29\n sdot z10.s, z1.b, z6.b\n movprfx z11, z29\n sdot z11.s, z3.b, z7.b\n" \
    "ld1rd {z18.d}, p2/z, [x12, #32]\n ld1rd {z19.d}, p2/z, [x12, #40]\n ld1rd {z20.d}, p2/z, [x12, #48]\n ld1rd {z21.d}, p2/z, [x12, #56]\n" \
    "movprfx z22, z29\n sdot z22.s, z14.b, z18.b\n movprfx z23, z29\n sdot z23.s, z16.b, z19.b\n movprfx z24, z29\n sdot z24.s, z15.b, z20.b\n movprfx z25, z29\n sdot z25.s, z17.b, z21.b\n"
KERNEL(k_sdotonly, 1, 0)

int main(void) {
    memset(scales, 60, sizeof scales);
    for (unsigned i = 0; i < sizeof codes; i++) codes[i] = (uint8_t)(i * 37);
    for (unsigned i = 0; i < sizeof actq; i++) actq[i] = (int8_t)(i * 13);
    for (unsigned i = 0; i < 320; i++) acts[i] = 1.0f;
    long reps = 20000;
    struct { const char *n; uint64_t (*f)(long); } k[] = {
        {"full", k_full}, {"no-tbl", k_notbl}, {"no-scale", k_noscale}, {"two-regsets", k_tworeg},
        {"tbl-only(8/iter)", k_tblonly}, {"sdot+ld1rd only(8/iter)", k_sdotonly}};
    for (int i = 0; i < 6; i++) {
        k[i].f(100);
        uint64_t t = k[i].f(reps);
        printf("%-26s %.2f cycles/pair\n", k[i].n, (double)t * 20.0 / (reps * 160.0));
    }
    return 0;
}
