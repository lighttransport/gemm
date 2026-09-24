/* Per-instruction throughput probe: 8 independent streams per iteration. */
#include <stdint.h>
#include <stdio.h>
static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static uint64_t freq(void) { uint64_t v; __asm__ volatile("mrs %0,cntfrq_el0" : "=r"(v)); return v; }
static int8_t buf[4096] __attribute__((aligned(256)));

#define R8(op) op(0) op(1) op(2) op(3) op(4) op(5) op(6) op(7)
#define LOOP(name, body)                                                             \
    static double name(long n) {                                                     \
        uint64_t t0 = ticks();                                                       \
        __asm__ volatile("ptrue p0.b\n ptrue p1.d\n ptrue p2.s\n"                     \
                         "dup z30.b, #1\n dup z31.b, #3\n"                           \
                         "1:\n" body "subs %0, %0, #1\n b.ne 1b\n"                   \
                         : "+r"(n) : "r"(buf)                                        \
                         : "memory", "cc", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", \
                           "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z27", "z28", "z29", "z30", "z31", "p0", "p1", "p2"); \
        return (double)(ticks() - t0);                                               \
    }
#define S(x) #x
#define TBL(i) "tbl z" S(i) ".b, {z30.b}, z31.b\n"
#define SDOT(i) "sdot z" S(i) ".s, z30.b, z31.b\n"
#define AND(i) "and z" S(i) ".b, z" S(i) ".b, #0xf\n"
#define LSR(i) "lsr z" S(i) ".b, z31.b, #4\n"
#define LD1RD(i) "ld1rd {z" S(i) ".d}, p1/z, [%1, #" S(i) "*8]\n"
#define LD1B(i) "ld1b {z" S(i) ".b}, p0/z, [%1, #" S(i) ", mul vl]\n"
#define SCVTF(i) "scvtf z" S(i) ".s, p2/m, z31.s\n"
#define FMLA(i) "fmla z" S(i) ".s, p2/m, z30.s, z31.s\n"
#define MOVPRFX_SDOT(i) "movprfx z" S(i) ", z29\n sdot z" S(i) ".s, z30.b, z31.b\n"
#define ZIP(i) "zip1 z" S(i) ".b, z30.b, z31.b\n"
#define ORR(i) "orr z" S(i) ".d, z30.d, z31.d\n"
#define LSL32(i) "lsl z" S(i) ".s, z31.s, #20\n"
#define FMUL(i) "fmul z" S(i) ".s, z30.s, z31.s\n"
#define MOVI(i) "movi v" S(i) ".2d, #0\n"
#define DUPZ(i) "dup z" S(i) ".s, #0\n"
#define LD1BS(i) "ld1b {z" S(i) ".s}, p2/z, [%1, #" S(i) ", mul vl]\n"
#define ADD(i) "add z" S(i) ".s, z30.s, z31.s\n"
#define ANDIMM_U(i) "movprfx z" S(i) ", z31\n and z" S(i) ".b, z" S(i) ".b, #0xf\n"
#define ANDVEC(i) "and z" S(i) ".d, z31.d, z30.d\n"
#define SCVTF_U(i) "movprfx z" S(i) ", z31\n scvtf z" S(i) ".s, p2/m, z31.s\n"
#define FMLA_U(i) "movprfx z" S(i) ", z29\n fmla z" S(i) ".s, p2/m, z28.s, z27.s\n"
#define FMLA_C(i) "fmla z" S(i) ".s, p2/m, z28.s, z27.s\n"
#define MOVZ(i) "mov z" S(i) ".d, z29.d\n"
#define EOR0(i) "eor z" S(i) ".d, z" S(i) ".d, z" S(i) ".d\n"
#define LSRV(i) "lsr z" S(i) ".b, z31.b, #4\n tbl z" S(i+8) ".b, {z30.b}, z" S(i) ".b\n"
#define SUNPK(i) "sunpklo z" S(i) ".h, z31.b\n"
#define UZP(i) "uzp1 z" S(i) ".s, z31.s, z30.s\n"
#define ADDV(i) "add z" S(i) ".s, z" S(i) ".s, z30.s\n"
LOOP(t_andimm_u, R8(ANDIMM_U))
LOOP(t_andvec, R8(ANDVEC))
LOOP(t_scvtf_u, R8(SCVTF_U))
LOOP(t_fmla_u, "fmov z27.s, #1.0\n fmov z28.s, #1.0\n fmov z29.s, #1.0\n" R8(FMLA_U))
LOOP(t_movz, R8(MOVZ))
LOOP(t_eor0, R8(EOR0))
LOOP(t_sunpk, R8(SUNPK))
LOOP(t_uzp, R8(UZP))
LOOP(t_tbl, R8(TBL))
LOOP(t_sdot, R8(SDOT))
LOOP(t_and, R8(AND))
LOOP(t_lsr, R8(LSR))
LOOP(t_ld1rd, R8(LD1RD))
LOOP(t_ld1b, R8(LD1B))
LOOP(t_scvtf, R8(SCVTF))
LOOP(t_fmla, R8(FMLA))
LOOP(t_mpsdot, R8(MOVPRFX_SDOT))
LOOP(t_zip, R8(ZIP))
LOOP(t_orr, R8(ORR))
LOOP(t_lsl32, R8(LSL32))
LOOP(t_fmul, R8(FMUL))
LOOP(t_movi, R8(MOVI))
LOOP(t_dupz, R8(DUPZ))
LOOP(t_ld1bs, R8(LD1BS))
LOOP(t_add, R8(ADD))
LOOP(t_tbl_sdot, R8(TBL) R8(SDOT))
LOOP(t_and_tbl, R8(AND) R8(TBL))
LOOP(t_sdot_ld1rd, R8(SDOT) R8(LD1RD))

int main(void) {
    long n = 20000000;
    double hz = (double)freq(), cyc = 2.0e9 / hz; /* 2 GHz core cycles per tick */
    struct { const char *name; double (*f)(long); int k; } t[] = {
        {"tbl", t_tbl, 8}, {"sdot", t_sdot, 8}, {"and.imm", t_and, 8}, {"lsr.imm", t_lsr, 8},
        {"ld1rd", t_ld1rd, 8}, {"ld1b", t_ld1b, 8}, {"scvtf", t_scvtf, 8}, {"fmla", t_fmla, 8},
        {"movprfx+sdot", t_mpsdot, 8}, {"zip1", t_zip, 8}, {"orr", t_orr, 8}, {"lsl.s", t_lsl32, 8},
        {"fmul", t_fmul, 8}, {"movi0", t_movi, 8}, {"dup0", t_dupz, 8}, {"ld1b.s", t_ld1bs, 8},
        {"add.s", t_add, 8}, {"and.imm(unchained,+movprfx)", t_andimm_u, 8}, {"and.vec", t_andvec, 8},
        {"scvtf(unchained)", t_scvtf_u, 8}, {"fmla(unchained,normal)", t_fmla_u, 8}, {"mov z,z", t_movz, 8},
        {"eor zero", t_eor0, 8}, {"sunpklo", t_sunpk, 8}, {"uzp1", t_uzp, 8}, {"tbl+sdot", t_tbl_sdot, 16}, {"and+tbl", t_and_tbl, 16},
        {"sdot+ld1rd", t_sdot_ld1rd, 16}};
    for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        double tk = t[i].f(n);
        printf("%-14s %.3f cycles/insn (%.2f insn/cycle)\n", t[i].name, tk * cyc / n / t[i].k,
               n * (double)t[i].k / (tk * cyc));
    }
    return 0;
}
