/* bench_w8a16_ceiling.c — isolate the COMPUTE-pipe cost of the w8a16 int16-SDOT widen.
 * Real K-loop over an L1-resident tile (so loads/widens actually execute — no invariant
 * hoisting), tile replayed `reps` times, single core, cycle-counted. This is the prefill-GEMM
 * regime (weight tile reused across tokens) where the widen fixup competes with svdot_s64 on
 * the FLA/FLB pipes. Decode matvec is memory-BW bound and does NOT benefit from this.
 *
 *   4row_5x : 4 weight rows x 5 token vectors = 20 svdot_s64 + 4 widen per K-step (prefill)
 *   8row_mv : 8 weight rows x 1 token vector  =  8 svdot_s64 + 8 widen per K-step (decode tile)
 *
 * variants: sub (ld1ub+sub), biasfold (ld1ub only), unpk (ld1_u8 + uunpklo/hi, 2 K-vecs/load).
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast bench_w8a16_ceiling.c -o bench_w8a16_ceiling
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_sve.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

#define KT 1024                 /* L1-resident K tile (int16 elems): 4*2KB W + 5*2KB x = 18KB < 64KB L1 */
#define WSUB(P)  svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,(P))),128)
#define WBF(P)   svreinterpret_s16_u16(svld1ub_u16(pg,(P)))

/* 4row x 5tok prefill shape */
#define K4X5(WIDEN) \
    svint64_t d00=svdup_s64(0),d01=svdup_s64(0),d02=svdup_s64(0),d03=svdup_s64(0),d04=svdup_s64(0); \
    svint64_t d10=svdup_s64(0),d11=svdup_s64(0),d12=svdup_s64(0),d13=svdup_s64(0),d14=svdup_s64(0); \
    svint64_t d20=svdup_s64(0),d21=svdup_s64(0),d22=svdup_s64(0),d23=svdup_s64(0),d24=svdup_s64(0); \
    svint64_t d30=svdup_s64(0),d31=svdup_s64(0),d32=svdup_s64(0),d33=svdup_s64(0),d34=svdup_s64(0); \
    svbool_t pg=svptrue_b16(); int vh=(int)svcnth(); \
    for(int64_t rp=0;rp<reps;rp++) for(int k=0;k<KT;k+=vh){ \
        svint16_t v0=WIDEN(w0+k),v1=WIDEN(w1+k),v2=WIDEN(w2+k),v3=WIDEN(w3+k); \
        svint16_t x0=svld1_s16(pg,xq0+k),x1=svld1_s16(pg,xq1+k),x2=svld1_s16(pg,xq2+k),x3=svld1_s16(pg,xq3+k),x4=svld1_s16(pg,xq4+k); \
        d00=svdot_s64(d00,v0,x0);d01=svdot_s64(d01,v0,x1);d02=svdot_s64(d02,v0,x2);d03=svdot_s64(d03,v0,x3);d04=svdot_s64(d04,v0,x4); \
        d10=svdot_s64(d10,v1,x0);d11=svdot_s64(d11,v1,x1);d12=svdot_s64(d12,v1,x2);d13=svdot_s64(d13,v1,x3);d14=svdot_s64(d14,v1,x4); \
        d20=svdot_s64(d20,v2,x0);d21=svdot_s64(d21,v2,x1);d22=svdot_s64(d22,v2,x2);d23=svdot_s64(d23,v2,x3);d24=svdot_s64(d24,v2,x4); \
        d30=svdot_s64(d30,v3,x0);d31=svdot_s64(d31,v3,x1);d32=svdot_s64(d32,v3,x2);d33=svdot_s64(d33,v3,x3);d34=svdot_s64(d34,v3,x4); \
    } \
    svint64_t r0=svadd_s64_x(pg,svadd_s64_x(pg,d00,d01),svadd_s64_x(pg,d02,d03)); \
    svint64_t r1=svadd_s64_x(pg,svadd_s64_x(pg,d04,d10),svadd_s64_x(pg,d11,d12)); \
    svint64_t r2=svadd_s64_x(pg,svadd_s64_x(pg,d13,d14),svadd_s64_x(pg,d20,d21)); \
    svint64_t r3=svadd_s64_x(pg,svadd_s64_x(pg,d22,d23),svadd_s64_x(pg,d24,d30)); \
    svint64_t r4=svadd_s64_x(pg,svadd_s64_x(pg,d31,d32),svadd_s64_x(pg,d33,d34)); \
    svint64_t s=svadd_s64_x(pg,svadd_s64_x(pg,r0,r1),svadd_s64_x(pg,svadd_s64_x(pg,r2,r3),r4)); \
    return svaddv_s64(svptrue_b64(),s);

static long k4x5_sub(const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const int16_t*xq0,const int16_t*xq1,const int16_t*xq2,const int16_t*xq3,const int16_t*xq4,int64_t reps){ K4X5(WSUB) }
static long k4x5_bf(const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const int16_t*xq0,const int16_t*xq1,const int16_t*xq2,const int16_t*xq3,const int16_t*xq4,int64_t reps){ K4X5(WBF) }

/* unpk 4row: one ld1_u8 (64B) per weight row -> lo/hi cover 2 K-vectors. Token vecs also 64-wide.
 * 4 rows x 5 tok but each svdot covers a half; 40 svdot per K-step of 64, i.e. per 2 K-vecs. */
static long k4x5_unpk(const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const int16_t*xq0,const int16_t*xq1,const int16_t*xq2,const int16_t*xq3,const int16_t*xq4,int64_t reps){
    svbool_t pg=svptrue_b16(),p8=svptrue_b8(); int vb=(int)svcntb(),vh=(int)svcnth();
    svint64_t d00=svdup_s64(0),d01=svdup_s64(0),d02=svdup_s64(0),d03=svdup_s64(0),d04=svdup_s64(0);
    svint64_t d10=svdup_s64(0),d11=svdup_s64(0),d12=svdup_s64(0),d13=svdup_s64(0),d14=svdup_s64(0);
    svint64_t d20=svdup_s64(0),d21=svdup_s64(0),d22=svdup_s64(0),d23=svdup_s64(0),d24=svdup_s64(0);
    svint64_t d30=svdup_s64(0),d31=svdup_s64(0),d32=svdup_s64(0),d33=svdup_s64(0),d34=svdup_s64(0);
    for(int64_t rp=0;rp<reps;rp++) for(int k=0;k<KT;k+=vb){
        svuint8_t r0=svld1_u8(p8,w0+k),r1=svld1_u8(p8,w1+k),r2=svld1_u8(p8,w2+k),r3=svld1_u8(p8,w3+k);
        svint16_t v0l=svreinterpret_s16_u16(svunpklo_u16(r0)),v0h=svreinterpret_s16_u16(svunpkhi_u16(r0));
        svint16_t v1l=svreinterpret_s16_u16(svunpklo_u16(r1)),v1h=svreinterpret_s16_u16(svunpkhi_u16(r1));
        svint16_t v2l=svreinterpret_s16_u16(svunpklo_u16(r2)),v2h=svreinterpret_s16_u16(svunpkhi_u16(r2));
        svint16_t v3l=svreinterpret_s16_u16(svunpklo_u16(r3)),v3h=svreinterpret_s16_u16(svunpkhi_u16(r3));
        svint16_t x0l=svld1_s16(pg,xq0+k),x0h=svld1_s16(pg,xq0+k+vh);
        svint16_t x1l=svld1_s16(pg,xq1+k),x1h=svld1_s16(pg,xq1+k+vh);
        svint16_t x2l=svld1_s16(pg,xq2+k),x2h=svld1_s16(pg,xq2+k+vh);
        svint16_t x3l=svld1_s16(pg,xq3+k),x3h=svld1_s16(pg,xq3+k+vh);
        svint16_t x4l=svld1_s16(pg,xq4+k),x4h=svld1_s16(pg,xq4+k+vh);
        d00=svdot_s64(d00,v0l,x0l);d00=svdot_s64(d00,v0h,x0h);d01=svdot_s64(d01,v0l,x1l);d01=svdot_s64(d01,v0h,x1h);
        d02=svdot_s64(d02,v0l,x2l);d02=svdot_s64(d02,v0h,x2h);d03=svdot_s64(d03,v0l,x3l);d03=svdot_s64(d03,v0h,x3h);d04=svdot_s64(d04,v0l,x4l);d04=svdot_s64(d04,v0h,x4h);
        d10=svdot_s64(d10,v1l,x0l);d10=svdot_s64(d10,v1h,x0h);d11=svdot_s64(d11,v1l,x1l);d11=svdot_s64(d11,v1h,x1h);
        d12=svdot_s64(d12,v1l,x2l);d12=svdot_s64(d12,v1h,x2h);d13=svdot_s64(d13,v1l,x3l);d13=svdot_s64(d13,v1h,x3h);d14=svdot_s64(d14,v1l,x4l);d14=svdot_s64(d14,v1h,x4h);
        d20=svdot_s64(d20,v2l,x0l);d20=svdot_s64(d20,v2h,x0h);d21=svdot_s64(d21,v2l,x1l);d21=svdot_s64(d21,v2h,x1h);
        d22=svdot_s64(d22,v2l,x2l);d22=svdot_s64(d22,v2h,x2h);d23=svdot_s64(d23,v2l,x3l);d23=svdot_s64(d23,v2h,x3h);d24=svdot_s64(d24,v2l,x4l);d24=svdot_s64(d24,v2h,x4h);
        d30=svdot_s64(d30,v3l,x0l);d30=svdot_s64(d30,v3h,x0h);d31=svdot_s64(d31,v3l,x1l);d31=svdot_s64(d31,v3h,x1h);
        d32=svdot_s64(d32,v3l,x2l);d32=svdot_s64(d32,v3h,x2h);d33=svdot_s64(d33,v3l,x3l);d33=svdot_s64(d33,v3h,x3h);d34=svdot_s64(d34,v3l,x4l);d34=svdot_s64(d34,v3h,x4h);
    }
    svbool_t pg2=svptrue_b16();
    svint64_t r0=svadd_s64_x(pg2,svadd_s64_x(pg2,d00,d01),svadd_s64_x(pg2,d02,d03));
    svint64_t r1=svadd_s64_x(pg2,svadd_s64_x(pg2,d04,d10),svadd_s64_x(pg2,d11,d12));
    svint64_t r2=svadd_s64_x(pg2,svadd_s64_x(pg2,d13,d14),svadd_s64_x(pg2,d20,d21));
    svint64_t r3=svadd_s64_x(pg2,svadd_s64_x(pg2,d22,d23),svadd_s64_x(pg2,d24,d30));
    svint64_t r4=svadd_s64_x(pg2,svadd_s64_x(pg2,d31,d32),svadd_s64_x(pg2,d33,d34));
    svint64_t s=svadd_s64_x(pg2,svadd_s64_x(pg2,r0,r1),svadd_s64_x(pg2,svadd_s64_x(pg2,r2,r3),r4));
    return svaddv_s64(svptrue_b64(),s);
}

/* 8row x 1tok decode tile shape */
#define K8X1(WIDEN) \
    svint64_t d0=svdup_s64(0),d1=svdup_s64(0),d2=svdup_s64(0),d3=svdup_s64(0); \
    svint64_t d4=svdup_s64(0),d5=svdup_s64(0),d6=svdup_s64(0),d7=svdup_s64(0); \
    svbool_t pg=svptrue_b16(); int vh=(int)svcnth(); \
    for(int64_t rp=0;rp<reps;rp++) for(int k=0;k<KT;k+=vh){ \
        svint16_t xv=svld1_s16(pg,xq0+k); \
        d0=svdot_s64(d0,WIDEN(w0+k),xv);d1=svdot_s64(d1,WIDEN(w1+k),xv); \
        d2=svdot_s64(d2,WIDEN(w2+k),xv);d3=svdot_s64(d3,WIDEN(w3+k),xv); \
        d4=svdot_s64(d4,WIDEN(w4+k),xv);d5=svdot_s64(d5,WIDEN(w5+k),xv); \
        d6=svdot_s64(d6,WIDEN(w6+k),xv);d7=svdot_s64(d7,WIDEN(w7+k),xv); \
    } \
    svint64_t s=svadd_s64_x(pg,svadd_s64_x(pg,svadd_s64_x(pg,d0,d1),svadd_s64_x(pg,d2,d3)),svadd_s64_x(pg,svadd_s64_x(pg,d4,d5),svadd_s64_x(pg,d6,d7))); \
    return svaddv_s64(svptrue_b64(),s);
static long k8x1_sub(const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,const int16_t*xq0,int64_t reps){ K8X1(WSUB) }
static long k8x1_bf(const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,const int16_t*xq0,int64_t reps){ K8X1(WBF) }

int main(int argc,char**argv){
    int64_t reps=argc>1?atoll(argv[1]):200000;
    double freq=(double)rdfreq();
    uint8_t*w[8]; int16_t*xq[5];
    for(int i=0;i<8;i++){ w[i]=(uint8_t*)aligned_alloc(256,KT); for(int k=0;k<KT;k++) w[i][k]=(uint8_t)((i*7+k*131+3)&0xff); }
    for(int i=0;i<5;i++){ xq[i]=(int16_t*)aligned_alloc(256,KT*2); for(int k=0;k<KT;k++) xq[i][k]=(int16_t)((i*3+k*97+11)%2001-1000); }
    double ksteps=(double)reps*(KT/32.0);

    #define BENCH(NAME,CALL,SDOTS) do{ CALL; \
        uint64_t c0=rdcyc(); volatile long v=(CALL); (void)v; uint64_t c1=rdcyc(); \
        double cyc=(double)(c1-c0)/freq*2.0e9; double cps=cyc/ksteps; double sdpc=(double)(SDOTS)/cps; \
        printf("  %-11s %6.2f cyc/Kstep  %.2f SDOT/cyc  (%d SDOT/Kstep of 32, peak 2.0)\n",NAME,cps,sdpc,SDOTS); }while(0)

    printf("w8a16 int16-SDOT compute ceiling (L1 tile KT=%d, 1 core, reps=%lld)\n",KT,(long long)reps);
    printf("4row x 5tok (prefill, 20 SDOT + 4 widen / 32-Kstep):\n");
    BENCH("sub",     k4x5_sub(w[0],w[1],w[2],w[3],xq[0],xq[1],xq[2],xq[3],xq[4],reps), 20);
    BENCH("biasfold",k4x5_bf (w[0],w[1],w[2],w[3],xq[0],xq[1],xq[2],xq[3],xq[4],reps), 20);
    BENCH("unpk",    k4x5_unpk(w[0],w[1],w[2],w[3],xq[0],xq[1],xq[2],xq[3],xq[4],reps), 20); /* 40 SDOT / 64-Kstep = 20 / 32 */
    printf("8row x 1tok (decode tile, 8 SDOT + 8 widen / 32-Kstep):\n");
    BENCH("sub",     k8x1_sub(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],xq[0],reps), 8);
    BENCH("biasfold",k8x1_bf (w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],xq[0],reps), 8);
    for(int i=0;i<8;i++) free(w[i]); for(int i=0;i<5;i++) free(xq[i]);
    return 0;
}
