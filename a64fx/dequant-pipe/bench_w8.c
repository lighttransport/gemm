#define _GNU_SOURCE
#include "w8.h"
#include "fused_opt.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

enum { K = 128, MAX_CORES = 12 };
static const char *formats[] = {"int8", "e4m3", "e5m2"};
static void (*const half_kernels[])(const uint8_t *, const _Float16 *, _Float16 *) = {
    w8_i8_f16, w8_e4m3_f16, w8_e5m2_f16
};
static void (*const float_kernels[])(const uint8_t *, const float *, float *) = {
    w8_i8_f32, w8_e4m3_f32, w8_e5m2_f32
};
extern void hbm_read_256_sve(const uint8_t *, size_t);
static uint64_t ticks(void) { uint64_t t; __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(t)); return t; }
static uint64_t frequency(void) { uint64_t t; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(t)); return t; }
static int pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); return pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
static _Float16 half_fma(_Float16 w, _Float16 a, _Float16 c)
{
    _Float16 r;
    __asm__("fmadd %h0, %h1, %h2, %h3" : "=w"(r) : "w"(w), "w"(a), "w"(c));
    return r;
}
static float decode(unsigned q, int format)
{
    if (!format) return (float)(int8_t)q;
    int mb = format == 1 ? 3 : 2, bias = format == 1 ? 7 : 15;
    unsigned e = (q & 127) >> mb, m = q & ((1u << mb) - 1);
    float v;
    if (format == 1 && (q & 127) == 127) v = NAN;
    else if (format == 2 && e == 31) v = m ? NAN : INFINITY;
    else v = e ? ldexpf(1.0f + (float)m / (1u << mb), (int)e - bias) :
                 ldexpf((float)m, 1 - bias - mb);
    return copysignf(v, q & 128 ? -1.0f : 1.0f);
}
int verify_w8(void)
{
    uint8_t w[K * 256] __attribute__((aligned(256)));
    _Float16 ah[K], yh[256], native[256];
    float af[K], yf[128];
    uint32_t rng = 39127;
    /* Exhaust all codes in isolation, then exercise sequential accumulation
     * across K, including fractional inputs, cancellation and special values. */
    for (int f = 0; f < 3; ++f) for (int pass = 0; pass < 276; ++pass) {
        for (size_t i = 0; i < sizeof(w); ++i) {
            rng = rng * 1664525u + 1013904223u;
            unsigned q = pass < 256 ? (unsigned)pass : rng >> 24;
            if (pass >= 256 && pass != 271 && f &&
                ((f == 1 && (q & 127) == 127) || (f == 2 && (q & 127) >= 124))) q = 56;
            w[i] = (uint8_t)q;
        }
        for (int k = 0; k < K; ++k) {
            rng = rng * 1664525u + 1013904223u;
            uint16_t bits = (uint16_t)((rng & 0x83ff) | ((pass % 16) << 10));
            if (pass < 256) bits = k == K - 1 ? 0x3c00 : 0;
            if (pass == 268) bits = (k & 1) ? 0x8001 : 1;
            if (pass == 269) bits = (k & 1) ? 0xbc00 : 0x3c00;
            if (pass == 270) bits = 0x7bff;
            if (pass == 272) bits = (k & 1) ? 0x8bff : 0x0bff; /* unsafe guard boundary */
            if (pass == 273) bits = (k & 1) ? 0x8c00 : 0x0c00; /* exact normal /4 */
            if (pass == 274) bits = (k & 1) ? 0xfc00 : 0x7c00;
            if (pass == 275) bits = (k & 1) ? 0x8000 : 0;
            memcpy(ah+k, &bits, 2);
            af[k] = (float)ah[k];
            if (pass == 267) af[k] = (k & 1) ? -0x1.234567p-120f : 0x1.234567p-120f;
            if (pass == 266) af[k] = (k & 1) ? -0x1.234567p100f : 0x1.234567p100f;
        }
        if (pass < 256) memset(w, 0, (K - 1) * 256);
        half_kernels[f](w, ah, yh);
        if (f == 2) {
            w8_e5m2_f16_native(w, ah, native);
            for (int n = 0; n < 256; ++n)
                if (!(isnan((float)yh[n]) && isnan((float)native[n])) && memcmp(yh+n, native+n, 2))
                    return fprintf(stderr, "E5M2 rescale/native mismatch pass=%d n=%d\n", pass, n), 1;
        }
        for (int n = 0; n < 256; ++n) {
            _Float16 ref = 0;
            for (int k = 0; k < K; ++k) ref = half_fma((_Float16)decode(w[k*256+n], f), ah[k], ref);
            if (!(isnan((float)ref) && isnan((float)yh[n])) && memcmp(&ref, yh+n, 2))
                return fprintf(stderr, "W8 FP16 mismatch format=%s pass=%d n=%d\n", formats[f], pass, n), 1;
        }
        if (pass < 256) {
            memset(w, 0, (K - 1) * 128);
            memset(w + (K - 1) * 128, pass, 128);
        }
        float_kernels[f](w, af, yf);
        for (int n = 0; n < 128; ++n) {
            float ref = 0;
            for (int k = 0; k < K; ++k) ref = fmaf(decode(w[k*128+n], f), af[k], ref);
            if (!(isnan(ref) && isnan(yf[n])) && memcmp(&ref, yf+n, 4))
                return fprintf(stderr, "W8 FP32 mismatch format=%s pass=%d n=%d got=%a ref=%a\n", formats[f], pass, n, yf[n], ref), 1;
        }
    }
    puts("W8 correctness: PASS (all 256 codes, sequential FP16/FP32 bit-exact finite results; NaN classification)");
    return 0;
}
typedef struct {
    uint8_t *weights;
    size_t bytes;
    int cpu, format, bits, iterations, read_only, error;
    atomic_int *ready, *start;
    uint64_t begin, end, checksum;
} worker;
static void *run(void *arg)
{
    worker *w = arg;
    _Float16 ah[K], yh[256]; float af[K], yf[128];
    for (int k = 0; k < K; ++k) af[k] = (float)(ah[k] = (_Float16)((k % 7 - 3) * 0.03125f));
    w->error = pin(w->cpu);
    atomic_fetch_add_explicit(w->ready, 1, memory_order_release);
    while (!atomic_load_explicit(w->start, memory_order_acquire)) __asm__ volatile("yield");
    w->begin = ticks();
    if (!w->error) for (int it = 0; it < w->iterations; ++it) {
        if (w->read_only) { hbm_read_256_sve(w->weights, w->bytes); continue; }
        size_t group = w->bits == 16 ? 32768 : 16384;
        for (size_t off = 0; off < w->bytes; off += group) {
            uint32_t value = 0;
            if (w->bits == 16) {
                half_kernels[w->format](w->weights+off, ah, yh);
                memcpy(&value, yh + (off / group % 256), 2);
            } else {
                float_kernels[w->format](w->weights+off, af, yf);
                memcpy(&value, yf + (off / group % 128), 4);
            }
            w->checksum += value;
        }
    }
    w->end = ticks(); return NULL;
}
int main(int argc, char **argv)
{
    int format = 0, bits = 16, cores = 12, core_base = 12, iterations = 10, trials = 5, verify = 0, normal_only = 0;
    size_t mib = 240;
    static const struct option opts[] = {
        {"normal-weights",0,0,'n'}, {"format",1,0,'f'}, {"bits",1,0,'b'}, {"cores",1,0,'c'}, {"core-base",1,0,'p'},
        {"iterations",1,0,'i'}, {"trials",1,0,'t'}, {"mib",1,0,'m'}, {"verify",0,0,'v'}, {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc,argv,"f:b:c:p:i:t:m:vn",opts,NULL)) != -1) {
        switch(c) {
        case 'f': for (format=0; format<3 && strcmp(optarg,formats[format]); ++format) {} break;
        case 'b': bits=atoi(optarg); break; case 'c': cores=atoi(optarg); break;
        case 'p': core_base=atoi(optarg); break; case 'i': iterations=atoi(optarg); break;
        case 't': trials=atoi(optarg); break; case 'm': mib=strtoull(optarg,NULL,0); break;
        case 'v': verify=1; break; case 'n': normal_only=1; break; default: return 2;
        }
    }
    if (format >= 3 || (bits != 16 && bits != 32) || cores < 1 || cores > MAX_CORES ||
        trials < 1 || trials > 32 || iterations < 1 || core_base < 0 || core_base + cores > CPU_SETSIZE ||
        !mib || mib > 1024) {
        fprintf(stderr,"usage: %s --format int8|e4m3|e5m2 --bits 16|32 [--verify] [--normal-weights] [--cores 1..12 --core-base 12 --mib 1..1024 --iterations 10 --trials 5]\n",argv[0]); return 2;
    }
    if (verify && verify_w8()) return 1;
    size_t group = bits == 16 ? 32768 : 16384, per_core = mib*1024*1024/cores/group*group;
    size_t bytes = per_core*cores;
    if (!bytes || pin(core_base)) return 1;
    uint8_t *w;
    if (posix_memalign((void **)&w, 2*1024*1024, bytes)) return 1;
    uint32_t rng = 113;
    for (size_t j = 0; j < bytes; ++j) {
        rng = rng*1664525u+1013904223u;
        unsigned q = rng >> 24;
        /* Time finite weights, including all subnormal codes. NaNs/infinities
         * are exhaustively tested above rather than poisoning every output. */
        if (format && ((format == 1 && (q&127)==127) || (format == 2 && (q&127)>=124))) q=56;
        if (normal_only && format && (q & 127) < (format == 1 ? 8u : 4u)) q = (q & 128) | 56;
        w[j]=(uint8_t)q;
    }
    printf("# weight_distribution=%s\n", normal_only ? "normal-only" : "all-finite");
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    printf("# fpcr=0x%lx\n", (unsigned long)fpcr);
    report_fused_mapping(w);
    printf("# weight_bytes=%zu group_bytes=%zu bits=%d accumulation=fp%d K=128 N=%d scales=omitted\n",bytes,group,bits,bits,bits==16?256:128);
    double medians[3];
    for (int phase=0; phase<3; ++phase) {
        double values[32];
        for (int trial=0; trial<=trials; ++trial) {
            pthread_t threads[MAX_CORES]; worker ws[MAX_CORES];
            atomic_int ready, start; atomic_init(&ready,0); atomic_init(&start,0);
            for (int j=0;j<cores;++j) {
                ws[j]=(worker){.weights=w+j*per_core,.bytes=per_core,.cpu=core_base+j,.format=format,
                    .bits=bits,.iterations=iterations,.read_only=phase!=1,.ready=&ready,.start=&start};
                int rc=pthread_create(threads+j,NULL,run,ws+j);
                if (rc) { fprintf(stderr,"pthread_create: %s\n",strerror(rc)); return 1; }
            }
            while (atomic_load_explicit(&ready,memory_order_acquire)!=cores) __asm__ volatile("yield");
            atomic_store_explicit(&start,1,memory_order_release);
            uint64_t first=UINT64_MAX,last=0,sum=0;
            for (int j=0;j<cores;++j) {
                pthread_join(threads[j],NULL);
                if (ws[j].error) { fprintf(stderr,"affinity: %s\n",strerror(ws[j].error)); return 1; }
                if (ws[j].begin<first) first=ws[j].begin;
                if (ws[j].end>last) last=ws[j].end;
                sum+=ws[j].checksum;
            }
            double bw=(double)bytes*iterations*frequency()/(last-first)/1e9;
            if (trial) values[trial-1]=bw;
            printf("format=%s bits=%d phase=%d trial=%d GB/s=%.2f checksum=%lu\n",formats[format],bits,phase,trial,bw,(unsigned long)sum);
        }
        for (int i=0;i<trials;++i) for(int j=i+1;j<trials;++j) if(values[j]<values[i]) { double t=values[i];values[i]=values[j];values[j]=t; }
        medians[phase]=(values[(trials-1)/2]+values[trials/2])/2;
        printf("summary format=%s bits=%d path=%s GB/s_median=%.2f best=%.2f\n",formats[format],bits,phase==1?"fma":"read",medians[phase],values[trials-1]);
    }
    printf("paired_read_before=%.2f after=%.2f qualified=%s target=%s\n",medians[0],medians[2],
           medians[0]>=220 && medians[2]>=220?"yes":"no",medians[1]>=220?"PASS":"FAIL");
    free(w); return 0;
}
