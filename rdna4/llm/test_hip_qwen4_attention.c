/* Exercise the production attention kernels with deliberately delayed waves.
 * --unsafe removes only the max-reader barrier: the same oracle must fail. */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "hip_llm_runner.c"
#include <assert.h>
#define REQUIRE(call) do { if ((call) != 0) { fprintf(stderr, "attention test line %d: %s\n", __LINE__, #call); exit(1); } } while (0)

static char *replace_all(const char *src, const char *from, const char *to) {
    size_t count = 0, fl = strlen(from), tl = strlen(to);
    for (const char *p = src; (p = strstr(p, from)); p += fl) ++count;
    char *out = malloc(strlen(src) + count * tl + 1), *d = out;
    assert(out && count);
    const char *p = src, *next;
    while ((next = strstr(p, from))) {
        size_t n = (size_t)(next - p); memcpy(d, p, n); d += n;
        memcpy(d, to, tl); d += tl; p = next + fl;
    }
    strcpy(d, p); return out;
}

int main(int argc, char **argv) {
    char *delayed = replace_all(hip_kernel_source, "float nm=fmaxf(mi,red[0])",
        "if(tid>=32){for(volatile int spin=0;spin<512;++spin){}} float nm=fmaxf(mi,red[0])");
    char *unsafe = NULL;
    if (argc > 1 && !strcmp(argv[1], "--unsafe"))
        unsafe = replace_all(delayed, "/* QWEN4_ATTN_MAX_READERS */ __syncthreads();", "");
    hip_kernel_source = unsafe ? unsafe : delayed;
    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) return 77;
    unsetenv("LLM_ATTN_GQA8"); unsetenv("LLM_ATTN_PREFILL_I8_WARP");
    unsetenv("LLM_ATTN_PREFILL_I8_GQA4");
    enum { M = 259, H = 24, HK = 2, D = 256, KD = HK*D, QD = H*D };
    uint16_t *kv = malloc((size_t)M*KD*2);
    float *out = malloc((size_t)M*QD*4), *ref = malloc((size_t)M*KD*4);
    assert(kv && out && ref);
    int rc=0;
    REQUIRE(hipMalloc(&r->d_position,4));
    float *scales=malloc((size_t)M*HK*(D/32)*4); assert(scales);
    for (int i=0;i<M*HK*(D/32);++i) scales[i]=0.015625f;
    void *ds=NULL; REQUIRE(hipMalloc(&ds,(size_t)M*HK*(D/32)*4));
    REQUIRE(hipMemcpy(ds,scales,(size_t)M*HK*(D/32)*4,hipMemcpyHostToDevice));
    setenv("LLM_QWEN4_DECODE_ATTN_SHARDS", "1", 1);
    for (int format=0;format<2;++format) {
        for (int m = 0; m < M; ++m) for (int j = 0; j < KD; ++j) {
            uint16_t v = (uint16_t)(0x3000 + ((m*29+j*11) % 2048));
            if ((m+j)%3 == 0) v |= 0x8000;
            if (!format) kv[(size_t)m*KD+j] = v;
            else ((signed char *)kv)[(size_t)m*KD+j]=(signed char)((m*29+j*11)%127-63);
            double sum = 0;
            for (int t = 0; t <= m; ++t) sum += format ?
                (float)((signed char *)kv)[(size_t)t*KD+j]*0.015625f : ggml_fp16_to_fp32(kv[(size_t)t*KD+j]);
            ref[(size_t)m*KD+j] = (float)(sum/(m+1));
        }
        void *dq=NULL, *dkv=NULL, *dout=NULL;
        REQUIRE(hipMalloc(&dq,(size_t)M*QD*4)); REQUIRE(hipMalloc(&dkv,(size_t)M*KD*2));
        REQUIRE(hipMalloc(&dout,(size_t)M*QD*4));
        REQUIRE(hipMemset(dq,0,(size_t)M*QD*4));
        REQUIRE(hipMemcpy(dkv,kv,(size_t)M*KD*2,hipMemcpyHostToDevice));
        for (int decode=0; decode<2; ++decode) {
            if (!decode) {
                if (!format) launch_attn_prefill_flash_f16(r,dout,dq,dkv,dkv,H,HK,D,KD,M,0,0.0625f);
                else launch_attn_prefill_i8(r,dout,dq,dkv,dkv,ds,ds,H,HK,D,KD,M,0,0.0625f,0);
            }
            else {
                int pos=M-1; REQUIRE(hipMemcpy(r->d_position,&pos,4,hipMemcpyHostToDevice));
                if (!format) launch_attn_decode_flash_f16(r,dout,dq,dkv,dkv,H,HK,D,KD,0.0625f);
                else launch_attn_decode_i8(r,dout,dq,dkv,dkv,ds,ds,H,HK,D,KD,0.0625f,0);
            }
            REQUIRE(hipStreamSynchronize(r->stream));
            REQUIRE(hipMemcpy(out,dout,(size_t)(decode?1:M)*QD*4,hipMemcpyDeviceToHost));
            double worst=0;
            for (int m=0; m<(decode?1:M); ++m) for(int h=0;h<H;++h) for(int d=0;d<D;++d) {
                float expected=ref[(size_t)(decode?M-1:m)*KD+(h/(H/HK))*D+d];
                float actual=out[(size_t)m*QD+h*D+d];
                double err=fabs((double)actual-expected);
                if(!isfinite(actual) || err>2e-6) rc=1;
                if(err>worst)worst=err;
            }
            printf("Qwen4 %s %s delayed-wave oracle: max_abs=%.9g %s\n",format?"I8":"F16",decode?"decode":"prefill",worst,worst<=2e-6?"PASS":"FAIL");
        }
        if (!format) {
            float *query = malloc((size_t)QD*4), *baseline = malloc((size_t)QD*4);
            assert(query && baseline);
            for (int i=0;i<QD;++i) query[i]=sinf(i*0.013f)+0.125f*cosf(i*0.071f);
            REQUIRE(hipMemcpy(dq,query,(size_t)QD*4,hipMemcpyHostToDevice));
            r->is_qwen4exp=1;
            int positions[]={31,255,M-1};
            const char *shards[]={"1","2","4","8"};
            for(int pi=0;pi<3;++pi) {
                REQUIRE(hipMemcpy(r->d_position,&positions[pi],4,hipMemcpyHostToDevice));
                for(int si=0;si<4;++si) {
                    setenv("LLM_QWEN4_DECODE_ATTN_SHARDS",shards[si],1);
                    launch_attn_decode_flash_f16(r,dout,dq,dkv,dkv,H,HK,D,KD,0.0625f);
                    REQUIRE(hipStreamSynchronize(r->stream));
                    REQUIRE(hipMemcpy(out,dout,(size_t)QD*4,hipMemcpyDeviceToHost));
                    if (!si) memcpy(baseline,out,(size_t)QD*4);
                    else if (memcmp(baseline,out,(size_t)QD*4)) {
                        fprintf(stderr,"attention shard mismatch pos=%d shards=%s\n",positions[pi],shards[si]);
                        rc=1;
                    }
                }
            }
            setenv("LLM_QWEN4_DECODE_ATTN_SHARDS","1",1);
            free(query); free(baseline);
        }
        REQUIRE(hipFree(dq)); REQUIRE(hipFree(dkv)); REQUIRE(hipFree(dout));
    }
    REQUIRE(hipFree(ds)); free(scales);
    hip_llm_free(r); free(delayed); free(unsafe); free(kv); free(out); free(ref);
    return rc;
}
