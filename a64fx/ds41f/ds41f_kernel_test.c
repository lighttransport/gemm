#include "ds41f_kernels.h"
#include "ds41f_model.h"
#include "ds41f_moe.h"
#include "ds41f_kv.h"
#include "ds41f_sve.h"
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *s) { fprintf(stderr, "FAIL: %s\n", s); exit(1); }
static void near(float a, float b, float tol, const char *s)
{ if (!isfinite(a) || !isfinite(b) || fabsf(a - b) > tol) { fprintf(stderr, "FAIL: %s %.8g %.8g\n", s, a, b); exit(1); } }

static void test_bf16_round(void)
{
    const uint32_t low[]={0,1,0x7fff,0x8000,0x8001,0xffff};
    float x[258];uint32_t expected[256];
    size_t checked=0;
    /* Every BF16 sign/exponent/mantissa, at both sides of the FP32 midpoint:
     * includes zeros, subnormals, overflow, infinities and NaN payloads. */
    for(size_t k=0;k<sizeof low/sizeof low[0];++k)for(uint32_t base=0;base<65536;base+=256){
        x[0]=12345;x[257]=-12345;
        for(uint32_t i=0;i<256;++i){uint32_t bits=((base+i)<<16)|low[k];
            memcpy(x+1+i,&bits,4);expected[i]=(uint32_t)ds41f_f32_to_bf16(x[1+i])<<16;}
        ds41f_round_bf16(x+1,256);
        for(size_t i=0;i<256;++i){uint32_t bits;memcpy(&bits,x+1+i,4);
            if(bits!=expected[i])fail("BF16 vector rounding vs scalar bits");}
        if(x[0]!=12345||x[257]!=-12345)fail("BF16 rounding canary");
        checked+=256;
    }
    uint32_t rng=1234567;
    for(size_t n=0;n<=256;++n){x[0]=12345;x[n+1]=-12345;
        for(size_t i=0;i<n;++i){rng=rng*1664525u+1013904223u;memcpy(x+1+i,&rng,4);
            expected[i]=(uint32_t)ds41f_f32_to_bf16(x[1+i])<<16;}
        ds41f_round_bf16(x+1,n);
        for(size_t i=0;i<n;++i){uint32_t bits;memcpy(&bits,x+1+i,4);
            if(bits!=expected[i])fail("BF16 rounding tail/random bits");}
        if(x[0]!=12345||x[n+1]!=-12345)fail("BF16 rounding tail canary");
        checked+=n;
    }
    printf("BF16_ROUND PASS bit_exact=%zu tails=0..256\n",checked);
}

static void test_dispatch(void)
{
    const int dims[] = {1, 7, 15, 16, 17, 31, 33, 127, 1280, 2304, 5120, 5121};
    for (size_t d = 0; d < sizeof dims / sizeof dims[0]; ++d) {
        int n = dims[d];
        uint16_t *w = malloc((size_t)7 * n * sizeof *w);
        uint16_t *x = malloc((size_t)n * sizeof *x);
        float *f = malloc((size_t)n * sizeof *f);
        float *out = malloc((size_t)(n + 1) * sizeof *out);
        if (!w || !x || !f || !out) fail("allocation");
        double ss = 0;
        for (int i = 0; i < n; ++i) {
            x[i] = ds41f_f32_to_bf16((float)(i % 23 - 11) / 16);
            f[i] = ds41f_bf16_to_f32(x[i]);
            ss += (double)f[i] * f[i];
        }
        for (int i = 0; i < 7 * n; ++i)
            w[i] = ds41f_f32_to_bf16((float)(i % 19 - 9) / 8);
        float y[8]; y[7] = 12345;
        ds41f_bf16_matvec(y, w, x, 7, n);
        for (int r = 0; r < 7; ++r) {
            double expected = 0;
            for (int i = 0; i < n; ++i)
                expected += (double)ds41f_bf16_to_f32(w[r*n+i]) * f[i];
            near(y[r], (float)expected, 1e-4f, "dispatch matvec vs FP64");
        }
        near(y[7], 12345, 0, "matvec canary");
        ds41f_bf16_f32_matvec(y,w,f,7,n);
        for(int r=0;r<7;++r){double expected=0;
            for(int i=0;i<n;++i)expected+=(double)ds41f_bf16_to_f32(w[r*n+i])*f[i];
            near(y[r],(float)expected,1e-4f,"BF16 weights FP32 input SVE");}
        near(y[7],12345,0,"BF16 FP32 matvec canary");
        out[n] = 12345;
        ds41f_rmsnorm_fast(out, f, w, n, 1e-6f);
        for (int i = 0; i < n; ++i) {
            float expected = (float)(f[i] * ds41f_bf16_to_f32(w[i]) /
                                     sqrt(ss / n + 1e-6));
            near(out[i], expected, 2e-5f, "dispatch RMSNorm vs FP64");
        }
        near(out[n], 12345, 0, "RMSNorm canary");
        free(w); free(x); free(f); free(out);
    }
#if defined(__ARM_FEATURE_SVE)
    printf("DISPATCH PASS backend=SVE vector_bits=%lu cases=12\n", (unsigned long)svcntb()*8);
#else
    puts("DISPATCH PASS backend=scalar cases=12");
#endif
}

int main(void)
{
    test_bf16_round();
    test_dispatch();
    for (int code=0;code<256;++code) {
        if ((code&127)==127) continue;
        if (ds41f_f32_to_fp8(ds41f_fp8_e4m3_to_f32((uint8_t)code))!=code)
            fail("FP8 exhaustive finite roundtrip");
    }
    for (int code=0;code<126;++code) {
        float a=ds41f_fp8_e4m3_to_f32((uint8_t)code);
        float b=ds41f_fp8_e4m3_to_f32((uint8_t)(code+1));
        if (ds41f_f32_to_fp8((a+b)*.5f)!=(code&1?code+1:code))
            fail("FP8 midpoint ties to even");
    }
    {
        float x[64]={0},q[64]; x[0]=448;x[1]=1.0625f;x[32]=896;
        if (ds41f_act_quant(q,x,64)) fail("activation quant");
        near(q[0],448,0,"activation scale1");
        near(q[1],1,0,"activation tie");
        near(q[32],896,0,"activation scale2");
        if (!ds41f_act_quant(q,x,63)) fail("activation shape");
    }
    {
        uint8_t packed[32], scales[2]={127,128};
        float x[32]={0}, y[2];
        for (int i=0;i<32;++i) packed[i]=0x71;
        x[0]=2; x[1]=3;
        if (ds41f_mxfp4_matvec_ref(y,packed,scales,x,2,32)) fail("MXFP4 call");
        near(y[0],19,0,"MXFP4 adjacent nibbles");
        near(y[1],38,0,"MXFP4 row scales");
        if (!ds41f_mxfp4_matvec_ref(y,packed,scales,x,2,31)) fail("MXFP4 shape rejection");
    }
    ds41f_model_config c; ds41f_model_config_init(&c);
    if (!ds41f_model_config_valid(&c)) fail("model config");
    if (ds41f_owned_expert_count(0, 12) != 32 || ds41f_owned_expert_count(11, 12) != 32)
        fail("expert ownership");
    near(ds41f_fp8_e4m3_to_f32(0x38), 1.0f, 0.0f, "fp8 one");
    near(ds41f_fp8_e4m3_to_f32(0x78), 256.0f, 0.0f, "fp8 exponent15");
    near(ds41f_fp8_e4m3_to_f32(0x7e), 448.0f, 0.0f, "fp8 max");
    near(ds41f_fp8_e4m3_to_f32(0x01), 1.0f/512, 0.0f, "fp8 subnormal");
    if (!isnan(ds41f_fp8_e4m3_to_f32(0x7f)) || !isnan(ds41f_e8m0_to_f32(255)))
        fail("quant NaN codes");
    {
        uint8_t weights[33*33], scales[4] = {127,128,129,130};
        float input[33], output[33];
        for (int i=0; i<33*33; ++i) weights[i]=0x38;
        for (int i=0; i<33; ++i) input[i]=1;
        ds41f_fp8_matvec_ref(output, weights, scales, input, 33, 33, 32);
        for (int i=0; i<33; ++i)
            near(output[i], i<32 ? 34 : 136, 0, "FP8 2D scale blocks");
        if (ds41f_fp8_matvec(output,weights,scales,input,33,33)) fail("FP8 dispatch");
        for (int i=0; i<33; ++i)
            near(output[i], i<32 ? 34 : 136, 0, "FP8 SVE scale blocks and tails");
    }
    {
        uint8_t w[256],s[4]={127,127,127,127};
        float x[128],ref[2],out[2];
        for (int i=0;i<256;++i) w[i]=(i&127)==127?0:(uint8_t)i;
        for (int i=0;i<128;++i) x[i]=(float)(i%7-3)/8;
        ds41f_fp8_matvec_ref(ref,w,s,x,2,128,32);
        if (ds41f_fp8_matvec(out,w,s,x,2,128)) fail("FP8 all codes dispatch");
        near(out[0],ref[0],1e-5,"FP8 all positive codes");
        near(out[1],ref[1],1e-5,"FP8 all negative codes");
    }
    near(ds41f_e8m0_to_f32(127), 1.0f, 0.0f, "e8m0 one");
    uint8_t fw[4] = { 0x38, 0x40, 0xb8, 0x00 }, fs[1] = { 127 };
    float x[4] = { 1, 2, 3, 4 }, y[1];
    ds41f_fp8_matvec_ref(y, fw, fs, x, 1, 4, 32);
    near(y[0], 1 + 4 - 3, 1e-6f, "fp8 matvec");
    int8_t iw[4] = { 1, 2, -1, 0 };
    ds41f_i8_matvec_ref(y, iw, fs, x, 1, 4, 32);
    near(y[0], 1 + 4 - 3, 1e-6f, "int8 matvec");
    uint16_t bw[2] = { ds41f_f32_to_bf16(2.0f), ds41f_f32_to_bf16(4.0f) };
    float z[2]; ds41f_rmsnorm_ref(z, x, bw, 2, 0.0f);
    near(z[0], 1.2649111f, 1e-5f, "rmsnorm");
    float gate[5] = { .2f, .9f, .9f, -.1f, .4f }; ds41f_route r[3];
    if (ds41f_route_topk(gate, 5, 3, r) || r[0].expert != 1 || r[1].expert != 2 || r[2].expert != 4)
        fail("topk tie order");
    ds41f_route own[3];
    if (ds41f_route_owner(r, 3, 1, 12, own) != 1 || own[0].expert != 1)
        fail("route owner");
    ds41f_kv_plan kp;
    if (ds41f_kv_plan_init(&kp,&c,1048576,128,512) ||
        kp.compressed_value_bytes != (size_t)2621440*256 ||
        kp.compressed_scale_bytes != (size_t)2621440*32 ||
        kp.index_value_bytes != (size_t)2621440*64 ||
        kp.index_scale_bytes != (size_t)2621440*4 ||
        kp.window_value_bytes != (size_t)40*128*528)
        fail("1M packed cache exact geometry");
    if (ds41f_kv_plan_init(&kp, &c, 1048576, 4096, 64) != 0 ||
        kp.total_bytes == 0 || kp.compressed_value_bytes == 0)
        fail("kv plan");
    puts("DS41F_KERNEL_TEST PASS");
    return 0;
}
