/* Smoke test for triton_spconv_bridge.h: replicates test_launch.c but goes
 * through the packaged API. Confirms bit-exact match vs PyTorch for the
 * (N=8452, Ci=Co=512) shape.
 *
 * Build:
 *   gcc -O2 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ \
 *       -o test_bridge test_bridge.c \
 *       -L/opt/rocm/lib -lamdhip64 -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <hip/hip_runtime.h>

#define TRITON_SPCONV_BRIDGE_IMPL
#include "triton_spconv_bridge.h"

#define HIPCK(call) do { hipError_t e=(call); if (e!=hipSuccess) { \
    fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#call,hipGetErrorString(e)); exit(1);} } while(0)

static void *read_npy(const char *path, int *ndim, int *shape, size_t *elt_sz, char *dt)
{
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return NULL; }
    char magic[6]; fread(magic,1,6,f);
    unsigned char vmaj,vmin; fread(&vmaj,1,1,f); fread(&vmin,1,1,f);
    uint32_t hl; if (vmaj==1) { uint16_t h; fread(&h,2,1,f); hl=h; } else fread(&hl,4,1,f);
    char *hdr = malloc(hl+1); fread(hdr,1,hl,f); hdr[hl]=0;
    char *p = strstr(hdr, "'descr':"); p += strlen("'descr':");
    while (*p && *p != '\'') p++; p++;
    if (*p=='<'||*p=='>'||*p=='|'||*p=='=') p++;
    *dt = *p++; *elt_sz = (size_t)(*p - '0');
    p = strstr(hdr, "'shape':"); p = strchr(p,'(')+1;
    *ndim = 0;
    while (*p && *p != ')') {
        while (*p==' '||*p==',') p++;
        if (!isdigit((unsigned char)*p)) break;
        shape[(*ndim)++] = atoi(p);
        while (isdigit((unsigned char)*p)) p++;
    }
    free(hdr);
    size_t total = *elt_sz; for (int i=0;i<*ndim;i++) total *= shape[i];
    void *buf = malloc(total); fread(buf,1,total,f); fclose(f);
    return buf;
}

int main(void)
{
    const char *DD = "/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/test_data";
    const char *KD = "/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/kernels";
    int nd, sh[4]; size_t es; char dt;
    char p[512];

    /* HIP context. */
    HIPCK(hipInit(0));
    hipDevice_t dev; HIPCK(hipDeviceGet(&dev,0));
    hipCtx_t ctx; HIPCK(hipCtxCreate(&ctx,0,dev));

    /* Init bridge — registers all shapes. */
    if (t2_triton_init(KD) != 0) { fprintf(stderr,"bridge init failed\n"); return 1; }

    /* Load PyTorch-captured tensors. */
    snprintf(p,sizeof p,"%s/spconv_input.npy",DD);
    uint16_t *input_h = read_npy(p,&nd,sh,&es,&dt);
    int N = sh[0], Ci = sh[1];
    snprintf(p,sizeof p,"%s/spconv_weight.npy",DD);
    uint16_t *weight_h = read_npy(p,&nd,sh,&es,&dt);
    int Co = sh[0], V = sh[1];
    snprintf(p,sizeof p,"%s/spconv_bias.npy",DD);
    uint16_t *bias_h = read_npy(p,&nd,sh,&es,&dt);
    snprintf(p,sizeof p,"%s/spconv_neighbor.npy",DD);
    uint32_t *nbr_h = read_npy(p,&nd,sh,&es,&dt);
    snprintf(p,sizeof p,"%s/spconv_output.npy",DD);
    uint16_t *out_ref_h = read_npy(p,&nd,sh,&es,&dt);
    printf("loaded N=%d Ci=%d Co=%d V=%d\n", N, Ci, Co, V);

    /* Upload. */
    void *d_input,*d_weight,*d_bias,*d_nbr,*d_output;
    HIPCK(hipMalloc(&d_input,  (size_t)N*Ci*2));
    HIPCK(hipMalloc(&d_weight, (size_t)Co*V*Ci*2));
    HIPCK(hipMalloc(&d_bias,   (size_t)Co*2));
    HIPCK(hipMalloc(&d_nbr,    (size_t)N*V*4));
    HIPCK(hipMalloc(&d_output, (size_t)N*Co*2));
    HIPCK(hipMemcpy(d_input,  input_h,  (size_t)N*Ci*2, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_weight, weight_h, (size_t)Co*V*Ci*2, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_bias,   bias_h,   (size_t)Co*2, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_nbr,    nbr_h,    (size_t)N*V*4, hipMemcpyHostToDevice));
    HIPCK(hipMemset(d_output, 0, (size_t)N*Co*2));

    /* Dispatch through bridge API. */
    int rc = t2_triton_spconv(N, Ci, Co, d_input, d_weight, d_bias, d_nbr, d_output, 0);
    if (rc != 0) { fprintf(stderr,"t2_triton_spconv returned %d\n", rc); return 1; }
    HIPCK(hipDeviceSynchronize());

    /* Compare. */
    uint16_t *out_h = malloc((size_t)N*Co*2);
    HIPCK(hipMemcpy(out_h, d_output, (size_t)N*Co*2, hipMemcpyDeviceToHost));
    int total = N*Co, mm_strict = 0;
    double max_d = 0, sum_d = 0, sum_r = 0;
    for (int i = 0; i < total; i++) {
        if (out_h[i] != out_ref_h[i]) mm_strict++;
        uint16_t a16=out_h[i], b16=out_ref_h[i];
        uint32_t af = ((uint32_t)(a16 & 0x8000) << 16) | (((uint32_t)(a16 & 0x7c00)+0x1c000) << 13) | (((uint32_t)a16 & 0x03ff) << 13);
        uint32_t bf = ((uint32_t)(b16 & 0x8000) << 16) | (((uint32_t)(b16 & 0x7c00)+0x1c000) << 13) | (((uint32_t)b16 & 0x03ff) << 13);
        float fa,fb; memcpy(&fa,&af,4); memcpy(&fb,&bf,4);
        if ((a16 & 0x7c00)==0) fa = (a16 & 0x8000)? -0.f:0.f;
        if ((b16 & 0x7c00)==0) fb = (b16 & 0x8000)? -0.f:0.f;
        double d = fabs((double)fa - (double)fb);
        sum_d += d; sum_r += fabs((double)fb);
        if (d > max_d) max_d = d;
    }
    printf("strict mismatches: %d / %d (%.3f%%)\n", mm_strict, total, 100.*mm_strict/total);
    printf("max_abs_diff=%g  rel_l1=%g\n", max_d, sum_d/sum_r);

    t2_triton_release();
    return mm_strict == 0 ? 0 : 1;
}
