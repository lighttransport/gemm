/* Microbenchmark: time the Triton-AOT spconv bridge on the captured shape
 * (N=8452, Ci=Co=512). Compare hot-path ms against the in-tree WMMA x8_db
 * baseline numbers from the perf memo to gauge the win.
 *
 * Build:
 *   gcc -O2 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ \
 *       -o bench_bridge bench_bridge.c \
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

    HIPCK(hipInit(0));
    hipDevice_t dev; HIPCK(hipDeviceGet(&dev,0));
    hipCtx_t ctx; HIPCK(hipCtxCreate(&ctx,0,dev));
    if (t2_triton_init(KD) != 0) return 1;

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

    /* Warmup (also seeds derived-buffer cache). */
    for (int i = 0; i < 5; i++)
        t2_triton_spconv(N, Ci, Co, d_input, d_weight, d_bias, d_nbr, d_output, 0);
    HIPCK(hipDeviceSynchronize());

    /* Timed loop: 200 iters. */
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    const int ITERS = 200;
    hipEventRecord(e0, 0);
    for (int i = 0; i < ITERS; i++)
        t2_triton_spconv(N, Ci, Co, d_input, d_weight, d_bias, d_nbr, d_output, 0);
    hipEventRecord(e1, 0);
    HIPCK(hipDeviceSynchronize());
    float total_ms = 0; hipEventElapsedTime(&total_ms, e0, e1);
    double per_ms = total_ms / ITERS;
    /* spconv FLOPs = M * outC * 2 * V * inC = N * Co * 2 * 27 * Ci. */
    double flops = (double)N * Co * 2.0 * 27.0 * Ci;
    double gflops = flops / (per_ms * 1e6);
    printf("Triton spconv N=%d Ci=%d Co=%d : %.3f ms/call, %.1f GFLOP/s (%d iters)\n",
           N, Ci, Co, per_ms, gflops, ITERS);

    t2_triton_release();
    return 0;
}
