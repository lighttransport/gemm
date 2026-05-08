/* End-to-end test: launch the AOT-extracted Triton spconv hsaco for shape
 * (N=8452, Ci=Co=512), compare output bit-exact (or near-bit) with PyTorch.
 *
 * Build:
 *   gcc -O2 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -o test_launch \
 *       test_launch.c -L/opt/rocm/lib -lamdhip64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "triton_spconv_pp.h"

#define HIPCK(call) do { hipError_t e=(call); if (e!=hipSuccess) { \
    fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#call,hipGetErrorString(e)); exit(1);} } while(0)

/* same npy reader as test_pp.c */
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

static void *load_hsaco(const char *path, size_t *out_sz)
{
    FILE *f = fopen(path,"rb"); if (!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    void *b = malloc(sz); fread(b,1,sz,f); fclose(f);
    *out_sz = sz; return b;
}

int main(void)
{
    const char *DD = "/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/test_data";
    const char *KD = "/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/kernels/N8452_Ci512_Co512_SPLITK1";
    int nd, sh[4]; size_t es; char dt;
    char p[512];

    snprintf(p,sizeof p,"%s/spconv_input.npy",DD);
    uint16_t *input_h = read_npy(p,&nd,sh,&es,&dt);
    int N = sh[0], Ci = sh[1];
    printf("input: (%d, %d) %c%zu\n", N, Ci, dt, es);

    snprintf(p,sizeof p,"%s/spconv_weight.npy",DD);
    uint16_t *weight_h = read_npy(p,&nd,sh,&es,&dt);
    int Co = sh[0], V = sh[1];
    printf("weight: (%d, %d, %d) %c%zu\n", Co, V, sh[2], dt, es);

    snprintf(p,sizeof p,"%s/spconv_bias.npy",DD);
    uint16_t *bias_h = read_npy(p,&nd,sh,&es,&dt);

    snprintf(p,sizeof p,"%s/spconv_neighbor.npy",DD);
    uint32_t *nbr_h = read_npy(p,&nd,sh,&es,&dt);
    printf("neighbor: (%d, %d) %c%zu\n", sh[0], sh[1], dt, es);

    snprintf(p,sizeof p,"%s/spconv_output.npy",DD);
    uint16_t *out_ref_h = read_npy(p,&nd,sh,&es,&dt);

    /* Compute sorted_idx + vk + vkseg from neighbor in C. */
    uint32_t *gray = malloc(N*4), *binary = malloc(N*4);
    int64_t *sorted = malloc(N*8);
    t2_neigh_to_gray_binary(N, V, (int32_t*)nbr_h, gray, binary);
    t2_argsort_binary(N, binary, sorted);
    int32_t *vk=NULL, *vkseg=NULL; int vk_len=0;
    int B1 = 64;
    t2_build_valid_kernel(N, B1, gray, sorted, &vk, &vkseg, &vk_len);
    printf("derived: vk_len=%d num_blocks=%d\n", vk_len, (N+B1-1)/B1);

    /* HIP setup. */
    HIPCK(hipInit(0));
    hipDevice_t dev; HIPCK(hipDeviceGet(&dev,0));
    hipCtx_t ctx; HIPCK(hipCtxCreate(&ctx,0,dev));

    size_t hsaco_sz; void *hsaco = load_hsaco("/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/kernels/N8452_Ci512_Co512_SPLITK1/kernel.hsaco", &hsaco_sz);
    hipModule_t mod; HIPCK(hipModuleLoadData(&mod, hsaco));
    hipFunction_t kfn;
    HIPCK(hipModuleGetFunction(&kfn, mod, "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel"));

    /* Allocate device buffers. */
    void *d_input,*d_weight,*d_bias,*d_nbr,*d_sorted,*d_output,*d_vk,*d_vkseg;
    HIPCK(hipMalloc(&d_input,  (size_t)N*Ci*2));
    HIPCK(hipMalloc(&d_weight, (size_t)Co*V*Ci*2));
    HIPCK(hipMalloc(&d_bias,   (size_t)Co*2));
    HIPCK(hipMalloc(&d_nbr,    (size_t)N*V*4));
    HIPCK(hipMalloc(&d_sorted, (size_t)N*8));
    HIPCK(hipMalloc(&d_output, (size_t)N*Co*2));
    HIPCK(hipMalloc(&d_vk,     (size_t)vk_len*4));
    int num_blocks = (N+B1-1)/B1;
    HIPCK(hipMalloc(&d_vkseg,  (size_t)(num_blocks+1)*4));
    HIPCK(hipMemcpy(d_input,  input_h,  (size_t)N*Ci*2, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_weight, weight_h, (size_t)Co*V*Ci*2, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_bias,   bias_h,   (size_t)Co*2, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_nbr,    nbr_h,    (size_t)N*V*4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_sorted, sorted,   (size_t)N*8, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_vk,     vk,       (size_t)vk_len*4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_vkseg,  vkseg,    (size_t)(num_blocks+1)*4, hipMemcpyHostToDevice));
    HIPCK(hipMemset(d_output, 0, (size_t)N*Co*2));

    /* Launch. */
    int LOGN = (int)log2((double)N);  /* int(log2(8452)) = 13 */
    void *null_ptr = NULL;
    void *args[] = {
        &d_input, &d_weight, &d_bias, &d_nbr, &d_sorted, &d_output,
        &N, &LOGN, &Ci, &Co,
        &d_vk, &d_vkseg,
        &null_ptr, &null_ptr,
    };
    int B2 = 128;
    int grid_x = ((Co+B2-1)/B2) * ((N+B1-1)/B1);  /* 4 * 133 = 532 */
    int block_x = 4 * 32;                          /* num_warps=4, warpSize=32 */
    int shared = 12288;
    printf("launch: grid=(%d,1,1) block=(%d,1,1) shared=%d LOGN=%d\n",
           grid_x, block_x, shared, LOGN);
    HIPCK(hipModuleLaunchKernel(kfn, grid_x,1,1, block_x,1,1, shared, 0, args, NULL));
    HIPCK(hipDeviceSynchronize());

    /* Copy back + compare. */
    uint16_t *out_h = malloc((size_t)N*Co*2);
    HIPCK(hipMemcpy(out_h, d_output, (size_t)N*Co*2, hipMemcpyDeviceToHost));

    /* Convert fp16 to fp32 for comparison. */
    int total = N*Co, mismatch_strict=0;
    double sum_abs_diff=0, max_abs_diff=0, sum_abs_ref=0;
    for (int i = 0; i < total; i++) {
        if (out_h[i] != out_ref_h[i]) mismatch_strict++;
        /* fp16 -> fp32 */
        uint16_t a16=out_h[i], b16=out_ref_h[i];
        uint32_t af = ((uint32_t)(a16 & 0x8000) << 16) | (((uint32_t)(a16 & 0x7c00)+0x1c000) << 13) | (((uint32_t)a16 & 0x03ff) << 13);
        uint32_t bf = ((uint32_t)(b16 & 0x8000) << 16) | (((uint32_t)(b16 & 0x7c00)+0x1c000) << 13) | (((uint32_t)b16 & 0x03ff) << 13);
        float fa,fb; memcpy(&fa,&af,4); memcpy(&fb,&bf,4);
        if ((a16 & 0x7c00)==0) fa = (a16 & 0x8000)? -0.f:0.f;
        if ((b16 & 0x7c00)==0) fb = (b16 & 0x8000)? -0.f:0.f;
        double d = fabs((double)fa - (double)fb);
        sum_abs_diff += d; sum_abs_ref += fabs((double)fb);
        if (d > max_abs_diff) max_abs_diff = d;
    }
    printf("strict mismatches: %d / %d (%.3f%%)\n", mismatch_strict, total, 100.*mismatch_strict/total);
    printf("max_abs_diff=%g  mean_abs_diff=%g  rel_l1=%g\n",
           max_abs_diff, sum_abs_diff/total, sum_abs_diff/sum_abs_ref);
    return 0;
}
