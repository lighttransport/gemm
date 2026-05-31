/* Decisive measurement for GPU MHR pose_correctives: the 166M-FMA hot path is
 * out[55317] = LW[55317,3000] @ h[3000] + Lb, i.e. gemm_f32_bias with N=1,
 * D_in=3000, D_out=55317 (LW is row-major [OUT,HID] = W[D_out,D_in]).
 *
 * This bench:
 *  (a) correctness: GPU gemm_f32_bias vs a CPU reference matvec (synthetic LW/h),
 *  (b) perf: GPU matvec time with LW RESIDENT (upload once) — the realistic
 *      per-call cost — vs the CPU OpenMP matvec, to decide if GPU MHR is worth
 *      wiring. (663MB LW read @ HBM BW is the floor.)
 */
#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "../hip_kernels_common.h"
#include "hip_sam3d_body_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef HIP_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)(uintptr_t)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void *)(uintptr_t)0x02)
#define HIP_LAUNCH_PARAM_END            ((void *)(uintptr_t)0x03)
#endif

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0 + t.tv_nsec*1e-6; }

int main(void){
    const int HID = 3000, V = 18439, OUT = V*3; /* 55317 */
    if (rocewInit(ROCEW_INIT_HIP|ROCEW_INIT_HIPRTC)!=ROCEW_SUCCESS){fprintf(stderr,"init fail\n");return 2;}
    size_t freeb=0,totb=0; hipMemGetInfo(&freeb,&totb);
    printf("VRAM free=%.0f MiB; LW f32=%.0f MiB\n", freeb/1048576.0,
           (double)OUT*HID*4/1048576.0);

    size_t la=strlen(hip_kernels_common_src), lb=strlen(hip_sam3d_body_kernels_src);
    char *src=malloc(la+lb+1); memcpy(src,hip_kernels_common_src,la); memcpy(src+la,hip_sam3d_body_kernels_src,lb+1);
    hipModule_t mod; int rc=hip_compile_kernels(&mod,0,src,"pc_bench",1,"pc"); free(src);
    if(rc<0){fprintf(stderr,"compile fail\n");return 3;}
    hipFunction_t fn; if(hipModuleGetFunction(&fn,mod,"gemm_f32_bias")!=hipSuccess){fprintf(stderr,"no gemm_f32_bias\n");return 4;}

    float *LW=malloc((size_t)OUT*HID*sizeof(float));
    float *Lb=malloc((size_t)OUT*sizeof(float));
    float *h =malloc((size_t)HID*sizeof(float));
    float *ref=malloc((size_t)OUT*sizeof(float));
    float *gpu=malloc((size_t)OUT*sizeof(float));
    unsigned s=99u;
    for(size_t i=0;i<(size_t)OUT*HID;i++){ s=s*1103515245u+12345u; LW[i]=((float)((s>>16)&0x7fff)/16384.f-1.f)*0.05f; }
    for(int i=0;i<OUT;i++){ s=s*1103515245u+12345u; Lb[i]=((float)((s>>16)&0x7fff)/16384.f-1.f)*0.01f; }
    for(int i=0;i<HID;i++){ s=s*1103515245u+12345u; h[i]=fmaxf(0.f,(float)((s>>16)&0x7fff)/16384.f-1.f); } /* ReLU'd */

    /* CPU reference matvec (threaded, matches the production path). */
    int nthr = 1;
#if defined(_OPENMP)
    nthr = omp_get_max_threads();
#endif
    double tc0=now_ms();
    int reps=10;
    for(int r=0;r<reps;r++){
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(nthr)
#endif
        for(int row=0;row<OUT;row++){ const float*W=LW+(size_t)row*HID; float a=Lb[row];
            for(int c=0;c<HID;c++) a+=W[c]*h[c]; ref[row]=a; }
    }
    double cpu_ms=(now_ms()-tc0)/reps;

    /* GPU: LW + Lb resident (upload once); per-call = upload h + launch + download out. */
    void *dW,*dLb,*dh,*dY;
    hipMalloc(&dW,(size_t)OUT*HID*sizeof(float)); hipMalloc(&dLb,(size_t)OUT*sizeof(float));
    hipMalloc(&dh,(size_t)HID*sizeof(float));     hipMalloc(&dY,(size_t)OUT*sizeof(float));
    hipMemcpy(dW,LW,(size_t)OUT*HID*sizeof(float),hipMemcpyHostToDevice);
    hipMemcpy(dLb,Lb,(size_t)OUT*sizeof(float),hipMemcpyHostToDevice);

    int N=1, Din=HID, Dout=OUT;
    struct __attribute__((packed)){ void*Y; const void*X,*W,*b; int N,Din,Dout; }
        p={ dY, dh, dW, dLb, N, Din, Dout };
    unsigned gx=(N+15)/16, gy=(Dout+15)/16;
    void *cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&p,HIP_LAUNCH_PARAM_BUFFER_SIZE,(void*)0,HIP_LAUNCH_PARAM_END};
    size_t pb=sizeof(p); cfg[3]=&pb;

    /* warm + correctness */
    hipMemcpy(dh,h,(size_t)HID*sizeof(float),hipMemcpyHostToDevice);
    hipModuleLaunchKernel(fn,gx,gy,1,16,16,1,0,0,NULL,cfg); hipDeviceSynchronize();
    hipMemcpy(gpu,dY,(size_t)OUT*sizeof(float),hipMemcpyDeviceToHost);
    double maxd=0,dot=0,na=0,nb=0;
    for(int i=0;i<OUT;i++){ double a=ref[i],b=gpu[i]; double d=fabs(a-b); if(d>maxd)maxd=d; dot+=a*b;na+=a*a;nb+=b*b; }
    printf("correctness: cosine=%.8f max_abs=%.4e  %s\n", dot/(sqrt(na*nb)+1e-30), maxd,
           (dot/(sqrt(na*nb)+1e-30)>0.9999)?"PASS":"FAIL");

    /* perf: per-call = h upload + launch + out download (LW already resident). */
    double tg0=now_ms();
    for(int r=0;r<reps;r++){
        hipMemcpy(dh,h,(size_t)HID*sizeof(float),hipMemcpyHostToDevice);
        hipModuleLaunchKernel(fn,gx,gy,1,16,16,1,0,0,NULL,cfg);
        hipMemcpy(gpu,dY,(size_t)OUT*sizeof(float),hipMemcpyDeviceToHost);
    }
    hipDeviceSynchronize();
    double gpu_ms=(now_ms()-tg0)/reps;

    /* launch-only (no copies) to isolate kernel time */
    hipDeviceSynchronize(); double tk0=now_ms();
    for(int r=0;r<reps;r++) hipModuleLaunchKernel(fn,gx,gy,1,16,16,1,0,0,NULL,cfg);
    hipDeviceSynchronize(); double kern_ms=(now_ms()-tk0)/reps;

    printf("matvec 55317x3000 (avg of %d):\n", reps);
    printf("  CPU (%d thr)     : %7.3f ms\n", nthr, cpu_ms);
    printf("  GPU per-call     : %7.3f ms  (h up + launch + out down, LW resident)\n", gpu_ms);
    printf("  GPU kernel-only  : %7.3f ms\n", kern_ms);
    printf("  speedup (per-call vs CPU): %.2fx\n", cpu_ms/gpu_ms);
    return 0;
}
