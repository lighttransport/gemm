/* Benchmark every registered Triton spconv shape with synthetic nmap (random
 * valid neighbors). Output table: shape, ms, GFLOP/s, % of 95 TF/s peak. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <hip/hip_runtime.h>

#define TRITON_SPCONV_BRIDGE_IMPL
#include "triton_spconv_bridge.h"

#define HIPCK(c) do { hipError_t e=(c); if (e!=hipSuccess) { fprintf(stderr,"%s\n",hipGetErrorString(e)); exit(1);} } while(0)

extern t2_tspconv_shape g_shapes[];
extern int g_n_shapes;

int main(void)
{
    HIPCK(hipInit(0));
    hipDevice_t dev; HIPCK(hipDeviceGet(&dev,0));
    hipCtx_t ctx; HIPCK(hipCtxCreate(&ctx,0,dev));
    if (t2_triton_init("/mnt/disk1/work/gemm/trellis2/rdna4/trellis2/triton_aot/kernels") != 0)
        return 1;

    printf("\n%-6s %-6s %-6s %12s %12s %8s\n", "N","Ci","Co","ms/call","GFLOP/s","%peak");
    printf("---------------------------------------------------------------\n");

    for (int s = 0; s < g_n_shapes; s++) {
        int N = g_shapes[s].N, Ci = g_shapes[s].Ci, Co = g_shapes[s].Co;
        size_t in_b = (size_t)N*Ci*2, w_b = (size_t)Co*27*Ci*2, b_b = (size_t)Co*2,
               n_b = (size_t)N*27*4, o_b = (size_t)N*Co*2;
        void *di,*dw,*db,*dn,*doo;
        HIPCK(hipMalloc(&di, in_b)); HIPCK(hipMemset(di, 0, in_b));
        HIPCK(hipMalloc(&dw, w_b )); HIPCK(hipMemset(dw, 0, w_b ));
        HIPCK(hipMalloc(&db, b_b )); HIPCK(hipMemset(db, 0, b_b ));
        HIPCK(hipMalloc(&dn, n_b ));
        /* Synthetic nmap: -1 (empty) to keep it simple — kernel still runs full tile loops. */
        int32_t *h = malloc(n_b);
        for (size_t i = 0; i < n_b/4; i++) h[i] = -1;
        HIPCK(hipMemcpy(dn, h, n_b, hipMemcpyHostToDevice)); free(h);
        HIPCK(hipMalloc(&doo, o_b));

        for (int w = 0; w < 5; w++) t2_triton_spconv(N, Ci, Co, di, dw, db, dn, doo, 0);
        HIPCK(hipDeviceSynchronize());

        hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
        int IT = (N > 100000) ? 50 : 200;
        hipEventRecord(e0,0);
        for (int i = 0; i < IT; i++) t2_triton_spconv(N, Ci, Co, di, dw, db, dn, doo, 0);
        hipEventRecord(e1,0); HIPCK(hipDeviceSynchronize());
        float ms_tot=0; hipEventElapsedTime(&ms_tot, e0, e1);
        double per_ms = ms_tot / IT;
        double flops = (double)N * Co * 2.0 * 27.0 * Ci;
        double gf = flops / (per_ms * 1e6);
        printf("%-6d %-6d %-6d %12.3f %12.1f %7.1f%%\n", N, Ci, Co, per_ms, gf, 100.0*gf/95000.0);

        hipFree(di); hipFree(dw); hipFree(db); hipFree(dn); hipFree(doo);
    }
    t2_triton_release();
    return 0;
}
