/* Opt-in BF16-WMMA GEMM dispatch for hip_sam3d.
 *
 * The scalar `gemm_f32_bias` kernel (one thread per output element) is the
 * default. When SAM3D_GEMM_WMMA is set (and a gfx12 `gemm_f32_bias_wmma`
 * handle is available and D_in is a multiple of 16), launches route to the
 * RDNA4 v_wmma_f32_16x16x16_bf16 kernel instead. The WMMA path is BF16-precision
 * (~1e-2 rel error) so it is OFF by default — existing fp32 verifiers exercise
 * the scalar path unchanged. Both kernels share the identical argument list
 * (Y, X, W, b, N, D_in, D_out), so callers reuse their existing args[] array.
 */
#ifndef HIP_SAM3D_WMMA_H_
#define HIP_SAM3D_WMMA_H_

#include "../rocew.h"

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if the WMMA GEMM path is requested via SAM3D_GEMM_WMMA, else 0.
 * Cached after first read (unset / "" / "0" → off). */
static inline int cs3d_use_gemm_wmma(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("SAM3D_GEMM_WMMA");
        cached = (e && *e && e[0] != '0') ? 1 : 0;
    }
    return cached;
}

/* Launch Y(N,D_out) = X(N,D_in) @ W^T + b. Picks the WMMA kernel when enabled
 * (and applicable), otherwise the scalar kernel; `args` is the shared
 * {&Y,&X,&W,&b,&N,&D_in,&D_out} array. `gemm_wmma` may be 0 (falls back).
 * Returns the hipModuleLaunchKernel status. */
static inline hipError_t cs3d_launch_gemm(hipFunction_t gemm_scalar,
                                          hipFunction_t gemm_wmma,
                                          void **args,
                                          int N, int D_in, int D_out,
                                          hipStream_t stream)
{
    { static int once = 0; if (!once && getenv("SAM3D_GEMM_WMMA_DEBUG")) { once = 1;
        fprintf(stderr, "[cs3d_launch_gemm] use_wmma=%d gemm_wmma=%p D_in=%d -> %s\n",
                cs3d_use_gemm_wmma(), (void*)gemm_wmma, D_in,
                (cs3d_use_gemm_wmma() && gemm_wmma && (D_in%16==0)) ? "WMMA" : "scalar"); } }
    if (cs3d_use_gemm_wmma() && gemm_wmma && (D_in % 16 == 0)) {
        unsigned gx = (unsigned)((D_out + 127) / 128);
        unsigned gy = (unsigned)((N + 127) / 128);
        return hipModuleLaunchKernel(gemm_wmma, gx, gy, 1, 256, 1, 1,
                                     0, stream, args, NULL);
    }
    unsigned gx = (unsigned)((N + 15) / 16);
    unsigned gy = (unsigned)((D_out + 15) / 16);
    return hipModuleLaunchKernel(gemm_scalar, gx, gy, 1, 16, 16, 1,
                                 0, stream, args, NULL);
}

/* Resolve (and cache) the gemm_f32_bias_wmma handle from a module. Lets call
 * sites that only have a hipModule_t (no pre-fetched fns struct) obtain the
 * WMMA handle cheaply. Returns NULL if absent (→ scalar fallback). */
static inline hipFunction_t cs3d_gemm_wmma(hipModule_t mod)
{
    static hipModule_t cached_mod = NULL;
    static hipFunction_t cached_fn = NULL;
    if (mod != cached_mod) {
        if (!mod || hipModuleGetFunction(&cached_fn, mod, "gemm_f32_bias_wmma") != hipSuccess)
            cached_fn = NULL;
        cached_mod = mod;
    }
    return cached_fn;
}

/* Convenience wrapper for the many call sites whose kernel-params array is the
 * canonical gemm_f32_bias layout {&Y,&X,&W,&b,&N,&D_in,&D_out}: derives N/D_in/
 * D_out from args[4..6] so inline launches convert with a single substitution.
 * Uses the default stream (matches the original hipModuleLaunchKernel sites). */
static inline hipError_t cs3d_launch_gemm_args(hipFunction_t gemm_scalar,
                                               hipFunction_t gemm_wmma,
                                               void **args)
{
    int N     = *(int *)args[4];
    int D_in  = *(int *)args[5];
    int D_out = *(int *)args[6];
    return cs3d_launch_gemm(gemm_scalar, gemm_wmma, args, N, D_in, D_out, 0);
}

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_WMMA_H_ */
