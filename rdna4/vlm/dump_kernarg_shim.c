/*
 * dump_kernarg_shim.c — LD_PRELOAD shim that intercepts hipExtModuleLaunchKernel,
 * dumps kernarg buffer + grid/block metadata to disk, then forwards the call.
 *
 * Used to capture exactly what hipBLASLt's runtime hands to algo 73624 for
 * mm0 (M=1024, N=4608, K=4608, BF16). Output:
 *   /tmp/mm0_kernarg_<idx>.bin     — raw kernarg bytes
 *   /tmp/mm0_kernarg_<idx>.meta    — text: grid, block, arg size, kernel name
 *
 * Only the first MAX_DUMPS launches are captured. Filter at run-site by
 * setting DUMP_KERNARG_FILTER=<substring> to match kernel name; otherwise
 * every launch is dumped. We also dump a host-side copy of the kernel name
 * via hipFuncGetName / hipModuleGetFunction reverse lookup (best-effort —
 * if it fails, name is left empty).
 *
 * Build:
 *   clang -shared -fPIC -O2 -o libdump_kernarg.so dump_kernarg_shim.c \
 *       -I/opt/rocm/include -ldl -lstdc++ \
 *       -L/opt/rocm/lib -lamdhip64
 *
 * Run:
 *   LD_PRELOAD=$PWD/libdump_kernarg.so DUMP_KERNARG_FILTER=Cijk_Alik_Bljk \
 *     ./bench_vlm_gemm_blaslt --shape mm0 --iters 1
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>

#define MAX_DUMPS 32

typedef hipError_t (*launch_fn_t)(hipFunction_t,
                                  uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t,
                                  size_t, hipStream_t,
                                  void**, void**,
                                  hipEvent_t, hipEvent_t);

static int g_dump_count = 0;

hipError_t hipExtModuleLaunchKernel(hipFunction_t f,
                                    uint32_t globalX, uint32_t globalY, uint32_t globalZ,
                                    uint32_t localX,  uint32_t localY,  uint32_t localZ,
                                    size_t sharedMemBytes, hipStream_t stream,
                                    void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent) {
    static launch_fn_t orig = NULL;
    if (!orig) {
        orig = (launch_fn_t)dlsym(RTLD_NEXT, "hipExtModuleLaunchKernel");
        if (!orig) {
            fprintf(stderr, "[dump_shim] dlsym failed: %s\n", dlerror());
            abort();
        }
    }

    if (extra && g_dump_count < MAX_DUMPS) {
        /* extra layout per HipSolutionAdapter:
         *   [HIP_LAUNCH_PARAM_BUFFER_POINTER, args_ptr,
         *    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &args_size,
         *    HIP_LAUNCH_PARAM_END]
         */
        void* args_buf  = NULL;
        size_t args_size = 0;
        for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END && i < 16; i += 2) {
            if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
                args_buf = extra[i+1];
            } else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
                args_size = *(size_t*)extra[i+1];
            }
        }

        const char* filter = getenv("DUMP_KERNARG_FILTER");
        const char* kernel_name = "";
        /* hipFuncGetName is best-effort; resolve via dlsym so we can compile
         * even when the header omits the prototype (older ROCm). */
        typedef hipError_t (*get_name_fn)(const char**, hipFunction_t);
        static get_name_fn s_get_name = NULL;
        static int s_get_name_resolved = 0;
        if (!s_get_name_resolved) {
            s_get_name = (get_name_fn)dlsym(RTLD_DEFAULT, "hipFuncGetName");
            s_get_name_resolved = 1;
        }
        const char* nm = NULL;
        if (s_get_name && s_get_name(&nm, f) == hipSuccess && nm) {
            kernel_name = nm;
        }
        int matches = 1;
        if (filter && filter[0]) {
            matches = (kernel_name[0] && strstr(kernel_name, filter) != NULL);
        }

        if (matches && args_buf && args_size > 0) {
            char binpath[256], metapath[256];
            snprintf(binpath,  sizeof binpath,  "/tmp/mm0_kernarg_%d.bin",  g_dump_count);
            snprintf(metapath, sizeof metapath, "/tmp/mm0_kernarg_%d.meta", g_dump_count);

            FILE* fp = fopen(binpath, "wb");
            if (fp) { fwrite(args_buf, 1, args_size, fp); fclose(fp); }

            FILE* mp = fopen(metapath, "w");
            if (mp) {
                fprintf(mp,
                        "kernel=%s\n"
                        "grid=%u,%u,%u\n"
                        "block=%u,%u,%u\n"
                        "shared_mem=%zu\n"
                        "args_size=%zu\n",
                        kernel_name,
                        globalX, globalY, globalZ,
                        localX,  localY,  localZ,
                        sharedMemBytes, args_size);
                fclose(mp);
            }
            fprintf(stderr, "[dump_shim] dump #%d kernel=%s grid=(%u,%u,%u) block=(%u,%u,%u) args=%zuB\n",
                    g_dump_count, kernel_name, globalX, globalY, globalZ, localX, localY, localZ, args_size);
            g_dump_count++;
        }
    }

    return orig(f, globalX, globalY, globalZ, localX, localY, localZ,
                sharedMemBytes, stream, kernelParams, extra, startEvent, stopEvent);
}
