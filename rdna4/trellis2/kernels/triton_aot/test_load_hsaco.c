/* Smoke test: can hipModuleLoadData consume a Triton-emitted .hsaco? */
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <hsaco> <kernel_name>\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz);
    fread(buf, 1, sz, f); fclose(f);
    printf("hsaco size = %ld bytes\n", sz);

    hipInit(0);
    hipDevice_t dev; hipDeviceGet(&dev, 0);
    hipCtx_t ctx; hipCtxCreate(&ctx, 0, dev);

    hipModule_t mod = NULL;
    hipError_t err = hipModuleLoadData(&mod, buf);
    if (err != hipSuccess) {
        fprintf(stderr, "hipModuleLoadData failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("hipModuleLoadData OK\n");

    hipFunction_t kfn = NULL;
    err = hipModuleGetFunction(&kfn, mod, argv[2]);
    if (err != hipSuccess) {
        fprintf(stderr, "hipModuleGetFunction(%s) failed: %s\n", argv[2], hipGetErrorString(err));
        return 1;
    }
    printf("hipModuleGetFunction OK -> %p\n", (void*)kfn);

    int sharedSizeBytes = 0;
    hipFuncGetAttribute(&sharedSizeBytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfn);
    int numRegs = 0;
    hipFuncGetAttribute(&numRegs, HIP_FUNC_ATTRIBUTE_NUM_REGS, kfn);
    printf("kernel: shared=%d bytes  numRegs=%d\n", sharedSizeBytes, numRegs);
    return 0;
}
