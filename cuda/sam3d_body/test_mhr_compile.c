/* Offline NVRTC syntax/codegen check; does not require a CUDA device. */
#include "../../common/sam3d_body_mhr.h"
#include "mhr_decode_cuda.h"

int main(void)
{
    if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    const char *sources[] = {mhr_cuda_source,
        "extern \"C\" {\n" CUDA_GEMM_F32_BIAS_SRC CUDA_GEMV_F32_SRC "}\n"};
    for (int i = 0; i < 2; i++) {
        nvrtcProgram program;
        if (nvrtcCreateProgram(&program, sources[i], "mhr_compile.cu", 0, NULL, NULL)) return 2;
        const char *options[] = {"--gpu-architecture=sm_120", "--std=c++11"};
        nvrtcResult result = nvrtcCompileProgram(program, 2, options);
        size_t size = 0;
        nvrtcGetProgramLogSize(program, &size);
        char *log = calloc(size + 1, 1);
        if (!log) { nvrtcDestroyProgram(&program); return 3; }
        nvrtcGetProgramLog(program, log);
        fprintf(stderr, "%s", log);
        free(log);
        if (result) { nvrtcDestroyProgram(&program); return 4; }
        nvrtcGetCUBINSize(program, &size);
        printf("source %d: sm_120 compile PASS (%zu bytes)\n", i, size);
        nvrtcDestroyProgram(&program);
    }
    return 0;
}
