#ifndef HV15_CUDA_ATTENTION_H
#define HV15_CUDA_ATTENTION_H
#include "stable-diffusion.h"
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
SD_API bool hv15_cuda_attention_install(ggml_backend_t backend);
SD_API uint64_t hv15_cuda_attention_calls(void);
#ifdef __cplusplus
}
#endif
#endif
