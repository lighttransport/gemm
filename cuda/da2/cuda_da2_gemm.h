/* Compatibility names for the DA2 hybrid runner. */
#include "../gemm/cuda_linear_f32.h"
#define da2_cuda_free cuda_linear_f32_free
#define da2_cuda_init cuda_linear_f32_init
#define da2_cuda_linear cuda_linear_f32_run
