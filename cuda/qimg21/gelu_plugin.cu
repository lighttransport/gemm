#include <cmath>
#include <cuda_runtime.h>

__device__ float q21_bf16_round_gelu(float value) {
    unsigned bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__global__ void q21_vision_gelu_exact_kernel(float *x, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = x[i];
        constexpr float alpha = M_SQRT1_2;
        x[i] = q21_bf16_round_gelu(value * 0.5f * (1.0f + ::erf(value * alpha)));
    }
}

extern "C" int q21_vision_gelu_exact(float *x, int count, cudaStream_t stream) {
    if (!x || count < 0) return cudaErrorInvalidValue;
    q21_vision_gelu_exact_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, count);
    return cudaGetLastError();
}
