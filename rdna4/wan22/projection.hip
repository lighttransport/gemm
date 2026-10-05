// SPDX-License-Identifier: MIT
// Q8_0 is block-scaled weight-only INT8. Decode one projection at a time,
// then use the same RDNA4 FP16 WMMA implementation as HunyuanVideo 1.5.
#include "../video_common/gemm.hip"

extern "C" __global__ void wan22_q8_decode(_Float16 *out, const unsigned char *packed,
                                          long count) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned char *block = packed + (i / 32) * 34;
        _Float16 scale = *reinterpret_cast<const _Float16 *>(block);
        out[i] = (_Float16)(scale * (_Float16)(signed char)block[2 + i % 32]);
    }
}

// All pointers belong to PyTorch; all launches use its current HIP stream.
// No allocation, context changes, synchronization or hidden default stream.
extern "C" int wan22_projection(float *out, const _Float16 *input,
                                const unsigned char *packed, _Float16 *scratch,
                                int m, int n, int k, void *stream_pointer) {
    if (!out || !input || !packed || !scratch || m <= 0 || n <= 0 || k <= 0 || k % 32)
        return (int)hipErrorInvalidValue;
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
    long count = (long)n * k;
    hipLaunchKernelGGL(wan22_q8_decode, dim3((count + 255) / 256), dim3(256), 0,
                       stream, scratch, packed, count);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess)
        return (int)status;
    if (m >= 128 && n >= 128) {
        hipLaunchKernelGGL(video_gemm_f16_tiled, dim3((n + 127) / 128, (m + 127) / 128),
                           dim3(128), 0, stream, out, input, scratch, m, n, k);
    } else {
        hipLaunchKernelGGL(video_gemm_f16, dim3((n + 31) / 32, (m + 31) / 32),
                           dim3(128), 0, stream, out, input, scratch, m, n, k);
    }
    return (int)hipGetLastError();
}
