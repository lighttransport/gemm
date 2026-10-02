#ifndef VHUMAN_RUNTIME_H
#define VHUMAN_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vh_cuda_runtime vh_cuda_runtime;
/* A negative borrowed flag creates an owned stream; otherwise borrow stream. */
vh_cuda_runtime *vh_cuda_open(int device, uintptr_t stream, int borrowed);
void vh_cuda_close(vh_cuda_runtime *runtime);
uintptr_t vh_cuda_stream(vh_cuda_runtime *runtime);
int vh_cuda_sync(vh_cuda_runtime *runtime);
uintptr_t vh_cuda_record(vh_cuda_runtime *runtime);
int vh_cuda_wait(vh_cuda_runtime *runtime, uintptr_t event);
int vh_cuda_elapsed(vh_cuda_runtime *runtime, uintptr_t begin, uintptr_t end, float *ms);
int vh_cuda_event_sync(vh_cuda_runtime *runtime, uintptr_t event);
void vh_cuda_event_free(vh_cuda_runtime *runtime, uintptr_t event);
int vh_cuda_download(vh_cuda_runtime *runtime, uintptr_t pointer, void *host, size_t bytes);
/* Premultiplied linear RGBA -> composited sRGB RGB8, or straight sRGB RGBA8.
 * Reuses device and pinned host staging; synchronous copy is the output boundary. */
int vh_cuda_pixels(vh_cuda_runtime *runtime, uintptr_t rgba, int width, int height,
                   float background, int straight_alpha, unsigned char *host);
int vh_pixels_cpu(const float *rgba, size_t pixels, float background,
                  int straight_alpha, unsigned char *host);
/* Compile embedded kernels without requiring a CUDA device (validation only). */
int vh_cuda_compile_probe(void);
#ifdef __cplusplus
}
#endif
#endif
