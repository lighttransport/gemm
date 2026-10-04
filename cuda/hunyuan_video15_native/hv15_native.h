#ifndef PIXAL3D_HV15_NATIVE_H
#define PIXAL3D_HV15_NATIVE_H
#include <stddef.h>
#include <stdint.h>
#define HV15N_DEFAULT_MODEL_DIR "/mnt/nvme02/data/models/hv15"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct hv15n_context hv15n_context;
typedef struct {
    const char *model_dir;
    int device, vram_budget_mib;
    const char *gemm;          /* repo (default), cublas */
    const char *gemm_fallback; /* cublas (default), error */
} hv15n_config;
typedef struct {
    const char *task, *preset, *prompt, *negative_prompt;
    const char *image, *vision_pixels; /* prepared RGB image and CHW F32 pixels */
    const char *noise_file, *dump_dir;
    int width, height, frames;
    int64_t seed;
} hv15n_request;
typedef struct {
    void (*progress)(int step, int total, void *user);
    int (*frame)(int index, int width, int height, const unsigned char *rgb, void *user);
    int (*cancelled)(void *user);
    void *user;
} hv15n_callbacks;
void hv15n_config_defaults(hv15n_config *config);
void hv15n_request_defaults(hv15n_request *request);
int hv15n_validate(const hv15n_request *request, char *error, size_t capacity);
hv15n_context *hv15n_load(const hv15n_config *config, char *error, size_t capacity);
int hv15n_generate(hv15n_context *context, const hv15n_request *request, const hv15n_callbacks *callbacks,
                   char *error, size_t capacity);
/* JSON valid until the next generation or destruction. */
const char *hv15n_metrics(const hv15n_context *context);
/* Optional ROCm attention provider; configure an idle context before generation.
 * The existing configuration struct and CUDA ABI remain unchanged. */
int hv15n_set_aotriton_bridge(hv15n_context *context, const char *path,
                            char *error, size_t capacity);
/* CUDA only: opt-in cuDNN SDPA for the DiT attention on an idle context. mode is
 * NULL/"off" (default, private FlashAttention-2), "auto" (libh3_cudnn.so beside this
 * library or the executable; falls back with a warning if unavailable) or a bridge path
 * (errors if unusable). cudnn_library is a libcudnn.so.9 path or NULL (loader search
 * path and fixed system locations). */
int hv15n_set_cudnn_attention(hv15n_context *context, const char *mode,
                              const char *cudnn_library, char *error, size_t capacity);
void hv15n_cancel(hv15n_context *context);
void hv15n_free(hv15n_context *context);
#ifdef __cplusplus
}
#endif
#endif
