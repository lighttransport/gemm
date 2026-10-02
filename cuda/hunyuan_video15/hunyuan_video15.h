#ifndef HUNYUAN_VIDEO15_H
#define HUNYUAN_VIDEO15_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One active context per process: the bootstrap library has global callbacks. */
typedef struct hv15_context hv15_context;
typedef struct {
    const char *diffusion, *vae, *qwen, *byt5, *vision, *tokenizer;
    int device, vram_budget_mib, threads;
    int allow_experimental; /* Required until official component parity is established. */
} hv15_model;
typedef struct {
    const char *task, *preset, *prompt, *negative_prompt, *image, *vision_pixels;
    int width, height, frames;
    int64_t seed;
} hv15_request;
typedef struct {
    void (*progress)(int step, int total, void *user);
    int (*frame)(int index, int width, int height, const unsigned char *rgb, void *user);
    int (*cancelled)(void *user);
    void *user;
} hv15_callbacks;
void hv15_defaults(hv15_request *request);
int hv15_validate(const hv15_request *request, char *error, unsigned capacity);
hv15_context *hv15_load(const hv15_model *model, char *error, unsigned capacity);
int hv15_generate(hv15_context *ctx, const hv15_request *request,
                  const hv15_callbacks *callbacks, char *error, unsigned capacity);
void hv15_cancel(hv15_context *ctx);
void hv15_free(hv15_context *ctx);
#ifdef __cplusplus
}
#endif
#endif
