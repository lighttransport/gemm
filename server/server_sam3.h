#ifndef SERVER_SAM3_H_
#define SERVER_SAM3_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *png;      /* caller frees via server_sam3_free_mask */
    int      png_len;
    int      width;
    int      height;
    float    score;
    float    box[4];   /* xyxy in source-image pixel coords */
} server_sam3_mask;

int  server_sam3_cpu_segment(const char *ckpt_path,
                              const char *vocab_path,
                              const char *merges_path,
                              const uint8_t *img_bytes, size_t img_len,
                              const char *phrase,
                              float score_thr, float mask_thr,
                              int threads,
                              server_sam3_mask *out_masks, int out_cap,
                              int *out_n,
                              char *err_buf, size_t err_cap);

#if defined(DIFFUSION_SERVER_ENABLE_SAM3_CUDA)
int  server_sam3_cuda_segment(const char *ckpt_path,
                               const char *vocab_path,
                               const char *merges_path,
                               const uint8_t *img_bytes, size_t img_len,
                               const char *phrase,
                               float score_thr, float mask_thr,
                               int device_ordinal,
                               const char *precision,
                               server_sam3_mask *out_masks, int out_cap,
                               int *out_n,
                               char *err_buf, size_t err_cap);
#endif

void server_sam3_free_mask(server_sam3_mask *m);

#ifdef __cplusplus
}
#endif

#endif
