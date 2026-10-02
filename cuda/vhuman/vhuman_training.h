#ifndef VHUMAN_TRAINING_H
#define VHUMAN_TRAINING_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct trainer vht_trainer;
const char *vht_error(void);
int vht_compile_probe(void);
vht_trainer *vht_open(int device,const float *parameters,int count,double lr,double decay,size_t budget_bytes);
void vht_close(vht_trainer *trainer);
int vht_parameters(vht_trainer *trainer,float *parameters,int upload);
int vht_optimizer(vht_trainer *trainer,double lr,double decay);
size_t vht_peak_bytes(vht_trainer *trainer);
size_t vht_overlaps(vht_trainer *trainer);
int vht_gemm(vht_trainer *trainer,float *out,const float *a,const float *b,int m,int n,int k,int ta,int tb);
/* All inputs are host arrays. Output/gradient can be NULL to skip downloads.
 * Truth NULL selects forward-only. Update requires truth and mask. A successful
 * call synchronizes its private stream; moments persist until close. */
int vht_cues(vht_trainer *trainer,const float *inputs,const float *truth,const float *mask,int n,int h,int w,
    float *output,float *gradient,double *loss,int update);
int vht_appearance(vht_trainer *trainer,const float *vertices,const int *attachments,const float *bary,
    const float *controls,const float *camera,int n,int v,int c,int width,int height,const float *truth,const float *mask,
    float *rgba,float *gradient,double *loss,int update);
#ifdef __cplusplus
}
#endif
#endif
