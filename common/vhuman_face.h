/* Native face mathematics. Caller buffers are contiguous row-major arrays.
 * Model storage is copied on open; outputs remain caller owned. No Python ABI. */
#ifndef VHUMAN_FACE_H
#define VHUMAN_FACE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vhf_gnm vhf_gnm;
int vhf_abi_version(void);
const char *vhf_error(void);
vhf_gnm *vhf_gnm_open(int vertices, int identities, int expressions, int threads,
    const int32_t *parents, const float *rest, const float *joints,
    const float *identity, const float *joint_identity, const float *expression,
    const float *correctives, const float *weights);
void vhf_gnm_close(vhf_gnm *model);
/* beta[I], expr[F,E], rotation[F,4,3], translation[F,3], residual[V,3].
 * Optional upstreams dp[F,V,3], dj[F,4,3] request a reverse pass. */
int vhf_gnm_eval(vhf_gnm *model, int frames, const float *beta, const float *expr,
    const float *rotation, const float *translation, const float *residual,
    float *positions, float *joints, const float *dp, const float *dj,
    float *dbeta, float *dexpr, float *drotation, float *dtranslation);
int vhf_gemm(float *out, const float *a, const float *b, int m, int n, int k,
             int transpose_a, int transpose_b, int threads);
/* Exact rotation matrix and its three derivatives, including at the origin. */
int vhf_rodrigues(const double *rotation, double *matrix, double *derivative, int count);

/* Return zero on success, -1 with vhf_error() on failure, except detector
 * decoding which returns the number of rows written. Buffers are caller owned. */
int vhf_image_sample(const uint8_t *rgb,int h,int w,const double *center,
    const double *size,double angle,int resolution,int zero_border,float *out);
int vhf_image_resize(const float *src,int h,int w,int channels,int oh,int ow,int nearest,int quantize,float *out);
int vhf_face_detections(const float *raw,float threshold,float *out,int capacity);
int vhf_adam(float *x,const float *g,float *m,float *v,const double *lr,int n,int step,double clip);
int vhf_adam64(double *x,const double *g,double *m,double *v,const double *lr,int n,int step,double clip);
int vhf_lbfgs(float *x,int n,int iterations,int max_evals,
    int (*objective)(void*,const float*,float*,double*),void *user);
int vhf_registration_loss(const double *x,int vertices,const int32_t *ids,int count,
    const double *target,const double *normal,const int32_t *edges,int edge_count,const double *length,
    double *gradient,double *energy);
int vhf_project(const float *points,const double *cameras,int frames,int count,float *pixels,float *depths);
int vhf_reprojection(const float *points,const double *cameras,const float *target,
    const float *weight,const float *ipd,int frames,int count,float *gradient,double *loss);
int vhf_quadratic(const float *x,const float *reference,int count,double weight,float *gradient,double *loss);
int vhf_area_loss(const float *rest,const float *posed,const int32_t *triangles,
    int vertices,int faces,int frames,float *gradient,double *loss);
int vhf_contact(const float *points,const float *joints,const float *rotations,const double *alignment,
    int frames,int anchors,const int32_t *pairs,int pair_count,const int32_t *eyes,const float *radii,int eye_count,
    const float *limit,float *distances,float *gp,float *gj,float *gr,double *loss);
int vhf_temporal(const float *expression,const float *rotations,const double *times,int frames,int modes,
    float *ge,float *gr,double *loss);
int vhf_silhouette(const float *vertices,int count,const int32_t *triangles,const float *bary,int anchors,
    const double *camera,const float *sdf,const float *allowed,int h,int w,double scale,
    const float *target,int targets,float *gradient,double *loss);
#ifdef __cplusplus
}
#endif
#endif
