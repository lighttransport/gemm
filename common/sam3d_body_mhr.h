/*
 * sam3d_body_mhr.h — MHR (Momentum Human Rig) skinning for sam-3d-body.
 *
 * Usage:
 *   #define SAM3D_BODY_MHR_IMPLEMENTATION
 *   #include "sam3d_body_mhr.h"
 *
 * Dependencies: safetensors.h, ggml_dequant.h (for qtensor typedef).
 *
 * This ports the pymomentum.mhr.MHR torch.jit ScriptModule that turns
 * an MHR parameter vector (shape, model_params, face_expr) into world-
 * space vertices and per-joint skel_state. Input assets come from
 * ref/sam3d-body/dump_mhr_assets.py; see cpu/sam3d_body/MHR_PORT.md.
 *
 * The 12-stage staircase (stages 1-5 live in the decoder head, 6-11 here):
 *   (6)  parameter_transform:           (B, 249)  -> (B, 889)
 *   (7)  joint_params -> local_skel:    (B, 127, 7) -> (B, 127, 8)
 *   (8)  local_skel  -> global_skel:    prefix-product walker (4 stages)
 *   (9)  quat -> rotmat (joint_global_rots)
 *   (10) shape/face blends + pose_correctives -> linear_model_unposed
 *   (11) LBS skin_points(global_skel, linear_model_unposed)
 *   (12) keypoint_mapping regression (lives in the decoder head)
 *
 * Skel-state convention: per-joint 8-vec (tx, ty, tz, qx, qy, qz, qw, s).
 * Quaternion convention: xyzw, Hamilton product, normalize() clamps by 1e-12.
 */

#ifndef SAM3D_BODY_MHR_H
#define SAM3D_BODY_MHR_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SAM3D_BODY_MHR_E_OK              (0)
#define SAM3D_BODY_MHR_E_INVAL           (-1)
#define SAM3D_BODY_MHR_E_LOAD            (-3)

/* Constants baked into mhr_model.pt — verified by dump_mhr_assets.py. */
#define S3DM_N_JOINTS         127
#define S3DM_N_VERTS          18439
#define S3DM_N_SKIN           51337
#define S3DM_N_MODEL_PARAMS   204         /* body + hands + scale   */
#define S3DM_N_PTRANS_IN      249         /* model_params + shape45 */
#define S3DM_N_PTRANS_OUT     889         /* joint_parameters       */
#define S3DM_N_SHAPE          45
#define S3DM_N_FACE           72
#define S3DM_N_PC_NNZ         53136       /* pose_correctives sparse */
#define S3DM_N_PC_IN          750         /* 125 joints * 6 = 750   */
#define S3DM_N_PC_H           3000        /* sparse hidden dim      */
#define S3DM_PMI_COLS         266         /* total prefix-prod steps */
#define S3DM_PMI_STAGES       4

#include "qtensor_utils.h"

/* ---- Asset container ---------------------------------------------- */

typedef struct sam3d_body_mhr_assets_t {
    /* Scalar constants. */
    int pmi_buffer_sizes[S3DM_PMI_STAGES];  /* {65, 56, 62, 83} */

    /* Identity + face shape blends. */
    qtensor blend_shape_vectors;     /* (45, 18439, 3) f32 */
    qtensor blend_base_shape;        /* (18439, 3)     f32 */
    qtensor face_shape_vectors;      /* (72, 18439, 3) f32 */

    /* Parameter transform (model_params + shape_pad -> joint_params). */
    qtensor parameter_transform;     /* (889, 249) f32 */

    /* Skeleton. */
    qtensor joint_translation_offsets;  /* (127, 3) f32     */
    qtensor joint_prerotations;         /* (127, 4) f32 xyzw */
    qtensor pmi;                        /* (2, 266) i64     */
    qtensor joint_parents;              /* (127,)   i32     */

    /* Linear blend skinning. */
    qtensor inverse_bind_pose;       /* (127, 8)  f32 */
    qtensor skin_indices_flat;       /* (51337,)  i32 */
    qtensor skin_weights_flat;       /* (51337,)  f32 */
    qtensor vert_indices_flat;       /* (51337,)  i64 */

    /* Pose correctives — COO sparse (3000,750) + dense Linear (55317,3000). */
    qtensor pc_sparse_indices;       /* (2, 53136) i64 */
    qtensor pc_sparse_weight;        /* (53136,)  f32 */
    qtensor pc_linear_weight;        /* (55317, 3000) f32 */

    /* Backing buffer for parsed safetensors (kept alive until free). */
    void *_st;
} sam3d_body_mhr_assets;

/* ---- Public API --------------------------------------------------- */

sam3d_body_mhr_assets *sam3d_body_mhr_load(const char *assets_sft,
                                           const char *assets_json);
void                   sam3d_body_mhr_free(sam3d_body_mhr_assets *a);

/*
 * Stage 6: parameter_transform(model_params [B, 204], shape [B, 45])
 *   -> joint_parameters [B, 889]
 *
 * The (889, 249) matrix is multiplied by cat(model_params, shape) along the
 * last axis. The shape coeffs zero out at the jit-level (character_torch
 * doesn't wire shape into parameter_transform in this rig), but the
 * upstream jit still appends zeros(45) — we replicate the same
 * shape = zeros(45) convention for bit-parity with the reference.
 */
int sam3d_body_mhr_parameter_transform(const sam3d_body_mhr_assets *a,
                                       const float *model_params,
                                       int B,
                                       int n_threads,
                                       float *out_joint_params);

/*
 * Stage 7: joint_params (B, 127, 7) -> local_skel_state (B, 127, 8).
 *
 *   local_t = jp[..., 0:3] + joint_translation_offsets
 *   local_q = qmul_normalized(joint_prerotations,
 *                             euler_xyz_to_quat(jp[..., 3:6]))
 *   local_s = exp(jp[..., 6] * ln(2))
 *
 * This is the canary — exercises euler→quat, quat multiply, and the
 * w-last ordering convention.
 */
int sam3d_body_mhr_joint_params_to_local_skel(
        const sam3d_body_mhr_assets *a,
        const float *joint_params,   /* (B, 889) */
        int B,
        float *out_local_skel);      /* (B, 127, 8) */

/*
 * Stage 8: local_skel_state -> global_skel_state via the 4-stage
 * prefix-product walker (fp64 accumulation to match jit).
 *
 *   for stage in 0..3:
 *     for each (source, target) pair in pmi[stage]:
 *       global[source] = skel_multiply(global[target], global[source])
 *
 * pmi is laid out as (2, 266) = [source_row; target_row], partitioned by
 * pmi_buffer_sizes = [65, 56, 62, 83] along the column axis.
 */
int sam3d_body_mhr_local_to_global_skel(
        const sam3d_body_mhr_assets *a,
        const float *local_skel,     /* (B, 127, 8) */
        int B,
        float *out_global_skel);     /* (B, 127, 8) */

/*
 * Stage 10A: blend_shape(shape_coeffs [B, 45]) -> (B, 18439, 3)
 *   out[v,d] = sum_n shape_vectors[n,v,d] * coeffs[n] + base_shape[v,d]
 */
int sam3d_body_mhr_blend_shape(const sam3d_body_mhr_assets *a,
                               const float *shape_coeffs,   /* (B, 45) */
                               int B,
                               int n_threads,
                               float *out_verts);           /* (B, 18439, 3) */

/*
 * Stage 10B: face_expressions(face_coeffs [B, 72]) -> (B, 18439, 3)
 *   out[v,d] = sum_n face_shape_vectors[n,v,d] * coeffs[n]   (no base)
 */
int sam3d_body_mhr_face_expressions(const sam3d_body_mhr_assets *a,
                                    const float *face_coeffs,  /* (B, 72) */
                                    int B,
                                    int n_threads,
                                    float *out_verts);         /* (B, 18439, 3) */

/*
 * Stage 10C: pose_correctives(joint_params [B, 889]) -> (B, 18439, 3)
 *   feat = batch6DFromXYZ(jp[:, 2:127, 3:6])                 (B, 125, 6)
 *   feat[:,:,0] -= 1; feat[:,:,4] -= 1                        (subtract id)
 *   feat = flatten(feat, dims 1,2) -> (B, 750)
 *   h = ReLU(SparseLinear(feat))                              (B, 3000)
 *   out = Linear(h)                                           (B, 55317)
 *   reshape -> (B, 18439, 3)
 */
int sam3d_body_mhr_pose_correctives(const sam3d_body_mhr_assets *a,
                                    const float *joint_params,  /* (B, 889) */
                                    int B,
                                    int n_threads,
                                    float *out_verts);          /* (B, 18439, 3) */

/*
 * Stage 11: LBS skin_points(global_skel, rest_vertex_positions)
 *   joint_state = skel_multiply(global_skel, inverse_bind_pose)
 *   skinned[v] = sum_{k: vert_indices[k]==v} skin_weights[k]
 *                * skel_transform_points(joint_state[skin_indices[k]],
 *                                        rest_verts[vert_indices[k]])
 */
int sam3d_body_mhr_skin_points(const sam3d_body_mhr_assets *a,
                               const float *global_skel,    /* (B, 127, 8)     */
                               const float *rest_verts,     /* (B, 18439, 3)   */
                               int B,
                               float *out_skinned);         /* (B, 18439, 3)   */

/*
 * Top-level forward — stages 6..11 composed.
 *
 *   in:  model_params (B, 204), shape (B, 45), face (B, 72)
 *   out: skinned_verts (B, 18439, 3), global_skel_state (B, 127, 8)
 *
 * Scratch: if non-NULL, must be at least
 *   B * (889 + 127*8*2 + 18439*3*3) floats.
 */
int sam3d_body_mhr_forward(const sam3d_body_mhr_assets *a,
                           const float *model_params,
                           const float *shape,
                           const float *face,
                           int B,
                           int apply_correctives,
                           int n_threads,
                           float *scratch,
                           float *out_skinned_verts,
                           float *out_global_skel);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef SAM3D_BODY_MHR_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "safetensors.h"

/* ---- tiny file-slurp helper for the JSON sidecar ------------------ */

static char *s3dm_slurp(const char *path, size_t *sz_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[n] = 0;
    if (sz_out) *sz_out = (size_t)n;
    return buf;
}

/* ---- tensor finders matching any of f32 / i32 / i64 --------------- */

static int s3dm_find(st_context *st, const char *name,
                     const char *want_dtype, qtensor *out)
{
    int i = safetensors_find(st, name);
    if (i < 0) {
        fprintf(stderr, "sam3d_body_mhr: missing tensor %s\n", name);
        return -1;
    }
    const char *dt = safetensors_dtype(st, i);
    if (strcmp(dt, want_dtype) != 0) {
        fprintf(stderr, "sam3d_body_mhr: %s dtype=%s (want %s)\n",
                name, dt, want_dtype);
        return -1;
    }
    out->data = (void *)safetensors_data(st, i);
    if (!strcmp(want_dtype, "F32"))      out->type = 1;
    else if (!strcmp(want_dtype, "I32")) out->type = 2;
    else if (!strcmp(want_dtype, "I64")) out->type = 3;
    else                                  out->type = 0;
    int nd = safetensors_ndims(st, i);
    const uint64_t *shape = safetensors_shape(st, i);
    out->n_dims = nd;
    for (int d = 0; d < nd && d < 4; d++) out->dims[d] = shape[d];
    out->n_rows = (nd >= 1) ? (int)out->dims[0] : 1;
    out->n_cols = (nd >= 2) ? (int)out->dims[1] : 1;
    return 0;
}

/* ---- Loader ------------------------------------------------------- */

sam3d_body_mhr_assets *sam3d_body_mhr_load(const char *assets_sft,
                                           const char *assets_json)
{
    if (!assets_sft) return NULL;

    sam3d_body_mhr_assets *a =
        (sam3d_body_mhr_assets *)calloc(1, sizeof(*a));
    if (!a) return NULL;

    st_context *st = safetensors_open(assets_sft);
    if (!st) {
        fprintf(stderr, "sam3d_body_mhr: safetensors_open(%s) failed\n",
                assets_sft);
        free(a);
        return NULL;
    }
    a->_st = st;

    int bad = 0;
    bad |= s3dm_find(st, "blend_shape.shape_vectors", "F32", &a->blend_shape_vectors);
    bad |= s3dm_find(st, "blend_shape.base_shape",    "F32", &a->blend_base_shape);
    bad |= s3dm_find(st, "face_expressions.shape_vectors", "F32",
                     &a->face_shape_vectors);
    bad |= s3dm_find(st, "parameter_transform",          "F32",
                     &a->parameter_transform);
    bad |= s3dm_find(st, "skeleton.joint_translation_offsets", "F32",
                     &a->joint_translation_offsets);
    bad |= s3dm_find(st, "skeleton.joint_prerotations",  "F32",
                     &a->joint_prerotations);
    bad |= s3dm_find(st, "skeleton.pmi",                 "I64", &a->pmi);
    bad |= s3dm_find(st, "skeleton.joint_parents",       "I32",
                     &a->joint_parents);
    bad |= s3dm_find(st, "lbs.inverse_bind_pose",        "F32",
                     &a->inverse_bind_pose);
    bad |= s3dm_find(st, "lbs.skin_indices_flattened",   "I32",
                     &a->skin_indices_flat);
    bad |= s3dm_find(st, "lbs.skin_weights_flattened",   "F32",
                     &a->skin_weights_flat);
    bad |= s3dm_find(st, "lbs.vert_indices_flattened",   "I64",
                     &a->vert_indices_flat);
    bad |= s3dm_find(st, "pose_correctives.sparse_indices", "I64",
                     &a->pc_sparse_indices);
    bad |= s3dm_find(st, "pose_correctives.sparse_weight",  "F32",
                     &a->pc_sparse_weight);
    bad |= s3dm_find(st, "pose_correctives.linear_weight",  "F32",
                     &a->pc_linear_weight);
    if (bad) {
        sam3d_body_mhr_free(a);
        return NULL;
    }

    /* Scalar sidecar — hand-parse since we only need the "scalars" object
     * and don't want to take a full json_val dep for four ints. */
    a->pmi_buffer_sizes[0] = 65;
    a->pmi_buffer_sizes[1] = 56;
    a->pmi_buffer_sizes[2] = 62;
    a->pmi_buffer_sizes[3] = 83;
    if (assets_json) {
        size_t sz = 0;
        char *buf = s3dm_slurp(assets_json, &sz);
        if (buf) {
            const char *p = strstr(buf, "\"pmi_buffer_sizes\"");
            if (p) {
                p = strchr(p, '[');
                if (p) {
                    int vals[S3DM_PMI_STAGES], i = 0;
                    while (*p && i < S3DM_PMI_STAGES) {
                        if ((*p >= '0' && *p <= '9') ||
                            (*p == '-' && p[1] >= '0' && p[1] <= '9')) {
                            char *end = NULL;
                            long v = strtol(p, &end, 10);
                            vals[i++] = (int)v;
                            p = end;
                            if (*p == ']') break;
                        } else {
                            p++;
                        }
                    }
                    if (i == S3DM_PMI_STAGES) {
                        for (int k = 0; k < S3DM_PMI_STAGES; k++)
                            a->pmi_buffer_sizes[k] = vals[k];
                    }
                }
            }
            free(buf);
        }
    }

    /* Sanity checks on shapes. */
    if (a->parameter_transform.dims[0] != S3DM_N_PTRANS_OUT ||
        a->parameter_transform.dims[1] != S3DM_N_PTRANS_IN) {
        fprintf(stderr, "sam3d_body_mhr: parameter_transform shape (%llu,%llu) "
                "expected (%d,%d)\n",
                (unsigned long long)a->parameter_transform.dims[0],
                (unsigned long long)a->parameter_transform.dims[1],
                S3DM_N_PTRANS_OUT, S3DM_N_PTRANS_IN);
        sam3d_body_mhr_free(a);
        return NULL;
    }
    if (a->pmi.dims[0] != 2 || a->pmi.dims[1] != S3DM_PMI_COLS) {
        fprintf(stderr, "sam3d_body_mhr: pmi shape mismatch\n");
        sam3d_body_mhr_free(a);
        return NULL;
    }

    return a;
}

void sam3d_body_mhr_free(sam3d_body_mhr_assets *a)
{
    if (!a) return;
    if (a->_st) safetensors_close((st_context *)a->_st);
    free(a);
}

/* ---- Primitive geometry ops --------------------------------------- */

/* quat (x,y,z,w) = euler_xyz_to_quaternion(roll, pitch, yaw) */
static inline void s3dm_quat_from_euler_xyz(const float *r, float *q)
{
    float hr = r[0] * 0.5f, hp = r[1] * 0.5f, hy = r[2] * 0.5f;
    float cr = cosf(hr), sr = sinf(hr);
    float cp = cosf(hp), sp = sinf(hp);
    float cy = cosf(hy), sy = sinf(hy);
    q[0] = sr * cp * cy - cr * sp * sy; /* x */
    q[1] = cr * sp * cy + sr * cp * sy; /* y */
    q[2] = cr * cp * sy - sr * sp * cy; /* z */
    q[3] = cr * cp * cy + sr * sp * sy; /* w */
}

static inline void s3dm_quat_mul(const float *q1, const float *q2, float *out)
{
    float x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
    float x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
    out[0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    out[3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
}

static inline void s3dm_quat_normalize(const float *q, float *out)
{
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    /* torch.nn.functional.normalize uses eps=1e-12 on the sqrt, then
     * divides; for torch "eps" the denominator is max(||q||, eps). */
    float n = sqrtf(n2);
    if (n < 1e-12f) n = 1e-12f;
    float inv = 1.0f / n;
    out[0] = q[0] * inv;
    out[1] = q[1] * inv;
    out[2] = q[2] * inv;
    out[3] = q[3] * inv;
}

/* rotate_vector_assume_normalized: v + 2 * (r * (axis × v) + axis × (axis × v)) */
static inline void s3dm_quat_rot_vec_norm(const float *q, const float *v,
                                          float *out)
{
    float ax = q[0], ay = q[1], az = q[2], r = q[3];
    float avx = ay * v[2] - az * v[1];
    float avy = az * v[0] - ax * v[2];
    float avz = ax * v[1] - ay * v[0];
    float aavx = ay * avz - az * avy;
    float aavy = az * avx - ax * avz;
    float aavz = ax * avy - ay * avx;
    out[0] = v[0] + 2.0f * (avx * r + aavx);
    out[1] = v[1] + 2.0f * (avy * r + aavy);
    out[2] = v[2] + 2.0f * (avz * r + aavz);
}

/* skel_state * skel_state (both already normalized) */
static inline void s3dm_skel_multiply(const float *s1, const float *s2, float *o)
{
    const float *t1 = s1, *q1 = s1 + 3; float sc1 = s1[7];
    const float *t2 = s2, *q2 = s2 + 3; float sc2 = s2[7];
    float s1t2[3] = { sc1 * t2[0], sc1 * t2[1], sc1 * t2[2] };
    float rot[3];
    s3dm_quat_rot_vec_norm(q1, s1t2, rot);
    o[0] = t1[0] + rot[0];
    o[1] = t1[1] + rot[1];
    o[2] = t1[2] + rot[2];
    s3dm_quat_mul(q1, q2, o + 3);
    o[7] = sc1 * sc2;
}

/* skel_state transform_points((t,q,s), p) = t + rotate(q, s*p) (q not assumed
 * normalized, but for our LBS input — joint_state = multiply(global, inv_bind)
 * — the quaternion is normalized by construction. We still normalize() to
 * match the jit's check path.) */
static inline void s3dm_skel_transform_point(const float *s, const float *p,
                                             float *out)
{
    float qn[4];
    s3dm_quat_normalize(s + 3, qn);
    float sp[3] = { s[7] * p[0], s[7] * p[1], s[7] * p[2] };
    float rot[3];
    s3dm_quat_rot_vec_norm(qn, sp, rot);
    out[0] = s[0] + rot[0];
    out[1] = s[1] + rot[1];
    out[2] = s[2] + rot[2];
}

/* ---- fp64 variants for the prefix-product walker ------------------ */

static inline void s3dm_quat_mul_d(const double *q1, const double *q2,
                                   double *out)
{
    double x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
    double x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
    out[0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    out[3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
}

static inline void s3dm_quat_normalize_d(const double *q, double *out)
{
    double n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    double n = sqrt(n2);
    if (n < 1e-12) n = 1e-12;
    double inv = 1.0 / n;
    out[0] = q[0] * inv;
    out[1] = q[1] * inv;
    out[2] = q[2] * inv;
    out[3] = q[3] * inv;
}

static inline void s3dm_quat_rot_vec_norm_d(const double *q, const double *v,
                                            double *out)
{
    double ax = q[0], ay = q[1], az = q[2], r = q[3];
    double avx = ay * v[2] - az * v[1];
    double avy = az * v[0] - ax * v[2];
    double avz = ax * v[1] - ay * v[0];
    double aavx = ay * avz - az * avy;
    double aavy = az * avx - ax * avz;
    double aavz = ax * avy - ay * avx;
    out[0] = v[0] + 2.0 * (avx * r + aavx);
    out[1] = v[1] + 2.0 * (avy * r + aavy);
    out[2] = v[2] + 2.0 * (avz * r + aavz);
}

/* Multiply with _normalize_split_skel_state applied to both (matches jit). */
static inline void s3dm_skel_multiply_norm_d(const double *s1, const double *s2,
                                             double *o)
{
    double q1n[4], q2n[4];
    s3dm_quat_normalize_d(s1 + 3, q1n);
    s3dm_quat_normalize_d(s2 + 3, q2n);
    double sc1 = s1[7], sc2 = s2[7];
    double s1t2[3] = { sc1 * s2[0], sc1 * s2[1], sc1 * s2[2] };
    double rot[3];
    s3dm_quat_rot_vec_norm_d(q1n, s1t2, rot);
    o[0] = s1[0] + rot[0];
    o[1] = s1[1] + rot[1];
    o[2] = s1[2] + rot[2];
    s3dm_quat_mul_d(q1n, q2n, o + 3);
    o[7] = sc1 * sc2;
}

/* ---- Stage 6: parameter_transform matvec -------------------------- */

int sam3d_body_mhr_parameter_transform(const sam3d_body_mhr_assets *a,
                                       const float *model_params,
                                       int B,
                                       int n_threads,
                                       float *out_joint_params)
{
    (void)n_threads;
    if (!a || !model_params || !out_joint_params)
        return SAM3D_BODY_MHR_E_INVAL;

    const float *W = (const float *)a->parameter_transform.data;
    const int D_out = S3DM_N_PTRANS_OUT; /* 889 */
    const int D_in  = S3DM_N_PTRANS_IN;  /* 249 */
    const int N_mp  = S3DM_N_MODEL_PARAMS; /* 204 */

    /* Input is cat(model_params(204), zeros(45)). Instead of materialising
     * the padded vector, we just skip the last 45 columns of W (they're
     * multiplied by zero). */
    for (int b = 0; b < B; b++) {
        const float *x = model_params + (size_t)b * N_mp;
        float *y = out_joint_params + (size_t)b * D_out;
        for (int d = 0; d < D_out; d++) {
            double acc = 0.0;
            const float *Wrow = W + (size_t)d * D_in;
            for (int i = 0; i < N_mp; i++)
                acc += (double)Wrow[i] * (double)x[i];
            y[d] = (float)acc;
        }
    }
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Stage 7: joint_params -> local_skel_state -------------------- */

int sam3d_body_mhr_joint_params_to_local_skel(
        const sam3d_body_mhr_assets *a,
        const float *joint_params,
        int B,
        float *out_local_skel)
{
    if (!a || !joint_params || !out_local_skel)
        return SAM3D_BODY_MHR_E_INVAL;

    const float *jto = (const float *)a->joint_translation_offsets.data; /* (127,3) */
    const float *jpr = (const float *)a->joint_prerotations.data;        /* (127,4) */
    const int J = S3DM_N_JOINTS;
    const float LN2 = 0.69314718246459961f;

    for (int b = 0; b < B; b++) {
        const float *jp = joint_params + (size_t)b * J * 7;
        float *ls = out_local_skel + (size_t)b * J * 8;
        for (int j = 0; j < J; j++) {
            const float *jpr_j = jpr + (size_t)j * 4;
            const float *jto_j = jto + (size_t)j * 3;
            const float *jpj = jp + (size_t)j * 7;
            float *ls_j = ls + (size_t)j * 8;

            /* translation: jp[0:3] + joint_translation_offsets */
            ls_j[0] = jpj[0] + jto_j[0];
            ls_j[1] = jpj[1] + jto_j[1];
            ls_j[2] = jpj[2] + jto_j[2];

            /* quaternion: multiply_assume_normalized(prerot, euler_xyz_to_q(jp[3:6])) */
            float q_euler[4];
            s3dm_quat_from_euler_xyz(jpj + 3, q_euler);
            s3dm_quat_mul(jpr_j, q_euler, ls_j + 3);

            /* scale: exp(jp[6] * ln(2)) */
            ls_j[7] = expf(jpj[6] * LN2);
        }
    }
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Stage 8: prefix-product walker (local -> global) ------------- */

int sam3d_body_mhr_local_to_global_skel(
        const sam3d_body_mhr_assets *a,
        const float *local_skel,
        int B,
        float *out_global_skel)
{
    if (!a || !local_skel || !out_global_skel)
        return SAM3D_BODY_MHR_E_INVAL;

    const int J = S3DM_N_JOINTS;
    const int64_t *pmi = (const int64_t *)a->pmi.data; /* (2, 266) row-major */
    /* Row 0 = source_idx; Row 1 = target_idx. The jit uses:
     *   source = select(pmi, 0, 0);   (row 0)
     *   target = select(pmi, 0, 1);   (row 1)
     * Then: global[source] = multiply(global[target], global[source]).
     */
    const int64_t *src_all = pmi;
    const int64_t *tgt_all = pmi + S3DM_PMI_COLS;

    /* fp64 scratch for one batch */
    double *acc = (double *)malloc((size_t)J * 8 * sizeof(double));
    if (!acc) return SAM3D_BODY_MHR_E_LOAD;

    for (int b = 0; b < B; b++) {
        const float *lsb = local_skel + (size_t)b * J * 8;
        float *gsb = out_global_skel + (size_t)b * J * 8;
        /* promote fp32 local state into fp64 accumulator */
        for (int i = 0; i < J * 8; i++) acc[i] = (double)lsb[i];

        int col = 0;
        for (int stage = 0; stage < S3DM_PMI_STAGES; stage++) {
            int n = a->pmi_buffer_sizes[stage];
            for (int k = 0; k < n; k++) {
                int s = (int)src_all[col + k];
                int t = (int)tgt_all[col + k];
                double out_s[8];
                s3dm_skel_multiply_norm_d(&acc[(size_t)t * 8],
                                          &acc[(size_t)s * 8],
                                          out_s);
                memcpy(&acc[(size_t)s * 8], out_s, sizeof(out_s));
            }
            col += n;
        }

        for (int i = 0; i < J * 8; i++) gsb[i] = (float)acc[i];
    }
    free(acc);
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Stage 10A: blend_shape --------------------------------------- */

int sam3d_body_mhr_blend_shape(const sam3d_body_mhr_assets *a,
                               const float *shape_coeffs,
                               int B,
                               int n_threads,
                               float *out_verts)
{
    (void)n_threads;
    if (!a || !shape_coeffs || !out_verts)
        return SAM3D_BODY_MHR_E_INVAL;

    const float *SV = (const float *)a->blend_shape_vectors.data; /* (45, V, 3) */
    const float *BS = (const float *)a->blend_base_shape.data;    /* (V, 3)     */
    const int N = S3DM_N_SHAPE, V = S3DM_N_VERTS;

    for (int b = 0; b < B; b++) {
        const float *c = shape_coeffs + (size_t)b * N;
        float *o = out_verts + (size_t)b * V * 3;
        memcpy(o, BS, (size_t)V * 3 * sizeof(float));
        for (int n = 0; n < N; n++) {
            float cn = c[n];
            if (cn == 0.0f) continue;
            const float *SVn = SV + (size_t)n * V * 3;
            for (int i = 0; i < V * 3; i++) o[i] += cn * SVn[i];
        }
    }
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Stage 10B: face_expressions ---------------------------------- */

int sam3d_body_mhr_face_expressions(const sam3d_body_mhr_assets *a,
                                    const float *face_coeffs,
                                    int B,
                                    int n_threads,
                                    float *out_verts)
{
    (void)n_threads;
    if (!a || !face_coeffs || !out_verts)
        return SAM3D_BODY_MHR_E_INVAL;

    const float *SV = (const float *)a->face_shape_vectors.data; /* (72, V, 3) */
    const int N = S3DM_N_FACE, V = S3DM_N_VERTS;

    for (int b = 0; b < B; b++) {
        const float *c = face_coeffs + (size_t)b * N;
        float *o = out_verts + (size_t)b * V * 3;
        memset(o, 0, (size_t)V * 3 * sizeof(float));
        for (int n = 0; n < N; n++) {
            float cn = c[n];
            if (cn == 0.0f) continue;
            const float *SVn = SV + (size_t)n * V * 3;
            for (int i = 0; i < V * 3; i++) o[i] += cn * SVn[i];
        }
    }
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Stage 10C: pose_correctives ---------------------------------- */

/* batch6DFromXYZ for one (3,) euler -> (6,) 6D feat.
 *
 * jit (pymomentum/mhr/utils.py): build R (3,3) from XYZ-Euler, then
 *   col0 = R[..., :, 0]   (3,)
 *   col1 = R[..., :, 1]   (3,)
 *   out  = cat([col0, col1], -1)   -> column-stacked
 *
 * So the flat 6D is [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]
 * which equals [cy*cz, cy*sz, -sy, -cx*sz+sx*sy*cz, cx*cz+sx*sy*sz, sx*cy].
 * Subtract-identity on cols 0 and 4 still hits R[0,0] and R[1,1].
 */
static inline void s3dm_batch6D_from_xyz(const float *e, float *out6)
{
    float cx = cosf(e[0]), sx = sinf(e[0]);
    float cy = cosf(e[1]), sy = sinf(e[1]);
    float cz = cosf(e[2]), sz = sinf(e[2]);
    /* col 0 */
    out6[0] = cy * cz;                          /* R[0,0] */
    out6[1] = cy * sz;                          /* R[1,0] */
    out6[2] = -sy;                              /* R[2,0] */
    /* col 1 */
    out6[3] = -cx * sz + sx * sy * cz;          /* R[0,1] */
    out6[4] = cx * cz + sx * sy * sz;           /* R[1,1] */
    out6[5] = sx * cy;                          /* R[2,1] */
}

int sam3d_body_mhr_pose_correctives(const sam3d_body_mhr_assets *a,
                                    const float *joint_params,
                                    int B,
                                    int n_threads,
                                    float *out_verts)
{
    if (!a || !joint_params || !out_verts)
        return SAM3D_BODY_MHR_E_INVAL;

    const int V = S3DM_N_VERTS;
    const int J = S3DM_N_JOINTS;
    const int FEAT_IN = S3DM_N_PC_IN;    /* 750 */
    const int HID     = S3DM_N_PC_H;     /* 3000 */
    const int NNZ     = S3DM_N_PC_NNZ;   /* 53136 */

    const int64_t *spi = (const int64_t *)a->pc_sparse_indices.data; /* (2, NNZ) row-major */
    const int64_t *spi_row = spi;
    const int64_t *spi_col = spi + NNZ;
    const float   *spw = (const float *)a->pc_sparse_weight.data;    /* (NNZ,) */
    const float   *LW  = (const float *)a->pc_linear_weight.data;    /* (55317, 3000) */

    float *feat = (float *)malloc((size_t)FEAT_IN * sizeof(float));
    float *h    = (float *)malloc((size_t)HID     * sizeof(float));
    if (!feat || !h) { free(feat); free(h); return SAM3D_BODY_MHR_E_LOAD; }

    /* jit semantics (see /tmp/mhr_jit_code/__torch__/pymomentum/mhr/mhr.py):
     *   jp_reshape = joint_params.reshape(B, -1, 7)      -> (B, 127, 7)
     *   joint_euler_angles = jp_reshape[:, 2:127, 3:6]    -> (B, 125, 3)
     *   feat6d = batch6DFromXYZ(joint_euler_angles)       -> (B, 125, 6)
     *   feat6d[..., 0] -= 1; feat6d[..., 4] -= 1          (subtract identity)
     *   feat = flatten(feat6d, 1, 2)                      -> (B, 750)
     *   h = SparseLinear(feat)                            -> (B, 3000)
     *       where the sparse (3000, 750) weight is
     *       sparse_coo_tensor(indices=(2,NNZ), values=(NNZ)).
     *       matmul(W, x.T).T  ==  out[b, row] += w[k] * x[b, col]
     *       for each (row, col, w) in (sp_row[k], sp_col[k], sp_w[k]).
     *   h = ReLU(h)
     *   out = h @ LW.T                                   -> (B, 55317)
     *   reshape -> (B, V, 3)
     */
    for (int b = 0; b < B; b++) {
        const float *jp = joint_params + (size_t)b * J * 7;

        /* batch6D for joints 2..126 (125 joints), euler at jp[j, 3:6]. */
        for (int jj = 0; jj < 125; jj++) {
            const float *e = jp + (size_t)(jj + 2) * 7 + 3;
            s3dm_batch6D_from_xyz(e, feat + (size_t)jj * 6);
        }
        /* Subtract identity on columns 0 and 4 of the last axis. */
        for (int jj = 0; jj < 125; jj++) {
            feat[(size_t)jj * 6 + 0] -= 1.0f;
            feat[(size_t)jj * 6 + 4] -= 1.0f;
        }
        /* feat already flattened to 750 above. */

        /* Sparse matvec: h[row] = sum_k sp_w[k] * feat[sp_col[k]] where row==sp_row[k]. */
        memset(h, 0, (size_t)HID * sizeof(float));
        for (int k = 0; k < NNZ; k++) {
            int row = (int)spi_row[k];
            int col = (int)spi_col[k];
            h[row] += spw[k] * feat[col];
        }
        /* ReLU */
        for (int i = 0; i < HID; i++)
            if (h[i] < 0.0f) h[i] = 0.0f;

        /* Dense matvec: out[b, row] = sum_c LW[row, c] * h[c], row in [0..55317).
         * 55317×3000 = 166M FMAs — the only nontrivial cost in MHR forward.
         * Parallelize across rows; rows are independent. */
        float *o = out_verts + (size_t)b * V * 3;
        const int OUT = V * 3; /* 55317 */
#if defined(_OPENMP)
        if (n_threads > 1) {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (int row = 0; row < OUT; row++) {
                const float *Wrow = LW + (size_t)row * HID;
                float acc = 0.0f;
                for (int c = 0; c < HID; c++)
                    acc += Wrow[c] * h[c];
                o[row] = acc;
            }
        } else
#endif
        {
            for (int row = 0; row < OUT; row++) {
                const float *Wrow = LW + (size_t)row * HID;
                float acc = 0.0f;
                for (int c = 0; c < HID; c++)
                    acc += Wrow[c] * h[c];
                o[row] = acc;
            }
        }
    }

    free(feat);
    free(h);
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Stage 11: LBS skin_points ------------------------------------ */

int sam3d_body_mhr_skin_points(const sam3d_body_mhr_assets *a,
                               const float *global_skel,
                               const float *rest_verts,
                               int B,
                               float *out_skinned)
{
    if (!a || !global_skel || !rest_verts || !out_skinned)
        return SAM3D_BODY_MHR_E_INVAL;

    const int J = S3DM_N_JOINTS;
    const int V = S3DM_N_VERTS;
    const int K = S3DM_N_SKIN;
    const float   *IBP = (const float *)a->inverse_bind_pose.data;   /* (J, 8)    */
    const int32_t *si  = (const int32_t *)a->skin_indices_flat.data; /* (K,)      */
    const float   *sw  = (const float *)a->skin_weights_flat.data;   /* (K,)      */
    const int64_t *vi  = (const int64_t *)a->vert_indices_flat.data; /* (K,)      */

    float *jstate = (float *)malloc((size_t)J * 8 * sizeof(float));
    if (!jstate) return SAM3D_BODY_MHR_E_LOAD;

    for (int b = 0; b < B; b++) {
        const float *gs = global_skel + (size_t)b * J * 8;
        const float *rv = rest_verts  + (size_t)b * V * 3;
        float *out = out_skinned + (size_t)b * V * 3;

        /* joint_state = skel_multiply(global, inverse_bind_pose) */
        for (int j = 0; j < J; j++) {
            s3dm_skel_multiply(gs + (size_t)j * 8,
                               IBP + (size_t)j * 8,
                               jstate + (size_t)j * 8);
        }

        memset(out, 0, (size_t)V * 3 * sizeof(float));
        for (int k = 0; k < K; k++) {
            int j = (int)si[k];
            int v = (int)vi[k];
            float w = sw[k];
            float tp[3];
            s3dm_skel_transform_point(jstate + (size_t)j * 8,
                                      rv + (size_t)v * 3,
                                      tp);
            out[(size_t)v * 3 + 0] += w * tp[0];
            out[(size_t)v * 3 + 1] += w * tp[1];
            out[(size_t)v * 3 + 2] += w * tp[2];
        }
    }
    free(jstate);
    return SAM3D_BODY_MHR_E_OK;
}

/* ---- Top-level forward -------------------------------------------- */

int sam3d_body_mhr_forward(const sam3d_body_mhr_assets *a,
                           const float *model_params,
                           const float *shape,
                           const float *face,
                           int B,
                           int apply_correctives,
                           int n_threads,
                           float *scratch,
                           float *out_skinned_verts,
                           float *out_global_skel)
{
    if (!a || !model_params || !shape || !out_skinned_verts || !out_global_skel)
        return SAM3D_BODY_MHR_E_INVAL;

    const int V = S3DM_N_VERTS;
    const int J = S3DM_N_JOINTS;

    /* Carve scratch: joint_params (B, 889) | local_skel (B, J, 8)
     * | rest_identity (B, V, 3) | face_offsets (B, V, 3) | pc_offsets (B, V, 3) */
    size_t need = (size_t)B * (S3DM_N_PTRANS_OUT + J * 8 + V * 3 * 3);
    float *buf = scratch;
    int owned = 0;
    if (!buf) {
        buf = (float *)malloc(need * sizeof(float));
        if (!buf) return SAM3D_BODY_MHR_E_LOAD;
        owned = 1;
    }
    float *jp      = buf;
    float *lskel   = jp + (size_t)B * S3DM_N_PTRANS_OUT;
    float *rest_id = lskel + (size_t)B * J * 8;
    float *face_v  = rest_id + (size_t)B * V * 3;
    float *pc_v    = face_v + (size_t)B * V * 3;

    int r;
    r = sam3d_body_mhr_parameter_transform(a, model_params, B, n_threads, jp);
    if (r) goto done;
    r = sam3d_body_mhr_joint_params_to_local_skel(a, jp, B, lskel);
    if (r) goto done;
    r = sam3d_body_mhr_local_to_global_skel(a, lskel, B, out_global_skel);
    if (r) goto done;
    r = sam3d_body_mhr_blend_shape(a, shape, B, n_threads, rest_id);
    if (r) goto done;
    if (face) {
        r = sam3d_body_mhr_face_expressions(a, face, B, n_threads, face_v);
        if (r) goto done;
    } else {
        memset(face_v, 0, (size_t)B * V * 3 * sizeof(float));
    }
    if (apply_correctives) {
        r = sam3d_body_mhr_pose_correctives(a, jp, B, n_threads, pc_v);
        if (r) goto done;
    } else {
        memset(pc_v, 0, (size_t)B * V * 3 * sizeof(float));
    }
    /* linear_model_unposed = rest_identity + face + pose_correctives. */
    size_t Ntot = (size_t)B * V * 3;
    for (size_t i = 0; i < Ntot; i++)
        rest_id[i] += face_v[i] + pc_v[i];

    r = sam3d_body_mhr_skin_points(a, out_global_skel, rest_id, B,
                                   out_skinned_verts);
done:
    if (owned) free(buf);
    return r;
}

#endif /* SAM3D_BODY_MHR_IMPLEMENTATION */
#endif /* SAM3D_BODY_MHR_H */
